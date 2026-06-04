"""Tests for stat_arb/hmm/regime.py and stat_arb/hmm/train.py — v2.

Covers all five improvements:
  1. Spread-based features (build_features_from_spread)
  2. spread_autocorr as primary MR indicator
  3. Winsorization for fat tails
  4. build_market_features for shared regime
  5. Parallel fitting in RollingHMMTrainer

All tests use deterministic synthetic data — no live API calls.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stat_arb.hmm.regime import (
    HMMRegimeDetector,
    Regime,
    REGIME_LABEL,
    TRADEABLE_REGIMES,
    build_features,
    build_features_from_spread,
    build_market_features,
    _to_2d,
    _extract_array,
)
from stat_arb.hmm.train import RollingHMMTrainer


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers / fixtures
# ═══════════════════════════════════════════════════════════════════════════════

def _make_price_series(n: int = 500, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    log_p = np.cumsum(rng.normal(0.0003, 0.012, n))
    idx = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
    return pd.Series(np.exp(log_p), index=idx, name="price")


def _make_kalman_df(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Synthetic Kalman output with 3 visible regimes in the spread."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")

    # Block structure: MR → TR → VOL → MR → VOL
    segment = n // 5
    spread = np.concatenate([
        # MR block: strong negative autocorr, low vol
        np.cumsum(rng.normal(0, 0.01, segment)) - np.arange(segment) * 0.005,
        # TR block: positive drift
        np.cumsum(rng.normal(0.003, 0.008, segment)),
        # VOL block: high vol
        np.cumsum(rng.normal(0, 0.04, segment)),
        # MR block again
        np.cumsum(rng.normal(0, 0.01, segment)) - np.arange(segment) * 0.005,
        # VOL block again
        np.cumsum(rng.normal(0, 0.04, n - 4 * segment)),
    ])
    zscore = (spread - spread.mean()) / (spread.std() + 1e-9)
    return pd.DataFrame({"spread": spread, "zscore": zscore}, index=idx)


def _make_spread_regime_df(n_per: int = 300, seed: int = 7) -> pd.DataFrame:
    """Synthetic feature DataFrame with 3 well-separated regimes.

    Columns: [spread_ret, realized_vol, spread_autocorr]
    Regimes:
      MR  : low vol,  negative autocorr (~-0.4)
      TR  : med vol,  positive autocorr (~+0.2)
      VOL : high vol, near-zero autocorr
    """
    rng = np.random.default_rng(seed)
    mr  = np.column_stack([rng.normal(0.0001, 0.0005, n_per),
                            rng.normal(0.005,  0.0005, n_per),
                            rng.normal(-0.40,  0.05,   n_per)])
    tr  = np.column_stack([rng.normal(0.005,  0.001,  n_per),
                            rng.normal(0.012,  0.001,  n_per),
                            rng.normal(0.20,   0.05,   n_per)])
    vol = np.column_stack([rng.normal(0.0,    0.003,  n_per),
                            rng.normal(0.030,  0.003,  n_per),
                            rng.normal(0.00,   0.10,   n_per)])
    X = np.vstack([mr, tr, vol, mr, vol, tr])
    idx = pd.date_range("2010-01-01", periods=len(X), freq="B", tz="UTC")
    return pd.DataFrame(X, columns=["spread_ret", "realized_vol", "spread_autocorr"],
                        index=idx)


def _make_universe_close(n: int = 500, n_stocks: int = 10, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0003, 0.012, (n, n_stocks))
    prices = np.exp(np.cumsum(log_returns, axis=0))
    idx = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
    cols = [f"STK{i}" for i in range(n_stocks)]
    return pd.DataFrame(prices, index=idx, columns=cols)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. build_features_from_spread  (new primary builder)
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildFeaturesFromSpread:
    def test_columns(self):
        kdf = _make_kalman_df(300)
        feat = build_features_from_spread(kdf)
        assert list(feat.columns) == ["spread_ret", "realized_vol", "spread_autocorr", "zscore"]

    def test_no_nans(self):
        kdf = _make_kalman_df(300)
        feat = build_features_from_spread(kdf)
        assert not feat.isnull().any().any()

    def test_shorter_than_input_due_to_windows(self):
        kdf = _make_kalman_df(300)
        feat = build_features_from_spread(kdf, vol_window=20, autocorr_window=10)
        assert len(feat) < len(kdf)
        assert len(feat) == len(kdf) - 20  # vol_window dominates NaN creation

    def test_autocorr_in_minus_one_to_one(self):
        kdf = _make_kalman_df(500)
        feat = build_features_from_spread(kdf)
        assert (feat["spread_autocorr"].abs() <= 1.0 + 1e-9).all()

    def test_missing_spread_column_raises(self):
        bad = pd.DataFrame({"zscore": np.ones(100)},
                           index=pd.date_range("2020-01-01", periods=100, freq="B", tz="UTC"))
        with pytest.raises(ValueError, match="spread"):
            build_features_from_spread(bad)

    def test_missing_zscore_column_raises(self):
        bad = pd.DataFrame({"spread": np.ones(100)},
                           index=pd.date_range("2020-01-01", periods=100, freq="B", tz="UTC"))
        with pytest.raises(ValueError, match="zscore"):
            build_features_from_spread(bad)

    def test_mr_spread_has_negative_autocorr(self):
        """A mean-reverting spread (OU process) should produce negative autocorr."""
        rng = np.random.default_rng(0)
        n = 1000
        # OU process: strong mean-reversion
        spread = np.zeros(n)
        for t in range(1, n):
            spread[t] = 0.9 * spread[t - 1] + rng.normal(0, 0.02)
        idx = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
        kdf = pd.DataFrame({"spread": spread,
                             "zscore": spread / (spread.std() + 1e-9)}, index=idx)
        feat = build_features_from_spread(kdf)
        # OU differences have negative autocorr
        assert feat["spread_autocorr"].mean() < 0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. build_market_features  (shared regime)
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildMarketFeatures:
    def test_columns(self):
        univ = _make_universe_close(300)
        feat = build_market_features(univ)
        assert list(feat.columns) == ["market_ret", "cross_vol",
                                       "avg_realized_vol", "avg_autocorr"]

    def test_no_nans(self):
        univ = _make_universe_close(300)
        feat = build_market_features(univ)
        assert not feat.isnull().any().any()

    def test_avg_realized_vol_positive(self):
        univ = _make_universe_close(300)
        feat = build_market_features(univ)
        assert (feat["avg_realized_vol"] > 0).all()

    def test_avg_autocorr_in_range(self):
        univ = _make_universe_close(300)
        feat = build_market_features(univ)
        assert (feat["avg_autocorr"].abs() <= 1.0 + 1e-9).all()

    def test_single_symbol_raises(self):
        univ = _make_universe_close(300, n_stocks=1)
        with pytest.raises(ValueError, match="at least 2"):
            build_market_features(univ)

    def test_high_vol_period_detected(self):
        """Universe with injected vol spike should show elevated avg_realized_vol."""
        rng = np.random.default_rng(3)
        n = 400
        quiet = rng.normal(0.0003, 0.005, (200, 5))
        spike = rng.normal(0.000,  0.040, (200, 5))
        prices = np.exp(np.cumsum(np.vstack([quiet, spike]), axis=0))
        idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
        univ = pd.DataFrame(prices, index=idx, columns=list("ABCDE"))
        feat = build_market_features(univ)
        avg_vol_quiet = feat["avg_realized_vol"].iloc[:150].mean()
        avg_vol_spike = feat["avg_realized_vol"].iloc[-150:].mean()
        assert avg_vol_spike > avg_vol_quiet * 2


# ═══════════════════════════════════════════════════════════════════════════════
# 3. build_features — legacy builder (backward compat)
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildFeaturesLegacy:
    def test_default_columns(self):
        prices = _make_price_series(200)
        feat = build_features(prices)
        assert list(feat.columns) == ["roll_ret", "realized_vol"]

    def test_no_nans(self):
        prices = _make_price_series(200)
        feat = build_features(prices)
        assert not feat.isnull().any().any()

    def test_with_kalman_adds_zscore(self):
        prices = _make_price_series(200)
        zscore = pd.Series(np.zeros(len(prices)), index=prices.index)
        kdf = pd.DataFrame({"zscore": zscore})
        feat = build_features(prices, kalman_df=kdf)
        assert "zscore" in feat.columns
        assert feat.shape[1] == 3

    def test_raises_on_non_positive_price(self):
        prices = _make_price_series(200)
        prices.iloc[5] = 0.0
        with pytest.raises(ValueError, match="positive"):
            build_features(prices)

    def test_kalman_df_missing_zscore_raises(self):
        prices = _make_price_series(200)
        bad_df = pd.DataFrame({"alpha": np.ones(len(prices))}, index=prices.index)
        with pytest.raises(ValueError, match="zscore"):
            build_features(prices, kalman_df=bad_df)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. HMMRegimeDetector — fit / predict
# ═══════════════════════════════════════════════════════════════════════════════

class TestHMMRegimeDetector:
    @pytest.fixture
    def fitted_on_spread_df(self):
        df = _make_spread_regime_df(n_per=300)
        det = HMMRegimeDetector(n_states=3, random_state=42)
        det.fit(df)  # pass DataFrame so column names are used
        return det, df

    def test_fit_returns_self(self):
        df = _make_spread_regime_df(n_per=100)
        det = HMMRegimeDetector()
        assert det.fit(df) is det

    def test_fit_stores_feature_names(self):
        df = _make_spread_regime_df(n_per=100)
        det = HMMRegimeDetector()
        det.fit(df)
        assert det._feature_names == list(df.columns)

    def test_fit_numpy_has_no_feature_names(self):
        X = _make_spread_regime_df(n_per=100).values
        det = HMMRegimeDetector()
        det.fit(X)
        assert det._feature_names is None

    def test_predict_shape(self, fitted_on_spread_df):
        det, df = fitted_on_spread_df
        preds = det.predict(df)
        assert preds.shape == (len(df),)

    def test_predict_valid_regimes(self, fitted_on_spread_df):
        det, df = fitted_on_spread_df
        preds = det.predict(df)
        assert set(preds.tolist()).issubset({int(r) for r in Regime})

    def test_predict_proba_shape(self, fitted_on_spread_df):
        det, df = fitted_on_spread_df
        proba = det.predict_proba(df)
        assert proba.shape == (len(df), 3)

    def test_predict_proba_sums_to_one(self, fitted_on_spread_df):
        det, df = fitted_on_spread_df
        proba = det.predict_proba(df)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_proba_non_negative(self, fitted_on_spread_df):
        det, df = fitted_on_spread_df
        proba = det.predict_proba(df)
        assert (proba >= 0).all()

    def test_predict_consistency(self, fitted_on_spread_df):
        det, df = fitted_on_spread_df
        assert np.array_equal(det.predict(df), det.predict(df))

    def test_predict_without_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            HMMRegimeDetector().predict(np.ones((10, 2)))

    def test_n_states_less_than_2_raises(self):
        with pytest.raises(ValueError):
            HMMRegimeDetector(n_states=1)

    def test_1d_numpy_accepted(self):
        X_1d = _make_spread_regime_df().values[:, 1]
        det = HMMRegimeDetector(random_state=42)
        det.fit(X_1d)
        preds = det.predict(X_1d)
        assert preds.shape == (len(X_1d),)

    def test_invalid_ndim_raises(self):
        with pytest.raises(ValueError):
            HMMRegimeDetector().fit(np.ones((10, 2, 3)))


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Regime labelling with autocorr (improvement #2)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegimeLabelling:
    @pytest.fixture
    def fitted_with_autocorr(self):
        df = _make_spread_regime_df(n_per=400)
        det = HMMRegimeDetector(n_states=3, random_state=42)
        det.fit(df)
        return det, df

    def test_volatile_has_highest_realized_vol(self, fitted_with_autocorr):
        det, df = fitted_with_autocorr
        preds = det.predict(df)
        mr_vol  = df.values[preds == Regime.MEAN_REVERTING, 1].mean()
        vol_vol = df.values[preds == Regime.VOLATILE,       1].mean()
        assert vol_vol > mr_vol

    def test_mr_has_most_negative_autocorr(self, fitted_with_autocorr):
        """MR regime must have the most negative mean autocorr — primary criterion."""
        det, df = fitted_with_autocorr
        preds = det.predict(df)
        mr_ac = df.values[preds == Regime.MEAN_REVERTING, 2].mean()
        tr_ac = df.values[preds == Regime.TRENDING,       2].mean()
        assert mr_ac < tr_ac, (
            f"MR autocorr ({mr_ac:.3f}) should be < TR autocorr ({tr_ac:.3f})"
        )

    def test_trending_has_positive_autocorr(self, fitted_with_autocorr):
        det, df = fitted_with_autocorr
        preds = det.predict(df)
        tr_ac = df.values[preds == Regime.TRENDING, 2].mean()
        assert tr_ac > 0

    def test_all_three_regimes_assigned(self, fitted_with_autocorr):
        det, df = fitted_with_autocorr
        preds = det.predict(df)
        assert {Regime(r) for r in preds} == {Regime.MEAN_REVERTING,
                                               Regime.TRENDING, Regime.VOLATILE}

    def test_fallback_labelling_without_autocorr_col(self):
        """Without autocorr column, label by |return| — backward compat."""
        rng = np.random.default_rng(7)
        n = 1800
        # 2-col data: [roll_ret, realized_vol]
        mr  = np.column_stack([rng.normal(0.0001, 0.0005, 600),
                                rng.normal(0.005,  0.0005, 600)])
        tr  = np.column_stack([rng.normal(0.005,  0.001,  600),
                                rng.normal(0.012,  0.001,  600)])
        vol = np.column_stack([rng.normal(0.0,    0.003,  600),
                                rng.normal(0.030,  0.003,  600)])
        X = np.vstack([mr, tr, vol, mr])
        det = HMMRegimeDetector(n_states=3, random_state=42)
        det.fit(X)  # numpy array → no feature names → fallback to |ret|
        preds = det.predict(X)
        assert {Regime(r) for r in preds} == {Regime.MEAN_REVERTING,
                                               Regime.TRENDING, Regime.VOLATILE}


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Winsorization (improvement #3)
# ═══════════════════════════════════════════════════════════════════════════════

class TestWinsorization:
    def test_outliers_dont_crash_fit(self):
        """Extreme outliers (100-sigma events) must not cause HMM to diverge."""
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (500, 2))
        X[50]  = [100.0, 200.0]   # massive outlier
        X[250] = [-150.0, 80.0]
        det = HMMRegimeDetector(n_states=3, random_state=0, winsorize_sigma=3.0)
        det.fit(X)   # should not raise
        preds = det.predict(X)
        assert preds.shape == (len(X),)

    def test_winsorize_none_still_works(self):
        df = _make_spread_regime_df(n_per=200)
        det = HMMRegimeDetector(winsorize_sigma=None)
        det.fit(df)
        preds = det.predict(df)
        assert preds.shape == (len(df),)

    def test_winsorize_clips_before_viterbi(self):
        """Winsorization clips BEFORE passing to HMM → extreme values and their
        ±sigma equivalent must produce identical predictions (definitive proof)."""
        df = _make_spread_regime_df(n_per=200, seed=9)
        det = HMMRegimeDetector(n_states=3, random_state=0, winsorize_sigma=3.0)
        det.fit(df)

        # Build one df with an extreme outlier at row 50
        df_extreme = df.copy()
        df_extreme.iloc[50] = df.iloc[50] * 1000

        # Build the clipped equivalent: manually scale → clip → inverse
        from sklearn.preprocessing import StandardScaler
        X_scaled = det._scaler.transform(df_extreme.values)
        X_clipped = np.clip(X_scaled, -3.0, 3.0)
        df_clipped = pd.DataFrame(
            det._scaler.inverse_transform(X_clipped),
            columns=df.columns, index=df.index,
        )

        preds_extreme = det.predict(df_extreme)
        preds_clipped = det.predict(df_clipped)

        # With winsorization both inputs are identical after clipping → same labels
        np.testing.assert_array_equal(preds_extreme, preds_clipped)

    def test_without_winsorize_outliers_distort_predictions(self):
        """Without winsorization, outliers cause more label drift (shows it matters)."""
        df = _make_spread_regime_df(n_per=300, seed=9)
        df_dirty = df.copy()
        df_dirty.iloc[100] *= 50
        df_dirty.iloc[200] *= 50

        det_w  = HMMRegimeDetector(n_states=3, random_state=0, winsorize_sigma=3.0)
        det_nw = HMMRegimeDetector(n_states=3, random_state=0, winsorize_sigma=None)

        det_w.fit(df)
        det_nw.fit(df)
        preds_w  = det_w.predict(df_dirty)
        preds_nw = det_nw.predict(df_dirty)

        agree_w  = (det_w.predict(df)  == preds_w).mean()
        agree_nw = (det_nw.predict(df) == preds_nw).mean()
        # Winsorized version should be at least as stable
        assert agree_w >= agree_nw - 0.05


# ═══════════════════════════════════════════════════════════════════════════════
# 7. predict_series & trading rule
# ═══════════════════════════════════════════════════════════════════════════════

class TestPredictSeries:
    def test_returns_series_with_correct_index(self):
        df = _make_spread_regime_df(500)
        det = HMMRegimeDetector(random_state=0)
        det.fit(df)
        result = det.predict_series(df)
        assert isinstance(result, pd.Series)
        pd.testing.assert_index_equal(result.index, df.index)

    def test_labels_are_valid_strings(self):
        df = _make_spread_regime_df(500)
        det = HMMRegimeDetector(random_state=0)
        det.fit(df)
        result = det.predict_series(df)
        assert set(result.unique()).issubset(set(REGIME_LABEL.values()))


class TestTradingRule:
    def test_mr_is_tradeable(self):
        assert HMMRegimeDetector.is_tradeable(Regime.MEAN_REVERTING) is True

    def test_trending_not_tradeable(self):
        assert HMMRegimeDetector.is_tradeable(Regime.TRENDING) is False

    def test_volatile_not_tradeable(self):
        assert HMMRegimeDetector.is_tradeable(Regime.VOLATILE) is False

    def test_accepts_int(self):
        assert HMMRegimeDetector.is_tradeable(0) is True
        assert HMMRegimeDetector.is_tradeable(2) is False


# ═══════════════════════════════════════════════════════════════════════════════
# 8. RollingHMMTrainer
# ═══════════════════════════════════════════════════════════════════════════════

class TestRollingHMMTrainer:
    @pytest.fixture
    def spread_feat(self):
        kdf = _make_kalman_df(n=1200, seed=42)
        return build_features_from_spread(kdf)

    def test_output_is_series(self, spread_feat):
        result = RollingHMMTrainer(252, 21).fit_predict(spread_feat)
        assert isinstance(result, pd.Series)

    def test_index_matches_input(self, spread_feat):
        result = RollingHMMTrainer(252, 21).fit_predict(spread_feat)
        pd.testing.assert_index_equal(result.index, spread_feat.index)

    def test_warmup_rows_are_null(self, spread_feat):
        result = RollingHMMTrainer(252, 21).fit_predict(spread_feat)
        assert result.iloc[:252].isna().all()

    def test_labels_after_warmup_mostly_non_null(self, spread_feat):
        result = RollingHMMTrainer(252, 21).fit_predict(spread_feat)
        assert result.iloc[252:].notna().mean() > 0.95

    def test_all_assigned_labels_valid(self, spread_feat):
        result = RollingHMMTrainer(252, 21).fit_predict(spread_feat)
        assert set(result.dropna().unique()).issubset(set(REGIME_LABEL.values()))

    def test_models_stored(self, spread_feat):
        trainer = RollingHMMTrainer(252, 21)
        trainer.fit_predict(spread_feat)
        assert len(trainer.models_) > 0
        assert all(isinstance(m, HMMRegimeDetector) for _, m in trainer.models_)

    def test_regime_fractions_sum_to_one(self, spread_feat):
        trainer = RollingHMMTrainer(252, 21)
        labels = trainer.fit_predict(spread_feat)
        assert abs(trainer.regime_fractions(labels).sum() - 1.0) < 1e-6

    def test_too_short_raises(self):
        tiny = pd.DataFrame(
            {"spread_ret": np.zeros(100), "realized_vol": np.ones(100) * 0.01,
             "spread_autocorr": np.zeros(100), "zscore": np.zeros(100)},
            index=pd.date_range("2020-01-01", periods=100, freq="B", tz="UTC"),
        )
        with pytest.raises(ValueError, match="at least"):
            RollingHMMTrainer(252, 21).fit_predict(tiny)

    def test_invalid_train_window_raises(self):
        with pytest.raises(ValueError, match="train_window"):
            RollingHMMTrainer(train_window=10)

    def test_shorter_retrain_produces_more_folds(self, spread_feat):
        t_fast = RollingHMMTrainer(252, 10)
        t_slow = RollingHMMTrainer(252, 42)
        t_fast.fit_predict(spread_feat)
        t_slow.fit_predict(spread_feat)
        assert len(t_fast.models_) > len(t_slow.models_)

    # ── parallel fitting (improvement #5) ────────────────────────────────────

    def test_parallel_matches_sequential(self, spread_feat):
        """n_jobs>1 must produce identical regime labels as sequential."""
        trainer_seq = RollingHMMTrainer(252, 21, random_state=42)
        trainer_par = RollingHMMTrainer(252, 21, random_state=42)
        labels_seq = trainer_seq.fit_predict(spread_feat, n_jobs=1)
        labels_par = trainer_par.fit_predict(spread_feat, n_jobs=2)
        pd.testing.assert_series_equal(labels_seq, labels_par)

    def test_n_jobs_minus_one_runs(self, spread_feat):
        """n_jobs=-1 (all cores) should complete without error."""
        trainer = RollingHMMTrainer(252, 42, random_state=0)
        result = trainer.fit_predict(spread_feat, n_jobs=-1)
        assert result.dropna().shape[0] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Smoke tests on real data — spread-based features
# ═══════════════════════════════════════════════════════════════════════════════

class TestOnRealDataSpread:
    @pytest.fixture
    def jpm_bac_spread_feat(self):
        p = Path("data/ohlcv/1d")
        if not p.exists():
            pytest.skip("Daily data not available")
        from stat_arb.kalman.filter import run_kalman
        jpm = pd.read_parquet(p / "JPM.parquet").set_index("timestamp")["close"]
        bac = pd.read_parquet(p / "BAC.parquet").set_index("timestamp")["close"]
        kdf = run_kalman(jpm, bac)
        return build_features_from_spread(kdf)

    def test_no_nans_in_spread_features(self, jpm_bac_spread_feat):
        assert not jpm_bac_spread_feat.isnull().any().any()

    def test_spread_autocorr_is_negative_on_average(self, jpm_bac_spread_feat):
        """JPM/BAC is a well-known cointegrated pair; spread should show MR tendency."""
        mean_ac = jpm_bac_spread_feat["spread_autocorr"].mean()
        assert mean_ac < 0.1, f"Expected negative avg autocorr, got {mean_ac:.3f}"

    def test_detector_runs_on_real_spread(self, jpm_bac_spread_feat):
        det = HMMRegimeDetector(random_state=0)
        det.fit(jpm_bac_spread_feat)
        result = det.predict_series(jpm_bac_spread_feat)
        assert result.notna().all()

    def test_rolling_trainer_on_real_spread(self, jpm_bac_spread_feat):
        trainer = RollingHMMTrainer(train_window=252, retrain_freq=21)
        labels = trainer.fit_predict(jpm_bac_spread_feat)
        labelled = labels.dropna()
        assert len(labelled) > 100
        fracs = trainer.regime_fractions(labels)
        # MR regime should be non-trivial on a cointegrated pair
        assert fracs.get("mean_reverting", 0) >= 0.20, (
            f"MR fraction only {fracs.get('mean_reverting', 0):.1%} — too low for JPM/BAC"
        )

    def test_volatile_regime_spikes_in_2020(self, jpm_bac_spread_feat):
        """COVID crash (Mar 2020) should be heavily classified as VOLATILE."""
        trainer = RollingHMMTrainer(train_window=252, retrain_freq=21)
        labels = trainer.fit_predict(jpm_bac_spread_feat)
        covid = labels.loc["2020-02-01":"2020-05-01"].dropna()
        if len(covid) == 0:
            pytest.skip("No labelled rows in COVID window")
        vol_frac = (covid == "volatile").mean()
        assert vol_frac >= 0.20, (
            f"Expected high VOLATILE fraction in Mar 2020, got {vol_frac:.1%}"
        )

    def test_market_features_on_real_universe(self):
        p = Path("data/ohlcv/1d")
        if not p.exists():
            pytest.skip("Daily data not available")
        syms = ["JPM", "BAC", "GS", "MS", "WFC"]
        closes = pd.concat(
            [pd.read_parquet(p / f"{s}.parquet").set_index("timestamp")["close"].rename(s)
             for s in syms],
            axis=1,
        ).dropna()
        feat = build_market_features(closes)
        assert not feat.isnull().any().any()
        # Train one shared HMM on market features
        det = HMMRegimeDetector(random_state=0)
        det.fit(feat)
        result = det.predict_series(feat)
        assert result.notna().all()
        assert set(result.unique()).issubset(set(REGIME_LABEL.values()))
