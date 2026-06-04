"""Tests for stat_arb/kalman/filter.py — v2.

Covers original tests plus three new areas:
  TestWarmup       : warmup_bars masking in run()
  TestDeltaTuning  : tune_delta() MLE calibration
  TestOnRealData   : updated real-data smoke tests with warmup + tuning

All tests use deterministic synthetic data — no live API calls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stat_arb.kalman.filter import (
    KalmanFilterHedge,
    KalmanState,
    _innovation_log_likelihood,
    run_kalman,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_pair():
    """Synthetic cointegrated pair: log_y = 0.3 + 1.5 * log_x + noise."""
    rng = np.random.default_rng(42)
    n = 1_000
    true_alpha, true_beta = 0.3, 1.5
    log_x = np.cumsum(rng.normal(0, 0.01, n))
    noise = rng.normal(0, 0.02, n)
    log_y = true_alpha + true_beta * log_x + noise
    return log_y, log_x, true_alpha, true_beta


@pytest.fixture
def price_series(synthetic_pair):
    log_y, log_x, _, _ = synthetic_pair
    idx = pd.date_range("2015-01-01", periods=len(log_y), freq="B", tz="UTC")
    return (
        pd.Series(np.exp(log_y), index=idx, name="Y"),
        pd.Series(np.exp(log_x), index=idx, name="X"),
    )


# ── output shape & schema ─────────────────────────────────────────────────────

class TestOutputShape:
    def test_run_returns_correct_shape(self, synthetic_pair):
        log_y, log_x, _, _ = synthetic_pair
        result = KalmanFilterHedge().run(log_y, log_x)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (len(log_y), 5)
        assert list(result.columns) == ["alpha", "beta", "spread", "spread_std", "zscore"]

    def test_no_nans_without_warmup(self, synthetic_pair):
        log_y, log_x, _, _ = synthetic_pair
        result = KalmanFilterHedge().run(log_y, log_x, warmup_bars=0)
        assert not result.isnull().any().any()

    def test_spread_std_is_positive(self, synthetic_pair):
        log_y, log_x, _, _ = synthetic_pair
        result = KalmanFilterHedge().run(log_y, log_x)
        assert (result["spread_std"].dropna() > 0).all()

    def test_zscore_equals_spread_over_std(self, synthetic_pair):
        log_y, log_x, _, _ = synthetic_pair
        result = KalmanFilterHedge().run(log_y, log_x, warmup_bars=0)
        np.testing.assert_allclose(
            result["zscore"].values,
            (result["spread"] / result["spread_std"]).values,
            rtol=1e-10,
        )


# ── convergence ───────────────────────────────────────────────────────────────

class TestConvergence:
    def test_beta_converges_to_true_value(self, synthetic_pair):
        log_y, log_x, _, true_beta = synthetic_pair
        result = KalmanFilterHedge(delta=1e-4, vt=1e-3).run(log_y, log_x)
        assert abs(result["beta"].iloc[-200:].mean() - true_beta) < 0.15

    def test_alpha_converges_to_true_value(self, synthetic_pair):
        log_y, log_x, true_alpha, _ = synthetic_pair
        result = KalmanFilterHedge(delta=1e-4, vt=1e-3).run(log_y, log_x)
        assert abs(result["alpha"].iloc[-200:].mean() - true_alpha) < 0.15

    def test_spread_near_zero_mean_after_warmup(self, synthetic_pair):
        log_y, log_x, _, _ = synthetic_pair
        result = KalmanFilterHedge(delta=1e-4, vt=1e-3).run(log_y, log_x)
        assert abs(result["spread"].iloc[200:].mean()) < 0.05

    def test_larger_delta_adapts_faster(self, synthetic_pair):
        log_y, log_x, _, _ = synthetic_pair
        slow = KalmanFilterHedge(delta=1e-6).run(log_y, log_x, warmup_bars=0)
        fast = KalmanFilterHedge(delta=1e-3).run(log_y, log_x, warmup_bars=0)
        assert fast["beta"].std() > slow["beta"].std()


# ── step() vs run() consistency ───────────────────────────────────────────────

class TestStepConsistency:
    def test_step_matches_run_element_by_element(self, synthetic_pair):
        log_y, log_x, _, _ = synthetic_pair
        kf = KalmanFilterHedge()
        result_run = kf.run(log_y, log_x, warmup_bars=0)

        kf.reset()
        spreads, stds, betas, alphas = [], [], [], []
        for t in range(len(log_y)):
            e, std = kf.step(log_y[t], log_x[t])
            spreads.append(e)
            stds.append(std)
            betas.append(kf.beta)
            alphas.append(kf.alpha)

        np.testing.assert_allclose(result_run["spread"].values,     spreads, rtol=1e-12)
        np.testing.assert_allclose(result_run["spread_std"].values, stds,    rtol=1e-12)
        np.testing.assert_allclose(result_run["beta"].values,       betas,   rtol=1e-12)
        np.testing.assert_allclose(result_run["alpha"].values,      alphas,  rtol=1e-12)


# ── state snapshot / restore ──────────────────────────────────────────────────

class TestStateManagement:
    def test_snapshot_and_restore(self, synthetic_pair):
        log_y, log_x, _, _ = synthetic_pair
        kf = KalmanFilterHedge()
        mid = len(log_y) // 2
        kf.run(log_y[:mid], log_x[:mid])
        saved = kf.snapshot()
        e1, s1 = kf.step(log_y[mid], log_x[mid])
        kf.restore(saved)
        e2, s2 = kf.step(log_y[mid], log_x[mid])
        assert e1 == pytest.approx(e2, rel=1e-12)
        assert s1 == pytest.approx(s2, rel=1e-12)

    def test_snapshot_is_independent_copy(self):
        kf = KalmanFilterHedge()
        kf._theta = np.array([1.0, 2.0])
        saved = kf.snapshot()
        kf._theta[0] = 999.0
        assert saved.theta[0] == pytest.approx(1.0)

    def test_reset_clears_state(self, synthetic_pair):
        log_y, log_x, _, _ = synthetic_pair
        kf = KalmanFilterHedge()
        kf.run(log_y, log_x)
        kf.reset()
        np.testing.assert_array_equal(kf._theta, np.zeros(2))


# ── run_kalman convenience wrapper ────────────────────────────────────────────

class TestRunKalman:
    def test_preserves_index(self, price_series):
        py, px = price_series
        result = run_kalman(py, px, warmup_bars=0)
        pd.testing.assert_index_equal(result.index, py.index)

    def test_aligns_mismatched_index(self, price_series):
        py, px = price_series
        px_shorter = px.iloc[10:]
        result = run_kalman(py, px_shorter, warmup_bars=0)
        assert len(result) == len(px_shorter)

    def test_raises_on_nan(self, price_series):
        py, px = price_series
        py_nan = py.copy()
        py_nan.iloc[5] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            run_kalman(py_nan, px)

    def test_raises_on_non_positive_price(self, price_series):
        py, px = price_series
        py_bad = py.copy()
        py_bad.iloc[0] = 0.0
        with pytest.raises(ValueError, match="positive"):
            run_kalman(py_bad, px)

    def test_default_warmup_is_60(self, price_series):
        """run_kalman default warmup_bars=60 — first 60 rows must be NaN."""
        py, px = price_series
        result = run_kalman(py, px)
        assert result.iloc[:60].isnull().all().all()
        assert result.iloc[60:].notnull().all().all()

    def test_beta_close_to_ols_on_stationary_data(self, price_series):
        py, px = price_series
        result = run_kalman(py, px, delta=1e-4, warmup_bars=0)
        log_y = np.log(py.values)
        log_x = np.log(px.values)
        ols_beta = np.cov(log_y, log_x)[0, 1] / np.var(log_x)
        assert abs(result["beta"].iloc[-1] - ols_beta) < 0.3


# ── input validation ──────────────────────────────────────────────────────────

class TestInputValidation:
    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            KalmanFilterHedge().run(np.ones(10), np.ones(11))

    def test_2d_input_raises(self):
        with pytest.raises(ValueError):
            KalmanFilterHedge().run(np.ones((10, 1)), np.ones((10, 1)))

    def test_single_observation(self):
        result = KalmanFilterHedge().run(np.array([1.0]), np.array([1.0]),
                                         warmup_bars=0)
        assert result.shape == (1, 5)
        assert not result.isnull().any().any()


# ── warmup masking (fix #1) ───────────────────────────────────────────────────

class TestWarmup:
    def test_first_n_rows_are_nan(self, synthetic_pair):
        log_y, log_x, _, _ = synthetic_pair
        w = 60
        result = KalmanFilterHedge().run(log_y, log_x, warmup_bars=w)
        assert result.iloc[:w].isnull().all().all(), \
            f"Expected first {w} rows to be all NaN"

    def test_rows_after_warmup_are_not_nan(self, synthetic_pair):
        log_y, log_x, _, _ = synthetic_pair
        w = 60
        result = KalmanFilterHedge().run(log_y, log_x, warmup_bars=w)
        assert result.iloc[w:].notnull().all().all(), \
            "Rows after warmup window must not contain NaN"

    def test_warmup_zero_produces_no_nans(self, synthetic_pair):
        log_y, log_x, _, _ = synthetic_pair
        result = KalmanFilterHedge().run(log_y, log_x, warmup_bars=0)
        assert not result.isnull().any().any()

    def test_warmup_all_five_columns_nanned(self, synthetic_pair):
        log_y, log_x, _, _ = synthetic_pair
        result = KalmanFilterHedge().run(log_y, log_x, warmup_bars=50)
        for col in ["alpha", "beta", "spread", "spread_std", "zscore"]:
            assert result[col].iloc[:50].isnull().all(), \
                f"Column '{col}' should be NaN during warmup"

    def test_warmup_larger_than_series_nans_all(self, synthetic_pair):
        log_y, log_x, _, _ = synthetic_pair
        result = KalmanFilterHedge().run(log_y[:30], log_x[:30], warmup_bars=100)
        assert result.isnull().all().all()

    def test_warmup_does_not_alter_post_warmup_values(self, synthetic_pair):
        """Values after warmup must equal the no-warmup result."""
        log_y, log_x, _, _ = synthetic_pair
        w = 60
        result_no_w  = KalmanFilterHedge().run(log_y, log_x, warmup_bars=0)
        result_with_w = KalmanFilterHedge().run(log_y, log_x, warmup_bars=w)
        pd.testing.assert_frame_equal(
            result_no_w.iloc[w:].reset_index(drop=True),
            result_with_w.iloc[w:].reset_index(drop=True),
        )

    def test_signal_count_after_warmup_less_than_without(self, synthetic_pair):
        """Fewer signal bars are available after applying warmup masking."""
        log_y, log_x, _, _ = synthetic_pair
        result_no_w  = run_kalman(
            pd.Series(np.exp(log_y)), pd.Series(np.exp(log_x)),
            warmup_bars=0,
        )
        result_with_w = run_kalman(
            pd.Series(np.exp(log_y)), pd.Series(np.exp(log_x)),
            warmup_bars=60,
        )
        valid_no_w   = result_no_w["zscore"].notna().sum()
        valid_with_w = result_with_w["zscore"].notna().sum()
        assert valid_with_w == valid_no_w - 60


# ── delta tuning (fix #2) ─────────────────────────────────────────────────────

class TestDeltaTuning:
    def test_tune_delta_returns_float_in_grid(self, synthetic_pair):
        log_y, log_x, _, _ = synthetic_pair
        grid = [1e-6, 1e-5, 1e-4, 1e-3]
        kf = KalmanFilterHedge()
        best = kf.tune_delta(log_y, log_x, delta_grid=grid)
        assert best in grid

    def test_tune_delta_does_not_modify_self_delta(self, synthetic_pair):
        log_y, log_x, _, _ = synthetic_pair
        kf = KalmanFilterHedge(delta=1e-5)
        _ = kf.tune_delta(log_y, log_x)
        assert kf.delta == 1e-5

    def test_tuned_delta_improves_log_likelihood(self, synthetic_pair):
        """Log-likelihood under tuned delta must be >= log-likelihood under default."""
        log_y, log_x, _, _ = synthetic_pair
        kf = KalmanFilterHedge(delta=1e-5, vt=1e-3)
        best_delta = kf.tune_delta(log_y, log_x)

        ll_default = _innovation_log_likelihood(log_y, log_x, 1e-5, 1e-3)
        ll_tuned   = _innovation_log_likelihood(log_y, log_x, best_delta, 1e-3)
        assert ll_tuned >= ll_default, (
            f"Tuned delta {best_delta:.2e} ll={ll_tuned:.1f} "
            f"should be >= default ll={ll_default:.1f}"
        )

    def test_tune_delta_custom_grid(self, synthetic_pair):
        log_y, log_x, _, _ = synthetic_pair
        custom_grid = [1e-4, 5e-4, 1e-3]
        kf = KalmanFilterHedge()
        best = kf.tune_delta(log_y, log_x, delta_grid=custom_grid)
        assert best in custom_grid

    def test_run_kalman_auto_tune_runs_without_error(self, price_series):
        py, px = price_series
        result = run_kalman(py, px, auto_tune_delta=True, warmup_bars=60)
        assert isinstance(result, pd.DataFrame)
        assert result.iloc[60:].notnull().all().all()

    def test_auto_tuned_result_has_same_shape_as_non_tuned(self, price_series):
        py, px = price_series
        r_default = run_kalman(py, px, auto_tune_delta=False, warmup_bars=0)
        r_tuned   = run_kalman(py, px, auto_tune_delta=True,  warmup_bars=0)
        assert r_default.shape == r_tuned.shape

    def test_innovation_log_likelihood_helper(self, synthetic_pair):
        """_innovation_log_likelihood must return a finite scalar.

        Note: Gaussian log-density (not log-probability) can be positive when
        the density value exceeds 1, which happens for small innovations and
        small innovation variance. Only finiteness is required.
        """
        log_y, log_x, _, _ = synthetic_pair
        ll = _innovation_log_likelihood(log_y, log_x, delta=1e-5, vt=1e-3)
        assert np.isfinite(ll)
        assert isinstance(float(ll), float)

    def test_higher_delta_better_for_fast_moving_pair(self):
        """For a pair with regime-shifting beta, higher delta should win."""
        rng = np.random.default_rng(7)
        n = 600
        log_x = np.cumsum(rng.normal(0, 0.01, n))
        # Beta shifts from 1.0 to 2.0 at midpoint
        beta = np.concatenate([np.ones(n // 2) * 1.0, np.ones(n // 2) * 2.0])
        log_y = 0.1 + beta * log_x + rng.normal(0, 0.01, n)

        kf = KalmanFilterHedge()
        best = kf.tune_delta(log_y, log_x)
        # For a pair with structural break, faster delta should be preferred
        assert best > 1e-6, f"Expected faster delta for shifting pair, got {best:.2e}"


# ── on real data ──────────────────────────────────────────────────────────────

class TestOnRealData:
    @pytest.fixture
    def jpm_bac_prices(self):
        from pathlib import Path
        p = Path("data/ohlcv/1d")
        if not p.exists():
            pytest.skip("Daily data not available")
        jpm = pd.read_parquet(p / "JPM.parquet").set_index("timestamp")["close"]
        bac = pd.read_parquet(p / "BAC.parquet").set_index("timestamp")["close"]
        return jpm, bac

    def test_default_warmup_masks_first_60_rows(self, jpm_bac_prices):
        jpm, bac = jpm_bac_prices
        result = run_kalman(jpm, bac)   # default warmup_bars=60
        assert result.iloc[:60].isnull().all().all()
        assert result.iloc[60:].notnull().all().all()

    def test_real_pair_bounded_zscore_after_warmup(self, jpm_bac_prices):
        jpm, bac = jpm_bac_prices
        result = run_kalman(jpm, bac)
        zscore = result["zscore"].dropna()
        assert (zscore.abs() < 6).mean() > 0.95

    def test_real_beta_positive_and_reasonable(self, jpm_bac_prices):
        jpm, bac = jpm_bac_prices
        result = run_kalman(jpm, bac)
        beta_tail = result["beta"].dropna().iloc[-500:].median()
        assert 0.1 < beta_tail < 5.0

    def test_real_data_no_nans_after_warmup(self, jpm_bac_prices):
        jpm, bac = jpm_bac_prices
        result = run_kalman(jpm, bac)
        assert result.iloc[60:].notnull().all().all()

    def test_real_tune_delta_returns_valid_value(self, jpm_bac_prices):
        jpm, bac = jpm_bac_prices
        log_y = np.log(jpm.values.astype(float))
        log_x = np.log(bac.values.astype(float))
        kf = KalmanFilterHedge()
        best = kf.tune_delta(log_y, log_x)
        assert 1e-8 <= best <= 1e-1, f"Tuned delta {best:.2e} out of expected range"

    def test_tuned_delta_improves_ll_on_real_pair(self, jpm_bac_prices):
        jpm, bac = jpm_bac_prices
        log_y = np.log(jpm.values.astype(float))
        log_x = np.log(bac.values.astype(float))
        kf = KalmanFilterHedge(delta=1e-5, vt=1e-3)
        best_delta = kf.tune_delta(log_y, log_x)
        ll_default = _innovation_log_likelihood(log_y, log_x, 1e-5, 1e-3)
        ll_tuned   = _innovation_log_likelihood(log_y, log_x, best_delta, 1e-3)
        assert ll_tuned >= ll_default

    def test_real_auto_tune_run_kalman(self, jpm_bac_prices):
        jpm, bac = jpm_bac_prices
        result = run_kalman(jpm, bac, auto_tune_delta=True, warmup_bars=60)
        assert result.iloc[60:].notnull().all().all()
        assert (result["zscore"].dropna().abs() < 10).all()

    def test_zscore_entry_signals_exist_after_warmup(self, jpm_bac_prices):
        """After warmup, z-score must cross ±2 at least once in 15 years."""
        jpm, bac = jpm_bac_prices
        result = run_kalman(jpm, bac, auto_tune_delta=True)
        zscore = result["zscore"].dropna()
        n_signals = (zscore.abs() >= 2.0).sum()
        assert n_signals >= 1, (
            f"No z-score crossings of ±2.0 found in {len(zscore)} bars. "
            f"Max |z| = {zscore.abs().max():.3f}. "
            "Check delta tuning or entry threshold."
        )
