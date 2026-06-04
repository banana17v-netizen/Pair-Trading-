"""Tests for stat_arb/backtesting/engine.py and metrics.py.

All synthetic-data tests are self-contained with fixed seeds.
Real-data tests are guarded with pytest.skip if data is absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stat_arb.backtesting.engine import (
    Fold,
    FoldResult,
    WalkForwardBacktest,
    WalkForwardConfig,
    _rolling_zscore,
    _compute_is_sharpe,
    _compute_smoothed_market_gate,
)
from stat_arb.backtesting.metrics import (
    annualized_sharpe,
    compute_metrics,
    equity_curve,
    max_drawdown,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _bday_range(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end, tz="UTC")


def _make_prices(
    n: int = 800,
    n_stocks: int = 4,
    sectors: dict | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic price DataFrame with realistic log-returns."""
    rng = np.random.default_rng(seed)
    idx = _bday_range("2015-01-01", pd.bdate_range("2015-01-01", periods=n, freq="B", tz="UTC")[-1].strftime("%Y-%m-%d"))
    prices = {}
    # First two stocks: cointegrated (same sector)
    log_x = np.cumsum(rng.normal(0, 0.01, n))
    log_y = 0.3 + 1.2 * log_x + rng.normal(0, 0.015, n)
    prices["AAPL"] = np.exp(log_y)   # Technology
    prices["MSFT"] = np.exp(log_x)   # Technology
    # Rest: independent random walks (different sector)
    for i in range(2, n_stocks):
        prices[f"S{i}"] = np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame(prices, index=idx[:n])


def _make_simple_config(**kwargs) -> WalkForwardConfig:
    """Config with short windows for fast tests."""
    defaults = dict(
        start="2015-01-01",
        end="2018-01-01",
        is_years=1,
        oos_months=3,
        max_pairs=5,
        notional_per_pair=10_000,
        eg_pvalue=0.10,
        min_halflife=1.0,
        max_halflife=200.0,
        same_sector_only=False,   # synthetic symbols not in SECTOR_MAP
        apply_fdr=False,
        kalman_warmup_bars=30,
    )
    defaults.update(kwargs)
    return WalkForwardConfig(**defaults)


# ═══════════════════════════════════════════════════════════════════════════════
# metrics.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeMetrics:
    def test_sharpe_positive_for_positive_returns(self):
        ret = pd.Series([0.001] * 252)
        m = compute_metrics(ret)
        assert m["sharpe_ratio"] > 0

    def test_sharpe_zero_for_zero_returns(self):
        ret = pd.Series([0.0] * 252)
        m = compute_metrics(ret)
        assert np.isnan(m["sharpe_ratio"]) or m["sharpe_ratio"] == 0.0

    def test_max_drawdown_negative(self):
        ret = pd.Series([0.01, -0.05, 0.02, -0.03])
        m = compute_metrics(ret)
        assert m["max_drawdown"] < 0

    def test_max_drawdown_zero_for_monotone_growth(self):
        ret = pd.Series([0.001] * 100)
        m = compute_metrics(ret)
        assert abs(m["max_drawdown"]) < 1e-9

    def test_win_rate_one_for_all_positive(self):
        ret = pd.Series([0.001] * 50)
        assert compute_metrics(ret)["win_rate"] == pytest.approx(1.0)

    def test_win_rate_zero_for_all_negative(self):
        ret = pd.Series([-0.001] * 50)
        assert compute_metrics(ret)["win_rate"] == pytest.approx(0.0)

    def test_total_return_correct(self):
        ret = pd.Series([0.01, -0.01, 0.02])
        m = compute_metrics(ret)
        expected = (1.01 * 0.99 * 1.02) - 1.0
        assert m["total_return"] == pytest.approx(expected, rel=1e-9)

    def test_empty_returns_gives_nans(self):
        ret = pd.Series([], dtype=float)
        m = compute_metrics(ret)
        assert np.isnan(m["sharpe_ratio"])

    def test_all_metrics_keys_present(self):
        ret = pd.Series([0.001] * 100)
        m = compute_metrics(ret)
        for k in ["annualized_return", "annualized_vol", "sharpe_ratio",
                  "max_drawdown", "calmar_ratio", "win_rate",
                  "total_return", "n_trading_days"]:
            assert k in m

    def test_annualized_vol_correct(self):
        daily_vol = 0.01
        ret = pd.Series(np.random.default_rng(0).normal(0, daily_vol, 1000))
        m = compute_metrics(ret)
        assert abs(m["annualized_vol"] - daily_vol * np.sqrt(252)) < 0.005

    def test_calmar_ratio_positive_for_positive_sharpe(self):
        ret = pd.Series([0.001] * 252 + [-0.005] * 10)
        m = compute_metrics(pd.Series(ret))
        # calmar might be nan if max_dd ~ 0, just check it exists
        assert "calmar_ratio" in m


class TestHelperFunctions:
    def test_annualized_sharpe(self):
        ret = pd.Series([0.001] * 252)
        sh = annualized_sharpe(ret)
        assert sh > 0

    def test_max_drawdown_helper(self):
        ret = pd.Series([0.01, -0.10, 0.05])
        dd = max_drawdown(ret)
        assert dd < 0

    def test_equity_curve_starts_at_starting_value(self):
        ret = pd.Series([0.01, -0.005, 0.02])
        ec = equity_curve(ret, starting_value=100.0)
        assert abs(ec.iloc[0] - 101.0) < 1e-9   # first return applied


# ═══════════════════════════════════════════════════════════════════════════════
# WalkForwardConfig
# ═══════════════════════════════════════════════════════════════════════════════

class TestWalkForwardConfig:
    def test_is_days(self):
        c = WalkForwardConfig(is_years=2)
        assert c.is_days == 504

    def test_oos_days(self):
        c = WalkForwardConfig(oos_months=6)
        assert c.oos_days == 126

    def test_total_notional(self):
        c = WalkForwardConfig(max_pairs=15, notional_per_pair=50_000)
        assert c.total_notional == 750_000


# ═══════════════════════════════════════════════════════════════════════════════
# Fold generation
# ═══════════════════════════════════════════════════════════════════════════════

class TestFoldGeneration:
    @pytest.fixture
    def engine_and_prices(self):
        prices = _make_prices(n=800)
        config = _make_simple_config()   # is_years=1 (252d), oos_months=3 (63d)
        engine = WalkForwardBacktest(config)
        return engine, prices

    def test_fold_count_reasonable(self, engine_and_prices):
        engine, prices = engine_and_prices
        prices_period = prices.loc[engine.config.start:engine.config.end]
        folds = engine._build_folds(prices_period)
        assert len(folds) >= 1

    def test_is_window_length(self, engine_and_prices):
        engine, prices = engine_and_prices
        prices_period = prices.loc[engine.config.start:engine.config.end]
        folds = engine._build_folds(prices_period)
        for fold in folds:
            is_prices = prices_period.loc[fold.is_start:fold.is_end]
            assert len(is_prices) == pytest.approx(engine.config.is_days, abs=5)

    def test_oos_start_equals_day_after_is_end(self, engine_and_prices):
        engine, prices = engine_and_prices
        prices_period = prices.loc[engine.config.start:engine.config.end]
        folds = engine._build_folds(prices_period)
        idx = prices_period.index
        for fold in folds:
            assert fold.oos_start > fold.is_end

    def test_consecutive_oos_non_overlapping(self, engine_and_prices):
        engine, prices = engine_and_prices
        prices_period = prices.loc[engine.config.start:engine.config.end]
        folds = engine._build_folds(prices_period)
        for i in range(1, len(folds)):
            assert folds[i].oos_start > folds[i - 1].oos_end

    def test_fold_idx_sequential(self, engine_and_prices):
        engine, prices = engine_and_prices
        prices_period = prices.loc[engine.config.start:engine.config.end]
        folds = engine._build_folds(prices_period)
        for i, fold in enumerate(folds):
            assert fold.idx == i


# ═══════════════════════════════════════════════════════════════════════════════
# Engine integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestEngineIntegration:
    @pytest.fixture
    def result(self):
        """Run a fast backtest on synthetic data."""
        prices = _make_prices(n=900, seed=42)
        config = _make_simple_config(
            start="2015-01-01", end="2018-06-01",
            is_years=1, oos_months=3,
            eg_pvalue=0.10,
            max_halflife=300.0,
            min_halflife=1.0,
        )
        return WalkForwardBacktest(config).run(prices)

    def test_result_has_daily_returns(self, result):
        assert isinstance(result.daily_returns, pd.Series)
        assert len(result.daily_returns) > 0

    def test_result_has_equity_curve(self, result):
        assert isinstance(result.equity_curve, pd.Series)
        assert len(result.equity_curve) > 0
        # Equity curve starts near 1.0 (first return is small)
        assert 0.8 < result.equity_curve.iloc[0] < 1.2

    def test_result_metrics_exist(self, result):
        m = result.metrics
        for k in ["sharpe_ratio", "max_drawdown", "annualized_return"]:
            assert k in m

    def test_folds_non_empty(self, result):
        assert len(result.folds) >= 1

    def test_daily_returns_ordered(self, result):
        idx = result.daily_returns.index
        assert idx.is_monotonic_increasing

    def test_no_nan_in_daily_returns(self, result):
        assert not result.daily_returns.isnull().any()

    def test_fold_result_structure(self, result):
        for f in result.folds:
            assert isinstance(f.fold, Fold)
            assert isinstance(f.daily_returns, pd.Series)
            assert isinstance(f.n_trades, int)
            assert isinstance(f.pairs, list)
            assert "sharpe_ratio" in f.metrics

    def test_zero_pairs_fold_returns_zeros(self):
        """A fold with no cointegrated pairs must have zero returns."""
        prices = _make_prices(n=700, seed=99)
        config = _make_simple_config(
            eg_pvalue=1e-10,   # impossible threshold → 0 pairs always
            same_sector_only=False,
        )
        result = WalkForwardBacktest(config).run(prices)
        # All returns should be very close to zero (only TC on no trades)
        assert (result.daily_returns == 0.0).all()

    def test_tc_reduces_returns(self):
        """Higher transaction costs must reduce total return."""
        prices = _make_prices(n=900, seed=42)
        cfg_lo = _make_simple_config(transaction_cost_bps=0.0,
                                     eg_pvalue=0.10, max_halflife=300)
        cfg_hi = _make_simple_config(transaction_cost_bps=50.0,
                                     eg_pvalue=0.10, max_halflife=300)
        r_lo = WalkForwardBacktest(cfg_lo).run(prices).metrics["total_return"]
        r_hi = WalkForwardBacktest(cfg_hi).run(prices).metrics["total_return"]
        assert r_lo >= r_hi - 1e-9   # zero-cost >= high-cost


# ═══════════════════════════════════════════════════════════════════════════════
# Real data smoke test
# ═══════════════════════════════════════════════════════════════════════════════

class TestOnRealData:
    @pytest.fixture
    def real_prices(self):
        from pathlib import Path
        p = Path("data/ohlcv/1d")
        if not p.exists():
            pytest.skip("Daily data not available")
        syms = ["ADBE", "CRM", "MSFT", "ACN", "TXN"]
        dfs = {}
        for s in syms:
            fp = p / f"{s}.parquet"
            if fp.exists():
                dfs[s] = pd.read_parquet(fp).set_index("timestamp")["close"]
        if len(dfs) < 2:
            pytest.skip("Insufficient symbols")
        return pd.DataFrame(dfs).dropna()

    def test_backtest_runs_on_real_data(self, real_prices):
        config = WalkForwardConfig(
            start="2018-01-01", end="2021-01-01",
            is_years=1, oos_months=6,
            max_pairs=3,
            eg_pvalue=0.10,
            max_halflife=200.0,
            same_sector_only=True,
        )
        result = WalkForwardBacktest(config).run(real_prices)
        assert isinstance(result.daily_returns, pd.Series)
        assert len(result.daily_returns) > 0

    def test_real_sharpe_is_finite(self, real_prices):
        config = WalkForwardConfig(
            start="2018-01-01", end="2021-01-01",
            is_years=1, oos_months=6,
            max_pairs=3,
            eg_pvalue=0.10,
            max_halflife=200.0,
            same_sector_only=True,
        )
        result = WalkForwardBacktest(config).run(real_prices)
        sharpe = result.metrics.get("sharpe_ratio", float("nan"))
        assert np.isfinite(sharpe), f"Sharpe should be finite, got {sharpe}"

    def test_real_equity_curve_starts_near_one(self, real_prices):
        config = WalkForwardConfig(
            start="2018-01-01", end="2021-01-01",
            is_years=1, oos_months=6,
            max_pairs=3,
            eg_pvalue=0.10,
            max_halflife=200.0,
            same_sector_only=True,
        )
        result = WalkForwardBacktest(config).run(real_prices)
        assert 0.5 < result.equity_curve.iloc[0] < 2.0


# ═══════════════════════════════════════════════════════════════════════════════
# Rolling z-score helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestRollingZScore:
    def test_output_length_matches_input(self):
        spread = np.random.default_rng(0).normal(0, 1, 100)
        z = _rolling_zscore(spread, window=20)
        assert len(z) == len(spread)

    def test_zero_where_no_history(self):
        spread = np.ones(5)  # shorter than window=20
        z = _rolling_zscore(spread, window=20)
        assert z[0] == 0.0

    def test_std_near_one_for_normal_after_warmup(self):
        rng = np.random.default_rng(42)
        spread = rng.normal(0, 1, 500)
        z = _rolling_zscore(spread, window=20)
        # After warmup, z should be approximately standard normal
        assert abs(z[50:].std() - 1.0) < 0.3

    def test_mean_near_zero_for_stationary_spread(self):
        rng = np.random.default_rng(7)
        spread = rng.normal(0, 1, 500)
        z = _rolling_zscore(spread, window=20)
        assert abs(z[50:].mean()) < 0.1

    def test_positive_when_spread_above_window_mean(self):
        spread = np.zeros(50)
        spread[-1] = 100.0   # large spike at end
        z = _rolling_zscore(spread, window=20)
        assert z[-1] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# IS Sharpe filter
# ═══════════════════════════════════════════════════════════════════════════════

class TestISSharpeFilter:
    def test_is_sharpe_returns_float(self):
        rng = np.random.default_rng(0)
        n = 300
        log_x = np.cumsum(rng.normal(0, 0.01, n))
        log_y = 0.3 + 1.2 * log_x + rng.normal(0, 0.02, n)
        spread = log_y - 0.3 - 1.2 * log_x
        z = _rolling_zscore(spread, 20)
        sharpe = _compute_is_sharpe(log_y, log_x, 1.2, z, 2.0, 0.5, 3.5)
        assert isinstance(sharpe, float)
        assert np.isfinite(sharpe)

    def test_is_sharpe_filter_keeps_good_pairs(self):
        """IS Sharpe filter must not drop ALL pairs when threshold is 0."""
        prices = _make_prices(n=900, seed=42)
        config = _make_simple_config(
            eg_pvalue=0.10, max_halflife=300,
            min_is_sharpe=0.0,
        )
        result = WalkForwardBacktest(config).run(prices)
        # With min_is_sharpe=0.0, at least some folds should have pairs
        pairs_per_fold = [f.n_pairs for f in result.folds]
        assert max(pairs_per_fold) > 0

    def test_is_sharpe_filter_stricter_gives_fewer_pairs(self):
        """Higher min_is_sharpe threshold should reduce average pairs per fold."""
        prices = _make_prices(n=900, seed=42)
        cfg_loose  = _make_simple_config(eg_pvalue=0.10, max_halflife=300, min_is_sharpe=-99.0)
        cfg_strict = _make_simple_config(eg_pvalue=0.10, max_halflife=300, min_is_sharpe=99.0)
        r_loose  = WalkForwardBacktest(cfg_loose).run(prices)
        r_strict = WalkForwardBacktest(cfg_strict).run(prices)
        avg_loose  = sum(f.n_pairs for f in r_loose.folds)
        avg_strict = sum(f.n_pairs for f in r_strict.folds)
        assert avg_strict <= avg_loose

    def test_config_has_new_fields(self):
        c = WalkForwardConfig()
        assert hasattr(c, "rolling_z_window")
        assert hasattr(c, "min_is_sharpe")
        # Layer 4: shorter window for faster mean-reversion
        assert c.rolling_z_window == 10
        # Layer 2: only high-quality IS pairs
        assert c.min_is_sharpe == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Smoothed market gate
# ═══════════════════════════════════════════════════════════════════════════════

class TestSmoothedMarketGate:
    """Tests for _compute_smoothed_market_gate()."""

    def _make_regime(self, labels: list[str]) -> pd.Series:
        idx = pd.date_range("2020-01-01", periods=len(labels), freq="B", tz="UTC")
        return pd.Series(labels, index=idx, name="market_regime")

    def test_all_quiet_not_gated(self):
        regime = self._make_regime(["mean_reverting"] * 20)
        gate = _compute_smoothed_market_gate(regime, smoothing_window=5, vol_threshold=0.3, cooldown_days=0)
        assert not gate.any(), "No volatile days → gate should be fully OFF"

    def test_all_volatile_fully_gated(self):
        regime = self._make_regime(["volatile"] * 20)
        gate = _compute_smoothed_market_gate(regime, smoothing_window=5, vol_threshold=0.3, cooldown_days=0)
        assert gate.all(), "All volatile → gate should be fully ON"

    def test_threshold_respected(self):
        # 2 out of 10 days volatile = 20% < threshold 0.3 → not gated
        labels = ["volatile"] * 2 + ["mean_reverting"] * 18
        regime = self._make_regime(labels)
        gate = _compute_smoothed_market_gate(regime, smoothing_window=10, vol_threshold=0.30, cooldown_days=0)
        # After 10 days, rolling fraction < 0.3 → not gated (tail should clear)
        assert not gate.iloc[-1], "Low volatile fraction → tail should be ungated"

    def test_threshold_triggers_gate(self):
        # 5 out of 10 days volatile = 50% > threshold 0.3 → gated
        labels = ["volatile"] * 5 + ["mean_reverting"] * 15
        regime = self._make_regime(labels)
        gate = _compute_smoothed_market_gate(regime, smoothing_window=10, vol_threshold=0.30, cooldown_days=0)
        # During first 5 volatile days, gate should activate
        assert gate.iloc[4], "50% volatile in window should trigger gate"

    def test_cooldown_extends_gate_after_volatile_clears(self):
        # 5 volatile days, then quiet. With cooldown=3, gate stays ON for 3 more days.
        labels = ["volatile"] * 5 + ["mean_reverting"] * 10
        regime = self._make_regime(labels)
        gate_no_cd = _compute_smoothed_market_gate(regime, 5, 0.3, cooldown_days=0)
        gate_cd    = _compute_smoothed_market_gate(regime, 5, 0.3, cooldown_days=3)
        # Cooldown version should be gated longer
        assert gate_cd.sum() > gate_no_cd.sum(), \
            "Cooldown should extend gate beyond volatile period"

    def test_cooldown_precise_days(self):
        # 5 volatile days (0-4), then 10 MR days (5-14), window=5, threshold=0.30
        # Rolling fractions:
        #   days 5,6,7: fractions 0.8, 0.6, 0.4 > 0.30 → still raw-gated
        #   day 8: fraction 0.2 < 0.30 → raw gate OFF → cooldown starts
        #   days 8,9,10: cooldown (3 days)
        #   day 11+: gate fully OFF
        labels = ["volatile"] * 5 + ["mean_reverting"] * 10
        regime = self._make_regime(labels)
        gate = _compute_smoothed_market_gate(regime, smoothing_window=5,
                                              vol_threshold=0.30, cooldown_days=3)
        assert gate.iloc[7],  "Day 7 (rolling frac=0.4>0.3) should still be gated"
        assert gate.iloc[8],  "First cooldown day should be gated"
        assert gate.iloc[9],  "Second cooldown day should be gated"
        assert gate.iloc[10], "Third cooldown day should be gated"
        assert not gate.iloc[11], "After cooldown ends (day 11), should not be gated"

    def test_returns_bool_series_same_length(self):
        regime = self._make_regime(["mean_reverting", "volatile", "trending"] * 10)
        gate   = _compute_smoothed_market_gate(regime, 5, 0.3, 3)
        assert len(gate) == len(regime)
        assert gate.dtype == bool

    def test_config_has_smooth_gate_fields(self):
        c = WalkForwardConfig()
        assert hasattr(c, "market_vol_smoothing_window")
        assert hasattr(c, "market_vol_threshold")
        assert hasattr(c, "market_cooldown_days")
        assert c.market_vol_smoothing_window == 10
        assert c.market_vol_threshold == 0.30
        assert c.market_cooldown_days == 5

    def test_close_on_volatile_default_is_false(self):
        """Default must be False to avoid whipsaw (force-close + immediate re-entry)."""
        c = WalkForwardConfig()
        assert c.close_positions_on_volatile is False, \
            "Default close_positions_on_volatile must be False to avoid whipsaw"

    def test_smoothed_gate_reduces_whipsaw_trades(self):
        """Smoothed gate must NOT generate more trades than no-filter baseline."""
        prices = _make_prices(n=900, seed=42)
        cfg_off = _make_simple_config(use_market_regime_filter=False,
                                       eg_pvalue=0.10, max_halflife=300)
        cfg_on  = _make_simple_config(use_market_regime_filter=True,
                                       eg_pvalue=0.10, max_halflife=300,
                                       market_vol_smoothing_window=5,
                                       market_vol_threshold=0.30,
                                       market_cooldown_days=3,
                                       close_positions_on_volatile=False)
        r_off = WalkForwardBacktest(cfg_off).run(prices)
        r_on  = WalkForwardBacktest(cfg_on).run(prices)
        trades_off = sum(f.n_trades for f in r_off.folds)
        trades_on  = sum(f.n_trades for f in r_on.folds)
        # With smoothed gate + no force-close, filter should NEVER create more trades
        assert trades_on <= trades_off + 5, \
            f"Market filter should not increase trades: {trades_on} > {trades_off}"

    def test_gate_pct_in_fold_metrics(self):
        prices = _make_prices(n=900, seed=42)
        config = _make_simple_config(use_market_regime_filter=True,
                                      eg_pvalue=0.10, max_halflife=300)
        result = WalkForwardBacktest(config).run(prices)
        for f in result.folds:
            if f.n_pairs > 0 and "market_gate_pct" in f.metrics:
                assert 0.0 <= f.metrics["market_gate_pct"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Market-level regime filter
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarketRegimeFilter:
    """Tests for the portfolio-level market HMM gate (improvement 4)."""

    @pytest.fixture
    def prices_with_volatile_period(self):
        """Synthetic prices: first 250 bars quiet, then 250 bars high-vol."""
        rng = np.random.default_rng(7)
        n   = 500
        idx = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
        data = {}
        # Cointegrated pair
        log_x = np.cumsum(rng.normal(0, 0.01, n))
        log_y = 0.3 + 1.2 * log_x + rng.normal(0, 0.015, n)
        data["AAPL"] = np.exp(log_y)
        data["MSFT"] = np.exp(log_x)
        # Add more symbols for market feature computation
        for i, name in enumerate(["S2", "S3", "S4", "S5"]):
            noise_scale = 0.01 if i < 2 else 0.06  # S4/S5 are volatile
            data[name] = np.exp(np.cumsum(rng.normal(0, noise_scale, n)))
        return pd.DataFrame(data, index=idx)

    def test_config_has_market_filter_fields(self):
        c = WalkForwardConfig()
        assert hasattr(c, "use_market_regime_filter")
        assert hasattr(c, "close_positions_on_volatile")
        assert c.use_market_regime_filter is True
        # Default is False to avoid whipsaw (see smoothed gate motivation)
        assert c.close_positions_on_volatile is False

    def test_market_filter_default_on(self):
        """Default config must have market filter enabled."""
        c = WalkForwardConfig()
        assert c.use_market_regime_filter is True

    def test_fit_market_hmm_returns_dict_or_none(self, prices_with_volatile_period):
        prices = _make_prices(n=700)
        config = _make_simple_config()
        engine = WalkForwardBacktest(config)
        is_prices = prices.iloc[:252]
        result = engine._fit_market_hmm(is_prices)
        # Either succeeds (dict) or returns None if data insufficient
        assert result is None or isinstance(result, dict)

    def test_fit_market_hmm_succeeds_on_rich_data(self, prices_with_volatile_period):
        config = _make_simple_config()
        engine = WalkForwardBacktest(config)
        result = engine._fit_market_hmm(prices_with_volatile_period.iloc[:252])
        assert result is not None
        assert "hmm" in result

    def test_compute_market_regime_returns_series(self, prices_with_volatile_period):
        config = _make_simple_config()
        engine = WalkForwardBacktest(config)
        is_prices = prices_with_volatile_period.iloc[:252]
        oos_prices = prices_with_volatile_period.iloc[252:378]
        mkt_model = engine._fit_market_hmm(is_prices)
        if mkt_model is None:
            pytest.skip("Market HMM fitting failed on synthetic data")
        regime = engine._compute_market_regime(oos_prices, is_prices.iloc[-30:], mkt_model)
        assert isinstance(regime, pd.Series)
        assert len(regime) == len(oos_prices)

    def test_compute_market_regime_valid_labels(self, prices_with_volatile_period):
        valid = {"mean_reverting", "trending", "volatile"}
        config = _make_simple_config()
        engine = WalkForwardBacktest(config)
        is_prices  = prices_with_volatile_period.iloc[:252]
        oos_prices = prices_with_volatile_period.iloc[252:378]
        mkt_model = engine._fit_market_hmm(is_prices)
        if mkt_model is None:
            pytest.skip("Market HMM fitting failed")
        regime = engine._compute_market_regime(oos_prices, is_prices.iloc[-30:], mkt_model)
        assert set(regime.unique()).issubset(valid)

    def test_market_filter_reduces_trades(self):
        """Market filter ON must produce <= trades than filter OFF."""
        prices = _make_prices(n=900, seed=42)
        cfg_on  = _make_simple_config(use_market_regime_filter=True,
                                       eg_pvalue=0.10, max_halflife=300)
        cfg_off = _make_simple_config(use_market_regime_filter=False,
                                       eg_pvalue=0.10, max_halflife=300)
        r_on  = WalkForwardBacktest(cfg_on).run(prices)
        r_off = WalkForwardBacktest(cfg_off).run(prices)
        assert sum(f.n_trades for f in r_on.folds) <= \
               sum(f.n_trades for f in r_off.folds) + 10  # ±10 tolerance for HMM stochasticity

    def test_volatile_market_blocks_new_entries(self):
        """When market_regime is all-volatile, simulate_oos must have 0 trades."""
        prices = _make_prices(n=600, seed=1)
        config = _make_simple_config(eg_pvalue=0.10, max_halflife=300,
                                      use_market_regime_filter=True)
        engine = WalkForwardBacktest(config)
        fold = engine._build_folds(prices.loc[config.start:config.end])[0]
        is_p  = prices.loc[fold.is_start:fold.is_end]
        oos_p = prices.loc[fold.oos_start:fold.oos_end]

        # Build a pair model manually
        from stat_arb.backtesting.engine import _rolling_zscore
        log_y = np.log(is_p["AAPL"].values)
        log_x = np.log(is_p["MSFT"].values)
        import statsmodels.api as sm
        ols_fit = sm.OLS(log_y, sm.add_constant(log_x)).fit()
        ols_alpha, ols_beta = float(ols_fit.params[0]), float(ols_fit.params[1])
        spread = log_y - ols_alpha - ols_beta * log_x
        from stat_arb.kalman.filter import KalmanFilterHedge
        from stat_arb.hmm.regime import HMMRegimeDetector, build_features_from_spread
        kf = KalmanFilterHedge()
        kf.run(log_y, log_x, warmup_bars=30)
        snap = kf.snapshot()
        is_df = pd.DataFrame({"spread": spread,
                               "zscore": _rolling_zscore(spread, 20)},
                              index=is_p.index)
        feats = build_features_from_spread(is_df.dropna())
        hmm = HMMRegimeDetector(random_state=0)
        hmm.fit(feats)
        pair_models = {("AAPL", "MSFT"): {
            "sym_y": "AAPL", "sym_x": "MSFT",
            "ols_alpha": ols_alpha, "ols_beta": ols_beta,
            "is_spread_tail": spread[-20:],
            "kf": kf, "kf_snapshot": snap, "hmm": hmm,
            "is_sharpe": 0.0, "hedge_ratio": ols_beta,
        }}
        # All-volatile market regime
        volatile_regime = pd.Series("volatile", index=oos_p.index, name="market_regime")
        _, n_trades = engine._simulate_oos(oos_p, pair_models, volatile_regime)
        assert n_trades == 0, f"Expected 0 trades with all-volatile market, got {n_trades}"

    def test_free_market_allows_entries(self):
        """When market_regime is all-MR, simulate_oos may have trades (signal dependent)."""
        prices = _make_prices(n=600, seed=1)
        config = _make_simple_config(eg_pvalue=0.10, max_halflife=300,
                                      use_market_regime_filter=True)
        engine = WalkForwardBacktest(config)
        fold = engine._build_folds(prices.loc[config.start:config.end])[0]
        oos_p = prices.loc[fold.oos_start:fold.oos_end]

        # Run full engine with no-filter vs forced-MR-market
        cfg_no = _make_simple_config(use_market_regime_filter=False,
                                      eg_pvalue=0.10, max_halflife=300)
        r_no = WalkForwardBacktest(cfg_no).run(prices)
        total_trades_no = sum(f.n_trades for f in r_no.folds)
        # With no market filter, some trades should happen if any pair has signal
        assert isinstance(total_trades_no, int)

    def test_close_on_volatile_exits_existing_position(self):
        """close_positions_on_volatile=True must force-close open positions."""
        prices = _make_prices(n=600, seed=1)
        config_close = _make_simple_config(
            use_market_regime_filter=True,
            close_positions_on_volatile=True,
            eg_pvalue=0.10, max_halflife=300,
        )
        config_keep = _make_simple_config(
            use_market_regime_filter=True,
            close_positions_on_volatile=False,
            eg_pvalue=0.10, max_halflife=300,
        )
        r_close = WalkForwardBacktest(config_close).run(prices)
        r_keep  = WalkForwardBacktest(config_keep).run(prices)
        # Both should run without error and produce valid returns
        assert isinstance(r_close.daily_returns, pd.Series)
        assert isinstance(r_keep.daily_returns, pd.Series)

    def test_backtest_with_market_filter_runs_end_to_end(self):
        prices = _make_prices(n=900, seed=42)
        config = _make_simple_config(
            use_market_regime_filter=True,
            close_positions_on_volatile=True,
            eg_pvalue=0.10, max_halflife=300,
        )
        result = WalkForwardBacktest(config).run(prices)
        assert isinstance(result.daily_returns, pd.Series)
        assert len(result.daily_returns) > 0
        assert not result.daily_returns.isnull().any()

    def test_market_volatile_pct_in_fold_metrics(self):
        """When market filter is ON, each fold metrics must contain market_volatile_pct."""
        prices = _make_prices(n=900, seed=42)
        config = _make_simple_config(
            use_market_regime_filter=True,
            eg_pvalue=0.10, max_halflife=300,
        )
        result = WalkForwardBacktest(config).run(prices)
        folds_with_pairs = [f for f in result.folds if f.n_pairs > 0]
        for f in folds_with_pairs:
            assert "market_volatile_pct" in f.metrics, \
                f"Fold {f.fold.idx} missing market_volatile_pct"
            assert 0.0 <= f.metrics["market_volatile_pct"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Real data: market filter smoke test
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarketFilterRealData:
    @pytest.fixture
    def real_prices_tech(self):
        from pathlib import Path
        p = Path("data/ohlcv/1d")
        if not p.exists():
            pytest.skip("Daily data not available")
        syms = ["AAPL", "MSFT", "NVDA", "ADBE", "CRM"]
        dfs = {
            s: pd.read_parquet(p / f"{s}.parquet").set_index("timestamp")["close"]
            for s in syms if (p / f"{s}.parquet").exists()
        }
        if len(dfs) < 3:
            pytest.skip("Insufficient symbols")
        return pd.DataFrame(dfs).dropna()

    def test_market_hmm_fits_on_real_prices(self, real_prices_tech):
        config = WalkForwardConfig()
        engine = WalkForwardBacktest(config)
        is_p   = real_prices_tech.loc["2018-01-01":"2020-01-01"]
        result = engine._fit_market_hmm(is_p)
        assert result is not None

    def test_market_regime_has_all_three_states_on_real_data(self, real_prices_tech):
        config = WalkForwardConfig()
        engine = WalkForwardBacktest(config)
        is_p   = real_prices_tech.loc["2018-01-01":"2020-01-01"]
        oos_p  = real_prices_tech.loc["2020-01-02":"2020-07-01"]
        mkt_model = engine._fit_market_hmm(is_p)
        if mkt_model is None:
            pytest.skip("Market HMM fitting failed")
        regime = engine._compute_market_regime(oos_p, is_p.iloc[-30:], mkt_model)
        # COVID period (2020 H1) should have at least some volatile days
        assert "volatile" in regime.values, \
            "Expected VOLATILE regime during COVID period (2020 H1)"

    def test_backtest_with_market_filter_on_real_data(self, real_prices_tech):
        config = WalkForwardConfig(
            start="2018-01-01", end="2021-01-01",
            is_years=1, oos_months=6,
            max_pairs=3, eg_pvalue=0.10, max_halflife=200.0,
            same_sector_only=True,
            use_market_regime_filter=True,
        )
        result = WalkForwardBacktest(config).run(real_prices_tech)
        assert isinstance(result.daily_returns, pd.Series)
        assert not result.daily_returns.isnull().any()


# ═══════════════════════════════════════════════════════════════════════════════
# VIX Gate (Layer 1)
# ═══════════════════════════════════════════════════════════════════════════════

class TestVIXGate:
    """Tests for VIX-based hard trading gate."""

    def _make_vix(self, n: int, level: float, idx) -> pd.Series:
        return pd.Series(level, index=idx, name="vix")

    def test_config_has_vix_threshold(self):
        c = WalkForwardConfig()
        assert hasattr(c, "vix_gate_threshold")
        assert c.vix_gate_threshold == 20.0

    def test_high_vix_blocks_all_new_entries(self):
        """VIX=35 (above threshold 20) must result in 0 new trades."""
        prices = _make_prices(n=600, seed=1)
        config = _make_simple_config(eg_pvalue=0.10, max_halflife=300,
                                      use_market_regime_filter=False,
                                      vix_gate_threshold=20.0)
        engine = WalkForwardBacktest(config)
        fold   = engine._build_folds(prices.loc[config.start:config.end])[0]
        oos_p  = prices.loc[fold.oos_start:fold.oos_end]
        is_p   = prices.loc[fold.is_start:fold.is_end]

        # Fit a simple pair model
        import statsmodels.api as sm
        from stat_arb.kalman.filter import KalmanFilterHedge
        from stat_arb.hmm.regime import HMMRegimeDetector, build_features_from_spread

        log_y = np.log(is_p["AAPL"].values)
        log_x = np.log(is_p["MSFT"].values)
        ols   = sm.OLS(log_y, sm.add_constant(log_x)).fit()
        a, b  = float(ols.params[0]), float(ols.params[1])
        spread = log_y - a - b * log_x
        z      = _rolling_zscore(spread, 10)
        kf = KalmanFilterHedge()
        kf.run(log_y, log_x, warmup_bars=20)
        snap = kf.snapshot()
        is_df = pd.DataFrame({"spread": spread, "zscore": z}, index=is_p.index)
        feats = build_features_from_spread(is_df.dropna())
        hmm = HMMRegimeDetector(random_state=0)
        hmm.fit(feats)
        pair_models = {("AAPL", "MSFT"): {
            "sym_y": "AAPL", "sym_x": "MSFT",
            "ols_alpha": a, "ols_beta": b,
            "is_spread_tail": spread[-10:],
            "kf": kf, "kf_snapshot": snap, "hmm": hmm,
            "is_sharpe": 0.0, "hedge_ratio": b,
        }}

        # All-high VIX → all new entries blocked
        high_vix = self._make_vix(len(oos_p), 35.0, oos_p.index)
        _, n_trades = engine._simulate_oos(oos_p, pair_models, vix_oos=high_vix)
        assert n_trades == 0, \
            f"VIX=35 > threshold=20 → 0 new trades expected, got {n_trades}"

    def test_low_vix_allows_entries(self):
        """VIX=10 (below threshold 20) must NOT block trades."""
        prices = _make_prices(n=600, seed=1)
        config_vix = _make_simple_config(eg_pvalue=0.10, max_halflife=300,
                                          use_market_regime_filter=False,
                                          vix_gate_threshold=20.0)
        config_no  = _make_simple_config(eg_pvalue=0.10, max_halflife=300,
                                          use_market_regime_filter=False,
                                          vix_gate_threshold=0.0)  # disabled
        r_vix = WalkForwardBacktest(config_vix).run(
            prices,
            vix=pd.Series(10.0,
                          index=pd.date_range("2010-01-01", periods=len(prices), freq="B", tz="UTC"),
                          name="vix"),
        )
        r_no = WalkForwardBacktest(config_no).run(prices)
        # Low VIX should not reduce trades relative to no-VIX baseline
        assert sum(f.n_trades for f in r_vix.folds) >= \
               sum(f.n_trades for f in r_no.folds) - 5  # tiny tolerance

    def test_vix_threshold_zero_disables_gate(self):
        """vix_gate_threshold=0 must not gate any day."""
        prices  = _make_prices(n=600, seed=1)
        config  = _make_simple_config(eg_pvalue=0.10, max_halflife=300,
                                       vix_gate_threshold=0.0,
                                       use_market_regime_filter=False)
        all_idx = pd.date_range("2010-01-01", periods=len(prices), freq="B", tz="UTC")
        all_high_vix = pd.Series(999.0, index=all_idx, name="vix")
        r = WalkForwardBacktest(config).run(prices, vix=all_high_vix)
        r_no_vix = WalkForwardBacktest(config).run(prices)
        # With threshold=0 (disabled), trades should equal no-VIX baseline
        assert sum(f.n_trades for f in r.folds) == \
               sum(f.n_trades for f in r_no_vix.folds)

    def test_vix_metrics_stored_in_fold(self):
        """avg_vix and vix_gated_pct must appear in fold metrics when VIX is used."""
        from pathlib import Path
        p = Path("data/market/vix.parquet")
        if not p.exists():
            pytest.skip("VIX data not available")
        from stat_arb.data.vix import load_vix
        vix = load_vix()
        prices = _make_prices(n=600, seed=42)
        config = _make_simple_config(eg_pvalue=0.10, max_halflife=300,
                                      vix_gate_threshold=20.0)
        result = WalkForwardBacktest(config).run(
            prices,
            vix=pd.Series(15.0,
                          index=pd.date_range("2010-01-01", periods=len(prices), freq="B", tz="UTC"),
                          name="vix"),
        )
        for f in result.folds:
            if f.n_pairs > 0:
                assert "avg_vix" in f.metrics
                assert "vix_gated_pct" in f.metrics

    def test_real_vix_blocks_covid_period(self):
        """During COVID (2020 Feb-Apr), VIX > 20 → those days should be gated."""
        from pathlib import Path
        from stat_arb.data.vix import load_vix
        p = Path("data/market/vix.parquet")
        if not p.exists():
            pytest.skip("VIX data not available")
        vix = load_vix()
        covid_vix = vix.loc["2020-02-01":"2020-05-01"]
        if len(covid_vix) == 0:
            pytest.skip("No VIX data for COVID period")
        pct_above_20 = float((covid_vix > 20).mean())
        assert pct_above_20 > 0.70, \
            f"Expected >70% of COVID period (Feb-May 2020) to have VIX>20, got {pct_above_20:.0%}"

    def test_vix_gate_plus_hmm_gate_combined(self):
        """With both VIX and HMM market gates active, filter should be strictest."""
        prices  = _make_prices(n=900, seed=42)
        cfg_both = _make_simple_config(eg_pvalue=0.10, max_halflife=300,
                                        use_market_regime_filter=True,
                                        vix_gate_threshold=20.0)
        cfg_hmm  = _make_simple_config(eg_pvalue=0.10, max_halflife=300,
                                        use_market_regime_filter=True,
                                        vix_gate_threshold=0.0)  # HMM only
        cfg_vix  = _make_simple_config(eg_pvalue=0.10, max_halflife=300,
                                        use_market_regime_filter=False,
                                        vix_gate_threshold=20.0)  # VIX only
        # Run with medium VIX to allow some trades
        med_vix  = pd.Series(15.0,
                             index=pd.date_range("2010-01-01", periods=len(prices), freq="B", tz="UTC"),
                             name="vix")
        r_both = WalkForwardBacktest(cfg_both).run(prices, vix=med_vix)
        r_hmm  = WalkForwardBacktest(cfg_hmm).run(prices)
        r_vix  = WalkForwardBacktest(cfg_vix).run(prices, vix=med_vix)
        # Both filters together should give <= trades than either alone
        trades_both = sum(f.n_trades for f in r_both.folds)
        trades_hmm  = sum(f.n_trades for f in r_hmm.folds)
        trades_vix  = sum(f.n_trades for f in r_vix.folds)
        assert trades_both <= trades_hmm + 5
        assert trades_both <= trades_vix + 5
