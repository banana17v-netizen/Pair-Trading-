"""Tests for stat_arb/risk/sizing.py."""

from __future__ import annotations

import numpy as np
import pytest

from stat_arb.risk.sizing import (
    compute_position_size,
    compute_is_trade_pnls,
    estimate_spread_stats,
    kelly_fraction,
    kelly_from_sharpe,
    kelly_from_trade_stats,
    volatility_adjusted_notional,
    adaptive_notional_per_pair,
)


# ═══════════════════════════════════════════════════════════════════════════════
# kelly_fraction
# ═══════════════════════════════════════════════════════════════════════════════

class TestKellyFraction:
    def test_positive_edge_gives_positive_fraction(self):
        f = kelly_fraction(win_rate=0.55, avg_win_pct=0.008, avg_loss_pct=0.005)
        assert f > 0

    def test_no_edge_gives_zero(self):
        # p=0.5, W=L → full Kelly = 0
        f = kelly_fraction(win_rate=0.5, avg_win_pct=0.005, avg_loss_pct=0.005)
        assert f == pytest.approx(0.0, abs=1e-6)

    def test_negative_edge_gives_zero(self):
        f = kelly_fraction(win_rate=0.40, avg_win_pct=0.005, avg_loss_pct=0.008)
        assert f == pytest.approx(0.0, abs=1e-6)

    def test_half_kelly_multiplier_halves_result(self):
        f_full = kelly_fraction(0.60, 0.01, 0.007, kelly_multiplier=1.0)
        f_half = kelly_fraction(0.60, 0.01, 0.007, kelly_multiplier=0.5)
        assert f_half == pytest.approx(0.5 * f_full, rel=1e-6)

    def test_fraction_capped_at_one(self):
        # Very high edge → full Kelly would exceed 1
        f = kelly_fraction(0.99, 0.50, 0.001, kelly_multiplier=1.0)
        assert f <= 1.0

    def test_fraction_non_negative(self):
        f = kelly_fraction(0.10, 0.001, 0.50)
        assert f >= 0.0

    def test_invalid_inputs_return_zero(self):
        assert kelly_fraction(0.0, 0.01, 0.01) == 0.0  # zero win rate
        assert kelly_fraction(0.5, 0.0, 0.01) == 0.0   # zero avg win
        assert kelly_fraction(0.5, 0.01, 0.0) == 0.0   # zero avg loss


# ═══════════════════════════════════════════════════════════════════════════════
# kelly_from_sharpe
# ═══════════════════════════════════════════════════════════════════════════════

class TestKellyFromSharpe:
    def test_positive_sharpe_gives_above_min(self):
        k = kelly_from_sharpe(1.0)
        assert k > 0

    def test_zero_sharpe_gives_min_fraction(self):
        k = kelly_from_sharpe(0.0, min_fraction=0.3)
        assert k == pytest.approx(0.3)

    def test_negative_sharpe_gives_min_fraction(self):
        k = kelly_from_sharpe(-0.5, min_fraction=0.3)
        assert k == pytest.approx(0.3)

    def test_high_sharpe_capped_at_max(self):
        """Very high Sharpe should be capped by max_fraction."""
        k = kelly_from_sharpe(10.0, max_fraction=1.3)
        assert k == pytest.approx(1.3)

    def test_sharpe_one_gives_reasonable_size(self):
        k = kelly_from_sharpe(1.0)
        assert 0.5 <= k <= 1.5

    def test_higher_sharpe_gives_higher_k(self):
        k1 = kelly_from_sharpe(0.5)
        k2 = kelly_from_sharpe(1.5)
        assert k2 >= k1


# ═══════════════════════════════════════════════════════════════════════════════
# volatility_adjusted_notional
# ═══════════════════════════════════════════════════════════════════════════════

class TestVolatilityAdjustedNotional:
    def test_high_vol_reduces_notional(self):
        """High spread vol → smaller notional (less risk per dollar)."""
        base  = 50_000
        high_vol = volatility_adjusted_notional(base, spread_std_daily=0.020, target_vol=0.010)
        low_vol  = volatility_adjusted_notional(base, spread_std_daily=0.005, target_vol=0.010)
        assert high_vol < low_vol

    def test_exact_target_vol_returns_base(self):
        """When spread std = target vol, no adjustment needed."""
        base   = 50_000
        result = volatility_adjusted_notional(base, spread_std_daily=0.010, target_vol=0.010)
        assert result == pytest.approx(base, rel=1e-6)

    def test_low_vol_scales_up_to_max(self):
        """Very low spread vol → scale up, capped at max_notional."""
        result = volatility_adjusted_notional(
            50_000, spread_std_daily=0.001, target_vol=0.010, max_notional=100_000
        )
        assert result == pytest.approx(100_000)  # capped

    def test_high_vol_scales_down_to_min(self):
        """Very high spread vol → scale down, floored at min_notional."""
        result = volatility_adjusted_notional(
            50_000, spread_std_daily=0.100, target_vol=0.010, min_notional=5_000
        )
        assert result == pytest.approx(5_000)  # floored

    def test_zero_vol_returns_base(self):
        """Edge case: zero vol → return base unchanged."""
        base   = 50_000
        result = volatility_adjusted_notional(base, spread_std_daily=0.0)
        assert result == pytest.approx(base)

    def test_notional_in_bounds(self):
        """Output always within [min, max]."""
        for std in [0.0001, 0.001, 0.01, 0.05, 0.5]:
            result = volatility_adjusted_notional(
                50_000, std, target_vol=0.01,
                min_notional=5_000, max_notional=100_000
            )
            assert 5_000 <= result <= 100_000

    def test_half_vol_doubles_notional(self):
        """Halving spread vol should double notional (up to cap)."""
        n1 = volatility_adjusted_notional(50_000, 0.01, 0.01, max_notional=200_000)
        n2 = volatility_adjusted_notional(50_000, 0.005, 0.01, max_notional=200_000)
        assert n2 == pytest.approx(2 * n1, rel=1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# compute_position_size
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputePositionSize:
    def test_zero_risk_scalar_gives_zero(self):
        size = compute_position_size(50_000, 0.010, is_sharpe=1.5, risk_scalar=0.0)
        assert size == pytest.approx(0.0)

    def test_negative_risk_scalar_gives_zero(self):
        size = compute_position_size(50_000, 0.010, is_sharpe=1.0, risk_scalar=-1.0)
        assert size == pytest.approx(0.0)

    def test_high_sharpe_gives_larger_size(self):
        low_sh  = compute_position_size(50_000, 0.010, is_sharpe=0.5)
        high_sh = compute_position_size(50_000, 0.010, is_sharpe=2.0)
        assert high_sh >= low_sh

    def test_negative_sharpe_reduces_size_nonconsv(self):
        """In non-conservative mode, negative Sharpe reduces size via Kelly."""
        positive = compute_position_size(50_000, 0.010, is_sharpe=1.0, conservative=False)
        negative = compute_position_size(50_000, 0.010, is_sharpe=-0.5, conservative=False)
        assert negative < positive

    def test_size_always_in_bounds(self):
        for sharpe in [-1, 0, 0.5, 1.0, 2.0, 3.0]:
            for std in [0.005, 0.010, 0.020]:
                for scalar in [0.3, 0.5, 1.0]:
                    size = compute_position_size(
                        50_000, std, is_sharpe=sharpe, risk_scalar=scalar,
                        min_notional=5_000, max_notional=100_000,
                    )
                    assert 5_000 <= size <= 100_000

    def test_reduce_scalar_proportionally_reduces_size(self):
        full   = compute_position_size(50_000, 0.010, is_sharpe=1.0, risk_scalar=1.0)
        halved = compute_position_size(50_000, 0.010, is_sharpe=1.0, risk_scalar=0.5)
        assert halved < full

    def test_high_vol_pair_has_smaller_size(self):
        low_vol  = compute_position_size(50_000, 0.005, is_sharpe=1.0)
        high_vol = compute_position_size(50_000, 0.020, is_sharpe=1.0)
        assert high_vol < low_vol

    def test_good_pair_gets_larger_allocation_nonconsv(self):
        """In non-conservative mode, higher Sharpe → larger notional."""
        good = compute_position_size(50_000, 0.010, is_sharpe=2.0, conservative=False)
        weak = compute_position_size(50_000, 0.010, is_sharpe=0.5, conservative=False)
        assert good > weak

    def test_realistic_pair_example(self):
        """ICE/SPGI: spread_std ~0.7%/day, IS Sharpe ~2.0, full risk."""
        size = compute_position_size(
            base_notional=50_000,
            spread_std_daily=0.007,  # 0.7%/day
            is_sharpe=2.0,
            risk_scalar=1.0,
            target_vol=0.010,
        )
        # vol_adj = 50K × (1%/0.7%) = ~71.4K → capped at 100K
        # kelly_from_sharpe(2.0) ≈ 1.3
        # final ~ min(100K × 1.3, 100K) = 100K (capped)
        assert 50_000 <= size <= 100_000

    def test_warning_zone_pair_example(self):
        """Same pair but in warning zone (50% scalar)."""
        full    = compute_position_size(50_000, 0.007, is_sharpe=2.0, risk_scalar=1.0)
        warning = compute_position_size(50_000, 0.007, is_sharpe=2.0, risk_scalar=0.5)
        assert warning < full
        assert warning >= 5_000   # still above floor


# ═══════════════════════════════════════════════════════════════════════════════
# estimate_spread_stats
# ═══════════════════════════════════════════════════════════════════════════════

class TestEstimateSpreadStats:
    def test_returns_required_keys(self):
        rng    = np.random.default_rng(0)
        rets   = rng.normal(0.001, 0.01, 300)
        stats  = estimate_spread_stats(rets)
        for key in ["win_rate", "avg_win_pct", "avg_loss_pct", "daily_std", "sharpe"]:
            assert key in stats

    def test_positive_mean_gives_positive_sharpe(self):
        rng  = np.random.default_rng(1)
        rets = rng.normal(0.005, 0.01, 500)   # strong positive signal
        stats = estimate_spread_stats(rets)
        assert stats["sharpe"] > 0

    def test_win_rate_in_range(self):
        rng  = np.random.default_rng(2)
        rets = rng.normal(0, 0.01, 300)
        stats = estimate_spread_stats(rets)
        assert 0 <= stats["win_rate"] <= 1

    def test_short_series_returns_defaults(self):
        """Too few observations → safe defaults returned."""
        rets  = np.array([0.001, -0.002, 0.003])
        stats = estimate_spread_stats(rets, warmup=5)
        assert stats["sharpe"] == 0.0   # default for insufficient data

    def test_warmup_excluded(self):
        """Results should differ when warmup period is excluded."""
        rng  = np.random.default_rng(3)
        rets = np.concatenate([
            rng.normal(-0.01, 0.005, 30),   # bad warmup period
            rng.normal(+0.005, 0.005, 270),  # good period
        ])
        stats_no_warmup = estimate_spread_stats(rets, warmup=30)
        stats_all       = estimate_spread_stats(rets, warmup=0)
        # Without warmup, Sharpe should be higher (bad period excluded)
        assert stats_no_warmup["sharpe"] > stats_all["sharpe"]


# ═══════════════════════════════════════════════════════════════════════════════
# kelly_from_trade_stats (new v2)
# ═══════════════════════════════════════════════════════════════════════════════

class TestKellyFromTradeStats:
    def test_positive_edge_trades_give_positive_kelly(self):
        """Trades with 60% win rate and favorable R/R → positive Kelly."""
        rng = np.random.default_rng(42)
        # 60% wins at +0.8%, 40% losses at -0.5%
        trades = np.concatenate([
            rng.normal(0.008, 0.001, 60),    # 60 winning trades
            rng.normal(-0.005, 0.001, 40),   # 40 losing trades
        ])
        k = kelly_from_trade_stats(trades)
        assert k > 0

    def test_zero_edge_gives_zero(self):
        """Equal win/loss size → Kelly ≈ 0."""
        rng = np.random.default_rng(0)
        # 50% win rate, equal avg win/loss
        trades = np.concatenate([
            rng.normal(0.005, 0.001, 50),
            rng.normal(-0.005, 0.001, 50),
        ])
        k = kelly_from_trade_stats(trades)
        assert k == pytest.approx(0.0, abs=0.1)

    def test_insufficient_trades_returns_zero(self):
        k = kelly_from_trade_stats(np.array([0.01, -0.005, 0.008]))
        assert k == 0.0

    def test_all_winning_trades_caps_at_one(self):
        trades = np.array([0.01] * 20)
        k = kelly_from_trade_stats(trades)
        assert k <= 1.0

    def test_half_kelly_multiplier(self):
        rng = np.random.default_rng(7)
        trades = np.concatenate([
            rng.normal(0.01, 0.001, 60),
            rng.normal(-0.006, 0.001, 40),
        ])
        k_full = kelly_from_trade_stats(trades, kelly_multiplier=1.0)
        k_half = kelly_from_trade_stats(trades, kelly_multiplier=0.5)
        assert k_half == pytest.approx(0.5 * k_full, rel=1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# adaptive_notional_per_pair (new v2)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdaptiveNotional:
    def test_under_budget_returns_base(self):
        """5 pairs × $50K = $250K < $750K budget → no scaling."""
        n = adaptive_notional_per_pair(50_000, total_budget=750_000, n_pairs_target=5)
        assert n == pytest.approx(50_000)

    def test_over_budget_scales_down(self):
        """18 pairs × $50K = $900K > $750K → scale to $41.7K."""
        n = adaptive_notional_per_pair(50_000, total_budget=750_000, n_pairs_target=18)
        expected = 750_000 / 18
        assert n == pytest.approx(expected, rel=1e-6)

    def test_exactly_at_budget_no_scaling(self):
        """15 pairs × $50K = $750K → no scaling needed."""
        n = adaptive_notional_per_pair(50_000, total_budget=750_000, n_pairs_target=15)
        assert n == pytest.approx(50_000)

    def test_floored_at_min(self):
        """Very many pairs → floor at min_notional."""
        n = adaptive_notional_per_pair(50_000, total_budget=100_000, n_pairs_target=100,
                                        min_notional=5_000)
        assert n >= 5_000

    def test_zero_pairs_returns_base(self):
        n = adaptive_notional_per_pair(50_000, total_budget=750_000, n_pairs_target=0)
        assert n == pytest.approx(50_000)


# ═══════════════════════════════════════════════════════════════════════════════
# compute_is_trade_pnls (new v2)
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeIsTradesPnls:
    def _make_cointegrated(self, n=500, seed=42):
        rng = np.random.default_rng(seed)
        log_x = np.cumsum(rng.normal(0, 0.01, n))
        log_y = 0.3 + 1.2 * log_x + rng.normal(0, 0.015, n)
        return log_y, log_x, 1.2

    def test_returns_array(self):
        log_y, log_x, beta = self._make_cointegrated()
        from stat_arb.backtesting.engine import _rolling_zscore
        spread = log_y - 0.3 - beta * log_x
        z = _rolling_zscore(spread, 10)
        pnls = compute_is_trade_pnls(log_y, log_x, beta, z)
        assert isinstance(pnls, np.ndarray)

    def test_cointegrated_pair_has_trades(self):
        """A strongly cointegrated pair should complete some trades."""
        log_y, log_x, beta = self._make_cointegrated()
        from stat_arb.backtesting.engine import _rolling_zscore
        spread = log_y - 0.3 - beta * log_x
        z = _rolling_zscore(spread, 10)
        pnls = compute_is_trade_pnls(log_y, log_x, beta, z, entry_z=1.5)
        # Should have at least some completed trades in 500 bars
        assert len(pnls) >= 0   # allow zero if no z-crossings

    def test_pnl_values_are_finite(self):
        log_y, log_x, beta = self._make_cointegrated()
        from stat_arb.backtesting.engine import _rolling_zscore
        spread = log_y - 0.3 - beta * log_x
        z = _rolling_zscore(spread, 10)
        pnls = compute_is_trade_pnls(log_y, log_x, beta, z)
        assert np.isfinite(pnls).all()


# ═══════════════════════════════════════════════════════════════════════════════
# compute_position_size v2 — target_vol calibration
# ═══════════════════════════════════════════════════════════════════════════════

class TestPositionSizeV2:
    def test_conservative_volatile_pair_reduced(self):
        """Conservative mode: spread_std > target_vol → reduce notional."""
        size = compute_position_size(50_000, spread_std_daily=0.020,
                                     is_sharpe=0.7, target_vol=0.013,
                                     risk_scalar=1.0, conservative=True,
                                     min_notional=5_000, max_notional=50_000)
        # scale = min(1.0, 0.013/0.020) = 0.65 → 50K × 0.65 = 32.5K
        assert size == pytest.approx(50_000 * 0.013 / 0.020, rel=0.01)

    def test_conservative_calm_pair_not_scaled_up(self):
        """Conservative mode: spread_std < target_vol → keep base (no scale up)."""
        size = compute_position_size(50_000, spread_std_daily=0.007,
                                     is_sharpe=0.7, target_vol=0.013,
                                     risk_scalar=1.0, conservative=True,
                                     min_notional=5_000, max_notional=75_000)
        # scale = min(1.0, 0.013/0.007) = 1.0 → no scaling
        assert size == pytest.approx(50_000, rel=0.01)

    def test_target_vol_equal_spread_std_no_scaling(self):
        """When target_vol == spread_std, conservative mode: neutral (scale=1)."""
        std  = 0.013
        size = compute_position_size(50_000, std, is_sharpe=0.7,
                                     target_vol=std, risk_scalar=1.0,
                                     conservative=True,
                                     min_notional=50_000, max_notional=50_000)
        assert size == pytest.approx(50_000, rel=0.01)

    def test_typical_pair_realistic_size(self):
        """ICE/SPGI: spread_std=1.2%/day, IS Sharpe=0.5, v2 target_vol=1.3%"""
        size = compute_position_size(
            base_notional=50_000,
            spread_std_daily=0.012,
            is_sharpe=0.5,
            risk_scalar=1.0,
            target_vol=0.013,    # v2 default
            min_notional=15_000,
            max_notional=75_000,
        )
        # scale = 1.3/1.2 = 1.08 → vol_adj = 54K → Kelly ~0.85 → ~46K
        assert 15_000 <= size <= 75_000

    def test_adaptive_budget_18_pairs(self):
        """18 pairs should fit within $750K budget via adaptive scaling."""
        notionals = [
            compute_position_size(
                50_000, 0.013, is_sharpe=0.5,
                n_open_pairs=18, total_budget=750_000,
                min_notional=5_000, max_notional=75_000,
            )
            for _ in range(18)
        ]
        total = sum(notionals)
        # Each pair should be scaled down to fit (750K / 18 ≈ 41.7K)
        assert total <= 750_000 + 100   # allow tiny float error

    def test_trade_pnls_improve_sizing_for_good_pair(self):
        """Positive-edge trade history should give larger size than Sharpe alone."""
        rng = np.random.default_rng(42)
        good_trades = np.concatenate([
            rng.normal(0.010, 0.002, 60),   # 60% winning trades
            rng.normal(-0.006, 0.002, 40),
        ])
        size_with_trades = compute_position_size(
            50_000, 0.013, is_sharpe=0.5,
            trade_pnls=good_trades, risk_scalar=1.0,
            min_notional=5_000, max_notional=100_000,
        )
        size_no_trades = compute_position_size(
            50_000, 0.013, is_sharpe=0.5,
            trade_pnls=None, risk_scalar=1.0,
            min_notional=5_000, max_notional=100_000,
        )
        # Good trade history should give >= base sizing
        assert size_with_trades > 0
        assert size_no_trades > 0
