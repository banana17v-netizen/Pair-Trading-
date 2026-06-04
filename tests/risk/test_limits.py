"""Tests for stat_arb/risk/limits.py."""

from __future__ import annotations

import numpy as np
import pytest

from stat_arb.risk.limits import (
    PortfolioState,
    RiskAction,
    RiskChecker,
    RiskLimits,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_state(**kwargs) -> PortfolioState:
    """Convenience: return PortfolioState with custom attributes."""
    state = PortfolioState()
    for k, v in kwargs.items():
        setattr(state, k, v)
    return state


def _make_checker(**limit_kwargs) -> RiskChecker:
    return RiskChecker(RiskLimits(**limit_kwargs))


# ═══════════════════════════════════════════════════════════════════════════════
# RiskLimits
# ═══════════════════════════════════════════════════════════════════════════════

class TestRiskLimits:
    def test_default_values_match_settings(self):
        L = RiskLimits()
        assert L.max_notional_per_pair == 50_000
        assert L.max_pairs_open == 15
        assert L.max_portfolio_drawdown == 0.15
        assert L.max_daily_loss_pct == 0.03

    def test_from_settings_returns_instance(self):
        L = RiskLimits.from_settings()
        assert isinstance(L, RiskLimits)
        assert L.max_notional_per_pair > 0


# ═══════════════════════════════════════════════════════════════════════════════
# PortfolioState
# ═══════════════════════════════════════════════════════════════════════════════

class TestPortfolioState:
    def test_initial_state_no_drawdown(self):
        state = PortfolioState()
        assert state.drawdown == pytest.approx(0.0)

    def test_drawdown_calculation(self):
        state = PortfolioState(peak_equity=1.0, current_equity=0.85)
        assert state.drawdown == pytest.approx(-0.15, rel=1e-6)

    def test_update_equity_tracks_peak(self):
        state = PortfolioState()
        state.update_equity(0.05)    # +5%
        assert state.peak_equity == pytest.approx(1.05, rel=1e-6)
        state.update_equity(-0.03)   # -3% from new level
        assert state.peak_equity == pytest.approx(1.05, rel=1e-6)  # peak unchanged
        assert state.current_equity == pytest.approx(1.05 * 0.97, rel=1e-6)

    def test_drawdown_after_loss(self):
        state = PortfolioState()
        state.update_equity(0.10)    # grow to 1.1
        state.update_equity(-0.20)   # drop 20% → 1.1 × 0.8 = 0.88
        expected_dd = (0.88 - 1.10) / 1.10
        assert state.drawdown == pytest.approx(expected_dd, rel=1e-4)

    def test_n_positions(self):
        state = PortfolioState()
        assert state.n_positions == 0
        state.add_position("AAPL/MSFT", 50_000)
        assert state.n_positions == 1
        state.add_position("JPM/BAC", -50_000)
        assert state.n_positions == 2

    def test_total_notional(self):
        state = PortfolioState()
        state.add_position("A/B", 50_000)
        state.add_position("C/D", -30_000)
        assert state.total_notional == pytest.approx(80_000)

    def test_remove_position(self):
        state = PortfolioState()
        state.add_position("A/B", 50_000)
        state.remove_position("A/B")
        assert state.n_positions == 0

    def test_daily_pnl_pct(self):
        state = PortfolioState(portfolio_value=1_000_000, daily_pnl_usd=-30_000)
        assert state.daily_pnl_pct == pytest.approx(-0.03, rel=1e-6)

    def test_reset_daily_pnl(self):
        state = PortfolioState(daily_pnl_usd=-20_000)
        state.reset_daily_pnl()
        assert state.daily_pnl_usd == 0.0

    def test_pair_concentration(self):
        state = PortfolioState(portfolio_value=500_000)
        conc = state.pair_concentration(50_000)
        assert conc == pytest.approx(0.10, rel=1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# RiskChecker — portfolio action
# ═══════════════════════════════════════════════════════════════════════════════

class TestRiskCheckerPortfolioAction:
    def test_normal_state_returns_normal(self):
        checker = _make_checker()
        state   = PortfolioState()   # no drawdown, no daily loss
        assert checker.check_portfolio_action(state) == RiskAction.NORMAL

    def test_max_drawdown_triggers_close_all(self):
        checker = _make_checker(max_portfolio_drawdown=0.15)
        state   = PortfolioState(peak_equity=1.0, current_equity=0.84)  # -16%
        assert checker.check_portfolio_action(state) == RiskAction.CLOSE_ALL

    def test_exactly_at_max_drawdown_triggers_close_all(self):
        checker = _make_checker(max_portfolio_drawdown=0.15)
        state   = PortfolioState(peak_equity=1.0, current_equity=0.85)  # exactly -15%
        assert checker.check_portfolio_action(state) == RiskAction.CLOSE_ALL

    def test_just_below_max_drawdown_does_not_trigger_close(self):
        checker = _make_checker(max_portfolio_drawdown=0.15)
        state   = PortfolioState(peak_equity=1.0, current_equity=0.86)  # -14%
        # Should be REDUCE (in soft zone) or NORMAL
        action = checker.check_portfolio_action(state)
        assert action != RiskAction.CLOSE_ALL

    def test_daily_loss_triggers_halt(self):
        checker = _make_checker(max_daily_loss_pct=0.03)
        state   = PortfolioState(portfolio_value=1_000_000, daily_pnl_usd=-35_000)  # -3.5%
        assert checker.check_portfolio_action(state) == RiskAction.HALT_TRADING

    def test_soft_drawdown_triggers_reduce(self):
        checker = _make_checker(soft_drawdown_pct=0.08, max_portfolio_drawdown=0.15)
        state   = PortfolioState(peak_equity=1.0, current_equity=0.90)  # -10%
        assert checker.check_portfolio_action(state) == RiskAction.REDUCE

    def test_soft_daily_loss_triggers_reduce(self):
        checker = _make_checker(soft_daily_loss_pct=0.015, max_daily_loss_pct=0.03)
        state   = PortfolioState(portfolio_value=1_000_000, daily_pnl_usd=-20_000)  # -2%
        assert checker.check_portfolio_action(state) == RiskAction.REDUCE

    def test_priority_close_all_over_halt(self):
        """CLOSE_ALL (drawdown) has higher priority than HALT_TRADING (daily loss)."""
        checker = _make_checker(max_portfolio_drawdown=0.15, max_daily_loss_pct=0.03)
        state   = PortfolioState(
            peak_equity=1.0, current_equity=0.83,   # -17% → CLOSE_ALL
            portfolio_value=1_000_000, daily_pnl_usd=-40_000,  # -4% → HALT
        )
        assert checker.check_portfolio_action(state) == RiskAction.CLOSE_ALL


# ═══════════════════════════════════════════════════════════════════════════════
# RiskChecker — check_new_position
# ═══════════════════════════════════════════════════════════════════════════════

class TestRiskCheckerNewPosition:
    def test_normal_state_allows_position(self):
        checker = _make_checker()
        state   = PortfolioState()
        ok, msg = checker.check_new_position(state, notional=50_000)
        assert ok
        assert msg == "OK"

    def test_max_pairs_blocks_new_position(self):
        checker = _make_checker(max_pairs_open=2)
        state   = PortfolioState()
        state.add_position("A/B", 50_000)
        state.add_position("C/D", 50_000)
        ok, msg = checker.check_new_position(state, notional=50_000)
        assert not ok
        assert "Max positions" in msg

    def test_notional_limit_blocks_oversized(self):
        checker = _make_checker(max_notional_per_pair=50_000)
        state   = PortfolioState()
        ok, msg = checker.check_new_position(state, notional=60_000)
        assert not ok
        assert "max" in msg.lower()

    def test_total_notional_blocks_when_full(self):
        checker = _make_checker(max_total_notional=100_000)
        state   = PortfolioState()
        state.add_position("A/B", 90_000)
        ok, msg = checker.check_new_position(state, notional=20_000)
        assert not ok
        assert "Total" in msg

    def test_drawdown_blocks_new_position(self):
        checker = _make_checker(max_portfolio_drawdown=0.15)
        state   = PortfolioState(peak_equity=1.0, current_equity=0.83)  # -17%
        ok, msg = checker.check_new_position(state, notional=50_000)
        assert not ok
        assert "CLOSE_ALL" in msg or "action" in msg.lower()

    def test_concentration_limit_blocks(self):
        checker = _make_checker(max_concentration=0.10)  # 10% max
        state   = PortfolioState(portfolio_value=200_000)  # small portfolio
        # 50K / 200K = 25% → exceeds 10% limit
        ok, msg = checker.check_new_position(state, notional=50_000)
        assert not ok
        assert "concentration" in msg.lower() or "portfolio" in msg.lower()

    def test_exactly_at_concentration_limit_allowed(self):
        checker = _make_checker(max_concentration=0.25)
        state   = PortfolioState(portfolio_value=200_000)
        # 50K / 200K = 25% → exactly at limit
        ok, _ = checker.check_new_position(state, notional=50_000)
        assert ok


# ═══════════════════════════════════════════════════════════════════════════════
# RiskChecker — sizing scalar
# ═══════════════════════════════════════════════════════════════════════════════

class TestSizingScalar:
    def test_normal_state_scalar_is_one(self):
        checker = _make_checker()
        state   = PortfolioState()
        assert checker.get_sizing_scalar(state) == pytest.approx(1.0)

    def test_close_all_scalar_is_zero(self):
        checker = _make_checker(max_portfolio_drawdown=0.15)
        state   = PortfolioState(peak_equity=1.0, current_equity=0.80)  # -20%
        assert checker.get_sizing_scalar(state) == pytest.approx(0.0)

    def test_halt_trading_scalar_is_zero(self):
        checker = _make_checker(max_daily_loss_pct=0.03)
        state   = PortfolioState(portfolio_value=1_000_000, daily_pnl_usd=-50_000)
        assert checker.get_sizing_scalar(state) == pytest.approx(0.0)

    def test_reduce_scalar_between_zero_and_one(self):
        checker = _make_checker(soft_drawdown_pct=0.08, max_portfolio_drawdown=0.15)
        state   = PortfolioState(peak_equity=1.0, current_equity=0.89)  # -11%
        scalar  = checker.get_sizing_scalar(state)
        assert 0.0 < scalar < 1.0

    def test_scalar_decreases_as_drawdown_worsens(self):
        checker = _make_checker(soft_drawdown_pct=0.05, max_portfolio_drawdown=0.15)
        state_mild  = PortfolioState(peak_equity=1.0, current_equity=0.92)  # -8%
        state_worse = PortfolioState(peak_equity=1.0, current_equity=0.87)  # -13%
        s_mild  = checker.get_sizing_scalar(state_mild)
        s_worse = checker.get_sizing_scalar(state_worse)
        assert s_mild >= s_worse

    def test_summary_contains_expected_keys(self):
        checker = _make_checker()
        state   = PortfolioState()
        summary = checker.summary(state)
        for key in ["action", "n_positions", "drawdown_pct", "sizing_scalar"]:
            assert key in summary


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: full risk pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class TestRiskPipeline:
    """End-to-end tests simulating a trading day."""

    def test_normal_day_full_sizing(self):
        """Normal market → full sizing, positions allowed."""
        limits  = RiskLimits()
        checker = RiskChecker(limits)
        state   = PortfolioState(portfolio_value=750_000)

        state.update_equity(0.002)   # +0.2% — good day
        action = checker.check_portfolio_action(state)
        scalar = checker.get_sizing_scalar(state)
        ok, _  = checker.check_new_position(state, notional=50_000)

        assert action == RiskAction.NORMAL
        assert scalar == pytest.approx(1.0)
        assert ok

    def test_drawdown_day_reduces_then_halts(self):
        """Simulate portfolio hitting drawdown limits over multiple small bad days.

        Note: update_equity() sets daily_pnl_usd to that day's return, so each
        call represents one trading day. We use returns < max_daily_loss_pct (3%)
        to avoid triggering HALT_TRADING before CLOSE_ALL.
        """
        limits  = RiskLimits(
            soft_drawdown_pct=0.05,
            max_portfolio_drawdown=0.15,
            max_daily_loss_pct=0.03,   # daily limit = 3%
        )
        checker = RiskChecker(limits)
        state   = PortfolioState(portfolio_value=750_000)

        # Small losses accumulate drawdown without hitting daily limit (each < 3%)
        for _ in range(3):
            state.update_equity(-0.02)   # -2%/day, 3 days → ~-6%
        state.reset_daily_pnl()          # reset daily for next check

        # At -6% cumulative: in REDUCE zone (soft_dd=5%)
        action = checker.check_portfolio_action(state)
        assert action == RiskAction.REDUCE
        assert 0 < checker.get_sizing_scalar(state) < 1.0

        # Continue with more small losses to exceed hard limit (total >-15%)
        for _ in range(6):
            state.update_equity(-0.02)   # total ~-17%
        state.reset_daily_pnl()

        assert checker.check_portfolio_action(state) == RiskAction.CLOSE_ALL
        assert checker.get_sizing_scalar(state) == pytest.approx(0.0)
        ok, _ = checker.check_new_position(state, notional=50_000)
        assert not ok

    def test_peak_resets_correctly(self):
        """Drawdown is always measured from the highest peak.

        We use small daily losses to avoid triggering daily-loss HALT_TRADING.
        After the loss sequence we reset daily_pnl to check drawdown-based action.
        """
        checker = _make_checker(
            max_portfolio_drawdown=0.15,
            soft_drawdown_pct=0.08,
            max_daily_loss_pct=0.03,
        )
        state = PortfolioState()

        # Grow to peak=1.2 over multiple days
        for _ in range(10):
            state.update_equity(0.02)
        assert state.peak_equity == pytest.approx(1.02 ** 10, rel=1e-3)

        # Small losses bring us to ~10% drawdown from peak (not triggering daily halt)
        for _ in range(5):
            state.update_equity(-0.02)
        state.reset_daily_pnl()   # reset so daily limit doesn't trigger

        # At ~-10% from peak → REDUCE
        assert checker.check_portfolio_action(state) == RiskAction.REDUCE

        # Recover to a new peak
        for _ in range(10):
            state.update_equity(0.02)
        state.reset_daily_pnl()

        # Now current equity > old peak → new peak established → dd = 0 → NORMAL
        assert state.drawdown >= -0.01   # very small drawdown
        assert checker.check_portfolio_action(state) == RiskAction.NORMAL
