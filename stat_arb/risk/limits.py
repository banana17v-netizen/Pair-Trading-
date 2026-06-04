"""Hard position and drawdown limits for the pairs trading system.

This module provides the portfolio-level risk guardrails that CANNOT be
bypassed by signal logic.  They are the final safety layer before any trade
reaches execution.

Risk hierarchy (outer to inner):
  1. Portfolio drawdown limit   → CLOSE_ALL when equity drops > max_dd from peak
  2. Daily loss limit           → HALT_TRADING when daily P&L < -max_daily_loss
  3. Soft drawdown warning      → REDUCE sizing proportionally in the warning zone
  4. Position count limit       → Block new positions when at max_pairs_open
  5. Per-pair notional limit    → Block oversized trades
  6. Total notional limit       → Block when portfolio would be over-leveraged
  7. Concentration limit        → Block single pair from dominating portfolio

All limits are loaded from config/settings.yaml defaults but can be overridden
at runtime for paper-trading or live-trading contexts.

Safety rules (CLAUDE.md):
  - limits.py defines max notional, max drawdown, max pairs
  - Never change these limits without explicit user instruction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


# ── Risk action enum ──────────────────────────────────────────────────────────

class RiskAction(Enum):
    """Portfolio-level risk action returned by RiskChecker.check_portfolio_action().

    NORMAL       : All systems go — normal trade sizing.
    REDUCE       : Approaching limits — scale down new position sizes (50%).
    CLOSE_ALL    : Hard limit breached — close ALL open positions immediately.
    HALT_TRADING : Daily loss exceeded — no new entries for the rest of the day.
    """
    NORMAL       = "normal"
    REDUCE       = "reduce"
    CLOSE_ALL    = "close_all"
    HALT_TRADING = "halt_trading"


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class RiskLimits:
    """All configurable risk parameters.

    Defaults match config/settings.yaml.
    """

    # ── Hard position limits ──────────────────────────────────────────────────
    max_notional_per_pair: float = 50_000.0
    max_pairs_open:        int   = 15
    max_total_notional:    float = 750_000.0   # = max_pairs × max_notional

    # ── Loss limits ───────────────────────────────────────────────────────────
    max_portfolio_drawdown: float = 0.15        # 15% from peak → CLOSE_ALL
    max_daily_loss_pct:     float = 0.03        # 3% daily P&L → HALT_TRADING

    # ── Soft / warning thresholds ─────────────────────────────────────────────
    soft_drawdown_pct:   float = 0.08           # 8% drawdown → REDUCE to 50%
    soft_daily_loss_pct: float = 0.015          # 1.5% daily loss → REDUCE

    # ── Concentration limit ───────────────────────────────────────────────────
    max_concentration: float = 0.25             # max 25% of portfolio in one pair

    @classmethod
    def from_settings(cls) -> "RiskLimits":
        """Load from config/settings.yaml (fallback to defaults if file missing)."""
        try:
            import yaml
            from pathlib import Path
            cfg_path = Path("config/settings.yaml")
            if cfg_path.exists():
                with open(cfg_path) as f:
                    cfg = yaml.safe_load(f).get("risk", {})
                return cls(
                    max_notional_per_pair=cfg.get("max_notional_per_pair", 50_000),
                    max_pairs_open=cfg.get("max_pairs_open", 15),
                    max_portfolio_drawdown=cfg.get("max_portfolio_drawdown", 0.15),
                    max_daily_loss_pct=cfg.get("max_daily_loss", 0.03),
                )
        except Exception as exc:
            logger.debug(f"Could not load risk settings: {exc}. Using defaults.")
        return cls()


# ── Portfolio state ───────────────────────────────────────────────────────────

@dataclass
class PortfolioState:
    """Mutable snapshot of the portfolio at the current moment.

    Updated daily by the execution layer (or backtesting engine).
    """

    # Position inventory
    open_positions:  dict[str, float] = field(default_factory=dict)
    # Key = "SYMB_Y/SYMB_X", value = signed notional ($)

    # Equity tracking (normalised, starting at 1.0)
    peak_equity:    float = 1.0
    current_equity: float = 1.0

    # Daily P&L tracking (reset at start of each trading day)
    daily_pnl_usd:  float = 0.0
    portfolio_value: float = 750_000.0   # total capital ($)

    # Timestamps
    date: Optional[pd.Timestamp] = None

    # ── Computed properties ───────────────────────────────────────────────────

    @property
    def n_positions(self) -> int:
        return len(self.open_positions)

    @property
    def total_notional(self) -> float:
        return sum(abs(v) for v in self.open_positions.values())

    @property
    def drawdown(self) -> float:
        """Current drawdown from peak (0 to -1, negative = loss)."""
        if self.peak_equity <= 0:
            return 0.0
        return (self.current_equity - self.peak_equity) / self.peak_equity

    @property
    def daily_pnl_pct(self) -> float:
        """Today's P&L as % of total portfolio value."""
        if self.portfolio_value <= 0:
            return 0.0
        return self.daily_pnl_usd / self.portfolio_value

    # ── State update methods ──────────────────────────────────────────────────

    def update_equity(self, daily_return: float) -> None:
        """Apply today's portfolio return and update peak."""
        self.current_equity *= (1.0 + daily_return)
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
        self.daily_pnl_usd = daily_return * self.portfolio_value

    def add_position(self, pair_key: str, notional: float) -> None:
        """Record a new position (direction encoded in sign of notional)."""
        self.open_positions[pair_key] = notional

    def remove_position(self, pair_key: str) -> None:
        """Remove a closed position."""
        self.open_positions.pop(pair_key, None)

    def reset_daily_pnl(self) -> None:
        """Call at start of each trading day."""
        self.daily_pnl_usd = 0.0

    def pair_concentration(self, notional: float) -> float:
        """Fraction of total portfolio value this position represents."""
        if self.portfolio_value <= 0:
            return 1.0
        return abs(notional) / self.portfolio_value


# ── Risk checker ──────────────────────────────────────────────────────────────

class RiskChecker:
    """Evaluates risk limits against the current portfolio state.

    Usage
    -----
    limits  = RiskLimits.from_settings()
    checker = RiskChecker(limits)

    # Before opening a new position:
    ok, reason = checker.check_new_position(state, notional=50_000)

    # Each day, determine portfolio-level action:
    action = checker.check_portfolio_action(state)
    if action == RiskAction.CLOSE_ALL:
        # … close everything
    """

    def __init__(self, limits: RiskLimits) -> None:
        self.limits = limits

    # ── Position-level checks ─────────────────────────────────────────────────

    def check_new_position(
        self,
        state: PortfolioState,
        notional: float,
        pair_key: str = "",
    ) -> tuple[bool, str]:
        """Can we open a new position with this notional?

        Returns
        -------
        (allowed: bool, reason: str)
        """
        L = self.limits

        # Portfolio-level action must be NORMAL or REDUCE to allow new trades
        action = self.check_portfolio_action(state)
        if action in (RiskAction.CLOSE_ALL, RiskAction.HALT_TRADING):
            return False, f"Portfolio action is {action.value} — no new positions"

        # Position count
        if state.n_positions >= L.max_pairs_open:
            return False, f"Max positions {L.max_pairs_open} already open"

        # Per-pair notional
        if notional > L.max_notional_per_pair:
            return False, (f"Notional ${notional:,.0f} > max "
                           f"${L.max_notional_per_pair:,.0f} per pair")

        # Total notional
        if state.total_notional + notional > L.max_total_notional:
            return False, (f"Total notional ${state.total_notional + notional:,.0f} "
                           f"would exceed ${L.max_total_notional:,.0f}")

        # Concentration
        conc = state.pair_concentration(notional)
        if conc > L.max_concentration:
            return False, (f"Position would be {conc:.1%} of portfolio "
                           f"> max {L.max_concentration:.0%}")

        return True, "OK"

    # ── Portfolio-level check ─────────────────────────────────────────────────

    def check_portfolio_action(self, state: PortfolioState) -> RiskAction:
        """Determine what the portfolio must do right now.

        Priority (highest to lowest):
          CLOSE_ALL    → portfolio drawdown ≥ max_dd
          HALT_TRADING → daily loss ≥ max_daily_loss
          REDUCE       → soft drawdown or soft daily loss
          NORMAL       → everything within limits
        """
        L  = self.limits
        dd = state.drawdown           # negative value
        dl = state.daily_pnl_pct      # negative when losing

        if dd <= -L.max_portfolio_drawdown:
            logger.warning(
                f"CLOSE_ALL: drawdown {dd:.1%} ≤ -{L.max_portfolio_drawdown:.0%}"
            )
            return RiskAction.CLOSE_ALL

        if dl <= -L.max_daily_loss_pct:
            logger.warning(
                f"HALT_TRADING: daily loss {dl:.1%} ≤ -{L.max_daily_loss_pct:.0%}"
            )
            return RiskAction.HALT_TRADING

        if dd <= -L.soft_drawdown_pct:
            logger.debug(f"REDUCE: drawdown {dd:.1%} in warning zone")
            return RiskAction.REDUCE

        if dl <= -L.soft_daily_loss_pct:
            logger.debug(f"REDUCE: daily loss {dl:.1%} in warning zone")
            return RiskAction.REDUCE

        return RiskAction.NORMAL

    # ── Sizing scalar ─────────────────────────────────────────────────────────

    def get_sizing_scalar(self, state: PortfolioState) -> float:
        """Multiplicative scalar [0, 1] for position sizing.

        1.0 = full size (NORMAL)
        0.5 = half size (REDUCE — smooth taper through warning zone)
        0.0 = no new trades (CLOSE_ALL / HALT_TRADING)
        """
        L      = self.limits
        action = self.check_portfolio_action(state)

        if action in (RiskAction.CLOSE_ALL, RiskAction.HALT_TRADING):
            return 0.0

        if action == RiskAction.REDUCE:
            # Linear taper from 1.0 at soft threshold to 0.3 approaching hard limit
            dd   = abs(state.drawdown)
            soft = L.soft_drawdown_pct
            hard = L.max_portfolio_drawdown
            if hard > soft and dd > soft:
                progress = min(1.0, (dd - soft) / (hard - soft))
                return max(0.3, 1.0 - 0.7 * progress)
            return 0.5

        return 1.0

    # ── Convenience summary ───────────────────────────────────────────────────

    def summary(self, state: PortfolioState) -> dict:
        """Return a dict snapshot of current risk status (for logging/display)."""
        action = self.check_portfolio_action(state)
        return {
            "action":           action.value,
            "n_positions":      state.n_positions,
            "total_notional":   state.total_notional,
            "drawdown_pct":     f"{state.drawdown:.2%}",
            "daily_pnl_pct":    f"{state.daily_pnl_pct:.2%}",
            "sizing_scalar":    self.get_sizing_scalar(state),
            "peak_equity":      state.peak_equity,
            "current_equity":   state.current_equity,
        }
