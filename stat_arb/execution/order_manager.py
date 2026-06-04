"""Order manager — translates TradeDecisions into Alpaca orders.

Each pairs trade = 2 legs:
  Long spread  (direction=+1): BUY sym_y, SELL sym_x
  Short spread (direction=-1): SELL sym_y, BUY sym_x
  Close spread (direction=0):  reverse the opening trade

SAFETY: This is LIVE CODE. All orders are logged. Paper trading only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from loguru import logger

from stat_arb.execution.alpaca_client import (
    close_position,
    get_latest_prices,
    submit_market_order,
)
from stat_arb.execution.portfolio_manager import TradeDecision


@dataclass
class OrderResult:
    """Result of submitting one pairs trade (two legs)."""
    decision:       TradeDecision
    leg_y_order_id: str | None
    leg_x_order_id: str | None
    success:        bool
    error:          str | None = None


class OrderManager:
    """Submits order pairs to Alpaca."""

    def __init__(self, dry_run: bool = True) -> None:
        self.dry_run = dry_run
        if dry_run:
            logger.info("OrderManager in DRY-RUN mode — no orders will be submitted")

    def execute_decisions(
        self,
        decisions: list[TradeDecision],
        prices:    dict[str, float] | None = None,
    ) -> list[OrderResult]:
        """Submit orders for all OPEN and CLOSE decisions.

        KEEP decisions are ignored (no action needed).
        """
        results = []
        to_act  = [d for d in decisions if d.action in ("OPEN", "CLOSE")]

        if not to_act:
            logger.info("No orders needed this run")
            return []

        logger.info(f"{'[DRY-RUN] ' if self.dry_run else ''}Processing {len(to_act)} decisions")

        for decision in to_act:
            result = self._execute_one(decision, prices)
            results.append(result)
            if not self.dry_run:
                time.sleep(0.3)   # rate limit

        n_ok  = sum(1 for r in results if r.success)
        n_err = sum(1 for r in results if not r.success)
        logger.info(f"Orders: {n_ok} OK, {n_err} errors")
        return results

    def _execute_one(
        self,
        decision: TradeDecision,
        prices:   dict[str, float] | None,
    ) -> OrderResult:
        """Execute one OPEN or CLOSE decision."""
        sym_y = decision.sym_y
        sym_x = decision.sym_x

        if decision.action == "CLOSE":
            return self._close_pair(decision)
        else:
            return self._open_pair(decision, prices)

    def _open_pair(
        self,
        decision: TradeDecision,
        prices:   dict[str, float] | None,
    ) -> OrderResult:
        """Open a new pairs position."""
        sym_y     = decision.sym_y
        sym_x     = decision.sym_x
        notional  = decision.notional
        direction = decision.direction

        # direction=+1: long spread → BUY Y, SELL X
        # direction=-1: short spread → SELL Y, BUY X
        side_y = "buy"  if direction == 1 else "sell"
        side_x = "sell" if direction == 1 else "buy"

        logger.info(
            f"{'[DRY-RUN] ' if self.dry_run else ''}"
            f"OPEN {sym_y}/{sym_x}: {side_y.upper()} {sym_y} + {side_x.upper()} {sym_x}  "
            f"notional=${notional:,.0f}  z={decision.z_score:.2f}"
        )

        if self.dry_run:
            return OrderResult(
                decision=decision,
                leg_y_order_id="DRY_RUN",
                leg_x_order_id="DRY_RUN",
                success=True,
            )

        # Alpaca rule: fractional (notional) orders work for BUY;
        # short selling requires WHOLE shares (integer qty).
        # Fetch current prices to compute share counts for short legs.
        current_prices = get_latest_prices([sym_y, sym_x])
        price_y = current_prices.get(sym_y, 0)
        price_x = current_prices.get(sym_x, 0)

        try:
            oy_id = None
            ox_id = None

            # Leg Y
            if side_y == "sell" and price_y > 0:
                # Short sell: use whole shares
                qty_y = max(1, int(notional / price_y))
                oy = submit_market_order(sym_y, qty=qty_y, side="sell", notional=None)
            else:
                # Buy: use notional (fractional OK)
                oy = submit_market_order(sym_y, qty=0, side=side_y, notional=notional)
            oy_id = oy["order_id"]

            # Leg X
            if side_x == "sell" and price_x > 0:
                qty_x = max(1, int(notional / price_x))
                ox = submit_market_order(sym_x, qty=qty_x, side="sell", notional=None)
            else:
                ox = submit_market_order(sym_x, qty=0, side=side_x, notional=notional)
            ox_id = ox["order_id"]

            return OrderResult(
                decision=decision,
                leg_y_order_id=oy_id,
                leg_x_order_id=ox_id,
                success=True,
            )
        except Exception as exc:
            logger.error(f"OPEN {sym_y}/{sym_x} failed: {exc}")
            # Best effort: cancel any partial orders
            from stat_arb.execution.alpaca_client import cancel_all_orders
            cancel_all_orders()
            return OrderResult(
                decision=decision,
                leg_y_order_id=None,
                leg_x_order_id=None,
                success=False,
                error=str(exc),
            )

    def _close_pair(self, decision: TradeDecision) -> OrderResult:
        """Close an existing pairs position."""
        sym_y = decision.sym_y
        sym_x = decision.sym_x

        logger.info(
            f"{'[DRY-RUN] ' if self.dry_run else ''}"
            f"CLOSE {sym_y}/{sym_x}  reason={decision.reason}"
        )

        if self.dry_run:
            return OrderResult(
                decision=decision,
                leg_y_order_id="DRY_RUN",
                leg_x_order_id="DRY_RUN",
                success=True,
            )

        try:
            ry = close_position(sym_y)
            rx = close_position(sym_x)
            return OrderResult(
                decision=decision,
                leg_y_order_id=ry["order_id"],
                leg_x_order_id=rx["order_id"],
                success=True,
            )
        except Exception as exc:
            logger.error(f"CLOSE {sym_y}/{sym_x} failed: {exc}")
            return OrderResult(
                decision=decision,
                leg_y_order_id=None,
                leg_x_order_id=None,
                success=False,
                error=str(exc),
            )

    def print_summary(self, decisions: list[TradeDecision]) -> None:
        """Print a human-readable order summary."""
        opens  = [d for d in decisions if d.action == "OPEN"]
        closes = [d for d in decisions if d.action == "CLOSE"]
        keeps  = [d for d in decisions if d.action == "KEEP"]

        prefix = "[DRY-RUN] " if self.dry_run else ""
        print(f"\n{prefix}=== Order Summary ===")
        print(f"  OPEN  : {len(opens)}  pairs")
        print(f"  CLOSE : {len(closes)} pairs")
        print(f"  KEEP  : {len(keeps)}  pairs (no action)")

        for d in opens:
            side = "LONG SPREAD" if d.direction == 1 else "SHORT SPREAD"
            print(f"    → OPEN  {d.sym_y}/{d.sym_x} [{side}]  z={d.z_score:.2f}  ${d.notional:,.0f}")
        for d in closes:
            print(f"    → CLOSE {d.sym_y}/{d.sym_x}  {d.reason}")
        print()
