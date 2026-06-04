"""Live paper-trading execution loop.

Run daily after market close (4:30pm+):
    python -m stat_arb.execution.run              # dry-run (default, no orders)
    python -m stat_arb.execution.run --live       # submit real paper orders
    python -m stat_arb.execution.run --status     # show portfolio status only

SAFETY:
  - Paper trading ONLY (hard-coded to paper-api.alpaca.markets)
  - Default is --dry-run (preview only)
  - All trades logged to data/trades/
  - Risk limits enforced before every order
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from stat_arb.execution.alpaca_client import (
    cancel_all_orders,
    close_all_positions,
    get_account,
    get_all_positions,
)
from stat_arb.execution.order_manager import OrderManager
from stat_arb.execution.portfolio_manager import PortfolioManager
from stat_arb.execution.signal_generator import SignalGenerator
from stat_arb.risk.limits import PortfolioState, RiskChecker, RiskLimits, RiskAction

# ── Logging setup ─────────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    level="INFO",
)
logger.add(
    "data/execution/execution.log",
    rotation="10 MB", retention=30, level="DEBUG",
    encoding="utf-8",
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Live paper-trading execution loop for Statistical Arbitrage System."
    )
    p.add_argument("--live",   action="store_true", default=False,
                   help="Submit real paper orders (default: dry-run preview only).")
    p.add_argument("--status", action="store_true", default=False,
                   help="Show portfolio status only, no signal computation.")
    p.add_argument("--date",   default=None,
                   help="Compute signals for this date (YYYY-MM-DD). Default: today.")
    p.add_argument("--min-portfolio-sharpe", type=float, default=1.28,
                   help="IS portfolio Sharpe threshold (default 1.28).")
    p.add_argument("--close-all", action="store_true", default=False,
                   help="Close ALL open positions and cancel all orders. USE WITH CARE.")
    p.add_argument("--update-data", action="store_true", default=False,
                   help="Download latest daily prices before computing signals.")
    return p.parse_args()


def _print_account(account: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  ALPACA PAPER TRADING ACCOUNT")
    print(f"{'='*60}")
    print(f"  Status       : {account['status']}")
    print(f"  Equity       : ${account['equity']:>12,.2f}")
    print(f"  Cash         : ${account['cash']:>12,.2f}")
    print(f"  Buying power : ${account['buying_power']:>12,.2f}")
    print(f"{'='*60}\n")


def _print_positions(positions: dict, state_summary: dict) -> None:
    print(f"  Open Alpaca positions: {len(positions)}")
    for sym, p in positions.items():
        sign = "+" if p["unrealized_pl"] >= 0 else ""
        print(f"    {sym:<8} {p['side']:<6} qty={p['qty']:.2f}  "
              f"value=${p['market_value']:>9,.2f}  P&L={sign}{p['unrealized_pl']:.2f}")
    print()
    print(f"  Internal state: {state_summary['n_open']} pairs open")
    for pos in state_summary.get("positions", []):
        d_str = "LONG" if pos["direction"] == 1 else "SHORT"
        print(f"    {pos['pair']:<20} [{d_str}]  entry={pos['entry_date']}")
    print()


def main() -> None:
    args = _parse_args()
    dry_run = not args.live

    if dry_run and not args.status and not args.close_all:
        print("\n[DRY-RUN MODE] No orders will be submitted. Use --live to execute.\n")

    # ── Optional data refresh ────────────────────────────────────────────────
    if args.update_data:
        logger.info("Updating daily price data...")
        import subprocess
        subprocess.run(
            ["python", "-m", "stat_arb.data.run",
             "--interval", "1d", "--start", "2010-01-01", "--end", "2025-12-31"],
            check=False,
        )
        logger.info("Data update complete")

    # ── Account check ─────────────────────────────────────────────────────────
    try:
        account = get_account()
    except Exception as exc:
        logger.error(f"Cannot connect to Alpaca: {exc}")
        sys.exit(1)

    _print_account(account)

    pm = PortfolioManager()

    # ── Status-only mode ──────────────────────────────────────────────────────
    if args.status:
        positions = get_all_positions()
        _print_positions(positions, pm.summary())
        return

    # ── Close-all mode ────────────────────────────────────────────────────────
    if args.close_all:
        if dry_run:
            print("[DRY-RUN] Would close all positions and cancel all orders.")
            print("Use --live --close-all to actually execute.")
            return
        cancel_all_orders()
        close_all_positions()
        pm.close_all(reason="Manual --close-all command")
        print("All positions closed.")
        return

    # ── Risk check: portfolio drawdown ────────────────────────────────────────
    risk_limits  = RiskLimits(
        max_portfolio_drawdown=0.15,
        max_daily_loss_pct=0.03,
    )
    risk_checker = RiskChecker(risk_limits)
    port_state   = PortfolioState(
        portfolio_value=account["equity"],
        current_equity=1.0,
        peak_equity=1.0,
    )
    # Estimate drawdown from account (compare equity to $100K initial paper capital)
    initial_capital = 100_000.0
    equity_ratio    = account["equity"] / initial_capital
    port_state.current_equity = equity_ratio
    port_state.peak_equity    = max(1.0, equity_ratio)

    risk_action = risk_checker.check_portfolio_action(port_state)
    if risk_action == RiskAction.CLOSE_ALL:
        logger.warning("RISK: Portfolio drawdown exceeded limit → CLOSE_ALL")
        if not dry_run:
            cancel_all_orders()
            close_all_positions()
        pm.close_all(reason="Portfolio drawdown limit")
        print("RISK LIMIT: All positions closed due to drawdown.")
        return

    if risk_action == RiskAction.HALT_TRADING:
        logger.warning("RISK: Daily loss limit → HALT_TRADING. No new entries today.")
        print("RISK LIMIT: Daily loss too large. No new entries today.")
        # Allow exits to proceed but no new entries
        desired = []
    else:
        # ── Signal generation ─────────────────────────────────────────────────
        as_of = (datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                 if args.date else datetime.now(tz=timezone.utc))

        gen = SignalGenerator()
        desired, diagnostics = gen.compute(
            as_of=as_of,
            min_portfolio_is_sharpe=args.min_portfolio_sharpe,
        )

        logger.info(
            f"Signals: {len(desired)} entries desired  "
            f"VIX={diagnostics.get('vix', 'N/A')}"
        )
        if diagnostics.get("gated"):
            print(f"Market gate active: {diagnostics['gated']} → no new entries")
            desired = []

    # ── Reconciliation ────────────────────────────────────────────────────────
    decisions = pm.reconcile(desired)

    # ── Order submission ──────────────────────────────────────────────────────
    om = OrderManager(dry_run=dry_run)
    om.print_summary(decisions)

    if decisions:
        results = om.execute_decisions(decisions)
        n_ok  = sum(1 for r in results if r.success)
        n_err = sum(1 for r in results if not r.success)
        if n_err:
            logger.error(f"{n_err} orders failed")

    # ── Final status ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Portfolio state after this run:")
    summary = pm.summary()
    print(f"  Open pairs   : {summary['n_open']}")
    print(f"  Total notional: ${summary['total_notional']:>10,.0f}")
    print(f"  Account equity: ${account['equity']:>10,.2f}")
    print(f"{'='*60}\n")

    if dry_run:
        print("Run with --live to submit real paper orders.\n")


if __name__ == "__main__":
    main()
