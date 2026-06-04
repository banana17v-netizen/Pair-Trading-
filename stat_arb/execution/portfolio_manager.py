"""Portfolio manager — tracks open pairs positions and reconciles with Alpaca.

State persistence: data/execution/state.json
Audit log:        data/trades/YYYYMMDD.json

SAFETY: This is LIVE CODE. State changes must be logged permanently.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

from stat_arb.execution.signal_generator import DesiredPosition
from stat_arb.risk.limits import PortfolioState, RiskChecker, RiskLimits

STATE_FILE = Path("data/execution/state.json")
TRADES_DIR = Path("data/trades")


@dataclass
class OpenPosition:
    """A currently open pairs position."""
    sym_y:       str
    sym_x:       str
    direction:   int     # +1 long spread, -1 short spread
    entry_date:  str
    entry_z:     float
    ols_alpha:   float
    ols_beta:    float
    notional:    float
    sector:      str


@dataclass
class TradeDecision:
    """A single trade action to execute."""
    action:    str      # "OPEN" or "CLOSE" or "KEEP"
    sym_y:     str
    sym_x:     str
    direction: int      # +1 or -1 (for OPEN); 0 (for CLOSE)
    notional:  float
    reason:    str
    z_score:   float = 0.0


class PortfolioManager:
    """Manages the state of open pairs positions.

    Usage
    -----
    pm = PortfolioManager()
    decisions = pm.reconcile(desired_positions, current_z_scores)
    """

    def __init__(
        self,
        max_pairs: int   = 15,
        max_notional: float = 750_000.0,
    ) -> None:
        self.max_pairs    = max_pairs
        self.max_notional = max_notional
        self._positions   = self._load_state()

    # ── State persistence ─────────────────────────────────────────────────────

    def _load_state(self) -> dict[str, OpenPosition]:
        if not STATE_FILE.exists():
            return {}
        try:
            data = json.loads(STATE_FILE.read_text())
            return {
                k: OpenPosition(**v)
                for k, v in data.get("positions", {}).items()
            }
        except Exception as exc:
            logger.error(f"Could not load state: {exc}. Starting fresh.")
            return {}

    def _save_state(self) -> None:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "positions": {k: asdict(v) for k, v in self._positions.items()},
            "last_update": datetime.now(tz=timezone.utc).isoformat(),
        }
        STATE_FILE.write_text(json.dumps(data, indent=2))

    def _log_trade(self, decision: TradeDecision) -> None:
        """Append trade to permanent daily audit log."""
        TRADES_DIR.mkdir(parents=True, exist_ok=True)
        today = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
        log_file = TRADES_DIR / f"{today}.json"

        record = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "action":    decision.action,
            "sym_y":     decision.sym_y,
            "sym_x":     decision.sym_x,
            "direction": decision.direction,
            "notional":  decision.notional,
            "z_score":   decision.z_score,
            "reason":    decision.reason,
        }

        # Append to daily log (newline-delimited JSON)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def open_positions(self) -> dict[str, OpenPosition]:
        return dict(self._positions)

    @property
    def n_open(self) -> int:
        return len(self._positions)

    @property
    def total_notional(self) -> float:
        return sum(abs(p.notional) for p in self._positions.values())

    def pair_key(self, sym_y: str, sym_x: str) -> str:
        return f"{sym_y}/{sym_x}"

    # ── Reconciliation ────────────────────────────────────────────────────────

    def reconcile(
        self,
        desired:        list[DesiredPosition],
        exit_z_scores:  dict[str, float] | None = None,
        exit_z:         float = 0.3,
        stop_z:         float = 3.5,
    ) -> list[TradeDecision]:
        """Compare desired vs current → return list of trade decisions.

        Parameters
        ----------
        desired       : List of DesiredPosition from SignalGenerator.
        exit_z_scores : Current z-scores for open positions (key=sym_y/sym_x).
        exit_z        : Exit when |z| drops below this.
        stop_z        : Stop-loss when |z| exceeds this.

        Returns
        -------
        List of TradeDecision (OPEN, CLOSE, KEEP).
        """
        decisions: list[TradeDecision] = []
        desired_keys = {self.pair_key(d.sym_y, d.sym_x): d for d in desired}

        # ── Step 1: check exits on current positions ──────────────────────────
        for key, pos in list(self._positions.items()):
            z = (exit_z_scores or {}).get(key, None)

            # Check exit conditions
            should_exit = False
            reason      = ""

            if z is not None:
                if abs(z) < exit_z:
                    should_exit = True
                    reason      = f"z={z:.2f} crossed back through exit_z={exit_z}"
                elif abs(z) > stop_z:
                    should_exit = True
                    reason      = f"STOP LOSS: z={z:.2f} > stop_z={stop_z}"

            # Also exit if pair no longer in desired (regime changed, etc.)
            if key not in desired_keys and z is None:
                should_exit = True
                reason      = "Pair no longer in desired portfolio"

            if should_exit:
                decision = TradeDecision(
                    action="CLOSE", sym_y=pos.sym_y, sym_x=pos.sym_x,
                    direction=0, notional=pos.notional,
                    reason=reason, z_score=z or 0.0,
                )
                decisions.append(decision)
                self._log_trade(decision)
                del self._positions[key]
            else:
                decisions.append(TradeDecision(
                    action="KEEP", sym_y=pos.sym_y, sym_x=pos.sym_x,
                    direction=pos.direction, notional=pos.notional,
                    reason="Within limits, regime OK", z_score=z or 0.0,
                ))

        # ── Step 2: check new entries from desired ────────────────────────────
        for key, desired_pos in desired_keys.items():
            if key in self._positions:
                continue   # already open

            # Capacity check
            if self.n_open >= self.max_pairs:
                logger.debug(f"Skip {key}: max pairs {self.max_pairs} reached")
                continue
            if self.total_notional + desired_pos.notional > self.max_notional:
                logger.debug(f"Skip {key}: total notional limit")
                continue

            decision = TradeDecision(
                action="OPEN", sym_y=desired_pos.sym_y, sym_x=desired_pos.sym_x,
                direction=desired_pos.direction, notional=desired_pos.notional,
                reason=f"z={desired_pos.z_score:.2f}, regime={desired_pos.regime}",
                z_score=desired_pos.z_score,
            )
            decisions.append(decision)
            self._log_trade(decision)

            # Record in state
            self._positions[key] = OpenPosition(
                sym_y=desired_pos.sym_y, sym_x=desired_pos.sym_x,
                direction=desired_pos.direction,
                entry_date=datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
                entry_z=desired_pos.z_score,
                ols_alpha=desired_pos.ols_alpha,
                ols_beta=desired_pos.ols_beta,
                notional=desired_pos.notional,
                sector=desired_pos.sector,
            )

        self._save_state()
        return decisions

    def close_all(self, reason: str = "CLOSE_ALL risk action") -> list[TradeDecision]:
        """Close all positions (used for CLOSE_ALL risk action)."""
        decisions = []
        for key, pos in list(self._positions.items()):
            d = TradeDecision(
                action="CLOSE", sym_y=pos.sym_y, sym_x=pos.sym_x,
                direction=0, notional=pos.notional, reason=reason,
            )
            decisions.append(d)
            self._log_trade(d)
        self._positions.clear()
        self._save_state()
        logger.warning(f"CLOSE_ALL: closed {len(decisions)} positions. Reason: {reason}")
        return decisions

    def summary(self) -> dict:
        return {
            "n_open":        self.n_open,
            "total_notional": self.total_notional,
            "positions":      [
                {"pair": k, "direction": v.direction, "entry_date": v.entry_date}
                for k, v in self._positions.items()
            ],
        }
