"""Volatility-adjusted and Kelly-optimal position sizing for pairs trading.

Position size pipeline (v2)
---------------------------
  1. Base notional         : Fixed starting point (e.g. $50,000 per pair)
  2. Vol-adjustment        : Scale so each pair contributes equal dollar risk
                             target_vol = mean(spread_std) across portfolio
                             → scale ≈ 1 for typical pairs, <1 for volatile pairs
  3. Kelly scaling         : Scale using TRADE-LEVEL win/loss stats from IS simulation
                             (NOT daily spread returns which have near-zero edge)
  4. Risk scalar           : Further reduce if portfolio is in warning zone
  5. Adaptive notional     : Scale down proportionally when many pairs open (budget constraint)
  6. Clip to [min, max]    : Hard floor and ceiling

Key fixes over v1
-----------------
  v1 bug 1: target_vol=1.0% vs actual spread_std=1.337% → scale DOWN notional
            → Use target_vol=spread_std_daily to neutralise vol-adj when desired
            → Or use target_vol slightly below spread_std to achieve modest reduction

  v1 bug 2: Kelly from DAILY spread returns (win_rate≈49% → Kelly≈0)
            → Use TRADE-LEVEL stats: P&L per completed round-trip trade
            → Trade-level has positive edge if z-score timing works (win_rate 55-65%)

  v1 bug 3: Total notional limit blocks entries when n_pairs > max_pairs
            → Use adaptive_notional_per_pair() to fit all pairs within budget
"""

from __future__ import annotations

import numpy as np


# ── Kelly criterion ───────────────────────────────────────────────────────────

def kelly_fraction(
    win_rate: float,
    avg_win_pct: float,
    avg_loss_pct: float,
    kelly_multiplier: float = 0.5,
) -> float:
    """Kelly criterion for optimal bet sizing.

    Full Kelly: f* = (p·W − (1−p)·L) / W
    Applied fraction: kelly_multiplier × f*  (default half-Kelly)

    Parameters
    ----------
    win_rate        : Fraction of TRADES (not days) that are profitable.
    avg_win_pct     : Average profit per winning trade (positive, e.g. 0.005).
    avg_loss_pct    : Average loss per losing trade (positive magnitude).
    kelly_multiplier : Safety factor — 0.5 (half-Kelly) standard for live trading.

    Returns
    -------
    Kelly fraction ∈ [0, 1].
    """
    if avg_win_pct <= 0 or avg_loss_pct <= 0 or not (0 < win_rate < 1):
        return 0.0
    p = float(win_rate)
    W = float(avg_win_pct)
    L = float(avg_loss_pct)
    f_full = (p * W - (1 - p) * L) / W
    return float(np.clip(kelly_multiplier * f_full, 0.0, 1.0))


def kelly_from_trade_stats(
    trade_pnls: np.ndarray,
    kelly_multiplier: float = 0.5,
    min_trades: int = 5,
) -> float:
    """Compute Kelly fraction from actual trade P&L array.

    Each element = P&L of one completed round-trip trade (entry → exit).

    Parameters
    ----------
    trade_pnls      : 1-D array of trade P&Ls as fraction of notional.
    kelly_multiplier : Half-Kelly safety factor.
    min_trades      : Minimum trades required for reliable Kelly estimate.

    Returns
    -------
    Kelly fraction ∈ [0, 1].  0 if insufficient data or no edge.
    """
    t = np.asarray(trade_pnls, dtype=float)
    t = t[~np.isnan(t)]
    if len(t) < min_trades:
        return 0.0

    wins   = t[t > 0]
    losses = t[t < 0]

    if len(wins) == 0 or len(losses) == 0:
        return 0.0

    win_rate  = float(len(wins) / len(t))
    avg_win   = float(wins.mean())
    avg_loss  = float(abs(losses.mean()))

    return kelly_fraction(win_rate, avg_win, avg_loss, kelly_multiplier)


def kelly_from_sharpe(
    sharpe: float,
    kelly_multiplier: float = 0.5,
    min_fraction: float = 0.5,
    max_fraction: float = 1.5,
) -> float:
    """Fallback: approximate Kelly multiplier from IS Sharpe.

    Used when trade-level P&L stats are not available.
    Returns a SIZING MULTIPLIER (not a capital fraction).

    Sharpe ≥ 1.4 → multiplier > 1.0 (size up relative to base)
    Sharpe ≈ 0.5 → multiplier ≈ min_fraction (size at floor)
    Sharpe < 0   → multiplier = min_fraction (don't penalise further)
    """
    if sharpe <= 0:
        return float(min_fraction)
    # sqrt(Sharpe / 0.7) gives 1.0 at Sharpe=0.7, 1.4 at Sharpe=1.4
    raw = np.sqrt(np.clip(sharpe, 0, 3) / 0.7)
    return float(np.clip(kelly_multiplier * raw / 0.5, min_fraction, max_fraction))


# ── Volatility-adjusted sizing ────────────────────────────────────────────────

def volatility_adjusted_notional(
    base_notional:    float,
    spread_std_daily: float,
    target_vol:       float = 0.013,   # ~mean spread_std for S&P500 same-sector pairs
    min_notional:     float = 5_000.0,
    max_notional:     float = 100_000.0,
) -> float:
    """Scale base notional so each pair contributes equal dollar risk.

    Formula: notional_adj = base × (target_vol / spread_std)

    When target_vol == spread_std → scale = 1.0 → no change (safe default).
    When target_vol < spread_std  → scale < 1.0 → reduce volatile pairs.
    When target_vol > spread_std  → scale > 1.0 → increase calm pairs.

    v2 calibration: target_vol=0.013 ≈ mean spread_std of S&P 500 same-sector pairs.
    This neutralises vol-adj for typical pairs and modestly reduces outliers.

    Parameters
    ----------
    base_notional    : Starting notional ($).
    spread_std_daily : Daily std of the OLS spread (from IS period).
    target_vol       : Target spread vol. Default 1.3% ≈ observed S&P 500 mean.
    min_notional     : Floor on adjusted notional.
    max_notional     : Cap on adjusted notional.
    """
    if spread_std_daily <= 1e-8:
        return float(base_notional)
    scale    = target_vol / spread_std_daily
    adjusted = base_notional * scale
    return float(np.clip(adjusted, min_notional, max_notional))


def adaptive_notional_per_pair(
    base_notional:   float,
    total_budget:    float,
    n_pairs_target:  int,
    min_notional:    float = 5_000.0,
) -> float:
    """Scale notional so all target pairs fit within total budget.

    When n_pairs_target × base_notional > total_budget, each pair's notional
    is reduced proportionally so the portfolio stays within budget.

    Example
    -------
    18 pairs × $50K = $900K > budget $750K
    → scale = 750K / (18 × 50K) = 0.833
    → per-pair notional = 50K × 0.833 = $41.7K
    → 18 × $41.7K = $750K (fits budget)
    """
    if n_pairs_target <= 0:
        return float(base_notional)
    required = n_pairs_target * base_notional
    if required <= total_budget:
        return float(base_notional)
    scale = total_budget / required
    return float(max(base_notional * scale, min_notional))


# ── Full sizing pipeline ──────────────────────────────────────────────────────

def compute_position_size(
    base_notional:    float,
    spread_std_daily: float,
    is_sharpe:        float,
    risk_scalar:      float = 1.0,
    target_vol:       float = 0.013,   # v2: calibrated to actual spread_std
    kelly_multiplier: float = 0.5,
    min_notional:     float = 5_000.0,
    max_notional:     float = 100_000.0,
    trade_pnls:       "np.ndarray | None" = None,
    n_open_pairs:     int = 0,
    total_budget:     float = 0.0,
    conservative:     bool = True,
) -> float:
    """Full position sizing pipeline (v2).

    Steps
    -----
    1. Adaptive notional: scale down if many pairs open (budget constraint).
    2. Conservative vol-adjustment: reduce volatile pairs, never scale up.
    3. Risk scalar: reduce if portfolio approaching drawdown limits.
    4. Clip to [min, max].

    conservative=True (default — recommended)
    -----------------------------------------
    Scale = min(1.0, target_vol / spread_std).
    - Volatile pair (spread_std > target_vol): reduce notional proportionally.
    - Calm pair (spread_std ≤ target_vol): keep base notional (no scaling up).
    Rationale: IS Sharpe is NOT a reliable predictor of OOS performance.
    Using IS Sharpe to SIZE UP "good" pairs amplifies IS→OOS overfitting.
    The safer approach is to only PENALISE unusually volatile pairs.

    conservative=False
    ------------------
    Full pipeline: vol-adj × Kelly(IS Sharpe or trade P&Ls) × risk_scalar.
    Use only when you have evidence that IS Sharpe reliably predicts OOS.
    """
    if risk_scalar <= 0:
        return 0.0

    # Step 1: adaptive notional (budget constraint)
    if total_budget > 0 and n_open_pairs > 0:
        base_notional = adaptive_notional_per_pair(
            base_notional, total_budget, n_open_pairs, min_notional
        )

    if conservative:
        # Conservative: only reduce volatile pairs, never scale up calm pairs
        if spread_std_daily > 1e-8 and target_vol > 0:
            scale = min(1.0, target_vol / spread_std_daily)
            sized = base_notional * scale * risk_scalar
        else:
            sized = base_notional * risk_scalar
        return float(np.clip(sized, 0 if risk_scalar <= 0 else min_notional, max_notional))

    # Full pipeline (non-conservative)
    # Step 2: vol-adjusted notional
    vol_adj = volatility_adjusted_notional(
        base_notional, spread_std_daily, target_vol,
        min_notional, max_notional,
    )

    # Step 3: Kelly scale — prefer trade-level, fallback to Sharpe approximation
    if trade_pnls is not None and len(trade_pnls) >= 5:
        k = 1.0 + kelly_from_trade_stats(trade_pnls, kelly_multiplier) * 2.0
        k = float(np.clip(k, 0.3, 2.0))
    else:
        k = kelly_from_sharpe(is_sharpe, kelly_multiplier, min_fraction=0.5)

    # Step 4: risk scalar
    sized = vol_adj * k * risk_scalar

    # Step 5: clip
    return float(np.clip(sized, 0 if risk_scalar <= 0 else min_notional, max_notional))


# ── Spread statistics from IS data ───────────────────────────────────────────

def estimate_spread_stats(
    spread_returns: np.ndarray,
    warmup: int = 20,
) -> dict[str, float]:
    """Compute raw P&L statistics from IS spread DAILY returns.

    Note: These are DAILY return stats, not trade-level stats.
    For proper Kelly, use kelly_from_trade_stats() with trade P&L.

    Returns dict: win_rate, avg_win_pct, avg_loss_pct, daily_std, sharpe.
    """
    r = np.asarray(spread_returns, dtype=float)[warmup:]
    r = r[~np.isnan(r)]

    if len(r) < 10:
        return {"win_rate": 0.5, "avg_win_pct": 0.01, "avg_loss_pct": 0.01,
                "daily_std": 0.01, "sharpe": 0.0}

    wins   = r[r > 0]
    losses = r[r < 0]

    win_rate   = float(len(wins) / len(r)) if len(r) > 0 else 0.5
    avg_win    = float(wins.mean())        if len(wins) > 0 else 0.01
    avg_loss   = float(abs(losses.mean())) if len(losses) > 0 else 0.01
    daily_std  = float(r.std()) if r.std() > 0 else 0.01
    sharpe_val = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0

    return {
        "win_rate":    win_rate,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "daily_std":   daily_std,
        "sharpe":      sharpe_val,
    }


def compute_is_trade_pnls(
    log_y:    np.ndarray,
    log_x:    np.ndarray,
    ols_beta: float,
    zscores:  np.ndarray,
    entry_z:  float = 1.5,
    exit_z:   float = 0.3,
    stop_z:   float = 3.0,
    warmup:   int = 30,
) -> np.ndarray:
    """Compute P&L per completed trade from IS simulation.

    Each element = total log P&L of one round-trip (entry → exit).
    Uses the same z-score strategy as the backtesting engine.

    Returns
    -------
    np.ndarray of trade P&Ls (as fraction of notional).
    Empty array if no trades completed.
    """
    n = len(log_y)
    position  = 0
    entry_t   = 0
    cum_pnl   = 0.0
    trade_pnls = []

    for t in range(warmup + 1, n):
        z     = float(zscores[t])
        new_p = position

        if position == 0:
            if z < -entry_z:
                new_p, entry_t, cum_pnl = 1, t, 0.0
            elif z > entry_z:
                new_p, entry_t, cum_pnl = -1, t, 0.0
        else:
            if abs(z) < exit_z or abs(z) > stop_z:
                new_p = 0
                trade_pnls.append(cum_pnl)

        if position != 0:
            sr = (log_y[t] - log_y[t - 1]) - ols_beta * (log_x[t] - log_x[t - 1])
            cum_pnl += position * sr

        position = new_p

    return np.array(trade_pnls, dtype=float)
