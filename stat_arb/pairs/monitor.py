"""Ongoing cointegration health checks for active pairs.

Called periodically (every retrain_freq days = 21) during live trading
to detect pairs that have lost their cointegration relationship.

Uses a looser EG threshold (0.10) to avoid premature position exits.
A pair is flagged as unhealthy only if it fails the test on the most
recent `window_days` of data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from stat_arb.pairs.select import engle_granger_test, compute_half_life


def check_pair_health(
    prices_y: pd.Series,
    prices_x: pd.Series,
    window_days: int = 252,
    eg_threshold: float = 0.10,
    max_half_life: float = 60.0,
) -> dict:
    """Check whether a single pair is still cointegrated.

    Uses the most recent `window_days` of data. Applies a looser EG
    threshold (0.10) than initial selection (0.05) to reduce false exits.

    Parameters
    ----------
    prices_y, prices_x : Adjusted close price series (DatetimeIndex).
    window_days        : Rolling window in trading days (default 252 = 1 year).
    eg_threshold       : EG p-value cutoff for health check (default 0.10).
    max_half_life      : Maximum acceptable half-life in days (default 60).

    Returns
    -------
    Dict with keys: healthy (bool), eg_pvalue, half_life, n_obs, reason.
    """
    pair = pd.DataFrame({"y": prices_y, "x": prices_x}).dropna()

    if len(pair) < window_days // 2:
        return {
            "healthy": False,
            "eg_pvalue": np.nan,
            "half_life": np.nan,
            "n_obs": len(pair),
            "reason": f"insufficient data ({len(pair)} < {window_days // 2})",
        }

    # Use most recent window
    recent = pair.iloc[-window_days:]
    log_y  = np.log(recent["y"].values)
    log_x  = np.log(recent["x"].values)
    n_obs  = len(recent)

    try:
        p_val, _, ols_beta = engle_granger_test(log_y, log_x)
    except Exception as exc:
        return {
            "healthy": False,
            "eg_pvalue": np.nan,
            "half_life": np.nan,
            "n_obs": n_obs,
            "reason": f"EG test failed: {exc}",
        }

    spread    = log_y - ols_beta * log_x
    half_life = compute_half_life(spread)

    eg_ok = p_val < eg_threshold
    hl_ok = np.isfinite(half_life) and 0 < half_life <= max_half_life
    healthy = eg_ok and hl_ok

    reason = "OK"
    if not eg_ok:
        reason = f"cointegration lost (EG p={p_val:.3f} > {eg_threshold})"
    elif not hl_ok:
        reason = f"half-life out of range ({half_life:.1f}d, max={max_half_life}d)"

    return {
        "healthy":   healthy,
        "eg_pvalue": float(p_val),
        "half_life": float(half_life),
        "n_obs":     n_obs,
        "reason":    reason,
    }


def monitor_pairs(
    active_pairs: list[tuple[str, str]],
    universe_close: pd.DataFrame,
    window_days: int = 252,
    eg_threshold: float = 0.10,
    max_half_life: float = 60.0,
) -> pd.DataFrame:
    """Check health of all active pairs.

    Parameters
    ----------
    active_pairs   : List of (symbol_y, symbol_x) tuples currently in portfolio.
    universe_close : DataFrame of adjusted close prices (columns = symbols).
    window_days    : Rolling window for each health check.
    eg_threshold   : Loose EG threshold for ongoing monitoring.
    max_half_life  : Max acceptable half-life.

    Returns
    -------
    DataFrame indexed by (symbol_y, symbol_x) with health check results.
    Column 'healthy' is the primary signal — False means consider closing.
    """
    rows = []
    for sym_y, sym_x in active_pairs:
        if sym_y not in universe_close.columns or sym_x not in universe_close.columns:
            logger.warning(f"monitor_pairs: {sym_y} or {sym_x} not in universe")
            continue

        health = check_pair_health(
            universe_close[sym_y],
            universe_close[sym_x],
            window_days=window_days,
            eg_threshold=eg_threshold,
            max_half_life=max_half_life,
        )
        rows.append({
            "symbol_y":  sym_y,
            "symbol_x":  sym_x,
            "healthy":   health["healthy"],
            "eg_pvalue": health["eg_pvalue"],
            "half_life": health["half_life"],
            "n_obs":     health["n_obs"],
            "reason":    health["reason"],
        })
        status = "OK" if health["healthy"] else f"UNHEALTHY — {health['reason']}"
        logger.debug(f"{sym_y}/{sym_x}: {status}")

    if not rows:
        empty = pd.DataFrame(
            columns=["healthy", "eg_pvalue", "half_life", "n_obs", "reason"]
        )
        empty.index = pd.MultiIndex.from_tuples(
            [], names=["symbol_y", "symbol_x"]
        )
        return empty

    df = pd.DataFrame(rows).set_index(["symbol_y", "symbol_x"])
    n_unhealthy = (~df["healthy"]).sum()
    if n_unhealthy:
        logger.warning(f"monitor_pairs: {n_unhealthy}/{len(df)} pairs are unhealthy")
    return df
