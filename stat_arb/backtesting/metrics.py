"""Performance metrics for walk-forward backtesting results.

All metrics assume daily return series (252 trading days per year).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(
    returns: pd.Series,
    risk_free_annual: float = 0.0,
    ann_factor: int = 252,
) -> dict:
    """Compute standard backtest performance metrics from a daily return series.

    Parameters
    ----------
    returns          : Daily portfolio return series (decimal, e.g. 0.01 = 1%).
    risk_free_annual : Annual risk-free rate (default 0). Used for Sharpe.
    ann_factor       : Trading days per year (default 252).

    Returns
    -------
    dict with keys:
      annualized_return, annualized_vol, sharpe_ratio, max_drawdown,
      calmar_ratio, win_rate, total_return, n_trading_days, avg_daily_return,
      skewness, kurtosis.
    """
    returns = returns.dropna()
    n = len(returns)

    if n < 2:
        return {k: np.nan for k in [
            "annualized_return", "annualized_vol", "sharpe_ratio",
            "max_drawdown", "calmar_ratio", "win_rate",
            "total_return", "n_trading_days", "avg_daily_return",
            "skewness", "kurtosis",
        ]}

    rf_daily = risk_free_annual / ann_factor

    ann_return = float(returns.mean() * ann_factor)
    ann_vol    = float(returns.std() * np.sqrt(ann_factor))
    sharpe     = float((returns.mean() - rf_daily) / returns.std() * np.sqrt(ann_factor))

    cum = (1.0 + returns).cumprod()
    roll_max = cum.cummax()
    drawdown  = (cum - roll_max) / roll_max
    max_dd    = float(drawdown.min())

    calmar = float(ann_return / abs(max_dd)) if max_dd != 0 else np.nan
    win_rate = float((returns > 0).mean())
    total_ret = float(cum.iloc[-1] - 1.0)

    return {
        "annualized_return": ann_return,
        "annualized_vol":    ann_vol,
        "sharpe_ratio":      sharpe,
        "max_drawdown":      max_dd,
        "calmar_ratio":      calmar,
        "win_rate":          win_rate,
        "total_return":      total_ret,
        "n_trading_days":    n,
        "avg_daily_return":  float(returns.mean()),
        "skewness":          float(returns.skew()),
        "kurtosis":          float(returns.kurtosis()),
    }


def annualized_sharpe(returns: pd.Series, ann_factor: int = 252) -> float:
    """Annualised Sharpe ratio (zero risk-free rate)."""
    r = returns.dropna()
    if r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(ann_factor))


def max_drawdown(returns: pd.Series) -> float:
    """Peak-to-trough maximum drawdown (negative number)."""
    cum = (1.0 + returns.dropna()).cumprod()
    return float(((cum - cum.cummax()) / cum.cummax()).min())


def equity_curve(returns: pd.Series, starting_value: float = 1.0) -> pd.Series:
    """Convert daily returns to an equity curve starting at starting_value."""
    return starting_value * (1.0 + returns.dropna()).cumprod()
