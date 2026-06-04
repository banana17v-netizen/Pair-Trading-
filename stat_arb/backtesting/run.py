"""CLI entrypoint — walk-forward backtest.

Usage
-----
# Default settings (2y IS / 6m OOS, 2010-2025, same-sector pairs)
python -m stat_arb.backtesting.run

# Custom window / signals
python -m stat_arb.backtesting.run --start 2015-01-01 --end 2025-01-01 \\
    --is-years 2 --oos-months 6 --entry-z 2.0 --exit-z 0.5

# Parallel pair selection (speeds up IS screening per fold)
python -m stat_arb.backtesting.run --n-jobs 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from stat_arb.backtesting.engine import WalkForwardBacktest, WalkForwardConfig

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    level="INFO",
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run walk-forward backtest for the pairs trading system."
    )
    p.add_argument("--data-dir",    default="data/ohlcv/1d",   help="Daily parquet folder.")
    p.add_argument("--out",         default="data/backtest_results.parquet",
                   help="Output parquet for daily returns.")
    p.add_argument("--start",       default="2010-01-01",      help="Backtest start date.")
    p.add_argument("--end",         default="2025-01-01",      help="Backtest end date.")
    p.add_argument("--is-years",    type=int,   default=2,     help="In-sample window (years).")
    p.add_argument("--oos-months",  type=int,   default=6,     help="OOS window (months).")
    p.add_argument("--max-pairs",   type=int,   default=15,    help="Max concurrent pairs.")
    p.add_argument("--notional",    type=float, default=50_000,help="Notional per pair ($).")
    p.add_argument("--entry-z",     type=float, default=2.0,   help="Entry z-score threshold.")
    p.add_argument("--exit-z",      type=float, default=0.5,   help="Exit z-score threshold.")
    p.add_argument("--stop-z",      type=float, default=3.5,   help="Stop-loss z-score.")
    p.add_argument("--tc-bps",      type=float, default=10.0,  help="Transaction cost (bps/side).")
    p.add_argument("--eg-pvalue",   type=float, default=0.05,  help="EG p-value threshold.")
    p.add_argument("--max-halflife",type=float, default=120.0, help="Max half-life (days).")
    p.add_argument("--n-jobs",        type=int,   default=1,     help="Parallel workers for pair screening.")
    p.add_argument("--rolling-z-win", type=int,   default=10,    help="Rolling z-score window (days). Layer 4 default: 10d.")
    p.add_argument("--min-is-sharpe",        type=float, default=1.0,  help="Min per-pair IS Sharpe. Layer 2 default: 1.0.")
    p.add_argument("--min-portfolio-sharpe", type=float, default=0.0,  help="Min portfolio IS Sharpe to trade a fold (0=any positive). Default: 0.0.")
    p.add_argument("--strict-regime",   dest="strict_regime",   action="store_true",  default=False,
                   help="Strict per-pair HMM: enter only in MEAN_REVERTING.")
    p.add_argument("--no-strict-regime",dest="strict_regime",   action="store_false",
                   help="Loosen per-pair HMM: enter if not VOLATILE (default).")
    # Market-level regime filter
    p.add_argument("--market-filter",    dest="market_filter",   action="store_true",  default=True,
                   help="Enable market-level HMM gate (default ON).")
    p.add_argument("--no-market-filter", dest="market_filter",   action="store_false",
                   help="Disable market HMM gate (baseline comparison).")
    p.add_argument("--close-volatile",   dest="close_volatile",  action="store_true",  default=False,
                   help="Force-close pairs when market is VOLATILE (default: block entries only).")
    p.add_argument("--no-close-volatile",dest="close_volatile",  action="store_false",
                   help="Block new entries only on VOLATILE market (default).")
    p.add_argument("--vol-window",     type=int,   default=10,    help="Smoothing window for volatile fraction (days). Default 10.")
    p.add_argument("--vol-threshold",  type=float, default=0.30,  help="Gate ON when >=X of window is volatile. Default 0.30.")
    p.add_argument("--cooldown",       type=int,   default=5,     help="Days to stay gated after volatile clears. Default 5.")
    # VIX gate (Layer 1)
    p.add_argument("--vix-threshold",  type=float, default=20.0,
                   help="VIX > threshold → block (stress/crisis). Default 20.")
    p.add_argument("--vix-min",        type=float, default=11.0,
                   help="VIX < min → block (extreme complacency = bull mkt). Default 11.")
    p.add_argument("--no-vix",         dest="no_vix", action="store_true", default=False,
                   help="Skip loading VIX data (use HMM market gate only).")
    p.add_argument("--no-etfs",        dest="no_etfs", action="store_true", default=False,
                   help="Exclude sector ETFs from universe (stock-vs-stock only).")
    p.add_argument("--momentum-window",    type=int,   default=60,    help="Market momentum window (days). Default 60.")
    p.add_argument("--momentum-threshold", type=float, default=0.0,   help="Gate when market up >X in window (0=off). Default 0.")
    p.add_argument("--min-dispersion",     type=float, default=0.006,
                   help="Block when cross-sectional daily std < X (0=off). Default 0.006.")
    return p.parse_args()


def _load_prices(data_dir: str, include_etfs: bool = True) -> pd.DataFrame:
    """Load adjusted close prices for stocks (and optionally sector ETFs).

    Sector ETFs are loaded from data/ohlcv/etf/ when available.
    Stock-vs-ETF pairs have near-perfect cointegration (stock IS in ETF),
    making the spread a pure idiosyncratic return with faster mean-reversion.
    """
    p = Path(data_dir)
    if not p.exists():
        logger.error(f"Data directory not found: {p}")
        sys.exit(1)
    dfs = {}
    for f in sorted(p.glob("*.parquet")):
        df = pd.read_parquet(f).set_index("timestamp")["close"]
        df.index = pd.to_datetime(df.index, utc=True)
        dfs[f.stem] = df

    # Load sector ETFs if available
    etf_dir = Path("data/ohlcv/etf")
    n_etfs = 0
    if include_etfs and etf_dir.exists():
        for f in sorted(etf_dir.glob("*.parquet")):
            ticker = f.stem
            df = pd.read_parquet(f)
            # ETF parquets have column "close" (from fetch_etfs.py)
            col = "close" if "close" in df.columns else df.columns[0]
            s = df[col].dropna()
            s.index = pd.to_datetime(s.index, utc=True)
            s.name = ticker
            dfs[ticker] = s
            n_etfs += 1
        if n_etfs:
            logger.info(f"Loaded {n_etfs} sector ETFs from {etf_dir}")

    return pd.DataFrame(dfs)


def _print_report(result, elapsed: float) -> None:
    m = result.metrics
    W = 70
    SEP = "+" + "-" * W + "+"

    def row(text): return f"| {text:<{W-1}}|"

    n_folds  = len(result.folds)
    n_trades = sum(f.n_trades for f in result.folds)
    avg_pairs = sum(f.n_pairs for f in result.folds) / max(n_folds, 1)

    lines = [SEP, row("Walk-Forward Backtest Results  (v2: rolling z-score + IS Sharpe filter)"), SEP]
    lines += [
        row(f"Period        : {result.daily_returns.index[0].date()} -> "
            f"{result.daily_returns.index[-1].date()}"),
        row(f"Folds         : {n_folds}  |  IS={result.config.is_years}y  "
            f"OOS={result.config.oos_months}m"),
        row(f"Avg pairs     : {avg_pairs:.1f} / fold  |  Total trades: {n_trades}"),
        row(f"Rolling-z win : {result.config.rolling_z_window}d  |  "
            f"Min IS Sharpe: {result.config.min_is_sharpe:.1f}  |  "
            f"Entry z: {result.config.entry_zscore:.1f}"),
        row(f"Market filter : "
            f"{'ON' if result.config.use_market_regime_filter else 'OFF'}  |  "
            f"vol_win={result.config.market_vol_smoothing_window}d  "
            f"thresh={result.config.market_vol_threshold:.0%}  "
            f"cooldown={result.config.market_cooldown_days}d  "
            f"close={'yes' if result.config.close_positions_on_volatile else 'no'}"),
        SEP,
        row(f"  Annualized return  : {m['annualized_return']*100:>8.2f}%"),
        row(f"  Annualized vol     : {m['annualized_vol']*100:>8.2f}%"),
        row(f"  Sharpe ratio       : {m['sharpe_ratio']:>8.3f}"),
        row(f"  Max drawdown       : {m['max_drawdown']*100:>8.2f}%"),
        row(f"  Calmar ratio       : {m.get('calmar_ratio', float('nan')):>8.3f}"),
        row(f"  Win rate (daily)   : {m['win_rate']*100:>8.1f}%"),
        row(f"  Total return       : {m['total_return']*100:>8.2f}%"),
        row(f"  Trading days       : {m['n_trading_days']:>8,}"),
        SEP,
        row("Per-fold Sharpe breakdown:"),
    ]

    for f in result.folds:
        sh       = f.metrics.get("sharpe_ratio", float("nan"))
        raw_vol  = f.metrics.get("market_volatile_pct", float("nan"))
        gate_pct = f.metrics.get("market_gate_pct", float("nan"))
        avg_vix  = f.metrics.get("avg_vix", float("nan"))
        vix_pct  = f.metrics.get("vix_gated_pct", float("nan"))
        vol_str  = ""
        if not np.isnan(raw_vol):
            vol_str = f"  hmm_vol={raw_vol:.0%}"
        if not np.isnan(gate_pct):
            vol_str += f"  gated={gate_pct:.0%}"
        if not np.isnan(avg_vix):
            vol_str += f"  avgVIX={avg_vix:.0f}"
        if not np.isnan(vix_pct):
            vol_str += f"  vix_gt={vix_pct:.0%}"
        lines.append(row(
            f"  Fold {f.fold.idx+1:>2}  OOS={f.fold.oos_start.date()}->"
            f"{f.fold.oos_end.date()}  "
            f"pairs={f.n_pairs}  trades={f.n_trades}  "
            f"Sharpe={sh:.2f}{vol_str}"
        ))

    m_str, s_str = divmod(int(elapsed), 60)
    lines += [
        SEP,
        row(f"Elapsed : {m_str}m {s_str:02d}s"),
        SEP,
    ]

    output = "\n" + "\n".join(lines) + "\n"
    sys.stdout.buffer.write(output.encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")
    sys.stdout.buffer.flush()


def main() -> None:
    args = _parse_args()

    prices = _load_prices(args.data_dir, include_etfs=not args.no_etfs)
    logger.info(
        f"Loaded {len(prices.columns)} symbols  |  "
        f"{prices.index[0].date()} -> {prices.index[-1].date()}"
    )

    config = WalkForwardConfig(
        start=args.start,
        end=args.end,
        is_years=args.is_years,
        oos_months=args.oos_months,
        max_pairs=args.max_pairs,
        notional_per_pair=args.notional,
        entry_zscore=args.entry_z,
        exit_zscore=args.exit_z,
        stop_zscore=args.stop_z,
        transaction_cost_bps=args.tc_bps,
        eg_pvalue=args.eg_pvalue,
        max_halflife=args.max_halflife,
        strict_regime_filter=args.strict_regime,
        rolling_z_window=args.rolling_z_win,
        min_is_sharpe=args.min_is_sharpe,
        same_sector_only=True,         # Layer 2: same-sector by default
        use_market_regime_filter=args.market_filter,
        close_positions_on_volatile=args.close_volatile,
        market_vol_smoothing_window=args.vol_window,
        market_vol_threshold=args.vol_threshold,
        market_cooldown_days=args.cooldown,
        vix_gate_threshold=args.vix_threshold,
        vix_min_threshold=args.vix_min,
        min_portfolio_is_sharpe=args.min_portfolio_sharpe,
        market_momentum_window=args.momentum_window,
        market_momentum_threshold=args.momentum_threshold,
        min_dispersion=args.min_dispersion,
    )

    # Load VIX data (Layer 1)
    vix = None
    if not args.no_vix and args.vix_threshold > 0:
        try:
            from stat_arb.data.vix import ensure_vix
            vix = ensure_vix()
            logger.info(
                f"VIX loaded: {len(vix)} days  "
                f"mean={vix.mean():.1f}  "
                f"gate_threshold={args.vix_threshold:.0f}"
            )
        except Exception as exc:
            logger.warning(f"VIX load failed: {exc} — running without VIX gate")

    t0 = time.perf_counter()
    result = WalkForwardBacktest(config).run(prices, vix=vix)
    elapsed = time.perf_counter() - t0

    # Save daily returns
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.daily_returns.to_frame("portfolio_return").to_parquet(out)
    logger.info(f"Daily returns saved to {out}")

    _print_report(result, elapsed)


if __name__ == "__main__":
    main()
