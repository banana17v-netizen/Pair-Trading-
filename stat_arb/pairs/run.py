"""CLI entrypoint — cointegration screening and pair selection.

Usage
-----
# Full universe, default settings (FDR + same-sector)
python -m stat_arb.pairs.run

# All pairs (cross-sector allowed), no FDR
python -m stat_arb.pairs.run --no-same-sector --no-fdr --max-halflife 120

# Parallel, custom thresholds
python -m stat_arb.pairs.run --n-jobs -1 --eg-pvalue 0.05 --fdr-alpha 0.10 --out data/pairs.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from loguru import logger

from stat_arb.pairs.select import screen_pairs

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    level="INFO",
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Screen S&P 500 universe for cointegrated pairs."
    )
    p.add_argument("--data-dir",      default="data/ohlcv/1d",  help="Parquet folder.")
    p.add_argument("--out",           default="data/pairs.csv", help="Output CSV.")
    p.add_argument("--start",         default=None,             help="Start date YYYY-MM-DD.")
    p.add_argument("--end",           default=None,             help="End date   YYYY-MM-DD.")
    p.add_argument("--eg-pvalue",     type=float, default=0.05, help="Raw EG p-value threshold.")
    p.add_argument("--min-halflife",  type=float, default=3.0,  help="Min half-life (days).")
    p.add_argument("--max-halflife",  type=float, default=120.0,help="Max half-life (days).")
    p.add_argument("--min-history",   type=int,   default=504,  help="Min shared bars.")
    p.add_argument("--max-pairs",     type=int,   default=25,   help="Top pairs to report.")
    p.add_argument("--n-jobs",        type=int,   default=1,    help="Parallel workers (-1=all).")
    # Sector filter
    p.add_argument("--same-sector",   dest="same_sector", action="store_true",  default=True,
                   help="Only test same-sector pairs (default ON).")
    p.add_argument("--no-same-sector",dest="same_sector", action="store_false",
                   help="Allow cross-sector pairs.")
    # FDR correction
    # Default OFF for full-period (15y) screening: signal too weak (min p=0.002 > BH
    # threshold 0.0003). Enable for short walk-forward windows (2y) where signal is stronger.
    p.add_argument("--fdr",           dest="apply_fdr", action="store_true",  default=False,
                   help="Apply BH-FDR correction (recommended for <=2y windows, OFF by default).")
    p.add_argument("--no-fdr",        dest="apply_fdr", action="store_false",
                   help="Use raw EG p-value (default for full-period screening).")
    p.add_argument("--fdr-alpha",     type=float, default=0.05,
                   help="FDR level (default 0.05).")
    return p.parse_args()


def _load_universe(data_dir: str, start: str | None, end: str | None) -> pd.DataFrame:
    p = Path(data_dir)
    if not p.exists():
        logger.error(f"Data directory not found: {p}")
        sys.exit(1)

    dfs = {}
    for f in sorted(p.glob("*.parquet")):
        df = pd.read_parquet(f).set_index("timestamp")["close"]
        dfs[f.stem] = df

    prices = pd.DataFrame(dfs)
    if start:
        prices = prices.loc[start:]
    if end:
        prices = prices.loc[:end]
    return prices


def _print_report(
    df: pd.DataFrame,
    args: argparse.Namespace,
    elapsed: float,
    prices: pd.DataFrame,
) -> None:
    W   = 80
    SEP = "+" + "-" * W + "+"

    def row(text: str) -> str:
        return f"| {text:<{W - 1}}|"

    n_symbols  = len(prices.columns)
    n_all      = n_symbols * (n_symbols - 1) // 2
    n_tested   = int((df["n_obs"] > 0).sum())
    n_eg_raw   = int(df["eg_passed"].sum())
    n_eg_fdr   = int(df["eg_passed_fdr"].sum())
    n_joh      = int((df["eg_passed"] & df["johansen_passed"]).sum())
    n_passed   = int(df["passed_all_fdr"].sum())
    top        = df[df["passed_all_fdr"]].head(args.max_pairs)

    m, s = divmod(int(elapsed), 60)
    elapsed_str = f"{m}m {s:02d}s" if m else f"{s}s"

    lines = [SEP, row("Cointegration Screening Report (v2: FDR + Sector)"), SEP]
    lines += [
        row(f"Universe    : {n_symbols} symbols  |  "
            f"All pairs: {n_all:,}  |  Tested: {n_tested:,}"),
        row(f"Period      : {prices.index[0].date()}  ->  {prices.index[-1].date()}"),
        row(f"Filters     : EG p<{args.eg_pvalue}  |  "
            f"FDR={'ON' if args.apply_fdr else 'OFF'} (alpha={args.fdr_alpha})  |  "
            f"Same-sector={'ON' if args.same_sector else 'OFF'}"),
        row(f"Half-life   : [{args.min_halflife:.0f}, {args.max_halflife:.0f}] days"),
        SEP,
        row(f"  Stage 1 - Engle-Granger (raw p < {args.eg_pvalue})    :"
            f" {n_eg_raw:>5} / {n_tested:>5} passed  ({n_eg_raw/max(n_tested,1)*100:.1f}%)"),
        row(f"  Stage 2 - BH-FDR correction (alpha={args.fdr_alpha})   :"
            f" {n_eg_fdr:>5} / {n_eg_raw:>5} retained"
            f"  (removed {n_eg_raw - n_eg_fdr} likely false positives)"),
        row(f"  Stage 3 - Johansen (95% confidence)             :"
            f" {n_joh:>5} / {n_eg_raw:>5} confirmed"),
        row(f"  Stage 4 - Half-life [{args.min_halflife:.0f},{args.max_halflife:.0f}]d + FDR  :"
            f" {n_passed:>5} / {n_joh:>5} passed"),
        row(f"  Final selection (top {args.max_pairs})                    :"
            f" {len(top):>5} pairs"),
        SEP,
    ]

    if not top.empty:
        hdr = (f"  {'#':<3} {'Pair':<12} {'Sector':<24} "
               f"{'EG raw':>8} {'FDR adj':>8} {'HL (d)':>7} {'Beta':>6} {'Score':>6}")
        lines.append(row(hdr))
        lines.append(row(f"  {'-'*3} {'-'*12} {'-'*24} {'-'*8} {'-'*8} {'-'*7} {'-'*6} {'-'*6}"))
        for i, (_, r) in enumerate(top.iterrows(), 1):
            pair_str   = f"{r['symbol_y']}/{r['symbol_x']}"
            sector_str = (r["sector_y"] if r["same_sector"]
                          else f"{r['sector_y'][:10]}/{r['sector_x'][:10]}")
            hl_str     = f"{r['half_life']:.1f}" if pd.notna(r["half_life"]) else "n/a"
            beta_str   = f"{r['hedge_ratio']:.3f}" if pd.notna(r["hedge_ratio"]) else "n/a"
            fdr_str    = f"{r['eg_pvalue_fdr']:.4f}" if pd.notna(r["eg_pvalue_fdr"]) else "n/a"
            line = (f"  {i:<3} {pair_str:<12} {sector_str:<24} "
                    f"{r['eg_pvalue']:>8.4f} {fdr_str:>8} {hl_str:>7} {beta_str:>6} {r['score']:>6.3f}")
            lines.append(row(line))
        lines.append(SEP)

    lines += [
        row(f"Elapsed : {elapsed_str}  |  Output : {args.out}"),
        SEP,
    ]

    output = "\n" + "\n".join(lines) + "\n"
    sys.stdout.buffer.write(output.encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")
    sys.stdout.buffer.flush()


def main() -> None:
    args   = _parse_args()
    prices = _load_universe(args.data_dir, args.start, args.end)
    logger.info(
        f"Loaded {len(prices.columns)} symbols  |  "
        f"{prices.index[0].date()} -> {prices.index[-1].date()}  |  "
        f"{len(prices):,} rows"
    )

    t0 = time.perf_counter()
    df = screen_pairs(
        prices,
        eg_pvalue_threshold=args.eg_pvalue,
        min_half_life=args.min_halflife,
        max_half_life=args.max_halflife,
        min_history=args.min_history,
        n_jobs=args.n_jobs,
        same_sector_only=args.same_sector,
        apply_fdr=args.apply_fdr,
        fdr_alpha=args.fdr_alpha,
    )
    elapsed = time.perf_counter() - t0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    _print_report(df, args, elapsed, prices)


if __name__ == "__main__":
    main()
