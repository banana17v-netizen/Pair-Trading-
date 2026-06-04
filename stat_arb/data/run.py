"""CLI entrypoint — fetch OHLCV data via yfinance and save as Parquet.

Intervals
---------
  5m  (default) last 59 days only — yfinance hard limit for intraday
  1d             unlimited history — use for cointegration, Kalman, HMM, backtest

Usage examples
--------------
# 5m, default universe, last 59 days
python -m stat_arb.data.run

# Daily, full 15-year history for backtesting
python -m stat_arb.data.run --interval 1d --start 2010-01-01 --end 2025-01-01

# Custom symbols, daily
python -m stat_arb.data.run --interval 1d --symbols AAPL MSFT GOOGL --start 2010-01-01

# Force re-download
python -m stat_arb.data.run --interval 1d --start 2010-01-01 --force
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

from loguru import logger

from stat_arb.data.fetch import fetch_ohlcv, SymbolResult, OUT_DIRS
from stat_arb.data.universe import load_symbols, SP500_LIQUID

# ── Logging config ────────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    level="INFO",
)
logger.add(
    "data/fetch.log",
    rotation="10 MB",
    retention=5,
    level="DEBUG",
    encoding="utf-8",
)


def _parse_args() -> argparse.Namespace:
    # Parse interval first (without consuming other args) to set smart defaults
    import sys as _sys
    _interval = "5m"
    for i, arg in enumerate(_sys.argv[1:]):
        if arg == "--interval" and i + 1 < len(_sys.argv) - 1:
            _interval = _sys.argv[i + 2]
            break

    today = date.today()
    if _interval == "1d":
        default_start = "2010-01-01"
        default_end   = "2025-01-01"
        default_out   = str(OUT_DIRS["1d"])
    else:
        default_start = (today - timedelta(days=59)).isoformat()
        default_end   = today.isoformat()
        default_out   = str(OUT_DIRS["5m"])

    p = argparse.ArgumentParser(
        description="Fetch OHLCV bars via yfinance and save as Parquet. "
                    "Use --interval 1d for full history (backtest/Kalman/HMM), "
                    "5m for intraday execution (last 59 days only)."
    )
    sym_group = p.add_mutually_exclusive_group()
    sym_group.add_argument(
        "--symbols", nargs="+", metavar="TICK",
        help="Space-separated ticker list (e.g. AAPL MSFT GOOGL).",
    )
    sym_group.add_argument(
        "--file", metavar="PATH",
        help="Path to a file with one ticker per line (or CSV, first column).",
    )

    p.add_argument("--interval", default="5m", choices=["5m", "1d"],
                   help="Bar interval: 5m (intraday, last 59 days) or 1d (daily, unlimited). Default: 5m.")
    p.add_argument("--start", default=default_start, help=f"Start date YYYY-MM-DD (default: {default_start})")
    p.add_argument("--end",   default=default_end,   help=f"End date   YYYY-MM-DD (default: {default_end})")
    p.add_argument("--out",   default=default_out,   help=f"Output folder (default based on interval).")
    p.add_argument("--batch-size", type=int, default=10,
                   help="Symbols per yfinance request. Default: 10.")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if the parquet file already exists.")
    return p.parse_args()


def _print_report(
    results: list[SymbolResult],
    start: str,
    end: str,
    out_dir: Path,
    elapsed: float,
    interval: str = "5m",
) -> None:
    ok      = [r for r in results if r.ok]
    skipped = [r for r in results if r.skipped]
    failed  = [r for r in results if r.error and not r.skipped]

    total_bars = sum(r.bars for r in ok)
    total_mb   = sum(r.size_mb for r in ok)
    elapsed_str = _fmt_elapsed(elapsed)

    W = 72  # box width
    SEP = "+" + "-" * W + "+"

    def row(text: str) -> str:
        return f"| {text:<{W - 1}}|"

    lines: list[str] = []
    lines.append(SEP)
    lines.append(row(f"OHLCV Fetch Complete -- {interval} bars"))
    lines.append(SEP)
    lines.append(row(f"Period     : {start}  ->  {end}"))
    lines.append(row(f"Output dir : {str(out_dir)}"))
    lines.append(row(f"Requested  : {len(results)} symbols"))
    lines.append(SEP)
    lines.append(row(f"  [OK]     Success : {len(ok):>4} symbols"))
    lines.append(row(f"  [--]     Skipped : {len(skipped):>4} symbols  (parquet already exists)"))
    lines.append(row(f"  [ERR]    Failed  : {len(failed):>4} symbols"))

    if ok:
        lines.append(SEP)
        lines.append(row(f"  {'Symbol':<8}  {'Bars':>9}  {'Size':>8}  File"))
        lines.append(row(f"  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*30}"))
        for r in sorted(ok, key=lambda x: x.symbol):
            line = f"  {r.symbol:<8}  {r.bars:>9,}  {r.size_mb:>7.2f}M  {r.path.name if r.path else ''}"
            lines.append(row(line))

    if skipped:
        lines.append(SEP)
        lines.append(row("Skipped (already on disk):"))
        chunk = ", ".join(r.symbol for r in skipped)
        lines.append(row(f"  {chunk}"))

    if failed:
        lines.append(SEP)
        lines.append(row("Failed:"))
        for r in failed:
            lines.append(row(f"  {r.symbol:<8}  {r.error}"[:W]))

    lines.append(SEP)
    lines.append(row(f"Total bars  : {total_bars:>12,}"))
    lines.append(row(f"Total size  : {total_mb:>11.2f} MB"))
    lines.append(row(f"Elapsed     : {elapsed_str}"))
    lines.append(SEP)

    output = "\n" + "\n".join(lines) + "\n"
    sys.stdout.buffer.write(output.encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")
    sys.stdout.buffer.flush()


def _fmt_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def main() -> None:
    args = _parse_args()

    # Resolve symbol list
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    elif args.file:
        symbols = load_symbols(source="file", path=args.file)
    else:
        symbols = load_symbols(source="default")

    start_date = date.fromisoformat(args.start)
    end_date   = date.fromisoformat(args.end)
    out_dir    = Path(args.out)

    logger.info(f"StatArb OHLCV Fetcher (yfinance) | {len(symbols)} symbols | {args.interval} | {args.start} -> {args.end}")
    logger.info(f"BatchSize={args.batch_size}  Force={args.force}")

    t0 = time.perf_counter()
    results = fetch_ohlcv(
        symbols=symbols,
        start=start_date,
        end=end_date,
        interval=args.interval,
        out_dir=out_dir,
        batch_size=args.batch_size,
        force=args.force,
    )
    elapsed = time.perf_counter() - t0

    _print_report(results, args.start, args.end, out_dir, elapsed, interval=args.interval)

    failed_count = sum(1 for r in results if r.error and not r.skipped)
    sys.exit(1 if failed_count else 0)


if __name__ == "__main__":
    main()
