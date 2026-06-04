from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
from loguru import logger

BATCH_SIZE = 10
MAX_5M_DAYS = 59  # yfinance hard limit for 5-minute intraday data

# Output dirs per interval
OUT_DIRS: dict[str, Path] = {
    "5m": Path("data/ohlcv/5m"),
    "1d": Path("data/ohlcv/1d"),
}
OUT_DIR = OUT_DIRS["5m"]  # kept for backwards-compat import in run.py


@dataclass
class SymbolResult:
    symbol: str
    bars: int = 0
    path: Optional[Path] = None
    error: Optional[str] = None
    skipped: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None and not self.skipped and self.bars > 0

    @property
    def size_mb(self) -> float:
        if self.path and self.path.exists():
            return self.path.stat().st_size / 1_048_576
        return 0.0


def fetch_ohlcv(
    symbols: list[str],
    start: date | datetime,
    end: date | datetime,
    interval: str = "5m",
    out_dir: Path | None = None,
    batch_size: int = BATCH_SIZE,
    force: bool = False,
) -> list[SymbolResult]:
    """Fetch OHLCV bars via yfinance and save each symbol to parquet.

    interval="5m"  — capped at last 59 days by yfinance; start is clipped automatically.
    interval="1d"  — no limit; supports 10+ years.
    """
    if out_dir is None:
        out_dir = OUT_DIRS.get(interval, Path(f"data/ohlcv/{interval}"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_d = start.date() if isinstance(start, datetime) else start
    end_d   = end.date()   if isinstance(end,   datetime) else end

    # Enforce yfinance 59-day limit for 5m data only
    if interval == "5m":
        hard_start = date.today() - timedelta(days=MAX_5M_DAYS)
        if start_d < hard_start:
            logger.warning(
                f"yfinance limits 5m data to {MAX_5M_DAYS} days. "
                f"Clipping start: {start_d} -> {hard_start}"
            )
            start_d = hard_start

    results: dict[str, SymbolResult] = {s: SymbolResult(symbol=s) for s in symbols}

    # Skip symbols whose parquet already exists (unless --force)
    to_fetch: list[str] = []
    for sym in symbols:
        path = out_dir / f"{sym}.parquet"
        if path.exists() and not force:
            n = _row_count(path)
            results[sym].skipped = True
            results[sym].path    = path
            results[sym].bars    = n
            logger.debug(f"{sym}: skipped ({n:,} bars exist). Use --force to overwrite.")
        else:
            to_fetch.append(sym)

    total_batches = (len(to_fetch) + batch_size - 1) // batch_size
    logger.info(
        f"Fetching {len(to_fetch)} symbols in {total_batches} batch(es) "
        f"({start_d} -> {end_d}, {interval}, yfinance)"
    )

    for batch_idx, b_start in enumerate(range(0, len(to_fetch), batch_size), 1):
        batch = to_fetch[b_start : b_start + batch_size]
        logger.info(f"[{batch_idx}/{total_batches}] {batch}")

        try:
            raw = yf.download(
                tickers=batch,
                start=str(start_d),
                end=str(end_d),
                interval=interval,
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception as exc:
            logger.error(f"  Batch failed: {exc}")
            for sym in batch:
                results[sym].error = str(exc)
            time.sleep(1.0)
            continue

        for sym in batch:
            results[sym] = _save_symbol(sym, raw, batch, out_dir, results[sym])

        if batch_idx < total_batches:
            time.sleep(0.3)

    return list(results.values())


# ── helpers ───────────────────────────────────────────────────────────────────

def _save_symbol(
    sym: str,
    raw: pd.DataFrame,
    batch: list[str],
    out_dir: Path,
    result: SymbolResult,
) -> SymbolResult:
    try:
        # Single-ticker download has flat columns; multi-ticker has (ticker, price) MultiIndex
        if len(batch) == 1:
            df = raw.copy()
        else:
            top = raw.columns.get_level_values(0).unique()
            if sym not in top:
                result.error = "no data returned"
                logger.warning(f"  {sym}: no data")
                return result
            df = raw[sym].copy()

        df = df.dropna(how="all")
        if df.empty:
            result.error = "empty after dropna"
            logger.warning(f"  {sym}: empty DataFrame")
            return result

        df.columns = df.columns.str.lower()
        df.index.name = "timestamp"
        df = df.reset_index()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        path = out_dir / f"{sym}.parquet"
        df.to_parquet(path, index=False, compression="snappy")

        result.bars = len(df)
        result.path = path
        logger.success(f"  {sym}: {len(df):,} bars -> {path.name}")

    except Exception as exc:
        result.error = str(exc)
        logger.error(f"  {sym}: {exc}")

    return result


def _row_count(path: Path) -> int:
    try:
        import pyarrow.parquet as pq
        return pq.read_metadata(path).num_rows
    except Exception:
        return len(pd.read_parquet(path))
