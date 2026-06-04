"""Download sector ETF data for stock-vs-ETF pairs trading.

Stock-vs-ETF pairs are the most reliable form of statistical arbitrage:
  spread = log(stock) - beta × log(sector_ETF)

This spread represents the stock's IDIOSYNCRATIC return component — the part
not explained by sector-wide movements. Idiosyncratic returns are:
  1. More stationary (less drift) than stock-vs-stock spreads
  2. Shorter half-life (1-10 days vs 50-120 days for stock-vs-stock)
  3. More stable across time (ETF composition changes slowly)

Sector ETFs:
  XLK  — Technology Select Sector SPDR
  XLF  — Financial Select Sector SPDR
  XLV  — Health Care Select Sector SPDR
  XLI  — Industrial Select Sector SPDR
  XLE  — Energy Select Sector SPDR
  XLY  — Consumer Discretionary Select Sector SPDR
  XLP  — Consumer Staples Select Sector SPDR
  XLU  — Utilities Select Sector SPDR
  XLB  — Materials Select Sector SPDR
  XLRE — Real Estate Select Sector SPDR
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger

# Sector ETF → sector label mapping (matches stat_arb/data/universe.py SECTOR_MAP)
SECTOR_ETFS: dict[str, str] = {
    "XLK":  "Technology",
    "XLF":  "Financials",
    "XLV":  "Health Care",
    "XLI":  "Industrials",
    "XLE":  "Energy",
    "XLY":  "Consumer Discretionary",
    "XLP":  "Consumer Staples",
    "XLU":  "Utilities",
    "XLB":  "Materials",
}

ETF_DIR = Path("data/ohlcv/etf")


def fetch_etf(ticker: str, start: str = "2009-01-01", end: str = "2025-01-01") -> pd.Series:
    """Download one sector ETF from yfinance."""
    df = yf.download(ticker, start=start, end=end,
                     interval="1d", auto_adjust=True, progress=False)
    if hasattr(df.columns, "levels"):
        df.columns = df.columns.droplevel(1)
    close = df["Close"].dropna()
    close.index = pd.to_datetime(close.index, utc=True)
    close.name = ticker
    return close


def fetch_all_etfs(
    start: str = "2009-01-01",
    end:   str = "2025-01-01",
    out_dir: Path = ETF_DIR,
) -> pd.DataFrame:
    """Download all sector ETFs and save to parquet files.

    Returns a DataFrame with all ETFs aligned on the same DatetimeIndex.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    series = {}
    for ticker in SECTOR_ETFS:
        logger.info(f"Downloading {ticker} ...")
        try:
            s = fetch_etf(ticker, start, end)
            # Save individual parquet
            s.to_frame("close").to_parquet(out_dir / f"{ticker}.parquet")
            series[ticker] = s
            logger.info(f"  {ticker}: {len(s)} days  range [{s.min():.1f}, {s.max():.1f}]")
        except Exception as exc:
            logger.warning(f"  {ticker}: failed — {exc}")
    return pd.DataFrame(series)


def load_etfs(out_dir: Path = ETF_DIR) -> pd.DataFrame:
    """Load all cached sector ETF parquet files."""
    dfs = {}
    for ticker in SECTOR_ETFS:
        fp = out_dir / f"{ticker}.parquet"
        if fp.exists():
            s = pd.read_parquet(fp)["close"]
            s.index = pd.to_datetime(s.index, utc=True)
            s.name = ticker
            dfs[ticker] = s
    return pd.DataFrame(dfs)


if __name__ == "__main__":
    import sys
    start = sys.argv[1] if len(sys.argv) > 1 else "2009-01-01"
    end   = sys.argv[2] if len(sys.argv) > 2 else "2025-01-01"
    df = fetch_all_etfs(start, end)
    print(f"Downloaded {len(df.columns)} ETFs: {list(df.columns)}")
    print(df.tail(3))
