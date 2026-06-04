"""VIX (CBOE Volatility Index) data utilities.

VIX is the market's expectation of 30-day S&P 500 volatility derived from
options prices.  It is a leading indicator:

  VIX < 15   : low volatility, range-bound market → ideal for pairs trading
  VIX 15-20  : normal range → pairs trading works
  VIX 20-30  : elevated stress → caution, reduce position sizing
  VIX > 30   : crisis mode (2008, 2020 COVID, 2022 rate hikes) → stop trading

Using VIX as a hard gate is more reliable than the HMM market regime alone
because it reflects forward-looking option prices rather than backward-looking
realized volatility.  The HMM approximates VIX from realized vol; using the
actual VIX removes the approximation error.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger

VIX_TICKER    = "^VIX"
DEFAULT_DIR   = Path("data/market")
VIX_FILE      = DEFAULT_DIR / "vix.parquet"


def fetch_vix(start: str = "2008-01-01", end: str = "2025-01-01") -> pd.Series:
    """Download VIX daily closing prices from Yahoo Finance.

    Returns a UTC-indexed Series named 'vix'.
    """
    df = yf.download(VIX_TICKER, start=start, end=end,
                     interval="1d", auto_adjust=True, progress=False)
    # Newer yfinance returns MultiIndex (Price, Ticker) columns — flatten
    if hasattr(df.columns, "levels"):
        df.columns = df.columns.droplevel(1)
    col = "Close" if "Close" in df.columns else df.columns[0]
    vix = df[col].dropna()
    vix.index = pd.to_datetime(vix.index, utc=True)
    vix.name  = "vix"
    return vix


def save_vix(vix: pd.Series, path: Path = VIX_FILE) -> None:
    """Save VIX Series to parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    vix.to_frame("vix").to_parquet(path)
    logger.info(f"VIX saved to {path} ({len(vix)} days)")


def load_vix(path: Path = VIX_FILE) -> pd.Series | None:
    """Load VIX from parquet cache; returns None if file not found."""
    if not Path(path).exists():
        return None
    df = pd.read_parquet(path)
    vix = df["vix"].dropna()
    vix.index = pd.to_datetime(vix.index, utc=True)
    vix.name  = "vix"
    return vix


def ensure_vix(
    start: str = "2008-01-01",
    end:   str = "2025-01-01",
    path:  Path = VIX_FILE,
) -> pd.Series:
    """Return cached VIX or download it if missing."""
    cached = load_vix(path)
    if cached is not None:
        logger.debug(f"VIX loaded from cache ({len(cached)} days)")
        return cached
    logger.info("VIX not cached — downloading from Yahoo Finance …")
    vix = fetch_vix(start, end)
    save_vix(vix, path)
    return vix


if __name__ == "__main__":
    import sys
    start = sys.argv[1] if len(sys.argv) > 1 else "2008-01-01"
    end   = sys.argv[2] if len(sys.argv) > 2 else "2025-01-01"
    vix   = fetch_vix(start, end)
    save_vix(vix)
    print(f"VIX: {len(vix)} days, range [{vix.min():.1f}, {vix.max():.1f}], mean={vix.mean():.1f}")
