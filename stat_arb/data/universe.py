from __future__ import annotations

from pathlib import Path

import pandas as pd

# 119 S&P 500 stocks: original 53 large-caps + 66 mid-caps added in expansion.
# Mid-cap expansion rationale:
#   - Same-sector pairs increase from ~170 → ~880 (5× more cointegration candidates)
#   - Mid-caps have less single-stock momentum dominance than mega-caps
#   - More stock-vs-stock and stock-vs-ETF pairs within each sector
#   - All stocks have continuous listing from 2010+ (or 2012+ for recent IPOs)
SP500_LIQUID: list[str] = [
    # ── Technology (22 stocks) ────────────────────────────────────────────────
    # Large-caps (original 12)
    "AAPL", "MSFT", "NVDA", "AVGO", "ADBE", "CRM", "INTC", "AMD",
    "QCOM", "TXN", "CSCO", "ACN",
    # Mid-caps added (10)
    "ORCL", "IBM",                          # enterprise software/cloud
    "AMAT", "LRCX", "KLAC",                 # semiconductor equipment
    "CDNS", "SNPS",                         # EDA software
    "FTNT", "PANW",                         # cybersecurity (PANW IPO Jul 2012)
    "NOW",                                  # ServiceNow (IPO Jun 2012)

    # ── Communication Services (10 stocks) ───────────────────────────────────
    # Large-caps (original 5)
    "GOOGL", "META", "NFLX", "VZ", "DIS",
    # Mid-caps added (5)
    "T",                                    # AT&T (telecom)
    "CMCSA",                                # Comcast (cable/media)
    "CHTR",                                 # Charter Comm (cable, listed Jul 2010)
    "TMUS",                                 # T-Mobile US (listed May 2013)
    "EA",                                   # Electronic Arts (gaming)

    # ── Consumer Discretionary (13 stocks) ───────────────────────────────────
    # Large-caps (original 6)
    "AMZN", "TSLA", "HD", "LOW", "MCD", "NKE",
    # Mid-caps added (7)
    "TJX",                                  # off-price retail
    "CMG",                                  # fast casual (IPO Jan 2006)
    "DHI",                                  # homebuilder
    "LEN",                                  # homebuilder
    "YUM",                                  # restaurants (KFC/Pizza Hut/Taco Bell)
    "MAR",                                  # hotels
    "BKNG",                                 # online travel (Booking/Priceline)

    # ── Consumer Staples (11 stocks) ─────────────────────────────────────────
    # Large-caps (original 6)
    "WMT", "COST", "PG", "KO", "PEP", "PM",
    # Mid-caps added (5)
    "CL",                                   # Colgate-Palmolive
    "GIS",                                  # General Mills
    "MO",                                   # Altria (tobacco)
    "STZ",                                  # Constellation Brands (beer/wine)
    "HSY",                                  # Hershey (confectionery)

    # ── Financials (15 stocks) ───────────────────────────────────────────────
    # Large-caps (original 7)
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA",
    # Mid-caps added (8)
    "SCHW",                                 # Charles Schwab (brokerage)
    "COF",                                  # Capital One (credit cards)
    "AXP",                                  # American Express
    "SPGI",                                 # S&P Global (financial info)
    "ICE",                                  # Intercontinental Exchange
    "CB",                                   # Chubb (insurance; was ACE Ltd pre-2016)
    "PGR",                                  # Progressive (auto insurance)
    "BK",                                   # Bank of New York Mellon (custody bank)

    # ── Health Care (17 stocks) ──────────────────────────────────────────────
    # Large-caps (original 9)
    "JNJ", "PFE", "MRK", "ABBV", "LLY", "TMO", "MDT", "AMGN", "BMY",
    # Mid-caps added (8)
    "ISRG",                                 # Intuitive Surgical (robotics surgery)
    "REGN",                                 # Regeneron (biotech)
    "VRTX",                                 # Vertex Pharmaceuticals (CF drugs)
    "BIIB",                                 # Biogen (neurology biotech)
    "BSX",                                  # Boston Scientific (med devices)
    "SYK",                                  # Stryker (orthopedic implants)
    "BDX",                                  # Becton Dickinson (diagnostics)
    "EW",                                   # Edwards Lifesciences (heart valves)

    # ── Industrials (12 stocks) ──────────────────────────────────────────────
    # Large-caps (original 4)
    "HON", "UPS", "CAT", "RTX",
    # Mid-caps added (8)
    "BA",                                   # Boeing (aerospace)
    "LMT",                                  # Lockheed Martin (defense)
    "GE",                                   # GE Aerospace
    "MMM",                                  # 3M (diversified industrial)
    "EMR",                                  # Emerson Electric (automation)
    "ETN",                                  # Eaton (power management)
    "PH",                                   # Parker Hannifin (motion control)
    "ROK",                                  # Rockwell Automation (factory automation)

    # ── Energy (7 stocks) ────────────────────────────────────────────────────
    # Large-caps (original 2)
    "CVX", "XOM",
    # Mid-caps added (5)
    "COP",                                  # ConocoPhillips (E&P)
    "EOG",                                  # EOG Resources (shale E&P)
    "SLB",                                  # SLB/Schlumberger (oilfield services)
    "MPC",                                  # Marathon Petroleum (refining; IPO Jun 2011)
    "VLO",                                  # Valero Energy (refining)

    # ── Materials (6 stocks) ─────────────────────────────────────────────────
    # Large-caps (original 1)
    "LIN",
    # Mid-caps added (5)
    "APD",                                  # Air Products (industrial gases)
    "ECL",                                  # Ecolab (water/hygiene)
    "NEM",                                  # Newmont (gold mining)
    "FCX",                                  # Freeport-McMoRan (copper/gold)
    "SHW",                                  # Sherwin-Williams (paints)

    # ── Utilities (6 stocks) ─────────────────────────────────────────────────
    # Large-caps (original 1)
    "NEE",
    # Mid-caps added (5)
    "DUK",                                  # Duke Energy (electric/gas utility)
    "SO",                                   # Southern Company (electric utility)
    "AEP",                                  # American Electric Power
    "EXC",                                  # Exelon (nuclear/electric utility)
    "XEL",                                  # Xcel Energy (electric/gas utility)
]

# GICS sector map for ALL 119 stocks + 9 ETFs.
SECTOR_MAP: dict[str, str] = {
    # ── Technology (22) ───────────────────────────────────────────────────────
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "AVGO": "Technology", "ADBE": "Technology", "CRM":  "Technology",
    "INTC": "Technology", "AMD":  "Technology", "QCOM": "Technology",
    "TXN":  "Technology", "CSCO": "Technology", "ACN":  "Technology",
    "ORCL": "Technology", "IBM":  "Technology",
    "AMAT": "Technology", "LRCX": "Technology", "KLAC": "Technology",
    "CDNS": "Technology", "SNPS": "Technology",
    "FTNT": "Technology", "PANW": "Technology", "NOW":  "Technology",

    # ── Communication Services (10) ───────────────────────────────────────────
    "GOOGL": "Communication", "META":  "Communication", "NFLX":  "Communication",
    "VZ":    "Communication", "DIS":   "Communication",
    "T":     "Communication", "CMCSA": "Communication", "CHTR":  "Communication",
    "TMUS":  "Communication", "EA":    "Communication",

    # ── Consumer Discretionary (13) ───────────────────────────────────────────
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD":   "Consumer Discretionary", "LOW":  "Consumer Discretionary",
    "MCD":  "Consumer Discretionary", "NKE":  "Consumer Discretionary",
    "TJX":  "Consumer Discretionary", "CMG":  "Consumer Discretionary",
    "DHI":  "Consumer Discretionary", "LEN":  "Consumer Discretionary",
    "YUM":  "Consumer Discretionary", "MAR":  "Consumer Discretionary",
    "BKNG": "Consumer Discretionary",

    # ── Consumer Staples (11) ─────────────────────────────────────────────────
    "WMT": "Consumer Staples", "COST": "Consumer Staples",
    "PG":  "Consumer Staples", "KO":   "Consumer Staples",
    "PEP": "Consumer Staples", "PM":   "Consumer Staples",
    "CL":  "Consumer Staples", "GIS":  "Consumer Staples",
    "MO":  "Consumer Staples", "STZ":  "Consumer Staples",
    "HSY": "Consumer Staples",

    # ── Financials (15) ───────────────────────────────────────────────────────
    "JPM":  "Financials", "BAC":  "Financials", "WFC":  "Financials",
    "GS":   "Financials", "MS":   "Financials", "V":    "Financials",
    "MA":   "Financials",
    "SCHW": "Financials", "COF":  "Financials", "AXP":  "Financials",
    "SPGI": "Financials", "ICE":  "Financials", "CB":   "Financials",
    "PGR":  "Financials", "BK":   "Financials",

    # ── Health Care (17) ──────────────────────────────────────────────────────
    "JNJ":  "Health Care", "PFE":  "Health Care", "MRK":  "Health Care",
    "ABBV": "Health Care", "LLY":  "Health Care", "TMO":  "Health Care",
    "MDT":  "Health Care", "AMGN": "Health Care", "BMY":  "Health Care",
    "ISRG": "Health Care", "REGN": "Health Care", "VRTX": "Health Care",
    "BIIB": "Health Care", "BSX":  "Health Care", "SYK":  "Health Care",
    "BDX":  "Health Care", "EW":   "Health Care",

    # ── Industrials (12) ──────────────────────────────────────────────────────
    "HON": "Industrials", "UPS": "Industrials",
    "CAT": "Industrials", "RTX": "Industrials",
    "BA":  "Industrials", "LMT": "Industrials", "GE":  "Industrials",
    "MMM": "Industrials", "EMR": "Industrials", "ETN": "Industrials",
    "PH":  "Industrials", "ROK": "Industrials",

    # ── Energy (7) ────────────────────────────────────────────────────────────
    "CVX": "Energy", "XOM": "Energy",
    "COP": "Energy", "EOG": "Energy", "SLB": "Energy",
    "MPC": "Energy", "VLO": "Energy",

    # ── Materials (6) ─────────────────────────────────────────────────────────
    "LIN": "Materials",
    "APD": "Materials", "ECL": "Materials", "NEM": "Materials",
    "FCX": "Materials", "SHW": "Materials",

    # ── Utilities (6) ─────────────────────────────────────────────────────────
    "NEE": "Utilities",
    "DUK": "Utilities", "SO":  "Utilities", "AEP": "Utilities",
    "EXC": "Utilities", "XEL": "Utilities",
}


def get_sector(symbol: str) -> str:
    """Return the GICS sector for a symbol, or 'Unknown' if not in SECTOR_MAP."""
    return SECTOR_MAP.get(symbol.upper(), "Unknown")


# ── Sector ETF tickers ────────────────────────────────────────────────────────
ETF_SECTOR_MAP: dict[str, str] = {
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

SECTOR_MAP.update(ETF_SECTOR_MAP)


def load_symbols(source: str = "default", path: str | None = None) -> list[str]:
    """Return the symbol list to fetch.

    source="default"  → SP500_LIQUID (119 stocks)
    source="file"     → first column of a CSV/TXT at `path`
    """
    if source == "default":
        return list(SP500_LIQUID)
    if source == "file":
        if not path:
            raise ValueError("path is required when source='file'")
        p = Path(path)
        if p.suffix.lower() == ".csv":
            df = pd.read_csv(p, header=None)
        else:
            df = pd.read_csv(p, header=None, sep=r"\s+")
        return df.iloc[:, 0].str.strip().str.upper().tolist()
    raise ValueError(f"Unknown source '{source}'. Use 'default' or 'file'.")
