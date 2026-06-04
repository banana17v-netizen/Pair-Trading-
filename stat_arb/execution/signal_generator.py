"""Daily signal generation for live paper trading.

Replicates the backtesting engine logic for a single "today" date:
  1. Load IS prices (last is_years of daily parquet data)
  2. Screen pairs using the expanded 119-stock universe
  3. For each pair: OLS regression → rolling z-score → regime filter
  4. Apply VIX gate (load from cache or fetch)
  5. Apply market HMM gate
  6. Return list of DesiredPosition objects

This module is designed to run once daily after market close (4pm+).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from loguru import logger

from stat_arb.backtesting.engine import _rolling_zscore
from stat_arb.data.universe import SP500_LIQUID, get_sector
from stat_arb.data.vix import load_vix, ensure_vix
from stat_arb.hmm.regime import HMMRegimeDetector, build_features_from_spread
from stat_arb.pairs.select import screen_pairs

warnings.filterwarnings("ignore")

# ── Config defaults (match backtesting engine optimal params) ─────────────────
IS_YEARS         = 2       # in-sample window
ROLLING_Z_WIN    = 10      # rolling z-score window
ENTRY_Z          = 1.5     # z-score entry threshold
EXIT_Z           = 0.3     # z-score exit threshold
STOP_Z           = 3.5     # stop-loss z-score
VIX_MAX          = 25.0    # VIX > 25 → no new entries
VIX_MIN          = 0.0     # VIX min (0=disabled)
EG_PVALUE        = 0.05
MAX_HALFLIFE     = 120.0
NOTIONAL         = 50_000.0


@dataclass
class DesiredPosition:
    """A desired pairs position after signal generation."""
    sym_y:      str
    sym_x:      str
    direction:  int     # +1 = long spread (long Y, short X); -1 = short spread
    z_score:    float
    ols_alpha:  float
    ols_beta:   float
    notional:   float
    sector:     str
    regime:     str     # "mean_reverting" / "trending" / "volatile"
    is_sharpe:  float


class SignalGenerator:
    """Computes daily desired positions from IS history + today's prices."""

    def __init__(
        self,
        data_dir:      str   = "data/ohlcv/1d",
        etf_dir:       str   = "data/ohlcv/etf",
        is_years:      int   = IS_YEARS,
        rolling_z_win: int   = ROLLING_Z_WIN,
        entry_z:       float = ENTRY_Z,
        exit_z:        float = EXIT_Z,
        stop_z:        float = STOP_Z,
        vix_max:       float = VIX_MAX,
        eg_pvalue:     float = EG_PVALUE,
        max_halflife:  float = MAX_HALFLIFE,
        notional:      float = NOTIONAL,
    ) -> None:
        self.data_dir      = Path(data_dir)
        self.etf_dir       = Path(etf_dir)
        self.is_years      = is_years
        self.rolling_z_win = rolling_z_win
        self.entry_z       = entry_z
        self.exit_z        = exit_z
        self.stop_z        = stop_z
        self.vix_max       = vix_max
        self.eg_pvalue     = eg_pvalue
        self.max_halflife  = max_halflife
        self.notional      = notional

    # ── Main entry point ──────────────────────────────────────────────────────

    def compute(
        self,
        as_of: datetime | None = None,
        min_portfolio_is_sharpe: float = 1.28,
    ) -> tuple[list[DesiredPosition], dict]:
        """Compute desired positions as of `as_of` date.

        Parameters
        ----------
        as_of                   : Date to compute for. Defaults to today.
        min_portfolio_is_sharpe : IS portfolio Sharpe filter (from backtesting).

        Returns
        -------
        (desired_positions, diagnostics_dict)
        """
        if as_of is None:
            as_of = datetime.now(tz=timezone.utc)

        logger.info(f"SignalGenerator: computing as_of={as_of.date()}")

        # ── 1. Load prices ────────────────────────────────────────────────────
        prices = self._load_prices_with_today(as_of)
        if prices.empty:
            logger.error("No price data available")
            return [], {}

        # Use last available trading date in the parquets as the signal date
        # (handles weekends/holidays and data lag)
        last_available = prices.index[-1].date()
        as_of_effective = min(as_of.date(), last_available)
        logger.info(f"Effective signal date: {as_of_effective} (latest data)")

        # IS window = last is_years years ending on effective date
        is_end   = as_of_effective
        is_start = as_of_effective - timedelta(days=self.is_years * 365 + 30)

        # Only keep dates where most stocks have data (handles ETF/stock overlap)
        first_good = (
            prices.dropna(thresh=int(len(prices.columns) * 0.9)).index[0].date()
            if len(prices) > 0 else is_start
        )
        is_cutoff = max(is_start, first_good)
        is_prices = prices.loc[str(is_cutoff):str(is_end)]
        if len(is_prices) < 200:
            logger.warning(f"IS data only {len(is_prices)} bars — need at least 200")
            return [], {"error": "insufficient_IS_data"}

        logger.info(f"IS window: {is_prices.index[0].date()} → {is_prices.index[-1].date()} "
                    f"({len(is_prices)} bars)")

        # ── 2. Screen IS pairs ────────────────────────────────────────────────
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pairs_df = screen_pairs(
                is_prices.dropna(axis=1, how="any"),
                eg_pvalue_threshold=self.eg_pvalue,
                max_half_life=self.max_halflife,
                min_history=100,
                same_sector_only=True,
                n_jobs=1,
            )
        selected = pairs_df[pairs_df["passed_all"]].head(20)
        logger.info(f"Pairs selected from IS: {len(selected)}")
        if selected.empty:
            return [], {"n_pairs": 0}

        # ── 3. VIX gate ───────────────────────────────────────────────────────
        vix_today = self._get_vix_today()
        if vix_today and self.vix_max > 0 and vix_today > self.vix_max:
            logger.warning(f"VIX={vix_today:.1f} > {self.vix_max} → NO NEW ENTRIES")
            return [], {"vix": vix_today, "gated": "VIX_HIGH"}

        # ── 4. Compute signals per pair ───────────────────────────────────────
        desired: list[DesiredPosition] = []
        today_prices = prices.loc[str(as_of.date()):] if str(as_of.date()) in prices.index.astype(str) else None

        for _, row in selected.iterrows():
            sym_y, sym_x = row["symbol_y"], row["symbol_x"]
            pos = self._compute_pair_signal(
                sym_y, sym_x, row, is_prices, prices, as_of
            )
            if pos is not None:
                desired.append(pos)

        diagnostics = {
            "n_pairs_screened": len(selected),
            "n_signals":        len(desired),
            "vix":              vix_today,
            "as_of":            str(as_of.date()),
        }
        logger.info(f"Signals: {len(desired)} active positions desired")
        return desired, diagnostics

    # ── Pair signal computation ───────────────────────────────────────────────

    def _compute_pair_signal(
        self,
        sym_y:    str,
        sym_x:    str,
        row:      pd.Series,
        is_prices: pd.DataFrame,
        all_prices: pd.DataFrame,
        as_of:    datetime,
    ) -> DesiredPosition | None:
        """Compute signal for one pair. Returns DesiredPosition or None."""
        try:
            pair = is_prices[[sym_y, sym_x]].dropna()
            if len(pair) < 50:
                return None

            ly = np.log(pair[sym_y].values)
            lx = np.log(pair[sym_x].values)

            # OLS on IS data
            ols = sm.OLS(ly, sm.add_constant(lx)).fit()
            ols_alpha = float(ols.params[0])
            ols_beta  = float(ols.params[1])

            # Build IS spread
            spread_is = ly - ols_alpha - ols_beta * lx

            # Add today's price if available
            today_str = str(as_of.date())
            today_avail = (
                today_str in all_prices.index.astype(str)
                and sym_y in all_prices.columns
                and sym_x in all_prices.columns
            )

            if today_avail:
                today_row = all_prices.loc[all_prices.index.astype(str) == today_str]
                if not today_row.empty:
                    ly_today = float(np.log(today_row[sym_y].iloc[0]))
                    lx_today = float(np.log(today_row[sym_x].iloc[0]))
                    spread_today = ly_today - ols_alpha - ols_beta * lx_today
                    spread_full = np.append(spread_is, spread_today)
                else:
                    spread_full = spread_is
            else:
                spread_full = spread_is

            # Rolling z-score on IS tail + today
            is_tail  = spread_is[-self.rolling_z_win:]
            combined = np.concatenate([is_tail, [spread_full[-1]]])
            z_all    = _rolling_zscore(combined, self.rolling_z_win)
            z_today  = float(z_all[-1])

            # HMM regime on IS spread features
            is_df = pd.DataFrame(
                {"spread": spread_is,
                 "zscore": _rolling_zscore(spread_is, self.rolling_z_win)},
                index=pair.index,
            )
            features_is = build_features_from_spread(is_df.dropna())
            if len(features_is) < 30:
                return None
            hmm = HMMRegimeDetector(random_state=42)
            hmm.fit(features_is)

            # Regime on today's context
            spread_recent = spread_full[-max(50, self.rolling_z_win * 2):]
            z_recent = _rolling_zscore(spread_recent, self.rolling_z_win)
            oos_df = pd.DataFrame({
                "spread": [spread_recent[-1]],
                "zscore": [z_recent[-1]],
            }, index=[as_of])
            features_context = build_features_from_spread(
                pd.concat([features_is.tail(30).assign(
                    zscore=features_is["zscore"].tail(30)
                ), oos_df])
            )
            try:
                regime_series = hmm.predict_series(features_context.tail(1))
                regime = str(regime_series.iloc[-1]) if len(regime_series) > 0 else "volatile"
            except Exception:
                regime = "mean_reverting"   # conservative fallback

            # IS Sharpe (from pairs screening)
            is_sharpe = float(row.get("is_sharpe", 0.5) if "is_sharpe" in row else 0.5)

            # Signal logic
            if abs(z_today) < self.entry_z:
                return None   # no entry signal
            if regime == "volatile":
                return None   # blocked by pair regime

            direction = 1 if z_today < -self.entry_z else -1

            return DesiredPosition(
                sym_y=sym_y, sym_x=sym_x,
                direction=direction,
                z_score=z_today,
                ols_alpha=ols_alpha,
                ols_beta=ols_beta,
                notional=self.notional,
                sector=get_sector(sym_y),
                regime=regime,
                is_sharpe=is_sharpe,
            )

        except Exception as exc:
            logger.debug(f"Signal failed {sym_y}/{sym_x}: {exc}")
            return None

    # ── Data loading helpers ──────────────────────────────────────────────────

    def _load_prices_with_today(self, as_of: datetime) -> pd.DataFrame:
        """Load IS daily prices and append today's close from yfinance."""
        dfs = {}

        # Load stocks
        for f in sorted(self.data_dir.glob("*.parquet")):
            df = pd.read_parquet(f).set_index("timestamp")["close"]
            df.index = pd.to_datetime(df.index, utc=True)
            dfs[f.stem] = df

        # Load ETFs
        if self.etf_dir.exists():
            for f in sorted(self.etf_dir.glob("*.parquet")):
                df = pd.read_parquet(f)
                col = "close" if "close" in df.columns else df.columns[0]
                s   = df[col].dropna()
                s.index = pd.to_datetime(s.index, utc=True)
                s.name  = f.stem
                dfs[f.stem] = s

        prices = pd.DataFrame(dfs)
        prices.index = pd.to_datetime(prices.index, utc=True)

        # Fetch today's prices from yfinance if market has closed
        today_str = as_of.strftime("%Y-%m-%d")
        if today_str not in prices.index.strftime("%Y-%m-%d"):
            logger.info(f"Fetching today's prices ({today_str}) from yfinance...")
            try:
                syms = list(prices.columns)
                today_data = yf.download(
                    syms, start=today_str, end=today_str,
                    interval="1d", auto_adjust=True, progress=False,
                    group_by="ticker",
                )
                if not today_data.empty:
                    today_row = {}
                    for sym in syms:
                        try:
                            if isinstance(today_data.columns, pd.MultiIndex):
                                p = float(today_data[sym]["Close"].iloc[0])
                            else:
                                p = float(today_data["Close"].iloc[0])
                            today_row[sym] = p
                        except Exception:
                            pass
                    if today_row:
                        ts = pd.Timestamp(today_str, tz="UTC")
                        new_row = pd.DataFrame(today_row, index=[ts])
                        prices = pd.concat([prices, new_row])
                        logger.info(f"Added today's prices for {len(today_row)} symbols")
            except Exception as exc:
                logger.warning(f"Could not fetch today's prices: {exc}")

        return prices.sort_index()

    def _get_vix_today(self) -> float | None:
        """Get today's VIX level from cache or yfinance."""
        try:
            vix = load_vix()
            if vix is not None and len(vix) > 0:
                # Use most recent available value
                return float(vix.iloc[-1])
        except Exception:
            pass
        # Fallback: fetch from yfinance
        try:
            vix_df = yf.download("^VIX", period="5d", interval="1d",
                                  auto_adjust=True, progress=False)
            if hasattr(vix_df.columns, "levels"):
                vix_df.columns = vix_df.columns.droplevel(1)
            if not vix_df.empty:
                return float(vix_df["Close"].iloc[-1])
        except Exception as exc:
            logger.warning(f"VIX fetch failed: {exc}")
        return None
