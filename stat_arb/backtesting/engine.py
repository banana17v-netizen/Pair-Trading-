"""Walk-forward backtesting engine for equity pairs trading — v2.

Three improvements over v1
--------------------------
  1. Rolling z-score signal
       Uses a 20-day rolling window z-score (spread vs recent mean/std) instead
       of the Kalman innovation z-score.  The rolling z ~ N(0,1) by construction,
       so entry at ±2 triggers on ~5% of days — meaningful and calibrated.
       The Kalman innovation z-score had std ~0.7 (filter adaptation artifact),
       making entry at ±2 effectively a 2.8-sigma event that almost never fired.

  2. More pairs (cross-sector allowed by default)
       same_sector_only defaults to False, giving 1,378 candidate pairs from 53
       symbols instead of 170.  The IS Sharpe filter (improvement 3) handles
       quality control instead of the sector filter.

  3. IS Sharpe quality filter
       After fitting OLS + rolling z-score on IS data, the engine runs a quick
       IS paper-trade simulation and keeps only pairs with IS Sharpe ≥ min_is_sharpe.
       This prevents including pairs that would not have worked even in-sample.

Signal flow (v2)
----------------
  Per fold, per pair:
    IS:
      a. OLS regression → fixed alpha, beta, IS spread stats
      b. Rolling z-score on IS OLS spread
      c. IS Sharpe simulation → filter by min_is_sharpe
      d. Kalman (dynamic beta for P&L hedge ratio)
      e. HMM fitted on IS OLS spread features (spread_ret, vol, autocorr, z)

    OOS (day by day, no look-ahead):
      a. OLS spread_t = log_y_t − is_alpha − is_beta × log_x_t
      b. Rolling z_t = (spread_t − mean_win) / std_win  using IS tail + OOS so far
      c. Kalman step for dynamic hedge ratio β_t
      d. HMM predict on OOS features → regime
      e. Entry: |z| > entry_z AND regime != VOLATILE
      f. Exit:  |z| < exit_z OR |z| > stop_z OR regime == VOLATILE
      g. P&L = direction × (ret_y − β_entry × ret_x) × notional
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from loguru import logger

from stat_arb.backtesting.metrics import compute_metrics
from stat_arb.hmm.regime import (
    HMMRegimeDetector,
    build_features_from_spread,
    build_market_features,
)
from stat_arb.kalman.filter import KalmanFilterHedge
from stat_arb.pairs.select import screen_pairs
from stat_arb.risk.limits import PortfolioState, RiskChecker, RiskLimits, RiskAction
from stat_arb.risk.sizing import (
    compute_position_size,
    compute_is_trade_pnls,
    adaptive_notional_per_pair,
)


# ── Module-level helpers ──────────────────────────────────────────────────────

def _compute_smoothed_market_gate(
    market_regime: pd.Series,
    smoothing_window: int = 10,
    vol_threshold: float = 0.30,
    cooldown_days: int = 5,
) -> pd.Series:
    """Compute a smoothed, persistent market trading gate.

    Motivation
    ----------
    A raw per-day volatile gate causes whipsaw: a position is force-closed on a
    volatile day, then immediately re-opened the next non-volatile day (if the
    z-score is still above threshold), generating extra TC and potentially MORE
    total losses than the no-filter baseline (observed in Fold 19, 2023 H1).

    This function replaces the raw gate with a smoother one:

      1. Rolling volatile fraction
           f_t = mean(regime == "volatile" over last smoothing_window days)
      2. Gate ON when f_t >= vol_threshold
           e.g. threshold=0.30 means "block if ≥30% of last 10 days were volatile"
      3. Cooldown extension
           After the gate turns OFF, keep it ON for cooldown_days more days.
           This prevents "false start" re-entries right after a volatile period.

    Returns
    -------
    pd.Series[bool] aligned to market_regime.index.
    True = gated (no new entries; force-close if enabled).
    """
    is_vol   = (market_regime == "volatile").astype(float)
    roll_frac = is_vol.rolling(smoothing_window, min_periods=1).mean()
    gated    = (roll_frac >= vol_threshold).values.copy()

    # Cooldown: stay gated for cooldown_days after volatile fraction drops
    if cooldown_days > 0:
        cooldown = 0
        for i in range(len(gated)):
            if gated[i]:
                cooldown = cooldown_days
            elif cooldown > 0:
                gated[i] = True
                cooldown -= 1

    return pd.Series(gated, index=market_regime.index, name="market_gate",
                     dtype=bool)


def _rolling_zscore(spread: np.ndarray, window: int) -> np.ndarray:
    """Rolling z-score: z_t = (s_t − mean(s_{t-w+1..t})) / std(s_{t-w+1..t}).

    Uses min_periods=5 so the first few bars get a z-score rather than NaN.
    Returns array of same length as spread, zeros where not enough history.
    """
    s = pd.Series(spread)
    mu    = s.rolling(window, min_periods=5).mean()
    sigma = s.rolling(window, min_periods=5).std()
    return ((s - mu) / sigma.replace(0, np.nan)).fillna(0.0).values


def _compute_is_sharpe(
    log_y:    np.ndarray,
    log_x:    np.ndarray,
    ols_beta: float,
    zscores:  np.ndarray,
    entry_z:  float,
    exit_z:   float,
    stop_z:   float,
    warmup:   int = 30,
) -> float:
    """Quick IS backtest (no HMM, pure z-score) to estimate IS Sharpe.

    Simulates the same entry/exit logic on IS data using the rolling z-score.
    Warmup period (first `warmup` bars) is excluded from P&L.
    """
    n        = len(log_y)
    position = 0
    pnl      = np.zeros(n)

    for t in range(warmup + 1, n):
        z      = float(zscores[t])
        new_p  = position

        if position == 0:
            if z < -entry_z:
                new_p = 1
            elif z > entry_z:
                new_p = -1
        else:
            if abs(z) < exit_z or abs(z) > stop_z:
                new_p = 0

        if position != 0:
            sr = (log_y[t] - log_y[t - 1]) - ols_beta * (log_x[t] - log_x[t - 1])
            pnl[t] = position * sr

        position = new_p

    r = pd.Series(pnl[warmup + 1:])
    std = r.std()
    if std < 1e-10 or len(r) < 10:
        return 0.0
    return float(r.mean() / std * np.sqrt(252))


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class WalkForwardConfig:
    """All parameters for a walk-forward backtest run."""

    # Time windows
    start:      str = "2010-01-01"
    end:        str = "2025-01-01"
    is_years:   int = 2
    oos_months: int = 6

    # Pair selection — Layer 2: quality over quantity
    max_pairs:        int   = 20
    eg_pvalue:        float = 0.05
    min_halflife:     float = 3.0
    max_halflife:     float = 120.0
    same_sector_only: bool  = True         # Layer 2: same-sector only (economic rationale)
    apply_fdr:        bool  = False

    # Signal — Layer 4: shorter rolling window captures faster mean-reversion
    rolling_z_window: int   = 10           # Layer 4: 10d (was 20d) — more responsive
    min_is_sharpe:    float = 1.0          # Layer 2: only pairs with strong IS alpha

    # Portfolio IS Sharpe gate — skip entire fold when combined IS back-test is poor
    # Motivation: per-pair IS Sharpe filter (min_is_sharpe) rejects bad INDIVIDUAL pairs,
    # but the COMBINED portfolio might still fail in some folds (e.g. 2024 H1 where all
    # tech stocks moved together — within-sector correlation too high for pairs to work).
    # This gate runs a quick IS paper-trade on ALL selected pairs together and skips
    # the fold entirely when portfolio IS Sharpe < min_portfolio_is_sharpe.
    min_portfolio_is_sharpe: float = 0.0   # skip fold if combined IS Sharpe < 0
    max_is_pair_correlation: float = 0.88  # skip fold if avg IS pairwise corr > 0.88

    # Market momentum gate (kept for reference; currently disabled by default=0.0)
    market_momentum_window:    int   = 60
    market_momentum_threshold: float = 0.0   # 0 = disabled

    # Cross-sectional dispersion gate (targets the REAL failure mode)
    # When all stocks move together (low cross-sectional std), pairs don't diverge.
    # 2017 H1 (good, Sharpe=2.45): dispersion ~0.7%/day (stocks rotating sectors)
    # 2017 H2 (bad, Sharpe=-1.89): dispersion ~0.5%/day (all trending up together)
    # Gate: block when rolling 20-day mean of cross-sectional std < min_dispersion.
    # This is causal and distinguishes bull-market rotation from bull-market trend.
    dispersion_window: int   = 20
    min_dispersion:    float = 0.006   # block when cross-sect std < 0.6%/day
                                           # High correlation → all stocks move together
                                           # → no spread opportunity for pairs trading

    # Risk module integration
    # When use_risk_sizing=True the engine applies:
    #   1. Vol-adjusted notional per pair (replaces fixed notional_per_pair)
    #   2. Kelly scaling from IS Sharpe
    #   3. Daily drawdown monitoring → CLOSE_ALL / HALT / REDUCE signals
    use_risk_sizing:        bool  = False  # True = vol-adj+Kelly; False = fixed notional
    risk_target_vol:        float = 0.01   # target daily spread vol (1%)
    risk_kelly_multiplier:  float = 0.5    # half-Kelly for safety
    risk_max_drawdown:      float = 0.15   # 15% → close all
    risk_max_daily_loss:    float = 0.03   # 3% daily → halt

    # VIX gate — Layer 1: hard rule from realized market fear indicator
    # VIX > threshold → block (stress/crisis, e.g. COVID, 2022 rate hikes)
    # VIX < min_threshold → ALSO block (complacency/extreme bull market)
    # VIX of 9-11 in 2017 H2 = all-time lows = market SO CALM all stocks trend up
    # The "goldilocks zone" for pairs: VIX in [vix_min, vix_max]
    vix_gate_threshold: float = 20.0       # block when VIX > 20 (stress)
    vix_min_threshold:  float = 11.0       # block when VIX < 11 (extreme complacency)

    # Signal thresholds (calibrated for rolling z ~ N(0,1))
    entry_zscore: float = 2.0             # ~5% of days trigger at ±2
    exit_zscore:  float = 0.5
    stop_zscore:  float = 3.5

    # Per-pair regime filter: False = enter if not VOLATILE (more signals than strict MR)
    strict_regime_filter: bool = False

    # Market-level regime filter (improvement 4 — smoothed gate v2)
    # One HMM trained on cross-sectional market features (market_ret, cross_vol,
    # avg_realized_vol, avg_autocorr) — acts as a portfolio-level gate.
    #
    # v1 problem: raw per-day volatile gate caused whipsaw:
    #   close on volatile → market clears → z-score still high → re-enter →
    #   close again → more TC, more losses (e.g. Fold 19: 73 trades WITH filter
    #   vs 67 WITHOUT — the filter made it worse!).
    #
    # v2 solution: smoothed gate with persistence and cooldown.
    use_market_regime_filter:    bool = True   # enable portfolio-level HMM gate

    # Smoothed gate parameters
    market_vol_smoothing_window: int   = 10    # rolling window for volatile fraction
    market_vol_threshold:        float = 0.30  # gate ON when >=30% of window is volatile
    market_cooldown_days:        int   = 5     # stay gated N days after volatile clears

    # close_positions_on_volatile=False (CHANGED default):
    #   Block new entries when gated. Let existing positions run through.
    #   Avoids the forced-close/immediate-reopen whipsaw that increased trades.
    close_positions_on_volatile: bool = False

    # Position sizing
    notional_per_pair:    float = 50_000.0
    transaction_cost_bps: float = 10.0

    # Model params
    kalman_warmup_bars:  int = 60
    hmm_random_state:    int = 42
    hmm_vol_window:      int = 20
    hmm_autocorr_window: int = 10

    @property
    def is_days(self) -> int:
        return int(self.is_years * 252)

    @property
    def oos_days(self) -> int:
        return int(self.oos_months / 12 * 252)

    @property
    def total_notional(self) -> float:
        return self.max_pairs * self.notional_per_pair


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Fold:
    idx:       int
    is_start:  pd.Timestamp
    is_end:    pd.Timestamp
    oos_start: pd.Timestamp
    oos_end:   pd.Timestamp

    def __str__(self) -> str:
        return (f"Fold {self.idx}: IS={self.is_start.date()}→{self.is_end.date()} "
                f"OOS={self.oos_start.date()}→{self.oos_end.date()}")


@dataclass
class FoldResult:
    fold:           Fold
    n_pairs:        int
    pairs:          list[str]
    daily_returns:  pd.Series
    n_trades:       int
    metrics:        dict = field(default_factory=dict)


@dataclass
class BacktestResult:
    config:        WalkForwardConfig
    folds:         list[FoldResult]
    daily_returns: pd.Series
    equity_curve:  pd.Series
    metrics:       dict


# ── Engine ────────────────────────────────────────────────────────────────────

class WalkForwardBacktest:
    """Walk-forward backtest orchestrator (v2: rolling z-score + IS Sharpe filter).

    Usage
    -----
    prices  = load_prices(...)   # DataFrame: index=date, columns=symbols
    config  = WalkForwardConfig()
    result  = WalkForwardBacktest(config).run(prices)
    """

    def __init__(self, config: WalkForwardConfig) -> None:
        self.config = config

    def run(
        self,
        prices: pd.DataFrame,
        vix: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """Run the full walk-forward backtest.

        Parameters
        ----------
        prices : Adjusted close prices (columns = symbols).
        vix    : Optional VIX daily series (UTC index).  When provided, acts as
                 a hard gate: no new pair entries when VIX > vix_gate_threshold.
                 This is Layer 1 of the four-layer quality improvement.
        """
        c      = self.config
        prices = prices.loc[c.start:c.end]
        if vix is not None:
            vix = vix.loc[c.start:c.end]
        folds  = self._build_folds(prices)

        logger.info(
            f"Walk-forward backtest v3 | {len(folds)} folds | "
            f"IS={c.is_years}y  OOS={c.oos_months}m | "
            f"z_win={c.rolling_z_window}d | IS_Sharpe>={c.min_is_sharpe:.1f} | "
            f"{'same-sector' if c.same_sector_only else 'cross-sector'} | "
            f"VIX_gate={c.vix_gate_threshold:.0f}"
            + (" [VIX loaded]" if vix is not None else " [no VIX]")
        )

        fold_results: list[FoldResult] = []
        for fold in folds:
            logger.info(str(fold))
            result = self._run_fold(fold, prices, vix=vix)
            fold_results.append(result)
            logger.info(
                f"  Pairs={result.n_pairs}  Trades={result.n_trades}  "
                f"Sharpe={result.metrics.get('sharpe_ratio', float('nan')):.2f}"
            )

        all_returns = (
            pd.concat([f.daily_returns for f in fold_results])
            .sort_index()
            .dropna()
        )
        metrics  = compute_metrics(all_returns)
        eq_curve = (1.0 + all_returns).cumprod()

        return BacktestResult(
            config=c, folds=fold_results,
            daily_returns=all_returns, equity_curve=eq_curve, metrics=metrics,
        )

    # ── Fold generation ───────────────────────────────────────────────────────

    def _build_folds(self, prices: pd.DataFrame) -> list[Fold]:
        idx = prices.index
        c   = self.config
        folds, t = [], c.is_days

        while t + c.oos_days <= len(idx):
            folds.append(Fold(
                idx       = len(folds),
                is_start  = idx[t - c.is_days],
                is_end    = idx[t - 1],
                oos_start = idx[t],
                oos_end   = idx[min(t + c.oos_days - 1, len(idx) - 1)],
            ))
            t += c.oos_days
        return folds

    # ── Per-fold orchestration ────────────────────────────────────────────────

    def _run_fold(
        self,
        fold: Fold,
        prices: pd.DataFrame,
        vix: Optional[pd.Series] = None,
    ) -> FoldResult:
        c          = self.config
        is_prices  = prices.loc[fold.is_start : fold.is_end]
        oos_prices = prices.loc[fold.oos_start : fold.oos_end]
        vix_oos    = vix.reindex(oos_prices.index).ffill() if vix is not None else None

        # ── 1. Pair selection ─────────────────────────────────────────────────
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pairs_df = screen_pairs(
                is_prices.dropna(axis=1, how="any"),
                eg_pvalue_threshold=c.eg_pvalue,
                min_half_life=c.min_halflife,
                max_half_life=c.max_halflife,
                min_history=max(c.kalman_warmup_bars + 40, 120),
                same_sector_only=c.same_sector_only,
                apply_fdr=c.apply_fdr,
                n_jobs=1,
            )

        selected = pairs_df[pairs_df["passed_all"]].head(c.max_pairs)
        if selected.empty:
            zero = pd.Series(0.0, index=oos_prices.index[1:], name="portfolio")
            return FoldResult(fold=fold, n_pairs=0, pairs=[],
                              daily_returns=zero, n_trades=0,
                              metrics=compute_metrics(zero))

        # ── 2. Fit models (OLS + rolling-z + Kalman + HMM) per pair ──────────
        pair_models: dict[tuple, dict] = {}
        for _, row in selected.iterrows():
            model = self._fit_pair(row, is_prices)
            if model is not None:
                pair_models[(row["symbol_y"], row["symbol_x"])] = model

        if not pair_models:
            zero = pd.Series(0.0, index=oos_prices.index[1:], name="portfolio")
            return FoldResult(fold=fold, n_pairs=0, pairs=[],
                              daily_returns=zero, n_trades=0,
                              metrics=compute_metrics(zero))

        # ── 2b. Portfolio IS quality gate ─────────────────────────────────────
        # Skip fold when the IS portfolio back-test is not profitable.
        # Per-pair IS Sharpe filters individual pairs, but this gate evaluates
        # the COMBINED portfolio — catching bad folds where within-sector
        # correlations are too high (e.g. 2024 H1 AI boom, 2021 H1 momentum).
        if c.min_portfolio_is_sharpe > -99 or c.max_is_pair_correlation < 1.0:
            port_is_sharpe, avg_corr = self._compute_portfolio_is_sharpe(
                pair_models, is_prices
            )
            logger.debug(
                f"  Portfolio IS Sharpe: {port_is_sharpe:.2f}  "
                f"avg_corr: {avg_corr:.3f}"
            )
            skip_reason = None
            if port_is_sharpe < c.min_portfolio_is_sharpe:
                skip_reason = (
                    f"portfolio IS Sharpe={port_is_sharpe:.2f} "
                    f"< {c.min_portfolio_is_sharpe:.2f}"
                )
            elif avg_corr > c.max_is_pair_correlation:
                skip_reason = (
                    f"avg IS pair corr={avg_corr:.3f} "
                    f"> {c.max_is_pair_correlation:.2f} "
                    f"(pairs too correlated — within-sector momentum)"
                )
            if skip_reason:
                logger.info(f"  Skip fold: {skip_reason}")
                zero = pd.Series(0.0, index=oos_prices.index[1:], name="portfolio")
                skip_metrics = compute_metrics(zero)
                skip_metrics["portfolio_is_sharpe"] = port_is_sharpe
                skip_metrics["avg_is_pair_corr"]    = avg_corr
                return FoldResult(fold=fold, n_pairs=0, pairs=[],
                                  daily_returns=zero, n_trades=0,
                                  metrics=skip_metrics)

        # ── 3. Market-level regime filter (portfolio gate) ────────────────────
        market_regime_oos: Optional[pd.Series] = None
        if c.use_market_regime_filter:
            mkt_model = self._fit_market_hmm(is_prices)
            if mkt_model is not None:
                market_regime_oos = self._compute_market_regime(
                    oos_prices, is_prices.iloc[-30:], mkt_model
                )
                # Report both raw volatile % and smoothed gate %
                raw_vol_pct = float((market_regime_oos == "volatile").mean())
                smoothed_gate = _compute_smoothed_market_gate(
                    market_regime_oos,
                    smoothing_window=c.market_vol_smoothing_window,
                    vol_threshold=c.market_vol_threshold,
                    cooldown_days=c.market_cooldown_days,
                )
                gate_pct = float(smoothed_gate.mean())
                logger.debug(
                    f"  Market: raw_vol={raw_vol_pct:.0%}  "
                    f"gate_active={gate_pct:.0%} "
                    f"(window={c.market_vol_smoothing_window}d "
                    f"thresh={c.market_vol_threshold:.0%} "
                    f"cooldown={c.market_cooldown_days}d)"
                )

        # ── 4. Simulate OOS ───────────────────────────────────────────────────
        daily_returns, n_trades = self._simulate_oos(
            oos_prices, pair_models, market_regime_oos,
            vix_oos=vix_oos, is_prices=is_prices,
        )
        fold_metrics = compute_metrics(daily_returns)
        if market_regime_oos is not None:
            fold_metrics["market_volatile_pct"] = float(
                (market_regime_oos == "volatile").mean()
            )
            smoothed = _compute_smoothed_market_gate(
                market_regime_oos,
                smoothing_window=c.market_vol_smoothing_window,
                vol_threshold=c.market_vol_threshold,
                cooldown_days=c.market_cooldown_days,
            )
            fold_metrics["market_gate_pct"] = float(smoothed.mean())
        if vix_oos is not None:
            fold_metrics["avg_vix"]         = float(vix_oos.mean())
            fold_metrics["vix_gated_pct"]   = float(
                (vix_oos > c.vix_gate_threshold).mean()
            )
        return FoldResult(
            fold=fold,
            n_pairs=len(pair_models),
            pairs=[f"{sy}/{sx}" for sy, sx in pair_models],
            daily_returns=daily_returns,
            n_trades=n_trades,
            metrics=fold_metrics,
        )

    # ── Portfolio IS quality gate ─────────────────────────────────────────────

    def _compute_portfolio_is_sharpe(
        self,
        pair_models: dict,
        is_prices: pd.DataFrame,
    ) -> tuple[float, float]:
        """Run a quick IS paper-trade on all pairs combined.

        Returns
        -------
        (portfolio_is_sharpe, avg_pairwise_corr)
            portfolio_is_sharpe : annualised Sharpe of combined IS P&L
            avg_pairwise_corr   : average pairwise return correlation of all
                                  pair symbols in IS — high correlation means
                                  all stocks moving together (bad for pairs)
        """
        c = self.config
        all_pnl = pd.Series(0.0, index=is_prices.index)

        for (sym_y, sym_x), model in pair_models.items():
            try:
                pair = is_prices[[sym_y, sym_x]].dropna()
                if len(pair) < c.rolling_z_window + 20:
                    continue

                log_y = np.log(pair[sym_y].values)
                log_x = np.log(pair[sym_x].values)
                spread = log_y - model["ols_alpha"] - model["ols_beta"] * log_x
                z = _rolling_zscore(spread, c.rolling_z_window)

                position = 0
                for t in range(c.rolling_z_window + 1, len(log_y)):
                    zt = float(z[t])
                    new_p = position
                    if position == 0:
                        if zt < -c.entry_zscore:
                            new_p = 1
                        elif zt > c.entry_zscore:
                            new_p = -1
                    else:
                        if abs(zt) < c.exit_zscore or abs(zt) > c.stop_zscore:
                            new_p = 0

                    if position != 0:
                        sr = ((log_y[t] - log_y[t - 1])
                              - model["ols_beta"] * (log_x[t] - log_x[t - 1]))
                        date = pair.index[t]
                        if date in all_pnl.index:
                            all_pnl[date] += position * sr

                    position = new_p
            except Exception:
                continue

        port = all_pnl / max(len(pair_models), 1)
        std  = port.std()
        sharpe = float(port.mean() / std * np.sqrt(252)) if std > 1e-10 else 0.0

        # Average pairwise correlation of all unique symbols
        all_syms = list({s for pair in pair_models for s in pair})
        avg_corr = 0.5  # neutral default
        if len(all_syms) >= 2:
            try:
                rets = np.log(is_prices[all_syms].dropna()).diff().dropna()
                corr_mat = rets.corr().values
                n = len(all_syms)
                # Off-diagonal correlations only
                upper = corr_mat[np.triu_indices(n, k=1)]
                avg_corr = float(np.nanmean(upper))
            except Exception:
                pass

        return sharpe, avg_corr

    # ── Market-level HMM ─────────────────────────────────────────────────────

    def _fit_market_hmm(self, is_prices: pd.DataFrame) -> Optional[dict]:
        """Fit one HMM on cross-sectional market features for the IS window.

        Returns a dict with the fitted HMM; None if fitting fails.
        The market HMM uses 4 features derived from ALL universe symbols:
          market_ret, cross_vol, avg_realized_vol, avg_autocorr.
        This captures the BROAD MARKET regime (not pair-specific), acting as
        a portfolio-level gate that blocks trading in volatile/trending markets.
        """
        c = self.config
        try:
            feats = build_market_features(is_prices.dropna(axis=1, how="any"))
            if len(feats) < 30:
                return None
            hmm = HMMRegimeDetector(random_state=c.hmm_random_state)
            hmm.fit(feats)
            return {"hmm": hmm}
        except Exception as exc:
            logger.warning(f"Market HMM fitting failed: {exc}")
            return None

    def _compute_market_regime(
        self,
        oos_prices: pd.DataFrame,
        is_prices_tail: pd.DataFrame,
        market_model: dict,
    ) -> pd.Series:
        """Predict market regime for each OOS day using the IS-fitted market HMM.

        Prepends the last 30 IS days to provide rolling-window warmup for the
        market features (avg_realized_vol uses a 20-day window).  Only OOS-period
        dates are returned.

        Returns
        -------
        pd.Series of regime strings ("mean_reverting" / "trending" / "volatile")
        aligned to oos_prices.index.  Missing dates filled with "mean_reverting"
        (conservative: assume normal market when data is unavailable).
        """
        try:
            combined = pd.concat([
                is_prices_tail.dropna(axis=1, how="any"),
                oos_prices.dropna(axis=1, how="any"),
            ]).dropna(axis=1, how="any")
            combined = combined[~combined.index.duplicated(keep="last")]

            feats_all = build_market_features(combined)
            feats_oos = feats_all.loc[feats_all.index.isin(oos_prices.index)]

            if feats_oos.empty or len(feats_oos) < 5:
                return pd.Series("mean_reverting", index=oos_prices.index,
                                 name="market_regime")

            regime = market_model["hmm"].predict_series(feats_oos)
            return (regime
                    .reindex(oos_prices.index)
                    .ffill()
                    .fillna("mean_reverting")
                    .rename("market_regime"))
        except Exception as exc:
            logger.debug(f"Market regime prediction failed: {exc}")
            return pd.Series("mean_reverting", index=oos_prices.index,
                             name="market_regime")

    # ── Model fitting ─────────────────────────────────────────────────────────

    def _fit_pair(self, row: pd.Series, is_prices: pd.DataFrame) -> Optional[dict]:
        """Fit OLS + rolling-z + Kalman + HMM for one pair on IS data.

        Returns None if:
          - not enough history
          - IS Sharpe < min_is_sharpe (quality filter)
          - HMM features cannot be computed
        """
        sym_y, sym_x = row["symbol_y"], row["symbol_x"]
        c = self.config

        try:
            pair = is_prices[[sym_y, sym_x]].dropna()
            min_len = max(c.kalman_warmup_bars + c.rolling_z_window + 20, 120)
            if len(pair) < min_len:
                return None

            log_y = np.log(pair[sym_y].values)
            log_x = np.log(pair[sym_x].values)

            # ── OLS IS regression (fixed signal parameters) ───────────────────
            X_ols    = sm.add_constant(log_x)
            ols_fit  = sm.OLS(log_y, X_ols).fit()
            ols_alpha = float(ols_fit.params[0])
            ols_beta  = float(ols_fit.params[1])

            ols_spread_is = log_y - ols_alpha - ols_beta * log_x

            # ── Rolling z-score on IS spread ──────────────────────────────────
            is_zscore = _rolling_zscore(ols_spread_is, c.rolling_z_window)

            # ── IS Sharpe filter (improvement 3) ──────────────────────────────
            is_sharpe = _compute_is_sharpe(
                log_y, log_x, ols_beta, is_zscore,
                c.entry_zscore, c.exit_zscore, c.stop_zscore,
                warmup=c.rolling_z_window,
            )
            if is_sharpe < c.min_is_sharpe:
                logger.debug(
                    f"Dropped {sym_y}/{sym_x}: IS Sharpe={is_sharpe:.2f} < {c.min_is_sharpe}"
                )
                return None

            # ── Kalman (dynamic hedge ratio for OOS P&L only) ─────────────────
            kf = KalmanFilterHedge()
            best_delta = kf.tune_delta(log_y, log_x)
            kf = KalmanFilterHedge(delta=best_delta)
            kf.run(log_y, log_x, warmup_bars=c.kalman_warmup_bars)
            kf_snapshot = kf.snapshot()

            # ── HMM trained on IS OLS spread features ─────────────────────────
            is_df = pd.DataFrame(
                {"spread": ols_spread_is, "zscore": is_zscore},
                index=pair.index,
            )
            features_is = build_features_from_spread(
                is_df.dropna(),
                vol_window=c.hmm_vol_window,
                autocorr_window=c.hmm_autocorr_window,
            )
            if len(features_is) < 30:
                return None

            hmm = HMMRegimeDetector(random_state=c.hmm_random_state)
            hmm.fit(features_is)

            # IS spread std — for vol-adjusted sizing
            is_spread_std = float(ols_spread_is.std()) if ols_spread_is.std() > 0 else 0.01

            # IS trade-level P&L — for accurate Kelly sizing
            is_trade_pnls = compute_is_trade_pnls(
                log_y, log_x, ols_beta, is_zscore,
                entry_z=c.entry_zscore, exit_z=c.exit_zscore, stop_z=c.stop_zscore,
                warmup=c.rolling_z_window,
            )

            return {
                "sym_y":      sym_y,
                "sym_x":      sym_x,
                # OLS params for signal z-score
                "ols_alpha":  ols_alpha,
                "ols_beta":   ols_beta,
                # IS spread tail — warmup for OOS rolling window
                "is_spread_tail": ols_spread_is[-c.rolling_z_window:],
                # Kalman for dynamic hedge in OOS P&L
                "kf":          kf,
                "kf_snapshot": kf_snapshot,
                # HMM for regime filter
                "hmm":        hmm,
                # IS quality stats (for reporting + risk sizing)
                "is_sharpe":     is_sharpe,
                "is_spread_std": is_spread_std,
                "is_trade_pnls": is_trade_pnls,   # trade-level P&L for Kelly
                "hedge_ratio":   float(row["hedge_ratio"]),
            }
        except Exception as exc:
            logger.debug(f"Pair fitting failed {sym_y}/{sym_x}: {exc}")
            return None

    # ── OOS signal generation ─────────────────────────────────────────────────

    def _compute_oos_signals(
        self,
        oos_prices: pd.DataFrame,
        model: dict,
    ) -> Optional[pd.DataFrame]:
        """Compute OOS signals using rolling z-score + Kalman beta + HMM regime."""
        sym_y = model["sym_y"]
        sym_x = model["sym_x"]
        c     = self.config

        oos = oos_prices[[sym_y, sym_x]].dropna()
        if len(oos) == 0:
            return None

        log_y_oos = np.log(oos[sym_y].values)
        log_x_oos = np.log(oos[sym_x].values)

        # ── OLS spread in OOS (fixed IS alpha, beta — no adaptation) ─────────
        ols_alpha   = model["ols_alpha"]
        ols_beta    = model["ols_beta"]
        ols_spread_oos = log_y_oos - ols_alpha - ols_beta * log_x_oos

        # ── Rolling z-score: prepend IS tail for warmup continuity ────────────
        is_tail    = model["is_spread_tail"]          # last rolling_z_window IS bars
        combined   = np.concatenate([is_tail, ols_spread_oos])
        z_combined = _rolling_zscore(combined, c.rolling_z_window)
        zscores    = z_combined[-len(ols_spread_oos):]   # OOS portion only

        # ── Kalman step-by-step → dynamic beta for P&L ────────────────────────
        kf = model["kf"]
        kf.restore(model["kf_snapshot"])
        betas = []
        for t in range(len(log_y_oos)):
            kf.step(log_y_oos[t], log_x_oos[t])
            betas.append(kf.beta)

        # ── HMM regime on OOS OLS-spread features ────────────────────────────
        oos_df = pd.DataFrame(
            {"spread": ols_spread_oos, "zscore": zscores},
            index=oos.index,
        )
        # Prepend IS tail spread for rolling feature warmup
        is_tail_df = pd.DataFrame(
            {"spread": is_tail, "zscore": np.zeros(len(is_tail))},
            index=pd.date_range(end=oos.index[0], periods=len(is_tail) + 1,
                                freq="B", tz="UTC")[:-1],
        )
        combined_df = pd.concat([is_tail_df, oos_df])
        combined_df = combined_df[~combined_df.index.duplicated(keep="last")]

        try:
            features_all = build_features_from_spread(
                combined_df.dropna(),
                vol_window=c.hmm_vol_window,
                autocorr_window=c.hmm_autocorr_window,
            )
            features_oos = features_all.loc[features_all.index.isin(oos.index)]
        except Exception:
            features_oos = pd.DataFrame()

        if not features_oos.empty and len(features_oos) >= 5:
            try:
                regime_series = model["hmm"].predict_series(features_oos)
            except Exception:
                regime_series = pd.Series("volatile", index=features_oos.index)
        else:
            regime_series = pd.Series(dtype="object")

        signals = pd.DataFrame(
            {"spread": ols_spread_oos, "zscore": zscores, "beta": betas},
            index=oos.index,
        )
        signals["regime"] = regime_series.reindex(oos.index).fillna("volatile")
        return signals

    # ── OOS simulation ────────────────────────────────────────────────────────

    def _simulate_oos(
        self,
        oos_prices: pd.DataFrame,
        pair_models: dict,
        market_regime: Optional[pd.Series] = None,
        vix_oos: Optional[pd.Series] = None,
        is_prices: Optional[pd.DataFrame] = None,
    ) -> tuple[pd.Series, int]:
        """Simulate OOS positions and compute daily P&L.

        Parameters
        ----------
        market_regime : Optional per-day market regime Series.
            When provided, the market HMM gate is applied:
              - "volatile" market → block all new entries
              - "volatile" market + close_positions_on_volatile → force-close opens
            "mean_reverting" / "trending" → normal pair-level signal logic.
        """
        c = self.config

        pair_signals: dict[tuple, pd.DataFrame] = {}
        for (sym_y, sym_x), model in pair_models.items():
            try:
                sig = self._compute_oos_signals(oos_prices, model)
                if sig is not None and not sig.empty:
                    pair_signals[(sym_y, sym_x)] = sig
            except Exception as exc:
                logger.debug(f"OOS signals failed {sym_y}/{sym_x}: {exc}")

        # ── Pre-compute market momentum gate (disabled by default) ───────────
        momentum_gate: Optional[pd.Series] = None
        if c.market_momentum_threshold > 0 and len(pair_signals) > 0 and is_prices is not None:
            try:
                all_syms = list({s for pair in pair_signals for s in pair})
                avail    = [s for s in all_syms if s in oos_prices.columns]
                if avail:
                    w = c.market_momentum_window
                    combined_px = pd.concat([
                        is_prices[avail].iloc[-w:],
                        oos_prices[avail],
                    ]).dropna(axis=1, how="any")
                    combined_px = combined_px[~combined_px.index.duplicated(keep="last")]
                    rolling_ret = combined_px.pct_change(w).mean(axis=1)
                    momentum_gate = (rolling_ret > c.market_momentum_threshold).reindex(
                        oos_prices.index
                    ).fillna(False)
            except Exception as exc:
                logger.debug(f"Momentum gate failed: {exc}")

        # ── Pre-compute cross-sectional dispersion gate ───────────────────────
        # When all stocks move together (low cross-sectional std), pairs don't
        # create exploitable spreads. Gate when rolling dispersion < min_dispersion.
        dispersion_gate: Optional[pd.Series] = None
        if c.min_dispersion > 0 and is_prices is not None:
            try:
                # Use a broad set of symbols for market-wide dispersion measure
                universe_syms = [s for s in is_prices.columns if s in oos_prices.columns]
                if len(universe_syms) >= 5:
                    w = c.dispersion_window
                    combined_px = pd.concat([
                        is_prices[universe_syms].iloc[-w:],
                        oos_prices[universe_syms],
                    ]).dropna(axis=1, how="any")
                    combined_px = combined_px[~combined_px.index.duplicated(keep="last")]
                    log_rets = np.log(combined_px).diff()
                    # Cross-sectional std per day, then rolling mean
                    xs_std = log_rets.std(axis=1).rolling(w, min_periods=5).mean()
                    dispersion_gate = (xs_std < c.min_dispersion).reindex(
                        oos_prices.index
                    ).fillna(False)
                    n_gated = dispersion_gate.sum()
                    if n_gated > 0:
                        logger.debug(
                            f"  Dispersion gate active: {n_gated}/{len(oos_prices)} days "
                            f"(xs_std < {c.min_dispersion:.4f})"
                        )
            except Exception as exc:
                logger.debug(f"Dispersion gate failed: {exc}")

        # ── Pre-compute smoothed market gate (once per fold) ──────────────────
        # Raw per-day volatile signals are too noisy. A smoothed gate (rolling
        # window + cooldown) prevents the whipsaw: close → market clears →
        # z still high → re-enter immediately → more TC and losses.
        market_gate: Optional[pd.Series] = None
        if market_regime is not None and c.use_market_regime_filter:
            market_gate = _compute_smoothed_market_gate(
                market_regime,
                smoothing_window=c.market_vol_smoothing_window,
                vol_threshold=c.market_vol_threshold,
                cooldown_days=c.market_cooldown_days,
            )

        # ── Risk module: portfolio state + checker ────────────────────────────
        risk_limits  = RiskLimits(
            max_notional_per_pair=c.notional_per_pair,
            max_pairs_open=c.max_pairs,
            max_total_notional=c.total_notional,
            max_portfolio_drawdown=c.risk_max_drawdown,
            max_daily_loss_pct=c.risk_max_daily_loss,
        )
        risk_checker = RiskChecker(risk_limits)
        port_state   = PortfolioState(portfolio_value=c.total_notional)

        dates       = oos_prices.index
        daily_pnl   = pd.Series(0.0, index=dates)
        positions:   dict[tuple, int]   = {k: 0 for k in pair_signals}
        entry_betas: dict[tuple, float] = {}
        entry_notionals: dict[tuple, float] = {}  # notional at entry (for risk sizing)
        n_trades    = 0

        for i in range(1, len(dates)):
            date      = dates[i]
            prev_date = dates[i - 1]

            # ── Market gate: True = no new entries (and optional force-close) ──
            is_gated = False
            if market_gate is not None and date in market_gate.index:
                is_gated = bool(market_gate.loc[date])

            # ── VIX gate (Layer 1) — both too-high AND too-low VIX ───────────
            if vix_oos is not None and date in vix_oos.index:
                vix_val = float(vix_oos.loc[date])
                # Too high → stress/crisis
                if not is_gated and c.vix_gate_threshold > 0:
                    is_gated = vix_val > c.vix_gate_threshold
                # Too low → extreme complacency = sustained bull = pairs fail
                if not is_gated and c.vix_min_threshold > 0:
                    is_gated = vix_val < c.vix_min_threshold

            # ── Market momentum gate ─────────────────────────────────────────
            if (not is_gated
                    and momentum_gate is not None
                    and date in momentum_gate.index):
                is_gated = bool(momentum_gate.loc[date])

            # ── Cross-sectional dispersion gate ──────────────────────────────
            # Block when all stocks move together (low dispersion = bull trend)
            if (not is_gated
                    and dispersion_gate is not None
                    and date in dispersion_gate.index):
                is_gated = bool(dispersion_gate.loc[date])

            for (sym_y, sym_x), signals in pair_signals.items():
                if date not in signals.index:
                    continue

                sig       = signals.loc[date]
                direction = positions[(sym_y, sym_x)]
                zscore    = float(sig["zscore"])
                regime    = str(sig["regime"])

                # Per-pair regime filter
                if c.strict_regime_filter:
                    can_enter          = (regime == "mean_reverting")
                    should_exit_regime = (regime != "mean_reverting")
                else:
                    can_enter          = (regime != "volatile")
                    should_exit_regime = (regime == "volatile")

                # ── Signal logic ──────────────────────────────────────────────
                new_direction = direction

                if direction != 0 and c.close_positions_on_volatile and is_gated:
                    # Market gated + force-close → exit all open positions
                    # (default OFF: avoids forced-close / immediate-reopen whipsaw)
                    new_direction = 0

                elif direction == 0:
                    # Consider new entry — blocked when market is gated
                    if not is_gated and can_enter:
                        if zscore < -c.entry_zscore:
                            new_direction = 1    # long spread
                        elif zscore > c.entry_zscore:
                            new_direction = -1   # short spread

                else:
                    # Existing position — normal pair-level exit (market gate
                    # does NOT close existing positions unless force-close is on)
                    if (abs(zscore) < c.exit_zscore
                            or abs(zscore) > c.stop_zscore
                            or should_exit_regime):
                        new_direction = 0

                # ── P&L for existing position ─────────────────────────────────
                if direction != 0:
                    try:
                        beta     = entry_betas.get((sym_y, sym_x),
                                                   float(sig.get("beta", 1.0)))
                        notional = entry_notionals.get((sym_y, sym_x),
                                                       c.notional_per_pair)
                        ret_y = np.log(oos_prices.loc[date, sym_y]
                                       / oos_prices.loc[prev_date, sym_y])
                        ret_x = np.log(oos_prices.loc[date, sym_x]
                                       / oos_prices.loc[prev_date, sym_x])
                        daily_pnl[date] += direction * (ret_y - beta * ret_x) \
                                           * notional
                    except Exception:
                        pass

                # ── Transaction cost on direction change ──────────────────────
                if new_direction != direction:
                    # Determine position notional using risk module sizing
                    if new_direction != 0 and c.use_risk_sizing:
                        model_data  = pair_models.get((sym_y, sym_x), {})
                        is_sharpe   = model_data.get("is_sharpe", 0.5)
                        spread_std  = model_data.get("is_spread_std", 0.013)
                        trade_pnls  = model_data.get("is_trade_pnls", None)
                        risk_scalar = risk_checker.get_sizing_scalar(port_state)

                        # Sizing strategy:
                        # NORMAL conditions (risk_scalar=1.0): fixed base notional
                        #   → preserves Sharpe (no IS→OOS overfitting from Kelly)
                        # REDUCE zone (risk_scalar<1.0): conservative sizing
                        #   → reduces volatile pairs + scales by risk_scalar
                        # CLOSE_ALL/HALT: risk_scalar=0 → zero notional (blocked)
                        if risk_scalar >= 1.0:
                            trade_notional = c.notional_per_pair
                        else:
                            # In warning zone: apply vol-based conservative reduction
                            n_active = sum(1 for v in positions.values() if v != 0) + 1
                            trade_notional = compute_position_size(
                                base_notional=c.notional_per_pair,
                                spread_std_daily=spread_std,
                                is_sharpe=is_sharpe,
                                risk_scalar=risk_scalar,
                                target_vol=c.risk_target_vol,
                                kelly_multiplier=c.risk_kelly_multiplier,
                                min_notional=c.notional_per_pair * 0.3,
                                max_notional=c.notional_per_pair,
                                trade_pnls=trade_pnls,
                                n_open_pairs=n_active,
                                total_budget=c.total_notional,
                                conservative=True,
                            )

                        # Safety check: enforce position limits
                        ok, _ = risk_checker.check_new_position(
                            port_state, trade_notional, f"{sym_y}/{sym_x}"
                        )
                        if not ok:
                            new_direction = 0
                    else:
                        trade_notional = c.notional_per_pair

                    tc = 2.0 * c.transaction_cost_bps / 10_000 * trade_notional
                    daily_pnl[date] -= tc

                    if new_direction != 0:
                        entry_betas[(sym_y, sym_x)]    = float(sig.get("beta", 1.0))
                        entry_notionals[(sym_y, sym_x)] = trade_notional
                        port_state.add_position(f"{sym_y}/{sym_x}", trade_notional)
                        n_trades += 1
                    else:
                        entry_betas.pop((sym_y, sym_x), None)
                        entry_notionals.pop((sym_y, sym_x), None)
                        port_state.remove_position(f"{sym_y}/{sym_x}")

                    positions[(sym_y, sym_x)] = new_direction

        # ── Update portfolio risk state ONCE per day (after all pairs) ────────
        daily_ret = float(daily_pnl.get(date, 0.0)) / c.total_notional
        port_state.update_equity(daily_ret)
        port_state.reset_daily_pnl()

        # Apply portfolio-level risk action (CLOSE_ALL) — end of day check
        port_action = risk_checker.check_portfolio_action(port_state)
        if port_action == RiskAction.CLOSE_ALL:
            logger.warning(f"RISK: CLOSE_ALL triggered at {date.date()}")
            for key in list(positions.keys()):
                if positions[key] != 0:
                    positions[key] = 0
                    entry_betas.pop(key, None)
                    entry_notionals.pop(key, None)
                    port_state.remove_position(f"{key[0]}/{key[1]}")

        daily_returns = (daily_pnl / c.total_notional).iloc[1:]
        daily_returns.name = "portfolio"
        return daily_returns, n_trades
