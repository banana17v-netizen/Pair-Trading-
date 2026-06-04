"""Cointegration-based pair selection for statistical arbitrage.

Pipeline (4 stages):
  Stage 1 — Engle-Granger (fast, all pairs or same-sector only):
      Tests every candidate pair. Retains raw p-value < eg_pvalue_threshold.
      Also computes OLS hedge ratio and spread std.

  Stage 2 — Benjamini-Hochberg FDR correction (new):
      Controls the false-discovery rate across all simultaneously tested pairs.
      With 1,378 pairs at alpha=0.05, EG alone expects ~69 false positives.
      BH-FDR reduces this to <= 5% of the truly-rejected set.
      Adds columns: eg_pvalue_fdr, eg_passed_fdr.

  Stage 3 — Johansen confirmation (only on FDR survivors):
      Confirms cointegration at 95% confidence via trace statistic.
      Extracts hedge ratio from the first eigenvector (more robust than OLS).

  Stage 4 — Half-life filter:
      Computes mean-reversion speed via AR(1)+intercept on the spread.
      Realistic range for daily data over long periods: [3, 120] days.

  Scoring & selection:
      Composite score = EG strength × Johansen strength × HL quality × sector bonus.
      Top N pairs are returned.

Fixes over v1:
  - FDR correction (reduces false positives from ~69 to ~5% of true rejections)
  - Same-sector filter option (170 vs 1,378 pairs; better economic rationale)
  - Sector info in every result (sector_y, sector_x, same_sector)
  - Updated scoring: Johansen trace margin + sector 1.2× bonus
  - optimal_hl updated from 15→70d (reflects empirical findings on 15-year data)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from loguru import logger
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from stat_arb.data.universe import get_sector, SECTOR_MAP


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class PairResult:
    """All statistics for one candidate pair (symbol_y, symbol_x)."""
    symbol_y: str
    symbol_x: str

    # Sector info
    sector_y:    str  = "Unknown"
    sector_x:    str  = "Unknown"
    same_sector: bool = False

    # Engle-Granger (raw)
    eg_pvalue:    float = np.nan
    eg_statistic: float = np.nan
    eg_passed:    bool  = False

    # FDR-adjusted EG
    eg_pvalue_fdr: float = np.nan
    eg_passed_fdr: bool  = False

    # Johansen
    johansen_passed:     bool  = False
    johansen_trace_stat: float = np.nan
    johansen_trace_cv95: float = np.nan

    # Spread characteristics
    ols_beta:     float = np.nan
    johansen_beta: float = np.nan
    half_life:    float = np.nan
    spread_std:   float = np.nan

    # Metadata
    n_obs:        int   = 0
    score:        float = 0.0

    @property
    def hedge_ratio(self) -> float:
        """Preferred hedge ratio: Johansen if available, else OLS."""
        if self.johansen_passed and np.isfinite(self.johansen_beta):
            return self.johansen_beta
        return self.ols_beta

    @property
    def passed_all(self) -> bool:
        """Passes raw EG + Johansen + finite half-life."""
        return (self.eg_passed and self.johansen_passed
                and np.isfinite(self.half_life) and self.half_life > 0)

    @property
    def passed_all_fdr(self) -> bool:
        """Passes FDR-adjusted EG + Johansen + finite half-life."""
        return (self.eg_passed_fdr and self.johansen_passed
                and np.isfinite(self.half_life) and self.half_life > 0)

    def __str__(self) -> str:
        return (f"{self.symbol_y}/{self.symbol_x}"
                f"  [{self.sector_y if self.same_sector else self.sector_y+'/'+self.sector_x}]"
                f"  EG={self.eg_pvalue:.4f}(fdr={self.eg_pvalue_fdr:.4f})"
                f"  HL={self.half_life:.1f}d  beta={self.hedge_ratio:.3f}"
                f"  score={self.score:.3f}")


# ── Core statistical tests ────────────────────────────────────────────────────

def engle_granger_test(
    log_y: np.ndarray,
    log_x: np.ndarray,
    trend: str = "c",
) -> tuple[float, float, float]:
    """Engle-Granger cointegration test.

    Returns (p_value, t_stat, ols_beta).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t_stat, p_value, _ = coint(log_y, log_x, trend=trend)

    X = sm.add_constant(log_x)
    ols = sm.OLS(log_y, X).fit()
    ols_beta = float(ols.params[1])

    return float(p_value), float(t_stat), ols_beta


def johansen_test(
    log_y: np.ndarray,
    log_x: np.ndarray,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> tuple[bool, float, float, float]:
    """Johansen cointegration test (trace statistic, 95% confidence).

    Returns (passed, johansen_beta, trace_stat, trace_cv95).
    """
    data = np.column_stack([log_y, log_x])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = coint_johansen(data, det_order, k_ar_diff)

    trace_stat = float(result.lr1[0])
    trace_cv95 = float(result.cvt[0, 1])
    passed     = trace_stat > trace_cv95

    v_y = result.evec[0, 0]
    v_x = result.evec[1, 0]
    if abs(v_y) > 1e-10:
        beta = float(-v_x / v_y)
        if beta < 0:
            beta = -beta
    else:
        beta = np.nan

    return passed, beta, trace_stat, trace_cv95


def compute_half_life(spread: np.ndarray) -> float:
    """Mean-reversion half-life via AR(1)+intercept regression.

    Fits: Δspread_t = a + λ · spread_{t-1} + ε_t
    Half-life = log(0.5) / log(1 + λ)

    The intercept term handles spreads with a non-zero long-run mean,
    preventing the bias that inflates half-life toward infinity.
    """
    spread = np.asarray(spread, dtype=float)
    if len(spread) < 10:
        return np.nan

    delta = np.diff(spread)
    lag   = spread[:-1]

    try:
        result  = sm.OLS(delta, sm.add_constant(lag)).fit()
        lambda_ = float(result.params[1])
    except Exception:
        return np.nan

    if lambda_ >= 0:
        return np.inf

    phi = 1.0 + lambda_
    if phi <= 0:
        return np.nan

    return float(np.log(0.5) / np.log(phi))


# ── FDR correction ────────────────────────────────────────────────────────────

def apply_fdr_correction(
    df: pd.DataFrame,
    alpha: float = 0.05,
    pvalue_col: str = "eg_pvalue",
) -> pd.DataFrame:
    """Apply Benjamini-Hochberg FDR correction to EG p-values.

    Only pairs that were actually tested (n_obs > 0) enter the BH pool.
    Untested pairs (e.g. cross-sector when same_sector_only=True) receive
    eg_pvalue_fdr=NaN and eg_passed_fdr=False.

    Including untested pairs (with NaN→1.0) in the pool would dilute the
    BH threshold — e.g. 1,208 untested pairs + 170 tested pairs makes the
    k=1 threshold 0.05/1378 = 3.6e-5 instead of the correct 0.05/170 = 2.9e-4.

    Parameters
    ----------
    df        : Output of screen_pairs() (must have eg_pvalue and n_obs columns).
    alpha     : FDR level. Default 0.05.
    pvalue_col: Name of the raw p-value column.

    Returns
    -------
    DataFrame with two new columns: eg_pvalue_fdr (BH-adjusted) and eg_passed_fdr.
    """
    df = df.copy()
    df["eg_pvalue_fdr"] = np.nan
    df["eg_passed_fdr"] = False

    # Only run BH on pairs that were actually tested
    # Fall back to all rows if n_obs column is absent (standalone use)
    tested = df["n_obs"] > 0 if "n_obs" in df.columns else pd.Series(True, index=df.index)
    if not tested.any():
        return df

    pvals = df.loc[tested, pvalue_col].fillna(1.0).values
    rejected, pvals_adj, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")

    df.loc[tested, "eg_pvalue_fdr"] = pvals_adj
    df.loc[tested, "eg_passed_fdr"] = rejected
    return df


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score(
    eg_pvalue: float,
    half_life: float,
    same_sector: bool,
    johansen_trace_stat: float = np.nan,
    johansen_trace_cv95: float = np.nan,
    optimal_hl: float = 70.0,
    sigma_hl: float = 30.0,
) -> float:
    """Composite pair ranking score (higher = better).

    Components:
      EG strength       : -log10(p)  [strength of cointegration evidence]
      Johansen margin   : (trace - cv) / cv  [how far above critical value]
      Half-life quality : Gaussian centred at optimal_hl  [mean-reversion speed]
      Sector bonus      : 1.2× multiplier for same-sector pairs

    optimal_hl is set to 70 days — midpoint of the empirically observed
    54–120 day range in the 15-year full-period screening. In 2-year
    walk-forward windows, half-lives will be shorter.
    """
    eg_score = -np.log10(max(eg_pvalue, 1e-10))

    # Johansen trace margin above 95% critical value (normalised)
    if (np.isfinite(johansen_trace_stat) and np.isfinite(johansen_trace_cv95)
            and johansen_trace_cv95 > 0):
        johansen_margin = max(0.0,
                              (johansen_trace_stat - johansen_trace_cv95) / johansen_trace_cv95)
    else:
        johansen_margin = 0.0

    hl_quality   = np.exp(-0.5 * ((half_life - optimal_hl) / sigma_hl) ** 2)
    sector_bonus = 1.2 if same_sector else 1.0

    return float((eg_score + johansen_margin) * hl_quality * sector_bonus)


# ── Pair-level pipeline ───────────────────────────────────────────────────────

def _test_one_pair(
    sym_y: str,
    sym_x: str,
    prices: pd.DataFrame,
    eg_threshold: float,
    min_history: int,
    same_sector_only: bool,
) -> PairResult:
    """Run the full test for a single pair. Called inside Parallel."""
    sector_y = get_sector(sym_y)
    sector_x = get_sector(sym_x)
    is_same  = (sector_y == sector_x and sector_y != "Unknown")

    res = PairResult(
        symbol_y=sym_y,
        symbol_x=sym_x,
        sector_y=sector_y,
        sector_x=sector_x,
        same_sector=is_same,
    )

    # Skip cross-sector pairs when same_sector_only is requested
    if same_sector_only and not is_same:
        return res

    # Align on shared non-NaN observations
    pair = prices[[sym_y, sym_x]].dropna()
    if len(pair) < min_history:
        return res

    log_y = np.log(pair[sym_y].values)
    log_x = np.log(pair[sym_x].values)
    res.n_obs = len(pair)

    # Stage 1: Engle-Granger
    try:
        p_val, t_stat, ols_beta = engle_granger_test(log_y, log_x)
        res.eg_pvalue    = p_val
        res.eg_statistic = t_stat
        res.ols_beta     = ols_beta
        res.eg_passed    = p_val < eg_threshold
    except Exception as exc:
        logger.debug(f"EG failed {sym_y}/{sym_x}: {exc}")
        return res

    if not res.eg_passed:
        return res

    spread_ols    = log_y - ols_beta * log_x
    res.spread_std = float(spread_ols.std())
    res.half_life  = compute_half_life(spread_ols)

    # Stage 3: Johansen
    try:
        passed, jbeta, tr_stat, tr_cv = johansen_test(log_y, log_x)
        res.johansen_passed     = passed
        res.johansen_beta       = jbeta
        res.johansen_trace_stat = tr_stat
        res.johansen_trace_cv95 = tr_cv
    except Exception as exc:
        logger.debug(f"Johansen failed {sym_y}/{sym_x}: {exc}")

    return res


# ── Public API ────────────────────────────────────────────────────────────────

def screen_pairs(
    universe_close: pd.DataFrame,
    eg_pvalue_threshold: float = 0.05,
    min_half_life: float = 3.0,
    max_half_life: float = 120.0,
    min_history: int = 504,
    n_jobs: int = 1,
    same_sector_only: bool = True,
    apply_fdr: bool = False,
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    """Screen all candidate pairs for cointegration.

    Parameters
    ----------
    universe_close      : DataFrame of adjusted close prices (columns = symbols).
    eg_pvalue_threshold : Raw EG p-value cutoff for initial screen (default 0.05).
    min_half_life       : Min mean-reversion half-life in days (default 3).
    max_half_life       : Max mean-reversion half-life in days (default 120).
                          Empirical finding: full-period screening on 15y daily
                          data gives minimum ~54 days. Use 45d only in short windows.
    min_history         : Minimum shared observations required (default 504 = 2yr).
    n_jobs              : Parallel workers (-1 = all cores).
    same_sector_only    : If True, only test same-sector pairs (170 vs 1,378).
                          Reduces multiple-testing burden and improves economic rationale.
    apply_fdr           : If True, apply Benjamini-Hochberg FDR correction and add
                          eg_pvalue_fdr / eg_passed_fdr columns.
    fdr_alpha           : FDR level. Default 0.05.

    Returns
    -------
    DataFrame with one row per pair tested, sorted by score descending.
    Key columns: symbol_y, symbol_x, sector_y, sector_x, same_sector,
                 eg_pvalue, eg_pvalue_fdr, eg_passed, eg_passed_fdr,
                 johansen_passed, half_life, hedge_ratio, score,
                 passed_all (raw), passed_all_fdr (FDR+sector).
    """
    symbols   = list(universe_close.columns)
    all_pairs = list(combinations(symbols, 2))
    n_pairs   = len(all_pairs)

    logger.info(
        f"Screening {'same-sector' if same_sector_only else 'all'} pairs "
        f"({len(symbols)} symbols) | "
        f"EG={eg_pvalue_threshold}  FDR={apply_fdr}  "
        f"HL=[{min_half_life},{max_half_life}]d  n_jobs={n_jobs}"
    )

    raw_results: list[PairResult] = Parallel(n_jobs=n_jobs)(
        delayed(_test_one_pair)(
            sym_y, sym_x, universe_close,
            eg_pvalue_threshold, min_history, same_sector_only,
        )
        for sym_y, sym_x in all_pairs
    )

    # Convert to DataFrame
    rows = [
        {
            "symbol_y":            r.symbol_y,
            "symbol_x":            r.symbol_x,
            "sector_y":            r.sector_y,
            "sector_x":            r.sector_x,
            "same_sector":         r.same_sector,
            "eg_pvalue":           r.eg_pvalue,
            "eg_statistic":        r.eg_statistic,
            "eg_passed":           r.eg_passed,
            "eg_pvalue_fdr":       r.eg_pvalue_fdr,
            "eg_passed_fdr":       r.eg_passed_fdr,
            "johansen_passed":     r.johansen_passed,
            "johansen_trace_stat": r.johansen_trace_stat,
            "johansen_trace_cv95": r.johansen_trace_cv95,
            "ols_beta":            r.ols_beta,
            "johansen_beta":       r.johansen_beta,
            "hedge_ratio":         r.hedge_ratio,
            "half_life":           r.half_life,
            "spread_std":          r.spread_std,
            "n_obs":               r.n_obs,
            "score":               0.0,
            "passed_all":          False,
            "passed_all_fdr":      False,
        }
        for r in raw_results
    ]

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No pairs returned — check data or thresholds")
        return df

    # Stage 2: FDR correction on ALL EG p-values
    if apply_fdr:
        df = apply_fdr_correction(df, alpha=fdr_alpha)
    else:
        df["eg_pvalue_fdr"] = df["eg_pvalue"]
        df["eg_passed_fdr"] = df["eg_passed"]

    # Half-life filter flag
    hl = df["half_life"]
    df["half_life_ok"] = (hl >= min_half_life) & (hl <= max_half_life) & hl.notna()

    # passed_all: raw EG + Johansen + half-life
    df["passed_all"] = (
        df["eg_passed"] & df["johansen_passed"] & df["half_life_ok"]
    )

    # passed_all_fdr: FDR-adjusted EG + Johansen + half-life
    df["passed_all_fdr"] = (
        df["eg_passed_fdr"] & df["johansen_passed"] & df["half_life_ok"]
    )

    # Recompute scores with updated formula
    df["score"] = df.apply(
        lambda r: _score(
            eg_pvalue=r["eg_pvalue"],
            half_life=r["half_life"],
            same_sector=r["same_sector"],
            johansen_trace_stat=r["johansen_trace_stat"],
            johansen_trace_cv95=r["johansen_trace_cv95"],
        ) if (r["johansen_passed"] and np.isfinite(r["half_life"]) and r["half_life"] > 0)
        else 0.0,
        axis=1,
    )

    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    # Logging
    n_eg_raw  = int(df["eg_passed"].sum())
    n_eg_fdr  = int(df["eg_passed_fdr"].sum())
    n_joh     = int((df["eg_passed"] & df["johansen_passed"]).sum())
    n_raw     = int(df["passed_all"].sum())
    n_fdr     = int(df["passed_all_fdr"].sum())
    n_tested  = sum(1 for r in raw_results if r.n_obs > 0)

    logger.info(
        f"EG raw={n_eg_raw}  FDR={n_eg_fdr}  "
        f"Johansen={n_joh}  "
        f"Passed(raw)={n_raw}  Passed(FDR)={n_fdr}"
    )

    return df


def select_pairs(
    universe_close: pd.DataFrame,
    max_pairs: int = 25,
    eg_pvalue_threshold: float = 0.05,
    min_half_life: float = 3.0,
    max_half_life: float = 120.0,
    min_history: int = 504,
    n_jobs: int = 1,
    same_sector_only: bool = True,
    apply_fdr: bool = False,
    fdr_alpha: float = 0.05,
) -> list[PairResult]:
    """Screen all pairs and return the top `max_pairs` that pass all FDR filters.

    Returns list of PairResult sorted by score descending.
    Uses passed_all_fdr (FDR + half-life) as the selection criterion.
    """
    df = screen_pairs(
        universe_close,
        eg_pvalue_threshold=eg_pvalue_threshold,
        min_half_life=min_half_life,
        max_half_life=max_half_life,
        min_history=min_history,
        n_jobs=n_jobs,
        same_sector_only=same_sector_only,
        apply_fdr=apply_fdr,
        fdr_alpha=fdr_alpha,
    )

    if df.empty:
        return []

    top = df[df["passed_all_fdr"]].head(max_pairs)

    results = []
    for _, row in top.iterrows():
        r = PairResult(
            symbol_y=row["symbol_y"],
            symbol_x=row["symbol_x"],
            sector_y=row["sector_y"],
            sector_x=row["sector_x"],
            same_sector=bool(row["same_sector"]),
            eg_pvalue=row["eg_pvalue"],
            eg_statistic=row["eg_statistic"],
            eg_passed=bool(row["eg_passed"]),
            eg_pvalue_fdr=row["eg_pvalue_fdr"],
            eg_passed_fdr=bool(row["eg_passed_fdr"]),
            johansen_passed=bool(row["johansen_passed"]),
            johansen_trace_stat=row["johansen_trace_stat"],
            johansen_trace_cv95=row["johansen_trace_cv95"],
            ols_beta=row["ols_beta"],
            johansen_beta=row["johansen_beta"],
            half_life=row["half_life"],
            spread_std=row["spread_std"],
            n_obs=int(row["n_obs"]),
            score=row["score"],
        )
        results.append(r)

    return results
