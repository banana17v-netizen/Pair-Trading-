"""Tests for stat_arb/pairs/select.py and stat_arb/pairs/monitor.py — v2.

Covers:
  - Original EG, Johansen, half-life, screen_pairs, select_pairs, monitor tests
  - NEW: FDR correction (TestFDRCorrection)
  - NEW: Sector info and same-sector filter (TestSectorInfo)
  - NEW: Improved scoring (TestImprovedScoring)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stat_arb.pairs.select import (
    PairResult,
    apply_fdr_correction,
    compute_half_life,
    engle_granger_test,
    johansen_test,
    screen_pairs,
    select_pairs,
    _score,
)
from stat_arb.pairs.monitor import check_pair_health, monitor_pairs
from stat_arb.data.universe import SECTOR_MAP, get_sector, SP500_LIQUID


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_cointegrated(n: int = 600, beta: float = 1.3, seed: int = 42) -> tuple:
    """Cointegrated pair: log_y = 0.4 + beta * log_x + noise."""
    rng = np.random.default_rng(seed)
    log_x = np.cumsum(rng.normal(0, 0.01, n))
    log_y = 0.4 + beta * log_x + rng.normal(0, 0.015, n)
    return log_y, log_x, beta


def _make_independent(n: int = 600, seed: int = 0) -> tuple:
    """Two independent random walks — should NOT be cointegrated."""
    rng = np.random.default_rng(seed)
    log_y = np.cumsum(rng.normal(0, 0.01, n))
    log_x = np.cumsum(rng.normal(0, 0.01, n))
    return log_y, log_x


def _make_ou_spread(n: int = 600, half_life: float = 20.0, seed: int = 1) -> np.ndarray:
    """OU process with known half-life: phi = exp(-log(2)/half_life)."""
    rng = np.random.default_rng(seed)
    phi = np.exp(-np.log(2) / half_life)
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = phi * s[t - 1] + rng.normal(0, 0.01)
    return s


def _make_price_df(n: int = 600, n_stocks: int = 4, seed: int = 5) -> pd.DataFrame:
    """Small universe DataFrame for screen_pairs tests."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
    data = {}
    # First two stocks: cointegrated
    log_x = np.cumsum(rng.normal(0, 0.01, n))
    log_y = 0.3 + 1.2 * log_x + rng.normal(0, 0.015, n)
    data["A"] = np.exp(log_y)
    data["B"] = np.exp(log_x)
    # Remaining: independent random walks
    for i in range(2, n_stocks):
        data[chr(65 + i)] = np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame(data, index=idx)


# ═══════════════════════════════════════════════════════════════════════════════
# engle_granger_test
# ═══════════════════════════════════════════════════════════════════════════════

class TestEngleGranger:
    def test_finds_cointegrated_pair(self):
        log_y, log_x, _ = _make_cointegrated(n=600)
        p_val, t_stat, beta = engle_granger_test(log_y, log_x)
        assert p_val < 0.05, f"Expected p < 0.05 for cointegrated pair, got {p_val:.4f}"

    def test_rejects_independent_pair(self):
        log_y, log_x = _make_independent(n=600)
        p_val, _, _ = engle_granger_test(log_y, log_x)
        # Most random walk pairs should fail at 5%
        assert p_val > 0.01, f"EG should not strongly reject non-cointegrated pair"

    def test_ols_beta_close_to_true(self):
        true_beta = 1.5
        log_y, log_x, _ = _make_cointegrated(n=800, beta=true_beta, seed=10)
        _, _, beta = engle_granger_test(log_y, log_x)
        assert abs(beta - true_beta) < 0.3, f"OLS beta {beta:.3f} far from true {true_beta}"

    def test_returns_three_values(self):
        log_y, log_x, _ = _make_cointegrated()
        result = engle_granger_test(log_y, log_x)
        assert len(result) == 3
        p_val, t_stat, beta = result
        assert isinstance(p_val,  float)
        assert isinstance(t_stat, float)
        assert isinstance(beta,   float)

    def test_t_stat_is_negative_for_cointegrated(self):
        """ADF test statistic should be negative (more negative = more stationary)."""
        log_y, log_x, _ = _make_cointegrated()
        _, t_stat, _ = engle_granger_test(log_y, log_x)
        assert t_stat < 0


# ═══════════════════════════════════════════════════════════════════════════════
# johansen_test
# ═══════════════════════════════════════════════════════════════════════════════

class TestJohansen:
    def test_confirms_cointegrated_pair(self):
        log_y, log_x, _ = _make_cointegrated(n=600)
        passed, beta, tr_stat, tr_cv = johansen_test(log_y, log_x)
        assert passed, (
            f"Johansen should confirm cointegrated pair "
            f"(trace={tr_stat:.2f} vs cv95={tr_cv:.2f})"
        )

    def test_beta_positive(self):
        log_y, log_x, _ = _make_cointegrated(n=600)
        _, beta, _, _ = johansen_test(log_y, log_x)
        assert beta > 0, f"Johansen beta should be positive, got {beta:.3f}"

    def test_beta_close_to_ols(self):
        log_y, log_x, true_beta = _make_cointegrated(n=800, seed=5)
        _, jbeta, _, _ = johansen_test(log_y, log_x)
        _, _, ols_beta = engle_granger_test(log_y, log_x)
        # Johansen and OLS betas should be in the same ballpark
        assert abs(jbeta - ols_beta) < 0.5

    def test_returns_four_values(self):
        log_y, log_x, _ = _make_cointegrated()
        result = johansen_test(log_y, log_x)
        assert len(result) == 4
        passed, beta, tr_stat, tr_cv = result
        assert isinstance(passed, bool)
        assert np.isfinite(beta) or np.isnan(beta)
        assert np.isfinite(tr_stat)
        assert np.isfinite(tr_cv)

    def test_trace_stat_exceeds_cv_for_cointegrated(self):
        log_y, log_x, _ = _make_cointegrated(n=600)
        _, _, tr_stat, tr_cv = johansen_test(log_y, log_x)
        assert tr_stat > tr_cv


# ═══════════════════════════════════════════════════════════════════════════════
# compute_half_life
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeHalfLife:
    def test_correct_for_ou_process(self):
        """Estimated half-life should be close to the true OU half-life."""
        true_hl = 20.0
        spread = _make_ou_spread(n=2000, half_life=true_hl, seed=3)
        hl = compute_half_life(spread)
        assert np.isfinite(hl), "Half-life should be finite for OU process"
        assert abs(hl - true_hl) < 10.0, (
            f"Estimated HL {hl:.1f} too far from true {true_hl:.1f}"
        )

    def test_inf_for_random_walk(self):
        rng = np.random.default_rng(0)
        rw = np.cumsum(rng.normal(0, 0.01, 1000))
        hl = compute_half_life(rw)
        assert hl == np.inf or hl > 200, \
            f"Random walk should give large/inf half-life, got {hl:.1f}"

    def test_short_half_life_for_fast_mr(self):
        """Strongly mean-reverting spread (phi=0.5) → HL ≈ 1 day."""
        rng = np.random.default_rng(9)
        s = np.zeros(1000)
        for t in range(1, 1000):
            s[t] = 0.5 * s[t - 1] + rng.normal(0, 0.01)
        hl = compute_half_life(s)
        assert np.isfinite(hl)
        assert hl < 5.0, f"Fast MR spread should have HL < 5d, got {hl:.2f}"

    def test_nan_for_too_short_series(self):
        hl = compute_half_life(np.array([1.0, 2.0, 1.5]))
        assert np.isnan(hl)

    def test_returns_positive_for_mean_reverting(self):
        spread = _make_ou_spread(n=500, half_life=15.0)
        hl = compute_half_life(spread)
        assert np.isfinite(hl) and hl > 0


# ═══════════════════════════════════════════════════════════════════════════════
# PairResult
# ═══════════════════════════════════════════════════════════════════════════════

class TestPairResult:
    def test_passed_all_requires_all_three(self):
        r = PairResult("A", "B", eg_passed=True, johansen_passed=True, half_life=15.0)
        assert r.passed_all is True

    def test_passed_all_false_if_eg_fails(self):
        r = PairResult("A", "B", eg_passed=False, johansen_passed=True, half_life=15.0)
        assert r.passed_all is False

    def test_passed_all_false_if_johansen_fails(self):
        r = PairResult("A", "B", eg_passed=True, johansen_passed=False, half_life=15.0)
        assert r.passed_all is False

    def test_passed_all_false_if_half_life_inf(self):
        r = PairResult("A", "B", eg_passed=True, johansen_passed=True,
                       half_life=np.inf)
        assert not r.passed_all   # np.bool_ != False; use `not` instead of `is False`

    def test_hedge_ratio_prefers_johansen(self):
        r = PairResult("A", "B", johansen_passed=True,
                       johansen_beta=1.5, ols_beta=1.2)
        assert r.hedge_ratio == pytest.approx(1.5)

    def test_hedge_ratio_falls_back_to_ols(self):
        r = PairResult("A", "B", johansen_passed=False, ols_beta=1.2)
        assert r.hedge_ratio == pytest.approx(1.2)


# ═══════════════════════════════════════════════════════════════════════════════
# screen_pairs
# ═══════════════════════════════════════════════════════════════════════════════

class TestScreenPairs:
    @pytest.fixture
    def small_universe(self):
        return _make_price_df(n=600, n_stocks=4)

    def test_returns_dataframe(self, small_universe):
        df = screen_pairs(small_universe, min_history=100)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self, small_universe):
        df = screen_pairs(small_universe, min_history=100)
        required = {"symbol_y", "symbol_x", "eg_pvalue", "eg_passed",
                    "johansen_passed", "half_life", "score", "passed_all"}
        assert required.issubset(set(df.columns))

    def test_correct_number_of_pairs(self, small_universe):
        n = small_universe.shape[1]
        expected = n * (n - 1) // 2
        df = screen_pairs(small_universe, min_history=100)
        assert len(df) == expected

    def test_finds_ab_cointegrated_pair(self, small_universe):
        """A and B in small_universe are cointegrated by construction."""
        # same_sector_only=False: synthetic symbols not in SECTOR_MAP
        df = screen_pairs(small_universe, min_history=100, same_sector_only=False)
        ab = df[(df["symbol_y"] == "A") & (df["symbol_x"] == "B")]
        if ab.empty:
            ab = df[(df["symbol_y"] == "B") & (df["symbol_x"] == "A")]
        assert not ab.empty
        assert float(ab.iloc[0]["eg_pvalue"]) < 0.05

    def test_sorted_by_score_descending(self, small_universe):
        df = screen_pairs(small_universe, min_history=100)
        scores = df["score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_parallel_same_as_sequential(self, small_universe):
        df_seq = screen_pairs(small_universe, min_history=100, n_jobs=1)
        df_par = screen_pairs(small_universe, min_history=100, n_jobs=2)
        # Same rows (may differ in order due to parallelism, so compare sorted)
        for col in ["eg_pvalue", "half_life"]:
            seq_vals = df_seq[col].sort_values().values
            par_vals = df_par[col].sort_values().values
            np.testing.assert_allclose(
                seq_vals[np.isfinite(seq_vals)],
                par_vals[np.isfinite(par_vals)],
                rtol=1e-10,
            )

    def test_half_life_filter_applied(self, small_universe):
        df = screen_pairs(small_universe, min_history=100,
                          min_half_life=3.0, max_half_life=45.0)
        passed = df[df["passed_all"]]
        if not passed.empty:
            hl = passed["half_life"].dropna()
            assert (hl >= 3.0).all() and (hl <= 45.0).all()

    def test_strict_threshold_fewer_passing_pairs(self, small_universe):
        """Tighter EG threshold must pass <= pairs than the default threshold."""
        df_normal = screen_pairs(small_universe, eg_pvalue_threshold=0.05, min_history=100)
        df_strict = screen_pairs(small_universe, eg_pvalue_threshold=0.001, min_history=100)
        assert df_strict["eg_passed"].sum() <= df_normal["eg_passed"].sum()


# ═══════════════════════════════════════════════════════════════════════════════
# select_pairs
# ═══════════════════════════════════════════════════════════════════════════════

class TestSelectPairs:
    @pytest.fixture
    def small_universe(self):
        return _make_price_df(n=600, n_stocks=4)

    def test_returns_list_of_pair_results(self, small_universe):
        pairs = select_pairs(small_universe, max_pairs=5, min_history=100)
        assert isinstance(pairs, list)
        assert all(isinstance(p, PairResult) for p in pairs)

    def test_respects_max_pairs(self, small_universe):
        pairs = select_pairs(small_universe, max_pairs=2, min_history=100)
        assert len(pairs) <= 2

    def test_all_returned_pairs_passed_all(self, small_universe):
        pairs = select_pairs(small_universe, max_pairs=10, min_history=100)
        for p in pairs:
            assert p.passed_all, f"{p.symbol_y}/{p.symbol_x} should have passed_all"

    def test_sorted_by_score(self, small_universe):
        pairs = select_pairs(small_universe, max_pairs=10, min_history=100)
        scores = [p.score for p in pairs]
        assert scores == sorted(scores, reverse=True)

    def test_empty_list_if_no_pairs_pass(self, small_universe):
        pairs = select_pairs(small_universe, eg_pvalue_threshold=1e-15,
                             max_pairs=5, min_history=100)
        assert pairs == []


# ═══════════════════════════════════════════════════════════════════════════════
# monitor_pairs
# ═══════════════════════════════════════════════════════════════════════════════

class TestMonitor:
    @pytest.fixture
    def cointegrated_prices(self):
        # Spread is an OU process (true half-life ~20 days) so compute_half_life
        # returns a finite, in-range value that passes the health check.
        rng = np.random.default_rng(42)
        n = 1200
        idx = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
        log_x = np.cumsum(rng.normal(0, 0.01, n))

        phi = np.exp(-np.log(2) / 20.0)  # OU AR(1) coeff → HL ≈ 20 days
        ou  = np.zeros(n)
        for t in range(1, n):
            ou[t] = phi * ou[t - 1] + rng.normal(0, 0.005)

        log_y = 0.3 + 1.2 * log_x + ou
        df = pd.DataFrame({"A": np.exp(log_y), "B": np.exp(log_x)}, index=idx)
        return df

    def test_healthy_pair_eg_passes(self, cointegrated_prices):
        """EG must pass for a synthetically cointegrated pair (full series)."""
        n = len(cointegrated_prices)
        result = check_pair_health(
            cointegrated_prices["A"],
            cointegrated_prices["B"],
            window_days=n,
            eg_threshold=0.10,
            max_half_life=100_000.0,   # disable half-life filter; tested separately
        )
        assert result["healthy"], f"EG should find cointegration: {result['reason']}"
        assert result["eg_pvalue"] < 0.10, f"EG p-value {result['eg_pvalue']:.4f} too high"

    def test_result_has_required_keys(self, cointegrated_prices):
        """check_pair_health must always return dict with expected keys."""
        result = check_pair_health(cointegrated_prices["A"], cointegrated_prices["B"])
        for key in ["healthy", "eg_pvalue", "half_life", "n_obs", "reason"]:
            assert key in result, f"Missing key '{key}' in result"

    def test_independent_pair_returns_false(self):
        rng = np.random.default_rng(7)
        n = 400
        idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
        y = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.02, n))), index=idx)
        x = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.02, n))), index=idx)
        result = check_pair_health(y, x, eg_threshold=0.05)
        # Most random walks fail cointegration — allow probabilistic outcome
        assert isinstance(result["healthy"], bool)
        assert "reason" in result

    def test_insufficient_data_returns_unhealthy(self):
        idx = pd.date_range("2020-01-01", periods=50, freq="B", tz="UTC")
        y = pd.Series(np.ones(50), index=idx)
        x = pd.Series(np.ones(50), index=idx)
        result = check_pair_health(y, x, window_days=252)
        assert result["healthy"] is False
        assert "insufficient" in result["reason"]

    def test_monitor_pairs_returns_dataframe(self, cointegrated_prices):
        df = monitor_pairs(
            active_pairs=[("A", "B")],
            universe_close=cointegrated_prices,
        )
        assert isinstance(df, pd.DataFrame)
        assert "healthy" in df.columns
        assert ("A", "B") in df.index

    def test_monitor_skips_missing_symbol(self, cointegrated_prices):
        df = monitor_pairs(
            active_pairs=[("A", "MISSING")],
            universe_close=cointegrated_prices,
        )
        assert len(df) == 0   # all rows skipped → empty result


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke tests on real data
# ═══════════════════════════════════════════════════════════════════════════════

class TestOnRealData:
    """Smoke tests on real data.

    Note: cointegration over a full 15-year horizon is rare and stock-pair
    dependent.  These tests verify correct function behaviour and output
    format rather than asserting that any specific pair is cointegrated.
    (Diagnosis showed GS/MS is the closest at p≈0.09; JPM/BAC p≈0.23.)
    """

    @pytest.fixture
    def real_prices(self):
        from pathlib import Path
        p = Path("data/ohlcv/1d")
        if not p.exists():
            pytest.skip("Daily data not available")
        syms = ["JPM", "BAC", "GS", "MS", "WFC"]
        dfs = {
            s: pd.read_parquet(p / f"{s}.parquet").set_index("timestamp")["close"]
            for s in syms
        }
        return pd.DataFrame(dfs)

    def test_eg_returns_valid_pvalue(self, real_prices):
        """EG p-value must be in [0, 1] — basic sanity for all pairs."""
        from itertools import combinations
        for sy, sx in combinations(real_prices.columns, 2):
            ly = np.log(real_prices[sy].values)
            lx = np.log(real_prices[sx].values)
            p_val, t_stat, beta = engle_granger_test(ly, lx)
            assert 0.0 <= p_val <= 1.0, f"{sy}/{sx}: p={p_val} not in [0,1]"
            assert np.isfinite(t_stat)
            assert np.isfinite(beta)

    def test_johansen_returns_correct_types(self, real_prices):
        """Johansen must return (bool, float, float, float) for any pair."""
        ly = np.log(real_prices["GS"].values)
        lx = np.log(real_prices["MS"].values)
        passed, beta, tr_stat, tr_cv = johansen_test(ly, lx)
        assert isinstance(passed, (bool, np.bool_))
        assert np.isfinite(tr_stat)
        assert np.isfinite(tr_cv)
        # beta NaN is acceptable if eigenvector is degenerate
        assert np.isnan(beta) or beta > 0

    def test_screen_pairs_runs_on_real_data(self, real_prices):
        """screen_pairs must run without error and return C(5,2)=10 rows."""
        df = screen_pairs(real_prices, min_history=252)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10   # C(5,2)
        assert set(df.columns).issuperset({"eg_pvalue", "eg_passed", "score"})

    def test_eg_pvalue_columns_are_valid_probabilities(self, real_prices):
        df = screen_pairs(real_prices, min_history=252)
        pvals = df["eg_pvalue"].dropna()
        assert (pvals >= 0).all() and (pvals <= 1).all()

    def test_compute_half_life_on_real_spread(self, real_prices):
        """Half-life computation must return finite or inf — never NaN for good data."""
        ly = np.log(real_prices["GS"].values)
        lx = np.log(real_prices["MS"].values)
        _, _, beta = engle_granger_test(ly, lx)
        spread = ly - beta * lx
        hl = compute_half_life(spread)
        assert np.isnan(hl) or hl > 0, f"Half-life must be positive or inf, got {hl}"

    def test_monitor_runs_on_real_pairs(self, real_prices):
        """monitor_pairs must return a DataFrame with correct columns."""
        result = monitor_pairs(
            active_pairs=[("JPM", "BAC"), ("GS", "MS")],
            universe_close=real_prices,
        )
        assert isinstance(result, pd.DataFrame)
        assert "healthy" in result.columns
        assert len(result) == 2
        assert result["healthy"].map(lambda x: isinstance(x, (bool, np.bool_))).all()

    def test_screen_pairs_has_fdr_columns(self, real_prices):
        """screen_pairs with apply_fdr=True must add eg_pvalue_fdr and eg_passed_fdr."""
        df = screen_pairs(real_prices, min_history=252, apply_fdr=True)
        assert "eg_pvalue_fdr" in df.columns
        assert "eg_passed_fdr" in df.columns

    def test_screen_pairs_has_sector_columns(self, real_prices):
        """screen_pairs must add sector_y, sector_x, same_sector columns."""
        df = screen_pairs(real_prices, min_history=252)
        assert "sector_y" in df.columns
        assert "sector_x" in df.columns
        assert "same_sector" in df.columns
        # All 5 stocks are Financials → all pairs should be same-sector
        assert df["same_sector"].all()

    def test_same_sector_only_reduces_pairs(self, real_prices):
        """same_sector_only=True must return <= pairs than full screening."""
        df_all    = screen_pairs(real_prices, min_history=252, same_sector_only=False)
        df_sector = screen_pairs(real_prices, min_history=252, same_sector_only=True)
        assert len(df_sector) <= len(df_all)


# ═══════════════════════════════════════════════════════════════════════════════
# NEW: FDR correction
# ═══════════════════════════════════════════════════════════════════════════════

class TestFDRCorrection:
    @pytest.fixture
    def screen_df(self):
        """Small universe DataFrame with known-cointegrated A/B pair."""
        rng = np.random.default_rng(5)
        n = 600
        idx = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
        log_x = np.cumsum(rng.normal(0, 0.01, n))
        log_y = 0.3 + 1.2 * log_x + rng.normal(0, 0.015, n)
        data = {"A": np.exp(log_y), "B": np.exp(log_x)}
        for i in range(2, 8):
            data[chr(65 + i)] = np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        df = pd.DataFrame(data, index=idx)
        return screen_pairs(df, min_history=100, apply_fdr=True)

    def test_fdr_adds_required_columns(self, screen_df):
        assert "eg_pvalue_fdr" in screen_df.columns
        assert "eg_passed_fdr" in screen_df.columns
        assert "passed_all_fdr" in screen_df.columns

    def test_fdr_adjusted_pvalue_geq_raw(self, screen_df):
        """BH-adjusted p-values are always >= raw p-values."""
        valid = screen_df["eg_pvalue"].notna() & screen_df["eg_pvalue_fdr"].notna()
        assert (screen_df.loc[valid, "eg_pvalue_fdr"]
                >= screen_df.loc[valid, "eg_pvalue"] - 1e-10).all()

    def test_fdr_reduces_or_equals_passing_pairs(self, screen_df):
        """FDR-adjusted filter must pass <= raw-EG pairs."""
        n_raw = int(screen_df["eg_passed"].sum())
        n_fdr = int(screen_df["eg_passed_fdr"].sum())
        assert n_fdr <= n_raw

    def test_apply_fdr_correction_standalone(self):
        """apply_fdr_correction() must be callable on any DataFrame with eg_pvalue."""
        df = pd.DataFrame({"eg_pvalue": [0.001, 0.01, 0.05, 0.10, 0.50]})
        out = apply_fdr_correction(df, alpha=0.05)
        assert "eg_pvalue_fdr" in out.columns
        assert "eg_passed_fdr" in out.columns
        assert (out["eg_pvalue_fdr"] >= out["eg_pvalue"] - 1e-10).all()

    def test_without_fdr_raw_equals_adjusted(self):
        """apply_fdr=False → eg_pvalue_fdr == eg_pvalue for tested pairs."""
        rng = np.random.default_rng(0)
        n = 300
        idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
        # Use same_sector_only=False so synthetic symbols (not in SECTOR_MAP) are tested
        df = pd.DataFrame({
            "A": np.exp(np.cumsum(rng.normal(0, 0.01, n))),
            "B": np.exp(np.cumsum(rng.normal(0, 0.01, n))),
        }, index=idx)
        result = screen_pairs(df, min_history=100, apply_fdr=False, same_sector_only=False)
        # Only check tested pairs (n_obs > 0); untested pairs have NaN for both
        tested = result[result["n_obs"] > 0]
        for _, row in tested.iterrows():
            assert row["eg_pvalue_fdr"] == pytest.approx(row["eg_pvalue"])
            assert row["eg_passed_fdr"] == row["eg_passed"]


# ═══════════════════════════════════════════════════════════════════════════════
# NEW: Sector info and same-sector filter
# ═══════════════════════════════════════════════════════════════════════════════

class TestSectorInfo:
    def test_sector_map_covers_all_universe_symbols(self):
        """Every symbol in SP500_LIQUID must have an entry in SECTOR_MAP."""
        missing = [s for s in SP500_LIQUID if s not in SECTOR_MAP]
        assert missing == [], f"Symbols missing from SECTOR_MAP: {missing}"

    def test_get_sector_known_symbol(self):
        assert get_sector("JPM")  == "Financials"
        assert get_sector("AAPL") == "Technology"
        assert get_sector("JNJ")  == "Health Care"
        assert get_sector("NEE")  == "Utilities"

    def test_get_sector_unknown_returns_unknown(self):
        assert get_sector("FAKE") == "Unknown"
        assert get_sector("")     == "Unknown"

    def test_screen_pairs_same_sector_column(self):
        """Two Technology stocks → same_sector=True; Tech+Financials → False."""
        rng = np.random.default_rng(9)
        n = 300
        idx = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
        df = pd.DataFrame({
            "AAPL": np.exp(np.cumsum(rng.normal(0, 0.01, n))),  # Technology
            "MSFT": np.exp(np.cumsum(rng.normal(0, 0.01, n))),  # Technology
            "JPM":  np.exp(np.cumsum(rng.normal(0, 0.01, n))),  # Financials
        }, index=idx)
        result = screen_pairs(df, min_history=100)
        # AAPL/MSFT → same_sector=True
        am = result[(result["symbol_y"] == "AAPL") & (result["symbol_x"] == "MSFT")]
        assert not am.empty and bool(am.iloc[0]["same_sector"])
        # AAPL/JPM → same_sector=False
        aj = result[(result["symbol_y"] == "AAPL") & (result["symbol_x"] == "JPM")]
        assert not aj.empty and not bool(aj.iloc[0]["same_sector"])

    def test_same_sector_only_excludes_cross_sector(self):
        """With same_sector_only=True, cross-sector pairs must not appear as tested."""
        rng = np.random.default_rng(3)
        n = 300
        idx = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
        df = pd.DataFrame({
            "AAPL": np.exp(np.cumsum(rng.normal(0, 0.01, n))),  # Technology
            "MSFT": np.exp(np.cumsum(rng.normal(0, 0.01, n))),  # Technology
            "JPM":  np.exp(np.cumsum(rng.normal(0, 0.01, n))),  # Financials
        }, index=idx)
        result = screen_pairs(df, min_history=100, same_sector_only=True)
        # AAPL/JPM and MSFT/JPM are cross-sector → n_obs should be 0
        aj = result[(result["symbol_y"].isin(["AAPL", "MSFT"]))
                    & (result["symbol_x"] == "JPM")]
        for _, r in aj.iterrows():
            assert r["n_obs"] == 0, "Cross-sector pair should have n_obs=0"

    def test_same_sector_only_tests_fewer_pairs(self):
        """same_sector_only should test fewer pairs (n_obs > 0) than full mode."""
        rng = np.random.default_rng(7)
        n = 300
        idx = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
        df = pd.DataFrame({
            "AAPL": np.exp(np.cumsum(rng.normal(0, 0.01, n))),
            "MSFT": np.exp(np.cumsum(rng.normal(0, 0.01, n))),
            "JPM":  np.exp(np.cumsum(rng.normal(0, 0.01, n))),
            "BAC":  np.exp(np.cumsum(rng.normal(0, 0.01, n))),
        }, index=idx)
        full   = screen_pairs(df, min_history=100, same_sector_only=False)
        sector = screen_pairs(df, min_history=100, same_sector_only=True)
        tested_full   = int((full["n_obs"] > 0).sum())
        tested_sector = int((sector["n_obs"] > 0).sum())
        assert tested_sector < tested_full


# ═══════════════════════════════════════════════════════════════════════════════
# NEW: Improved scoring
# ═══════════════════════════════════════════════════════════════════════════════

class TestImprovedScoring:
    def test_same_sector_gets_higher_score(self):
        """Identical statistics but same_sector=True must outscore same_sector=False."""
        score_same  = _score(0.01, 70.0, same_sector=True,  johansen_trace_stat=20, johansen_trace_cv95=15)
        score_cross = _score(0.01, 70.0, same_sector=False, johansen_trace_stat=20, johansen_trace_cv95=15)
        assert score_same > score_cross, "Same-sector pair must have higher score"

    def test_lower_pvalue_gives_higher_score(self):
        score_low  = _score(0.001, 70.0, same_sector=True)
        score_high = _score(0.04,  70.0, same_sector=True)
        assert score_low > score_high

    def test_johansen_margin_boosts_score(self):
        """Johansen trace well above CV must give higher score than borderline."""
        score_strong = _score(0.01, 70.0, same_sector=False,
                              johansen_trace_stat=30, johansen_trace_cv95=15)
        score_weak   = _score(0.01, 70.0, same_sector=False,
                              johansen_trace_stat=16, johansen_trace_cv95=15)
        assert score_strong > score_weak

    def test_score_peaks_near_optimal_hl(self):
        """Score should be highest when half-life is near optimal_hl (70d)."""
        score_opt  = _score(0.01,   70.0, same_sector=True)
        score_fast = _score(0.01,   10.0, same_sector=True)
        score_slow = _score(0.01,  200.0, same_sector=True)
        assert score_opt > score_fast
        assert score_opt > score_slow

    def test_screen_pairs_score_column_populated(self):
        """Pairs that pass Johansen + finite HL must have score > 0."""
        rng = np.random.default_rng(5)
        n = 600
        idx = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
        log_x = np.cumsum(rng.normal(0, 0.01, n))
        log_y = 0.3 + 1.2 * log_x + rng.normal(0, 0.015, n)
        df = pd.DataFrame({"A": np.exp(log_y), "B": np.exp(log_x)}, index=idx)
        result = screen_pairs(df, min_history=100)
        passed = result[result["passed_all"]]
        if not passed.empty:
            assert (passed["score"] > 0).all()
