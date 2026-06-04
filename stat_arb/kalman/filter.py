"""Kalman Filter for dynamic hedge ratio estimation in equity pairs trading.

Models the relationship:
    log(P_y[t]) = alpha[t] + beta[t] * log(P_x[t]) + eps[t]

where alpha[t] and beta[t] evolve as independent random walks:
    alpha[t] = alpha[t-1] + w_a[t],  w_a ~ N(0, delta)
    beta[t]  = beta[t-1]  + w_b[t],  w_b ~ N(0, delta)

The spread (innovation) at each step is computed BEFORE updating the state,
so there is zero look-ahead bias when used for live signal generation.

Fixes applied (v2):
  1. warmup_bars  : First N bars of run() are NaN — filter state is unreliable
                    until the covariance P has converged. Default warmup=60 in
                    run_kalman() prevents using these bars in signal generation.
  2. tune_delta() : MLE-based per-pair delta tuning via innovation log-likelihood
                    grid search. One-size-fits-all delta=1e-5 is replaced by the
                    value that best explains this specific pair's price dynamics.
  3. run_kalman() : warmup_bars=60 and auto_tune_delta=False are now exposed so
                    callers can opt into safer, better-calibrated output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class KalmanState:
    """Serialisable snapshot of filter state for save/resume."""
    theta: np.ndarray  # shape (2,): [alpha, beta]
    P: np.ndarray      # shape (2, 2): state covariance


class KalmanFilterHedge:
    """Online Kalman Filter for dynamic hedge ratio between two assets.

    Parameters
    ----------
    delta : float
        Process-noise variance added per step (Q = delta * I_2).
        Smaller → β adapts slowly (smooth, may lag structural breaks).
        Larger  → β adapts fast (responsive, noisier spread).
        Default 1e-5; use tune_delta() for per-pair MLE calibration.
    vt : float
        Observation-noise variance R.
        Default 1e-3.
    """

    def __init__(self, delta: float = 1e-5, vt: float = 1e-3) -> None:
        self.delta = delta
        self.vt = float(vt)
        self.Q = delta * np.eye(2)
        self.reset()

    # ── state management ─────────────────────────────────────────────────────

    def reset(self) -> None:
        """Re-initialise to an uninformed prior."""
        self._theta = np.zeros(2)
        self._P = np.eye(2)

    def snapshot(self) -> KalmanState:
        """Return a copy of the current state (for mid-series checkpointing)."""
        return KalmanState(theta=self._theta.copy(), P=self._P.copy())

    def restore(self, state: KalmanState) -> None:
        """Resume from a previously saved snapshot."""
        self._theta = state.theta.copy()
        self._P = state.P.copy()

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def alpha(self) -> float:
        return float(self._theta[0])

    @property
    def beta(self) -> float:
        return float(self._theta[1])

    # ── core filter step ─────────────────────────────────────────────────────

    def step(self, log_y: float, log_x: float) -> tuple[float, float]:
        """Process one observation and return the spread before state update.

        Parameters
        ----------
        log_y : log price of dependent asset.
        log_x : log price of independent asset.

        Returns
        -------
        spread     : Innovation = log_y − (alpha_pred + beta_pred * log_x).
                     Computed BEFORE state update → no look-ahead.
        spread_std : sqrt(S_t), innovation standard deviation.
                     Z-score = spread / spread_std.
        """
        H = np.array([1.0, log_x])
        P_pred = self._P + self.Q
        e = log_y - float(H @ self._theta)
        S = float(H @ P_pred @ H) + self.vt
        K = P_pred @ H / S
        I_KH = np.eye(2) - np.outer(K, H)
        self._theta = self._theta + K * e
        self._P = I_KH @ P_pred @ I_KH.T + np.outer(K, K) * self.vt
        return e, float(np.sqrt(max(S, 1e-12)))

    # ── batch run ────────────────────────────────────────────────────────────

    def run(
        self,
        log_y: np.ndarray,
        log_x: np.ndarray,
        warmup_bars: int = 0,
    ) -> pd.DataFrame:
        """Run the filter over a full log-price series.

        Parameters
        ----------
        log_y, log_x : 1-D arrays of log prices (equal length).
        warmup_bars  : Number of leading bars to mask as NaN.
                       The Kalman covariance P takes ~30-100 bars to converge
                       from the initial identity prior. Setting warmup_bars=60
                       prevents unreliable early estimates from reaching
                       downstream signal logic. Default 0 (backward-compat).

        Returns
        -------
        DataFrame with columns: alpha, beta, spread, spread_std, zscore.
        First warmup_bars rows are NaN when warmup_bars > 0.
        """
        log_y = np.asarray(log_y, dtype=float)
        log_x = np.asarray(log_x, dtype=float)

        if log_y.shape != log_x.shape or log_y.ndim != 1:
            raise ValueError(
                f"log_y and log_x must be 1-D arrays of equal length, "
                f"got {log_y.shape} vs {log_x.shape}"
            )

        n = len(log_y)
        alphas      = np.empty(n)
        betas       = np.empty(n)
        spreads     = np.empty(n)
        spread_stds = np.empty(n)

        self.reset()

        for t in range(n):
            e, std = self.step(log_y[t], log_x[t])
            alphas[t]      = self._theta[0]
            betas[t]       = self._theta[1]
            spreads[t]     = e
            spread_stds[t] = std

        # ── mask warmup period ────────────────────────────────────────────────
        if warmup_bars > 0:
            w = min(warmup_bars, n)
            alphas[:w]      = np.nan
            betas[:w]       = np.nan
            spreads[:w]     = np.nan
            spread_stds[:w] = np.nan

        return pd.DataFrame(
            {
                "alpha":      alphas,
                "beta":       betas,
                "spread":     spreads,
                "spread_std": spread_stds,
                "zscore":     spreads / np.where(spread_stds > 0, spread_stds, np.nan),
            }
        )

    # ── delta tuning (MLE via innovation log-likelihood) ─────────────────────

    def tune_delta(
        self,
        log_y: np.ndarray,
        log_x: np.ndarray,
        delta_grid: list[float] | None = None,
    ) -> float:
        """Find the delta that maximises innovation log-likelihood (MLE).

        Runs a fresh filter for each candidate delta and returns the value
        that best explains the observed price sequence in terms of Gaussian
        innovation log-likelihood:

            LL = Σ_t  -½ [log(2π S_t) + e_t² / S_t]

        This is the standard MLE objective for Kalman filter hyperparameters.

        Does NOT modify self.delta. Construct a new KalmanFilterHedge with
        the returned delta before calling run().

        Parameters
        ----------
        log_y, log_x : Log price arrays (equal length).
        delta_grid   : Candidate delta values to search.
                       Default: 12 log-spaced values in [1e-7, 1e-2].

        Returns
        -------
        best_delta : float — the delta maximising log-likelihood.

        Example
        -------
        kf = KalmanFilterHedge()
        best = kf.tune_delta(log_y, log_x)
        result = KalmanFilterHedge(delta=best).run(log_y, log_x, warmup_bars=60)
        """
        if delta_grid is None:
            delta_grid = np.logspace(-7, -2, 12).tolist()

        log_y = np.asarray(log_y, dtype=float)
        log_x = np.asarray(log_x, dtype=float)

        best_delta: float = self.delta
        best_ll: float = -np.inf

        for delta in delta_grid:
            ll = _innovation_log_likelihood(log_y, log_x, delta, self.vt)
            if ll > best_ll:
                best_ll = ll
                best_delta = float(delta)

        logger.debug(f"tune_delta: best={best_delta:.2e}  ll={best_ll:.1f}")
        return best_delta


# ── module-level helpers ──────────────────────────────────────────────────────

def _innovation_log_likelihood(
    log_y: np.ndarray,
    log_x: np.ndarray,
    delta: float,
    vt: float,
) -> float:
    """Total Gaussian innovation log-likelihood for (delta, vt).

    Used internally by KalmanFilterHedge.tune_delta().
    """
    kf = KalmanFilterHedge(delta=delta, vt=vt)
    ll = 0.0
    log2pi = np.log(2 * np.pi)
    for t in range(len(log_y)):
        e, std = kf.step(log_y[t], log_x[t])
        S = std * std
        ll -= 0.5 * (log2pi + np.log(max(S, 1e-30)) + e * e / max(S, 1e-30))
    return ll


# ── convenience wrapper ───────────────────────────────────────────────────────

def run_kalman(
    prices_y: pd.Series,
    prices_x: pd.Series,
    delta: float = 1e-5,
    vt: float = 1e-3,
    warmup_bars: int = 60,
    auto_tune_delta: bool = False,
) -> pd.DataFrame:
    """Convenience function: raw price Series → spread DataFrame.

    Aligns the two series on their shared index, converts to log prices,
    and runs KalmanFilterHedge. The result carries the aligned index.

    Parameters
    ----------
    prices_y        : Close price series for the dependent asset.
    prices_x        : Close price series for the independent asset.
    delta           : Process noise. Use tune_delta() for per-pair MLE.
    vt              : Observation noise variance.
    warmup_bars     : Leading bars set to NaN (filter warm-up period).
                      Default 60 — safe conservative choice. Set 0 to disable.
    auto_tune_delta : If True, run MLE grid search to find the best delta
                      for this specific pair before computing the spread.
                      Adds ~0.5-1s per pair. Default False.

    Returns
    -------
    DataFrame with columns: alpha, beta, spread, spread_std, zscore.
    Index matches the inner-join of the two input series.
    First warmup_bars rows are NaN.
    """
    if prices_y.isnull().any() or prices_x.isnull().any():
        raise ValueError("Input series must not contain NaN values.")
    if (prices_y <= 0).any() or (prices_x <= 0).any():
        raise ValueError("Prices must be strictly positive for log transform.")

    y_aligned, x_aligned = prices_y.align(prices_x, join="inner")

    log_y = np.log(y_aligned.to_numpy(dtype=float))
    log_x = np.log(x_aligned.to_numpy(dtype=float))

    kf = KalmanFilterHedge(delta=delta, vt=vt)

    if auto_tune_delta:
        best_delta = kf.tune_delta(log_y, log_x)
        if best_delta != delta:
            logger.info(
                f"run_kalman: auto-tuned delta {delta:.2e} → {best_delta:.2e}"
            )
        kf = KalmanFilterHedge(delta=best_delta, vt=vt)

    result = kf.run(log_y, log_x, warmup_bars=warmup_bars)
    result.index = y_aligned.index
    return result
