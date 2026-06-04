"""Rauch-Tung-Striebel (RTS) backward smoother for offline hedge ratio analysis.

The RTS smoother runs a backward pass over the Kalman forward-filter outputs
to produce minimum-variance estimates that use ALL observations in the series,
not just past observations.

WARNING: Smoothed estimates use future information and must NEVER be used for
live signal generation or backtesting entry/exit decisions. Use only for:
  - Post-hoc analysis and diagnostics
  - Parameter tuning (choosing delta / vt)
  - Research and visualisation
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stat_arb.kalman.filter import KalmanFilterHedge


def run_with_smoother(
    log_y: np.ndarray,
    log_x: np.ndarray,
    delta: float = 1e-5,
    vt: float = 1e-3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Kalman forward filter then RTS backward smoother.

    Parameters
    ----------
    log_y, log_x : 1-D arrays of log prices.
    delta, vt    : Kalman filter noise parameters.

    Returns
    -------
    filtered : DataFrame — causal estimates (no look-ahead). Use for trading.
    smoothed : DataFrame — non-causal estimates (uses full series). Use for analysis only.

    Both DataFrames have columns: alpha, beta, spread, spread_std, zscore.
    """
    log_y = np.asarray(log_y, dtype=float)
    log_x = np.asarray(log_x, dtype=float)
    n = len(log_y)

    kf = KalmanFilterHedge(delta=delta, vt=vt)
    Q = kf.Q

    # ── Forward pass: store all intermediate states ───────────────────────────
    thetas   = np.empty((n, 2))       # filtered state means
    Ps       = np.empty((n, 2, 2))    # filtered state covariances
    P_preds  = np.empty((n, 2, 2))    # predicted covariances (before update)
    spreads  = np.empty(n)
    spread_stds = np.empty(n)

    kf.reset()
    for t in range(n):
        # Replicate step() internals to capture P_pred before update
        H = np.array([1.0, log_x[t]])
        P_pred = kf._P + Q
        e = log_y[t] - float(H @ kf._theta)
        S = float(H @ P_pred @ H) + kf.vt
        K = P_pred @ H / S
        I_KH = np.eye(2) - np.outer(K, H)
        kf._theta = kf._theta + K * e
        kf._P = I_KH @ P_pred @ I_KH.T + np.outer(K, K) * kf.vt

        thetas[t]   = kf._theta
        Ps[t]       = kf._P
        P_preds[t]  = P_pred
        spreads[t]  = e
        spread_stds[t] = float(np.sqrt(max(S, 1e-12)))

    filtered = pd.DataFrame({
        "alpha":      thetas[:, 0],
        "beta":       thetas[:, 1],
        "spread":     spreads,
        "spread_std": spread_stds,
        "zscore":     spreads / spread_stds,
    })

    # ── Backward (RTS) pass ───────────────────────────────────────────────────
    thetas_s = thetas.copy()
    Ps_s     = Ps.copy()

    for t in range(n - 2, -1, -1):
        P_pred_next = P_preds[t + 1]                            # P_{t+1|t}
        G = Ps[t] @ np.linalg.solve(P_pred_next.T, np.eye(2)).T  # smoother gain
        thetas_s[t] = thetas[t] + G @ (thetas_s[t + 1] - thetas[t])
        Ps_s[t] = Ps[t] + G @ (Ps_s[t + 1] - P_pred_next) @ G.T

    # Smoothed spread: recompute as log_y - smoothed prediction (for reference only)
    H_all = np.column_stack([np.ones(n), log_x])
    spreads_s = log_y - (H_all * thetas_s).sum(axis=1)
    spread_stds_s = np.array([
        float(np.sqrt(max(float(H_all[t] @ Ps_s[t] @ H_all[t]) + vt, 1e-12)))
        for t in range(n)
    ])

    smoothed = pd.DataFrame({
        "alpha":      thetas_s[:, 0],
        "beta":       thetas_s[:, 1],
        "spread":     spreads_s,
        "spread_std": spread_stds_s,
        "zscore":     spreads_s / spread_stds_s,
    })

    return filtered, smoothed
