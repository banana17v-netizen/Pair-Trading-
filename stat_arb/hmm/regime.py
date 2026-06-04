"""HMM-based market regime detector for equity pairs trading.

Three feature builders are provided, ordered by preference:

  build_features_from_spread(kalman_df)   ← PRIMARY for pairs trading
      Uses spread directly from Kalman output. All features describe the pair,
      not the individual assets. Most economically meaningful.
      Columns: [spread_ret, realized_vol, spread_autocorr, zscore]

  build_market_features(universe_close)   ← SHARED regime (performance)
      One HMM trained on the whole universe → same label for all pairs.
      Faster, more stable. Use when running 1000+ pairs.
      Columns: [market_ret, cross_vol, avg_realized_vol, avg_autocorr]

  build_features(prices, kalman_df=None)  ← LEGACY single-stock fallback
      Retained for backward compatibility and single-stock diagnostics.
      Columns: [roll_ret, realized_vol] or [roll_ret, realized_vol, zscore]

Regimes (auto-labelled after each fit):
  MEAN_REVERTING — most negative spread autocorrelation, low vol  → enter new trades
  TRENDING       — positive autocorrelation, directional drift     → avoid new entries
  VOLATILE       — highest realized vol, stress period            → reduce positions

Improvements over v1:
  1. Spread-based features instead of single-stock prices
  2. spread_autocorr as primary MR indicator (negative = mean-reverting)
  3. Winsorization at ±winsorize_sigma std devs — robust to fat tails / crisis
  4. build_market_features() for shared regime across the full universe
  5. fit() accepts pd.DataFrame → column names drive state labelling automatically
"""

from __future__ import annotations

from enum import IntEnum
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from loguru import logger
from sklearn.preprocessing import StandardScaler


# ── Regime enum ───────────────────────────────────────────────────────────────

class Regime(IntEnum):
    MEAN_REVERTING = 0
    TRENDING       = 1
    VOLATILE       = 2


REGIME_LABEL: dict[Regime, str] = {
    Regime.MEAN_REVERTING: "mean_reverting",
    Regime.TRENDING:       "trending",
    Regime.VOLATILE:       "volatile",
}

TRADEABLE_REGIMES = {Regime.MEAN_REVERTING}

# Feature column names used by _label_states to find the right indices
_COL_VOL      = "realized_vol"
_COL_AUTOCORR = "spread_autocorr"
_COL_RET      = "spread_ret"


# ── Feature builders ──────────────────────────────────────────────────────────

def build_features_from_spread(
    kalman_df: pd.DataFrame,
    vol_window: int = 20,
    autocorr_window: int = 10,
) -> pd.DataFrame:
    """PRIMARY: Build 4-feature HMM matrix from Kalman filter output.

    All features describe the pair spread, not individual assets.
    This is the correct input when using HMM for pairs trading regime detection.

    Parameters
    ----------
    kalman_df       : Output of run_kalman() — must have columns 'spread' and 'zscore'.
    vol_window      : Rolling window for realized vol of spread returns (default 20).
    autocorr_window : Rolling window for lag-1 autocorrelation (default 10).

    Returns
    -------
    DataFrame with columns [spread_ret, realized_vol, spread_autocorr, zscore], no NaNs.

    Interpretation of spread_autocorr:
      Negative → spread mean-reverts (good for pairs entry)
      Positive → spread is trending (avoid entry)
      Near 0   → no clear structure (volatile or random)
    """
    missing = {"spread", "zscore"} - set(kalman_df.columns)
    if missing:
        raise ValueError(f"kalman_df missing required columns: {missing}")

    spread     = kalman_df["spread"]
    spread_ret = spread.diff()
    spread_ret_lag = spread_ret.shift(1)

    feat = pd.DataFrame(
        {
            "spread_ret":     spread_ret,
            "realized_vol":   spread_ret.rolling(vol_window).std(),
            "spread_autocorr": spread_ret.rolling(autocorr_window).corr(spread_ret_lag),
            "zscore":         kalman_df["zscore"],
        },
        index=kalman_df.index,
    )
    return feat.dropna()


def build_market_features(
    universe_close: pd.DataFrame,
    ret_window: int = 5,
    vol_window: int = 20,
    autocorr_window: int = 10,
) -> pd.DataFrame:
    """SHARED: Build market-wide HMM features from a universe of close prices.

    Train ONE HMM on these cross-sectional features and apply the resulting regime
    labels to ALL pairs in the universe. This avoids fitting 1,000+ per-pair HMMs
    and produces more stable, market-aware regime labels.

    Parameters
    ----------
    universe_close  : DataFrame where each column is one symbol's adjusted close.
    ret_window      : Rolling window for mean return smoothing (default 5).
    vol_window      : Rolling window for individual realized vol (default 20).
    autocorr_window : Rolling window for lag-1 autocorr (default 10).

    Returns
    -------
    DataFrame with columns [market_ret, cross_vol, avg_realized_vol, avg_autocorr].
    """
    if universe_close.shape[1] < 2:
        raise ValueError("universe_close must contain at least 2 symbols")

    log_rets = np.log(universe_close).diff()

    # Per-symbol rolling autocorr then averaged across universe
    autocorr_cols = {
        col: log_rets[col].rolling(autocorr_window).corr(log_rets[col].shift(1))
        for col in log_rets.columns
    }
    avg_autocorr = pd.DataFrame(autocorr_cols).mean(axis=1)

    feat = pd.DataFrame(
        {
            "market_ret":       log_rets.mean(axis=1).rolling(ret_window).mean(),
            "cross_vol":        log_rets.std(axis=1).rolling(ret_window).mean(),
            "avg_realized_vol": log_rets.rolling(vol_window).std().mean(axis=1),
            "avg_autocorr":     avg_autocorr,
        },
        index=universe_close.index,
    )
    return feat.dropna()


def build_features(
    prices: pd.Series,
    kalman_df: Optional[pd.DataFrame] = None,
    ret_window: int = 5,
    vol_window: int = 20,
) -> pd.DataFrame:
    """LEGACY: Build HMM features from a single price series.

    Retained for backward compatibility and single-stock diagnostics.
    For pairs trading, prefer build_features_from_spread().

    Returns columns [roll_ret, realized_vol] or [roll_ret, realized_vol, zscore].
    """
    if (prices <= 0).any():
        raise ValueError("prices must be strictly positive")

    log_rets = np.log(prices).diff()
    feat = pd.DataFrame(
        {
            "roll_ret":     log_rets.rolling(ret_window).mean(),
            "realized_vol": log_rets.rolling(vol_window).std(),
        },
        index=prices.index,
    )

    if kalman_df is not None:
        if "zscore" not in kalman_df.columns:
            raise ValueError("kalman_df must contain a 'zscore' column")
        feat["zscore"] = kalman_df["zscore"].reindex(prices.index)

    return feat.dropna()


# ── Detector ──────────────────────────────────────────────────────────────────

class HMMRegimeDetector:
    """Gaussian HMM regime detector with automatic, economically-grounded state labelling.

    Parameters
    ----------
    n_states        : Hidden states (default 3: MR / Trending / Volatile).
    n_iter          : Max EM iterations (default 200).
    random_state    : Seed for reproducibility.
    winsorize_sigma : Clip features at ±N std devs in scaled space before fitting/predicting.
                      Reduces influence of crisis-period outliers (fat tails).
                      Set None to disable. Default 3.0.
    """

    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 200,
        random_state: int = 42,
        winsorize_sigma: Optional[float] = 3.0,
    ) -> None:
        if n_states < 2:
            raise ValueError("n_states must be >= 2")
        self.n_states        = n_states
        self.n_iter          = n_iter
        self.random_state    = random_state
        self.winsorize_sigma = winsorize_sigma

        self._model: Optional[GaussianHMM] = None
        self._scaler: Optional[StandardScaler] = None
        self._feature_names: Optional[list[str]] = None
        self._state_to_regime: dict[int, Regime] = {}

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray | pd.DataFrame) -> "HMMRegimeDetector":
        """Fit HMM on feature matrix.

        Parameters
        ----------
        X : (T × n_features) array or DataFrame.
            If DataFrame, column names are used to identify realized_vol and
            spread_autocorr for automatic state labelling.

        Features are standardised then optionally winsorised internally.
        """
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
            X_arr = X.to_numpy(dtype=float)
        else:
            self._feature_names = None
            X_arr = _to_2d(np.asarray(X, dtype=float))

        self._scaler = StandardScaler().fit(X_arr)
        X_scaled = self._scaler.transform(X_arr)

        if self.winsorize_sigma is not None:
            X_scaled = np.clip(X_scaled, -self.winsorize_sigma, self.winsorize_sigma)

        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=False,
        )
        model.fit(X_scaled)
        self._model = model
        self._state_to_regime = self._label_states(X_scaled)
        return self

    # ── predict ──────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return array of Regime int values for each row of X."""
        self._check_fitted()
        X_arr = _extract_array(X)
        X_scaled = self._scaler.transform(X_arr)
        if self.winsorize_sigma is not None:
            X_scaled = np.clip(X_scaled, -self.winsorize_sigma, self.winsorize_sigma)
        raw = self._model.predict(X_scaled)
        return np.array([self._state_to_regime[s] for s in raw], dtype=int)

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return posterior state probabilities, shape (T, n_states).

        Columns ordered: [MEAN_REVERTING, TRENDING, VOLATILE].
        """
        self._check_fitted()
        X_arr = _extract_array(X)
        X_scaled = self._scaler.transform(X_arr)
        if self.winsorize_sigma is not None:
            X_scaled = np.clip(X_scaled, -self.winsorize_sigma, self.winsorize_sigma)
        _, posteriors = self._model.score_samples(X_scaled)

        ordered = np.zeros_like(posteriors)
        for raw_state, regime in self._state_to_regime.items():
            if int(regime) < posteriors.shape[1]:
                ordered[:, int(regime)] = posteriors[:, raw_state]
        return ordered

    def predict_series(self, features: pd.DataFrame) -> pd.Series:
        """Predict regimes and return a named Series preserving the input index."""
        regimes = self.predict(features.values)
        labels = pd.array([REGIME_LABEL[Regime(r)] for r in regimes], dtype="string")
        return pd.Series(labels, index=features.index, name="regime")

    # ── trading rule ─────────────────────────────────────────────────────────

    @staticmethod
    def is_tradeable(regime: Regime | int) -> bool:
        """Return True only when regime is MEAN_REVERTING."""
        return Regime(regime) in TRADEABLE_REGIMES

    # ── internals ────────────────────────────────────────────────────────────

    def _label_states(self, X_scaled: np.ndarray) -> dict[int, Regime]:
        """Map raw HMM state indices → Regime using economically-grounded rules.

        Priority:
          1. State with highest mean realized_vol          → VOLATILE
          2. Among remaining states:
             a. If spread_autocorr column present:
                  most negative autocorr mean             → MEAN_REVERTING
                  (negative autocorr = spread mean-reverts)
             b. Fallback (no autocorr column):
                  lowest |mean roll_ret|                  → MEAN_REVERTING
          3. Remaining state                              → TRENDING
        """
        raw_states = self._model.predict(X_scaled)
        n_feat = X_scaled.shape[1]
        fn = self._feature_names or []

        # Per-state feature means on training data
        state_means = {
            s: (X_scaled[raw_states == s].mean(axis=0)
                if (raw_states == s).any() else np.zeros(n_feat))
            for s in range(self.n_states)
        }

        # Resolve column indices from names (or fall back to positional)
        vol_idx      = fn.index(_COL_VOL)      if _COL_VOL      in fn else min(1, n_feat - 1)
        autocorr_idx = fn.index(_COL_AUTOCORR) if _COL_AUTOCORR in fn else None
        ret_idx      = fn.index(_COL_RET)      if _COL_RET      in fn else 0

        sorted_by_vol = sorted(range(self.n_states),
                               key=lambda s: state_means[s][vol_idx])
        state_map: dict[int, Regime] = {}

        if self.n_states == 3:
            volatile_state = sorted_by_vol[2]
            low_vol_pair   = sorted_by_vol[:2]

            if autocorr_idx is not None:
                # Primary criterion: most negative autocorr → MR
                low_vol_pair.sort(key=lambda s: state_means[s][autocorr_idx])
                state_map[low_vol_pair[0]] = Regime.MEAN_REVERTING  # most negative
                state_map[low_vol_pair[1]] = Regime.TRENDING
            else:
                # Fallback: lowest |return| → MR
                low_vol_pair.sort(key=lambda s: abs(state_means[s][ret_idx]))
                state_map[low_vol_pair[0]] = Regime.MEAN_REVERTING
                state_map[low_vol_pair[1]] = Regime.TRENDING

            state_map[volatile_state] = Regime.VOLATILE
        else:
            state_map[sorted_by_vol[0]]  = Regime.MEAN_REVERTING
            state_map[sorted_by_vol[-1]] = Regime.VOLATILE
            for s in sorted_by_vol[1:-1]:
                state_map[s] = Regime.TRENDING

        return state_map

    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_2d(X: np.ndarray) -> np.ndarray:
    if X.ndim == 1:
        return X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"X must be 1-D or 2-D, got shape {X.shape}")
    return X


def _extract_array(X: np.ndarray | pd.DataFrame) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=float)
    return _to_2d(np.asarray(X, dtype=float))
