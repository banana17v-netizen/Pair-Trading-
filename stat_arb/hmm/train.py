"""Rolling HMM re-training manager.

At each retrain checkpoint (every `retrain_freq` days), a fresh HMMRegimeDetector
is trained on the previous `train_window` days of features, then used to label the
NEXT `retrain_freq` days (out-of-sample).  This mirrors the walk-forward backtest
structure and prevents any look-ahead bias in regime labels.

Settings (from config/settings.yaml):
  hmm.train_window_days : 252   (1 year in-sample)
  hmm.retrain_freq_days : 21    (~1 month OOS interval)
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from joblib import Parallel, delayed
from loguru import logger

from stat_arb.hmm.regime import (
    HMMRegimeDetector,
    Regime,
    REGIME_LABEL,
)


class RollingHMMTrainer:
    """Walk-forward rolling HMM trainer.

    Parameters
    ----------
    train_window  : Number of in-sample rows to train each model on (default 252).
    retrain_freq  : Rows between re-training checkpoints (default 21).
    n_states      : Passed to HMMRegimeDetector.
    n_iter        : Max EM iterations per fit.
    random_state  : Seed — fixed for reproducibility.
    fallback      : Regime to assign when HMM fitting fails (default VOLATILE).
    """

    def __init__(
        self,
        train_window: int = 252,
        retrain_freq: int = 21,
        n_states: int = 3,
        n_iter: int = 200,
        random_state: int = 42,
        fallback: Regime = Regime.VOLATILE,
    ) -> None:
        if train_window < 50:
            raise ValueError("train_window must be >= 50 for stable HMM fitting")
        if retrain_freq < 1:
            raise ValueError("retrain_freq must be >= 1")

        self.train_window  = train_window
        self.retrain_freq  = retrain_freq
        self.n_states      = n_states
        self.n_iter        = n_iter
        self.random_state  = random_state
        self.fallback      = fallback

        # Stored for post-hoc inspection
        self.models_: list[tuple[int, HMMRegimeDetector]] = []

    # ── main API ─────────────────────────────────────────────────────────────

    def fit_predict(self, features: pd.DataFrame, n_jobs: int = 1) -> pd.Series:
        """Run rolling train → predict and return a regime label Series.

        Parameters
        ----------
        features : Feature DataFrame with DatetimeIndex (output of any build_features*).
                   Must have >= train_window + retrain_freq rows.
        n_jobs   : Number of parallel jobs for fold fitting.
                   1 = sequential (default). -1 = all CPU cores.
                   Parallel mode is useful when running many folds (large datasets).

        Returns
        -------
        pd.Series of regime strings ("mean_reverting" / "trending" / "volatile"),
        same index as features. Rows before the first train window are NaN.
        """
        if len(features) < self.train_window + self.retrain_freq:
            raise ValueError(
                f"Need at least {self.train_window + self.retrain_freq} rows, "
                f"got {len(features)}"
            )

        n = len(features)
        labels = pd.Series(pd.NA, index=features.index, dtype="string", name="regime")
        self.models_ = []

        checkpoints = list(range(self.train_window, n, self.retrain_freq))
        total = len(checkpoints)

        # ── run folds (sequential or parallel) ───────────────────────────────
        fold_results = Parallel(n_jobs=n_jobs)(
            delayed(self._fit_one_fold)(features, t, fold_idx, total)
            for fold_idx, t in enumerate(checkpoints, 1)
        )

        # ── collect results in original order ─────────────────────────────────
        for (t, chunk_labels, detector) in fold_results:
            pred_end = min(t + self.retrain_freq, n)
            labels.iloc[t:pred_end] = chunk_labels
            if detector is not None:
                self.models_.append((t, detector))

        return labels

    def _fit_one_fold(
        self,
        features: pd.DataFrame,
        t: int,
        fold_idx: int,
        total: int,
    ) -> tuple[int, list[str], Optional[HMMRegimeDetector]]:
        """Fit one fold and return (t, label_list, detector).  Thread-safe."""
        n = len(features)
        train_slice = features.iloc[t - self.train_window : t]
        pred_end    = min(t + self.retrain_freq, n)
        pred_slice  = features.iloc[t : pred_end]

        if pred_slice.empty:
            return t, [], None

        detector = HMMRegimeDetector(
            n_states=self.n_states,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )

        try:
            # Pass DataFrame so column names drive automatic state labelling
            detector.fit(train_slice)
            regime_series = detector.predict_series(pred_slice)
            chunk = list(regime_series.values)

            if fold_idx % 10 == 0 or fold_idx == total:
                logger.debug(
                    f"HMM fold {fold_idx}/{total} | "
                    f"train {train_slice.index[0].date()}–{train_slice.index[-1].date()} | "
                    f"pred rows={len(pred_slice)}"
                )
            return t, chunk, detector

        except Exception as exc:
            logger.warning(
                f"HMM fit failed at fold {fold_idx} (t={t}): {exc}. "
                f"Filling with fallback='{REGIME_LABEL[self.fallback]}'"
            )
            return t, [REGIME_LABEL[self.fallback]] * len(pred_slice), None

    # ── convenience ──────────────────────────────────────────────────────────

    def regime_counts(self, labels: pd.Series) -> pd.Series:
        """Return value counts of non-null regime labels."""
        return labels.dropna().value_counts()

    def regime_fractions(self, labels: pd.Series) -> pd.Series:
        """Return fraction of time in each regime (non-null rows only)."""
        counts = self.regime_counts(labels)
        return (counts / counts.sum()).round(4)
