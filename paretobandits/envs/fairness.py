"""Fairness contextual bandit environment.

Reframes a tabular fair-classification problem as a multi-objective
contextual bandit. At each step the environment samples a row from a
dataset (one (features, group, outcome) triple), the algorithm picks
one of K threshold-based prediction strategies, and rewards are returned
as a 2-vector:

  reward[0] = accuracy_indicator   (1 if prediction matches outcome else 0)
  reward[1] = fairness_indicator   (running 1 - |demographic_parity_gap|)

The fairness component uses a rolling-window estimate of the
demographic-parity gap (or any user-selected fairness metric — the env
exposes the choice via `fairness_metric=`). Demographic parity, equal
opportunity, and predictive parity are supported.

Distribution shift comes from the data ordering: pre-shift the
environment over-samples one demographic group, post-shift the other.
This creates a real Pareto-front shift because the optimal threshold
varies by group, so an algorithm that doesn't adapt loses.

Two ways to use the environment:
  - With a built-in semi-synthetic generator (no external deps) — useful
    for CI and the smoke benchmark.
  - With a real CSV (Adult / COMPAS / German Credit) — pass `csv_path=`.
    Documented schema: `feature_*` columns, exactly one `group` column
    (binary), exactly one `outcome` column (binary).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from paretobandits.core.environment import Environment

_FAIRNESS_METRICS = ("demographic_parity", "equal_opportunity", "predictive_parity")


class FairnessBandit(Environment):
    """Multi-objective contextual bandit over a fair-classification task.

    Args:
        n_arms: K threshold strategies; thresholds linearly spaced in [0,1].
        n_features: dimension of feature vector. Used by the synthetic
                    generator only; ignored if `csv_path` is given.
        sigma: noise std-dev added to per-step rewards (default 0.05).
        csv_path: optional CSV with columns `feature_*`, `group`, `outcome`.
                  When None, falls back to the synthetic generator.
        fairness_metric: which fairness signal to use as objective 2.
                         One of "demographic_parity", "equal_opportunity",
                         "predictive_parity". Default "demographic_parity".
        window: rolling-window size for the running fairness estimate.
                Default 200.
        schedule: one of "none", "single", "multi". With "single", the
                  group sampling probability shifts at the change point
                  (one group dominates pre-shift, the other post-shift).
        shift_times: change-point times.
        seed: RNG seed.

    Notes:
        Per-step "fairness reward" is `1 - |gap|` where `gap` is the
        demographic-parity gap estimated over the most recent `window`
        samples for which the same arm `k` was played. This makes
        rewards continuous in [0, 1] and naturally tied to per-arm
        behavior — playing arm k more reduces the variance of its gap
        estimate, encouraging adaptive policies to invest in arms that
        are both accurate and fair.
    """

    n_objectives = 2

    def __init__(
        self,
        n_arms: int = 10,
        n_features: int = 6,
        sigma: float = 0.05,
        csv_path: str | None = None,
        fairness_metric: str = "demographic_parity",
        window: int = 200,
        schedule: str = "single",
        shift_times: Sequence[int] | None = None,
        seed: int | None = None,
    ):
        super().__init__(rng=seed)
        if fairness_metric not in _FAIRNESS_METRICS:
            raise ValueError(
                f"unknown fairness_metric {fairness_metric!r}, "
                f"choose from {_FAIRNESS_METRICS}"
            )
        if schedule not in ("none", "single", "multi"):
            raise ValueError(f"unknown schedule {schedule!r}")
        self.n_arms = int(n_arms)
        self.fairness_metric = fairness_metric
        self.window = int(window)
        self.schedule = schedule
        self.sigma = float(sigma)

        if csv_path is not None:
            self._features, self._group, self._outcome = self._load_csv(csv_path)
            self.n_features = self._features.shape[1]
        else:
            self.n_features = int(n_features)
            self._features = None
            self._group = None
            self._outcome = None

        self.context_dim = self.n_features

        if schedule == "none":
            self._shifts = []
        elif schedule == "single":
            self._shifts = list(shift_times or [5000])
        else:
            if shift_times is None:
                raise ValueError("multi schedule requires shift_times")
            self._shifts = list(shift_times)

        # Per-arm rolling buffers for fairness estimation: stores
        # (group, predicted_label, true_label) tuples for the most
        # recent `window` plays of each arm.
        self._buffers: dict[int, deque] = {
            k: deque(maxlen=self.window) for k in range(self.n_arms)
        }
        # Cached prediction probabilities per arm (lazy, set on first call).
        self._cached_score: float | None = None
        self._cached_group: int | None = None
        self._cached_outcome: int | None = None

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._buffers = {
            k: deque(maxlen=self.window) for k in range(self.n_arms)
        }
        self._cached_score = None
        self._cached_group = None
        self._cached_outcome = None

    # ─── Synthetic data generator ──────────────────────────────────

    def _generate_row(self, group_bias: float) -> tuple[np.ndarray, int, int, float]:
        """Generate one synthetic (features, group, outcome, score) tuple.

        Args:
            group_bias: P(group=1). 0.5 = balanced; the schedule sets this
                        per-step so pre-shift draws mostly group=0 and
                        post-shift mostly group=1.

        Returns:
            features: (d,) float vector in roughly [0, 1].
            group:    binary protected attribute.
            outcome:  binary true label.
            score:    underlying probability — used to test thresholds.
        """
        d = self.n_features
        group = int(self.rng.uniform() < group_bias)
        # Two demographic profiles with shifted feature distributions and
        # different optimal score thresholds.
        if group == 0:
            features = np.clip(self.rng.normal(0.45, 0.18, size=d), 0.0, 1.0)
        else:
            features = np.clip(self.rng.normal(0.65, 0.20, size=d), 0.0, 1.0)
        # Score = mean of features with a small group-specific offset.
        # Optimal threshold differs by group (~0.45 for group 0, ~0.65 for 1).
        score = float(np.mean(features) + (0.05 if group == 1 else -0.05))
        score = float(np.clip(score, 0.0, 1.0))
        # True outcome is Bernoulli(score) — score predicts well by design.
        outcome = int(self.rng.uniform() < score)
        return features, group, outcome, score

    def _load_csv(self, path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load a CSV with `feature_*`, `group`, `outcome` columns."""
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "Loading external fair-classification CSVs requires pandas. "
                "Install with: pip install paretobandits[warfarin]"
            ) from e
        df = pd.read_csv(Path(path))
        feat_cols = [c for c in df.columns if c.startswith("feature_")]
        if not feat_cols:
            raise ValueError(
                "CSV must contain at least one 'feature_*' column"
            )
        if "group" not in df.columns or "outcome" not in df.columns:
            raise ValueError("CSV must contain 'group' and 'outcome' columns")
        # Min-max normalize features to [0, 1].
        feats = df[feat_cols].to_numpy(dtype=float)
        lo, hi = feats.min(0), feats.max(0)
        rng = np.maximum(hi - lo, 1e-9)
        feats = (feats - lo) / rng
        return (
            feats,
            df["group"].to_numpy(dtype=int),
            df["outcome"].to_numpy(dtype=int),
        )

    def _sample_row(self, t: int) -> tuple[np.ndarray, int, int, float]:
        """Sample one row, applying the schedule to bias group draw."""
        if self.schedule == "none":
            group_bias = 0.5
        elif self.schedule == "single":
            tp = self._shifts[0]
            group_bias = 0.2 if t < tp else 0.8
        else:  # multi: alternate
            phase = sum(1 for ts in self._shifts if t >= ts) % 2
            group_bias = 0.2 if phase == 0 else 0.8

        if self._features is None:
            return self._generate_row(group_bias)
        # CSV mode: oversample by group.
        n = self._features.shape[0]
        target_group = 1 if self.rng.uniform() < group_bias else 0
        candidates = np.where(self._group == target_group)[0]
        if len(candidates) == 0:
            i = int(self.rng.integers(0, n))
        else:
            i = int(self.rng.choice(candidates))
        features = self._features[i]
        group = int(self._group[i])
        outcome = int(self._outcome[i])
        # Use feature mean as a synthetic score for threshold prediction.
        score = float(np.clip(np.mean(features), 0.0, 1.0))
        return features, group, outcome, score

    # ─── Environment interface ─────────────────────────────────────

    def context(self, t: int) -> np.ndarray:
        features, group, outcome, score = self._sample_row(t)
        # Stash so step() / true_means() see the same row.
        self._cached_score = score
        self._cached_group = group
        self._cached_outcome = outcome
        return features

    def _arm_threshold(self, k: int) -> float:
        """Threshold for arm k: linearly spaced in [0, 1]."""
        return float(k / max(self.n_arms - 1, 1))

    def _arm_predict(self, k: int, score: float) -> int:
        return int(score >= self._arm_threshold(k))

    def true_means(self, context: np.ndarray) -> np.ndarray:
        """Best estimate of (accuracy, fairness) per arm at this context.

        Accuracy: P(prediction matches outcome | this score). For a fixed
        threshold k and a context with score s and outcome y, accuracy
        is deterministic — 1 if (s ≥ thresh_k) == y else 0. We expose
        per-arm 0/1 means here; the noise comes from `step()`'s additive
        Gaussian perturbation.

        Fairness: per-arm running estimate of `1 - |gap|` over the
        rolling window. Initialized to 0.5 to avoid bias before data.
        """
        if self._cached_score is None:
            # Should only happen if true_means is called before context.
            return np.full((self.n_arms, 2), 0.5)
        means = np.zeros((self.n_arms, 2))
        score = self._cached_score
        outcome = self._cached_outcome
        for k in range(self.n_arms):
            pred = self._arm_predict(k, score)
            means[k, 0] = float(pred == outcome)
            means[k, 1] = self._fairness_estimate(k)
        return np.clip(means, 1e-3, 1.0 - 1e-3)

    def _fairness_estimate(self, k: int) -> float:
        """Estimate of `1 - |gap|` for arm k from its rolling buffer."""
        buf = self._buffers[k]
        if len(buf) < 10:
            return 0.5
        groups = np.array([b[0] for b in buf])
        preds = np.array([b[1] for b in buf])
        outcomes = np.array([b[2] for b in buf])
        if self.fairness_metric == "demographic_parity":
            p0 = preds[groups == 0].mean() if (groups == 0).any() else 0.5
            p1 = preds[groups == 1].mean() if (groups == 1).any() else 0.5
            gap = abs(p0 - p1)
        elif self.fairness_metric == "equal_opportunity":
            mask0 = (groups == 0) & (outcomes == 1)
            mask1 = (groups == 1) & (outcomes == 1)
            tpr0 = preds[mask0].mean() if mask0.any() else 0.5
            tpr1 = preds[mask1].mean() if mask1.any() else 0.5
            gap = abs(tpr0 - tpr1)
        else:  # predictive_parity
            mask0 = (groups == 0) & (preds == 1)
            mask1 = (groups == 1) & (preds == 1)
            ppv0 = outcomes[mask0].mean() if mask0.any() else 0.5
            ppv1 = outcomes[mask1].mean() if mask1.any() else 0.5
            gap = abs(ppv0 - ppv1)
        return float(np.clip(1.0 - gap, 0.0, 1.0))

    def step(self, t: int, action: int) -> np.ndarray:
        """Apply arm `action` to the cached row, update buffers, return reward."""
        if self._cached_score is None:
            self.context(t)
        score = self._cached_score
        group = self._cached_group
        outcome = self._cached_outcome
        pred = self._arm_predict(action, score)
        # Update rolling buffer for this arm.
        self._buffers[action].append((group, pred, outcome))
        # Reward = (accuracy_indicator, fairness_estimate) + sigma noise.
        accuracy = float(pred == outcome)
        fairness = self._fairness_estimate(action)
        reward = np.array([accuracy, fairness]) + self.sigma * self.rng.standard_normal(2)
        return np.clip(reward, 0.0, 1.0)

    # ─── Shift hooks ───────────────────────────────────────────────

    def is_shifted(self, t: int) -> bool:
        return self.schedule != "none" and t >= self._shifts[0]

    def shift_times(self) -> list:
        return list(self._shifts)
