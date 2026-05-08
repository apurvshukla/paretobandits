"""Baseline algorithms for comparison.

Two simple baselines:
  - RandomPlay: pick uniformly at random. The "you must beat this" floor.
  - ScalarizedUCB: project rewards via a fixed weight vector and run UCB1
                   on the scalarized reward. The "weighted-sum" baseline
                   the paper argues against in its motivation.

More sophisticated baselines (Türgay et al. 2018, Suk & Kpotufe 2020,
Auer et al. 2016) live in `paretobandits.algos.legacy.*` — those wrap the
existing implementations from the original `script/` directory and are
included in the reproducibility track of the benchmark.
"""

from __future__ import annotations

import numpy as np

from paretobandits.core.algorithm import Algorithm
from paretobandits.core.preference import Preference


class RandomPlay(Algorithm):
    """Pick uniformly at random. Used as a regret-floor benchmark."""

    name = "RandomPlay"

    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        n_objectives: int,
        preference: Preference,
        delta: float = 0.05,
        horizon: int | None = None,
        rng=None,
    ):
        super().__init__(
            n_arms=n_arms,
            context_dim=context_dim,
            n_objectives=n_objectives,
            preference=preference,
            delta=delta,
            horizon=horizon,
            rng=rng,
        )

    def act(self, context: np.ndarray) -> int:
        return int(self.rng.integers(0, self.n_arms))

    def update(
        self, context: np.ndarray, action: int, reward: np.ndarray
    ) -> None:
        self._t += 1

    def pareto_estimate(self, context: np.ndarray) -> set[int]:
        return set(range(self.n_arms))


class ScalarizedUCB(Algorithm):
    """UCB1 on weighted-sum scalarized rewards.

    `weights`: (M,) non-negative; defaults to uniform 1/M across objectives.

    This is intentionally simple — context-agnostic, no covariate shift
    handling.  Demonstrates the failure mode the paper's motivation
    section calls out: optimal scalarization to identify Pareto points
    is itself a hard problem.
    """

    name = "ScalarizedUCB"

    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        n_objectives: int,
        preference: Preference,
        weights: np.ndarray | None = None,
        delta: float = 0.05,
        horizon: int | None = None,
        rng=None,
    ):
        super().__init__(
            n_arms=n_arms,
            context_dim=context_dim,
            n_objectives=n_objectives,
            preference=preference,
            delta=delta,
            horizon=horizon,
            rng=rng,
        )
        if weights is None:
            weights = np.ones(n_objectives) / n_objectives
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (n_objectives,):
            raise ValueError(
                f"weights shape must be ({n_objectives},), got {weights.shape}"
            )
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")
        self.weights = weights / np.sum(weights)

        # Per-arm UCB state — context-agnostic (this is the baseline's
        # weakness, on purpose).
        self._sums = np.zeros(n_arms)
        self._counts = np.zeros(n_arms, dtype=int)

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._sums = np.zeros(self.n_arms)
        self._counts = np.zeros(self.n_arms, dtype=int)

    def act(self, context: np.ndarray) -> int:
        # Initialization: play each arm once.
        unplayed = np.where(self._counts == 0)[0]
        if len(unplayed) > 0:
            return int(unplayed[0])
        means = self._sums / self._counts
        # UCB1 bonus: sqrt(2 log t / n_k).
        bonus = np.sqrt(2 * np.log(max(self._t, 2)) / self._counts)
        return int(np.argmax(means + bonus))

    def update(
        self, context: np.ndarray, action: int, reward: np.ndarray
    ) -> None:
        scalar = float(np.dot(self.weights, reward))
        self._sums[action] += scalar
        self._counts[action] += 1
        self._t += 1

    def pareto_estimate(self, context: np.ndarray) -> set[int]:
        # Scalarized algorithms collapse to a single arm — return only
        # the current best.  This is how they score against d_p regret.
        if np.any(self._counts == 0):
            return set(range(self.n_arms))
        means = self._sums / self._counts
        return {int(np.argmax(means))}
