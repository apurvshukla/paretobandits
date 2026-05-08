"""Algorithm base class.

Every bandit algorithm in paretobandits implements four methods:

    reset(seed)                    : reset internal state
    act(context) -> int            : select an arm given a context
    update(context, action, reward): update state from observed reward
    pareto_estimate(context) -> set: estimated Pareto-optimal arm ids

The `pareto_estimate` method is what makes preference-regret computable —
without it, evaluation would have to reverse-engineer the algorithm's
internal Pareto belief from action choices alone.

Contexts are np.ndarray of shape (context_dim,) and rewards are
np.ndarray of shape (M,).  Actions are arm ids in {0, ..., n_arms-1}.
"""

from __future__ import annotations

import numpy as np

from paretobandits.core.preference import Preference


class Algorithm:
    """Base class for multi-objective contextual bandit algorithms.

    Args:
        n_arms: number of discrete arms K.
        context_dim: dimension d of the context vector.
        n_objectives: dimension M of the reward vector.
        preference: a Preference object defining the dominance order.
        delta: confidence parameter (failure probability budget).
        horizon: optional horizon T — some algorithms (e.g., level-tuning)
                 use this; pass None for anytime variants.
        rng: numpy Generator or seed for reproducibility.
    """

    name: str = "Algorithm"

    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        n_objectives: int,
        preference: Preference,
        delta: float = 0.05,
        horizon: int | None = None,
        rng: np.random.Generator | None = None,
    ):
        if n_arms < 1:
            raise ValueError("n_arms must be >= 1")
        if context_dim < 1:
            raise ValueError("context_dim must be >= 1")
        if n_objectives != preference.M:
            raise ValueError(
                f"n_objectives={n_objectives} doesn't match preference.M={preference.M}"
            )
        if not 0 < delta < 1:
            raise ValueError("delta must be in (0, 1)")
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.n_objectives = n_objectives
        self.preference = preference
        self.delta = delta
        self.horizon = horizon
        self.rng = np.random.default_rng(rng)
        self._t = 0

    # ─── Required interface ──────────────────────────────────────────

    def reset(self, seed: int | None = None) -> None:
        """Reset internal state. Called once per run."""
        self.rng = np.random.default_rng(seed)
        self._t = 0

    def act(self, context: np.ndarray) -> int:
        """Choose an arm given the current context."""
        raise NotImplementedError

    def update(
        self, context: np.ndarray, action: int, reward: np.ndarray
    ) -> None:
        """Update internal state from observed (context, action, reward)."""
        raise NotImplementedError

    def pareto_estimate(self, context: np.ndarray) -> set:
        """Set of arm ids the algorithm currently believes are Pareto-optimal.

        Used by preference-regret computation.  Default implementation:
        returns all arms (i.e., "no information yet"); subclasses should
        override.
        """
        return set(range(self.n_arms))

    # ─── Convenience ────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(n_arms={self.n_arms}, "
            f"context_dim={self.context_dim}, M={self.n_objectives})"
        )
