"""Environment base class.

An Environment defines the data-generating process: how contexts arrive,
how rewards are generated, and (optionally) when distributions shift.
It also exposes oracle information — true mean rewards and true Pareto
sets — which the runner uses to compute regret.

Required methods:
    reset(seed)             : reset RNG / shift state
    context(t) -> ndarray   : generate the t-th context
    step(t, action) -> ndarray  : sample a noisy reward vector
    true_means(context) -> ndarray  : (K, M) oracle means at this context

Optional methods:
    is_shifted(t) -> bool   : whether t is post-shift (default False)
    shift_times() -> list   : known shift times (for evaluation only)
"""

from __future__ import annotations

import numpy as np


class Environment:
    """Base class for multi-objective contextual bandit environments.

    Subclasses must set `n_arms`, `context_dim`, `n_objectives` in __init__
    (or as class attributes) and implement `context`, `step`, `true_means`.
    """

    n_arms: int = 0
    context_dim: int = 0
    n_objectives: int = 0

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = np.random.default_rng(rng)

    # ─── Required interface ──────────────────────────────────────────

    def reset(self, seed: int | None = None) -> None:
        """Reset the environment for a fresh run."""
        self.rng = np.random.default_rng(seed)

    def context(self, t: int) -> np.ndarray:
        """Generate the context at time t. Returns shape (context_dim,)."""
        raise NotImplementedError

    def step(self, t: int, action: int) -> np.ndarray:
        """Sample a noisy reward vector for `action` at time t.

        Default implementation: true_means + sigma * standard_normal,
        where sigma defaults to 0.1.  Subclasses can override.
        """
        means = self.true_means(self.context(t))
        sigma = getattr(self, "sigma", 0.1)
        return means[action] + sigma * self.rng.standard_normal(self.n_objectives)

    def true_means(self, context: np.ndarray) -> np.ndarray:
        """Oracle (K, M) array of true mean rewards at this context."""
        raise NotImplementedError

    # ─── Optional shift hooks ───────────────────────────────────────

    def is_shifted(self, t: int) -> bool:
        """Whether the data-generating distribution has shifted by time t.

        Default: never. Subclasses with shifts should override.
        """
        return False

    def shift_times(self) -> list:
        """Sorted list of (known) change-points. Empty if no shifts.

        Used only by evaluation code (e.g., RecoveryTime metric).  The
        algorithm itself never sees this.
        """
        return []

    # ─── Convenience ────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(K={self.n_arms}, M={self.n_objectives}, "
            f"d={self.context_dim})"
        )
