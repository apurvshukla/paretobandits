"""Synthetic environment with configurable distribution shifts.

This is the canonical synthetic benchmark used to validate the paper's
theorems.  The reward function for each arm is a Hölder-β-continuous
function of the 1D context, with peaks at different positions per arm
so a non-trivial Pareto front exists at every context.

Shift schedules
---------------
- "none":     Stationary. Single context distribution for the whole run.
- "single":   One change point t_p; pre-shift Beta(1, ν+1), post Beta(ν+1, 1).
- "multi":    Multiple change points at user-specified times (alternates
              between two distributions).
- "gradual":  Smooth interpolation from source to target over [t0, t1].
- "tree":     Tree-discretized D(γ, C_γ) family from Theorem 2.

The same `SyntheticShift` class supports all schedules — pass `schedule=`
to switch.

Reward model
------------
Two objectives by default. For arm k ∈ {0, ..., K-1} and context x ∈ [0,1]:

    μ_k^1(x) = max(ε, 1 - L * |k/K - peak_1(x)|)        (objective 1)
    μ_k^2(x) = max(ε, 1 - L * |k/K - peak_2(x)|)        (objective 2)

where peak_1(x), peak_2(x) move smoothly with x and never coincide,
guaranteeing a multi-arm Pareto front for every context.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from paretobandits.core.environment import Environment


class SyntheticShift(Environment):
    """Synthetic bandit environment with configurable Hölder smoothness
    and distribution shift.

    Args:
        n_arms: K
        n_objectives: M (default 2)
        beta: Hölder exponent of mean rewards (default 1.0).
        lipschitz_L: Hölder constant (default 4.0).
        sigma: noise std-dev for each objective (default 0.1).
        schedule: shift schedule — see module docstring.
        shift_times: change-point times. Single int for "single",
                     sequence for "multi", pair for "gradual".
        nu: Beta-distribution skewness (default 5.0).
        seed: RNG seed.
    """

    def __init__(
        self,
        n_arms: int = 10,
        n_objectives: int = 2,
        context_dim: int = 1,
        beta: float = 1.0,
        lipschitz_L: float = 4.0,
        sigma: float = 0.1,
        schedule: str = "single",
        shift_times: Sequence[int] | None = None,
        nu: float = 5.0,
        seed: int | None = None,
    ):
        super().__init__(rng=seed)
        if n_objectives != 2:
            raise NotImplementedError(
                "SyntheticShift currently supports n_objectives=2. "
                "Generalizing to M>2 is mostly cosmetic — see _arm_means."
            )
        if context_dim < 1:
            raise ValueError(f"context_dim must be >= 1, got {context_dim}")
        self.n_arms = n_arms
        self.n_objectives = n_objectives
        self.context_dim = context_dim
        self.beta = beta
        self.lipschitz_L = lipschitz_L
        self.sigma = sigma
        self.nu = nu

        if schedule not in ("none", "single", "multi", "gradual", "tree"):
            raise ValueError(f"unknown schedule {schedule!r}")
        self.schedule = schedule

        if schedule == "none":
            self._shifts = []
        elif schedule == "single":
            if shift_times is None:
                shift_times = [1500]
            elif isinstance(shift_times, int):
                shift_times = [shift_times]
            self._shifts = list(shift_times)
        elif schedule == "multi":
            if shift_times is None:
                raise ValueError("multi schedule requires shift_times")
            self._shifts = list(shift_times)
        elif schedule == "gradual":
            if shift_times is None or len(shift_times) != 2:
                raise ValueError("gradual schedule requires (t0, t1)")
            self._shifts = list(shift_times)
        elif schedule == "tree":
            if shift_times is None:
                shift_times = [1500]
            self._shifts = list(shift_times)

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)

    # ─── Context generation ──────────────────────────────────────────

    def context(self, t: int) -> np.ndarray:
        """Draw an i.i.d. d-dim context. Each axis follows the same
        shift schedule (independent Beta marginals).
        """
        d = self.context_dim
        if self.schedule == "none":
            a, b = 1.0, self.nu + 1
            x = self.rng.beta(a, b, size=d)
        elif self.schedule == "single":
            tp = self._shifts[0]
            if t < tp:
                x = self.rng.beta(1.0, self.nu + 1, size=d)
            else:
                x = self.rng.beta(self.nu + 1, 1.0, size=d)
        elif self.schedule == "multi":
            phase = sum(1 for ts in self._shifts if t >= ts) % 2
            if phase == 0:
                x = self.rng.beta(1.0, self.nu + 1, size=d)
            else:
                x = self.rng.beta(self.nu + 1, 1.0, size=d)
        elif self.schedule == "gradual":
            t0, t1 = self._shifts
            if t < t0:
                a, b = 1.0, self.nu + 1
            elif t > t1:
                a, b = self.nu + 1, 1.0
            else:
                w = (t - t0) / max(t1 - t0, 1)
                a = (1 - w) * 1.0 + w * (self.nu + 1)
                b = (1 - w) * (self.nu + 1) + w * 1.0
            x = self.rng.beta(a, b, size=d)
        elif self.schedule == "tree":
            tp = self._shifts[0]
            if t < tp:
                x = self.rng.beta(1.0 + self.beta, 1.0, size=d)
            else:
                x = self.rng.uniform(size=d)
        else:
            x = self.rng.uniform(size=d)
        return np.clip(x, 1e-3, 1 - 1e-3).astype(float)

    # ─── True means ─────────────────────────────────────────────────

    def true_means(self, context: np.ndarray) -> np.ndarray:
        """Mean rewards as a function of context.

        For d > 1 we project to a 1-D summary (mean of the per-axis
        contexts) and use the same per-arm reward shape as the 1D case.
        This keeps the Pareto landscape directly comparable across
        dimensions while still letting the algorithm benefit from
        d-dim partitioning of the context distribution.
        """
        x_vec = np.atleast_1d(np.asarray(context, dtype=float)).reshape(-1)
        x = float(np.mean(x_vec))
        return self._arm_means(x)

    def _arm_means(self, x: float) -> np.ndarray:
        """(K, M=2) mean rewards. Two peaks that move with x.

        peak_1(x) = (8 - 8x) / 10  — drifts left as x grows.
        peak_2(x) = (10 - 8x) / 10 — drifts left more slowly.
        """
        K = self.n_arms
        eps = 1e-3
        peak_1 = (8 - 8 * x) / 10
        peak_2 = (10 - 8 * x) / 10
        means = np.zeros((K, 2))
        idxs = np.arange(K) / max(K - 1, 1)            # in [0, 1]
        # Triangle-shaped peaks with Hölder-β slope.
        means[:, 0] = np.maximum(
            eps, 1 - self.lipschitz_L * np.abs(idxs - peak_1) ** self.beta
        )
        # Asymmetric for objective 2: gentler decay above peak_2.
        below = idxs <= peak_2
        means[:, 1] = np.where(
            below,
            np.maximum(
                eps,
                1 - self.lipschitz_L * (peak_2 - idxs) ** self.beta,
            ),
            np.maximum(
                eps,
                1 - 0.25 * self.lipschitz_L * (idxs - peak_2) ** self.beta,
            ),
        )
        return np.clip(means, eps, 1.0)

    # ─── Shift hooks ────────────────────────────────────────────────

    def is_shifted(self, t: int) -> bool:
        if self.schedule == "none":
            return False
        return t >= self._shifts[0]

    def shift_times(self) -> list:
        return list(self._shifts)
