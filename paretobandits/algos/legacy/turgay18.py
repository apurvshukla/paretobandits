"""Turgay18: Pareto Contextual Zooming (PCZ) baseline.

Faithful implementation of:
  Türgay, E., Öner, D. and Tekin, C. (2018). "Multi-objective contextual
  bandit problem with similarity information." arXiv:1803.04015.

Algorithm (Algorithm 1 in the paper)
------------------------------------
PCZ adaptively partitions the joint context–arm similarity space using a
collection of "balls" (axis-aligned squares under sup-norm here). Each
ball maintains per-objective sample means; the algorithm picks balls from
the Pareto front of "indices" (mean + sample uncertainty + radius), with
similarity-propagation tightening across nearby balls.

Mechanism per step
~~~~~~~~~~~~~~~~~~
  1. Observe context x_t.
  2. Find the relevant balls — those whose domain dom(B) intersects the
     vertical slice at x_t.  dom(B) = B \\ {smaller balls} carves children
     out of parents so each (x, y) point is owned by the deepest ball
     containing it.
  3. For each ball, compute the index vector
        g_B^i = r(B) + min_{B'} (μ̂_{B'}^i + u_{B'} + r(B') + D(B', B))
     where u_B = sqrt(2 * A_B / N_B), A_B = 1 + 2 log(2√2 d_r T^{3/2} / δ).
     The min over all balls B' is the similarity-propagation step.
  4. Pareto-select: keep balls whose index vector is not dominated.
  5. Sample (x_t, y_t) uniformly from the union of domains.
  6. Update the chosen ball's mean.
  7. Activation: if u_B ≤ r(B), spawn a new child ball at (x_t, y_t) with
     half the radius.

Differences from the paper
--------------------------
- Sup-norm metric on [0,1]^2 (paper allows any metric; sup-norm makes
  ball intersection trivially fast for our axis-aligned setup).
- Finite arms K mapped to y_k = k / max(K-1, 1).  The continuum-arm
  variant in the paper specializes to this by halving radius until balls
  isolate single arms.
- A ball's "domain" is computed lazily from current ball list rather
  than incrementally maintained; with B balls and K arms this is O(B·K)
  per step which is acceptable for benchmark settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from paretobandits.core.algorithm import Algorithm
from paretobandits.core.preference import Preference


@dataclass
class _Ball:
    """A single ball in joint context-arm space (sup-norm)."""
    cx: float
    cy: float
    r: float
    n_objectives: int
    mean: np.ndarray = field(default_factory=lambda: np.zeros(2))
    M2: np.ndarray = field(default_factory=lambda: np.zeros(2))
    n: int = 0
    sigma: np.ndarray = field(default_factory=lambda: np.ones(2))

    def contains(self, x: float, y: float) -> bool:
        return abs(self.cx - x) <= self.r and abs(self.cy - y) <= self.r

    def distance(self, other: _Ball) -> float:
        """Sup-norm distance between ball centers."""
        return max(abs(self.cx - other.cx), abs(self.cy - other.cy))


class Turgay18(Algorithm):
    """Pareto Contextual Zooming (Turgay, Öner, Tekin 2018).

    Args:
        n_arms, context_dim, n_objectives, preference, delta, horizon, rng:
            Standard Algorithm arguments.
        initial_radius: starting ball radius (default 0.5 — covers [0,1]^2
                        under sup-norm with center (0.5, 0.5)).
    """

    name = "Turgay18"

    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        n_objectives: int,
        preference: Preference,
        delta: float = 0.05,
        horizon: int | None = None,
        initial_radius: float = 0.5,
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
        if context_dim != 1:
            raise NotImplementedError(
                "Turgay18 currently supports context_dim=1 only."
            )
        if horizon is None:
            raise ValueError(
                "Turgay18 requires a horizon T to compute its A_B constant."
            )
        self.initial_radius = float(initial_radius)
        # Map arm id k to y_k = k / max(K-1, 1) in [0, 1].
        self._arm_y = np.linspace(0.0, 1.0, n_arms) if n_arms > 1 else np.array([0.5])
        # Universal A_B constant from the paper (per-step uncertainty multiplier).
        # A_B = 1 + 2 log(2 sqrt(2) * d_r * T^{3/2} / delta).
        T = horizon
        self._A_B = 1.0 + 2.0 * np.log(
            2.0 * np.sqrt(2.0) * self.n_objectives * (T**1.5) / self.delta
        )

        self._balls: list[_Ball] = []
        self._last_chosen: _Ball | None = None
        self._last_xy = (0.0, 0.0)
        self._build()

    def _build(self) -> None:
        # Single root ball covering [0, 1]^2 under sup-norm.
        self._balls = [
            _Ball(
                cx=0.5,
                cy=0.5,
                r=self.initial_radius,
                n_objectives=self.n_objectives,
                mean=np.zeros(self.n_objectives),
                M2=np.zeros(self.n_objectives),
                sigma=np.ones(self.n_objectives),
            )
        ]
        self._last_chosen = None

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._build()

    # ─── Domain ownership: deepest ball containing (x, y) wins ───────

    def _owner(self, x: float, y: float) -> _Ball | None:
        """Return the ball whose domain contains (x, y).

        dom(B) = B \\ ∪_{B' : r(B') < r(B)} B'. So (x, y) ∈ dom(B) iff B
        contains (x, y) and no strictly smaller ball does. Equivalently,
        the deepest ball (smallest radius) containing (x, y).
        """
        best: _Ball | None = None
        best_r = float("inf")
        for B in self._balls:
            if B.contains(x, y) and B.r < best_r:
                best = B
                best_r = B.r
        return best

    # ─── Index computation (similarity propagation) ──────────────────

    def _u(self, B: _Ball) -> float:
        """Sample uncertainty u_B = sqrt(2 A_B / N_B)."""
        if B.n < 1:
            return float("inf")
        return float(np.sqrt(2.0 * self._A_B / B.n))

    def _pre_index(self, B: _Ball) -> np.ndarray:
        """g_B^pre = μ̂_B + u_B + r(B), per objective (M-dim vector)."""
        return B.mean + self._u(B) + B.r

    def _index(self, B: _Ball) -> np.ndarray:
        """g_B^i = r(B) + min_{B'} (g_{B'}^pre^i + D(B', B))."""
        if not self._balls:
            return np.full(self.n_objectives, np.inf)
        pre = np.stack([self._pre_index(Bp) for Bp in self._balls])  # (N, M)
        dists = np.array([Bp.distance(B) for Bp in self._balls])     # (N,)
        return B.r + np.min(pre + dists[:, np.newaxis], axis=0)

    # ─── Algorithm interface ────────────────────────────────────────

    def act(self, context: np.ndarray) -> int:
        x = float(np.atleast_1d(context)[0])

        # 1) Identify owner ball for each arm; collect unique relevant balls.
        owners = [self._owner(x, float(y)) for y in self._arm_y]
        # If for some reason no owner (shouldn't happen with full-coverage
        # root), fall back to root.
        for i, o in enumerate(owners):
            if o is None:
                owners[i] = self._balls[0]
        relevant_balls = list({id(o): o for o in owners}.values())

        # 2) Index vector per relevant ball.
        indices = np.stack([self._index(B) for B in relevant_balls])  # (R, M)

        # 3) Pareto front of indices under the active preference.
        mask = self.preference.pareto_set(indices)
        pareto_balls = [relevant_balls[i] for i in range(len(relevant_balls)) if mask[i]]

        # 4) Among arms whose owner is in the Pareto-front, sample uniformly.
        candidate_arms = [
            k for k in range(self.n_arms)
            if id(owners[k]) in {id(B) for B in pareto_balls}
        ]
        if not candidate_arms:
            # Numerical fallback: pick any arm.
            candidate_arms = list(range(self.n_arms))
        action = int(self.rng.choice(candidate_arms))
        self._last_chosen = owners[action]
        self._last_xy = (x, float(self._arm_y[action]))
        return action

    def update(
        self, context: np.ndarray, action: int, reward: np.ndarray
    ) -> None:
        if self._last_chosen is None:
            x = float(np.atleast_1d(context)[0])
            y = float(self._arm_y[action])
            self._last_chosen = self._owner(x, y) or self._balls[0]
            self._last_xy = (x, y)
        B = self._last_chosen
        B.n += 1
        # Welford online mean + variance.
        d = reward - B.mean
        B.mean = B.mean + d / B.n
        B.M2 = B.M2 + d * (reward - B.mean)
        if B.n >= 2:
            B.sigma = np.sqrt(np.maximum(B.M2 / (B.n - 1), 1e-6))

        # Activation: u_B <= r(B)  → spawn a child ball at the played point.
        if self._u(B) <= B.r:
            x_t, y_t = self._last_xy
            child = _Ball(
                cx=x_t,
                cy=y_t,
                r=B.r / 2.0,
                n_objectives=self.n_objectives,
                mean=B.mean.copy(),
                M2=np.zeros(self.n_objectives),
                sigma=B.sigma.copy(),
            )
            # The child gets a fresh play count so the activation rule
            # doesn't immediately fire again with a stale n.
            self._balls.append(child)

        self._t += 1

    def pareto_estimate(self, context: np.ndarray) -> set[int]:
        x = float(np.atleast_1d(context)[0])
        owners = [self._owner(x, float(y)) for y in self._arm_y]
        for i, o in enumerate(owners):
            if o is None:
                owners[i] = self._balls[0]
        relevant_balls = list({id(o): o for o in owners}.values())
        indices = np.stack([self._index(B) for B in relevant_balls])
        mask = self.preference.pareto_set(indices)
        pareto_set = {id(relevant_balls[i]) for i in range(len(relevant_balls)) if mask[i]}
        return {k for k in range(self.n_arms) if id(owners[k]) in pareto_set}

    @property
    def n_balls(self) -> int:
        """For diagnostics: how many balls has the algorithm allocated?"""
        return len(self._balls)
