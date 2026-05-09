"""AnnealingPareto: temperature-annealed multi-objective bandit.

Faithful port of:
  Yahyaa, S. Q., Drugan, M. M., and Manderick, B. (2014). "Annealing-
  Pareto multi-objective multi-armed bandit algorithm." ADPRL 2014.

Setting: K stochastic arms, M-objective rewards, no contexts. A softmax
over Pareto-optimality scores with a time-decaying temperature, so early
exploration anneals to greedy Pareto-front exploitation.

Mechanism per step
~~~~~~~~~~~~~~~~~~
  1. For each arm, compute Pareto-front membership of its empirical means
     (1 if on the empirical Pareto front, 0 otherwise).
  2. Softmax over (membership / temperature) gives a sampling distribution.
  3. Temperature τ_t = c / log(t + 1) — decreases as t grows.
  4. Sample one arm from the softmax distribution.

Differences from the paper
--------------------------
- The paper proposes several variants; this is the basic Annealing-Pareto
  with the standard 1/log(t) annealing schedule.
- Membership uses the active `Preference` so non-orthant cones work.
"""

from __future__ import annotations

import numpy as np

from paretobandits.core.algorithm import Algorithm
from paretobandits.core.preference import Preference


class AnnealingPareto(Algorithm):
    """Yahyaa et al. (2014) Annealing-Pareto multi-objective bandit.

    Args:
        n_arms, context_dim, n_objectives, preference, delta, horizon, rng:
            Standard Algorithm arguments. context_dim is accepted but
            ignored — context-free.
        temperature_const: c in τ_t = c / log(t+1) (default 1.0).
    """

    name = "AnnealingPareto"

    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        n_objectives: int,
        preference: Preference,
        delta: float = 0.05,
        horizon: int | None = None,
        temperature_const: float = 1.0,
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
        self.temperature_const = float(temperature_const)
        self._mean = np.zeros((n_arms, n_objectives))
        self._n = np.zeros(n_arms, dtype=int)

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._mean = np.zeros((self.n_arms, self.n_objectives))
        self._n = np.zeros(self.n_arms, dtype=int)

    def _temperature(self) -> float:
        # τ_t = c / log(t + 2), so τ → 0 as t → ∞.
        return self.temperature_const / np.log(self._t + 2)

    def act(self, context: np.ndarray) -> int:
        # Warmup: pull each arm once.
        if self._t < self.n_arms:
            return int(self._t % self.n_arms)
        # Score = 1 if on empirical Pareto front, 0 otherwise.
        mask = self.preference.pareto_set(self._mean).astype(float)
        if not mask.any():
            mask = np.ones(self.n_arms)
        # Annealed softmax.
        tau = max(self._temperature(), 1e-6)
        logits = mask / tau
        # Stabilize.
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        return int(self.rng.choice(self.n_arms, p=probs))

    def update(
        self, context: np.ndarray, action: int, reward: np.ndarray
    ) -> None:
        self._n[action] += 1
        n = self._n[action]
        self._mean[action] = self._mean[action] + (reward - self._mean[action]) / n
        self._t += 1

    def pareto_estimate(self, context: np.ndarray) -> set[int]:
        if self._t < self.n_arms:
            return set(range(self.n_arms))
        mask = self.preference.pareto_set(self._mean)
        result = {int(i) for i in np.where(mask)[0]}
        return result if result else set(range(self.n_arms))
