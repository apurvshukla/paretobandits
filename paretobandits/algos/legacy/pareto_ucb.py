"""ParetoUCB1: multi-objective UCB with Pareto-optimal candidate set.

Faithful port of:
  Drugan, M. M. and Nowe, A. (2013). "Designing multi-objective
  multi-armed bandits algorithms: A study." In IJCNN.

Setting: K stochastic arms, M-objective rewards, no contexts. The
algorithm maintains UCB indices per arm per objective and at each step
picks uniformly at random from the *Pareto-optimal set of UCB vectors*.
This is the canonical multi-objective UCB baseline; missing it from a
multi-objective bandit benchmark would be an oversight.

Mechanism per step
~~~~~~~~~~~~~~~~~~
  1. Pull each arm once (warmup).
  2. For each arm i ∈ [K], compute UCB_i = μ̂_i + sqrt(2 log(t · M^{1/4}) / N_i).
  3. Find the Pareto front of the UCB vectors under the active preference.
  4. Pick uniformly at random from the Pareto front.

Differences from the paper
--------------------------
- The paper uses M=2 in its experiments; the implementation here works
  for arbitrary M via the active `Preference`.
- Confidence radius uses `sqrt(2 log(t · M^{1/4}) / N_i)` per the paper's
  Theorem 1; some implementations use a slightly different log term.
"""

from __future__ import annotations

import numpy as np

from paretobandits.core.algorithm import Algorithm
from paretobandits.core.preference import Preference


class ParetoUCB(Algorithm):
    """Drugan & Nowe (2013) multi-objective UCB.

    Args:
        n_arms, context_dim, n_objectives, preference, delta, horizon, rng:
            Standard Algorithm arguments. context_dim is accepted but
            ignored — the algorithm is context-free.
    """

    name = "ParetoUCB"

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
        self._mean = np.zeros((n_arms, n_objectives))
        self._n = np.zeros(n_arms, dtype=int)

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._mean = np.zeros((self.n_arms, self.n_objectives))
        self._n = np.zeros(self.n_arms, dtype=int)

    def _ucb(self) -> np.ndarray:
        """UCB vector per arm. Same scalar bonus broadcast over M dims."""
        t = max(self._t + 1, 2)
        log_term = np.log(t * (self.n_objectives ** 0.25))
        n = np.maximum(self._n, 1).astype(float)
        bonus = np.sqrt(2 * log_term / n)
        return self._mean + bonus[:, np.newaxis]

    def act(self, context: np.ndarray) -> int:
        # Warmup: pull each arm once.
        if self._t < self.n_arms:
            return int(self._t % self.n_arms)
        ucb = self._ucb()
        mask = self.preference.pareto_set(ucb)
        candidates = np.where(mask)[0]
        if candidates.size == 0:
            candidates = np.arange(self.n_arms)
        return int(self.rng.choice(candidates))

    def update(
        self, context: np.ndarray, action: int, reward: np.ndarray
    ) -> None:
        self._n[action] += 1
        n = self._n[action]
        # Welford mean update.
        self._mean[action] = self._mean[action] + (reward - self._mean[action]) / n
        self._t += 1

    def pareto_estimate(self, context: np.ndarray) -> set[int]:
        if self._t < self.n_arms:
            return set(range(self.n_arms))
        ucb = self._ucb()
        mask = self.preference.pareto_set(ucb)
        result = {int(i) for i in np.where(mask)[0]}
        return result if result else set(range(self.n_arms))
