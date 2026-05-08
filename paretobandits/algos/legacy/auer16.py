"""Auer16: Pareto front identification from stochastic bandit feedback.

Faithful port of:
  Auer, P., Chiang, C.-K., Ortner, R., and Drugan, M. (2016). "Pareto
  front identification from stochastic bandit feedback." In AISTATS.

Setting and caveats
-------------------
This is the **context-free** baseline. Auer et al. consider K stochastic
arms with M-objective rewards; no contexts, no covariate shift. Wrapping
it in this library's contextual API simply ignores the context — it is
included as a baseline because the paper is widely cited in the
multi-objective bandit literature, and any benchmark missing it would
look like an oversight.

Algorithm sketch (Algorithm 1 in the paper)
-------------------------------------------
Maintain:
  A_t : currently active arms (start with all K).
  P_t : confirmed Pareto-optimal arms.
  N_t : number of pulls of each arm.
  μ̂_t : empirical mean per arm (per objective).

At each step:
  1. For all i ∈ A_t, compute UCB_i = μ̂_i + β_i,  LCB_i = μ̂_i - β_i,
     where β_i = sqrt(log(c·K·M·t² / δ) / N_i).
  2. Remove from A_t any arm i for which there exists j ∈ A_t with
     LCB_j[d] ≥ UCB_i[d] for ALL d  (j strictly dominates i in CI sense).
  3. Promote to P_t any arm i ∈ A_t such that LCB_i is not weakly
     dominated by UCB_j of any other arm — i.e., i is "confirmed Pareto".
  4. Sample one arm uniformly at random from A_t and pull it.

Differences from the paper
--------------------------
- The paper's exact form has a two-phase structure: identification +
  termination. This implementation runs the identification phase
  indefinitely (no early stopping); the regret framework expects T
  steps regardless.
- The cone-aware Pareto checks defer to the active `Preference`,
  matching the paper's positive-orthant case when used with the
  default `PositiveOrthant`.
"""

from __future__ import annotations

import numpy as np

from paretobandits.core.algorithm import Algorithm
from paretobandits.core.preference import Preference


class Auer16(Algorithm):
    """Auer-Chiang-Ortner-Drugan Pareto front identification (2016).

    Args:
        n_arms, context_dim, n_objectives, preference, delta, horizon, rng:
            Standard Algorithm arguments. context_dim is accepted but
            ignored — the algorithm is context-free.
        c_const: log-term constant (paper uses c=4); larger → more
                 conservative elimination.
    """

    name = "Auer16"

    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        n_objectives: int,
        preference: Preference,
        delta: float = 0.05,
        horizon: int | None = None,
        c_const: float = 4.0,
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
        self.c_const = float(c_const)
        # Per-arm running statistics — context-free, so no per-bin tables.
        self._mean = np.zeros((n_arms, n_objectives))
        self._M2 = np.zeros((n_arms, n_objectives))
        self._n = np.zeros(n_arms, dtype=int)
        self._sigma = np.ones((n_arms, n_objectives))
        self._active: set[int] = set(range(n_arms))
        self._confirmed: set[int] = set()

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._mean = np.zeros((self.n_arms, self.n_objectives))
        self._M2 = np.zeros((self.n_arms, self.n_objectives))
        self._n = np.zeros(self.n_arms, dtype=int)
        self._sigma = np.ones((self.n_arms, self.n_objectives))
        self._active = set(range(self.n_arms))
        self._confirmed = set()

    # ─── Confidence bounds and elimination ──────────────────────────

    def _beta(self, t: int) -> np.ndarray:
        """Per-arm CI half-width (vector of length K, broadcast over M).

        β_i(t) = σ̂_i * sqrt( log(c·K·M·t² / δ) / N_i )
        """
        log_term = np.log(
            max(self.c_const * self.n_arms * self.n_objectives * (t**2) / self.delta, 2.0)
        )
        n = np.maximum(self._n, 1).astype(float)
        return self._sigma * np.sqrt(log_term / n[:, np.newaxis])

    def _refresh_active_and_confirmed(self) -> None:
        """Recompute A_t and P_t from current statistics."""
        if self._t < self.n_arms:
            return  # still in the per-arm-once warmup
        active = list(self._active)
        if len(active) <= 1:
            return
        beta = self._beta(self._t + 1)             # (K, M)
        ucb = self._mean + beta
        lcb = self._mean - beta

        # Eliminate strictly-dominated arms (LCB[j] >= UCB[i] all dims).
        new_active: set[int] = set()
        for i in active:
            dominated = False
            for j in active:
                if j == i:
                    continue
                if self.preference.dominates(lcb[j], ucb[i]):
                    dominated = True
                    break
            if not dominated:
                new_active.add(i)
        if not new_active:
            new_active = set(active)
        self._active = new_active

        # Promote confirmed Pareto arms: i is confirmed if its LCB is not
        # dominated by ANY other arm's UCB.  Once confirmed, an arm stays.
        for i in self._active:
            if i in self._confirmed:
                continue
            ok = True
            for j in self._active:
                if j == i:
                    continue
                if self.preference.dominates(ucb[j], lcb[i]):
                    ok = False
                    break
            if ok:
                self._confirmed.add(i)

    # ─── Algorithm interface ────────────────────────────────────────

    def act(self, context: np.ndarray) -> int:
        # Per-arm warmup: pull each arm once.
        if self._t < self.n_arms:
            return int(self._t % self.n_arms)
        self._refresh_active_and_confirmed()
        candidates = sorted(self._active)
        if not candidates:
            candidates = list(range(self.n_arms))
        # Round-robin within the active set ensures all surviving arms
        # gather data evenly.
        action = candidates[self._t % len(candidates)]
        return int(action)

    def update(
        self, context: np.ndarray, action: int, reward: np.ndarray
    ) -> None:
        self._n[action] += 1
        n = self._n[action]
        d = reward - self._mean[action]
        self._mean[action] = self._mean[action] + d / n
        self._M2[action] = self._M2[action] + d * (reward - self._mean[action])
        if n >= 2:
            self._sigma[action] = np.sqrt(
                np.maximum(self._M2[action] / (n - 1), 1e-6)
            )
        self._t += 1

    def pareto_estimate(self, context: np.ndarray) -> set[int]:
        if self._t < self.n_arms:
            return set(range(self.n_arms))
        # Estimated Pareto = confirmed-Pareto arms ∪ active arms whose
        # empirical mean is on the Pareto front of all active means.
        if self._confirmed:
            base = set(self._confirmed)
        else:
            base = set()
        active = sorted(self._active) if self._active else list(range(self.n_arms))
        means = self._mean[active]
        mask = self.preference.pareto_set(means)
        front = {active[i] for i in range(len(active)) if mask[i]}
        result = base | front
        return result if result else set(active)
