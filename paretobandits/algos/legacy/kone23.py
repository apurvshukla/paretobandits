"""Kone23: Adaptive algorithms for relaxed Pareto set identification.

Faithful (with caveats) port of:
  Kone, C., Kaufmann, E., and Richert, L. (2023). "Adaptive algorithms
  for relaxed Pareto set identification." NeurIPS 2023.

Setting in the paper
--------------------
**Pareto set identification (PSI), not regret minimization.** The original
goal is "identify the ε-relaxed Pareto-optimal set with high probability,
in as few samples as possible." Stochastic K-arm bandit, M-objective
rewards, **no contexts**, **stationary** (no distribution shift). The
algorithm has a stopping rule and a sample-allocation rule designed to
minimize expected sample complexity, not cumulative regret.

What this port does in our framework
-------------------------------------
The benchmark expects algorithms that act for `horizon` steps and produce
per-step Pareto estimates. We adapt:

  - **Active-set tracking** (faithful): UCB/LCB intervals per arm, with an
    arm declared "outside the Pareto set" when its UCB is dominated by
    some LCB-Pareto element, and "inside" when its LCB is not dominated
    by any UCB.
  - **Sampling rule** (faithful): pull the most-uncertain still-undecided
    arm — i.e. the arm with maximum CI-half-width along any objective.
  - **Stopping** (added): once no arm is still undecided, switch to
    uniform play over the identified Pareto set. The remaining horizon
    is "stationary play"; cumulative regret continues to accrue but the
    algorithm has finished its identification phase.

What this port DOESN'T do, and why
----------------------------------
- **Distribution shift**: the algorithm is designed for stationary
  rewards. On shift configs, the identified Pareto set becomes wrong
  post-shift and the algorithm has no way to detect this. Expected to
  perform poorly on shift configs — that's informative for the benchmark.
- **Contextual rewards**: the algorithm is context-free; we ignore the
  context. Expected to perform poorly on configs where the optimal arm
  varies by context.
- **The paper's exact ε-relaxation**: we use ε=0 (strict Pareto). Adding
  the relaxation is a one-line change but doesn't matter for benchmark
  rankings.

Including this baseline is informative even though it's outside its
design regime: it shows that PSI algorithms don't transfer to streaming
regret-minimization-with-shift settings, motivating PCBShift.
"""

from __future__ import annotations

import numpy as np

from paretobandits.core.algorithm import Algorithm
from paretobandits.core.preference import Preference


class Kone23(Algorithm):
    """Kone-Kaufmann-Richert (2023) adaptive Pareto set identification.

    Args:
        n_arms, context_dim, n_objectives, preference, delta, horizon, rng:
            Standard. context_dim is accepted but ignored.
        c_const: log-term constant in the CI (default 4).
    """

    name = "Kone23"

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
        self._mean = np.zeros((n_arms, n_objectives))
        self._M2 = np.zeros((n_arms, n_objectives))
        self._n = np.zeros(n_arms, dtype=int)
        self._sigma = np.ones((n_arms, n_objectives))
        # Arm classification: "in" / "out" / "undecided" (latter = not yet classified).
        self._classified_in: set[int] = set()
        self._classified_out: set[int] = set()

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._mean = np.zeros((self.n_arms, self.n_objectives))
        self._M2 = np.zeros((self.n_arms, self.n_objectives))
        self._n = np.zeros(self.n_arms, dtype=int)
        self._sigma = np.ones((self.n_arms, self.n_objectives))
        self._classified_in = set()
        self._classified_out = set()

    def _ci(self, t: int) -> np.ndarray:
        """Per-arm CI half-width vector. Same form as Auer16."""
        log_term = np.log(
            max(self.c_const * self.n_arms * self.n_objectives * (t**2) / self.delta, 2.0)
        )
        n = np.maximum(self._n, 1).astype(float)
        return self._sigma * np.sqrt(log_term / n[:, np.newaxis])

    def _refresh_classification(self) -> None:
        """Update classified_in / classified_out / undecided sets."""
        if self._t < self.n_arms:
            return
        ci = self._ci(self._t + 1)
        ucb = self._mean + ci
        lcb = self._mean - ci

        # i is "in" the Pareto set if its LCB is not strictly dominated by any UCB.
        # i is "out" if its UCB is strictly dominated by some LCB.
        # Otherwise undecided.
        for i in range(self.n_arms):
            if i in self._classified_in or i in self._classified_out:
                continue
            ucb_i = ucb[i]
            lcb_i = lcb[i]
            is_out = False
            for j in range(self.n_arms):
                if j == i:
                    continue
                if self.preference.dominates(lcb[j], ucb_i):
                    is_out = True
                    break
            if is_out:
                self._classified_out.add(i)
                continue
            is_in = True
            for j in range(self.n_arms):
                if j == i:
                    continue
                if self.preference.dominates(ucb[j], lcb_i):
                    is_in = False
                    break
            if is_in:
                self._classified_in.add(i)

    @property
    def _undecided(self) -> set[int]:
        return set(range(self.n_arms)) - self._classified_in - self._classified_out

    def act(self, context: np.ndarray) -> int:
        # Per-arm warmup.
        if self._t < self.n_arms:
            return int(self._t % self.n_arms)
        self._refresh_classification()
        undecided = self._undecided

        if undecided:
            # Sampling rule: pick the arm with the largest CI half-width
            # along any objective among undecided arms.
            ci = self._ci(self._t + 1)
            cand = sorted(undecided)
            widths = np.max(ci[cand], axis=1)
            return int(cand[int(np.argmax(widths))])

        # Identification phase ended — uniform play over identified Pareto.
        if self._classified_in:
            return int(self.rng.choice(sorted(self._classified_in)))
        # Fallback (shouldn't happen with reasonable data).
        return int(self.rng.integers(0, self.n_arms))

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
        # Estimated Pareto set = classified_in ∪ undecided that are
        # currently on the empirical Pareto front (best-effort while
        # identification is in progress).
        if self._classified_in and not self._undecided:
            return set(self._classified_in)
        # Mid-identification: report empirical Pareto from non-out arms.
        live = sorted(set(range(self.n_arms)) - self._classified_out)
        if not live:
            live = list(range(self.n_arms))
        means = self._mean[live]
        mask = self.preference.pareto_set(means)
        result = {live[i] for i in range(len(live)) if mask[i]}
        return result if result else set(live)
