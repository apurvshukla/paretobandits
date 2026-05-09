"""Cai24: Transfer learning for contextual multi-armed bandits (simplified).

Adaptation (with caveats) of:
  Cai, C., Cai, T. T., and Li, H. (2024). "Transfer learning for
  contextual multi-armed bandits." The Annals of Statistics, 52(1).

Setting in the paper
--------------------
**Single-objective contextual MABs with transfer learning.** Two related
tasks: a *source* task with abundant historical data and a *target* task
where we collect data online. Mean reward functions are "close" between
source and target; the algorithm uses source data to warm-start target
learning. The paper assumes:
  - One reward objective (scalar).
  - Linear contextual structure (LinUCB-style).
  - **A given source dataset** is available offline before the target run.
  - **Stationary** within each task — no shift during the target task.

What this port does in our framework
-------------------------------------
The benchmark expects multi-objective + streaming + (sometimes) shift.
We translate Cai24's "use source data to inform target" idea into a
streaming setting:

  - **Recent vs all-time estimates**: maintain two empirical-mean
    estimates per arm-objective — one weighted toward the last `window`
    samples, one over all samples.
  - **Transferred prior**: the all-time estimate plays the role of
    "source data" (historical), the recent estimate plays "target".
  - **Shrinkage UCB**: use a convex combination of the two estimates,
    shrinking toward the recent one as more recent data accumulates.
    Pareto-front selection over the resulting per-arm vectors.
  - **No contextual structure**: we drop the linear-context assumption
    (the paper's machinery doesn't trivially generalize to vector rewards).

What this port DOESN'T do, and why
----------------------------------
- **The paper's actual transfer mechanism** is a regularizer in a LinUCB
  optimization that depends on having two distinct datasets (source and
  target). In a single streaming run there's no clear separation, so we
  approximate via the "old vs recent" decomposition.
- **Multi-objective generalization**: the paper is scalar; we run their
  shrinkage logic per objective independently, then take Pareto fronts.
- **Distribution shift**: not designed for. The all-time estimate becomes
  stale post-shift, biasing the shrinkage toward the wrong values until
  the recent window dominates. Expected to perform poorly on shift configs.

This is the "v0.8 simplified port" — a faithful LinUCB-with-transfer
version is on the v0.9 roadmap and would require adding a linear-model
infrastructure to the library.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from paretobandits.core.algorithm import Algorithm
from paretobandits.core.preference import Preference


class Cai24(Algorithm):
    """Cai-Cai-Li (2024) transfer learning bandit, simplified port.

    Args:
        n_arms, context_dim, n_objectives, preference, delta, horizon, rng:
            Standard. context_dim is accepted but ignored.
        window_size: per-arm recent-history window (the "target" set).
                     Default 200.
        transfer_strength: λ in the shrinkage formula
            μ̂_used = (1 - λ_t) · μ̂_recent + λ_t · μ̂_all_time
            where λ_t = transfer_strength / sqrt(n_recent + 1). Larger →
            more reliance on the all-time (source) estimate. Default 1.0.
    """

    name = "Cai24"

    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        n_objectives: int,
        preference: Preference,
        delta: float = 0.05,
        horizon: int | None = None,
        window_size: int = 200,
        transfer_strength: float = 1.0,
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
        self.window_size = int(window_size)
        self.transfer_strength = float(transfer_strength)
        # All-time per-arm running statistics.
        self._mean_all = np.zeros((n_arms, n_objectives))
        self._n_all = np.zeros(n_arms, dtype=int)
        # Per-arm rolling reward window.
        self._windows: list[deque] = [
            deque(maxlen=self.window_size) for _ in range(n_arms)
        ]

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._mean_all = np.zeros((self.n_arms, self.n_objectives))
        self._n_all = np.zeros(self.n_arms, dtype=int)
        self._windows = [
            deque(maxlen=self.window_size) for _ in range(self.n_arms)
        ]

    def _recent_mean(self, k: int) -> np.ndarray:
        w = self._windows[k]
        if len(w) == 0:
            return np.zeros(self.n_objectives)
        return np.mean(np.stack(list(w)), axis=0)

    def _shrunk_mean_and_ucb(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        n_recent = len(self._windows[k])
        n_all = self._n_all[k]
        if n_all == 0:
            return np.zeros(self.n_objectives), np.full(self.n_objectives, np.inf)
        mu_recent = self._recent_mean(k) if n_recent > 0 else self._mean_all[k]
        mu_all = self._mean_all[k]
        # Shrinkage weight: more weight on all-time when recent is small.
        lam = self.transfer_strength / np.sqrt(n_recent + 1)
        lam = float(np.clip(lam, 0.0, 1.0))
        mu = (1.0 - lam) * mu_recent + lam * mu_all
        # CI based on recent-window count (more honest given streaming).
        log_term = np.log(max(self.n_arms * self.n_objectives / self.delta, 2.0))
        cr = np.sqrt(2 * log_term / max(n_recent, 1))
        return mu, np.full(self.n_objectives, cr)

    def act(self, context: np.ndarray) -> int:
        # Per-arm warmup.
        if self._t < self.n_arms:
            return int(self._t % self.n_arms)
        # Compute UCB vectors via shrinkage.
        ucb = np.zeros((self.n_arms, self.n_objectives))
        for k in range(self.n_arms):
            mu, cr = self._shrunk_mean_and_ucb(k)
            ucb[k] = mu + cr
        mask = self.preference.pareto_set(ucb)
        candidates = np.where(mask)[0]
        if candidates.size == 0:
            candidates = np.arange(self.n_arms)
        # Tie-break to least-pulled (recent window) for under-sampled arms.
        recent_counts = np.array([len(self._windows[k]) for k in candidates])
        return int(candidates[int(np.argmin(recent_counts))])

    def update(
        self, context: np.ndarray, action: int, reward: np.ndarray
    ) -> None:
        # All-time update.
        self._n_all[action] += 1
        n = self._n_all[action]
        d = reward - self._mean_all[action]
        self._mean_all[action] = self._mean_all[action] + d / n
        # Recent window update.
        self._windows[action].append(reward.copy())
        self._t += 1

    def pareto_estimate(self, context: np.ndarray) -> set[int]:
        if self._t < self.n_arms:
            return set(range(self.n_arms))
        means = np.zeros((self.n_arms, self.n_objectives))
        for k in range(self.n_arms):
            mu, _ = self._shrunk_mean_and_ucb(k)
            means[k] = mu
        mask = self.preference.pareto_set(means)
        result = {int(i) for i in np.where(mask)[0]}
        return result if result else set(range(self.n_arms))
