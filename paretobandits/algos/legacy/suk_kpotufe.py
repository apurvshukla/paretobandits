"""SukKpotufe20: self-tuning bandits over unknown covariate shifts.

Faithful port of the multi-objective adaptation of:
  Suk, J. and Kpotufe, S. (2021). Self-Tuning Bandits over Unknown
  Covariate-Shifts. In ALT 2021.  arXiv:2007.08584.

The original paper considers scalar rewards. This port matches the
adaptation already in the experiments code (`script/classes.py:SK_MO`):
the Pareto-based elimination replaces scalar successive elimination,
using the same pairwise-CI test as PCBShift's `pairwise` mode.

Mechanism
---------
  - Dyadic partition tree over [0,1] with widths r ∈ {1, 1/2, 1/4, ...}.
  - Adaptive level selection: choose the smallest h with
        n_h(x_t) ≥ 8K log(KM/δ) · 2^{2h}
  - Pareto elimination per bin: drop arm i if some j satisfies
        f̂_j - f̂_i  >  β_{i,j}    componentwise across all M objectives,
    where β_{i,j} = σ_pool · sqrt(2 log(KM/δ) · (1/n_i + 1/n_j)).
  - Play uniformly at random from surviving candidates.
  - Self-tuning: no knowledge of shift time or magnitude needed.

What it doesn't have (relative to PCBShift)
------------------------------------------
  - No optimistic-Pareto second-phase elimination.
  - No tree splitting on uncertainty — the tree is a fixed dyadic grid.
  - Elimination cascades through ancestors (an arm dropped at level h
    stays dropped at deeper levels), but estimates aren't shared via
    Lipschitz cross-bin tightening.

These are exactly the ablations the paper expects: PCBShift's wins over
SukKpotufe20 isolate the contributions of (a) preference-based metric,
(b) optimistic Pareto elimination, (c) Lipschitz-aware splitting.
"""

from __future__ import annotations

import numpy as np

from paretobandits.core.algorithm import Algorithm
from paretobandits.core.preference import Preference


class SukKpotufe20(Algorithm):
    """Suk-Kpotufe self-tuning multi-objective contextual bandit.

    Args:
        n_arms, context_dim, n_objectives, preference, delta, horizon, rng:
            Standard Algorithm arguments.
        max_depth: maximum dyadic depth (default 8 — covers T up to ~100k).
        explore_factor: multiplier on the warm-up phase length. The total
            warm-up plays are explore_factor * K * log(KM/δ). Default 8,
            matching the paper.
        sigma_floor: minimum sigma estimate; numerical safety.
    """

    name = "SukKpotufe20"

    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        n_objectives: int,
        preference: Preference,
        delta: float = 0.05,
        horizon: int | None = None,
        max_depth: int = 8,
        explore_factor: float = 8.0,
        sigma_floor: float = 1e-3,
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
                "SukKpotufe20 currently supports context_dim=1 only "
                "(matches the original paper's scalar context setting)."
            )
        self.max_depth = int(max_depth)
        self.explore_factor = float(explore_factor)
        self.sigma_floor = sigma_floor

        # Cached log term used for the level selector and elimination.
        self._log_term = np.log(
            max(self.n_arms * self.n_objectives / self.delta, 2.0)
        )
        self.explore_rounds = int(
            np.ceil(self.explore_factor * self.n_arms * self._log_term)
        )

        # Per-bin data: keyed by (level, bin_idx).
        self._bin_data: dict = {}
        self._last_level = 0
        self._last_action = 0
        self._last_context = 0.0

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._bin_data = {}
        self._last_level = 0
        self._last_action = 0

    # ─── Bin bookkeeping ────────────────────────────────────────────

    def _bin_idx(self, x: float, level: int) -> int:
        num_bins = 2**level
        idx = int(x * num_bins)
        return min(max(idx, 0), num_bins - 1)

    def _bin(self, level: int, bin_idx: int) -> dict:
        key = (level, bin_idx)
        bd = self._bin_data.get(key)
        if bd is None:
            bd = {
                "n_contexts": 0,
                "candidates": set(range(self.n_arms)),
                "arm": {
                    k: {
                        "mean": np.zeros(self.n_objectives),
                        "M2": np.zeros(self.n_objectives),
                        "n": 0,
                        "sigma": np.ones(self.n_objectives),
                    }
                    for k in range(self.n_arms)
                },
            }
            self._bin_data[key] = bd
        return bd

    def _select_level(self, x: float) -> int:
        """Smallest h such that the bin at level h has enough samples.

            n_h(x) ≥ 8K log(KM/δ) · 2^{2h}    ⟺    selectable.
        """
        threshold_const = self.explore_factor * self.n_arms * self._log_term
        best = 0
        for h in range(self.max_depth + 1):
            bd = self._bin(h, self._bin_idx(x, h))
            r_h = 2.0 ** (-h)
            threshold = threshold_const / (r_h**2) if r_h > 0 else float("inf")
            if bd["n_contexts"] >= threshold:
                best = h
            else:
                break
        return best

    # ─── Pareto-aware elimination (the multi-objective bit) ─────────

    def _eliminate(self, candidates: set[int], arm_est: dict) -> set[int]:
        cands = list(candidates)
        nc = len(cands)
        if nc <= 1:
            return candidates
        means = np.stack([arm_est[k]["mean"] for k in cands])
        sigmas = np.stack([arm_est[k]["sigma"] for k in cands])
        counts = np.array([arm_est[k]["n"] for k in cands])
        if (counts >= 2).sum() <= 1:
            return candidates

        # diff[i, j, :] = means[i] - means[j].
        diff = means[:, np.newaxis, :] - means[np.newaxis, :, :]
        sig_pool = np.maximum(
            sigmas[:, np.newaxis, :], sigmas[np.newaxis, :, :]
        )
        inv_n = 1.0 / np.maximum(counts, 1)
        inv_sum = inv_n[:, np.newaxis] + inv_n[np.newaxis, :]
        beta = sig_pool * np.sqrt(2 * self._log_term * inv_sum[:, :, np.newaxis])

        # dominates[i, j] = "i dominates j" by margin beta in all objectives.
        dominates = np.all(diff > beta, axis=2)
        np.fill_diagonal(dominates, False)
        insufficient = counts < 2
        dominates[insufficient, :] = False
        dominates[:, insufficient] = False
        # j is eliminated if any i dominates it.
        is_dominated = np.any(dominates, axis=0)
        return {cands[i] for i in range(nc) if not is_dominated[i]}

    # ─── Algorithm interface ────────────────────────────────────────

    def act(self, context: np.ndarray) -> int:
        x = float(np.atleast_1d(context)[0])
        self._last_context = x

        # Register the context up to (current best level + 2).
        max_reg = min(self._last_level + 2, self.max_depth + 1)
        for h in range(max_reg):
            self._bin(h, self._bin_idx(x, h))["n_contexts"] += 1

        # Round-robin warm-up.
        if self._t < self.explore_rounds:
            action = self._t % self.n_arms
            self._last_action = action
            return action

        level = self._select_level(x)
        self._last_level = level
        bd = self._bin(level, self._bin_idx(x, level))

        # Inherit eliminations from ancestors.
        for h in range(level):
            anc = self._bin(h, self._bin_idx(x, h))
            bd["candidates"] &= anc["candidates"]
            if not bd["candidates"]:
                bd["candidates"] = set(range(self.n_arms))

        # Eliminate using bin-local estimates.
        bd["candidates"] = self._eliminate(bd["candidates"], bd["arm"])
        cands = sorted(bd["candidates"])
        action = int(cands[int(self.rng.integers(0, len(cands)))])
        self._last_action = action
        return action

    def update(
        self, context: np.ndarray, action: int, reward: np.ndarray
    ) -> None:
        x = float(np.atleast_1d(context)[0])
        # Update estimates at the relevant levels.
        max_lvl = min(self._last_level + 2, self.max_depth + 1)
        for h in range(max_lvl):
            bd = self._bin(h, self._bin_idx(x, h))
            est = bd["arm"][action]
            est["n"] += 1
            n = est["n"]
            delta = reward - est["mean"]
            est["mean"] = est["mean"] + delta / n
            est["M2"] = est["M2"] + delta * (reward - est["mean"])
            if n >= 2:
                est["sigma"] = np.sqrt(
                    np.maximum(est["M2"] / (n - 1), self.sigma_floor**2)
                )
        self._t += 1

    def pareto_estimate(self, context: np.ndarray) -> set[int]:
        x = float(np.atleast_1d(context)[0])
        # Use the deepest bin that has enough data; otherwise fall back
        # to all arms during warmup.
        if self._t < self.explore_rounds:
            return set(range(self.n_arms))
        level = self._select_level(x)
        bd = self._bin(level, self._bin_idx(x, level))
        cands = list(bd["candidates"])
        if not cands:
            return set(range(self.n_arms))
        # Pareto front *of the surviving candidates* by their estimated means.
        means = np.stack([bd["arm"][k]["mean"] for k in cands])
        # Skip arms that have never been played in this bin.
        played = np.array([bd["arm"][k]["n"] >= 1 for k in cands])
        if played.sum() == 0:
            return set(cands)
        means = means[played]
        active_cands = [c for c, ok in zip(cands, played) if ok]
        mask = self.preference.pareto_set(means)
        result = {active_cands[i] for i in range(len(active_cands)) if mask[i]}
        return result if result else set(active_cands)
