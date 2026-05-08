"""PCBShift: preference-based contextual bandit under distribution shift.

Implements Algorithm 1 from Shukla & Kumar (2024), "Vector preference-based
contextual bandits under distributional shifts."

Mechanism (one sentence): adaptive dyadic discretization of the context
space + per-bin pairwise-CI elimination of dominated arms + uniform play
from the current Pareto candidates, with bin splits triggered when
statistical uncertainty drops below the bin's Lipschitz bias.

The algorithm self-tunes to:
  - the change point t_p (no input)
  - the margin parameter α (no input)
  - the source-target dissimilarity ρ (no input)
  - the context dimension d (no input — built into bin geometry)

Inputs (`__init__`):
  n_arms, context_dim, n_objectives, preference, delta, horizon
  beta:           Hölder exponent of the reward function (default 1.0).
  lipschitz_L:    Hölder constant C_β (default 4.0).
  sigma_floor:    minimum sigma estimate to avoid divide-by-zero.
  warmup_per_arm: round-robin plays at the root before splitting (default 2).

Notation map (paper → code):
  β            → self.beta
  C_β          → self.lipschitz_L
  δ            → self.delta
  V_h          → leaf.width
  μ̂_k         → leaf.estimates[k]["mean"]
  ū_k,t        → confidence radius `cr` (see _confidence_radius)
  u_k,t        → ucb = mean + cr
  L_t          → tree.leaves
  Δ            → preference.gap (paper Definition 7)
  d_p          → eval.metrics.PreferenceRegret

context_dim
-----------
Supports any `context_dim >= 1`. For d=1 the default `branching="doubling"`
preserves v0.2 numbers (legacy from the original code). For d > 1, standard
dyadic branching (2^d children per split, bisecting every axis) is used.
"""

from __future__ import annotations

import numpy as np

from paretobandits.core.algorithm import Algorithm
from paretobandits.core.preference import Preference
from paretobandits.utils.tree import DyadicNode, DyadicTree


class PCBShift(Algorithm):
    """Preference-based contextual bandits with adaptive discretization.

    See module docstring for algorithm details.
    """

    name = "PCBShift"

    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        n_objectives: int,
        preference: Preference,
        delta: float = 0.05,
        horizon: int | None = None,
        beta: float = 1.0,
        lipschitz_L: float = 4.0,
        sigma_floor: float = 1e-3,
        warmup_per_arm: int = 2,
        elim_mode: str = "pairwise_pareto",
        split_mode: str = "pairwise",
        branching: str | None = None,
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
        self.beta = beta
        self.lipschitz_L = lipschitz_L
        self.sigma_floor = sigma_floor
        self.warmup_per_arm = warmup_per_arm
        if elim_mode not in ("individual", "pairwise", "pairwise_pareto"):
            raise ValueError(f"unknown elim_mode {elim_mode!r}")
        if split_mode not in ("pairwise", "data_driven"):
            raise ValueError(f"unknown split_mode {split_mode!r}")
        # Default branching: legacy "doubling" for d=1 (matches v0.2 numbers),
        # standard "dyadic" for d>1.
        if branching is None:
            branching = "doubling" if context_dim == 1 else "dyadic"
        if branching not in ("dyadic", "doubling"):
            raise ValueError(f"unknown branching {branching!r}")
        if branching == "doubling" and context_dim != 1:
            raise ValueError(
                "branching='doubling' is only valid for context_dim=1"
            )
        self.elim_mode = elim_mode
        self.split_mode = split_mode
        self.branching = branching

        # Built lazily on first use to allow reset() to rebuild.
        self.tree: DyadicTree | None = None
        self._last_leaf: DyadicNode | None = None
        self._last_action: int | None = None
        self._build_tree()

    # ─── Lifecycle ───────────────────────────────────────────────────

    def _build_tree(self) -> None:
        active = list(range(self.n_arms))
        self.tree = DyadicTree(
            active_arms=active,
            n_objectives=self.n_objectives,
            context_dim=self.context_dim,
            branching=self.branching,
        )
        self._last_leaf = None
        self._last_action = None

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._build_tree()

    # ─── Action selection ────────────────────────────────────────────

    def act(self, context: np.ndarray) -> int:
        x = np.atleast_1d(np.asarray(context, dtype=float)).reshape(-1)
        leaf = self.tree.find_leaf(x)
        self._last_leaf = leaf
        leaf.num_visit += 1

        # Round-robin warmup at the start.
        if self._t < self.warmup_per_arm * self.n_arms:
            action = self._t % self.n_arms
        else:
            # Refine the active-arm set by elimination.
            leaf.active_arms = self._eliminate_inferior(leaf, leaf.active_arms)
            if not leaf.active_arms:
                # Safety: never let active set go empty.
                leaf.active_arms = list(range(self.n_arms))

            # Estimated Pareto front from current means.
            pareto = self._estimated_pareto(leaf, leaf.active_arms)
            if not pareto:
                pareto = leaf.active_arms[:1]

            # Least-pulled tie-break for exploration among Pareto candidates.
            plays = np.array([leaf.estimates[k]["n"] for k in pareto])
            action = int(pareto[int(np.argmin(plays))])

        self._last_action = action
        return action

    # ─── Reward update ───────────────────────────────────────────────

    def update(
        self, context: np.ndarray, action: int, reward: np.ndarray
    ) -> None:
        if self._last_leaf is None or self._last_action != action:
            # Robustness: re-locate the leaf if act() wasn't called in pair.
            x = np.atleast_1d(np.asarray(context, dtype=float)).reshape(-1)
            self._last_leaf = self.tree.find_leaf(x)

        leaf = self._last_leaf
        est = leaf.estimates[action]
        est["n"] += 1
        n = est["n"]
        # Welford online mean + variance.
        delta_r = reward - est["mean"]
        est["mean"] = est["mean"] + delta_r / n
        delta_r2 = reward - est["mean"]
        est["M2"] = est["M2"] + delta_r * delta_r2
        if n >= 2:
            est["sigma"] = np.sqrt(np.maximum(est["M2"] / (n - 1), self.sigma_floor**2))
        # Confidence radius — see _confidence_radius for the formula.
        est["cr"] = self._confidence_radius(est["sigma"], n)
        est["ucb"] = est["mean"] + est["cr"]
        self._t += 1

        # Maybe split this leaf.
        if self._should_split(leaf):
            self.tree.split(leaf)

    # ─── Pareto-estimate API (used by metrics) ──────────────────────

    def pareto_estimate(self, context: np.ndarray) -> set[int]:
        x = np.atleast_1d(np.asarray(context, dtype=float)).reshape(-1)
        leaf = self.tree.find_leaf(x)
        if not leaf.active_arms:
            return set(range(self.n_arms))
        pareto = self._estimated_pareto(leaf, leaf.active_arms)
        return set(pareto) if pareto else set(leaf.active_arms[:1])

    # ─── Internal: confidence radii and elimination ─────────────────

    def _confidence_radius(self, sigma: np.ndarray, n: int) -> np.ndarray:
        """Per-arm CI half-width (Equation 5 in the paper, statistical part).

            cr = sigma * sqrt(2 * log(K * M / delta) / n)

        The Lipschitz bias C_β * V_h^β is handled separately — it cancels
        for *pairwise* comparisons within a bin (as both arms see the same
        bias) and only enters the splitting decision.
        """
        n = max(n, 1)
        log_term = np.log(max(self.n_arms * self.n_objectives / self.delta, 2.0))
        return sigma * np.sqrt(2 * log_term / n)

    def _pairwise_beta(
        self,
        sigma_i: np.ndarray,
        sigma_j: np.ndarray,
        n_i: int,
        n_j: int,
    ) -> np.ndarray:
        """Pairwise CI for the *difference* μ_i - μ_j.

            β_{i,j} = σ_pool * sqrt(2 * log(K^2 * M / δ) * (1/n_i + 1/n_j))

        Tighter than two individual CRs because the bin-bias cancels.
        """
        sigma_pool = np.maximum(sigma_i, sigma_j)
        log_term = np.log(
            max(self.n_arms**2 * self.n_objectives / self.delta, 2.0)
        )
        inv_sum = 1.0 / max(n_i, 1) + 1.0 / max(n_j, 1)
        return sigma_pool * np.sqrt(2 * log_term * inv_sum)

    def _eliminate_inferior(
        self, leaf: DyadicNode, arms: list[int]
    ) -> list[int]:
        """Remove arms provably dominated by another arm in the same bin.

        Two-phase: pairwise elimination, then optimistic-Pareto elimination
        if elim_mode == 'pairwise_pareto'.
        """
        if len(arms) <= 1:
            return arms
        n = len(arms)
        means = np.stack([leaf.estimates[a]["mean"] for a in arms])
        sigmas = np.stack([leaf.estimates[a]["sigma"] for a in arms])
        counts = np.array([leaf.estimates[a]["n"] for a in arms])

        has_data = counts >= 2
        if has_data.sum() <= 1:
            return arms

        # Phase 1: pairwise (tighter) or individual CI elimination.
        if self.elim_mode == "individual":
            cr = np.stack(
                [self._confidence_radius(sigmas[i], counts[i]) for i in range(n)]
            )
            lb = means - cr
            ub = means + cr
            # j dominates i: lb[j] >= ub[i] componentwise + at least one strict.
            lb_e = lb[:, np.newaxis, :]   # (n, 1, M)
            ub_e = ub[np.newaxis, :, :]   # (1, n, M)
            ge = np.all(lb_e >= ub_e, axis=2)
            gt = np.any(lb_e > ub_e, axis=2)
            dominates = ge & gt          # dominates[j, i] = j dominates i
        else:
            # Pairwise CI: dominates[j, i] = means[j] - means[i] > beta_{i,j}
            diff = means[:, np.newaxis, :] - means[np.newaxis, :, :]  # j - i
            sig_pool = np.maximum(
                sigmas[:, np.newaxis, :], sigmas[np.newaxis, :, :]
            )
            log_term = np.log(
                max(self.n_arms**2 * self.n_objectives / self.delta, 2.0)
            )
            inv_n = 1.0 / np.maximum(counts, 1)
            inv_sum = inv_n[:, np.newaxis] + inv_n[np.newaxis, :]
            beta = sig_pool * np.sqrt(2 * log_term * inv_sum[:, :, np.newaxis])
            dominates = np.all(diff > beta, axis=2)

        np.fill_diagonal(dominates, False)
        insufficient = ~has_data
        dominates[insufficient, :] = False
        dominates[:, insufficient] = False
        # arm i is dominated if any j dominates i (column i has any True).
        is_dominated = np.any(dominates, axis=0)

        survivors = [arms[i] for i in range(n) if not is_dominated[i]]
        if len(survivors) <= 1 or self.elim_mode != "pairwise_pareto":
            return survivors if survivors else arms

        # Phase 2: optimistic Pareto elimination.
        # Drop arms whose UCB is dominated by some pessimistic-Pareto LCB.
        ns = len(survivors)
        s_means = np.stack([leaf.estimates[a]["mean"] for a in survivors])
        s_sigmas = np.stack([leaf.estimates[a]["sigma"] for a in survivors])
        s_counts = np.array([leaf.estimates[a]["n"] for a in survivors])
        s_cr = np.stack(
            [
                self._confidence_radius(s_sigmas[i], s_counts[i])
                for i in range(ns)
            ]
        )
        ucb = s_means + s_cr
        lcb = s_means - s_cr
        lcb_pareto_mask = self.preference.pareto_set(lcb)
        lcb_pareto_vals = lcb[lcb_pareto_mask]

        keep = np.ones(ns, dtype=bool)
        for i in range(ns):
            if lcb_pareto_mask[i] or s_counts[i] < 2:
                continue
            ge = np.all(lcb_pareto_vals >= ucb[i], axis=1)
            gt = np.any(lcb_pareto_vals > ucb[i], axis=1)
            if np.any(ge & gt):
                keep[i] = False
        result = [survivors[i] for i in range(ns) if keep[i]]
        return result if result else survivors

    def _estimated_pareto(
        self, leaf: DyadicNode, arms: list[int]
    ) -> list[int]:
        """Pareto front of empirical means under the active preference."""
        if not arms:
            return []
        means = np.stack([leaf.estimates[a]["mean"] for a in arms])
        mask = self.preference.pareto_set(means)
        return [arms[i] for i in range(len(arms)) if mask[i]]

    # ─── Splitting rule ──────────────────────────────────────────────

    def _should_split(self, leaf: DyadicNode) -> bool:
        """Split when statistical uncertainty < Lipschitz bias of children.

        Specifically (paper Algorithm 1, line 14):

            max_{i,j ∈ Pareto(leaf)} β_{i,j}  <  L * child_max_width^β

        This trades off bin Lipschitz error against statistical error;
        once stat error is tighter than the child-bin error, splitting
        is information-positive.

        For d > 1 the child max-width is the per-axis half-width
        (uniform across axes under standard dyadic splitting).

        `data_driven` mode adds a second condition (numvisit >=
        K * numChild^2 * log(KM/delta)) that prevents over-splitting at
        large K.
        """
        # Need enough visits to be worth checking.
        if leaf.num_visit < 2 * len(leaf.active_arms):
            return False
        pareto = self._estimated_pareto(leaf, leaf.active_arms)
        if len(pareto) < 2:
            return False

        num_child = leaf.num_children
        # Child max-width (per axis). For doubling (1D legacy):
        # width / num_child along the single axis. For dyadic (d>=1):
        # max_width / 2 since every axis is bisected once.
        if self.branching == "doubling":
            child_max_width = float(leaf.width[0]) / num_child
        else:
            child_max_width = leaf.max_width / 2.0
        child_bias = self.lipschitz_L * (child_max_width**self.beta)

        max_beta = 0.0
        for ii in range(len(pareto)):
            for jj in range(ii + 1, len(pareto)):
                ei = leaf.estimates[pareto[ii]]
                ej = leaf.estimates[pareto[jj]]
                b_ij = self._pairwise_beta(
                    ei["sigma"], ej["sigma"], ei["n"], ej["n"]
                )
                max_beta = max(max_beta, float(np.max(b_ij)))
        ci_ok = max_beta < child_bias

        if self.split_mode == "data_driven":
            log_term = np.log(
                max(self.n_arms * self.n_objectives / self.delta, 2.0)
            )
            data_threshold = self.n_arms * (num_child**2) * log_term
            return ci_ok and (leaf.num_visit >= data_threshold)
        return ci_ok
