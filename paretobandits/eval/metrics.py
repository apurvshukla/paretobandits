"""Evaluation metrics over `RunResult` objects.

All metrics follow the same protocol:

    metric = MetricClass(... config ...)
    values = metric.compute(result, env)   # returns array shape (n_seeds, T)
    summary = metric.summarize(values)     # returns dict of scalar stats

The benchmark report renders all six metrics by default — the philosophy
is that "report only the metric your algorithm wins on" is exactly what
makes benchmarks untrustworthy.

Metrics
-------
- PreferenceRegret      : the paper's d_p (Definition 8). Primary metric.
- HausdorffRegret       : classical Hausdorff over Pareto sets in objective space.
- DominanceCoverage     : |P_π ∩ P*| / |P*|, fraction of true Pareto arms played.
- ParetoPrecisionRecall : precision / recall of the algorithm's estimated set.
- RecoveryTime          : steps after each shift to return to within ε of oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from paretobandits.core.preference import PositiveOrthant, Preference

if TYPE_CHECKING:
    from paretobandits.core.environment import Environment
    from paretobandits.eval.runner import RunResult


@dataclass
class MetricSummary:
    """Per-metric summary across seeds."""

    name: str
    mean_curve: np.ndarray   # (T,) — mean across seeds
    std_curve: np.ndarray    # (T,) — std across seeds
    final_mean: float        # mean at horizon
    final_std: float         # std at horizon
    cumulative_mean: float   # mean of total summed metric
    cumulative_std: float


def _summarize(name: str, values: np.ndarray) -> MetricSummary:
    """values: (n_seeds, T) → MetricSummary."""
    mean_curve = values.mean(axis=0)
    std_curve = values.std(axis=0)
    cum = values.sum(axis=1)
    return MetricSummary(
        name=name,
        mean_curve=mean_curve,
        std_curve=std_curve,
        final_mean=float(mean_curve[-1]),
        final_std=float(std_curve[-1]),
        cumulative_mean=float(cum.mean()),
        cumulative_std=float(cum.std()),
    )


# ─── Primary: preference regret ─────────────────────────────────────


class PreferenceRegret:
    """The paper's d_p regret (Definition 8 + Equation 2).

    For each step t > t_p:
        regret_t = d_p(P_π(X_t), P*(X_t))

    where d_p is the max scale-independent log-ratio gap between Pareto
    sets.  Pre-shift values are reported as zero by convention (see paper);
    pass post_shift_only=False to include them anyway.
    """

    name = "preference_regret"

    def __init__(
        self,
        preference: Preference | None = None,
        post_shift_only: bool = True,
    ):
        self.preference = preference
        self.post_shift_only = post_shift_only

    def compute(self, result: RunResult, env: Environment) -> np.ndarray:
        n_seeds, T = result.actions.shape
        regret = np.zeros((n_seeds, T))
        pref = self.preference or PositiveOrthant(env.n_objectives)

        for s in range(n_seeds):
            for t in range(T):
                if self.post_shift_only and not env.is_shifted(t):
                    continue
                ctx = result.contexts[s, t]
                true_means = env.true_means(ctx)
                true_mask = pref.pareto_set(true_means)
                est_ids = result.pareto_estimates[s][t]
                if not est_ids:
                    est_ids = {result.actions[s, t]}
                est_mask = np.zeros(env.n_arms, dtype=bool)
                for k in est_ids:
                    if k < env.n_arms:
                        est_mask[k] = True
                regret[s, t] = self._d_p(
                    true_means, true_mask, est_mask, pref
                )
        return regret

    @staticmethod
    def _d_p(
        means: np.ndarray,
        true_mask: np.ndarray,
        est_mask: np.ndarray,
        preference: Preference,
    ) -> float:
        """Symmetric scale-independent Pareto-set distance.

        Implements the "skip arms in both sets" convention from the
        original experiments code: arms that appear in both Pareto sets
        contribute zero. This mirrors paper Definition 7's property that
        Δ(k, P) = 0 when k ∈ P, which the simplified gap formula does
        not satisfy on its own.
        """
        true_ids = set(np.where(true_mask)[0].tolist())
        est_ids = set(np.where(est_mask)[0].tolist())
        only_true = true_ids - est_ids
        only_est = est_ids - true_ids
        if not only_true and not only_est:
            return 0.0
        true_pts = means[list(true_ids)] if true_ids else means[:0]
        est_pts = means[list(est_ids)] if est_ids else means[:0]
        d1 = max(
            (preference.gap(means[i], true_pts) for i in only_est),
            default=0.0,
        )
        d2 = max(
            (preference.gap(means[i], est_pts) for i in only_true),
            default=0.0,
        )
        return float(max(d1, d2))

    def summarize(self, values: np.ndarray) -> MetricSummary:
        return _summarize(self.name, values)


# ─── Hausdorff (for comparison with prior work) ─────────────────────


class HausdorffRegret:
    """Classical Hausdorff distance between Pareto sets in objective space.

    Reported alongside PreferenceRegret to demonstrate the paper's
    Example 2 — that Hausdorff convergence does not imply preference
    convergence.
    """

    name = "hausdorff_regret"

    def __init__(self, preference: Preference | None = None):
        self.preference = preference

    def compute(self, result: RunResult, env: Environment) -> np.ndarray:
        n_seeds, T = result.actions.shape
        regret = np.zeros((n_seeds, T))
        pref = self.preference or PositiveOrthant(env.n_objectives)
        for s in range(n_seeds):
            for t in range(T):
                ctx = result.contexts[s, t]
                means = env.true_means(ctx)
                true_mask = pref.pareto_set(means)
                est_ids = result.pareto_estimates[s][t]
                est_mask = np.zeros(env.n_arms, dtype=bool)
                for k in est_ids:
                    if k < env.n_arms:
                        est_mask[k] = True
                regret[s, t] = self._hausdorff(
                    means[true_mask], means[est_mask]
                )
        return regret

    @staticmethod
    def _hausdorff(A: np.ndarray, B: np.ndarray) -> float:
        if A.size == 0 or B.size == 0:
            return 0.0
        # Symmetric Hausdorff in L2.
        diffs_AB = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
        d1 = diffs_AB.min(axis=1).max()
        d2 = diffs_AB.min(axis=0).max()
        return float(max(d1, d2))

    def summarize(self, values: np.ndarray) -> MetricSummary:
        return _summarize(self.name, values)


# ─── Dominance coverage ─────────────────────────────────────────────


class DominanceCoverage:
    """Fraction of true Pareto arms in the algorithm's estimated set.

    DC = |P_estimated ∩ P_true| / |P_true|

    Higher is better. 1.0 means all true Pareto arms are recognized.
    """

    name = "dominance_coverage"

    def __init__(self, preference: Preference | None = None):
        self.preference = preference

    def compute(self, result: RunResult, env: Environment) -> np.ndarray:
        n_seeds, T = result.actions.shape
        coverage = np.zeros((n_seeds, T))
        pref = self.preference or PositiveOrthant(env.n_objectives)
        for s in range(n_seeds):
            for t in range(T):
                ctx = result.contexts[s, t]
                means = env.true_means(ctx)
                true_mask = pref.pareto_set(means)
                true_ids = set(np.where(true_mask)[0])
                est_ids = result.pareto_estimates[s][t]
                if not true_ids:
                    coverage[s, t] = 1.0
                else:
                    coverage[s, t] = len(true_ids & est_ids) / len(true_ids)
        return coverage

    def summarize(self, values: np.ndarray) -> MetricSummary:
        return _summarize(self.name, values)


# ─── Pareto precision/recall ───────────────────────────────────────


class ParetoPrecisionRecall:
    """Precision and recall of the estimated Pareto set vs the true one.

    Returns the F1 score per step.  Use this when you care about both
    over-inclusion (false positives) and under-inclusion (false negatives).
    """

    name = "pareto_f1"

    def __init__(self, preference: Preference | None = None):
        self.preference = preference

    def compute(self, result: RunResult, env: Environment) -> np.ndarray:
        n_seeds, T = result.actions.shape
        f1 = np.zeros((n_seeds, T))
        pref = self.preference or PositiveOrthant(env.n_objectives)
        for s in range(n_seeds):
            for t in range(T):
                ctx = result.contexts[s, t]
                means = env.true_means(ctx)
                true_mask = pref.pareto_set(means)
                true_ids = set(np.where(true_mask)[0])
                est_ids = result.pareto_estimates[s][t]
                tp = len(true_ids & est_ids)
                if not est_ids:
                    p = 0.0
                else:
                    p = tp / len(est_ids)
                r = tp / len(true_ids) if true_ids else 1.0
                f1[s, t] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return f1

    def summarize(self, values: np.ndarray) -> MetricSummary:
        return _summarize(self.name, values)


# ─── Recovery time ──────────────────────────────────────────────────


class RecoveryTime:
    """Time after each shift for the algorithm to recover.

    Defined as the smallest k such that the rolling-mean preference regret
    over [t_p+1, t_p+k] falls below ε.  Reported per shift per seed.

    Args:
        epsilon: regret threshold (default 0.05).
        window: rolling-mean window size (default 100).
    """

    name = "recovery_time"

    def __init__(
        self,
        epsilon: float = 0.05,
        window: int = 100,
        preference: Preference | None = None,
    ):
        self.epsilon = epsilon
        self.window = window
        self.preference = preference

    def compute(self, result: RunResult, env: Environment) -> np.ndarray:
        # Reuse PreferenceRegret to compute per-step values.
        pr = PreferenceRegret(preference=self.preference, post_shift_only=False)
        regret = pr.compute(result, env)
        n_seeds, T = regret.shape
        shifts = env.shift_times()
        if not shifts:
            # No shifts → recovery undefined; return zeros for shape compat.
            return np.zeros((n_seeds, T))

        out = np.full((n_seeds, T), np.nan)
        for s in range(n_seeds):
            for tp in shifts:
                # Rolling mean from tp onwards; first window where mean < eps.
                tail = regret[s, tp:]
                if tail.size < self.window:
                    continue
                cumsum = np.cumsum(tail)
                # Sliding window mean.
                k = self.window
                roll = (cumsum[k - 1 :] - np.concatenate([[0.0], cumsum[:-k]])) / k
                hits = np.where(roll < self.epsilon)[0]
                rt = int(hits[0]) + k if len(hits) > 0 else T - tp
                # Encode as a constant value over [tp, end] for plotting.
                out[s, tp:] = rt
        return out

    def summarize(self, values: np.ndarray) -> MetricSummary:
        # nanmean / nanstd because recovery is only defined post-shift.
        with np.errstate(invalid="ignore"):
            mean_curve = np.nanmean(values, axis=0)
            std_curve = np.nanstd(values, axis=0)
        # Cumulative is meaningless for recovery — use per-shift mean RT.
        first_rt = np.array([
            row[~np.isnan(row)][0] if np.any(~np.isnan(row)) else np.nan
            for row in values
        ])
        with np.errstate(invalid="ignore"):
            cum_mean = float(np.nanmean(first_rt))
            cum_std = float(np.nanstd(first_rt))
            final_mean = float(np.nanmean(values[:, -1]))
            final_std = float(np.nanstd(values[:, -1]))
        return MetricSummary(
            name=self.name,
            mean_curve=mean_curve,
            std_curve=std_curve,
            final_mean=final_mean,
            final_std=final_std,
            cumulative_mean=cum_mean,
            cumulative_std=cum_std,
        )
