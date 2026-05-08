"""Preference orders over reward vectors.

This module is the main new abstraction relative to existing bandit libraries.
A `Preference` defines a partial order over R^M and provides the three
operations every multi-objective algorithm needs:

    - dominates(a, b)          : True iff a dominates b under the order
    - pareto_set(points)       : boolean mask of non-dominated points
    - gap(point, pareto_pts)   : scale-independent gap (Δ in the paper)

The default order in multi-objective bandits is component-wise (positive
orthant), but the paper's framework supports arbitrary polyhedral cones
C = {x : Ax <= 0}.  Halfspace cones (M=2) are the most common practical
generalization — e.g., a clinician who is willing to trade safety for
efficacy at a fixed exchange rate.

Conventions
-----------
- All reward vectors are MAXIMIZED.  Higher is better.
- Domination: a dominates b iff (a - b) is in the cone interior in the
  paper's notation; equivalently iff (a - b) is a non-zero point of C.
- For the positive orthant (default), this reduces to the standard
  "componentwise >= and at least one strict >".
"""

from __future__ import annotations

import numpy as np


class Preference:
    """Abstract preference order over R^M reward vectors.

    Subclasses must implement `dominates`, `pareto_set`, and `gap`.
    The default `gap` provided is the scale-independent log-ratio gap from
    the paper (Definition 7); subclasses can override for efficiency.
    """

    def __init__(self, M: int):
        if M < 1:
            raise ValueError(f"Number of objectives M must be >= 1, got {M}")
        self.M = M

    # ─── Required interface ──────────────────────────────────────────

    def dominates(self, a: np.ndarray, b: np.ndarray) -> bool:
        """True iff `a` weakly dominates `b` and they're not equal."""
        raise NotImplementedError

    def pareto_set(self, points: np.ndarray) -> np.ndarray:
        """Boolean mask of length K marking the Pareto-optimal rows.

        Args:
            points: (K, M) array of reward vectors.
        Returns:
            (K,) boolean array — True for non-dominated rows.
        """
        raise NotImplementedError

    # ─── Default (paper's) gap ───────────────────────────────────────

    def gap(self, point: np.ndarray, pareto_points: np.ndarray) -> float:
        """Scale-independent log-ratio gap (approximation to Definition 7).

        Approximates the paper's Δ(k, P) by

            Δ(k, P) ≈ max_{p ∈ P} || log10(p / k) ||_∞

        This is the same simplification used in the original experiments
        codebase. The exact Definition 7 requires solving a small LP per
        query and is identical for *strictly* dominated arms; the values
        differ for non-strictly-dominated arms (the paper definition
        returns 0 there, this approximation may not).

        For d_p computation between Pareto sets the approximation is
        mostly equivalent in practice — the original code uses a
        "skip arms in both fronts" trick to get clean zeros when sets
        coincide; that trick lives in `eval.metrics.PreferenceRegret`.

        Args:
            point: (M,) array — the arm's mean reward.
            pareto_points: (P, M) array — Pareto front mean rewards.
        Returns:
            Non-negative float.
        """
        if pareto_points.size == 0:
            return 0.0
        # Default safe-positive log-ratio (matches existing _gap_log_ratio_vectorized).
        # Subclasses with negative or zero rewards should override.
        eps = 1e-12
        p = np.clip(pareto_points, eps, None)
        a = np.clip(point, eps, None)
        log_ratios = np.log10(p / a)              # (P, M)
        linf = np.max(np.abs(log_ratios), axis=1)  # (P,)
        return float(np.max(linf))

    # ─── Convenience ────────────────────────────────────────────────

    def pareto_indices(self, points: np.ndarray) -> np.ndarray:
        """Indices of Pareto-optimal points."""
        return np.where(self.pareto_set(points))[0]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(M={self.M})"


class PolyhedralCone(Preference):
    """Preference induced by a polyhedral cone C = {x : A x <= 0}.

    `a` dominates `b` iff (a - b) ∈ C \\ {0}, i.e. A(a-b) <= 0 and a != b.

    Args:
        A: (m, M) constraint matrix defining the cone via Ax <= 0.
           For the positive orthant pass A = -I.  For a halfspace cone
           in 2D with weight vector w, pass A = [[-w_1, -w_2]].
        M: number of objectives (inferred from A.shape[1] if not given).

    Notes:
        Pareto computation is O(K^2 * m) — fine for K,m <= a few hundred.
        For the positive orthant, prefer `PositiveOrthant` which uses an
        O(K^2 * M) componentwise check.
    """

    def __init__(self, A: np.ndarray, M: int | None = None):
        A = np.asarray(A, dtype=float)
        if A.ndim != 2:
            raise ValueError(f"A must be 2D, got shape {A.shape}")
        inferred_M = A.shape[1]
        if M is not None and M != inferred_M:
            raise ValueError(f"M={M} does not match A.shape[1]={inferred_M}")
        super().__init__(inferred_M)
        self.A = A

    def dominates(self, a: np.ndarray, b: np.ndarray) -> bool:
        if np.allclose(a, b):
            return False
        diff = a - b
        return bool(np.all(self.A @ diff <= 1e-12))

    def pareto_set(self, points: np.ndarray) -> np.ndarray:
        K = points.shape[0]
        is_pareto = np.ones(K, dtype=bool)
        # For each pair, check if i dominates j → mark j as dominated.
        for i in range(K):
            if not is_pareto[i]:
                continue
            diffs = points[i] - points                   # (K, M)
            constraints = diffs @ self.A.T               # (K, m)
            in_cone = np.all(constraints <= 1e-12, axis=1)
            not_equal = np.any(diffs != 0, axis=1)
            dominated = in_cone & not_equal
            dominated[i] = False
            is_pareto[dominated] = False
        return is_pareto


class PositiveOrthant(PolyhedralCone):
    """The default cone: a dominates b iff a >= b componentwise and a != b.

    Equivalent to PolyhedralCone with A = -I_M, but uses a fast vectorized
    Pareto check that's O(K^2 * M) without matrix multiplication.
    """

    def __init__(self, M: int):
        A = -np.eye(M)
        super().__init__(A=A, M=M)

    def dominates(self, a: np.ndarray, b: np.ndarray) -> bool:
        return bool(np.all(a >= b) and np.any(a > b))

    def pareto_set(self, points: np.ndarray) -> np.ndarray:
        K = points.shape[0]
        is_pareto = np.ones(K, dtype=bool)
        for i in range(K):
            if not is_pareto[i]:
                continue
            ge_all = np.all(points[i] >= points, axis=1)
            gt_any = np.any(points[i] > points, axis=1)
            dominated = ge_all & gt_any
            dominated[i] = False
            is_pareto[dominated] = False
        return is_pareto


class HalfspaceCone(PolyhedralCone):
    """A halfspace cone in M=2: weighted-sum preference with weight w.

    `a` dominates `b` iff w · (a - b) > 0  and  a != b.

    This is a lexicographic-soft cone: the user has decided objectives
    can be traded off at exchange rate w_2 / w_1 (weight on objective 2
    relative to objective 1).  Not the same as scalarization for arm
    selection — Pareto sets are still computed, just under this order.

    Args:
        w: (M,) weight vector; will be normalized to unit L2 norm.
    """

    def __init__(self, w: np.ndarray):
        w = np.asarray(w, dtype=float)
        if w.ndim != 1 or np.all(w == 0):
            raise ValueError(f"w must be a non-zero 1D vector, got {w}")
        w = w / np.linalg.norm(w)
        # Ax <= 0  with  A = [-w]  means  -w·x <= 0  ⟺  w·x >= 0.
        A = -w.reshape(1, -1)
        super().__init__(A=A, M=w.size)
        self.w = w
