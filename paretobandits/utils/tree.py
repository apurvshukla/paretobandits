"""Dyadic tree partitioning of [0, 1]^d with per-bin arm estimates.

This is the data structure underlying Algorithm 1 of the paper.  Each
node maintains:
  - axis-aligned hyper-rectangular bounds (`bounds`: (d, 2) array)
  - geometric center (`center`: (d,) array)
  - per-dim widths (`width`: (d,) array)
  - per-arm running estimates (Welford mean, online sigma, play count)
  - active-arm set inherited from parent
  - children (lazily created when split)

The tree starts as a single root covering [0, 1]^d.  When a leaf's
statistical uncertainty drops below its diameter (under a configurable
norm), it is split into children. Two splitting schemes are supported:

  - "dyadic"  (default for d>1, d=1):  bisect every axis simultaneously,
                                       producing 2^d children of equal
                                       size. Standard adaptive partition.
  - "doubling" (legacy, opt-in for d=1): produce 2^(level+1) children
                                       along the single axis, matching
                                       the original `script/classes.py`
                                       behavior. Provided for backward
                                       compatibility with v0.2 numbers.

The `split` rule defaults to `dyadic` regardless of dimension; pass
`branching="doubling"` to `DyadicTree` to recover the legacy behavior
in 1D.
"""

from __future__ import annotations

import numpy as np


class DyadicNode:
    """Single node in the dyadic tree.

    Args:
        level: depth in the tree (root = 0).
        bounds: (d, 2) array — per-dim [low, high].
        active_arms: list of arm ids active in this bin.
        n_objectives: M.
        parent: parent node, or None for root.
        num_children: branching factor on next split. If None, defaults
                      to 2^d (standard dyadic).
    """

    __slots__ = (
        "level",
        "bounds",
        "center",
        "width",
        "active_arms",
        "M",
        "parent",
        "children",
        "num_children",
        "num_visit",
        "estimates",
    )

    def __init__(
        self,
        level: int,
        bounds: np.ndarray,
        active_arms: list[int],
        n_objectives: int,
        parent: DyadicNode | None = None,
        num_children: int | None = None,
    ):
        bounds = np.asarray(bounds, dtype=float)
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError(
                f"bounds must have shape (d, 2), got {bounds.shape}"
            )
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError(f"bounds rows must satisfy low < high, got {bounds}")

        self.level = level
        self.bounds = bounds
        self.center = 0.5 * (bounds[:, 0] + bounds[:, 1])
        self.width = bounds[:, 1] - bounds[:, 0]
        self.active_arms = list(active_arms)
        self.M = n_objectives
        self.parent = parent
        self.children: list[DyadicNode] = []
        d = bounds.shape[0]
        self.num_children = (
            num_children if num_children is not None else 2**d
        )
        self.num_visit = 0
        # Per-arm Welford bookkeeping (mean, M2, sigma) + per-arm CIs.
        self.estimates: dict[int, dict] = {
            arm_id: {
                "mean": np.zeros(n_objectives),
                "M2": np.zeros(n_objectives),
                "n": 0,
                "sigma": np.ones(n_objectives),
                "cr": np.full(n_objectives, np.inf),
                "ucb": np.full(n_objectives, np.inf),
            }
            for arm_id in active_arms
        }

    def contains(self, x: np.ndarray) -> bool:
        """Whether this node's bin contains x (inclusive of bounds)."""
        x = np.asarray(x).reshape(-1)
        return bool(
            np.all(x >= self.bounds[:, 0]) and np.all(x <= self.bounds[:, 1])
        )

    @property
    def diameter(self) -> float:
        """L2 diameter of the bin (sqrt of sum of squared per-dim widths)."""
        return float(np.linalg.norm(self.width))

    @property
    def max_width(self) -> float:
        """Sup-norm width of the bin (max per-dim width)."""
        return float(np.max(self.width))

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def inherit_from(self, parent: DyadicNode) -> None:
        """Warm-start estimates from parent at split time."""
        for arm_id, est in parent.estimates.items():
            if arm_id in self.estimates:
                self.estimates[arm_id] = {
                    "mean": est["mean"].copy(),
                    "M2": est["M2"].copy(),
                    "n": est["n"],
                    "sigma": est["sigma"].copy(),
                    "cr": est["cr"].copy(),
                    "ucb": est["ucb"].copy(),
                }


class DyadicTree:
    """Dyadic tree over [0, 1]^d with leaf bookkeeping.

    Maintains the current set of leaves so context lookup is fast.

    Args:
        active_arms: list of arm ids initially active in the root bin.
        n_objectives: M.
        context_dim: d (default 1 for backward compatibility).
        branching: "dyadic" (bisect every axis, 2^d children, default) or
                   "doubling" (1D-only legacy: 2^(level+1) children
                   along the single axis).
    """

    def __init__(
        self,
        active_arms: list[int],
        n_objectives: int,
        context_dim: int = 1,
        branching: str = "dyadic",
    ):
        if branching not in ("dyadic", "doubling"):
            raise ValueError(f"unknown branching {branching!r}")
        if branching == "doubling" and context_dim != 1:
            raise ValueError("branching='doubling' is only valid for context_dim=1")
        self.context_dim = int(context_dim)
        self.branching = branching
        root_bounds = np.tile(np.array([0.0, 1.0]), (self.context_dim, 1))
        root_num_children = (
            2 if branching == "doubling" else 2**self.context_dim
        )
        self.root = DyadicNode(
            level=0,
            bounds=root_bounds,
            active_arms=active_arms,
            n_objectives=n_objectives,
            parent=None,
            num_children=root_num_children,
        )
        self.leaves: list[DyadicNode] = [self.root]

    def find_leaf(self, x: np.ndarray) -> DyadicNode:
        """Locate the leaf containing context x.

        Args:
            x: (d,) context vector. Will be clipped to [0, 1] per dim.
        """
        x = np.atleast_1d(np.asarray(x, dtype=float)).reshape(-1)
        if x.size != self.context_dim:
            raise ValueError(
                f"context dim mismatch: tree expects {self.context_dim}, "
                f"got {x.size}"
            )
        x = np.clip(x, 0.0, 1.0)
        for leaf in self.leaves:
            if leaf.contains(x):
                return leaf
        return self.root  # fallback (should not occur for valid trees)

    def split(self, node: DyadicNode) -> list[DyadicNode]:
        """Split `node` into children, update leaves, return new children.

        Behavior depends on `self.branching`:
          - "dyadic": bisect every axis, producing 2^d children.
          - "doubling" (1D only): produce `node.num_children` equal-width
            children along the single axis.
        """
        if not node.is_leaf():
            return node.children

        if self.branching == "doubling":
            children = self._split_doubling(node)
        else:
            children = self._split_dyadic(node)

        node.children = children
        self.leaves.remove(node)
        self.leaves.extend(children)
        return children

    # ─── Splitting strategies ──────────────────────────────────────

    def _split_dyadic(self, node: DyadicNode) -> list[DyadicNode]:
        """Bisect each axis, producing 2^d children of equal volume."""
        d = self.context_dim
        midpoints = node.center
        children: list[DyadicNode] = []
        for combo in range(2**d):
            child_bounds = np.empty_like(node.bounds)
            for dim in range(d):
                bit = (combo >> dim) & 1
                if bit == 0:
                    child_bounds[dim, 0] = node.bounds[dim, 0]
                    child_bounds[dim, 1] = midpoints[dim]
                else:
                    child_bounds[dim, 0] = midpoints[dim]
                    child_bounds[dim, 1] = node.bounds[dim, 1]
            child = DyadicNode(
                level=node.level + 1,
                bounds=child_bounds,
                active_arms=node.active_arms,
                n_objectives=node.M,
                parent=node,
                num_children=2**d,
            )
            child.inherit_from(node)
            children.append(child)
        return children

    def _split_doubling(self, node: DyadicNode) -> list[DyadicNode]:
        """1D legacy: produce `node.num_children` equal-width children
        along the single axis, with next level's children doubling again.
        """
        n = node.num_children
        low = node.bounds[0, 0]
        high = node.bounds[0, 1]
        child_width = (high - low) / n
        children: list[DyadicNode] = []
        for i in range(n):
            child_low = low + i * child_width
            child_high = child_low + child_width
            child_bounds = np.array([[child_low, child_high]])
            child = DyadicNode(
                level=node.level + 1,
                bounds=child_bounds,
                active_arms=node.active_arms,
                n_objectives=node.M,
                parent=node,
                num_children=2 ** (node.level + 2),
            )
            child.inherit_from(node)
            children.append(child)
        return children

    def all_leaves(self) -> list[DyadicNode]:
        return list(self.leaves)
