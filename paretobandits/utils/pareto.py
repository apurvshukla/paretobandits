"""Pareto-front utilities (positive-orthant fast path).

The cone-aware Pareto check lives on `Preference` subclasses; this module
provides fast vectorized helpers for the positive-orthant case used by
the hero algorithm.  Ports from the original `classes.py:_find_pareto_mask`
and gap functions, kept here to decouple algorithms from the legacy code.
"""

from __future__ import annotations

import numpy as np


def pareto_mask_orthant(means: np.ndarray) -> np.ndarray:
    """Fast vectorized Pareto front for the positive orthant.

    Args:
        means: (K, M) array. Maximization is assumed.
    Returns:
        (K,) boolean mask — True for non-dominated rows.
    """
    if means.ndim != 2:
        raise ValueError(f"means must be 2D (K, M), got shape {means.shape}")
    K = means.shape[0]
    is_pareto = np.ones(K, dtype=bool)
    for i in range(K):
        if not is_pareto[i]:
            continue
        ge_all = np.all(means[i] >= means, axis=1)
        gt_any = np.any(means[i] > means, axis=1)
        dominated = ge_all & gt_any
        dominated[i] = False
        is_pareto[dominated] = False
    return is_pareto


def log_ratio_gap(point: np.ndarray, pareto_points: np.ndarray) -> float:
    """Scale-independent log-ratio gap (Definition 7 in the paper).

    Δ(point, P) = max_{p ∈ P} || log10(p / point) ||_∞

    Both inputs must be component-wise positive (clip to a small floor
    if rewards may be zero).
    """
    if pareto_points.size == 0:
        return 0.0
    eps = 1e-12
    p = np.clip(pareto_points, eps, None)
    a = np.clip(point, eps, None)
    log_ratios = np.log10(p / a)
    return float(np.max(np.max(np.abs(log_ratios), axis=1)))


def arithmetic_gap(point: np.ndarray, pareto_points: np.ndarray) -> float:
    """Arithmetic gap: min_{p ∈ P} max(0, min_d (p_d - point_d)).

    Used by some legacy baselines (Türgay et al., 2018).  The paper's
    preferred metric is `log_ratio_gap` (above).
    """
    if pareto_points.size == 0:
        return 0.0
    diffs = pareto_points - point
    min_per_row = np.min(diffs, axis=1)
    clamped = np.maximum(min_per_row, 0.0)
    return float(np.min(clamped))
