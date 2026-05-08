"""Warfarin dosing environment — semi-synthetic from real IWPC data.

Wraps the existing `warfarin_data.py` module from the original `script/`
directory.  Exposes the IWPC patient pool, the dose arms (K=5), and the
two-objective (efficacy, safety) reward model through the standard
Environment interface.

Data source
-----------
IWPC, "Estimation of the Warfarin Dose with Clinical and Pharmacogenetic
Data," NEJM 2009; 360:753-764. Distributed via PharmGKB, accession PA162355460.

The XLS file (`iwpc_warfarin.xls`) should be present in either:
  - the `data/` subdirectory of this package, or
  - the path passed via `iwpc_path=`.

If unavailable, the environment falls back to synthetic patients matching
published IWPC marginals (5,000 patients by default).

Dependencies
------------
Requires `pandas` and an XLS reader (`xlrd` or `openpyxl`).  Install with:

    pip install paretobandits[warfarin]
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

import numpy as np

from paretobandits.core.environment import Environment

# ─── Locate and import the legacy module ────────────────────────────

def _import_warfarin_data():
    """Import the original `warfarin_data.py` module from script/.

    We bend the import path because the legacy module is a sibling
    directory; once the data is migrated into the package we'll drop
    this shim.
    """
    pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    candidates = [
        os.path.join(pkg_root, "..", "script"),                 # repo layout
        os.path.join(pkg_root, "data"),                          # in-package
        os.path.join(os.path.expanduser("~"), "covar", "script"),
    ]
    for path in candidates:
        path = os.path.abspath(path)
        if os.path.exists(os.path.join(path, "warfarin_data.py")):
            if path not in sys.path:
                sys.path.insert(0, path)
            try:
                import warfarin_data
                return warfarin_data
            except ImportError as e:
                # Re-raise with a clearer message about missing extras.
                raise ImportError(
                    "warfarin_data found but failed to import. Install "
                    "the warfarin extras: pip install paretobandits[warfarin]"
                ) from e
    raise ImportError(
        "warfarin_data.py not found. Expected locations:\n"
        + "\n".join(f"  - {p}" for p in candidates)
    )


# Imported lazily so the package import doesn't fail without pandas.
_wd = None


def _load():
    global _wd
    if _wd is None:
        _wd = _import_warfarin_data()
    return _wd


# ─── Environment ────────────────────────────────────────────────────


class Warfarin(Environment):
    """Warfarin dosing environment (K=5, M=2).

    Args:
        n_patients: size of the synthetic patient pool if IWPC data is
                    unavailable (default 5000).
        iwpc_path: optional override for the IWPC XLS file location.
        sigma: noise std-dev added to mean rewards (default 0.05).
        schedule: shift schedule — same options as SyntheticShift.
        shift_times: change-point times.
        nu: Beta-distribution skewness (default 5.0).
        seed: RNG seed.

    Note on the patient pool: the environment uses the existing legacy
    code's quantile-based mapping from context x ∈ [0,1] to patient.
    Pre-shift contexts concentrate on low-risk patients, post-shift on
    high-risk — modeling enrollment drift over a multi-month trial.
    """

    n_arms = 5
    n_objectives = 2
    context_dim = 1

    def __init__(
        self,
        n_patients: int = 5000,
        iwpc_path: str | None = None,
        sigma: float = 0.05,
        schedule: str = "single",
        shift_times: Sequence[int] | None = None,
        nu: float = 5.0,
        seed: int | None = None,
    ):
        super().__init__(rng=seed)
        wd = _load()
        # Build patient pool — prefer real IWPC, fall back to synthetic.
        if iwpc_path is not None and os.path.exists(iwpc_path):
            self._pool = wd.load_iwpc_patients(iwpc_path)
        else:
            try:
                self._pool = wd.get_real_patient_pool()
            except Exception:
                self._pool = wd.generate_patient_pool(n_patients, seed=seed or 42)
        wd.build_patient_index(self._pool)
        self._wd = wd

        self.sigma = sigma
        self.nu = nu

        if schedule not in ("none", "single", "multi", "gradual"):
            raise ValueError(f"unknown schedule {schedule!r}")
        self.schedule = schedule
        if schedule == "none":
            self._shifts = []
        elif schedule == "single":
            self._shifts = list(shift_times or [1500])
        elif schedule == "multi":
            if shift_times is None:
                raise ValueError("multi requires shift_times")
            self._shifts = list(shift_times)
        else:  # gradual
            if shift_times is None or len(shift_times) != 2:
                raise ValueError("gradual requires (t0, t1)")
            self._shifts = list(shift_times)

    # ─── Context generation ──────────────────────────────────────────

    def context(self, t: int) -> np.ndarray:
        if self.schedule == "none":
            x = self.rng.beta(1.0, self.nu + 1)
        elif self.schedule == "single":
            tp = self._shifts[0]
            x = self.rng.beta(1.0, self.nu + 1) if t < tp else self.rng.beta(self.nu + 1, 1.0)
        elif self.schedule == "multi":
            phase = sum(1 for ts in self._shifts if t >= ts) % 2
            x = self.rng.beta(1.0, self.nu + 1) if phase == 0 else self.rng.beta(self.nu + 1, 1.0)
        else:  # gradual
            t0, t1 = self._shifts
            if t < t0:
                a, b = 1.0, self.nu + 1
            elif t > t1:
                a, b = self.nu + 1, 1.0
            else:
                w = (t - t0) / max(t1 - t0, 1)
                a = (1 - w) * 1.0 + w * (self.nu + 1)
                b = (1 - w) * (self.nu + 1) + w * 1.0
            x = self.rng.beta(a, b)
        return np.array([float(np.clip(x, 1e-3, 1 - 1e-3))])

    # ─── True means and reward sampling ──────────────────────────────

    def true_means(self, context: np.ndarray) -> np.ndarray:
        x = float(np.atleast_1d(context)[0])
        patient = self._wd._context_to_patient(x, self._pool)
        return self._wd.get_arm_means(patient)

    def step(self, t: int, action: int) -> np.ndarray:
        means = self.true_means(self.context(t))
        return means[action] + self.sigma * self.rng.standard_normal(self.n_objectives)

    # ─── Shift hooks ────────────────────────────────────────────────

    def is_shifted(self, t: int) -> bool:
        return self.schedule != "none" and t >= self._shifts[0]

    def shift_times(self) -> list:
        return list(self._shifts)
