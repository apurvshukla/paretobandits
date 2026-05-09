"""Legacy baselines from prior work.

These are faithful implementations of the closest existing algorithms in
the multi-objective contextual bandit / covariate-shift literature, wrapped
in the new `Algorithm` API. They form the reproducibility track of the
benchmark — every paper in this neighborhood now has a one-line reason
to import this library.

Available:
    SukKpotufe20  — Suk & Kpotufe (2020), self-tuning bandits over unknown
                    covariate shifts. Adapted to multi-objective rewards
                    (the original is scalar).
    Turgay18      — Türgay, Öner & Tekin (2018), multi-objective contextual
                    bandits with similarity information. Stationary.
    Auer16        — Auer, Chiang, Ortner & Drugan (2016), Pareto front
                    identification from stochastic bandit feedback.
                    Context-free.
"""

from paretobandits.algos.legacy.annealing_pareto import AnnealingPareto
from paretobandits.algos.legacy.auer16 import Auer16
from paretobandits.algos.legacy.pareto_ucb import ParetoUCB
from paretobandits.algos.legacy.static_binning import (
    ATCBinning,
    CUSUMRestart,
    SlidingWindowBinning,
    StaticBinning,
)
from paretobandits.algos.legacy.suk_kpotufe import SukKpotufe20
from paretobandits.algos.legacy.turgay18 import Turgay18

__all__ = [
    "AnnealingPareto",
    "ATCBinning",
    "Auer16",
    "CUSUMRestart",
    "ParetoUCB",
    "SlidingWindowBinning",
    "StaticBinning",
    "SukKpotufe20",
    "Turgay18",
]
