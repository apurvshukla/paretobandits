"""paretobandits: multi-objective contextual bandits under distribution shift.

Top-level convenience exports. Most users will only need:

    from paretobandits import (
        PCBShift, RandomPlay, ScalarizedUCB,
        SyntheticShift, Warfarin,
        PositiveOrthant, PolyhedralCone,
        Run, PreferenceRegret,
    )
"""

from paretobandits.algos.baselines import RandomPlay, ScalarizedUCB
from paretobandits.algos.legacy import SukKpotufe20
from paretobandits.algos.pcb_shift import PCBShift
from paretobandits.core.algorithm import Algorithm
from paretobandits.core.environment import Environment
from paretobandits.core.preference import (
    HalfspaceCone,
    PolyhedralCone,
    PositiveOrthant,
    Preference,
)
from paretobandits.envs.fairness import FairnessBandit
from paretobandits.envs.rlhf import RLHFBandit
from paretobandits.envs.synthetic import SyntheticShift
from paretobandits.eval.metrics import (
    DominanceCoverage,
    HausdorffRegret,
    ParetoPrecisionRecall,
    PreferenceRegret,
    RecoveryTime,
)
from paretobandits.eval.runner import Run, RunResult

__version__ = "0.8.0"

__all__ = [
    "Preference",
    "PolyhedralCone",
    "PositiveOrthant",
    "HalfspaceCone",
    "Algorithm",
    "Environment",
    "PCBShift",
    "RandomPlay",
    "ScalarizedUCB",
    "SukKpotufe20",
    "SyntheticShift",
    "FairnessBandit",
    "RLHFBandit",
    "PreferenceRegret",
    "HausdorffRegret",
    "DominanceCoverage",
    "RecoveryTime",
    "ParetoPrecisionRecall",
    "Run",
    "RunResult",
    "__version__",
]
