"""Algorithms for multi-objective contextual bandits."""
from paretobandits.algos.baselines import RandomPlay, ScalarizedUCB
from paretobandits.algos.legacy import SukKpotufe20
from paretobandits.algos.pcb_shift import PCBShift

__all__ = ["PCBShift", "RandomPlay", "ScalarizedUCB", "SukKpotufe20"]

# Optional legacy baselines (only available if their dependencies / files
# are present; v0.2 ships SukKpotufe20, others land in v0.2.x).
try:
    from paretobandits.algos.legacy import Turgay18  # noqa: F401
    __all__.append("Turgay18")
except ImportError:
    pass

try:
    from paretobandits.algos.legacy import Auer16  # noqa: F401
    __all__.append("Auer16")
except ImportError:
    pass
