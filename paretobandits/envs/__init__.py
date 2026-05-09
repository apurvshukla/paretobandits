"""Benchmark environments."""
from paretobandits.envs.fairness import FairnessBandit
from paretobandits.envs.rlhf import RLHFBandit
from paretobandits.envs.synthetic import SyntheticShift

__all__ = ["SyntheticShift", "FairnessBandit", "RLHFBandit"]

# Warfarin is an optional import (requires pandas + xls reader).
try:
    from paretobandits.envs.warfarin import Warfarin  # noqa: F401
    __all__.append("Warfarin")
except ImportError:
    pass
