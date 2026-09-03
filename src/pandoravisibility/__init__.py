import os

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
TESTDIR = "/".join(PACKAGEDIR.split("/")[:-2]) + "/tests/"

# Find Version Number
import importlib.metadata
__version__ = importlib.metadata.version("pandoravisibility")
version = __version__

from .utils import analyze_target_yearly_visibility  # noqa: E402, F401
from .utils import (
    analyze_yearly_visibility,
    calculate_visibility_statistics,
    export_visibility_periods,
    find_continuous_periods,
    find_optimal_observation_windows,
    plot_observation_windows,
    plot_visibility_summary,
    plot_yearly_visibility,
)
from .visibility import Visibility  # noqa: E402, F401

__all__ = [
    "Visibility",
    "analyze_yearly_visibility",
    "find_continuous_periods",
    "calculate_visibility_statistics",
    "find_optimal_observation_windows",
    "export_visibility_periods",
    "analyze_target_yearly_visibility",
    "plot_yearly_visibility",
    "plot_visibility_summary",
    "plot_observation_windows",
]
