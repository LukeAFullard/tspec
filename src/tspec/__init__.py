"""tspec: Unevenly sampled time-series spectral analysis, change detection, and decomposition.

Clean-room implementation of LSSA, ALLSSA, JUST, and STPD with integrated
censored-data support via ``ndimpute``.
"""

from __future__ import annotations

from ._diagnostics import DiagnosticReport, RedNoiseDiagnostics, WarningItem
from ._robust import RobustMethod, SlopeEstimator
from .censored import CensoredConfig, CensoringProvenance, DlShiftEvent
from .exceptions import (
    CensoringError,
    InsufficientDataError,
    RankDeficiencyError,
    TspecError,
)
from .just import (
    JumpEvent,
    JustDecomposeResult,
    JustJumpResult,
    WindowCandidate,
    just_decompose,
    just_jumps,
)
from .monitor import MonitorResult, MonitorStep, just_monitor
from .pipeline import PipelineResult, SegmentSummary, analyze_series
from .spectral import AllssaResult, LssaResult, allssa, lssa
from .stpd import (
    FlaggedJumpCandidate,
    PiecewiseTrendResult,
    StpdResult,
    TurningPointEvent,
    detect_turning_points,
    fit_piecewise_trend,
    stpd,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # Core functions
    "lssa",
    "allssa",
    "just_jumps",
    "just_decompose",
    "just_monitor",
    "detect_turning_points",
    "stpd",
    "fit_piecewise_trend",
    "analyze_series",
    # Type aliases & config/result classes
    "RobustMethod",
    "SlopeEstimator",
    "CensoredConfig",
    "CensoringProvenance",
    "DlShiftEvent",
    "DiagnosticReport",
    "RedNoiseDiagnostics",
    "WarningItem",
    "LssaResult",
    "AllssaResult",
    "JumpEvent",
    "WindowCandidate",
    "JustJumpResult",
    "JustDecomposeResult",
    "MonitorStep",
    "MonitorResult",
    "TurningPointEvent",
    "FlaggedJumpCandidate",
    "PiecewiseTrendResult",
    "StpdResult",
    "SegmentSummary",
    "PipelineResult",
    # Exceptions
    "TspecError",
    "InsufficientDataError",
    "RankDeficiencyError",
    "CensoringError",
]
