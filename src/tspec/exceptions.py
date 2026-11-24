"""Custom exception hierarchy for tspec."""

from __future__ import annotations


class TspecError(Exception):
    """Base exception for all tspec errors."""


class InsufficientDataError(TspecError, ValueError):
    """Raised when a series or window has too few valid observations for estimation."""


class RankDeficiencyError(TspecError, ValueError):
    """Raised when a design or weight matrix is numerically singular or non-SPD."""


class CensoringError(TspecError, ValueError):
    """Raised when censored-data parsing, guardrails, or imputation fails."""
