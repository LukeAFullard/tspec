"""Shared base interface and helpers for all tspec result dataclasses."""

from __future__ import annotations

from typing import Any

from ._diagnostics import DiagnosticReport


def require_pandas() -> Any:
    """Lazily import pandas with an informative message for lite installs."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for `.to_dataframe()`. Install it with "
            "`pip install pandas` or reinstall tspec normally (`pip install tspec`)."
        ) from exc
    return pd


def require_matplotlib() -> Any:
    """Lazily import matplotlib.pyplot with an informative message."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for `.plot()`. Install it with "
            "`pip install tspec[plot]`."
        ) from exc
    return plt


def format_diagnostics_block(diag: DiagnosticReport) -> list[str]:
    """Render concise diagnostic and warning lines for `.summary()`."""
    lines: list[str] = [
        f"Observations : {diag.n_valid} valid / {diag.n_input} total"
        + (f" ({diag.n_dropped_nan} NaN dropped)" if diag.n_dropped_nan else "")
    ]
    if diag.red_noise is not None and diag.red_noise.lag1_autocorr == diag.red_noise.lag1_autocorr:
        rn = diag.red_noise
        lines.append(
            f"Red noise    : lag-1 r = {rn.lag1_autocorr:+.3f}, "
            f"OU r = {rn.ou_decay_corr:.3f}, n_eff ≈ {rn.effective_sample_size:.1f} "
            f"({rn.severity})"
        )
    if diag.condition_number_max > 1e8:
        lines.append(f"Conditioning : max cond(R) = {diag.condition_number_max:.2e}")
    if diag.warnings:
        lines.append(f"Diagnostics  : {len(diag.warnings)} notice(s)")
        for w in diag.warnings:
            lines.append(f"  [{w.severity.upper()}:{w.code}] {w.message}")
    return lines


def format_explain_warnings(diag: DiagnosticReport) -> list[str]:
    """Render plain-English caveats and next steps for `.explain()`."""
    if not diag.warnings:
        return ["No statistical red flags were triggered on this series."]
    out: list[str] = ["Important caveats to keep in mind:"]
    for w in diag.warnings:
        tag = {"warning": "WARNING", "caution": "Caution", "info": "Note"}[w.severity]
        line = f"  - {tag}: {w.message}"
        if w.suggestion:
            line += f" Suggestion: {w.suggestion}"
        out.append(line)
    return out


class BaseResult:
    """Common protocol for all tspec result objects."""

    diagnostics: DiagnosticReport

    def summary(self) -> str:
        """Return a concise technical summary table."""
        raise NotImplementedError

    def explain(self) -> str:
        """Return a plain-English narrative interpretation for practitioners."""
        raise NotImplementedError

    def to_dataframe(self) -> Any:
        """Export structured results as a tidy pandas DataFrame."""
        raise NotImplementedError

    def plot(self, ax: Any = None, **kwargs: Any) -> tuple[Any, Any]:
        """Generate a diagnostic Matplotlib figure."""
        raise NotImplementedError

    def __str__(self) -> str:
        return self.summary()
