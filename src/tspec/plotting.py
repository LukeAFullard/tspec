"""Publication-ready diagnostic plots for tspec results (lazy Matplotlib import).

Uses a colorblind-safe Okabe-Ito palette, never mutates global ``rcParams``,
and visually distinguishes censored/imputed observations (hollow downward
triangles for left-censored ``<DL``, hollow upward triangles for right-censored
``>UL``, and error whiskers for interval bounds).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .results import require_matplotlib

if TYPE_CHECKING:
    from .censored import CensoringProvenance
    from .just import JustDecomposeResult, JustJumpResult
    from .monitor import MonitorResult
    from .pipeline import PipelineResult
    from .spectral import AllssaResult, LssaResult
    from .stpd import PiecewiseTrendResult, StpdResult

# Okabe-Ito colorblind-safe palette
_BLUE = "#0072B2"
_ORANGE = "#D55E00"
_GREEN = "#009E73"
_PURPLE = "#CC79A7"
_SKY = "#56B4E9"
_DARK = "#222222"
_GRAY = "#7F7F7F"


def _plot_observations_with_censoring(
    ax: Any,
    t: Any,
    y: np.ndarray[Any, Any],
    censoring: CensoringProvenance | None,
    *,
    label_obs: str = "Observed",
) -> None:
    """Scatter observations, visually distinguishing censored/imputed entries."""
    if censoring is None or censoring.n_censored == 0:
        ax.plot(t, y, "o", color=_DARK, markersize=3.5, alpha=0.75, label=label_obs)
        return

    obs_mask = ~censoring.is_censored
    if np.any(obs_mask):
        ax.plot(
            t[obs_mask],
            y[obs_mask],
            "o",
            color=_DARK,
            markersize=3.8,
            alpha=0.8,
            label=label_obs,
        )

    left_mask = censoring.status_codes == -1
    if np.any(left_mask):
        ax.plot(
            t[left_mask],
            y[left_mask],
            "v",
            markerfacecolor="white",
            markeredgecolor=_ORANGE,
            markeredgewidth=1.2,
            markersize=5.0,
            linestyle="none",
            label="Imputed (<DL)",
        )

    right_mask = censoring.status_codes == 1
    if np.any(right_mask):
        ax.plot(
            t[right_mask],
            y[right_mask],
            "^",
            markerfacecolor="white",
            markeredgecolor=_PURPLE,
            markeredgewidth=1.2,
            markersize=5.0,
            linestyle="none",
            label="Imputed (>UL)",
        )

    int_mask = censoring.status_codes == 2
    if np.any(int_mask) and censoring.original_left is not None and censoring.original_right is not None:
        lo = censoring.original_left[int_mask]
        hi = censoring.original_right[int_mask]
        yv = y[int_mask]
        ax.errorbar(
            t[int_mask],
            yv,
            yerr=[np.maximum(0.0, yv - lo), np.maximum(0.0, hi - yv)],
            fmt="s",
            color=_SKY,
            markerfacecolor="white",
            markersize=4.0,
            capsize=2.0,
            linewidth=0.9,
            label="Imputed [L, R]",
        )

    for shift in censoring.dl_shifts:
        ax.axvline(
            shift.time,
            color=_GRAY,
            linestyle=":",
            linewidth=1.2,
            alpha=0.85,
            label="DL shift" if shift == censoring.dl_shifts[0] else None,
        )


def plot_lssa(result: LssaResult, ax: Any = None, **kwargs: Any) -> tuple[Any, Any]:
    """Plot LSSA percentage-variance spectrum and critical threshold line."""
    plt = require_matplotlib()
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (8, 3.8)))
    else:
        fig = ax.figure

    ax.plot(result.frequencies, result.spectrum * 100.0, color=_BLUE, lw=1.4, label="LSS (%)")
    if np.isfinite(result.critical_value):
        ax.axhline(
            result.critical_value * 100.0,
            color=_ORANGE,
            ls="--",
            lw=1.2,
            label=f"Critical ({100 * (1 - result.alpha):.1f}%)",
        )
    ax.set_xlabel("Cyclic Frequency")
    ax.set_ylabel("Percentage Variance (%)")
    ax.set_title("Least-Squares Spectrum (LSSA)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    return fig, ax


def plot_allssa(result: AllssaResult, ax: Any = None, **kwargs: Any) -> tuple[Any, Any]:
    """Plot ALLSSA selected harmonic amplitudes."""
    plt = require_matplotlib()
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (8, 3.8)))
    else:
        fig = ax.figure

    if result.frequencies.size > 0:
        ax.stem(result.frequencies, result.amplitudes, basefmt=" ")
    ax.set_xlabel("Cyclic Frequency")
    ax.set_ylabel("Amplitude")
    ax.set_title("ALLSSA Significant Harmonics")
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_just_jumps(result: JustJumpResult, ax: Any = None, **kwargs: Any) -> tuple[Any, Any]:
    """Plot observed series with JUST detected jumps, magnitudes, and censoring markers."""
    plt = require_matplotlib()
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (9, 4.2)))
    else:
        fig = ax.figure

    _plot_observations_with_censoring(ax, result.times_raw, result.values, result.censoring)
    for k, j in enumerate(result.jumps):
        ax.axvline(
            j.time,
            color=_ORANGE,
            ls="--",
            lw=1.5,
            label="JUST jump" if k == 0 else None,
        )
        badge = f"Δ={j.magnitude:+.3g}"
        if j.coincides_with_dl_shift:
            badge += "\n(DL shift!)"
        y_top = float(np.nanmax(result.values))
        ax.text(
            j.time,
            y_top,
            badge,
            fontsize=8,
            color=_ORANGE,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": _ORANGE, "pad": 2},
        )
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("JUST Abrupt Jump Detection")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="best")
    return fig, ax


def plot_just_decompose(
    result: JustDecomposeResult, ax: Any = None, **kwargs: Any
) -> tuple[Any, Any]:
    """Plot 4-panel JUST decomposition: observations, trend, seasonal, remainder."""
    plt = require_matplotlib()
    if ax is None:
        fig, axes = plt.subplots(
            4, 1, figsize=kwargs.pop("figsize", (9, 7.5)), sharex=True
        )
    else:
        axes = ax
        fig = axes[0].figure

    _plot_observations_with_censoring(axes[0], result.times_raw, result.values, result.censoring)
    axes[0].set_ylabel("Value")
    axes[0].set_title("JUST Decomposition")
    axes[0].legend(frameon=False, loc="best", ncol=3)

    if result.trend_reg is not None and result.eval_times_raw is not None:
        axes[1].plot(result.eval_times_raw, result.trend_reg, color=_BLUE, lw=1.6, label="Trend (reg)")
    else:
        axes[1].plot(result.times_raw, result.trend, color=_BLUE, lw=1.6, label="Trend")
    axes[1].set_ylabel("Trend")

    if result.seasonal_reg is not None and result.eval_times_raw is not None:
        axes[2].plot(result.eval_times_raw, result.seasonal_reg, color=_GREEN, lw=1.3)
    else:
        axes[2].plot(result.times_raw, result.seasonal, color=_GREEN, lw=1.3)
    axes[2].axhline(0.0, color=_GRAY, lw=0.7, ls=":")
    axes[2].set_ylabel("Seasonal")

    axes[3].plot(result.times_raw, result.remainder, "o-", color=_GRAY, lw=0.7, ms=2.5)
    axes[3].axhline(0.0, color=_DARK, lw=0.7, ls=":")
    axes[3].set_ylabel("Remainder")
    axes[3].set_xlabel("Time")

    for a in axes:
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, axes


def plot_piecewise_trend(
    result: PiecewiseTrendResult, ax: Any = None, **kwargs: Any
) -> tuple[Any, Any]:
    """Plot connected piecewise-linear trend and turning points."""
    plt = require_matplotlib()
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (8.5, 4.0)))
    else:
        fig = ax.figure

    y_obs = result.fitted_trend + result.residuals
    ax.plot(result.times_raw, y_obs, "o", color=_DARK, ms=3.5, alpha=0.7, label="Series")
    ax.plot(result.times_raw, result.fitted_trend, color=_BLUE, lw=2.0, label="Piecewise trend")
    if result.tp_valid_indices.size > 0:
        ax.plot(
            result.times_raw[result.tp_valid_indices],
            result.fitted_trend[result.tp_valid_indices],
            "D",
            color=_ORANGE,
            ms=7,
            label="Turning point",
        )
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("Connected Piecewise Linear Trend")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    return fig, ax


def plot_stpd(result: StpdResult, ax: Any = None, **kwargs: Any) -> tuple[Any, Any]:
    """Plot STPD series, connected piecewise trend, turning points, and NDRI badges."""
    plt = require_matplotlib()
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (9, 4.4)))
    else:
        fig = ax.figure

    _plot_observations_with_censoring(ax, result.times_raw, result.values, result.censoring)
    ax.plot(
        result.times_raw,
        result.piecewise_trend.fitted_trend,
        color=_BLUE,
        lw=2.0,
        label="STPD trend",
    )

    for k, tp in enumerate(result.turning_points):
        y_tp = float(result.piecewise_trend.fitted_trend[tp.valid_index])
        ax.plot(
            tp.time,
            y_tp,
            "D",
            color=_GREEN,
            ms=7.5,
            zorder=5,
            label="Turning point" if k == 0 else None,
        )
        ndri_val = tp.ndri_rmse if result.ndri_mode == "rmse" else tp.ndri_l2
        badge = f"Δslope={tp.direction:+.3g}\nNDRI={ndri_val:.2f}"
        ax.annotate(
            badge,
            xy=(tp.time, y_tp),
            xytext=(6, 10),
            textcoords="offset points",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": _GREEN, "pad": 2},
        )

    for k, fj in enumerate(result.flagged_jumps):
        ax.axvline(
            fj.time,
            color=_ORANGE,
            ls=":",
            lw=1.3,
            label="Flagged high-NDRI jump" if k == 0 else None,
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("STPD Continuous Turning Point Detection")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="best")
    return fig, ax


def plot_pipeline(result: PipelineResult, ax: Any = None, **kwargs: Any) -> tuple[Any, Any]:
    """Plot 2- or 3-panel summary of the guided JUST -> STPD pipeline."""
    plt = require_matplotlib()
    has_season = result.decomposition is not None
    n_panels = 3 if has_season else 2
    if ax is None:
        fig, axes = plt.subplots(
            n_panels, 1, figsize=kwargs.pop("figsize", (9.5, 2.8 * n_panels)), sharex=True
        )
    else:
        axes = ax
        fig = axes[0].figure

    # Panel 1: Observed series + combined trend + jumps + TPs
    _plot_observations_with_censoring(axes[0], result.times_raw, result.values, result.censoring)
    axes[0].plot(result.times_raw, result.combined_trend, color=_BLUE, lw=2.0, label="Combined trend")
    if result.jumps is not None:
        for k, j in enumerate(result.jumps.jumps):
            axes[0].axvline(
                j.time, color=_ORANGE, ls="--", lw=1.4, label="JUST jump" if k == 0 else None
            )
    if result.turning_points is not None:
        for k, tp in enumerate(result.turning_points.turning_points):
            axes[0].plot(
                tp.time,
                result.combined_trend[tp.valid_index],
                "D",
                color=_GREEN,
                ms=7,
                zorder=5,
                label="STPD turning point" if k == 0 else None,
            )
    axes[0].set_ylabel("Value")
    axes[0].set_title("Guided JUST -> STPD Analysis")
    axes[0].legend(frameon=False, loc="best", ncol=3)

    # Panel 2: Deseasonalized & jump-adjusted series with STPD trend
    axes[1].plot(
        result.times_raw, result.adjusted_series, "o", color=_DARK, ms=3.0, alpha=0.65, label="Adjusted"
    )
    if result.turning_points is not None:
        axes[1].plot(
            result.times_raw,
            result.turning_points.piecewise_trend.fitted_trend,
            color=_GREEN,
            lw=1.8,
            label="STPD trend",
        )
    axes[1].set_ylabel("Adjusted")
    axes[1].legend(frameon=False, loc="best")

    if has_season and result.decomposition is not None:
        axes[2].plot(result.times_raw, result.decomposition.seasonal, color=_PURPLE, lw=1.3)
        axes[2].axhline(0.0, color=_GRAY, lw=0.7, ls=":")
        axes[2].set_ylabel("Seasonal")
        axes[2].set_xlabel("Time")
    else:
        axes[1].set_xlabel("Time")

    for a in axes:
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, axes


def plot_monitor(result: MonitorResult, ax: Any = None, **kwargs: Any) -> tuple[Any, Any]:
    """Plot 2-panel Near-Real-Time (NRT) monitoring diagnostic figure."""
    plt = require_matplotlib()
    if ax is None:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=kwargs.pop("figsize", (9.5, 5.8)),
            sharex=True,
            gridspec_kw={"height_ratios": [1.5, 1.0]},
        )
    else:
        axes = ax
        fig = axes[0].figure

    ax_top, ax_bot = axes[0], axes[1]
    n_act = len(result.values)

    # Slice censoring provenance to the active [history:] tail if present
    cen_slice: CensoringProvenance | None = None
    if result.censoring is not None:
        from dataclasses import replace

        c = result.censoring
        sl = slice(len(c.is_censored) - n_act, len(c.is_censored))
        t0_act = float(result.time_years[0]) if n_act > 0 else float("-inf")
        active_shifts = tuple(sh for sh in c.dl_shifts if sh.time_year >= t0_act)
        cen_slice = replace(
            c,
            is_censored=c.is_censored[sl],
            status_codes=c.status_codes[sl],
            original_values=c.original_values[sl],
            original_left=c.original_left[sl] if c.original_left is not None else None,
            original_right=c.original_right[sl] if c.original_right is not None else None,
            imputed_values=c.imputed_values[sl],
            dl_shifts=active_shifts,
        )

    # Shade monitoring zone on both panels
    t_end = result.times[-1]
    ax_top.axvspan(result.start_time, t_end, color=_SKY, alpha=0.14, label="Monitoring period")
    ax_bot.axvspan(result.start_time, t_end, color=_SKY, alpha=0.14)
    ax_top.axvline(result.start_time, color=_BLUE, ls=":", lw=1.2)
    ax_bot.axvline(result.start_time, color=_BLUE, ls=":", lw=1.2)

    _plot_observations_with_censoring(ax_top, result.times, result.values, cen_slice)

    # Split forecast into historical fit (solid) and forward forecast (dashed)
    n_his = int(np.searchsorted(result.time_years, float(result.steps[0].time_year), side="left")) if result.steps else n_act
    if n_his > 0:
        ax_top.plot(
            result.times[:n_his],
            result.forecast[:n_his],
            color=_BLUE,
            lw=1.8,
            label="Historical model",
        )
    if n_his < n_act:
        s0 = max(0, n_his - 1)
        ax_top.plot(
            result.times[s0:],
            result.forecast[s0:],
            color=_BLUE,
            lw=1.8,
            ls="--",
            label="Forecast",
        )

    if result.detected and result.disturbance_time is not None and result.disturbance_time_year is not None:
        pos = int(np.argmin(np.abs(result.time_years - result.disturbance_time_year)))
        col = _ORANGE if result.disturbance_type == "abrupt" else _PURPLE
        lbl = (
            f"Abrupt disturbance (mag={result.magnitude:+.3g})"
            if result.disturbance_type == "abrupt" and result.magnitude is not None
            else f"{result.disturbance_type.capitalize()} disturbance"
        )
        ax_top.plot(
            result.times[pos],
            result.values[pos],
            "*",
            color=col,
            markersize=13,
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=6,
            label=lbl,
        )
        ax_top.axvline(result.disturbance_time, color=col, ls="--", lw=1.2, alpha=0.8)
        ax_bot.axvline(result.disturbance_time, color=col, ls="--", lw=1.2, alpha=0.8)

    ax_top.set_ylabel("Value")
    ax_top.set_title(
        f"JUST Near-Real-Time Monitoring ({result.freq_correction} / {result.sequential_control}, "
        f"alpha={result.alpha:g})"
    )
    ax_top.legend(frameon=False, loc="best", ncol=2, fontsize=8.5)
    ax_top.grid(True, alpha=0.3)

    # Bottom panel: per-step low-freq & high-freq max LSS vs critical boundary
    if result.steps:
        st_times = [s.time for s in result.steps]
        low_s = [s.max_low_freq_spec for s in result.steps]
        high_s = [s.max_high_freq_spec for s in result.steps]
        crits = [s.critical_value for s in result.steps]
        ax_bot.plot(st_times, low_s, "-o", color=_ORANGE, ms=4.0, lw=1.6, label="Low-freq max LSS (<1 cy/yr, abrupt)")
        ax_bot.plot(st_times, high_s, "-s", color=_PURPLE, ms=3.5, lw=1.3, alpha=0.85, label="High-freq max LSS (>=1 cy/yr, phenological)")
        ax_bot.plot(st_times, crits, "--", color=_DARK, lw=1.5, label="Sequential critical boundary")
    ax_bot.set_ylabel("Max Residual LSS")
    ax_bot.set_xlabel("Time")
    ax_bot.set_ylim(0.0, 1.02)
    ax_bot.legend(frameon=False, loc="best", fontsize=8.5)
    ax_bot.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, axes

