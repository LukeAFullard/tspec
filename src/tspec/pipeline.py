"""Guided two-stage JUST -> STPD time-series analysis pipeline.

Implements the recommended retrospective workflow (Ghaderpour et al., 2024):
1. Preprocess timestamps, transparent NaNs, and optional ``ndimpute`` censoring.
2. Detect abrupt jumps & slope shifts via JUST (``just_jumps``).
3. Separate seasonality and trend via JUST decomposition (``just_decompose``).
4. Adjust for seasonality and abrupt vertical step offsets (or split by epoch).
5. Detect continuous trend turning points via STPD (``detect_turning_points``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ._diagnostics import DiagnosticReport, WarningItem
from ._linalg import prepare_weights
from ._time import format_timestamp
from ._types import ArrayLike, FloatArray, TimeLike, ValueLike, WeightLike
from .censored import (
    CensoredConfig,
    CensoringProvenance,
    has_censored_input,
    prepare_series_with_censoring,
)
from .exceptions import InsufficientDataError
from .just import JustDecomposeResult, JustJumpResult, just_decompose, just_jumps
from .results import (
    BaseResult,
    format_diagnostics_block,
    format_explain_warnings,
    require_pandas,
)
from .spectral import _lssa_core
from .stpd import (
    StpdResult,
    TurningPointEvent,
    fit_piecewise_trend,
)
from .stpd import (
    detect_turning_points as _detect_turning_points,
)


@dataclass(frozen=True)
class PipelineResult(BaseResult):
    """Unified result object returned by :func:`analyze_series`."""

    jumps: JustJumpResult | None
    decomposition: JustDecomposeResult | None
    turning_points: StpdResult | None
    times_numeric: FloatArray
    times_raw: np.ndarray[Any, Any]
    values: FloatArray
    adjusted_series: FloatArray
    combined_trend: FloatArray
    step_offsets: FloatArray
    is_datetime: bool
    jump_adjustment: str
    censoring: CensoringProvenance | None
    diagnostics: DiagnosticReport

    def summary(self) -> str:
        n_j = len(self.jumps.jumps) if self.jumps is not None else 0
        n_tp = len(self.turning_points.turning_points) if self.turning_points is not None else 0
        lines = [
            f"tspec Guided Pipeline Result (jump_adjustment={self.jump_adjustment!r})",
            f"Abrupt jumps (JUST)     : {n_j}",
            f"Turning points (STPD)   : {n_tp}",
        ]
        if self.censoring is not None:
            lines.append(
                f"Censoring (ndimpute)    : {self.censoring.n_censored}/{self.diagnostics.n_valid} "
                f"({self.censoring.censor_fraction:.1%}, method={self.censoring.method!r}, "
                f"dist={self.censoring.dist!r})"
            )
        if self.jumps is not None and self.jumps.jumps:
            lines.append("--- JUST Abrupt Jumps ---")
            for j in self.jumps.jumps:
                t_str = format_timestamp(j.time, self.is_datetime)
                lines.append(
                    f"  idx={j.index:<5d} time={t_str:<12s} mag={j.magnitude:+.4f} "
                    f"dir={j.direction:+.4f} p={j.jump_pvalue:.2e}"
                )
        if self.turning_points is not None and self.turning_points.turning_points:
            lines.append("--- STPD Continuous Turning Points ---")
            for tp in self.turning_points.turning_points:
                t_str = format_timestamp(tp.time, self.is_datetime)
                lines.append(
                    f"  idx={tp.index:<5d} time={t_str:<12s} "
                    f"slope {tp.slope_before:+.4f} -> {tp.slope_after:+.4f} "
                    f"NDRI={tp.ndri_rmse:.2f} p={tp.pvalue:.2e}"
                )
        lines.extend(format_diagnostics_block(self.diagnostics))
        return "\n".join(lines)

    def explain(self) -> str:
        lines: list[str] = [
            f"Analyzed {self.diagnostics.n_valid} unevenly spaced observations using the "
            "guided JUST -> STPD workflow (with zero gap interpolation)."
        ]
        if self.censoring is not None and self.censoring.n_censored > 0:
            lines.append(
                f"Censored data: {self.censoring.n_censored} observation(s) "
                f"({self.censoring.censor_fraction:.1%}) were imputed via ndimpute "
                f"({self.censoring.method}, {self.censoring.dist}, {self.censoring.impute_type})."
            )
        if self.jumps is not None:
            if self.jumps.jumps:
                lines.append(f"Stage 1 (JUST) found {len(self.jumps.jumps)} abrupt jump(s):")
                for j in self.jumps.jumps:
                    t_str = format_timestamp(j.time, self.is_datetime)
                    step_dir = "upward" if j.magnitude >= 0 else "downward"
                    msg = (
                        f"  - Around {t_str}: {step_dir} step of {j.magnitude:+.4g} "
                        f"(slope change {j.direction:+.4g}/unit time, p={j.jump_pvalue:.2e})."
                    )
                    if j.stability_rate is not None:
                        msg += f" Stable in {j.stability_rate:.0%} of censoring draws."
                    if j.coincides_with_dl_shift:
                        msg += " WARNING: coincides with a detection-limit shift!"
                    lines.append(msg)
            else:
                lines.append("Stage 1 (JUST) found no abrupt vertical jumps.")

        if self.turning_points is not None:
            if self.turning_points.turning_points:
                lines.append(
                    f"Stage 2 (STPD on deseasonalized/jump-adjusted series) found "
                    f"{len(self.turning_points.turning_points)} smooth turning point(s):"
                )
                for tp in self.turning_points.turning_points:
                    t_str = format_timestamp(tp.time, self.is_datetime)
                    lines.append(
                        f"  - Around {t_str}: slope shifted from {tp.slope_before:+.4g} "
                        f"to {tp.slope_after:+.4g}/unit time (NDRI={tp.ndri_rmse:.2f}, "
                        f"p={tp.pvalue:.2e})."
                    )
            else:
                lines.append(
                    "Stage 2 (STPD) found no additional continuous trend turning points."
                )

        lines.extend(format_explain_warnings(self.diagnostics))
        return "\n".join(lines)

    def to_dataframe(self) -> Any:
        pd = require_pandas()
        data: dict[str, Any] = {
            "time": self.times_raw,
            "time_year": self.times_numeric,
            "value": self.values,
            "adjusted_value": self.adjusted_series,
            "combined_trend": self.combined_trend,
            "step_offset": self.step_offsets,
        }
        if self.decomposition is not None:
            data["just_trend"] = self.decomposition.trend
            data["seasonal"] = self.decomposition.seasonal
            data["remainder"] = self.decomposition.remainder
        if self.censoring is not None:
            data["is_censored"] = self.censoring.is_censored
            data["censor_status"] = self.censoring.status_codes
            data["original_value"] = self.censoring.original_values
        return pd.DataFrame(data)

    def plot(self, ax: Any = None, **kwargs: Any) -> tuple[Any, Any]:
        from .plotting import plot_pipeline

        return plot_pipeline(self, ax=ax, **kwargs)


def _estimate_vertical_step_offsets(
    t_valid: FloatArray,
    y_deseas: FloatArray,
    jump_valid_indices: list[int],
    weights_valid: Any,
) -> FloatArray:
    """Estimate cumulative vertical step offsets at JUST jump indices via piecewise OLS."""
    n = int(t_valid.size)
    offsets = np.zeros(n, dtype=np.float64)
    if not jump_valid_indices:
        return offsets

    j_arr = np.sort(np.unique(np.asarray(jump_valid_indices, dtype=np.intp)))
    t0 = t_valid - t_valid[0]
    pw = prepare_weights(weights_valid, n)
    fit = _lssa_core(
        t0,
        y_deseas,
        pw,
        np.empty(0, dtype=np.float64),
        j_arr,
        0.05,
        "linear",
        False,
        np.empty(0, dtype=np.float64),
    )
    m = int(j_arr.size)
    intercepts = fit.coeff[: m + 1]
    slopes = fit.coeff[m + 1 : 2 * m + 2]

    for k, j in enumerate(j_arr):
        left_val = float(intercepts[k] + slopes[k] * t0[j])
        right_val = float(intercepts[k + 1] + slopes[k + 1] * t0[j])
        step_k = right_val - left_val
        offsets[j:] += step_k

    return offsets


def analyze_series(
    t: TimeLike,
    y: ValueLike,
    *,
    weights: WeightLike | None = None,
    detect_jumps: bool = True,
    detect_turning_points: bool = True,
    remove_seasonality: bool = True,
    jump_adjustment: Literal["offset", "segment"] = "offset",
    censor_status: ArrayLike | None = None,
    censoring: CensoredConfig | None = None,
    just_kwargs: dict[str, Any] | None = None,
    decompose_kwargs: dict[str, Any] | None = None,
    stpd_kwargs: dict[str, Any] | None = None,
) -> PipelineResult:
    """Run the guided two-stage JUST -> STPD change-detection and decomposition pipeline.

    Designed for applied practitioners and researchers who want the recommended
    end-to-end sequence without manual glue code:

    1. Handles irregular timestamps, ``NaN``s, and optional censored data
       (defaulting to 20 stochastic stability draws when censoring is detected).
    2. Detects abrupt jumps via :func:`just_jumps` (enabling scale-free
       ``jump_alpha=0.05`` screening by default).
    3. Extracts seasonality and trend via :func:`just_decompose`.
    4. Removes seasonality and vertical jump offsets (or segments by epoch)
       and runs :func:`detect_turning_points` (STPD) to isolate smooth changes
       in trend slope.
    """
    from dataclasses import replace

    if jump_adjustment not in ("offset", "segment"):
        raise ValueError("`jump_adjustment` must be 'offset' or 'segment'.")

    # Enable 20-draw censoring stability checks by default in the guided pipeline
    eff_censoring = censoring
    if eff_censoring is None and has_censored_input(y, censor_status, None):
        eff_censoring = CensoredConfig(n_stability_draws=20)

    prep = prepare_series_with_censoring(
        t, y, censor_status=censor_status, censoring=eff_censoring
    )
    series = prep.series
    prov = prep.provenance

    j_kw: dict[str, Any] = dict(just_kwargs or {})
    d_kw: dict[str, Any] = dict(decompose_kwargs or {})
    s_kw: dict[str, Any] = dict(stpd_kwargs or {})

    # Practitioner-friendly scale-free defaults in analyze_series unless overridden:
    # - Bonferroni-adjusted within-window screening p-values suppress post-selection noise.
    # - When STPD is also enabled, step_alpha isolates genuine vertical discontinuities in
    #   Stage 1 (JUST) so pure continuous slope bends pass cleanly to Stage 2 (STPD).
    if "jump_alpha" not in j_kw:
        j_kw["jump_alpha"] = 0.05
    if "bonferroni_window" not in j_kw:
        j_kw["bonferroni_window"] = True
    if detect_turning_points and "step_alpha" not in j_kw:
        j_kw["step_alpha"] = float(j_kw["jump_alpha"]) if j_kw["jump_alpha"] is not None else 0.05
    if "mag_th" not in j_kw and "dir_th" not in j_kw and j_kw["jump_alpha"] is not None:
        j_kw["mag_th"] = 0.0
        j_kw["dir_th"] = 0.0
    if not remove_seasonality:
        j_kw.setdefault("season", "none")
        d_kw.setdefault("season", "none")

    all_warnings: list[WarningItem] = []
    seen_codes: set[tuple[str, str]] = set()

    def _add_warnings(items: tuple[WarningItem, ...] | list[WarningItem]) -> None:
        for item in items:
            key = (item.code, item.message)
            if key not in seen_codes:
                seen_codes.add(key)
                all_warnings.append(item)

    _add_warnings(prep.warnings)

    # Stage 1: JUST abrupt jump detection
    jumps_res: JustJumpResult | None = None
    jump_valid_indices: list[int] = []
    if detect_jumps:
        jumps_res = just_jumps(
            t,
            y,
            weights=weights,
            censor_status=censor_status,
            censoring=eff_censoring,
            **j_kw,
        )
        jump_valid_indices = [j.valid_index for j in jumps_res.jumps]
        _add_warnings(jumps_res.diagnostics.warnings)

    # Use consistent imputed series for downstream decomposition & STPD
    y_work_valid = jumps_res.values if jumps_res is not None else series.values
    t_valid = series.times_numeric
    t_raw_valid = series.times_raw

    # Prepare weights for valid entries if passed
    w_valid_raw: Any = None
    if weights is not None:
        pw_full = prepare_weights(weights, series.n_input)
        pw_sub = pw_full.subset(series.valid_mask) if series.n_dropped_nan else pw_full
        if pw_sub.kind == "vector":
            w_valid_raw = pw_sub.vector
        elif pw_sub.kind == "matrix":
            w_valid_raw = pw_sub.matrix
        elif pw_sub.kind == "scalar":
            w_valid_raw = pw_sub.scalar

    # Stage 2: JUST decomposition (on valid non-NaN series with detected jump indices)
    decomp_res: JustDecomposeResult | None = None
    if remove_seasonality or detect_jumps:
        try:
            decomp_res = just_decompose(
                t_raw_valid,
                y_work_valid,
                weights=w_valid_raw,
                jump_indices=jump_valid_indices,
                **d_kw,
            )
            _add_warnings(decomp_res.diagnostics.warnings)
        except InsufficientDataError as exc:
            _add_warnings(
                [
                    WarningItem(
                        code="DECOMPOSE_SKIPPED",
                        severity="caution",
                        message=f"JUST decomposition skipped: {exc}",
                    )
                ]
            )

    # Stage 3: Deseasonalized & jump-adjusted series for STPD
    y_deseas = (
        y_work_valid - np.nan_to_num(decomp_res.seasonal, nan=0.0)
        if (remove_seasonality and decomp_res is not None)
        else y_work_valid.copy()
    )

    step_offsets = np.zeros(series.n_valid, dtype=np.float64)
    if jump_valid_indices and jump_adjustment == "offset":
        step_offsets = _estimate_vertical_step_offsets(
            t_valid, y_deseas, jump_valid_indices, w_valid_raw
        )
        y_adj = y_deseas - step_offsets
    else:
        y_adj = y_deseas.copy()

    # Stage 4: STPD continuous turning-point detection
    stpd_res: StpdResult | None = None
    w_stpd = w_valid_raw if (w_valid_raw is not None and np.ndim(w_valid_raw) <= 1) else None
    if detect_turning_points:
        if jump_adjustment == "segment" and jump_valid_indices:
            bounds = [0, *sorted(jump_valid_indices), series.n_valid]
            seg_tps: list[TurningPointEvent] = []
            seg_valid_idx: list[int] = []
            seg_fitted = np.empty(series.n_valid, dtype=np.float64)
            for seg_i in range(len(bounds) - 1):
                s0, s1 = bounds[seg_i], bounds[seg_i + 1]
                seg_w = (
                    w_stpd[s0:s1]
                    if (w_stpd is not None and np.ndim(w_stpd) == 1)
                    else w_stpd
                )
                if s1 - s0 >= 12:
                    try:
                        seg_res = _detect_turning_points(
                            t_raw_valid[s0:s1],
                            y_adj[s0:s1],
                            weights=seg_w,
                            **s_kw,
                        )
                        seg_fitted[s0:s1] = seg_res.piecewise_trend.fitted_trend
                        for tp in seg_res.turning_points:
                            g_valid = s0 + int(tp.valid_index)
                            seg_valid_idx.append(g_valid)
                            seg_tps.append(
                                replace(
                                    tp,
                                    valid_index=g_valid,
                                    index=int(series.valid_to_orig[g_valid]),
                                )
                            )
                        _add_warnings(seg_res.diagnostics.warnings)
                        continue
                    except InsufficientDataError:
                        pass
                pw_seg = fit_piecewise_trend(t_raw_valid[s0:s1], y_adj[s0:s1], [], weights=seg_w)
                seg_fitted[s0:s1] = pw_seg.fitted_trend

            pw_global = fit_piecewise_trend(
                t_raw_valid,
                y_adj,
                seg_valid_idx,
                weights=w_stpd,
                mode=s_kw.get("refit_mode", "sequential"),
            )
            pw_global = replace(
                pw_global,
                fitted_trend=seg_fitted,
                residuals=y_adj - seg_fitted,
            )
            stpd_res = StpdResult(
                turning_points=tuple(seg_tps),
                flagged_jumps=(),
                piecewise_trend=pw_global,
                times_numeric=t_valid,
                times_raw=t_raw_valid,
                values=y_adj,
                is_datetime=series.is_datetime,
                window_mode=str(s_kw.get("window_mode", "count")),
                snr_mode=str(s_kw.get("snr_mode", "centered")),
                ndri_mode=str(s_kw.get("ndri_mode", "rmse")),
                alpha=float(s_kw.get("alpha", 0.01)),
                snr_th=float(s_kw.get("snr_th", 1.0)),
                ndri_th=float(s_kw.get("ndri_th", 0.3)),
                dir_th=float(s_kw.get("dir_th", 0.0)),
                tp_th=float(s_kw.get("tp_th", 1.0)),
                censoring=prov,
                diagnostics=pw_global.diagnostics,
            )
        else:
            try:
                raw_stpd = _detect_turning_points(
                    t_raw_valid,
                    y_adj,
                    weights=w_stpd,
                    **s_kw,
                )
                # Exclude turning points sitting directly on an already-detected JUST jump
                if jump_valid_indices:
                    j_th = float(j_kw.get("jump_th", 0.75))
                    excl_tol = 0.5 * j_th
                    jump_times = [float(t_valid[j]) for j in jump_valid_indices]
                    kept_tps = tuple(
                        tp
                        for tp in raw_stpd.turning_points
                        if all(abs(tp.time_year - jt) > excl_tol for jt in jump_times)
                    )
                    if len(kept_tps) != len(raw_stpd.turning_points):
                        kept_valid_idx = [tp.valid_index for tp in kept_tps]
                        pw_refit = fit_piecewise_trend(
                            t_raw_valid,
                            y_adj,
                            kept_valid_idx,
                            weights=w_stpd,
                            mode=s_kw.get("refit_mode", "sequential"),
                        )
                        remapped_tps = tuple(
                            replace(
                                tp,
                                index=int(series.valid_to_orig[tp.valid_index]),
                                slope_before=float(pw_refit.slopes[k]),
                                slope_after=float(pw_refit.slopes[k + 1]),
                                direction=float(pw_refit.directions[k]),
                            )
                            for k, tp in enumerate(kept_tps)
                        )
                        stpd_res = replace(
                            raw_stpd,
                            turning_points=remapped_tps,
                            piecewise_trend=pw_refit,
                        )
                    else:
                        remapped_tps = tuple(
                            replace(tp, index=int(series.valid_to_orig[tp.valid_index]))
                            for tp in raw_stpd.turning_points
                        )
                        stpd_res = replace(raw_stpd, turning_points=remapped_tps)
                else:
                    remapped_tps = tuple(
                        replace(tp, index=int(series.valid_to_orig[tp.valid_index]))
                        for tp in raw_stpd.turning_points
                    )
                    stpd_res = replace(raw_stpd, turning_points=remapped_tps)
                _add_warnings(stpd_res.diagnostics.warnings)
            except InsufficientDataError as exc:
                _add_warnings(
                    [
                        WarningItem(
                            code="STPD_SKIPPED",
                            severity="caution",
                            message=f"STPD turning-point detection skipped: {exc}",
                        )
                    ]
                )

    if stpd_res is not None:
        combined_trend = stpd_res.piecewise_trend.fitted_trend + step_offsets
        cond_max = max(
            stpd_res.diagnostics.condition_number_max,
            jumps_res.diagnostics.condition_number_max if jumps_res is not None else 1.0,
            decomp_res.diagnostics.condition_number_max if decomp_res is not None else 1.0,
        )
    elif decomp_res is not None:
        combined_trend = decomp_res.trend
        cond_max = decomp_res.diagnostics.condition_number_max
    else:
        combined_trend = y_work_valid.copy()
        cond_max = jumps_res.diagnostics.condition_number_max if jumps_res is not None else 1.0

    # Replace intermediate-stage autocorrelation warnings with one clean final check
    # on the combined residuals (after both JUST jumps/seasonality and STPD trend).
    from ._diagnostics import compute_red_noise_diagnostics

    seasonal_part = (
        np.nan_to_num(decomp_res.seasonal, nan=0.0)
        if (remove_seasonality and decomp_res is not None)
        else np.zeros_like(y_work_valid)
    )
    final_residuals = y_work_valid - combined_trend - seasonal_part
    rn_final, final_rn_warns = compute_red_noise_diagnostics(t_valid, final_residuals)
    filtered_warnings = [
        w
        for w in all_warnings
        if w.code not in ("MODERATE_AUTOCORRELATION", "STRONG_AUTOCORRELATION")
    ]
    filtered_warnings.extend(final_rn_warns)

    diag = DiagnosticReport(
        red_noise=rn_final,
        condition_number_max=cond_max,
        n_input=series.n_input,
        n_valid=series.n_valid,
        n_dropped_nan=series.n_dropped_nan,
        sampling_rate_ratio=series.sampling_rate_ratio,
        warnings=tuple(filtered_warnings),
    )

    return PipelineResult(
        jumps=jumps_res,
        decomposition=decomp_res,
        turning_points=stpd_res,
        times_numeric=t_valid,
        times_raw=t_raw_valid,
        values=y_work_valid,
        adjusted_series=y_adj,
        combined_trend=combined_trend,
        step_offsets=step_offsets,
        is_datetime=series.is_datetime,
        jump_adjustment=jump_adjustment,
        censoring=prov,
        diagnostics=diag,
    )
