"""Guided two-stage JUST -> STPD time-series analysis pipeline.

Implements the recommended retrospective workflow (Ghaderpour et al., 2024):
1. Preprocess timestamps, transparent NaNs, and optional ``ndimpute`` censoring.
2. Detect abrupt jumps & slope shifts via JUST (``just_jumps``).
3. Separate seasonality and trend via JUST decomposition (``just_decompose``).
4. Adjust for seasonality and abrupt vertical step offsets (or split by epoch).
5. Detect continuous trend turning points via STPD (``detect_turning_points``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from scipy.stats import t as t_dist

from ._diagnostics import DiagnosticReport, WarningItem, compute_red_noise_diagnostics
from ._linalg import PreparedWeights, prepare_weights, solve_weighted_ols
from ._robust import (
    RobustMethod,
    SlopeEstimator,
    fit_l1_piecewise,
    fit_theil_sen_segment,
    solve_huber_irls,
    solve_mm_biweight,
    tukey_weight,
    validate_robust_args,
)
from ._serial import (
    SerialMode,
    SerialSpec,
    deflate_t_pvalue,
    resolve_serial_spec,
    sandwich_vif,
    serial_warning_message,
    validate_serial_mode,
)
from ._time import format_timestamp
from ._types import ArrayLike, FloatArray, IntArray, TimeLike, ValueLike, WeightLike
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
from .spectral import _lssa_core, build_design_matrix
from .stpd import (
    SNR_TH_DEFAULT,
    FlaggedJumpCandidate,
    PiecewiseTrendResult,
    RefitMode,
    StpdResult,
    TurningPointEvent,
    _batched_window_knots,
    _bootstrap_tp_curves,
    _fit_piecewise_core,
    _tpd_window,
    fit_piecewise_trend,
)
from .stpd import (
    detect_turning_points as _detect_turning_points,
)


@dataclass(frozen=True)
class SegmentSummary:
    """Per-segment linear trend estimate across detected jumps and turning points.

    Attributes
    ----------
    segment_index : int
        0-based segment number.
    start_index, end_index : int
        First/last observation of the segment on the input axis.
    start_valid_index, end_valid_index : int
        The same on the valid (NaN-dropped) axis.
    start_time, end_time : Any
        Segment bounds in the original time representation.
    start_time_year, end_time_year : float
        Segment bounds in decimal years.
    n_obs, n_censored : int
        Observations and censored observations in the segment.
    estimator : {"ols", "ols_mi", "sen", "ats", "l1", "huber"}
        Estimator used (``"ols_mi"`` = Rubin-pooled multiple imputation over censored
        values; ``"ats"`` = Akritas-Theil-Sen).
    slope, intercept : float
        Slope per time unit and intercept. For ``"ols"``/``"ols_mi"``/``"sen"`` each
        segment is fitted separately (the fits need not meet at turning points), so
        these slopes can differ slightly from ``TurningPointEvent.slope_before/after``,
        which come from the connected piecewise refit. ``"huber"`` and ``"l1"`` segment
        slopes come from a connected fit within each epoch.
    slope_ci_low, slope_ci_high : float or None
        Confidence interval for the slope (conditional on the segment boundaries).
    pvalue : float or None
        Two-sided p-value for zero slope (Mann-Kendall for ``"sen"``/``"ats"``).
    """

    segment_index: int
    start_index: int
    end_index: int
    start_valid_index: int
    end_valid_index: int
    start_time: Any
    end_time: Any
    start_time_year: float
    end_time_year: float
    n_obs: int
    n_censored: int
    estimator: Literal["ols", "ols_mi", "sen", "ats", "l1", "huber"]
    slope: float
    intercept: float
    slope_ci_low: float | None
    slope_ci_high: float | None
    pvalue: float | None


@dataclass(frozen=True)
class PipelineResult(BaseResult):
    """Unified result object returned by :func:`analyze_series`.

    Attributes
    ----------
    jumps, decomposition, turning_points
        Stage results (:class:`JustJumpResult`, :class:`JustDecomposeResult`,
        :class:`StpdResult`), or ``None`` when a stage was skipped.
    times_numeric, times_raw, values, is_datetime
        Valid timestamps and (imputed) values.
    adjusted_series : ndarray
        Deseasonalized and jump-offset-adjusted series passed to STPD.
    combined_trend : ndarray
        Final trend on the original scale (STPD trend + step offsets, or the
        estimator-specific segment fits).
    step_offsets : ndarray
        Cumulative vertical step offsets estimated at the JUST jumps.
    jump_adjustment : str
        ``"offset"`` or ``"segment"``.
    censoring : CensoringProvenance or None
        Censored-data provenance.
    diagnostics : DiagnosticReport
        Consolidated warnings and final red-noise diagnostics.
    segments : tuple of SegmentSummary
        Per-segment slope, CI and p-value.
    robust, slope_estimator : str
        Settings used.
    ci_level : float
        Confidence level.
    trend_ci_low, trend_ci_high : ndarray or None
        Pointwise trend band, conditional on the detected structure (``None`` for
        ``"sen"`` and ``"l1"``).
    slope_fan_low, slope_fan_high : ndarray or None
        Median-pivoted Sen/ATS slope envelope for ``"sen"``. Not a pointwise trend
        CI: it has zero width at each segment's median time.
    band_method : str
        How ``trend_ci_*`` and the OLS segment-slope CIs were computed: ``"bootstrap"``
        (detect-then-fit bootstrap including turning-point location, jump-step and
        seasonal-estimation uncertainty) or ``"conditional"`` (treats the detected
        structure as known).
    """

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
    segments: tuple[SegmentSummary, ...] = ()
    robust: str = "none"
    slope_estimator: str = "ols"
    ci_level: float = 0.95
    trend_ci_low: FloatArray | None = None
    trend_ci_high: FloatArray | None = None
    slope_fan_low: FloatArray | None = None
    slope_fan_high: FloatArray | None = None
    band_method: str = "conditional"
    requested_band_method: str = "bootstrap"
    serial_correlation: str = "none"
    serial_rho: float = 0.0
    serial_rho_used: float = 0.0

    def summary(self) -> str:
        n_j = len(self.jumps.jumps) if self.jumps is not None else 0
        n_tp = len(self.turning_points.turning_points) if self.turning_points is not None else 0
        ci_pct = f"{100.0 * self.ci_level:g}"
        lines = [
            f"tspec Guided Pipeline Result (jump_adjustment={self.jump_adjustment!r}, "
            f"robust={self.robust!r}, slope_estimator={self.slope_estimator!r})",
            f"Abrupt jumps (JUST)     : {n_j}",
            f"Turning points (STPD)   : {n_tp}",
            f"Trend segments          : {len(self.segments)}",
        ]
        if self.censoring is not None:
            cen_engine = (
                "Tobit MLE"
                if self.censoring.method == "censored_regression"
                else "ndimpute"
            )
            lines.append(
                f"Censoring ({cen_engine:<9s})   : {self.censoring.n_censored}/{self.diagnostics.n_valid} "
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
        if self.segments:
            lines.append("--- Segment Trend Estimates ---")
            for seg in self.segments:
                t0_s = format_timestamp(seg.start_time, self.is_datetime)
                t1_s = format_timestamp(seg.end_time, self.is_datetime)
                ci_s = (
                    f"[{seg.slope_ci_low:+.4f}, {seg.slope_ci_high:+.4f}]"
                    if (seg.slope_ci_low is not None and seg.slope_ci_high is not None)
                    else "n/a"
                )
                p_s = f"{seg.pvalue:.2e}" if seg.pvalue is not None else "n/a"
                lines.append(
                    f"  seg={seg.segment_index:<2d} [{t0_s}..{t1_s}] n={seg.n_obs:<4d} "
                    f"({seg.estimator}) slope={seg.slope:+.4f} CI{ci_pct}={ci_s} p={p_s}"
                )
        lines.extend(format_diagnostics_block(self.diagnostics))
        return "\n".join(lines)

    def explain(self) -> str:
        ci_pct = f"{100.0 * self.ci_level:g}"
        lines: list[str] = [
            f"Analyzed {self.diagnostics.n_valid} unevenly spaced observations using the "
            f"guided JUST -> STPD workflow (robust={self.robust!r}, "
            f"slope_estimator={self.slope_estimator!r}, zero gap interpolation)."
        ]
        if self.censoring is not None and self.censoring.n_censored > 0:
            cen_desc = (
                f"tspec censored-regression MLE ({self.censoring.dist}, {self.censoring.impute_type})"
                if self.censoring.method == "censored_regression"
                else f"ndimpute ({self.censoring.method}, {self.censoring.dist}, {self.censoring.impute_type})"
            )
            lines.append(
                f"Censored data: {self.censoring.n_censored} observation(s) "
                f"({self.censoring.censor_fraction:.1%}) were imputed via {cen_desc}."
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
                        f"to {tp.slope_after:+.4g}/unit time (p={tp.pvalue:.2e}, "
                        f"step-vs-bend p={tp.step_pvalue:.2g}, NDRI={tp.ndri_rmse:.2f})."
                    )
            else:
                lines.append(
                    "Stage 2 (STPD) found no additional continuous trend turning points."
                )
            if self.turning_points.flagged_jumps:
                fl = ", ".join(
                    format_timestamp(fj.time, self.is_datetime)
                    for fj in self.turning_points.flagged_jumps[:4]
                )
                lines.append(
                    f"WARNING: STPD found {len(self.turning_points.flagged_jumps)} significant "
                    f"break(s) that look like abrupt vertical steps rather than smooth bends ({fl}) "
                    "but were not detected by Stage 1 (JUST). They are NOT reported as turning "
                    "points; the trend and segment slopes do not account for them. Consider a longer "
                    "JUST window (just_kwargs=dict(size=...)) or supplying known jump dates."
                )

        if self.segments:
            lines.append(f"Segment trend slopes ({len(self.segments)} segment(s)):")
            for seg in self.segments:
                t0_s = format_timestamp(seg.start_time, self.is_datetime)
                t1_s = format_timestamp(seg.end_time, self.is_datetime)
                msg = (
                    f"  - Segment {seg.segment_index} ({t0_s} to {t1_s}, n={seg.n_obs}, "
                    f"{seg.estimator}): slope = {seg.slope:+.4g}/unit time"
                )
                if seg.slope_ci_low is not None and seg.slope_ci_high is not None:
                    msg += f" ({ci_pct}% CI [{seg.slope_ci_low:+.4g}, {seg.slope_ci_high:+.4g}])"
                if seg.pvalue is not None:
                    msg += f", p={seg.pvalue:.2e}"
                lines.append(msg + ".")
            if self.band_method == "bootstrap":
                has_mi = any(s.estimator == "ols_mi" for s in self.segments)
                mi_note = (
                    " (censored 'ols_mi' segments retain Rubin-pooled multiple-imputation CIs)"
                    if has_mi
                    else ""
                )
                lines.append(
                    f"Segment CIs and the trend band are detect-then-fit bootstrap intervals{mi_note}: "
                    "they include turning-point location, jump-step and seasonal-estimation "
                    "uncertainty, but are conditional on the number of breakpoints found."
                )
            elif len(self.segments) > 1 or self.trend_ci_low is not None:
                if self.requested_band_method == "bootstrap":
                    lines.append(
                        f"Note: conditional/analytical intervals were used (slope_estimator={self.slope_estimator!r}, "
                        f"jump_adjustment={self.jump_adjustment!r}) because bootstrap trend bands apply to "
                        "OLS/Huber STPD trend fits."
                    )
                else:
                    lines.append(
                        "Note: segment CIs and the trend band are conditional on the detected "
                        "breakpoints and omit their location uncertainty (in simulation the trend "
                        "band covered about 83-92% at nominal 95% even when the structure was found "
                        "correctly); band_method='bootstrap' includes it."
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
        if self.trend_ci_low is not None and self.trend_ci_high is not None:
            data["trend_ci_low"] = self.trend_ci_low
            data["trend_ci_high"] = self.trend_ci_high
        if self.slope_fan_low is not None and self.slope_fan_high is not None:
            data["slope_fan_low"] = self.slope_fan_low
            data["slope_fan_high"] = self.slope_fan_high
        if self.decomposition is not None:
            data["just_trend"] = self.decomposition.trend
            data["seasonal"] = self.decomposition.seasonal
            if (
                self.decomposition.seasonal_ci_low is not None
                and self.decomposition.seasonal_ci_high is not None
            ):
                data["seasonal_ci_low"] = self.decomposition.seasonal_ci_low
                data["seasonal_ci_high"] = self.decomposition.seasonal_ci_high
            data["remainder"] = self.decomposition.remainder
        if self.censoring is not None:
            data["is_censored"] = self.censoring.is_censored
            data["censor_status"] = self.censoring.status_codes
            data["original_value"] = self.censoring.original_values
        return pd.DataFrame(data)

    def segments_to_dataframe(self) -> Any:
        """Return per-segment trend slopes, CIs, and p-values as a pandas DataFrame."""
        pd = require_pandas()
        rows = [
            {
                "segment_index": s.segment_index,
                "start_index": s.start_index,
                "end_index": s.end_index,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "start_time_year": s.start_time_year,
                "end_time_year": s.end_time_year,
                "n_obs": s.n_obs,
                "n_censored": s.n_censored,
                "estimator": s.estimator,
                "slope": s.slope,
                "intercept": s.intercept,
                "slope_ci_low": s.slope_ci_low,
                "slope_ci_high": s.slope_ci_high,
                "pvalue": s.pvalue,
            }
            for s in self.segments
        ]
        return pd.DataFrame(rows)

    def plot(self, ax: Any = None, **kwargs: Any) -> tuple[Any, Any]:
        from .plotting import plot_pipeline

        return plot_pipeline(self, ax=ax, **kwargs)


def _estimate_vertical_step_offsets(
    t_valid: FloatArray,
    y_deseas: FloatArray,
    jump_valid_indices: list[int],
    weights_valid: Any,
    *,
    robust: RobustMethod = "none",
    huber_k: float = 1.345,
    local_half_window: int | None = None,
    tp_valid_indices: list[int] | None = None,
) -> FloatArray:
    """Estimate cumulative vertical step offsets at JUST jump indices.

    Before STPD runs (``tp_valid_indices is None``), each jump's vertical step is
    estimated on a local window around the jump (clamped inside adjacent jumps) so
    that an undetected turning point elsewhere in an inter-jump epoch cannot tilt
    the fitted line and corrupt the step offset (BLK-01). Once STPD has identified
    ``tp_valid_indices``, passing them performs a unified joint fit of vertical
    step indicators and continuous trend hinges over the full series.
    """
    n = int(t_valid.size)
    offsets = np.zeros(n, dtype=np.float64)
    if not jump_valid_indices:
        return offsets

    j_arr = np.sort(np.unique(np.asarray(jump_valid_indices, dtype=np.intp)))
    j_arr = j_arr[(j_arr > 0) & (j_arr < n)]
    if j_arr.size == 0:
        return offsets
    t0 = t_valid - t_valid[0]
    pw_full = prepare_weights(weights_valid, n)

    if tp_valid_indices is not None:
        # Unified post-STPD joint model: [1, u, (u - u_tp)_+, ..., I(t >= t_j), ...]
        s_t = float(t0[-1]) if float(t0[-1]) > 0.0 else 1.0
        u = t0 / s_t
        tp_arr = np.sort(
            np.unique(
                np.asarray(
                    [tp for tp in tp_valid_indices if 0 < tp < n and int(tp) not in set(j_arr.tolist())],
                    dtype=np.intp,
                )
            )
        )
        cols: list[FloatArray] = [np.ones(n, dtype=np.float64), u]
        for tp in tp_arr:
            cols.append(np.maximum(0.0, u - u[int(tp)]))
        for j in j_arr:
            step_col = np.zeros(n, dtype=np.float64)
            step_col[int(j) :] = 1.0
            cols.append(step_col)
        D_joint = np.column_stack(cols)
        if robust == "mm":
            coeff = solve_mm_biweight(D_joint, y_deseas, pw_full, k=huber_k, compute_cov=False).ols.coeff
        elif robust == "huber":
            coeff = solve_huber_irls(D_joint, y_deseas, pw_full, k=huber_k, compute_cov=False).ols.coeff
        else:
            coeff = solve_weighted_ols(D_joint, y_deseas, pw_full, compute_cov=False).coeff
        m_j = int(j_arr.size)
        step_coefs = coeff[-m_j:]
        for k, j in enumerate(j_arr):
            offsets[int(j) :] += float(step_coefs[k])
        return offsets

    # Pre-STPD local two-piece linear step estimation around each jump
    half_w = (
        max(6, int(local_half_window))
        if local_half_window is not None
        else max(10, min(20, n // 6))
    )
    m = int(j_arr.size)
    for k, j in enumerate(j_arr):
        j_int = int(j)
        prev_j = int(j_arr[k - 1]) if k > 0 else 0
        next_j = int(j_arr[k + 1]) if k + 1 < m else n
        lo = max(prev_j, j_int - half_w)
        hi = min(next_j, j_int + half_w)
        j_loc = j_int - lo
        n_loc = hi - lo
        if j_loc < 1 or (n_loc - j_loc) < 1:
            continue
        t_loc = t0[lo:hi] - t0[lo]
        y_loc = y_deseas[lo:hi]
        pw_loc = pw_full.subset(np.arange(lo, hi, dtype=np.intp))
        if j_loc >= 3 and (n_loc - j_loc) >= 3 and float(t_loc[j_loc - 1]) > 0.0 and float(t_loc[-1] - t_loc[j_loc]) > 0.0:
            fit_loc = _lssa_core(
                t_loc,
                y_loc,
                pw_loc,
                np.empty(0, dtype=np.float64),
                np.array([j_loc], dtype=np.intp),
                0.05,
                "linear",
                False,
                np.empty(0, dtype=np.float64),
                robust=robust,
                huber_k=huber_k,
            )
            left_val = float(fit_loc.coeff[0] + fit_loc.coeff[2] * t_loc[j_loc])
            right_val = float(fit_loc.coeff[1] + fit_loc.coeff[3] * t_loc[j_loc])
            step_k = right_val - left_val
        else:
            step_k = float(np.median(y_loc[j_loc:]) - np.median(y_loc[:j_loc]))
        offsets[j_int:] += step_k

    return offsets


def _mi_segment_slope(
    t_s: FloatArray,
    prov: CensoringProvenance,
    seasonal_part: FloatArray,
    s0: int,
    s1: int,
    D_s: FloatArray,
    s_t: float,
    pw_s: PreparedWeights,
    alpha: float,
    m: int,
    seed: int,
) -> tuple[float, float, float, float, float] | None:
    """Rubin-pooled OLS slope over proper multiple imputations of a censored segment.

    Imputation model: censored linear regression in time on the deseasonalized
    values and limits using the resolved censoring distribution (``"lognormal"``
    when requested/resolved and strictly positive, else ``"normal"``).
    Returns ``(slope, intercept, ci_low, ci_high, pvalue)`` or ``None`` on failure.
    """
    from ._cenreg import (
        fit_censored_regression_auto,
        impute_censored_regression,
        rubin_pool,
    )
    from .censored import _resolve_trend_dist

    codes = prov.status_codes[s0:s1]
    seas = seasonal_part[s0:s1]
    vals = prov.original_values[s0:s1] - seas
    left = prov.original_left[s0:s1] - seas if prov.original_left is not None else None
    right = prov.original_right[s0:s1] - seas if prov.original_right is not None else None
    w = pw_s.vector if pw_s.kind == "vector" else None
    n_s = int(t_s.size)

    # Honor resolved censoring distribution (BLK-03): use lognormal whenever requested
    # or auto-resolved AND segment deseasonalized values/limits remain strictly positive.
    req_dist = prov.dist if prov.dist in ("normal", "lognormal") else "auto"
    if req_dist == "lognormal":
        seg_dist = _resolve_trend_dist(vals, codes, left, right, "auto")
    elif req_dist == "normal":
        seg_dist = "normal"
    else:
        seg_dist = "auto"

    try:
        fit, used_dist = fit_censored_regression_auto(
            vals, codes, D_s, left=left, right=right, weights=w, dist=seg_dist
        )
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not fit.converged:
        return None
    rng = np.random.default_rng(seed)
    slopes = np.empty(m)
    vars_ = np.empty(m)
    icpts = np.empty(m)
    # Precompute weighted leverage/bread for HC3 sandwich variance under heteroscedastic lognormal Y
    use_hc3 = used_dist == "lognormal" and pw_s.kind != "matrix" and n_s > 3
    if use_hc3:
        sw_hc = pw_s.sqrt_vector if pw_s.sqrt_vector is not None else np.ones(n_s, dtype=np.float64)
        Dw_hc = D_s * sw_hc[:, None]
        xtx_inv = np.linalg.pinv(Dw_hc.T @ Dw_hc)
        h_diag = np.clip(np.sum((Dw_hc @ xtx_inv) * Dw_hc, axis=1), 0.0, 0.99)
        bread_slope = xtx_inv[1, :] @ Dw_hc.T  # shape (n_s,)
    for i in range(m):
        y_i = impute_censored_regression(
            fit, vals, codes, D_s, rng, left=left, right=right, weights=w, draw_params=True
        )
        ols_i = solve_weighted_ols(D_s, y_i, pw_s, compute_cov=True)
        slopes[i] = float(ols_i.coeff[1]) / s_t
        if use_hc3:
            e_w = ols_i.residuals * sw_hc
            u_hc3 = e_w / np.maximum(1.0 - h_diag, 0.05)
            vars_[i] = float(np.sum((bread_slope * u_hc3) ** 2)) / (s_t * s_t)
        else:
            vars_[i] = float(ols_i.cov[1, 1]) / (s_t * s_t)
        icpts[i] = float(ols_i.coeff[0]) - slopes[i] * float(t_s[0])
    if not (np.all(np.isfinite(slopes)) and np.all(np.isfinite(vars_))):
        return None
    q, t_var, df = rubin_pool(slopes, vars_, float(n_s - 2))
    se = float(np.sqrt(max(t_var, 0.0)))
    if se <= 0.0:
        return q, float(np.mean(icpts)), q, q, (0.0 if q != 0.0 else 1.0)
    t_crit = float(t_dist.ppf(1.0 - 0.5 * alpha, df))
    pval = float(2.0 * t_dist.sf(abs(q) / se, df))
    return q, float(np.mean(icpts)), q - t_crit * se, q + t_crit * se, pval


def _build_segment_summaries(
    t_valid: FloatArray,
    t_raw_valid: np.ndarray[Any, Any],
    valid_to_orig: IntArray,
    y_deseas: FloatArray,
    seasonal_part: FloatArray,
    jump_valid_indices: list[int],
    tp_valid_indices: list[int],
    w_stpd: Any,
    *,
    slope_estimator: SlopeEstimator,
    huber_k: float,
    prov: CensoringProvenance | None,
    alpha: float = 0.05,
    mk_correction: Literal["none", "hamed_rao", "ar1"] = "ar1",
    n_imputations: int = 20,
    mi_seed: int = 0,
    mi_warnings: list[WarningItem] | None = None,
) -> tuple[tuple[SegmentSummary, ...], FloatArray | None, FloatArray | None, FloatArray | None]:
    """Compute per-segment trend slopes, CIs, p-values, and pointwise trend CI bands.

    For ``slope_estimator="ols"`` segments containing censored observations, slopes,
    CIs and p-values come from ``n_imputations`` proper multiple imputations under a
    segment-congenial censored linear regression (deseasonalized values and limits),
    pooled with Rubin's rules and Barnard-Rubin degrees of freedom (``estimator="ols_mi"``).
    """
    n = int(t_valid.size)
    if n == 0:
        return (), None, None, None

    jump_bps = sorted({j for j in jump_valid_indices if 0 < j < n})
    tp_bps = sorted({tp for tp in tp_valid_indices if 0 < tp < n and tp not in jump_bps})
    epoch_bounds = [0, *jump_bps, n]

    segments: list[SegmentSummary] = []
    override_trend: FloatArray | None = (
        np.empty(n, dtype=np.float64) if slope_estimator in ("sen", "l1", "huber") else None
    )
    override_ci_low: FloatArray | None = (
        np.full(n, np.nan, dtype=np.float64) if slope_estimator in ("sen", "huber") else None
    )
    override_ci_high: FloatArray | None = (
        np.full(n, np.nan, dtype=np.float64) if slope_estimator in ("sen", "huber") else None
    )
    seg_counter = 0

    def _slice_w(a: int, b: int) -> Any:
        if w_stpd is not None and np.ndim(w_stpd) == 1:
            return w_stpd[a:b]
        return w_stpd

    def _make_seg(
        s0: int,
        s1: int,
        est: Literal["ols", "ols_mi", "sen", "ats", "l1", "huber"],
        slope: float,
        intercept: float,
        ci_low: float | None,
        ci_high: float | None,
        pval: float | None,
    ) -> SegmentSummary:
        nonlocal seg_counter
        n_cen = int(np.sum(prov.is_censored[s0:s1])) if prov is not None else 0
        seg = SegmentSummary(
            segment_index=seg_counter,
            start_index=int(valid_to_orig[s0]),
            end_index=int(valid_to_orig[s1 - 1]),
            start_valid_index=s0,
            end_valid_index=s1 - 1,
            start_time=t_raw_valid[s0],
            end_time=t_raw_valid[s1 - 1],
            start_time_year=float(t_valid[s0]),
            end_time_year=float(t_valid[s1 - 1]),
            n_obs=s1 - s0,
            n_censored=n_cen,
            estimator=est,
            slope=slope,
            intercept=intercept,
            slope_ci_low=ci_low,
            slope_ci_high=ci_high,
            pvalue=pval,
        )
        seg_counter += 1
        return seg

    for ep_i in range(len(epoch_bounds) - 1):
        e0, e1 = epoch_bounds[ep_i], epoch_bounds[ep_i + 1]
        if e1 <= e0:
            continue
        tp_epoch = [tp for tp in tp_bps if e0 < tp < e1]
        sub_bounds = [e0, *tp_epoch, e1]

        if slope_estimator == "sen":
            assert (
                override_trend is not None
                and override_ci_low is not None
                and override_ci_high is not None
            )
            for k in range(len(sub_bounds) - 1):
                s0, s1 = sub_bounds[k], sub_bounds[k + 1]
                seg_codes = prov.status_codes[s0:s1] if prov is not None else None
                seg_orig = (
                    prov.original_values[s0:s1] - seasonal_part[s0:s1]
                    if prov is not None
                    else None
                )
                ts_fit = fit_theil_sen_segment(
                    t_valid[s0:s1],
                    y_deseas[s0:s1],
                    alpha=alpha,
                    status_codes=seg_codes,
                    original_values=seg_orig,
                    autocorr_correction=mk_correction,
                )
                if mi_warnings is not None:
                    n_seg = s1 - s0
                    seg_has_cen = seg_codes is not None and bool(np.any(seg_codes != 0))
                    if ts_fit.ci_low is None and n_seg > 2000:
                        mi_warnings.append(
                            WarningItem(
                                code="SEN_CI_UNAVAILABLE",
                                severity="caution",
                                message=(
                                    f"Segment {k} has {n_seg} observations (> 2000): its Theil-Sen slope "
                                    "uses sampled pairs and no CI or Mann-Kendall p-value is reported."
                                ),
                            )
                        )
                    if seg_has_cen and ts_fit.estimator == "sen":
                        mi_warnings.append(
                            WarningItem(
                                code="ATS_UNAVAILABLE",
                                severity="caution",
                                message=(
                                    f"Segment {k} contains censored values but Akritas-Theil-Sen was not "
                                    "applicable (interval censoring or > 2000 observations); Theil-Sen was "
                                    "computed on imputed values instead."
                                ),
                            )
                        )
                t_seg = t_valid[s0:s1]
                override_trend[s0:s1] = ts_fit.intercept + ts_fit.slope * t_seg
                if ts_fit.ci_low is not None and ts_fit.ci_high is not None:
                    # Median-pivoted Sen/ATS slope-CI fan (Helsel 2012 / MannKS convention)
                    t_med = float(np.median(t_seg))
                    y_med = ts_fit.intercept + ts_fit.slope * t_med
                    dt_seg = t_seg - t_med
                    b_lo = y_med + ts_fit.ci_low * dt_seg
                    b_hi = y_med + ts_fit.ci_high * dt_seg
                    override_ci_low[s0:s1] = np.minimum(b_lo, b_hi)
                    override_ci_high[s0:s1] = np.maximum(b_lo, b_hi)
                segments.append(
                    _make_seg(
                        s0,
                        s1,
                        ts_fit.estimator,
                        ts_fit.slope,
                        ts_fit.intercept,
                        ts_fit.ci_low,
                        ts_fit.ci_high,
                        ts_fit.pvalue,
                    )
                )
        elif slope_estimator == "l1":
            assert override_trend is not None
            n_ep = e1 - e0
            if n_ep < 2 or float(t_valid[e1 - 1] - t_valid[e0]) <= 0.0:
                med_y = float(np.median(y_deseas[e0:e1]))
                override_trend[e0:e1] = med_y
                segments.append(_make_seg(e0, e1, "l1", 0.0, med_y, None, None, None))
            else:
                tp_rel = np.array([tp - e0 for tp in tp_epoch], dtype=np.intp)
                w_ep = _slice_w(e0, e1)
                w_vec = np.asarray(w_ep, dtype=np.float64) if (w_ep is not None and np.ndim(w_ep) == 1) else None
                l1_fb: list[bool] = []
                slopes_ep, intercepts_ep, y_fit_ep, _ = fit_l1_piecewise(
                    t_valid[e0:e1], y_deseas[e0:e1], tp_rel, w=w_vec, fallback_flag=l1_fb
                )
                if l1_fb and l1_fb[0] and mi_warnings is not None:
                    mi_warnings.append(
                        WarningItem(
                            code="L1_SOLVER_FALLBACK",
                            severity="warning",
                            message=(
                                f"HiGHS L1 linear-program solver did not converge cleanly on epoch {ep_i}; "
                                "fell back to reweighted L1 IRLS approximation."
                            ),
                        )
                    )
                override_trend[e0:e1] = y_fit_ep
                for k in range(len(sub_bounds) - 1):
                    s0, s1 = sub_bounds[k], sub_bounds[k + 1]
                    segments.append(
                        _make_seg(
                            s0,
                            s1,
                            "l1",
                            float(slopes_ep[k]),
                            float(intercepts_ep[k]),
                            None,
                            None,
                            None,
                        )
                    )
        elif slope_estimator == "huber":
            assert (
                override_trend is not None
                and override_ci_low is not None
                and override_ci_high is not None
            )
            n_ep = e1 - e0
            if n_ep < 2 or float(t_valid[e1 - 1] - t_valid[e0]) <= 0.0:
                med_y = float(np.median(y_deseas[e0:e1]))
                override_trend[e0:e1] = med_y
                segments.append(_make_seg(e0, e1, "huber", 0.0, med_y, None, None, None))
            else:
                t_ep = t_valid[e0:e1]
                y_ep = y_deseas[e0:e1]
                t0_ep = t_ep - t_ep[0]
                s_t = float(t0_ep[-1]) if float(t0_ep[-1]) > 0.0 else 1.0
                u_ep = t0_ep / s_t
                m_ep = len(tp_epoch)
                D_ep = np.empty((n_ep, m_ep + 2), dtype=np.float64)
                D_ep[:, 0] = 1.0
                D_ep[:, 1] = u_ep
                for k_i, tp in enumerate(tp_epoch):
                    D_ep[:, 2 + k_i] = np.maximum(0.0, u_ep - u_ep[tp - e0])
                pw_ep = prepare_weights(_slice_w(e0, e1), n_ep)
                hr_ep = solve_huber_irls(D_ep, y_ep, pw_ep, k=huber_k, compute_cov=True)
                coeff = hr_ep.ols.coeff
                y_fit_ep = D_ep @ coeff
                override_trend[e0:e1] = y_fit_ep
                dof_ep = n_ep - hr_ep.ols.rank
                if (
                    hr_ep.ols.cov.size > 0
                    and np.all(np.isfinite(hr_ep.ols.cov))
                    and dof_ep > 0
                ):
                    var_ep = np.maximum(0.0, np.sum((D_ep @ hr_ep.ols.cov) * D_ep, axis=1))
                    t_crit_ep = float(t_dist.ppf(1.0 - 0.5 * alpha, dof_ep))
                    hw_ep = t_crit_ep * np.sqrt(var_ep)
                    override_ci_low[e0:e1] = y_fit_ep - hw_ep
                    override_ci_high[e0:e1] = y_fit_ep + hw_ep
                for k in range(len(sub_bounds) - 1):
                    s0, s1 = sub_bounds[k], sub_bounds[k + 1]
                    v = np.zeros(m_ep + 2, dtype=np.float64)
                    v[1 : 2 + k] = 1.0 / s_t
                    slope_k = float(v @ coeff)
                    anchor_idx = 0 if k == 0 else (tp_epoch[k - 1] - e0)
                    intercept_k = float(y_fit_ep[anchor_idx]) - slope_k * float(t_ep[anchor_idx])
                    ci_lo: float | None = None
                    ci_hi: float | None = None
                    pval_k: float | None = None
                    if hr_ep.ols.cov.size > 0 and np.all(np.isfinite(hr_ep.ols.cov)) and dof_ep > 0:
                        var_k = max(0.0, float(v @ hr_ep.ols.cov @ v))
                        se_k = float(np.sqrt(var_k))
                        if se_k > 0.0:
                            t_crit = float(t_dist.ppf(1.0 - 0.5 * alpha, dof_ep))
                            ci_lo = slope_k - t_crit * se_k
                            ci_hi = slope_k + t_crit * se_k
                            pval_k = float(2.0 * t_dist.sf(abs(slope_k) / se_k, dof_ep))
                        else:
                            ci_lo = slope_k
                            ci_hi = slope_k
                            pval_k = 0.0 if abs(slope_k) > 0.0 else 1.0
                    segments.append(
                        _make_seg(s0, s1, "huber", slope_k, intercept_k, ci_lo, ci_hi, pval_k)
                    )
        else:
            # Default OLS per-segment slopes, t-CIs, and p-values
            for k in range(len(sub_bounds) - 1):
                s0, s1 = sub_bounds[k], sub_bounds[k + 1]
                n_s = s1 - s0
                if n_s < 2 or float(t_valid[s1 - 1] - t_valid[s0]) <= 0.0:
                    mean_y = float(np.mean(y_deseas[s0:s1])) if n_s > 0 else 0.0
                    segments.append(_make_seg(s0, s1, "ols", 0.0, mean_y, None, None, None))
                    continue
                t_s = t_valid[s0:s1]
                y_s = y_deseas[s0:s1]
                t0_s = t_s - t_s[0]
                s_t = float(t0_s[-1]) if float(t0_s[-1]) > 0.0 else 1.0
                D_s = np.column_stack([np.ones(n_s, dtype=np.float64), t0_s / s_t])
                pw_s: PreparedWeights = prepare_weights(_slice_w(s0, s1), n_s)
                if (
                    prov is not None
                    and n_imputations >= 2
                    and bool(np.any(prov.status_codes[s0:s1] != 0))
                ):
                    mi = _mi_segment_slope(
                        t_s, prov, seasonal_part, s0, s1, D_s, s_t, pw_s, alpha,
                        n_imputations, mi_seed + 7919 * k,
                    )
                    cen_frac_k = float(np.mean(prov.status_codes[s0:s1] != 0))
                    if mi is not None and cen_frac_k > 0.4 and mi_warnings is not None:
                        # AUD-15: Rubin-pooled OLS CIs covered ~90.5-91% at 50% censoring in
                        # simulation, Akritas-Theil-Sen ~95%.
                        mi_warnings.append(
                            WarningItem(
                                code="HEAVY_CENSORING_OLS_MI",
                                severity="caution",
                                message=(
                                    f"Segment {k} is {cen_frac_k:.0%} censored; multiple-imputation OLS "
                                    "slope CIs were slightly too narrow (~91% at nominal 95%) at this "
                                    "level of censoring in simulation."
                                ),
                                suggestion="Use slope_estimator='sen' (Akritas-Theil-Sen for censored data).",
                            )
                        )
                    if mi is not None:
                        segments.append(_make_seg(s0, s1, "ols_mi", *mi))
                        continue
                    if mi_warnings is not None:
                        mi_warnings.append(
                            WarningItem(
                                code="MI_SEGMENT_FALLBACK",
                                severity="warning",
                                message=(
                                    f"Multiple imputation failed for censored segment {k}; its OLS "
                                    "CI ignores imputation uncertainty."
                                ),
                            )
                        )
                ols_s = solve_weighted_ols(D_s, y_s, pw_s, compute_cov=True)
                slope_k = float(ols_s.coeff[1]) / s_t
                intercept_k = float(ols_s.coeff[0]) - slope_k * float(t_s[0])
                dof_s = n_s - ols_s.rank
                ci_lo = None
                ci_hi = None
                pval_k = None
                if ols_s.cov.size > 0 and np.all(np.isfinite(ols_s.cov)) and dof_s > 0:
                    se_k = float(np.sqrt(max(0.0, float(ols_s.cov[1, 1])))) / s_t
                    if se_k > 0.0:
                        t_crit = float(t_dist.ppf(1.0 - 0.5 * alpha, dof_s))
                        ci_lo = slope_k - t_crit * se_k
                        ci_hi = slope_k + t_crit * se_k
                        pval_k = float(2.0 * t_dist.sf(abs(slope_k) / se_k, dof_s))
                    else:
                        ci_lo = slope_k
                        ci_hi = slope_k
                        pval_k = 0.0 if abs(slope_k) > 0.0 else 1.0
                segments.append(
                    _make_seg(s0, s1, "ols", slope_k, intercept_k, ci_lo, ci_hi, pval_k)
                )

    return tuple(segments), override_trend, override_ci_low, override_ci_high


# Pointwise interval construction for the trend band: "percentile" (default; about 1 point
# closer to nominal than "basic" in the MC7b design) or "basic".
_BAND_INTERVAL = "percentile"


def _replicate_windowed_seasonal(
    t_valid: FloatArray,
    ystar: FloatArray,
    decomp: JustDecomposeResult,
    jump_valid_indices: list[int],
    w: FloatArray | None,
) -> FloatArray:
    """Re-run ``just_decompose``'s seasonal estimator on bootstrap replicates (columns).

    Same windows, local design (piecewise-linear trend with the window's jumps plus the
    window's selected harmonics) and Gaussian blending as the original decomposition,
    by weighted least squares vectorised over replicates. Frequency selection is held
    fixed at the original windows' choices.
    """
    n, nb = ystar.shape
    num = np.zeros((n, nb))
    den = np.zeros(n)
    jv = np.asarray(sorted(jump_valid_indices), dtype=np.intp)
    for (lb, ub), freqs in zip(decomp.window_bounds, decomp.window_frequencies, strict=True):
        size0 = ub - lb
        if size0 < 4:
            continue
        ty = t_valid[lb:ub]
        ty_loc = ty - ty[0]
        local_j = np.array([int(j - lb) for j in jv if lb + 2 <= j < ub - 1], dtype=np.intp)
        fr = np.asarray(freqs, dtype=np.float64)
        d_w, _cs = build_design_matrix(ty_loc, "linear", local_j, False, fr)
        sw = np.sqrt(w[lb:ub]) if w is not None else np.ones(size0)
        coef, *_ = np.linalg.lstsq(d_w * sw[:, None], ystar[lb:ub] * sw[:, None], rcond=None)
        n_h = 2 * int(fr.size)
        season_w = d_w[:, d_w.shape[1] - n_h :] @ coef[d_w.shape[1] - n_h :] if n_h else 0.0 * ystar[lb:ub]
        tau0 = float(ty[size0 // 2]) if (size0 % 2 == 1) else 0.5 * float(ty[size0 // 2] + ty[size0 // 2 - 1])
        g = np.exp(decomp.window_dilation * (ty - tau0) ** 2)
        num[lb:ub] += season_w * g[:, None]
        den[lb:ub] += g
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den[:, None] > 0, num / np.where(den > 0, den, 1.0)[:, None], 0.0)
    return np.asarray(out, dtype=np.float64)


def _windowed_seasonal_leverage(
    t_valid: FloatArray,
    decomp: JustDecomposeResult,
    jump_valid_indices: list[int],
    w: FloatArray | None,
) -> FloatArray:
    """Diagonal of the (linear) windowed seasonal estimator's hat matrix."""
    n = int(t_valid.size)
    num = np.zeros(n)
    den = np.zeros(n)
    jv = np.asarray(sorted(jump_valid_indices), dtype=np.intp)
    for (lb, ub), freqs in zip(decomp.window_bounds, decomp.window_frequencies, strict=True):
        size0 = ub - lb
        fr = np.asarray(freqs, dtype=np.float64)
        n_h = 2 * int(fr.size)
        if size0 < 4 or n_h == 0:
            continue
        ty = t_valid[lb:ub]
        ty_loc = ty - ty[0]
        local_j = np.array([int(j - lb) for j in jv if lb + 2 <= j < ub - 1], dtype=np.intp)
        d_w, _cs = build_design_matrix(ty_loc, "linear", local_j, False, fr)
        sw = np.sqrt(w[lb:ub]) if w is not None else np.ones(size0)
        pinv = np.linalg.pinv(d_w * sw[:, None]) * sw[None, :]  # coef = pinv @ y
        h_cols = slice(d_w.shape[1] - n_h, d_w.shape[1])
        diag_w = np.sum(d_w[:, h_cols] * pinv[h_cols, :].T, axis=1)
        tau0 = float(ty[size0 // 2]) if (size0 % 2 == 1) else 0.5 * float(ty[size0 // 2] + ty[size0 // 2 - 1])
        g = np.exp(decomp.window_dilation * (ty - tau0) ** 2)
        num[lb:ub] += diag_w * g
        den[lb:ub] += g
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


def _replicate_step_offsets(
    t_valid: FloatArray,
    y_deseas: FloatArray,
    jump_valid_indices: list[int],
    w: FloatArray | None,
    *,
    local_half_window: int | None = None,
) -> FloatArray:
    """Vectorised local-window :func:`_estimate_vertical_step_offsets` over columns."""
    n, nb = y_deseas.shape
    out = np.zeros((n, nb), dtype=np.float64)
    if not jump_valid_indices:
        return out
    j_arr = np.sort(np.unique(np.asarray(jump_valid_indices, dtype=np.intp)))
    j_arr = j_arr[(j_arr > 0) & (j_arr < n)]
    if j_arr.size == 0:
        return out
    t0 = t_valid - t_valid[0]
    sw_all = np.sqrt(w) if w is not None else np.ones(n, dtype=np.float64)
    half_w = (
        max(6, int(local_half_window))
        if local_half_window is not None
        else max(10, min(20, n // 6))
    )
    m = int(j_arr.size)
    for k, j in enumerate(j_arr):
        j_int = int(j)
        prev_j = int(j_arr[k - 1]) if k > 0 else 0
        next_j = int(j_arr[k + 1]) if k + 1 < m else n
        lo = max(prev_j, j_int - half_w)
        hi = min(next_j, j_int + half_w)
        j_loc = j_int - lo
        n_loc = hi - lo
        if j_loc < 1 or (n_loc - j_loc) < 1:
            continue
        t_loc = t0[lo:hi] - t0[lo]
        sw_loc = sw_all[lo:hi]
        y_loc = y_deseas[lo:hi]
        if j_loc >= 3 and (n_loc - j_loc) >= 3 and float(t_loc[j_loc - 1]) > 0.0 and float(t_loc[-1] - t_loc[j_loc]) > 0.0:
            d_m, cs = build_design_matrix(
                t_loc, "linear", np.array([j_loc], dtype=np.intp), False, np.empty(0, dtype=np.float64)
            )
            d_raw = d_m / cs[None, :]
            coef, *_ = np.linalg.lstsq(d_raw * sw_loc[:, None], y_loc * sw_loc[:, None], rcond=None)
            left = coef[0] + coef[2] * t_loc[j_loc]
            right = coef[1] + coef[3] * t_loc[j_loc]
            out[j_int:] += (right - left)[None, :]
        else:
            step_k = np.median(y_loc[j_loc:], axis=0) - np.median(y_loc[:j_loc], axis=0)
            out[j_int:] += step_k[None, :]
    return out


def _joint_step_offsets_single(
    t_valid: FloatArray,
    y_deseas_1d: FloatArray,
    jump_arr: IntArray,
    tp_arr: IntArray,
    sw: FloatArray,
) -> FloatArray:
    """Fast weighted joint step-offset fit [1, u, hinges(tp), steps(j)] for one replicate."""
    n = int(t_valid.size)
    offsets = np.zeros(n, dtype=np.float64)
    if jump_arr.size == 0:
        return offsets
    t0 = t_valid - t_valid[0]
    s_t = float(t0[-1]) if float(t0[-1]) > 0.0 else 1.0
    u = t0 / s_t
    j_set = set(jump_arr.tolist())
    cols: list[FloatArray] = [np.ones(n, dtype=np.float64), u]
    for tp in tp_arr:
        ti = int(tp)
        if 0 < ti < n and ti not in j_set:
            cols.append(np.maximum(0.0, u - u[ti]))
    for j in jump_arr:
        col = np.zeros(n, dtype=np.float64)
        col[int(j) :] = 1.0
        cols.append(col)
    D_j = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(D_j * sw[:, None], y_deseas_1d * sw, rcond=None)
    m_j = int(jump_arr.size)
    step_c = coef[-m_j:]
    for k, j in enumerate(jump_arr):
        offsets[int(j) :] += float(step_c[k])
    return offsets


def _robust_bootstrap_weights(
    raw_resid: FloatArray,
    w_stpd: FloatArray | None,
    robust: RobustMethod,
    huber_k: float,
) -> FloatArray | None:
    """Down-weight isolated outliers during vectorized bootstrap WLS nuisance refits (CRT-09)."""
    if robust == "none":
        return w_stpd
    mad_s = float(1.482602218505602 * np.median(np.abs(raw_resid)))
    rms_r = float(np.sqrt(np.mean(raw_resid * raw_resid)))
    if mad_s <= 1e-14 * max(rms_r, 1e-300):
        return w_stpd
    u_r = raw_resid / mad_s
    au = np.abs(u_r)
    w_rob = (
        np.maximum(tukey_weight(u_r), 1e-12)
        if robust == "mm"
        else np.where(au <= huber_k, 1.0, huber_k / np.maximum(au, huber_k))
    )
    return w_rob if w_stpd is None else np.asarray(w_stpd * w_rob, dtype=np.float64)


def _pipeline_bootstrap(
    t_valid: FloatArray,
    t_raw_valid: np.ndarray[Any, Any],
    series: Any,
    y_work: FloatArray,
    stpd_res: StpdResult,
    step_offsets: FloatArray,
    seasonal_part: FloatArray,
    jump_valid_indices: list[int],
    decomp: JustDecomposeResult | None,
    w_stpd: FloatArray | None,
    refit_mode: RefitMode,
    *,
    robust: RobustMethod,
    huber_k: float,
    ci_level: float,
    n_boot: int,
    segments: tuple[SegmentSummary, ...],
    jump_adjustment: Literal["offset", "segment"] = "offset",
    local_half_window: int | None = None,
) -> tuple[FloatArray, FloatArray, tuple[SegmentSummary, ...]] | None:
    """Detect-then-fit bootstrap band and OLS segment-slope CIs for ``analyze_series``.

    Wild residual bootstrap of the fitted structure (connected or segmented trend +
    jump steps + seasonal harmonics). Supports turning-point series, jump-only /
    seasonal-only series (0 turning points), both ``jump_adjustment='offset'`` and
    ``'segment'``, and robust M/MM down-weighting of isolated outliers in nuisance
    refits (BLK-01, CRT-04, CRT-09). Returns ``(band_low, band_high, segments)`` or ``None``.
    """
    tps = stpd_res.turning_points
    if any(tp.detection_window is None for tp in tps):
        return None
    n = int(t_valid.size)
    f_trend = stpd_res.piecewise_trend.fitted_trend
    use_seas = decomp is not None and bool(decomp.window_bounds) and bool(np.any(seasonal_part != 0.0))
    jumps_sorted = sorted({int(j) for j in jump_valid_indices if 0 < j < n})
    if not tps and not use_seas and not jumps_sorted:
        return None

    tp_win = [
        (int(tp.valid_index), *tp.detection_window)  # type: ignore[misc]
        for tp in tps
    ]
    combined = f_trend + step_offsets
    raw_full = y_work - combined - seasonal_part
    w_boot = _robust_bootstrap_weights(raw_full, w_stpd, robust, huber_k)
    sw_boot = np.sqrt(w_boot) if w_boot is not None else np.ones(n, dtype=np.float64)
    a_lvl = 1.0 - ci_level

    if jump_adjustment == "segment" and jumps_sorted:
        # Epoch-by-epoch detect-then-fit wild bootstrap with full-series seasonal re-estimation
        bounds = [0, *jumps_sorted, n]
        n_ep = len(bounds) - 1
        lev = np.zeros(n, dtype=np.float64)
        p_tr_tot = 0
        for ep_i in range(n_ep):
            e0, e1 = bounds[ep_i], bounds[ep_i + 1]
            ne = e1 - e0
            if ne < 2:
                continue
            te = t_valid[e0:e1]
            sp = float(te[-1] - te[0]) if float(te[-1] - te[0]) > 0.0 else 1.0
            ue = (te - te[0]) / sp
            ep_tps = [gi - e0 for gi, _, _, _ in tp_win if e0 < gi < e1]
            x_e = np.column_stack([np.ones(ne), ue, *[np.maximum(0.0, ue - ue[k]) for k in ep_tps]])
            q_e, _ = np.linalg.qr(x_e * sw_boot[e0:e1, None], mode="reduced")
            lev[e0:e1] = np.sum(q_e * q_e, axis=1)
            p_tr_tot += 2 + len(ep_tps)
        extra_lev = (
            _windowed_seasonal_leverage(t_valid, decomp, jumps_sorted, w_boot)
            if use_seas and decomp is not None
            else np.zeros(n, dtype=np.float64)
        )
        lev_tot = lev + extra_lev
        if np.all(np.isfinite(lev_tot)) and float(np.max(lev_tot)) < 0.99:
            resid = raw_full / np.sqrt(1.0 - lev_tot)
        else:
            resid = raw_full * np.sqrt(n / max(1.0, float(n - p_tr_tot)))
        rng = np.random.default_rng((20261127, n, *jumps_sorted, *[x[0] for x in tp_win]))
        v = rng.choice(np.array([-1.0, 1.0]), size=(n, int(n_boot)))
        ystar = (f_trend + seasonal_part)[:, None] + resid[:, None] * v
        seas_b = (
            _replicate_windowed_seasonal(t_valid, ystar, decomp, jumps_sorted, w_boot)
            if use_seas and decomp is not None
            else np.zeros_like(ystar)
        )
        y_deseas_b = ystar - seas_b
        y_fixed_deseas = ystar - seasonal_part[:, None]

        # Re-locate turning points in their detecting windows (clamped inside each epoch)
        k_n = len(tp_win)
        knots = np.empty((int(n_boot), k_n), dtype=np.intp)
        for j_i, (_gi, w0, w1, m_eff) in enumerate(tp_win):
            t_w = t_valid[w0:w1]
            w_w = w_stpd[w0:w1] if w_stpd is not None else None
            if robust == "none":
                knots[:, j_i] = w0 + _batched_window_knots(t_w, y_deseas_b[w0:w1], m_eff, w_w)
            else:
                for b in range(int(n_boot)):
                    st = _tpd_window(t_w, y_deseas_b[w0:w1, b], m_eff, w_w, robust=robust, huber_k=huber_k)
                    knots[b, j_i] = w0 + int(st.local_tp)

        curves = np.empty((n, int(n_boot)), dtype=np.float64)
        fixed = np.empty((n, int(n_boot)), dtype=np.float64)
        dev_scale = np.ones(n, dtype=np.float64)
        pw_ci_lo = stpd_res.piecewise_trend.trend_ci_low
        pw_ci_hi = stpd_res.piecewise_trend.trend_ci_high
        for ep_i in range(n_ep):
            e0, e1 = bounds[ep_i], bounds[ep_i + 1]
            ne = e1 - e0
            w_ep = w_stpd[e0:e1] if w_stpd is not None else None
            ep_tp_cols = [j_i for j_i, (gi, _, _, _) in enumerate(tp_win) if e0 < gi < e1]
            k0_ep = np.array([tp_win[j_i][0] - e0 for j_i in ep_tp_cols], dtype=np.intp)
            k0_ep = np.unique(np.clip(k0_ep, 1, max(1, ne - 2))) if k0_ep.size > 0 and ne >= 3 else np.empty(0, dtype=np.intp)
            for b in range(int(n_boot)):
                if ep_tp_cols and ne >= 3:
                    kb_ep = np.unique(np.clip(knots[b, ep_tp_cols] - e0, 1, ne - 2))
                else:
                    kb_ep = np.empty(0, dtype=np.intp)
                curves[e0:e1, b] = _fit_piecewise_core(
                    t_valid[e0:e1], t_raw_valid[e0:e1], series.is_datetime, y_deseas_b[e0:e1, b], kb_ep,
                    series.valid_to_orig[e0:e1], ne, 0, series.sampling_rate_ratio,
                    w=w_ep, mode=refit_mode, huber_k=huber_k, ci_level=ci_level,
                ).fitted_trend
                fixed[e0:e1, b] = _fit_piecewise_core(
                    t_valid[e0:e1], t_raw_valid[e0:e1], series.is_datetime, y_fixed_deseas[e0:e1, b], k0_ep,
                    series.valid_to_orig[e0:e1], ne, 0, series.sampling_rate_ratio,
                    w=w_ep, mode=refit_mode, huber_k=huber_k, ci_level=ci_level,
                ).fitted_trend
            if pw_ci_lo is not None and pw_ci_hi is not None:
                dof_ep = max(1, ne - 2 - int(k0_ep.size))
                a_sd_ep = 0.5 * (pw_ci_hi[e0:e1] - pw_ci_lo[e0:e1]) / float(
                    t_dist.ppf(0.5 + 0.5 * ci_level, df=dof_ep)
                )
                sd_fix_ep = np.std(fixed[e0:e1] - f_trend[e0:e1, None], axis=1, ddof=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio_ep = np.where(sd_fix_ep > 0, a_sd_ep / sd_fix_ep, 1.0)
                dev_scale[e0:e1] = np.clip(np.nan_to_num(ratio_ep, nan=1.0), 1.0, 2.0)

        dev = (curves - combined[:, None]) * dev_scale[:, None]
        if _BAND_INTERVAL == "percentile":
            band_lo = combined + np.quantile(dev, 0.5 * a_lvl, axis=1)
            band_hi = combined + np.quantile(dev, 1.0 - 0.5 * a_lvl, axis=1)
        else:
            band_lo = combined - np.quantile(dev, 1.0 - 0.5 * a_lvl, axis=1)
            band_hi = combined - np.quantile(dev, 0.5 * a_lvl, axis=1)
        boot_knots = knots
        boot_ystar = ystar
    else:
        # Offset mode (or no jumps): wild bootstrap with local pre-STPD steps + joint post-STPD refit
        extra_df = 2.0 * len(jumps_sorted)
        if use_seas and decomp is not None:
            probes = np.random.default_rng((20261128, n)).choice(np.array([-1.0, 1.0]), size=(n, 32))
            extra_df += float(np.mean(np.sum(probes * _replicate_windowed_seasonal(
                t_valid, probes, decomp, jumps_sorted, w_boot), axis=0)))

        j_arr_b = np.asarray(jumps_sorted, dtype=np.intp)

        def _adjust(ystar_in: FloatArray) -> tuple[FloatArray, FloatArray]:
            seas_in = (
                _replicate_windowed_seasonal(t_valid, ystar_in, decomp, jumps_sorted, w_boot)
                if use_seas and decomp is not None
                else np.zeros_like(ystar_in)
            )
            steps_in = _replicate_step_offsets(
                t_valid, ystar_in - seas_in, jumps_sorted, w_boot, local_half_window=local_half_window
            )
            return seas_in + steps_in, steps_in

        def _refit(yb: FloatArray, kb: IntArray) -> FloatArray:
            return _fit_piecewise_core(
                t_valid, t_raw_valid, series.is_datetime, yb, kb, series.valid_to_orig,
                series.n_input, series.n_dropped_nan, series.sampling_rate_ratio,
                w=w_stpd, mode=refit_mode, huber_k=huber_k, ci_level=ci_level,
            ).fitted_trend

        def _refit_joint(
            ystar_b: FloatArray,
            nuis_fit_b: FloatArray,
            nuis_trend_b: FloatArray,
            kb: IntArray,
            _b: int,
        ) -> FloatArray:
            y_des_b = ystar_b - (nuis_fit_b - nuis_trend_b)
            if j_arr_b.size > 0:
                steps_j_b = _joint_step_offsets_single(t_valid, y_des_b, j_arr_b, kb, sw_boot)
                return _refit(y_des_b - steps_j_b, kb) + steps_j_b
            return _refit(y_des_b, kb)

        boot = _bootstrap_tp_curves(
            t_valid, y_work, f_trend, tp_win, _refit, w=w_stpd, robust=robust, huber_k=huber_k,
            n_boot=int(n_boot), seed=(20261127, n, *jumps_sorted, *[x[0] for x in tp_win]),
            other_fit=step_offsets + seasonal_part, adjust_fn=_adjust, extra_df=extra_df,
            extra_leverage=(
                _windowed_seasonal_leverage(t_valid, decomp, jumps_sorted, w_boot)
                if use_seas and decomp is not None
                else None
            ),
            analytic_sd=(
                0.5 * (stpd_res.piecewise_trend.trend_ci_high - stpd_res.piecewise_trend.trend_ci_low)
                / float(t_dist.ppf(0.5 + 0.5 * ci_level, df=max(1, n - 2 - len(tp_win))))
                if stpd_res.piecewise_trend.trend_ci_low is not None
                and stpd_res.piecewise_trend.trend_ci_high is not None
                else None
            ),
            allow_empty_tps=True,
            l1_refit=(refit_mode == "l1"),
            refit_with_nuisance=_refit_joint if j_arr_b.size > 0 else None,
        )
        if boot is None:
            return None
        dev = (boot.curves - combined[:, None]) * boot.dev_scale[:, None]
        if _BAND_INTERVAL == "percentile":
            band_lo = combined + np.quantile(dev, 0.5 * a_lvl, axis=1)
            band_hi = combined + np.quantile(dev, 1.0 - 0.5 * a_lvl, axis=1)
        else:
            band_lo = combined - np.quantile(dev, 1.0 - 0.5 * a_lvl, axis=1)
            band_hi = combined - np.quantile(dev, 0.5 * a_lvl, axis=1)
        y_deseas_b = boot.ystar - (boot.nuisance_fit - boot.nuisance_trend)
        boot_knots = boot.knots
        boot_ystar = boot.ystar

    # OLS segment slopes between the replicate's breakpoints
    sw = np.sqrt(w_stpd) if w_stpd is not None else np.ones(n, dtype=np.float64)

    def _ols_slope(y_seg: FloatArray, a: int, b: int) -> float:
        if b - a < 3 or t_valid[b - 1] <= t_valid[a]:
            return float("nan")
        x = np.column_stack([np.ones(b - a), t_valid[a:b] - t_valid[a]]) * sw[a:b, None]
        coef, *_ = np.linalg.lstsq(x, y_seg * sw[a:b], rcond=None)
        return float(coef[1])

    new_segments = list(segments)
    truth_curve = combined
    seg_bounds = sorted(
        {0, n, *jumps_sorted, *[int(tp.valid_index) for tp in tps if 0 < tp.valid_index < n]}
    )
    if len(seg_bounds) - 1 != len(segments):
        return band_lo, band_hi, segments
    d = np.full((int(n_boot), len(segments)), np.nan)
    d_fixed = np.full((int(n_boot), len(segments)), np.nan)
    y_fixed_deseas = boot_ystar - seasonal_part[:, None]
    for b in range(int(n_boot)):
        for k in range(len(segments)):
            d_fixed[b, k] = _ols_slope(
                y_fixed_deseas[seg_bounds[k] : seg_bounds[k + 1], b], seg_bounds[k], seg_bounds[k + 1]
            )
        kb = sorted(int(k) for k in np.unique(boot_knots[b]) if 0 < k < n) if boot_knots.shape[1] > 0 else []
        bb = sorted({0, n, *jumps_sorted, *kb})
        if len(bb) != len(seg_bounds):
            continue
        for k in range(len(segments)):
            d[b, k] = _ols_slope(y_deseas_b[bb[k] : bb[k + 1], b], bb[k], bb[k + 1])
    for k, seg in enumerate(segments):
        if seg.estimator != "ols" or seg.slope_ci_low is None:
            continue
        a, b = seg_bounds[k], seg_bounds[k + 1]
        s_true = _ols_slope(truth_curve[a:b], a, b)
        dk = d[:, k][np.isfinite(d[:, k])] - s_true
        if dk.size < 0.8 * int(n_boot) or not np.isfinite(s_true):
            continue
        # Variance calibration (as for the band): scale up to the analytical OLS slope SE
        # where the fixed-structure bootstrap under-disperses.
        sd_fix = float(np.nanstd(d_fixed[:, k], ddof=1))
        dof_k = max(1, (b - a) - 2)
        se_an = (
            (float(seg.slope_ci_high) - float(seg.slope_ci_low))
            / (2.0 * float(t_dist.ppf(0.5 + 0.5 * ci_level, df=dof_k)))
            if seg.slope_ci_high is not None
            else float("nan")
        )
        if sd_fix > 0 and np.isfinite(se_an):
            dk = dk * float(np.clip(se_an / sd_fix, 1.0, 2.0))
        lo = seg.slope - float(np.quantile(dk, 1.0 - 0.5 * a_lvl))
        hi = seg.slope - float(np.quantile(dk, 0.5 * a_lvl))
        shifted = seg.slope - dk
        frac = min(float(np.mean(shifted <= 0.0)), float(np.mean(shifted >= 0.0)))
        pval = seg.pvalue if frac == 0.0 else float(min(1.0, 2.0 * frac))
        new_segments[k] = replace(seg, slope_ci_low=lo, slope_ci_high=hi, pvalue=pval)
    return band_lo, band_hi, tuple(new_segments)


def _inflate_for_serial_correlation(
    segments: tuple[SegmentSummary, ...],
    t_valid: FloatArray,
    w: FloatArray | None,
    serial: SerialSpec,
    combined_trend: FloatArray,
    band_lo: FloatArray | None,
    band_hi: FloatArray | None,
) -> tuple[tuple[SegmentSummary, ...], FloatArray | None, FloatArray | None]:
    """Scale OLS-type segment CIs/p-values and the trend band for AR(1)/OU errors (AUD-02).

    For each segment the exact sandwich variance of the weighted straight-line fit under
    the fitted correlation (:func:`tspec._serial.sandwich_vif`) gives a variance inflation
    factor ``VIF`` for the slope and for every fitted value. Slope CI half-widths (about
    the estimate, preserving any bootstrap asymmetry) and band half-widths (about the
    combined trend) are multiplied by ``sqrt(VIF)``; p-values are recomputed with the
    t statistic divided by ``sqrt(VIF)``. Theil-Sen / ATS segments carry their own AR(1)
    Mann-Kendall correction and are left unchanged.
    """
    new_segments = list(segments)
    lo = band_lo.copy() if band_lo is not None else None
    hi = band_hi.copy() if band_hi is not None else None
    for k, seg in enumerate(segments):
        a, b = int(seg.start_valid_index), int(seg.end_valid_index) + 1
        n_s = b - a
        if n_s < 4:
            continue
        t_s = t_valid[a:b]
        span = float(t_s[-1] - t_s[0]) if t_s[-1] > t_s[0] else 1.0
        X = np.column_stack([np.ones(n_s), (t_s - t_s[0]) / span])
        w_s = w[a:b] if w is not None else None
        vif_c, vif_f = sandwich_vif(t_s, X, w_s, serial)
        if seg.estimator in ("ols", "ols_mi", "huber") and seg.slope_ci_low is not None:
            f = float(np.sqrt(max(vif_c[1], 1.0)))
            seg_new = replace(
                seg,
                slope_ci_low=seg.slope - f * (seg.slope - float(seg.slope_ci_low)),
                slope_ci_high=seg.slope + f * (float(seg.slope_ci_high) - seg.slope)
                if seg.slope_ci_high is not None
                else None,
                pvalue=deflate_t_pvalue(float(seg.pvalue), n_s - 2, float(max(vif_c[1], 1.0)))
                if seg.pvalue is not None
                else None,
            )
            new_segments[k] = seg_new
        if lo is not None and hi is not None:
            g = np.sqrt(np.maximum(vif_f, 1.0))
            c = combined_trend[a:b]
            lo[a:b] = c - g * (c - lo[a:b])
            hi[a:b] = c + g * (hi[a:b] - c)
    return tuple(new_segments), lo, hi


def analyze_series(
    t: TimeLike,
    y: ValueLike,
    *,
    weights: WeightLike | None = None,
    detect_jumps: bool = True,
    detect_turning_points: bool = True,
    remove_seasonality: bool = True,
    jump_adjustment: Literal["offset", "segment"] = "offset",
    ci_level: float = 0.95,
    robust: RobustMethod = "none",
    slope_estimator: SlopeEstimator = "ols",
    huber_k: float = 1.345,
    mk_correction: Literal["none", "hamed_rao", "ar1"] = "ar1",
    censor_status: ArrayLike | None = None,
    censoring: CensoredConfig | None = None,
    just_kwargs: dict[str, Any] | None = None,
    decompose_kwargs: dict[str, Any] | None = None,
    stpd_kwargs: dict[str, Any] | None = None,
    band_method: Literal["bootstrap", "conditional"] = "bootstrap",
    n_boot: int = 499,
    serial_correlation: SerialMode = "ar1",
) -> PipelineResult:
    """Run the guided two-stage JUST -> STPD change-detection and decomposition pipeline.

    Designed for applied practitioners and researchers who want the recommended
    end-to-end sequence without manual glue code:

    1. Handles irregular timestamps, ``NaN``s, and optional censored data
       (defaulting to 20 stochastic stability draws when censoring is detected).
    2. Detects abrupt jumps via :func:`just_jumps` with family-wise calibrated
       selection (``jump_alpha=0.05``, Monte Carlo split-search calibration).
    3. Extracts seasonality and trend via :func:`just_decompose` with analytical
       pointwise confidence intervals at level ``ci_level``.
    4. Removes seasonality and vertical jump offsets (or segments by epoch)
       and runs :func:`detect_turning_points` (STPD) to isolate smooth changes
       in trend slope.
    5. Computes per-segment trend slopes, confidence intervals, p-values
       (:attr:`PipelineResult.segments`), and pointwise trend bands
       (:attr:`PipelineResult.trend_ci_low` / ``trend_ci_high``) using
       ``slope_estimator``:

       - ``'ols'``: weighted ordinary least squares with Student-t CIs.
       - ``'sen'``: exact **Theil–Sen** (uncensored) or **Akritas–Theil–Sen**
         (left/right censored) with distribution-free CIs and Mann–Kendall
         p-values. For uncensored segments ``mk_correction="ar1"`` (default)
         inflates Var(S) for serial correlation via a bias-corrected AR(1)
         effective-sample-size factor; ``"hamed_rao"`` is the significant-lags
         Hamed & Rao (1998) correction (liberal in small samples); ``"none"``
         assumes independence.
         The median-pivoted slope envelope is exported as ``slope_fan_low/high``
         (it is not a pointwise trend CI; ``trend_ci_*`` is ``None``).
       - ``'huber'``: continuous **Huber IRLS** hinge-spline with Huber (1981)
         asymptotic CIs.
       - ``'l1'``: globally optimal continuous **L1 (LAD)** hinge-spline via
         sparse HiGHS linear programming (robust to isolated outliers; design-dependent
         breakdown well below 50%; CIs are
         ``None``—use ``'sen'`` or ``'huber'`` for segment CIs).

    Parameters
    ----------
    t:
        1D monotonically ascending timestamps (numeric decimal years or datetime).
    y:
        1D observations (numeric, censored strings such as ``'<0.05'``, or
        ``(N, 2)`` intervals). ``NaN`` entries are dropped without interpolation.
    weights:
        Optional scalar, 1D, or 2D inverse-covariance weights (STPD and segment
        fits use scalar/1D weights only).
    detect_jumps:
        Run Stage 1 (:func:`just_jumps`). Default ``True``.
    detect_turning_points:
        Run Stage 4 (:func:`detect_turning_points`). Default ``True``; also enables
        the ``step_alpha`` filter in Stage 1.
    remove_seasonality:
        Subtract the JUST seasonal component before STPD and segment fits.
    jump_adjustment:
        ``'offset'`` (subtract estimated vertical step offsets, then fit one
        connected trend) or ``'segment'`` (fit each inter-jump epoch separately).
    ci_level:
        Confidence level for all bands and segment CIs.
    robust:
        ``'none'``, ``'huber'``, or ``'mm'`` for all stages.
    slope_estimator:
        ``'ols'``, ``'sen'``, ``'l1'``, or ``'huber'`` (see above).
    huber_k:
        Huber tuning constant.
    mk_correction:
        Serial-correlation correction for ``slope_estimator='sen'``: ``'ar1'``
        (default), ``'hamed_rao'``, or ``'none'``.
    censor_status, censoring:
        Optional censoring indicators and :class:`CensoredConfig`. When censored
        input is detected and ``censoring`` is ``None``,
        ``CensoredConfig(n_stability_draws=20)`` is used. OLS segments containing
        censored values are estimated by Rubin-pooled multiple imputation
        (``estimator='ols_mi'``).
    just_kwargs, decompose_kwargs, stpd_kwargs:
        Extra keyword arguments forwarded to :func:`just_jumps`,
        :func:`just_decompose`, and :func:`detect_turning_points`.
    band_method : {"bootstrap", "conditional"}, default "bootstrap"
        ``"bootstrap"``: when turning points are detected, the trend band and the OLS
        segment-slope CIs come from a detect-then-fit wild bootstrap. Each replicate
        re-estimates the jump steps and seasonal harmonics, re-locates every turning
        point with STPD's knot estimator over its detecting window, and refits the
        trend and segment slopes. ``"conditional"``: analytical intervals that treat the
        detected structure as known. Bootstrap intervals apply with
        ``slope_estimator="ols"`` and ``jump_adjustment="offset"``; otherwise, and when
        no turning point is found, the analytical intervals are used.
    n_boot : int, default 499
        Bootstrap replicates for ``band_method="bootstrap"``.
    serial_correlation : {"ar1", "ou", "none"}, default "ar1"
        Error model for calibrating the stages (audit finding AUD-02). It is passed to
        ``just_jumps`` and ``stpd`` (unless set in their kwargs). The OLS / ``ols_mi`` /
        Huber segment-slope CIs and p-values and the trend band are scaled by the exact
        sandwich variance inflation of a straight-line fit under the fitted AR(1)
        (``"ar1"``, observation order) or Ornstein-Uhlenbeck (``"ou"``, continuous time)
        correlation. ``"none"`` assumes independent errors: the OLS slope test then
        rejected 24% (rho = 0.5) and 49% (rho = 0.8) of trend-free AR(1) series at a
        nominal 5%. Theil-Sen/ATS segments use ``mk_correction`` instead.

    Returns
    -------
    PipelineResult
        Stage results, adjusted series, combined trend and bands, and per-segment
        slope summaries (:attr:`PipelineResult.segments`). With the default
        ``band_method="bootstrap"`` the trend band and OLS segment-slope CIs include
        turning-point location, jump-step and seasonal-estimation uncertainty. They
        remain conditional on the number of breakpoints found.

    Setting ``robust='huber'`` (convex Huber IRLS) or ``robust='mm'``
    (Huber-initialized Tukey biweight MM-type estimation, recommended for extreme
    isolated spikes) switches Phase 1 detection and decomposition
    (``just_jumps``, ``just_decompose``, step-offset estimation, and STPD) to
    outlier-resistant M/MM-estimation with Schrader–Hettmansperger (1980) robust
    nested F-tests.
    """

    if jump_adjustment not in ("offset", "segment"):
        raise ValueError("`jump_adjustment` must be 'offset' or 'segment'.")
    if not (0.0 < ci_level < 1.0):
        raise ValueError(f"`ci_level` must lie in (0, 1); got {ci_level}.")
    rob_method = validate_robust_args(robust, huber_k)
    if mk_correction not in ("none", "hamed_rao", "ar1"):
        raise ValueError("`mk_correction` must be 'none', 'hamed_rao', or 'ar1'.")
    if slope_estimator not in ("ols", "sen", "l1", "huber"):
        raise ValueError(
            f"Unknown `slope_estimator` {slope_estimator!r}. "
            "Options: 'ols', 'sen', 'l1', 'huber'."
        )

    # Enable 20-draw censoring stability checks by default in the guided pipeline
    eff_censoring = censoring
    if eff_censoring is None and has_censored_input(y, censor_status, None):
        eff_censoring = CensoredConfig(n_stability_draws=20)

    # Resolve non-integer random_state (None or Generator) to a single concrete integer
    # seed so Stage 0 preprocessing and Stage 1 just_jumps share identical draws.
    if eff_censoring is not None and not isinstance(eff_censoring.random_state, (int, np.integer)):
        seed_rng = (
            eff_censoring.random_state
            if isinstance(eff_censoring.random_state, np.random.Generator)
            else np.random.default_rng()
        )
        eff_censoring = replace(
            eff_censoring,
            random_state=int(seed_rng.integers(0, 2**31 - 1)),
        )

    prep = prepare_series_with_censoring(
        t, y, censor_status=censor_status, censoring=eff_censoring
    )
    series = prep.series
    prov = prep.provenance

    j_kw: dict[str, Any] = dict(just_kwargs or {})
    d_kw: dict[str, Any] = dict(decompose_kwargs or {})
    s_kw: dict[str, Any] = dict(stpd_kwargs or {})

    # Propagate robust & CI configuration unless explicitly overridden in stage kwargs
    serial_mode = validate_serial_mode(serial_correlation)
    j_kw.setdefault("serial_correlation", serial_mode)
    s_kw.setdefault("serial_correlation", serial_mode)
    j_kw.setdefault("robust", rob_method)
    j_kw.setdefault("huber_k", huber_k)
    d_kw.setdefault("robust", rob_method)
    d_kw.setdefault("huber_k", huber_k)
    d_kw.setdefault("ci_level", ci_level)
    s_kw.setdefault("robust", rob_method)
    s_kw.setdefault("huber_k", huber_k)
    s_kw.setdefault("ci_level", ci_level)
    if slope_estimator in ("l1", "huber"):
        s_kw.setdefault("refit_mode", slope_estimator)
    if band_method not in ("bootstrap", "conditional"):
        raise ValueError(f"Unknown `band_method` {band_method!r}. Options: 'bootstrap', 'conditional'.")
    if int(n_boot) < 99:
        raise ValueError(f"`n_boot` must be >= 99; got {n_boot}.")
    # The pipeline computes its own detect-then-fit bootstrap (it also re-estimates jump
    # steps and seasonal harmonics), so the internal STPD call uses analytical bands.
    s_kw.setdefault("band_method", "conditional")

    # Scale-free defaults in analyze_series unless overridden:
    # - just_jumps' family-wise calibrated selection (jump_alpha=0.05, Monte Carlo
    #   max-over-splits calibration + Holm across candidate locations).
    # - When STPD is also enabled, step_alpha isolates genuine vertical discontinuities in
    #   Stage 1 (JUST) so pure continuous slope bends pass cleanly to Stage 2 (STPD).
    if "jump_alpha" not in j_kw:
        j_kw["jump_alpha"] = 0.05
    if detect_turning_points and "step_alpha" not in j_kw:
        j_kw["step_alpha"] = float(j_kw["jump_alpha"]) if j_kw["jump_alpha"] is not None else 0.05
    # - Multi-scale windows (AUD-04): longer windows give Stage 1 power for small steps;
    #   all scales form one Holm family and the curvature-robust step test keeps smooth
    #   trends from being reported as jumps.
    if not j_kw.get("parity_mode", False):
        j_kw.setdefault("window_scales", (1.0, 2.0, 4.0))
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
        if jumps_res.censoring is not None:
            prov = jumps_res.censoring
        _add_warnings(jumps_res.diagnostics.warnings)

    # Use consistent imputed series for downstream decomposition & STPD
    y_work_valid = jumps_res.values if jumps_res is not None else series.values
    t_valid = series.times_numeric
    t_raw_valid = series.times_raw

    # Prepare weights for valid entries, propagating censoring imputed_weight_factor
    w_valid_raw: Any = None
    pw_work = prepare_weights(weights, series.n_input)
    if series.n_dropped_nan:
        pw_work = pw_work.subset(series.valid_mask)
    if prep.weight_factors is not None:
        pw_work = pw_work.scale_entries(prep.weight_factors)
    if pw_work.kind == "vector":
        w_valid_raw = pw_work.vector
    elif pw_work.kind == "matrix":
        w_valid_raw = pw_work.matrix
    elif pw_work.kind == "scalar":
        w_valid_raw = pw_work.scalar

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
            if series.n_dropped_nan > 0:
                orig_j_arr = (
                    series.valid_to_orig[decomp_res.jump_indices]
                    if decomp_res.jump_indices.size > 0
                    else decomp_res.jump_indices
                )
                decomp_res = replace(
                    decomp_res,
                    jump_indices=orig_j_arr,
                    diagnostics=replace(
                        decomp_res.diagnostics,
                        n_input=series.n_input,
                        n_dropped_nan=series.n_dropped_nan,
                    ),
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
    seasonal_part = (
        np.nan_to_num(decomp_res.seasonal, nan=0.0)
        if (remove_seasonality and decomp_res is not None)
        else np.zeros_like(y_work_valid)
    )
    y_deseas = y_work_valid - seasonal_part

    step_offsets = np.zeros(series.n_valid, dtype=np.float64)
    if decomp_res is not None and decomp_res.window_bounds:
        med_win = int(np.median([ub - lb for lb, ub in decomp_res.window_bounds]))
        local_half_w = max(8, min(med_win // 2, max(8, series.n_valid // 6)))
    else:
        local_half_w = max(10, min(20, max(10, series.n_valid // 6)))
    if jump_valid_indices and jump_adjustment == "offset":
        step_offsets = _estimate_vertical_step_offsets(
            t_valid,
            y_deseas,
            jump_valid_indices,
            w_valid_raw,
            robust=rob_method,
            huber_k=huber_k,
            local_half_window=local_half_w,
        )
        y_adj = y_deseas - step_offsets
    else:
        y_adj = y_deseas.copy()

    def _remap_stpd_to_orig(res: StpdResult) -> StpdResult:
        """Remap STPD turning points, flagged jumps, piecewise trend, and diagnostics to input axis."""
        remapped_tps = tuple(
            replace(tp, index=int(series.valid_to_orig[tp.valid_index]))
            for tp in res.turning_points
        )
        remapped_flagged = tuple(
            replace(fj, index=int(series.valid_to_orig[fj.valid_index]))
            for fj in res.flagged_jumps
        )
        pw = res.piecewise_trend
        pw_tp_orig = (
            series.valid_to_orig[pw.tp_valid_indices]
            if pw.tp_valid_indices.size > 0
            else np.empty(0, dtype=np.intp)
        )
        pw_diag = replace(
            pw.diagnostics,
            n_input=series.n_input,
            n_dropped_nan=series.n_dropped_nan,
        )
        pw_remapped = replace(pw, tp_indices=pw_tp_orig, diagnostics=pw_diag)
        stpd_diag = replace(
            res.diagnostics,
            n_input=series.n_input,
            n_dropped_nan=series.n_dropped_nan,
        )
        return replace(
            res,
            turning_points=remapped_tps,
            flagged_jumps=remapped_flagged,
            piecewise_trend=pw_remapped,
            censoring=prov,
            diagnostics=stpd_diag,
        )

    # Stage 4: STPD continuous turning-point detection
    stpd_res: StpdResult | None = None
    if w_valid_raw is not None and np.ndim(w_valid_raw) == 2:
        _add_warnings(
            [
                WarningItem(
                    code="TWO_D_WEIGHTS_DROPPED_FOR_STPD",
                    severity="caution",
                    message=(
                        "2D inverse-covariance weights were applied in JUST stages, but STPD and "
                        "per-segment trend fits support scalar/1D weights only and were run unweighted."
                    ),
                    suggestion="Pass 1D observation weights if heteroscedastic weighting is needed in STPD.",
                )
            ]
        )
    w_stpd = w_valid_raw if (w_valid_raw is not None and np.ndim(w_valid_raw) <= 1) else None
    if detect_turning_points:
        eff_stpd_ci = float(s_kw.get("ci_level", ci_level))
        if jump_adjustment == "segment" and jump_valid_indices:
            bounds = [0, *sorted(jump_valid_indices), series.n_valid]
            seg_tps: list[TurningPointEvent] = []
            seg_flagged: list[FlaggedJumpCandidate] = []
            seg_valid_idx: list[int] = []
            seg_fitted = np.empty(series.n_valid, dtype=np.float64)
            seg_ci_low_arr = np.full(series.n_valid, np.nan, dtype=np.float64)
            seg_ci_high_arr = np.full(series.n_valid, np.nan, dtype=np.float64)
            seg_has_ci = True
            seg_slopes_list: list[FloatArray] = []
            seg_intercepts_list: list[FloatArray] = []
            seg_dirs_list: list[FloatArray] = []
            seg_snr_raw_list: list[FloatArray] = []
            seg_snr_cent_list: list[FloatArray] = []
            seg_ndri_l2_list: list[FloatArray] = []
            seg_ndri_rmse_list: list[FloatArray] = []
            seg_cond_max = 1.0
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
                        pw_s = seg_res.piecewise_trend
                        seg_fitted[s0:s1] = pw_s.fitted_trend
                        if pw_s.trend_ci_low is not None and pw_s.trend_ci_high is not None:
                            seg_ci_low_arr[s0:s1] = pw_s.trend_ci_low
                            seg_ci_high_arr[s0:s1] = pw_s.trend_ci_high
                        else:
                            seg_has_ci = False
                        seg_slopes_list.append(pw_s.slopes)
                        seg_intercepts_list.append(pw_s.intercepts)
                        seg_dirs_list.append(pw_s.directions)
                        seg_snr_raw_list.append(pw_s.local_snr_raw)
                        seg_snr_cent_list.append(pw_s.local_snr_centered)
                        seg_ndri_l2_list.append(pw_s.local_ndri_l2)
                        seg_ndri_rmse_list.append(pw_s.local_ndri_rmse)
                        seg_cond_max = max(seg_cond_max, seg_res.diagnostics.condition_number_max)
                        for tp in seg_res.turning_points:
                            g_valid = s0 + int(tp.valid_index)
                            seg_valid_idx.append(g_valid)
                            g_win = (
                                (s0 + tp.detection_window[0], s0 + tp.detection_window[1], tp.detection_window[2])
                                if tp.detection_window is not None
                                else None
                            )
                            seg_tps.append(
                                replace(
                                    tp,
                                    valid_index=g_valid,
                                    index=int(series.valid_to_orig[g_valid]),
                                    detection_window=g_win,
                                )
                            )
                        for fj in seg_res.flagged_jumps:
                            g_valid_fj = s0 + int(fj.valid_index)
                            seg_flagged.append(
                                replace(
                                    fj,
                                    valid_index=g_valid_fj,
                                    index=int(series.valid_to_orig[g_valid_fj]),
                                )
                            )
                        _add_warnings(seg_res.diagnostics.warnings)
                        continue
                    except InsufficientDataError as exc:
                        _add_warnings(
                            [
                                WarningItem(
                                    code="STPD_SEGMENT_SKIPPED",
                                    severity="info",
                                    message=(
                                        f"Turning-point detection skipped in epoch {seg_i} ({exc}); a "
                                        "single linear trend was fitted there."
                                    ),
                                )
                            ]
                        )
                eff_refit_mode: RefitMode = s_kw.get("refit_mode") or (
                    "huber" if rob_method != "none" else "sequential"
                )
                pw_seg = fit_piecewise_trend(
                    t_raw_valid[s0:s1],
                    y_adj[s0:s1],
                    [],
                    weights=seg_w,
                    mode=eff_refit_mode,
                    huber_k=huber_k,
                    ci_level=eff_stpd_ci,
                )
                seg_fitted[s0:s1] = pw_seg.fitted_trend
                if pw_seg.trend_ci_low is not None and pw_seg.trend_ci_high is not None:
                    seg_ci_low_arr[s0:s1] = pw_seg.trend_ci_low
                    seg_ci_high_arr[s0:s1] = pw_seg.trend_ci_high
                else:
                    seg_has_ci = False
                seg_slopes_list.append(pw_seg.slopes)
                seg_intercepts_list.append(pw_seg.intercepts)
                seg_cond_max = max(seg_cond_max, pw_seg.diagnostics.condition_number_max)

            seg_resid = y_adj - seg_fitted
            rn_seg, rn_seg_warns = compute_red_noise_diagnostics(t_valid, seg_resid)
            seg_diag = DiagnosticReport(
                red_noise=rn_seg,
                condition_number_max=seg_cond_max,
                n_input=series.n_input,
                n_valid=series.n_valid,
                n_dropped_nan=series.n_dropped_nan,
                sampling_rate_ratio=series.sampling_rate_ratio,
                warnings=rn_seg_warns,
            )
            tp_valid_arr = np.array(seg_valid_idx, dtype=np.intp)
            tp_orig_arr = (
                series.valid_to_orig[tp_valid_arr]
                if tp_valid_arr.size > 0
                else np.empty(0, dtype=np.intp)
            )
            eff_sel_dir: Literal["forward", "backward", "joint"] = (
                "joint"
                if (s_kw.get("refit_mode") in ("joint", "l1", "huber") or rob_method != "none")
                else "forward"
            )
            pw_global = PiecewiseTrendResult(
                tp_indices=tp_orig_arr,
                tp_valid_indices=tp_valid_arr,
                slopes=np.concatenate(seg_slopes_list) if seg_slopes_list else np.empty(0, dtype=np.float64),
                intercepts=np.concatenate(seg_intercepts_list) if seg_intercepts_list else np.empty(0, dtype=np.float64),
                fitted_trend=seg_fitted,
                residuals=seg_resid,
                directions=np.concatenate(seg_dirs_list) if seg_dirs_list else np.empty(0, dtype=np.float64),
                local_snr_raw=np.concatenate(seg_snr_raw_list) if seg_snr_raw_list else np.empty(0, dtype=np.float64),
                local_snr_centered=np.concatenate(seg_snr_cent_list) if seg_snr_cent_list else np.empty(0, dtype=np.float64),
                local_ndri_l2=np.concatenate(seg_ndri_l2_list) if seg_ndri_l2_list else np.empty(0, dtype=np.float64),
                local_ndri_rmse=np.concatenate(seg_ndri_rmse_list) if seg_ndri_rmse_list else np.empty(0, dtype=np.float64),
                mode="segmented",
                selected_direction=eff_sel_dir,
                times_numeric=t_valid,
                times_raw=t_raw_valid,
                is_datetime=series.is_datetime,
                diagnostics=seg_diag,
                ci_level=eff_stpd_ci,
                trend_ci_low=seg_ci_low_arr if seg_has_ci else None,
                trend_ci_high=seg_ci_high_arr if seg_has_ci else None,
            )
            eff_seg_snr = (
                (1.0 if s_kw.get("parity_mode") else SNR_TH_DEFAULT)
                if s_kw.get("snr_th") is None
                else float(s_kw["snr_th"])
            )
            eff_seg_tp_cal = (
                "none"
                if s_kw.get("parity_mode")
                else str(
                    s_kw.get("tp_calibration")
                    or ("bonferroni" if rob_method != "none" else "montecarlo")
                )
            )
            stpd_res = StpdResult(
                turning_points=tuple(seg_tps),
                flagged_jumps=tuple(seg_flagged),
                piecewise_trend=pw_global,
                times_numeric=t_valid,
                times_raw=t_raw_valid,
                values=y_adj,
                is_datetime=series.is_datetime,
                window_mode=str(s_kw.get("window_mode", "count")),
                snr_mode=str(s_kw.get("snr_mode", "centered")),
                ndri_mode=str(s_kw.get("ndri_mode", "rmse")),
                alpha=float(s_kw.get("alpha", 0.01)),
                snr_th=eff_seg_snr,
                ndri_th=float(s_kw.get("ndri_th", 0.3)),
                dir_th=float(s_kw.get("dir_th", 0.0)),
                tp_th=float(s_kw.get("tp_th", 1.0)),
                censoring=prov,
                diagnostics=seg_diag,
                tp_calibration=eff_seg_tp_cal,
                robust=rob_method,
                jump_test=(
                    "ndri" if s_kw.get("parity_mode") else str(s_kw.get("jump_test") or "step")
                ),
            )
        else:
            try:
                raw_stpd = _detect_turning_points(
                    t_raw_valid,
                    y_adj,
                    weights=w_stpd,
                    **s_kw,
                )
                # Exclude turning points sitting directly on an already-detected JUST jump,
                # then jointly refine vertical step offsets + connected trend (BLK-01).
                if jump_valid_indices:
                    j_th = float(j_kw.get("jump_th", 0.75))
                    excl_tol = 0.5 * j_th
                    jump_times = [float(t_valid[j]) for j in jump_valid_indices]
                    kept_tps = tuple(
                        tp
                        for tp in raw_stpd.turning_points
                        if all(abs(tp.time_year - jt) > excl_tol for jt in jump_times)
                    )
                    kept_valid_idx = [tp.valid_index for tp in kept_tps]
                    step_offsets = _estimate_vertical_step_offsets(
                        t_valid,
                        y_deseas,
                        jump_valid_indices,
                        w_valid_raw,
                        robust=rob_method,
                        huber_k=huber_k,
                        tp_valid_indices=kept_valid_idx,
                    )
                    y_adj = y_deseas - step_offsets
                    eff_refit_mode_j: RefitMode = s_kw.get("refit_mode") or (
                        "huber" if rob_method != "none" else "sequential"
                    )
                    pw_refit = fit_piecewise_trend(
                        t_raw_valid,
                        y_adj,
                        kept_valid_idx,
                        weights=w_stpd,
                        mode=eff_refit_mode_j,
                        huber_k=huber_k,
                        ci_level=eff_stpd_ci,
                    )
                    updated_tps = tuple(
                        replace(
                            tp,
                            slope_before=float(pw_refit.slopes[k]),
                            slope_after=float(pw_refit.slopes[k + 1]),
                            direction=float(pw_refit.directions[k]),
                        )
                        for k, tp in enumerate(kept_tps)
                    )
                    raw_stpd = replace(
                        raw_stpd,
                        turning_points=updated_tps,
                        piecewise_trend=pw_refit,
                        values=y_adj,
                    )
                stpd_res = _remap_stpd_to_orig(raw_stpd)
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

    trend_ci_low: FloatArray | None = None
    trend_ci_high: FloatArray | None = None
    if stpd_res is not None:
        combined_trend = stpd_res.piecewise_trend.fitted_trend + step_offsets
        if (
            stpd_res.piecewise_trend.trend_ci_low is not None
            and stpd_res.piecewise_trend.trend_ci_high is not None
        ):
            trend_ci_low = stpd_res.piecewise_trend.trend_ci_low + step_offsets
            trend_ci_high = stpd_res.piecewise_trend.trend_ci_high + step_offsets
        cond_max = max(
            stpd_res.diagnostics.condition_number_max,
            jumps_res.diagnostics.condition_number_max if jumps_res is not None else 1.0,
            decomp_res.diagnostics.condition_number_max if decomp_res is not None else 1.0,
        )
    elif decomp_res is not None:
        combined_trend = decomp_res.trend
        trend_ci_low = decomp_res.trend_ci_low
        trend_ci_high = decomp_res.trend_ci_high
        cond_max = decomp_res.diagnostics.condition_number_max
    else:
        combined_trend = y_work_valid.copy()
        cond_max = jumps_res.diagnostics.condition_number_max if jumps_res is not None else 1.0

    # Stage 5: Per-segment trend slopes, CIs, and p-values (OLS / Theil-Sen / ATS / L1 / Huber)
    mi_warns: list[WarningItem] = []
    tp_valid_indices = (
        [tp.valid_index for tp in stpd_res.turning_points] if stpd_res is not None else []
    )
    segments, override_trend, override_ci_low, override_ci_high = _build_segment_summaries(
        t_valid,
        t_raw_valid,
        series.valid_to_orig,
        y_deseas,
        seasonal_part,
        jump_valid_indices,
        tp_valid_indices,
        w_stpd,
        slope_estimator=slope_estimator,
        huber_k=huber_k,
        prov=prov,
        alpha=1.0 - ci_level,
        mk_correction=mk_correction,
        n_imputations=int(eff_censoring.n_imputations) if eff_censoring is not None else 0,
        mi_seed=int(eff_censoring.random_state) if (
            eff_censoring is not None and isinstance(eff_censoring.random_state, (int, np.integer))
        ) else 0,
        mi_warnings=mi_warns,
    )
    slope_fan_low: FloatArray | None = None
    slope_fan_high: FloatArray | None = None
    if override_trend is not None:
        combined_trend = override_trend
        if slope_estimator == "sen":
            # Median-pivoted slope envelope: NOT a pointwise CI for the trend (zero width at
            # each segment's median time), so it is exported separately.
            slope_fan_low = override_ci_low
            slope_fan_high = override_ci_high
            trend_ci_low = None
            trend_ci_high = None
        else:
            trend_ci_low = override_ci_low
            trend_ci_high = override_ci_high

    band_used = "conditional"
    has_boot_structure = (
        stpd_res is not None
        and (
            bool(stpd_res.turning_points)
            or bool(jump_valid_indices)
            or (remove_seasonality and decomp_res is not None and bool(np.any(seasonal_part != 0.0)))
        )
    )
    if (
        band_method == "bootstrap"
        and stpd_res is not None
        and has_boot_structure
        and override_trend is None
        and trend_ci_low is not None
    ):
        refit_mode_b: RefitMode = s_kw.get("refit_mode") or (
            "huber" if rob_method != "none" else "sequential"
        )
        boot_out = _pipeline_bootstrap(
            t_valid, t_raw_valid, series, y_work_valid, stpd_res, step_offsets, seasonal_part,
            jump_valid_indices,
            decomp_res if remove_seasonality else None,
            w_stpd if (w_stpd is None or np.ndim(w_stpd) == 1) else None,
            refit_mode_b, robust=rob_method, huber_k=huber_k,
            ci_level=float(s_kw.get("ci_level", ci_level)), n_boot=int(n_boot),
            segments=segments,
            jump_adjustment=jump_adjustment,
            local_half_window=local_half_w,
        )
        if boot_out is not None:
            trend_ci_low, trend_ci_high, segments = boot_out
            band_used = "bootstrap"

    # Serial-correlation adjustment of OLS segment inference and the trend band (AUD-02):
    # the analytical and bootstrap intervals above assume independent errors; scale them
    # by the exact sandwich variance inflation under the fitted AR(1)/OU model.
    serial_spec = resolve_serial_spec(
        serial_mode,
        parity_mode=bool(j_kw.get("parity_mode", False) or s_kw.get("parity_mode", False)),
        t=t_valid,
        y=y_deseas - step_offsets,
        weight_kind="vector" if (w_stpd is not None and np.ndim(w_stpd) == 1) else (
            "matrix" if (w_stpd is not None and np.ndim(w_stpd) == 2) else "none"
        ),
        weight_vector=w_stpd if (w_stpd is not None and np.ndim(w_stpd) == 1) else None,
        season="none",
        omega=None,
    )
    if serial_spec.active:
        segments, trend_ci_low, trend_ci_high = _inflate_for_serial_correlation(
            segments,
            t_valid,
            w_stpd if (w_stpd is not None and np.ndim(w_stpd) == 1) else None,
            serial_spec,
            combined_trend,
            trend_ci_low,
            trend_ci_high,
        )
        all_warnings.append(
            WarningItem(
                code="SERIAL_CORRELATION_ADJUSTED",
                severity="info",
                message=serial_warning_message(serial_spec)
                + " OLS segment-slope CIs/p-values and the trend band include the variance "
                "inflation of the fitted AR(1) errors.",
            )
        )

    # Replace intermediate-stage autocorrelation warnings with one clean final check
    # on the combined residuals (after both JUST jumps/seasonality and STPD trend).
    final_residuals = y_work_valid - combined_trend - seasonal_part
    rn_final, final_rn_warns = compute_red_noise_diagnostics(t_valid, final_residuals)
    filtered_warnings = [
        w
        for w in all_warnings
        if w.code not in ("MODERATE_AUTOCORRELATION", "STRONG_AUTOCORRELATION")
        and not (w.code == "SERIAL_CORRELATION_ADJUSTED" and "OLS segment-slope" not in w.message)
    ]
    if serial_spec.active:
        final_rn_warns = tuple(
            replace(
                w,
                suggestion=(
                    f"Jump, turning-point and OLS segment-slope tests were already calibrated for "
                    f"AR(1) errors (rho = {serial_spec.rho:.2f}); other dependence structures "
                    "(long memory, seasonal correlation) are not covered."
                ),
            )
            if w.code in ("MODERATE_AUTOCORRELATION", "STRONG_AUTOCORRELATION")
            else w
            for w in final_rn_warns
        )
    filtered_warnings.extend(final_rn_warns)
    filtered_warnings.extend(mi_warns)

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
        segments=segments,
        robust=rob_method,
        slope_estimator=slope_estimator,
        ci_level=ci_level,
        trend_ci_low=trend_ci_low,
        trend_ci_high=trend_ci_high,
        slope_fan_low=slope_fan_low,
        slope_fan_high=slope_fan_high,
        band_method=band_used,
        requested_band_method=band_method,
        serial_correlation=serial_spec.mode,
        serial_rho=float(serial_spec.rho_hat),
        serial_rho_used=float(serial_spec.rho),
    )
