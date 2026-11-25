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
from scipy.stats import t as t_dist

from ._diagnostics import DiagnosticReport, WarningItem, compute_red_noise_diagnostics
from ._linalg import PreparedWeights, prepare_weights, solve_weighted_ols
from ._robust import (
    RobustMethod,
    SlopeEstimator,
    fit_l1_piecewise,
    fit_theil_sen_segment,
    solve_huber_irls,
    validate_robust_args,
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
from .spectral import _lssa_core
from .stpd import (
    FlaggedJumpCandidate,
    PiecewiseTrendResult,
    RefitMode,
    StpdResult,
    TurningPointEvent,
    fit_piecewise_trend,
)
from .stpd import (
    detect_turning_points as _detect_turning_points,
)


@dataclass(frozen=True)
class SegmentSummary:
    """Per-segment linear trend estimate across detected jumps and turning points."""

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
    segments: tuple[SegmentSummary, ...] = ()
    robust: str = "none"
    slope_estimator: str = "ols"
    ci_level: float = 0.95
    trend_ci_low: FloatArray | None = None
    trend_ci_high: FloatArray | None = None
    slope_fan_low: FloatArray | None = None
    slope_fan_high: FloatArray | None = None

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
) -> FloatArray:
    """Estimate cumulative vertical step offsets at JUST jump indices."""
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
        robust=robust,
        huber_k=huber_k,
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

    Imputation model: normal censored linear regression in time on the deseasonalized
    values and limits (congenial with the segment OLS analysis model).
    Returns ``(slope, intercept, ci_low, ci_high, pvalue)`` or ``None`` on failure.
    """
    from ._cenreg import fit_censored_regression, impute_censored_regression, rubin_pool

    codes = prov.status_codes[s0:s1]
    seas = seasonal_part[s0:s1]
    vals = prov.original_values[s0:s1] - seas
    left = prov.original_left[s0:s1] - seas if prov.original_left is not None else None
    right = prov.original_right[s0:s1] - seas if prov.original_right is not None else None
    w = pw_s.vector if pw_s.kind == "vector" else None
    n_s = int(t_s.size)
    try:
        fit = fit_censored_regression(vals, codes, D_s, left=left, right=right, weights=w, dist="normal")
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not fit.converged:
        return None
    rng = np.random.default_rng(seed)
    slopes = np.empty(m)
    vars_ = np.empty(m)
    icpts = np.empty(m)
    for i in range(m):
        y_i = impute_censored_regression(
            fit, vals, codes, D_s, rng, left=left, right=right, weights=w, draw_params=True
        )
        ols_i = solve_weighted_ols(D_s, y_i, pw_s, compute_cov=True)
        slopes[i] = float(ols_i.coeff[1]) / s_t
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
    mk_correction: Literal["none", "hamed_rao"] = "hamed_rao",
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
                slopes_ep, intercepts_ep, y_fit_ep, _ = fit_l1_piecewise(
                    t_valid[e0:e1], y_deseas[e0:e1], tp_rel, w=w_vec
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
    mk_correction: Literal["none", "hamed_rao"] = "hamed_rao",
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
         p-values. For uncensored segments ``mk_correction="hamed_rao"``
         (default) inflates Var(S) for serial correlation (Hamed & Rao, 1998).
         The median-pivoted slope envelope is exported as ``slope_fan_low/high``
         (it is not a pointwise trend CI; ``trend_ci_*`` is ``None``).
       - ``'huber'``: continuous **Huber IRLS** hinge-spline with Huber (1981)
         asymptotic CIs.
       - ``'l1'``: globally optimal continuous **L1 (LAD)** hinge-spline via
         sparse HiGHS linear programming (50% vertical breakdown; CIs are
         ``None``—use ``'sen'`` or ``'huber'`` for segment CIs).

    Setting ``robust='huber'`` (convex Huber IRLS) or ``robust='mm'`` (Yohai
    1987 Huber-initialized Tukey biweight MM-estimation, recommended for extreme
    isolated spikes) switches Phase 1 detection and decomposition
    (``just_jumps``, ``just_decompose``, step-offset estimation, and STPD) to
    outlier-resistant M/MM-estimation with Schrader–Hettmansperger (1980) robust
    nested F-tests.
    """
    from dataclasses import replace

    if jump_adjustment not in ("offset", "segment"):
        raise ValueError("`jump_adjustment` must be 'offset' or 'segment'.")
    if not (0.0 < ci_level < 1.0):
        raise ValueError(f"`ci_level` must lie in (0, 1); got {ci_level}.")
    rob_method = validate_robust_args(robust, huber_k)
    if mk_correction not in ("none", "hamed_rao"):
        raise ValueError("`mk_correction` must be 'none' or 'hamed_rao'.")
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

    # Scale-free defaults in analyze_series unless overridden:
    # - just_jumps' family-wise calibrated selection (jump_alpha=0.05, Monte Carlo
    #   max-over-splits calibration + Holm across candidate locations).
    # - When STPD is also enabled, step_alpha isolates genuine vertical discontinuities in
    #   Stage 1 (JUST) so pure continuous slope bends pass cleanly to Stage 2 (STPD).
    if "jump_alpha" not in j_kw:
        j_kw["jump_alpha"] = 0.05
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
    if jump_valid_indices and jump_adjustment == "offset":
        step_offsets = _estimate_vertical_step_offsets(
            t_valid,
            y_deseas,
            jump_valid_indices,
            w_valid_raw,
            robust=rob_method,
            huber_k=huber_k,
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
                            seg_tps.append(
                                replace(
                                    tp,
                                    valid_index=g_valid,
                                    index=int(series.valid_to_orig[g_valid]),
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
                    except InsufficientDataError:
                        pass
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
                snr_th=float(s_kw.get("snr_th", 1.0)),
                ndri_th=float(s_kw.get("ndri_th", 0.3)),
                dir_th=float(s_kw.get("dir_th", 0.0)),
                tp_th=float(s_kw.get("tp_th", 1.0)),
                censoring=prov,
                diagnostics=seg_diag,
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

    # Replace intermediate-stage autocorrelation warnings with one clean final check
    # on the combined residuals (after both JUST jumps/seasonality and STPD trend).
    final_residuals = y_work_valid - combined_trend - seasonal_part
    rn_final, final_rn_warns = compute_red_noise_diagnostics(t_valid, final_residuals)
    filtered_warnings = [
        w
        for w in all_warnings
        if w.code not in ("MODERATE_AUTOCORRELATION", "STRONG_AUTOCORRELATION")
    ]
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
    )
