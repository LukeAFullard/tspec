"""Sequential Turning Point Detection (STPD) and Piecewise Linear Trend Estimation.

Clean-room implementation grounded in:
- Ghaderpour, E., Antonielli, B., Bozzano, F., Scarascia Mugnozza, G., &
  Mazzanti, P. (2024). A fast and robust method for detecting trend turning
  points in InSAR displacement time series. *Computers & Geosciences*, 185,
  105546. https://doi.org/10.1016/j.cageo.2024.105546
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.stats import t as student_t

from ._diagnostics import DiagnosticReport, WarningItem, compute_red_noise_diagnostics
from ._linalg import prepare_weights, solve_weighted_ols
from ._time import format_timestamp, prepare_numeric_series
from ._types import ArrayLike, FloatArray, IntArray, TimeLike, ValueLike
from .censored import (
    CensoredConfig,
    CensoringProvenance,
    check_dl_shift_coincidence,
    generate_stability_draws,
    impute_window_local,
    prepare_series_with_censoring,
)
from .exceptions import InsufficientDataError
from .just import WindowMode, _build_duration_windows
from .results import (
    BaseResult,
    format_diagnostics_block,
    format_explain_warnings,
    require_pandas,
)

SnrMode = Literal["raw", "centered"]
NdriMode = Literal["l2", "rmse"]
RefitMode = Literal["sequential", "joint"]


@dataclass(frozen=True)
class TurningPointEvent:
    """Accepted continuous trend turning point detected by STPD."""

    index: int
    valid_index: int
    time: Any
    time_year: float
    slope_before: float
    slope_after: float
    direction: float
    pvalue: float
    snr_raw: float
    snr_centered: float
    ndri_l2: float
    ndri_rmse: float
    fb_diff: int
    window_center_distance: float
    window_censored_fraction: float = 0.0
    coincides_with_dl_shift: bool = False
    stability_rate: float | None = None


@dataclass(frozen=True)
class FlaggedJumpCandidate:
    """High-NDRI window candidate rejected as a turning point (likely a datum jump)."""

    index: int
    valid_index: int
    time: Any
    time_year: float
    direction: float
    pvalue: float
    ndri_l2: float
    ndri_rmse: float
    fb_diff: int


@dataclass(frozen=True)
class PiecewiseTrendResult(BaseResult):
    """Global connected piecewise-linear trend fit across turning points."""

    tp_indices: IntArray
    tp_valid_indices: IntArray
    slopes: FloatArray
    intercepts: FloatArray
    fitted_trend: FloatArray
    residuals: FloatArray
    directions: FloatArray
    local_snr_raw: FloatArray
    local_snr_centered: FloatArray
    local_ndri_l2: FloatArray
    local_ndri_rmse: FloatArray
    mode: str
    selected_direction: Literal["forward", "backward", "joint"]
    times_numeric: FloatArray
    times_raw: np.ndarray[Any, Any]
    is_datetime: bool
    diagnostics: DiagnosticReport

    def summary(self) -> str:
        lines = [
            f"Piecewise Linear Trend Result (mode={self.mode!r}, "
            f"fit={self.selected_direction!r}, segments={self.slopes.size})",
            f"Turning points : {self.tp_indices.tolist()}",
            f"Segment slopes : {[round(float(s), 4) for s in self.slopes]}",
        ]
        if self.tp_indices.size > 0:
            lines.append(
                "  TP_Idx  Post-Slope   Direction    SNR(cent)   NDRI(rmse)  NDRI(L2)"
            )
            for k in range(self.tp_indices.size):
                lines.append(
                    f"  {int(self.tp_indices[k]):<7d} {self.slopes[k + 1]:+11.4f}  "
                    f"{self.directions[k]:+11.4f}  {self.local_snr_centered[k]:10.3f}  "
                    f"{self.local_ndri_rmse[k]:10.3f}  {self.local_ndri_l2[k]:8.3f}"
                )
        lines.extend(format_diagnostics_block(self.diagnostics))
        return "\n".join(lines)

    def explain(self) -> str:
        lines = [
            f"Fitted a continuous connected trend with {self.slopes.size} linear segment(s) "
            f"across {self.diagnostics.n_valid} observations."
        ]
        for k, s in enumerate(self.slopes, start=1):
            lines.append(f"  - Segment {k} slope: {s:+.4g} per unit time.")
        lines.extend(format_explain_warnings(self.diagnostics))
        return "\n".join(lines)

    def to_dataframe(self) -> Any:
        pd = require_pandas()
        return pd.DataFrame(
            {
                "time": self.times_raw,
                "time_year": self.times_numeric,
                "fitted_trend": self.fitted_trend,
                "residual": self.residuals,
            }
        )

    def plot(self, ax: Any = None, **kwargs: Any) -> tuple[Any, Any]:
        from .plotting import plot_piecewise_trend

        return plot_piecewise_trend(self, ax=ax, **kwargs)


@dataclass(frozen=True)
class StpdResult(BaseResult):
    """Result object returned by :func:`detect_turning_points` (:func:`stpd`)."""

    turning_points: tuple[TurningPointEvent, ...]
    flagged_jumps: tuple[FlaggedJumpCandidate, ...]
    piecewise_trend: PiecewiseTrendResult
    times_numeric: FloatArray
    times_raw: np.ndarray[Any, Any]
    values: FloatArray
    is_datetime: bool
    window_mode: str
    snr_mode: str
    ndri_mode: str
    alpha: float
    snr_th: float
    ndri_th: float
    dir_th: float
    tp_th: float
    censoring: CensoringProvenance | None
    diagnostics: DiagnosticReport

    @property
    def tp_indices(self) -> IntArray:
        """0-based turning-point indices mapped onto the original input axis."""
        return np.array([tp.index for tp in self.turning_points], dtype=np.intp)

    @property
    def tp_times(self) -> np.ndarray[Any, Any]:
        """Native timestamps of detected turning points."""
        if not self.turning_points:
            return np.empty(0, dtype=self.times_raw.dtype)
        return np.array([tp.time for tp in self.turning_points], dtype=self.times_raw.dtype)

    def summary(self) -> str:
        lines = [
            f"STPD Turning Point Result (window_mode={self.window_mode!r}, "
            f"snr_mode={self.snr_mode!r}, ndri_mode={self.ndri_mode!r}, alpha={self.alpha})",
            f"Thresholds     : SNR > {self.snr_th}, NDRI < {self.ndri_th}, "
            f"|dir| > {self.dir_th}, min separation >= {self.tp_th}",
            f"Turning points : {len(self.turning_points)} accepted "
            f"({len(self.flagged_jumps)} high-NDRI jump candidate(s) flagged)",
        ]
        if self.turning_points:
            lines.append(
                "  Idx   Time         Slope1      Slope2      Dir         p-value     "
                "NDRI(rmse) NDRI(L2) Flags"
            )
            for tp in self.turning_points:
                t_str = format_timestamp(tp.time, self.is_datetime)
                flags: list[str] = []
                if tp.coincides_with_dl_shift:
                    flags.append("DL-SHIFT")
                if tp.stability_rate is not None:
                    flags.append(f"stab={tp.stability_rate:.0%}")
                flag_str = ",".join(flags) if flags else "-"
                lines.append(
                    f"  {tp.index:<5d} {t_str:<12s} {tp.slope_before:+10.4f}  "
                    f"{tp.slope_after:+10.4f}  {tp.direction:+10.4f}  {tp.pvalue:<11.3e} "
                    f"{tp.ndri_rmse:9.3f}  {tp.ndri_l2:7.3f}  {flag_str}"
                )
        lines.extend(format_diagnostics_block(self.diagnostics))
        return "\n".join(lines)

    def explain(self) -> str:
        lines: list[str] = []
        if not self.turning_points:
            lines.append(
                f"No continuous trend turning points were detected across "
                f"{self.diagnostics.n_valid} observations."
            )
        else:
            lines.append(
                f"Detected {len(self.turning_points)} continuous turning point(s) "
                "where the rate of change shifted smoothly without a sudden vertical jump:"
            )
            for k, tp in enumerate(self.turning_points, start=1):
                t_str = format_timestamp(tp.time, self.is_datetime)
                ndri_disp = tp.ndri_rmse if self.ndri_mode == "rmse" else tp.ndri_l2
                desc = (
                    f"  {k}. Around {t_str} (index {tp.index}): slope changed from "
                    f"{tp.slope_before:+.4g} to {tp.slope_after:+.4g} per unit time "
                    f"(change {tp.direction:+.4g}, screening t-test p={tp.pvalue:.2e}, "
                    f"NDRI={ndri_disp:.2f} — low NDRI confirms a smooth bend)."
                )
                if tp.stability_rate is not None:
                    desc += f" Recovered in {tp.stability_rate:.0%} of censoring stability draws."
                if tp.coincides_with_dl_shift:
                    desc += (
                        " WARNING: coincides with a detection-limit change — verify this is not "
                        "an analytical method artifact."
                    )
                lines.append(desc)

        if self.flagged_jumps:
            t_list = ", ".join(
                f"{format_timestamp(fj.time, self.is_datetime)} (NDRI={fj.ndri_rmse:.2f})"
                for fj in self.flagged_jumps[:4]
            )
            lines.append(
                f"Jump Alarm Bell (high NDRI): {len(self.flagged_jumps)} candidate change(s) "
                f"were rejected as smooth turning points because residual balance was strongly "
                f"asymmetric ({t_list}). This usually indicates an abrupt vertical jump — run "
                "`just_jumps()` or `analyze_series()` to check!"
            )

        lines.extend(format_explain_warnings(self.diagnostics))
        return "\n".join(lines)

    def to_dataframe(self) -> Any:
        pd = require_pandas()
        rows = [
            {
                "index": tp.index,
                "valid_index": tp.valid_index,
                "time": tp.time,
                "time_year": tp.time_year,
                "slope_before": tp.slope_before,
                "slope_after": tp.slope_after,
                "direction": tp.direction,
                "pvalue": tp.pvalue,
                "snr_raw": tp.snr_raw,
                "snr_centered": tp.snr_centered,
                "ndri_l2": tp.ndri_l2,
                "ndri_rmse": tp.ndri_rmse,
                "fb_diff": tp.fb_diff,
                "window_center_distance": tp.window_center_distance,
                "window_censored_fraction": tp.window_censored_fraction,
                "coincides_with_dl_shift": tp.coincides_with_dl_shift,
                "stability_rate": tp.stability_rate,
            }
            for tp in self.turning_points
        ]
        return pd.DataFrame(rows)

    def plot(self, ax: Any = None, **kwargs: Any) -> tuple[Any, Any]:
        from .plotting import plot_stpd

        return plot_stpd(self, ax=ax, **kwargs)


def _stpd_pvalue(
    t: FloatArray,
    y: FloatArray,
    ind: int,
    intercept1: float,
    slope1: float,
    slope2: float,
    w: FloatArray | None = None,
) -> float:
    """Two-sided Student t-test comparing slopes of two connected linear pieces."""
    n = int(t.size)
    if ind <= 1 or n - ind <= 1 or n <= 2:
        return float("nan")

    t1 = t[: ind + 1]
    y1 = y[: ind + 1]
    g1 = y1 - (intercept1 + slope1 * t1)

    d2 = t[ind:] - t[ind]
    y2 = y[ind:] - (intercept1 + slope1 * t[ind])
    g2 = y2 - slope2 * d2

    if w is None:
        s1_num = float(np.dot(g1, g1)) / (ind - 1)
        s1_den = float(np.linalg.norm(t1 - np.mean(t1)))
        s2_num = float(np.dot(g2, g2)) / (n - ind - 1)
        s2_den = float(np.linalg.norm(d2 - np.mean(d2)))
    else:
        w1 = w[: ind + 1]
        w2 = w[ind:]
        t1_bar = float(np.sum(w1 * t1) / np.sum(w1))
        d2_bar = float(np.sum(w2 * d2) / np.sum(w2))
        s1_num = float(np.sum(w1 * g1 * g1)) / (ind - 1)
        s1_den = float(np.sqrt(np.sum(w1 * (t1 - t1_bar) ** 2)))
        s2_num = float(np.sum(w2 * g2 * g2)) / (n - ind - 1)
        s2_den = float(np.sqrt(np.sum(w2 * (d2 - d2_bar) ** 2)))

    if s1_den <= 1e-15 or s2_den <= 1e-15:
        return float("nan")

    se1 = float(np.sqrt(max(0.0, s1_num))) / s1_den
    se2 = float(np.sqrt(max(0.0, s2_num))) / s2_den
    se_comb = float(np.hypot(se1, se2))
    if se_comb <= 1e-24:
        return 0.0 if abs(slope1 - slope2) > 1e-15 else 1.0
    t_stat = abs(slope1 - slope2) / se_comb
    return float(2.0 * student_t.sf(t_stat, df=n - 2))


def _fit_two_piece_sequential_scan(
    t: FloatArray,
    y: FloatArray,
    margin: int,
    w: FloatArray | None = None,
) -> tuple[int, float, float, float, float]:
    """Scan ``ind in [margin, n - margin)`` via O(n) vectorized sequential 2-step OLS/WLS.

    Step 1 fits unconstrained linear piece on ``0..ind``; Step 2 anchors the
    second linear piece at ``(t[ind], b1 + a1 * t[ind])`` and estimates ``a2``
    over ``ind..n-1``. Uses exact prefix-sum identities to evaluate all split
    indices simultaneously in O(n) time while preserving exact forward/backward
    asymmetry (Ghaderpour et al., 2024).
    """
    n = int(t.size)
    lo = max(2, margin)
    hi = min(n - 2, n - margin)
    if lo >= hi:
        raise InsufficientDataError(
            f"Window size n={n} too small for margin={margin} (need at least 2*margin + 1 >= 5)."
        )

    # Center y once for numerical stability in prefix sums of squares (exact shift invariance)
    y_mean = float(np.mean(y))
    yc = y - y_mean
    idx = np.arange(lo, hi, dtype=np.intp)

    if w is None:
        s_w = np.arange(1, n + 1, dtype=np.float64)
        s_t = np.cumsum(t)
        s_y = np.cumsum(yc)
        s_tt = np.cumsum(t * t)
        s_ty = np.cumsum(t * yc)
        s_yy = np.cumsum(yc * yc)
        w_at_idx = np.ones(idx.size, dtype=np.float64)
    else:
        s_w = np.cumsum(w)
        s_t = np.cumsum(w * t)
        s_y = np.cumsum(w * yc)
        s_tt = np.cumsum(w * t * t)
        s_ty = np.cumsum(w * t * yc)
        s_yy = np.cumsum(w * yc * yc)
        w_at_idx = w[idx]

    # Segment 1: 0..ind (inclusive)
    w1 = s_w[idx]
    t1_sum = s_t[idx]
    y1_sum = s_y[idx]
    t1_m = t1_sum / w1
    y1_m = y1_sum / w1
    stt = s_tt[idx] - w1 * t1_m * t1_m
    sty = s_ty[idx] - w1 * t1_m * y1_m
    syy1 = s_yy[idx] - w1 * y1_m * y1_m

    valid = stt > 1e-24
    a1 = np.zeros_like(stt)
    a1[valid] = sty[valid] / stt[valid]
    b1_c = y1_m - a1 * t1_m
    rss1 = np.maximum(0.0, syy1 - a1 * sty)

    # Segment 2: ind..n-1 (inclusive), anchored at (tau, h_c)
    tau = t[idx]
    h_c = b1_c + a1 * tau
    prev = idx - 1
    w2 = s_w[-1] - s_w[prev]
    t2_sum = s_t[-1] - s_t[prev]
    y2_sum = s_y[-1] - s_y[prev]
    tt2_sum = s_tt[-1] - s_tt[prev]
    ty2_sum = s_ty[-1] - s_ty[prev]
    yy2_sum = s_yy[-1] - s_yy[prev]

    dd2 = tt2_sum - 2.0 * tau * t2_sum + w2 * tau * tau
    dy2 = ty2_sum - tau * y2_sum - h_c * t2_sum + w2 * tau * h_c
    yy2 = yy2_sum - 2.0 * h_c * y2_sum + w2 * h_c * h_c

    valid &= dd2 > 1e-24
    if not np.any(valid):
        raise InsufficientDataError("Collinear timestamps prevented STPD window estimation.")

    a2 = np.zeros_like(dd2)
    a2[valid] = dy2[valid] / dd2[valid]
    rss2 = np.maximum(0.0, yy2 - a2 * dy2)

    # Endpoint `ind` is included in both segment sums with identical residual (yc[ind] - h_c)
    r_hinge_sq = w_at_idx * (yc[idx] - h_c) ** 2
    total_rss = np.where(valid, np.maximum(0.0, rss1 + rss2 - r_hinge_sq), np.inf)

    best_pos = int(np.argmin(total_rss))
    best_ind = int(idx[best_pos])
    best_a1 = float(a1[best_pos])
    best_b1 = float(b1_c[best_pos] + y_mean)
    best_a2 = float(a2[best_pos])
    best_norm = float(np.sqrt(total_rss[best_pos]))
    return best_ind, best_b1, best_a1, best_a2, best_norm


@dataclass(frozen=True)
class _WindowTpdStats:
    local_tp: int
    slope_before: float
    slope_after: float
    direction: float
    pval: float
    fb_diff: int
    fb_time_diff: float
    snr_raw: float
    snr_centered: float
    ndri_l2: float
    ndri_rmse: float
    fitted: FloatArray
    n_left: int
    n_right: int


def _tpd_window(
    t: FloatArray,
    y: FloatArray,
    margin: int,
    w: FloatArray | None = None,
) -> _WindowTpdStats:
    """Run forward and backward sequential turning-point detection on one window."""
    n = int(t.size)
    t_fwd = t - t[0]
    ind_f, b1_f, a1_f, a2_f, norm_f = _fit_two_piece_sequential_scan(t_fwd, y, margin, w)
    pval_f = _stpd_pvalue(t_fwd, y, ind_f, b1_f, a1_f, a2_f, w)

    t_rev = t_fwd[-1] - t_fwd[::-1]
    y_rev = y[::-1]
    w_rev = w[::-1] if w is not None else None
    ind_b_rev, b1_b, a1_b, a2_b, norm_b = _fit_two_piece_sequential_scan(
        t_rev, y_rev, margin, w_rev
    )
    ind_b = n - 1 - ind_b_rev
    pval_b = _stpd_pvalue(t_rev, y_rev, ind_b_rev, b1_b, a1_b, a2_b, w_rev)

    fb_diff = abs(ind_b - ind_f)
    fb_time_diff = float(abs(t_fwd[ind_b] - t_fwd[ind_f]))

    y_fit = np.empty(n, dtype=np.float64)
    if norm_f <= norm_b:
        tp = ind_f
        g1 = b1_f + a1_f * t_fwd[: tp + 1]
        g2 = a2_f * (t_fwd[tp:] - t_fwd[tp]) + b1_f + a1_f * t_fwd[tp]
        y_fit[: tp + 1] = g1
        y_fit[tp:] = g2
        slope_before = a1_f
        slope_after = a2_f
        pval = pval_f
    else:
        tp_rev = ind_b_rev
        g1_r = b1_b + a1_b * t_rev[: tp_rev + 1]
        g2_r = a2_b * (t_rev[tp_rev:] - t_rev[tp_rev]) + b1_b + a1_b * t_rev[tp_rev]
        y_fit_rev = np.empty(n, dtype=np.float64)
        y_fit_rev[: tp_rev + 1] = g1_r
        y_fit_rev[tp_rev:] = g2_r
        y_fit = y_fit_rev[::-1]
        tp = ind_b
        # Reversing time flips slope signs and swaps before/after segments
        slope_before = -a2_b
        slope_after = -a1_b
        pval = pval_b

    res_all = y - y_fit
    res_norm = float(np.linalg.norm(res_all))
    if res_norm <= 1e-24:
        snr_raw = np.inf
        snr_centered = np.inf
    else:
        snr_raw = float(np.linalg.norm(y_fit)) / res_norm
        snr_centered = float(np.linalg.norm(y_fit - np.mean(y_fit))) / res_norm

    r1 = y[: tp + 1] - y_fit[: tp + 1]
    r2 = y[tp:] - y_fit[tp:]
    nr1 = float(np.linalg.norm(r1))
    nr2 = float(np.linalg.norm(r2))
    ndri_l2 = abs(nr1 - nr2) / (nr1 + nr2) if (nr1 + nr2) > 1e-24 else 0.0

    n_left = tp + 1
    n_right = n - tp
    rmse1 = nr1 / np.sqrt(n_left)
    rmse2 = nr2 / np.sqrt(n_right)
    ndri_rmse = abs(rmse1 - rmse2) / (rmse1 + rmse2) if (rmse1 + rmse2) > 1e-24 else 0.0

    return _WindowTpdStats(
        local_tp=tp,
        slope_before=float(slope_before),
        slope_after=float(slope_after),
        direction=float(slope_after - slope_before),
        pval=pval,
        fb_diff=fb_diff,
        fb_time_diff=fb_time_diff,
        snr_raw=snr_raw,
        snr_centered=snr_centered,
        ndri_l2=float(ndri_l2),
        ndri_rmse=float(ndri_rmse),
        fitted=y_fit,
        n_left=n_left,
        n_right=n_right,
    )


def _chained_forward_fit(
    t: FloatArray,
    y: FloatArray,
    tp_sorted: IntArray,
    w: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Sequential chained connected OLS/WLS across sorted turning points."""
    n = int(t.size)
    m = int(tp_sorted.size)
    y_fit = np.empty(n, dtype=np.float64)

    if m == 0:
        if w is None:
            tm = float(np.mean(t))
            ym = float(np.mean(y))
            dt = t - tm
            a = float(np.dot(dt, y - ym)) / float(np.dot(dt, dt))
            b = ym - a * tm
        else:
            sw = float(np.sum(w))
            tm = float(np.sum(w * t)) / sw
            ym = float(np.sum(w * y)) / sw
            dt = t - tm
            a = float(np.sum(w * dt * (y - ym))) / float(np.sum(w * dt * dt))
            b = ym - a * tm
        y_fit[:] = b + a * t
        return np.array([a], dtype=np.float64), np.array([b], dtype=np.float64), y_fit

    bounds = np.append(tp_sorted, n - 1)
    slopes = np.empty(m + 1, dtype=np.float64)
    intercepts = np.empty(m + 1, dtype=np.float64)

    # First segment: unconstrained 2-param fit on 0..tp_0
    end0 = int(bounds[0]) + 1
    t1 = t[:end0]
    y1 = y[:end0]
    if w is None:
        tm = float(np.mean(t1))
        ym = float(np.mean(y1))
        dt = t1 - tm
        a0 = float(np.dot(dt, y1 - ym)) / float(np.dot(dt, dt))
        b0 = ym - a0 * tm
    else:
        w1 = w[:end0]
        sw1 = float(np.sum(w1))
        tm = float(np.sum(w1 * t1)) / sw1
        ym = float(np.sum(w1 * y1)) / sw1
        dt = t1 - tm
        a0 = float(np.sum(w1 * dt * (y1 - ym))) / float(np.sum(w1 * dt * dt))
        b0 = ym - a0 * tm
    slopes[0] = a0
    intercepts[0] = b0
    y_fit[:end0] = b0 + a0 * t1

    # Subsequent anchored segments
    for k in range(m):
        i_start = int(bounds[k])
        i_stop = int(bounds[k + 1]) + 1
        d_seg = t[i_start:i_stop] - t[i_start]
        y_seg = y[i_start:i_stop] - y_fit[i_start]
        if w is None:
            ak = float(np.dot(d_seg, y_seg)) / float(np.dot(d_seg, d_seg))
        else:
            wk = w[i_start:i_stop]
            ak = float(np.sum(wk * d_seg * y_seg)) / float(np.sum(wk * d_seg * d_seg))
        slopes[k + 1] = ak
        intercepts[k + 1] = y_fit[i_start] - ak * t[i_start]
        y_fit[i_start:i_stop] = y_fit[i_start] + ak * d_seg

    return slopes, intercepts, y_fit


def _fit_piecewise_core(
    t_num: FloatArray,
    t_raw: np.ndarray[Any, Any],
    is_dt: bool,
    y: FloatArray,
    tp_valid: IntArray,
    valid_to_orig: IntArray,
    n_input: int,
    n_dropped_nan: int,
    sampling_rate_ratio: float,
    *,
    w: FloatArray | None = None,
    mode: RefitMode = "sequential",
) -> PiecewiseTrendResult:
    """Internal piecewise connected linear trend fitter (sequential or joint hinge-spline)."""
    n = int(t_num.size)
    tp_sorted = np.sort(np.unique(tp_valid.astype(np.intp)))
    m = int(tp_sorted.size)
    if m > 0 and (int(tp_sorted[0]) < 1 or int(tp_sorted[-1]) > n - 2):
        raise ValueError("`tp_indices` must lie strictly inside the series (1 <= tp <= n - 2).")

    t0 = t_num - t_num[0]

    if mode == "joint":
        # Continuous piecewise linear hinge basis: [1, t0, (t0 - tau_1)_+, ..., (t0 - tau_m)_+]
        D = np.empty((n, m + 2), dtype=np.float64)
        D[:, 0] = 1.0
        D[:, 1] = t0
        for k, tp in enumerate(tp_sorted):
            D[:, 2 + k] = np.maximum(0.0, t0 - t0[tp])
        pw = prepare_weights(w, n) if w is not None else prepare_weights(None, n)
        ols = solve_weighted_ols(D, y, pw)
        y_fit = y - ols.residuals
        b0 = float(ols.coeff[0])
        a0 = float(ols.coeff[1])
        deltas = ols.coeff[2:]
        slopes = np.empty(m + 1, dtype=np.float64)
        intercepts = np.empty(m + 1, dtype=np.float64)
        slopes[0] = a0
        intercepts[0] = b0 - a0 * t_num[0]
        cum_slope = a0
        for k, tp in enumerate(tp_sorted):
            cum_slope += float(deltas[k])
            slopes[k + 1] = cum_slope
            intercepts[k + 1] = float(y_fit[tp]) - cum_slope * float(t_num[tp])
        selected_dir: Literal["forward", "backward", "joint"] = "joint"
    else:
        slopes_f, intercepts_f, y_f = _chained_forward_fit(t0, y, tp_sorted, w)
        t_rev = t0[-1] - t0[::-1]
        y_rev = y[::-1]
        w_rev = w[::-1] if w is not None else None
        tp_rev = np.sort(n - 1 - tp_sorted)
        slopes_b_rev, _int_b_rev, y_b_rev = _chained_forward_fit(t_rev, y_rev, tp_rev, w_rev)
        slopes_b = -slopes_b_rev[::-1]
        y_b = y_b_rev[::-1]

        err_f = (
            float(np.linalg.norm(y - y_f))
            if w is None
            else float(np.sqrt(np.sum(w * (y - y_f) ** 2)))
        )
        err_b = (
            float(np.linalg.norm(y_rev - y_b_rev))
            if w_rev is None
            else float(np.sqrt(np.sum(w_rev * (y_rev - y_b_rev) ** 2)))
        )
        if err_f < err_b:
            slopes = slopes_f
            y_fit = y_f
            selected_dir = "forward"
        else:
            slopes = slopes_b
            y_fit = y_b
            selected_dir = "backward"

        intercepts = np.empty(m + 1, dtype=np.float64)
        bounds_full = [0, *(int(tp) for tp in tp_sorted), n - 1]
        for k in range(m + 1):
            anchor = bounds_full[k]
            intercepts[k] = float(y_fit[anchor]) - float(slopes[k]) * float(t_num[anchor])

    # Local per-TP statistics across adjacent segments [tp_{k-1}..tp_k] and [tp_k..tp_{k+1}]
    ext = np.concatenate(([0], tp_sorted, [n - 1]))
    directions = np.empty(m, dtype=np.float64)
    snr_raw_arr = np.empty(m, dtype=np.float64)
    snr_cent_arr = np.empty(m, dtype=np.float64)
    ndri_l2_arr = np.empty(m, dtype=np.float64)
    ndri_rmse_arr = np.empty(m, dtype=np.float64)

    for k in range(1, m + 1):
        i0, i1, i2 = int(ext[k - 1]), int(ext[k]), int(ext[k + 1])
        r1 = y[i0 : i1 + 1] - y_fit[i0 : i1 + 1]
        r2 = y[i1 : i2 + 1] - y_fit[i1 : i2 + 1]
        r_pair = y[i0 : i2 + 1] - y_fit[i0 : i2 + 1]
        y_pair = y_fit[i0 : i2 + 1]

        nr1 = float(np.linalg.norm(r1))
        nr2 = float(np.linalg.norm(r2))
        nr_all = float(np.linalg.norm(r_pair))

        directions[k - 1] = slopes[k] - slopes[k - 1]
        if nr_all <= 1e-24:
            snr_raw_arr[k - 1] = np.inf
            snr_cent_arr[k - 1] = np.inf
        else:
            snr_raw_arr[k - 1] = float(np.linalg.norm(y_pair)) / nr_all
            snr_cent_arr[k - 1] = float(np.linalg.norm(y_pair - np.mean(y_pair))) / nr_all
        ndri_l2_arr[k - 1] = abs(nr2 - nr1) / (nr1 + nr2) if (nr1 + nr2) > 1e-24 else 0.0
        rmse1 = nr1 / np.sqrt(i1 - i0 + 1)
        rmse2 = nr2 / np.sqrt(i2 - i1 + 1)
        ndri_rmse_arr[k - 1] = (
            abs(rmse2 - rmse1) / (rmse1 + rmse2) if (rmse1 + rmse2) > 1e-24 else 0.0
        )

    residuals = y - y_fit
    rn, rn_warns = compute_red_noise_diagnostics(t_num, residuals)
    diag = DiagnosticReport(
        red_noise=rn,
        condition_number_max=1.0,
        n_input=n_input,
        n_valid=n,
        n_dropped_nan=n_dropped_nan,
        sampling_rate_ratio=sampling_rate_ratio,
        warnings=rn_warns,
    )

    return PiecewiseTrendResult(
        tp_indices=valid_to_orig[tp_sorted] if m > 0 else np.empty(0, dtype=np.intp),
        tp_valid_indices=tp_sorted,
        slopes=slopes,
        intercepts=intercepts,
        fitted_trend=y_fit,
        residuals=residuals,
        directions=directions,
        local_snr_raw=snr_raw_arr,
        local_snr_centered=snr_cent_arr,
        local_ndri_l2=ndri_l2_arr,
        local_ndri_rmse=ndri_rmse_arr,
        mode=mode,
        selected_direction=selected_dir,
        times_numeric=t_num,
        times_raw=t_raw,
        is_datetime=is_dt,
        diagnostics=diag,
    )


def fit_piecewise_trend(
    t: TimeLike,
    y: ArrayLike,
    tp_indices: Sequence[int],
    *,
    weights: ArrayLike | None = None,
    mode: RefitMode = "sequential",
) -> PiecewiseTrendResult:
    """Fit a global connected piecewise-linear trend across known turning points.

    Parameters
    ----------
    t, y:
        Time series observations (``NaN`` entries dropped cleanly).
    tp_indices:
        0-based turning-point indices on the input axis.
    weights:
        Optional 1D strictly positive observation weights.
    mode:
        ``'sequential'`` (Ghaderpour et al., 2024 ``TPTR`` forward/backward
        chained OLS) or ``'joint'`` (globally optimal simultaneous continuous
        hinge-spline regression).
    """
    if mode not in ("sequential", "joint"):
        raise ValueError(f"Unknown `mode` {mode!r}. Options: 'sequential', 'joint'.")
    series = prepare_numeric_series(t, y)
    w_arr: FloatArray | None = None
    if weights is not None:
        pw_full = prepare_weights(weights, series.n_input)
        if pw_full.kind == "matrix":
            raise ValueError("`fit_piecewise_trend` supports scalar or 1D `weights` only.")
        pw_sub = pw_full.subset(series.valid_mask) if series.n_dropped_nan else pw_full
        w_arr = pw_sub.vector

    orig_tp = np.asarray(tp_indices, dtype=np.intp)
    if orig_tp.size > 0:
        if np.any(orig_tp < 0) or np.any(orig_tp >= series.n_input):
            raise ValueError("`tp_indices` contains out-of-range indices.")
        tp_valid = (
            np.array(
                [int(np.searchsorted(series.valid_to_orig, idx)) for idx in orig_tp],
                dtype=np.intp,
            )
            if series.n_dropped_nan
            else orig_tp
        )
    else:
        tp_valid = np.empty(0, dtype=np.intp)

    return _fit_piecewise_core(
        series.times_numeric,
        series.times_raw,
        series.is_datetime,
        series.values,
        tp_valid,
        series.valid_to_orig,
        series.n_input,
        series.n_dropped_nan,
        series.sampling_rate_ratio,
        w=w_arr,
        mode=mode,
    )


def _run_stpd_windows(
    t_valid: FloatArray,
    y_valid: FloatArray,
    w_valid: FloatArray | None,
    windows: list[tuple[int, int]],
    margin: int,
    snr_th: float,
    snr_mode: SnrMode,
    ndri_th: float,
    ndri_mode: NdriMode,
    alpha: float,
    dir_th: float,
    tp_th: float,
    max_fb_diff: int,
    max_fb_time_diff: float | None,
    *,
    provenance: CensoringProvenance | None = None,
    censor_cfg: CensoredConfig | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[
    list[tuple[int, float, _WindowTpdStats, int, int]],
    list[tuple[int, _WindowTpdStats]],
    bool,
    int,
]:
    """Run STPD across sliding windows, returning accepted TPs and high-NDRI jump candidates."""
    accepted_raw: list[tuple[int, float, _WindowTpdStats, int, int]] = []
    flagged_raw: list[tuple[int, _WindowTpdStats]] = []
    density_imbalanced = False
    window_fallbacks = 0

    use_local_censor = (
        provenance is not None
        and censor_cfg is not None
        and censor_cfg.scope == "windowed"
        and provenance.n_censored > 0
        and rng is not None
    )

    for w_start, w_stop in windows:
        n_w = w_stop - w_start
        t_w = t_valid[w_start:w_stop]
        if use_local_censor:
            assert provenance is not None and censor_cfg is not None and rng is not None
            y_w, fb = impute_window_local(provenance, censor_cfg, w_start, w_stop, rng)
            if fb:
                window_fallbacks += 1
        else:
            y_w = y_valid[w_start:w_stop]
        w_w = w_valid[w_start:w_stop] if w_valid is not None else None
        m_eff = min(margin, (n_w - 3) // 2)
        if m_eff < 2:
            continue

        stats = _tpd_window(t_w, y_w, m_eff, w_w)
        ratio_lr = max(stats.n_left, stats.n_right) / max(1, min(stats.n_left, stats.n_right))
        if ratio_lr > 1.5:
            density_imbalanced = True

        snr_val = stats.snr_centered if snr_mode == "centered" else stats.snr_raw
        ndri_val = stats.ndri_rmse if ndri_mode == "rmse" else stats.ndri_l2
        fb_ok = (
            (stats.fb_time_diff <= max_fb_time_diff)
            if max_fb_time_diff is not None
            else (stats.fb_diff <= max_fb_diff)
        )

        sig_and_strong = (
            (not np.isnan(stats.pval))
            and (stats.pval < alpha)
            and (snr_val > snr_th)
            and (abs(stats.direction) > dir_th)
        )

        g_idx = w_start + stats.local_tp
        if fb_ok and sig_and_strong and ndri_val < ndri_th:
            dist_center = abs(stats.local_tp - (n_w - 1) / 2.0)
            accepted_raw.append((g_idx, dist_center, stats, w_start, w_stop))
        elif sig_and_strong and ndri_val >= ndri_th:
            flagged_raw.append((g_idx, stats))

    # Merge nearby accepted TPs within tp_th, keeping closest to window center
    accepted_raw.sort(key=lambda item: item[0])
    r = 1
    while r < len(accepted_raw):
        if t_valid[accepted_raw[r][0]] - t_valid[accepted_raw[r - 1][0]] < tp_th:
            if accepted_raw[r][1] < accepted_raw[r - 1][1]:
                accepted_raw[r - 1] = accepted_raw[r]
            del accepted_raw[r]
        else:
            r += 1

    # Deduplicate nearby flagged jump candidates within tp_th, keeping highest NDRI
    flagged_raw.sort(key=lambda item: item[0])
    r = 1
    while r < len(flagged_raw):
        if t_valid[flagged_raw[r][0]] - t_valid[flagged_raw[r - 1][0]] < tp_th:
            nd_curr = flagged_raw[r][1].ndri_rmse if ndri_mode == "rmse" else flagged_raw[r][1].ndri_l2
            nd_prev = flagged_raw[r - 1][1].ndri_rmse if ndri_mode == "rmse" else flagged_raw[r - 1][1].ndri_l2
            if nd_curr > nd_prev:
                flagged_raw[r - 1] = flagged_raw[r]
            del flagged_raw[r]
        else:
            r += 1

    return accepted_raw, flagged_raw, density_imbalanced, window_fallbacks


def detect_turning_points(
    t: TimeLike,
    y: ValueLike,
    *,
    weights: ArrayLike | None = None,
    size: int | None = None,
    step: int | None = None,
    window_mode: WindowMode = "count",
    window_duration: float | None = None,
    step_duration: float | None = None,
    min_window_obs: int | None = None,
    snr_th: float = 1.0,
    snr_mode: SnrMode = "centered",
    ndri_th: float = 0.3,
    ndri_mode: NdriMode = "rmse",
    alpha: float = 0.01,
    dir_th: float = 0.0,
    tp_th: float = 1.0,
    margin: int | None = None,
    max_fb_diff: int = 3,
    max_fb_time_diff: float | None = None,
    refit_mode: RefitMode = "sequential",
    censor_status: ArrayLike | None = None,
    censoring: CensoredConfig | None = None,
    parity_mode: bool = False,
) -> StpdResult:
    """Detect continuous trend turning points via STPD (Ghaderpour et al., 2024).

    Fits connected two-piece linear trends forward and backward within sliding
    windows, testing slope-change significance (Student t-test), forward/backward
    agreement, signal-to-noise ratio (SNR), and the Normalized Difference
    Residual Index (NDRI) to separate genuine continuous turning points
    (``NDRI ≈ 0``) from abrupt datum jumps (``NDRI -> 1``).

    Parameters
    ----------
    t, y:
        Unevenly or regularly spaced timestamps and observations (numeric or
        censored). ``NaN`` entries are dropped without interpolation.
    weights:
        Optional 1D strictly positive observation weights.
    size, step:
        Observation-count window size and step (defaults: ``5*M`` and ``M``).
    window_mode:
        ``'count'`` (default) or ``'duration'`` (fixed calendar span).
    snr_th, snr_mode:
        SNR threshold (default ``1.0``) and mode: ``'centered'`` (default,
        shift-invariant for non-zero baselines) or ``'raw'`` (Ghaderpour 2024).
        ``parity_mode=True`` forces ``'raw'``.
    ndri_th, ndri_mode:
        NDRI threshold (default ``0.3``) and mode: ``'rmse'`` (default,
        unbiased under uneven sampling density) or ``'l2'`` (Ghaderpour 2024).
        ``parity_mode=True`` forces ``'l2'``.
    alpha:
        Significance level for the two-sided Student t-test (default ``0.01``).
    dir_th:
        Minimum absolute slope change ``|slope2 - slope1|`` (default ``0.0``).
    tp_th:
        Minimum time separation between accepted turning points (default ``1.0``).
    margin:
        Number of edge observations excluded on each side of a window (enforced
        ``>= 2`` to avoid 0/0 variance singularities).
    max_fb_diff, max_fb_time_diff:
        Maximum allowed index difference (default ``3``, matching upstream ``< 4``)
        or time difference between forward and backward turning-point estimates.
    refit_mode:
        Global piecewise trend refit mode: ``'sequential'`` (TPTR) or ``'joint'``.
    censor_status, censoring:
        Optional censoring indicator array and :class:`CensoredConfig`.
    parity_mode:
        If ``True``, uses exact Ghaderpour et al. (2024) conventions
        (``snr_mode='raw'``, ``ndri_mode='l2'``).
    """
    if window_mode not in ("count", "duration"):
        raise ValueError(f"Unknown `window_mode` {window_mode!r}. Options: 'count', 'duration'.")
    if snr_mode not in ("raw", "centered"):
        raise ValueError(f"Unknown `snr_mode` {snr_mode!r}. Options: 'raw', 'centered'.")
    if ndri_mode not in ("l2", "rmse"):
        raise ValueError(f"Unknown `ndri_mode` {ndri_mode!r}. Options: 'l2', 'rmse'.")
    if refit_mode not in ("sequential", "joint"):
        raise ValueError(f"Unknown `refit_mode` {refit_mode!r}. Options: 'sequential', 'joint'.")
    if snr_th <= 0.0:
        raise ValueError("`snr_th` must be strictly positive.")
    if not (0.0 <= ndri_th <= 1.0):
        raise ValueError("`ndri_th` must lie in [0, 1].")
    if not (0.0 < alpha < 1.0):
        raise ValueError("`alpha` must lie in (0, 1).")
    if dir_th < 0.0:
        raise ValueError("`dir_th` must be non-negative.")
    if tp_th <= 0.0:
        raise ValueError("`tp_th` must be strictly positive.")
    if max_fb_diff < 0:
        raise ValueError("`max_fb_diff` must be non-negative.")

    eff_snr_mode: SnrMode = "raw" if parity_mode else snr_mode
    eff_ndri_mode: NdriMode = "l2" if parity_mode else ndri_mode

    prep = prepare_series_with_censoring(t, y, censor_status=censor_status, censoring=censoring)
    series = prep.series
    prov = prep.provenance
    cfg = prep.config

    w_valid: FloatArray | None = None
    if weights is not None:
        pw_full = prepare_weights(weights, series.n_input)
        if pw_full.kind == "matrix":
            raise ValueError("STPD supports scalar or 1D observation `weights` only.")
        pw_sub = pw_full.subset(series.valid_mask) if series.n_dropped_nan else pw_full
        w_valid = pw_sub.vector
    if prep.weight_factors is not None:
        w_valid = (
            prep.weight_factors
            if w_valid is None
            else (w_valid * prep.weight_factors)
        )

    warnings_list: list[WarningItem] = list(prep.warnings)
    n = series.n_valid
    span = float(series.times_numeric[-1] - series.times_numeric[0])
    if span <= 0.0:
        raise InsufficientDataError("Time span must be strictly positive.")

    if window_mode == "count":
        rate = max(1, int(n / span))
        win_size = 5 * rate if size is None else int(size)
        if win_size > n or win_size < 5:
            raise InsufficientDataError(
                f"STPD window `size` ({win_size}) must satisfy 5 <= size <= n ({n})."
            )
        win_step = min(rate, max(1, win_size - 1)) if step is None else int(step)
        if win_step < 1 or win_step >= win_size:
            raise ValueError(
                f"Translation `step` ({win_step}) must satisfy 1 <= step < size ({win_size})."
            )
        windows: list[tuple[int, int]] = []
        a = 0
        b = win_size
        while True:
            if b >= n:
                windows.append((n - win_size, n))
                break
            windows.append((a, b))
            a += win_step
            b = a + win_size

        if not parity_mode and series.sampling_rate_ratio > 2.0:
            warnings_list.append(
                WarningItem(
                    code="VARIABLE_SAMPLING_RATE",
                    severity="caution",
                    message=(
                        f"Sampling density varies by ~{series.sampling_rate_ratio:.1f}x across "
                        "the record while using fixed observation-count windows."
                    ),
                    suggestion="Consider `window_mode='duration'` for uniform calendar windows.",
                )
            )
        default_margin = max(2, min(rate, win_size // 5)) if not parity_mode else 2
        eff_margin = default_margin if margin is None else int(margin)
        if eff_margin < 2:
            if not parity_mode:
                warnings_list.append(
                    WarningItem(
                        code="MARGIN_CLAMPED",
                        severity="info",
                        message=(
                            f"Requested margin={eff_margin} raised to minimum safe value 2 "
                            "to avoid 0/0 slope-variance singularity."
                        ),
                    )
                )
            eff_margin = 2
        if eff_margin >= win_size / 2:
            raise ValueError(f"`margin` ({eff_margin}) must be < size / 2 ({win_size / 2}).")
    else:
        min_obs = int(min_window_obs) if min_window_obs is not None else 10
        windows, _dur, skipped = _build_duration_windows(
            series.times_numeric,
            window_duration if window_duration is not None else min(5.0, span),
            step_duration,
            max(6, min_obs),
        )
        if skipped > 0:
            warnings_list.append(
                WarningItem(
                    code="SPARSE_WINDOW_SKIPPED",
                    severity="info",
                    message=f"Skipped {skipped} duration window(s) below min_window_obs={min_obs}.",
                )
            )
        med_w = int(np.median([b - a for a, b in windows]))
        eff_margin = max(2, med_w // 6) if margin is None else max(2, int(margin))

    # Diagnostic warnings for raw SNR on large-baseline series
    y_std = float(np.std(series.values))
    y_mean = float(np.mean(series.values))
    if eff_snr_mode == "raw" and y_std > 0.0 and abs(y_mean) > 5.0 * y_std:
        warnings_list.append(
            WarningItem(
                code="RAW_SNR_LARGE_BASELINE",
                severity="caution",
                message=(
                    f"Series has a large non-zero baseline (|mean|={abs(y_mean):.4g} > "
                    f"5 * std={5 * y_std:.4g}) with snr_mode='raw', which inflates SNR."
                ),
                suggestion="Use `snr_mode='centered'` (default) for baseline-invariant SNR.",
            )
        )

    if cfg is not None and isinstance(cfg.random_state, np.random.Generator):
        rng: np.random.Generator = cfg.random_state
    else:
        rng = np.random.default_rng(cfg.random_state if cfg is not None else 0)

    accepted_raw, flagged_raw, density_imbalanced, win_fb = _run_stpd_windows(
        series.times_numeric,
        series.values,
        w_valid,
        windows,
        eff_margin,
        snr_th,
        eff_snr_mode,
        ndri_th,
        eff_ndri_mode,
        alpha,
        dir_th,
        tp_th,
        max_fb_diff,
        max_fb_time_diff,
        provenance=prov,
        censor_cfg=cfg,
        rng=rng,
    )

    if eff_ndri_mode == "l2" and density_imbalanced:
        warnings_list.append(
            WarningItem(
                code="NDRI_L2_DENSITY_IMBALANCE",
                severity="caution",
                message=(
                    "Left/right sample counts around turning points differed by >1.5x while "
                    "using ndri_mode='l2', which can bias NDRI upward."
                ),
                suggestion="Use `ndri_mode='rmse'` (default) for unevenly sampled records.",
            )
        )
    if win_fb > 0:
        warnings_list.append(
            WarningItem(
                code="WINDOW_IMPUTATION_FALLBACK",
                severity="info",
                message=(
                    f"Windowed censoring imputation fell back to global imputation in "
                    f"{win_fb} heavily censored window(s)."
                ),
            )
        )

    # Stability sweep for censored series if requested
    stability_counts: dict[int, int] = {item[0]: 0 for item in accepted_raw}
    n_stab = cfg.n_stability_draws if (cfg is not None and prov is not None and prov.n_censored > 0) else 0
    if n_stab > 0 and accepted_raw and prov is not None and cfg is not None:
        dt_med = float(np.median(np.diff(series.times_numeric))) if n > 1 else 0.1
        match_tol = max(0.4, 3.0 * dt_med)
        for y_draw in generate_stability_draws(prov, cfg, series.times_numeric):
            acc_d, _, _, _ = _run_stpd_windows(
                series.times_numeric,
                y_draw,
                w_valid,
                windows,
                eff_margin,
                snr_th,
                eff_snr_mode,
                ndri_th,
                eff_ndri_mode,
                alpha,
                dir_th,
                tp_th,
                max_fb_diff,
                max_fb_time_diff,
            )
            draw_times = [float(series.times_numeric[it[0]]) for it in acc_d]
            for item in accepted_raw:
                t_ref = float(series.times_numeric[item[0]])
                if any(abs(td - t_ref) <= match_tol for td in draw_times):
                    stability_counts[item[0]] += 1

    # Global connected piecewise-linear refit across accepted TPs
    tp_valid_arr = np.array([item[0] for item in accepted_raw], dtype=np.intp)
    pw_trend = _fit_piecewise_core(
        series.times_numeric,
        series.times_raw,
        series.is_datetime,
        series.values,
        tp_valid_arr,
        series.valid_to_orig,
        series.n_input,
        series.n_dropped_nan,
        series.sampling_rate_ratio,
        w=w_valid,
        mode=refit_mode,
    )

    tp_events: list[TurningPointEvent] = []
    for k, (g_idx, dist_c, st, w_start, w_stop) in enumerate(accepted_raw):
        orig_idx = int(series.valid_to_orig[g_idx])
        t_yr = float(series.times_numeric[g_idx])
        win_cen = (
            float(np.mean(prov.is_censored[w_start:w_stop])) if prov is not None else 0.0
        )
        if win_cen > 0.50:
            warnings_list.append(
                WarningItem(
                    code="WINDOW_HEAVY_CENSORING",
                    severity="warning",
                    message=(
                        f"Turning point near {format_timestamp(series.times_raw[g_idx], series.is_datetime)} "
                        f"is supported by a window that is {win_cen:.0%} censored."
                    ),
                )
            )
        coincides_dl = (
            check_dl_shift_coincidence(t_yr, prov.dl_shifts, series.times_numeric)
            if prov is not None
            else False
        )
        if coincides_dl:
            warnings_list.append(
                WarningItem(
                    code="DL_SHIFT_COINCIDENCE",
                    severity="warning",
                    message=(
                        f"Turning point near {format_timestamp(series.times_raw[g_idx], series.is_datetime)} "
                        "coincides with a detection-limit shift."
                    ),
                )
            )
        stab_rate = (stability_counts[g_idx] / n_stab) if n_stab > 0 else None
        tp_events.append(
            TurningPointEvent(
                index=orig_idx,
                valid_index=g_idx,
                time=series.times_raw[g_idx],
                time_year=t_yr,
                slope_before=float(pw_trend.slopes[k]),
                slope_after=float(pw_trend.slopes[k + 1]),
                direction=float(pw_trend.directions[k]),
                pvalue=st.pval,
                snr_raw=float(pw_trend.local_snr_raw[k]),
                snr_centered=float(pw_trend.local_snr_centered[k]),
                ndri_l2=float(pw_trend.local_ndri_l2[k]),
                ndri_rmse=float(pw_trend.local_ndri_rmse[k]),
                fb_diff=st.fb_diff,
                window_center_distance=dist_c,
                window_censored_fraction=win_cen,
                coincides_with_dl_shift=coincides_dl,
                stability_rate=stab_rate,
            )
        )

    flagged_events = tuple(
        FlaggedJumpCandidate(
            index=int(series.valid_to_orig[g_idx]),
            valid_index=g_idx,
            time=series.times_raw[g_idx],
            time_year=float(series.times_numeric[g_idx]),
            direction=st.direction,
            pvalue=st.pval,
            ndri_l2=st.ndri_l2,
            ndri_rmse=st.ndri_rmse,
            fb_diff=st.fb_diff,
        )
        for g_idx, st in flagged_raw
    )
    if flagged_events:
        warnings_list.append(
            WarningItem(
                code="HIGH_NDRI_JUMP_CANDIDATE",
                severity="caution",
                message=(
                    f"STPD flagged {len(flagged_events)} high-NDRI candidate(s) that look more "
                    "like abrupt vertical jumps than continuous turning points."
                ),
                suggestion="Run `just_jumps()` or `analyze_series()` to test for datum jumps.",
            )
        )

    warnings_list.extend(pw_trend.diagnostics.warnings)
    diag = DiagnosticReport(
        red_noise=pw_trend.diagnostics.red_noise,
        condition_number_max=pw_trend.diagnostics.condition_number_max,
        n_input=series.n_input,
        n_valid=series.n_valid,
        n_dropped_nan=series.n_dropped_nan,
        sampling_rate_ratio=series.sampling_rate_ratio,
        warnings=tuple(warnings_list),
    )

    return StpdResult(
        turning_points=tuple(tp_events),
        flagged_jumps=flagged_events,
        piecewise_trend=pw_trend,
        times_numeric=series.times_numeric,
        times_raw=series.times_raw,
        values=series.values,
        is_datetime=series.is_datetime,
        window_mode=window_mode,
        snr_mode=eff_snr_mode,
        ndri_mode=eff_ndri_mode,
        alpha=alpha,
        snr_th=snr_th,
        ndri_th=ndri_th,
        dir_th=dir_th,
        tp_th=tp_th,
        censoring=prov,
        diagnostics=diag,
    )


# Concise top-level alias
stpd = detect_turning_points
