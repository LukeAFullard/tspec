"""Sequential Turning Point Detection (STPD) and Piecewise Linear Trend Estimation.

Clean-room implementation grounded in:
- Ghaderpour, E., Antonielli, B., Bozzano, F., Scarascia Mugnozza, G., &
  Mazzanti, P. (2024). A fast and robust method for detecting trend turning
  points in InSAR displacement time series. *Computers & Geosciences*, 185,
  105546. https://doi.org/10.1016/j.cageo.2024.105546
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from scipy.stats import t as student_t

from ._cenreg import pool_pvalues, window_congenial_imputations
from ._diagnostics import (
    DiagnosticReport,
    WarningItem,
    compute_red_noise_diagnostics,
    time_unit_warning,
)
from ._linalg import prepare_weights, solve_weighted_ols
from ._robust import (
    RobustMethod,
    fit_l1_piecewise,
    huber_nested_ftest,
    huber_rho,
    solve_huber_irls,
    tukey_weight,
    validate_robust_args,
)
from ._serial import (
    NO_SERIAL,
    SerialMode,
    SerialSpec,
    correlated_normals,
    deflate_t_pvalue,
    heavy_tail_warning,
    resolve_serial_spec,
    serial_warning_message,
)
from ._time import format_timestamp, prepare_numeric_series
from ._types import ArrayLike, FloatArray, IntArray, TimeLike, ValueLike
from .censored import (
    CensoredConfig,
    CensoringProvenance,
    _spawn_stability_rngs,
    check_dl_shift_coincidence,
    generate_stability_draws,
    impute_window_local,
    prepare_series_with_censoring,
    resolve_imputation_dist,
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
# Default SNR gate outside parity mode (the published STPD gate is 1.0)
SNR_TH_DEFAULT = 0.0
NdriMode = Literal["l2", "rmse"]
RefitMode = Literal["sequential", "joint", "l1", "huber"]
JumpTest = Literal["step", "ndri", "both"]
# Minimum share of a break's explained variance that only a vertical step captures before
# a significant step test classifies the break as a jump (see _step_vs_hinge_pvalue).
STEP_SHARE_MIN = 0.25


@dataclass(frozen=True)
class TurningPointEvent:
    """Accepted continuous trend turning point detected by STPD.

    Attributes
    ----------
    index, valid_index : int
        Turning-point observation on the input / valid axis.
    time, time_year
        Its timestamp (original representation / decimal years).
    slope_before, slope_after, direction : float
        Adjacent segment slopes from the global connected refit and their difference
        (``analyze_series`` segment slopes from separate per-segment fits can differ
        slightly).
    pvalue : float
        p-value of the slope change. With ``tp_calibration='montecarlo'`` (default) or
        ``'bonferroni'`` it accounts for the knot search within the window and is
        Holm-adjusted across candidate locations; with ``'none'`` (parity mode) it is
        the raw fixed-knot screening p-value at the selected knot (optimistic).
    snr_raw, snr_centered, ndri_l2, ndri_rmse : float
        Local SNR and NDRI over the adjacent segments.
    fb_diff : int
        Index difference between forward and backward estimates.
    window_center_distance : float
        Distance from the detecting window's center.
    window_censored_fraction, coincides_with_dl_shift, stability_rate
        Censored-data diagnostics.
    detection_window : tuple of int or None
        ``(start, stop, margin)`` of the detecting window on the valid axis (used by
        the bootstrap bands).
    """

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
    detection_window: tuple[int, int, int] | None = None
    step_pvalue: float = float("nan")
    step_share: float = float("nan")


@dataclass(frozen=True)
class FlaggedJumpCandidate:
    """Significant break rejected as a turning point because it looks like an abrupt jump.

    Attributes
    ----------
    index, valid_index : int
        0-based candidate index on the input / valid axis.
    time : Any
        Candidate timestamp in the original time representation.
    time_year : float
        Candidate timestamp in continuous decimal years.
    direction : float
        Estimated slope change (post minus pre, per time unit).
    pvalue : float
        Screening or calibrated slope-change p-value.
    ndri_l2, ndri_rmse : float
        Normalized difference residual index (L2 norm and RMSE variants).
    fb_diff : int
        Forward-minus-backward sequential knot index discrepancy.
    step_pvalue : float
        Bonferroni-corrected p-value of the step-vs-continuous-hinge test (small values
        mean a vertical step explains the break better than any continuous bend).
    step_share : float
        Fraction of the break's explained variance that only a vertical step captures.
    """

    index: int
    valid_index: int
    time: Any
    time_year: float
    direction: float
    pvalue: float
    ndri_l2: float
    ndri_rmse: float
    fb_diff: int
    step_pvalue: float = float("nan")
    step_share: float = float("nan")


@dataclass(frozen=True)
class PiecewiseTrendResult(BaseResult):
    """Global connected piecewise-linear trend fit across turning points.

    Attributes
    ----------
    tp_indices, tp_valid_indices : ndarray
        Turning-point indices on the input axis / valid (NaN-dropped) axis.
    slopes, intercepts : ndarray
        Per-segment slopes (per time unit) and intercepts.
    fitted_trend, residuals : ndarray
        Fitted connected trend and residuals on the valid observations.
    directions : ndarray
        Slope change at each turning point.
    local_snr_raw, local_snr_centered, local_ndri_l2, local_ndri_rmse : ndarray
        SNR and NDRI over the two segments adjacent to each turning point.
    mode : str
        ``"sequential"``, ``"joint"``, ``"l1"``, ``"huber"`` (or ``"segmented"``
        from ``analyze_series``).
    selected_direction : {"forward", "backward", "joint"}
        Which sequential fit direction was retained.
    times_numeric, times_raw, is_datetime
        Valid timestamps.
    diagnostics : DiagnosticReport
        Warnings and red-noise diagnostics.
    ci_level : float
        Pointwise confidence level.
    trend_ci_low, trend_ci_high : ndarray or None
        Pointwise band. From :func:`fit_piecewise_trend` it is conditional on the
        given turning points (``None`` for ``mode="l1"``); from
        :func:`detect_turning_points` with ``band_method='bootstrap'`` it includes
        knot-location uncertainty.
    """

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
    ci_level: float = 0.95
    trend_ci_low: FloatArray | None = None
    trend_ci_high: FloatArray | None = None

    def summary(self) -> str:
        lines = [
            f"Piecewise Linear Trend Result (mode={self.mode!r}, "
            f"fit={self.selected_direction!r}, segments={self.slopes.size})",
            f"Turning points : {self.tp_indices.tolist()}",
            f"Segment slopes : {[round(float(s), 4) for s in self.slopes]}",
        ]
        if self.tp_indices.size > 0:
            if self.slopes.size == self.tp_indices.size + 1:
                lines.append(
                    "  TP_Idx  Post-Slope   Direction    SNR(cent)   NDRI(rmse)  NDRI(L2)"
                )
                for k in range(self.tp_indices.size):
                    lines.append(
                        f"  {int(self.tp_indices[k]):<7d} {self.slopes[k + 1]:+11.4f}  "
                        f"{self.directions[k]:+11.4f}  {self.local_snr_centered[k]:10.3f}  "
                        f"{self.local_ndri_rmse[k]:10.3f}  {self.local_ndri_l2[k]:8.3f}"
                    )
            else:
                lines.append("  TP_Idx   Direction    SNR(cent)   NDRI(rmse)  NDRI(L2)")
                for k in range(self.tp_indices.size):
                    lines.append(
                        f"  {int(self.tp_indices[k]):<7d} {self.directions[k]:+11.4f}  "
                        f"{self.local_snr_centered[k]:10.3f}  "
                        f"{self.local_ndri_rmse[k]:10.3f}  {self.local_ndri_l2[k]:8.3f}"
                    )
        lines.extend(format_diagnostics_block(self.diagnostics))
        return "\n".join(lines)

    def explain(self) -> str:
        if self.mode == "segmented":
            lead = (
                f"Fitted a segmented piecewise-linear trend across epochs separated by abrupt jumps "
                f"({self.slopes.size} linear segment(s) across {self.diagnostics.n_valid} observations)."
            )
        else:
            lead = (
                f"Fitted a continuous connected trend with {self.slopes.size} linear segment(s) "
                f"across {self.diagnostics.n_valid} observations."
            )
        lines = [lead]
        for k, s in enumerate(self.slopes, start=1):
            lines.append(f"  - Segment {k} slope: {s:+.4g} per unit time.")
        lines.extend(format_explain_warnings(self.diagnostics))
        return "\n".join(lines)

    def to_dataframe(self) -> Any:
        pd = require_pandas()
        data: dict[str, Any] = {
            "time": self.times_raw,
            "time_year": self.times_numeric,
            "fitted_trend": self.fitted_trend,
            "residual": self.residuals,
        }
        if self.trend_ci_low is not None and self.trend_ci_high is not None:
            data["trend_ci_low"] = self.trend_ci_low
            data["trend_ci_high"] = self.trend_ci_high
        return pd.DataFrame(data)

    def plot(self, ax: Any = None, **kwargs: Any) -> tuple[Any, Any]:
        from .plotting import plot_piecewise_trend

        return plot_piecewise_trend(self, ax=ax, **kwargs)


@dataclass(frozen=True)
class StpdResult(BaseResult):
    """Result object returned by :func:`detect_turning_points` (:func:`stpd`).

    Attributes
    ----------
    turning_points : tuple of TurningPointEvent
        Accepted continuous turning points.
    flagged_jumps : tuple of FlaggedJumpCandidate
        Significant candidates rejected because of high NDRI (likely vertical jumps).
    piecewise_trend : PiecewiseTrendResult
        Global connected piecewise-linear refit across the accepted turning points.
    times_numeric, times_raw, values, is_datetime
        Valid timestamps and (imputed) values.
    window_mode, snr_mode, ndri_mode : str
        Settings used.
    alpha, snr_th, ndri_th, dir_th, tp_th : float
        Thresholds used (``snr_th`` is the resolved value).
    censoring : CensoringProvenance or None
        Censored-data provenance.
    diagnostics : DiagnosticReport
        Warnings and red-noise diagnostics.
    tp_calibration : str
        How turning-point p-values were calibrated for the knot search:
        ``'montecarlo'``, ``'bonferroni'`` or ``'none'``.
    n_imputations : int
        Number of imputations whose candidate p-values were pooled (Licht–Rubin) for
        censored data; ``1`` means a single imputation.
    band_method : str
        How ``piecewise_trend.trend_ci_*`` was computed: ``'bootstrap'`` (detect-then-fit
        bootstrap including knot-location uncertainty) or ``'conditional'``.
    """

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
    tp_calibration: str = "none"
    n_imputations: int = 1
    band_method: str = "conditional"
    robust: str = "none"
    jump_test: str = "ndri"
    serial_correlation: str = "none"
    serial_rho: float = 0.0
    serial_rho_used: float = 0.0

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
            f"Thresholds     : SNR > {self.snr_th}, "
            + (f"NDRI < {self.ndri_th}, " if self.jump_test in ("ndri", "both") else "")
            + ("step-vs-bend test not significant, " if self.jump_test in ("step", "both") else "")
            + f"|dir| > {self.dir_th}, min separation >= {self.tp_th}",
            f"Turning points : {len(self.turning_points)} accepted "
            f"({len(self.flagged_jumps)} jump candidate(s) flagged, jump_test={self.jump_test!r})",
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
            test_name = "robust nested F-test" if self.robust != "none" else "t-test"
            p_lbl = (
                f"family-wise calibrated {test_name}"
                if self.tp_calibration in ("montecarlo", "bonferroni")
                else f"uncalibrated screening {test_name}"
            )
            if self.n_imputations > 1:
                p_lbl += f" pooled over {self.n_imputations} imputations"
            for k, tp in enumerate(self.turning_points, start=1):
                t_str = format_timestamp(tp.time, self.is_datetime)
                ndri_disp = tp.ndri_rmse if self.ndri_mode == "rmse" else tp.ndri_l2
                if self.jump_test == "ndri":
                    shape = f"NDRI={ndri_disp:.2f} — low NDRI is consistent with a smooth bend"
                else:
                    shape = (
                        f"step-vs-bend p={tp.step_pvalue:.2g}, step share {tp.step_share:.2f} — "
                        "consistent with a smooth bend rather than a vertical step"
                    )
                desc = (
                    f"  {k}. Around {t_str} (index {tp.index}): slope changed from "
                    f"{tp.slope_before:+.4g} to {tp.slope_after:+.4g} per unit time "
                    f"(change {tp.direction:+.4g}, {p_lbl} p={tp.pvalue:.2e}, {shape})."
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
            if self.jump_test == "ndri":
                t_list = ", ".join(
                    f"{format_timestamp(fj.time, self.is_datetime)} (NDRI={fj.ndri_rmse:.2f})"
                    for fj in self.flagged_jumps[:4]
                )
                why = "residual balance was strongly asymmetric"
            else:
                t_list = ", ".join(
                    f"{format_timestamp(fj.time, self.is_datetime)} (step p={fj.step_pvalue:.2g}, "
                    f"share {fj.step_share:.2f})"
                    for fj in self.flagged_jumps[:4]
                )
                why = "a vertical step fits significantly better than any continuous bend"
            lines.append(
                f"Jump Alarm Bell: {len(self.flagged_jumps)} candidate change(s) were rejected "
                f"as smooth turning points because {why} ({t_list}). This usually indicates an "
                "abrupt vertical jump — run `just_jumps()` or `analyze_series()` to check!"
            )

        if self.turning_points and self.band_method != "bootstrap":
            lines.append(
                "Note: the piecewise-trend band is conditional on the detected turning points "
                "and omits their location uncertainty (in simulation it covered about 85-96% "
                "at nominal 95% after correct detection); use band_method='bootstrap'."
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
                "step_pvalue": tp.step_pvalue,
                "step_share": tp.step_share,
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
    *,
    parity_mode: bool = False,
) -> float:
    """Two-sided Student-t p-value testing H0: slope1 == slope2 at knot ``ind``.

    In default mode (``parity_mode=False``), computes the exact linear-propagated
    standard error of ``a1 - a2`` under the sequential 2-step OLS estimator
    (accounting for the negative covariance between ``a1`` and ``a2`` induced by
    the shared hinge anchor ``h = b1 + a1 * t[ind]``) with ``df = n - 3``.
    In ``parity_mode=True``, reproduces Ghaderpour et al. (2024)'s approximate
    independent-segment screening formula with ``df = n - 2``.
    """
    n = int(t.size)
    if ind <= 1 or n - ind <= 1 or n <= 3:
        return float("nan")

    t1 = t[: ind + 1]
    y1 = y[: ind + 1]
    g1 = y1 - (intercept1 + slope1 * t1)

    d2 = t[ind:] - t[ind]
    y2 = y[ind:] - (intercept1 + slope1 * t[ind])
    g2 = y2 - slope2 * d2

    if parity_mode:
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

        if s1_den <= 0.0 or s2_den <= 0.0:
            return float("nan")

        se1 = float(np.sqrt(max(0.0, s1_num))) / s1_den
        se2 = float(np.sqrt(max(0.0, s2_num))) / s2_den
        se_comb = float(np.hypot(se1, se2))
        if se_comb <= 1e-24 * max(1.0, abs(slope1), abs(slope2)):
            return 0.0 if abs(slope1 - slope2) > 0.0 else 1.0
        t_stat = abs(slope1 - slope2) / se_comb
        return float(2.0 * student_t.sf(t_stat, df=n - 2))

    # Exact linear-propagated variance of a1 - a2 = v^T y (accounts for Cov(a1, a2) < 0)
    w_arr = np.ones(n, dtype=np.float64) if w is None else w
    w1 = w_arr[: ind + 1]
    w2 = w_arr[ind:]
    w1_sum = float(np.sum(w1))
    t1_bar = float(np.sum(w1 * t1) / w1_sum)
    stt1 = float(np.sum(w1 * (t1 - t1_bar) ** 2))
    sd2 = float(np.sum(w2 * d2 * d2))
    md2 = float(np.sum(w2 * d2))
    if stt1 <= 0.0 or sd2 <= 0.0:
        return float("nan")

    c1 = w1 * (t1 - t1_bar) / stt1
    h_w = (w1 / w1_sum) + (t[ind] - t1_bar) * c1
    v_left = c1 + (md2 / sd2) * h_w
    v_right = -(w2[1:] * d2[1:]) / sd2

    # Sum v_i^2 / w_i under Var(y_i) = sigma^2 / w_i
    var_factor = float(np.sum((v_left * v_left) / w1) + np.sum((v_right * v_right) / w2[1:]))
    # Combined residual sum of squares (avoiding double-counting hinge index `ind`)
    rss_tot = float(np.sum(w1 * g1 * g1) + np.sum(w2[1:] * g2[1:] * g2[1:]))
    dof = n - 3
    sigma2_hat = max(0.0, rss_tot) / dof
    se_exact = float(np.sqrt(sigma2_hat * var_factor))
    slope_scale = max(abs(slope1), abs(slope2), float(np.finfo(np.float64).tiny))
    if se_exact <= 1e-14 * slope_scale:
        return 0.0 if abs(slope1 - slope2) > 1e-14 * slope_scale else 1.0
    t_stat = abs(slope1 - slope2) / se_exact
    return float(2.0 * student_t.sf(t_stat, df=dof))


def _fit_two_piece_sequential_scan(
    t: FloatArray,
    y: FloatArray,
    margin: int,
    w: FloatArray | None = None,
) -> tuple[int, float, float, float, float]:
    """Scan ``ind in [margin, n - margin)`` via O(n) vectorized sequential 2-step OLS/WLS.

    Step 1 fits unconstrained linear piece on ``0..ind``; Step 2 anchors the
    second linear piece at ``(t[ind], b1 + a1 * t[ind])`` and estimates ``a2``
    over ``ind..n-1``. Normalizes ``t`` to ``[0, 1]`` and removes the global
    linear trend from ``y`` prior to prefix sums (exact algebraic invariance) to
    eliminate catastrophic cancellation on steep trends, huge baselines, or tiny
    time spans while preserving O(n) speed and exact forward/backward asymmetry.
    """
    n = int(t.size)
    lo = max(2, margin)
    hi = min(n - 2, n - margin)
    if lo >= hi:
        raise InsufficientDataError(
            f"Window size n={n} too small for margin={margin} (need at least 2*margin + 1 >= 5)."
        )

    span = float(t[-1] - t[0])
    if span <= 0.0:
        raise InsufficientDataError("Collinear timestamps prevented STPD window estimation.")
    u = (t - t[0]) / span

    # Remove global linear trend b_g + a_g * u (leaves all piecewise-linear residuals invariant!)
    if w is None:
        u_mean = float(np.mean(u))
        y_mean = float(np.mean(y))
        du = u - u_mean
        suu_g = float(np.dot(du, du))
        a_g = float(np.dot(du, y - y_mean) / suu_g) if suu_g > 0.0 else 0.0
        b_g = y_mean - a_g * u_mean
    else:
        w_sum_g = float(np.sum(w))
        u_mean = float(np.sum(w * u) / w_sum_g)
        y_mean = float(np.sum(w * y) / w_sum_g)
        du = u - u_mean
        suu_g = float(np.sum(w * du * du))
        a_g = float(np.sum(w * du * (y - y_mean)) / suu_g) if suu_g > 0.0 else 0.0
        b_g = y_mean - a_g * u_mean

    yc = y - (b_g + a_g * u)
    idx = np.arange(lo, hi, dtype=np.intp)

    if w is None:
        s_w = np.arange(1, n + 1, dtype=np.float64)
        s_t = np.cumsum(u)
        s_y = np.cumsum(yc)
        s_tt = np.cumsum(u * u)
        s_ty = np.cumsum(u * yc)
        s_yy = np.cumsum(yc * yc)
        w_at_idx = np.ones(idx.size, dtype=np.float64)
    else:
        s_w = np.cumsum(w)
        s_t = np.cumsum(w * u)
        s_y = np.cumsum(w * yc)
        s_tt = np.cumsum(w * u * u)
        s_ty = np.cumsum(w * u * yc)
        s_yy = np.cumsum(w * yc * yc)
        w_at_idx = w[idx]

    # Segment 1: 0..ind (inclusive) on dimensionless u in [0, 1]
    w1 = s_w[idx]
    t1_sum = s_t[idx]
    y1_sum = s_y[idx]
    t1_m = t1_sum / w1
    y1_m = y1_sum / w1
    stt = s_tt[idx] - w1 * t1_m * t1_m
    sty = s_ty[idx] - w1 * t1_m * y1_m
    syy1 = s_yy[idx] - w1 * y1_m * y1_m

    valid = stt > 1e-24 * w1
    a1 = np.zeros_like(stt)
    a1[valid] = sty[valid] / stt[valid]
    b1_c = y1_m - a1 * t1_m
    rss1 = np.maximum(0.0, syy1 - a1 * sty)

    # Segment 2: ind..n-1 (inclusive), anchored at (tau, h_c)
    tau = u[idx]
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

    valid &= dd2 > 1e-24 * w2
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
    best_a1 = float((a1[best_pos] + a_g) / span)
    best_b1 = float(b1_c[best_pos] + b_g - best_a1 * t[0])
    best_a2 = float((a2[best_pos] + a_g) / span)
    best_norm = float(np.sqrt(total_rss[best_pos]))
    return best_ind, best_b1, best_a1, best_a2, best_norm



def _seq_scan_batch(
    t: FloatArray,
    Y: FloatArray,
    margin: int,
    w: FloatArray | None,
) -> tuple[IntArray, FloatArray, FloatArray]:
    """Batched sequential two-step scan over the columns of ``Y`` (n x B).

    Same estimator as :func:`_fit_two_piece_sequential_scan`, vectorized over many
    response vectors. Returns ``(best_ind, slope_diff, total_rss)`` per column, where
    ``slope_diff = a1 - a2`` in units of ``y`` per unit of ``t``.
    """
    n = int(t.size)
    lo = max(2, margin)
    hi = min(n - 2, n - margin)
    span = float(t[-1] - t[0])
    u = (t - t[0]) / span
    W = np.ones(n, dtype=np.float64) if w is None else np.asarray(w, dtype=np.float64)
    sw = float(np.sum(W))
    um = float(W @ u) / sw
    ym = (W @ Y) / sw
    du = u - um
    suu = float(np.sum(W * du * du))
    ag = ((W * du) @ (Y - ym[None, :])) / suu
    bg = ym - ag * um
    Yc = Y - (bg[None, :] + ag[None, :] * u[:, None])
    idx = np.arange(lo, hi)
    s_w = np.cumsum(W)
    s_t = np.cumsum(W * u)
    s_tt = np.cumsum(W * u * u)
    s_y = np.cumsum(W[:, None] * Yc, axis=0)
    s_ty = np.cumsum((W * u)[:, None] * Yc, axis=0)
    s_yy = np.cumsum(W[:, None] * Yc * Yc, axis=0)
    w1 = s_w[idx]
    t1m = s_t[idx] / w1
    stt = s_tt[idx] - w1 * t1m * t1m
    y1m = s_y[idx] / w1[:, None]
    sty = s_ty[idx] - (w1 * t1m)[:, None] * y1m
    syy1 = s_yy[idx] - w1[:, None] * y1m * y1m
    ok1 = stt > 1e-24 * w1
    stt_safe = np.where(ok1, stt, 1.0)
    a1 = sty / stt_safe[:, None]
    b1c = y1m - a1 * t1m[:, None]
    rss1 = np.maximum(0.0, syy1 - a1 * sty)
    tau = u[idx]
    hc = b1c + a1 * tau[:, None]
    prev = idx - 1
    w2 = s_w[-1] - s_w[prev]
    t2 = s_t[-1] - s_t[prev]
    tt2 = s_tt[-1] - s_tt[prev]
    y2 = s_y[-1][None, :] - s_y[prev]
    ty2 = s_ty[-1][None, :] - s_ty[prev]
    yy2 = s_yy[-1][None, :] - s_yy[prev]
    dd2 = tt2 - 2.0 * tau * t2 + w2 * tau * tau
    ok2 = ok1 & (dd2 > 1e-24 * w2)
    dd2_safe = np.where(ok2, dd2, 1.0)
    dy2 = ty2 - tau[:, None] * y2 - hc * t2[:, None] + (w2 * tau)[:, None] * hc
    yy2c = yy2 - 2.0 * hc * y2 + w2[:, None] * hc * hc
    a2 = dy2 / dd2_safe[:, None]
    rss2 = np.maximum(0.0, yy2c - a2 * dy2)
    r_h = W[idx][:, None] * (Yc[idx] - hc) ** 2
    tot = np.where(ok2[:, None], np.maximum(0.0, rss1 + rss2 - r_h), np.inf)
    best = np.argmin(tot, axis=0)
    cols = np.arange(Y.shape[1])
    diff = (a1[best, cols] - a2[best, cols]) / span
    return idx[best], diff, tot[best, cols]


def _seq_var_factors(t: FloatArray, margin: int, w: FloatArray | None) -> FloatArray:
    """``Var(a1 - a2) / sigma^2`` of the sequential estimator for every scanned knot."""
    n = int(t.size)
    lo = max(2, margin)
    hi = min(n - 2, n - margin)
    W = np.ones(n, dtype=np.float64) if w is None else np.asarray(w, dtype=np.float64)
    out = np.full(n, np.nan, dtype=np.float64)
    for ind in range(lo, hi):
        t1 = t[: ind + 1]
        w1 = W[: ind + 1]
        w2 = W[ind:]
        d2 = t[ind:] - t[ind]
        w1_sum = float(np.sum(w1))
        t1_bar = float(np.sum(w1 * t1) / w1_sum)
        stt1 = float(np.sum(w1 * (t1 - t1_bar) ** 2))
        sd2 = float(np.sum(w2 * d2 * d2))
        md2 = float(np.sum(w2 * d2))
        if stt1 <= 0.0 or sd2 <= 0.0:
            continue
        c1 = w1 * (t1 - t1_bar) / stt1
        h_w = (w1 / w1_sum) + (t[ind] - t1_bar) * c1
        v_left = c1 + (md2 / sd2) * h_w
        v_right = -(w2[1:] * d2[1:]) / sd2
        out[ind] = float(np.sum(v_left * v_left / w1) + np.sum(v_right * v_right / w2[1:]))
    return out


def _stpd_window_tstats(
    t: FloatArray,
    Y: FloatArray,
    margin: int,
    w: FloatArray | None,
    vf_f: FloatArray,
    vf_b: FloatArray,
) -> FloatArray:
    """Window slope-change t statistic of STPD (direction chosen by smaller RSS)."""
    n = int(t.size)
    t_fwd = t - t[0]
    t_rev = t_fwd[-1] - t_fwd[::-1]
    w_rev = w[::-1] if w is not None else None
    ind_f, diff_f, rss_f = _seq_scan_batch(t_fwd, Y, margin, w)
    ind_b, diff_b, rss_b = _seq_scan_batch(t_rev, Y[::-1], margin, w_rev)
    use_f = rss_f <= rss_b
    diff = np.where(use_f, diff_f, diff_b)
    rss = np.where(use_f, rss_f, rss_b)
    vf = np.where(use_f, vf_f[ind_f], vf_b[ind_b])
    se = np.sqrt(np.maximum(rss, 0.0) / (n - 3) * vf)
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.where(se > 0, np.abs(diff) / se, 0.0)
    return np.asarray(np.nan_to_num(tstat, nan=0.0), dtype=np.float64)


@dataclass(frozen=True)
class _TpBootstrap:
    curves: FloatArray
    knots: IntArray
    ystar: FloatArray
    nuisance_fit: FloatArray
    nuisance_trend: FloatArray
    dev_scale: FloatArray


def _batched_window_knots(
    t: FloatArray, Y: FloatArray, margin: int, w: FloatArray | None
) -> IntArray:
    """STPD knot estimate (local index) of each column of ``Y`` over one window.

    Same estimator as :func:`_tpd_window` (non-robust): forward and backward sequential
    scans, keeping the direction with the smaller residual sum of squares.
    """
    n = int(t.size)
    t_fwd = t - t[0]
    t_rev = t_fwd[-1] - t_fwd[::-1]
    w_rev = w[::-1] if w is not None else None
    ind_f, _, rss_f = _seq_scan_batch(t_fwd, Y, margin, w)
    ind_b, _, rss_b = _seq_scan_batch(t_rev, Y[::-1], margin, w_rev)
    return np.asarray(np.where(rss_f <= rss_b, ind_f, n - 1 - ind_b), dtype=np.intp)


def _bootstrap_tp_curves(
    t: FloatArray,
    y: FloatArray,
    trend_fit: FloatArray,
    tp_windows: list[tuple[int, int, int, int]],
    refit: Any,
    *,
    w: FloatArray | None = None,
    robust: RobustMethod = "none",
    huber_k: float = 1.345,
    n_boot: int = 499,
    seed: tuple[int, ...] = (20261126,),
    other_fit: FloatArray | None = None,
    nuisance_cols: FloatArray | None = None,
    nuisance_trend: np.ndarray | None = None,
    adjust_fn: Any = None,
    extra_df: float = 0.0,
    extra_leverage: FloatArray | None = None,
    analytic_sd: FloatArray | None = None,
    allow_empty_tps: bool = False,
    l1_refit: bool = False,
    refit_with_nuisance: Any = None,
) -> _TpBootstrap | None:
    """Wild residual bootstrap of the detect-then-fit STPD trend (audit CRT-04).

    Data are simulated from the fitted structure, ``y* = f + g + e v``: ``f`` is the
    fitted connected trend, ``g`` any other fitted structure (``other_fit``), ``e`` the
    residuals inflated by ``sqrt(n/(n - p))`` and ``v`` Rademacher signs (wild bootstrap,
    which keeps heteroscedasticity and the weights). In each replicate:

    - ``nuisance_cols`` (seasonal harmonics, jump steps) are re-estimated by weighted
      least squares jointly with the trend at the detected knots and removed;
    - every turning point is re-located with STPD's own knot estimator on the
      replicate's data over the window that detected it (forward/backward sequential
      scans; the robust variant for ``robust != 'none'``);
    - the trend is refitted with ``refit(y_adjusted, knots)`` (or
      ``refit_with_nuisance(ystar, nuis_fit, nuis_trend, knots, b)`` when provided).
      ``nuisance_trend`` flags nuisance columns whose fitted contribution belongs to
      the reported trend (jump steps).

    ``adjust_fn(ystar) -> (nuisance_fit, nuisance_trend_fit)`` replaces the
    least-squares nuisance step with a caller-supplied replica of the estimator that
    removed the nuisance structure in the original analysis (used by
    ``analyze_series`` to re-run its windowed seasonal and jump-step estimators).

    Returns the replicate curves ``(n, B)`` (reported-trend scale), the re-estimated
    knots ``(B, K)``, the replicate series ``y*`` and their fitted nuisance part
    ``(n, B)``, or ``None`` if no turning points are given and ``allow_empty_tps`` is False.
    """
    n = int(t.size)
    k_n = len(tp_windows)
    if k_n == 0 and not allow_empty_tps:
        return None
    g = other_fit if other_fit is not None else np.zeros(n)
    sw = np.sqrt(w) if w is not None else np.ones(n)
    ncols = nuisance_cols if nuisance_cols is not None else np.empty((n, 0))
    # Leverage-adjusted (HC3-type) wild-bootstrap residuals e_i / sqrt(1 - h_ii): the
    # trend's hat diagonal (connected hinge design at the detected knots) plus any
    # nuisance-estimator leverage supplied by the caller. Falls back to a global
    # sqrt(n / (n - p)) inflation if the leverages are degenerate.
    span_l = float(t[-1] - t[0]) if t[-1] > t[0] else 1.0
    u_l = (t - t[0]) / span_l
    x_tr = np.column_stack(
        [np.ones(n), u_l, *[np.maximum(0.0, u_l - u_l[gi]) for gi, _, _, _ in tp_windows]]
    )
    q_tr, _ = np.linalg.qr(x_tr * sw[:, None], mode="reduced")
    lev = np.sum(q_tr * q_tr, axis=1)
    if extra_leverage is not None:
        lev = lev + np.asarray(extra_leverage, dtype=np.float64)
    raw = y - trend_fit - g
    p_par = 2 + k_n + ncols.shape[1] + float(extra_df)
    if np.all(np.isfinite(lev)) and float(np.max(lev)) < 0.99:
        resid = raw / np.sqrt(1.0 - lev)
    else:
        resid = raw * np.sqrt(n / max(1, n - p_par))
    rng = np.random.default_rng(seed)
    v = rng.choice(np.array([-1.0, 1.0]), size=(n, int(n_boot)))
    if l1_refit:
        # Exact L1 (LP) hinge fits interpolate p = k + 2 observations with identically zero
        # residuals, pinning those points in every wild-bootstrap draw (CRT-09). Unpin exact
        # interpolation points by drawing replacement magnitudes from the non-zero residuals,
        # and inflate for the p exact-zero degrees of freedom.
        r_abs = np.abs(raw)
        max_r = float(np.max(r_abs)) if n > 0 else 0.0
        nz = r_abs > 1e-11 * max(max_r, 1e-300)
        n_nz = int(np.sum(nz))
        if 0 < n_nz < n:
            scale_l1 = float(np.sqrt(n / max(1.0, float(n_nz) - float(p_par))))
            r_pool = np.abs(raw[nz]) * scale_l1
            r_mat = np.broadcast_to((np.abs(raw) * scale_l1)[:, None], (n, int(n_boot))).copy()
            zero_idx = np.flatnonzero(~nz)
            r_mat[zero_idx, :] = rng.choice(r_pool, size=(zero_idx.size, int(n_boot)), replace=True)
            ystar = (trend_fit + g)[:, None] + r_mat * v
        else:
            ystar = (trend_fit + g)[:, None] + resid[:, None] * v
    else:
        ystar = (trend_fit + g)[:, None] + resid[:, None] * v

    nuis_trend = np.zeros((n, int(n_boot)))
    nuis_fit = np.zeros((n, int(n_boot)))
    if adjust_fn is not None:
        nuis_fit, nuis_trend = adjust_fn(ystar)
        y_adj = ystar - nuis_fit
    elif ncols.shape[1] > 0:
        span = float(t[-1] - t[0]) if t[-1] > t[0] else 1.0
        u = (t - t[0]) / span
        hinges = [np.maximum(0.0, u - u[gi]) for gi, _, _, _ in tp_windows]
        x_full = np.column_stack([np.ones(n), u, *hinges, ncols])
        coef, *_ = np.linalg.lstsq(x_full * sw[:, None], ystar * sw[:, None], rcond=None)
        c_n = coef[-ncols.shape[1] :]
        nuis_fit = ncols @ c_n
        tmask = nuisance_trend if nuisance_trend is not None else np.zeros(ncols.shape[1], dtype=bool)
        if np.any(tmask):
            nuis_trend = ncols[:, tmask] @ c_n[tmask]
        y_adj = ystar - nuis_fit
    else:
        y_adj = ystar

    knots = np.empty((int(n_boot), k_n), dtype=np.intp)
    for j, (_gi, w0, w1, m_eff) in enumerate(tp_windows):
        t_w = t[w0:w1]
        w_w = w[w0:w1] if w is not None else None
        if robust == "none":
            knots[:, j] = w0 + _batched_window_knots(t_w, y_adj[w0:w1], m_eff, w_w)
        else:
            for b in range(int(n_boot)):
                st = _tpd_window(t_w, y_adj[w0:w1, b], m_eff, w_w, robust=robust, huber_k=huber_k)
                knots[b, j] = w0 + int(st.local_tp)
    curves = np.empty((n, int(n_boot)), dtype=np.float64)
    for b in range(int(n_boot)):
        kb = np.unique(np.clip(knots[b], 1, n - 2)) if k_n > 0 else np.empty(0, dtype=np.intp)
        if refit_with_nuisance is not None:
            curves[:, b] = refit_with_nuisance(ystar[:, b], nuis_fit[:, b], nuis_trend[:, b], kb, b)
        else:
            curves[:, b] = refit(y_adj[:, b], kb) + nuis_trend[:, b]

    # Variance calibration: the wild bootstrap under-disperses the sequential trend
    # estimator even with the structure fixed (by about 6% in width in simulation). The
    # same replicates refitted with the original knots and nuisance structure give the
    # bootstrap's fixed-structure dispersion; where it falls below the exact analytical
    # conditional standard error, deviations are scaled up by the ratio (clipped to
    # [1, 2]), so the location and nuisance uncertainty is added on top of the correct
    # fixed-structure variance.
    dev_scale = np.ones(n)
    if analytic_sd is not None and np.all(np.isfinite(analytic_sd)):
        k0 = (
            np.unique(np.clip(np.array([gi for gi, _, _, _ in tp_windows], dtype=np.intp), 1, n - 2))
            if k_n > 0
            else np.empty(0, dtype=np.intp)
        )
        y_fixed = ystar - g[:, None]
        fixed = np.column_stack([refit(y_fixed[:, b], k0) for b in range(int(n_boot))])
        sd_fixed = np.std(fixed - trend_fit[:, None], axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(sd_fixed > 0, np.asarray(analytic_sd) / sd_fixed, 1.0)
        dev_scale = np.clip(np.nan_to_num(ratio, nan=1.0), 1.0, 2.0)
    return _TpBootstrap(
        curves=curves, knots=knots, ystar=ystar, nuisance_fit=nuis_fit, nuisance_trend=nuis_trend,
        dev_scale=dev_scale,
    )


def _stpd_mc_pvalue(
    t: FloatArray,
    y: FloatArray,
    margin: int,
    w: FloatArray | None,
    n_mc: int,
    seed: tuple[int, ...],
    serial: SerialSpec = NO_SERIAL,
) -> float:
    """Single-series wrapper of :func:`_stpd_mc_pvalues`."""
    return float(_stpd_mc_pvalues(t, y[:, None], margin, w, n_mc, seed, serial=serial)[0])


def _stpd_mc_pvalues(
    t: FloatArray,
    Y: FloatArray,
    margin: int,
    w: FloatArray | None,
    n_mc: int,
    seed: tuple[int, ...],
    skip_above: float | None = None,
    serial: SerialSpec = NO_SERIAL,
) -> FloatArray:
    """Exact Monte Carlo p-value of the STPD window test against the max-over-knots null.

    Under a single straight line with Gaussian errors (variances ``sigma^2 / w``), the
    sequential estimator's slope-change t statistic, including the data-driven choice
    of knot and fit direction, depends only on the errors, so simulating standard
    errors gives its exact null distribution. Below Monte Carlo resolution the
    Bonferroni bound over knots and directions is reported (capped at ``1/(n_mc+1)``).
    ``Y`` holds one series per column (e.g. multiple imputations); the null draws are
    shared across columns. With ``skip_above``, if every column's raw (fixed-knot)
    p-value is at least ``skip_above``, ones are returned without simulating (each
    calibrated p-value would exceed its raw value).

    With an active ``serial`` model the errors are simulated with AR(1)/OU correlation
    (parametric bootstrap under the fitted correlation, AUD-02), the Bonferroni tail bound
    uses ``t / kappa`` and the raw-p skip is disabled.
    """
    n = int(t.size)
    t_fwd = t - t[0]
    t_rev = t_fwd[-1] - t_fwd[::-1]
    w_rev = w[::-1] if w is not None else None
    vf_f = _seq_var_factors(t_fwd, margin, w)
    vf_b = _seq_var_factors(t_rev, margin, w_rev)
    t_obs = np.asarray(_stpd_window_tstats(t, Y, margin, w, vf_f, vf_b), dtype=np.float64)
    if skip_above is not None and bool(
        np.all(2.0 * student_t.sf(t_obs, df=n - 3) >= skip_above)
    ):
        # A calibrated p-value can only exceed its raw value (also under positive
        # serial correlation, which widens the null).
        return np.ones(t_obs.size, dtype=np.float64)
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(w) if w is not None else np.ones(n)
    exceed = np.zeros(t_obs.size, dtype=np.int64)
    done = 0
    while done < n_mc:
        m = min(256, n_mc - done)
        E = correlated_normals(rng, t, m, serial) * scale[:, None]
        null_t = _stpd_window_tstats(t, E, margin, w, vf_f, vf_b)
        exceed += np.sum(null_t[None, :] >= t_obs[:, None], axis=1)
        done += m
    p_mc = (1.0 + exceed) / (n_mc + 1.0)
    n_knots = 2 * max(1, min(n - 2, n - margin) - max(2, margin))
    p_bonf = np.minimum(
        1.0, n_knots * 2.0 * student_t.sf(t_obs / np.sqrt(serial.kappa2), df=n - 3)
    )
    return np.asarray(np.where(exceed == 0, np.minimum(p_mc, p_bonf), p_mc), dtype=np.float64)


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
    step_pval: float = float("nan")
    step_share: float = float("nan")


def _huber_line_2param(
    u: FloatArray,
    y: FloatArray,
    w_base: FloatArray | None,
    k: float,
    max_iter: int = 12,
    tol: float = 1e-6,
) -> tuple[float, float]:
    """Closed-form weighted 2-parameter Huber IRLS fit ``y ~ b + a * u``."""
    w0 = np.ones_like(u) if w_base is None else w_base
    sw0 = np.sqrt(w0)
    y_ref = float(np.sqrt(np.mean((y * sw0) ** 2)))
    floor = 1e-14 * max(y_ref, 1e-300)

    def _wls2(wt: FloatArray) -> tuple[float, float]:
        s0 = float(np.sum(wt))
        if s0 <= 0.0:
            return 0.0, 0.0
        su = float(np.dot(wt, u))
        sy = float(np.dot(wt, y))
        u_bar = su / s0
        y_bar = sy / s0
        du = u - u_bar
        s_uu = float(np.dot(wt * du, du))
        if s_uu <= 1e-30 * s0:
            return y_bar, 0.0
        a = float(np.dot(wt * du, y - y_bar)) / s_uu
        return y_bar - a * u_bar, a

    b, a = _wls2(w0)
    rw = (y - (b + a * u)) * sw0
    scale = float(1.482602218505602 * np.median(np.abs(rw)))
    if scale <= floor:
        scale = float(1.2533141373155001 * np.mean(np.abs(rw)))
        if scale <= floor:
            return b, a

    for it in range(1, max_iter + 1):
        au = np.abs(rw) / scale
        wh = np.where(au <= k, 1.0, k / np.maximum(au, k))
        nb, na = _wls2(w0 * wh)
        rw = (y - (nb + na * u)) * sw0
        if it <= 3:
            ns = float(1.482602218505602 * np.median(np.abs(rw)))
            if ns > floor:
                scale = ns
        diff = max(abs(nb - b), abs(na - a))
        ref = max(abs(nb), abs(na), 1e-300)
        b, a = nb, na
        if diff <= tol * ref:
            break
    return b, a


def _huber_slope_1param(
    d: FloatArray,
    y: FloatArray,
    w_base: FloatArray | None,
    k: float,
    max_iter: int = 12,
    tol: float = 1e-6,
) -> float:
    """Closed-form weighted 1-parameter zero-intercept Huber IRLS fit ``y ~ a * d``."""
    w0 = np.ones_like(d) if w_base is None else w_base
    sw0 = np.sqrt(w0)
    y_ref = float(np.sqrt(np.mean((y * sw0) ** 2)))
    floor = 1e-14 * max(y_ref, 1e-300)
    d2 = d * d

    den0 = float(np.dot(w0, d2))
    if den0 <= 0.0:
        return 0.0
    a = float(np.dot(w0 * d, y)) / den0
    rw = (y - a * d) * sw0
    scale = float(1.482602218505602 * np.median(np.abs(rw)))
    if scale <= floor:
        scale = float(1.2533141373155001 * np.mean(np.abs(rw)))
        if scale <= floor:
            return a

    for it in range(1, max_iter + 1):
        au = np.abs(rw) / scale
        wh = np.where(au <= k, 1.0, k / np.maximum(au, k))
        wt = w0 * wh
        den = float(np.dot(wt, d2))
        if den <= 0.0:
            break
        na = float(np.dot(wt * d, y)) / den
        rw = (y - na * d) * sw0
        if it <= 3:
            ns = float(1.482602218505602 * np.median(np.abs(rw)))
            if ns > floor:
                scale = ns
        if abs(na - a) <= tol * max(abs(na), 1e-300):
            a = na
            break
        a = na
    return a


def _fit_two_piece_sequential_huber(
    t: FloatArray,
    y: FloatArray,
    margin: int,
    w: FloatArray | None,
    huber_k: float,
    method: RobustMethod = "huber",
) -> tuple[int, float, float, float, float, FloatArray]:
    """Huber/MM-IRLS asymmetric sequential two-piece scan preserving STPD NDRI geometry."""
    n = int(t.size)
    span = float(t[-1] - t[0])
    if span <= 0.0:
        raise InsufficientDataError("Window time span must be strictly positive.")
    u = (t - t[0]) / span
    y_med = float(np.median(y))
    y_c = y - y_med
    sqrt_w = np.sqrt(w) if w is not None else None

    cands: list[tuple[int, float, float, float, FloatArray]] = []
    min_mad = np.inf

    for ind in range(margin, n - margin):
        u1 = u[: ind + 1]
        w1 = w[: ind + 1] if w is not None else None
        b1_c, a1_u = _huber_line_2param(u1, y_c[: ind + 1], w1, huber_k)
        y_anchor = b1_c + a1_u * float(u[ind])

        d2_u = u[ind:] - u[ind]
        w2 = w[ind:] if w is not None else None
        a2_u = _huber_slope_1param(d2_u, y_c[ind:] - y_anchor, w2, huber_k)

        y_fit_c = np.empty(n, dtype=np.float64)
        y_fit_c[: ind + 1] = b1_c + a1_u * u1
        y_fit_c[ind:] = y_anchor + a2_u * d2_u
        r_j = y_c - y_fit_c
        rw_j = r_j * sqrt_w if sqrt_w is not None else r_j
        mad_j = float(1.482602218505602 * np.median(np.abs(rw_j)))
        if mad_j < min_mad:
            min_mad = mad_j
        cands.append((ind, b1_c, a1_u, a2_u, rw_j))

    y_ref = float(np.sqrt(np.mean((y_c * sqrt_w if sqrt_w is not None else y_c) ** 2)))
    s_win = min_mad if min_mad > 1e-14 * max(y_ref, 1e-300) else 0.0

    best_loss = np.inf
    best_ind = margin
    best_b1 = 0.0
    best_a1 = 0.0
    best_a2 = 0.0
    best_rw = np.zeros(n, dtype=np.float64)

    for ind, b1_c, a1_u, a2_u, rw_j in cands:
        loss_j = (
            float(np.sum(huber_rho(rw_j / s_win, huber_k)))
            if s_win > 0.0
            else float(np.dot(rw_j, rw_j))
        )
        if loss_j < best_loss:
            best_loss = loss_j
            best_ind = ind
            best_b1 = y_med + b1_c
            best_a1 = a1_u / span
            best_a2 = a2_u / span
            best_rw = rw_j

    if s_win > 0.0:
        u_signed = best_rw / s_win
        u_abs = np.abs(u_signed)
        is_ext = u_abs >= 4.685061
        ext_frac = float(np.mean(is_ext))
        max_run = 0
        cur_run = 0
        for flag in is_ext:
            if flag:
                cur_run += 1
                if cur_run > max_run:
                    max_run = cur_run
            else:
                cur_run = 0
        if method == "mm" and ext_frac <= 0.20 and max_run < 3:
            # Isolated spikes only: zero extreme spikes without masking contiguous step cliffs
            w_huber = np.maximum(tukey_weight(u_signed), 1e-12)
        else:
            w_huber = np.where(u_abs <= huber_k, 1.0, huber_k / np.maximum(u_abs, huber_k))
    else:
        w_huber = np.ones(n, dtype=np.float64)

    return best_ind, best_b1, best_a1, best_a2, best_loss, w_huber


def _step_vs_hinge_pvalue(
    t: FloatArray,
    y: FloatArray,
    w: FloatArray | None,
    tp: int,
    margin: int,
    *,
    n_grid: int = 17,
    kappa2: float = 1.0,
) -> tuple[float, float]:
    """p-value that a window's break is a vertical step rather than a continuous bend.

    Compares the best discontinuous two-line model (separate intercepts and slopes, split
    between samples ``j - 1`` and ``j``) with the best *continuous* hinge whose knot ``c`` may
    lie anywhere between samples, over the scanned splits. For a split ``j`` the constrained least-squares identity gives
    ``RSS_hinge(c) = RSS_disc(j) + g(c)^2 / v(c)`` for ``c`` in ``[t[j-1], t[j]]``, where
    ``g(c)`` is the estimated vertical gap between the two lines at ``c`` and ``v(c)`` its
    variance factor, so the continuous knot is optimized exactly within each gap (a precise
    bend whose knot falls between samples is *not* mistaken for a step). The 1-df statistic
    ``F = (min RSS_hinge - min RSS_disc) / (min RSS_disc / (n - 4))`` is Bonferroni-corrected
    over the splits scanned (conservative, so genuine bends keep their power). The
    discontinuous split ranges over ``[margin, n - margin]``; ``tp`` is kept for API
    symmetry with the slope-change test.

    Returns ``(p_value, step_share)`` where ``step_share = (min RSS_hinge - min RSS_disc) /
    (RSS_line - min RSS_disc)`` is the fraction of the break's explained variance that only
    a vertical step can capture (an effect size: about 0.6-0.9 for genuine steps, below
    about 0.1 for genuine bends). At very high signal-to-noise a tiny misspecification makes
    the p-value significant for a genuine bend, so callers should require both.
    """
    nan2 = (float("nan"), float("nan"))
    n = int(t.size)
    if n < 8:
        return nan2
    m_lo = max(3, int(margin))
    lo_all, hi_all = m_lo, n - m_lo  # split j: left = [0, j), right = [j, n)
    if hi_all <= lo_all:
        return nan2
    # The discontinuous alternative may split anywhere in the scan range: a window that
    # sees a step off-centre often places its best hinge knot many samples away from it.
    lo, hi = lo_all, hi_all
    span = float(t[-1] - t[0]) if t[-1] > t[0] else 1.0
    u = (t - t[0]) / span
    sd = float(np.std(y))
    if not np.isfinite(sd) or sd <= 0.0:
        return 1.0, 0.0
    yc = (y - float(np.mean(y))) / sd
    ww = np.ones(n, dtype=np.float64) if w is None else np.asarray(w, dtype=np.float64)

    def _cum(v: FloatArray) -> FloatArray:
        return np.concatenate(([0.0], np.cumsum(v)))

    c0, c1, c2 = _cum(ww), _cum(ww * u), _cum(ww * u * u)
    cy, cuy, cyy = _cum(ww * yc), _cum(ww * u * yc), _cum(ww * yc * yc)
    def _side(a: IntArray, b: IntArray) -> tuple[FloatArray, ...]:
        s0, s1, s2 = c0[b] - c0[a], c1[b] - c1[a], c2[b] - c2[a]
        sy, suy, syy = cy[b] - cy[a], cuy[b] - cuy[a], cyy[b] - cyy[a]
        det = s0 * s2 - s1 * s1
        det = np.where(det > 0.0, det, np.nan)
        i00, i01, i11 = s2 / det, -s1 / det, s0 / det  # inverse of [[s0, s1], [s1, s2]]
        a0 = i00 * sy + i01 * suy
        a1 = i01 * sy + i11 * suy
        rss = np.maximum(syy - (a0 * sy + a1 * suy), 0.0)
        return a0, a1, i00, i01, i11, rss

    # Continuous hinge: knot anywhere between u[1] and u[n-2] (every gap with >= 2 points per
    # side), so the null model is never handicapped by the scan margin.
    jh = np.arange(2, n - 1)
    aL, bL, l00, l01, l11, rL = _side(np.zeros_like(jh), jh)
    aR, bR, r00, r01, r11, rR = _side(jh, np.full_like(jh, n))
    rss_disc_all = rL + rR
    frac = np.linspace(0.0, 1.0, n_grid)
    cgrid = u[jh - 1][:, None] + (u[jh] - u[jh - 1])[:, None] * frac[None, :]
    g = (aL - aR)[:, None] + (bL - bR)[:, None] * cgrid
    v = (l00 + r00)[:, None] + 2.0 * (l01 + r01)[:, None] * cgrid + (l11 + r11)[:, None] * cgrid**2
    pen = np.where(v > 0.0, g * g / v, np.inf)
    # A sign change of the gap inside the interval means the lines meet there exactly
    crosses = (g[:, 0] * g[:, -1]) <= 0.0
    pen_min = np.where(crosses, 0.0, np.min(pen, axis=1))
    rss_hinge = rss_disc_all + pen_min
    # Discontinuous alternative: splits within the scan range [margin, n - margin]
    in_loc = (jh >= lo) & (jh < hi)
    ok_h = np.isfinite(rss_hinge)
    ok_d = in_loc & np.isfinite(rss_disc_all)
    if not (np.any(ok_h) and np.any(ok_d)):
        return nan2
    sst = float(np.sum(ww * yc * yc))
    rd = float(np.min(rss_disc_all[ok_d]))
    rh = float(np.min(rss_hinge[ok_h]))
    num = max(rh - rd, 0.0)
    rl = float(_side(np.array([0]), np.array([n]))[5][0])
    share = float(num / (rl - rd)) if (rl - rd) > 1e-300 else 0.0
    if num <= 1e-12 * max(sst, 1e-300):
        return 1.0, share
    dof = n - 4
    den = max(rd, 1e-28 * max(sst, 1e-300)) / dof
    # Under serially correlated errors the F statistic is deflated by kappa^2 (AUD-02)
    f_stat = num / den / max(kappa2, 1.0)
    from scipy.stats import f as f_dist

    p_raw = float(f_dist.sf(f_stat, 1, dof))
    return float(min(1.0, p_raw * int(np.sum(ok_d)))), share


def _tpd_window(
    t: FloatArray,
    y: FloatArray,
    margin: int,
    w: FloatArray | None = None,
    *,
    parity_mode: bool = False,
    robust: RobustMethod = "none",
    huber_k: float = 1.345,
    kappa2: float = 1.0,
) -> _WindowTpdStats:
    """Run forward and backward sequential turning-point detection on one window.

    ``kappa2`` deflates the step-vs-bend F statistic under serial correlation (AUD-02).
    """
    n = int(t.size)
    t_fwd = t - t[0]
    eff_robust: RobustMethod = "none" if parity_mode else robust

    if eff_robust != "none":
        ind_f, b1_f, a1_f, a2_f, norm_f, wh_f = _fit_two_piece_sequential_huber(
            t_fwd, y, margin, w, huber_k, method=eff_robust
        )
        t_rev = t_fwd[-1] - t_fwd[::-1]
        y_rev = y[::-1]
        w_rev = w[::-1] if w is not None else None
        ind_b_rev, b1_b, a1_b, a2_b, norm_b, wh_b_rev = _fit_two_piece_sequential_huber(
            t_rev, y_rev, margin, w_rev, huber_k, method=eff_robust
        )
        ind_b = n - 1 - ind_b_rev
        fb_diff = abs(ind_b - ind_f)
        fb_time_diff = float(abs(t_fwd[ind_b] - t_fwd[ind_f]))

        y_fit = np.empty(n, dtype=np.float64)
        if norm_f <= norm_b:
            tp = ind_f
            y_fit[: tp + 1] = b1_f + a1_f * t_fwd[: tp + 1]
            y_fit[tp:] = a2_f * (t_fwd[tp:] - t_fwd[tp]) + b1_f + a1_f * t_fwd[tp]
            slope_before = a1_f
            slope_after = a2_f
            w_h_win = wh_f
        else:
            tp_rev = ind_b_rev
            y_fit_rev = np.empty(n, dtype=np.float64)
            y_fit_rev[: tp_rev + 1] = b1_b + a1_b * t_rev[: tp_rev + 1]
            y_fit_rev[tp_rev:] = a2_b * (t_rev[tp_rev:] - t_rev[tp_rev]) + b1_b + a1_b * t_rev[tp_rev]
            y_fit = y_fit_rev[::-1]
            tp = ind_b
            slope_before = -a2_b
            slope_after = -a1_b
            w_h_win = wh_b_rev[::-1]

        span_w = float(t_fwd[-1]) if float(t_fwd[-1]) > 0.0 else 1.0
        u_w = t_fwd / span_w
        D_null = np.column_stack([np.ones(n, dtype=np.float64), u_w])
        D_hinge = np.column_stack([np.ones(n, dtype=np.float64), u_w, np.maximum(0.0, u_w - u_w[tp])])
        pw_w = prepare_weights(w, n)
        _, pval, _, _ = huber_nested_ftest(
            D_null, D_hinge, y, pw_w, method=eff_robust, k=huber_k
        )
        w_stat: FloatArray | None = w_h_win if w is None else (w * w_h_win)
    else:
        ind_f, b1_f, a1_f, a2_f, norm_f = _fit_two_piece_sequential_scan(t_fwd, y, margin, w)
        pval_f = _stpd_pvalue(t_fwd, y, ind_f, b1_f, a1_f, a2_f, w, parity_mode=parity_mode)

        t_rev = t_fwd[-1] - t_fwd[::-1]
        y_rev = y[::-1]
        w_rev = w[::-1] if w is not None else None
        ind_b_rev, b1_b, a1_b, a2_b, norm_b = _fit_two_piece_sequential_scan(
            t_rev, y_rev, margin, w_rev
        )
        ind_b = n - 1 - ind_b_rev
        pval_b = _stpd_pvalue(t_rev, y_rev, ind_b_rev, b1_b, a1_b, a2_b, w_rev, parity_mode=parity_mode)

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
        w_stat = w

    res_all = y - y_fit
    r1 = res_all[: tp + 1]
    r2 = res_all[tp:]
    n_left = tp + 1
    n_right = n - tp

    if w_stat is None:
        res_norm = float(np.linalg.norm(res_all))
        fit_norm = float(np.linalg.norm(y_fit))
        fit_cent_norm = float(np.linalg.norm(y_fit - np.mean(y_fit)))
        nr1 = float(np.linalg.norm(r1))
        nr2 = float(np.linalg.norm(r2))
        rmse1 = nr1 / np.sqrt(n_left)
        rmse2 = nr2 / np.sqrt(n_right)
    else:
        w1 = w_stat[: tp + 1]
        w2 = w_stat[tp:]
        w_sum = float(np.sum(w_stat))
        y_fit_wmean = float(np.sum(w_stat * y_fit) / w_sum)
        res_norm = float(np.sqrt(np.sum(w_stat * res_all * res_all)))
        fit_norm = float(np.sqrt(np.sum(w_stat * y_fit * y_fit)))
        fit_cent_norm = float(np.sqrt(np.sum(w_stat * (y_fit - y_fit_wmean) ** 2)))
        nr1 = float(np.sqrt(np.sum(w1 * r1 * r1)))
        nr2 = float(np.sqrt(np.sum(w2 * r2 * r2)))
        rmse1 = nr1 / np.sqrt(float(n_left))
        rmse2 = nr2 / np.sqrt(float(n_right))

    zero_tol = 1e-14 * max(fit_norm, fit_cent_norm, float(np.finfo(np.float64).tiny))
    if res_norm <= zero_tol:
        snr_raw = np.inf
        snr_centered = np.inf
    else:
        snr_raw = fit_norm / res_norm
        snr_centered = fit_cent_norm / res_norm

    ndri_l2 = abs(nr1 - nr2) / (nr1 + nr2) if (nr1 + nr2) > zero_tol else 0.0
    ndri_rmse = abs(rmse1 - rmse2) / (rmse1 + rmse2) if (rmse1 + rmse2) > zero_tol else 0.0
    step_p, step_share = _step_vs_hinge_pvalue(
        t, y, w_stat, int(tp), int(margin), kappa2=kappa2
    )

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
        step_pval=step_p,
        step_share=step_share,
    )


def _chained_forward_fit(
    t: FloatArray,
    y: FloatArray,
    tp_sorted: IntArray,
    w: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Sequential chained connected OLS/WLS across sorted turning points.

    Also returns the exact dimensionless pointwise variance factors
    ``v[i] = Var(y_hat[i]) / sigma^2`` via an O(n) disjoint-support recursion.
    """
    n = int(t.size)
    m = int(tp_sorted.size)
    y_fit = np.empty(n, dtype=np.float64)
    v_fac = np.empty(n, dtype=np.float64)

    if m == 0:
        if w is None:
            sw = float(n)
            tm = float(np.mean(t))
            ym = float(np.mean(y))
            dt = t - tm
            stt = float(np.dot(dt, dt))
            a = float(np.dot(dt, y - ym)) / stt if stt > 0.0 else 0.0
            b = ym - a * tm
        else:
            sw = float(np.sum(w))
            tm = float(np.sum(w * t)) / sw
            ym = float(np.sum(w * y)) / sw
            dt = t - tm
            stt = float(np.sum(w * dt * dt))
            a = float(np.sum(w * dt * (y - ym))) / stt if stt > 0.0 else 0.0
            b = ym - a * tm
        y_fit[:] = b + a * t
        v_fac[:] = (1.0 / sw) + (dt * dt / stt if stt > 0.0 else 0.0)
        return np.array([a], dtype=np.float64), np.array([b], dtype=np.float64), y_fit, v_fac

    bounds = np.append(tp_sorted, n - 1)
    slopes = np.empty(m + 1, dtype=np.float64)
    intercepts = np.empty(m + 1, dtype=np.float64)

    # First segment: unconstrained 2-param fit on 0..tp_0
    end0 = int(bounds[0]) + 1
    t1 = t[:end0]
    y1 = y[:end0]
    if w is None:
        sw1 = float(end0)
        tm = float(np.mean(t1))
        ym = float(np.mean(y1))
        dt = t1 - tm
        stt1 = float(np.dot(dt, dt))
        a0 = float(np.dot(dt, y1 - ym)) / stt1 if stt1 > 0.0 else 0.0
        b0 = ym - a0 * tm
    else:
        w1 = w[:end0]
        sw1 = float(np.sum(w1))
        tm = float(np.sum(w1 * t1)) / sw1
        ym = float(np.sum(w1 * y1)) / sw1
        dt = t1 - tm
        stt1 = float(np.sum(w1 * dt * dt))
        a0 = float(np.sum(w1 * dt * (y1 - ym))) / stt1 if stt1 > 0.0 else 0.0
        b0 = ym - a0 * tm
    slopes[0] = a0
    intercepts[0] = b0
    y_fit[:end0] = b0 + a0 * t1
    v_fac[:end0] = (1.0 / sw1) + (dt * dt / stt1 if stt1 > 0.0 else 0.0)

    # Subsequent anchored segments: exact disjoint-support variance recursion
    # y_hat_i = (1 - d_i * gamma_k) * y_hat_{tau_k} + d_i * (c_k^T y), where
    # supp(c_k) = {tau_k+1..tau_{k+1}} is disjoint from supp(y_hat_{tau_k}) = {0..tau_k}.
    for k in range(m):
        i_start = int(bounds[k])
        i_stop = int(bounds[k + 1]) + 1
        d_seg = t[i_start:i_stop] - t[i_start]
        y_seg = y[i_start:i_stop] - y_fit[i_start]
        if w is None:
            s_dd = float(np.dot(d_seg, d_seg))
            s_d = float(np.sum(d_seg))
            ak = float(np.dot(d_seg, y_seg)) / s_dd if s_dd > 0.0 else 0.0
        else:
            wk = w[i_start:i_stop]
            s_dd = float(np.sum(wk * d_seg * d_seg))
            s_d = float(np.sum(wk * d_seg))
            ak = float(np.sum(wk * d_seg * y_seg)) / s_dd if s_dd > 0.0 else 0.0
        slopes[k + 1] = ak
        intercepts[k + 1] = y_fit[i_start] - ak * t[i_start]
        y_fit[i_start:i_stop] = y_fit[i_start] + ak * d_seg
        v_anchor = float(v_fac[i_start])
        if s_dd > 0.0:
            gamma_k = s_d / s_dd
            v_fac[i_start:i_stop] = (1.0 - d_seg * gamma_k) ** 2 * v_anchor + (d_seg * d_seg) / s_dd
        else:
            v_fac[i_start:i_stop] = v_anchor

    return slopes, intercepts, y_fit, v_fac


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
    huber_k: float = 1.345,
    ci_level: float = 0.95,
) -> PiecewiseTrendResult:
    """Internal piecewise connected linear trend fitter (sequential, joint, L1, or Huber)."""
    n = int(t_num.size)
    tp_sorted = np.sort(np.unique(tp_valid.astype(np.intp)))
    m = int(tp_sorted.size)
    if m > 0 and (int(tp_sorted[0]) < 1 or int(tp_sorted[-1]) > n - 2):
        raise ValueError("`tp_indices` must lie strictly inside the series (1 <= tp <= n - 2).")

    t0 = t_num - t_num[0]
    cond_num = 1.0
    huber_converged = True
    l1_fallback_flag: list[bool] = []
    trend_ci_low: FloatArray | None = None
    trend_ci_high: FloatArray | None = None

    if mode == "l1":
        slopes, intercepts, y_fit, cond_num = fit_l1_piecewise(
            t_num, y, tp_sorted, w=w, fallback_flag=l1_fallback_flag
        )
        selected_dir: Literal["forward", "backward", "joint"] = "joint"
    elif mode in ("joint", "huber"):
        # Continuous piecewise linear hinge basis scaled by span s_t for numerical conditioning:
        # [1, t0/s_t, ((t0 - tau_1)/s_t)_+, ..., ((t0 - tau_m)/s_t)_+]
        s_t = float(t0[-1]) if float(t0[-1]) > 0.0 else 1.0
        u = t0 / s_t
        D = np.empty((n, m + 2), dtype=np.float64)
        D[:, 0] = 1.0
        D[:, 1] = u
        for k, tp in enumerate(tp_sorted):
            D[:, 2 + k] = np.maximum(0.0, u - u[tp])
        pw = prepare_weights(w, n) if w is not None else prepare_weights(None, n)
        if mode == "huber":
            hr = solve_huber_irls(D, y, pw, k=huber_k, compute_cov=True)
            ols = hr.ols
            huber_converged = bool(hr.converged)
        else:
            ols = solve_weighted_ols(D, y, pw, compute_cov=True)
        cond_num = float(ols.condition_number)
        y_fit = y - ols.residuals
        dof_j = n - ols.rank
        if ols.cov.size > 0 and np.all(np.isfinite(ols.cov)) and dof_j > 0:
            var_fit = np.maximum(0.0, np.sum((D @ ols.cov) * D, axis=1))
            t_crit = float(student_t.ppf(0.5 + 0.5 * ci_level, df=dof_j))
            hw = t_crit * np.sqrt(var_fit)
            trend_ci_low = y_fit - hw
            trend_ci_high = y_fit + hw
        b0 = float(ols.coeff[0])
        a0 = float(ols.coeff[1]) / s_t
        deltas = ols.coeff[2:] / s_t
        slopes = np.empty(m + 1, dtype=np.float64)
        intercepts = np.empty(m + 1, dtype=np.float64)
        slopes[0] = a0
        intercepts[0] = b0 - a0 * t_num[0]
        cum_slope = a0
        for k, tp in enumerate(tp_sorted):
            cum_slope += float(deltas[k])
            slopes[k + 1] = cum_slope
            intercepts[k + 1] = float(y_fit[tp]) - cum_slope * float(t_num[tp])
        selected_dir = "joint"
    else:
        slopes_f, intercepts_f, y_f, v_f = _chained_forward_fit(t0, y, tp_sorted, w)
        t_rev = t0[-1] - t0[::-1]
        y_rev = y[::-1]
        w_rev = w[::-1] if w is not None else None
        tp_rev = np.sort(n - 1 - tp_sorted)
        slopes_b_rev, _int_b_rev, y_b_rev, v_b_rev = _chained_forward_fit(
            t_rev, y_rev, tp_rev, w_rev
        )
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
            v_sel = v_f
            rss_sel = err_f * err_f
            selected_dir = "forward"
        else:
            slopes = slopes_b
            y_fit = y_b
            v_sel = v_b_rev[::-1]
            rss_sel = err_b * err_b
            selected_dir = "backward"

        dof_s = n - (m + 2)
        if dof_s > 0:
            sigma2_hat = max(0.0, rss_sel) / dof_s
            t_crit = float(student_t.ppf(0.5 + 0.5 * ci_level, df=dof_s))
            hw = t_crit * np.sqrt(np.maximum(0.0, sigma2_hat * v_sel))
            trend_ci_low = y_fit - hw
            trend_ci_high = y_fit + hw

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

        if w is None:
            nr1 = float(np.linalg.norm(r1))
            nr2 = float(np.linalg.norm(r2))
            nr_all = float(np.linalg.norm(r_pair))
            ny_raw = float(np.linalg.norm(y_pair))
            ny_cent = float(np.linalg.norm(y_pair - np.mean(y_pair)))
            rmse1 = nr1 / np.sqrt(i1 - i0 + 1)
            rmse2 = nr2 / np.sqrt(i2 - i1 + 1)
        else:
            w1 = w[i0 : i1 + 1]
            w2 = w[i1 : i2 + 1]
            wp = w[i0 : i2 + 1]
            nr1 = float(np.sqrt(np.sum(w1 * r1 * r1)))
            nr2 = float(np.sqrt(np.sum(w2 * r2 * r2)))
            nr_all = float(np.sqrt(np.sum(wp * r_pair * r_pair)))
            ny_raw = float(np.sqrt(np.sum(wp * y_pair * y_pair)))
            wp_sum = float(np.sum(wp))
            y_wmean = float(np.sum(wp * y_pair)) / wp_sum if wp_sum > 0.0 else 0.0
            ny_cent = float(np.sqrt(np.sum(wp * (y_pair - y_wmean) ** 2)))
            rmse1 = nr1 / np.sqrt(float(i1 - i0 + 1))
            rmse2 = nr2 / np.sqrt(float(i2 - i1 + 1))

        zero_tol = 1e-14 * max(ny_raw, ny_cent, 1e-300)
        directions[k - 1] = slopes[k] - slopes[k - 1]
        if nr_all <= zero_tol:
            snr_raw_arr[k - 1] = np.inf
            snr_cent_arr[k - 1] = np.inf
        else:
            snr_raw_arr[k - 1] = ny_raw / nr_all
            snr_cent_arr[k - 1] = ny_cent / nr_all
        ndri_l2_arr[k - 1] = abs(nr2 - nr1) / (nr1 + nr2) if (nr1 + nr2) > zero_tol else 0.0
        ndri_rmse_arr[k - 1] = (
            abs(rmse2 - rmse1) / (rmse1 + rmse2) if (rmse1 + rmse2) > zero_tol else 0.0
        )

    residuals = y - y_fit
    rn, rn_warns = compute_red_noise_diagnostics(t_num, residuals)
    if not huber_converged:
        rn_warns = (
            *rn_warns,
            WarningItem(
                code="ROBUST_NOT_CONVERGED",
                severity="caution",
                message="Huber IRLS hinge-spline fit reached its iteration limit before converging.",
                suggestion="Treat the robust trend and its CI as approximate.",
            ),
        )
    if l1_fallback_flag and l1_fallback_flag[0]:
        rn_warns = (
            *rn_warns,
            WarningItem(
                code="L1_SOLVER_FALLBACK",
                severity="warning",
                message=(
                    "HiGHS L1 linear-program solver did not converge cleanly; fell back to "
                    "reweighted L1 IRLS approximation."
                ),
            ),
        )
    diag = DiagnosticReport(
        red_noise=rn,
        condition_number_max=cond_num,
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
        ci_level=ci_level,
        trend_ci_low=trend_ci_low,
        trend_ci_high=trend_ci_high,
    )


def fit_piecewise_trend(
    t: TimeLike,
    y: ArrayLike,
    tp_indices: Sequence[int],
    *,
    weights: ArrayLike | None = None,
    mode: RefitMode = "sequential",
    huber_k: float = 1.345,
    ci_level: float = 0.95,
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
        - ``'sequential'``: Ghaderpour et al. (2024) ``TPTR`` forward/backward chained OLS.
        - ``'joint'``: globally optimal simultaneous continuous hinge-spline OLS.
        - ``'l1'``: globally optimal continuous L1 (LAD) hinge-spline via linear
          programming (robust to isolated outliers, design-dependent breakdown well
          below 50%, strictly continuous at knots;
          pointwise CIs are ``None``).
        - ``'huber'``: globally optimal continuous Huber IRLS hinge-spline.
    huber_k:
        Huber tuning constant used when ``mode='huber'`` (default ``1.345``).
    ci_level:
        Nominal pointwise confidence level in ``(0, 1)`` (default ``0.95``).
    """
    if mode not in ("sequential", "joint", "l1", "huber"):
        raise ValueError(
            f"Unknown `mode` {mode!r}. Options: 'sequential', 'joint', 'l1', 'huber'."
        )
    if not np.isfinite(huber_k) or huber_k <= 0.0:
        raise ValueError(f"`huber_k` must be finite and strictly positive; got {huber_k}.")
    if not (0.0 < ci_level < 1.0):
        raise ValueError(f"`ci_level` must lie in (0, 1); got {ci_level}.")
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
        huber_k=huber_k,
        ci_level=ci_level,
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
    parity_mode: bool = False,
    robust: RobustMethod = "none",
    huber_k: float = 1.345,
    provenance: CensoringProvenance | None = None,
    censor_cfg: CensoredConfig | None = None,
    rng: np.random.Generator | None = None,
    tp_calibration: str = "none",
    n_mc: int = 999,
    mi_spec: tuple[int, str] | None = None,
    mi_counter: list[int] | None = None,
    jump_test: str = "ndri",
    step_alpha: float | None = None,
    serial: SerialSpec = NO_SERIAL,
) -> tuple[
    list[tuple[int, float, _WindowTpdStats, int, int]],
    list[tuple[int, _WindowTpdStats]],
    bool,
    int,
]:
    """Run STPD across sliding windows, returning accepted TPs and high-NDRI jump candidates.

    With ``tp_calibration`` ``"montecarlo"`` or ``"bonferroni"``, each candidate window's
    slope-change p-value is corrected for the data-driven choice of knot and direction,
    candidates are merged within ``tp_th``, and the merged p-values are Holm-adjusted so
    ``alpha`` bounds the probability of accepting any false turning point.
    ``mi_spec = (n_imputations, dist)`` pools each candidate window's calibrated
    p-value (Licht–Rubin) over ``n_imputations`` proper imputations of the window drawn
    from a censored straight-line regression (the window's null model); the other
    gates use the primary imputation. ``mi_counter`` records
    ``[n_pooled, n_fallback]``.

    ``jump_test`` decides whether a candidate break is a vertical jump (rejected as a
    turning point and reported as a flagged jump candidate): ``"step"`` uses the nested
    discontinuous-vs-continuous-hinge test (:func:`_step_vs_hinge_pvalue` below
    ``step_alpha``, default 0.05), ``"ndri"`` the published NDRI threshold, ``"both"`` flags
    a jump when either does.
    """
    eff_step_alpha = 0.05 if step_alpha is None else float(step_alpha)
    k2 = serial.kappa2
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

        stats = _tpd_window(
            t_w, y_w, m_eff, w_w, parity_mode=parity_mode, robust=robust, huber_k=huber_k,
            kappa2=k2,
        )
        ratio_lr = max(stats.n_left, stats.n_right) / max(1, min(stats.n_left, stats.n_right))
        if ratio_lr > 1.5:
            density_imbalanced = True

        snr_val = stats.snr_centered if snr_mode == "centered" else stats.snr_raw
        ndri_val = stats.ndri_rmse if ndri_mode == "rmse" else stats.ndri_l2
        ndri_jump = ndri_val >= ndri_th
        step_jump = bool(
            np.isfinite(stats.step_pval)
            and stats.step_pval < eff_step_alpha
            and stats.step_share >= STEP_SHARE_MIN
        )
        if jump_test == "step":
            is_jump = step_jump
        elif jump_test == "both":
            is_jump = step_jump or ndri_jump
        else:
            is_jump = ndri_jump
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
        if tp_calibration == "none":
            if fb_ok and sig_and_strong and not is_jump:
                dist_center = abs(stats.local_tp - (n_w - 1) / 2.0)
                accepted_raw.append((g_idx, dist_center, stats, w_start, w_stop))
        else:
            gates_ok = (
                fb_ok
                and (snr_val > snr_th)
                and (abs(stats.direction) > dir_th)
                and not is_jump
                and not np.isnan(stats.pval)
            )
            if gates_ok:
                n_knots = 2 * max(1, min(n_w - 2, n_w - m_eff) - max(2, m_eff))
                imps: list[FloatArray] | None = None
                if (
                    mi_spec is not None
                    and provenance is not None
                    and bool(np.any(provenance.status_codes[w_start:w_stop] != 0))
                ):
                    span_w = float(t_w[-1] - t_w[0]) if t_w[-1] > t_w[0] else 1.0
                    X_imp = np.column_stack([np.ones(n_w), (t_w - t_w[0]) / span_w])
                    imps = window_congenial_imputations(
                        provenance.original_values[w_start:w_stop],
                        provenance.status_codes[w_start:w_stop],
                        provenance.original_left[w_start:w_stop]
                        if provenance.original_left is not None
                        else None,
                        provenance.original_right[w_start:w_stop]
                        if provenance.original_right is not None
                        else None,
                        X_imp,
                        mi_spec[1],
                        mi_spec[0],
                        (20261126, int(w_start), int(w_stop)),
                        weights=w_w,
                    )
                    if mi_counter is not None:
                        mi_counter[0 if imps is not None else 1] += 1
                if imps is not None:
                    if tp_calibration == "montecarlo" and robust == "none":
                        p_m = _stpd_mc_pvalues(
                            t_w, np.column_stack(imps), m_eff, w_w, n_mc,
                            (20261125, int(w_start), int(w_stop)), skip_above=alpha,
                            serial=serial,
                        )
                        p_cal = pool_pvalues(p_m) if bool(np.any(p_m < 1.0)) else 1.0
                    else:
                        # Pool the fixed-knot p-values, then apply the Bonferroni factor
                        raw = [
                            deflate_t_pvalue(
                                _tpd_window(
                                    t_w, y_m, m_eff, w_w,
                                    parity_mode=parity_mode, robust=robust, huber_k=huber_k,
                                ).pval,
                                n_w - 3,
                                k2,
                            )
                            for y_m in imps
                        ]
                        p_cal = float(min(1.0, n_knots * pool_pvalues(np.asarray(raw))))
                elif stats.pval >= alpha:
                    p_cal = 1.0  # a calibrated p-value can only be larger
                elif tp_calibration == "montecarlo" and robust == "none":
                    p_cal = _stpd_mc_pvalue(
                        t_w, y_w, m_eff, w_w, n_mc, (20261125, int(w_start), int(w_stop)),
                        serial=serial,
                    )
                else:
                    p_cal = float(min(1.0, n_knots * deflate_t_pvalue(stats.pval, n_w - 3, k2)))
                dist_center = abs(stats.local_tp - (n_w - 1) / 2.0)
                accepted_raw.append(
                    (g_idx, dist_center, replace(stats, pval=p_cal), w_start, w_stop)
                )
        if sig_and_strong and is_jump:
            flagged_raw.append((g_idx, stats))

    # Group nearby candidates within tp_th and deduplicate (CRT-01):
    # When calibrated, compute each merged cluster's Simes p-value min_j (c / j) * p_(j)
    # so an insignificant window closer to center can never overwrite a significant turning
    # point, while preserving exact FWER <= alpha under positively dependent overlapping windows
    # without over-penalizing Holm's denominator with redundant overlapping windows.
    accepted_raw.sort(key=lambda item: item[0])
    if tp_calibration != "none" and accepted_raw:
        clusters: list[list[tuple[int, float, _WindowTpdStats, int, int]]] = [[accepted_raw[0]]]
        for item in accepted_raw[1:]:
            if t_valid[item[0]] - t_valid[clusters[-1][-1][0]] < tp_th:
                clusters[-1].append(item)
            else:
                clusters.append([item])
        merged: list[tuple[int, float, _WindowTpdStats, int, int]] = []
        for cl in clusters:
            c_len = len(cl)
            p_sorted = np.sort(np.array([it[2].pval for it in cl], dtype=np.float64))
            p_simes = float(
                np.min(np.minimum(1.0, (c_len / np.arange(1, c_len + 1, dtype=np.float64)) * p_sorted))
            )
            # Pick representative closest to window center among significant windows (or min pval)
            sig_members = [it for it in cl if it[2].pval < alpha]
            if sig_members:
                rep = min(sig_members, key=lambda it: (it[1], it[2].pval))
            else:
                rep = min(cl, key=lambda it: (it[2].pval, it[1]))
            merged.append((rep[0], rep[1], replace(rep[2], pval=p_simes), rep[3], rep[4]))

        # Holm step-down across merged candidate clusters, then keep p_adj < alpha
        raw_p = np.array([item[2].pval for item in merged], dtype=np.float64)
        m_c = int(raw_p.size)
        order = np.argsort(raw_p)
        adj = np.empty(m_c, dtype=np.float64)
        running = 0.0
        for rank, i_c in enumerate(order):
            running = max(running, min(1.0, (m_c - rank) * raw_p[i_c]))
            adj[i_c] = running
        accepted_raw = [
            (it[0], it[1], replace(it[2], pval=float(p_a)), it[3], it[4])
            for it, p_a in zip(merged, adj, strict=True)
            if p_a < alpha
        ]
    else:
        r = 1
        while r < len(accepted_raw):
            if t_valid[accepted_raw[r][0]] - t_valid[accepted_raw[r - 1][0]] < tp_th:
                if accepted_raw[r][1] < accepted_raw[r - 1][1]:
                    accepted_raw[r - 1] = accepted_raw[r]
                del accepted_raw[r]
            else:
                r += 1

    # Deduplicate nearby flagged jump candidates within tp_th, keeping the strongest
    # jump evidence (smallest step p-value, or highest NDRI under jump_test="ndri")
    def _jump_strength(st: _WindowTpdStats) -> float:
        if jump_test == "ndri":
            return st.ndri_rmse if ndri_mode == "rmse" else st.ndri_l2
        return -st.step_pval if np.isfinite(st.step_pval) else -np.inf

    flagged_raw.sort(key=lambda item: item[0])
    # Every window that flagged a step, before de-duplication (used for reconciliation)
    all_flag_times = [float(t_valid[g]) for g, _ in flagged_raw]
    r = 1
    while r < len(flagged_raw):
        if t_valid[flagged_raw[r][0]] - t_valid[flagged_raw[r - 1][0]] < tp_th:
            nd_curr = _jump_strength(flagged_raw[r][1])
            nd_prev = _jump_strength(flagged_raw[r - 1][1])
            if nd_curr > nd_prev:
                flagged_raw[r - 1] = flagged_raw[r]
            del flagged_raw[r]
        else:
            r += 1

    # Cross-window reconciliation (AUD-03): overlapping windows see the same break from
    # different positions. A window with the break near its edge has little power for the
    # step test and may accept a hinge, while a better-centred window shows that a vertical
    # step explains it. Significant step evidence wins: drop turning points within tp_th
    # of a flagged step.
    if jump_test != "ndri" and all_flag_times and accepted_raw:
        accepted_raw = [
            it
            for it in accepted_raw
            if all(abs(float(t_valid[it[0]]) - ft) >= tp_th for ft in all_flag_times)
        ]

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
    snr_th: float | None = None,
    snr_mode: SnrMode = "centered",
    ndri_th: float = 0.3,
    ndri_mode: NdriMode = "rmse",
    jump_test: JumpTest | None = None,
    step_alpha: float | None = None,
    serial_correlation: SerialMode = "ar1",
    alpha: float = 0.01,
    dir_th: float = 0.0,
    tp_th: float = 1.0,
    margin: int | None = None,
    max_fb_diff: int = 3,
    max_fb_time_diff: float | None = None,
    refit_mode: RefitMode | None = None,
    ci_level: float = 0.95,
    robust: RobustMethod = "none",
    huber_k: float = 1.345,
    censor_status: ArrayLike | None = None,
    censoring: CensoredConfig | None = None,
    parity_mode: bool = False,
    tp_calibration: Literal["montecarlo", "bonferroni", "none"] | None = None,
    n_mc: int = 999,
    band_method: Literal["bootstrap", "conditional"] | None = None,
    n_boot: int = 499,
) -> StpdResult:
    """Detect continuous trend turning points via STPD (Ghaderpour et al., 2024).

    Fits connected two-piece linear trends forward and backward within sliding
    windows, testing slope-change significance (Student t-test, or
    Schrader–Hettmansperger robust F-test when ``robust != 'none'``),
    forward/backward agreement, signal-to-noise ratio (SNR), and the
    Normalized Difference Residual Index (NDRI) to separate genuine continuous
    turning points (``NDRI ≈ 0``) from abrupt datum jumps (``NDRI -> 1``).

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
    window_duration, step_duration, min_window_obs:
        Window span (default ``min(5, span)`` years), step (default
        ``window_duration/3``) and minimum observations per window (default ``10``)
        for ``window_mode='duration'``.
    snr_th, snr_mode:
        SNR threshold and mode. ``snr_th=None`` (default) resolves to ``0`` (no SNR
        gate: the calibrated slope-change test controls false detections) or to the
        published ``1.0`` with ``parity_mode=True``. A positive threshold is a hard
        gate independent of sample size, so turning points whose within-window trend
        range is comparable to the noise level are rejected however small their
        p-value. Mode: ``'centered'`` (default, shift-invariant for non-zero
        baselines) or ``'raw'`` (Ghaderpour 2024); ``parity_mode=True`` forces
        ``'raw'``.
    ndri_th, ndri_mode:
        NDRI threshold (default ``0.3``) and mode: ``'rmse'`` (default,
        unbiased under uneven sampling density) or ``'l2'`` (Ghaderpour 2024).
        ``parity_mode=True`` forces ``'l2'``. NDRI is always computed and reported; it
        only gates turning points when ``jump_test`` is ``'ndri'`` or ``'both'``.
    jump_test:
        How a significant break is classified as an abrupt vertical jump (rejected as a
        turning point and reported in ``flagged_jumps``). ``'step'`` (default) applies a
        nested F-test of a discontinuous two-line fit against the best *continuous*
        hinge (knot optimized continuously between samples, Bonferroni over the scanned
        splits) at level ``step_alpha``, and additionally requires the step to capture at
        least 25% of the break's explained variance (``step_share``), so that tiny model
        misspecification at high signal-to-noise does not turn a genuine bend into a
        "jump". ``'ndri'`` uses the published NDRI threshold
        (forced by ``parity_mode=True``); it fails to recognise most pure steps and
        rejects very precise bends. ``'both'`` flags a jump if either rule does.
    serial_correlation:
        Error model used to calibrate the slope-change and step tests (audit finding
        AUD-02): ``"ar1"`` (default) estimates a lag-1 residual autocorrelation ``rho``
        (difference-based, insensitive to jumps), simulates the Monte Carlo knot-search
        null with AR(1) errors and deflates analytical t/F references by
        ``kappa^2 = (1 + rho) / (1 - rho)``; ``"ou"`` uses continuous-time correlation
        ``rho ** (dt / median_dt)``; ``"none"`` assumes independent errors (forced by
        ``parity_mode=True``). With independent-error calibration, 27.5% of straight-line
        series with AR(1) ``rho = 0.5`` noise gave a false turning point at ``alpha=0.01``.
    step_alpha:
        Level of the (Bonferroni-corrected, conservative) step-vs-bend test for
        ``jump_test='step'``/``'both'`` (default ``0.05``). Reporting a step as a smooth
        turning point is the costlier error, so the default leans towards flagging; in
        simulation it rejected 0.5–2% of genuine bends.
    alpha:
        Significance level for the slope-change test (default ``0.01``). With a
        calibrated ``tp_calibration`` it bounds the probability of reporting any
        turning point on a straight-line series (about 0.2–0.6% in simulation).
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
        Global piecewise trend refit mode: ``'sequential'`` (TPTR), ``'joint'``,
        ``'l1'`` (continuous LAD hinge-spline), or ``'huber'`` (continuous Huber
        hinge-spline). Defaults to ``'huber'`` when ``robust != 'none'``
        (``'huber'`` or ``'mm'``) and ``'sequential'`` otherwise.
    ci_level:
        Nominal pointwise confidence level in ``(0, 1)`` for the global
        piecewise trend fit (default ``0.95``).
    robust:
        ``'none'`` (default WLS/OLS), ``'huber'`` (Huber-weighted asymmetric
        sequential scan + Schrader–Hettmansperger robust test), or ``'mm'``
        (Huber-initialized Tukey biweight MM test). Forced to ``'none'`` when
        ``parity_mode=True``.
    huber_k:
        Huber tuning constant (default ``1.345``).
    censor_status, censoring:
        Optional censoring indicator array and :class:`CensoredConfig`.
    parity_mode:
        If ``True``, uses exact Ghaderpour et al. (2024) conventions
        (``snr_mode='raw'``, ``ndri_mode='l2'``, ``robust='none'``, ``snr_th=1.0``,
        ``tp_calibration='none'``).
    tp_calibration:
        How the slope-change p-value accounts for searching every knot in a window:
        ``'montecarlo'`` (default for non-robust fits) refers the maximum t statistic
        over knots and scan directions to its exact Monte Carlo null under a straight
        line with Gaussian errors; ``'bonferroni'`` (default for robust fits)
        multiplies the raw p-value by the number of candidate knots; ``'none'``
        (forced by ``parity_mode``) uses raw screening p-values. Calibrated p-values
        are Holm-adjusted across candidate locations.
    n_mc:
        Monte Carlo replicates per window for ``tp_calibration='montecarlo'``
        (default ``999``). Candidates whose raw p-value already exceeds ``alpha`` are
        not simulated.
    band_method:
        Pointwise band of the piecewise trend. ``'bootstrap'`` (default, forced to
        ``'conditional'`` by ``parity_mode``) is a wild residual bootstrap of the
        detect-then-fit procedure: every replicate re-locates each turning point with
        STPD's own knot estimator over the window that detected it and refits the
        trend, so knot-location uncertainty is included. ``'conditional'`` is the
        analytical band that treats the turning points as known (it under-covers after
        detection: about 85-92% at nominal 95% in simulation). Without turning points
        the analytical band is used.
    n_boot:
        Bootstrap replicates for ``band_method='bootstrap'`` (default ``499``).
    """
    if window_mode not in ("count", "duration"):
        raise ValueError(f"Unknown `window_mode` {window_mode!r}. Options: 'count', 'duration'.")
    if snr_mode not in ("raw", "centered"):
        raise ValueError(f"Unknown `snr_mode` {snr_mode!r}. Options: 'raw', 'centered'.")
    if ndri_mode not in ("l2", "rmse"):
        raise ValueError(f"Unknown `ndri_mode` {ndri_mode!r}. Options: 'l2', 'rmse'.")
    eff_robust: RobustMethod = "none" if parity_mode else validate_robust_args(robust, huber_k)
    eff_refit_mode: RefitMode = (
        ("huber" if eff_robust != "none" else "sequential")
        if refit_mode is None
        else refit_mode
    )
    if eff_refit_mode not in ("sequential", "joint", "l1", "huber"):
        raise ValueError(
            f"Unknown `refit_mode` {eff_refit_mode!r}. "
            "Options: 'sequential', 'joint', 'l1', 'huber'."
        )
    eff_snr_th = (1.0 if parity_mode else SNR_TH_DEFAULT) if snr_th is None else float(snr_th)
    if eff_snr_th < 0.0:
        raise ValueError("`snr_th` must be non-negative.")
    if tp_calibration is not None and tp_calibration not in ("montecarlo", "bonferroni", "none"):
        raise ValueError(
            f"Unknown `tp_calibration` {tp_calibration!r}. Options: 'montecarlo', 'bonferroni', 'none'."
        )
    if not parity_mode and int(n_mc) < 99:
        raise ValueError(f"`n_mc` must be >= 99; got {n_mc}.")
    if band_method is not None and band_method not in ("bootstrap", "conditional"):
        raise ValueError(
            f"Unknown `band_method` {band_method!r}. Options: 'bootstrap', 'conditional'."
        )
    if not parity_mode and int(n_boot) < 99:
        raise ValueError(f"`n_boot` must be >= 99; got {n_boot}.")
    eff_band = "conditional" if parity_mode else (band_method or "bootstrap")
    if not (0.0 <= ndri_th <= 1.0):
        raise ValueError("`ndri_th` must lie in [0, 1].")
    if not (0.0 < alpha < 1.0):
        raise ValueError("`alpha` must lie in (0, 1).")
    if not (0.0 < ci_level < 1.0):
        raise ValueError(f"`ci_level` must lie in (0, 1); got {ci_level}.")
    if dir_th < 0.0:
        raise ValueError("`dir_th` must be non-negative.")
    if tp_th <= 0.0:
        raise ValueError("`tp_th` must be strictly positive.")
    if max_fb_diff < 0:
        raise ValueError("`max_fb_diff` must be non-negative.")

    eff_snr_mode: SnrMode = "raw" if parity_mode else snr_mode
    if parity_mode:
        eff_tp_cal = "none"
    elif tp_calibration is not None:
        eff_tp_cal = tp_calibration
    else:
        eff_tp_cal = "bonferroni" if eff_robust != "none" else "montecarlo"
    eff_ndri_mode: NdriMode = "l2" if parity_mode else ndri_mode
    if jump_test is not None and jump_test not in ("step", "ndri", "both"):
        raise ValueError(f"Unknown `jump_test` {jump_test!r}. Options: 'step', 'ndri', 'both'.")
    if step_alpha is not None and not (0.0 < step_alpha < 1.0):
        raise ValueError(f"`step_alpha` must lie in (0, 1); got {step_alpha}.")
    eff_jump_test: str = "ndri" if parity_mode else (jump_test or "step")

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
    warnings_list.extend(time_unit_warning(series.times_numeric, series.is_datetime))
    n = series.n_valid
    span = float(series.times_numeric[-1] - series.times_numeric[0])
    if span <= 0.0:
        raise InsufficientDataError("Time span must be strictly positive.")
    # Residual serial correlation (AUD-02). The smooth global fit includes annual
    # harmonics when the record spans >= 2 time units so that residual seasonality in a
    # non-deseasonalized input is not mistaken for autocorrelation.
    if eff_robust == "none" and not parity_mode:
        ht_warn = heavy_tail_warning(
            series.times_numeric, series.values, w_valid, harmonics=np.array([1.0, 2.0, 3.0, 4.0]) if span >= 2.0 else None
        )
        if ht_warn is not None:
            warnings_list.append(ht_warn)
    serial_spec = resolve_serial_spec(
        serial_correlation,
        parity_mode=parity_mode,
        t=series.times_numeric,
        y=series.values,
        weight_kind="vector" if w_valid is not None else "none",
        weight_vector=w_valid,
        season="ols" if span >= 2.0 else "none",
        omega=np.array([1.0, 2.0, 3.0, 4.0]),
    )
    if serial_spec.active:
        warnings_list.append(
            WarningItem(
                code="SERIAL_CORRELATION_ADJUSTED",
                severity="info",
                message=serial_warning_message(serial_spec),
            )
        )

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

    # Multiple imputation of censored values: candidate p-values are pooled over
    # `n_imputations` congenial imputations of each window (see _run_stpd_windows).
    mi_spec: tuple[int, str] | None = None
    if (
        not parity_mode
        and cfg is not None
        and prov is not None
        and prov.n_censored > 0
        and cfg.detection_imputations > 1
        and cfg.scope != "windowed"
        and eff_tp_cal != "none"
    ):
        mi_dist = resolve_imputation_dist(prov, cfg)
        if mi_dist is not None:
            mi_spec = (int(cfg.detection_imputations), mi_dist)
    mi_counter = [0, 0]

    accepted_raw, flagged_raw, density_imbalanced, win_fb = _run_stpd_windows(
        series.times_numeric,
        series.values,
        w_valid,
        windows,
        eff_margin,
        eff_snr_th,
        eff_snr_mode,
        ndri_th,
        eff_ndri_mode,
        alpha,
        dir_th,
        tp_th,
        max_fb_diff,
        max_fb_time_diff,
        parity_mode=parity_mode,
        robust=eff_robust,
        huber_k=huber_k,
        provenance=prov,
        censor_cfg=cfg,
        rng=rng,
        tp_calibration=eff_tp_cal,
        n_mc=int(n_mc),
        mi_spec=mi_spec,
        mi_counter=mi_counter,
        jump_test=eff_jump_test,
        step_alpha=step_alpha,
        serial=serial_spec,
    )
    if mi_counter[1] > 0:
        warnings_list.append(
            WarningItem(
                code="MI_DETECTION_FALLBACK",
                severity="caution",
                message=(
                    f"{mi_counter[1]} candidate window(s) could not be multiply imputed (censored "
                    "regression did not converge); their p-values condition on the single "
                    "primary imputation."
                ),
            )
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
        if prov is not None:
            prov = replace(prov, window_fallback_count=prov.window_fallback_count + win_fb)
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
        stab_fb = [0]
        stab_draws = generate_stability_draws(prov, cfg, series.times_numeric, stab_fb)
        if stab_fb[0] > 0:
            warnings_list.append(
                WarningItem(
                    code="STABILITY_DRAW_FALLBACK",
                    severity="caution",
                    message=(
                        f"{stab_fb[0]} censoring stability draw(s) could not refit the imputation "
                        "model and reused fixed parameters, so stability rates understate "
                        "parameter uncertainty."
                    ),
                )
            )
        win_rngs = (
            _spawn_stability_rngs(cfg.random_state, len(stab_draws))
            if cfg.scope == "windowed"
            else [None] * len(stab_draws)
        )
        for y_draw, draw_rng in zip(stab_draws, win_rngs, strict=True):
            cfg_draw = (
                CensoredConfig(
                    method=cfg.method,
                    censoring_type=cfg.censoring_type,
                    dist=cfg.dist,
                    impute_type="stochastic",
                    scope=cfg.scope,
                    random_state=draw_rng,
                    clamp=cfg.clamp,
                    ndimpute_kwargs=cfg.ndimpute_kwargs,
                )
                if cfg.scope == "windowed"
                else None
            )
            acc_d, _, _, _ = _run_stpd_windows(
                series.times_numeric,
                y_draw,
                w_valid,
                windows,
                eff_margin,
                eff_snr_th,
                eff_snr_mode,
                ndri_th,
                eff_ndri_mode,
                alpha,
                dir_th,
                tp_th,
                max_fb_diff,
                max_fb_time_diff,
                parity_mode=parity_mode,
                robust=eff_robust,
                huber_k=huber_k,
                provenance=prov if cfg.scope == "windowed" else None,
                censor_cfg=cfg_draw,
                rng=draw_rng,
                tp_calibration=eff_tp_cal,
                n_mc=int(n_mc),
                jump_test=eff_jump_test,
                step_alpha=step_alpha,
                serial=serial_spec,
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
        mode=eff_refit_mode,
        huber_k=huber_k,
        ci_level=ci_level,
    )

    band_used = "conditional"
    if eff_band == "bootstrap" and accepted_raw:
        tp_win = [
            (int(it[0]), int(it[3]), int(it[4]), max(2, min(eff_margin, (int(it[4]) - int(it[3]) - 3) // 2)))
            for it in accepted_raw
        ]

        def _refit(yb: FloatArray, kb: IntArray) -> FloatArray:
            return _fit_piecewise_core(
                series.times_numeric, series.times_raw, series.is_datetime, yb, kb,
                series.valid_to_orig, series.n_input, series.n_dropped_nan,
                series.sampling_rate_ratio, w=w_valid, mode=eff_refit_mode,
                huber_k=huber_k, ci_level=ci_level,
            ).fitted_trend

        a_sd = None
        if pw_trend.trend_ci_low is not None and pw_trend.trend_ci_high is not None:
            dof_a = max(1, int(series.n_valid) - 2 - len(tp_win))
            a_sd = 0.5 * (pw_trend.trend_ci_high - pw_trend.trend_ci_low) / float(
                student_t.ppf(0.5 + 0.5 * ci_level, df=dof_a)
            )
        boot = _bootstrap_tp_curves(
            series.times_numeric, series.values, pw_trend.fitted_trend, tp_win, _refit,
            w=w_valid, robust=eff_robust, huber_k=huber_k, n_boot=int(n_boot),
            seed=(20261126, int(series.n_valid), *[int(x[0]) for x in tp_win]),
            analytic_sd=a_sd,
            l1_refit=(eff_refit_mode == "l1"),
        )
        if boot is not None:
            dev = (boot.curves - pw_trend.fitted_trend[:, None]) * boot.dev_scale[:, None]
            a_lvl = 1.0 - ci_level
            # Percentile intervals of the (variance-calibrated) replicate curves; they were
            # slightly closer to nominal than basic intervals near the turning points.
            pw_trend = replace(
                pw_trend,
                trend_ci_low=pw_trend.fitted_trend + np.quantile(dev, 0.5 * a_lvl, axis=1),
                trend_ci_high=pw_trend.fitted_trend + np.quantile(dev, 1.0 - 0.5 * a_lvl, axis=1),
            )
            band_used = "bootstrap"

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
                detection_window=(
                    int(w_start),
                    int(w_stop),
                    max(2, min(eff_margin, (int(w_stop) - int(w_start) - 3) // 2)),
                ),
                step_pvalue=float(st.step_pval),
                step_share=float(st.step_share),
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
            step_pvalue=float(st.step_pval),
            step_share=float(st.step_share),
        )
        for g_idx, st in flagged_raw
    )
    if flagged_events:
        by_step = eff_jump_test != "ndri"
        warnings_list.append(
            WarningItem(
                code="STEP_JUMP_CANDIDATE" if by_step else "HIGH_NDRI_JUMP_CANDIDATE",
                severity="caution",
                message=(
                    f"STPD flagged {len(flagged_events)} candidate(s) that look more like abrupt "
                    "vertical jumps than continuous turning points "
                    + ("(significant step-vs-continuous-hinge test)." if by_step else "(high NDRI).")
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
        snr_th=eff_snr_th,
        ndri_th=ndri_th,
        dir_th=dir_th,
        tp_th=tp_th,
        censoring=prov,
        diagnostics=diag,
        tp_calibration=eff_tp_cal,
        n_imputations=int(mi_spec[0]) if (mi_spec is not None and mi_counter[0] > 0) else 1,
        band_method=band_used,
        robust=eff_robust,
        jump_test=eff_jump_test,
        serial_correlation=serial_spec.mode,
        serial_rho=float(serial_spec.rho_hat),
        serial_rho_used=float(serial_spec.rho),
    )


# Concise top-level alias
stpd = detect_turning_points
