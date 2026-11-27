"""Jumps Upon Spectrum and Trend (JUST) jump detection and decomposition.

Clean-room implementation grounded in:
- Ghaderpour, E., & Vujadinovic, T. (2020). Change Detection within Remotely
  Sensed Satellite Image Time Series via Spectral Analysis. *Remote Sensing*,
  12(23), 4001. https://doi.org/10.3390/rs12234001
- Ghaderpour, E. (2021). JUST: MATLAB and Python Software for Change Detection
  and Time Series Analysis. *GPS Solutions*, 25(3), 85.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.stats import f as f_dist
from scipy.stats import t as t_dist

from ._cenreg import pool_pvalues, window_congenial_imputations
from ._diagnostics import (
    DiagnosticReport,
    WarningItem,
    compute_red_noise_diagnostics,
    time_unit_warning,
)
from ._linalg import PreparedWeights, prepare_weights, whiten_matrix, whiten_vector
from ._robust import (
    RobustMethod,
    huber_nested_ftest,
    huber_rho,
    solve_robust_irls,
    tukey_rho,
    validate_robust_args,
)
from ._time import convert_eval_times, format_timestamp
from ._types import ArrayLike, FloatArray, IntArray, TimeLike, ValueLike, WeightLike
from .censored import (
    CensoredConfig,
    CensoringProvenance,
    check_dl_shift_coincidence,
    generate_stability_draws,
    impute_window_local,
    prepare_series_with_censoring,
    recensor_to_common_limit,
    resolve_imputation_dist,
)
from .exceptions import InsufficientDataError
from .results import (
    BaseResult,
    format_diagnostics_block,
    format_explain_warnings,
    require_pandas,
)
from .spectral import (
    _allssa_core,
    _lssa_core,
    _validate_jump_indices,
    build_design_matrix,
)

SeasonMode = Literal["allssa", "ols", "none"]
WindowMode = Literal["count", "duration"]
JumpCalibration = Literal["montecarlo", "bonferroni", "none"]
SeasonSignificance = Literal["sidak", "montecarlo", "per_frequency"]


def _short_seasonal_window_warning(
    t: FloatArray, windows: list[tuple[int, int]], omega: FloatArray, season: str, sig: str
) -> WarningItem | None:
    """Warn when windows cover fewer than two cycles of the lowest searched frequency."""
    if season != "allssa" or sig == "per_frequency" or omega.size == 0 or not windows:
        return None
    spans = [float(t[b - 1] - t[a]) for a, b in windows if b - a > 1]
    if not spans:
        return None
    cycles = float(np.median(spans)) * float(np.min(omega))
    if cycles >= 2.0:
        return None
    return WarningItem(
        code="SHORT_SEASONAL_WINDOW",
        severity="caution",
        message=(
            f"Windows span about {cycles:.1f} cycle(s) of the lowest searched frequency, so real "
            "harmonics may not be family-wise identifiable and seasonality can leak into the "
            "trend/jump estimates."
        ),
        suggestion=(
            "Use longer windows (`size`/`window_duration`), `season='ols'` with known frequencies, "
            "or `season_significance='per_frequency'` if seasonality is known to be present."
        ),
    )


def _resolve_season_significance(value: str, parity_mode: bool) -> str:
    if value not in ("sidak", "montecarlo", "per_frequency"):
        raise ValueError(
            f"Unknown `season_significance` {value!r}. Options: 'sidak', 'montecarlo', 'per_frequency'."
        )
    return "per_frequency" if parity_mode else value


@dataclass(frozen=True)
class WindowCandidate:
    """Single-window jump candidate produced by sliding-window JUST search."""

    window_start: int
    window_stop: int
    valid_index: int
    index: int
    magnitude: float
    direction: float
    center_distance: float
    norm_res: float
    selected_freqs: FloatArray


@dataclass(frozen=True)
class JumpEvent:
    """Detected abrupt jump and slope change returned by :func:`just_jumps`.

    Attributes
    ----------
    index, valid_index : int
        First post-jump observation on the input / valid axis.
    time, time_year
        Its timestamp (original representation / decimal years).
    magnitude : float
        Vertical step at the jump (post minus pre fitted trend).
    direction : float
        Slope change (post minus pre, per time unit).
    occurrences : int
        Number of windows whose best split fell on this index.
    window_center_distance : float
        Distance (observations) from the representative window's center.
    jump_pvalue : float
        p-value of the level-and-slope change (2 df), calibrated per
        ``JustJumpResult.jump_calibration``.
    step_pvalue : float
        p-value of a vertical discontinuity beyond a continuous hinge (1 df).
    window_censored_fraction, coincides_with_dl_shift, dl_artifact_likely, stability_rate
        Censored-data diagnostics.
    """

    index: int
    valid_index: int
    time: Any
    time_year: float
    magnitude: float
    direction: float
    occurrences: int
    window_center_distance: float
    jump_pvalue: float
    step_pvalue: float = float("nan")
    window_censored_fraction: float = 0.0
    coincides_with_dl_shift: bool = False
    dl_artifact_likely: bool | None = None
    stability_rate: float | None = None


@dataclass(frozen=True)
class JustJumpResult(BaseResult):
    """Result object returned by :func:`just_jumps`.

    Attributes
    ----------
    jumps : tuple of JumpEvent
        Accepted jumps (index, time, magnitude, slope change, p-values, flags).
    all_window_candidates : tuple of WindowCandidate
        Best split found in every window before aggregation and filtering.
    times_numeric, times_raw, values, is_datetime
        Valid timestamps and (imputed) values.
    season, window_mode : str
        Seasonal model and windowing mode used.
    alpha : float
        Per-frequency ALLSSA level inside windows.
    mag_th, dir_th, jump_th : float
        Effect-size thresholds and minimum separation (as resolved).
    jump_alpha : float or None
        Family-wise significance level applied to ``jump_pvalue`` (``None`` = no
        statistical filter).
    censoring : CensoringProvenance or None
        Censored-data provenance.
    diagnostics : DiagnosticReport
        Warnings and red-noise diagnostics.
    jump_calibration : str
        ``"montecarlo"``, ``"bonferroni"``, or ``"none"``: how ``jump_pvalue``
        accounts for the split search (plus Holm adjustment across candidates).
    n_imputations : int
        Number of imputations whose per-candidate p-values were pooled (Licht–Rubin)
        for censored data; ``1`` means a single imputation.
    """

    jumps: tuple[JumpEvent, ...]
    all_window_candidates: tuple[WindowCandidate, ...]
    times_numeric: FloatArray
    times_raw: np.ndarray[Any, Any]
    values: FloatArray
    is_datetime: bool
    season: str
    window_mode: str
    alpha: float
    mag_th: float
    dir_th: float
    jump_th: float
    jump_alpha: float | None
    censoring: CensoringProvenance | None
    diagnostics: DiagnosticReport
    jump_calibration: str = "none"
    n_imputations: int = 1

    @property
    def jump_indices(self) -> IntArray:
        """0-based jump indices mapped onto the original input array axis."""
        return np.array([j.index for j in self.jumps], dtype=np.intp)

    @property
    def jump_times(self) -> np.ndarray[Any, Any]:
        """Native timestamps of detected jumps."""
        if not self.jumps:
            return np.empty(0, dtype=self.times_raw.dtype)
        return np.array([j.time for j in self.jumps], dtype=self.times_raw.dtype)

    def summary(self) -> str:
        lines = [
            f"JUST Jump Detection Result (season={self.season!r}, "
            f"window_mode={self.window_mode!r}, alpha={self.alpha}, "
            f"jump_alpha={self.jump_alpha}, jump_calibration={self.jump_calibration!r})",
            f"Thresholds   : |mag| >= {self.mag_th} or |dir| >= {self.dir_th}, "
            f"min separation >= {self.jump_th}",
            f"Jumps found  : {len(self.jumps)}",
        ]
        if self.jumps:
            lines.append(
                "  Idx   Time         Magnitude    Direction    Occ  p-value     Flags"
            )
            for j in self.jumps:
                t_str = format_timestamp(j.time, self.is_datetime)
                flags: list[str] = []
                if j.coincides_with_dl_shift:
                    flags.append("DL-SHIFT")
                if j.dl_artifact_likely:
                    flags.append("DL-ARTIFACT?")
                if j.stability_rate is not None:
                    flags.append(f"stab={j.stability_rate:.0%}")
                flag_str = ",".join(flags) if flags else "-"
                lines.append(
                    f"  {j.index:<5d} {t_str:<12s} {j.magnitude:+11.4f}  "
                    f"{j.direction:+11.4f}  {j.occurrences:<4d} {j.jump_pvalue:<11.3e} {flag_str}"
                )
        lines.extend(format_diagnostics_block(self.diagnostics))
        return "\n".join(lines)

    def explain(self) -> str:
        lines: list[str] = []
        if not self.jumps:
            lines.append(
                f"No abrupt jumps met the selection criteria across {self.diagnostics.n_valid} "
                f"observations (using {self.season.upper()} seasonal modeling)."
            )
        else:
            lines.append(
                f"Detected {len(self.jumps)} abrupt change(s) in the trend component "
                f"(using {self.season.upper()} seasonal modeling without interpolating gaps):"
            )
            p_label = (
                "family-wise calibrated F-test"
                if self.jump_calibration in ("montecarlo", "bonferroni")
                else "uncalibrated screening F-test"
            )
            if self.n_imputations > 1:
                p_label += f" pooled over {self.n_imputations} imputations"
            for k, j in enumerate(self.jumps, start=1):
                t_str = format_timestamp(j.time, self.is_datetime)
                step_word = "upward" if j.magnitude >= 0 else "downward"
                slope_word = "steepened/increased" if j.direction >= 0 else "declined/decreased"
                desc = (
                    f"  {k}. Around {t_str} (index {j.index}): {step_word} step of "
                    f"{j.magnitude:+.4g} and slope {slope_word} by {j.direction:+.4g}/unit time "
                    f"(seen in {j.occurrences} window(s); {p_label} p={j.jump_pvalue:.2e})."
                )
                if j.stability_rate is not None:
                    desc += f" Recovered in {j.stability_rate:.0%} of censoring stability draws."
                if j.coincides_with_dl_shift:
                    desc += (
                        " WARNING: coincides with a laboratory detection-limit change — verify "
                        "this is a real physical shift rather than a lab method upgrade!"
                    )
                if j.dl_artifact_likely:
                    desc += (
                        " Common-limit recensoring indicates this jump disappears once detection "
                        "limits are harmonized (likely a detection-limit artifact)."
                    )
                lines.append(desc)
            if self.jump_calibration in ("montecarlo", "bonferroni"):
                if self.jump_alpha is not None:
                    lines.append(
                        f"Jump p-values account for the data-driven split search "
                        f"({self.jump_calibration}) and are Holm-adjusted across candidate "
                        f"locations; reported jumps are significant at family-wise "
                        f"alpha={self.jump_alpha:g}."
                    )
            else:
                lines.append(
                    "Note: these are uncalibrated screening p-values computed at the "
                    "RSS-minimizing split index; they are far more optimistic than "
                    "fixed-date tests and do not control false positives."
                )
        lines.extend(format_explain_warnings(self.diagnostics))
        return "\n".join(lines)

    def to_dataframe(self) -> Any:
        pd = require_pandas()
        rows = [
            {
                "index": j.index,
                "valid_index": j.valid_index,
                "time": j.time,
                "time_year": j.time_year,
                "magnitude": j.magnitude,
                "direction": j.direction,
                "occurrences": j.occurrences,
                "window_center_distance": j.window_center_distance,
                "jump_pvalue": j.jump_pvalue,
                "step_pvalue": j.step_pvalue,
                "window_censored_fraction": j.window_censored_fraction,
                "coincides_with_dl_shift": j.coincides_with_dl_shift,
                "dl_artifact_likely": j.dl_artifact_likely,
                "stability_rate": j.stability_rate,
            }
            for j in self.jumps
        ]
        return pd.DataFrame(rows)

    def plot(self, ax: Any = None, **kwargs: Any) -> tuple[Any, Any]:
        from .plotting import plot_just_jumps

        return plot_just_jumps(self, ax=ax, **kwargs)


@dataclass(frozen=True)
class JustDecomposeResult(BaseResult):
    """Result object returned by :func:`just_decompose`.

    Attributes
    ----------
    times_numeric, times_raw, values, is_datetime
        Valid timestamps and (imputed) values.
    trend, seasonal, remainder : ndarray
        Blended components on the observed times; they sum to ``values`` exactly.
    eval_times_numeric, eval_times_raw : ndarray or None
        Evaluation grid when ``eval_times`` was given.
    trend_reg, seasonal_reg : ndarray or None
        Components evaluated on ``eval_times`` (``NaN`` over uncovered gaps).
    jump_indices : ndarray
        Jump indices used (input axis).
    window_frequencies : tuple of ndarray
        Seasonal frequencies fitted in each window.
    window_bounds : tuple of (int, int)
        Valid-axis ``[start, stop)`` of each window, aligned with ``window_frequencies``.
    window_dilation : float
        Exponent factor of the Gaussian window weights ``exp(dilation * (t - tau)^2)``.
    season, window_mode : str
        Settings used.
    censoring : CensoringProvenance or None
        Censored-data provenance.
    diagnostics : DiagnosticReport
        Warnings and red-noise diagnostics.
    ci_level : float
        Pointwise confidence level.
    trend_ci_low, trend_ci_high : ndarray or None
        Trend band. With ``trend_ci_method="bias_corrected"`` it is centered on
        ``trend_bc``, not on ``trend``.
    seasonal_ci_low, seasonal_ci_high : ndarray or None
        Seasonal band (conditional on the selected frequencies).
    trend_reg_ci_low, trend_reg_ci_high, seasonal_reg_ci_low, seasonal_reg_ci_high
        The same bands on ``eval_times``.
    trend_ci_method : str
        ``"bias_corrected"`` or ``"model"``.
    trend_bc, trend_reg_bc : ndarray or None
        Bias-corrected (local-quadratic) trend estimate on observed times / ``eval_times``.
    """

    times_numeric: FloatArray
    times_raw: np.ndarray[Any, Any]
    values: FloatArray
    trend: FloatArray
    seasonal: FloatArray
    remainder: FloatArray
    eval_times_numeric: FloatArray | None
    eval_times_raw: np.ndarray[Any, Any] | None
    trend_reg: FloatArray | None
    seasonal_reg: FloatArray | None
    jump_indices: IntArray
    window_frequencies: tuple[FloatArray, ...]
    is_datetime: bool
    season: str
    window_mode: str
    censoring: CensoringProvenance | None
    diagnostics: DiagnosticReport
    ci_level: float = 0.95
    trend_ci_low: FloatArray | None = None
    trend_ci_high: FloatArray | None = None
    seasonal_ci_low: FloatArray | None = None
    seasonal_ci_high: FloatArray | None = None
    trend_reg_ci_low: FloatArray | None = None
    trend_reg_ci_high: FloatArray | None = None
    seasonal_reg_ci_low: FloatArray | None = None
    seasonal_reg_ci_high: FloatArray | None = None
    trend_ci_method: str = "model"
    trend_bc: FloatArray | None = None
    trend_reg_bc: FloatArray | None = None
    window_bounds: tuple[tuple[int, int], ...] = ()
    window_dilation: float = float("nan")

    def summary(self) -> str:
        t_range = float(np.ptp(self.trend)) if self.trend.size else 0.0
        s_amp = 0.5 * float(np.ptp(self.seasonal)) if self.seasonal.size else 0.0
        r_rms = float(np.sqrt(np.mean(self.remainder**2))) if self.remainder.size else 0.0
        lines = [
            f"JUST Decomposition Result (season={self.season!r}, window_mode={self.window_mode!r})",
            f"Jump indices : {self.jump_indices.tolist()}",
            f"Trend range  : {t_range:.4g}  |  Seasonal half-range: {s_amp:.4g}  "
            f"|  Remainder RMS: {r_rms:.4g}",
        ]
        if self.eval_times_numeric is not None:
            lines.append(f"Regular grid : {self.eval_times_numeric.size} evaluation points")
        lines.extend(format_diagnostics_block(self.diagnostics))
        return "\n".join(lines)

    def explain(self) -> str:
        s_amp = 0.5 * float(np.ptp(self.seasonal)) if self.seasonal.size else 0.0
        r_std = float(np.std(self.remainder)) if self.remainder.size else 0.0
        lines = [
            f"Decomposed {self.diagnostics.n_valid} unevenly spaced observations into trend, "
            f"seasonal, and remainder components using {self.season.upper()} without "
            "interpolating input gaps.",
            f"Estimated seasonal wave half-amplitude is ~{s_amp:.4g}, and unexplained remainder "
            f"standard deviation is ~{r_std:.4g}.",
        ]
        if self.jump_indices.size > 0:
            lines.append(
                f"Piecewise trend explicitly preserved {self.jump_indices.size} jump "
                f"discontinuity(ies) at index(es) {self.jump_indices.tolist()}."
            )
        lines.extend(format_explain_warnings(self.diagnostics))
        return "\n".join(lines)

    def to_dataframe(self) -> Any:
        pd = require_pandas()
        data: dict[str, Any] = {
            "time": self.times_raw,
            "time_year": self.times_numeric,
            "value": self.values,
            "trend": self.trend,
            "seasonal": self.seasonal,
            "remainder": self.remainder,
        }
        if self.trend_ci_low is not None and self.trend_ci_high is not None:
            data["trend_ci_low"] = self.trend_ci_low
            data["trend_ci_high"] = self.trend_ci_high
        if self.trend_bc is not None:
            data["trend_bias_corrected"] = self.trend_bc
        if self.seasonal_ci_low is not None and self.seasonal_ci_high is not None:
            data["seasonal_ci_low"] = self.seasonal_ci_low
            data["seasonal_ci_high"] = self.seasonal_ci_high
        if self.censoring is not None:
            data["is_censored"] = self.censoring.is_censored
            data["censor_status"] = self.censoring.status_codes
            data["original_value"] = self.censoring.original_values
        return pd.DataFrame(data)

    def plot(self, ax: Any = None, **kwargs: Any) -> tuple[Any, Any]:
        from .plotting import plot_just_decompose

        return plot_just_decompose(self, ax=ax, **kwargs)


def _default_just_omega(season: str) -> FloatArray:
    if season == "allssa":
        return np.arange(4, 20, dtype=np.float64) / 5.0
    return np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)


def _build_count_windows(
    n: int,
    t: FloatArray,
    size: int | None,
    step: int | None,
    *,
    parity_mode: bool,
) -> tuple[list[tuple[int, int]], int, int, int]:
    """Generate count-based sliding window index pairs ``(start, stop)``."""
    span = float(t[-1] - t[0])
    if span <= 0.0:
        raise InsufficientDataError("Time span must be strictly positive.")
    rate = int(n / span)
    rate_eff = max(1, rate)

    if size is None:
        win_size = 3 * rate_eff
    else:
        win_size = int(size)
        if win_size > n or win_size < max(6, rate if rate >= 1 else 6):
            if parity_mode:
                if win_size > n or win_size < rate:
                    raise ValueError(
                        f"Window `size` ({win_size}) must lie between M={rate} and n={n}."
                    )
            elif win_size > n or win_size < 6:
                raise ValueError(f"Window `size` ({win_size}) must lie between 6 and n={n}.")

    if win_size > n:
        raise InsufficientDataError(
            f"Default/requested window size ({win_size}) exceeds series length n={n}. "
            "Pass a smaller `size` or collect a longer record."
        )
    if size is None and rate < 2:
        raise InsufficientDataError(
            f"Default window size is 3 x samples per unit time, but the series has only "
            f"{n / span:.3g} samples per unit time (window = {win_size}). Defaults assume time "
            "in decimal years with at least ~2 samples per year; convert t to years or pass `size`."
        )

    if step is None:
        win_step = min(rate_eff, max(1, win_size - 1))
    else:
        win_step = int(step)
        if win_step < 1 or win_step >= win_size:
            raise ValueError(
                f"Translation `step` ({win_step}) must satisfy 1 <= step < size ({win_size})."
            )

    windows: list[tuple[int, int]] = []
    start = 0
    stop = win_size
    while True:
        if parity_mode:
            if stop > n:
                windows.append((n - win_size, n))
                break
            windows.append((start, stop))
            start += win_step
            stop = start + win_size
            if start > n:
                break
        else:
            if stop >= n:
                windows.append((n - win_size, n))
                break
            windows.append((start, stop))
            start += win_step
            stop = start + win_size

    return windows, win_size, win_step, rate_eff


def _build_duration_windows(
    t: FloatArray,
    window_duration: float | None,
    step_duration: float | None,
    min_window_obs: int,
) -> tuple[list[tuple[int, int]], float, int]:
    """Generate calendar/time-duration sliding windows ``(start, stop)``."""
    t_start = float(t[0])
    t_end = float(t[-1])
    span = t_end - t_start
    if span <= 0.0:
        raise InsufficientDataError("Time span must be strictly positive.")

    dur = float(window_duration) if window_duration is not None else min(3.0, span)
    if dur <= 0.0:
        raise ValueError("`window_duration` must be strictly positive.")
    if dur > span:
        dur = span
    step_d = float(step_duration) if step_duration is not None else dur / 3.0
    if step_d <= 0.0 or step_d >= dur and dur < span:
        raise ValueError("`step_duration` must satisfy 0 < step_duration < window_duration.")

    windows: list[tuple[int, int]] = []
    skipped = 0
    left_t = t_start
    tol = 1e-12 * max(1.0, span)
    while True:
        right_t = left_t + dur
        is_last = right_t >= t_end - tol
        if is_last:
            left_t = t_end - dur
            right_t = t_end
        i0 = int(np.searchsorted(t, left_t - tol, side="left"))
        i1 = int(np.searchsorted(t, right_t + tol, side="right"))
        if i1 - i0 >= min_window_obs:
            if not windows or windows[-1] != (i0, i1):
                windows.append((i0, i1))
        else:
            skipped += 1
        if is_last:
            break
        left_t += step_d

    if not windows:
        raise InsufficientDataError(
            f"No duration window contained at least {min_window_obs} observations. "
            "Increase `window_duration` or decrease `min_window_obs`."
        )
    return windows, dur, skipped


def _detect_jump_in_window(
    t_win: FloatArray,
    y_win: FloatArray,
    pw_win: PreparedWeights,
    season: str,
    omega: FloatArray,
    alpha: float,
    *,
    parity_mode: bool,
    robust: RobustMethod = "none",
    huber_k: float = 1.345,
    season_significance: str = "per_frequency",
) -> tuple[int, float, float, float, FloatArray]:
    """Scan all interior split indices ``3 .. n_win - 3`` in a single window."""
    eff_robust: RobustMethod = "none" if parity_mode else robust
    n_win = int(t_win.size)
    t0 = t_win - t_win[0]
    best_ind = -1
    best_mag = 0.0
    best_dir = 0.0
    best_rss = np.inf
    best_freqs = np.empty(0, dtype=np.float64)

    robust_cands: list[tuple[int, FloatArray, FloatArray, FloatArray]] = []
    min_mad = np.inf
    min_seg = (
        min(max(6, min(n_win // 5, 8)), max(3, (n_win - 1) // 2))
        if eff_robust != "none"
        else 3
    )
    split_stop = (n_win - min_seg + 1) if eff_robust != "none" else (n_win - 2)

    for ind in range(min_seg, split_stop):
        j_arr = np.array([ind], dtype=np.intp)
        if season == "allssa":
            res_a = _allssa_core(
                t0,
                y_win,
                pw_win,
                omega,
                j_arr,
                alpha,
                "linear",
                False,
                1,
                max(25, 2 * int(omega.size)),
                parity_mode=parity_mode,
                compute_cov=False,
                robust=eff_robust,
                huber_k=huber_k,
                significance=season_significance,
            )
            rss = res_a.norm_res
            tr_c = res_a.trend_coeff
            freqs = res_a.frequencies
            res_vec = res_a.residuals
        elif season == "ols":
            res_l = _lssa_core(
                t0,
                y_win,
                pw_win,
                np.empty(0, dtype=np.float64),
                j_arr,
                alpha,
                "linear",
                False,
                omega,
                compute_cov=False,
                robust=eff_robust,
                huber_k=huber_k,
            )
            rss = res_l.norm_res
            tr_c = res_l.coeff[:4]
            freqs = omega
            res_vec = res_l.residuals
        else:
            res_l = _lssa_core(
                t0,
                y_win,
                pw_win,
                np.empty(0, dtype=np.float64),
                j_arr,
                alpha,
                "linear",
                False,
                np.empty(0, dtype=np.float64),
                compute_cov=False,
                robust=eff_robust,
                huber_k=huber_k,
            )
            rss = res_l.norm_res
            tr_c = res_l.coeff[:4]
            freqs = np.empty(0, dtype=np.float64)
            res_vec = res_l.residuals

        if eff_robust != "none":
            rw_base = whiten_vector(res_vec, pw_win)
            p_est = 4 + 2 * int(freqs.size)
            dof_fac = float(np.sqrt(n_win / max(1, n_win - p_est)))
            mad_j = float(dof_fac * 1.482602218505602 * np.median(np.abs(rw_base)))
            if mad_j < min_mad:
                min_mad = mad_j
            robust_cands.append((ind, rw_base, tr_c, freqs))
        elif rss < best_rss:
            best_rss = rss
            best_ind = ind
            best_mag = float((tr_c[1] - tr_c[0]) + (tr_c[3] - tr_c[2]) * t0[ind])
            best_dir = float(tr_c[3] - tr_c[2])
            best_freqs = freqs

    if eff_robust != "none":
        if not robust_cands:
            raise InsufficientDataError(f"Window of length {n_win} is too short for jump detection.")
        # Evaluate all candidate splits on a shared window scale so robust losses
        # are directly comparable across split locations without masking jumps.
        y_ref = float(np.sqrt(np.mean(whiten_vector(y_win - np.median(y_win), pw_win) ** 2)))
        s_win = min_mad if min_mad > 1e-14 * max(y_ref, 1e-300) else 0.0
        for ind, rw_base, tr_c, freqs in robust_cands:
            if s_win > 0.0:
                u_j = rw_base / s_win
                loss_j = float(
                    np.sum(tukey_rho(u_j) if eff_robust == "mm" else huber_rho(u_j, huber_k))
                )
            else:
                loss_j = float(np.dot(rw_base, rw_base))
            if loss_j < best_rss:
                best_rss = loss_j
                best_ind = ind
                best_mag = float((tr_c[1] - tr_c[0]) + (tr_c[3] - tr_c[2]) * t0[ind])
                best_dir = float(tr_c[3] - tr_c[2])
                best_freqs = freqs

    if best_ind < 0:
        raise InsufficientDataError(f"Window of length {n_win} is too short for jump detection.")

    return best_ind, best_mag, best_dir, best_rss, best_freqs


def _mc_maxf_null(
    t0: FloatArray,
    pw_win: PreparedWeights,
    test_freqs: FloatArray,
    j_lo: int,
    j_hi: int,
    n_mc: int,
    seed: int | Sequence[int],
) -> tuple[FloatArray, int, int] | None:
    """Monte Carlo draws of the null maximum jump F over splits ``j_lo <= j < j_hi``.

    Returns ``(max_f_draws, n_splits, dof2)`` or ``None`` when the design is degenerate.
    The draws depend only on the window design, weights, harmonics and seed, so they
    can be reused for several response vectors (e.g. multiple imputations).
    """
    n = int(t0.size)
    if j_hi <= j_lo:
        return None
    span = float(t0[-1] - t0[0]) if t0[-1] > t0[0] else 1.0
    u = (t0 - t0[0]) / span
    n_f = int(test_freqs.size)
    X0 = np.empty((n, 2 + 2 * n_f), dtype=np.float64)
    X0[:, 0] = 1.0
    X0[:, 1] = u
    if n_f > 0:
        ph = np.outer((2.0 * np.pi) * t0, test_freqs)
        X0[:, 2::2] = np.cos(ph)
        X0[:, 3::2] = np.sin(ph)
    dof2 = n - X0.shape[1] - 2
    if dof2 <= 0:
        return None
    q0, _ = np.linalg.qr(whiten_matrix(X0, pw_win), mode="reduced")
    splits = np.arange(j_lo, j_hi)
    n_j = int(splits.size)
    step = (np.arange(n)[:, None] >= splits[None, :]).astype(np.float64)
    zw = whiten_matrix(np.hstack([step, step * u[:, None]]), pw_win)
    zp = zw - q0 @ (q0.T @ zw)
    a = zp[:, :n_j]
    b = zp[:, n_j:]
    s11 = np.sum(a * a, axis=0)
    s22 = np.sum(b * b, axis=0)
    s12 = np.sum(a * b, axis=0)
    det = s11 * s22 - s12 * s12
    ok = (s11 > 0.0) & (s22 > 0.0) & (det > 1e-10 * s11 * s22)
    if not np.any(ok):
        return None
    a, b, s11, s22, s12, det = a[:, ok], b[:, ok], s11[ok], s22[ok], s12[ok], det[ok]
    rng = np.random.default_rng(seed)
    draws = np.empty(int(n_mc), dtype=np.float64)
    done = 0
    chunk = 256
    while done < n_mc:
        m = min(chunk, n_mc - done)
        e = rng.standard_normal((n, m))
        r = e - q0 @ (q0.T @ e)
        rss0 = np.sum(r * r, axis=0)
        g1 = a.T @ r
        g2 = b.T @ r
        red = (s22[:, None] * g1 * g1 - 2.0 * s12[:, None] * g1 * g2 + s11[:, None] * g2 * g2) / det[:, None]
        red = np.minimum(red, rss0[None, :] * (1.0 - 1e-12))
        f_null = (0.5 * red) / ((rss0[None, :] - red) / dof2)
        draws[done : done + m] = np.max(f_null, axis=0)
        done += m
    return draws, n_j, int(dof2)


def _best_jump_split(
    t0: FloatArray,
    y: FloatArray,
    pw_win: PreparedWeights,
    test_freqs: FloatArray,
    j_lo: int,
    j_hi: int,
) -> int | None:
    """Split ``j_lo <= j < j_hi`` maximising the jump F statistic (level + slope change)
    against a linear trend plus fixed harmonics, via 2-column Schur complements."""
    n = int(t0.size)
    if j_hi <= j_lo:
        return None
    span = float(t0[-1] - t0[0]) if t0[-1] > t0[0] else 1.0
    u = (t0 - t0[0]) / span
    n_f = int(test_freqs.size)
    X0 = np.empty((n, 2 + 2 * n_f), dtype=np.float64)
    X0[:, 0] = 1.0
    X0[:, 1] = u
    if n_f > 0:
        ph = np.outer((2.0 * np.pi) * t0, test_freqs)
        X0[:, 2::2] = np.cos(ph)
        X0[:, 3::2] = np.sin(ph)
    q0, _ = np.linalg.qr(whiten_matrix(X0, pw_win), mode="reduced")
    splits = np.arange(j_lo, j_hi)
    n_j = int(splits.size)
    step = (np.arange(n)[:, None] >= splits[None, :]).astype(np.float64)
    zw = whiten_matrix(np.hstack([step, step * u[:, None]]), pw_win)
    zp = zw - q0 @ (q0.T @ zw)
    a_m = zp[:, :n_j]
    b_m = zp[:, n_j:]
    s11 = np.sum(a_m * a_m, axis=0)
    s22 = np.sum(b_m * b_m, axis=0)
    s12 = np.sum(a_m * b_m, axis=0)
    det = s11 * s22 - s12 * s12
    ok = (s11 > 0.0) & (s22 > 0.0) & (det > 1e-10 * s11 * s22)
    if not np.any(ok):
        return None
    yw = whiten_vector(y, pw_win)
    r = yw - q0 @ (q0.T @ yw)
    g1 = a_m.T @ r
    g2 = b_m.T @ r
    red = np.where(ok, (s22 * g1 * g1 - 2.0 * s12 * g1 * g2 + s11 * g2 * g2) / np.where(ok, det, 1.0), -np.inf)
    return int(splits[int(np.argmax(red))])


def _mc_maxf_pvalue(
    t0: FloatArray,
    pw_win: PreparedWeights,
    test_freqs: FloatArray,
    f_obs: float,
    j_lo: int,
    j_hi: int,
    n_mc: int,
    seed: int | Sequence[int],
    cache: dict[Any, Any] | None = None,
) -> float:
    """Exact Monte Carlo p-value of an observed jump F statistic against the null
    distribution of the **maximum** F over all candidate splits ``j_lo <= j < j_hi``.

    Under Gaussian errors with the supplied weights, the split-wise jump F statistics
    (null: linear trend + fixed harmonics; alternative: adds a level and slope change at
    ``j``) are pivotal, so simulating standard-normal whitened errors yields the exact
    finite-sample null distribution of ``max_j F_j`` for this window design. This corrects
    the anti-conservative fixed-split F reference used for a data-chosen split
    (Andrews, 1993). ``cache`` (keyed by seed, design size and harmonics) reuses the
    null draws across response vectors on the same window.
    """
    if not np.isfinite(f_obs):
        return float("nan")
    key = (
        tuple(seed) if isinstance(seed, Sequence) else seed,
        int(t0.size), j_lo, j_hi, int(n_mc), test_freqs.tobytes(),
    )
    null = cache.get(key) if cache is not None else None
    if null is None:
        null = _mc_maxf_null(t0, pw_win, test_freqs, j_lo, j_hi, n_mc, seed)
        if cache is not None:
            cache[key] = null
    if null is None:
        return float("nan")
    draws, n_j, dof2 = null
    exceed = int(np.sum(draws >= f_obs))
    p_mc = float((1 + exceed) / (n_mc + 1))
    if exceed == 0:
        # Below Monte Carlo resolution: report the (valid) Bonferroni bound over the
        # scanned splits, capped at 1/(n_mc+1). The test is unchanged for any
        # alpha >= 1/(n_mc+1) and Bonferroni-valid below it.
        p_bonf = float(min(1.0, n_j * f_dist.sf(f_obs, 2, dof2)))
        return min(p_mc, p_bonf)
    return p_mc


def _jump_split_count(n_win: int, robust: str) -> int:
    """Number of candidate splits used by the Bonferroni corrections in the jump tests."""
    if robust != "none":
        min_seg = min(max(6, min(n_win // 5, 8)), max(3, (n_win - 1) // 2))
        return max(1, n_win - 2 * min_seg + 1)
    return max(1, n_win - 5)


def _nested_jump_ftests(
    t_win: FloatArray,
    y_win: FloatArray,
    pw_win: PreparedWeights,
    local_jump_idx: int,
    freqs: FloatArray,
    *,
    season: str = "ols",
    omega: FloatArray | None = None,
    alpha: float = 0.01,
    bonferroni: bool = False,
    parity_mode: bool = False,
    robust: RobustMethod = "none",
    huber_k: float = 1.345,
    calibration: JumpCalibration | None = None,
    n_mc: int = 999,
    mc_seed: int | Sequence[int] = 0,
    season_significance: str = "per_frequency",
    mc_cache: dict[Any, Any] | None = None,
    raw_step: bool = False,
) -> tuple[float, float]:
    """Nested partial F-tests comparing discontinuous 2-piece vs null & continuous hinge.

    Returns ``(jump_pvalue, step_pvalue)``:
    - ``jump_pvalue`` (2 df): 2-piece discontinuous linear vs 1-piece linear trend.
    - ``step_pvalue`` (1 df): 2-piece discontinuous linear vs continuous 2-piece
      hinge ``[1, t0, (t0 - t_j)_+]``, isolating whether a genuine vertical step
      discontinuity exists beyond a smooth slope bend.

    When ``season == "allssa"``, harmonics are selected strictly under the
    no-jump null model first so adaptive frequency selection cannot be mined
    jointly with the candidate split to artificially deflate ``rss1`` (fixing
    post-selection ALLSSA Type I error inflation).

    When ``robust in ("huber", "mm")`` (and ``parity_mode=False``), uses the
    Schrader–Hettmansperger (1980) robust drop-in-dispersion F-test at the
    shared full-model scale so outlier spikes do not inflate false positives.

    ``calibration`` controls the split-selection correction of ``jump_pvalue``:
    ``"montecarlo"`` (exact Monte Carlo null of the max-over-splits F statistic;
    non-robust fits only), ``"bonferroni"`` (multiply raw p-values by the number of
    candidate splits), or ``"none"`` (raw fixed-split p-values). ``bonferroni=True`` is
    the legacy spelling of ``calibration="bonferroni"``.
    """
    from ._linalg import solve_weighted_ols

    eff_robust: RobustMethod = "none" if parity_mode else robust
    n_win = int(t_win.size)
    t0 = t_win - t_win[0]

    # For adaptive ALLSSA, select harmonics under the no-jump null model so the
    # nested F-test conditions on a null-selected subspace without harmonic mining.
    if season == "allssa" and omega is not None and omega.size > 0 and n_win >= 5:
        null_allssa = _allssa_core(
            t0,
            y_win,
            pw_win,
            omega,
            np.empty(0, dtype=np.intp),
            alpha,
            "linear",
            False,
            1,
            max(25, 2 * int(omega.size)),
            parity_mode=parity_mode,
            compute_cov=False,
            robust=eff_robust,
            huber_k=huber_k,
            significance=season_significance,
        )
        test_freqs = null_allssa.frequencies
    else:
        test_freqs = freqs

    n_f = int(test_freqs.size)
    nc_full = 4 + 2 * n_f
    dof2 = n_win - nc_full
    if dof2 <= 0:
        return float("nan"), float("nan")

    min_seg = (
        min(max(6, min(n_win // 5, 8)), max(3, (n_win - 1) // 2))
        if eff_robust != "none"
        else 3
    )
    calib: JumpCalibration = calibration if calibration is not None else (
        "bonferroni" if bonferroni else "none"
    )
    if calib == "montecarlo" and eff_robust != "none":
        calib = "bonferroni"
    n_cands = max(1, (n_win - 2 * min_seg + 1) if eff_robust != "none" else (n_win - 5))
    mult = float(n_cands) if calib == "bonferroni" else 1.0
    if float(np.ptp(y_win)) <= 1e-13 * abs(float(np.mean(y_win))):
        return 1.0, 1.0

    # Continuous hinge basis [1, t0/s, max(0, t0 - t0[k])/s, cos/sin(test_freqs)]
    s_t = float(t0[-1] - t0[0]) if t0[-1] > t0[0] else 1.0
    D_hinge = np.empty((n_win, 3 + 2 * n_f), dtype=np.float64)
    D_hinge[:, 0] = 1.0
    D_hinge[:, 1] = t0 / s_t
    if n_f > 0:
        phases = np.outer((2.0 * np.pi) * t0, test_freqs)
        D_hinge[:, 3::2] = np.cos(phases)
        D_hinge[:, 4::2] = np.sin(phases)

    if eff_robust != "none":
        D_full, cs_full = build_design_matrix(
            t0, "linear", np.array([local_jump_idx], dtype=np.intp), False, test_freqs
        )
        D_null, cs_null = build_design_matrix(
            t0, "linear", np.empty(0, dtype=np.intp), False, test_freqs
        )
        _, p_jump_raw, res_full, _ = huber_nested_ftest(
            D_null,
            D_full,
            y_win,
            pw_win,
            method=eff_robust,
            k=huber_k,
            col_scales_null=cs_null,
            col_scales_full=cs_full,
        )
        if np.isnan(p_jump_raw):
            return float("nan"), float("nan")
        knot_stop = n_win - min_seg + 1
        if res_full.scale <= 0.0 or res_full.lambda_scale <= 0.0:
            # Noiseless jump regime
            rss_h = np.inf
            for k_knot in range(min_seg, knot_stop):
                D_hinge[:, 2] = np.maximum(0.0, t0 - t0[k_knot]) / s_t
                rss_k = solve_weighted_ols(D_hinge, y_win, pw_win, compute_cov=False).norm_res
                if rss_k < rss_h:
                    rss_h = rss_k
            tol_r = 1e-26 * max(rss_h, res_full.ols.y_w_norm_sq, float(np.finfo(np.float64).tiny))
            p_step_raw = 0.0 if rss_h > tol_r else 1.0
        else:
            loss_hinge = np.inf
            for k_knot in range(min_seg, knot_stop):
                D_hinge[:, 2] = np.maximum(0.0, t0 - t0[k_knot]) / s_t
                h_k = solve_robust_irls(
                    D_hinge,
                    y_win,
                    pw_win,
                    method=eff_robust,
                    k=huber_k,
                    fixed_scale=res_full.scale,
                    compute_cov=False,
                )
                if h_k.loss < loss_hinge:
                    loss_hinge = h_k.loss
            f_step = max(0.0, 2.0 * (loss_hinge - res_full.loss)) / res_full.lambda_scale
            p_step_raw = float(f_dist.sf(f_step, 1, dof2))
        return float(min(1.0, mult * p_jump_raw)), float(min(1.0, mult * p_step_raw))

    res_full_ols = _lssa_core(
        t0,
        y_win,
        pw_win,
        np.empty(0, dtype=np.float64),
        np.array([local_jump_idx], dtype=np.intp),
        0.05,
        "linear",
        False,
        test_freqs,
        compute_cov=False,
    )
    res_null_ols = _lssa_core(
        t0,
        y_win,
        pw_win,
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.intp),
        0.05,
        "linear",
        False,
        test_freqs,
        compute_cov=False,
    )

    rss_hinge = np.inf
    for k_knot in range(3, n_win - 2):
        D_hinge[:, 2] = np.maximum(0.0, t0 - t0[k_knot]) / s_t
        rss_k = solve_weighted_ols(D_hinge, y_win, pw_win, compute_cov=False).norm_res
        if rss_k < rss_hinge:
            rss_hinge = rss_k

    rss1 = res_full_ols.norm_res
    rss0 = res_null_ols.norm_res

    # Scale-invariant noiseless guard (handles scaled noiseless jumps down to 1e-13 cleanly).
    tol = 1e-26 * max(rss0, rss_hinge, float(np.finfo(np.float64).tiny))
    if rss1 <= tol:
        p_jump = 0.0 if rss0 > tol else 1.0
        p_step = 0.0 if rss_hinge > tol else 1.0
        return p_jump, p_step

    den = rss1 / dof2
    f_jump = max(0.0, (rss0 - rss1) / 2.0) / den
    f_step = max(0.0, rss_hinge - rss1) / den
    if calib == "montecarlo":
        p_jump = _mc_maxf_pvalue(
            t0, pw_win, test_freqs, f_jump, 3, n_win - 2, n_mc, mc_seed, cache=mc_cache
        )
        # The step test (jump vs best continuous hinge) is also evaluated at the
        # data-chosen split; its null (an unknown-knot hinge) is not pivotal, so use the
        # analytical Bonferroni correction over the scanned splits.
        step_mult = 1.0 if raw_step else float(n_cands)
        p_step = float(min(1.0, step_mult * float(f_dist.sf(f_step, 1, dof2))))
        return float(p_jump), p_step
    p_jump = float(min(1.0, mult * float(f_dist.sf(f_jump, 2, dof2))))
    p_step = float(min(1.0, mult * float(f_dist.sf(f_step, 1, dof2))))
    return p_jump, p_step


def _holm_adjust_candidates(
    tested: list[tuple[int, int, float, float, float, float, float, WindowCandidate]],
) -> list[tuple[int, int, float, float, float, float, float, WindowCandidate]]:
    """Holm (1979) step-down adjustment of ``jump_pvalue`` and ``step_pvalue`` across
    candidate locations."""

    def _holm(raw: FloatArray) -> FloatArray:
        m = int(raw.size)
        order = np.argsort(raw)
        adj = np.empty(m, dtype=np.float64)
        running = 0.0
        for rank, idx in enumerate(order):
            running = max(running, min(1.0, (m - rank) * raw[idx]))
            adj[idx] = running
        return adj

    pj = np.array([np.nan_to_num(it[5], nan=1.0) for it in tested], dtype=np.float64)
    ps = np.array([np.nan_to_num(it[6], nan=1.0) for it in tested], dtype=np.float64)
    adj_j = _holm(pj)
    adj_s = _holm(ps)
    out = []
    for k, it in enumerate(tested):
        pj_new = float(adj_j[k]) if np.isfinite(it[5]) else it[5]
        ps_new = float(adj_s[k]) if np.isfinite(it[6]) else it[6]
        out.append((it[0], it[1], it[2], it[3], it[4], pj_new, ps_new, it[7]))
    return out


def _run_just_jumps_array(
    t_valid: FloatArray,
    y_valid: FloatArray,
    pw: PreparedWeights,
    windows: list[tuple[int, int]],
    size_for_bucket: float,
    season: str,
    omega: FloatArray,
    alpha: float,
    mag_th: float,
    dir_th: float,
    jump_th: float,
    jump_alpha: float | None,
    *,
    step_alpha: float | None = None,
    bonferroni_window: bool = False,
    jump_calibration: JumpCalibration | None = None,
    n_mc: int = 999,
    season_significance: str = "per_frequency",
    parity_mode: bool,
    robust: RobustMethod = "none",
    huber_k: float = 1.345,
    provenance: CensoringProvenance | None = None,
    censor_cfg: CensoredConfig | None = None,
    rng: np.random.Generator | None = None,
    mi_spec: tuple[int, str] | None = None,
    mi_counter: list[int] | None = None,
) -> tuple[
    list[tuple[int, int, float, float, float, float, float, WindowCandidate]],
    list[WindowCandidate],
    int,
]:
    """Core sliding-window JUST jump search + aggregation + merging.

    ``mi_spec = (n_imputations, dist)`` pools each candidate's jump and step p-values
    (Licht–Rubin) over ``n_imputations`` proper imputations of the candidate's window
    drawn from a censored regression of the window's null model (trend plus
    harmonics), before the Holm adjustment. (Imputing with the candidate jump in the
    model builds the data-selected location into every imputation and was
    anti-conservative in simulation.) Candidate locations come from the primary
    imputation ``y_valid``. ``mi_counter`` (``[n_pooled, n_fallback]``) records how
    many candidates were pooled and how many fell back to the primary imputation.
    """
    candidates: list[WindowCandidate] = []
    window_fallbacks = 0
    window_y_cache: dict[tuple[int, int], FloatArray] = {}

    use_local_censor = (
        provenance is not None
        and censor_cfg is not None
        and censor_cfg.scope == "windowed"
        and provenance.n_censored > 0
        and rng is not None
    )

    for w_start, w_stop in windows:
        t_w = t_valid[w_start:w_stop]
        if use_local_censor:
            assert provenance is not None and censor_cfg is not None and rng is not None
            y_w, fb = impute_window_local(provenance, censor_cfg, w_start, w_stop, rng)
            window_y_cache[(w_start, w_stop)] = y_w
            if fb:
                window_fallbacks += 1
        else:
            y_w = y_valid[w_start:w_stop]
        pw_w = pw.slice_range(w_start, w_stop)
        loc_ind, mag, direction, rss, freqs = _detect_jump_in_window(
            t_w,
            y_w,
            pw_w,
            season,
            omega,
            alpha,
            parity_mode=parity_mode,
            robust=robust,
            huber_k=huber_k,
            season_significance=season_significance,
        )
        n_w = w_stop - w_start
        dist_center = abs(loc_ind - (n_w - 1) / 2.0)
        candidates.append(
            WindowCandidate(
                window_start=w_start,
                window_stop=w_stop,
                valid_index=w_start + loc_ind,
                index=w_start + loc_ind,  # remapped to orig by caller
                magnitude=mag,
                direction=direction,
                center_distance=dist_center,
                norm_res=rss,
                selected_freqs=freqs,
            )
        )

    # Sort by jump index ascending and aggregate occurrences
    candidates_sorted = sorted(candidates, key=lambda c: c.valid_index)
    counts = Counter(c.valid_index for c in candidates_sorted)

    # Group per unique valid_index; pick representative with smallest distance to window center
    grouped: list[list[Any]] = []
    pos = 0
    for j_idx in sorted(counts):
        occ = counts[j_idx]
        group_cands = candidates_sorted[pos : pos + occ]
        pos += occ
        best_c = min(group_cands, key=lambda c: c.center_distance)
        mag_val = round(best_c.magnitude, 2) if parity_mode else best_c.magnitude
        dir_val = round(best_c.direction, 2) if parity_mode else best_c.direction
        dis_raw = float(int(best_c.center_distance)) if parity_mode else best_c.center_distance
        bucket = int(int(best_c.center_distance) / (size_for_bucket / 6.0))
        grouped.append([j_idx, occ, mag_val, dir_val, bucket, dis_raw, best_c])

    # Greedy merge adjacent jumps closer than jump_th
    r = 1
    while r < len(grouped):
        if t_valid[grouped[r][0]] - t_valid[grouped[r - 1][0]] < jump_th:
            pair = [grouped[r - 1], grouped[r]]
            pair.sort(key=lambda item: (-item[1], item[4], -abs(item[2])))
            grouped[r - 1] = pair[0]
            del grouped[r]
        else:
            r += 1

    # Compute nested partial F-test p-values & apply magnitude/direction/statistical filters
    calib: JumpCalibration = jump_calibration if jump_calibration is not None else (
        "bonferroni" if bonferroni_window else "none"
    )
    tested: list[tuple[int, int, float, float, float, float, float, WindowCandidate]] = []
    mc_cache: dict[Any, Any] = {}
    for item in grouped:
        j_idx, occ, mag_val, dir_val, _bucket, dis_raw, best_c = item
        if abs(mag_val) < mag_th and abs(dir_val) < dir_th:
            continue
        t_w = t_valid[best_c.window_start : best_c.window_stop]
        y_w = window_y_cache.get(
            (best_c.window_start, best_c.window_stop),
            y_valid[best_c.window_start : best_c.window_stop],
        )
        pw_w = pw.slice_range(best_c.window_start, best_c.window_stop)
        p_jump, p_step = _nested_jump_ftests(
            t_w,
            y_w,
            pw_w,
            j_idx - best_c.window_start,
            best_c.selected_freqs,
            season=season,
            omega=omega,
            alpha=alpha,
            parity_mode=parity_mode,
            robust=robust,
            huber_k=huber_k,
            calibration=calib,
            n_mc=n_mc,
            mc_seed=(20261125, int(best_c.window_start), int(best_c.window_stop)),
            season_significance=season_significance,
            mc_cache=mc_cache,
        )
        ws_c, we_c = int(best_c.window_start), int(best_c.window_stop)
        if (
            mi_spec is not None
            and provenance is not None
            and pw_w.kind != "matrix"
            and bool(np.any(provenance.status_codes[ws_c:we_c] != 0))
        ):
            n_imp, mi_dist = mi_spec
            n_w_c = int(t_w.size)
            j_cand = j_idx - ws_c
            u_w = (t_w - t_w[0]) / (float(t_w[-1] - t_w[0]) if t_w[-1] > t_w[0] else 1.0)
            fr = best_c.selected_freqs
            imps = None
            # Imputation model: the window's null model (trend + harmonics); harmonics are
            # dropped when the window has too few detected values.
            for fr_imp in (fr, fr[:1], fr[:0]) if fr.size > 1 else (fr, fr[:0]):
                cols = [np.ones(n_w_c), u_w]
                for f_h in fr_imp:
                    cols += [np.cos(2.0 * np.pi * f_h * t_w), np.sin(2.0 * np.pi * f_h * t_w)]
                imps = window_congenial_imputations(
                    provenance.original_values[ws_c:we_c],
                    provenance.status_codes[ws_c:we_c],
                    provenance.original_left[ws_c:we_c] if provenance.original_left is not None else None,
                    provenance.original_right[ws_c:we_c] if provenance.original_right is not None else None,
                    np.column_stack(cols),
                    mi_dist,
                    n_imp,
                    (20261126, ws_c, we_c),
                    weights=pw_w.vector if pw_w.kind == "vector" else None,
                )
                if imps is not None:
                    break
            if imps is None:
                if mi_counter is not None:
                    mi_counter[1] += 1
            else:
                if mi_counter is not None:
                    mi_counter[0] += 1
                # Jump test: pool calibrated Monte Carlo p-values (each imputation at its
                # own best split, the statistic the max-over-splits null describes), or
                # pool fixed-split p-values and apply the Bonferroni split count. Step
                # test: always the latter (Bonferroni-corrected in both calibrations).
                n_sp = _jump_split_count(n_w_c, "none" if parity_mode else robust)
                mc_jump = calib == "montecarlo" and robust == "none"
                pj: list[float] = []
                ps: list[float] = []
                for y_m_w in imps:
                    j_loc = j_cand
                    if robust == "none":
                        j_best = _best_jump_split(t_w - t_w[0], y_m_w, pw_w, fr, 3, n_w_c - 2)
                        if j_best is not None:
                            j_loc = j_best
                    pj_m, ps_m = _nested_jump_ftests(
                        t_w,
                        y_m_w,
                        pw_w,
                        j_loc,
                        fr,
                        season=season,
                        omega=omega,
                        alpha=alpha,
                        parity_mode=parity_mode,
                        robust=robust,
                        huber_k=huber_k,
                        calibration=calib if mc_jump else "none",
                        n_mc=n_mc,
                        mc_seed=(20261125, ws_c, we_c),
                        season_significance=season_significance,
                        mc_cache=mc_cache,
                        raw_step=True,
                    )
                    pj.append(pj_m)
                    ps.append(ps_m)
                pj_pool = pool_pvalues(np.asarray(pj, dtype=np.float64))
                p_jump = pj_pool if mc_jump else float(min(1.0, n_sp * pj_pool))
                p_step = float(min(1.0, n_sp * pool_pvalues(np.asarray(ps, dtype=np.float64))))
        tested.append((j_idx, occ, mag_val, dir_val, dis_raw, p_jump, p_step, best_c))

    # Family-wise control across the distinct candidate locations (Holm step-down) when
    # a calibrated jump test is requested.
    if calib != "none" and tested:
        tested = _holm_adjust_candidates(tested)

    accepted: list[tuple[int, int, float, float, float, float, float, WindowCandidate]] = []
    for cand in tested:
        p_jump, p_step = cand[5], cand[6]
        if jump_alpha is not None and (np.isnan(p_jump) or p_jump >= jump_alpha):
            continue
        if step_alpha is not None and (np.isnan(p_step) or p_step >= step_alpha):
            continue
        accepted.append(cand)

    return accepted, candidates_sorted, window_fallbacks


def just_jumps(
    t: TimeLike,
    y: ValueLike,
    *,
    weights: WeightLike | None = None,
    size: int | None = None,
    step: int | None = None,
    window_mode: WindowMode = "count",
    window_duration: float | None = None,
    step_duration: float | None = None,
    min_window_obs: int | None = None,
    season: SeasonMode = "allssa",
    omega: ArrayLike | None = None,
    alpha: float = 0.01,
    mag_th: float | None = None,
    dir_th: float | None = None,
    jump_th: float = 0.75,
    jump_alpha: float | None | Literal["auto"] = "auto",
    step_alpha: float | None = None,
    bonferroni_window: bool = False,
    jump_calibration: JumpCalibration | None = None,
    n_mc: int = 999,
    season_significance: SeasonSignificance = "sidak",
    robust: RobustMethod = "none",
    huber_k: float = 1.345,
    censor_status: ArrayLike | None = None,
    censoring: CensoredConfig | None = None,
    parity_mode: bool = False,
) -> JustJumpResult:
    """Detect abrupt jumps and trend slope shifts via JUST (Ghaderpour & Vujadinovic, 2020).

    Slides a window across an unevenly sampled time series with **zero
    interpolation**, simultaneously estimating a 2-piece discontinuous linear
    trend and seasonal harmonics (via ALLSSA, fixed OLS, or trend-only) to
    locate abrupt jumps, their step magnitudes, slope changes, and screening
    nested F-test p-values.

    Parameters
    ----------
    t:
        1D monotonically ascending timestamps (numeric or datetime).
    y:
        1D observations (numeric, censored string markers like ``'<0.05'``, or
        ``(N, 2)`` intervals). ``NaN`` entries are dropped without interpolation.
    weights:
        Optional scalar, 1D positive weight vector (O(n) memory), or 2D SPD
        inverse-covariance matrix ``P = C_f^{-1}``.
    size, step:
        Observation-count window size and step (used when ``window_mode='count'``).
        Default to ``3*M`` and ``M`` where ``M`` is the average sampling rate.
    window_mode:
        ``'count'`` (default, Ghaderpour parity) or ``'duration'`` (fixed calendar
        span; recommended when sampling frequency changes sharply over time).
    window_duration, step_duration, min_window_obs:
        Parameters for ``window_mode='duration'``.
    season:
        Seasonal component model: ``'allssa'`` (default, anti-leakage), ``'ols'``
        (fixed frequencies), or ``'none'`` (trend only).
    omega:
        Cyclic frequency set. Defaults to ``[0.8, 1.0, ..., 3.8]`` for ALLSSA and
        ``[1, 2, 3, 4]`` for OLS.
    alpha:
        Significance level for ALLSSA seasonal peak detection (default ``0.01``).
    mag_th, dir_th:
        Minimum absolute jump magnitude or slope-change thresholds (keeps jumps
        satisfying ``|mag| >= mag_th or |dir| >= dir_th``). Default ``None``
        resolves to ``0.0`` (no effect-size filter; selection is controlled by
        ``jump_alpha``), or to Ghaderpour's NDVI-scale ``0.05`` / ``0.01`` when
        ``parity_mode=True``.
    jump_th:
        Minimum time separation between distinct jumps (default ``0.75``).
    jump_alpha:
        Family-wise significance level for reported jumps. ``"auto"`` (default)
        resolves to ``0.05``, or to ``None`` (no statistical filter, Ghaderpour
        parity) when ``parity_mode=True``. Pass ``None`` explicitly to disable the
        filter; ``jump_pvalue`` is always computed and stored on every
        :class:`JumpEvent`.
    step_alpha:
        Optional additional cutoff on ``step_pvalue`` (discontinuous jump vs.
        best continuous hinge), isolating vertical steps from smooth bends.
    jump_calibration:
        Correction of ``jump_pvalue`` for the data-driven choice of split inside
        each window: ``"montecarlo"`` (default for non-robust fits; exact Monte
        Carlo null of the max-over-splits F statistic, ``n_mc`` draws),
        ``"bonferroni"`` (analytical, conservative; default for robust fits), or
        ``"none"`` (raw fixed-split screening p-values, forced by
        ``parity_mode=True``). With ``"montecarlo"`` or ``"bonferroni"``, p-values
        are additionally Holm-adjusted across the distinct candidate locations so
        ``jump_alpha`` bounds the probability of reporting any false jump.
    bonferroni_window:
        Legacy alias for ``jump_calibration="bonferroni"``.
    n_mc:
        Monte Carlo draws for ``jump_calibration="montecarlo"`` (default ``999``).
    season_significance:
        Acceptance rule for harmonics selected by ALLSSA inside each window when
        ``season='allssa'``: ``"sidak"`` (default; family-wise over the searched
        frequencies, analytical), ``"montecarlo"`` (exact but slow inside the split
        scan), or ``"per_frequency"`` (published rule, forced by ``parity_mode``; it
        admits spurious harmonics in about 20% of noise windows).
    robust:
        ``'none'`` (default WLS/OLS), ``'huber'`` (convex Huber IRLS window
        fits + Schrader–Hettmansperger robust nested F-tests), or ``'mm'``
        (Huber-initialized Tukey biweight MM-type estimation; not Yohai's high-breakdown
        S-started MM, so breakdown is design-dependent; recommended
        for extreme isolated spikes without masking genuine sustained jumps).
        Forced to ``'none'`` when ``parity_mode=True``.
    huber_k:
        Huber tuning constant (default ``1.345``).
    censor_status, censoring:
        Optional censoring indicator array and :class:`CensoredConfig`.
    parity_mode:
        If ``True``, matches legacy upstream rounding and window-termination
        semantics for Tier 1 numerical parity verification.
    """
    season_l = season.lower()
    if season_l not in ("allssa", "ols", "none"):
        raise ValueError(f"Unknown `season` {season!r}. Options: 'allssa', 'ols', 'none'.")
    if window_mode not in ("count", "duration"):
        raise ValueError(f"Unknown `window_mode` {window_mode!r}. Options: 'count', 'duration'.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"`alpha` must lie in (0, 1); got {alpha}.")
    eff_jump_alpha: float | None
    if isinstance(jump_alpha, str):
        if jump_alpha != "auto":
            raise ValueError(f"`jump_alpha` must be a float in (0, 1), None, or 'auto'; got {jump_alpha!r}.")
        eff_jump_alpha = None if parity_mode else 0.05
    else:
        eff_jump_alpha = jump_alpha
    if eff_jump_alpha is not None and not (0.0 < eff_jump_alpha < 1.0):
        raise ValueError(f"`jump_alpha` must lie in (0, 1); got {eff_jump_alpha}.")
    if step_alpha is not None and not (0.0 < step_alpha < 1.0):
        raise ValueError(f"`step_alpha` must lie in (0, 1); got {step_alpha}.")
    mag_th = (0.05 if parity_mode else 0.0) if mag_th is None else float(mag_th)
    dir_th = (0.01 if parity_mode else 0.0) if dir_th is None else float(dir_th)
    if mag_th < 0.0 or dir_th < 0.0 or jump_th < 0.0:
        raise ValueError("`mag_th`, `dir_th`, and `jump_th` must be non-negative.")
    if jump_calibration is not None and jump_calibration not in ("montecarlo", "bonferroni", "none"):
        raise ValueError(
            f"Unknown `jump_calibration` {jump_calibration!r}. Options: 'montecarlo', 'bonferroni', 'none'."
        )
    if int(n_mc) < 99:
        raise ValueError(f"`n_mc` must be >= 99; got {n_mc}.")
    eff_robust: RobustMethod = "none" if parity_mode else validate_robust_args(robust, huber_k)
    eff_season_sig = _resolve_season_significance(season_significance, parity_mode)
    calib_warn: WarningItem | None = None
    if parity_mode:
        eff_calib: JumpCalibration = "none"
    elif jump_calibration is not None:
        eff_calib = jump_calibration
    elif bonferroni_window:
        eff_calib = "bonferroni"
    else:
        eff_calib = "bonferroni" if eff_robust != "none" else "montecarlo"
    if eff_calib == "montecarlo" and eff_robust != "none":
        eff_calib = "bonferroni"
        calib_warn = WarningItem(
            code="CALIBRATION_FALLBACK",
            severity="info",
            message="Monte Carlo jump calibration is available for non-robust fits only; used Bonferroni.",
        )

    prep = prepare_series_with_censoring(t, y, censor_status=censor_status, censoring=censoring)
    series = prep.series
    prov = prep.provenance
    cfg = prep.config

    pw_full = prepare_weights(weights, series.n_input)
    pw = pw_full.subset(series.valid_mask) if series.n_dropped_nan else pw_full
    if prep.weight_factors is not None:
        pw = pw.scale_entries(prep.weight_factors)

    om_arr = (
        np.sort(np.asarray(omega, dtype=np.float64).ravel())
        if omega is not None and len(omega) > 0
        else _default_just_omega(season_l)
    )

    warnings_list: list[WarningItem] = list(prep.warnings)
    warnings_list.extend(time_unit_warning(series.times_numeric, series.is_datetime))
    if calib_warn is not None:
        warnings_list.append(calib_warn)
    if not parity_mode and eff_jump_alpha is None:
        warnings_list.append(
            WarningItem(
                code="UNCALIBRATED_JUMP_SELECTION",
                severity="warning",
                message=(
                    "jump_alpha=None: jumps are selected by magnitude/direction thresholds only, "
                    "with no false-positive control."
                ),
                suggestion="Use the default jump_alpha (0.05) for family-wise calibrated selection.",
            )
        )
    elif not parity_mode and eff_calib == "none":
        warnings_list.append(
            WarningItem(
                code="UNCALIBRATED_JUMP_PVALUES",
                severity="warning",
                message=(
                    "jump_calibration='none': jump p-values are raw fixed-split screening p-values at "
                    "data-chosen splits and are strongly anti-conservative."
                ),
                suggestion="Use jump_calibration='montecarlo' (default) or 'bonferroni'.",
            )
        )

    # Scale-mismatch diagnostic for default NDVI-tuned mag_th
    iqr = float(np.percentile(series.values, 75) - np.percentile(series.values, 25))
    if (
        not parity_mode
        and eff_jump_alpha is None
        and mag_th == 0.05
        and iqr > 0.0
        and (0.01 * iqr > 0.05 or 2.0 * iqr < 0.05)
    ):
        warnings_list.append(
                WarningItem(
                    code="THRESHOLD_SCALE_MISMATCH",
                    severity="caution",
                    message=(
                        f"Default mag_th=0.05 may be poorly scaled for this series "
                        f"(interquartile range ≈ {iqr:.4g})."
                    ),
                    suggestion=(
                        f"Consider setting mag_th≈{0.25 * iqr:.3g} (domain-meaningful jump size) "
                        "or using the calibrated default `jump_alpha=0.05`."
                    ),
                )
            )

    if window_mode == "count":
        windows, win_size, _win_step, _rate = _build_count_windows(
            series.n_valid, series.times_numeric, size, step, parity_mode=parity_mode
        )
        size_for_bucket = float(win_size)
        if not parity_mode and series.sampling_rate_ratio > 2.0:
            warnings_list.append(
                WarningItem(
                    code="VARIABLE_SAMPLING_RATE",
                    severity="caution",
                    message=(
                        f"Sampling density varies by ~{series.sampling_rate_ratio:.1f}x across "
                        "the record while using fixed observation-count windows."
                    ),
                    suggestion=(
                        "Consider `window_mode='duration'` (e.g., `window_duration=3.0`) so "
                        "each window spans a consistent calendar duration."
                    ),
                )
            )
    else:
        min_obs_default = 4 + (2 * int(om_arr.size) if season_l == "ols" else 0) + 5
        min_obs = int(min_window_obs) if min_window_obs is not None else max(12, min_obs_default)
        windows, _dur, skipped = _build_duration_windows(
            series.times_numeric, window_duration, step_duration, min_obs
        )
        size_for_bucket = float(np.median([b - a for a, b in windows]))
        if skipped > 0:
            warnings_list.append(
                WarningItem(
                    code="SPARSE_WINDOW_SKIPPED",
                    severity="info",
                    message=(
                        f"Skipped {skipped} duration window(s) containing fewer than "
                        f"{min_obs} observations."
                    ),
                )
            )

    sw_warn = _short_seasonal_window_warning(series.times_numeric, windows, om_arr, season_l, eff_season_sig)
    if sw_warn is not None:
        warnings_list.append(sw_warn)

    if cfg is not None and isinstance(cfg.random_state, np.random.Generator):
        rng: np.random.Generator = cfg.random_state
    else:
        rng = np.random.default_rng(cfg.random_state if cfg is not None else 0)

    # Multiple imputation of censored values: each candidate's p-values are pooled over
    # `detection_imputations` congenial imputations of its window (see _run_just_jumps_array).
    mi_spec: tuple[int, str] | None = None
    if (
        not parity_mode
        and cfg is not None
        and prov is not None
        and prov.n_censored > 0
        and cfg.detection_imputations > 1
        and eff_calib != "none"
    ):
        mi_skip_reason: str | None = None
        if cfg.scope == "windowed":
            mi_skip_reason = "scope='windowed' uses local marginal imputation per window"
        elif pw.kind == "matrix":
            mi_skip_reason = "2D weight matrices are not supported by censored-regression MI"
        else:
            mi_dist = resolve_imputation_dist(prov, cfg)
            if mi_dist is None:
                mi_skip_reason = f"dist={cfg.dist!r} is not supported by censored-regression MI"
            else:
                mi_spec = (int(cfg.detection_imputations), mi_dist)
        if mi_skip_reason is not None:
            warnings_list.append(
                WarningItem(
                    code="MI_DETECTION_FALLBACK",
                    severity="caution",
                    message=(
                        f"`detection_imputations={cfg.detection_imputations}` was skipped because "
                        f"{mi_skip_reason}; jump p-values condition on the single primary imputation."
                    ),
                )
            )
    mi_counter = [0, 0]

    accepted, all_cands_valid, win_fb = _run_just_jumps_array(
        series.times_numeric,
        series.values,
        pw,
        windows,
        size_for_bucket,
        season_l,
        om_arr,
        alpha,
        mag_th,
        dir_th,
        jump_th,
        eff_jump_alpha,
        step_alpha=step_alpha,
        jump_calibration=eff_calib,
        n_mc=int(n_mc),
        season_significance=eff_season_sig,
        parity_mode=parity_mode,
        robust=eff_robust,
        huber_k=huber_k,
        provenance=prov,
        censor_cfg=cfg,
        rng=rng,
        mi_spec=mi_spec,
        mi_counter=mi_counter,
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

    # Optional stability sweep across censoring realizations
    stability_counts: dict[int, int] = {item[0]: 0 for item in accepted}
    n_stab = cfg.n_stability_draws if (cfg is not None and prov is not None and prov.n_censored > 0) else 0
    if n_stab > 0 and accepted and prov is not None and cfg is not None:
        from .censored import _spawn_stability_rngs

        dt_med = float(np.median(np.diff(series.times_numeric))) if series.n_valid > 1 else 0.1
        match_tol = max(0.35, 2.0 * dt_med)
        stab_fb = [0]
        draw_series = generate_stability_draws(prov, cfg, series.times_numeric, stab_fb)
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
        win_rngs = _spawn_stability_rngs(cfg.random_state, len(draw_series)) if cfg.scope == "windowed" else None
        for idx_d, y_draw in enumerate(draw_series):
            draw_cfg = (
                CensoredConfig(
                    method=cfg.method,
                    censoring_type=cfg.censoring_type,
                    dist=cfg.dist,
                    impute_type="stochastic",
                    scope="windowed",
                    clamp=cfg.clamp,
                    ndimpute_kwargs=cfg.ndimpute_kwargs,
                )
                if cfg.scope == "windowed"
                else None
            )
            acc_d, _, _ = _run_just_jumps_array(
                series.times_numeric,
                y_draw,
                pw,
                windows,
                size_for_bucket,
                season_l,
                om_arr,
                alpha,
                mag_th,
                dir_th,
                jump_th,
                eff_jump_alpha,
                step_alpha=step_alpha,
                jump_calibration=eff_calib,
                n_mc=int(n_mc),
                season_significance=eff_season_sig,
                parity_mode=parity_mode,
                robust=eff_robust,
                huber_k=huber_k,
                provenance=prov if cfg.scope == "windowed" else None,
                censor_cfg=draw_cfg,
                rng=win_rngs[idx_d] if win_rngs is not None else None,
            )
            draw_times = [float(series.times_numeric[it[0]]) for it in acc_d]
            for item in accepted:
                t_ref = float(series.times_numeric[item[0]])
                if any(abs(td - t_ref) <= match_tol for td in draw_times):
                    stability_counts[item[0]] += 1

    # Assemble JumpEvent objects with original-index mapping, censoring & DL-shift audits
    events: list[JumpEvent] = []
    for j_valid, occ, mag_val, dir_val, dis_raw, pval, p_step, best_c in accepted:
        orig_idx = int(series.valid_to_orig[j_valid])
        t_yr = float(series.times_numeric[j_valid])
        win_cen_frac = (
            float(np.mean(prov.is_censored[best_c.window_start : best_c.window_stop]))
            if prov is not None
            else 0.0
        )
        if win_cen_frac > 0.50:
            warnings_list.append(
                WarningItem(
                    code="WINDOW_HEAVY_CENSORING",
                    severity="warning",
                    message=(
                        f"Jump at {format_timestamp(series.times_raw[j_valid], series.is_datetime)} "
                        f"is supported by a window that is {win_cen_frac:.0%} censored."
                    ),
                )
            )
        coincides_dl = (
            check_dl_shift_coincidence(t_yr, prov.dl_shifts, series.times_numeric)
            if prov is not None
            else False
        )
        dl_artifact: bool | None = None
        if coincides_dl:
            warnings_list.append(
                WarningItem(
                    code="DL_SHIFT_COINCIDENCE",
                    severity="warning",
                    message=(
                        f"Detected jump near {format_timestamp(series.times_raw[j_valid], series.is_datetime)} "
                        "coincides with a shift in the active censoring detection limit!"
                    ),
                    suggestion=(
                        "Re-run with `CensoredConfig(recensor_check=True)` or verify laboratory "
                        "method change history before concluding a physical shift occurred."
                    ),
                )
            )
        if cfg is not None and cfg.recensor_check and prov is not None and prov.n_censored > 0:
            y_recen = recensor_to_common_limit(
                prov, cfg, best_c.window_start, best_c.window_stop, rng
            )
            if y_recen is not None:
                t_w = series.times_numeric[best_c.window_start : best_c.window_stop]
                pw_w = pw.slice_range(best_c.window_start, best_c.window_stop)
                loc_r, mag_r, dir_r, _, freqs_r = _detect_jump_in_window(
                    t_w,
                    y_recen,
                    pw_w,
                    season_l,
                    om_arr,
                    alpha,
                    parity_mode=parity_mode,
                    robust=eff_robust,
                    huber_k=huber_k,
                    season_significance=eff_season_sig,
                )
                p_r, _ = _nested_jump_ftests(
                    t_w,
                    y_recen,
                    pw_w,
                    loc_r,
                    freqs_r,
                    season=season_l,
                    omega=om_arr,
                    alpha=alpha,
                    parity_mode=parity_mode,
                    robust=eff_robust,
                    huber_k=huber_k,
                    calibration=eff_calib,
                    n_mc=int(n_mc),
                    mc_seed=(20261128, int(best_c.window_start), int(best_c.window_stop)),
                    season_significance=eff_season_sig,
                )
                recen_alpha = eff_jump_alpha if eff_jump_alpha is not None else 0.05
                vanished = (abs(mag_r) < mag_th and abs(dir_r) < dir_th) or (p_r >= recen_alpha)
                dl_artifact = bool(vanished)
                if dl_artifact:
                    warnings_list.append(
                        WarningItem(
                            code="DL_ARTIFACT_LIKELY",
                            severity="warning",
                            message=(
                                f"Jump near {format_timestamp(series.times_raw[j_valid], series.is_datetime)} "
                                "disappears under common-limit recensoring (p="
                                f"{p_r:.3f}) — likely an artifact of changing detection limits."
                            ),
                        )
                    )

        stab_rate = (stability_counts[j_valid] / n_stab) if n_stab > 0 else None
        events.append(
            JumpEvent(
                index=orig_idx,
                valid_index=j_valid,
                time=series.times_raw[j_valid],
                time_year=t_yr,
                magnitude=mag_val,
                direction=dir_val,
                occurrences=occ,
                window_center_distance=dis_raw,
                jump_pvalue=pval,
                step_pvalue=p_step,
                window_censored_fraction=win_cen_frac,
                coincides_with_dl_shift=coincides_dl,
                dl_artifact_likely=dl_artifact,
                stability_rate=stab_rate,
            )
        )

    # Remap all_window_candidates to original index axis
    mapped_cands = tuple(
        WindowCandidate(
            window_start=int(series.valid_to_orig[c.window_start]),
            window_stop=int(series.valid_to_orig[c.window_stop - 1]) + 1,
            valid_index=c.valid_index,
            index=int(series.valid_to_orig[c.valid_index]),
            magnitude=c.magnitude,
            direction=c.direction,
            center_distance=c.center_distance,
            norm_res=c.norm_res,
            selected_freqs=c.selected_freqs,
        )
        for c in all_cands_valid
    )

    # Red-noise check on global piecewise-trend + seasonal residuals
    j_valid_arr = np.array([e.valid_index for e in events], dtype=np.intp)
    t0_full = series.times_numeric - series.times_numeric[0]
    try:
        if season_l == "allssa":
            fit_global = _allssa_core(
                t0_full,
                series.values,
                pw,
                om_arr,
                j_valid_arr,
                alpha,
                "linear",
                False,
                1,
                max(25, 2 * int(om_arr.size)),
                parity_mode=parity_mode,
                robust=eff_robust,
                huber_k=huber_k,
                significance=eff_season_sig,
            )
            global_res = fit_global.residuals
            cond_max = fit_global.condition_number_max
        elif season_l == "ols":
            fit_l = _lssa_core(
                t0_full,
                series.values,
                pw,
                np.empty(0, dtype=np.float64),
                j_valid_arr,
                alpha,
                "linear",
                False,
                om_arr,
                robust=eff_robust,
                huber_k=huber_k,
            )
            global_res = fit_l.residuals
            cond_max = fit_l.condition_number
        else:
            fit_l = _lssa_core(
                t0_full,
                series.values,
                pw,
                np.empty(0, dtype=np.float64),
                j_valid_arr,
                alpha,
                "linear",
                False,
                np.empty(0, dtype=np.float64),
                robust=eff_robust,
                huber_k=huber_k,
            )
            global_res = fit_l.residuals
            cond_max = fit_l.condition_number
        rn, rn_warns = compute_red_noise_diagnostics(series.times_numeric, global_res)
        warnings_list.extend(rn_warns)
    except InsufficientDataError:
        rn = None
        cond_max = 1.0

    diag = DiagnosticReport(
        red_noise=rn,
        condition_number_max=cond_max,
        n_input=series.n_input,
        n_valid=series.n_valid,
        n_dropped_nan=series.n_dropped_nan,
        sampling_rate_ratio=series.sampling_rate_ratio,
        warnings=tuple(warnings_list),
    )

    return JustJumpResult(
        jumps=tuple(events),
        all_window_candidates=mapped_cands,
        times_numeric=series.times_numeric,
        times_raw=series.times_raw,
        values=series.values,
        is_datetime=series.is_datetime,
        season=season_l,
        window_mode=window_mode,
        alpha=alpha,
        mag_th=mag_th,
        dir_th=dir_th,
        jump_th=jump_th,
        jump_alpha=eff_jump_alpha,
        censoring=prov,
        diagnostics=diag,
        jump_calibration=eff_calib,
        n_imputations=int(mi_spec[0]) if (mi_spec is not None and mi_counter[0] > 0) else 1,
    )


def _eval_piecewise_trend(
    t_obs: FloatArray,
    t_eval: FloatArray,
    local_jumps: IntArray,
    tr_coeff: FloatArray,
    *,
    parity_mode: bool = False,
) -> FloatArray:
    """Evaluate a piecewise linear trend analytically on ``t_eval``."""
    n_eval = int(t_eval.size)
    n_j = int(local_jumps.size)
    if n_j == 0:
        return np.asarray(tr_coeff[0] + tr_coeff[1] * t_eval, dtype=np.float64)

    eval_jumps: list[int] = []
    for j in local_jumps:
        idx_e = int(np.searchsorted(t_eval, t_obs[j], side="left"))
        if parity_mode:
            idx_e = min(n_eval, idx_e + 1)
        eval_jumps.append(idx_e)

    D, col_scales = build_design_matrix(
        t_eval,
        "linear",
        np.array(eval_jumps, dtype=np.intp),
        False,
        np.empty(0, dtype=np.float64),
    )
    return np.asarray(D @ (tr_coeff / col_scales), dtype=np.float64)


def _eval_seasonal(t_eval: FloatArray, freqs: FloatArray, cs_coeff: FloatArray) -> FloatArray:
    """Evaluate sinusoidal seasonal component analytically on ``t_eval``."""
    if freqs.size == 0:
        return np.zeros(t_eval.size, dtype=np.float64)
    phases = np.outer((2.0 * np.pi) * t_eval, freqs)
    return np.asarray(
        np.cos(phases) @ cs_coeff[:, 0] + np.sin(phases) @ cs_coeff[:, 1],
        dtype=np.float64,
    )


def _window_component_half_widths(
    t_obs: FloatArray,
    t_eval: FloatArray,
    local_jumps: IntArray,
    freqs: FloatArray,
    cov: FloatArray,
    n_obs: int,
    ci_level: float,
    *,
    is_obs_grid: bool = True,
    parity_mode: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """Compute pointwise Student-t half-widths for local trend and seasonal components.

    In ``parity_mode=True``, accounts for Ghaderpour's within-window sample-mean
    seasonal shifting (``trend_eff = trend + mean(seasonal)``,
    ``seasonal_eff = seasonal - mean(seasonal)``). In default mode
    (``parity_mode=False``), pure sinusoidal harmonics already have analytical
    cycle mean zero and are not sample-centered (preventing sampling-density
    bias on irregular grids), so exact block covariances apply directly.
    """
    n_eval = int(t_eval.size)
    n_tr = 2 * (int(local_jumps.size) + 1)
    n_seas = 2 * int(freqs.size)
    nc = n_tr + n_seas

    if cov.size == 0 or cov.shape != (nc, nc) or not np.all(np.isfinite(cov)) or n_obs <= nc:
        nan_arr = np.full(n_eval, np.nan, dtype=np.float64)
        return nan_arr, (np.zeros(n_eval, dtype=np.float64) if n_seas == 0 else nan_arr.copy())

    if is_obs_grid:
        eval_jumps = local_jumps
    elif local_jumps.size > 0:
        mapped_j: list[int] = []
        for j in local_jumps:
            idx_e = int(np.searchsorted(t_eval, t_obs[j], side="left"))
            if parity_mode:
                idx_e = min(n_eval, idx_e + 1)
            mapped_j.append(idx_e)
        eval_jumps = np.array(mapped_j, dtype=np.intp)
    else:
        eval_jumps = np.empty(0, dtype=np.intp)

    D_full, col_scales = build_design_matrix(t_eval, "linear", eval_jumps, False, freqs)
    D_u = D_full / col_scales
    t_crit = float(t_dist.ppf(0.5 + 0.5 * ci_level, df=max(1, n_obs - nc)))

    if n_seas == 0:
        var_tr = np.maximum(0.0, np.sum((D_u @ cov) * D_u, axis=1))
        return t_crit * np.sqrt(var_tr), np.zeros(n_eval, dtype=np.float64)

    D_tr = D_u[:, :n_tr]
    D_seas = D_u[:, n_tr:]
    cov_ss = cov[n_tr:, n_tr:]

    if parity_mode:
        d_seas_mean = np.mean(D_seas, axis=0)
        D_seas_cent = D_seas - d_seas_mean[None, :]
        var_seas = np.maximum(0.0, np.sum((D_seas_cent @ cov_ss) * D_seas_cent, axis=1))

        D_tr_eff = np.empty_like(D_u)
        D_tr_eff[:, :n_tr] = D_tr
        D_tr_eff[:, n_tr:] = d_seas_mean[None, :]
        var_tr = np.maximum(0.0, np.sum((D_tr_eff @ cov) * D_tr_eff, axis=1))
    else:
        cov_tt = cov[:n_tr, :n_tr]
        var_tr = np.maximum(0.0, np.sum((D_tr @ cov_tt) * D_tr, axis=1))
        var_seas = np.maximum(0.0, np.sum((D_seas @ cov_ss) * D_seas, axis=1))

    return t_crit * np.sqrt(var_tr), t_crit * np.sqrt(var_seas)


def _window_bias_corrected_trend(
    t_obs: FloatArray,
    y_obs: FloatArray,
    pw_w: PreparedWeights,
    local_jumps: IntArray,
    freqs: FloatArray,
    t_targets: list[FloatArray],
    target_is_obs: list[bool],
    ci_level: float,
    *,
    robust: RobustMethod,
    huber_k: float,
) -> list[tuple[FloatArray, FloatArray]] | None:
    """Local-quadratic (bias-corrected) window trend and pointwise t half-widths.

    Refits the window with a quadratic trend term (piecewise intercepts/slopes at the
    local jumps plus one global quadratic) holding the selected harmonics fixed, and
    returns ``(trend_bc, half_width)`` on each requested target grid. Centering the band
    on the bias-corrected estimate with its own (larger) variance removes the leading
    local-linear smoothing bias (Calonico, Cattaneo & Farrell, 2018).
    Returns ``None`` when the window cannot support the extra parameter.
    """
    n_obs = int(t_obs.size)
    n_j = int(local_jumps.size)
    n_trq = 2 * (n_j + 1) + 1
    nc_q = n_trq + 2 * int(freqs.size)
    if n_obs <= nc_q + 1:
        return None
    try:
        res_q = _lssa_core(
            t_obs, y_obs, pw_w, np.empty(0, dtype=np.float64), local_jumps, 0.05,
            "quadratic", False, freqs, compute_cov=True, robust=robust, huber_k=huber_k,
        )
    except InsufficientDataError:
        return None
    cov = res_q.cov
    if cov.size == 0 or not np.all(np.isfinite(cov)) or res_q.rank_deficient:
        return None
    coeff_tr = res_q.coeff[:n_trq]
    cov_tt = cov[:n_trq, :n_trq]
    t_crit = float(t_dist.ppf(0.5 + 0.5 * ci_level, df=max(1, n_obs - nc_q)))
    out: list[tuple[FloatArray, FloatArray]] = []
    for tt, is_obs in zip(t_targets, target_is_obs, strict=True):
        if is_obs or n_j == 0:
            jj = local_jumps
        else:
            jj = np.array([int(np.searchsorted(tt, t_obs[j], side="left")) for j in local_jumps], dtype=np.intp)
        D, cs = build_design_matrix(tt, "quadratic", jj, False, np.empty(0, dtype=np.float64))
        D_u = D / cs
        est = D_u @ coeff_tr
        var = np.maximum(0.0, np.sum((D_u @ cov_tt) * D_u, axis=1))
        out.append((np.asarray(est, dtype=np.float64), t_crit * np.sqrt(var)))
    return out


def just_decompose(
    t: TimeLike,
    y: ValueLike,
    *,
    weights: WeightLike | None = None,
    eval_times: TimeLike | None = None,
    jump_indices: Sequence[int] | None = None,
    size: int | None = None,
    step: int | None = None,
    window_mode: WindowMode = "count",
    window_duration: float | None = None,
    step_duration: float | None = None,
    min_window_obs: int | None = None,
    season: SeasonMode = "allssa",
    omega: ArrayLike | None = None,
    alpha: float = 0.01,
    ci_level: float = 0.95,
    trend_ci_method: Literal["bias_corrected", "model"] | None = None,
    season_significance: SeasonSignificance = "sidak",
    robust: RobustMethod = "none",
    huber_k: float = 1.345,
    censor_status: ArrayLike | None = None,
    censoring: CensoredConfig | None = None,
    parity_mode: bool = False,
) -> JustDecomposeResult:
    """Decompose an unevenly sampled time series into trend, seasonal, and remainder via JUST.

    Uses Gaussian-weighted overlapping window estimation with zero interpolation
    of the input observations. Optional ``eval_times`` evaluates the fitted
    parametric trend and seasonal models analytically onto any target grid.
    Set ``robust='huber'`` or ``robust='mm'`` to fit each window via outlier-resistant
    M/MM-estimation so isolated extreme spikes do not distort the local trend or
    seasonality.

    Pointwise confidence intervals at level ``ci_level`` (default ``0.95``) are
    computed analytically inside each window and blended across overlapping
    Gaussian windows via the Cauchy–Schwarz half-width blend (a valid upper bound on
    the standard deviation of the blended estimator).

    ``trend_ci_method`` controls the trend band. ``"bias_corrected"`` (default)
    refits every window with an added quadratic trend term and centers the band on
    that bias-corrected estimate (``trend_bc``) with its own variance, so smooth
    curvature within a window does not make the band miss the true trend (robust
    bias correction, Calonico, Cattaneo & Farrell, 2018). ``"model"`` (forced by
    ``parity_mode=True``) gives the band of the local-linear window model around
    ``trend``; it ignores the smoothing bias and under-covers curved trends (e.g.
    ~70% for a 9-year cycle). Seasonal bands are model-based in both cases.
    In default mode (``parity_mode=False``), pure sinusoidal seasonal terms are
    not sample-mean shifted onto the trend, eliminating seasonal-balance bias on
    irregularly sampled timestamps (``parity_mode=True`` retains Ghaderpour's
    sample-mean centering for exact upstream parity).

    Parameters
    ----------
    t:
        1D monotonically ascending timestamps (numeric decimal years or datetime).
    y:
        1D observations (numeric, censored string markers like ``'<0.05'``, or
        ``(N, 2)`` intervals). ``NaN`` entries are dropped without interpolation.
    weights:
        Optional scalar, 1D positive weight vector, or 2D SPD inverse-covariance
        matrix ``P = C_f^{-1}`` (relative precisions; the residual variance factor is
        estimated).
    eval_times:
        Optional evaluation grid inside the observed time span; fills the
        ``*_reg`` fields (e.g. a regular monthly grid).
    jump_indices:
        0-based indices (input axis) of known jumps, typically
        ``[j.index for j in just_jumps(...).jumps]``; each window fits a separate
        piece on either side.
    size, step:
        Observation-count window size and step for ``window_mode='count'``
        (defaults ``3*M`` and ``M``, ``M`` = samples per year).
    window_mode:
        ``'count'`` (default) or ``'duration'`` (fixed calendar span).
    window_duration, step_duration, min_window_obs:
        Window span, step and minimum observations for ``window_mode='duration'``
        (defaults ``min(3, span)`` years, ``window_duration/3``, and
        ``max(12, n_params + 5)``).
    season:
        ``'allssa'`` (default; anti-leakage frequency selection per window, see
        ``season_significance``), ``'ols'`` (fixed frequencies ``omega``), or ``'none'``.
    omega:
        Candidate (ALLSSA) or fixed (OLS) cyclic frequencies in cycles/year.
        Defaults to ``[0.8, 1.0, ..., 3.8]`` for ALLSSA and ``[1, 2, 3, 4]`` for OLS.
    alpha:
        Per-frequency significance level for ALLSSA selection inside windows.
    ci_level:
        Pointwise confidence level of the trend and seasonal bands.
    trend_ci_method:
        ``'bias_corrected'`` (default) or ``'model'``; see above.
    season_significance:
        Acceptance rule for ALLSSA-selected harmonics in each window
        (``season='allssa'``): ``'sidak'`` (default, family-wise over the searched
        frequencies), ``'montecarlo'``, or ``'per_frequency'`` (published rule,
        forced by ``parity_mode``).
    robust:
        ``'none'``, ``'huber'``, or ``'mm'`` window fits (forced to ``'none'`` in
        ``parity_mode``).
    huber_k:
        Huber tuning constant (default ``1.345``).
    censor_status, censoring:
        Optional censoring indicators and :class:`CensoredConfig`.
    parity_mode:
        Reproduce Ghaderpour's upstream JUST decomposition exactly (seasonal
        sample-mean centering, legacy rounding, model-based trend band).

    Returns
    -------
    JustDecomposeResult
        ``trend``, ``seasonal`` and ``remainder`` on the observed times (they sum to
        the input exactly), pointwise bands, and optional ``eval_times`` versions.
        Seasonal bands are conditional on the selected frequencies.
    """
    season_l = season.lower()
    if season_l not in ("allssa", "ols", "none"):
        raise ValueError(f"Unknown `season` {season!r}. Options: 'allssa', 'ols', 'none'.")
    if window_mode not in ("count", "duration"):
        raise ValueError(f"Unknown `window_mode` {window_mode!r}. Options: 'count', 'duration'.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"`alpha` must lie in (0, 1); got {alpha}.")
    if not (0.0 < ci_level < 1.0):
        raise ValueError(f"`ci_level` must lie in (0, 1); got {ci_level}.")
    if trend_ci_method is not None and trend_ci_method not in ("bias_corrected", "model"):
        raise ValueError(
            f"Unknown `trend_ci_method` {trend_ci_method!r}. Options: 'bias_corrected', 'model'."
        )
    eff_ci_method = "model" if parity_mode else (trend_ci_method or "bias_corrected")
    eff_season_sig = _resolve_season_significance(season_significance, parity_mode)
    eff_robust: RobustMethod = "none" if parity_mode else validate_robust_args(robust, huber_k)

    prep = prepare_series_with_censoring(t, y, censor_status=censor_status, censoring=censoring)
    series = prep.series
    prov = prep.provenance

    pw_full = prepare_weights(weights, series.n_input)
    pw = pw_full.subset(series.valid_mask) if series.n_dropped_nan else pw_full
    if prep.weight_factors is not None:
        pw = pw.scale_entries(prep.weight_factors)

    # Map jump_indices (original axis) onto valid axis
    if jump_indices is not None and len(jump_indices) > 0:
        orig_j = np.asarray(jump_indices, dtype=np.intp)
        if np.any(orig_j < 0) or np.any(orig_j >= series.n_input):
            raise ValueError("`jump_indices` contains out-of-range indices.")
        if series.n_dropped_nan:
            mapped = [int(np.searchsorted(series.valid_to_orig, j)) for j in orig_j]
            j_valid = _validate_jump_indices(mapped, series.n_valid)
        else:
            j_valid = _validate_jump_indices(orig_j.tolist(), series.n_valid)
    else:
        j_valid = np.empty(0, dtype=np.intp)

    om_arr = (
        np.sort(np.asarray(omega, dtype=np.float64).ravel())
        if omega is not None and len(omega) > 0
        else _default_just_omega(season_l)
    )

    has_eval = eval_times is not None and len(eval_times) > 0
    if has_eval:
        assert eval_times is not None
        tt_num, tt_raw = convert_eval_times(
            eval_times,
            float(series.times_numeric[0]),
            float(series.times_numeric[-1]),
            expect_datetime=series.is_datetime,
        )
    else:
        tt_num = series.times_numeric
        tt_raw = series.times_raw

    warnings_list: list[WarningItem] = list(prep.warnings)
    warnings_list.extend(time_unit_warning(series.times_numeric, series.is_datetime))
    t_origin = float(tt_num[0])
    t0 = series.times_numeric - t_origin
    tt0 = tt_num - t_origin
    n = series.n_valid
    n_tt = int(tt0.size)

    if window_mode == "count":
        windows_tt, win_size, _win_step, rate_eff = _build_count_windows(
            n_tt, tt_num, size, step, parity_mode=parity_mode
        )
        if parity_mode:
            rate_for_dil = float(rate_eff)
        else:
            span_tt = float(tt_num[-1] - tt_num[0]) if n_tt > 1 else 1.0
            rate_for_dil = (n_tt / span_tt) if span_tt > 0.0 else float(rate_eff)
        dil_fac = float(np.log(0.1) / ((win_size / (6.0 * rate_for_dil)) ** 2))
        if not parity_mode and series.sampling_rate_ratio > 2.0:
            warnings_list.append(
                WarningItem(
                    code="VARIABLE_SAMPLING_RATE",
                    severity="caution",
                    message=(
                        f"Sampling density varies by ~{series.sampling_rate_ratio:.1f}x across "
                        "the record while using fixed observation-count windows."
                    ),
                    suggestion="Consider `window_mode='duration'` for uniform calendar smoothing.",
                )
            )
    else:
        min_obs_default = 4 + (2 * int(om_arr.size) if season_l == "ols" else 0) + 5
        min_obs = int(min_window_obs) if min_window_obs is not None else max(12, min_obs_default)
        windows_tt, dur, skipped = _build_duration_windows(
            tt_num, window_duration, step_duration, min_obs if not has_eval else 2
        )
        dil_fac = float(np.log(0.1) / ((dur / 6.0) ** 2))
        if skipped > 0:
            warnings_list.append(
                WarningItem(
                    code="SPARSE_WINDOW_SKIPPED",
                    severity="info",
                    message=f"Skipped {skipped} sparse duration window(s).",
                )
            )

    sw_warn = _short_seasonal_window_warning(tt_num, windows_tt, om_arr, season_l, eff_season_sig)
    if sw_warn is not None:
        warnings_list.append(sw_warn)

    cfg = prep.config
    use_local_censor = (
        prov is not None
        and cfg is not None
        and cfg.scope == "windowed"
        and prov.n_censored > 0
    )
    if cfg is not None and isinstance(cfg.random_state, np.random.Generator):
        rng_dec: np.random.Generator = cfg.random_state
    else:
        rng_dec = np.random.default_rng(cfg.random_state if cfg is not None else 0)
    win_censor_fb = 0
    eval_gap_skipped = 0

    trends0 = np.zeros(n, dtype=np.float64)
    seasonals0 = np.zeros(n, dtype=np.float64)
    remainders0 = np.zeros(n, dtype=np.float64)
    sum_w0 = np.zeros(n, dtype=np.float64)
    trend_hw0 = np.zeros(n, dtype=np.float64)
    seas_hw0 = np.zeros(n, dtype=np.float64)
    sum_w0_tr_ci = np.zeros(n, dtype=np.float64)
    sum_w0_s_ci = np.zeros(n, dtype=np.float64)

    trends_reg_acc = np.zeros(n_tt, dtype=np.float64)
    seasonals_reg_acc = np.zeros(n_tt, dtype=np.float64)
    sum_w_reg = np.zeros(n_tt, dtype=np.float64)
    trend_reg_hw_acc = np.zeros(n_tt, dtype=np.float64)
    seas_reg_hw_acc = np.zeros(n_tt, dtype=np.float64)
    sum_w_reg_tr_ci = np.zeros(n_tt, dtype=np.float64)
    sum_w_reg_s_ci = np.zeros(n_tt, dtype=np.float64)

    bc_on = eff_ci_method == "bias_corrected"
    trend_bc0 = np.zeros(n, dtype=np.float64)
    hw_bc0 = np.zeros(n, dtype=np.float64)
    sum_w_bc0 = np.zeros(n, dtype=np.float64)
    trend_bc_reg = np.zeros(n_tt, dtype=np.float64)
    hw_bc_reg = np.zeros(n_tt, dtype=np.float64)
    sum_w_bc_reg = np.zeros(n_tt, dtype=np.float64)
    bc_fallbacks = 0

    win_freqs_list: list[FloatArray] = []
    win_bounds_list: list[tuple[int, int]] = []
    cond_max = 1.0

    for lb, ub in windows_tt:
        tty = tt0[lb:ub]
        if has_eval:
            lb0 = int(np.searchsorted(t0, tt0[lb], side="left"))
            ub0 = min(n, int(np.searchsorted(t0, tt0[ub - 1], side="left")) + 1)
        else:
            lb0, ub0 = lb, ub

        size0 = ub0 - lb0
        local_j = np.array(
            [int(j - lb0) for j in j_valid if lb0 + 2 <= j < ub0 - 1], dtype=np.intp
        )
        base_nc = 2 * (int(local_j.size) + 1)
        if season_l == "ols":
            min_needed = base_nc + 2 * int(om_arr.size) + 1
        elif season_l == "allssa":
            min_needed = base_nc + 3
        else:
            min_needed = base_nc + 1
        if size0 <= 0 or (has_eval and size0 < min_needed):
            eval_gap_skipped += 1
            continue

        ty = t0[lb0:ub0]
        if use_local_censor:
            assert prov is not None and cfg is not None
            yy, fb = impute_window_local(prov, cfg, lb0, ub0, rng_dec)
            if fb:
                win_censor_fb += 1
        else:
            yy = series.values[lb0:ub0]
        pw_w = pw.slice_range(lb0, ub0)

        # Shift local window time to ty[0] for numerical conditioning when evaluating
        ty_loc = ty - ty[0]
        tty_loc = tty - ty[0]

        if season_l == "allssa":
            res_a = _allssa_core(
                ty_loc,
                yy,
                pw_w,
                om_arr,
                local_j,
                alpha,
                "linear",
                False,
                1,
                max(25, 2 * int(om_arr.size)),
                parity_mode=parity_mode,
                compute_cov=True,
                robust=eff_robust,
                huber_k=huber_k,
                significance=eff_season_sig,
            )
            tr_coeff = res_a.trend_coeff
            cs_coeff = res_a.cos_sin_coeffs
            rem_w = res_a.residuals
            freqs_w = res_a.frequencies
            cov_w = res_a.cov
            cond_max = max(cond_max, res_a.condition_number_max)
        elif season_l == "ols":
            res_l = _lssa_core(
                ty_loc,
                yy,
                pw_w,
                np.empty(0, dtype=np.float64),
                local_j,
                alpha,
                "linear",
                False,
                om_arr,
                compute_cov=True,
                robust=eff_robust,
                huber_k=huber_k,
            )
            rem_w = res_l.residuals
            n_f = int(om_arr.size)
            tr_coeff = res_l.coeff[: res_l.coeff.size - 2 * n_f]
            cs_coeff = res_l.coeff[res_l.coeff.size - 2 * n_f :].reshape(n_f, 2)
            freqs_w = om_arr
            cov_w = res_l.cov
            cond_max = max(cond_max, res_l.condition_number)
        else:
            res_l = _lssa_core(
                ty_loc,
                yy,
                pw_w,
                np.empty(0, dtype=np.float64),
                local_j,
                alpha,
                "linear",
                False,
                np.empty(0, dtype=np.float64),
                compute_cov=True,
                robust=eff_robust,
                huber_k=huber_k,
            )
            rem_w = res_l.residuals
            tr_coeff = res_l.coeff
            cs_coeff = np.empty((0, 2), dtype=np.float64)
            freqs_w = np.empty(0, dtype=np.float64)
            cov_w = res_l.cov
            cond_max = max(cond_max, res_l.condition_number)

        win_freqs_list.append(freqs_w)
        win_bounds_list.append((int(lb0), int(ub0)))
        season_t = _eval_seasonal(ty_loc, freqs_w, cs_coeff)
        trend_t = yy - season_t - rem_w
        if parity_mode:
            mean_s = float(np.mean(season_t))
            trend_t = trend_t + mean_s
            season_t = season_t - mean_s

        hw_tr_w, hw_s_w = _window_component_half_widths(
            ty_loc,
            ty_loc,
            local_j,
            freqs_w,
            cov_w,
            size0,
            ci_level,
            is_obs_grid=True,
            parity_mode=parity_mode,
        )

        tau0 = float(ty[size0 // 2]) if (size0 % 2 == 1) else 0.5 * float(ty[size0 // 2] + ty[size0 // 2 - 1])
        gauss_w = np.exp(dil_fac * (ty - tau0) ** 2)
        trends0[lb0:ub0] += trend_t * gauss_w
        seasonals0[lb0:ub0] += season_t * gauss_w
        remainders0[lb0:ub0] += rem_w * gauss_w
        sum_w0[lb0:ub0] += gauss_w

        fin_tr = np.isfinite(hw_tr_w)
        if np.any(fin_tr):
            trend_hw0[lb0:ub0] += np.where(fin_tr, hw_tr_w * gauss_w, 0.0)
            sum_w0_tr_ci[lb0:ub0] += np.where(fin_tr, gauss_w, 0.0)
        fin_s = np.isfinite(hw_s_w)
        if np.any(fin_s):
            seas_hw0[lb0:ub0] += np.where(fin_s, hw_s_w * gauss_w, 0.0)
            sum_w0_s_ci[lb0:ub0] += np.where(fin_s, gauss_w, 0.0)

        bc_win: list[tuple[FloatArray, FloatArray]] | None = None
        if bc_on:
            targets = [ty_loc, tty_loc] if has_eval else [ty_loc]
            bc_win = _window_bias_corrected_trend(
                ty_loc, yy, pw_w, local_j, freqs_w, targets, [True, False][: len(targets)],
                ci_level, robust=eff_robust, huber_k=huber_k,
            )
            if bc_win is None:
                bc_fallbacks += 1
                bc_obs = (trend_t, hw_tr_w)
            else:
                bc_obs = bc_win[0]
            fin_bc = np.isfinite(bc_obs[1])
            trend_bc0[lb0:ub0] += np.where(fin_bc, bc_obs[0] * gauss_w, 0.0)
            hw_bc0[lb0:ub0] += np.where(fin_bc, bc_obs[1] * gauss_w, 0.0)
            sum_w_bc0[lb0:ub0] += np.where(fin_bc, gauss_w, 0.0)

        if has_eval:
            trend_tt = _eval_piecewise_trend(
                ty_loc, tty_loc, local_j, tr_coeff, parity_mode=parity_mode
            )
            season_tt = _eval_seasonal(tty_loc, freqs_w, cs_coeff)
            if parity_mode:
                mean_s_tt = float(np.mean(season_tt))
                trend_tt = trend_tt + mean_s_tt
                season_tt = season_tt - mean_s_tt
            hw_tr_tt, hw_s_tt = _window_component_half_widths(
                ty_loc,
                tty_loc,
                local_j,
                freqs_w,
                cov_w,
                size0,
                ci_level,
                is_obs_grid=False,
                parity_mode=parity_mode,
            )
            w_len = ub - lb
            tau_tt = (
                float(tty[w_len // 2])
                if (w_len % 2 == 1)
                else 0.5 * float(tty[w_len // 2] + tty[w_len // 2 - 1])
            )
            gw_tt = np.exp(dil_fac * (tty - tau_tt) ** 2)
            trends_reg_acc[lb:ub] += trend_tt * gw_tt
            seasonals_reg_acc[lb:ub] += season_tt * gw_tt
            sum_w_reg[lb:ub] += gw_tt

            fin_tr_tt = np.isfinite(hw_tr_tt)
            if np.any(fin_tr_tt):
                trend_reg_hw_acc[lb:ub] += np.where(fin_tr_tt, hw_tr_tt * gw_tt, 0.0)
                sum_w_reg_tr_ci[lb:ub] += np.where(fin_tr_tt, gw_tt, 0.0)
            fin_s_tt = np.isfinite(hw_s_tt)
            if np.any(fin_s_tt):
                seas_reg_hw_acc[lb:ub] += np.where(fin_s_tt, hw_s_tt * gw_tt, 0.0)
                sum_w_reg_s_ci[lb:ub] += np.where(fin_s_tt, gw_tt, 0.0)
            if bc_on:
                bc_reg = bc_win[1] if bc_win is not None else (trend_tt, hw_tr_tt)
                fin_bcr = np.isfinite(bc_reg[1])
                trend_bc_reg[lb:ub] += np.where(fin_bcr, bc_reg[0] * gw_tt, 0.0)
                hw_bc_reg[lb:ub] += np.where(fin_bcr, bc_reg[1] * gw_tt, 0.0)
                sum_w_bc_reg[lb:ub] += np.where(fin_bcr, gw_tt, 0.0)

    if eval_gap_skipped > 0:
        warnings_list.append(
            WarningItem(
                code="SPARSE_WINDOW_SKIPPED",
                severity="info",
                message=(
                    f"Skipped {eval_gap_skipped} evaluation window(s) spanning a data gap "
                    "with insufficient observations; uncovered `eval_times` entries are NaN."
                ),
            )
        )
    if win_censor_fb > 0:
        warnings_list.append(
            WarningItem(
                code="WINDOW_IMPUTATION_FALLBACK",
                severity="info",
                message=(
                    f"Windowed censoring imputation fell back to global imputation in "
                    f"{win_censor_fb} heavily censored window(s)."
                ),
            )
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        trend = np.where(sum_w0 > 0, trends0 / sum_w0, np.nan)
        seasonal = np.where(sum_w0 > 0, seasonals0 / sum_w0, np.nan)
        remainder = np.where(sum_w0 > 0, remainders0 / sum_w0, np.nan)
        tr_hw = np.where(sum_w0_tr_ci > 0, trend_hw0 / sum_w0_tr_ci, np.nan)
        s_hw = np.where(sum_w0_s_ci > 0, seas_hw0 / sum_w0_s_ci, np.nan)
        trend_bc_arr: FloatArray | None = None
        if bc_on:
            trend_bc_arr = np.where(sum_w_bc0 > 0, trend_bc0 / sum_w_bc0, np.nan)
            hw_bc = np.where(sum_w_bc0 > 0, hw_bc0 / sum_w_bc0, np.nan)
            trend_ci_low = trend_bc_arr - hw_bc
            trend_ci_high = trend_bc_arr + hw_bc
        else:
            trend_ci_low = trend - tr_hw
            trend_ci_high = trend + tr_hw
        seasonal_ci_low = seasonal - s_hw
        seasonal_ci_high = seasonal + s_hw

        if has_eval:
            trend_reg: FloatArray | None = np.where(
                sum_w_reg > 0, trends_reg_acc / sum_w_reg, np.nan
            )
            seasonal_reg: FloatArray | None = np.where(
                sum_w_reg > 0, seasonals_reg_acc / sum_w_reg, np.nan
            )
            tr_reg_hw = np.where(sum_w_reg_tr_ci > 0, trend_reg_hw_acc / sum_w_reg_tr_ci, np.nan)
            s_reg_hw = np.where(sum_w_reg_s_ci > 0, seas_reg_hw_acc / sum_w_reg_s_ci, np.nan)
            trend_reg_bc: FloatArray | None = None
            if bc_on:
                trend_reg_bc = np.where(sum_w_bc_reg > 0, trend_bc_reg / sum_w_bc_reg, np.nan)
                hw_bcr = np.where(sum_w_bc_reg > 0, hw_bc_reg / sum_w_bc_reg, np.nan)
                trend_reg_ci_low: FloatArray | None = trend_reg_bc - hw_bcr
                trend_reg_ci_high: FloatArray | None = trend_reg_bc + hw_bcr
            else:
                trend_reg_ci_low = trend_reg - tr_reg_hw
                trend_reg_ci_high = trend_reg + tr_reg_hw
            seasonal_reg_ci_low: FloatArray | None = seasonal_reg - s_reg_hw
            seasonal_reg_ci_high: FloatArray | None = seasonal_reg + s_reg_hw
        else:
            trend_reg = None
            seasonal_reg = None
            trend_reg_bc = None
            trend_reg_ci_low = None
            trend_reg_ci_high = None
            seasonal_reg_ci_low = None
            seasonal_reg_ci_high = None

    if bc_on and bc_fallbacks > 0:
        warnings_list.append(
            WarningItem(
                code="BIAS_CORRECTION_FALLBACK",
                severity="info",
                message=(
                    f"{bc_fallbacks} window(s) were too small for the local-quadratic bias-corrected "
                    "trend band; their model-based band was used instead."
                ),
            )
        )
    valid_rem = remainder[~np.isnan(remainder)]
    rn, rn_warns = compute_red_noise_diagnostics(
        series.times_numeric[~np.isnan(remainder)], valid_rem
    )
    warnings_list.extend(rn_warns)

    diag = DiagnosticReport(
        red_noise=rn,
        condition_number_max=cond_max,
        n_input=series.n_input,
        n_valid=series.n_valid,
        n_dropped_nan=series.n_dropped_nan,
        sampling_rate_ratio=series.sampling_rate_ratio,
        warnings=tuple(warnings_list),
    )

    return JustDecomposeResult(
        times_numeric=series.times_numeric,
        times_raw=series.times_raw,
        values=series.values,
        trend=trend,
        seasonal=seasonal,
        remainder=remainder,
        eval_times_numeric=tt_num if has_eval else None,
        eval_times_raw=tt_raw if has_eval else None,
        trend_reg=trend_reg,
        seasonal_reg=seasonal_reg,
        jump_indices=series.valid_to_orig[j_valid] if j_valid.size else j_valid,
        window_frequencies=tuple(win_freqs_list),
        window_bounds=tuple(win_bounds_list),
        window_dilation=float(dil_fac),
        is_datetime=series.is_datetime,
        season=season_l,
        window_mode=window_mode,
        censoring=prov,
        diagnostics=diag,
        ci_level=ci_level,
        trend_ci_low=trend_ci_low,
        trend_ci_high=trend_ci_high,
        seasonal_ci_low=seasonal_ci_low,
        seasonal_ci_high=seasonal_ci_high,
        trend_reg_ci_low=trend_reg_ci_low,
        trend_reg_ci_high=trend_reg_ci_high,
        seasonal_reg_ci_low=seasonal_reg_ci_low,
        seasonal_reg_ci_high=seasonal_reg_ci_high,
        trend_ci_method=eff_ci_method,
        trend_bc=trend_bc_arr,
        trend_reg_bc=trend_reg_bc,
    )
