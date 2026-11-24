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

from ._diagnostics import DiagnosticReport, WarningItem, compute_red_noise_diagnostics
from ._linalg import PreparedWeights, prepare_weights
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
    """Detected abrupt jump and slope change returned by :func:`just_jumps`."""

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
    """Result object returned by :func:`just_jumps`."""

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
            f"jump_alpha={self.jump_alpha})",
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
            for k, j in enumerate(self.jumps, start=1):
                t_str = format_timestamp(j.time, self.is_datetime)
                step_word = "upward" if j.magnitude >= 0 else "downward"
                slope_word = "steepened/increased" if j.direction >= 0 else "declined/decreased"
                desc = (
                    f"  {k}. Around {t_str} (index {j.index}): {step_word} step of "
                    f"{j.magnitude:+.4g} and slope {slope_word} by {j.direction:+.4g}/unit time "
                    f"(seen in {j.occurrences} window(s); screening F-test p={j.jump_pvalue:.2e})."
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
            lines.append(
                "Note: Window screening p-values are computed at the RSS-minimizing split "
                "index and are more optimistic than fixed-date tests."
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
    """Result object returned by :func:`just_decompose`."""

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
) -> tuple[int, float, float, float, FloatArray]:
    """Scan all interior split indices ``3 .. n_win - 3`` in a single window."""
    n_win = int(t_win.size)
    t0 = t_win - t_win[0]
    best_ind = -1
    best_mag = 0.0
    best_dir = 0.0
    best_rss = np.inf
    best_freqs = np.empty(0, dtype=np.float64)

    for ind in range(3, n_win - 2):
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
            )
            rss = res_a.norm_res
            tr_c = res_a.trend_coeff
            freqs = res_a.frequencies
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
            )
            rss = res_l.norm_res
            tr_c = res_l.coeff[:4]
            freqs = omega
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
            )
            rss = res_l.norm_res
            tr_c = res_l.coeff[:4]
            freqs = np.empty(0, dtype=np.float64)

        if rss < best_rss:
            best_rss = rss
            best_ind = ind
            best_mag = float((tr_c[1] - tr_c[0]) + (tr_c[3] - tr_c[2]) * t0[ind])
            best_dir = float(tr_c[3] - tr_c[2])
            best_freqs = freqs

    if best_ind < 0:
        raise InsufficientDataError(f"Window of length {n_win} is too short for jump detection.")

    return best_ind, best_mag, best_dir, best_rss, best_freqs


def _nested_jump_ftests(
    t_win: FloatArray,
    y_win: FloatArray,
    pw_win: PreparedWeights,
    local_jump_idx: int,
    freqs: FloatArray,
    *,
    bonferroni: bool = False,
) -> tuple[float, float]:
    """Nested partial F-tests comparing discontinuous 2-piece vs null & continuous hinge.

    Returns ``(jump_pvalue, step_pvalue)``:
    - ``jump_pvalue`` (2 df): 2-piece discontinuous linear vs 1-piece linear trend.
    - ``step_pvalue`` (1 df): 2-piece discontinuous linear vs continuous 2-piece
      hinge ``[1, t0, (t0 - t_j)_+]``, isolating whether a genuine vertical step
      discontinuity exists beyond a smooth slope bend.

    If ``bonferroni=True``, multiplies raw p-values by the number of candidate
    interior split indices scanned in the window (``max(1, n_win - 5)``) to
    mitigate post-selection inflation.
    """
    from ._linalg import solve_weighted_ols

    n_win = int(t_win.size)
    t0 = t_win - t_win[0]
    n_f = int(freqs.size)
    nc_full = 4 + 2 * n_f
    dof2 = n_win - nc_full
    if dof2 <= 0:
        return float("nan"), float("nan")

    res_full = _lssa_core(
        t0,
        y_win,
        pw_win,
        np.empty(0, dtype=np.float64),
        np.array([local_jump_idx], dtype=np.intp),
        0.05,
        "linear",
        False,
        freqs,
    )
    res_null = _lssa_core(
        t0,
        y_win,
        pw_win,
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.intp),
        0.05,
        "linear",
        False,
        freqs,
    )

    # Optimal continuous hinge model across the window:
    # [1, t0/s, max(0, t0 - t0[k])/s, cos/sin(freqs)] for k in [3 .. n_win - 3].
    # Optimizing k fairly compares the best discontinuous jump against the best
    # continuous turning point rather than handicapping the hinge at a biased split.
    s_t = float(t0[-1] - t0[0]) if t0[-1] > t0[0] else 1.0
    D_hinge = np.empty((n_win, 3 + 2 * n_f), dtype=np.float64)
    D_hinge[:, 0] = 1.0
    D_hinge[:, 1] = t0 / s_t
    if n_f > 0:
        phases = np.outer((2.0 * np.pi) * t0, freqs)
        D_hinge[:, 3::2] = np.cos(phases)
        D_hinge[:, 4::2] = np.sin(phases)
    rss_hinge = np.inf
    for k_knot in range(3, n_win - 2):
        D_hinge[:, 2] = np.maximum(0.0, t0 - t0[k_knot]) / s_t
        rss_k = solve_weighted_ols(D_hinge, y_win, pw_win).norm_res
        if rss_k < rss_hinge:
            rss_hinge = rss_k

    rss1 = res_full.norm_res
    rss0 = res_null.norm_res
    mult = float(max(1, n_win - 5)) if bonferroni else 1.0

    if rss1 <= 1e-24:
        p_jump = 0.0 if rss0 > 1e-24 else 1.0
        p_step = 0.0 if rss_hinge > 1e-24 else 1.0
        return p_jump, p_step

    den = rss1 / dof2
    f_jump = max(0.0, (rss0 - rss1) / 2.0) / den
    f_step = max(0.0, rss_hinge - rss1) / den
    p_jump = float(min(1.0, mult * float(f_dist.sf(f_jump, 2, dof2))))
    p_step = float(min(1.0, mult * float(f_dist.sf(f_step, 1, dof2))))
    return p_jump, p_step


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
    parity_mode: bool,
    provenance: CensoringProvenance | None = None,
    censor_cfg: CensoredConfig | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[
    list[tuple[int, int, float, float, float, float, float, WindowCandidate]],
    list[WindowCandidate],
    int,
]:
    """Core sliding-window JUST jump search + aggregation + merging."""
    candidates: list[WindowCandidate] = []
    window_fallbacks = 0

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
            if fb:
                window_fallbacks += 1
        else:
            y_w = y_valid[w_start:w_stop]
        pw_w = pw.slice_range(w_start, w_stop)
        loc_ind, mag, direction, rss, freqs = _detect_jump_in_window(
            t_w, y_w, pw_w, season, omega, alpha, parity_mode=parity_mode
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
    accepted: list[tuple[int, int, float, float, float, float, float, WindowCandidate]] = []
    for item in grouped:
        j_idx, occ, mag_val, dir_val, _bucket, dis_raw, best_c = item
        if abs(mag_val) < mag_th and abs(dir_val) < dir_th:
            continue
        t_w = t_valid[best_c.window_start : best_c.window_stop]
        y_w = y_valid[best_c.window_start : best_c.window_stop]
        pw_w = pw.slice_range(best_c.window_start, best_c.window_stop)
        p_jump, p_step = _nested_jump_ftests(
            t_w,
            y_w,
            pw_w,
            j_idx - best_c.window_start,
            best_c.selected_freqs,
            bonferroni=bonferroni_window,
        )
        if jump_alpha is not None and (np.isnan(p_jump) or p_jump >= jump_alpha):
            continue
        if step_alpha is not None and (np.isnan(p_step) or p_step >= step_alpha):
            continue
        accepted.append((j_idx, occ, mag_val, dir_val, dis_raw, p_jump, p_step, best_c))

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
    mag_th: float = 0.05,
    dir_th: float = 0.01,
    jump_th: float = 0.75,
    jump_alpha: float | None = None,
    step_alpha: float | None = None,
    bonferroni_window: bool = False,
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
        satisfying ``|mag| >= mag_th or |dir| >= dir_th``). Note: defaults
        ``0.05`` / ``0.01`` follow Ghaderpour (2020) for NDVI/EVI scale; a
        diagnostic warning is emitted automatically if mismatched to your data.
    jump_th:
        Minimum time separation between distinct jumps (default ``0.75``).
    jump_alpha:
        Optional nested partial F-test significance cutoff (e.g. ``0.05``).
        Defaults to ``None`` for Ghaderpour parity; ``jump_pvalue`` is always
        computed and stored on every :class:`JumpEvent`.
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
    if jump_alpha is not None and not (0.0 < jump_alpha < 1.0):
        raise ValueError(f"`jump_alpha` must lie in (0, 1); got {jump_alpha}.")
    if mag_th < 0.0 or dir_th < 0.0 or jump_th < 0.0:
        raise ValueError("`mag_th`, `dir_th`, and `jump_th` must be non-negative.")

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

    # Scale-mismatch diagnostic for default NDVI-tuned mag_th
    iqr = float(np.percentile(series.values, 75) - np.percentile(series.values, 25))
    if (
        not parity_mode
        and jump_alpha is None
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
                        "or enabling `jump_alpha=0.05` (or using `analyze_series()`)."
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

    if cfg is not None and isinstance(cfg.random_state, np.random.Generator):
        rng: np.random.Generator = cfg.random_state
    else:
        rng = np.random.default_rng(cfg.random_state if cfg is not None else 0)

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
        jump_alpha,
        step_alpha=step_alpha,
        bonferroni_window=bonferroni_window,
        parity_mode=parity_mode,
        provenance=prov,
        censor_cfg=cfg,
        rng=rng,
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
        dt_med = float(np.median(np.diff(series.times_numeric))) if series.n_valid > 1 else 0.1
        match_tol = max(0.35, 2.0 * dt_med)
        draw_series = generate_stability_draws(prov, cfg, series.times_numeric)
        for y_draw in draw_series:
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
                jump_alpha,
                step_alpha=step_alpha,
                bonferroni_window=bonferroni_window,
                parity_mode=parity_mode,
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
                    t_w, y_recen, pw_w, season_l, om_arr, alpha, parity_mode=parity_mode
                )
                p_r, _ = _nested_jump_ftests(t_w, y_recen, pw_w, loc_r, freqs_r)
                vanished = (abs(mag_r) < mag_th and abs(dir_r) < dir_th) or (p_r >= 0.05)
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
        jump_alpha=jump_alpha,
        censoring=prov,
        diagnostics=diag,
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
    censor_status: ArrayLike | None = None,
    censoring: CensoredConfig | None = None,
    parity_mode: bool = False,
) -> JustDecomposeResult:
    """Decompose an unevenly sampled time series into trend, seasonal, and remainder via JUST.

    Uses Gaussian-weighted overlapping window estimation with zero interpolation
    of the input observations. Optional ``eval_times`` evaluates the fitted
    parametric trend and seasonal models analytically onto any target grid.
    """
    season_l = season.lower()
    if season_l not in ("allssa", "ols", "none"):
        raise ValueError(f"Unknown `season` {season!r}. Options: 'allssa', 'ols', 'none'.")
    if window_mode not in ("count", "duration"):
        raise ValueError(f"Unknown `window_mode` {window_mode!r}. Options: 'count', 'duration'.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"`alpha` must lie in (0, 1); got {alpha}.")

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
    t_origin = float(tt_num[0])
    t0 = series.times_numeric - t_origin
    tt0 = tt_num - t_origin
    n = series.n_valid
    n_tt = int(tt0.size)

    if window_mode == "count":
        windows_tt, win_size, _win_step, rate_eff = _build_count_windows(
            n_tt, tt_num, size, step, parity_mode=parity_mode
        )
        dil_fac = float(np.log(0.1) / ((win_size / (6.0 * rate_eff)) ** 2))
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

    trends0 = np.zeros(n, dtype=np.float64)
    seasonals0 = np.zeros(n, dtype=np.float64)
    remainders0 = np.zeros(n, dtype=np.float64)
    sum_w0 = np.zeros(n, dtype=np.float64)

    trends_reg_acc = np.zeros(n_tt, dtype=np.float64)
    seasonals_reg_acc = np.zeros(n_tt, dtype=np.float64)
    sum_w_reg = np.zeros(n_tt, dtype=np.float64)

    win_freqs_list: list[FloatArray] = []
    cond_max = 1.0

    for lb, ub in windows_tt:
        tty = tt0[lb:ub]
        if has_eval:
            lb0 = int(np.searchsorted(t0, tt0[lb], side="left"))
            ub0 = min(n, int(np.searchsorted(t0, tt0[ub - 1], side="left")) + 1)
        else:
            lb0, ub0 = lb, ub

        size0 = ub0 - lb0
        ty = t0[lb0:ub0]
        yy = series.values[lb0:ub0]
        pw_w = pw.slice_range(lb0, ub0)
        local_j = np.array(
            [int(j - lb0) for j in j_valid if lb0 + 2 <= j < ub0 - 1], dtype=np.intp
        )

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
            )
            tr_coeff = res_a.trend_coeff
            cs_coeff = res_a.cos_sin_coeffs
            rem_w = res_a.residuals
            freqs_w = res_a.frequencies
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
            )
            rem_w = res_l.residuals
            n_f = int(om_arr.size)
            tr_coeff = res_l.coeff[: res_l.coeff.size - 2 * n_f]
            cs_coeff = res_l.coeff[res_l.coeff.size - 2 * n_f :].reshape(n_f, 2)
            freqs_w = om_arr
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
            )
            rem_w = res_l.residuals
            tr_coeff = res_l.coeff
            cs_coeff = np.empty((0, 2), dtype=np.float64)
            freqs_w = np.empty(0, dtype=np.float64)
            cond_max = max(cond_max, res_l.condition_number)

        win_freqs_list.append(freqs_w)
        season_t = _eval_seasonal(ty_loc, freqs_w, cs_coeff)
        trend_t = yy - season_t - rem_w
        mean_s = float(np.mean(season_t))
        trend_t = trend_t + mean_s
        season_t = season_t - mean_s

        tau0 = float(ty[size0 // 2]) if (size0 % 2 == 1) else 0.5 * float(ty[size0 // 2] + ty[size0 // 2 - 1])
        gauss_w = np.exp(dil_fac * (ty - tau0) ** 2)
        trends0[lb0:ub0] += trend_t * gauss_w
        seasonals0[lb0:ub0] += season_t * gauss_w
        remainders0[lb0:ub0] += rem_w * gauss_w
        sum_w0[lb0:ub0] += gauss_w

        if has_eval:
            trend_tt = _eval_piecewise_trend(
                ty_loc, tty_loc, local_j, tr_coeff, parity_mode=parity_mode
            )
            season_tt = _eval_seasonal(tty_loc, freqs_w, cs_coeff)
            mean_s_tt = float(np.mean(season_tt))
            trend_tt = trend_tt + mean_s_tt
            season_tt = season_tt - mean_s_tt
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

    with np.errstate(divide="ignore", invalid="ignore"):
        trend = np.where(sum_w0 > 0, trends0 / sum_w0, np.nan)
        seasonal = np.where(sum_w0 > 0, seasonals0 / sum_w0, np.nan)
        remainder = np.where(sum_w0 > 0, remainders0 / sum_w0, np.nan)
        if has_eval:
            trend_reg: FloatArray | None = np.where(
                sum_w_reg > 0, trends_reg_acc / sum_w_reg, np.nan
            )
            seasonal_reg: FloatArray | None = np.where(
                sum_w_reg > 0, seasonals_reg_acc / sum_w_reg, np.nan
            )
        else:
            trend_reg = None
            seasonal_reg = None

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
        is_datetime=series.is_datetime,
        season=season_l,
        window_mode=window_mode,
        censoring=prov,
        diagnostics=diag,
    )
