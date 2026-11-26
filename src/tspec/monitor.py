"""Near-Real-Time (NRT) disturbance monitoring via JUSTmonitor (v1.1).

Clean-room implementation of least-squares spectral near-real-time disturbance
detection (Ghaderpour & Vujadinovic, 2020b), enhanced with explicit
multi-frequency and sequential-peeking multiplicity controls, timestamp-aware
indexing, and integrated censored-data support via ``ndimpute``.

References
----------
- Ghaderpour, E., & Vujadinovic, T. (2020b). The Potential of the Least-Squares
  Spectral and Cross-Wavelet Analyses for Near-Real-Time Disturbance Detection.
  *Remote Sensing*, 12(15), 2446. https://doi.org/10.3390/rs12152446
- Lan, K. K. G., & DeMets, D. L. (1983). Discrete sequential boundaries for
  clinical trials. *Biometrika*, 70(3), 659-663.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import linalg
from scipy.stats import norm
from scipy.stats import t as t_dist

from ._diagnostics import (
    DiagnosticReport,
    WarningItem,
    compute_red_noise_diagnostics,
    time_unit_warning,
)
from ._linalg import (
    PreparedWeights,
    max_lss_batch,
    orthonormal_design_basis,
    prepare_weights,
    whiten_matrix,
    whiten_vector,
)
from ._robust import RobustMethod, tukey_weight, validate_robust_args
from ._time import TimeSeriesInput, convert_eval_times
from ._types import ArrayLike, FloatArray, TimeLike, ValueLike, WeightLike
from .censored import (
    CensoredConfig,
    CensoringProvenance,
    prepare_series_with_censoring,
)
from .exceptions import InsufficientDataError
from .results import BaseResult, require_pandas
from .spectral import _allssa_core, _lssa_core, build_design_matrix, critical_value_lss

DisturbanceType = Literal["none", "abrupt", "phenological"]
FreqCorrection = Literal["sidak", "bonferroni", "none"]
SequentialControl = Literal["pocock", "obrien_fleming", "consecutive_k", "none"]
MonitorWindowMode = Literal["expanding", "sliding"]

# sum_{k>=2} 1 / (k ln(k)^2)
_OPEN_SPEND_C = 2.1097428012368
# Number of resampled future sampling grids in the consecutive_k calibration
_CONSEC_GRID_GROUPS = 25


@dataclass(frozen=True)
class MonitorStep:
    """Per-observation diagnostic record during near-real-time monitoring.

    Attributes
    ----------
    step : int
        1-based monitoring step number (``d = 1, 2, ...``).
    index : int
        0-based index in the original input time axis.
    time : Any
        Observation timestamp in the user's original time representation.
    time_year : float
        Observation timestamp in continuous decimal years.
    max_low_freq_spec : float
        Maximum normalized LSS over low frequencies (``< 1.0`` cy/yr),
        sensitive to abrupt mean/level disturbances.
    max_high_freq_spec : float
        Maximum normalized LSS over intra-annual frequencies (``>= 1.0`` cy/yr),
        sensitive to phenological/seasonal disturbances.
    max_spec : float
        Overall maximum normalized LSS across ``monitor_omega``.
    peak_freq : float
        Frequency (cycles/year) attaining ``max_spec``.
    critical_value : float
        Applied Beta critical threshold at this step after any frequency
        and sequential multiplicity adjustments.
    alpha_step : float
        Per-step familywise significance budget before multi-frequency adjustment.
    alpha_adjusted : float
        Per-frequency significance level passed to the Beta critical formula.
    exceeded : bool
        Whether ``max_spec > critical_value`` at this step.
    candidate_type : {"none", "abrupt", "phenological"}
        Disturbance classification at this step before consecutive confirmation.
    consecutive_count : int
        Current streak of consecutive exceedances up to and including this step.
    """

    step: int
    index: int
    time: Any
    time_year: float
    max_low_freq_spec: float
    max_high_freq_spec: float
    max_spec: float
    peak_freq: float
    critical_value: float
    alpha_step: float
    alpha_adjusted: float
    exceeded: bool
    candidate_type: DisturbanceType
    consecutive_count: int


@dataclass
class MonitorResult(BaseResult):
    """Result object returned by :func:`just_monitor`.

    Attributes
    ----------
    detected : bool
        ``True`` if a confirmed disturbance (abrupt or phenological) was detected.
    disturbance_type : {"none", "abrupt", "phenological"}
        Plain-English classification of the detected disturbance.
    type_code : int
        Integer code matching Ghaderpour & Vujadinovic (2020b):
        ``0`` = none, ``1`` = abrupt change, ``2`` = phenological change.
    disturbance_index : int | None
        0-based index in the original input axis where the disturbance triggered.
    disturbance_time : Any | None
        Timestamp of the disturbance in the user's original time representation.
    disturbance_time_year : float | None
        Timestamp of the disturbance in decimal years.
    magnitude : float | None
        Estimated vertical step magnitude if ``disturbance_type == "abrupt"`` and
        at least 3 observations remain from the disturbance onward; else ``None``.
    direction : float | None
        Estimated post-minus-pre slope change if ``disturbance_type == "abrupt"``,
        ``trend == "linear"``, and enough post-disturbance points exist; else ``None``.
    dominant_frequency : float | None
        Frequency (cy/yr) that triggered the disturbance alarm.
    pending_confirmation : bool
        ``True`` when ``sequential_control="consecutive_k"`` saw an exceedance
        at the final observation whose streak has not yet reached ``consecutive_k``.
    history_index : int
        0-based original index where the stable historical period begins.
    start_index : int
        0-based original index where the monitoring period begins.
    history_time : Any
        Original timestamp of ``history_index``.
    start_time : Any
        Original timestamp of ``start_index``.
    times : Any
        Original timestamps over the active slice ``history:n``.
    time_years : FloatArray
        Continuous decimal years over the active slice ``history:n``.
    values : FloatArray
        Observed (or imputed) values over the active slice ``history:n``.
    forecast : FloatArray
        Historical trend + seasonal model evaluated across ``history:n``.
    residuals : FloatArray
        ``values - forecast`` over ``history:n``.
    historical_frequencies : FloatArray
        Harmonic frequencies (cy/yr) fitted on the stable historical period.
    historical_trend_coeff : FloatArray
        Intercept and slope ``[b0, b1]`` fitted on the stable historical period
        (relative to zero-shifted time).
    steps : list[MonitorStep]
        Step-by-step monitoring trace across evaluated monitoring observations.
    alpha : float
        Nominal significance level requested by the caller.
    freq_correction : {"sidak", "bonferroni", "none"}
        Multi-frequency adjustment applied at each monitoring step.
    sequential_control : {"pocock", "obrien_fleming", "consecutive_k", "none"}
        Across-time repeated-testing control applied across monitoring steps.
    horizon : int
        Monitoring horizon ``D_max`` used for alpha-spending rules (``0`` = open-ended
        spending because no horizon was supplied).
    diagnostics : DiagnosticReport
        Structured diagnostics and warnings.
    censoring : CensoringProvenance | None
        Censored-data provenance if censored inputs were provided.
    ci_level : float
        Confidence level of the forecast bands.
    forecast_ci_low, forecast_ci_high, forecast_pi_low, forecast_pi_high : FloatArray | None
        Pointwise mean-forecast confidence band and observation prediction interval.
    consecutive_k : int
        Number of consecutive exceedances required under ``"consecutive_k"``.
    consecutive_k_calibrated : bool
        ``True`` when the ``"consecutive_k"`` per-step level was calibrated by Monte
        Carlo over the planned ``horizon`` (reported as ``steps[i].alpha_step``).
    """

    detected: bool
    disturbance_type: DisturbanceType
    type_code: int
    disturbance_index: int | None
    disturbance_time: Any | None
    disturbance_time_year: float | None
    magnitude: float | None
    direction: float | None
    dominant_frequency: float | None
    pending_confirmation: bool
    history_index: int
    start_index: int
    history_time: Any
    start_time: Any
    times: Any
    time_years: FloatArray
    values: FloatArray
    forecast: FloatArray
    residuals: FloatArray
    historical_frequencies: FloatArray
    historical_trend_coeff: FloatArray
    steps: list[MonitorStep]
    alpha: float
    freq_correction: FreqCorrection
    sequential_control: SequentialControl
    horizon: int
    diagnostics: DiagnosticReport
    censoring: CensoringProvenance | None = None
    ci_level: float = 0.95
    forecast_ci_low: FloatArray | None = None
    forecast_ci_high: FloatArray | None = None
    forecast_pi_low: FloatArray | None = None
    forecast_pi_high: FloatArray | None = None
    consecutive_k: int = 2
    consecutive_k_calibrated: bool = False

    def summary(self) -> str:
        lines = [
            f"JUST Near-Real-Time Monitor (steps evaluated={len(self.steps)}, "
            f"alpha={self.alpha:g}, freq_correction={self.freq_correction}, "
            f"sequential_control={self.sequential_control})",
            f"  Stable history start: idx={self.history_index} (t={self.history_time})",
            f"  Monitoring start:     idx={self.start_index} (t={self.start_time})",
        ]
        if self.detected:
            mag_str = f"{self.magnitude:+.4g}" if self.magnitude is not None else "n/a"
            dir_str = f"{self.direction:+.4g}" if self.direction is not None else "n/a"
            freq_str = f"{self.dominant_frequency:.2f} cy/yr" if self.dominant_frequency is not None else "n/a"
            lines.append(
                f"  ALARM: {self.disturbance_type.upper()} disturbance (type={self.type_code}) "
                f"at idx={self.disturbance_index}, t={self.disturbance_time} "
                f"(peak_freq={freq_str}, mag={mag_str}, slope_change={dir_str})"
            )
        elif self.pending_confirmation:
            lines.append(
                "  Status: PENDING CONFIRMATION (final observation exceeded threshold; "
                "awaiting next observation(s) to confirm)."
            )
        else:
            lines.append("  Status: No statistically significant disturbance detected.")
        if self.diagnostics.warnings:
            lines.append(f"  Warnings ({len(self.diagnostics.warnings)}):")
            for w in self.diagnostics.warnings:
                lines.append(f"    - [{w.severity.upper()}] {w.code}: {w.message}")
        return "\n".join(lines)

    def explain(self) -> str:
        parts: list[str] = []
        if self.detected:
            if self.disturbance_type == "abrupt":
                detail = (
                    f"Estimated step magnitude is {self.magnitude:+.4g}"
                    if self.magnitude is not None
                    else "Too few observations follow the alarm yet to estimate step magnitude reliably"
                )
                if self.direction is not None:
                    detail += f" with slope change {self.direction:+.4g}/yr"
                parts.append(
                    f"An ABRUPT disturbance was flagged in near-real time at {self.disturbance_time} "
                    f"(dominant residual frequency {self.dominant_frequency:.2f} cy/yr < 1 cy/yr). "
                    f"{detail}."
                )
            else:
                parts.append(
                    f"A PHENOLOGICAL (intra-annual / seasonal) disturbance was flagged at "
                    f"{self.disturbance_time} (dominant residual frequency "
                    f"{self.dominant_frequency:.2f} cy/yr >= 1 cy/yr), indicating a shift in "
                    f"seasonal timing or amplitude rather than a flat baseline jump."
                )
        elif self.pending_confirmation:
            parts.append(
                "No disturbance is confirmed yet, but the most recent observation exceeded "
                "the monitoring threshold. Wait for the next incoming sample(s) to confirm "
                "whether this is a real disturbance or a single-sample outlier."
            )
        else:
            parts.append(
                f"No statistically significant disturbance was detected across {len(self.steps)} "
                f"monitoring observation(s) starting at {self.start_time}."
            )

        if self.freq_correction == "none" and self.sequential_control == "none":
            parts.append(
                "Note: repeated-testing multiplicity controls are disabled (parity/unadjusted mode). "
                "Over long monitoring horizons, unadjusted per-step testing increases cumulative "
                "false-alarm probability."
            )
        elif self.sequential_control == "consecutive_k" and self.consecutive_k_calibrated:
            a_step = self.steps[0].alpha_step if self.steps else float("nan")
            parts.append(
                f"Repeated-testing protection active: an alarm needs {self.consecutive_k} consecutive "
                f"exceedances at a Monte Carlo-calibrated per-step level {a_step:.3g}, which bounds "
                f"the familywise false-alarm rate at alpha={self.alpha:g} over the planned horizon of "
                f"{self.horizon} steps (assuming Gaussian errors and history-like sampling)."
            )
        elif self.sequential_control == "consecutive_k":
            parts.append(
                f"Note: sequential_control='consecutive_k' without `horizon` requires repeated "
                f"exceedances but does not bound the familywise false-alarm rate at "
                f"alpha={self.alpha:g}; it grows with the monitoring length (about 10% over 96 steps "
                "at alpha=0.05 in simulations)."
            )
        else:
            parts.append(
                f"Repeated-testing protection active: freq_correction='{self.freq_correction}' "
                f"and sequential_control='{self.sequential_control}' bound the familywise "
                f"false-alarm rate at alpha={self.alpha:g}."
            )

        for w in self.diagnostics.warnings:
            parts.append(f"Warning ({w.code}): {w.message}")
        return " ".join(parts)

    def to_dataframe(self) -> Any:
        pd = require_pandas()
        rows = [
            {
                "step": s.step,
                "index": s.index,
                "time": s.time,
                "time_year": s.time_year,
                "max_low_freq_spec": s.max_low_freq_spec,
                "max_high_freq_spec": s.max_high_freq_spec,
                "max_spec": s.max_spec,
                "peak_freq": s.peak_freq,
                "critical_value": s.critical_value,
                "alpha_step": s.alpha_step,
                "alpha_adjusted": s.alpha_adjusted,
                "exceeded": s.exceeded,
                "candidate_type": s.candidate_type,
                "consecutive_count": s.consecutive_count,
            }
            for s in self.steps
        ]
        return pd.DataFrame(rows)

    def plot(self, ax: Any = None, **kwargs: Any) -> tuple[Any, Any]:
        from .plotting import plot_monitor

        return plot_monitor(self, ax=ax, **kwargs)


def _resolve_boundary_index(
    bound: int | float | Any,
    series: TimeSeriesInput,
    *,
    name: str,
    default_valid_idx: int,
) -> int:
    """Resolve an integer original index OR timestamp/year into a valid-series index."""
    if bound is None:
        return default_valid_idx

    if isinstance(bound, (int, np.integer)) and not isinstance(bound, bool):
        orig_idx = int(bound)
        if orig_idx < 0:
            orig_idx += series.n_input
        if orig_idx < 0 or orig_idx >= series.n_input:
            raise ValueError(f"{name}={bound} is out of bounds for series of length {series.n_input}.")
        pos = int(np.searchsorted(series.valid_to_orig, orig_idx, side="left"))
        if pos >= series.n_valid:
            raise ValueError(f"{name}={bound} falls after all finite observations.")
        return pos

    # Treat as a timestamp or float decimal year
    t_num, _ = convert_eval_times(
        [bound],
        float(series.times_numeric[0]),
        float(series.times_numeric[-1]),
        expect_datetime=series.is_datetime,
    )
    target_year = float(t_num[0])
    pos = int(np.searchsorted(series.times_numeric, target_year, side="left"))
    if pos >= series.n_valid:
        raise ValueError(f"{name}={bound!r} falls after the last observation ({series.times_raw[-1]!r}).")
    return pos


def _step_alpha_budget(
    step_d: int,
    horizon: int,
    alpha: float,
    sequential_control: SequentialControl,
) -> float:
    """Return the per-step familywise alpha budget at 1-based step ``step_d``."""
    if sequential_control in ("none", "consecutive_k"):
        return alpha
    if horizon <= 0 and sequential_control in ("pocock", "obrien_fleming"):
        # Open-ended spending (no planned horizon): alpha_d = alpha / (C (d+1) ln^2(d+1)),
        # C = sum_{k>=2} 1/(k ln^2 k) = 2.1097428..., so sum_d alpha_d = alpha over an
        # unbounded horizon and thresholds never depend on future data.
        k = float(step_d) + 1.0
        return float(alpha / (_OPEN_SPEND_C * k * np.log(k) ** 2))
    h = max(1, int(horizon))
    if sequential_control == "pocock":
        # Exact Sidak-Pocock constant per-step allocation: 1 - (1 - alpha)^(1 / H)
        return float(1.0 - np.power(1.0 - alpha, 1.0 / h))
    if sequential_control == "obrien_fleming":
        # Lan-DeMets (1983) O'Brien-Fleming alpha-spending function
        # alpha*(u) = 2 * (1 - Phi(z_{alpha/2} / sqrt(u))), u in (0, 1].
        # Past the planned horizon (step_d > h), retain the final planned step's
        # budget rather than collapsing to 1e-15 (a HORIZON_EXCEEDED warning is emitted).
        z_half = float(norm.isf(0.5 * alpha))
        d_eff = min(step_d, h)
        u_curr = d_eff / h
        u_prev = (d_eff - 1) / h
        spend_curr = float(2.0 * norm.sf(z_half / np.sqrt(u_curr))) if u_curr > 0.0 else 0.0
        spend_prev = float(2.0 * norm.sf(z_half / np.sqrt(u_prev))) if u_prev > 0.0 else 0.0
        inc_alpha = max(1e-15, spend_curr - spend_prev)
        return min(alpha, inc_alpha)
    raise ValueError(
        "sequential_control must be one of 'pocock', 'obrien_fleming', 'consecutive_k', or 'none'."
    )


def _adjust_alpha_for_frequencies(
    alpha_step: float,
    n_freqs: int,
    freq_correction: FreqCorrection,
) -> float:
    """Adjust a per-step alpha budget across ``n_freqs`` scanned frequencies."""
    k = max(1, int(n_freqs))
    if freq_correction == "none" or k == 1:
        return alpha_step
    if freq_correction == "bonferroni":
        return max(1e-15, alpha_step / k)
    if freq_correction == "sidak":
        return max(1e-15, float(1.0 - np.power(1.0 - alpha_step, 1.0 / k)))
    raise ValueError("freq_correction must be 'sidak', 'bonferroni', or 'none'.")


def _step_pvalues_from_spectrum(
    smax: FloatArray, n_w: int, n_freqs: int, freq_correction: FreqCorrection
) -> FloatArray:
    """Per-step p-values (on the ``alpha_step`` scale) of the maximum normalized LSS.

    Inverts the monitoring rule ``smax > critical_value_lss(n_w, 1, alpha_adj)`` with
    ``alpha_adj`` the frequency-adjusted budget, so ``p <= alpha_step`` iff the step
    exceeds its threshold.
    """
    nu = n_w - 3
    if nu <= 0:
        return np.ones_like(smax)
    p_min = np.power(np.clip(1.0 - smax, 0.0, 1.0), 0.5 * nu)
    k = max(1, int(n_freqs))
    if freq_correction == "none" or k == 1:
        return np.asarray(p_min, dtype=np.float64)
    if freq_correction == "bonferroni":
        return np.asarray(np.minimum(1.0, k * p_min), dtype=np.float64)
    return np.asarray(-np.expm1(k * np.log1p(-np.minimum(p_min, 1.0 - 1e-16))), dtype=np.float64)


def _calibrate_consecutive_k(
    t_his: FloatArray,
    pw_his: PreparedWeights,
    his_freqs: FloatArray,
    t_first: float,
    horizon: int,
    lh: int,
    slide_width: int,
    window_mode: MonitorWindowMode,
    om_mon: FloatArray,
    freq_correction: FreqCorrection,
    k: int,
    alpha: float,
    n_mc: int,
    forecast_cov: bool,
    seed: int = 0,
) -> float | None:
    """Monte Carlo per-step level for ``consecutive_k`` with family-wise control.

    Simulates the monitoring residual process under the fitted history model with
    Gaussian errors (the history design is refitted in every replicate; selected
    harmonics are held fixed). Future sampling times start at the first monitoring
    time and follow gaps resampled from the history (a separate grid for each of up to
    25 groups of replicates; a regular history gives a regular grid), with the
    history's median weight, so the level depends only on the history, the first
    monitoring time and the planned ``horizon``.
    For each replicate the smallest per-step level at which ``k`` consecutive steps
    would all exceed is recorded; the returned level is a lower Monte Carlo quantile
    of these, so ``P(alarm within horizon) <= alpha`` under the simulated null.
    Returns ``None`` when the weights cannot be extrapolated (full weight matrices) or
    the history spacing is degenerate.
    """
    n_his = int(t_his.size)
    if pw_his.kind == "matrix":
        return None
    if pw_his.kind == "none":
        w_his = np.ones(n_his, dtype=np.float64)
    elif pw_his.kind == "scalar":
        w_his = np.full(n_his, pw_his.scalar, dtype=np.float64)
    else:
        assert pw_his.vector is not None
        w_his = np.asarray(pw_his.vector, dtype=np.float64)
    dt = np.diff(t_his)
    dt = dt[dt > 0.0]
    if dt.size == 0 or horizon < 1:
        return None
    if k > horizon:
        return alpha
    w_fut = np.full(horizon, float(np.median(w_his)))
    w_c = np.concatenate([w_his, w_fut])
    sw_c = np.sqrt(w_c)
    pw_c = PreparedWeights(kind="vector", vector=w_c, sqrt_vector=sw_c)
    rng = np.random.default_rng(seed)
    n_mc = int(n_mc)
    pvals = np.empty((horizon, n_mc), dtype=np.float64)
    # Future sampling times are unknown when monitoring starts: each group of replicates
    # uses its own grid built from gaps resampled from the history (a regular history
    # gives a regular grid).
    n_groups = 1 if np.ptp(dt) <= 1e-9 * float(np.median(dt)) else min(_CONSEC_GRID_GROUPS, n_mc)
    bounds = np.linspace(0, n_mc, n_groups + 1).astype(int)
    for g in range(n_groups):
        b0, b1 = int(bounds[g]), int(bounds[g + 1])
        gaps = dt if n_groups == 1 else rng.choice(dt, size=horizon)
        offs = np.concatenate([[0.0], np.cumsum(gaps[: horizon - 1])]) if n_groups > 1 else (
            float(np.median(dt)) * np.arange(horizon, dtype=np.float64)
        )
        t_c = np.concatenate([t_his, float(t_first) + offs])
        d_c, _ = build_design_matrix(t_c, "linear", np.empty(0, dtype=np.intp), False, his_freqs)
        wd_his = sw_c[:n_his, None] * d_c[:n_his]
        u_h, s_h, vt_h = linalg.svd(wd_his, full_matrices=False)
        if s_h.size == 0 or s_h[0] <= 0.0:
            return None
        inv_s = np.where(s_h > 1e-12 * s_h[0], 1.0 / s_h, 0.0)
        f_his = vt_h.T * inv_s[None, :]

        z = rng.standard_normal((t_c.size, b1 - b0))
        coef = f_his @ (u_h.T @ z[:n_his])
        resid = z / sw_c[:, None] - d_c @ coef
        v_out = np.asarray(d_c[n_his:] @ f_his, dtype=np.float64) if forecast_cov else None

        for d in range(1, horizon + 1):
            win_end = n_his + d
            win_start = lh if window_mode == "expanding" else max(0, win_end - slide_width)
            idx = np.arange(win_start, win_end, dtype=np.intp)
            pw_w = pw_c.subset(idx)
            if v_out is not None:
                n_in = max(0, n_his - win_start)
                pw_w = _build_predictive_window_weights(
                    pw_w, n_in, v_out[max(0, win_start - n_his) : d]
                )
            q = orthonormal_design_basis(np.ones((idx.size, 1), dtype=np.float64), pw_w)
            smax = max_lss_batch(t_c[idx], om_mon, q, pw_w, whiten_matrix(resid[idx], pw_w))
            pvals[d - 1, b0:b1] = _step_pvalues_from_spectrum(
                smax, int(idx.size), int(om_mon.size), freq_correction
            )

    # Smallest level at which k consecutive steps all exceed, per replicate
    run_level = np.full(n_mc, np.inf)
    for d in range(k - 1, horizon):
        run_level = np.minimum(run_level, np.max(pvals[d - k + 1 : d + 1], axis=0))
    j = int(np.floor(alpha * (n_mc + 1)))
    if j < 1:
        return None
    thr = float(np.sort(run_level)[j - 1])
    return float(min(1.0, np.nextafter(thr, 0.0)))


def _build_predictive_window_weights(
    pw_w: PreparedWeights,
    n_in: int,
    v_out: FloatArray,
) -> PreparedWeights:
    """Build exact predictive precision weights accounting for historical parameter uncertainty.

    For monitoring residuals ``e_out = y_out - D_out beta_hat``, orthogonality of
    historical OLS residuals implies ``Cov(e_in, e_out) = 0`` identically under
    independent errors, while
    ``Cov(e_out) = sigma^2 (P_out^{-1} + V_out V_out^T)``
    where ``V_out = D_out (D_his^T P_his D_his)^{-1/2}``.
    """
    n_out = int(v_out.shape[0])
    n_w = n_in + n_out
    if n_out == 0:
        return pw_w

    cov_extra = v_out @ v_out.T
    if pw_w.kind == "matrix":
        assert pw_w.matrix is not None and pw_w.cholesky_lower is not None
        sigma_w = linalg.cho_solve((pw_w.cholesky_lower, True), np.eye(n_w, dtype=np.float64))
        sigma_w[n_in:, n_in:] += cov_extra
        sigma_w = 0.5 * (sigma_w + sigma_w.T)
        l_sig = linalg.cholesky(sigma_w, lower=True)
        p_win = linalg.cho_solve((l_sig, True), np.eye(n_w, dtype=np.float64))
        p_win = 0.5 * (p_win + p_win.T)
        l_win = linalg.cholesky(p_win, lower=True)
        return PreparedWeights(kind="matrix", matrix=p_win, cholesky_lower=l_win)

    if pw_w.kind == "none":
        inv_w_out = np.ones(n_out, dtype=np.float64)
        w_in_diag = np.ones(n_in, dtype=np.float64)
    elif pw_w.kind == "scalar":
        inv_w_out = np.full(n_out, 1.0 / pw_w.scalar, dtype=np.float64)
        w_in_diag = np.full(n_in, pw_w.scalar, dtype=np.float64)
    else:
        assert pw_w.vector is not None
        inv_w_out = 1.0 / pw_w.vector[n_in:]
        w_in_diag = pw_w.vector[:n_in]

    sigma_out = cov_extra.copy()
    sigma_out[np.diag_indices(n_out)] += inv_w_out
    sigma_out = 0.5 * (sigma_out + sigma_out.T)
    l_sig_out = linalg.cholesky(sigma_out, lower=True)
    p_out = linalg.cho_solve((l_sig_out, True), np.eye(n_out, dtype=np.float64))
    p_out = 0.5 * (p_out + p_out.T)
    l_out = linalg.cholesky(p_out, lower=True)

    p_win = np.zeros((n_w, n_w), dtype=np.float64)
    l_win = np.zeros((n_w, n_w), dtype=np.float64)
    if n_in > 0:
        idx_in = np.arange(n_in)
        p_win[idx_in, idx_in] = w_in_diag
        l_win[idx_in, idx_in] = np.sqrt(w_in_diag)
    p_win[n_in:, n_in:] = p_out
    l_win[n_in:, n_in:] = l_out
    return PreparedWeights(kind="matrix", matrix=p_win, cholesky_lower=l_win)


def just_monitor(
    t: TimeLike,
    y: ValueLike,
    *,
    weights: WeightLike | None = None,
    start: int | float | Any | None = None,
    history: int | float | Any = 0,
    h: float = 0.25,
    season: Literal["allssa", "ols", "none"] = "allssa",
    omega: ArrayLike | None = None,
    monitor_omega: ArrayLike | None = None,
    trend: Literal["linear", "constant"] = "linear",
    alpha: float = 0.01,
    ci_level: float = 0.95,
    freq_correction: FreqCorrection = "sidak",
    sequential_control: SequentialControl = "pocock",
    consecutive_k: int = 2,
    horizon: int | None = None,
    window_mode: MonitorWindowMode = "expanding",
    stop_on_first: bool = True,
    account_for_forecast_uncertainty: bool = True,
    robust: RobustMethod = "none",
    huber_k: float = 1.345,
    censor_status: ArrayLike | None = None,
    censoring: CensoredConfig | None = None,
    parity_mode: bool = False,
    season_significance: Literal["sidak", "montecarlo", "per_frequency"] = "sidak",
    n_mc: int = 999,
) -> MonitorResult:
    """Near-Real-Time (NRT) disturbance detection and classification (JUSTmonitor).

    Fits a stable historical trend-and-seasonal model on ``t[history:start]``,
    forecasts it forward into the monitoring period ``t[start:]``, and applies
    Least-Squares Spectral Analysis (LSSA) to the forecast residuals as each
    new observation arrives.

    - A statistically significant spectral peak at **low frequencies**
      (``< 1.0`` cy/yr) flags an **abrupt disturbance** (step jump / rapid drop).
    - A statistically significant peak exclusively at **intra-annual frequencies**
      (``>= 1.0`` cy/yr) flags a **phenological disturbance** (seasonal timing
      or amplitude change).

    Unlike unadjusted repeated testing—where scanning 35 frequencies at every
    new observation inflates cumulative false alarms rapidly—``tspec`` enables
    multi-frequency Šidák correction (``freq_correction="sidak"``), Pocock
    sequential alpha-spending (``sequential_control="pocock"``), and exact
    out-of-sample predictive covariance weighting
    (``account_for_forecast_uncertainty=True``) by default so the familywise
    false-alarm rate over the monitoring horizon stays bounded by ``alpha``
    even when forecasting beyond a finite historical record. Set
    ``parity_mode=True`` for exact numerical parity with Ghaderpour &
    Vujadinovic (2020b).

    Parameters
    ----------
    t : array-like
        Strictly increasing timestamps (numeric years, ``datetime64``,
        ``DatetimeIndex``, or ``datetime``/``date``).
    y : array-like
        Observations (numeric or censored strings such as ``"<0.05"``).
    weights : float, 1D array, or 2D SPD array, optional
        Observation weights or inverse covariance matrix ``P``.
    start : int, float, or datetime-like, optional
        Start of the monitoring period, specified either as a 0-based integer
        index into ``t`` or as a timestamp/decimal year. Defaults to ``n - 2``.
    history : int, float, or datetime-like, default 0
        Start of the stable historical period (0-based index or timestamp/year).
    h : float, default 0.25
        Fraction of the stable historical period retained immediately before
        ``start`` as the monitoring baseline context. Must satisfy ``0 <= h <= 1``.
    season : {"allssa", "ols", "none"}, default "allssa"
        Seasonal model fitted on the stable historical period (and for post-alarm
        jump magnitude estimation).
    omega : array-like, optional
        Candidate cyclic frequencies (cy/yr) for the historical seasonal fit.
        Defaults to ``[0.8, 1.0, ..., 3.8]`` for ``"allssa"`` and ``[1, 2, 3, 4]``
        for ``"ols"``.
    monitor_omega : array-like, optional
        Frequencies (cy/yr) scanned in the monitoring residual LSS. Defaults to
        ``[0.1, 0.2, ..., 3.5]`` (35 frequencies).
    trend : {"linear", "constant"}, default "linear"
        Post-disturbance jump model: ``"linear"`` estimates both jump magnitude
        and slope change; ``"constant"`` estimates jump magnitude only.
    alpha : float, default 0.01
        Nominal significance level in ``(0, 1)``.
    ci_level : float, default 0.95
        Nominal pointwise confidence and prediction interval level in ``(0, 1)``
        for ``forecast_ci_low/high`` and ``forecast_pi_low/high``.
    freq_correction : {"sidak", "bonferroni", "none"}, default "sidak"
        Per-step correction across ``monitor_omega`` frequencies. Forced to
        ``"none"`` when ``parity_mode=True``.
    sequential_control : {"pocock", "obrien_fleming", "consecutive_k", "none"}, default "pocock"
        Repeated-testing control across monitoring steps:
        - ``"pocock"``: constant Šidák alpha-spending across ``horizon`` steps.
        - ``"obrien_fleming"``: Lan–DeMets O'Brien–Fleming alpha-spending
          (very conservative early, higher power later in the horizon).
        - ``"consecutive_k"``: requires ``consecutive_k`` consecutive exceedances. With
          ``horizon``, the constant per-step level is calibrated by Monte Carlo so that
          the probability of any alarm within the horizon is at most ``alpha`` under the
          fitted history model with Gaussian errors (future sampling gaps are resampled
          from the history, so the level uses no monitoring data; ALLSSA-selected
          harmonics are held fixed, and robust fits are simulated as weighted least
          squares with the robust history weights). Without ``horizon``, each step is tested at the full
          ``alpha``, which does **not** control the familywise rate (warning
          ``UNCALIBRATED_CONSECUTIVE_K``).
        - ``"none"``: unadjusted per-step testing (used by ``parity_mode=True``).
    consecutive_k : int, default 2
        Number of consecutive exceedances required when
        ``sequential_control="consecutive_k"``.
    horizon : int, optional
        Total number of planned monitoring steps ``D_max`` for ``"pocock"`` or
        ``"obrien_fleming"`` spending, and the calibration horizon for
        ``"consecutive_k"``. It must be fixed in advance: if omitted,
        open-ended spending ``alpha_d = alpha / (2.11 (d+1) ln^2(d+1))`` is used (reported as
        ``horizon=0``) so thresholds never depend on how many future observations
        exist. ``parity_mode=True`` keeps the legacy default (number of monitoring
        observations).
    window_mode : {"expanding", "sliding"}, default "expanding"
        Whether the monitoring residual window expands from ``Lh`` as new points
        arrive (``"expanding"``, matching Ghaderpour & Vujadinovic, 2020b) or
        slides with fixed sample count (``"sliding"``).
    stop_on_first : bool, default True
        If ``True``, halt monitoring on the first confirmed disturbance (standard
        online NRT behavior). If ``False``, continue recording ``steps`` through
        the end of the series while retaining the first confirmed alarm.
    account_for_forecast_uncertainty : bool, default True
        If ``True`` (default), weights each monitoring window using the exact
        predictive residual covariance
        ``Cov(e_out) = sigma^2 (P_out^{-1} + D_out (D_his^T P_his D_his)^{-1} D_out^T)``,
        preventing finite-sample historical slope/seasonal estimation errors
        from triggering false alarms when forecasting far past ``start``.
        Forced to ``False`` when ``parity_mode=True``.
    robust : {"none", "huber", "mm"}, default "none"
        If ``"huber"`` or ``"mm"``, fits the stable historical baseline via
        convex Huber IRLS or Huber-started Tukey biweight (down-weighting historical spikes
        without ever down-weighting incoming monitoring observations so new
        disturbances are never masked). Forced to ``"none"`` when
        ``parity_mode=True``.
    huber_k : float, default 1.345
        Huber tuning constant used when ``robust`` is ``"huber"`` or ``"mm"``.
    censor_status : array-like, optional
        Optional censoring indicators (``-1`` left, ``0`` detected, ``+1`` right).
    censoring : CensoredConfig, optional
        Configuration for ``ndimpute`` censored-data handling. Fitted strictly
        causally on ``t[history:start]`` so future monitoring observations never
        leak into historical imputation.
    season_significance : {"sidak", "montecarlo", "per_frequency"}, default "sidak"
        Acceptance rule for ALLSSA-selected historical harmonics (``season="allssa"``):
        family-wise Sidak (default), exact Monte Carlo, or the published per-frequency
        rule (forced by ``parity_mode=True``).
    n_mc : int, default 999
        Monte Carlo replicates for the ``"consecutive_k"`` calibration (requires
        ``(n_mc + 1) * alpha >= 5``).
    parity_mode : bool, default False
        If ``True``, disables multiplicity controls (``freq_correction="none"``,
        ``sequential_control="none"``), disables predictive covariance weighting
        (``account_for_forecast_uncertainty=False``), disables robust
        reweighting (``robust="none"``), and forces ``window_mode="expanding"``
        for exact numerical parity with Ghaderpour & Vujadinovic (2020b).

    Returns
    -------
    MonitorResult

    References
    ----------
    Ghaderpour, E., & Vujadinovic, T. (2020b). *Remote Sensing*, 12(15), 2446.
    https://doi.org/10.3390/rs12152446
    """
    if not (0.0 <= h <= 1.0):
        raise ValueError(f"h must be in [0, 1], got {h}.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if not (0.0 < ci_level < 1.0):
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}.")
    if consecutive_k < 1:
        raise ValueError(f"consecutive_k must be >= 1, got {consecutive_k}.")
    if sequential_control == "consecutive_k" and horizon is not None and (int(n_mc) + 1) * alpha < 5:
        raise ValueError(
            f"`n_mc` too small for alpha={alpha}: need (n_mc + 1) * alpha >= 5, got n_mc={n_mc}."
        )

    season_norm = season.lower()
    if season_norm not in ("allssa", "ols", "none"):
        raise ValueError("season must be 'allssa', 'ols', or 'none'.")
    trend_norm = trend.lower()
    if trend_norm not in ("linear", "constant"):
        raise ValueError("trend must be 'linear' or 'constant'.")

    eff_freq_corr: FreqCorrection = "none" if parity_mode else freq_correction
    eff_seq_ctrl: SequentialControl = "none" if parity_mode else sequential_control
    eff_win_mode: MonitorWindowMode = "expanding" if parity_mode else window_mode
    eff_forecast_cov = False if parity_mode else bool(account_for_forecast_uncertainty)
    eff_robust: RobustMethod = "none" if parity_mode else validate_robust_args(robust, huber_k)
    if season_significance not in ("sidak", "montecarlo", "per_frequency"):
        raise ValueError(
            f"Unknown `season_significance` {season_significance!r}. "
            "Options: 'sidak', 'montecarlo', 'per_frequency'."
        )
    season_sig = "per_frequency" if parity_mode else season_significance

    def _causal_split(pre_series: TimeSeriesInput) -> tuple[int, int]:
        n_pre = pre_series.n_valid
        if n_pre < 5:
            raise InsufficientDataError(
                f"just_monitor requires at least 5 observations, got n={n_pre}."
            )
        h_i = _resolve_boundary_index(history, pre_series, name="history", default_valid_idx=0)
        s_i = _resolve_boundary_index(start, pre_series, name="start", default_valid_idx=n_pre - 2)
        if s_i >= n_pre:
            raise ValueError("Monitoring start index must be strictly less than series length.")
        if h_i >= s_i:
            raise ValueError("history start must precede monitoring start.")
        return h_i, s_i

    prepared = prepare_series_with_censoring(
        t,
        y,
        censor_status=censor_status,
        censoring=censoring,
        causal_split_fn=_causal_split,
    )
    series = prepared.series
    t0_full = series.times_numeric - series.times_numeric[0]
    y_full = series.values

    pw_raw = prepare_weights(weights, series.n_input)
    pw_full = pw_raw.subset(series.valid_mask) if series.n_dropped_nan > 0 else pw_raw
    if prepared.weight_factors is not None:
        pw_full = pw_full.scale_entries(prepared.weight_factors)

    warnings_list: list[WarningItem] = list(prepared.warnings)
    warnings_list.extend(time_unit_warning(series.times_numeric, series.is_datetime))

    n = series.n_valid
    if n < 5:
        raise InsufficientDataError(f"just_monitor requires at least 5 observations, got n={n}.")

    hist_idx = _resolve_boundary_index(history, series, name="history", default_valid_idx=0)
    start_idx = _resolve_boundary_index(start, series, name="start", default_valid_idx=n - 2)

    if start_idx >= n:
        raise ValueError("Monitoring start index must be strictly less than series length.")
    if hist_idx >= start_idx:
        raise ValueError("history start must precede monitoring start.")

    n_his = start_idx - hist_idx
    total_span = float(t0_full[-1] - t0_full[0])
    if parity_mode:
        min_his = n / total_span if total_span > 0.0 else 4.0
    elif series.is_datetime and season_norm != "none" and total_span >= 1.0:
        min_his = n / total_span
    else:
        min_his = 4.0
    if n_his < min_his or n_his < 4:
        raise InsufficientDataError(
            f"Insufficient stable historical period (n_history={n_his}, need at least "
            f"{max(4, int(np.ceil(min_his)))} observations)."
        )

    if omega is None:
        if season_norm == "allssa":
            om_hist = np.arange(4, 20, dtype=np.float64) / 5.0
        elif season_norm == "ols":
            om_hist = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        else:
            om_hist = np.empty(0, dtype=np.float64)
    else:
        om_hist = np.asarray(omega, dtype=np.float64)

    if monitor_omega is None:
        om_mon = np.arange(1, 36, dtype=np.float64) / 10.0
    else:
        om_mon = np.asarray(monitor_omega, dtype=np.float64)
        if om_mon.ndim != 1 or om_mon.size == 0 or np.any(om_mon <= 0.0):
            raise ValueError("monitor_omega must be a non-empty 1D array of positive frequencies.")

    # Slice active interval [hist_idx : n]
    active_idx = np.arange(hist_idx, n, dtype=np.intp)
    t_act = t0_full[active_idx]
    y_act = y_full[active_idx]
    pw_act = pw_full.subset(active_idx)

    his_local = np.arange(0, n_his, dtype=np.intp)
    t_his = t_act[his_local]
    y_his = y_act[his_local]
    pw_his = pw_act.subset(his_local)

    empty_j = np.empty(0, dtype=np.intp)
    cond_max = 1.0

    pw_act_base = pw_act
    pw_his_base = pw_his

    # Fit stable historical linear trend + seasonal model (Ghaderpour & Vujadinovic, 2020b)
    if season_norm == "allssa":
        his_allssa = _allssa_core(
            t_his,
            y_his,
            pw_his,
            om_hist,
            empty_j,
            alpha,
            "linear",
            False,
            decimal=1,
            max_iter=200,
            parity_mode=parity_mode,
            compute_cov=True,
            robust=eff_robust,
            huber_k=huber_k,
            significance=season_sig,
        )
        his_freqs = his_allssa.frequencies
        his_trend_coeff = his_allssa.trend_coeff
        his_coeff = np.concatenate([his_trend_coeff, his_allssa.cos_sin_coeffs.ravel()])
        his_residuals = his_allssa.residuals
        his_cov = his_allssa.cov
        cond_max = max(cond_max, his_allssa.condition_number_max)
    elif season_norm == "ols":
        his_ols = _lssa_core(
            t_his,
            y_his,
            pw_his,
            np.empty(0, dtype=np.float64),
            empty_j,
            alpha,
            "linear",
            False,
            om_hist,
            compute_cov=True,
            robust=eff_robust,
            huber_k=huber_k,
        )
        his_freqs = om_hist.copy()
        his_trend_coeff = his_ols.coeff[:2]
        his_coeff = his_ols.coeff
        his_residuals = his_ols.residuals
        his_cov = his_ols.cov
        cond_max = max(cond_max, his_ols.condition_number)
    else:
        his_none = _lssa_core(
            t_his,
            y_his,
            pw_his,
            np.empty(0, dtype=np.float64),
            empty_j,
            alpha,
            "linear",
            False,
            np.empty(0, dtype=np.float64),
            compute_cov=True,
            robust=eff_robust,
            huber_k=huber_k,
        )
        his_freqs = np.empty(0, dtype=np.float64)
        his_trend_coeff = his_none.coeff[:2]
        his_coeff = his_none.coeff
        his_residuals = his_none.residuals
        his_cov = his_none.cov
        cond_max = max(cond_max, his_none.condition_number)

    # Under robust='huber' or 'mm', down-weight historical baseline spikes in [history:start]
    # while keeping all incoming monitoring observations at full weight 1.0 (zero masking).
    if eff_robust != "none":
        rw_his = whiten_vector(his_residuals, pw_his)
        s_his = float(1.482602218505602 * np.median(np.abs(rw_his)))
        if s_his > 1e-14 * max(float(np.sqrt(np.mean(rw_his * rw_his))), 1e-300):
            u_his = np.abs(rw_his) / s_his
            w_his_h = (
                np.maximum(tukey_weight(rw_his / s_his), 1e-12)
                if eff_robust == "mm"
                else np.where(u_his <= huber_k, 1.0, huber_k / np.maximum(u_his, huber_k))
            )
            act_factors = np.concatenate(
                [w_his_h, np.ones(int(t_act.size) - n_his, dtype=np.float64)]
            )
            pw_act = pw_act.scale_entries(act_factors)
            pw_his = pw_act.subset(his_local)

    # Forecast historical model across [hist_idx : n] (unscaling design columns to match his_coeff)
    D_act, col_scales_act = build_design_matrix(t_act, "linear", empty_j, False, his_freqs)
    D_act_u = D_act / col_scales_act
    forecast = np.asarray(D_act_u @ his_coeff, dtype=np.float64)
    residuals_act = np.asarray(y_act - forecast, dtype=np.float64)

    # Pointwise mean forecast CI and out-of-sample observation prediction interval (PI)
    nc_his = int(his_coeff.size)
    dof_his = n_his - nc_his
    forecast_ci_low: FloatArray | None = None
    forecast_ci_high: FloatArray | None = None
    forecast_pi_low: FloatArray | None = None
    forecast_pi_high: FloatArray | None = None
    if (
        his_cov.size > 0
        and his_cov.shape == (nc_his, nc_his)
        and np.all(np.isfinite(his_cov))
        and dof_his > 0
    ):
        var_mean = np.maximum(0.0, np.sum((D_act_u @ his_cov) * D_act_u, axis=1))
        rw_his_base = whiten_vector(his_residuals, pw_his_base)
        if eff_robust != "none":
            mad_his = float(1.482602218505602 * np.median(np.abs(rw_his_base))) * np.sqrt(
                n_his / max(1, dof_his)
            )
            sigma2_his = mad_his * mad_his
        else:
            sigma2_his = float(np.dot(rw_his_base, rw_his_base)) / dof_his

        n_act_len = int(t_act.size)
        if pw_act_base.kind == "none":
            inv_w_act = np.ones(n_act_len, dtype=np.float64)
        elif pw_act_base.kind == "scalar":
            inv_w_act = np.full(n_act_len, 1.0 / pw_act_base.scalar, dtype=np.float64)
        elif pw_act_base.kind == "vector":
            assert pw_act_base.vector is not None
            inv_w_act = 1.0 / pw_act_base.vector
        else:
            assert pw_act_base.cholesky_lower is not None
            sigma_mat = linalg.cho_solve(
                (pw_act_base.cholesky_lower, True), np.eye(n_act_len, dtype=np.float64)
            )
            inv_w_act = np.maximum(0.0, np.diag(sigma_mat))

        t_crit_his = float(t_dist.ppf(0.5 + 0.5 * ci_level, df=dof_his))
        hw_ci = t_crit_his * np.sqrt(var_mean)
        hw_pi = t_crit_his * np.sqrt(np.maximum(0.0, var_mean + sigma2_his * inv_w_act))
        forecast_ci_low = forecast - hw_ci
        forecast_ci_high = forecast + hw_ci
        forecast_pi_low = forecast - hw_pi
        forecast_pi_high = forecast + hw_pi

    # Precompute out-of-sample predictive covariance factor V_all = D_out (D_his^T P_his D_his)^{-1/2}
    v_out_all: FloatArray | None = None
    if eff_forecast_cov:
        d_his_s = D_act[:n_his]
        w_his_mat = whiten_matrix(d_his_s, pw_his)
        _, s_h, vt_h = linalg.svd(w_his_mat, full_matrices=False)
        if s_h.size > 0 and s_h[0] > 0.0:
            inv_s = np.where(s_h > 1e-12 * s_h[0], 1.0 / s_h, 0.0)
            f_his = vt_h.T * inv_s[None, :]
            v_out_all = np.asarray(D_act[n_his:] @ f_his, dtype=np.float64)

    # Historical residual red-noise diagnostics
    rn, rn_warns = compute_red_noise_diagnostics(t_his, his_residuals)
    warnings_list.extend(rn_warns)

    n_mon = n - start_idx
    if horizon is not None:
        eff_horizon = int(horizon)
        if eff_horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {eff_horizon}.")
    elif parity_mode or eff_seq_ctrl not in ("pocock", "obrien_fleming"):
        eff_horizon = n_mon
    else:
        # Never infer the spending horizon from the amount of data present: that would make
        # thresholds at past steps depend on future observations (look-ahead).
        eff_horizon = 0
        warnings_list.append(
            WarningItem(
                code="OPEN_ENDED_SPENDING",
                severity="caution",
                message=(
                    f"No `horizon` given for sequential_control={eff_seq_ctrl!r}; used open-ended "
                    "alpha spending alpha_d = alpha / (2.11 (d+1) ln^2(d+1)), which controls the familywise "
                    "false-alarm rate over any monitoring length without using future data."
                ),
                suggestion=(
                    "Open-ended spending is strict at later steps; pass the planned number of "
                    "monitoring steps as `horizon` (fixed in advance) for substantially more power."
                ),
            )
        )

    if (
        (not parity_mode)
        and eff_seq_ctrl in ("pocock", "obrien_fleming", "consecutive_k")
        and horizon is not None
        and 0 < eff_horizon < n_mon
    ):
        warnings_list.append(
            WarningItem(
                code="HORIZON_EXCEEDED",
                severity="warning",
                message=(
                    f"Monitoring length ({n_mon} steps) exceeds planned horizon ({eff_horizon}); "
                    f"familywise Type I error control under sequential_control={eff_seq_ctrl!r} "
                    "is not guaranteed beyond the planned horizon."
                ),
                suggestion=(
                    "Increase `horizon` to cover the monitoring period, or omit `horizon` "
                    "to use open-ended alpha spending."
                ),
            )
        )

    his_span = float(t_act[n_his - 1] - t_act[0])
    mon_span = float(t_act[-1] - t_act[n_his]) if n_mon > 1 else 0.0
    if (not parity_mode) and his_span > 0.0 and mon_span > his_span:
        warnings_list.append(
            WarningItem(
                code="LONG_MONITORING_HORIZON",
                severity="caution",
                message=(
                    f"Monitoring span ({mon_span:.3g}) exceeds the stable historical span "
                    f"({his_span:.3g}); extrapolated trend uncertainty grows quadratically "
                    "with forecast distance."
                ),
                suggestion=(
                    "Keep `account_for_forecast_uncertainty=True` (default) or periodically "
                    "re-baseline the stable historical model."
                ),
            )
        )

    if (not parity_mode) and eff_freq_corr == "none" and eff_seq_ctrl == "none" and n_mon > 1:
        warnings_list.append(
            WarningItem(
                code="UNADJUSTED_REPEATED_TESTING",
                severity="warning",
                message=(
                    f"Monitoring {n_mon} observations across {om_mon.size} frequencies without "
                    "multiplicity control (freq_correction='none', sequential_control='none'). "
                    "Cumulative false-alarm probability can greatly exceed nominal alpha."
                ),
            )
        )

    # Monitoring baseline offset Lh (matching Ghaderpour & Vujadinovic, 2020b)
    lh = n_his - int(h * n_his) - 1
    if lh > n_his - 3:
        lh -= 3
    elif lh == -1:
        lh += 1
    lh = max(0, lh)
    slide_width = max(5, n_his - lh + 1)

    low_mask = om_mon < 1.0 - 1e-12
    high_mask = ~low_mask

    # consecutive_k: calibrate a constant per-step level so that k consecutive exceedances
    # within the planned horizon occur with probability <= alpha under the fitted null.
    consec_alpha: float | None = None
    if eff_seq_ctrl == "consecutive_k":
        if horizon is not None:
            consec_alpha = _calibrate_consecutive_k(
                t_act[:n_his],
                pw_act.subset(his_local),
                his_freqs,
                float(t_act[n_his]),
                eff_horizon,
                lh,
                slide_width,
                eff_win_mode,
                om_mon,
                eff_freq_corr,
                int(consecutive_k),
                alpha,
                int(n_mc),
                v_out_all is not None,
            )
            if consec_alpha is None:
                warnings_list.append(
                    WarningItem(
                        code="CALIBRATION_FALLBACK",
                        severity="warning",
                        message=(
                            "consecutive_k could not be calibrated (full weight matrix or degenerate "
                            "history spacing); each step is tested at the full alpha, which does not "
                            "bound the familywise false-alarm rate."
                        ),
                    )
                )
        else:
            warnings_list.append(
                WarningItem(
                    code="UNCALIBRATED_CONSECUTIVE_K",
                    severity="caution",
                    message=(
                        "sequential_control='consecutive_k' without `horizon` tests each step at the "
                        "full alpha; the familywise false-alarm rate grows with monitoring length."
                    ),
                    suggestion=(
                        "Pass the planned number of monitoring steps as `horizon` to calibrate the "
                        "per-step level by Monte Carlo, or use sequential_control='pocock'."
                    ),
                )
            )

    steps: list[MonitorStep] = []
    detected = False
    dist_type: DisturbanceType = "none"
    type_code = 0
    dist_valid_idx: int | None = None
    dist_local_idx: int | None = None
    dom_freq: float | None = None
    consec_count = 0

    for d in range(1, n_mon + 1):
        win_end = n_his + d
        win_start = lh if eff_win_mode == "expanding" else max(0, win_end - slide_width)
        win_idx = np.arange(win_start, win_end, dtype=np.intp)

        t_w = t_act[win_idx]
        r_w = residuals_act[win_idx]
        pw_w = pw_act.subset(win_idx)
        if v_out_all is not None:
            n_in_w = max(0, n_his - win_start)
            out_start = max(0, win_start - n_his)
            pw_w = _build_predictive_window_weights(pw_w, n_in_w, v_out_all[out_start:d])

        alpha_step = (
            consec_alpha
            if consec_alpha is not None
            else _step_alpha_budget(d, eff_horizon, alpha, eff_seq_ctrl)
        )
        alpha_adj = _adjust_alpha_for_frequencies(alpha_step, int(om_mon.size), eff_freq_corr)

        res_step = _lssa_core(
            t_w,
            r_w,
            pw_w,
            om_mon,
            empty_j,
            alpha_adj,
            "constant",
            False,
            np.empty(0, dtype=np.float64),
            compute_cov=False,
        )
        cond_max = max(cond_max, res_step.condition_number)
        spec = res_step.spectrum
        crit = float(critical_value_lss(int(t_w.size), 1, alpha_adj))

        max_low = float(np.max(spec[low_mask])) if np.any(low_mask) else 0.0
        max_high = float(np.max(spec[high_mask])) if np.any(high_mask) else 0.0
        peak_i = int(np.argmax(spec))
        max_all = float(spec[peak_i])
        peak_f = float(om_mon[peak_i])

        if max_low > crit:
            cand_type: DisturbanceType = "abrupt"
            cand_freq = float(om_mon[low_mask][int(np.argmax(spec[low_mask]))])
            exceeded = True
        elif max_all > crit:
            cand_type = "phenological"
            cand_freq = peak_f
            exceeded = True
        else:
            cand_type = "none"
            cand_freq = peak_f
            exceeded = False

        if exceeded:
            consec_count += 1
        else:
            consec_count = 0

        cur_valid_idx = start_idx + d - 1
        cur_orig_idx = int(series.valid_to_orig[cur_valid_idx])

        steps.append(
            MonitorStep(
                step=d,
                index=cur_orig_idx,
                time=series.times_raw[cur_valid_idx],
                time_year=float(series.times_numeric[cur_valid_idx]),
                max_low_freq_spec=max_low,
                max_high_freq_spec=max_high,
                max_spec=max_all,
                peak_freq=cand_freq if exceeded else peak_f,
                critical_value=crit,
                alpha_step=alpha_step,
                alpha_adjusted=alpha_adj,
                exceeded=exceeded,
                candidate_type=cand_type,
                consecutive_count=consec_count,
            )
        )

        triggered_now = (
            exceeded
            if eff_seq_ctrl != "consecutive_k"
            else (exceeded and consec_count >= consecutive_k)
        )

        if triggered_now and not detected:
            detected = True
            # For consecutive_k, report onset index, type, and frequency consistently
            # from the first exceedance of the confirming streak.
            onset_step = steps[d - consecutive_k] if eff_seq_ctrl == "consecutive_k" else steps[-1]
            dist_type = onset_step.candidate_type
            type_code = 1 if dist_type == "abrupt" else 2
            dist_local_idx = n_his + onset_step.step - 1
            dist_valid_idx = start_idx + onset_step.step - 1
            dom_freq = onset_step.peak_freq
            if stop_on_first:
                break

    pending_confirmation = (
        (not detected)
        and (eff_seq_ctrl == "consecutive_k")
        and (consec_count > 0)
    )
    if pending_confirmation:
        warnings_list.append(
            WarningItem(
                code="PENDING_CONFIRMATION",
                severity="info",
                message=(
                    f"Latest observation(s) exceeded threshold ({consec_count}/{consecutive_k} "
                    "consecutive exceedances); awaiting additional observations to confirm."
                ),
            )
        )

    # Estimate magnitude and direction when an abrupt disturbance is detected
    magnitude: float | None = None
    direction: float | None = None
    if (
        detected
        and dist_type == "abrupt"
        and dist_local_idx is not None
        and dist_local_idx < int(t_act.size) - 2
    ):
        j_arr = np.array([dist_local_idx], dtype=np.intp)
        if season_norm == "allssa":
            j_res = _allssa_core(
                t_act,
                y_act,
                pw_act,
                om_hist,
                j_arr,
                alpha,
                trend_norm,
                False,
                decimal=1,
                max_iter=200,
                parity_mode=parity_mode,
                compute_cov=False,
                robust=eff_robust,
                huber_k=huber_k,
                significance=season_sig,
            )
            tr_c = j_res.trend_coeff
            cond_max = max(cond_max, j_res.condition_number_max)
        elif season_norm == "ols":
            j_ols = _lssa_core(
                t_act,
                y_act,
                pw_act,
                np.empty(0, dtype=np.float64),
                j_arr,
                alpha,
                trend_norm,
                False,
                om_hist,
                compute_cov=False,
                robust=eff_robust,
                huber_k=huber_k,
            )
            tr_c = j_ols.coeff
            cond_max = max(cond_max, j_ols.condition_number)
        else:
            j_none = _lssa_core(
                t_act,
                y_act,
                pw_act,
                np.empty(0, dtype=np.float64),
                j_arr,
                alpha,
                trend_norm,
                False,
                np.empty(0, dtype=np.float64),
                compute_cov=False,
                robust=eff_robust,
                huber_k=huber_k,
            )
            tr_c = j_none.coeff
            cond_max = max(cond_max, j_none.condition_number)

        if trend_norm == "linear":
            magnitude = float((tr_c[1] - tr_c[0]) + (tr_c[3] - tr_c[2]) * t_act[dist_local_idx])
            direction = float(tr_c[3] - tr_c[2])
        else:
            magnitude = float(tr_c[1] - tr_c[0])
            direction = None

    # Check if detected disturbance coincides with a lab DL shift
    if (
        detected
        and dist_valid_idx is not None
        and prepared.provenance is not None
        and prepared.provenance.dl_shifts
    ):
        dist_yr = float(series.times_numeric[dist_valid_idx])
        median_dt = float(np.median(np.diff(series.times_numeric))) if n > 1 else 0.1
        tol_yr = max(0.35, 3.0 * median_dt)
        for sh in prepared.provenance.dl_shifts:
            if abs(dist_yr - sh.time_year) <= tol_yr:
                warnings_list.append(
                    WarningItem(
                        code="DL_SHIFT_COINCIDENCE",
                        severity="warning",
                        message=(
                            f"Detected near-real-time disturbance at {series.times_raw[dist_valid_idx]} "
                            f"coincides with a detection-limit shift at {sh.time} "
                            f"({sh.old_limit:g} -> {sh.new_limit:g}). Verify this is not a lab artifact."
                        ),
                    )
                )
                break

    dist_orig_idx = int(series.valid_to_orig[dist_valid_idx]) if dist_valid_idx is not None else None
    dist_time_val = series.times_raw[dist_valid_idx] if dist_valid_idx is not None else None
    dist_year_val = float(series.times_numeric[dist_valid_idx]) if dist_valid_idx is not None else None

    diag = DiagnosticReport(
        warnings=tuple(warnings_list),
        red_noise=rn,
        condition_number_max=cond_max,
        n_input=series.n_input,
        n_valid=series.n_valid,
        n_dropped_nan=series.n_dropped_nan,
        sampling_rate_ratio=series.sampling_rate_ratio,
    )

    return MonitorResult(
        detected=detected,
        disturbance_type=dist_type,
        type_code=type_code,
        disturbance_index=dist_orig_idx,
        disturbance_time=dist_time_val,
        disturbance_time_year=dist_year_val,
        magnitude=magnitude,
        direction=direction,
        dominant_frequency=dom_freq,
        pending_confirmation=pending_confirmation,
        history_index=int(series.valid_to_orig[hist_idx]),
        start_index=int(series.valid_to_orig[start_idx]),
        history_time=series.times_raw[hist_idx],
        start_time=series.times_raw[start_idx],
        times=series.times_raw[active_idx],
        time_years=series.times_numeric[active_idx],
        values=y_act,
        forecast=forecast,
        residuals=residuals_act,
        historical_frequencies=his_freqs,
        historical_trend_coeff=his_trend_coeff,
        steps=steps,
        alpha=alpha,
        freq_correction=eff_freq_corr,
        sequential_control=eff_seq_ctrl,
        horizon=eff_horizon,
        diagnostics=diag,
        censoring=prepared.provenance,
        consecutive_k=int(consecutive_k),
        consecutive_k_calibrated=consec_alpha is not None,
        ci_level=ci_level,
        forecast_ci_low=forecast_ci_low,
        forecast_ci_high=forecast_ci_high,
        forecast_pi_low=forecast_pi_low,
        forecast_pi_high=forecast_pi_high,
    )
