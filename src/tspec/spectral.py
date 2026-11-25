"""Weighted Least-Squares Spectral Analysis (LSSA) and Anti-Leakage LSSA (ALLSSA).

Clean-room implementation grounded in:
- Ghaderpour, E., & Pagiatakis, S. D. (2017). Least-Squares Wavelet Analysis of
  Unequally Spaced and Non-stationary Time Series and Its Applications.
  *Mathematical Geosciences*, 49(7), 819–844.
- Ghaderpour, E., Liao, W., & Lamoureux, M. P. (2018). Antileakage least-squares
  spectral analysis for seismic data regularization and random noise attenuation.
  *Geophysics*, 83(3), V157–V170.
- Ghaderpour, E. (2021). JUST: MATLAB and Python Software for Change Detection
  and Time Series Analysis. *GPS Solutions*, 25(3), 85.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ._diagnostics import (
    DiagnosticReport,
    WarningItem,
    compute_red_noise_diagnostics,
    time_unit_warning,
)
from ._linalg import (
    PreparedWeights,
    compute_lss_vectorized,
    max_lss_null_draws,
    orthonormal_design_basis,
    prepare_weights,
    solve_weighted_ols,
)
from ._robust import RobustMethod, solve_robust_irls, validate_robust_args
from ._time import prepare_numeric_series
from ._types import ArrayLike, FloatArray, IntArray, TimeLike, WeightLike
from .exceptions import InsufficientDataError
from .results import (
    BaseResult,
    format_diagnostics_block,
    format_explain_warnings,
    require_pandas,
)

TrendType = Literal["none", "constant", "linear", "quadratic", "cubic"]
SignificanceMethod = Literal["montecarlo", "sidak", "per_frequency"]
_VALID_TRENDS: tuple[str, ...] = ("none", "constant", "linear", "quadratic", "cubic")


def _round_half_up(x: float, decimals: int = 0) -> float:
    """Round half away from zero for positive values (matches standard mathematical rounding)."""
    mult = float(10**decimals)
    return float(np.floor(x * mult + 0.5) / mult)


def _validate_jump_indices(
    jump_indices: Sequence[int] | None,
    n: int,
) -> IntArray:
    """Validate and sort internal 0-based jump split indices."""
    if jump_indices is None or len(jump_indices) == 0:
        return np.empty(0, dtype=np.intp)
    raw = np.asarray(jump_indices)
    if not np.all(np.floor(raw) == raw):
        raise ValueError("`jump_indices` must contain integers.")
    idx = np.sort(np.unique(raw.astype(np.intp)))
    if int(idx[0]) < 2 or int(idx[-1]) > n - 2:
        raise ValueError(
            f"`jump_indices` must satisfy 2 <= j <= n - 2 (got min={idx[0]}, max={idx[-1]}, n={n})."
        )
    return idx


def _count_constituents(
    trend: str,
    n_jumps: int,
    shared_slope: bool,
    n_known_freqs: int,
) -> int:
    """Count total columns in the known-constituents design matrix D."""
    nc = 2 * n_known_freqs
    if trend == "none":
        return nc
    if trend == "constant":
        return nc + n_jumps + 1
    slope_cols = 1 if (n_jumps == 0 or shared_slope) else (n_jumps + 1)
    poly_extra = {"linear": 0, "quadratic": 1, "cubic": 2}[trend]
    return nc + (n_jumps + 1) + slope_cols + poly_extra


def build_design_matrix(
    t0: FloatArray,
    trend: str,
    jump_indices: IntArray,
    shared_slope: bool,
    known_freqs: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Construct column-scaled design matrix ``D`` and diagonal back-scaling vector.

    Column ordering matches the published LSSA/JUST convention:
    1. Piece intercepts (``n_jumps + 1`` columns for constant/linear/quadratic/cubic)
    2. Piece slopes (``1`` column if ``shared_slope`` or no jumps, else ``n_jumps + 1``)
    3. Global quadratic term ``t0**2`` (for quadratic/cubic)
    4. Global cubic term ``t0**3`` (for cubic)
    5. Alternating ``cos(2*pi*f*t0), sin(2*pi*f*t0)`` for each frequency in ``known_freqs``
    """
    n = int(t0.size)
    n_jumps = int(jump_indices.size)
    n_freq = int(known_freqs.size)
    nc = _count_constituents(trend, n_jumps, shared_slope, n_freq)

    D = np.zeros((n, nc), dtype=np.float64)
    col_scales = np.ones(nc, dtype=np.float64)
    if nc == 0:
        return D, col_scales

    t_span = float(t0[-1] - t0[0]) if n > 1 else 1.0
    s_t = t_span if t_span > 0.0 else 1.0
    u = t0 / s_t

    q = 0
    # Segment boundaries [b_0, b_1), [b_1, b_2), ...
    bounds = [0, *(int(j) for j in jump_indices), n]

    # 1. Intercepts
    if trend in ("constant", "linear", "quadratic", "cubic"):
        for seg in range(len(bounds) - 1):
            a, b = bounds[seg], bounds[seg + 1]
            D[a:b, q] = 1.0
            q += 1

    # 2. Slopes
    if trend in ("linear", "quadratic", "cubic"):
        if n_jumps == 0 or shared_slope:
            D[:, q] = u
            col_scales[q] = 1.0 / s_t
            q += 1
        else:
            for seg in range(len(bounds) - 1):
                a, b = bounds[seg], bounds[seg + 1]
                D[a:b, q] = u[a:b]
                col_scales[q] = 1.0 / s_t
                q += 1

    # 3. Quadratic
    if trend in ("quadratic", "cubic"):
        D[:, q] = u * u
        col_scales[q] = 1.0 / (s_t * s_t)
        q += 1

    # 4. Cubic
    if trend == "cubic":
        D[:, q] = u * u * u
        col_scales[q] = 1.0 / (s_t * s_t * s_t)
        q += 1

    # 5. Sinusoids of known frequencies
    if n_freq > 0:
        phases = np.outer((2.0 * np.pi) * t0, known_freqs)
        D[:, q::2] = np.cos(phases)
        D[:, q + 1 :: 2] = np.sin(phases)

    return D, col_scales


def _default_allssa_omega(t0: FloatArray) -> FloatArray:
    """Default coarse cyclic-frequency search grid ``1 .. ceil(M/2) - 1_{M/2 in Z}``."""
    n = int(t0.size)
    span = float(t0[-1] - t0[0])
    if span <= 0.0:
        raise InsufficientDataError("Time span must be strictly positive.")
    rate = (n - 1) / span
    half = rate / 2.0
    upper = int(half) if int(half) == half else int(half) + 1
    if upper <= 1:
        return np.array([1.0], dtype=np.float64)
    return np.arange(1, upper, dtype=np.float64)


def _allssa_search_grid(t0: FloatArray, omega: FloatArray, decimal: int) -> tuple[FloatArray, float]:
    """Dense grid covering every frequency ALLSSA can select, and its effective size.

    Returns ``(grid, k_eff)`` where ``k_eff`` approximates the number of independent
    frequencies searched (three times the Rayleigh count ``T x bandwidth``, capped by the
    grid size), used for the analytical Sidak family-wise threshold.
    """
    span = float(t0[-1] - t0[0]) if t0.size > 1 else 1.0
    span = span if span > 0.0 else 1.0
    if omega.size == 0:
        return np.empty(0, dtype=np.float64), 1.0
    if decimal == 0:
        grid = np.unique(np.maximum(1.0, np.floor(omega + 0.5)))
        return grid.astype(np.float64), float(max(1, grid.size))
    step = 10.0 ** (-decimal)
    f_lo = max(step, float(np.min(omega)) - 0.5)
    f_hi = float(np.max(omega)) + 0.5
    d_step = max(step, 1.0 / (20.0 * span))
    grid = np.arange(f_lo, f_hi + 0.5 * d_step, d_step, dtype=np.float64)
    # The maximum over a continuous (refined) band behaves like ~2-3x the Rayleigh count
    # T * bandwidth of independent frequencies (calibrated by simulation, audit/post_sidak_null.py).
    k_eff = float(min(grid.size, int(np.ceil(3.0 * span * (f_hi - f_lo)))))
    return grid, max(1.0, k_eff)


def critical_value_lss(n: int, nc: int, alpha: float) -> float:
    """Exact Beta(1, (n - nc - 2)/2) critical value for the normalized LSS."""
    nu = n - nc - 2
    if nu <= 0:
        return float("nan")
    return float(1.0 - np.power(alpha, 2.0 / nu))


def _lss_pvalues(spectrum: FloatArray, n: int, nc: int) -> FloatArray:
    """Exact p-values under the Gaussian white-noise null: p = (1 - s)^(nu / 2)."""
    nu = n - nc - 2
    if nu <= 0:
        return np.full_like(spectrum, np.nan)
    return np.asarray(np.power(np.clip(1.0 - spectrum, 0.0, 1.0), 0.5 * nu), dtype=np.float64)


def _amplitudes_and_phases(cs: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Convert ``(cos, sin)`` coefficient pairs to amplitude and phase in radians."""
    if cs.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    c = cs[:, 0]
    s = cs[:, 1]
    amp = np.hypot(c, s)
    phase = np.arctan2(s, c)
    return np.asarray(amp, dtype=np.float64), np.asarray(phase, dtype=np.float64)


@dataclass(frozen=True)
class _LssaCoreOutput:
    frequencies: FloatArray
    spectrum: FloatArray
    critical_value: float
    pvalues: FloatArray
    cos_sin_coeffs: FloatArray
    residuals: FloatArray
    norm_res: float
    coeff: FloatArray
    cov: FloatArray
    condition_number: float
    rank_deficient: bool
    robust_converged: bool = True


def _lssa_core(
    t0: FloatArray,
    y: FloatArray,
    pw: PreparedWeights,
    omega: FloatArray,
    jump_indices: IntArray,
    alpha: float,
    trend: str,
    shared_slope: bool,
    known_freqs: FloatArray,
    *,
    compute_cov: bool = True,
    robust: RobustMethod = "none",
    huber_k: float = 1.345,
) -> _LssaCoreOutput:
    """Fast internal LSSA kernel operating on pre-validated, zero-shifted ``t0``."""
    n = int(t0.size)
    nc = _count_constituents(trend, int(jump_indices.size), shared_slope, int(known_freqs.size))

    min_obs = nc + 3 if omega.size > 0 else nc + 1
    if n < min_obs:
        raise InsufficientDataError(
            f"Series/window too short (n={n}) for {nc} known constituents "
            f"(need at least {min_obs} observations)."
        )

    D, col_scales = build_design_matrix(t0, trend, jump_indices, shared_slope, known_freqs)
    if robust != "none":
        hr = solve_robust_irls(
            D, y, pw, method=robust, k=huber_k, col_scales=col_scales, compute_cov=compute_cov
        )
        ols = hr.ols
        pw_spec = hr.pw_effective
        robust_ok = bool(hr.converged)
    else:
        robust_ok = True
        ols = solve_weighted_ols(
            D, y, pw, col_scales=col_scales, allow_rank_deficient=True, compute_cov=compute_cov
        )
        pw_spec = pw
    eff_nc = ols.rank if ols.rank_deficient else nc
    crit = critical_value_lss(n, eff_nc, alpha)

    if omega.size > 0:
        spec, cs = compute_lss_vectorized(t0, omega, ols, pw_spec, known_freqs=known_freqs)
        pvals = _lss_pvalues(spec, n, eff_nc)
    else:
        spec = np.empty(0, dtype=np.float64)
        cs = np.empty((0, 2), dtype=np.float64)
        pvals = np.empty(0, dtype=np.float64)

    return _LssaCoreOutput(
        frequencies=omega,
        spectrum=spec,
        critical_value=crit,
        pvalues=pvals,
        cos_sin_coeffs=cs,
        residuals=ols.residuals,
        norm_res=ols.norm_res,
        coeff=ols.coeff,
        cov=ols.cov,
        condition_number=ols.condition_number,
        rank_deficient=ols.rank_deficient,
        robust_converged=robust_ok,
    )


@dataclass(frozen=True)
class _AllssaCoreOutput:
    trend_coeff: FloatArray
    cos_sin_coeffs: FloatArray
    frequencies: FloatArray
    residuals: FloatArray
    norm_res: float
    cov: FloatArray
    n_iter: int
    converged: bool
    condition_number_max: float
    rank_deficient: bool


def _allssa_core(
    t0: FloatArray,
    y: FloatArray,
    pw: PreparedWeights,
    omega: FloatArray,
    jump_indices: IntArray,
    alpha: float,
    trend: str,
    shared_slope: bool,
    decimal: int,
    max_iter: int,
    *,
    parity_mode: bool = False,
    compute_cov: bool = True,
    robust: RobustMethod = "none",
    huber_k: float = 1.345,
    significance: str = "per_frequency",
    n_mc: int = 999,
) -> _AllssaCoreOutput:
    """Fast internal ALLSSA iterative anti-leakage kernel.

    ``significance`` selects the acceptance rule for each new frequency:
    ``"per_frequency"`` (published rule: peak > Beta critical value at ``alpha``),
    ``"sidak"`` (analytical family-wise threshold over the effective number of
    searched frequencies), or ``"montecarlo"`` (exact Monte Carlo null of the spectral
    maximum over the searched grid; peak accepted when its family-wise p <= alpha).
    """
    eff_robust: RobustMethod = "none" if parity_mode else robust
    inc = 10**decimal
    step = 1.0 / inc
    min_positive_freq = step

    selected: list[float] = []
    visited: set[tuple[float, ...]] = set()
    n_iter = 0
    converged = False
    cond_max = 1.0
    any_rank_def = False

    n = int(t0.size)
    base_nc = _count_constituents(trend, int(jump_indices.size), shared_slope, 0)
    fw_method = "per_frequency" if parity_mode else significance
    if fw_method != "per_frequency":
        search_grid, k_eff = _allssa_search_grid(t0, omega, decimal)
        alpha_sidak = float(1.0 - (1.0 - alpha) ** (1.0 / k_eff))

    res = _lssa_core(
        t0,
        y,
        pw,
        omega,
        jump_indices,
        alpha,
        trend,
        shared_slope,
        np.empty(0, dtype=np.float64),
        compute_cov=False,
        robust=eff_robust,
        huber_k=huber_k,
    )
    cond_max = max(cond_max, res.condition_number)
    any_rank_def = any_rank_def or res.rank_deficient

    for it in range(1, max_iter + 1):
        n_iter = it
        prev = selected.copy()
        visited.add(tuple(prev))

        if res.spectrum.size == 0:
            converged = True
            break

        max_idx = int(np.argmax(res.spectrum))
        om_center = _round_half_up(float(omega[max_idx]), 0)

        # Temporarily remove the highest previously selected frequency rounding to om_center
        matching = [idx for idx, fr in enumerate(selected) if _round_half_up(fr, 0) == om_center]
        if matching:
            del selected[matching[-1]]

        # Construct fine grid around om_center
        if inc > 1:
            om_fine = om_center - 0.5 + np.arange(inc + 1, dtype=np.float64) * step
            if not parity_mode:
                om_fine = om_fine[om_fine >= min_positive_freq]
                if om_fine.size == 0:
                    om_fine = np.array([min_positive_freq], dtype=np.float64)
        else:
            om_val = om_center if (parity_mode or om_center >= 1.0) else 1.0
            om_fine = np.array([om_val], dtype=np.float64)

        # Ensure sufficient degrees of freedom before evaluating fine grid
        nc_fine = base_nc + 2 * len(selected)
        if n < nc_fine + 3:
            selected = prev
            converged = True
            break

        res_fine = _lssa_core(
            t0,
            y,
            pw,
            om_fine,
            jump_indices,
            alpha,
            trend,
            shared_slope,
            np.asarray(selected, dtype=np.float64),
            compute_cov=False,
            robust=eff_robust,
            huber_k=huber_k,
        )
        cond_max = max(cond_max, res_fine.condition_number)
        any_rank_def = any_rank_def or res_fine.rank_deficient

        fine_idx = int(np.argmax(res_fine.spectrum))
        s_peak = float(res_fine.spectrum[fine_idx])
        accept = s_peak > res_fine.critical_value
        if accept and fw_method == "sidak":
            accept = s_peak > critical_value_lss(n, nc_fine, alpha_sidak)
        elif accept and fw_method == "montecarlo":
            sel_arr = np.asarray(selected, dtype=np.float64)
            D_cur, _ = build_design_matrix(t0, trend, jump_indices, shared_slope, sel_arr)
            q_cur = orthonormal_design_basis(D_cur, pw)
            draws = max_lss_null_draws(
                t0, search_grid, q_cur, pw, known_freqs=sel_arr, n_mc=n_mc,
                seed=(20261125, it, len(selected)),
            )
            p_fw = (1.0 + float(np.sum(draws >= s_peak))) / (n_mc + 1.0)
            accept = p_fw <= alpha
        if accept:
            om_new = _round_half_up(float(om_fine[fine_idx]), decimal)
            if any(abs(om_new - f_prev) < 0.5 * step for f_prev in prev):
                selected = prev
            else:
                selected.append(om_new)
                selected.sort()

        if selected == prev:
            converged = True
            break

        # Guard against oscillation cycles or running out of degrees of freedom
        if tuple(selected) in visited:
            res = _lssa_core(
                t0,
                y,
                pw,
                np.empty(0, dtype=np.float64),
                jump_indices,
                alpha,
                trend,
                shared_slope,
                np.asarray(selected, dtype=np.float64),
                compute_cov=compute_cov,
                robust=eff_robust,
                huber_k=huber_k,
            )
            converged = True
            break

        nc_next = base_nc + 2 * len(selected)
        if n < nc_next + 3:
            res = _lssa_core(
                t0,
                y,
                pw,
                np.empty(0, dtype=np.float64),
                jump_indices,
                alpha,
                trend,
                shared_slope,
                np.asarray(selected, dtype=np.float64),
                compute_cov=compute_cov,
                robust=eff_robust,
                huber_k=huber_k,
            )
            converged = True
            break

        res = _lssa_core(
            t0,
            y,
            pw,
            omega,
            jump_indices,
            alpha,
            trend,
            shared_slope,
            np.asarray(selected, dtype=np.float64),
            compute_cov=False,
            robust=eff_robust,
            huber_k=huber_k,
        )
        cond_max = max(cond_max, res.condition_number)
        any_rank_def = any_rank_def or res.rank_deficient

    freq_arr = np.asarray(selected, dtype=np.float64)
    if compute_cov and res.cov.size == 0:
        res = _lssa_core(
            t0,
            y,
            pw,
            np.empty(0, dtype=np.float64),
            jump_indices,
            alpha,
            trend,
            shared_slope,
            freq_arr,
            compute_cov=True,
            robust=eff_robust,
            huber_k=huber_k,
        )
    n_freq = int(freq_arr.size)
    if n_freq > 0:
        n_tr = res.coeff.size - 2 * n_freq
        tr_coeff = res.coeff[:n_tr]
        cs_coeff = res.coeff[n_tr:].reshape(n_freq, 2)
    else:
        tr_coeff = res.coeff
        cs_coeff = np.empty((0, 2), dtype=np.float64)

    return _AllssaCoreOutput(
        trend_coeff=tr_coeff,
        cos_sin_coeffs=cs_coeff,
        frequencies=freq_arr,
        residuals=res.residuals,
        norm_res=res.norm_res,
        cov=res.cov,
        n_iter=n_iter,
        converged=converged,
        condition_number_max=cond_max,
        rank_deficient=any_rank_def,
    )


@dataclass(frozen=True)
class LssaResult(BaseResult):
    """Result object returned by :func:`lssa`.

    Attributes
    ----------
    frequencies : ndarray
        Evaluated cyclic frequencies (cycles per time unit; cycles/year for datetimes).
    spectrum : ndarray
        Normalized least-squares spectrum in ``[0, 1]``: the fraction of residual
        variance explained by each frequency after removing the known constituents.
    critical_value : float
        Per-frequency Beta critical value at ``alpha``.
    pvalues : ndarray
        Per-frequency p-values (not corrected for the number of frequencies).
    cos_sin_coeffs, amplitudes, phases : ndarray
        Out-of-context cosine/sine coefficients, amplitude and phase (radians,
        relative to the first timestamp) at each frequency.
    residuals, norm_res : ndarray, float
        Residuals of the known-constituent fit and their weighted sum of squares.
    coeff, cov : ndarray
        Known-constituent coefficients (relative to the first timestamp) and their
        covariance.
    alpha, trend, known_freqs, jump_indices
        Inputs as resolved.
    times_numeric, times_raw, is_datetime
        Valid timestamps as decimal years / original representation.
    diagnostics : DiagnosticReport
        Warnings and red-noise diagnostics.
    peak_fap : float
        Family-wise false-alarm probability of the highest peak over the searched
        frequencies (``NaN`` when ``peak_fap="none"``).
    peak_fap_method : str
        ``"sidak"``, ``"montecarlo"``, or ``"none"``.
    """

    frequencies: FloatArray
    spectrum: FloatArray
    critical_value: float
    pvalues: FloatArray
    cos_sin_coeffs: FloatArray
    amplitudes: FloatArray
    phases: FloatArray
    residuals: FloatArray
    norm_res: float
    coeff: FloatArray
    cov: FloatArray
    alpha: float
    trend: str
    known_freqs: FloatArray
    jump_indices: IntArray
    times_numeric: FloatArray
    times_raw: np.ndarray[Any, Any]
    is_datetime: bool
    diagnostics: DiagnosticReport
    peak_fap: float = float("nan")
    peak_fap_method: str = "none"

    @property
    def significant_mask(self) -> np.ndarray[Any, np.dtype[np.bool_]]:
        """Boolean mask of frequencies exceeding the **per-frequency** critical value.

        Under pure noise about ``alpha * len(frequencies)`` points exceed it by chance;
        use :attr:`peak_fap` for the family-wise significance of the highest peak.
        """
        return np.asarray(self.spectrum > self.critical_value, dtype=np.bool_)

    def summary(self) -> str:
        n_sig = int(np.sum(self.significant_mask)) if self.frequencies.size else 0
        lines = [
            f"LSSA Result (trend={self.trend!r}, alpha={self.alpha})",
            f"Known constituents : {self.coeff.size} parameters "
            f"(known_freqs={self.known_freqs.tolist()})",
            f"Weighted RSS       : {self.norm_res:.6g}",
            f"Spectrum grid      : {self.frequencies.size} frequencies, "
            f"critical value = {self.critical_value:.4f} ({n_sig} significant)",
        ]
        if self.frequencies.size > 0:
            peak_idx = int(np.argmax(self.spectrum))
            lines.append(
                f"Peak frequency     : {self.frequencies[peak_idx]:.4f} "
                f"(LSS={self.spectrum[peak_idx]:.4f}, per-frequency p={self.pvalues[peak_idx]:.3e}, "
                f"family-wise FAP={self.peak_fap:.3e} [{self.peak_fap_method}])"
            )
        lines.extend(format_diagnostics_block(self.diagnostics))
        return "\n".join(lines)

    def explain(self) -> str:
        lines: list[str] = []
        if self.frequencies.size == 0:
            lines.append(
                f"Fitted a weighted {self.trend} model to {self.diagnostics.n_valid} "
                "observations without evaluating a frequency grid."
            )
        else:
            sig_idx = np.flatnonzero(self.significant_mask)
            peak_idx = int(np.argmax(self.spectrum))
            n_freq = self.frequencies.size
            fap_known = np.isfinite(self.peak_fap)
            if fap_known and self.peak_fap <= self.alpha:
                lines.append(
                    f"The highest spectral peak, near frequency {self.frequencies[peak_idx]:.3f} "
                    f"(explaining {100.0 * self.spectrum[peak_idx]:.1f}% of residual variance), is "
                    f"significant after accounting for the {n_freq} frequencies searched "
                    f"(family-wise false-alarm probability {self.peak_fap:.2e} <= alpha={self.alpha})."
                )
            elif fap_known:
                lines.append(
                    f"No periodic cycle is significant after accounting for the {n_freq} "
                    f"frequencies searched: the highest peak near {self.frequencies[peak_idx]:.3f} "
                    f"has family-wise false-alarm probability {self.peak_fap:.2f} > alpha={self.alpha}."
                )
            else:
                lines.append(
                    f"The highest spectral peak is near frequency {self.frequencies[peak_idx]:.3f} "
                    f"(per-frequency p = {self.pvalues[peak_idx]:.2e}; no family-wise correction applied)."
                )
            lines.append(
                f"{sig_idx.size} of {n_freq} frequencies exceed the per-frequency critical value "
                f"(LSS = {self.critical_value:.3f}); about {self.alpha * n_freq:.1f} are expected "
                "by chance under pure noise, so individual exceedances are not evidence of cycles."
            )
        lines.extend(format_explain_warnings(self.diagnostics))
        return "\n".join(lines)

    def to_dataframe(self) -> Any:
        pd = require_pandas()
        return pd.DataFrame(
            {
                "frequency": self.frequencies,
                "spectrum": self.spectrum,
                "pvalue": self.pvalues,
                "significant_per_frequency": self.significant_mask,
                "cos_coeff": self.cos_sin_coeffs[:, 0] if self.frequencies.size else [],
                "sin_coeff": self.cos_sin_coeffs[:, 1] if self.frequencies.size else [],
                "amplitude": self.amplitudes,
                "phase_rad": self.phases,
            }
        )

    def plot(self, ax: Any = None, **kwargs: Any) -> tuple[Any, Any]:
        from .plotting import plot_lssa

        return plot_lssa(self, ax=ax, **kwargs)


@dataclass(frozen=True)
class AllssaResult(BaseResult):
    """Result object returned by :func:`allssa`.

    Attributes
    ----------
    trend_coeff : ndarray
        Trend/jump coefficients of the final simultaneous fit.
    frequencies : ndarray
        Selected cyclic frequencies.
    cos_sin_coeffs, amplitudes, phases : ndarray
        Coefficients, amplitudes and phases (radians, relative to the first
        timestamp) of the selected sinusoids.
    residuals, norm_res, cov : ndarray, float, ndarray
        Final residuals, weighted residual sum of squares, and parameter covariance
        (conditional on the selected frequencies).
    alpha, trend : float, str
        Significance level and trend model.
    n_iter, converged : int, bool
        Iterations used and whether the frequency set stabilized.
    times_numeric, times_raw, is_datetime
        Valid timestamps.
    diagnostics : DiagnosticReport
        Warnings and red-noise diagnostics.
    significance : str
        Acceptance rule used: ``"montecarlo"``, ``"sidak"``, or ``"per_frequency"``.
    """

    trend_coeff: FloatArray
    cos_sin_coeffs: FloatArray
    frequencies: FloatArray
    amplitudes: FloatArray
    phases: FloatArray
    residuals: FloatArray
    norm_res: float
    cov: FloatArray
    alpha: float
    trend: str
    n_iter: int
    converged: bool
    times_numeric: FloatArray
    times_raw: np.ndarray[Any, Any]
    is_datetime: bool
    diagnostics: DiagnosticReport
    significance: str = "per_frequency"

    def summary(self) -> str:
        lines = [
            f"ALLSSA Result (trend={self.trend!r}, alpha={self.alpha}, "
            f"significance={self.significance!r}, iters={self.n_iter}, converged={self.converged})",
            f"Weighted RSS : {self.norm_res:.6g}",
            f"Selected frequencies ({self.frequencies.size}):",
        ]
        if self.frequencies.size == 0:
            lines.append("  (none statistically significant)")
        else:
            lines.append("  Freq       Amplitude   Cos         Sin         Phase(rad)")
            for k in range(self.frequencies.size):
                lines.append(
                    f"  {self.frequencies[k]:<10.4f} {self.amplitudes[k]:<11.4f} "
                    f"{self.cos_sin_coeffs[k, 0]:<11.4f} {self.cos_sin_coeffs[k, 1]:<11.4f} "
                    f"{self.phases[k]:+.4f}"
                )
        lines.extend(format_diagnostics_block(self.diagnostics))
        return "\n".join(lines)

    def explain(self) -> str:
        lines: list[str] = []
        if self.frequencies.size == 0:
            lines.append(
                f"Anti-leakage spectral search found no statistically significant periodic "
                f"cycles at alpha={self.alpha} after removing the {self.trend} trend."
            )
        else:
            freq_desc = ", ".join(
                f"{f:.3f} (amplitude {a:.3g})"
                for f, a in zip(self.frequencies, self.amplitudes, strict=True)
            )
            if self.significance == "per_frequency":
                lines.append(
                    f"Anti-leakage spectral search (ALLSSA) selected {self.frequencies.size} "
                    f"cycle(s) simultaneously with the {self.trend} trend: {freq_desc}. "
                    f"Selection used the per-frequency critical value at alpha={self.alpha}, "
                    "which does not account for the frequency search; spurious cycles are "
                    "selected far more often than alpha under pure noise."
                )
            else:
                lines.append(
                    f"Anti-leakage spectral search (ALLSSA) identified {self.frequencies.size} "
                    f"cycle(s) significant at family-wise alpha={self.alpha} "
                    f"({self.significance} control over the searched frequency grid), "
                    f"estimated simultaneously with the {self.trend} trend: {freq_desc}."
                )
        lines.extend(format_explain_warnings(self.diagnostics))
        return "\n".join(lines)

    def to_dataframe(self) -> Any:
        pd = require_pandas()
        return pd.DataFrame(
            {
                "frequency": self.frequencies,
                "amplitude": self.amplitudes,
                "cos_coeff": self.cos_sin_coeffs[:, 0] if self.frequencies.size else [],
                "sin_coeff": self.cos_sin_coeffs[:, 1] if self.frequencies.size else [],
                "phase_rad": self.phases,
            }
        )

    def plot(self, ax: Any = None, **kwargs: Any) -> tuple[Any, Any]:
        from .plotting import plot_allssa

        return plot_allssa(self, ax=ax, **kwargs)


def _validate_spectral_common(
    trend: str,
    alpha: float,
) -> str:
    trend_l = trend.lower()
    if trend_l not in _VALID_TRENDS:
        raise ValueError(f"Unknown `trend` {trend!r}. Options: {_VALID_TRENDS}.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"Significance level `alpha` must lie in (0, 1); got {alpha}.")
    return trend_l


def lssa(
    t: TimeLike,
    y: ArrayLike,
    *,
    weights: WeightLike | None = None,
    omega: ArrayLike | None = None,
    jump_indices: Sequence[int] | None = None,
    alpha: float = 0.01,
    trend: TrendType = "linear",
    shared_slope: bool = False,
    known_freqs: Sequence[float] | None = None,
    robust: RobustMethod = "none",
    huber_k: float = 1.345,
    peak_fap: Literal["sidak", "montecarlo", "none"] = "sidak",
    n_mc: int = 999,
) -> LssaResult:
    """Compute the weighted Least-Squares Spectrum (LSSA) of an unevenly sampled series.

    Simultaneously fits known trend/jump/sinusoid constituents via whitened QR
    weighted least squares (or optional Huber IRLS M-estimation when
    ``robust='huber'``) and evaluates the normalized percentage-variance
    spectrum across candidate cyclic frequencies ``omega`` with zero interpolation.

    Parameters
    ----------
    t:
        1D monotonically ascending timestamps (numeric or datetime).
    y:
        1D observation values (``NaN`` entries are dropped cleanly without interpolation).
    weights:
        Optional scalar, 1D positive weight vector (O(n) memory), or 2D SPD
        inverse-covariance matrix ``P = C_f^{-1}``.
    omega:
        Candidate cyclic frequencies to evaluate (cycles per unit time; cycles/year
        for datetime inputs). If ``None``, fits only known constituents.
    jump_indices:
        Optional 0-based indices (in input ``t`` coordinates) of known trend jumps.
    alpha:
        Significance level in ``(0, 1)`` for the exact Beta critical value.
    trend:
        Known trend form: ``'none'``, ``'constant'``, ``'linear'``, ``'quadratic'``,
        or ``'cubic'``.
    shared_slope:
        If ``True``, piecewise linear segments share a single global slope.
    known_freqs:
        Known cyclic frequencies removed simultaneously with the trend.
    robust:
        ``'none'`` (default WLS/OLS), ``'huber'`` (convex Huber IRLS
        M-estimation), or ``'mm'`` (Huber-initialized Tukey biweight MM-type estimation
        with zero weight on extreme isolated spikes; not Yohai's S-started high-breakdown MM).
    huber_k:
        Huber tuning constant (default ``1.345``, 95% Gaussian efficiency).
    peak_fap:
        Family-wise false-alarm probability of the highest peak over ``omega``:
        ``"sidak"`` (analytical, effective number of independent frequencies),
        ``"montecarlo"`` (exact Gaussian-null simulation of the spectral maximum,
        ``n_mc`` draws), or ``"none"``. Per-frequency ``pvalues`` are unaffected.
    n_mc:
        Monte Carlo draws for ``peak_fap="montecarlo"``.

    Returns
    -------
    LssaResult
        Structured spectral result with ``.summary()``, ``.explain()``,
        ``.to_dataframe()``, and ``.plot()``.
    """
    trend_l = _validate_spectral_common(trend, alpha)
    robust_m = validate_robust_args(robust, huber_k)
    series = prepare_numeric_series(t, y)
    pw_full = prepare_weights(weights, series.n_input)
    pw = pw_full.subset(series.valid_mask) if series.n_dropped_nan else pw_full

    # Map caller jump_indices (original axis) onto valid axis
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
        else np.empty(0, dtype=np.float64)
    )
    kf_arr = (
        np.asarray(known_freqs, dtype=np.float64).ravel()
        if known_freqs is not None and len(known_freqs) > 0
        else np.empty(0, dtype=np.float64)
    )

    t0 = series.times_numeric - series.times_numeric[0]
    core = _lssa_core(
        t0,
        series.values,
        pw,
        om_arr,
        j_valid,
        alpha,
        trend_l,
        shared_slope,
        kf_arr,
        robust=robust_m,
        huber_k=huber_k,
    )
    amps, phases = _amplitudes_and_phases(core.cos_sin_coeffs)

    if peak_fap not in ("sidak", "montecarlo", "none"):
        raise ValueError(f"Unknown `peak_fap` {peak_fap!r}. Options: 'sidak', 'montecarlo', 'none'.")
    fap = float("nan")
    if om_arr.size > 0 and peak_fap != "none" and np.isfinite(core.critical_value):
        k_peak = int(np.argmax(core.spectrum))
        if peak_fap == "sidak":
            span_t = float(t0[-1]) if t0[-1] > 0 else 1.0
            band = float(om_arr[-1] - om_arr[0])
            k_eff = float(min(om_arr.size, max(1, int(np.ceil(span_t * band)) + 1)))
            p_min = float(core.pvalues[k_peak])
            fap = float(-np.expm1(k_eff * np.log1p(-min(p_min, 1.0 - 1e-16))))
        else:
            D_l, _ = build_design_matrix(t0, trend_l, j_valid, shared_slope, kf_arr)
            draws = max_lss_null_draws(
                t0, om_arr, orthonormal_design_basis(D_l, pw), pw,
                known_freqs=kf_arr, n_mc=int(n_mc), seed=(20261125, int(om_arr.size)),
            )
            fap = float((1.0 + np.sum(draws >= core.spectrum[k_peak])) / (int(n_mc) + 1.0))

    rn, rn_warns = compute_red_noise_diagnostics(series.times_numeric, core.residuals)
    extra_warns = list(rn_warns)
    if not core.robust_converged:
        extra_warns.append(
            WarningItem(
                code="ROBUST_NOT_CONVERGED",
                severity="caution",
                message=f"Robust ({robust_m}) IRLS reached its iteration limit before converging.",
                suggestion="Treat robust coefficients and spectrum as approximate.",
            )
        )
    if core.rank_deficient:
        extra_warns.append(
            WarningItem(
                code="RANK_DEFICIENT_DESIGN",
                severity="warning",
                message=(
                    f"Design matrix was numerically rank-deficient (cond={core.condition_number:.2e}); "
                    "fell back to SVD pseudoinverse."
                ),
                suggestion="Check for duplicate timestamps, overly narrow jump segments, or collinear frequencies.",
            )
        )
    elif core.condition_number > 1e6:
        extra_warns.append(
            WarningItem(
                code="HIGH_CONDITION_NUMBER",
                severity="caution",
                message=(
                    f"High design-matrix condition number ({core.condition_number:.2e}); individual "
                    "coefficients/amplitudes (e.g. of near-identical known frequencies) are unstable, "
                    "although fitted values remain accurate."
                ),
            )
        )

    diag = DiagnosticReport(
        red_noise=rn,
        condition_number_max=core.condition_number,
        n_input=series.n_input,
        n_valid=series.n_valid,
        n_dropped_nan=series.n_dropped_nan,
        sampling_rate_ratio=series.sampling_rate_ratio,
        warnings=tuple(extra_warns),
    )

    return LssaResult(
        frequencies=core.frequencies,
        spectrum=core.spectrum,
        critical_value=core.critical_value,
        pvalues=core.pvalues,
        cos_sin_coeffs=core.cos_sin_coeffs,
        amplitudes=amps,
        phases=phases,
        residuals=core.residuals,
        norm_res=core.norm_res,
        coeff=core.coeff,
        cov=core.cov,
        alpha=alpha,
        trend=trend_l,
        known_freqs=kf_arr,
        jump_indices=series.valid_to_orig[j_valid] if j_valid.size else j_valid,
        times_numeric=series.times_numeric,
        times_raw=series.times_raw,
        is_datetime=series.is_datetime,
        diagnostics=diag,
        peak_fap=fap,
        peak_fap_method=peak_fap if om_arr.size > 0 else "none",
    )


def allssa(
    t: TimeLike,
    y: ArrayLike,
    *,
    weights: WeightLike | None = None,
    omega: ArrayLike | None = None,
    jump_indices: Sequence[int] | None = None,
    alpha: float = 0.01,
    trend: TrendType = "linear",
    shared_slope: bool = False,
    decimal: int = 1,
    max_iter: int | None = None,
    robust: RobustMethod = "none",
    huber_k: float = 1.345,
    parity_mode: bool = False,
    significance: SignificanceMethod | None = None,
    n_mc: int = 999,
) -> AllssaResult:
    """Antileakage Least-Squares Spectral Analysis (ALLSSA).

    Iteratively identifies and refines statistically significant sinusoidal
    components simultaneously with the trend and optional jump basis,
    suppressing spectral leakage on irregular records (Ghaderpour et al., 2018).

    Parameters
    ----------
    t, y, weights, jump_indices, alpha, trend, shared_slope, robust, huber_k:
        See :func:`lssa`.
    omega:
        Initial coarse cyclic-frequency search grid. Defaults to integer
        frequencies ``1 .. floor(M/2)``.
    decimal:
        Decimal precision for frequency refinement (integer in ``0..4``).
    max_iter:
        Maximum iterations (defaults to ``max(20, 2 * len(omega))``).
    parity_mode:
        If ``True``, disables robust reweighting and uses un-clamped fine grids
        around rounded integer centers to match legacy upstream behavior.
    significance:
        Acceptance rule for each new frequency. ``"montecarlo"`` (default) accepts a
        peak only if its family-wise p-value (exact Gaussian-null Monte Carlo of the
        spectral maximum over the whole searched grid, ``n_mc`` draws) is
        ``<= alpha``; ``"sidak"`` uses an analytical Sidak threshold over the
        effective number of searched frequencies; ``"per_frequency"`` is the
        published rule (Beta critical value at ``alpha`` for a single frequency;
        forced by ``parity_mode=True``), which selects spurious cycles in roughly
        20% of pure-noise series at ``alpha=0.01``.
    n_mc:
        Monte Carlo draws for ``significance="montecarlo"`` (default ``999``).
    """
    trend_l = _validate_spectral_common(trend, alpha)
    robust_m = validate_robust_args(robust, huber_k)
    if not isinstance(decimal, (int, np.integer)) or decimal < 0 or decimal > 4:
        raise ValueError("`decimal` must be an integer between 0 and 4.")
    decimal_int = int(decimal)
    if significance is not None and significance not in ("montecarlo", "sidak", "per_frequency"):
        raise ValueError(
            f"Unknown `significance` {significance!r}. Options: 'montecarlo', 'sidak', 'per_frequency'."
        )
    eff_sig: str = "per_frequency" if parity_mode else (significance or "montecarlo")
    if eff_sig == "montecarlo" and (int(n_mc) + 1) * alpha < 5.0:
        raise ValueError(
            f"`n_mc` too small for alpha={alpha}: need (n_mc + 1) * alpha >= 5 for Monte Carlo significance."
        )

    series = prepare_numeric_series(t, y)
    pw_full = prepare_weights(weights, series.n_input)
    pw = pw_full.subset(series.valid_mask) if series.n_dropped_nan else pw_full

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

    t0 = series.times_numeric - series.times_numeric[0]
    if omega is None or len(omega) == 0:
        om_arr = _default_allssa_omega(t0)
    else:
        om_arr = np.sort(np.asarray(omega, dtype=np.float64).ravel())

    iter_cap = max_iter if max_iter is not None else max(25, 2 * int(om_arr.size))
    core = _allssa_core(
        t0,
        series.values,
        pw,
        om_arr,
        j_valid,
        alpha,
        trend_l,
        shared_slope,
        decimal_int,
        iter_cap,
        parity_mode=parity_mode,
        robust=robust_m,
        huber_k=huber_k,
        significance=eff_sig,
        n_mc=int(n_mc),
    )
    amps, phases = _amplitudes_and_phases(core.cos_sin_coeffs)

    rn, rn_warns = compute_red_noise_diagnostics(series.times_numeric, core.residuals)
    extra_warns = list(rn_warns)
    if omega is None:
        extra_warns.extend(time_unit_warning(series.times_numeric, series.is_datetime))
    if om_arr.size <= 1 and omega is None:
        extra_warns.append(
            WarningItem(
                code="DEGENERATE_DEFAULT_GRID",
                severity="warning",
                message=(
                    f"The default integer frequency grid has {om_arr.size} frequency; it assumes "
                    "time in decimal years (cycles/year) and roughly >= 4 samples per unit time."
                ),
                suggestion="Pass `omega` explicitly in cycles per unit of your time axis, or convert t to years.",
            )
        )
    if not core.converged:
        extra_warns.append(
            WarningItem(
                code="ALLSSA_MAX_ITER",
                severity="caution",
                message=f"ALLSSA reached max_iter={iter_cap} before frequency set stabilized.",
                suggestion="Increase `max_iter` or use a stricter `alpha`.",
            )
        )
    if core.rank_deficient:
        extra_warns.append(
            WarningItem(
                code="RANK_DEFICIENT_DESIGN",
                severity="warning",
                message=(
                    f"Design matrix was numerically rank-deficient during ALLSSA "
                    f"(max cond={core.condition_number_max:.2e})."
                ),
            )
        )

    diag = DiagnosticReport(
        red_noise=rn,
        condition_number_max=core.condition_number_max,
        n_input=series.n_input,
        n_valid=series.n_valid,
        n_dropped_nan=series.n_dropped_nan,
        sampling_rate_ratio=series.sampling_rate_ratio,
        warnings=tuple(extra_warns),
    )

    return AllssaResult(
        trend_coeff=core.trend_coeff,
        cos_sin_coeffs=core.cos_sin_coeffs,
        frequencies=core.frequencies,
        amplitudes=amps,
        phases=phases,
        residuals=core.residuals,
        norm_res=core.norm_res,
        cov=core.cov,
        alpha=alpha,
        trend=trend_l,
        n_iter=core.n_iter,
        converged=core.converged,
        times_numeric=series.times_numeric,
        times_raw=series.times_raw,
        is_datetime=series.is_datetime,
        diagnostics=diag,
        significance=eff_sig,
    )
