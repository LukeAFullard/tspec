"""Diagnostic structures, red-noise autocorrelation checks, and warning engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import minimize_scalar

from ._types import FloatArray

WarningSeverity = Literal["info", "caution", "warning"]
AutocorrSeverity = Literal["none", "moderate", "strong"]


@dataclass(frozen=True)
class WarningItem:
    """Structured diagnostic notice attached to result objects.

    Attributes
    ----------
    code : str
        Machine-readable warning identifier (e.g. ``'STRONG_AUTOCORRELATION'``).
    severity : {"info", "caution", "warning"}
        Severity level of the diagnostic notice.
    message : str
        Human-readable description of the detected condition.
    suggestion : str
        Optional actionable guidance for mitigating the condition.
    """

    code: str
    severity: WarningSeverity
    message: str
    suggestion: str = ""


@dataclass(frozen=True)
class RedNoiseDiagnostics:
    """Residual autocorrelation (red-noise) diagnostics for uneven series.

    Attributes
    ----------
    lag1_autocorr : float
        Sequential lag-1 sample autocorrelation of residuals (``NaN`` if ``n < 5``).
    ou_decay_corr : float
        Equivalent lag-1 correlation at median time step under continuous-time
        Ornstein-Uhlenbeck decay ``exp(-dt / tau)``.
    effective_sample_size : float
        Autocorrelation-adjusted effective sample size ``n_eff``.
    severity : {"none", "moderate", "strong"}
        Qualitative red-noise classification.
    """

    lag1_autocorr: float
    ou_decay_corr: float
    effective_sample_size: float
    severity: AutocorrSeverity


@dataclass(frozen=True)
class DiagnosticReport:
    """Consolidated numerical, sampling, and statistical diagnostics.

    Attributes
    ----------
    red_noise : RedNoiseDiagnostics or None
        Residual autocorrelation diagnostics, when available.
    condition_number_max : float
        Maximum design-matrix condition number encountered during fitting.
    n_input : int
        Total number of input observations before dropping ``NaN`` entries.
    n_valid : int
        Number of finite observations retained for analysis.
    n_dropped_nan : int
        Number of non-finite observations dropped transparently.
    sampling_rate_ratio : float
        Ratio of upper-to-lower sampling density across the record.
    warnings : tuple of WarningItem
        Structured warnings and cautions emitted during analysis.
    """

    red_noise: RedNoiseDiagnostics | None
    condition_number_max: float
    n_input: int
    n_valid: int
    n_dropped_nan: int
    sampling_rate_ratio: float
    warnings: tuple[WarningItem, ...] = ()

    def with_warnings(self, extra: list[WarningItem] | tuple[WarningItem, ...]) -> DiagnosticReport:
        """Return a new report appending additional warnings."""
        if not extra:
            return self
        return DiagnosticReport(
            red_noise=self.red_noise,
            condition_number_max=self.condition_number_max,
            n_input=self.n_input,
            n_valid=self.n_valid,
            n_dropped_nan=self.n_dropped_nan,
            sampling_rate_ratio=self.sampling_rate_ratio,
            warnings=self.warnings + tuple(extra),
        )


def compute_red_noise_diagnostics(
    t: FloatArray,
    residuals: FloatArray,
    *,
    moderate_threshold: float = 0.30,
    strong_threshold: float = 0.60,
) -> tuple[RedNoiseDiagnostics, tuple[WarningItem, ...]]:
    """Compute lag-1 and irregular continuous-time AR(1) (OU) residual autocorrelation.

    Standard LSSA/ALLSSA Beta/F-tests and STPD t-tests assume independent Gaussian
    residuals. Strong positive serial correlation inflates false-positive rates;
    this routine quantifies both sequential lag-1 correlation and equivalent
    correlation at the median spacing ``dt_med`` under an Ornstein-Uhlenbeck
    decay ``corr(dt) = exp(-dt / tau)``.
    """
    n = int(residuals.size)
    if n < 5:
        return (
            RedNoiseDiagnostics(
                lag1_autocorr=float("nan"),
                ou_decay_corr=float("nan"),
                effective_sample_size=float(n),
                severity="none",
            ),
            (),
        )

    r = residuals - float(np.mean(residuals))
    var = float(np.dot(r, r)) / n
    if var <= 1e-24:
        return (
            RedNoiseDiagnostics(
                lag1_autocorr=0.0,
                ou_decay_corr=0.0,
                effective_sample_size=float(n),
                severity="none",
            ),
            (),
        )

    r0 = r[:-1]
    r1 = r[1:]
    denom = float(np.sqrt(np.dot(r0, r0) * np.dot(r1, r1)))
    phi1 = float(np.dot(r0, r1) / denom) if denom > 0.0 else 0.0
    phi1 = float(np.clip(phi1, -0.999, 0.999))

    # Continuous-time OU correlation estimate at median time step
    dt = np.diff(t)
    dt_pos = dt[dt > 0]
    if dt_pos.size >= 3 and phi1 > 0.0:
        dt_med = float(np.median(dt_pos))
        # When spacing is nearly uniform, continuous-time OU reduces analytically to lag-1 phi1
        if float(np.std(dt_pos)) <= 0.05 * dt_med:
            ou_corr = float(phi1)
        else:
            valid_dt = dt > 0
            dt_norm = dt[valid_dt] / dt_med
            prod = (r0[valid_dt] * r1[valid_dt]) / var

            def _ou_loss(phi_med: float) -> float:
                pred = np.power(phi_med, dt_norm)
                diff = prod - pred
                return float(np.dot(diff, diff))

            res_opt = minimize_scalar(_ou_loss, bounds=(0.0, 0.999), method="bounded")
            ou_corr = float(res_opt.x) if res_opt.success else max(0.0, phi1)
    else:
        ou_corr = max(0.0, phi1)

    # Exact O(n) continuous-time OU effective sample size:
    # n_eff = n^2 / sum_{i,j} exp(-|t_i - t_j| / tau), with exp(-dt_med / tau) = ou_corr
    if ou_corr > 0.0 and dt_pos.size >= 1:
        dt_med = float(np.median(dt_pos))
        if dt_med > 0.0:
            log_phi = float(np.log(np.clip(ou_corr, 1e-12, 0.999999)))
            step_corr = np.exp(log_phi * (np.maximum(dt, 0.0) / dt_med))
            # L_k = sum_{j=0}^{k-1} corr(t_j, t_k) satisfies L_k = r_k * (1 + L_{k-1})
            L_curr = 0.0
            sum_upper = 0.0
            for rk in step_corr:
                L_curr = float(rk) * (1.0 + L_curr)
                sum_upper += L_curr
            corr_sum = float(n) + 2.0 * sum_upper
            n_eff = float(np.clip((n * n) / max(corr_sum, float(n)), 1.0, float(n)))
        else:
            n_eff = float(n)
    elif phi1 > 0.0:
        n_eff = float(max(1.0, n * (1.0 - phi1) / (1.0 + phi1)))
    else:
        n_eff = float(n)

    metric = max(phi1, ou_corr)
    warnings_list: list[WarningItem] = []
    severity: AutocorrSeverity = "none"
    if metric >= strong_threshold:
        severity = "strong"
        warnings_list.append(
            WarningItem(
                code="STRONG_AUTOCORRELATION",
                severity="warning",
                message=(
                    f"Strong positive residual autocorrelation detected "
                    f"(lag-1 r={phi1:.2f}, median-step OU r={ou_corr:.2f}; "
                    f"effective sample size n_eff≈{n_eff:.0f} vs n={n})."
                ),
                suggestion=(
                    "Nominal p-values assume independent errors and may over-detect "
                    "spurious changes. Consider passing an inverse-covariance weight "
                    "matrix P = C_f^{-1}, tightening alpha (e.g., 0.001), or verifying "
                    "changes on coarser temporal aggregates."
                ),
            )
        )
    elif metric >= moderate_threshold:
        severity = "moderate"
        warnings_list.append(
            WarningItem(
                code="MODERATE_AUTOCORRELATION",
                severity="caution",
                message=(
                    f"Moderate positive residual autocorrelation detected "
                    f"(lag-1 r={phi1:.2f}, median-step OU r={ou_corr:.2f}; "
                    f"n_eff≈{n_eff:.0f} vs n={n})."
                ),
                suggestion=(
                    "False-positive rates may be modestly elevated above nominal alpha. "
                    "Consider using a stricter alpha (e.g., 0.005) or supplying correlated "
                    "error weights `weights`."
                ),
            )
        )

    return (
        RedNoiseDiagnostics(
            lag1_autocorr=phi1,
            ou_decay_corr=ou_corr,
            effective_sample_size=n_eff,
            severity=severity,
        ),
        tuple(warnings_list),
    )


def time_unit_warning(t_num: FloatArray, is_datetime: bool) -> tuple[WarningItem, ...]:
    """Warn when a numeric time axis does not look like decimal years.

    Defaults such as JUST/ALLSSA/monitor frequency grids (cycles/year), window sizes
    derived from samples per unit time, ``jump_th`` / ``tp_th`` (years), and the
    1 cycle/year abrupt/phenological split all assume decimal years.
    """
    if is_datetime or t_num.size < 2:
        return ()
    span = float(t_num[-1] - t_num[0])
    if span <= 0.0:
        return ()
    rate = (t_num.size - 1) / span
    if span > 300.0 or rate > 1000.0 or rate < 0.25:
        return (
            WarningItem(
                code="TIME_UNIT_ASSUMPTION",
                severity="warning",
                message=(
                    f"Numeric time axis spans {span:.4g} units at {rate:.3g} samples/unit; tspec "
                    "defaults (frequency grids in cycles/year, window sizes, jump_th/tp_th in years, "
                    "1 cy/yr monitoring split) assume decimal years."
                ),
                suggestion="Convert t to decimal years (or pass datetimes), or set omega/size/thresholds explicitly.",
            ),
        )
    return ()
