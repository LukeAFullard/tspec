"""Robust linear, piecewise-linear, and non-parametric trend estimators.

Provides clean-room, strictly ``numpy + scipy`` implementations (no extra
dependencies) of:

- **Huber IRLS M-estimation** (:func:`solve_huber_irls`) with MAD residual
  scale and Huber (1981) asymptotic variance estimation.
- **Schrader–Hettmansperger (1980) robust nested F-test**
  (:func:`huber_nested_ftest`) for outlier-resistant model comparison.
- **Exact L1 (LAD) continuous hinge-spline regression**
  (:func:`fit_l1_piecewise`) via sparse HiGHS linear programming.
- **Theil–Sen and Akritas–Theil–Sen (ATS)** segment slope, intercept,
  distribution-free confidence interval, and Mann–Kendall p-value estimation
  (:func:`fit_theil_sen_segment`).

References
----------
- Huber, P. J. (1981). *Robust Statistics*. John Wiley & Sons.
- Schrader, R. M., & Hettmansperger, T. P. (1980). Improving analysis of
  variance by using M-estimators. *Biometrika*, 67(1), 93-101.
- Sen, P. K. (1968). Estimates of the regression coefficient based on
  Kendall's tau. *JASA*, 63(324), 1379-1389.
- Akritas, M. G., Murphy, S. A., & LaValley, M. P. (1995). The Theil-Sen
  estimator with doubly censored data and applications to astronomy.
  *JASA*, 90(429), 170-177.
- Gilbert, R. O. (1987). *Statistical Methods for Environmental Pollution
  Monitoring*. Van Nostrand Reinhold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import linalg, sparse
from scipy.optimize import linprog
from scipy.stats import f as f_dist
from scipy.stats import norm

from ._linalg import (
    PreparedWeights,
    WeightedOLSResult,
    _detect_intercept_partition_cols,
    solve_weighted_ols,
    whiten_matrix,
    whiten_vector,
)
from ._types import FloatArray, IntArray

RobustMethod = Literal["none", "huber", "mm"]
SlopeEstimator = Literal["ols", "sen", "l1", "huber"]

_MAD_NORM = 1.482602218505602  # 1 / Phi^{-1}(0.75)
_MEAN_ABS_NORM = 1.2533141373155001  # sqrt(pi / 2)
TUKEY_BIWEIGHT_C = 4.685061  # 95% Gaussian efficiency


def validate_robust_args(robust: str, huber_k: float) -> RobustMethod:
    """Validate ``robust`` method and ``huber_k`` tuning constant."""
    if robust not in ("none", "huber", "mm"):
        raise ValueError(
            f"Unknown `robust` method {robust!r}. Options: 'none', 'huber', 'mm'."
        )
    if not np.isfinite(huber_k) or huber_k <= 0.0:
        raise ValueError(f"`huber_k` must be finite and strictly positive; got {huber_k}.")
    return robust  # type: ignore[return-value]


def huber_rho(u: FloatArray, k: float) -> FloatArray:
    """Evaluate Huber loss ``rho_k(u)`` elementwise."""
    au = np.abs(u)
    return np.where(au <= k, 0.5 * au * au, k * au - 0.5 * k * k)


def huber_psi(u: FloatArray, k: float) -> FloatArray:
    """Evaluate Huber score ``psi_k(u) = clip(u, -k, k)`` elementwise."""
    return np.clip(u, -k, k)


def tukey_rho(u: FloatArray, c: float = TUKEY_BIWEIGHT_C) -> FloatArray:
    """Evaluate Tukey bisquare (biweight) bounded loss ``rho_c(u)`` elementwise."""
    t = np.clip(np.abs(u) / c, 0.0, 1.0)
    return np.asarray((c * c / 6.0) * (1.0 - (1.0 - t * t) ** 3), dtype=np.float64)


def tukey_psi(u: FloatArray, c: float = TUKEY_BIWEIGHT_C) -> FloatArray:
    """Evaluate Tukey bisquare redescending score ``psi_c(u)`` elementwise."""
    t = u / c
    return np.where(np.abs(u) < c, u * (1.0 - t * t) ** 2, 0.0)


def tukey_weight(u: FloatArray, c: float = TUKEY_BIWEIGHT_C) -> FloatArray:
    """Evaluate Tukey bisquare IRLS weight ``w_c(u) = (1 - (u/c)^2)_+^2``."""
    t = u / c
    return np.where(np.abs(u) < c, (1.0 - t * t) ** 2, 0.0)


def _robust_scale_estimate(
    r_w: FloatArray,
    scale_ref: float,
    dof_factor: float = 1.0,
) -> float:
    """Estimate robust residual scale via finite-sample-adjusted MAD with fallback."""
    if r_w.size == 0:
        return 0.0
    mad = float(dof_factor * _MAD_NORM * np.median(np.abs(r_w)))
    floor = 1e-14 * max(scale_ref, 1e-300)
    if mad > floor:
        return mad
    mean_abs = float(dof_factor * _MEAN_ABS_NORM * np.mean(np.abs(r_w)))
    if mean_abs > floor:
        return mean_abs
    return 0.0


@dataclass(frozen=True)
class HuberIRLSResult:
    """Result of a Huber IRLS or MM-biweight M-estimation fit."""

    ols: WeightedOLSResult
    huber_weights: FloatArray
    pw_effective: PreparedWeights
    scale: float
    loss: float
    lambda_scale: float
    tau_sq: float
    n_iter: int
    converged: bool


def solve_huber_irls(
    D: FloatArray,
    y: FloatArray,
    pw: PreparedWeights,
    *,
    k: float = 1.345,
    fixed_scale: float | None = None,
    col_scales: FloatArray | None = None,
    rcond: float = 1e-12,
    max_iter: int = 25,
    tol: float = 1e-7,
    compute_cov: bool = True,
) -> HuberIRLSResult:
    """Fit linear model ``y ~ D c`` by convex Huber IRLS M-estimation.

    Supports all four :class:`PreparedWeights` modes (``none``, ``scalar``,
    ``vector``, and 2D SPD ``matrix``). When ``fixed_scale`` is supplied, the
    residual scale is held fixed across iterations to guarantee exact convex
    objective nesting across nested models.
    """
    n, nc = D.shape
    if n == 0:
        ols0 = solve_weighted_ols(D, y, pw, col_scales=col_scales, compute_cov=compute_cov)
        return HuberIRLSResult(
            ols=ols0,
            huber_weights=np.ones(0, dtype=np.float64),
            pw_effective=pw,
            scale=0.0,
            loss=0.0,
            lambda_scale=0.0,
            tau_sq=0.0,
            n_iter=0,
            converged=True,
        )

    # Shift-invariant centering when D spans 1_n so huge baselines do not lose digits
    k_int = _detect_intercept_partition_cols(D)
    y_shift = float(np.median(y)) if k_int > 0 else 0.0
    y_work = y - y_shift if y_shift != 0.0 else y

    # Pre-whiten once under base weights pw (avoids repeated Cholesky for 2D P)
    y_w0 = whiten_vector(y_work, pw)
    D_w0 = whiten_matrix(D, pw)
    scale_ref = float(np.sqrt(np.dot(y_w0, y_w0) / max(1, n)))

    # Initial fit via QR on pre-whitened (D_w0, y_w0); also checks conditioning once
    use_gram = False
    if nc == 0:
        prev_coeff_s = np.empty(0, dtype=np.float64)
        r_w = y_w0.copy()
    else:
        Q0, R0 = np.linalg.qr(D_w0, mode="reduced")
        d_abs0 = np.abs(np.diag(R0))
        d_max0 = float(np.max(d_abs0)) if d_abs0.size else 0.0
        d_min0 = float(np.min(d_abs0)) if d_abs0.size else 0.0
        if d_max0 > 0.0 and d_min0 > 1e-8 * d_max0:
            prev_coeff_s = np.asarray(
                linalg.solve_triangular(R0, Q0.T @ y_w0, lower=False, check_finite=False),
                dtype=np.float64,
            )
            use_gram = d_min0 > 1e-5 * d_max0
        else:
            prev_coeff_s = np.asarray(
                np.linalg.lstsq(D_w0, y_w0, rcond=rcond)[0], dtype=np.float64
            )
        r_w = y_w0 - D_w0 @ prev_coeff_s

    D_w0_T = D_w0.T

    def _fast_wls_step(w_vec: FloatArray) -> tuple[FloatArray, FloatArray]:
        if nc == 0:
            return np.empty(0, dtype=np.float64), y_w0.copy()
        if use_gram:
            DwT = D_w0_T * w_vec
            try:
                c_s = np.linalg.solve(DwT @ D_w0, DwT @ y_w0)
                if np.all(np.isfinite(c_s)):
                    return np.asarray(c_s, dtype=np.float64), y_w0 - D_w0 @ c_s
            except np.linalg.LinAlgError:
                pass
        sw = np.sqrt(w_vec)
        Dw = D_w0 * sw[:, None]
        yw = y_w0 * sw
        Q_i, R_i = np.linalg.qr(Dw, mode="reduced")
        d_abs = np.abs(np.diag(R_i))
        d_max = float(np.max(d_abs)) if d_abs.size else 0.0
        d_min = float(np.min(d_abs)) if d_abs.size else 0.0
        if d_max > 0.0 and d_min > 1e-8 * d_max:
            c_s = linalg.solve_triangular(R_i, Q_i.T @ yw, lower=False, check_finite=False)
        else:
            c_s = np.linalg.lstsq(Dw, yw, rcond=rcond)[0]
        return np.asarray(c_s, dtype=np.float64), y_w0 - D_w0 @ c_s

    dof_factor = float(np.sqrt(n / max(1, n - nc)))
    if fixed_scale is not None:
        scale = float(fixed_scale)
    else:
        scale = _robust_scale_estimate(r_w, scale_ref, dof_factor=dof_factor)

    if scale <= 1e-14 * max(scale_ref, 1e-300):
        # Noiseless / exact-fit regime: weights are identically 1
        ols_exact = solve_weighted_ols(
            D, y, pw, col_scales=col_scales, rcond=rcond, compute_cov=compute_cov
        )
        return HuberIRLSResult(
            ols=ols_exact,
            huber_weights=np.ones(n, dtype=np.float64),
            pw_effective=pw,
            scale=0.0,
            loss=0.0,
            lambda_scale=0.0,
            tau_sq=0.0,
            n_iter=0,
            converged=True,
        )

    w_h = np.ones(n, dtype=np.float64)
    converged = False
    n_iter = 0

    for it in range(1, max_iter + 1):
        n_iter = it
        u = np.abs(r_w) / scale
        w_h = np.where(u <= k, 1.0, k / np.maximum(u, k))
        cur_coeff_s, r_w = _fast_wls_step(w_h)

        # Allow MAD scale to stabilize on early iterations if not held fixed
        if fixed_scale is None and it <= 3:
            new_scale = _robust_scale_estimate(r_w, scale_ref, dof_factor=dof_factor)
            if new_scale > 1e-14 * max(scale_ref, 1e-300):
                scale = new_scale

        diff = float(np.max(np.abs(cur_coeff_s - prev_coeff_s))) if nc > 0 else 0.0
        ref = float(np.max(np.abs(cur_coeff_s))) if nc > 0 else 0.0
        prev_coeff_s = cur_coeff_s
        if diff <= tol * max(ref, 1e-300):
            converged = True
            break

    # Final effective weight structure & WeightedOLSResult for downstream spectral/Q use
    u_final = np.abs(r_w) / scale
    w_h = np.where(u_final <= k, 1.0, k / np.maximum(u_final, k))
    pw_eff = pw.scale_whitened(w_h)
    ols_final = solve_weighted_ols(
        D, y, pw_eff, col_scales=col_scales, rcond=rcond, compute_cov=compute_cov
    )

    # Recompute exact base-whitened residuals for Huber loss & asymptotic scale
    r_w_final = whiten_vector(ols_final.residuals, pw)
    u_signed = r_w_final / scale
    loss = float(np.sum(huber_rho(u_signed, k)))
    psi_vals = huber_psi(u_signed, k)
    m_inlier = max(0.1, float(np.mean(np.abs(u_signed) <= k)))
    dof = max(1, n - ols_final.rank)
    psi_sq_mean = float(np.sum(psi_vals * psi_vals)) / dof
    k_corr = 1.0 + (ols_final.rank / max(1, n)) * (1.0 - m_inlier) / m_inlier
    lambda_scale = k_corr * psi_sq_mean / m_inlier
    tau_sq = (k_corr * scale) ** 2 * psi_sq_mean / (m_inlier * m_inlier)

    # Replace OLS covariance with Huber (1981) asymptotic covariance tau^2 (D^T P D)^+
    if compute_cov and nc > 0 and n > ols_final.rank:
        _, s_b, vt_b = linalg.svd(D_w0, full_matrices=False)
        keep_b = s_b > rcond * s_b[0] if (s_b.size > 0 and s_b[0] > 0.0) else np.zeros(0, dtype=bool)
        if np.any(keep_b):
            sv = vt_b[keep_b, :].T / s_b[keep_b][None, :]
            cov_s = tau_sq * (sv @ sv.T)
            cov_rob = (
                (col_scales[:, None] * cov_s) * col_scales[None, :]
                if col_scales is not None
                else cov_s
            )
            ols_final = WeightedOLSResult(
                coeff=ols_final.coeff,
                residuals=ols_final.residuals,
                whitened_residuals=ols_final.whitened_residuals,
                norm_res=ols_final.norm_res,
                y_w_norm_sq=ols_final.y_w_norm_sq,
                cov=np.asarray(cov_rob, dtype=np.float64),
                Q=ols_final.Q,
                condition_number=ols_final.condition_number,
                rank=ols_final.rank,
                rank_deficient=ols_final.rank_deficient,
            )

    return HuberIRLSResult(
        ols=ols_final,
        huber_weights=w_h,
        pw_effective=pw_eff,
        scale=scale,
        loss=loss,
        lambda_scale=lambda_scale,
        tau_sq=tau_sq,
        n_iter=n_iter,
        converged=converged,
    )


def solve_mm_biweight(
    D: FloatArray,
    y: FloatArray,
    pw: PreparedWeights,
    *,
    k: float = 1.345,
    c: float = TUKEY_BIWEIGHT_C,
    fixed_scale: float | None = None,
    col_scales: FloatArray | None = None,
    rcond: float = 1e-12,
    max_iter: int = 15,
    tol: float = 1e-6,
    compute_cov: bool = True,
) -> HuberIRLSResult:
    """Fit linear model via MM-type estimation (Huber IRLS start + Tukey biweight).

    Unlike Yohai's (1987) MM-estimator, the start is a monotone Huber M-estimate rather
    than a high-breakdown S-estimate, so the breakdown point is design-dependent (it
    breaks down under ~35% sustained vertical contamination at one end of the record).

    Initializes from convex Huber IRLS (guaranteeing zero masking of genuine
    sustained jumps) and refines at the fixed finite-sample MAD scale with
    Tukey's bisquare redescending weight (``c = 4.685``, 95% Gaussian efficiency,
    zero influence for extreme isolated spikes ``|u| >= c``).
    """
    n, nc = D.shape
    hr_init = solve_huber_irls(
        D,
        y,
        pw,
        k=k,
        fixed_scale=fixed_scale,
        col_scales=col_scales,
        rcond=rcond,
        compute_cov=False,
    )
    if n == 0 or hr_init.scale <= 0.0:
        if compute_cov and n > 0:
            return solve_huber_irls(
                D,
                y,
                pw,
                k=k,
                fixed_scale=fixed_scale,
                col_scales=col_scales,
                rcond=rcond,
                compute_cov=True,
            )
        return hr_init

    scale = hr_init.scale
    k_int = _detect_intercept_partition_cols(D)
    y_shift = float(np.median(y)) if k_int > 0 else 0.0
    y_work = y - y_shift if y_shift != 0.0 else y
    y_w0 = whiten_vector(y_work, pw)
    D_w0 = whiten_matrix(D, pw)
    D_w0_T = D_w0.T

    # Check conditioning once for fast Gram solve vs QR fallback
    use_gram = False
    if nc > 0:
        _, R0 = np.linalg.qr(D_w0, mode="reduced")
        d_abs0 = np.abs(np.diag(R0))
        d_max0 = float(np.max(d_abs0)) if d_abs0.size else 0.0
        d_min0 = float(np.min(d_abs0)) if d_abs0.size else 0.0
        use_gram = d_max0 > 0.0 and d_min0 > 1e-5 * d_max0

    r_w = whiten_vector(hr_init.ols.residuals, pw)
    prev_c = hr_init.ols.coeff / col_scales if col_scales is not None else hr_init.ols.coeff.copy()
    if k_int > 0 and y_shift != 0.0:
        prev_c = prev_c.copy()
        prev_c[:k_int] -= y_shift

    w_bi = np.ones(n, dtype=np.float64)
    converged = False
    n_iter = 0
    for it in range(1, max_iter + 1):
        n_iter = it
        w_try = tukey_weight(r_w / scale, c)
        # Guard against total weight collapse when fitting a misspecified null model
        # (e.g., 1-piece null evaluated at small full-model scale on a huge real jump).
        if int(np.sum(w_try > 1e-6)) <= nc:
            w_bi = hr_init.huber_weights
            break
        w_bi = w_try
        if nc == 0:
            converged = True
            break
        solved = False
        if use_gram:
            DwT = D_w0_T * w_bi
            try:
                cur_c = np.linalg.solve(DwT @ D_w0, DwT @ y_w0)
                if np.all(np.isfinite(cur_c)):
                    solved = True
            except np.linalg.LinAlgError:
                solved = False
        if not solved:
            sw = np.sqrt(w_bi)
            Dw = D_w0 * sw[:, None]
            yw = y_w0 * sw
            cur_c = np.linalg.lstsq(Dw, yw, rcond=rcond)[0]
        r_w = y_w0 - D_w0 @ cur_c
        diff = float(np.max(np.abs(cur_c - prev_c)))
        ref = float(np.max(np.abs(cur_c)))
        prev_c = np.asarray(cur_c, dtype=np.float64)
        if diff <= tol * max(ref, 1e-300):
            converged = True
            break

    w_final = tukey_weight(r_w / scale, c)
    if int(np.sum(w_final > 1e-6)) <= nc:
        w_final = hr_init.huber_weights
    # Floor tiny positive weight (1e-12) so PreparedWeights stays strictly positive
    w_eff_vec = np.maximum(w_final, 1e-12)
    pw_eff = pw.scale_whitened(w_eff_vec)
    ols_final = solve_weighted_ols(
        D, y, pw_eff, col_scales=col_scales, rcond=rcond, compute_cov=compute_cov
    )

    r_w_final = whiten_vector(ols_final.residuals, pw)
    u_signed = r_w_final / scale
    loss = float(np.sum(tukey_rho(u_signed, c)))
    psi_vals = tukey_psi(u_signed, c)
    t2 = (u_signed / c) ** 2
    psi_prime = np.where(np.abs(u_signed) < c, (1.0 - t2) * (1.0 - 5.0 * t2), 0.0)
    m_inlier = max(0.1, float(np.mean(psi_prime)))
    dof = max(1, n - ols_final.rank)
    psi_sq_mean = float(np.sum(psi_vals * psi_vals)) / dof
    k_corr = 1.0 + (ols_final.rank / max(1, n)) * float(np.var(psi_prime)) / (m_inlier * m_inlier)
    lambda_scale = k_corr * psi_sq_mean / m_inlier
    tau_sq = (k_corr * scale) ** 2 * psi_sq_mean / (m_inlier * m_inlier)

    if compute_cov and nc > 0 and n > ols_final.rank:
        _, s_b, vt_b = linalg.svd(D_w0, full_matrices=False)
        keep_b = s_b > rcond * s_b[0] if (s_b.size > 0 and s_b[0] > 0.0) else np.zeros(0, dtype=bool)
        if np.any(keep_b):
            sv = vt_b[keep_b, :].T / s_b[keep_b][None, :]
            cov_s = tau_sq * (sv @ sv.T)
            cov_rob = (
                (col_scales[:, None] * cov_s) * col_scales[None, :]
                if col_scales is not None
                else cov_s
            )
            ols_final = WeightedOLSResult(
                coeff=ols_final.coeff,
                residuals=ols_final.residuals,
                whitened_residuals=ols_final.whitened_residuals,
                norm_res=ols_final.norm_res,
                y_w_norm_sq=ols_final.y_w_norm_sq,
                cov=np.asarray(cov_rob, dtype=np.float64),
                Q=ols_final.Q,
                condition_number=ols_final.condition_number,
                rank=ols_final.rank,
                rank_deficient=ols_final.rank_deficient,
            )

    return HuberIRLSResult(
        ols=ols_final,
        huber_weights=w_final,
        pw_effective=pw_eff,
        scale=scale,
        loss=loss,
        lambda_scale=lambda_scale,
        tau_sq=tau_sq,
        n_iter=hr_init.n_iter + n_iter,
        converged=converged,
    )


def solve_robust_irls(
    D: FloatArray,
    y: FloatArray,
    pw: PreparedWeights,
    *,
    method: RobustMethod = "huber",
    k: float = 1.345,
    fixed_scale: float | None = None,
    col_scales: FloatArray | None = None,
    rcond: float = 1e-12,
    compute_cov: bool = True,
) -> HuberIRLSResult:
    """Dispatch robust M-estimation to Huber IRLS (``'huber'``) or MM-biweight (``'mm'``)."""
    if method == "mm":
        return solve_mm_biweight(
            D,
            y,
            pw,
            k=k,
            fixed_scale=fixed_scale,
            col_scales=col_scales,
            rcond=rcond,
            compute_cov=compute_cov,
        )
    return solve_huber_irls(
        D,
        y,
        pw,
        k=k,
        fixed_scale=fixed_scale,
        col_scales=col_scales,
        rcond=rcond,
        compute_cov=compute_cov,
    )


def huber_nested_ftest(
    D_null: FloatArray,
    D_full: FloatArray,
    y: FloatArray,
    pw: PreparedWeights,
    *,
    method: RobustMethod = "huber",
    k: float = 1.345,
    col_scales_null: FloatArray | None = None,
    col_scales_full: FloatArray | None = None,
    full_fit: HuberIRLSResult | None = None,
) -> tuple[float, float, HuberIRLSResult, HuberIRLSResult]:
    """Schrader–Hettmansperger (1980) robust nested F-test for ``D_null subset D_full``.

    Evaluates both models at the shared full-model residual scale ``s_1`` using
    Huber (``method='huber'``) or Huber-started Tukey-biweight MM-type (``method='mm'``), returning
    ``(F_rob, pval, full_fit, null_fit)``.
    """
    n = int(y.size)
    res_full = (
        full_fit
        if full_fit is not None
        else solve_robust_irls(
            D_full, y, pw, method=method, k=k, col_scales=col_scales_full, compute_cov=False
        )
    )
    dof2 = n - res_full.ols.rank
    if dof2 <= 0:
        res_null = solve_robust_irls(
            D_null, y, pw, method=method, k=k, col_scales=col_scales_null, compute_cov=False
        )
        return float("nan"), float("nan"), res_full, res_null

    if float(np.ptp(y)) <= 1e-13 * abs(float(np.mean(y))):
        res_null = solve_robust_irls(
            D_null, y, pw, method=method, k=k, col_scales=col_scales_null, compute_cov=False
        )
        return 0.0, 1.0, res_full, res_null

    if res_full.scale <= 0.0 or res_full.lambda_scale <= 0.0:
        res_null = solve_robust_irls(
            D_null, y, pw, method=method, k=k, col_scales=col_scales_null, compute_cov=False
        )
        tol = 1e-26 * max(res_null.ols.norm_res, res_full.ols.y_w_norm_sq, float(np.finfo(np.float64).tiny))
        if res_null.ols.norm_res > tol:
            return float("inf"), 0.0, res_full, res_null
        return 0.0, 1.0, res_full, res_null

    res_null = solve_robust_irls(
        D_null,
        y,
        pw,
        method=method,
        k=k,
        fixed_scale=res_full.scale,
        col_scales=col_scales_null,
        compute_cov=False,
    )
    q = max(1, res_full.ols.rank - res_null.ols.rank)
    drop = max(0.0, 2.0 * (res_null.loss - res_full.loss))
    f_stat = (drop / q) / res_full.lambda_scale
    pval = float(f_dist.sf(f_stat, q, dof2))
    return f_stat, pval, res_full, res_null


def fit_l1_piecewise(
    t_num: FloatArray,
    y: FloatArray,
    tp_sorted: IntArray,
    w: FloatArray | None = None,
    *,
    fallback_flag: list[bool] | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray, float]:
    """Fit a globally optimal continuous L1 (LAD) piecewise-linear hinge-spline.

    Solves ``min sum_i w_i |y_i - (b0 + a0 * u_i + sum_k delta_k (u_i - u_tau_k)_+)|``
    exactly via HiGHS sparse linear programming. Strictly continuous at every
    turning point. Robust to isolated vertical outliers; its breakdown point is
    design-dependent (about 0.29 for an intercept + slope on a uniform design, He et al. 1990),
    not 50%.
    """
    n = int(t_num.size)
    m = int(tp_sorted.size)
    t0 = t_num - t_num[0]
    s_t = float(t0[-1]) if float(t0[-1]) > 0.0 else 1.0
    u = t0 / s_t

    p = m + 2
    D = np.empty((n, p), dtype=np.float64)
    D[:, 0] = 1.0
    D[:, 1] = u
    for k_i, tp in enumerate(tp_sorted):
        D[:, 2 + k_i] = np.maximum(0.0, u - u[tp])

    # Median-center y for shift-invariant LP precision
    y_med = float(np.median(y))
    y_c = y - y_med
    w_vec = np.ones(n, dtype=np.float64) if w is None else np.asarray(w, dtype=np.float64)

    # Variables: [c (p free), e_pos (n >= 0), e_neg (n >= 0)]
    c_obj = np.concatenate([np.zeros(p, dtype=np.float64), w_vec, w_vec])
    eye_n = sparse.eye(n, format="csc")
    A_eq = sparse.hstack([sparse.csc_matrix(D), eye_n, -eye_n], format="csc")
    bounds = [(None, None)] * p + [(0.0, None)] * (2 * n)

    lp = linprog(c_obj, A_eq=A_eq, b_eq=y_c, bounds=bounds, method="highs")
    if not lp.success or lp.x is None:
        # Fallback to Huber IRLS if LP fails on degenerate input
        if fallback_flag is not None:
            fallback_flag.append(True)
        pw = PreparedWeights(kind="vector", vector=w_vec, sqrt_vector=np.sqrt(w_vec)) if w is not None else PreparedWeights(kind="none")
        hr = solve_huber_irls(D, y_c, pw, compute_cov=False)
        coeff = hr.ols.coeff
        cond_num = hr.ols.condition_number
    else:
        coeff = np.asarray(lp.x[:p], dtype=np.float64)
        svals = linalg.svdvals(D)
        cond_num = float(svals[0] / svals[-1]) if (svals.size > 0 and svals[-1] > 0.0) else float("inf")

    y_fit = y_med + D @ coeff
    b0 = y_med + float(coeff[0])
    a0 = float(coeff[1]) / s_t
    deltas = coeff[2:] / s_t

    slopes = np.empty(m + 1, dtype=np.float64)
    intercepts = np.empty(m + 1, dtype=np.float64)
    slopes[0] = a0
    intercepts[0] = b0 - a0 * float(t_num[0])
    cum_slope = a0
    for k_i, tp in enumerate(tp_sorted):
        cum_slope += float(deltas[k_i])
        slopes[k_i + 1] = cum_slope
        intercepts[k_i + 1] = float(y_fit[tp]) - cum_slope * float(t_num[tp])

    return slopes, intercepts, np.asarray(y_fit, dtype=np.float64), cond_num


@dataclass(frozen=True)
class RobustSegmentFit:
    """Robust non-parametric linear segment summary (Theil–Sen or Akritas–Theil–Sen)."""

    slope: float
    intercept: float
    ci_low: float | None
    ci_high: float | None
    pvalue: float | None
    estimator: Literal["sen", "ats"]


def _kendall_var_s_ties(y: FloatArray, t: FloatArray) -> float:
    """Compute tie-adjusted Kendall S variance (Kendall 1975; Gilbert 1987 Eq. 16.8)."""
    n = int(y.size)
    if n < 2:
        return 0.0
    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    _, y_counts = np.unique(y, return_counts=True)
    y_ties = y_counts[y_counts > 1]
    if y_ties.size > 0:
        var_s -= float(np.sum(y_ties * (y_ties - 1) * (2 * y_ties + 5))) / 18.0
    _, t_counts = np.unique(t, return_counts=True)
    t_ties = t_counts[t_counts > 1]
    if t_ties.size > 0:
        var_s -= float(np.sum(t_ties * (t_ties - 1) * (2 * t_ties + 5))) / 18.0
    return max(0.0, float(var_s))


def hamed_rao_factor(t: FloatArray, y: FloatArray, slope: float, alpha_acf: float = 0.05) -> float:
    """Hamed & Rao (1998) variance inflation factor ``n / n*`` for the Mann-Kendall S.

    Ranks of the Sen-detrended series are autocorrelated at lags ``1..n-1`` and only
    lags significant at ``alpha_acf`` contribute (as in ``pymannkendall``'s
    ``hamed_rao_modification_test``). Lags are in observation order.
    """
    from scipy.stats import rankdata

    n = int(y.size)
    if n < 4:
        return 1.0
    ranks = rankdata(y - slope * t)
    r = ranks - ranks.mean()
    den = float(np.dot(r, r))
    if den <= 0.0:
        return 1.0
    acf = np.correlate(r, r, mode="full")[n - 1 :] / den
    bound = float(norm.ppf(1.0 - alpha_acf / 2.0) / np.sqrt(n))
    lags = np.arange(1, n)
    a = acf[1:]
    sig = np.abs(a) > bound
    sni = float(np.sum(((n - lags) * (n - lags - 1) * (n - lags - 2) * a)[sig]))
    factor = 1.0 + 2.0 * sni / (n * (n - 1) * (n - 2))
    return factor if factor > 0.0 else 1.0


def _rank_lag1_from_residuals(r_cols: FloatArray, *, exact_ties: bool = False) -> FloatArray:
    """Spearman lag-1 autocorrelation mapped to Gaussian AR(1) correlation ``2 sin(pi r_S / 6)``.

    Rank-based lag-1 autocorrelation is immune to isolated outliers and heavy-tailed
    innovations (Kendall, 1975) while remaining consistent for Gaussian AR(1) series.
    """
    if exact_ties:
        from scipy.stats import rankdata

        rk = np.asarray(rankdata(r_cols, axis=0), dtype=np.float64)
    else:
        rk = np.argsort(np.argsort(r_cols, axis=0), axis=0).astype(np.float64)
    r1 = rk[:-1] - np.mean(rk[:-1], axis=0, keepdims=True)
    r2 = rk[1:] - np.mean(rk[1:], axis=0, keepdims=True)
    den = np.sqrt(np.sum(r1 * r1, axis=0) * np.sum(r2 * r2, axis=0))
    rs = np.where(den > 0.0, np.sum(r1 * r2, axis=0) / np.where(den > 0.0, den, 1.0), 0.0)
    return np.asarray(2.0 * np.sin((np.pi / 6.0) * np.clip(rs, -1.0, 1.0)), dtype=np.float64)


def _robust_rank_lag1(
    y_cols: FloatArray, u: FloatArray, *, exact_ties: bool = False
) -> FloatArray:
    """Outlier-resistant lag-1 autocorrelation after robust linear detrending.

    Detrends each column with a shift/scale-invariant three-group median slope
    refined by one Huber WLS step, then computes the Gaussian-mapped Spearman
    lag-1 autocorrelation.
    """
    n, b = y_cols.shape
    m = max(1, n // 3)
    du = float(np.median(u[-m:]) - np.median(u[:m]))
    if du > 0.0:
        slope = (np.median(y_cols[-m:], axis=0) - np.median(y_cols[:m], axis=0)) / du
    else:
        slope = np.zeros(b, dtype=np.float64)
    r = y_cols - u[:, None] * slope[None, :]
    r = r - np.median(r, axis=0, keepdims=True)
    mad = 1.4826 * np.median(np.abs(r), axis=0)
    scale = np.where(mad > 1e-12, mad, 1.0)
    a = np.abs(r) / scale[None, :]
    w = np.where(a <= 1.345, 1.0, 1.345 / np.maximum(a, 1.345))
    sw = np.sum(w, axis=0)
    uw = np.sum(w * u[:, None], axis=0) / np.where(sw > 0.0, sw, 1.0)
    yw = np.sum(w * y_cols, axis=0) / np.where(sw > 0.0, sw, 1.0)
    du_w = u[:, None] - uw[None, :]
    den_s = np.sum(w * du_w * du_w, axis=0)
    slope2 = np.where(
        den_s > 0.0,
        np.sum(w * du_w * (y_cols - yw[None, :]), axis=0) / np.where(den_s > 0.0, den_s, 1.0),
        slope,
    )
    r_det = y_cols - u[:, None] * slope2[None, :]
    return _rank_lag1_from_residuals(r_det, exact_ties=exact_ties)


def ar1_rank_factor(t: FloatArray, y: FloatArray, slope: float) -> float:
    """AR(1) effective-sample-size inflation ``n / n*`` for the Mann-Kendall S.

    Estimates the lag-1 autocorrelation of the Sen-detrended series via the
    outlier-resistant Gaussian-mapped Spearman rank autocorrelation
    ``2 sin(pi r_S / 6)`` (Kendall, 1975), applies the Marriott & Pope (1954)
    small-sample bias correction ``(n r + 1)/(n - 4)`` truncated to ``[0, 0.95]``
    (the variance is never deflated), maps the implied Gaussian AR(1) correlations
    to rank correlations ``(6/pi) arcsin(rho^k / 2)``, and applies the Hamed & Rao
    (1998) summation.
    """
    n = int(y.size)
    if n < 6:
        return 1.0
    r = y - slope * t
    rho = float(_rank_lag1_from_residuals(r[:, None], exact_ties=True)[0])
    rho = float(np.clip((n * rho + 1.0) / (n - 4.0), 0.0, 0.95))
    if rho <= 0.0:
        return 1.0
    k = np.arange(1, n, dtype=np.float64)
    rs = (6.0 / np.pi) * np.arcsin(rho**k / 2.0)
    return float(1.0 + 2.0 * np.sum((n - k) * (n - k - 1) * (n - k - 2) * rs) / (n * (n - 1) * (n - 2)))


_AR1_BOOT_MAX_N = 300
_AR1_BOOT_HIGH_RHO_MAX_N = 1000
_AR1_BOOT_MIN_NEFF = 150.0
_AR1_BOOT_B = 999
# Spread of the autocorrelation draws in the bootstrap reference, in units of the
# asymptotic standard error sqrt((1 - rho^2)/n). Widening it beyond 1 compensates the
# plug-in error of the reference (tail probabilities are convex in rho).
_AR1_BOOT_JITTER = 1.0


def _should_use_ar1_bootstrap(n: int, rho_obs: float) -> bool:
    """Return True when finite-sample AR(1) bootstrap reference should be used.

    Always enabled for ``6 <= n <= 300``, and smoothly extended up to ``n <= 1000``
    whenever the implied effective sample size ``n_eff = n (1 - rho) / (1 + rho)``
    falls below 150 (preventing a switchover discontinuity at ``n = 300 -> 301``
    under strong autocorrelation).
    """
    if n < 6:
        return False
    if n <= _AR1_BOOT_MAX_N:
        return True
    if n > _AR1_BOOT_HIGH_RHO_MAX_N:
        return False
    rho_mp = float(np.clip((n * rho_obs + 1.0) / (n - 4.0), 0.0, 0.95))
    n_eff = float(n) * (1.0 - rho_mp) / (1.0 + rho_mp)
    return n_eff < _AR1_BOOT_MIN_NEFF


def _ar1_factor_from_rho(rho_hat: FloatArray, n: int) -> FloatArray:
    """Variance inflation of S for lag-1 estimates ``rho_hat`` (same mapping as
    :func:`ar1_rank_factor`: Marriott-Pope correction, truncation, Spearman map)."""
    rho = np.clip((n * np.asarray(rho_hat, dtype=np.float64) + 1.0) / (n - 4.0), 0.0, 0.95)
    k = np.arange(1, n, dtype=np.float64)
    w = (n - k) * (n - k - 1) * (n - k - 2) / (n * (n - 1) * (n - 2))
    rs = (6.0 / np.pi) * np.arcsin(np.power.outer(rho, k) / 2.0)
    return np.asarray(1.0 + 2.0 * (rs @ w), dtype=np.float64)


def _ar1_series_from_innov(e: FloatArray, rhos: FloatArray) -> FloatArray:
    """Build stationary Gaussian AR(1) columns from pre-drawn innovations ``e``."""
    n = e.shape[0]
    x = np.empty_like(e)
    x[0] = e[0] / np.sqrt(1.0 - rhos**2)
    for i in range(1, n):
        x[i] = rhos * x[i - 1] + e[i]
    return x


def _ar1_series(rng: np.random.Generator, n: int, rhos: FloatArray) -> FloatArray:
    """Stationary Gaussian AR(1) columns with per-column coefficients ``rhos``."""
    e = rng.standard_normal((n, rhos.size))
    return _ar1_series_from_innov(e, rhos)


def _mk_s_columns(y_cols: FloatArray) -> FloatArray:
    """Mann-Kendall S for each column (observations already in time order)."""
    s = np.zeros(y_cols.shape[1], dtype=np.float64)
    for i in range(y_cols.shape[0] - 1):
        s += np.sum(np.sign(y_cols[i + 1 :] - y_cols[i]), axis=0)
    return s


def ar1_bootstrap_reference(
    t: FloatArray, y: FloatArray, *, n_boot: int = _AR1_BOOT_B, seed: int = 20261126
) -> tuple[float, FloatArray]:
    """Parametric AR(1) bootstrap reference for the autocorrelation-corrected MK z.

    The lag-1 autocorrelation is estimated via the outlier-resistant rank lag-1
    estimator on robustly detrended residuals and debiased by simulation using
    common-random-number bisection (finding the AR(1) coefficient whose simulated
    estimator mean matches the observed estimate). Bootstrap series are simulated
    with coefficients drawn around that value (normal, standard error
    ``sqrt((1 - rho^2)/n)``), so uncertainty in the estimated autocorrelation is
    propagated; each is studentized with its own re-estimated variance factor
    exactly as the observed statistic is. The RNG seed depends only on the ranks
    of ``y`` and ``rho_obs``, guaranteeing exact invariance to shifts and positive
    rescalings. Returns the observed lag-1 estimate and the sorted bootstrap ``|z|``
    values.
    """
    from scipy.stats import rankdata

    n = int(y.size)
    span = float(t[-1] - t[0]) if t[-1] > t[0] else 1.0
    u = (t - t[0]) / span
    rho_obs = float(_robust_rank_lag1(y[:, None], u, exact_ties=True)[0])
    # Rank-based deterministic key: unique per ordering (avoids shared MC error across
    # independent series) while strictly invariant to shifts y + c and positive scales a * y.
    rk_int = np.rint(2.0 * np.asarray(rankdata(y), dtype=np.float64)).astype(np.uint64)
    data_key = int(
        np.sum(rk_int * np.arange(1, n + 1, dtype=np.uint64)) % 2**31
    )
    rho_key = int(round(rho_obs * 1e6)) % 2**31
    rng = np.random.default_rng((int(seed), n, data_key, rho_key))

    # Debias rho_obs via common-random-number bisection on [0.0, 0.98]
    e_deb = rng.standard_normal((n, 250))
    m_lo = float(np.mean(_robust_rank_lag1(_ar1_series_from_innov(e_deb, np.zeros(250)), u)))
    if rho_obs <= m_lo:
        r_t = 0.0
    else:
        lo_r, hi_r = 0.0, 0.98
        for _ in range(7):
            mid_r = 0.5 * (lo_r + hi_r)
            m_mid = float(
                np.mean(_robust_rank_lag1(_ar1_series_from_innov(e_deb, np.full(250, mid_r)), u))
            )
            if m_mid < rho_obs:
                lo_r = mid_r
            else:
                hi_r = mid_r
        r_t = 0.5 * (lo_r + hi_r)

    se0 = float(np.sqrt(max(1.0 - r_t * r_t, 1e-3) / n))
    se = _AR1_BOOT_JITTER * se0
    # Smooth small-sample Berger-Boos guard (0.40 SE at n=20, tapering to 0 at n>=50)
    small_n_shift = 0.40 * max(0.0, (50.0 - float(n)) / 30.0)
    rhos = np.clip(r_t + small_n_shift * se0 + se * rng.standard_normal(int(n_boot)), 0.0, 0.97)
    yb = _ar1_series(rng, n, rhos)
    sb = _mk_s_columns(yb)
    vb = n * (n - 1) * (2 * n + 5) / 18.0 * _ar1_factor_from_rho(_robust_rank_lag1(yb, u), n)
    zb = np.abs(sb - np.sign(sb)) / np.sqrt(vb)
    return rho_obs, np.sort(zb)


def _censored_kendall_score(
    t: FloatArray,
    vals: FloatArray,
    codes: IntArray,
    slope: float,
) -> float:
    """Compute censored Kendall S(slope) on residuals ``r_i = vals_i - slope * t_i``.

    Follows Akritas, Murphy & LaValley (1995) and Helsel (2012) concordance rules
    for left-censored (-1), observed (0), and right-censored (+1) observations.
    """
    r = vals - slope * t
    dt = t[None, :] - t[:, None]
    dr = r[None, :] - r[:, None]
    ci = codes[:, None]
    cj = codes[None, :]

    valid_t = np.triu(dt > 0.0, k=1)
    # Pair (i, j) with t_j > t_i is definitively concordant (+1) iff r_j > r_i
    # and i is not right-censored (<= r_i) and j is not left-censored (>= r_j).
    conc = valid_t & (dr > 0.0) & (ci <= 0) & (cj >= 0)
    # Definitively discordant (-1) iff r_j < r_i and i is not left-censored (>= r_i)
    # and j is not right-censored (<= r_j).
    disc = valid_t & (dr < 0.0) & (ci >= 0) & (cj <= 0)
    return float(np.sum(conc) - np.sum(disc))


def _censored_kendall_var(
    t: FloatArray,
    vals: FloatArray,
    codes: IntArray,
    slope: float,
) -> float:
    """Permutation (exchangeability) variance of the censored Kendall S at ``slope``.

    Uses Daniels' (1944) generalized-correlation variance for antisymmetric scores
    ``a_ij = sign(t_j - t_i)`` and ``b_ij`` = definite concordance of the residual
    responses ``r = vals - slope * t`` (``+1``/``-1`` when the censoring status allows a
    definite ordering, ``0`` otherwise):
    ``Var(Gamma) = 2 A2 B2 / (n(n-1)) + 4 (A3 - A2)(B3 - B2) / (n(n-1)(n-2))`` with
    ``Gamma = 2 S``, ``A2 = sum a_ij^2`` and ``A3 = sum_i (sum_j a_ij)^2`` (likewise B).
    Without censoring or ties this equals ``n(n-1)(2n+5)/18``; with censoring it is
    smaller, so CIs and p-values are no longer conservative.
    """
    n = int(t.size)
    if n < 3:
        return 0.0
    r = vals - slope * t
    a = np.sign(t[None, :] - t[:, None]).astype(np.int8)
    dr = r[None, :] - r[:, None]
    ci = codes[:, None]
    cj = codes[None, :]
    b = ((dr > 0.0) & (ci <= 0) & (cj >= 0)).astype(np.int8) - (
        (dr < 0.0) & (ci >= 0) & (cj <= 0)
    ).astype(np.int8)
    a2 = float(np.sum(a.astype(np.int64) ** 2))
    b2 = float(np.sum(b.astype(np.int64) ** 2))
    a3 = float(np.sum(np.sum(a, axis=1, dtype=np.int64).astype(np.float64) ** 2))
    b3 = float(np.sum(np.sum(b, axis=1, dtype=np.int64).astype(np.float64) ** 2))
    var_g = 2.0 * a2 * b2 / (n * (n - 1)) + 4.0 * (a3 - a2) * (b3 - b2) / (n * (n - 1) * (n - 2))
    return max(0.0, var_g / 4.0)


def _invert_censored_score(
    t: FloatArray,
    vals: FloatArray,
    codes: IntArray,
    target_s: float,
    lo_init: float,
    hi_init: float,
    n_bisect: int = 50,
) -> float:
    """Find midpoint where monotone non-increasing ``S(b)`` crosses ``target_s``."""
    lo, hi = lo_init, hi_init
    # Expand bracket if needed
    span = max(abs(hi - lo), 1.0)
    for _ in range(25):
        if _censored_kendall_score(t, vals, codes, lo) >= target_s:
            break
        lo -= span
        span *= 2.0
    span = max(abs(hi - lo), 1.0)
    for _ in range(25):
        if _censored_kendall_score(t, vals, codes, hi) <= target_s:
            break
        hi += span
        span *= 2.0

    # Upper crossing: inf { b : S(b) < target_s }
    a_u, b_u = lo, hi
    for _ in range(n_bisect):
        mid = 0.5 * (a_u + b_u)
        if _censored_kendall_score(t, vals, codes, mid) >= target_s:
            a_u = mid
        else:
            b_u = mid
    # Lower crossing: sup { b : S(b) > target_s }
    a_l, b_l = lo, hi
    for _ in range(n_bisect):
        mid = 0.5 * (a_l + b_l)
        if _censored_kendall_score(t, vals, codes, mid) > target_s:
            a_l = mid
        else:
            b_l = mid
    return 0.5 * (0.5 * (a_u + b_u) + 0.5 * (a_l + b_l))


def fit_theil_sen_segment(
    t: FloatArray,
    y: FloatArray,
    *,
    alpha: float = 0.05,
    status_codes: IntArray | None = None,
    original_values: FloatArray | None = None,
    max_exact_n: int = 2000,
    autocorr_correction: Literal["none", "hamed_rao", "ar1"] = "none",
) -> RobustSegmentFit:
    """Estimate robust segment slope, intercept, CI, and Mann–Kendall p-value.

    Uses exact **Theil–Sen** for uncensored segments and **Akritas–Theil–Sen**
    (Akritas et al., 1995) automatically when left/right censoring indicators
    are supplied.
    """
    n = int(t.size)
    if n < 2 or float(t[-1] - t[0]) <= 0.0:
        med_y = float(np.median(y)) if n > 0 else 0.0
        return RobustSegmentFit(
            slope=0.0,
            intercept=med_y,
            ci_low=None,
            ci_high=None,
            pvalue=None,
            estimator="sen",
        )

    has_cen = (
        status_codes is not None
        and original_values is not None
        and bool(np.any((status_codes == -1) | (status_codes == 1)))
        and not bool(np.any(status_codes == 2))
        and n <= max_exact_n
    )

    if has_cen:
        assert status_codes is not None and original_values is not None
        vals = original_values.astype(np.float64, copy=False)
        codes = status_codes.astype(np.intp, copy=False)

        # Bracket slopes using pairwise finite differences of observed/limit values
        i_idx, j_idx = np.triu_indices(n, k=1)
        dt = t[j_idx] - t[i_idx]
        valid = dt > 0.0
        if not np.any(valid):
            return RobustSegmentFit(0.0, float(np.median(y)), None, None, None, "ats")
        raw_slopes = (vals[j_idx[valid]] - vals[i_idx[valid]]) / dt[valid]
        s_lo = float(np.min(raw_slopes))
        s_hi = float(np.max(raw_slopes))
        if s_lo == s_hi:
            s_lo -= 1.0
            s_hi += 1.0

        slope = _invert_censored_score(t, vals, codes, 0.0, s_lo, s_hi)
        # Residual median intercept (uses imputed y when censored fraction >= 50%)
        intercept = float(np.median(y - slope * t))

        s0 = _censored_kendall_score(t, vals, codes, 0.0)
        # Censoring-aware permutation variances: at slope 0 for the trend test, and at
        # the ATS estimate for the confidence interval inversion.
        var_s0 = _censored_kendall_var(t, vals, codes, 0.0)
        var_s = _censored_kendall_var(t, vals, codes, slope)
        boot_z_ats: FloatArray | None = None
        if var_s0 > 0.0 and var_s > 0.0:
            if autocorr_correction == "hamed_rao":
                fac = hamed_rao_factor(t, y, slope)
                var_s0 *= fac
                var_s *= fac
            elif autocorr_correction == "ar1":
                span_c = float(t[-1] - t[0]) if t[-1] > t[0] else 1.0
                u_c = (t - t[0]) / span_c
                rho_c = float(_robust_rank_lag1(y[:, None], u_c, exact_ties=True)[0])
                if _should_use_ar1_bootstrap(n, rho_c):
                    rho_c, boot_z_ats = ar1_bootstrap_reference(t, y)
                    fac = float(_ar1_factor_from_rho(np.array([rho_c]), n)[0])
                else:
                    fac = ar1_rank_factor(t, y, slope)
                var_s0 *= fac
                var_s *= fac

            z_mk = (s0 - np.sign(s0)) / np.sqrt(var_s0) if abs(s0) > 0.0 else 0.0
            pval_ats = float(2.0 * norm.sf(abs(z_mk)))
            z_crit = float(norm.isf(0.5 * alpha))
            if boot_z_ats is not None:
                exceed = int(np.sum(boot_z_ats >= abs(z_mk)))
                p_mc = (1.0 + exceed) / (boot_z_ats.size + 1.0)
                pval_ats = float(min(p_mc, pval_ats)) if exceed == 0 else float(p_mc)
                z_crit = float(np.quantile(boot_z_ats, 1.0 - alpha))
            pval: float | None = pval_ats
            c_alpha = float(z_crit * np.sqrt(var_s))
            ci_lo_v = _invert_censored_score(t, vals, codes, c_alpha, s_lo, s_hi)
            ci_hi_v = _invert_censored_score(t, vals, codes, -c_alpha, s_lo, s_hi)
            if ci_lo_v > ci_hi_v:
                ci_lo_v, ci_hi_v = ci_hi_v, ci_lo_v
            ci_low: float | None = ci_lo_v
            ci_high: float | None = ci_hi_v
        else:
            pval = None
            ci_low = None
            ci_high = None

        return RobustSegmentFit(
            slope=slope,
            intercept=intercept,
            ci_low=ci_low,
            ci_high=ci_high,
            pvalue=pval,
            estimator="ats",
        )

    # Uncensored Theil–Sen estimator
    if n <= max_exact_n:
        i_idx, j_idx = np.triu_indices(n, k=1)
        dt = t[j_idx] - t[i_idx]
        valid = dt > 0.0
        if not np.any(valid):
            return RobustSegmentFit(0.0, float(np.median(y)), None, None, None, "sen")
        dy = y[j_idx[valid]] - y[i_idx[valid]]
        slopes_ij = dy / dt[valid]
        s_stat = float(np.sum(np.sign(dy)))
        var_s = _kendall_var_s_ties(y, t)
    else:
        # Deterministic large-n pair sampling (keeps memory O(1))
        rng = np.random.default_rng(0)
        n_pairs = 500_000
        a_i = rng.integers(0, n, size=n_pairs)
        b_i = rng.integers(0, n, size=n_pairs)
        lo_i = np.minimum(a_i, b_i)
        hi_i = np.maximum(a_i, b_i)
        dt = t[hi_i] - t[lo_i]
        valid = dt > 0.0
        dy = y[hi_i[valid]] - y[lo_i[valid]]
        slopes_ij = dy / dt[valid]
        s_stat = float("nan")
        var_s = 0.0

    slopes_sorted = np.sort(slopes_ij)
    n_p = int(slopes_sorted.size)
    slope = float(np.median(slopes_sorted))
    intercept = float(np.median(y - slope * t))
    boot_z: FloatArray | None = None
    if autocorr_correction == "hamed_rao" and var_s > 0.0:
        var_s *= hamed_rao_factor(t, y, slope)
    elif autocorr_correction == "ar1" and var_s > 0.0:
        span_u = float(t[-1] - t[0]) if t[-1] > t[0] else 1.0
        u_u = (t - t[0]) / span_u
        rho_pre = float(_robust_rank_lag1(y[:, None], u_u, exact_ties=True)[0])
        if _should_use_ar1_bootstrap(n, rho_pre):
            rho_obs, boot_z = ar1_bootstrap_reference(t, y)
            var_s *= float(_ar1_factor_from_rho(np.array([rho_obs]), n)[0])
        else:
            var_s *= ar1_rank_factor(t, y, slope)

    if n <= max_exact_n and var_s > 0.0:
        z_mk = (s_stat - np.sign(s_stat)) / np.sqrt(var_s) if abs(s_stat) > 0.0 else 0.0
        pval = float(2.0 * norm.sf(abs(z_mk)))
        z_crit = float(norm.isf(0.5 * alpha))
        if boot_z is not None:
            # Short / strongly autocorrelated series: refer the studentized statistic to its
            # AR(1) bootstrap distribution (MC p-value; below resolution keep the normal value).
            exceed = int(np.sum(boot_z >= abs(z_mk)))
            p_mc = (1.0 + exceed) / (boot_z.size + 1.0)
            pval = float(min(p_mc, pval)) if exceed == 0 else float(p_mc)
            z_crit = float(np.quantile(boot_z, 1.0 - alpha))
        c_alpha = float(z_crit * np.sqrt(var_s))
        # Gilbert (1987, Sec. 16.4): lower = M1-th, upper = (M2 + 1)-th ordered slope
        # with M1 = (N' - C)/2 and M2 = (N' + C)/2 (rounded, as in R trend::sens.slope).
        idx_lo = max(0, min(n_p - 1, int(np.round((n_p - c_alpha) / 2.0)) - 1))
        idx_hi = max(0, min(n_p - 1, int(np.round((n_p + c_alpha) / 2.0))))
        ci_low = float(slopes_sorted[idx_lo])
        ci_high = float(slopes_sorted[idx_hi])
    else:
        pval = None
        ci_low = None
        ci_high = None

    return RobustSegmentFit(
        slope=slope,
        intercept=intercept,
        ci_low=ci_low,
        ci_high=ci_high,
        pvalue=pval,
        estimator="sen",
    )
