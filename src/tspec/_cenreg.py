"""Censored (Tobit-type) linear regression MLE and conditional imputation.

Provides a time-aware imputation model for left-, right-, and interval-censored
observations: ``z_i = x_i^T beta + sigma * eps_i / sqrt(w_i)`` with Gaussian ``eps``,
where ``z = log(y)`` for the lognormal family and ``z = y`` for the normal family.
Censored values are imputed from the fitted *conditional* truncated distribution
given ``x_i`` (e.g., time trend and seasonality), so imputations are congenial with
trend analyses (Meng, 1994; Helsel, 2012, ch. 12). Optional parameter draws from
the asymptotic normal posterior give proper multiple imputations (Rubin, 1987).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize
from scipy.special import log_ndtr, ndtr, ndtri
from scipy.stats import truncnorm

from ._types import FloatArray, IntArray


@dataclass(frozen=True)
class CenRegFit:
    """Fitted censored regression (parameters in transformed ``z`` space)."""

    beta: FloatArray
    log_sigma: float
    cov: FloatArray  # covariance of (beta, log_sigma)
    lognormal: bool
    converged: bool


def _bounds_z(
    vals: FloatArray,
    codes: IntArray,
    left: FloatArray | None,
    right: FloatArray | None,
    lognormal: bool,
) -> tuple[FloatArray, FloatArray]:
    """Lower/upper bounds of each observation in z space (equal for observed values)."""
    lo = np.where(codes == -1, -np.inf, vals).astype(np.float64)
    hi = np.where(codes == 1, np.inf, vals).astype(np.float64)
    if left is not None and right is not None:
        iv = codes == 2
        lo = np.where(iv, left, lo)
        hi = np.where(iv, right, hi)
    if lognormal:
        with np.errstate(divide="ignore", invalid="ignore"):
            lo = np.where(np.isfinite(lo) & (lo > 0), np.log(np.where(lo > 0, lo, 1.0)), -np.inf)
            hi = np.where(np.isposinf(hi), np.inf, np.log(np.where(hi > 0, hi, np.nan)))
    return lo, hi


_HALF_LOG_2PI = 0.5 * float(np.log(2.0 * np.pi))


def _log_interval_prob(a: FloatArray, b: FloatArray) -> FloatArray:
    """Stable log(Phi(b) - Phi(a)) for a < b (possibly infinite)."""
    # Use upper tails when both bounds are positive for accuracy.
    flip = a > 0
    a2 = np.where(flip, -b, a)
    b2 = np.where(flip, -a, b)
    lb = log_ndtr(b2)
    la = log_ndtr(a2)
    with np.errstate(divide="ignore", invalid="ignore"):
        diff = lb + np.log1p(-np.exp(np.minimum(la - lb, 0.0)))
    # Narrow finite interval fallback (avoids cancellation when la == lb in float64)
    narrow = np.isfinite(a2) & np.isfinite(b2) & ((b2 - a2) < 1e-6)
    if np.any(narrow):
        mid = 0.5 * (a2 + b2)
        width = np.maximum(b2 - a2, 1e-300)
        approx = np.log(width) - 0.5 * mid * mid - _HALF_LOG_2PI
        diff = np.where(narrow, approx, diff)
    return np.where(np.isfinite(diff), diff, -745.0)


def _interval_mills_terms(a: FloatArray, b: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Stable ``(r1, r2)`` for truncated standard normal on ``(a, b)``.

    Returns:
      - ``r1 = (phi(b) - phi(a)) / (Phi(b) - Phi(a))``
      - ``r2 = (b * phi(b) - a * phi(a)) / (Phi(b) - Phi(a))``

    Uses sign-flip symmetry into ``a' <= 0``, log-space subtraction against
    ``_log_interval_prob``, exact narrow-interval Taylor limits, and Gordon's (1941)
    asymptotic inverse Mills ratio ``-x - 1/x + 2/x^3 - 10/x^5`` in the deep tail
    ``b' < -30`` so extreme tails never underflow to 0/0.
    """
    flip = a > 0
    a2 = np.where(flip, -b, a)
    b2 = np.where(flip, -a, b)
    lp = _log_interval_prob(a2, b2)

    fin_a = np.isfinite(a2)
    fin_b = np.isfinite(b2)
    a_safe = np.where(fin_a, a2, 0.0)
    b_safe = np.where(fin_b, b2, 0.0)
    lpa = np.where(fin_a, -0.5 * a_safe * a_safe - _HALF_LOG_2PI, -np.inf)
    lpb = np.where(fin_b, -0.5 * b_safe * b_safe - _HALF_LOG_2PI, -np.inf)

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        rb = np.where(fin_b, np.exp(np.clip(lpb - lp, -700.0, 700.0)), 0.0)
        ra = np.where(fin_a, np.exp(np.clip(lpa - lp, -700.0, 700.0)), 0.0)
        r1 = rb - ra
        r2 = b_safe * rb - a_safe * ra

    # Deep left tail (b2 < -30 and lower bound negligible): asymptotic Mills ratio
    deep_tail = fin_b & (b2 < -30.0) & ((~fin_a) | ((lpa - lpb) < -40.0))
    if np.any(deep_tail):
        x = np.where(deep_tail, b2, -30.0)
        inv_x = 1.0 / x
        inv_x2 = inv_x * inv_x
        mills = -x - inv_x * (1.0 - 2.0 * inv_x2 + 10.0 * inv_x2 * inv_x2)
        r1 = np.where(deep_tail, mills, r1)
        r2 = np.where(deep_tail, x * mills, r2)

    # Narrow finite intervals (b2 - a2 < 1e-6): exact midpoint Taylor limits
    narrow = fin_a & fin_b & ((b2 - a2) < 1e-6)
    if np.any(narrow):
        mid = 0.5 * (a2 + b2)
        r1 = np.where(narrow, -mid, r1)
        r2 = np.where(narrow, 1.0 - mid * mid, r2)

    r1_out = np.where(flip, -r1, r1)
    return np.asarray(r1_out, dtype=np.float64), np.asarray(r2, dtype=np.float64)


def _nll_grad(
    params: FloatArray,
    X: FloatArray,
    lo: FloatArray,
    hi: FloatArray,
    sw: FloatArray,
    exact: np.ndarray,
) -> tuple[float, FloatArray]:
    p = X.shape[1]
    beta = params[:p]
    th = float(np.clip(params[p], -35.0, 35.0))
    sig = float(np.exp(th))
    mu = X @ beta
    s_i = sig / sw  # per-observation sd
    nll = 0.0
    g_mu = np.zeros_like(mu)
    g_th = 0.0
    # exact observations
    if np.any(exact):
        r = (lo[exact] - mu[exact]) / s_i[exact]
        nll += float(np.sum(0.5 * r * r + np.log(s_i[exact])))
        g_mu[exact] = -r / s_i[exact]
        g_th += float(np.sum(1.0 - r * r))
    cen = ~exact
    if np.any(cen):
        a = (lo[cen] - mu[cen]) / s_i[cen]
        b = (hi[cen] - mu[cen]) / s_i[cen]
        lp = _log_interval_prob(a, b)
        nll -= float(np.sum(lp))
        r1, r2 = _interval_mills_terms(a, b)
        g_mu[cen] = r1 / s_i[cen]
        g_th += float(np.sum(r2))
    grad = np.concatenate([X.T @ g_mu, [g_th]])
    return nll, grad


def fit_censored_regression(
    vals: FloatArray,
    codes: IntArray,
    X: FloatArray,
    *,
    left: FloatArray | None = None,
    right: FloatArray | None = None,
    weights: FloatArray | None = None,
    dist: str = "normal",
) -> CenRegFit:
    """Maximum-likelihood censored linear regression (Tobit for left/right/interval).

    Standardizes ``z`` internally before BFGS optimization and numerical Hessian
    differentiation so MLE estimates, gradients, and covariances are strictly
    scale-invariant across arbitrary response magnitudes, and projects the inverse
    Hessian onto the strictly positive-definite cone.
    """
    from ._linalg import _detect_intercept_partition_cols

    lognormal = dist == "lognormal"
    lo, hi = _bounds_z(vals, codes, left, right, lognormal)
    if np.any(np.isnan(lo)) or np.any(np.isnan(hi)) or np.any(lo > hi):
        raise ValueError("Invalid censoring bounds for censored regression.")
    exact = (codes == 0) | (lo == hi)
    if int(np.sum(exact)) < X.shape[1] + 1:
        raise ValueError("Too few exact observations for censored regression.")
    sw = np.sqrt(weights) if weights is not None else np.ones(vals.size)

    # Internal affine standardization in z-space for exact scale/shift invariance
    k_int = _detect_intercept_partition_cols(X)
    z0_raw = np.where(
        exact,
        lo,
        np.where(
            np.isfinite(lo) & np.isfinite(hi),
            0.5 * (lo + hi),
            np.where(np.isfinite(hi), hi, lo),
        ),
    )
    z_center = float(np.median(lo[exact])) if k_int > 0 else 0.0
    z_scale = float(np.std(lo[exact])) if int(np.sum(exact)) > 1 else 0.0
    if not np.isfinite(z_scale) or z_scale <= 0.0:
        z_scale = float(np.std(z0_raw))
    if not np.isfinite(z_scale) or z_scale <= 0.0:
        z_scale = max(float(np.max(np.abs(z0_raw))), 1.0)

    lo_s = (lo - z_center) / z_scale
    hi_s = (hi - z_center) / z_scale
    z0_s = (z0_raw - z_center) / z_scale

    # Start: weighted OLS on midpoints / limits in standardized coordinates
    b0, *_ = np.linalg.lstsq(X * sw[:, None], z0_s * sw, rcond=None)
    r0 = (z0_s - X @ b0) * sw
    s0 = float(np.std(r0[exact])) if np.sum(exact) > 1 else 1.0
    x0 = np.concatenate([b0, [np.log(max(s0, 1e-4))]])
    res = optimize.minimize(
        _nll_grad,
        x0,
        args=(X, lo_s, hi_s, sw, exact),
        jac=True,
        method="BFGS",
        options={"maxiter": 500, "gtol": 1e-7},
    )
    xp = res.x

    # Numerical Hessian in standardized coordinates from analytic gradient
    k = xp.size
    H = np.empty((k, k), dtype=np.float64)
    for i in range(k):
        h = 1e-5 * max(1.0, abs(float(xp[i])))
        e = np.zeros(k, dtype=np.float64)
        e[i] = h
        gp = _nll_grad(xp + e, X, lo_s, hi_s, sw, exact)[1]
        gm = _nll_grad(xp - e, X, lo_s, hi_s, sw, exact)[1]
        H[:, i] = (gp - gm) / (2.0 * h)
    H = 0.5 * (H + H.T)

    # Guarantee strictly SPD covariance via eigendecomposition eigenvalue floor
    if np.all(np.isfinite(H)):
        evals, evecs = np.linalg.eigh(H)
        max_ev = max(float(np.max(evals)), 1e-12)
        evals_pos = np.maximum(evals, 1e-10 * max_ev)
        cov_s = (evecs / evals_pos[None, :]) @ evecs.T
        cov_s = 0.5 * (cov_s + cov_s.T)
    else:
        cov_s = np.eye(k, dtype=np.float64)

    g_final = _nll_grad(xp, X, lo_s, hi_s, sw, exact)[1]
    converged = bool(res.success) or float(np.max(np.abs(g_final))) < 1e-4 * max(1.0, float(vals.size))

    # Unscale parameters & covariance back to original z coordinates
    beta_orig = xp[:-1] * z_scale
    if k_int > 0 and z_center != 0.0:
        beta_orig = beta_orig.copy()
        beta_orig[:k_int] += z_center
    log_sigma_orig = float(xp[-1]) + float(np.log(z_scale))
    jac_diag = np.full(k, z_scale, dtype=np.float64)
    jac_diag[-1] = 1.0
    cov_orig = (jac_diag[:, None] * cov_s) * jac_diag[None, :]
    cov_orig = np.asarray(0.5 * (cov_orig + cov_orig.T), dtype=np.float64)

    return CenRegFit(
        beta=np.asarray(beta_orig, dtype=np.float64),
        log_sigma=log_sigma_orig,
        cov=cov_orig,
        lognormal=lognormal,
        converged=converged,
    )


def _fit_nll_y(
    fit: CenRegFit,
    vals: FloatArray,
    codes: IntArray,
    X: FloatArray,
    left: FloatArray | None,
    right: FloatArray | None,
    weights: FloatArray | None,
) -> float:
    """Observed-scale negative log-likelihood (including log-Jacobian for lognormal)."""
    lo, hi = _bounds_z(vals, codes, left, right, fit.lognormal)
    exact = (codes == 0) | (lo == hi)
    sw = np.sqrt(weights) if weights is not None else np.ones(vals.size)
    params = np.concatenate([fit.beta, [fit.log_sigma]])
    nll_z, _ = _nll_grad(params, X, lo, hi, sw, exact)
    if fit.lognormal:
        # Jacobian of z = log(y) on exact observations: -log(1/y) = +log(y)
        nll_z += float(np.sum(np.log(vals[exact])))
    return float(nll_z)


def fit_censored_regression_auto(
    vals: FloatArray,
    codes: IntArray,
    X: FloatArray,
    *,
    left: FloatArray | None = None,
    right: FloatArray | None = None,
    weights: FloatArray | None = None,
    dist: str = "auto",
) -> tuple[CenRegFit, str]:
    """Fit Tobit MLE with optional Jacobian-adjusted NLL selection for ``dist='auto'``."""
    from .censored import _resolve_trend_dist

    if dist in ("normal", "lognormal"):
        return (
            fit_censored_regression(
                vals, codes, X, left=left, right=right, weights=weights, dist=dist
            ),
            dist,
        )
    cand_dist = _resolve_trend_dist(vals, codes, left, right, dist)
    if cand_dist != "lognormal":
        return (
            fit_censored_regression(
                vals, codes, X, left=left, right=right, weights=weights, dist="normal"
            ),
            "normal",
        )
    # Positive data with dist='auto': compare normal vs lognormal on exact observed-scale NLL
    fit_norm: CenRegFit | None = None
    fit_ln: CenRegFit | None = None
    try:
        fit_norm = fit_censored_regression(
            vals, codes, X, left=left, right=right, weights=weights, dist="normal"
        )
    except (ValueError, np.linalg.LinAlgError):
        fit_norm = None
    try:
        fit_ln = fit_censored_regression(
            vals, codes, X, left=left, right=right, weights=weights, dist="lognormal"
        )
    except (ValueError, np.linalg.LinAlgError):
        fit_ln = None

    if fit_ln is not None and fit_ln.converged and (fit_norm is None or not fit_norm.converged):
        return fit_ln, "lognormal"
    if fit_norm is not None and fit_norm.converged and (fit_ln is None or not fit_ln.converged):
        return fit_norm, "normal"
    if fit_norm is None and fit_ln is None:
        raise ValueError("Censored regression failed for both normal and lognormal families.")
    assert fit_norm is not None and fit_ln is not None
    nll_n = _fit_nll_y(fit_norm, vals, codes, X, left, right, weights)
    nll_l = _fit_nll_y(fit_ln, vals, codes, X, left, right, weights)
    if nll_l < nll_n:
        return fit_ln, "lognormal"
    return fit_norm, "normal"


def impute_censored_regression(
    fit: CenRegFit,
    vals: FloatArray,
    codes: IntArray,
    X: FloatArray,
    rng: np.random.Generator,
    *,
    left: FloatArray | None = None,
    right: FloatArray | None = None,
    weights: FloatArray | None = None,
    mode: str = "stochastic",
    draw_params: bool = False,
) -> FloatArray:
    """Impute censored entries from the conditional truncated distribution given ``X``.

    ``mode``: ``"stochastic"`` (random draw), ``"mean"`` (conditional mean), or
    ``"quantile"`` (conditional median). ``draw_params=True`` first draws
    ``(beta, log_sigma)`` from their asymptotic normal distribution (proper MI).
    """
    beta = fit.beta
    th = fit.log_sigma
    if draw_params:
        mean_th = np.concatenate([beta, [th]])
        d_sd = np.sqrt(np.maximum(np.diag(fit.cov), 0.0))
        if np.all(d_sd > 0.0):
            corr = fit.cov / np.outer(d_sd, d_sd)
            corr = 0.5 * (corr + corr.T)
            evals, evecs = np.linalg.eigh(corr)
            max_ev = max(float(np.max(evals)), 1e-12)
            evals_pos = np.maximum(evals, 1e-12 * max_ev)
            z_std = rng.standard_normal(mean_th.size)
            theta = mean_th + d_sd * (evecs @ (np.sqrt(evals_pos) * z_std))
        else:
            theta = mean_th
        beta, th = theta[:-1], float(theta[-1])
    lo, hi = _bounds_z(vals, codes, left, right, fit.lognormal)
    exact = (codes == 0) | (lo == hi)
    out = vals.astype(np.float64).copy()
    cen = ~exact
    if not np.any(cen):
        return out
    sw = np.sqrt(weights) if weights is not None else np.ones(vals.size)
    mu = (X @ beta)[cen]
    s = np.exp(th) / sw[cen]
    a = (lo[cen] - mu) / s
    b = (hi[cen] - mu) / s
    if mode == "stochastic":
        z = truncnorm.rvs(a, b, loc=mu, scale=s, random_state=rng)
        z = np.clip(z, lo[cen], hi[cen])
        y = np.exp(z) if fit.lognormal else z
    elif mode == "quantile":
        pm = 0.5 * (ndtr(a) + ndtr(b))
        z = np.clip(mu + s * ndtri(np.clip(pm, 1e-300, 1 - 1e-16)), lo[cen], hi[cen])
        y = np.exp(z) if fit.lognormal else z
    else:  # conditional mean
        if fit.lognormal:
            lp = _log_interval_prob(a, b)
            lp_s = _log_interval_prob(a - s, b - s)
            y = np.exp(mu + 0.5 * s * s + lp_s - lp)
        else:
            r1, _ = _interval_mills_terms(a, b)
            y = mu - s * r1
        lo_y = np.exp(lo[cen]) if fit.lognormal else lo[cen]
        hi_y = np.exp(hi[cen]) if fit.lognormal else hi[cen]
        y = np.clip(y, lo_y, hi_y)
    out[cen] = y
    return out


def trend_design(
    t_num: FloatArray,
    harmonics: tuple[float, ...] = (),
) -> FloatArray:
    """Imputation design ``[1, u, cos/sin(2 pi f t)...]`` with ``u`` = time scaled to [0, 1]."""
    span = float(t_num[-1] - t_num[0]) if t_num.size > 1 else 1.0
    span = span if span > 0 else 1.0
    cols = [np.ones(t_num.size), (t_num - t_num[0]) / span]
    for f in harmonics:
        cols.append(np.cos(2.0 * np.pi * f * t_num))
        cols.append(np.sin(2.0 * np.pi * f * t_num))
    return np.column_stack(cols)


def rubin_pool(estimates: FloatArray, variances: FloatArray, df_complete: float) -> tuple[float, float, float]:
    """Rubin's rules with Barnard & Rubin (1999) small-sample df. Returns (Q, T, df)."""
    m = int(estimates.size)
    q = float(np.mean(estimates))
    u = float(np.mean(variances))
    b = float(np.var(estimates, ddof=1)) if m > 1 else 0.0
    t_var = u + (1.0 + 1.0 / m) * b
    if b <= 0.0 or t_var <= 0.0:
        return q, t_var, float(df_complete)
    lam = (1.0 + 1.0 / m) * b / t_var
    df_old = (m - 1) / lam**2
    df_obs = (df_complete + 1.0) / (df_complete + 3.0) * df_complete * (1.0 - lam)
    df = 1.0 / (1.0 / df_old + 1.0 / max(df_obs, 1e-12))
    return q, t_var, float(df)


def pool_pvalues(pvalues: FloatArray) -> float:
    """Pool one-sided evidence p-values across multiple imputations (Licht–Rubin).

    Each p-value is transformed to ``z = Phi^{-1}(1 - p)`` (standard normal under the
    null), and Rubin's rules are applied with unit within-imputation variance:
    ``T = 1 + (1 + 1/M) B`` with ``B`` the between-imputation variance of ``z``, and the
    pooled statistic ``mean(z) / sqrt(T)`` is referred to a t distribution with Rubin's
    (1987) degrees of freedom ``(M - 1) (1 + 1/r)^2``, ``r = (1 + 1/M) B``
    (Licht, 2010; van Buuren, 2018, Sec. 5.3). Non-finite p-values are ignored, and
    p-values are clipped to ``[1e-300, 1 - 1e-12]`` so exact ``0`` or ``1`` values
    (e.g. from Monte Carlo p-values) do not produce ``+-inf`` z-scores.
    """
    from scipy.stats import t as t_dist

    p = np.asarray(pvalues, dtype=np.float64)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return float("nan")
    if p.size == 1:
        return float(p[0])
    z = -ndtri(np.clip(p, 1e-300, 1.0 - 1e-12))
    m = int(z.size)
    b = float(np.var(z, ddof=1))
    r = (1.0 + 1.0 / m) * b
    stat = float(np.mean(z)) / np.sqrt(1.0 + r)
    if r <= 1e-12:
        return float(ndtr(-stat))
    nu = (m - 1) * (1.0 + 1.0 / r) ** 2
    return float(t_dist.sf(stat, nu))


def window_congenial_imputations(
    vals: FloatArray,
    codes: IntArray,
    left: FloatArray | None,
    right: FloatArray | None,
    X: FloatArray,
    dist: str,
    n_imputations: int,
    seed: int | tuple[int, ...],
    weights: FloatArray | None = None,
) -> list[FloatArray] | None:
    """Proper multiple imputations of one window from a censored regression on ``X``.

    Detection uses the window's null model as ``X``: including a structure whose
    location was selected from the data (a candidate jump or knot) builds that
    selection into every imputation and inflates false detections. Returns ``None``
    when the window has no censored values or the model cannot be fitted.
    """
    if not np.any(codes != 0):
        return None
    try:
        fit = fit_censored_regression(vals, codes, X, left=left, right=right, weights=weights, dist=dist)
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not fit.converged or not np.all(np.isfinite(fit.cov)):
        return None
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(int(n_imputations)):
        out.append(
            impute_censored_regression(
                fit, vals, codes, X, rng, left=left, right=right, weights=weights,
                mode="stochastic", draw_params=True,
            )
        )
    return out
