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
    return np.where(np.isfinite(diff), diff, -745.0)


def _phi(x: FloatArray) -> FloatArray:
    out = np.zeros_like(x)
    fin = np.isfinite(x)
    out[fin] = np.exp(-0.5 * x[fin] ** 2) / np.sqrt(2.0 * np.pi)
    return out


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
    th = float(params[p])
    sig = np.exp(th)
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
        pa = _phi(a)
        pb = _phi(b)
        P = np.exp(lp)
        P = np.maximum(P, 1e-300)
        g_mu[cen] = (pb - pa) / (s_i[cen] * P)
        aa = np.where(np.isfinite(a), a, 0.0)
        bb = np.where(np.isfinite(b), b, 0.0)
        g_th += float(np.sum((pb * bb - pa * aa) / P))
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
    """Maximum-likelihood censored linear regression (Tobit for left/right/interval)."""
    lognormal = dist == "lognormal"
    lo, hi = _bounds_z(vals, codes, left, right, lognormal)
    if np.any(np.isnan(lo)) or np.any(np.isnan(hi)) or np.any(lo > hi):
        raise ValueError("Invalid censoring bounds for censored regression.")
    exact = (codes == 0) | (lo == hi)
    if int(np.sum(exact)) < X.shape[1] + 1:
        raise ValueError("Too few exact observations for censored regression.")
    sw = np.sqrt(weights) if weights is not None else np.ones(vals.size)
    # start: weighted OLS on midpoints / limits
    z0 = np.where(exact, lo, np.where(np.isfinite(lo) & np.isfinite(hi), 0.5 * (lo + hi),
                                      np.where(np.isfinite(hi), hi, lo)))
    b0, *_ = np.linalg.lstsq(X * sw[:, None], z0 * sw, rcond=None)
    r0 = (z0 - X @ b0) * sw
    s0 = float(np.std(r0[exact])) if np.sum(exact) > 1 else 1.0
    x0 = np.concatenate([b0, [np.log(max(s0, 1e-8))]])
    res = optimize.minimize(_nll_grad, x0, args=(X, lo, hi, sw, exact), jac=True, method="BFGS",
                            options={"maxiter": 500, "gtol": 1e-7})
    xp = res.x
    # numerical Hessian from analytic gradient
    k = xp.size
    H = np.empty((k, k))
    for i in range(k):
        h = 1e-5 * max(1.0, abs(xp[i]))
        e = np.zeros(k)
        e[i] = h
        gp = _nll_grad(xp + e, X, lo, hi, sw, exact)[1]
        gm = _nll_grad(xp - e, X, lo, hi, sw, exact)[1]
        H[:, i] = (gp - gm) / (2.0 * h)
    H = 0.5 * (H + H.T)
    try:
        cov = np.linalg.inv(H)
        if not np.all(np.isfinite(cov)) or np.any(np.diag(cov) <= 0):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)
    g_final = _nll_grad(xp, X, lo, hi, sw, exact)[1]
    converged = bool(res.success) or float(np.max(np.abs(g_final))) < 1e-4 * max(1.0, float(vals.size))
    return CenRegFit(beta=xp[:-1].copy(), log_sigma=float(xp[-1]), cov=cov, lognormal=lognormal,
                     converged=converged)


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
        try:
            theta = rng.multivariate_normal(np.concatenate([beta, [th]]), fit.cov, method="cholesky")
        except (np.linalg.LinAlgError, ValueError):
            theta = rng.multivariate_normal(np.concatenate([beta, [th]]), fit.cov, method="eigh")
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
        lp = _log_interval_prob(a, b)
        if fit.lognormal:
            lp_s = _log_interval_prob(a - s, b - s)
            y = np.exp(mu + 0.5 * s * s + lp_s - lp)
        else:
            y = mu + s * (_phi(a) - _phi(b)) / np.maximum(np.exp(lp), 1e-300)
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
    p-values are capped at 0.999 so a degenerate ``p = 1`` (e.g. from a Monte Carlo
    p-value with no smaller null draw) cannot dominate the between-imputation variance;
    the cap can only make the pooled p-value larger.
    """
    from scipy.stats import t as t_dist

    p = np.asarray(pvalues, dtype=np.float64)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return float("nan")
    if p.size == 1:
        return float(p[0])
    z = -ndtri(np.clip(p, 1e-300, 0.999))
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
