"""Serial-correlation model for calibrating tests under red noise (audit finding AUD-02).

The change-point tests in ``tspec`` (JUST jump F-tests, STPD slope-change t-tests and OLS
segment slopes) were calibrated for independent errors. Under positively autocorrelated
residuals their false-positive rates explode (e.g. ~34% / 95% family-wise false jumps at
AR(1) rho = 0.3 / 0.6 for a nominal 5%). This module provides

* a robust, jump-insensitive estimate of a lag-1 autocorrelation from residuals of a
  smooth global model (difference-based estimator of Chakar, Lebarbier, Levy-Leduc &
  Robin, 2017, which is unaffected by level shifts);
* a generator of unit-variance AR(1) (observation order) or Ornstein-Uhlenbeck
  (continuous time, ``rho ** (dt / median_dt)``) error series for Monte Carlo nulls;
* the long-run variance factor ``kappa^2 = (1 + rho) / (1 - rho)`` for analytical
  (Bonferroni / F / t) calibrations, and an exact O(n) sandwich variance of weighted
  least-squares coefficients under the fitted AR(1)/OU correlation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ._types import FloatArray

SerialMode = Literal["ar1", "ou", "none"]

RHO_MAX = 0.95
TRIM = 0.05
# Sampling uncertainty of the difference-based estimator: SD ~ 2.0 / sqrt(n) for rho >= 0.3
# and ~0.9 / sqrt(n) at rho = 0 (simulation, n = 60-720). Because the long-run variance
# factor is convex in rho, plugging in an under-estimate is costlier than an over-estimate;
# when the estimate is distinguishable from zero, calibration uses rho_hat + RHO_MARGIN_SE
# standard errors.
RHO_SE_COEF = 2.0
RHO_SE0_COEF = 0.9
RHO_DETECT_Z = 2.0
RHO_MARGIN_SE = 1.0
SMOOTH_KNOTS = 4
JUMP_Z = 6.0


def _upper_trimmed_mean(v: FloatArray, trim: float) -> float:
    if v.size == 0:
        return float("nan")
    k = max(1, int(np.ceil((1.0 - trim) * v.size)))
    return float(np.mean(np.sort(v)[:k]))


@dataclass(frozen=True)
class SerialSpec:
    """Fitted serial-correlation model of the (whitened) residuals.

    ``mode`` is ``"ar1"`` (correlation ``rho`` between consecutive observations),
    ``"ou"`` (continuous time: correlation ``rho ** (dt / median_dt)``) or ``"none"``.
    ``rho`` is the value used for calibration (the point estimate ``rho_hat`` plus an
    uncertainty margin, see :func:`estimate_serial_spec`).
    """

    mode: SerialMode
    rho: float = 0.0
    median_dt: float = 1.0
    rho_hat: float = 0.0

    @property
    def active(self) -> bool:
        return self.mode != "none" and self.rho > 0.0

    @property
    def kappa2(self) -> float:
        """Long-run variance inflation ``(1 + rho) / (1 - rho)`` (1 when inactive)."""
        if not self.active:
            return 1.0
        return float((1.0 + self.rho) / (1.0 - self.rho))

    def lag_coefficients(self, t: FloatArray) -> FloatArray:
        """Lag-1 coefficients ``a_i`` (``i >= 1``) linking observation ``i`` to ``i - 1``."""
        n = int(t.size)
        if n < 2 or not self.active:
            return np.zeros(max(n - 1, 0), dtype=np.float64)
        if self.mode == "ar1":
            return np.full(n - 1, self.rho, dtype=np.float64)
        dt = np.diff(np.asarray(t, dtype=np.float64))
        md = self.median_dt if self.median_dt > 0.0 else 1.0
        return np.asarray(np.power(self.rho, np.maximum(dt, 0.0) / md), dtype=np.float64)


NO_SERIAL = SerialSpec("none")


def validate_serial_mode(mode: str) -> SerialMode:
    if mode not in ("ar1", "ou", "none"):
        raise ValueError(
            f"Unknown `serial_correlation` {mode!r}. Options: 'ar1', 'ou', 'none'."
        )
    return mode  # type: ignore[return-value]


def estimate_serial_spec(
    t: FloatArray,
    y: FloatArray,
    w: FloatArray | None,
    mode: SerialMode,
    *,
    harmonics: FloatArray | None = None,
) -> SerialSpec:
    """Robust lag-1 autocorrelation of residuals from a smooth global model.

    The residuals of a weighted fit of ``[1, u, u^2]``, up to 4 hinge terms
    ``max(u - c, 0)`` at equally spaced knots and the given harmonics are
    whitened by ``sqrt(w)``. With ``d_k = r_{i+k} - r_i``, ``Var(d_2) / Var(d_1) = 1 + rho``
    for AR(1) errors (the difference-based identity of Chakar et al., 2017). Differencing
    removes piecewise-constant mean shifts, so genuine jumps (and slowly varying trend
    misfit) hardly affect it, whereas a direct residual autocorrelation would be inflated
    by them. Steps that stand out of the first-pass residual differences by more than six
    robust standard deviations get their own level-shift column, so that a large jump is
    not smeared into a residual ramp by the smooth fit. Both variances are 5%-upper-trimmed
    means of squared differences (the
    trimming constant cancels in the ratio for Gaussian errors, and a jump only adds one
    or two extreme differences): about half the standard deviation of the median-based
    version. Clipped to ``[0, 0.95]``. The calibration value ``rho`` is ``0`` unless
    ``rho_hat`` exceeds 2 null standard errors (``0.9 / sqrt(n)``); then it adds one
    standard error (``2 / sqrt(n)``) to guard against under-estimation. ``rho_hat`` is kept
    for reporting.
    """
    if mode == "none":
        return NO_SERIAL
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = int(t.size)
    if n < 12:
        return SerialSpec(mode, 0.0, 1.0)
    span = float(t[-1] - t[0]) if t[-1] > t[0] else 1.0
    u = (t - t[0]) / span
    cols = [np.ones(n), u, u * u]
    # A few hinge terms let the smooth fit follow genuine bends, whose misfit would
    # otherwise masquerade as autocorrelation (jumps are handled by differencing).
    n_knots = min(SMOOTH_KNOTS, max(0, n // 25))
    for c in np.linspace(0.0, 1.0, n_knots + 2)[1:-1]:
        cols.append(np.maximum(u - c, 0.0))
    if harmonics is not None:
        for f in np.asarray(harmonics, dtype=np.float64).ravel():
            cols += [np.cos(2.0 * np.pi * f * t), np.sin(2.0 * np.pi * f * t)]
    X = np.column_stack(cols)
    sw = np.sqrt(w) if w is not None else np.ones(n)
    if X.shape[1] >= n // 2:
        X = X[:, :3]
    coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    r = (y - X @ coef) * sw
    # Large jumps: a smooth fit smears a big step into a long residual ramp, which looks
    # like autocorrelation. Give every step that stands out of the first-pass residual
    # differences (> JUMP_Z robust SDs) its own level-shift column and refit (twice).
    for _ in range(2):
        d = np.diff(r)
        mad = float(np.median(np.abs(d - np.median(d))))
        if not np.isfinite(mad) or mad <= 0.0:
            break
        big = np.flatnonzero(np.abs(d) > JUMP_Z * 1.4826 * mad) + 1
        big = big[(big >= 3) & (big <= n - 3)]
        if big.size == 0 or big.size > max(1, n // 20):
            break
        steps = (np.arange(n)[:, None] >= big[None, :]).astype(np.float64)
        Xs = np.hstack([X, steps])
        if Xs.shape[1] >= n // 2:
            break
        coef, *_ = np.linalg.lstsq(Xs * sw[:, None], y * sw, rcond=None)
        r_new = (y - Xs @ coef) * sw
        if np.allclose(r_new, r):
            break
        r = r_new
    q1 = _upper_trimmed_mean(np.diff(r) ** 2, TRIM)
    q2 = _upper_trimmed_mean((r[2:] - r[:-2]) ** 2, TRIM)
    if not np.isfinite(q1) or q1 <= 0.0:
        return SerialSpec(mode, 0.0, 1.0)
    rho = float(q2 / q1 - 1.0)
    rho_hat = float(np.clip(rho, 0.0, RHO_MAX)) if np.isfinite(rho) else 0.0
    se0 = RHO_SE0_COEF / np.sqrt(n)
    # Pre-test: an estimate indistinguishable from zero is treated as independent errors
    # (a tiny rho would only weaken the analytical tail bounds the calibrations rely on
    # below Monte Carlo resolution and cost power on white noise).
    rho_eff = 0.0
    if rho_hat > RHO_DETECT_Z * se0:
        rho_eff = float(min(RHO_MAX, rho_hat + RHO_MARGIN_SE * RHO_SE_COEF / np.sqrt(n)))
    dts = np.diff(t)
    md = float(np.median(dts[dts > 0])) if np.any(dts > 0) else 1.0
    return SerialSpec(mode, rho_eff, md, rho_hat)


def correlated_normals(
    rng: np.random.Generator, t: FloatArray, n_draws: int, spec: SerialSpec
) -> FloatArray:
    """``(n, n_draws)`` unit-variance Gaussian series with the AR(1)/OU correlation of ``spec``."""
    n = int(np.asarray(t).size)
    z = rng.standard_normal((n, int(n_draws)))
    if not spec.active or n < 2:
        return z
    a = spec.lag_coefficients(np.asarray(t, dtype=np.float64))
    s = np.sqrt(np.maximum(1.0 - a * a, 0.0))
    x = np.empty_like(z)
    x[0] = z[0]
    for i in range(1, n):
        x[i] = a[i - 1] * x[i - 1] + s[i - 1] * z[i]
    return x


def _apply_cholesky_transpose(t: FloatArray, M: FloatArray, spec: SerialSpec) -> FloatArray:
    """Return ``L^T M`` where ``R = L L^T`` is the AR(1)/OU correlation matrix of ``spec``.

    ``L`` is the lower-triangular generator ``x = L z`` of :func:`correlated_normals`;
    ``L^T M`` is obtained by back-substitution with its bidiagonal inverse in O(n k).
    """
    n = int(t.size)
    a = spec.lag_coefficients(t)
    s = np.sqrt(np.maximum(1.0 - a * a, 1e-300))
    # W = L^{-1}: (W x)_0 = x_0, (W x)_i = (x_i - a_i x_{i-1}) / s_i.
    # Solve W^T v = M (upper bidiagonal) backwards: v = W^{-T} M = L^T M.
    v = np.empty_like(M)
    s_full = np.concatenate(([1.0], s))
    a_full = np.concatenate((a, [0.0]))
    v[n - 1] = s_full[n - 1] * M[n - 1]
    for j in range(n - 2, -1, -1):
        v[j] = s_full[j] * (M[j] + a_full[j] * v[j + 1] / s_full[j + 1])
    return v


def sandwich_vif(
    t: FloatArray,
    X: FloatArray,
    w: FloatArray | None,
    spec: SerialSpec,
) -> tuple[FloatArray, FloatArray]:
    """Variance inflation of weighted least-squares coefficients under AR(1)/OU errors.

    For errors ``eps = D^{-1/2} eta`` (``D = diag(w)``, ``eta`` with correlation ``R``),
    ``Var(beta) = (X'WX)^{-1} X~' R X~ (X'WX)^{-1} sigma^2`` with ``X~ = sqrt(w) X``.
    Returns ``(vif_coef, vif_fitted)``: the ratio of the correlated to the independent-
    error variance for every coefficient and for every fitted value ``X beta``.
    """
    n, k = X.shape
    if not spec.active or n < 3:
        return np.ones(k), np.ones(n)
    sw = np.sqrt(w) if w is not None else np.ones(n)
    Xt = X * sw[:, None]
    xtx_inv = np.linalg.pinv(Xt.T @ Xt)
    LtX = _apply_cholesky_transpose(np.asarray(t, dtype=np.float64), Xt, spec)
    meat = LtX.T @ LtX
    v_corr = xtx_inv @ meat @ xtx_inv
    d_iid = np.diag(xtx_inv)
    vif_coef = np.where(d_iid > 0.0, np.diag(v_corr) / np.where(d_iid > 0.0, d_iid, 1.0), 1.0)
    fit_iid = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    fit_corr = np.einsum("ij,jk,ik->i", X, v_corr, X)
    vif_fit = np.where(fit_iid > 0.0, fit_corr / np.where(fit_iid > 0.0, fit_iid, 1.0), 1.0)
    return np.maximum(vif_coef, 0.0), np.maximum(vif_fit, 0.0)


def resolve_serial_spec(
    mode: str,
    *,
    parity_mode: bool,
    t: FloatArray,
    y: FloatArray,
    weight_kind: str,
    weight_vector: FloatArray | None,
    season: str,
    omega: FloatArray | None,
) -> SerialSpec:
    """Validate ``serial_correlation`` and estimate the residual AR(1)/OU model.

    Returns :data:`NO_SERIAL` for ``"none"``, in ``parity_mode`` and when the caller
    supplied a full 2D inverse-covariance ``weights`` matrix (the user's covariance is
    then already the error model). The smooth global fit uses the integer harmonics of
    ``omega`` (or none when ``season == "none"``).
    """
    m = validate_serial_mode(mode)
    if m == "none" or parity_mode or weight_kind == "matrix":
        return NO_SERIAL
    harm: FloatArray | None = None
    if season != "none" and omega is not None and np.asarray(omega).size > 0:
        h = np.unique(np.round(np.asarray(omega, dtype=np.float64)))
        h = h[h >= 1.0]
        harm = h if h.size > 0 else None
    w = weight_vector if weight_kind == "vector" else None
    return estimate_serial_spec(t, y, w, m, harmonics=harm)


def serial_warning_message(spec: SerialSpec) -> str:
    return (
        f"Tests calibrated for serially correlated errors ({spec.mode}, estimated lag-1 "
        f"rho = {spec.rho_hat:.2f}, calibrated with rho = {spec.rho:.2f} including its "
        f"sampling uncertainty; long-run variance factor {spec.kappa2:.2f})."
    )


def deflate_t_pvalue(p: float, df: float, kappa2: float) -> float:
    """Two-sided t p-value recomputed with the statistic divided by ``sqrt(kappa2)``."""
    if kappa2 <= 1.0 or not np.isfinite(p) or p <= 0.0 or df <= 0:
        return p
    from scipy.stats import t as t_dist

    t_obs = float(t_dist.isf(0.5 * min(p, 1.0), df))
    return float(2.0 * t_dist.sf(t_obs / np.sqrt(kappa2), df))
