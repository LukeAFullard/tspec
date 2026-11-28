"""Tests for the serial-correlation calibration (audit finding AUD-02)."""

from __future__ import annotations

import numpy as np
import pytest

import tspec
from tspec._serial import (
    NO_SERIAL,
    SerialSpec,
    _apply_cholesky_transpose,
    correlated_normals,
    deflate_t_pvalue,
    estimate_serial_spec,
    sandwich_vif,
)


def _ar1(rng: np.random.Generator, n: int, rho: float) -> np.ndarray:
    x = np.empty(n)
    x[0] = rng.normal()
    for i in range(1, n):
        x[i] = rho * x[i - 1] + np.sqrt(1 - rho**2) * rng.normal()
    return x


def _dense_cholesky(t: np.ndarray, spec: SerialSpec) -> np.ndarray:
    a = spec.lag_coefficients(t)
    s = np.sqrt(1 - a * a)
    n = t.size
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            L[i, j] = np.prod(a[j:i]) * (1.0 if j == 0 else s[j - 1])
    return L


@pytest.mark.parametrize("mode", ["ar1", "ou"])
def test_sandwich_vif_matches_dense_computation(mode: str) -> None:
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 10, 35))
    spec = SerialSpec(mode, 0.6, float(np.median(np.diff(t))))
    L = _dense_cholesky(t, spec)
    R = L @ L.T
    assert np.allclose(np.diag(R), 1.0)
    M = rng.normal(size=(t.size, 3))
    assert np.allclose(_apply_cholesky_transpose(t, M, spec), L.T @ M, atol=1e-12)
    X = np.column_stack([np.ones(t.size), t])
    w = rng.uniform(0.5, 2.0, t.size)
    vif_c, vif_f = sandwich_vif(t, X, w, spec)
    Xt = X * np.sqrt(w)[:, None]
    A = np.linalg.inv(Xt.T @ Xt)
    V = A @ Xt.T @ R @ Xt @ A
    assert np.allclose(vif_c, np.diag(V) / np.diag(A))
    assert np.allclose(vif_f, np.einsum("ij,jk,ik->i", X, V, X) / np.einsum("ij,jk,ik->i", X, A, X))
    assert np.all(sandwich_vif(t, X, w, NO_SERIAL)[0] == 1.0)


def test_correlated_normals_have_unit_variance_and_target_lag1() -> None:
    t = np.arange(50, dtype=float)
    E = correlated_normals(np.random.default_rng(1), t, 20000, SerialSpec("ar1", 0.7, 1.0))
    assert np.allclose(E.var(axis=1), 1.0, atol=0.05)
    assert np.allclose(np.mean(E[1:] * E[:-1], axis=1), 0.7, atol=0.05)


def test_estimator_recovers_rho_and_ignores_jumps() -> None:
    rng = np.random.default_rng(2)
    n = 3000
    t = np.arange(n) / 12.0
    e = _ar1(rng, n, 0.5)
    y = 0.02 * t + np.cos(2 * np.pi * t) + 0.5 * e
    rho = estimate_serial_spec(t, y, None, "ar1", harmonics=np.array([1.0, 2.0])).rho
    assert rho == pytest.approx(0.5, abs=0.06)
    y_jump = y + 3.0 * (t >= t[n // 2]) + 2.0 * (t >= t[n // 4])
    rho_j = estimate_serial_spec(t, y_jump, None, "ar1", harmonics=np.array([1.0, 2.0])).rho
    assert abs(rho_j - rho) < 0.03
    white = estimate_serial_spec(t, np.cos(2 * np.pi * t) + rng.normal(size=n), None, "ar1",
                                 harmonics=np.array([1.0])).rho
    assert white < 0.06


def test_deflate_t_pvalue() -> None:
    assert deflate_t_pvalue(0.01, 50, 1.0) == 0.01
    assert deflate_t_pvalue(0.01, 50, 4.0) > 0.01
    assert deflate_t_pvalue(0.0, 50, 4.0) == 0.0
    assert np.isnan(deflate_t_pvalue(float("nan"), 50, 4.0))


def _null_ar_series(seed: int, rho: float, n: int = 240) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, n / 12, n))
    y = 0.05 * t + np.cos(2 * np.pi * t) + 0.4 * np.sin(4 * np.pi * t) + 0.5 * _ar1(rng, n, rho)
    return t, y


def test_just_jumps_controls_false_jumps_under_ar1() -> None:
    fp_ar = fp_none = 0
    for seed in range(12):
        t, y = _null_ar_series(seed, 0.6)
        r = tspec.just_jumps(t, y, season="ols", jump_definition="level_or_slope")
        assert r.serial_correlation == "ar1" and r.serial_rho > 0.3 and r.serial_rho_used >= r.serial_rho
        assert any(w.code == "SERIAL_CORRELATION_ADJUSTED" for w in r.diagnostics.warnings)
        fp_ar += len(r.jumps) > 0
        fp_none += len(
            tspec.just_jumps(t, y, season="ols", serial_correlation="none", jump_definition="level_or_slope").jumps
        ) > 0
    assert fp_ar <= 2
    assert fp_none >= 8


def test_serial_options_and_parity() -> None:
    t, y = _null_ar_series(0, 0.5)
    assert tspec.just_jumps(t, y, season="ols", parity_mode=True).serial_correlation == "none"
    assert tspec.just_jumps(t, y, season="ols", serial_correlation="ou").serial_correlation == "ou"
    with pytest.raises(ValueError, match="serial_correlation"):
        tspec.just_jumps(t, y, serial_correlation="arma")  # type: ignore[arg-type]
    # A user-supplied 2D covariance is taken as the error model
    n = t.size
    C = 0.5 ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    r2 = tspec.just_jumps(t, y, season="ols", weights=np.linalg.inv(C))
    assert r2.serial_correlation == "none"


def test_stpd_controls_false_turning_points_under_ar1() -> None:
    fp_ar = fp_none = 0
    for seed in range(10):
        rng = np.random.default_rng(seed)
        t = np.sort(rng.uniform(0, 20, 200))
        y = 0.5 * t + 0.5 * _ar1(rng, 200, 0.7)
        r = tspec.stpd(t, y, band_method="conditional")
        assert r.serial_correlation == "ar1"
        fp_ar += len(r.turning_points) > 0
        fp_none += len(tspec.stpd(t, y, band_method="conditional", serial_correlation="none").turning_points) > 0
    assert fp_ar <= 1
    assert fp_none >= fp_ar + 3


def test_analyze_series_ols_slope_inflated_under_ar1() -> None:
    rng = np.random.default_rng(3)
    t = 2000 + np.arange(240) / 12
    y = 0.5 * _ar1(rng, 240, 0.7)
    kw = dict(detect_jumps=False, detect_turning_points=False, remove_seasonality=False,
              band_method="conditional")
    a = tspec.analyze_series(t, y, **kw)
    b = tspec.analyze_series(t, y, serial_correlation="none", **kw)
    sa, sb = a.segments[0], b.segments[0]
    assert a.serial_correlation == "ar1" and a.serial_rho > 0.4
    assert sa.slope == pytest.approx(sb.slope)
    assert (sa.slope_ci_high - sa.slope_ci_low) > 1.8 * (sb.slope_ci_high - sb.slope_ci_low)
    assert sa.pvalue > sb.pvalue
    assert "variance inflation of the fitted AR(1) errors" in a.explain()
