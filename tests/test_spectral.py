"""Unit tests for weighted LSSA and ALLSSA spectral analysis."""

from __future__ import annotations

import numpy as np
import pytest

from tspec import InsufficientDataError, allssa, lssa


def test_lssa_recovers_sinusoid_on_irregular_grid(rng: np.random.Generator) -> None:
    t = np.sort(rng.uniform(0.0, 6.0, size=140))
    y = 2.0 - 0.15 * t + 0.8 * np.cos(2 * np.pi * 1.25 * t) + rng.normal(0.0, 0.05, size=t.size)
    omega = np.linspace(0.2, 3.0, 141)

    res = lssa(t, y, omega=omega, trend="linear", alpha=0.01)
    peak_freq = float(res.frequencies[int(np.argmax(res.spectrum))])
    assert abs(peak_freq - 1.25) <= 0.025
    assert res.spectrum.max() > res.critical_value
    assert "LSSA Result" in res.summary()
    assert "highest spectral peak" in res.explain()
    assert res.peak_fap < 0.01


def test_lssa_trend_none_works_without_unbound_error() -> None:
    # Upstream LSSA.py crashed with UnboundLocalError on N12 when trend='none', freq=[]
    t = np.linspace(0.0, 5.0, 60)
    y = np.sin(2 * np.pi * 1.0 * t)
    res = lssa(t, y, omega=[0.5, 1.0, 1.5], trend="none")
    assert res.coeff.size == 0
    assert res.spectrum[1] > 0.99


def test_lssa_polynomial_trends_and_shift_invariance(rng: np.random.Generator) -> None:
    t = np.sort(rng.uniform(0.0, 4.0, size=80))
    y = 1.0 + 0.5 * t - 0.2 * t**2 + 0.05 * t**3 + rng.normal(0.0, 0.01, size=t.size)
    res = lssa(t, y, omega=[1.0, 2.0], trend="cubic")
    assert res.coeff.size == 4
    assert np.allclose(res.coeff, [1.0, 0.5, -0.2, 0.05], atol=0.05)

    # Shift invariance on linear trend
    y_lin = 3.0 + 0.4 * t + 0.5 * np.cos(2 * np.pi * 1.3 * t)
    r1 = lssa(t, y_lin, omega=[0.5, 1.3, 2.1], trend="linear")
    r2 = lssa(t + 2000.0, y_lin, omega=[0.5, 1.3, 2.1], trend="linear")
    assert np.allclose(r1.spectrum, r2.spectrum, atol=1e-12)


def test_allssa_recovers_non_integer_frequencies_without_leakage(rng: np.random.Generator) -> None:
    t = np.sort(rng.uniform(0.0, 8.0, size=180))
    y = (
        1.5
        + 0.1 * t
        + 0.7 * np.cos(2 * np.pi * 1.0 * t)
        + 0.5 * np.sin(2 * np.pi * 2.35 * t)
        + 0.4 * np.cos(2 * np.pi * 0.37 * t)
        + rng.normal(0.0, 0.03, size=t.size)
    )
    omega = np.arange(1, 20) / 5.0
    res = allssa(t, y, omega=omega, decimal=2, alpha=0.01)
    assert res.converged is True
    assert res.frequencies.size == 3
    for true_f in (0.37, 1.0, 2.35):
        assert np.min(np.abs(res.frequencies - true_f)) <= 0.02
    assert "ALLSSA Result" in res.summary()
    assert "ALLSSA" in res.explain()


def test_allssa_sub_half_frequency_does_not_crash() -> None:
    # Upstream ALLSSA.py crashed with singular matrix when frequency < 0.5 rounded to 0.0
    t = np.linspace(0.0, 10.0, 120)
    y = 0.5 * np.cos(2 * np.pi * 0.25 * t)
    res = allssa(t, y, omega=np.linspace(0.1, 2.0, 20), decimal=2, alpha=0.01)
    assert np.all(res.frequencies > 0.0)
    assert np.min(np.abs(res.frequencies - 0.25)) <= 0.02


def test_lssa_insufficient_dof_raises() -> None:
    # Upstream divided by zero at n == nc + 2
    t = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(InsufficientDataError):
        lssa(t, y, omega=[1.0], trend="linear")


def test_lssa_scalar_weight_invariance(rng: np.random.Generator) -> None:
    t = np.sort(rng.uniform(0.0, 5.0, size=60))
    y = 1.0 + 0.2 * t + 0.5 * np.sin(2 * np.pi * 1.1 * t) + rng.normal(0.0, 0.1, size=60)
    r1 = lssa(t, y, weights=1.0, omega=[0.5, 1.1, 2.0])
    r2 = lssa(t, y, weights=7.5, omega=[0.5, 1.1, 2.0])
    assert np.allclose(r1.spectrum, r2.spectrum, atol=1e-12)
    assert np.allclose(r1.coeff, r2.coeff, atol=1e-12)
    assert np.allclose(r1.cov, r2.cov, atol=1e-12)
    assert np.isclose(r2.norm_res, 7.5 * r1.norm_res, rtol=1e-12)
