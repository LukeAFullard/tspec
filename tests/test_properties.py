"""Property-based invariance & edge-case tests via Hypothesis (Tier 3)."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from tspec import detect_turning_points, just_decompose, just_jumps, lssa


@settings(max_examples=25, deadline=None)
@given(
    shift=st.floats(min_value=-5000.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
    scale=st.floats(min_value=0.05, max_value=500.0, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_lssa_shift_invariance_and_scale_equivariance(
    shift: float, scale: float, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, 5.0, size=60))
    y = 1.0 + 0.3 * t + 0.6 * np.cos(2 * np.pi * 1.2 * t) + rng.normal(0.0, 0.1, size=60)
    omega = np.array([0.5, 1.0, 1.2, 2.0])

    r0 = lssa(t, y, omega=omega, trend="linear")
    r_shift = lssa(t + shift, y, omega=omega, trend="linear")
    r_scale = lssa(t, scale * y, omega=omega, trend="linear")

    assert np.allclose(r0.spectrum, r_shift.spectrum, atol=1e-10)
    assert np.allclose(r0.spectrum, r_scale.spectrum, atol=1e-10)
    assert np.allclose(r_scale.amplitudes, scale * r0.amplitudes, rtol=1e-9, atol=1e-10)


@settings(max_examples=20, deadline=None)
@given(
    shift=st.floats(min_value=-2000.0, max_value=2000.0, allow_nan=False, allow_infinity=False),
    scale=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_just_and_stpd_shift_and_scale_properties(
    shift: float, scale: float, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 6.0, 72)
    y = np.where(t <= 3.0, 0.5 * t, 1.5 - 2.0 * (t - 3.0)) + rng.normal(0.0, 0.08, size=72)

    s0 = detect_turning_points(t, y, size=50, step=10, margin=8, snr_mode="centered")
    s_shift = detect_turning_points(t + shift, y, size=50, step=10, margin=8, snr_mode="centered")
    s_scale = detect_turning_points(t, scale * y, size=50, step=10, margin=8, snr_mode="centered")

    assert s0.tp_indices.tolist() == s_shift.tp_indices.tolist()
    assert s0.tp_indices.tolist() == s_scale.tp_indices.tolist()
    if s0.turning_points:
        assert np.isclose(
            s_scale.turning_points[0].direction,
            scale * s0.turning_points[0].direction,
            rtol=1e-8,
        )
        assert np.isclose(
            s0.turning_points[0].ndri_rmse, s_shift.turning_points[0].ndri_rmse, atol=1e-10
        )

    j0 = just_jumps(t, y, season="none", size=36, step=12, mag_th=0.0, dir_th=0.0)
    j_shift = just_jumps(t + shift, y, season="none", size=36, step=12, mag_th=0.0, dir_th=0.0)
    assert j0.jump_indices.tolist() == j_shift.jump_indices.tolist()


def test_degenerate_constant_and_noiseless_series() -> None:
    t = np.linspace(0.0, 6.0, 72)
    y_const = np.full_like(t, 4.2)

    r_lssa = lssa(t, y_const, omega=[1.0, 2.0])
    assert np.allclose(r_lssa.spectrum, 0.0)
    assert r_lssa.norm_res < 1e-24

    r_stpd = detect_turning_points(t, y_const, size=50, step=10, margin=8)
    assert len(r_stpd.turning_points) == 0

    r_dec = just_decompose(t, y_const, season="ols", size=36, step=12)
    assert np.allclose(r_dec.trend, 4.2, atol=1e-10)
    assert np.allclose(r_dec.seasonal, 0.0, atol=1e-10)
    assert np.allclose(r_dec.remainder, 0.0, atol=1e-10)
