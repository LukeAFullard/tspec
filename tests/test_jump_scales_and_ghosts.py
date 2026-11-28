"""Regression tests for batch 3 of the independent audit (AUD-04, AUD-17, AUD-18)."""

from __future__ import annotations

import numpy as np
import pytest

import tspec


def _series(seed: int, kind: str, jump: float = 0.0, n: int = 240) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 20, n))
    trend = {"line": 0.05 * t, "curved": 3.0 * np.sin(2 * np.pi * t / 9.0)}[kind]
    y = trend + jump * (t >= 10.3) + np.cos(2 * np.pi * t) + 0.4 * np.sin(4 * np.pi * t)
    return t, y + rng.normal(0, 0.5, n)


def test_smooth_curvature_is_not_a_vertical_step() -> None:
    """AUD-18: the 2-df jump test reports curvature; the step test does not."""
    for seed in range(3):
        t, y = _series(seed, "curved")
        default = tspec.just_jumps(t, y, season="ols")
        assert len(default.jumps) >= 1
        assert any(w.code == "NOT_A_VERTICAL_STEP" for w in default.diagnostics.warnings)
        steps = tspec.just_jumps(t, y, season="ols", step_alpha=0.05, window_scales=(1, 2, 4))
        assert steps.jumps == ()


def test_multiscale_step_detection_gains_power() -> None:
    """AUD-04: a 2-sigma step missed by single 3-cycle windows is found with scales (1, 2, 4)."""
    single = multi = 0
    for seed in range(6):
        t, y = _series(seed, "line", jump=1.0)
        single += any(
            abs(t[j.valid_index] - 10.3) < 0.5 for j in tspec.just_jumps(t, y, season="ols").jumps
        )
        r = tspec.just_jumps(t, y, season="ols", step_alpha=0.05, window_scales=(1, 2, 4))
        multi += any(abs(t[j.valid_index] - 10.3) < 0.5 for j in r.jumps)
    assert multi >= 4
    assert multi > single


def test_window_scales_validation_and_parity() -> None:
    t, y = _series(0, "line", jump=2.0)
    with pytest.raises(ValueError, match="window_scales"):
        tspec.just_jumps(t, y, window_scales=(0.5, 1.0))
    # Parity mode ignores extra scales (single-scale published algorithm)
    par = tspec.just_jumps(t, y, parity_mode=True, window_scales=(1, 2))
    assert isinstance(par.jumps, tuple)


def test_ghost_jumps_one_seasonal_cycle_from_a_real_jump() -> None:
    """AUD-17: in a 3-cycle window a step one seasonal period away from the split is nearly
    indistinguishable from "ramp + annual sawtooth" (harmonics), so single-scale JUST can
    report an alias ~1 yr from a real jump. Longer windows (window_scales) resolve it."""
    t = 2000 + (np.arange(240) + 0.5) / 12
    ghosts_single = ghosts_multi = missed_multi = 0
    for seed in range(100, 130):
        rng = np.random.default_rng(seed)
        y = 5 + 0.05 * (t - t[0]) + 2.0 * (t >= 2010) + np.cos(2 * np.pi * t) + 0.4 * np.sin(4 * np.pi * t)
        y = y + rng.normal(0, 0.3, t.size)
        single = tspec.just_jumps(t, y, season="ols")
        multi = tspec.just_jumps(t, y, season="ols", window_scales=(1, 2, 4))
        ghosts_single += sum(abs(single.times_numeric[e.valid_index] - 2010) >= 0.25 for e in single.jumps)
        ghosts_multi += sum(abs(multi.times_numeric[e.valid_index] - 2010) >= 0.25 for e in multi.jumps)
        missed_multi += not any(abs(multi.times_numeric[e.valid_index] - 2010) < 0.25 for e in multi.jumps)
    assert missed_multi == 0
    assert ghosts_multi <= 1
    assert ghosts_multi < ghosts_single


def test_conditional_retest_with_known_jump() -> None:
    """The conditional re-test (known_jumps) removes an alias once the real jump is modelled."""
    from tspec._linalg import prepare_weights
    from tspec.just import _nested_jump_ftests

    t = 2000 + (np.arange(240) + 0.5) / 12
    rng = np.random.default_rng(129)
    y = 5 + 0.05 * (t - t[0]) + 2.0 * (t >= 2010) + np.cos(2 * np.pi * t) + 0.4 * np.sin(4 * np.pi * t)
    y = y + rng.normal(0, 0.3, t.size)
    ws = int(np.searchsorted(t, 2010 - 1.96 - 1e-9))
    tw, yw = t[ws : ws + 36], y[ws : ws + 36]
    true_i = int(np.searchsorted(tw, 2010))
    alias_i = true_i - 12
    kw = dict(season="ols", calibration="bonferroni", step_quadratic=True)
    p_alone, _ = _nested_jump_ftests(tw, yw, prepare_weights(None, 36), alias_i, np.array([1.0, 2.0, 3.0, 4.0]), **kw)
    p_cond, _ = _nested_jump_ftests(
        tw, yw, prepare_weights(None, 36), alias_i, np.array([1.0, 2.0, 3.0, 4.0]), known_jumps=[true_i], **kw
    )
    assert p_alone < 1e-3
    assert p_cond > p_alone


def test_analyze_series_reports_small_step_as_jump_not_turning_point() -> None:
    t, y = _series(3, "line", jump=1.0)
    r = tspec.analyze_series(t, y, band_method="conditional")
    assert any(abs(j.time_year - 10.3) < 0.5 for j in r.jumps.jumps)
    assert not any(abs(p.time_year - 10.3) < 1.0 for p in r.turning_points.turning_points)
