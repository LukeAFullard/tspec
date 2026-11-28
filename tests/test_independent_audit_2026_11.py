"""Regression tests for the batch-1 fixes of the independent audit (2026-11).

See ``audit/INDEPENDENT_AUDIT_2026-11.md`` (AUD-01 is covered in ``test_time_adapter.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

import tspec
from tspec.exceptions import InsufficientDataError
from tspec.stpd import _step_vs_hinge_pvalue


def _t(seed: int = 7, n: int = 200, span: float = 20.0) -> np.ndarray:
    return np.sort(np.random.default_rng(seed).uniform(0.0, span, n))


# ----------------------------------------------------------------------------- AUD-03 / AUD-06
def test_step_vs_hinge_statistic_separates_steps_from_bends() -> None:
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 6, 72))
    tp = int(np.searchsorted(t, 3.0))
    bend = 0.5 * t - np.maximum(t - 3.0, 0.0)
    step = 0.5 * t + 2.0 * (t >= 3.0)
    # Noiseless / very precise bends: the continuous knot is optimized between samples
    assert _step_vs_hinge_pvalue(t, bend, None, tp, 5)[0] == 1.0
    assert _step_vs_hinge_pvalue(t, bend + rng.normal(0, 1e-3, t.size), None, tp, 5)[0] > 0.5
    # Steps: significant, and the step carries most of the break's explained variance
    p_step, share_step = _step_vs_hinge_pvalue(t, step + rng.normal(0, 0.5, t.size), None, tp, 5)
    assert p_step < 1e-4
    assert share_step > 0.5
    p0, share0 = _step_vs_hinge_pvalue(t, step, None, tp, 5)
    assert p0 == 0.0 and share0 > 0.5
    # Location/scale invariance
    p_off, _ = _step_vs_hinge_pvalue(t, 1e9 + 1e-3 * (step + rng.normal(0, 0.5, t.size)), None, tp, 5)
    assert p_off < 1e-4
    # A noisy bend has a small step share
    _, share_bend = _step_vs_hinge_pvalue(t, bend + rng.normal(0, 0.5, t.size), None, tp, 5)
    assert share_bend < 0.25


@pytest.mark.parametrize("sd", [0.0, 1e-3])
def test_precise_bend_is_a_turning_point_not_a_jump(sd: float) -> None:
    """AUD-06: NDRI rejected noiseless/high-SNR bends as jumps."""
    t = _t()
    y = 0.5 * t - np.maximum(t - 10.0, 0.0) + np.random.default_rng(1).normal(0, sd, t.size)
    res = tspec.stpd(t, y, band_method="conditional")
    assert any(abs(p.time_year - 10.0) < 0.5 for p in res.turning_points)
    assert res.flagged_jumps == ()
    assert res.jump_test == "step"


def test_pure_step_is_flagged_not_reported_as_turning_point() -> None:
    """AUD-03: a vertical step with no change of slope must not become a turning point."""
    t = _t(3)
    y = 0.5 * t + 3.0 * (t >= 10.0) + np.random.default_rng(3).normal(0, 0.5, t.size)
    res = tspec.stpd(t, y, band_method="conditional")
    assert not any(abs(p.time_year - 10.0) < 1.0 for p in res.turning_points)
    assert any(abs(f.time_year - 10.0) < 1.0 for f in res.flagged_jumps)
    fj = min(res.flagged_jumps, key=lambda f: abs(f.time_year - 10.0))
    assert fj.step_pvalue < 0.05 and fj.step_share >= 0.25
    assert any(w.code == "STEP_JUMP_CANDIDATE" for w in res.diagnostics.warnings)
    assert "vertical step fits significantly better" in res.explain()


def test_jump_test_options_and_parity() -> None:
    t = _t()
    y = 0.5 * t - np.maximum(t - 10.0, 0.0) + np.random.default_rng(2).normal(0, 0.3, t.size)
    assert tspec.stpd(t, y, jump_test="ndri", band_method="conditional").jump_test == "ndri"
    assert tspec.stpd(t, y, jump_test="both", band_method="conditional").jump_test == "both"
    assert tspec.stpd(t, y, parity_mode=True).jump_test == "ndri"
    with pytest.raises(ValueError, match="jump_test"):
        tspec.stpd(t, y, jump_test="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="step_alpha"):
        tspec.stpd(t, y, step_alpha=1.5)


def test_turning_point_records_step_statistics() -> None:
    t = _t()
    y = 0.5 * t - np.maximum(t - 10.0, 0.0) + np.random.default_rng(4).normal(0, 0.2, t.size)
    res = tspec.stpd(t, y, band_method="conditional")
    tp = res.turning_points[0]
    assert np.isfinite(tp.step_pvalue) and np.isfinite(tp.step_share)
    assert "step-vs-bend p=" in res.explain()


def test_analyze_series_warns_about_steps_missed_by_just() -> None:
    """A step that JUST misses must not be reported as a turning point by the pipeline."""
    t = _t(11, n=240)
    rng = np.random.default_rng(11)
    y = 0.5 * t + 3.0 * (t >= 10.0) + rng.normal(0, 0.5, t.size)
    res = tspec.analyze_series(t, y, remove_seasonality=False, band_method="conditional")
    jump_found = any(abs(j.time_year - 10.0) < 0.5 for j in res.jumps.jumps)
    tp_near = any(abs(p.time_year - 10.0) < 1.0 for p in res.turning_points.turning_points)
    assert jump_found or not tp_near
    if res.turning_points.flagged_jumps:
        assert "not detected by Stage 1 (JUST)" in res.explain()


# ----------------------------------------------------------------------------- AUD-09
def test_noiseless_line_has_no_jumps() -> None:
    t = _t()
    y = 3.0 + 0.2 * t
    assert tspec.just_jumps(t, y, season="none").jumps == ()
    assert len(tspec.analyze_series(t, y).segments) == 1
    # Noiseless steps, even at tiny scales, are still detected
    for scale in (1.0, 1e-13):
        jr = tspec.just_jumps(t, scale * (1.0 + 2.0 * (t > 10.0)), season="none")
        assert len(jr.jumps) == 1


# ----------------------------------------------------------------------------- AUD-10
def test_monitor_requires_adequate_history_for_numeric_time() -> None:
    t = _t()
    y = 0.1 * t + np.cos(2 * np.pi * t) + np.random.default_rng(5).normal(0, 0.3, t.size)
    with pytest.raises(InsufficientDataError, match="historical period"):
        tspec.just_monitor(t, y, start=5)
    # One seasonal cycle of history is enough to run
    res = tspec.just_monitor(t, y, start=int(np.searchsorted(t, 2.0)), horizon=10)
    assert len(res.steps) >= 1
