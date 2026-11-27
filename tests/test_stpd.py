"""Unit tests for STPD turning-point detection and global piecewise trend fitting."""

from __future__ import annotations

import numpy as np
import pytest

from tspec import detect_turning_points, fit_piecewise_trend, stpd


def test_stpd_detects_continuous_turning_point_and_low_ndri(rng: np.random.Generator) -> None:
    t = np.linspace(0.0, 6.0, 73)
    tp_true = 36  # t = 3.0
    y_true = np.where(t <= t[tp_true], -2.0 * t, -2.0 * t[tp_true] - 12.0 * (t - t[tp_true]))
    y = y_true + rng.normal(0.0, 1.0, size=t.size)

    res = detect_turning_points(t, y, size=60, step=12, margin=10, alpha=0.01)
    assert stpd is detect_turning_points
    assert len(res.turning_points) == 1
    tp = res.turning_points[0]
    assert abs(tp.index - tp_true) <= 3
    assert tp.ndri_rmse < 0.3
    assert tp.slope_before == pytest.approx(-2.0, abs=1.0)
    assert tp.slope_after == pytest.approx(-12.0, abs=1.2)
    assert "STPD Turning Point Result" in res.summary()
    assert "smooth bend" in res.explain()


def test_stpd_rejects_pure_datum_jump_and_flags_high_ndri(rng: np.random.Generator) -> None:
    t = np.linspace(0.0, 6.0, 73)
    y = np.where(t < 3.0, 1.0, 25.0) + rng.normal(0.0, 0.5, size=t.size)
    res = detect_turning_points(t, y, size=60, step=12, margin=8, ndri_th=0.3)
    # Pure step jump produces strongly asymmetric residual balance (high NDRI)
    assert len(res.turning_points) == 0
    assert len(res.flagged_jumps) >= 1
    assert "Jump Alarm Bell" in res.explain()


def test_stpd_centered_snr_handles_large_baseline_unlike_raw(rng: np.random.Generator) -> None:
    t = np.linspace(0.0, 6.0, 73)
    # Pure white noise around large baseline 300.0 (no turning point!)
    y_noise = 300.0 + rng.normal(0.0, 0.5, size=t.size)
    res_raw = detect_turning_points(
        t, y_noise, size=60, step=12, margin=8, snr_mode="raw", snr_th=1.0, ndri_mode="l2"
    )
    codes = {w.code for w in res_raw.diagnostics.warnings}
    assert "RAW_SNR_LARGE_BASELINE" in codes

    # Real turning point around large baseline 300.0 works cleanly with centered SNR
    y_tp = 300.0 + np.where(t <= 3.0, 0.5 * t, 1.5 - 3.0 * (t - 3.0)) + rng.normal(0.0, 0.2, size=t.size)
    res_cent = detect_turning_points(
        t, y_tp, size=60, step=12, margin=8, snr_mode="centered", snr_th=1.0
    )
    assert len(res_cent.turning_points) == 1
    assert abs(res_cent.turning_points[0].time_year - 3.0) < 0.25


def test_stpd_margin_one_does_not_crash_with_nan(rng: np.random.Generator) -> None:
    # Upstream STPD.py crashed with 0/0 NaN when default margin=1 was used
    t = np.linspace(0.0, 5.0, 61)
    y = np.where(t <= 2.5, t, 2.5 - 3.0 * (t - 2.5)) + rng.normal(0.0, 0.1, size=t.size)
    res = detect_turning_points(t, y, size=50, step=10, margin=1)
    assert any(w.code == "MARGIN_CLAMPED" for w in res.diagnostics.warnings)
    assert len(res.turning_points) == 1


def test_fit_piecewise_trend_sequential_vs_joint(rng: np.random.Generator) -> None:
    t = np.linspace(0.0, 6.0, 73)
    y = np.where(t <= 3.0, 1.0 + 2.0 * t, 7.0 - 4.0 * (t - 3.0)) + rng.normal(0.0, 0.3, size=t.size)

    r_seq = fit_piecewise_trend(t, y, [36], mode="sequential")
    r_jnt = fit_piecewise_trend(t, y, [36], mode="joint")

    assert r_seq.slopes.size == 2
    assert r_jnt.slopes.size == 2
    # Joint hinge spline is globally optimal in L2 residual norm
    assert float(np.linalg.norm(r_jnt.residuals)) <= float(np.linalg.norm(r_seq.residuals)) + 1e-10
    assert r_jnt.slopes[0] == pytest.approx(2.0, abs=0.25)
    assert r_jnt.slopes[1] == pytest.approx(-4.0, abs=0.25)
    # Continuity at TP is exact
    assert np.isclose(
        r_jnt.intercepts[0] + r_jnt.slopes[0] * t[36],
        r_jnt.intercepts[1] + r_jnt.slopes[1] * t[36],
        atol=1e-10,
    )


def test_stpd_weighted_heteroscedastic_ndri_rmse_not_inflated() -> None:
    """BLK-05: weighted NDRI_RMSE normalizes by sqrt(n_side), not sqrt(sum(w_side))."""
    rng = np.random.default_rng(20260102)
    t = np.linspace(2010.0, 2020.0, 120)
    tr = 2.0 + 0.5 * (t - 2010.0) - 1.5 * np.maximum(0.0, t - 2015.0)
    sigma = np.where(t < 2015.0, 0.2, 0.8)
    w = 1.0 / (sigma * sigma)
    y = tr + rng.normal(0.0, sigma)

    res = detect_turning_points(t, y, weights=w, alpha=0.01, band_method="conditional", n_mc=199)
    assert len(res.turning_points) == 1
    assert abs(res.turning_points[0].time_year - 2015.0) <= 0.75
    assert res.turning_points[0].ndri_rmse < 0.25
    assert len(res.flagged_jumps) == 0


def test_stpd_holm_before_merge_preserves_significant_tp() -> None:
    """CRT-01: insignificant overlapping windows cannot overwrite a significant TP."""
    rng = np.random.default_rng(77)
    t = np.linspace(2010.0, 2020.0, 120)
    # Replicates where finer stepping previously let a closer-to-center p_cal=1 window overwrite
    detected = 0
    for _ in range(10):
        y = np.where(t <= 2013.2, 0.0, 2.0 * (t - 2013.2)) + rng.normal(0.0, 0.7, size=120)
        res = detect_turning_points(
            t, y, size=48, step=6, alpha=0.01, band_method="conditional", n_mc=199
        )
        if res.turning_points:
            detected += 1
    assert detected >= 8


def test_stpd_robust_explain_and_parity_mode_guards() -> None:
    """MIN-04 & MIN-05: robust explain cites nested F-test; parity_mode skips n_mc/n_boot guard."""
    rng = np.random.default_rng(42)
    t = np.linspace(0.0, 6.0, 73)
    y = np.where(t <= 3.0, 1.0 + 1.5 * t, 5.5 - 3.0 * (t - 3.0)) + rng.normal(0.0, 0.25, size=73)

    r_rob = detect_turning_points(
        t, y, size=60, step=12, margin=8, robust="huber", band_method="conditional"
    )
    assert len(r_rob.turning_points) == 1
    assert "robust nested F-test" in r_rob.explain()

    r_par = detect_turning_points(
        t, y, size=60, step=12, margin=8, parity_mode=True, n_mc=1, n_boot=1
    )
    assert len(r_par.turning_points) == 1

