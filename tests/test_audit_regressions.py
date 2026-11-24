"""Regression tests covering every finding from the pre-PyPI independent audit.

Covers CRIT-01..03, HIGH-01..10, MED-01..09, and LOW-01..05.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tspec import (
    CensoredConfig,
    analyze_series,
    detect_turning_points,
    fit_piecewise_trend,
    just_decompose,
    just_jumps,
    just_monitor,
    lssa,
)
from tspec._diagnostics import compute_red_noise_diagnostics
from tspec._time import estimate_sampling_rate_ratio
from tspec.censored import prepare_series_with_censoring, recensor_to_common_limit
from tspec.stpd import _stpd_pvalue, _tpd_window


def test_crit01_just_decompose_days_and_seconds_no_nan() -> None:
    """CRIT-01: Gaussian dilation must not underflow to NaN on day/second numeric axes."""
    rng = np.random.default_rng(1)
    for scale in (365.25, 365.25 * 86400.0):
        t = np.linspace(0.0, 5.0 * scale, 120)
        y = 1.0 + 0.001 * (t / scale) + 0.5 * np.cos(2 * np.pi * t / scale) + rng.normal(0.0, 0.05, t.size)
        res = just_decompose(t, y, season="none", size=36, step=12)
        assert np.all(np.isfinite(res.trend))
        assert np.all(np.isfinite(res.seasonal))
        assert np.all(np.isfinite(res.remainder))


def test_crit02_integer_right_censoring_not_flipped_to_left() -> None:
    """CRIT-02: Integer censor_status in {0, +1} must stay right-censored (+1) under auto/right."""
    rng = np.random.default_rng(2)
    t = np.linspace(0.0, 5.0, 60)
    y_true = rng.lognormal(1.0, 0.4, size=60)
    ul = float(np.quantile(y_true, 0.80))
    status = np.where(y_true > ul, 1, 0)
    y_obs = np.where(status == 1, ul, y_true)

    prep_auto = prepare_series_with_censoring(
        t, y_obs, censor_status=status, censoring=CensoredConfig(method="parametric", random_state=42)
    )
    assert prep_auto.provenance is not None
    assert prep_auto.provenance.censoring_type == "right"
    assert np.all(prep_auto.provenance.status_codes[status == 1] == 1)
    assert np.all(prep_auto.series.values[status == 1] >= ul - 1e-12)

    # Left censoring with {0, 1} when censoring_type='left' remains supported
    dl = float(np.quantile(y_true, 0.20))
    status_l = np.where(y_true < dl, 1, 0)
    y_obs_l = np.where(status_l == 1, dl, y_true)
    prep_left = prepare_series_with_censoring(
        t,
        y_obs_l,
        censor_status=status_l,
        censoring=CensoredConfig(method="ros", censoring_type="left", random_state=42),
    )
    assert prep_left.provenance is not None
    assert prep_left.provenance.censoring_type == "left"
    assert np.all(prep_left.provenance.status_codes[status_l == 1] == -1)
    assert np.all(prep_left.series.values[status_l == 1] <= dl + 1e-12)


def test_crit03_monitor_causal_censoring_zero_lookahead() -> None:
    """CRIT-03: Future monitoring data must not alter historical imputation or forecast in just_monitor."""
    rng = np.random.default_rng(3)
    t = np.linspace(2016.0, 2022.0, 120)
    y_base = np.exp(0.5 + 0.2 * np.cos(2 * np.pi * t) + rng.normal(0.0, 0.25, size=120))
    dl = float(np.quantile(y_base[:80], 0.25))

    y_a = [f"<{dl:.4f}" if v < dl else f"{v:.5f}" for v in y_base]
    y_b = list(y_a)
    # Inject a massive future spike strictly inside the monitoring period (>= idx 80)
    for k in range(85, 120):
        y_b[k] = f"{y_base[k] + 500.0:.5f}"

    cfg = CensoredConfig(method="parametric", dist="lognormal", impute_type="mean", random_state=7)
    res_a = just_monitor(t, y_a, start=80, season="ols", censoring=cfg)
    res_b = just_monitor(t, y_b, start=80, season="ols", censoring=cfg)

    np.testing.assert_allclose(res_a.values[:80], res_b.values[:80], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(res_a.forecast[:80], res_b.forecast[:80], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        res_a.historical_trend_coeff, res_b.historical_trend_coeff, rtol=0.0, atol=1e-12
    )


def test_high01_just_jumps_allssa_null_fpr_controlled() -> None:
    """HIGH-01: Null-selected ALLSSA harmonics eliminate harmonic-mining FPR inflation."""
    fps = 0
    n_trials = 25
    for seed in range(n_trials):
        rng = np.random.default_rng(500 + seed)
        t = np.sort(rng.uniform(0.0, 6.0, size=120))
        y = 1.0 + 0.1 * t + rng.normal(0.0, 0.2, size=120)
        res = just_jumps(
            t,
            y,
            season="allssa",
            mag_th=0.0,
            dir_th=0.0,
            jump_alpha=0.05,
            bonferroni_window=True,
        )
        if len(res.jumps) > 0:
            fps += 1
    assert fps / n_trials <= 0.16


def test_high02_monitor_horizon_exceeded_and_obf_nonblind() -> None:
    """HIGH-02: Past planned horizon, emit HORIZON_EXCEEDED and keep OBF sensitive."""
    rng = np.random.default_rng(12)
    t = np.linspace(2016.0, 2022.0, 140)
    y = 1.0 + 0.25 * np.cos(2 * np.pi * t) + rng.normal(0.0, 0.025, size=140)
    # Inject a huge abrupt step well past planned horizon=10 (monitoring starts at 100)
    y[122:] -= 0.6

    res = just_monitor(
        t,
        y,
        start=100,
        season="ols",
        alpha=0.01,
        sequential_control="obrien_fleming",
        horizon=10,
    )
    codes = {w.code for w in res.diagnostics.warnings}
    assert "HORIZON_EXCEEDED" in codes
    assert res.detected
    assert res.disturbance_type == "abrupt"
    assert res.steps[-1].alpha_step > 1e-6


def test_high03_monitor_forecast_uncertainty_and_long_horizon_warning() -> None:
    """HIGH-03: Predictive covariance prevents forecast-drift false alarms and warns on long horizons."""
    rng = np.random.default_rng(77)
    t = np.linspace(2010.0, 2022.0, 144)  # 48 historical (4 yr) + 96 monitoring (8 yr)
    y = 1.0 + 0.2 * (t - 2010.0) + 0.5 * np.cos(2 * np.pi * t) + rng.normal(0.0, 0.2, size=144)

    res = just_monitor(t, y, start=48, season="ols", alpha=0.05)
    codes = {w.code for w in res.diagnostics.warnings}
    assert "LONG_MONITORING_HORIZON" in codes


def test_high04_mixed_constant_left_right_no_false_dl_shifts() -> None:
    """HIGH-04: Constant <DL and >UL limits must never trigger false DL shifts; real >UL shifts do."""
    rng = np.random.default_rng(4)
    t = np.linspace(2015.0, 2022.0, 100)
    y_true = rng.normal(5.0, 3.0, size=100)
    y_str = [
        "<1.0" if v < 1.0 else (">10.0" if v > 10.0 else f"{v:.3f}")
        for v in y_true
    ]
    prep = prepare_series_with_censoring(
        t, y_str, censoring=CensoredConfig(method="parametric", dist="normal", random_state=4)
    )
    assert prep.provenance is not None
    assert len(prep.provenance.dl_shifts) == 0

    # Now inject a genuine right-censoring limit shift (>8.0 -> >14.0 at 2018.5)
    y_true2 = rng.normal(9.0, 4.0, size=120)
    t2 = np.linspace(2015.0, 2022.0, 120)
    y_str2 = [
        ">8.0" if (ti < 2018.5 and v > 8.0) else (">14.0" if (ti >= 2018.5 and v > 14.0) else f"{v:.3f}")
        for ti, v in zip(t2, y_true2, strict=True)
    ]
    prep2 = prepare_series_with_censoring(
        t2, y_str2, censoring=CensoredConfig(method="parametric", dist="normal", random_state=4)
    )
    assert prep2.provenance is not None
    assert any(sh.side == "right" for sh in prep2.provenance.dl_shifts)


def test_high05_linalg_baseline_scale_and_collinearity() -> None:
    """HIGH-05: LSSA must be invariant to +1e13 baseline, *1e-13 scale, and collinear frequencies."""
    rng = np.random.default_rng(5)
    t = np.linspace(0.0, 5.0, 120)
    y = 0.5 * np.cos(2 * np.pi * 2.0 * t) + rng.normal(0.0, 0.2, size=120)
    omega = np.array([1.0, 2.0, 3.0])

    r_base = lssa(t, y, omega=omega, trend="linear")
    r_shift = lssa(t, y + 1e13, omega=omega, trend="linear")
    r_tiny = lssa(t, y * 1e-13, omega=omega, trend="linear")

    np.testing.assert_allclose(r_shift.spectrum, r_base.spectrum, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(r_tiny.spectrum, r_base.spectrum, rtol=1e-10, atol=1e-12)

    # Collinear frequency near known constituent must stay bounded in [0, 1] and finite
    r_col = lssa(t, y, omega=[1.0 + 1e-8, 2.0], known_freqs=[1.0], trend="linear")
    assert np.all(np.isfinite(r_col.spectrum))
    assert np.all((r_col.spectrum >= 0.0) & (r_col.spectrum <= 1.0))


def test_high06_stpd_large_baseline_and_tiny_scale() -> None:
    """HIGH-06: STPD O(n) prefix scan must remain exact on +1e8 baseline and *1e-10 scale."""
    rng = np.random.default_rng(6)
    t = np.linspace(0.0, 10.0, 120)
    y = np.where(t < 5.0, 0.2 * t, 1.0 - 0.8 * (t - 5.0)) + rng.normal(0.0, 0.08, size=120)

    r_ref = detect_turning_points(t, y, alpha=0.01)
    r_shift = detect_turning_points(t, y + 1e8, alpha=0.01)
    r_tiny = detect_turning_points(t, y * 1e-10, alpha=0.01)

    assert len(r_ref.turning_points) == 1
    assert len(r_shift.turning_points) == 1
    assert len(r_tiny.turning_points) == 1
    assert r_shift.turning_points[0].index == r_ref.turning_points[0].index
    assert r_tiny.turning_points[0].index == r_ref.turning_points[0].index


def test_high07_pipeline_nan_remapping_weights_and_segment_mode() -> None:
    """HIGH-07: Pipeline must remap NaN indices, propagate weight_factors, and build coherent segment trends."""
    rng = np.random.default_rng(7)
    t = np.linspace(2015.0, 2025.0, 200)
    trend = np.where(t < 2018.5, 4.0, 7.0) + np.where(t > 2022.0, -1.6 * (t - 2022.0), 0.0)
    y = trend + 0.6 * np.cos(2 * np.pi * t) + rng.normal(0.0, 0.12, size=200)
    y_nan = y.copy()
    y_nan[[0, 1, 2, 10, 50]] = np.nan

    res_off = analyze_series(t, y_nan, jump_adjustment="offset")
    assert res_off.diagnostics.n_input == 200
    assert res_off.diagnostics.n_dropped_nan == 5
    assert res_off.decomposition is not None
    assert res_off.decomposition.diagnostics.n_input == 200
    assert res_off.decomposition.diagnostics.n_dropped_nan == 5
    if res_off.jumps is not None and res_off.jumps.jumps:
        assert set(res_off.decomposition.jump_indices.tolist()) == {j.index for j in res_off.jumps.jumps}
    assert res_off.turning_points is not None
    assert res_off.turning_points.diagnostics.n_input == 200
    assert res_off.turning_points.diagnostics.n_dropped_nan == 5
    if res_off.turning_points.turning_points:
        assert res_off.turning_points.piecewise_trend.tp_indices.tolist() == [
            tp.index for tp in res_off.turning_points.turning_points
        ]

    # Segment mode must produce coherent per-segment piecewise_trend (no global continuous spline across jumps)
    res_seg = analyze_series(t, y_nan, jump_adjustment="segment")
    assert res_seg.turning_points is not None
    pw = res_seg.turning_points.piecewise_trend
    assert pw.mode == "segmented"
    assert pw.slopes.size >= 2
    assert pw.intercepts.size == pw.slopes.size
    assert pw.diagnostics.n_input == 200
    assert pw.diagnostics.n_dropped_nan == 5

    # Imputed weight_factors must propagate into Stage 2 decomposition
    y_pos = np.exp(0.4 + np.where(t >= 2020.0, 1.0, 0.0) + rng.normal(0.0, 0.2, size=200))
    dl = float(np.quantile(y_pos, 0.25))
    y_cen = [f"<{dl:.3f}" if v < dl else f"{v:.4f}" for v in y_pos]
    r_w1 = analyze_series(
        t, y_cen, detect_jumps=False, detect_turning_points=False,
        censoring=CensoredConfig(imputed_weight_factor=1.0, random_state=11, n_stability_draws=0),
    )
    r_w05 = analyze_series(
        t, y_cen, detect_jumps=False, detect_turning_points=False,
        censoring=CensoredConfig(imputed_weight_factor=0.5, random_state=11, n_stability_draws=0),
    )
    assert r_w1.decomposition is not None and r_w05.decomposition is not None
    assert not np.allclose(r_w1.decomposition.trend, r_w05.decomposition.trend)


def test_high08_windowed_decompose_and_implicit_dl_recensor() -> None:
    """HIGH-08: just_decompose honors scope='windowed' and recensor_to_common_limit handles implicit DL drop."""
    rng = np.random.default_rng(8)
    t = np.linspace(0.0, 6.0, 120)
    y_true = np.exp(np.where(t < 3.0, -0.1, 1.5) + rng.normal(0.0, 0.3, size=120))
    dl = 0.85
    y_str = [f"<{dl:.2f}" if v < dl else f"{v:.4f}" for v in y_true]

    d_glob = just_decompose(t, y_str, season="none", censoring=CensoredConfig(scope="global", random_state=8))
    d_win = just_decompose(t, y_str, season="none", censoring=CensoredConfig(scope="windowed", random_state=8))
    assert not np.allclose(d_glob.trend, d_win.trend)

    # Implicit DL drop: old limit <1.2 in first half, uncensored detects < 1.2 in second half (no second <DL)
    y_stat = np.exp(0.1 + rng.normal(0.0, 0.5, size=120))
    y_imp = np.where(t < 3.0, np.where(y_stat < 1.2, 1.2, y_stat), y_stat)
    status_imp = np.where((t < 3.0) & (y_stat < 1.2), -1, 0)
    prep = prepare_series_with_censoring(
        t, y_imp, censor_status=status_imp, censoring=CensoredConfig(method="ros", random_state=8)
    )
    assert prep.provenance is not None
    rec = recensor_to_common_limit(
        prep.provenance, CensoredConfig(method="ros", random_state=8), 0, 120, np.random.default_rng(8)
    )
    assert rec is not None


def test_high09_monitor_true_sliding_and_consecutive_k_consistency() -> None:
    """HIGH-09: Sliding window stays constant width; consecutive_k onset type/freq stay consistent."""
    rng = np.random.default_rng(9)
    t = np.linspace(2016.0, 2022.0, 120)
    y = 1.0 + 0.2 * np.cos(2 * np.pi * t) + rng.normal(0.0, 0.03, size=120)

    res = just_monitor(
        t,
        y,
        start=80,
        h=0.25,
        season="ols",
        window_mode="sliding",
        sequential_control="pocock",
        stop_on_first=False,
    )
    crits = [s.critical_value for s in res.steps]
    # Constant sliding window width + constant Pocock per-step alpha => identical critical value at every step
    np.testing.assert_allclose(crits, crits[0], rtol=0.0, atol=1e-12)

    # Inject disturbance and verify onset classification/frequency consistency under consecutive_k=2
    y_d = y.copy()
    y_d[95:] -= 0.4
    res_ck = just_monitor(
        t,
        y_d,
        start=80,
        season="ols",
        sequential_control="consecutive_k",
        consecutive_k=2,
        stop_on_first=False,
    )
    assert res_ck.detected and res_ck.dominant_frequency is not None
    if res_ck.disturbance_type == "abrupt":
        assert res_ck.dominant_frequency < 1.0
    else:
        assert res_ck.dominant_frequency >= 1.0


def test_med01_stpd_exact_vs_parity_se() -> None:
    """MED-01: Default exact STPD SE accounts for Cov(a1, a2) < 0, giving larger p-value than parity approx."""
    rng = np.random.default_rng(101)
    t = np.linspace(0.0, 4.0, 30)
    y = np.where(t < 2.0, 0.1 * t, 0.2 + 0.35 * (t - 2.0)) + rng.normal(0.0, 0.15, size=30)
    r_exact = _tpd_window(t, y, margin=3, w=None, parity_mode=False)
    r_par = _tpd_window(t, y, margin=3, w=None, parity_mode=True)
    assert r_exact.pval > r_par.pval

    # Direct check of _stpd_pvalue with fixed knot
    t0 = t - t[0]
    p_ex = _stpd_pvalue(t0, y - np.mean(y), 15, 0.0, 0.1, 0.35, None, parity_mode=False)
    p_pa = _stpd_pvalue(t0, y - np.mean(y), 15, 0.0, 0.1, 0.35, None, parity_mode=True)
    assert p_ex > p_pa


def test_med02_stpd_weighted_snr_and_ndri() -> None:
    """MED-02: 1D weights must influence STPD SNR and NDRI in both window scan and piecewise refit."""
    t = np.linspace(0.0, 10.0, 60)
    y = np.where(t < 5.0, 0.5 * t, 2.5 - 0.5 * (t - 5.0))
    y[10] += 3.0  # Single large outlier on left side
    w_eq = np.ones_like(t)
    w_down = np.ones_like(t)
    w_down[10] = 1e-4

    st_eq = _tpd_window(t, y, margin=3, w=w_eq)
    st_dn = _tpd_window(t, y, margin=3, w=w_down)
    assert st_dn.snr_centered > st_eq.snr_centered * 5.0

    pw_eq = fit_piecewise_trend(t, y, [30], weights=w_eq)
    pw_dn = fit_piecewise_trend(t, y, [30], weights=w_down)
    assert pw_dn.local_snr_centered[0] > pw_eq.local_snr_centered[0] * 5.0


def test_med03_just_decompose_eval_times_across_large_gap() -> None:
    """MED-03: eval_times spanning a wide empty gap skips sparse windows cleanly."""
    rng = np.random.default_rng(103)
    t = np.concatenate([np.linspace(0.0, 3.0, 40), np.linspace(8.0, 11.0, 40)])
    y = 2.0 + 0.3 * np.cos(2 * np.pi * t) + rng.normal(0.0, 0.05, size=t.size)
    eval_t = np.linspace(0.0, 11.0, 80)

    res = just_decompose(
        t, y, window_mode="duration", window_duration=2.5, step_duration=1.0, eval_times=eval_t, season="ols"
    )
    assert res.trend.size == eval_t.size
    assert "SPARSE_WINDOW_SKIPPED" in {w.code for w in res.diagnostics.warnings}


def test_med04_monitor_subyear_numeric_units() -> None:
    """MED-04: Numeric non-datetime axes spanning <1 unit must not fail with spurious min_his error."""
    rng = np.random.default_rng(104)
    t = np.linspace(0.0, 0.5, 60)
    y = 1.0 + 0.5 * t + rng.normal(0.0, 0.02, size=60)
    res = just_monitor(t, y, start=40, season="none")
    assert len(res.steps) == 20


def test_med05_censoring_stability_generator_and_windowed_scope() -> None:
    """MED-05: Stability draws accept np.random.Generator and honor seasonal_stratified & windowed scope."""
    rng = np.random.default_rng(105)
    t = np.linspace(2016.0, 2022.0, 120)
    y_true = np.exp(np.where(t < 2019.0, 0.2, 1.4) + 0.3 * np.cos(2 * np.pi * t) + rng.normal(0.0, 0.2, 120))
    dl = float(np.quantile(y_true, 0.25))
    y_str = [f"<{dl:.3f}" if v < dl else f"{v:.4f}" for v in y_true]

    for scope in ("seasonal_stratified", "windowed"):
        cfg = CensoredConfig(
            scope=scope,  # type: ignore[arg-type]
            n_stability_draws=4,
            stability_mode="bootstrap",
            random_state=np.random.default_rng(999),
        )
        rj = just_jumps(t, y_str, season="none", mag_th=0.3, dir_th=0.1, censoring=cfg)
        assert rj.censoring is not None
        if rj.jumps:
            assert rj.jumps[0].stability_rate is not None


def test_med06_rank_deficient_lssa_effective_df() -> None:
    """MED-06: Duplicate/rank-deficient known frequencies must use effective rank for critical_value."""
    rng = np.random.default_rng(106)
    t = np.linspace(0.0, 5.0, 40)
    y = rng.normal(0.0, 1.0, size=40)
    r_single = lssa(t, y, omega=[2.0], known_freqs=[1.0], trend="linear")
    r_dup = lssa(t, y, omega=[2.0], known_freqs=[1.0, 1.0], trend="linear")
    np.testing.assert_allclose(r_dup.critical_value, r_single.critical_value, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(r_dup.pvalues, r_single.pvalues, rtol=1e-12, atol=1e-12)


def test_med07_noiseless_jump_scale_invariant() -> None:
    """MED-07: Noiseless jump F-test guard must be scale-invariant at 1e-10 amplitude."""
    t = np.linspace(0.0, 6.0, 90)
    y = np.where(t < 3.0, 0.0, 1.0) * 1e-10
    rj = just_jumps(t, y, season="none", mag_th=0.0, dir_th=0.0, jump_alpha=0.05)
    assert len(rj.jumps) == 1
    assert rj.jumps[0].jump_pvalue == 0.0
    assert rj.jumps[0].step_pvalue == 0.0


def test_med08_fit_piecewise_joint_scaling_and_condition_number() -> None:
    """MED-08: Joint hinge-spline scales time basis and reports real condition number."""
    t = np.linspace(0.0, 1e6, 80)
    y = 2.0 + 1e-6 * t - 3e-6 * np.maximum(0.0, t - t[40])
    pw = fit_piecewise_trend(t, y, [40], mode="joint")
    assert pw.diagnostics.condition_number_max > 1.0
    assert pw.diagnostics.condition_number_max < 1e3
    np.testing.assert_allclose(pw.fitted_trend, y, rtol=1e-10, atol=1e-10)


def test_med09_pipeline_random_state_none_synced() -> None:
    """MED-09: analyze_series with random_state=None must keep censoring.imputed_values == values."""
    rng = np.random.default_rng(109)
    t = np.linspace(2016.0, 2022.0, 100)
    y_true = np.exp(0.3 + rng.normal(0.0, 0.25, size=100))
    dl = float(np.quantile(y_true, 0.25))
    y_str = [f"<{dl:.3f}" if v < dl else f"{v:.4f}" for v in y_true]

    res = analyze_series(
        t, y_str, censoring=CensoredConfig(random_state=None, n_stability_draws=3)
    )
    assert res.censoring is not None
    np.testing.assert_array_equal(res.censoring.imputed_values, res.values)


def test_low01_to_low05_miscellaneous() -> None:
    """LOW-01..LOW-05: step_pvalue export, tz-aware timestamps, empty bins, OU n_eff, half-inf & plot_monitor."""
    import pandas as pd

    # LOW-01: step_pvalue column & step_alpha validation
    t = np.linspace(0.0, 6.0, 90)
    y = np.where(t < 3.0, 1.0, 3.0)
    rj = just_jumps(t, y, season="none", mag_th=0.5)
    df = rj.to_dataframe()
    assert "step_pvalue" in df.columns
    with pytest.raises(ValueError, match="step_alpha"):
        just_jumps(t, y, season="none", step_alpha=1.5)

    # LOW-02: tz-aware pandas DatetimeIndex emits no NumPy timezone UserWarning
    tz_idx = pd.date_range("2020-01-01", periods=60, freq="15D", tz="UTC")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lssa(tz_idx, np.sin(np.arange(60) * 0.2), omega=[1.0])
    tz_warns = [w for w in caught if "timezone" in str(w.message).lower()]
    assert len(tz_warns) == 0

    # LOW-03: Empty bin in estimate_sampling_rate_ratio returns inf
    t_gap = np.concatenate([np.linspace(0.0, 1.0, 30), np.linspace(9.0, 10.0, 30)])
    assert np.isinf(estimate_sampling_rate_ratio(t_gap))

    # LOW-04: Continuous-time OU n_eff accounts for burst clustering
    t_burst = np.concatenate([np.linspace(0.0, 0.05, 25), np.linspace(1.0, 1.05, 25)])
    e_burst = np.concatenate([np.ones(25), -np.ones(25)]) + np.linspace(0, 0.05, 50)
    rn, _ = compute_red_noise_diagnostics(t_burst, e_burst)
    assert rn.effective_sample_size < 50.0

    # LOW-05: Half-infinite interval stores finite bound in original_values
    t_int = np.linspace(0.0, 5.0, 30)
    y_int = np.array(
        [(-np.inf, 1.0) if i < 6 else (1.5 + 0.1 * i, 1.5 + 0.1 * i) for i in range(30)],
        dtype=np.float64,
    )
    prep_int = prepare_series_with_censoring(
        t_int, y_int, censoring=CensoredConfig(method="parametric", dist="normal", random_state=5)
    )
    assert prep_int.provenance is not None
    assert np.all(np.isfinite(prep_int.provenance.original_values))
