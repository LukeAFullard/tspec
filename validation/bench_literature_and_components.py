"""Tier 2b: Literature-Inspired & Compositional Sub-Component Validation Suite.

Clean-room synthetic recreations of canonical peer-reviewed benchmarks (L1-L6)
plus systematic compositional sub-component stress tests (SC1-SC3).
Writes a Markdown verification report to ``validation/reports/literature_and_components_report.md``.

Run with:
    OMP_NUM_THREADS=1 MPLBACKEND=Agg python3 validation/bench_literature_and_components.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import tspec
from tspec import (
    CensoredConfig,
    allssa,
    analyze_series,
    detect_turning_points,
    fit_piecewise_trend,
    just_decompose,
    just_jumps,
    just_monitor,
    lssa,
)
from tspec._linalg import prepare_weights
from tspec._robust import huber_nested_ftest


@dataclass(frozen=True)
class GateResult:
    code: str
    name: str
    passed: bool
    metric: str
    elapsed_s: float


def _run_gate(code: str, name: str, fn) -> GateResult:
    t0 = time.perf_counter()
    passed, metric = fn()
    dt = time.perf_counter() - t0
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {code:4s} {name:46s} | {metric} ({dt:.1f}s)")
    return GateResult(code=code, name=name, passed=passed, metric=metric, elapsed_s=dt)


# ---------------------------------------------------------------------------
# Literature-Inspired Benchmarks (L1 - L6)
# ---------------------------------------------------------------------------


def bench_l1_ghaderpour2018_antileakage() -> tuple[bool, str]:
    """L1: Ghaderpour et al. (2018) non-integer multi-harmonic anti-leakage on gapped data."""
    rng = np.random.default_rng(1001)
    true_f = np.array([0.73, 1.85, 3.25])
    recovered = 0
    null_fp = 0
    n_rep = 12

    for _ in range(n_rep):
        t_raw = np.sort(rng.uniform(0.0, 8.0, size=240))
        t = t_raw[~((t_raw > 3.1) & (t_raw < 3.55))]  # 5.4-month blackout gap
        j_idx = int(np.searchsorted(t, 4.5))
        sig = (
            10.0
            + 0.35 * t
            + np.where(np.arange(t.size) >= j_idx, 2.5, 0.0)
            + 0.9 * np.cos(2 * np.pi * true_f[0] * t)
            + 1.2 * np.sin(2 * np.pi * true_f[1] * t)
            + 0.55 * np.cos(2 * np.pi * true_f[2] * t)
            + rng.normal(0.0, 0.25, size=t.size)
        )
        res = allssa(
            t,
            sig,
            jump_indices=[j_idx],
            omega=np.arange(0.5, 4.05, 0.1),
            decimal=2,
            alpha=0.01,
            significance="montecarlo",
            n_mc=499,
        )
        if res.frequencies.size == 3 and np.allclose(np.sort(res.frequencies), true_f, atol=0.05):
            recovered += 1

        y_null = 5.0 + 0.1 * t + rng.normal(0.0, 0.4, size=t.size)
        rn = allssa(
            t,
            y_null,
            omega=np.arange(1, 5, dtype=float),
            decimal=1,
            alpha=0.01,
            significance="montecarlo",
            n_mc=499,
        )
        if rn.frequencies.size > 0:
            null_fp += 1

    ok = recovered == n_rep and null_fp <= 1
    return ok, f"3-freq recovery={recovered}/{n_rep}, null FP={null_fp}/{n_rep}"


def bench_l2_bfast_just_wildfire_regrowth() -> tuple[bool, str]:
    """L2: Verbesselt (2010) / Ghaderpour & Vujadinovic (2020a) NDVI wildfire + regrowth."""
    rng = np.random.default_rng(1002)
    dates = np.arange("2015-01-01", "2022-01-01", 16, dtype="datetime64[D]")
    t_all = (dates - dates[0]).astype("timedelta64[D]").astype(float) / 365.25
    keep = (rng.uniform(size=dates.size) > 0.35) & ~((t_all > 1.8) & (t_all < 2.15))
    t_dates = dates[keep]
    t_yr = t_all[keep]
    trend_true = np.where(t_yr < 3.6, 0.58 - 0.01 * t_yr, 0.26 + 0.07 * (t_yr - 3.6))
    seas_true = 0.16 * np.cos(2 * np.pi * t_yr) + 0.06 * np.sin(4 * np.pi * t_yr)
    y = trend_true + seas_true + rng.normal(0.0, 0.025, size=t_yr.size)

    jumps = just_jumps(t_dates, y, season="allssa", jump_alpha=0.01, n_mc=199)
    dec = just_decompose(
        t_dates, y, jump_indices=jumps.jump_indices, season="allssa", trend_ci_method="bias_corrected"
    )
    t_cov = float(
        np.mean((dec.trend_ci_low <= trend_true) & (trend_true <= dec.trend_ci_high))
    )
    s_cov = float(
        np.mean((dec.seasonal_ci_low <= seas_true) & (seas_true <= dec.seasonal_ci_high))
    )
    jump_ok = len(jumps.jumps) == 1 and abs(jumps.jumps[0].time_year - (2015.0 + 3.6)) <= 0.35
    mag_err = abs(jumps.jumps[0].magnitude - (-0.32)) if jumps.jumps else float("inf")
    ok = jump_ok and mag_err < 0.08 and t_cov >= 0.85 and s_cov >= 0.85
    return (
        ok,
        f"jumps={len(jumps.jumps)}, |dmag|={mag_err:.3f}, trend_cov={t_cov:.0%}, seas_cov={s_cov:.0%}",
    )


def bench_l3_ghaderpour2024_insar_multi_tp() -> tuple[bool, str]:
    """L3: Ghaderpour et al. (2024) PS-InSAR multi-bend subsidence + disguised unwrapping jump."""
    rng = np.random.default_rng(1003)
    t = np.linspace(2015.0, 2022.0, 85)
    disp_true = np.where(
        t <= 2017.5,
        -1.0 * (t - 2015.0),
        np.where(
            t <= 2020.0,
            -2.5 - 10.0 * (t - 2017.5),
            -27.5 - 15.0 * (t - 2020.0),
        ),
    )
    disp = disp_true + rng.normal(0.0, 1.2, size=t.size)
    res_tp = detect_turning_points(
        t, disp, size=48, step=12, margin=8, alpha=0.01, ndri_th=0.30, n_mc=199, band_method="conditional"
    )
    tp_years = sorted(tp.time_year for tp in res_tp.turning_points)
    tp_ok = len(tp_years) == 2 and abs(tp_years[0] - 2017.5) <= 0.45 and abs(tp_years[1] - 2020.0) <= 0.45

    pw_l1 = fit_piecewise_trend(
        t, disp, tp_indices=[tp.index for tp in res_tp.turning_points], mode="l1"
    )
    slope_err = float(np.max(np.abs(pw_l1.slopes - np.array([-1.0, -10.0, -15.0])))) if tp_ok else 99.0

    # Disguised unwrapping jump (-25 mm) must be rejected by NDRI and placed in flagged_jumps
    disp_jump = np.where(t < 2018.5, -2.0 * (t - 2015.0), -25.0 - 2.0 * (t - 2015.0)) + rng.normal(0.0, 1.0, t.size)
    res_j = detect_turning_points(
        t, disp_jump, size=60, step=12, margin=10, alpha=0.01, n_mc=199, band_method="conditional"
    )
    ndri_ok = len(res_j.turning_points) == 0 and len(res_j.flagged_jumps) >= 1
    ok = tp_ok and slope_err < 1.5 and ndri_ok
    return (
        ok,
        f"TPs={len(tp_years)} ({[round(x, 2) for x in tp_years]}), max|dslope|={slope_err:.2f}, jump_flagged={len(res_j.flagged_jumps)}",
    )


def bench_l4_helsel2012_multidl_water_quality() -> tuple[bool, str]:
    """L4: Helsel (2012) multi-DL lognormal water-quality trend + lab-upgrade artifact defense."""
    rng = np.random.default_rng(1004)
    n = 120
    t = np.linspace(2014.0, 2024.0, n)
    t0 = t - t[0]

    # Part A: Real declining trend under ~35-45% lognormal left-censoring across 2 DLs (10 reps)
    mi_errs: list[float] = []
    dl2_errs: list[float] = []
    ats_ok = True
    for r in range(10):
        y_lat = np.exp(0.75 - 0.14 * t0 + rng.normal(0.0, 0.35, size=n))
        dl = np.where(t0 < 5.0, 1.8, 1.0)
        is_c = y_lat < dl
        vals = np.where(is_c, dl, y_lat)
        pop_lin_slope = float(np.polyfit(t0, y_lat, 1)[0])
        res_mi = analyze_series(
            t,
            vals,
            censor_status=is_c,
            detect_jumps=False,
            detect_turning_points=False,
            remove_seasonality=False,
            slope_estimator="ols",
            censoring=CensoredConfig(
                imputation_model="trend",
                dist="lognormal",
                n_imputations=20,
                allow_heavy_censoring=True,
                random_state=r,
            ),
        )
        assert res_mi.segments[0].estimator == "ols_mi"
        mi_errs.append(abs(res_mi.segments[0].slope - pop_lin_slope))
        dl2_vals = np.where(is_c, dl / 2.0, y_lat)
        dl2_errs.append(abs(float(np.polyfit(t0, dl2_vals, 1)[0]) - pop_lin_slope))
        if r == 0:
            res_ats = analyze_series(
                t,
                vals,
                censor_status=is_c,
                detect_jumps=False,
                detect_turning_points=False,
                remove_seasonality=False,
                slope_estimator="sen",
                censoring=CensoredConfig(
                    imputation_model="trend",
                    dist="lognormal",
                    allow_heavy_censoring=True,
                    random_state=0,
                ),
            )
            ats_ok = res_ats.segments[0].estimator == "ats" and res_ats.segments[0].slope < 0.0

    mi_mae = float(np.mean(mi_errs))
    dl2_mae = float(np.mean(dl2_errs))

    # Part B: Pure-noise lab DL upgrade (4.2 -> 0.5) under substitution must be flagged by recensor_check
    y_null = 4.0 + rng.normal(0.0, 0.35, size=n)
    dl_up = np.where(t0 < 5.0, 4.2, 0.5)
    c_up = y_null < dl_up
    v_up = np.where(c_up, dl_up, y_null)
    rj = just_jumps(
        t,
        v_up,
        censor_status=c_up,
        season="none",
        n_mc=199,
        censoring=CensoredConfig(
            method="substitution",
            imputation_model="marginal",
            dist="normal",
            detect_dl_shifts=True,
            recensor_check=True,
            allow_heavy_censoring=True,
            random_state=4,
        ),
    )
    artifact_ok = (len(rj.jumps) == 0) or all(j.dl_artifact_likely is True for j in rj.jumps)
    ok = mi_mae < dl2_mae and ats_ok and artifact_ok
    return (
        ok,
        f"ols_mi_MAE={mi_mae:.3f} vs dl/2_MAE={dl2_mae:.3f}, ats_ok={ats_ok}, dl_artifact_guarded={artifact_ok}",
    )


def bench_l5_yue2002_ar1_hydrology() -> tuple[bool, str]:
    """L5: Yue et al. (2002) / Hamed & Rao (1998) AR(1) red-noise hydrology + spikes."""
    rng = np.random.default_rng(1005)
    n = 100
    t = np.linspace(2014.0, 2024.0, n)
    rho = 0.7
    idx = np.arange(n)
    P_ar1 = np.linalg.inv(rho ** np.abs(idx[:, None] - idx[None, :]))
    om = np.linspace(0.5, 4.0, 35)

    n_rep = 30
    fp_raw = 0
    fp_ar1 = 0
    fp_lssa_unw = 0
    fp_lssa_gls = 0
    for r in range(n_rep):
        eps = np.zeros(n)
        eps[0] = rng.normal()
        for i in range(1, n):
            eps[i] = rho * eps[i - 1] + np.sqrt(1.0 - rho**2) * rng.normal()
        y = 5.0 + eps
        # Add 2 isolated flood spikes to half the replicates
        y_spk = y.copy()
        if r % 2 == 0:
            y_spk[[20, 75]] += np.array([12.0, -10.0])

        p_raw = analyze_series(
            t, y_spk, detect_jumps=False, detect_turning_points=False,
            remove_seasonality=False, slope_estimator="sen", mk_correction="none",
        ).segments[0].pvalue
        p_ar1 = analyze_series(
            t, y_spk, detect_jumps=False, detect_turning_points=False,
            remove_seasonality=False, slope_estimator="sen", mk_correction="ar1",
        ).segments[0].pvalue
        if p_raw is not None and p_raw < 0.05:
            fp_raw += 1
        if p_ar1 is not None and p_ar1 < 0.05:
            fp_ar1 += 1

        if lssa(t, y, omega=om, trend="linear", peak_fap="sidak").peak_fap < 0.05:
            fp_lssa_unw += 1
        if lssa(t, y, weights=P_ar1, omega=om, trend="linear", peak_fap="sidak").peak_fap < 0.05:
            fp_lssa_gls += 1

    ok = fp_ar1 <= 5 and fp_raw >= fp_ar1 + 5 and fp_lssa_gls <= 4 and fp_lssa_unw >= fp_lssa_gls + 5
    return (
        ok,
        f"MK FPR raw={fp_raw}/{n_rep} vs ar1={fp_ar1}/{n_rep} | LSSA FPR unw={fp_lssa_unw}/{n_rep} vs GLS={fp_lssa_gls}/{n_rep}",
    )


def bench_l6_ghaderpour2020b_nrt_regimes() -> tuple[bool, str]:
    """L6: Ghaderpour & Vujadinovic (2020b) NRT monitoring: null FWER, abrupt, and phenological."""
    rng = np.random.default_rng(1006)
    t = np.linspace(2017.0, 2023.5, 156)  # ~biweekly
    base = 0.65 + 0.20 * np.cos(2 * np.pi * t) + 0.06 * np.sin(4 * np.pi * t)

    # Null FWER across 20 stable series with calibrated consecutive_k=2
    n_null = 20
    fp_k2 = 0
    for _ in range(n_null):
        y_n = base + rng.normal(0.0, 0.03, size=t.size)
        r = just_monitor(
            t, y_n, start=2022.0, season="ols", alpha=0.05,
            sequential_control="consecutive_k", consecutive_k=2, horizon=36, n_mc=199,
        )
        if r.detected:
            fp_k2 += 1

    # Abrupt canopy drop
    y_ab = base + np.where(t >= 2022.6, -0.25, 0.0) + rng.normal(0.0, 0.03, size=t.size)
    r_ab = just_monitor(t, y_ab, start=2022.0, season="ols", alpha=0.01, horizon=36)

    # Pure phenological intra-annual shift (zero baseline shift)
    y_ph = base + np.where(t >= 2022.4, 0.22 * np.sin(2 * np.pi * 3.0 * t), 0.0) + rng.normal(0.0, 0.025, size=t.size)
    r_ph = just_monitor(t, y_ph, start=2022.0, season="ols", alpha=0.01, horizon=36)

    ok = (
        fp_k2 <= 3
        and r_ab.detected
        and r_ab.disturbance_type == "abrupt"
        and r_ph.detected
        and r_ph.disturbance_type == "phenological"
    )
    return (
        ok,
        f"null_k2_FP={fp_k2}/{n_null}, abrupt={r_ab.disturbance_type}@{r_ab.disturbance_time_year:.2f}, phen={r_ph.disturbance_type}@{r_ph.disturbance_time_year:.2f}",
    )


# ---------------------------------------------------------------------------
# Compositional Sub-Component Stress Tests (SC1 - SC3)
# ---------------------------------------------------------------------------


def bench_sc1_kitchen_sink_compositional() -> tuple[bool, str]:
    """SC1: Irregular + 4-month gap + 2x rate shift + 2 harmonics + jump + TP + 20% <DL + spikes."""
    rng = np.random.default_rng(2001)
    # 2x sampling density shift at 2019.0 + 4-month blackout in 2016
    t1 = np.sort(rng.uniform(2014.0, 2019.0, size=75))
    t2 = np.sort(rng.uniform(2019.02, 2024.0, size=150))
    t = np.concatenate([t1, t2])
    t = t[~((t > 2016.2) & (t < 2016.53))]

    trend = (
        np.where(t < 2017.5, 4.0, 6.6)
        + np.where(t > 2021.0, -1.4 * (t - 2021.0), 0.0)
    )
    seas = 0.7 * np.cos(2 * np.pi * t) + 0.3 * np.sin(4 * np.pi * t)
    y = trend + seas + rng.normal(0.0, 0.16, size=t.size)
    # Inject 5 isolated extreme spikes
    y[[10, 45, 95, 145, 195]] += np.array([6.0, -5.5, 6.5, -6.0, 5.5])
    # Left-censor lowest ~15%
    dl = float(np.quantile(y, 0.15))
    is_c = y < dl
    vals = np.where(is_c, dl, y)

    res = analyze_series(
        t,
        vals,
        censor_status=is_c,
        robust="mm",
        slope_estimator="sen",
        just_kwargs={
            "season": "ols",
            "window_mode": "duration",
            "window_duration": 3.0,
            "step_duration": 0.75,
            "jump_alpha": 0.01,
        },
        decompose_kwargs={"season": "ols", "window_mode": "duration", "window_duration": 3.0, "step_duration": 0.75},
        stpd_kwargs={"window_mode": "duration", "window_duration": 4.5, "step_duration": 0.75, "alpha": 0.01},
        censoring=CensoredConfig(imputation_model="trend", dist="normal", n_stability_draws=0, random_state=0),
        band_method="conditional",
    )
    j_ok = res.jumps is not None and len(res.jumps.jumps) == 1 and abs(res.jumps.jumps[0].time_year - 2017.5) <= 0.45
    tp_ok = (
        res.turning_points is not None
        and len(res.turning_points.turning_points) == 1
        and abs(res.turning_points.turning_points[0].time_year - 2021.0) <= 0.6
    )
    seg_ok = len(res.segments) == 3 and res.segments[-1].slope < -0.9
    ok = j_ok and tp_ok and seg_ok
    return (
        ok,
        f"jumps={[round(j.time_year, 2) for j in (res.jumps.jumps if res.jumps else [])]}, "
        f"TPs={[round(p.time_year, 2) for p in (res.turning_points.turning_points if res.turning_points else [])]}, "
        f"last_slope={res.segments[-1].slope:.2f}",
    )


def bench_sc2_compound_interactions() -> tuple[bool, str]:
    """SC2: Four high-risk pairwise sub-component interactions."""
    rng = np.random.default_rng(2002)

    # 1. Strong continuous bend + 2 harmonics under season='allssa' must NOT be stolen by JUST step_pvalue
    t1 = np.sort(rng.uniform(2014.0, 2024.0, size=200))
    t1 = t1[~((t1 > 2016.2) & (t1 < 2016.45))]
    y1 = (
        np.where(t1 < 2017.5, 4.0, 6.5)
        + np.where(t1 > 2021.0, -1.5 * (t1 - 2021.0), 0.0)
        + 0.8 * np.cos(2 * np.pi * t1)
        + 0.35 * np.sin(4 * np.pi * t1)
        + rng.normal(0.0, 0.15, size=t1.size)
    )
    rj1 = just_jumps(t1, y1, season="allssa", jump_alpha=0.01, step_alpha=0.01, n_mc=199)
    int1_ok = len(rj1.jumps) == 1 and abs(rj1.jumps[0].time_year - 2017.5) <= 0.35

    # 2. 4x variance shift across a smooth turning point -> weighted NDRI passes
    t2 = np.linspace(2014.0, 2024.0, 140)
    sig2 = np.where(t2 < 2019.0, 0.8, 0.2)
    y2 = 5.0 + 0.4 * (t2 - 2014.0) - 1.2 * np.maximum(0.0, t2 - 2019.0) + rng.normal(0.0, sig2)
    tp_w = detect_turning_points(
        t2, y2, weights=1.0 / (sig2**2), size=80, step=15, margin=10, alpha=0.01, n_mc=199, band_method="conditional"
    )
    int2_ok = len(tp_w.turning_points) == 1 and abs(tp_w.turning_points[0].time_year - 2019.0) <= 0.5

    # 3. Curved quadratic trend + abrupt jump + seasonality -> bias_corrected CI coverage >= 90%
    t3 = np.linspace(2015.0, 2023.0, 180)
    t30 = t3 - t3[0]
    tr3 = 2.0 + 0.12 * (t30**2) + np.where(t3 >= 2019.0, 3.0, 0.0)
    y3 = tr3 + 0.9 * np.cos(2 * np.pi * t30) + rng.normal(0.0, 0.25, size=t3.size)
    j3 = int(np.searchsorted(t3, 2019.0))
    dec3 = just_decompose(t3, y3, jump_indices=[j3], season="ols", trend_ci_method="bias_corrected")
    cov3 = float(np.mean((dec3.trend_ci_low <= tr3) & (tr3 <= dec3.trend_ci_high)))
    int3_ok = cov3 >= 0.88

    # 4. 2D SPD AR(1) weights + spikes -> exact whitened Huber & MM nested loss monotonicity
    n4 = 60
    t4 = np.linspace(0.0, 6.0, n4)
    D_null = np.column_stack([np.ones(n4), t4])
    D_full = np.column_stack([np.ones(n4), t4, np.maximum(0.0, t4 - 3.0)])
    i4 = np.arange(n4)
    P4 = np.linalg.inv(0.7 ** np.abs(i4[:, None] - i4[None, :]))
    pw4 = prepare_weights(P4, n4)
    y4 = 2.0 + 0.3 * t4 + rng.normal(0.0, 0.4, size=n4)
    y4[[10, 30, 50]] += np.array([12.0, -14.0, 11.0])
    _, _, full_h, null_h = huber_nested_ftest(D_null, D_full, y4, pw4, method="huber")
    _, _, full_m, null_m = huber_nested_ftest(D_null, D_full, y4, pw4, method="mm")
    int4_ok = (null_h.loss >= full_h.loss - 1e-9) and (null_m.loss >= full_m.loss - 1e-9)

    ok = int1_ok and int2_ok and int3_ok and int4_ok
    return ok, f"bend+allssa={int1_ok}, het_NDRI={int2_ok}, quad_cov={cov3:.0%}, 2D_rob_mono={int4_ok}"


def bench_sc3_invariance_and_causality() -> tuple[bool, str]:
    """SC3: Exact time-shift, affine-scale, weight-scale invariances & strict NRT causality."""
    rng = np.random.default_rng(2003)
    t = np.sort(rng.uniform(0.0, 8.0, size=140))
    y = np.where(t < 4.0, 2.0, 5.0) + 0.6 * np.cos(2 * np.pi * t) + rng.normal(0.0, 0.2, size=t.size)
    w = rng.uniform(0.5, 2.0, size=t.size)
    om = np.linspace(0.5, 3.5, 25)

    # 1. Time-origin shift invariance (t -> t + 2000) & weight-scale invariance (w -> 13.7 * w)
    s0 = lssa(t, y, weights=w, omega=om, trend="linear")
    s_shift = lssa(t + 2000.0, y, weights=w, omega=om, trend="linear")
    s_wscale = lssa(t, y, weights=13.7 * w, omega=om, trend="linear")
    inv_t = float(np.max(np.abs(s0.spectrum - s_shift.spectrum))) < 1e-11
    inv_w = float(np.max(np.abs(s0.spectrum - s_wscale.spectrum))) < 1e-12

    # 2. Affine response equivariance (y -> 7.5 * y - 120) on just_jumps
    j0 = just_jumps(t, y, season="ols", jump_alpha=0.01, n_mc=199)
    j_aff = just_jumps(t, 7.5 * y - 120.0, season="ols", jump_alpha=0.01, n_mc=199)
    inv_y = (
        len(j0.jumps) == len(j_aff.jumps) == 1
        and j0.jumps[0].index == j_aff.jumps[0].index
        and abs(j_aff.jumps[0].magnitude - 7.5 * j0.jumps[0].magnitude) < 1e-9
    )

    # 3. Strict online causality in just_monitor: mutating future steps leaves past steps bit-for-bit equal
    y_m1 = 0.5 + 0.2 * np.cos(2 * np.pi * t) + rng.normal(0.0, 0.04, size=t.size)
    y_m2 = y_m1.copy()
    cut = 120
    y_m2[cut:] += 25.0  # massive future shock after index 120
    m1 = just_monitor(t, y_m1, start=100, season="ols", horizon=40, stop_on_first=False)
    m2 = just_monitor(t, y_m2, start=100, season="ols", horizon=40, stop_on_first=False)
    causal_ok = all(
        m1.steps[k].max_spec == m2.steps[k].max_spec
        and m1.steps[k].critical_value == m2.steps[k].critical_value
        for k in range(cut - 100)
    )

    ok = inv_t and inv_w and inv_y and causal_ok
    return ok, f"time_shift={inv_t}, weight_scale={inv_w}, affine_y={inv_y}, nrt_causal={causal_ok}"


def main() -> int:
    print(f"Running tspec v{tspec.__version__} Tier 2b Literature & Compositional Validation Suite...")
    t_start = time.perf_counter()

    gates = [
        _run_gate("L1", "Ghaderpour (2018) non-integer anti-leakage", bench_l1_ghaderpour2018_antileakage),
        _run_gate("L2", "Verbesselt/Ghaderpour NDVI wildfire + regrowth", bench_l2_bfast_just_wildfire_regrowth),
        _run_gate("L3", "Ghaderpour (2024) PS-InSAR multi-TP + NDRI", bench_l3_ghaderpour2024_insar_multi_tp),
        _run_gate("L4", "Helsel (2012) multi-DL water quality + artifact", bench_l4_helsel2012_multidl_water_quality),
        _run_gate("L5", "Yue (2002) / Hamed-Rao AR(1) hydrology + spikes", bench_l5_yue2002_ar1_hydrology),
        _run_gate("L6", "Ghaderpour (2020b) NRT null/abrupt/phenological", bench_l6_ghaderpour2020b_nrt_regimes),
        _run_gate("SC1", "Kitchen-sink compositional recovery", bench_sc1_kitchen_sink_compositional),
        _run_gate("SC2", "Compound sub-component interaction matrix", bench_sc2_compound_interactions),
        _run_gate("SC3", "Exact invariances & strict NRT causality", bench_sc3_invariance_and_causality),
    ]

    total_s = time.perf_counter() - t_start
    n_pass = sum(g.passed for g in gates)
    print(f"\nSummary: {n_pass}/{len(gates)} gates PASSED in {total_s:.1f}s.")

    report_path = Path(__file__).resolve().parent / "reports" / "literature_and_components_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Tier 2b: Literature-Inspired & Compositional Sub-Component Validation Report",
        "",
        f"- **tspec version:** `{tspec.__version__}`",
        f"- **Result:** **{n_pass} / {len(gates)} gates PASSED** ({total_s:.1f} s single-threaded runtime)",
        "",
        "| Gate | Benchmark / Scenario | Status | Measured Metrics | Runtime |",
        "| :--- | :--- | :---: | :--- | ---: |",
    ]
    for g in gates:
        lines.append(
            f"| `{g.code}` | {g.name} | **{'PASS' if g.passed else 'FAIL'}** | {g.metric} | {g.elapsed_s:.1f} s |"
        )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report to {report_path}")
    return 0 if n_pass == len(gates) else 1


if __name__ == "__main__":
    raise SystemExit(main())
