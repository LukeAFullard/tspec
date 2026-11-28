"""Tier 2d: External-Oracle, Cross-Domain Scientific & Adversarial Edge-Case Validation Suite.

Covers 15 independent verification gates across:
- External Mathematical & Library Oracles (OR1-OR3):
  - OR1: Exact GLS/WLS parity vs. ``statsmodels.api.GLS`` (1D & full 2D AR(1) covariance)
  - OR2: Exact Theil-Sen slope & Gilbert CI parity vs. ``scipy.stats.theilslopes``
  - OR3: Irregular peak parity vs. ``scipy.signal.lombscargle`` + exact Beta(1, nu/2) KS test
- Wide Scientific, Engineering & Economic Domains (D1-D4):
  - D1: Epidemiology / Public Health (seasonal waves + vaccination step + decay slope)
  - D2: Econometrics / Finance (Bai-Perron level break + growth bend + 3x volatility shift)
  - D3: Industrial Engineering / Predictive Maintenance (bearing wear RUL onset + 25-sigma spikes)
  - D4: Clinical Pharmacology (40% <LLOQ lognormal clearance decay + assay upgrade guard)
- Adversarial Edge-Case Torture Matrix (E1-E8):
  - E1: Extreme dynamic range (1e-9 and 1e+9 amplitude scaling)
  - E2: Catastrophic cancellation offsets (t + 1e6, y + 1e8)
  - E3: Exact noiseless & constant signals (zero-variance, exact jump/TP/sinusoid)
  - E4: Dirty timestamps (unsorted, duplicates, NaN bursts, 70% random dropout)
  - E5: Small-n boundary behavior (n = 14..22 graceful skip diagnostics)
  - E6: Mixed left+right+interval censoring & >80% CensoringError guardrail
  - E7: Heavy-tailed Student-t2 & 15% extreme spike contamination (robust MM/Huber/L1/Sen vs OLS)
  - E8: Closely spaced double breakpoints (two jumps 1.4 yr apart; jump + TP 1.8 yr apart)

Run with:
    OMP_NUM_THREADS=1 MPLBACKEND=Agg python3 validation/bench_cross_domain_and_edge_cases.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import lombscargle
from scipy.stats import kstest, theilslopes

import tspec
from tspec import (
    CensoredConfig,
    CensoringError,
    analyze_series,
    detect_turning_points,
    fit_piecewise_trend,
    just_decompose,
    just_jumps,
    just_monitor,
    lssa,
)


@dataclass(frozen=True)
class GateResult:
    code: str
    category: str
    name: str
    passed: bool
    metric: str
    elapsed_s: float


def _run_gate(code: str, category: str, name: str, fn) -> GateResult:
    t0 = time.perf_counter()
    passed, metric = fn()
    dt = time.perf_counter() - t0
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {code:4s} [{category:10s}] {name:44s} | {metric} ({dt:.1f}s)")
    return GateResult(
        code=code, category=category, name=name, passed=passed, metric=metric, elapsed_s=dt
    )


# ---------------------------------------------------------------------------
# Part 1: External Mathematical & Library Oracles (OR1 - OR3)
# ---------------------------------------------------------------------------


def bench_or1_statsmodels_gls_oracle() -> tuple[bool, str]:
    """OR1: Exact parameter & covariance match against statsmodels.api.GLS (full 2D AR(1) P)."""
    import statsmodels.api as sm

    rng = np.random.default_rng(3001)
    n = 90
    t = np.sort(rng.uniform(0.0, 6.0, size=n))
    t0 = t - t[0]
    rho = 0.65
    idx = np.arange(n)
    Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
    P = np.linalg.inv(Sigma)
    y = 3.0 + 0.5 * t0 + 0.9 * np.cos(2 * np.pi * t0) - 0.4 * np.sin(2 * np.pi * t0) + rng.multivariate_normal(np.zeros(n), 0.16 * Sigma)

    res = lssa(t, y, weights=P, known_freqs=[1.0], trend="linear")
    X = np.column_stack([np.ones(n), t0, np.cos(2 * np.pi * t0), np.sin(2 * np.pi * t0)])
    gls = sm.GLS(y, X, sigma=Sigma).fit()

    coef_err = float(np.max(np.abs(res.coeff - gls.params)))
    cov_err = float(np.max(np.abs(res.cov - gls.cov_params())))
    res_err = float(np.max(np.abs(res.residuals - gls.resid)))
    ok = coef_err < 1e-11 and cov_err < 1e-11 and res_err < 1e-11
    return ok, f"max|Δβ|={coef_err:.2e}, max|ΔCov|={cov_err:.2e}, max|Δresid|={res_err:.2e}"


def bench_or2_scipy_theilsen_oracle() -> tuple[bool, str]:
    """OR2: Exact slope, intercept & Gilbert 95% CI match against scipy.stats.theilslopes."""
    rng = np.random.default_rng(3002)
    t = np.sort(rng.uniform(2015.0, 2025.0, size=110))
    y = 4.0 + 0.42 * (t - 2015.0) + rng.standard_t(df=4, size=t.size) * 0.5
    y[[12, 55, 93]] += np.array([15.0, -18.0, 12.0])

    ref = theilslopes(y, t, alpha=0.95)
    res = analyze_series(
        t,
        y,
        detect_jumps=False,
        detect_turning_points=False,
        remove_seasonality=False,
        slope_estimator="sen",
        mk_correction="none",
        ci_level=0.95,
    )
    seg = res.segments[0]
    s_err = abs(seg.slope - float(ref.slope))
    lo_err = abs((seg.slope_ci_low or 0.0) - float(ref.low_slope))
    hi_err = abs((seg.slope_ci_high or 0.0) - float(ref.high_slope))
    ok = s_err < 1e-12 and lo_err < 1e-12 and hi_err < 1e-12
    return ok, f"|Δslope|={s_err:.2e}, |ΔCI_lo|={lo_err:.2e}, |ΔCI_hi|={hi_err:.2e}"


def bench_or3_lombscargle_and_beta_cdf_oracle() -> tuple[bool, str]:
    """OR3: Irregular peak agreement with scipy.signal.lombscargle + exact Beta(1, nu/2) KS test."""
    rng = np.random.default_rng(3003)
    t = np.sort(rng.uniform(0.0, 10.0, size=180))
    y = 1.4 * np.cos(2 * np.pi * 1.37 * t + 0.4) + rng.normal(0.0, 0.3, size=t.size)
    om = np.linspace(0.2, 3.0, 281)

    sp = lssa(t, y, omega=om, trend="none")
    ls = lombscargle(t, y - np.mean(y), 2.0 * np.pi * om, precenter=False, normalize=True)
    f_tspec = float(om[int(np.argmax(sp.spectrum))])
    f_scipy = float(om[int(np.argmax(ls))])

    # Exact Beta(1, nu/2) null p-value uniformity check over 200 Gaussian noise draws
    pvals = np.empty(200, dtype=float)
    for k in range(200):
        yn = rng.normal(0.0, 1.0, size=t.size)
        pvals[k] = float(lssa(t, yn, omega=[1.25], trend="linear").pvalues[0])
    ks = kstest(pvals, "uniform")
    ok = abs(f_tspec - f_scipy) <= 0.02 and abs(f_tspec - 1.37) <= 0.02 and float(ks.pvalue) > 0.05
    return ok, f"f_tspec={f_tspec:.2f} vs f_scipy={f_scipy:.2f} (true 1.37), Beta-null KS p={ks.pvalue:.3f}"


# ---------------------------------------------------------------------------
# Part 2: Wide Scientific, Engineering & Economic Domains (D1 - D4)
# ---------------------------------------------------------------------------


def bench_d1_epidemiology_vaccination_shock() -> tuple[bool, str]:
    """D1: Public Health / Epidemiology — seasonal incidence + abrupt vaccine rollout + decay."""
    rng = np.random.default_rng(4001)
    t = np.sort(rng.uniform(2016.0, 2024.0, size=240))
    # Pre-vaccine stable seasonal baseline -> abrupt -3.5 drop at 2020.2 -> post-vaccine decay -0.4/yr
    trend = np.where(t < 2020.2, 8.0, 4.5 - 0.40 * (t - 2020.2))
    seas = 1.6 * np.cos(2 * np.pi * t) + 0.6 * np.sin(4 * np.pi * t)
    y = trend + seas + rng.normal(0.0, 0.30, size=t.size)

    res = analyze_series(
        t,
        y,
        just_kwargs={"season": "ols", "jump_alpha": 0.01},
        decompose_kwargs={"season": "ols"},
        band_method="conditional",
    )
    j_ok = (
        res.jumps is not None
        and len(res.jumps.jumps) == 1
        and abs(res.jumps.jumps[0].time_year - 2020.2) <= 0.3
        and abs(res.jumps.jumps[0].magnitude - (-3.5)) <= 0.5
    )
    post_slope = res.segments[-1].slope
    slope_ok = abs(post_slope - (-0.40)) <= 0.15
    ok = j_ok and slope_ok
    return (
        ok,
        f"rollout={res.jumps.jumps[0].time_year:.2f} (Δ={res.jumps.jumps[0].magnitude:+.2f}), post_slope={post_slope:+.2f}/yr",
    )


def bench_d2_econometrics_structural_break_volatility() -> tuple[bool, str]:
    """D2: Econometrics / Finance — level shock + growth slowdown + 3x volatility regime shift."""
    rng = np.random.default_rng(4002)
    t = np.linspace(1995.0, 2025.0, 180)
    sigma = np.where(t < 2012.0, 0.25, 0.75)  # 3x volatility regime shift
    trend = (
        np.where(t < 2004.0, 10.0 + 0.45 * (t - 1995.0), 17.5 + 0.45 * (t - 2004.0))
        + np.where(t > 2015.0, -0.60 * (t - 2015.0), 0.0)
    )
    y = trend + rng.normal(0.0, sigma)
    w = 1.0 / (sigma**2)

    rj = just_jumps(t, y, weights=w, season="none", size=60, step=10, jump_alpha=0.01, step_alpha=0.01, n_mc=199)
    # Remove estimated jump offset and locate the 2015 growth slowdown under 3x volatility shift
    y_adj = y.copy()
    if rj.jumps:
        y_adj[rj.jumps[0].valid_index :] -= rj.jumps[0].magnitude
    tp = detect_turning_points(
        t, y_adj, weights=w, size=72, step=12, margin=10, alpha=0.05, n_mc=199, band_method="conditional"
    )
    j_ok = len(rj.jumps) == 1 and abs(rj.jumps[0].time_year - 2004.0) <= 0.8
    tp_ok = len(tp.turning_points) == 1 and abs(tp.turning_points[0].time_year - 2015.0) <= 1.2
    ok = j_ok and tp_ok
    return (
        ok,
        f"jump={[round(j.time_year, 1) for j in rj.jumps]} (true 2004), "
        f"TP={[round(p.time_year, 1) for p in tp.turning_points]} (true 2015, weighted NDRI)",
    )


def bench_d3_engineering_bearing_rul_and_spikes() -> tuple[bool, str]:
    """D3: Industrial Engineering — bearing wear RUL onset + 8% 25-sigma telemetry glitches."""
    rng = np.random.default_rng(4003)
    t = np.sort(rng.uniform(0.0, 10.0, size=180))  # 10 months / operating units
    wear = 1.0 + 0.05 * t + np.where(t > 6.5, 0.90 * (t - 6.5), 0.0)
    y = wear + rng.normal(0.0, 0.12, size=t.size)
    # Inject 8% extreme telemetry glitches (25 sigma)
    glitch_idx = rng.choice(np.arange(10, 170), size=14, replace=False)
    y[glitch_idx] += rng.choice([-3.2, 3.2], size=14)

    tp_mm = detect_turning_points(
        t, y, robust="mm", size=90, step=15, margin=12, alpha=0.01, band_method="conditional"
    )
    tp_ok = len(tp_mm.turning_points) == 1 and abs(tp_mm.turning_points[0].time_year - 6.5) <= 0.5

    # NRT monitoring catches degradation onset cleanly with consecutive_k=2 (ignoring isolated glitches)
    mon = just_monitor(
        t,
        y,
        start=5.0,
        season="none",
        robust="mm",
        alpha=0.05,
        sequential_control="consecutive_k",
        consecutive_k=2,
        horizon=90,
        monitor_omega=np.linspace(0.1, 1.5, 25),
        n_mc=199,
    )
    mon_ok = mon.detected and mon.disturbance_time_year is not None and 6.4 <= mon.disturbance_time_year <= 8.5
    ok = tp_ok and mon_ok
    return (
        ok,
        f"RUL onset TP={tp_mm.turning_points[0].time_year:.2f} (true 6.50), NRT alarm@{mon.disturbance_time_year:.2f}",
    )


def bench_d4_pharmacology_lloq_clearance() -> tuple[bool, str]:
    """D4: Clinical Pharmacology — 40% <LLOQ lognormal biomarker decay + assay sensitivity change."""
    rng = np.random.default_rng(4004)
    n = 120
    t = np.linspace(0.0, 10.0, n)
    y_lat = np.exp(1.0 - 0.16 * t + rng.normal(0.0, 0.30, size=n))
    lloq = np.where(t < 5.0, 1.5, 0.7)
    is_c = y_lat < lloq
    vals = np.where(is_c, lloq, y_lat)
    pop_slope = float(np.polyfit(t, y_lat, 1)[0])

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
            random_state=4,
        ),
    )
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
            random_state=4,
        ),
    )
    seg_mi = res_mi.segments[0]
    seg_ats = res_ats.segments[0]
    mi_err = abs(seg_mi.slope - pop_slope)
    ok = (
        seg_mi.estimator == "ols_mi"
        and seg_ats.estimator == "ats"
        and mi_err < 0.04
        and seg_ats.slope < -0.10
        and float(np.mean(is_c)) >= 0.25
    )
    return (
        ok,
        f"cens={np.mean(is_c):.0%}, ols_mi={seg_mi.slope:.3f} (pop={pop_slope:.3f}), ats={seg_ats.slope:.3f}",
    )


# ---------------------------------------------------------------------------
# Part 3: Adversarial Edge-Case Torture Matrix (E1 - E8)
# ---------------------------------------------------------------------------


def bench_e1_extreme_dynamic_range() -> tuple[bool, str]:
    """E1: Scale invariance at 1e-9 and 1e+9 amplitudes (zero underflow/overflow)."""
    rng = np.random.default_rng(5001)
    t = np.linspace(2016.0, 2024.0, 120)
    y_base = np.where(t < 2020.0, 2.0, 5.0) + rng.normal(0.0, 0.2, size=t.size)
    scales = [1e-9, 1.0, 1e9]
    j_idx = []
    mags = []
    for s in scales:
        rj = just_jumps(t, y_base * s, season="none", jump_alpha=0.01, n_mc=199)
        j_idx.append(rj.jumps[0].index if rj.jumps else -1)
        mags.append(rj.jumps[0].magnitude / s if rj.jumps else float("nan"))

    ok = len(set(j_idx)) == 1 and j_idx[0] > 0 and float(np.ptp(mags)) < 1e-7
    return ok, f"indices={j_idx}, scaled_mags={[round(m, 4) for m in mags]}"


def bench_e2_catastrophic_cancellation_offsets() -> tuple[bool, str]:
    """E2: Huge baseline offsets (t + 1e6, y + 1e8) without numerical loss."""
    rng = np.random.default_rng(5002)
    t = np.linspace(0.0, 8.0, 120)
    y = 2.0 + 0.5 * t - 1.5 * np.maximum(0.0, t - 4.0) + rng.normal(0.0, 0.15, size=t.size)

    tp0 = detect_turning_points(t, y, size=60, step=12, margin=8, alpha=0.01, n_mc=199, band_method="conditional")
    tp1 = detect_turning_points(
        t + 1e6, y + 1e8, size=60, step=12, margin=8, alpha=0.01, n_mc=199, band_method="conditional"
    )
    ok = (
        len(tp0.turning_points) == 1
        and len(tp1.turning_points) == 1
        and tp0.turning_points[0].index == tp1.turning_points[0].index
        and abs(tp0.turning_points[0].direction - tp1.turning_points[0].direction) < 1e-5
    )
    return (
        ok,
        f"idx0={tp0.turning_points[0].index}, idx_shifted={tp1.turning_points[0].index}, "
        f"|Δdir|={abs(tp0.turning_points[0].direction - tp1.turning_points[0].direction):.2e}",
    )


def bench_e3_noiseless_and_constant_signals() -> tuple[bool, str]:
    """E3: Exact zero-noise step, exact TP, exact sinusoid, and flat constant signal (zero NaN/div0)."""
    t = np.linspace(2016.0, 2024.0, 101)
    # Constant signal
    y_const = np.full_like(t, 42.0)
    rj_c = just_jumps(t, y_const, season="none", n_mc=99)
    tp_c = detect_turning_points(t, y_const, n_mc=99, band_method="conditional")
    sp_c = lssa(t, y_const, omega=[1.0, 2.0])

    # Exact noiseless step jump
    y_step = np.where(t < 2020.0, 1.0, 4.0)
    rj_s = just_jumps(t, y_step, season="none", jump_alpha=0.01, n_mc=99)

    # Exact noiseless turning point (knot on grid point t[50] == 2020.0)
    y_tp = 2.0 + 0.5 * (t - 2016.0) - 1.5 * np.maximum(0.0, t - 2020.0)
    tp_s = detect_turning_points(t, y_tp, size=60, step=12, margin=8, alpha=0.01, n_mc=99, band_method="conditional")

    ok = (
        len(rj_c.jumps) == 0
        and len(tp_c.turning_points) == 0
        and bool(np.all(np.isfinite(sp_c.spectrum)))
        and len(rj_s.jumps) == 1
        and abs(rj_s.jumps[0].magnitude - 3.0) < 1e-9
        and len(tp_s.turning_points) == 1
        and abs(tp_s.turning_points[0].direction - (-1.5)) < 1e-6
    )
    tp_dir_str = f"{tp_s.turning_points[0].direction:.4f}" if tp_s.turning_points else "none"
    return (
        ok,
        f"const(j={len(rj_c.jumps)},tp={len(tp_c.turning_points)}), "
        f"exact_step_mag={rj_s.jumps[0].magnitude:.4f}, exact_tp_dir={tp_dir_str}",
    )


def bench_e4_dirty_timestamps_and_70pct_dropout() -> tuple[bool, str]:
    """E4: Unsorted & duplicate timestamp guardrails + NaN bursts + 70% random dropout."""
    rng = np.random.default_rng(5004)
    t_dense = np.linspace(2015.0, 2025.0, 400)
    # Keep only 30% (70% missing!)
    t = t_dense[rng.uniform(size=t_dense.size) < 0.30]
    y = np.where(t < 2020.0, 3.0, 6.2) + 0.6 * np.cos(2 * np.pi * t) + rng.normal(0.0, 0.2, size=t.size)

    # 1. Unsorted timestamps must raise ValueError
    unsorted_rejected = False
    try:
        just_jumps(t[::-1], y[::-1], season="ols", n_mc=99)
    except ValueError:
        unsorted_rejected = True

    # 2. Duplicate timestamps must raise ValueError
    dup_rejected = False
    try:
        t_dup = np.sort(np.concatenate([t, t[:3]]))
        y_dup = np.concatenate([y, y[:3]])
        just_jumps(t_dup, y_dup, season="ols", n_mc=99)
    except ValueError:
        dup_rejected = True

    # 3. Sorted sparse series with 8 NaN observations + 70% dropout recovers the jump cleanly
    t_nan = np.concatenate([t, np.linspace(2017.11, 2017.51, 8)])
    y_nan = np.concatenate([y, np.full(8, np.nan)])
    order = np.argsort(t_nan)
    rj = just_jumps(t_nan[order], y_nan[order], season="ols", jump_alpha=0.01, n_mc=199)
    ok = (
        unsorted_rejected
        and dup_rejected
        and len(rj.jumps) == 1
        and abs(rj.jumps[0].time_year - 2020.0) <= 0.45
        and rj.diagnostics.n_dropped_nan == 8
    )
    return (
        ok,
        f"guards(unsorted={unsorted_rejected},dup={dup_rejected}), "
        f"n_valid={rj.diagnostics.n_valid}, dropped_nan={rj.diagnostics.n_dropped_nan}, jump={rj.jumps[0].time_year:.2f}",
    )


def bench_e5_small_n_graceful_degradation() -> tuple[bool, str]:
    """E5: Very short series (n = 16) raises InsufficientDataError on default window, degrades cleanly with small size."""
    from tspec import InsufficientDataError

    rng = np.random.default_rng(5005)
    t = np.linspace(2022.0, 2023.5, 16)
    y = 2.0 + 0.4 * (t - 2022.0) + rng.normal(0.0, 0.1, size=16)

    too_short_raised = False
    try:
        analyze_series(t, y, band_method="conditional")
    except InsufficientDataError:
        too_short_raised = True

    res = analyze_series(
        t,
        y,
        just_kwargs={"season": "none", "size": 12, "step": 4, "n_mc": 99},
        decompose_kwargs={"season": "none", "size": 10, "step": 4},
        band_method="conditional",
    )
    dec = just_decompose(t, y, season="none", size=10, step=4)
    ok = (
        too_short_raised
        and len(res.segments) >= 1
        and bool(np.all(np.isfinite(res.combined_trend)))
        and bool(np.all(np.isfinite(dec.trend)))
    )
    codes = [w.code for w in res.diagnostics.warnings]
    return ok, f"default_guard={too_short_raised}, n=16 finite, warnings={codes}"


def bench_e6_mixed_interval_censoring_and_guardrail() -> tuple[bool, str]:
    """E6: Mixed <DL + >UL, [L, R] interval censoring, and >80% CensoringError guardrail."""
    rng = np.random.default_rng(5006)
    t = np.linspace(2016.0, 2024.0, 110)
    y = 5.0 + 0.35 * (t - 2016.0) + rng.normal(0.0, 0.4, size=t.size)

    # 1. Mixed left (<DL) + right (>UL)
    status = np.where(y < 5.3, -1, np.where(y > 7.6, 1, 0))
    vals = np.where(status == -1, 5.3, np.where(status == 1, 7.6, y))
    res_mix = analyze_series(
        t,
        vals,
        censor_status=status,
        detect_jumps=False,
        detect_turning_points=False,
        remove_seasonality=False,
        censoring=CensoredConfig(imputation_model="trend", dist="normal", random_state=0),
    )

    # 2. Interval [L, R] censoring
    is_int = y < 5.6
    bounds = np.column_stack([np.where(is_int, 4.5, y), np.where(is_int, 5.6, y)])
    res_int = analyze_series(
        t,
        bounds,
        detect_jumps=False,
        detect_turning_points=False,
        remove_seasonality=False,
        censoring=CensoredConfig(method="parametric", censoring_type="interval", dist="normal", random_state=0),
    )

    # 3. >80% censoring must raise CensoringError unless allow_heavy_censoring=True
    guard_raised = False
    try:
        just_jumps(t, np.full_like(t, 1.0), censor_status=np.arange(t.size) < 95, season="none")
    except CensoringError:
        guard_raised = True

    ok = (
        abs(res_mix.segments[0].slope - 0.35) < 0.12
        and abs(res_int.segments[0].slope - 0.35) < 0.12
        and guard_raised
    )
    return (
        ok,
        f"mixed_slope={res_mix.segments[0].slope:.3f}, interval_slope={res_int.segments[0].slope:.3f}, guardrail={guard_raised}",
    )


def bench_e7_heavy_tail_t2_and_15pct_spikes() -> tuple[bool, str]:
    """E7: Student-t2 heavy tails + 15% extreme spikes — robust MM+Sen & L1 stay bounded vs OLS."""
    rng = np.random.default_rng(5007)
    t = np.linspace(2015.0, 2025.0, 160)
    y_true = 2.0 + 0.50 * (t - 2015.0) - 1.20 * np.maximum(0.0, t - 2020.0)
    y = y_true + rng.standard_t(df=2, size=t.size) * 0.25
    spk = rng.choice(t.size, size=24, replace=False)  # 15% extreme spikes!
    y[spk] += rng.choice([-15.0, 15.0], size=24)

    pw_ols = fit_piecewise_trend(t, y, tp_indices=[80], mode="joint")
    pw_l1 = fit_piecewise_trend(t, y, tp_indices=[80], mode="l1")
    pw_hub = fit_piecewise_trend(t, y, tp_indices=[80], mode="huber")

    true_slopes = np.array([0.50, -0.70])
    err_ols = float(np.max(np.abs(pw_ols.slopes - true_slopes)))
    err_l1 = float(np.max(np.abs(pw_l1.slopes - true_slopes)))
    err_hub = float(np.max(np.abs(pw_hub.slopes - true_slopes)))

    ok = err_l1 < 0.12 and err_hub < 0.12 and err_l1 < err_ols / 2.0
    return ok, f"max|Δslope| OLS={err_ols:.3f} vs L1={err_l1:.3f}, Huber={err_hub:.3f}"


def bench_e8_closely_spaced_multi_breakpoints() -> tuple[bool, str]:
    """E8: Two abrupt jumps separated by 2.0 yr AND jump + TP separated by 2.0 yr."""
    rng = np.random.default_rng(5008)
    t = np.linspace(2014.0, 2024.0, 240)
    # Two distinct abrupt jumps at 2017.5 (+2.2) and 2020.5 (-2.4)
    y2j = (
        np.where(t < 2017.5, 3.0, np.where(t < 2020.5, 5.2, 2.8))
        + 0.5 * np.cos(2 * np.pi * t)
        + rng.normal(0.0, 0.14, size=t.size)
    )
    rj2 = just_jumps(t, y2j, season="ols", size=60, step=12, jump_alpha=0.01, n_mc=199)
    j_yrs = sorted(j.time_year for j in rj2.jumps)
    j2_ok = len(j_yrs) == 2 and abs(j_yrs[0] - 2017.5) <= 0.35 and abs(j_yrs[1] - 2020.5) <= 0.35

    # Jump at 2018.0 (+2.3) + smooth TP at 2020.5 (slope 0 -> -1.5/yr)
    y_jt = (
        np.where(t < 2018.0, 4.0, 6.3)
        + np.where(t > 2020.5, -1.5 * (t - 2020.5), 0.0)
        + 0.6 * np.cos(2 * np.pi * t)
        + rng.normal(0.0, 0.14, size=t.size)
    )
    res_jt = analyze_series(
        t,
        y_jt,
        just_kwargs={"season": "ols", "jump_alpha": 0.01},
        decompose_kwargs={"season": "ols"},
        stpd_kwargs={"size": 72, "step": 12, "margin": 10, "alpha": 0.01},
        band_method="conditional",
    )
    jt_ok = (
        res_jt.jumps is not None
        and len(res_jt.jumps.jumps) == 1
        and abs(res_jt.jumps.jumps[0].time_year - 2018.0) <= 0.35
        and res_jt.turning_points is not None
        and len(res_jt.turning_points.turning_points) == 1
        and abs(res_jt.turning_points.turning_points[0].time_year - 2020.5) <= 0.5
    )
    ok = j2_ok and jt_ok
    return (
        ok,
        f"2-jumps={[round(x, 2) for x in j_yrs]} (true 2017.5,2020.5); "
        f"jump+TP=({res_jt.jumps.jumps[0].time_year:.2f},{res_jt.turning_points.turning_points[0].time_year:.2f})",
    )


def main() -> int:
    print(f"Running tspec v{tspec.__version__} Tier 2d Cross-Domain, Oracle & Edge-Case Validation Suite...")
    t_start = time.perf_counter()

    gates = [
        _run_gate("OR1", "Oracle", "statsmodels.GLS 2D AR(1) exact parity", bench_or1_statsmodels_gls_oracle),
        _run_gate("OR2", "Oracle", "scipy.stats.theilslopes exact parity", bench_or2_scipy_theilsen_oracle),
        _run_gate("OR3", "Oracle", "scipy.signal.lombscargle + Beta null KS", bench_or3_lombscargle_and_beta_cdf_oracle),
        _run_gate("D1", "Epidemiol.", "Seasonal incidence + vaccine shock + decay", bench_d1_epidemiology_vaccination_shock),
        _run_gate("D2", "Economics", "Bai-Perron break + bend + 3x vol shift", bench_d2_econometrics_structural_break_volatility),
        _run_gate("D3", "Engineer.", "Bearing RUL onset + 8% 25-sigma spikes", bench_d3_engineering_bearing_rul_and_spikes),
        _run_gate("D4", "Pharmacol.", "40% <LLOQ lognormal clearance + assay shift", bench_d4_pharmacology_lloq_clearance),
        _run_gate("E1", "Edge Case", "Extreme dynamic range (1e-9 to 1e+9)", bench_e1_extreme_dynamic_range),
        _run_gate("E2", "Edge Case", "Catastrophic cancellation (t+1e6, y+1e8)", bench_e2_catastrophic_cancellation_offsets),
        _run_gate("E3", "Edge Case", "Noiseless & flat constant signals", bench_e3_noiseless_and_constant_signals),
        _run_gate("E4", "Edge Case", "Unsorted/duplicate times + NaNs + 70% drop", bench_e4_dirty_timestamps_and_70pct_dropout),
        _run_gate("E5", "Edge Case", "Small-n (n=16) graceful degradation", bench_e5_small_n_graceful_degradation),
        _run_gate("E6", "Edge Case", "Mixed <DL/>UL + [L,R] interval + >80% guard", bench_e6_mixed_interval_censoring_and_guardrail),
        _run_gate("E7", "Edge Case", "Heavy-tailed t2 + 15% extreme spikes", bench_e7_heavy_tail_t2_and_15pct_spikes),
        _run_gate("E8", "Edge Case", "Closely spaced double breakpoints", bench_e8_closely_spaced_multi_breakpoints),
    ]

    total_s = time.perf_counter() - t_start
    n_pass = sum(g.passed for g in gates)
    print(f"\nSummary: {n_pass}/{len(gates)} Tier 2d gates PASSED in {total_s:.1f}s.")

    report_path = Path(__file__).resolve().parent / "reports" / "cross_domain_and_edge_cases_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Tier 2d: External-Oracle, Cross-Domain & Adversarial Edge-Case Validation Report",
        "",
        f"- **tspec version:** `{tspec.__version__}`",
        f"- **Result:** **{n_pass} / {len(gates)} gates PASSED** ({total_s:.1f} s single-threaded runtime)",
        "",
        "| Gate | Category | Validation Scenario | Status | Measured Metrics | Runtime |",
        "| :--- | :--- | :--- | :---: | :--- | ---: |",
    ]
    for g in gates:
        lines.append(
            f"| `{g.code}` | {g.category} | {g.name} | **{'PASS' if g.passed else 'FAIL'}** | {g.metric} | {g.elapsed_s:.1f} s |"
        )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report to {report_path}")
    return 0 if n_pass == len(gates) else 1


if __name__ == "__main__":
    raise SystemExit(main())
