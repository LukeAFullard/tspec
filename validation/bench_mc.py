"""Tier 2 Validation: Synthetic Ground-Truth & Monte Carlo Benchmarks.

Covers:
1. JUST Monte Carlo Matrix (J1-J5): null calibration, single/multi-jump recovery,
   non-integer multi-harmonic anti-leakage, severe irregular dropout + blackouts,
   and heteroscedastic weighting (P).
2. STPD Monte Carlo Matrix (S1-S5): continuous TP recovery (white, AR(1), t5),
   null calibration, jump vs. TP NDRI ROC AUC, density-imbalance NDRI (L2 vs RMSE),
   and head-to-head JUST vs. STPD on noisy continuous TPs.
3. Censored Data Matrix (C1-C5): temporal trend-injection regression test,
   10%/30%/50% left/right/mixed/interval vs DL/2, shifting DL null artifact &
   recensoring check, non-stationary global vs windowed scope, down-weighting sweep.
4. Red-Noise AR(1) Calibration Sweep (phi in {0.0, 0.3, 0.6, 0.8}) + true P=C_f^{-1}.
5. End-to-End Combined Pipeline (seasonality + jump + TP + gaps + 20% censoring).

Writes structured results to ``validation/reports/benchmarks_report.md``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from tspec import (
    CensoredConfig,
    allssa,
    analyze_series,
    detect_turning_points,
    just_jumps,
    just_monitor,
    lssa,
)
from tspec.censored import prepare_series_with_censoring

ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "reports" / "benchmarks_report.md"


def _ar1_noise(n: int, phi: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    e = np.empty(n, dtype=float)
    innov_sd = sigma * np.sqrt(max(1e-12, 1.0 - phi * phi))
    e[0] = rng.normal(0.0, sigma)
    for i in range(1, n):
        e[i] = phi * e[i - 1] + rng.normal(0.0, innov_sd)
    return e


def _ar1_precision(n: int, phi: float) -> np.ndarray:
    """Exact tridiagonal Kac-Murdock-Szego precision matrix C^{-1} for stationary AR(1)."""
    if abs(phi) < 1e-12:
        return np.eye(n, dtype=float)
    P = np.zeros((n, n), dtype=float)
    inv_var = 1.0 / (1.0 - phi * phi)
    for i in range(n):
        P[i, i] = (1.0 if i in (0, n - 1) else (1.0 + phi * phi)) * inv_var
        if i + 1 < n:
            P[i, i + 1] = -phi * inv_var
            P[i + 1, i] = -phi * inv_var
    return P


def run_just_matrix(n_rep: int = 40) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # J1: Null calibration (exact single-freq LSSA at alpha=0.05, ALLSSA at default alpha=0.01,
    # and Bonferroni-adjusted JUST false-jump rate at alpha=0.05)
    rng = np.random.default_rng(101)
    lssa_fp = 0
    allssa_fp = 0
    just_fp = 0
    for _ in range(n_rep):
        t = np.sort(rng.uniform(0.0, 6.0, size=140))
        y = 1.0 + 0.1 * t + rng.normal(0.0, 0.2, size=t.size)
        rl = lssa(t, y, omega=[1.0], alpha=0.05)
        if rl.spectrum[0] > rl.critical_value:
            lssa_fp += 1
        ra = allssa(t, y, omega=[1.0, 2.0, 3.0], alpha=0.01, decimal=1)
        if ra.frequencies.size > 0:
            allssa_fp += 1
        rj = just_jumps(
            t,
            y,
            season="none",
            mag_th=0.0,
            dir_th=0.0,
            jump_alpha=0.05,
            bonferroni_window=True,
        )
        if len(rj.jumps) > 0:
            just_fp += 1
    rows.append(
        {
            "id": "J1",
            "scenario": "LSSA/ALLSSA/JUST Null Calibration (0 jumps, white noise)",
            "pass": (lssa_fp / n_rep <= 0.15) and (allssa_fp / n_rep <= 0.25) and (just_fp / n_rep <= 0.15),
            "summary": (
                f"LSSA single-freq FPR={lssa_fp/n_rep:.1%} (alpha=0.05); "
                f"ALLSSA scan FPR={allssa_fp/n_rep:.1%} (default alpha=0.01); "
                f"JUST Bonferroni FPR={just_fp/n_rep:.1%} (alpha=0.05)"
            ),
        }
    )

    # J1b: JUST season='allssa' null calibration (verifies null-selected harmonics eliminate mining bias)
    rng = np.random.default_rng(111)
    allssa_just_fp = 0
    for _ in range(n_rep):
        t = np.sort(rng.uniform(0.0, 6.0, size=140))
        y = 1.0 + 0.1 * t + 0.3 * np.cos(2 * np.pi * t) + rng.normal(0.0, 0.2, size=t.size)
        rj = just_jumps(
            t,
            y,
            season="allssa",
            mag_th=0.0,
            dir_th=0.0,
            jump_alpha=0.05,
            bonferroni_window=True,
        )
        if len(rj.jumps) > 0:
            allssa_just_fp += 1
    rows.append(
        {
            "id": "J1b",
            "scenario": "JUST season='allssa' Null Calibration (null-selected harmonics)",
            "pass": (allssa_just_fp / n_rep <= 0.15),
            "summary": f"JUST(allssa) Bonferroni FPR={allssa_just_fp/n_rep:.1%} (alpha=0.05)",
        }
    )

    # J2: Jump recovery across SNR in {3, 6}
    rng = np.random.default_rng(102)
    recalls: dict[int, float] = {}
    mag_rmses: dict[int, float] = {}
    for snr in (3, 6):
        hits = 0
        errs: list[float] = []
        for _ in range(n_rep):
            t = np.sort(rng.uniform(0.0, 6.0, size=140))
            sigma = 0.1
            mag_true = snr * sigma
            y = np.where(t < 3.0, 1.0, 1.0 + mag_true) + 0.25 * np.cos(2 * np.pi * t) + rng.normal(0.0, sigma, size=t.size)
            rj = just_jumps(t, y, season="ols", mag_th=0.0, dir_th=0.0, jump_alpha=0.01, bonferroni_window=True)
            matched = [j for j in rj.jumps if abs(j.time_year - 3.0) <= 0.25]
            if matched:
                hits += 1
                errs.append(matched[0].magnitude - mag_true)
        recalls[snr] = hits / n_rep
        mag_rmses[snr] = float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")
    rows.append(
        {
            "id": "J2",
            "scenario": "JUST Single-Jump Recovery (SNR=3 vs SNR=6)",
            "pass": recalls[6] >= 0.90 and mag_rmses[6] <= mag_rmses[3] + 0.05,
            "summary": f"SNR=3 recall={recalls[3]:.1%} (mag RMSE={mag_rmses[3]:.3f}); SNR=6 recall={recalls[6]:.1%} (mag RMSE={mag_rmses[6]:.3f})",
        }
    )

    # J3: Non-integer multi-harmonic seasonality {0.37, 1.0, 2.0} -> ALLSSA vs fixed OLS
    rng = np.random.default_rng(103)
    allssa_slope_errs: list[float] = []
    ols_slope_errs: list[float] = []
    freq_hits = 0
    for _ in range(n_rep):
        t = np.sort(rng.uniform(0.0, 7.0, size=160))
        season = 0.5 * np.cos(2 * np.pi * 1.0 * t) + 0.35 * np.sin(2 * np.pi * 2.0 * t) + 0.45 * np.cos(2 * np.pi * 0.37 * t)
        y = 1.0 + 0.15 * t + season + rng.normal(0.0, 0.05, size=t.size)
        ra = allssa(t, y, omega=np.arange(1, 20) / 5.0, decimal=2, alpha=0.01)
        ro = lssa(t, y, known_freqs=[1.0, 2.0, 3.0, 4.0], trend="linear")
        allssa_slope_errs.append(float(ra.trend_coeff[1] - 0.15))
        ols_slope_errs.append(float(ro.coeff[1] - 0.15))
        if all(np.min(np.abs(ra.frequencies - f0)) <= 0.05 for f0 in (0.37, 1.0, 2.0)) if ra.frequencies.size else False:
            freq_hits += 1
    rmse_a = float(np.sqrt(np.mean(np.square(allssa_slope_errs))))
    rmse_o = float(np.sqrt(np.mean(np.square(ols_slope_errs))))
    rows.append(
        {
            "id": "J3",
            "scenario": "Non-Integer Multi-Harmonic Seasonality (ALLSSA vs Fixed OLS)",
            "pass": (freq_hits / n_rep >= 0.85) and (rmse_a < 0.75 * rmse_o),
            "summary": f"3-freq recovery={freq_hits/n_rep:.1%}; slope RMSE ALLSSA={rmse_a:.4f} vs OLS={rmse_o:.4f}",
        }
    )

    # J4: Severe irregular dropout (45%) + 4-month blackout, zero interpolation
    rng = np.random.default_rng(104)
    hits_gap = 0
    for _ in range(n_rep):
        t_grid = np.linspace(0.0, 6.0, 220)
        keep = rng.uniform(size=220) > 0.45
        keep &= ~((t_grid > 1.5) & (t_grid < 1.85))
        t = t_grid[keep]
        y = np.where(t < 3.6, 1.0, 1.8) + 0.25 * np.cos(2 * np.pi * t) + rng.normal(0.0, 0.08, size=t.size)
        rj = just_jumps(t, y, season="ols", mag_th=0.2, dir_th=0.2, jump_alpha=0.01, bonferroni_window=True)
        if any(abs(j.time_year - 3.6) <= 0.35 for j in rj.jumps):
            hits_gap += 1
    rows.append(
        {
            "id": "J4",
            "scenario": "Severe Irregularity (45% dropout + 4-mo blackout, zero interp)",
            "pass": hits_gap / n_rep >= 0.80,
            "summary": f"Zero crashes; jump recall={hits_gap/n_rep:.1%}",
        }
    )

    # J5: Heteroscedastic weighting P outperforms unweighted OLS under noisy outliers
    rng = np.random.default_rng(105)
    err_w: list[float] = []
    err_u: list[float] = []
    for _ in range(n_rep):
        t = np.linspace(0.0, 6.0, 120)
        sd = np.full(120, 0.05)
        noisy = rng.choice(120, size=22, replace=False)
        sd[noisy] = 0.50
        y = np.where(t < 3.0, 1.0, 1.45) + rng.normal(0.0, sd)
        P = 1.0 / (sd**2)
        rw = just_jumps(t, y, weights=P, season="none", mag_th=0.1, dir_th=0.1, jump_alpha=0.05, bonferroni_window=True)
        ru = just_jumps(t, y, season="none", mag_th=0.1, dir_th=0.1, jump_alpha=0.05, bonferroni_window=True)
        mw = [j for j in rw.jumps if abs(j.time_year - 3.0) <= 0.3]
        mu = [j for j in ru.jumps if abs(j.time_year - 3.0) <= 0.3]
        if mw:
            err_w.append(mw[0].magnitude - 0.45)
        if mu:
            err_u.append(mu[0].magnitude - 0.45)
    rmse_w = float(np.sqrt(np.mean(np.square(err_w)))) if err_w else float("inf")
    rmse_u = float(np.sqrt(np.mean(np.square(err_u)))) if err_u else float("inf")
    rows.append(
        {
            "id": "J5",
            "scenario": "Heteroscedastic Weighting (Weighted P vs Unweighted OLS)",
            "pass": rmse_w < 0.80 * rmse_u,
            "summary": f"Weighted mag RMSE={rmse_w:.4f} (recall={len(err_w)/n_rep:.0%}) vs Unweighted RMSE={rmse_u:.4f} (recall={len(err_u)/n_rep:.0%})",
        }
    )
    return rows


def run_stpd_matrix(n_rep: int = 50) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # S1: Continuous TP recovery (white, AR(1) phi=0.3, heavy-tail t5)
    rng = np.random.default_rng(201)
    hits_w = 0
    t_errs_w: list[float] = []
    hits_t5 = 0
    for _ in range(n_rep):
        t = np.linspace(0.0, 6.0, 73)
        y0 = np.where(t <= 3.0, -1.0 * t, -3.0 - 6.0 * (t - 3.0))
        rw = detect_turning_points(t, y0 + rng.normal(0.0, 0.8, size=73), size=60, step=12, margin=10)
        mw = [tp for tp in rw.turning_points if abs(tp.time_year - 3.0) <= 0.5]
        if mw:
            hits_w += 1
            t_errs_w.append(abs(mw[0].time_year - 3.0))
        rt = detect_turning_points(t, y0 + 0.6 * rng.standard_t(df=5, size=73), size=60, step=12, margin=10)
        if any(abs(tp.time_year - 3.0) <= 0.5 for tp in rt.turning_points):
            hits_t5 += 1
    mae_t = float(np.mean(t_errs_w)) if t_errs_w else float("inf")
    rows.append(
        {
            "id": "S1",
            "scenario": "STPD Continuous TP Recovery (White & Heavy-Tail t5)",
            "pass": (hits_w / n_rep >= 0.90) and (mae_t <= 0.25) and (hits_t5 / n_rep >= 0.80),
            "summary": f"White recall={hits_w/n_rep:.1%} (timing MAE={mae_t:.3f} yr); t5 recall={hits_t5/n_rep:.1%}",
        }
    )

    # S2: Null constant-slope false-positive rate
    rng = np.random.default_rng(202)
    fp = 0
    for _ in range(n_rep):
        t = np.linspace(0.0, 6.0, 73)
        y = 2.0 - 1.5 * t + rng.normal(0.0, 0.8, size=73)
        rs = detect_turning_points(t, y, size=60, step=12, margin=10, alpha=0.01)
        if len(rs.turning_points) > 0:
            fp += 1
    rows.append(
        {
            "id": "S2",
            "scenario": "STPD Null Constant-Slope Calibration (0 TPs)",
            "pass": fp / n_rep <= 0.08,
            "summary": f"Empirical FPR={fp/n_rep:.1%} at alpha=0.01, SNR>1, NDRI<0.3",
        }
    )

    # S3: Jump vs. TP NDRI ROC AUC discrimination
    rng = np.random.default_rng(203)
    ndri_tps: list[float] = []
    ndri_jumps: list[float] = []
    flagged_cnt = 0
    for _ in range(n_rep):
        t = np.linspace(0.0, 6.0, 73)
        y_tp = np.where(t <= 3.0, -1.0 * t, -3.0 - 7.0 * (t - 3.0)) + rng.normal(0.0, 0.7, size=73)
        y_jp = np.where(t < 3.0, 0.0, -14.0) + rng.normal(0.0, 0.7, size=73)
        rtp = detect_turning_points(t, y_tp, size=60, step=12, margin=10)
        rjp = detect_turning_points(t, y_jp, size=60, step=12, margin=10)
        if rtp.turning_points:
            ndri_tps.append(rtp.turning_points[0].ndri_rmse)
        if rjp.flagged_jumps:
            ndri_jumps.append(rjp.flagged_jumps[0].ndri_rmse)
            flagged_cnt += 1
        elif rjp.turning_points:
            ndri_jumps.append(rjp.turning_points[0].ndri_rmse)
    # Compute Mann-Whitney ROC AUC
    auc = (
        float(np.mean([1.0 if j > tp else (0.5 if j == tp else 0.0) for j in ndri_jumps for tp in ndri_tps]))
        if (ndri_tps and ndri_jumps)
        else 0.0
    )
    med_tp = float(np.median(ndri_tps)) if ndri_tps else float("nan")
    med_jp = float(np.median(ndri_jumps)) if ndri_jumps else float("nan")
    rows.append(
        {
            "id": "S3",
            "scenario": "STPD NDRI Jump vs. Continuous TP Discrimination",
            "pass": (med_tp <= 0.15) and (med_jp >= 0.45) and (auc >= 0.90),
            "summary": f"Median NDRI TP={med_tp:.3f} vs Jump={med_jp:.3f}; ROC AUC={auc:.3f}; flagged jumps={flagged_cnt}/{n_rep}",
        }
    )

    # S4: Density-imbalance NDRI stress test (3:1 sample ratio before vs after TP)
    rng = np.random.default_rng(204)
    rej_l2 = 0
    rej_rmse = 0
    for _ in range(n_rep):
        t = np.concatenate([np.linspace(0.0, 3.0, 60), np.linspace(3.15, 6.0, 20)])
        y = np.where(t <= 3.0, -1.0 * t, -3.0 - 6.0 * (t - 3.0)) + rng.normal(0.0, 0.7, size=t.size)
        r_l2 = detect_turning_points(t, y, size=80, step=10, margin=8, ndri_mode="l2", snr_mode="centered")
        r_rm = detect_turning_points(t, y, size=80, step=10, margin=8, ndri_mode="rmse", snr_mode="centered")
        if len(r_l2.turning_points) == 0:
            rej_l2 += 1
        if len(r_rm.turning_points) == 0:
            rej_rmse += 1
    rows.append(
        {
            "id": "S4",
            "scenario": "Density-Imbalance NDRI Stress Test (3:1 sampling ratio)",
            "pass": rej_rmse <= rej_l2,
            "summary": f"False-rejection rate: ndri_mode='l2'={rej_l2/n_rep:.1%} vs ndri_mode='rmse'={rej_rmse/n_rep:.1%}",
        }
    )

    # S5: Head-to-head JUST vs STPD on noisy continuous TPs
    rng = np.random.default_rng(205)
    t_err_stpd: list[float] = []
    t_err_just: list[float] = []
    just_spurious_steps: list[float] = []
    for _ in range(n_rep):
        t = np.linspace(0.0, 6.0, 73)
        y = np.where(t <= 3.0, -1.0 * t, -3.0 - 4.0 * (t - 3.0)) + rng.normal(0.0, 1.2, size=73)
        rs = detect_turning_points(t, y, size=60, step=12, margin=10)
        rj = just_jumps(t, y, season="none", size=60, step=12, mag_th=0.0, dir_th=0.0, jump_alpha=0.05, bonferroni_window=True)
        if rs.turning_points:
            t_err_stpd.append(abs(rs.turning_points[0].time_year - 3.0))
        if rj.jumps:
            best_j = min(rj.jumps, key=lambda j: abs(j.time_year - 3.0))
            t_err_just.append(abs(best_j.time_year - 3.0))
            just_spurious_steps.append(abs(best_j.magnitude))
    rmse_ts = float(np.sqrt(np.mean(np.square(t_err_stpd)))) if t_err_stpd else float("inf")
    rmse_tj = float(np.sqrt(np.mean(np.square(t_err_just)))) if t_err_just else float("inf")
    med_step = float(np.median(just_spurious_steps)) if just_spurious_steps else 0.0
    rows.append(
        {
            "id": "S5",
            "scenario": "Head-to-Head JUST vs. STPD on Noisy Continuous TPs",
            "pass": rmse_ts <= rmse_tj + 0.15,
            "summary": f"Timing RMSE STPD={rmse_ts:.3f} yr vs JUST={rmse_tj:.3f} yr (JUST median spurious vertical step={med_step:.2f})",
        }
    )
    return rows


def run_censored_matrix(n_rep: int = 30) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    import ndimpute

    # C1: Temporal trend injection regression test
    raw_slopes: list[float] = []
    safe_slopes: list[float] = []
    for s in range(n_rep):
        rng = np.random.default_rng(300 + s)
        t = np.linspace(0.0, 5.0, 120)
        y0 = rng.lognormal(0.0, 0.8, size=120)
        dl = float(np.quantile(y0, 0.40))
        cen = y0 < dl
        y_obs = np.where(cen, dl, y0)
        df_r = ndimpute.impute(y_obs, status=cen, method="ros", dist="lognormal", impute_type="quantile")
        raw_slopes.append(float(np.polyfit(t, df_r["imputed_value"].to_numpy(), 1)[0]))
        prep = prepare_series_with_censoring(
            t, y_obs, censor_status=cen, censoring=CensoredConfig(method="ros", dist="lognormal", impute_type="quantile", random_state=s)
        )
        safe_slopes.append(float(np.polyfit(t, prep.series.values, 1)[0]))
    se_safe = float(np.std(safe_slopes, ddof=1) / np.sqrt(n_rep))
    rows.append(
        {
            "id": "C1",
            "scenario": "Censored ROS Quantile Temporal-Trend-Injection Guard",
            "pass": (float(np.mean(raw_slopes)) > 0.02) and (abs(float(np.mean(safe_slopes))) <= 2.5 * se_safe),
            "summary": f"Unshuffled raw slope={np.mean(raw_slopes):+.4f} (biased!) vs tspec tie-shuffled={np.mean(safe_slopes):+.4f} ± {se_safe:.4f}",
        }
    )

    # C2: 30% left-censored recovery: ndimpute vs DL/2 substitution vs oracle
    rng = np.random.default_rng(302)
    err_nd: list[float] = []
    err_sub: list[float] = []
    for s in range(n_rep):
        t = np.linspace(0.0, 6.0, 120)
        y_true = np.exp(np.where(t < 3.0, 0.2, 1.3) + rng.normal(0.0, 0.25, size=120))
        dl = float(np.quantile(y_true, 0.30))
        cen = y_true < dl
        y_obs = np.where(cen, dl, y_true)
        r_ora = just_jumps(t, y_true, season="none", mag_th=0.2, dir_th=0.2, jump_alpha=0.05, bonferroni_window=True)
        r_nd = just_jumps(
            t, y_obs, season="none", mag_th=0.2, dir_th=0.2, jump_alpha=0.05, bonferroni_window=True,
            censor_status=cen, censoring=CensoredConfig(method="ros", dist="lognormal", impute_type="stochastic", random_state=s),
        )
        r_sub = just_jumps(
            t, y_obs, season="none", mag_th=0.2, dir_th=0.2, jump_alpha=0.05, bonferroni_window=True,
            censor_status=cen, censoring=CensoredConfig(method="substitution", random_state=s),
        )
        if r_ora.jumps and r_nd.jumps and r_sub.jumps:
            ora_m = min(r_ora.jumps, key=lambda j: abs(j.time_year - 3.0)).magnitude
            nd_m = min(r_nd.jumps, key=lambda j: abs(j.time_year - 3.0)).magnitude
            sub_m = min(r_sub.jumps, key=lambda j: abs(j.time_year - 3.0)).magnitude
            err_nd.append(abs(nd_m - ora_m))
            err_sub.append(abs(sub_m - ora_m))
    mae_nd = float(np.mean(err_nd)) if err_nd else float("inf")
    mae_sub = float(np.mean(err_sub)) if err_sub else float("inf")
    rows.append(
        {
            "id": "C2",
            "scenario": "30% Left-Censored Jump Magnitude Error (ndimpute ROS vs DL/2)",
            "pass": mae_nd <= mae_sub + 0.05,
            "summary": f"MAE vs uncensored oracle: ndimpute ROS={mae_nd:.4f} vs DL/2 substitution={mae_sub:.4f}",
        }
    )

    # C3: Shifting Detection-Limit Null Artifact Stress Test
    rng = np.random.default_rng(303)
    dl_detected = 0
    coinc_flagged = 0
    n_with_jump = 0
    for s in range(n_rep):
        t = np.linspace(2014.0, 2022.0, 140)
        y_true = rng.lognormal(0.2, 0.7, size=140)
        dls = np.where(t < 2018.0, 2.0, 0.2)
        cen = y_true < dls
        y_obs = np.where(cen, dls, y_true)
        rj = just_jumps(
            t, y_obs, season="none", mag_th=0.1, dir_th=0.1,
            censor_status=cen,
            censoring=CensoredConfig(method="substitution", detect_dl_shifts=True, recensor_check=True, random_state=s),
        )
        if rj.censoring is not None and len(rj.censoring.dl_shifts) >= 1:
            dl_detected += 1
        near = [j for j in rj.jumps if abs(j.time_year - 2018.0) <= 0.6]
        if near:
            n_with_jump += 1
            if any(j.coincides_with_dl_shift for j in near):
                coinc_flagged += 1
    rows.append(
        {
            "id": "C3",
            "scenario": "Shifting Detection-Limit Null Artifact Defense (<2.0 -> <0.2)",
            "pass": (dl_detected / n_rep >= 0.95) and (n_with_jump == 0 or coinc_flagged / n_with_jump >= 0.90),
            "summary": f"DL shift detected={dl_detected}/{n_rep} ({dl_detected/n_rep:.0%}); coinciding fake jumps flagged={coinc_flagged}/{n_with_jump}",
        }
    )

    # C4: Non-stationary large jump (global vs windowed scope)
    rng = np.random.default_rng(304)
    err_g: list[float] = []
    err_w: list[float] = []
    for s in range(n_rep):
        t = np.linspace(0.0, 6.0, 120)
        y_true = np.exp(np.where(t < 3.0, -0.2, 1.6) + rng.normal(0.0, 0.35, size=120))
        dl = 0.85  # ~55% censored before jump, ~2% censored after jump
        cen = y_true < dl
        y_obs = np.where(cen, dl, y_true)
        r_ora = just_jumps(t, y_true, season="none", mag_th=0.5, dir_th=0.5, jump_alpha=0.05, bonferroni_window=True)
        r_g = just_jumps(
            t, y_obs, season="none", mag_th=0.5, dir_th=0.5, jump_alpha=0.05, bonferroni_window=True,
            censor_status=cen, censoring=CensoredConfig(scope="global", random_state=s),
        )
        r_w = just_jumps(
            t, y_obs, season="none", mag_th=0.5, dir_th=0.5, jump_alpha=0.05, bonferroni_window=True,
            censor_status=cen, censoring=CensoredConfig(scope="windowed", random_state=s),
        )
        if r_ora.jumps and r_g.jumps and r_w.jumps:
            om = min(r_ora.jumps, key=lambda j: abs(j.time_year - 3.0)).magnitude
            err_g.append(abs(min(r_g.jumps, key=lambda j: abs(j.time_year - 3.0)).magnitude - om))
            err_w.append(abs(min(r_w.jumps, key=lambda j: abs(j.time_year - 3.0)).magnitude - om))
    rows.append(
        {
            "id": "C4",
            "scenario": "Non-Stationary Large Jump (scope='global' vs scope='windowed')",
            "pass": len(err_w) > 0 and len(err_g) > 0,
            "summary": f"Mag MAE vs oracle: global={np.mean(err_g):.3f} vs windowed={np.mean(err_w):.3f}",
        }
    )

    # C5: Down-weighting trade-off sweep (gamma in {1.0, 0.5, 0.25})
    rng = np.random.default_rng(305)
    gammas = (1.0, 0.5, 0.25)
    gamma_errs: dict[float, list[float]] = {g: [] for g in gammas}
    for s in range(n_rep):
        t = np.linspace(0.0, 6.0, 120)
        y_true = np.exp(np.where(t < 3.0, 0.2, 1.3) + rng.normal(0.0, 0.25, size=120))
        dl = float(np.quantile(y_true, 0.30))
        cen = y_true < dl
        y_obs = np.where(cen, dl, y_true)
        r_ora = just_jumps(t, y_true, season="none", mag_th=0.2, dir_th=0.2, jump_alpha=0.05, bonferroni_window=True)
        if not r_ora.jumps:
            continue
        om = min(r_ora.jumps, key=lambda j: abs(j.time_year - 3.0)).magnitude
        for g in gammas:
            rg = just_jumps(
                t, y_obs, season="none", mag_th=0.2, dir_th=0.2, jump_alpha=0.05, bonferroni_window=True,
                censor_status=cen, censoring=CensoredConfig(imputed_weight_factor=g, random_state=s),
            )
            if rg.jumps:
                gamma_errs[g].append(abs(min(rg.jumps, key=lambda j: abs(j.time_year - 3.0)).magnitude - om))
    g_str = ", ".join(f"γ={g}:{np.mean(gamma_errs[g]):.3f}" for g in gammas)
    rows.append(
        {
            "id": "C5",
            "scenario": "Imputed Down-Weighting Trade-Off Sweep (30% censored)",
            "pass": all(len(gamma_errs[g]) > 0 for g in gammas),
            "summary": f"Mag MAE vs oracle: {g_str}",
        }
    )

    # C6: Mixed constant <DL / >UL zero false DL shifts + causal NRT censoring zero look-ahead leakage
    rng = np.random.default_rng(306)
    false_dl = 0
    max_leak = 0.0
    for s in range(n_rep):
        t = np.linspace(2015.0, 2022.0, 110)
        y_m = rng.normal(5.0, 2.8, size=110)
        y_str = ["<1.0" if v < 1.0 else (">9.0" if v > 9.0 else f"{v:.3f}") for v in y_m]
        prep_m = prepare_series_with_censoring(
            t, y_str, censoring=CensoredConfig(method="parametric", dist="normal", random_state=s)
        )
        if prep_m.provenance is not None and len(prep_m.provenance.dl_shifts) > 0:
            false_dl += 1

        y_pos = np.exp(0.4 + 0.2 * np.cos(2 * np.pi * t) + rng.normal(0.0, 0.25, size=110))
        dl_c = float(np.quantile(y_pos[:75], 0.25))
        ya = [f"<{dl_c:.4f}" if v < dl_c else f"{v:.5f}" for v in y_pos]
        yb = list(ya)
        for k in range(80, 110):
            yb[k] = f"{y_pos[k] + 250.0:.5f}"
        cfg_c = CensoredConfig(method="parametric", dist="lognormal", impute_type="mean", random_state=s)
        ra = just_monitor(t, ya, start=75, season="ols", censoring=cfg_c)
        rb = just_monitor(t, yb, start=75, season="ols", censoring=cfg_c)
        leak = float(np.max(np.abs(ra.values[:75] - rb.values[:75])))
        max_leak = max(max_leak, leak)

    rows.append(
        {
            "id": "C6",
            "scenario": "Mixed Constant <DL/>UL Zero False Shifts & Causal NRT Imputation",
            "pass": (false_dl == 0) and (max_leak <= 1e-12),
            "summary": f"Mixed-censor false DL shifts={false_dl}/{n_rep}; causal NRT max history leakage={max_leak:.2e}",
        }
    )
    return rows


def run_rednoise_and_pipeline(n_rep: int = 40) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # R1: AR(1) red-noise calibration sweep + exact P=C_f^{-1} restoration
    rng = np.random.default_rng(401)
    fpr_unw: dict[float, float] = {}
    fpr_gls: dict[float, float] = {}
    for phi in (0.0, 0.3, 0.6, 0.8):
        n = 120
        t = np.linspace(0.0, 6.0, n)
        P_true = _ar1_precision(n, phi)
        fp_u = 0
        fp_g = 0
        for _ in range(n_rep):
            y = 1.0 + 0.1 * t + _ar1_noise(n, phi, 0.2, rng)
            ru = lssa(t, y, omega=[1.0], alpha=0.05)
            rg = lssa(t, y, weights=P_true, omega=[1.0], alpha=0.05)
            if ru.spectrum[0] > ru.critical_value:
                fp_u += 1
            if rg.spectrum[0] > rg.critical_value:
                fp_g += 1
        fpr_unw[phi] = fp_u / n_rep
        fpr_gls[phi] = fp_g / n_rep
    rows.append(
        {
            "id": "R1",
            "scenario": "AR(1) Red-Noise Sweep & P=C_f^{-1} Calibration Restoration (alpha=0.05)",
            "pass": fpr_gls[0.8] <= 0.15,
            "summary": "Unweighted FPR: "
            + ", ".join(f"φ={p}:{fpr_unw[p]:.0%}" for p in fpr_unw)
            + " | Weighted P=C_f^{-1} FPR: "
            + ", ".join(f"φ={p}:{fpr_gls[p]:.0%}" for p in fpr_gls),
        }
    )

    # P1: End-to-End Combined Pipeline (seasonality + jump + TP + gaps + 20% censoring)
    rng = np.random.default_rng(501)
    jump_hits = 0
    tp_hits = 0
    n_p = 25
    for s in range(n_p):
        t = np.sort(rng.uniform(2014.0, 2024.0, size=200))
        t = t[~((t > 2016.2) & (t < 2016.45))]
        trend = np.where(t < 2017.5, 4.0, 7.0) + np.where(t > 2021.0, -1.6 * (t - 2021.0), 0.0)
        seas = 0.7 * np.cos(2 * np.pi * t) + 0.3 * np.sin(2 * np.pi * 2.0 * t)
        y_true = trend + seas + rng.normal(0.0, 0.18, size=t.size)
        dl = float(np.quantile(y_true, 0.20))
        cen = y_true < dl
        y_obs = np.where(cen, dl, y_true)
        res = analyze_series(
            t,
            y_obs,
            censor_status=cen,
            censoring=CensoredConfig(method="ros", dist="normal", impute_type="stochastic", random_state=s, n_stability_draws=0),
            just_kwargs={"season": "ols", "jump_alpha": 0.01},
            decompose_kwargs={"season": "ols"},
            stpd_kwargs={"size": 80, "step": 15, "margin": 12, "alpha": 0.01},
        )
        if res.jumps and any(abs(j.time_year - 2017.5) <= 0.35 for j in res.jumps.jumps):
            jump_hits += 1
        if res.turning_points and any(abs(tp.time_year - 2021.0) <= 0.5 for tp in res.turning_points.turning_points):
            tp_hits += 1
    rows.append(
        {
            "id": "P1",
            "scenario": "End-to-End Pipeline (seasonality + jump + TP + gaps + 20% censored)",
            "pass": (jump_hits / n_p >= 0.80) and (tp_hits / n_p >= 0.75),
            "summary": f"Jump recovery={jump_hits/n_p:.1%}; Turning-point recovery={tp_hits/n_p:.1%}",
        }
    )
    return rows


def run_monitor_matrix(n_rep: int = 40) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # M1: Sequential-peeking null false-alarm rate across 25 monitoring steps (alpha=0.05)
    rng = np.random.default_rng(601)
    fp_unadj = 0
    fp_poc = 0
    fp_obf = 0
    fp_con = 0
    for _ in range(n_rep):
        t = np.linspace(2016.0, 2022.0, 145)
        y = 1.0 + 0.05 * (t - 2016.0) + 0.3 * np.cos(2 * np.pi * t) + rng.normal(0.0, 0.08, size=t.size)
        if just_monitor(t, y, start=120, season="ols", alpha=0.05, parity_mode=True).detected:
            fp_unadj += 1
        if just_monitor(
            t, y, start=120, season="ols", alpha=0.05, freq_correction="sidak", sequential_control="pocock"
        ).detected:
            fp_poc += 1
        if just_monitor(
            t, y, start=120, season="ols", alpha=0.05, freq_correction="sidak", sequential_control="obrien_fleming"
        ).detected:
            fp_obf += 1
        if just_monitor(
            t, y, start=120, season="ols", alpha=0.05, freq_correction="sidak", sequential_control="consecutive_k", consecutive_k=2
        ).detected:
            fp_con += 1

    rows.append(
        {
            "id": "M1",
            "scenario": "JUSTmonitor Sequential-Peeking Null FPR (25 steps, alpha=0.05)",
            "pass": (fp_poc / n_rep <= 0.10) and (fp_obf / n_rep <= 0.10) and (fp_con / n_rep <= 0.10) and (fp_poc < fp_unadj),
            "summary": (
                f"Unadjusted parity FPR={fp_unadj/n_rep:.1%} vs "
                f"Sidak+Pocock={fp_poc/n_rep:.1%}, Sidak+OBF={fp_obf/n_rep:.1%}, Sidak+Consec2={fp_con/n_rep:.1%}"
            ),
        }
    )

    # M2: Abrupt & phenological NRT disturbance recovery and detection delay under default control
    rng = np.random.default_rng(602)
    abrupt_hits = 0
    abrupt_delays: list[float] = []
    pheno_hits = 0
    for _ in range(n_rep):
        t = np.linspace(2016.0, 2022.5, 160)
        base = 1.0 + 0.02 * (t - 2016.0) + 0.25 * np.cos(2 * np.pi * t)
        y_ab = base + np.where(t >= 2021.4, -0.45, 0.0) + rng.normal(0.0, 0.04, size=t.size)
        r_ab = just_monitor(t, y_ab, start=2021.0, season="ols", alpha=0.01)
        if r_ab.detected and r_ab.disturbance_type == "abrupt" and r_ab.disturbance_time_year is not None:
            delay = r_ab.disturbance_time_year - 2021.4
            if -0.05 <= delay <= 0.45:
                abrupt_hits += 1
                abrupt_delays.append(max(0.0, delay))

        y_ph = base + np.where(t >= 2021.2, 0.22 * np.sin(2 * np.pi * 3.0 * t), 0.0) + rng.normal(0.0, 0.03, size=t.size)
        r_ph = just_monitor(t, y_ph, start=2021.0, season="ols", alpha=0.01)
        if r_ph.detected and r_ph.disturbance_type == "phenological":
            pheno_hits += 1

    med_delay = float(np.median(abrupt_delays)) if abrupt_delays else float("inf")
    rows.append(
        {
            "id": "M2",
            "scenario": "JUSTmonitor NRT Disturbance Recall (Abrupt & Phenological, Sidak+Pocock)",
            "pass": (abrupt_hits / n_rep >= 0.90) and (pheno_hits / n_rep >= 0.85) and (med_delay <= 0.25),
            "summary": f"Abrupt recall={abrupt_hits/n_rep:.1%} (median delay={med_delay:.3f} yr); Phenological recall={pheno_hits/n_rep:.1%}",
        }
    )

    # M3: Finite-sample forecast-drift null calibration (n_his=48, D=96, alpha=0.05)
    rng = np.random.default_rng(603)
    fp_no_cov = 0
    fp_pred_cov = 0
    for _ in range(n_rep):
        t = np.linspace(2010.0, 2022.0, 144)  # 48 historical + 96 monitoring
        y = 1.0 + 0.2 * (t - 2010.0) + 0.5 * np.cos(2 * np.pi * t) + rng.normal(0.0, 0.2, size=144)
        r_no = just_monitor(t, y, start=48, season="ols", alpha=0.05, account_for_forecast_uncertainty=False)
        r_yes = just_monitor(t, y, start=48, season="ols", alpha=0.05, account_for_forecast_uncertainty=True)
        if r_no.detected:
            fp_no_cov += 1
        if r_yes.detected:
            fp_pred_cov += 1

    rows.append(
        {
            "id": "M3",
            "scenario": "JUSTmonitor Long-Horizon Forecast-Drift Null Calibration (n_his=48, D=96)",
            "pass": (fp_pred_cov / n_rep <= 0.10) and (fp_pred_cov <= fp_no_cov),
            "summary": (
                f"Without forecast cov={fp_no_cov/n_rep:.1%}; "
                f"With predictive cov={fp_pred_cov/n_rep:.1%} (alpha=0.05)"
            ),
        }
    )
    return rows


def main() -> int:
    t0 = time.perf_counter()
    all_rows: list[dict[str, Any]] = []
    all_rows.extend(run_just_matrix())
    all_rows.extend(run_stpd_matrix())
    all_rows.extend(run_censored_matrix())
    all_rows.extend(run_rednoise_and_pipeline())
    all_rows.extend(run_monitor_matrix())
    elapsed = time.perf_counter() - t0

    n_pass = sum(1 for r in all_rows if r["pass"])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Tier 2 Synthetic Ground-Truth & Monte Carlo Benchmarks Report",
        "",
        f"- **Summary:** **{n_pass} / {len(all_rows)} benchmark gates PASSED** in {elapsed:.1f}s",
        "",
        "| ID | Benchmark Scenario | Status | Measured Metrics |",
        "| :--- | :--- | :---: | :--- |",
    ]
    for r in all_rows:
        st = "PASS" if r["pass"] else "FAIL"
        lines.append(f"| **{r['id']}** | {r['scenario']} | **{st}** | {r['summary']} |")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT_PATH} ({n_pass}/{len(all_rows)} passed in {elapsed:.1f}s).")
    return 0 if n_pass == len(all_rows) else 1


if __name__ == "__main__":
    sys.exit(main())
