"""Tier 2e: Known-solution and real-world head-to-head benchmarks, tspec vs STL/MSTL.

Independent audit benchmark (2026-11). Every synthetic scenario has an exact analytic
ground truth (trend, periodic component, noise), so trend / seasonal / remainder errors
and CI coverage are measured against the truth, not against another method.

Scenarios
---------
S1  Regular monthly grid, smooth nonlinear trend + 3-harmonic seasonality  (STL home turf)
S2  Abrupt step jump between two months (sensor recalibration)               (smearing)
S3  35% random dropout + 5-month blackout gap (irregular sampling)          (gap handling)
S4a Non-integer period (2.37 cy/yr) + annual cycle
S4b Growing seasonal amplitude (0.5 -> 1.5)
S4c Hourly tide: M2, S2, K1, O1 constituents (incommensurate periods)
S5a 7% extreme spikes (15-30 sigma)
S5b 30% left-censored (<DL) lognormal-type environmental series
R1  Mauna Loa CO2 (Cleveland et al., 1990) - complete and artificially gapped
R2  AirPassengers (log scale, Box & Jenkins Series G)
R3  San Francisco tide gauge monthly MSL (PSMSL station 10), secular rise + gaps
R4  Nile at Aswan (known 1898/99 regime shift) - JUST vs ruptures (no seasonality)

Run:  OMP_NUM_THREADS=1 MPLBACKEND=Agg python validation/bench_stl_headtohead.py
      (BENCH_NREP=40 by default; figures -> validation/reports/figures_stl/,
       tables -> validation/reports/stl_headtohead_report.md + .json)
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import platform
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import statsmodels
from statsmodels.tsa.seasonal import MSTL, STL

import tspec

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
REP = HERE / "reports"
FIG = REP / "figures_stl"
FIG.mkdir(parents=True, exist_ok=True)
NREP = int(os.environ.get("BENCH_NREP", "40"))
PROCS = int(os.environ.get("BENCH_PROCS", "4"))

# Validated categorical palette (dataviz reference instance, light mode)
C_TRUE, C_TSPEC, C_STL, C_ALT, C_GRID = "#191919", "#257de0", "#f47230", "#00b894", "#e6e6e6"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 220, "axes.grid": True, "grid.color": C_GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#999999",
    "axes.labelcolor": "#333333", "xtick.color": "#555555", "ytick.color": "#555555",
    "lines.linewidth": 1.6, "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
})


# ----------------------------------------------------------------------------- helpers
def rmse(a, b, m=None):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if m is not None:
        a, b = a[m], b[m]
    return float(np.sqrt(np.nanmean((a - b) ** 2)))


def monthly(years=20, start=2000.0):
    return start + (np.arange(int(12 * years)) + 0.5) / 12.0


def stl_fit(y, robust=False, period=12):
    r = STL(y, period=period, robust=robust).fit()
    return np.asarray(r.trend), np.asarray(r.seasonal), np.asarray(r.resid)


def tspec_dec(t, y, **kw):
    kw.setdefault("season", "ols")
    d = tspec.just_decompose(t, y, **kw)
    return d


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t0


def agg(rows):
    keys = rows[0].keys()
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        out[k] = float(np.nanmean(v))
    return out


def pmap(fn, args):
    with mp.get_context("fork").Pool(PROCS) as p:
        return p.map(fn, args)


# ----------------------------------------------------------------------------- S1
def s1_truth(t):
    u = t - t[0]
    trend = 10 + 0.08 * u + 0.9 * np.sin(2 * np.pi * u / 13.0) + 0.002 * (u - 10) ** 2
    seas = 1.0 * np.cos(2 * np.pi * t) + 0.45 * np.sin(4 * np.pi * t + 0.3) + 0.2 * np.cos(6 * np.pi * t)
    return trend, seas


def s1_job(seed):
    rng = np.random.default_rng(seed)
    t = monthly(20)
    tr, se = s1_truth(t)
    e = rng.normal(0, 0.3, t.size)
    y = tr + se + e
    row = {}
    for lab, kw in [("tspec_ols", dict(season="ols")), ("tspec_allssa", dict(season="allssa"))]:
        d, dt = timed(tspec_dec, t, y, **kw)
        row[f"{lab}_trend_rmse"] = rmse(d.trend, tr)
        row[f"{lab}_trendbc_rmse"] = rmse(d.trend_bc, tr)
        row[f"{lab}_seas_rmse"] = rmse(d.seasonal, se)
        row[f"{lab}_resid_rmse"] = rmse(d.remainder, e)
        row[f"{lab}_trend_cov"] = float(np.mean((d.trend_ci_low <= tr) & (tr <= d.trend_ci_high)))
        row[f"{lab}_seas_cov"] = float(np.mean((d.seasonal_ci_low <= se) & (se <= d.seasonal_ci_high)))
        row[f"{lab}_sec"] = dt
    for lab, rb in [("stl", False), ("stl_robust", True)]:
        (T, S, R), dt = timed(stl_fit, y, rb)
        row[f"{lab}_trend_rmse"] = rmse(T, tr)
        row[f"{lab}_seas_rmse"] = rmse(S, se)
        row[f"{lab}_resid_rmse"] = rmse(R, e)
        row[f"{lab}_sec"] = dt
    return row


# ----------------------------------------------------------------------------- S2
T_JUMP = 2010.0  # falls exactly between two monthly samples (2009.958, 2010.042)


def s2_truth(t):
    u = t - t[0]
    trend = 5 + 0.05 * u + 2.0 * (t >= T_JUMP)
    seas = 1.0 * np.cos(2 * np.pi * t) + 0.4 * np.sin(4 * np.pi * t)
    return trend, seas


def rise_months(t, T, lo_level, hi_level):
    """10-90% rise duration (months) of an estimated trend across a step."""
    frac = (T - lo_level) / (hi_level - lo_level)
    near = np.abs(t - T_JUMP) < 3.0
    tt, ff = t[near], frac[near]
    try:
        t10 = tt[np.argmax(ff >= 0.1)]
        t90 = tt[np.argmax(ff >= 0.9)]
        return float((t90 - t10) * 12.0)
    except ValueError:
        return np.nan


def s2_job(seed):
    rng = np.random.default_rng(100 + seed)
    t = monthly(20)
    tr, se = s2_truth(t)
    e = rng.normal(0, 0.3, t.size)
    y = tr + se + e
    near = np.abs(t - T_JUMP) < 1.0
    row = {}
    (jr), dt_j = timed(tspec.just_jumps, t, y, season="ols")
    hits = [j for j in jr.jumps if abs(jr.times_numeric[j.valid_index] - T_JUMP) < 0.25]
    row["tspec_recall"] = float(bool(hits))
    row["tspec_false_jumps"] = float(len(jr.jumps) - len(hits))
    row["tspec_mag_err"] = float(hits[0].magnitude - 2.0) if hits else np.nan
    row["tspec_loc_err_months"] = float(abs(jr.times_numeric[hits[0].valid_index] - 2010.042) * 12) if hits else np.nan
    ji = [hits[0].index] if hits else []
    d, dt_d = timed(tspec_dec, t, y, jump_indices=ji)
    pre, post = 5 + 0.05 * (T_JUMP - t[0]), 7 + 0.05 * (T_JUMP - t[0])
    row["tspec_trend_rmse"] = rmse(d.trend, tr)
    row["tspec_trend_rmse_pm1yr"] = rmse(d.trend, tr, near)
    row["tspec_seas_rmse_pm1yr"] = rmse(d.seasonal, se, near)
    row["tspec_rise_months"] = rise_months(t, d.trend, pre, post)
    row["tspec_trend_cov_pm1yr"] = float(np.mean(((d.trend_ci_low <= tr) & (tr <= d.trend_ci_high))[near]))
    row["tspec_sec"] = dt_j + dt_d
    for lab, rb in [("stl", False), ("stl_robust", True)]:
        (T, S, R), dt = timed(stl_fit, y, rb)
        row[f"{lab}_trend_rmse"] = rmse(T, tr)
        row[f"{lab}_trend_rmse_pm1yr"] = rmse(T, tr, near)
        row[f"{lab}_seas_rmse_pm1yr"] = rmse(S, se, near)
        row[f"{lab}_rise_months"] = rise_months(t, T, pre, post)
        # STL implied jump: trend change across the two bracketing months
        k = int(np.searchsorted(t, T_JUMP))
        row[f"{lab}_mag_err"] = float(T[k] - T[k - 1] - 2.0)
        row[f"{lab}_sec"] = dt
    return row


# ----------------------------------------------------------------------------- S3
def s3_job(seed):
    rng = np.random.default_rng(200 + seed)
    t = monthly(20)
    tr, se = s1_truth(t)
    e = rng.normal(0, 0.3, t.size)
    y = tr + se + e
    keep = rng.random(t.size) > 0.35
    gap = (t > 2011.0) & (t < 2011.0 + 5 / 12)
    keep &= ~gap
    to, yo = t[keep], y[keep]
    row = {"frac_missing": 1 - keep.mean()}
    d, dt = timed(tspec_dec, to, yo, eval_times=t[(t >= to[0]) & (t <= to[-1])])
    tg = t[(t >= to[0]) & (t <= to[-1])]
    trg, seg = s1_truth(tg)
    in_gap = (tg > 2011.0) & (tg < 2011.0 + 5 / 12)
    row["tspec_trend_rmse_obs"] = rmse(d.trend, tr[keep])
    row["tspec_seas_rmse_obs"] = rmse(d.seasonal, se[keep])
    row["tspec_trend_rmse_gap"] = rmse(d.trend_reg, trg, in_gap)
    row["tspec_trend_bias_gap"] = float(np.mean((d.trend_reg - trg)[in_gap]))
    row["tspec_seas_rmse_grid"] = rmse(d.seasonal_reg, seg)
    row["tspec_trend_cov_obs"] = float(np.mean((d.trend_ci_low <= tr[keep]) & (tr[keep] <= d.trend_ci_high)))
    row["tspec_trend_cov_gap"] = float(np.mean(((d.trend_reg_ci_low <= trg) & (trg <= d.trend_reg_ci_high))[in_gap]))
    row["tspec_sec"] = dt
    for lab, kind in [("stl_linear", "linear"), ("stl_spline", "cubic")]:
        from scipy.interpolate import CubicSpline

        yi = np.interp(t, to, yo) if kind == "linear" else CubicSpline(to, yo)(t)
        (T, S, R), dt = timed(stl_fit, yi, True)
        row[f"{lab}_trend_rmse_obs"] = rmse(T[keep], tr[keep])
        row[f"{lab}_seas_rmse_obs"] = rmse(S[keep], se[keep])
        gm = (t > 2011.0) & (t < 2011.0 + 5 / 12)
        row[f"{lab}_trend_rmse_gap"] = rmse(T, tr, gm)
        row[f"{lab}_trend_bias_gap"] = float(np.mean((T - tr)[gm]))
        row[f"{lab}_seas_rmse_grid"] = rmse(S, se)
        row[f"{lab}_seas_rmse_gap"] = rmse(S, se, gm)
        row[f"{lab}_sec"] = dt
    return row


# ----------------------------------------------------------------------------- S4
def s4a_job(seed):
    rng = np.random.default_rng(300 + seed)
    t = monthly(20)
    tr = 3 + 0.05 * (t - t[0])
    per = 1.0 * np.cos(2 * np.pi * t) + 0.7 * np.cos(2 * np.pi * 2.37 * t + 0.5)
    e = rng.normal(0, 0.25, t.size)
    y = tr + per + e
    row = {}
    for lab, kw in [("tspec_allssa_win", dict(season="allssa")), ("tspec_ols_fixed124", dict(season="ols"))]:
        d, dt = timed(tspec_dec, t, y, **kw)
        row[f"{lab}_trend_rmse"] = rmse(d.trend, tr)
        row[f"{lab}_periodic_rmse"] = rmse(d.seasonal, per)
        row[f"{lab}_sec"] = dt
    a, dt = timed(tspec.allssa, t, y, decimal=2, n_mc=499)
    fit = y - a.residuals
    trend_part = a.trend_coeff[0] + a.trend_coeff[1] * (t - t[0])
    row["tspec_global_allssa_periodic_rmse"] = rmse(fit - trend_part, per)
    row["tspec_global_allssa_found_2.37"] = float(np.any(np.abs(a.frequencies - 2.37) < 0.02))
    row["tspec_global_allssa_found_1.0"] = float(np.any(np.abs(a.frequencies - 1.0) < 0.02))
    row["tspec_global_allssa_sec"] = dt
    (T, S, R), dt = timed(stl_fit, y, True)
    row["stl_trend_rmse"] = rmse(T, tr)
    row["stl_periodic_rmse"] = rmse(S, per)
    row["stl_sec"] = dt
    m, dt = timed(lambda: MSTL(y, periods=(5, 12)).fit())
    row["mstl_5_12_trend_rmse"] = rmse(m.trend, tr)
    row["mstl_5_12_periodic_rmse"] = rmse(np.asarray(m.seasonal).sum(axis=1), per)
    row["mstl_5_12_sec"] = dt
    return row


def s4b_job(seed):
    rng = np.random.default_rng(400 + seed)
    t = monthly(20)
    u = (t - t[0]) / 20.0
    tr = 3 + 0.05 * (t - t[0])
    amp = 0.5 + 1.0 * u
    se = amp * np.cos(2 * np.pi * t) + 0.3 * amp * np.sin(4 * np.pi * t)
    e = rng.normal(0, 0.25, t.size)
    y = tr + se + e
    row = {}
    for lab, kw in [("tspec_ols", dict(season="ols")), ("tspec_allssa", dict(season="allssa"))]:
        d, dt = timed(tspec_dec, t, y, **kw)
        row[f"{lab}_trend_rmse"] = rmse(d.trend, tr)
        row[f"{lab}_seas_rmse"] = rmse(d.seasonal, se)
        row[f"{lab}_seas_cov"] = float(np.mean((d.seasonal_ci_low <= se) & (se <= d.seasonal_ci_high)))
        row[f"{lab}_sec"] = dt
    a = tspec.lssa(t, y, known_freqs=[1.0, 2.0])  # global fixed-amplitude harmonic fit (baseline)
    row["tspec_global_lssa_seas_rmse"] = rmse(y - a.residuals - a.coeff[0] - a.coeff[1] * (t - t[0]), se)
    for lab, rb in [("stl", False), ("stl_robust", True)]:
        (T, S, R), dt = timed(stl_fit, y, rb)
        row[f"{lab}_trend_rmse"] = rmse(T, tr)
        row[f"{lab}_seas_rmse"] = rmse(S, se)
        row[f"{lab}_sec"] = dt
    return row


TIDE = {"M2": 24 / 12.4206012, "S2": 24 / 12.0, "K1": 24 / 23.9344696, "O1": 24 / 25.8193417}  # cycles/day
TAMP = {"M2": 1.0, "S2": 0.35, "K1": 0.45, "O1": 0.30}


def s4c_job(seed):
    rng = np.random.default_rng(500 + seed)
    t = np.arange(0, 60, 1 / 24)  # 60 days hourly, time unit = days
    tide = sum(TAMP[k] * np.cos(2 * np.pi * f * t + 0.3 * i) for i, (k, f) in enumerate(TIDE.items()))
    msl = 0.002 * t
    y = msl + tide + rng.normal(0, 0.1, t.size)
    keep = rng.random(t.size) > 0.3  # 30% random gauge dropouts
    row = {}
    a, dt = timed(tspec.allssa, t[keep], y[keep], omega=np.arange(0.5, 4.01, 0.5), decimal=3, n_mc=499)
    fit_t = y[keep] - a.residuals - (a.trend_coeff[0] + a.trend_coeff[1] * (t[keep] - t[keep][0]))
    row["tspec_allssa_n_constituents"] = float(a.frequencies.size)
    row["tspec_allssa_tide_rmse"] = rmse(fit_t, tide[keep])
    row["tspec_allssa_sec"] = dt
    k, dt = timed(tspec.lssa, t[keep], y[keep], known_freqs=list(TIDE.values()))
    cs = k.coeff[2:].reshape(-1, 2)
    t0 = t[keep][0]
    fit_k = sum(cs[i, 0] * np.cos(2 * np.pi * f * (t[keep] - t0)) + cs[i, 1] * np.sin(2 * np.pi * f * (t[keep] - t0))
                for i, f in enumerate(TIDE.values()))
    row["tspec_known_freqs_tide_rmse"] = rmse(fit_k, tide[keep])
    for i, kname in enumerate(TIDE):
        row[f"tspec_known_amp_err_{kname}"] = float(np.hypot(*cs[i]) - TAMP[kname])
    row["tspec_known_freqs_sec"] = dt
    yi = np.interp(t, t[keep], y[keep])
    m, dt = timed(lambda: MSTL(yi, periods=(12, 25)).fit())
    row["mstl_12_25h_tide_rmse"] = rmse(np.asarray(m.seasonal).sum(axis=1)[keep], tide[keep])
    row["mstl_sec"] = dt
    return row


# ----------------------------------------------------------------------------- S5
def s5a_job(seed):
    rng = np.random.default_rng(600 + seed)
    t = monthly(20)
    tr, se = s1_truth(t)
    e = rng.normal(0, 0.3, t.size)
    out_mask = rng.random(t.size) < 0.07
    spikes = np.where(out_mask, rng.choice([-1, 1], t.size) * rng.uniform(15, 30, t.size) * 0.3, 0.0)
    y = tr + se + e + spikes
    clean = ~out_mask
    row = {}
    for lab, kw in [("tspec_none", dict()), ("tspec_huber", dict(robust="huber")), ("tspec_mm", dict(robust="mm"))]:
        d, dt = timed(tspec_dec, t, y, **kw)
        row[f"{lab}_trend_rmse"] = rmse(d.trend, tr)
        row[f"{lab}_seas_rmse"] = rmse(d.seasonal, se)
        row[f"{lab}_resid_rmse_clean"] = rmse(d.remainder, e, clean)
        row[f"{lab}_trend_cov"] = float(np.mean((d.trend_ci_low <= tr) & (tr <= d.trend_ci_high)))
        row[f"{lab}_sec"] = dt
    for lab, rb in [("stl", False), ("stl_robust", True)]:
        (T, S, R), dt = timed(stl_fit, y, rb)
        row[f"{lab}_trend_rmse"] = rmse(T, tr)
        row[f"{lab}_seas_rmse"] = rmse(S, se)
        row[f"{lab}_resid_rmse_clean"] = rmse(R, e, clean)
        row[f"{lab}_sec"] = dt
    return row


def s5b_job(seed):
    rng = np.random.default_rng(700 + seed)
    t = monthly(15)
    tr = 1.0 + 0.04 * (t - t[0])
    se = 0.5 * np.cos(2 * np.pi * t)
    y = tr + se + rng.normal(0, 0.3, t.size)
    dl = float(np.quantile(y, 0.30))
    cens = y < dl
    status = np.where(cens, -1, 0)
    yc = np.where(cens, dl, y)
    row = {"censored_frac": float(cens.mean())}
    cfg = tspec.CensoredConfig(random_state=seed, dist="normal")
    d, dt = timed(tspec_dec, t, yc, censor_status=status, censoring=cfg)
    row["tspec_trend_rmse"] = rmse(d.trend, tr)
    row["tspec_trend_bias"] = float(np.mean(d.trend - tr))
    row["tspec_seas_rmse"] = rmse(d.seasonal, se)
    row["tspec_trend_cov"] = float(np.mean((d.trend_ci_low <= tr) & (tr <= d.trend_ci_high)))
    row["tspec_sec"] = dt
    d2 = tspec_dec(t, np.where(cens, dl / 2, y))
    row["tspec_dl2_trend_rmse"] = rmse(d2.trend, tr)
    row["tspec_dl2_trend_bias"] = float(np.mean(d2.trend - tr))
    for lab, yy in [("stl_dl2", np.where(cens, dl / 2, y)), ("stl_dl", yc)]:
        (T, S, R), dt = timed(stl_fit, yy, True)
        row[f"{lab}_trend_rmse"] = rmse(T, tr)
        row[f"{lab}_trend_bias"] = float(np.mean(T - tr))
        row[f"{lab}_seas_rmse"] = rmse(S, se)
        row[f"{lab}_sec"] = dt
    (T, S, R) = stl_fit(y, True)
    row["stl_uncensored_oracle_trend_rmse"] = rmse(T, tr)
    return row


# ----------------------------------------------------------------------------- figures for synthetic scenarios
def fig_scenario(name, t, y, truth_tr, truth_se, d, stl, extra=None, obs_t=None, title="", vline=None, shade=None):
    T, S, R = stl
    fig, ax = plt.subplots(3, 1, figsize=(8.0, 7.2), sharex=True, constrained_layout=True)
    to = obs_t if obs_t is not None else t
    ax[0].plot(to, y, ".", ms=2.5, color="#8a8a8a", label="observations")
    ax[0].set_ylabel("data")
    ax[1].plot(t, truth_tr, color=C_TRUE, lw=2.2, label="true trend")
    ax[1].plot(d.times_numeric, d.trend, color=C_TSPEC, label="tspec trend")
    if d.trend_ci_low is not None:
        ax[1].fill_between(d.times_numeric, d.trend_ci_low, d.trend_ci_high, color=C_TSPEC, alpha=0.18, lw=0, label="tspec 95% CI")
    if extra is not None:
        ax[1].plot(extra[0], extra[1], color=C_TSPEC, ls=":", lw=1.4, label=extra[2])
    ax[1].plot(t, T, color=C_STL, label="STL trend")
    ax[1].set_ylabel("trend")
    ax[2].plot(t, truth_se, color=C_TRUE, lw=2.2, label="true seasonal")
    ax[2].plot(d.times_numeric, d.seasonal, color=C_TSPEC, label="tspec seasonal")
    ax[2].plot(t, S, color=C_STL, label="STL seasonal")
    ax[2].set_ylabel("seasonal")
    for a in ax:
        if vline is not None:
            a.axvline(vline, color="#777777", ls="--", lw=0.8)
        if shade is not None:
            a.axvspan(*shade, color="#bbbbbb", alpha=0.25, lw=0)
    ax[1].legend(ncol=3, loc="upper left", fontsize=8)
    ax[2].legend(ncol=3, loc="upper left", fontsize=8)
    ax[0].set_title(title)
    ax[2].set_xlabel("time (decimal years)")
    p = FIG / f"{name}.png"
    fig.savefig(p)
    plt.close(fig)
    return p.name


def make_synthetic_figures():
    figs = {}
    rng = np.random.default_rng(1)
    t = monthly(20)
    tr, se = s1_truth(t)
    y = tr + se + rng.normal(0, 0.3, t.size)
    figs["S1"] = fig_scenario("s1_regular_grid", t, y, tr, se, tspec_dec(t, y), stl_fit(y, True),
                              title="S1 - regular monthly grid, smooth nonlinear trend (STL home turf)")
    # S2 + zoom
    rng = np.random.default_rng(101)
    tr, se = s2_truth(t)
    y = tr + se + rng.normal(0, 0.3, t.size)
    jr = tspec.just_jumps(t, y, season="ols")
    d = tspec_dec(t, y, jump_indices=[j.index for j in jr.jumps])
    st = stl_fit(y, True)
    figs["S2"] = fig_scenario("s2_step_jump", t, y, tr, se, d, st, title="S2 - abrupt +2.0 step at 2010.0 (dashed)", vline=T_JUMP)
    fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.2), constrained_layout=True)
    z = np.abs(t - T_JUMP) < 2.0
    ax[0].plot(t[z], tr[z], color=C_TRUE, lw=2.2, label="true")
    ax[0].plot(t[z], d.trend[z], color=C_TSPEC, marker="o", ms=3, label="tspec (JUST)")
    ax[0].plot(t[z], st[0][z], color=C_STL, marker="o", ms=3, label="STL robust")
    ax[0].set_title("trend around the step (zoom)")
    ax[0].legend(fontsize=8)
    ax[1].plot(t[z], (d.seasonal - se)[z], color=C_TSPEC, marker="o", ms=3, label="tspec seasonal error")
    ax[1].plot(t[z], (st[1] - se)[z], color=C_STL, marker="o", ms=3, label="STL seasonal error")
    ax[1].axhline(0, color=C_TRUE, lw=0.8)
    ax[1].set_title("seasonal error around the step")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.axvline(T_JUMP, color="#777777", ls="--", lw=0.8)
        a.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1.0))
        a.set_xlabel("time (decimal years)")
    fig.savefig(FIG / "s2_step_jump_zoom.png")
    plt.close(fig)
    figs["S2_zoom"] = "s2_step_jump_zoom.png"
    # S3
    rng = np.random.default_rng(201)
    tr, se = s1_truth(t)
    y = tr + se + rng.normal(0, 0.3, t.size)
    keep = (rng.random(t.size) > 0.35) & ~((t > 2011.0) & (t < 2011.0 + 5 / 12))
    to, yo = t[keep], y[keep]
    tg = t[(t >= to[0]) & (t <= to[-1])]
    d = tspec_dec(to, yo, eval_times=tg)
    st = stl_fit(np.interp(t, to, yo), True)
    figs["S3"] = fig_scenario("s3_gaps", t, yo, tr, se, d, st, obs_t=to, extra=(tg, d.trend_reg, "tspec trend on full grid"),
                              title="S3 - 35% random dropout + 5-month blackout (shaded); STL needs linear pre-interpolation",
                              shade=(2011.0, 2011.0 + 5 / 12))
    # S5a
    rng = np.random.default_rng(601)
    tr, se = s1_truth(t)
    om = rng.random(t.size) < 0.07
    y = tr + se + rng.normal(0, 0.3, t.size) + np.where(om, rng.choice([-1, 1], t.size) * rng.uniform(15, 30, t.size) * 0.3, 0)
    figs["S5a"] = fig_scenario("s5a_outliers", t, y, tr, se, tspec_dec(t, y, robust="mm"), stl_fit(y, True),
                               title="S5a - 7% spikes of 15-30 sigma: tspec robust='mm' vs STL(robust=True)")
    return figs


# ----------------------------------------------------------------------------- real data
def load_csv(name, cols):
    a = np.genfromtxt(DATA / name, delimiter=",", names=True, dtype=float)
    return [np.asarray(a[c], float) for c in cols]


def real_co2():
    _, _, t, y = load_csv("mauna_loa_co2_monthly.csv", ["year", "month", "decimal_date", "average"])
    m = (t >= 1959) & (t < 2025) & (y > 0)
    t, y = t[m], y[m]
    res = {}
    (T, S, R), dt = timed(stl_fit, y, True)
    d, dt2 = timed(tspec_dec, t, y, season="allssa")
    res["complete"] = dict(n=int(t.size), stl_sec=dt, tspec_sec=dt2,
                           trend_rmse_tspec_vs_stl=rmse(d.trend, T), seas_rmse_tspec_vs_stl=rmse(d.seasonal, S),
                           resid_sd_tspec=float(np.std(d.remainder)), resid_sd_stl=float(np.std(R)),
                           seas_amp_tspec_ptp_1960s=float(np.ptp(d.seasonal[(t > 1960) & (t < 1961)])),
                           seas_amp_tspec_ptp_2010s=float(np.ptp(d.seasonal[(t > 2015) & (t < 2016)])),
                           seas_amp_stl_ptp_1960s=float(np.ptp(S[(t > 1960) & (t < 1961)])),
                           seas_amp_stl_ptp_2010s=float(np.ptp(S[(t > 2015) & (t < 2016)])))
    rng = np.random.default_rng(11)
    keep = (rng.random(t.size) > 0.35) & ~((t > 1990.0) & (t < 1990.5))
    to, yo = t[keep], y[keep]
    dg = tspec_dec(to, yo, season="allssa", eval_times=t[(t >= to[0]) & (t <= to[-1])])
    tg = t[(t >= to[0]) & (t <= to[-1])]
    Tg, Sg, _ = stl_fit(np.interp(t, to, yo), True)
    gap = (t > 1990.0) & (t < 1990.5)
    gapg = (tg > 1990.0) & (tg < 1990.5)
    d_full_on_g = np.interp(tg, t, d.trend)
    res["gapped_self_consistency"] = dict(
        frac_missing=float(1 - keep.mean()),
        tspec_trend_rmse_vs_own_complete=rmse(dg.trend_reg, d_full_on_g),
        stl_trend_rmse_vs_own_complete=rmse(Tg, T),
        tspec_trend_rmse_in_gap=rmse(dg.trend_reg, d_full_on_g, gapg),
        stl_trend_rmse_in_gap=rmse(Tg, T, gap),
        tspec_seas_rmse_vs_own_complete=rmse(dg.seasonal_reg, np.interp(tg, t, d.seasonal)),
        stl_seas_rmse_vs_own_complete=rmse(Sg, S),
        tspec_seas_rmse_in_gap=rmse(dg.seasonal_reg, np.interp(tg, t, d.seasonal), gapg),
        stl_seas_rmse_in_gap=rmse(Sg, S, gap),
    )
    # Non-default tspec settings that are better suited to long blackouts
    for lab, kw in [("ols", dict(season="ols")),
                    ("ols_duration4y", dict(season="ols", window_mode="duration", window_duration=4.0, step_duration=1.0))]:
        full = tspec_dec(t, y, **kw)
        gg = tspec_dec(to, yo, eval_times=tg, **kw)
        rt, rs = np.interp(tg, t, full.trend), np.interp(tg, t, full.seasonal)
        res["gapped_self_consistency"].update({
            f"tspec_{lab}_trend_rmse_vs_own_complete": rmse(gg.trend_reg, rt),
            f"tspec_{lab}_trend_rmse_in_gap": rmse(gg.trend_reg, rt, gapg),
            f"tspec_{lab}_seas_rmse_vs_own_complete": rmse(gg.seasonal_reg, rs),
            f"tspec_{lab}_seas_rmse_in_gap": rmse(gg.seasonal_reg, rs, gapg),
        })
    fig, ax = plt.subplots(3, 2, figsize=(10, 7.2), constrained_layout=True)
    for col, (lab, tt, yy, dd, TT, SS) in enumerate([("complete", t, y, d, T, S), ("35% dropout + 6-month blackout", to, yo, dg, Tg, Sg)]):
        ax[0, col].plot(tt, yy, ".", ms=1.5, color="#8a8a8a")
        ax[0, col].set_title(f"Mauna Loa CO2 - {lab}")
        tt_tr = dd.times_numeric
        ax[1, col].plot(t, TT, color=C_STL, label="STL robust")
        ax[1, col].plot(tt_tr, dd.trend, color=C_TSPEC, ls="--", label="tspec (allssa)")
        ax[1, col].set_ylabel("trend (ppm)")
        z = (t > 1986) & (t < 1994)
        ax[2, col].plot(t[z], SS[z], color=C_STL, label="STL seasonal")
        zz = (tt_tr > 1986) & (tt_tr < 1994)
        ax[2, col].plot(tt_tr[zz], dd.seasonal[zz], color=C_TSPEC, marker=".", ms=3, ls="--", label="tspec seasonal")
        if col == 1:
            for a in ax[:, 1]:
                a.axvspan(1990.0, 1990.5, color="#bbbbbb", alpha=0.35, lw=0)
            ax[2, 1].plot(tg[(tg > 1986) & (tg < 1994)], dg.seasonal_reg[(tg > 1986) & (tg < 1994)], color=C_ALT, lw=1.0, label="tspec on full grid")
        ax[2, col].set_ylabel("seasonal (ppm)")
        ax[2, col].set_xlim(1986, 1994)
        ax[2, col].set_title("seasonal component, 1986-1994 zoom", fontsize=9)
        ax[1, col].legend(fontsize=8)
        ax[2, col].legend(fontsize=8, ncol=2)
    fig.savefig(FIG / "r1_mauna_loa_co2.png")
    plt.close(fig)
    return res


def real_air():
    t, y = load_csv("airpassengers_monthly.csv", ["decimal_year", "passengers_thousands"])
    t = t + 1 / 24
    ly = np.log(y)
    (T, S, R), dt = timed(stl_fit, ly, True)
    d, dt2 = timed(tspec_dec, t, ly, season="allssa")
    dols = tspec_dec(t, ly, season="ols", omega=[1, 2, 3, 4, 5, 6])
    res = dict(n=int(t.size), stl_sec=dt, tspec_sec=dt2, trend_rmse_tspec_vs_stl=rmse(d.trend, T),
               seas_rmse_tspec_vs_stl=rmse(d.seasonal, S), seas_rmse_tspec_ols6_vs_stl=rmse(dols.seasonal, S),
               resid_sd_tspec_allssa=float(np.std(d.remainder)), resid_sd_tspec_ols6=float(np.std(dols.remainder)),
               resid_sd_stl=float(np.std(R)),
               window_freqs_example=[float(f) for f in d.window_frequencies[0]] if d.window_frequencies else [])
    fig, ax = plt.subplots(3, 1, figsize=(8, 6.5), sharex=True, constrained_layout=True)
    ax[0].plot(t, ly, color="#8a8a8a", lw=1)
    ax[0].set_title("AirPassengers (log) - tspec vs STL")
    ax[1].plot(t, T, color=C_STL, label="STL robust")
    ax[1].plot(t, d.trend, color=C_TSPEC, ls="--", label="tspec allssa")
    ax[1].plot(t, dols.trend, color=C_ALT, ls=":", label="tspec ols, 6 harmonics")
    ax[2].plot(t, S, color=C_STL, label="STL")
    ax[2].plot(t, d.seasonal, color=C_TSPEC, ls="--", label="tspec allssa")
    ax[2].plot(t, dols.seasonal, color=C_ALT, ls=":", label="tspec ols 6h")
    for a in ax[1:]:
        a.legend(fontsize=8, ncol=3)
    fig.savefig(FIG / "r2_airpassengers.png")
    plt.close(fig)
    return res


def real_sf():
    t, y = load_csv("psmsl_san_francisco_msl_monthly.csv", ["decimal_year", "msl_mm_rlr"])
    m = t >= 1900
    t, y = t[m], y[m]
    ok = np.isfinite(y)
    (T, S, R), dt = timed(stl_fit, np.interp(t, t[ok], y[ok]), True, 12)
    d, dt2 = timed(tspec_dec, t, y, season="ols", omega=[1, 2])
    sl_t = float(np.polyfit(d.times_numeric, d.trend, 1)[0])
    sl_s = float(np.polyfit(t, T, 1)[0])
    a = tspec.analyze_series(t, y, detect_jumps=False, detect_turning_points=False, slope_estimator="sen", band_method="conditional")
    seg = a.segments[0]
    res = dict(n=int(t.size), n_missing=int((~ok).sum()), stl_sec=dt, tspec_sec=dt2,
               secular_rate_mm_yr_tspec_trend_ls=sl_t, secular_rate_mm_yr_stl_trend_ls=sl_s,
               sen_rate_mm_yr=float(seg.slope), sen_ci=[float(seg.slope_ci_low), float(seg.slope_ci_high)], mk_p=float(seg.pvalue),
               seasonal_ptp_tspec=float(np.ptp(d.seasonal[(d.times_numeric > 2000) & (d.times_numeric < 2001)])),
               seasonal_ptp_stl=float(np.ptp(S[(t > 2000) & (t < 2001)])),
               remainder_1983_tspec=float(np.mean(d.remainder[(d.times_numeric > 1982.9) & (d.times_numeric < 1983.4)])),
               remainder_1998_tspec=float(np.mean(d.remainder[(d.times_numeric > 1997.9) & (d.times_numeric < 1998.4)])),
               remainder_1983_stl=float(np.mean(R[(t > 1982.9) & (t < 1983.4)])),
               remainder_1998_stl=float(np.mean(R[(t > 1997.9) & (t < 1998.4)])),
               resid_sd_tspec=float(np.std(d.remainder)), resid_sd_stl=float(np.std(R)))
    fig, ax = plt.subplots(2, 1, figsize=(8, 5.5), constrained_layout=True)
    ax[0].plot(t, y, color="#8a8a8a", lw=0.6, label="monthly MSL")
    ax[0].plot(t, T, color=C_STL, label=f"STL trend ({sl_s:.2f} mm/yr)")
    ax[0].plot(d.times_numeric, d.trend, color=C_TSPEC, ls="--", label=f"tspec trend ({sl_t:.2f} mm/yr)")
    ax[0].set_ylabel("MSL (mm, RLR)")
    ax[0].set_title("San Francisco tide gauge (PSMSL 10), 1900-2025")
    ax[0].legend(fontsize=8)
    ax[1].plot(t, R, color=C_STL, lw=0.8, label="STL remainder")
    ax[1].plot(d.times_numeric, d.remainder, color=C_TSPEC, lw=0.8, alpha=0.8, label="tspec remainder")
    for yr in (1983.0, 1998.0):
        ax[1].axvline(yr, color="#777777", ls="--", lw=0.8)
    ax[1].set_title("remainders - El Nino 1982-83 / 1997-98 anomalies (dashed)")
    ax[1].legend(fontsize=8)
    fig.savefig(FIG / "r3_sf_tide_gauge.png")
    plt.close(fig)
    return res


def real_nile():
    import ruptures as rpt

    t, y = load_csv("nile_aswan_1871_1970.csv", ["year", "volume"])
    j, dt = timed(tspec.just_jumps, t, y, season="none", size=30, step=5, jump_th=5)
    j_lv = tspec.just_jumps(t, y, season="none", size=30, step=5, jump_th=5, jump_definition="level_or_slope")
    j_iid = tspec.just_jumps(
        t, y, season="none", size=30, step=5, jump_th=5, jump_definition="level_or_slope",
        serial_correlation="none",
    )
    algo = rpt.Binseg(model="l2").fit(y.reshape(-1, 1))
    bk = algo.predict(n_bkps=1)
    pelt = rpt.Pelt(model="l2").fit(y.reshape(-1, 1)).predict(pen=3 * np.var(y) * np.log(y.size))
    return dict(tspec_jump_years=[float(t[e.valid_index]) for e in j.jumps], tspec_mags=[float(e.magnitude) for e in j.jumps],
                tspec_pvals=[float(e.jump_pvalue) for e in j.jumps], tspec_sec=dt,
                serial_rho=float(j.serial_rho),
                level_or_slope_years=[float(t[e.valid_index]) for e in j_lv.jumps],
                level_or_slope_iid_years=[float(t[e.valid_index]) for e in j_iid.jumps],
                level_or_slope_iid_pvals=[float(e.jump_pvalue) for e in j_iid.jumps],
                ruptures_binseg_1bkp_year=float(t[bk[0]]), ruptures_pelt_years=[float(t[b]) for b in pelt[:-1]],
                known="1898/1899 (Cobb 1978)")


# ----------------------------------------------------------------------------- tide figure
def tide_figure():
    rng = np.random.default_rng(501)
    t = np.arange(0, 60, 1 / 24)
    tide = sum(TAMP[k] * np.cos(2 * np.pi * f * t + 0.3 * i) for i, (k, f) in enumerate(TIDE.items()))
    y = 0.002 * t + tide + rng.normal(0, 0.1, t.size)
    keep = rng.random(t.size) > 0.3
    a = tspec.allssa(t[keep], y[keep], omega=np.arange(0.5, 4.01, 0.5), decimal=3, n_mc=499)
    k = tspec.lssa(t[keep], y[keep], known_freqs=list(TIDE.values()))
    fa = y[keep] - a.residuals - (a.trend_coeff[0] + a.trend_coeff[1] * (t[keep] - t[keep][0]))
    fk = y[keep] - k.residuals - (k.coeff[0] + k.coeff[1] * (t[keep] - t[keep][0]))
    m = MSTL(np.interp(t, t[keep], y[keep]), periods=(12, 25)).fit()
    ms = np.asarray(m.seasonal).sum(axis=1)
    z = (t[keep] > 20) & (t[keep] < 24)
    zf = (t > 20) & (t < 24)
    fig, ax = plt.subplots(1, 1, figsize=(8, 3.4), constrained_layout=True)
    ax.plot(t[zf], tide[zf], color=C_TRUE, lw=2.2, label="true tide (M2+S2+K1+O1)")
    ax.plot(t[keep][z], fk[z], color=C_TSPEC, label="tspec lssa(known_freqs)")
    ax.plot(t[keep][z], fa[z], color=C_ALT, ls="--", label=f"tspec allssa ({a.frequencies.size} freqs: {', '.join(f'{f:.3f}' for f in a.frequencies)} cpd)")
    ax.plot(t[zf], ms[zf], color=C_STL, ls=":", label="MSTL(12h, 25h) after interpolation")
    ax.set_xlabel("days")
    ax.set_title("S4c - hourly tide, 30% dropouts (4-day zoom)")
    ax.legend(fontsize=7.5, ncol=2, loc="lower left")
    fig.savefig(FIG / "s4c_tides.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- report
def fmt_table(res, methods, metrics, labels):
    hdr = "| method | " + " | ".join(labels) + " |\n|---|" + "---|" * len(labels) + "\n"
    rows = []
    for m in methods:
        cells = []
        for met in metrics:
            v = res.get(f"{m}_{met}")
            cells.append("—" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:.3f}" if abs(v) < 1000 else f"{v:.0f}"))
        rows.append(f"| `{m}` | " + " | ".join(cells) + " |")
    return hdr + "\n".join(rows) + "\n"


def main():
    t_all = time.time()
    versions = dict(tspec=tspec.__version__, numpy=np.__version__, scipy=scipy.__version__, statsmodels=statsmodels.__version__,
                    python=platform.python_version())
    try:
        from importlib.metadata import version as _v

        versions["ndimpute"] = _v("ndimpute")
    except Exception:
        versions["ndimpute"] = "not installed"
    results = {"versions": versions, "NREP": NREP}
    for name, fn in [("S1", s1_job), ("S2", s2_job), ("S3", s3_job), ("S4a", s4a_job), ("S4b", s4b_job), ("S4c", s4c_job),
                     ("S5a", s5a_job), ("S5b", s5b_job)]:
        t0 = time.time()
        rows = pmap(fn, list(range(NREP)))
        results[name] = agg(rows)
        print(f"{name} done in {time.time() - t0:.1f}s", flush=True)
    results["figures"] = make_synthetic_figures()
    tide_figure()
    results["R1_co2"] = real_co2()
    results["R2_air"] = real_air()
    results["R3_sf"] = real_sf()
    results["R4_nile"] = real_nile()
    results["wall_sec"] = time.time() - t_all
    (REP / "stl_headtohead_results.json").write_text(json.dumps(results, indent=1, default=float))
    write_report(results)
    print(f"total {time.time() - t_all:.0f}s")


def write_report(r):
    L = []
    v = r["versions"]
    L.append("# Tier 2e — Known-solution & real-world head-to-head: `tspec` vs STL / MSTL\n")
    L.append(f"Generated by `validation/bench_stl_headtohead.py` (independent audit, 2026-11). "
             f"`OMP_NUM_THREADS=1 MPLBACKEND=Agg`, {r['NREP']} Monte Carlo replicates per synthetic scenario; "
             f"metrics are replicate means. Versions: " + ", ".join(f"{k} {x}" for k, x in v.items()) + ".\n")
    L.append("All synthetic truths are exact analytic functions. RMSE is against the truth; `cov` is the fraction of "
             "time points whose pointwise 95% CI contains the truth (STL has no native CI: —). Runtime `sec` is "
             "wall-clock per fit, single thread.\n")
    S = r["S1"]
    L.append("## S1 — Regular monthly grid, smooth nonlinear trend + 3 harmonics (STL home turf)\n")
    L.append(fmt_table(S, ["tspec_ols", "tspec_allssa", "stl", "stl_robust"],
                       ["trend_rmse", "trendbc_rmse", "seas_rmse", "resid_rmse", "trend_cov", "seas_cov", "sec"],
                       ["trend RMSE", "trend_bc RMSE", "seasonal RMSE", "remainder vs true noise", "trend CI cov", "seasonal CI cov", "sec"]))
    L.append(f"\n![S1](figures_stl/{r['figures']['S1']})\n")
    S = r["S2"]
    L.append("## S2 — Abrupt +2.0 step between two months (noise σ = 0.3)\n")
    L.append(f"`tspec` = `just_jumps(season='ols')` → `just_decompose(jump_indices=...)`. Jump recall {S['tspec_recall']:.2f}, "
             f"false jumps per series {S['tspec_false_jumps']:.2f}, magnitude error {S['tspec_mag_err']:+.3f}, "
             f"location error {S['tspec_loc_err_months']:.2f} months, trend CI coverage within ±1 yr of the step "
             f"{S['tspec_trend_cov_pm1yr']:.3f}.\n")
    L.append(fmt_table(S, ["tspec", "stl", "stl_robust"], ["trend_rmse", "trend_rmse_pm1yr", "seas_rmse_pm1yr", "rise_months", "mag_err", "sec"],
                       ["trend RMSE", "trend RMSE ±1 yr", "seasonal RMSE ±1 yr", "10–90% rise (months)", "jump-magnitude error", "sec"]))
    L.append(f"\n![S2](figures_stl/{r['figures']['S2']})\n![S2 zoom](figures_stl/{r['figures']['S2_zoom']})\n")
    S = r["S3"]
    L.append(f"## S3 — Irregular sampling: 35% random dropout + 5-month blackout ({S['frac_missing']*100:.0f}% missing)\n")
    L.append("`tspec` runs natively on the observed timestamps (zero interpolation; `eval_times` = full monthly grid). "
             "STL requires a regular grid, so the gaps are first linearly or cubic-spline interpolated.\n")
    L.append(fmt_table(S, ["tspec", "stl_linear", "stl_spline"],
                       ["trend_rmse_obs", "seas_rmse_obs", "trend_rmse_gap", "trend_bias_gap", "seas_rmse_grid", "trend_cov_obs", "trend_cov_gap", "sec"],
                       ["trend RMSE (obs)", "seasonal RMSE (obs)", "trend RMSE in gap", "trend bias in gap", "seasonal RMSE (grid)", "trend CI cov (obs)", "trend CI cov (gap)", "sec"]))
    L.append(f"\nSTL seasonal RMSE inside the blackout (linear fill): {S['stl_linear_seas_rmse_gap']:.3f}; (spline fill): {S['stl_spline_seas_rmse_gap']:.3f}.\n")
    L.append(f"\n![S3](figures_stl/{r['figures']['S3']})\n")
    S = r["S4a"]
    L.append("## S4a — Non-integer period (2.37 cy/yr, 5.06 months) + annual cycle\n")
    L.append(fmt_table(S, ["tspec_allssa_win", "tspec_ols_fixed124", "stl", "mstl_5_12"], ["trend_rmse", "periodic_rmse", "sec"],
                       ["trend RMSE", "total periodic RMSE", "sec"]))
    L.append(f"\nGlobal `allssa(decimal=2)`: periodic RMSE {S['tspec_global_allssa_periodic_rmse']:.3f}; found 2.37 cy/yr in "
             f"{S['tspec_global_allssa_found_2.37']*100:.0f}% and 1.0 cy/yr in {S['tspec_global_allssa_found_1.0']*100:.0f}% of replicates "
             f"({S['tspec_global_allssa_sec']:.2f} s).\n")
    S = r["S4b"]
    L.append("## S4b — Seasonal amplitude growing linearly 0.5 → 1.5 over 20 years\n")
    L.append(fmt_table(S, ["tspec_ols", "tspec_allssa", "stl", "stl_robust"], ["trend_rmse", "seas_rmse", "seas_cov", "sec"],
                       ["trend RMSE", "seasonal RMSE", "seasonal CI cov", "sec"]))
    L.append(f"\nBaseline global fixed-amplitude harmonic fit (`lssa(known_freqs=[1,2])`): seasonal RMSE {S['tspec_global_lssa_seas_rmse']:.3f}.\n")
    S = r["S4c"]
    L.append("## S4c — Hourly tide gauge: M2 + S2 + K1 + O1, 30% random dropouts, 60 days (time unit = days)\n")
    tide_rows = ["| method | tidal-signal RMSE | constituents | sec |", "|---|---|---|---|"]
    tide_rows.append(f"| `tspec.lssa(known_freqs=[M2,S2,K1,O1])` (classical harmonic analysis, no interpolation) | {S['tspec_known_freqs_tide_rmse']:.4f} | 4 (given) | {S['tspec_known_freqs_sec']:.3f} |")
    tide_rows.append(f"| `tspec.allssa(decimal=3)` (blind search) | {S['tspec_allssa_tide_rmse']:.4f} | {S['tspec_allssa_n_constituents']:.1f} found | {S['tspec_allssa_sec']:.2f} |")
    tide_rows.append(f"| `MSTL(periods=(12, 25))` after linear interpolation | {S['mstl_12_25h_tide_rmse']:.4f} | 2 integer periods | {S['mstl_sec']:.2f} |")
    L.append("\n".join(tide_rows) + "\n")
    L.append("\nAmplitude errors with `known_freqs`: " + ", ".join(f"{k} {S[f'tspec_known_amp_err_{k}']:+.4f}" for k in TIDE) + ".\n")
    L.append("\n![S4c](figures_stl/s4c_tides.png)\n")
    S = r["S5a"]
    L.append("## S5a — 7% extreme spikes (15–30σ, random sign)\n")
    L.append(fmt_table(S, ["tspec_none", "tspec_huber", "tspec_mm", "stl", "stl_robust"], ["trend_rmse", "seas_rmse", "resid_rmse_clean", "trend_cov", "sec"],
                       ["trend RMSE", "seasonal RMSE", "remainder err (clean pts)", "trend CI cov", "sec"]))
    L.append(f"\n![S5a](figures_stl/{r['figures']['S5a']})\n")
    S = r["S5b"]
    L.append(f"## S5b — {S['censored_frac']*100:.0f}% left-censored (<DL) observations\n")
    L.append(fmt_table(S, ["tspec", "tspec_dl2", "stl_dl2", "stl_dl", "stl_uncensored_oracle"], ["trend_rmse", "trend_bias", "seas_rmse", "trend_cov", "sec"],
                       ["trend RMSE", "trend bias", "seasonal RMSE", "trend CI cov", "sec"]))
    L.append("\n`tspec` = `just_decompose(censor_status=..., CensoredConfig(dist='normal'))` (trend-aware Tobit imputation); "
             "`*_dl2` = naive DL/2 substitution; `stl_dl` = substitute DL; oracle = STL on the uncensored values.\n")
    R = r["R1_co2"]
    L.append("## R1 — Mauna Loa CO₂ (Cleveland et al., 1990), 1959–2024\n")
    L.append("No ground truth; we report agreement with STL on the complete record and each method's *self-consistency* "
             "when 35% of months plus a 6-month blackout (1990.0–1990.5) are removed (RMSE of the gapped decomposition vs "
             "the same method on complete data).\n")
    c = R["complete"]
    L.append(f"Complete: trend RMSE(tspec vs STL) = {c['trend_rmse_tspec_vs_stl']:.3f} ppm, seasonal RMSE = {c['seas_rmse_tspec_vs_stl']:.3f} ppm; "
             f"remainder SD tspec {c['resid_sd_tspec']:.3f} vs STL {c['resid_sd_stl']:.3f} ppm; seasonal peak-to-peak 1960 "
             f"tspec {c['seas_amp_tspec_ptp_1960s']:.2f} / STL {c['seas_amp_stl_ptp_1960s']:.2f}, 2015 tspec {c['seas_amp_tspec_ptp_2010s']:.2f} / STL {c['seas_amp_stl_ptp_2010s']:.2f} ppm.\n")
    g = R["gapped_self_consistency"]
    L.append("| method | trend RMSE vs own complete | in blackout | seasonal RMSE vs own complete | in blackout |\n|---|---|---|---|---|\n"
             f"| tspec default `season='allssa'` (native gaps) | {g['tspec_trend_rmse_vs_own_complete']:.3f} | {g['tspec_trend_rmse_in_gap']:.3f} | {g['tspec_seas_rmse_vs_own_complete']:.3f} | {g['tspec_seas_rmse_in_gap']:.3f} |\n"
             f"| STL robust (linear fill) | {g['stl_trend_rmse_vs_own_complete']:.3f} | {g['stl_trend_rmse_in_gap']:.3f} | {g['stl_seas_rmse_vs_own_complete']:.3f} | {g['stl_seas_rmse_in_gap']:.3f} |\n"
             f"| tspec `season='ols'` | {g['tspec_ols_trend_rmse_vs_own_complete']:.3f} | {g['tspec_ols_trend_rmse_in_gap']:.3f} | {g['tspec_ols_seas_rmse_vs_own_complete']:.3f} | {g['tspec_ols_seas_rmse_in_gap']:.3f} |\n"
             f"| tspec `season='ols'`, 4-yr duration windows | {g['tspec_ols_duration4y_trend_rmse_vs_own_complete']:.3f} | {g['tspec_ols_duration4y_trend_rmse_in_gap']:.3f} | {g['tspec_ols_duration4y_seas_rmse_vs_own_complete']:.3f} | {g['tspec_ols_duration4y_seas_rmse_in_gap']:.3f} |\n")
    L.append("\nThe default ALLSSA-per-window seasonal model is the weakest `tspec` option across a long blackout on this record "
             "(windows straddling the gap select different harmonic sets); fixed OLS harmonics with calendar-duration windows "
             "beat linear-fill STL inside the gap.\n")
    L.append("\n![R1](figures_stl/r1_mauna_loa_co2.png)\n")
    R = r["R2_air"]
    L.append("## R2 — AirPassengers (log), Box & Jenkins Series G\n")
    L.append(f"trend RMSE(tspec allssa vs STL) = {R['trend_rmse_tspec_vs_stl']:.4f}, seasonal RMSE = {R['seas_rmse_tspec_vs_stl']:.4f} "
             f"(tspec OLS with 6 harmonics: {R['seas_rmse_tspec_ols6_vs_stl']:.4f}); remainder SD: tspec allssa {R['resid_sd_tspec_allssa']:.4f}, "
             f"tspec ols6 {R['resid_sd_tspec_ols6']:.4f}, STL {R['resid_sd_stl']:.4f}.\n")
    L.append("\n![R2](figures_stl/r2_airpassengers.png)\n")
    R = r["R3_sf"]
    L.append("## R3 — San Francisco tide gauge (PSMSL 10), monthly MSL 1900–2025\n")
    L.append(f"{R['n']} months, {R['n_missing']} missing (tspec: dropped; STL: linear fill). Secular rate from a straight line through the "
             f"extracted trend: tspec {R['secular_rate_mm_yr_tspec_trend_ls']:.2f} mm/yr, STL {R['secular_rate_mm_yr_stl_trend_ls']:.2f} mm/yr; "
             f"`analyze_series(slope_estimator='sen')` Sen slope {R['sen_rate_mm_yr']:.2f} mm/yr (95% CI {R['sen_ci'][0]:.2f}–{R['sen_ci'][1]:.2f}, AR(1)-corrected MK p = {R['mk_p']:.2g}). "
             f"NOAA CO-OPS publishes ≈2 mm/yr for station 9414290. Remainder means Dec–May of the 1982-83 / 1997-98 El Niños: "
             f"tspec {R['remainder_1983_tspec']:+.0f} / {R['remainder_1998_tspec']:+.0f} mm; STL {R['remainder_1983_stl']:+.0f} / {R['remainder_1998_stl']:+.0f} mm.\n")
    L.append("\n![R3](figures_stl/r3_sf_tide_gauge.png)\n")
    R = r["R4_nile"]
    L.append("## R4 — Nile at Aswan (known 1898/99 regime shift; no seasonality, STL not applicable)\n")
    L.append(f"`just_jumps(season='none', size=30, step=5)` with the defaults (step definition, AR(1) calibration; estimated "
             f"rho = {R['serial_rho']:.2f}): jumps at {R['tspec_jump_years']}. Published level-or-slope definition: "
             f"{R['level_or_slope_years']}. Level-or-slope with independent errors (the assumption of the classic analyses): "
             f"{R['level_or_slope_iid_years']} (calibrated p {[f'{p:.2g}' for p in R['level_or_slope_iid_pvals']]}). "
             f"`ruptures` Binseg(1 bkp): {R['ruptures_binseg_1bkp_year']:.0f}; Pelt(BIC-type penalty): {R['ruptures_pelt_years']}. "
             f"Known: {R['known']}.\n\nThe Nile drop is about 1.5 sigma within a 30-year window: under the default step definition a "
             "continuous bend with curvature explains it as well as a vertical step, and allowing for the record's mild "
             "persistence makes the evidence borderline (p = 0.05). Methods that assume independent errors and a flat mean "
             "(ruptures, the classic analyses) find it; tspec's defaults are deliberately more conservative.\n")
    L.append(f"\nTotal wall time {r['wall_sec']:.0f} s.\n")
    (REP / "stl_headtohead_report.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
