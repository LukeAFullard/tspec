"""Pre-registered held-out validation of ``just_decompose`` vs STL.

Implements ``validation/decomposition_heldout_plan.md`` exactly (designs, configurations,
seeds, metrics, decision rules). Do not change the registered parts after the first run;
anything added later must be labelled exploratory.

Run from the repository root::

    OMP_NUM_THREADS=1 MPLBACKEND=Agg python validation/bench_decomp_heldout.py

Outputs: validation/reports/decomp_heldout_report.md, decomp_heldout_results.json and
figures in validation/reports/figures_decomp/. Environment: DECOMP_NREP (default 100),
DECOMP_PROCS (default 4), DECOMP_ONLY (comma-separated cell labels, for debugging only).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import platform
import subprocess
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import statsmodels  # noqa: E402
from scipy.special import i0  # noqa: E402
from statsmodels.nonparametric.smoothers_lowess import lowess  # noqa: E402
from statsmodels.tsa.seasonal import STL  # noqa: E402

import tspec  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "validation" / "reports"
FIG = REPORTS / "figures_decomp"
NREP = int(os.environ.get("DECOMP_NREP", "100"))
PROCS = int(os.environ.get("DECOMP_PROCS", "4"))
YEAR_DAYS = 365.2425
SIGMA = 0.3
DATA = ROOT / "validation" / "data" / "choptank_01491000_nitrate.csv"


# ----------------------------------------------------------------------------- truths
def trend_t1(t, t0):
    return 2.0 + 0.03 * (t - t0) + 0.5 * np.sin(2 * np.pi * (t - t0) / 13.0)


def trend_t2(t, t0):
    return trend_t1(t, t0) + 0.9 * (t >= t0 + 11.4)


def seas_s1(t, t0, span):
    return 0.8 * np.cos(2 * np.pi * t) + 0.3 * np.sin(4 * np.pi * t + 0.5)


def seas_s2(t, t0, span):
    u = (t - t0) / span
    amp = 0.5 + 0.7 * u
    phi = 0.6 * u
    return amp * (np.cos(2 * np.pi * t - phi) + 0.3 * np.sin(4 * np.pi * t + 0.5 - 2 * phi))


def seas_s3(t, t0, span, kappa=4.0):
    num = np.exp(kappa * np.cos(2 * np.pi * (t - 0.25))) - i0(kappa)
    return 1.2 * num / (np.exp(kappa) - i0(kappa))


TRENDS = {"T1": trend_t1, "T2": trend_t2}
SEASONS = {"S1": seas_s1, "S2": seas_s2, "S3": seas_s3}


# ----------------------------------------------------------------------------- sampling
def to_decimal_year(dt):
    delta = pd.to_datetime(pd.Series(dt)) - pd.Timestamp("2000-01-01")
    return 2000.0 + delta.dt.total_seconds().to_numpy() / (YEAR_DAYS * 86400.0)


_CHOPTANK = None


def choptank():
    global _CHOPTANK
    if _CHOPTANK is None:
        d = pd.read_csv(DATA, parse_dates=["date"])
        d = d[(d["date"] >= "1985-01-01") & (d["date"] <= "2023-09-30")].reset_index(drop=True)
        d["t"] = to_decimal_year(d["date"])
        _CHOPTANK = d
    return _CHOPTANK


def sample_times(regime, rng):
    t0 = 2000.0
    monthly = t0 + (np.arange(240) + 0.5) / 12
    if regime == "R1":
        return monthly
    if regime == "R2":
        keep = rng.random(240) > 0.30
        keep &= ~((monthly >= t0 + 12.3) & (monthly < t0 + 12.8))
        return monthly[keep]
    if regime == "R3":
        return np.sort(monthly + rng.uniform(-10, 10, 240) / YEAR_DAYS)
    if regime == "R4":
        return np.sort(t0 + rng.uniform(0, 20, 180))
    if regime == "R5":
        out = []
        while len(out) < 200:  # rejection sampling, density ~ 1 + 0.8 cos(2 pi (t - 0.55))
            c = t0 + rng.uniform(0, 20)
            if rng.uniform(0, 1.8) < 1 + 0.8 * np.cos(2 * np.pi * (c - 0.55)):
                out.append(c)
        return np.sort(np.array(out))
    if regime == "R6":
        weekly = t0 + np.arange(0, 6, 7 / YEAR_DAYS)
        mon = t0 + 6 + (np.arange(120) + 0.5) / 12
        qtr = t0 + 16 + (np.arange(16) + 0.5) / 4
        return np.concatenate([weekly, mon, qtr])
    if regime == "R7":
        return choptank()["t"].to_numpy().copy()
    raise ValueError(regime)


def ou(t, tau_years, sd, rng):
    e = np.empty(t.size)
    e[0] = sd * rng.normal()
    for i in range(1, t.size):
        phi = np.exp(-(t[i] - t[i - 1]) / tau_years)
        e[i] = phi * e[i - 1] + sd * np.sqrt(1 - phi * phi) * rng.normal()
    return e


def noise(kind, t, rng):
    if kind == "N1":
        return SIGMA * rng.normal(size=t.size)
    if kind == "N2":
        return ou(t, 45 / YEAR_DAYS, SIGMA, rng)
    if kind == "N3":
        return ou(t, 10 / YEAR_DAYS, 0.25, rng) + ou(t, 1.0, 0.17, rng)
    if kind == "N4":
        return SIGMA * rng.standard_t(3, t.size) / np.sqrt(3.0)
    raise ValueError(kind)


# ----------------------------------------------------------------------------- cells
def build_cells():
    cells = []
    for r in ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]:
        for s in ["S1", "S2", "S3"]:
            cells.append(dict(group="core", regime=r, trend="T1", season=s, noise="N1"))
    for r in ["R1", "R4", "R7"]:
        for n in ["N2", "N3", "N4"]:
            cells.append(dict(group="noise", regime=r, trend="T1", season="S1", noise=n))
    for r in ["R1", "R4", "R7"]:
        cells.append(dict(group="step", regime=r, trend="T2", season="S1", noise="N1"))
    for i, c in enumerate(cells, start=1):
        c["id"] = i
        c["label"] = f"{c['regime']}_{c['trend']}{c['season']}_{c['noise']}"
    return cells


# ----------------------------------------------------------------------------- methods
def month_grid(t):
    """Mid-month grid covering every month that contains data (STL input grid)."""
    k0 = int(np.floor((t[0] - 2000.0) * 12))
    k1 = int(np.floor((t[-1] - 2000.0) * 12))
    return 2000.0 + (np.arange(k0, k1 + 1) + 0.5) / 12


def eval_grid(t):
    """Mid-month evaluation grid inside the data span (secondary metrics, plan section 4)."""
    g = month_grid(t)
    return g[(g >= t[0]) & (g <= t[-1])]


def stl_grid_series(t, y, mode):
    g = month_grid(t)
    if mode == "lin":
        return g, np.interp(g, t, y)
    k = np.floor((t - 2000.0) * 12).astype(int)
    k0 = int(np.floor((g[0] - 2000.0) * 12))
    sums = np.bincount(k - k0, weights=y, minlength=g.size)
    cnt = np.bincount(k - k0, minlength=g.size)
    have = cnt > 0
    gy = np.empty(g.size)
    gy[have] = sums[have] / cnt[have]
    gy[~have] = np.interp(g[~have], g[have], gy[have])
    return g, gy


def run_stl(t, y, mode, robust=False):
    g, gy = stl_grid_series(t, y, mode)
    r = STL(gy, period=12, robust=robust).fit()
    gt, gs = np.asarray(r.trend), np.asarray(r.seasonal)
    return dict(trend=np.interp(t, g, gt), seasonal=np.interp(t, g, gs), grid=g, trend_grid=gt, seas_grid=gs)


def harm_design(t):
    cols = [np.ones_like(t)]
    for k in range(1, 5):
        cols += [np.cos(2 * np.pi * k * t), np.sin(2 * np.pi * k * t)]
    return np.column_stack(cols)


def run_harm_lowess(t, y):
    frac = min(1.0, 3.0 / (t[-1] - t[0]))
    X = harm_design(t)
    trend = lowess(y, t, frac=frac, it=0, return_sorted=False)
    for _ in range(3):
        beta = np.linalg.lstsq(X, y - trend, rcond=None)[0]
        seas = X[:, 1:] @ beta[1:]
        trend = lowess(y - seas, t, frac=frac, it=0, return_sorted=False)
    g = month_grid(t)
    return dict(trend=trend, seasonal=seas, grid=g, trend_grid=np.interp(g, t, trend),
                seas_grid=harm_design(g)[:, 1:] @ beta[1:])


def run_tspec(t, y, cfg, censor_status=None, eval_times=None, jump_indices=None):
    kw = {}
    if censor_status is not None:
        kw["censor_status"] = censor_status
    if cfg == "tspec_default":
        d = tspec.just_decompose(t, y, eval_times=eval_times, **kw)
    elif cfg == "tspec_ols":
        d = tspec.just_decompose(t, y, season="ols", eval_times=eval_times, **kw)
    elif cfg == "tspec_ols_dur":
        d = tspec.just_decompose(t, y, season="ols", window_mode="duration", eval_times=eval_times, **kw)
    elif cfg == "tspec_jump":
        if jump_indices is None:
            jr = tspec.just_jumps(t, y, season="ols")
            jump_indices = [(int(j.index), float(jr.times_numeric[j.valid_index])) for j in jr.jumps]
        ji = sorted(i for i, _ in jump_indices)
        d = tspec.just_decompose(t, y, season="ols", jump_indices=ji, eval_times=eval_times, **kw)
        return d, jump_indices
    else:
        raise ValueError(cfg)
    return d, None


TSPEC_CFGS = ["tspec_default", "tspec_ols", "tspec_ols_dur"]
OTHER_CFGS = ["stl_lin", "stl_bin", "stl_lin_robust", "harm_lowess"]


def rmse(a, b, m=None):
    if m is not None:
        if not np.any(m):
            return np.nan
        a, b = a[m], b[m]
    return float(np.sqrt(np.mean((a - b) ** 2)))


def rmse_finite(a, b):
    ok = np.isfinite(a)
    return (float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2))) if ok.any() else np.nan), float(np.mean(~ok))


def corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def gap_mask(t_obs, grid, min_gap=0.25):
    m = np.zeros(grid.size, bool)
    d = np.diff(t_obs)
    for i in np.flatnonzero(d > min_gap):
        m |= (grid > t_obs[i]) & (grid < t_obs[i + 1])
    return m


def metrics(t, tr, se, e, est_tr, est_se, grid=None, tr_g=None, se_g=None, est_tr_g=None, est_se_g=None, gaps=None):
    rem = (tr + se + e) - est_tr - est_se
    out = dict(
        trend_rmse=rmse(est_tr, tr), seas_rmse=rmse(est_se, se), rem_rmse=rmse(rem, e),
        total_rmse=rmse(est_tr + est_se, tr + se),
        leak_seas=corr(rem, se), leak_trend=corr(rem, tr - tr.mean()),
    )
    if grid is not None:
        out["grid_trend_rmse"] = rmse(est_tr_g, tr_g)
        out["grid_seas_rmse"] = rmse(est_se_g, se_g)
        out["gap_trend_rmse"] = rmse(est_tr_g, tr_g, gaps)
        out["gap_seas_rmse"] = rmse(est_se_g, se_g, gaps)
    return out


def cell_job(args):
    cell, rep = args
    rng = np.random.default_rng(910_000 + 1_000 * cell["id"] + rep)
    t = sample_times(cell["regime"], rng)
    t0, span = float(t[0]), float(t[-1] - t[0])
    tr = TRENDS[cell["trend"]](t, t0)
    se = SEASONS[cell["season"]](t, t0, span)
    e = noise(cell["noise"], t, rng)
    y = tr + se + e
    grid = eval_grid(t)
    tr_g = TRENDS[cell["trend"]](grid, t0)
    se_g = SEASONS[cell["season"]](grid, t0, span)
    gaps = gap_mask(t, grid)
    row = {"n": int(t.size)}
    cfgs = TSPEC_CFGS + (["tspec_jump"] if cell["trend"] == "T2" else [])
    for cfg in cfgs:
        # Primary metrics: the registered call without eval_times. eval_times changes the
        # observation-time results (audit finding AUD-19b), so the grid metrics come from a
        # separate call with eval_times.
        t1 = time.perf_counter()
        try:
            d, jumps = run_tspec(t, y, cfg)
        except Exception as exc:  # noqa: BLE001 - failures are a registered outcome
            row[f"{cfg}_fail"] = 1.0
            row[f"{cfg}_error"] = type(exc).__name__
            continue
        row[f"{cfg}_sec"] = time.perf_counter() - t1
        row[f"{cfg}_fail"] = 0.0
        try:
            dg, _ = run_tspec(t, y, cfg, eval_times=grid, jump_indices=jumps)
            tr_reg, se_reg = dg.trend_reg, dg.seasonal_reg
            row[f"{cfg}_grid_nan"] = float(np.mean(~np.isfinite(tr_reg + se_reg)))
            row[f"{cfg}_grid_fail"] = 0.0
        except Exception as exc:  # noqa: BLE001
            tr_reg = se_reg = np.full(grid.size, np.nan)
            row[f"{cfg}_grid_fail"] = 1.0
            row[f"{cfg}_grid_error"] = type(exc).__name__
        m = metrics(t, tr, se, e, d.trend, d.seasonal, grid, tr_g, se_g, tr_reg, se_reg, gaps)
        m["trend_cov"] = float(np.mean((d.trend_ci_low <= tr) & (tr <= d.trend_ci_high)))
        m["seas_cov"] = float(np.mean((d.seasonal_ci_low <= se) & (se <= d.seasonal_ci_high)))
        if cell["trend"] == "T2":
            near = np.abs(t - (t0 + 11.4)) < 1.0
            m["step_trend_rmse"] = rmse(d.trend, tr, near)
            if cfg == "tspec_jump":
                m["jump_hit"] = float(any(abs(x - (t0 + 11.4)) < 0.5 for _, x in jumps))
                m["false_jumps"] = float(sum(abs(x - (t0 + 11.4)) >= 0.5 for _, x in jumps))
        row.update({f"{cfg}_{k}": v for k, v in m.items()})
    for cfg in OTHER_CFGS:
        t1 = time.perf_counter()
        if cfg == "harm_lowess":
            r = run_harm_lowess(t, y)
        else:
            r = run_stl(t, y, "bin" if cfg == "stl_bin" else "lin", robust=cfg == "stl_lin_robust")
        row[f"{cfg}_sec"] = time.perf_counter() - t1
        row[f"{cfg}_fail"] = 0.0
        est_tr_g = np.interp(grid, r["grid"], r["trend_grid"])
        est_se_g = np.interp(grid, r["grid"], r["seas_grid"])
        m = metrics(t, tr, se, e, r["trend"], r["seasonal"], grid, tr_g, se_g, est_tr_g, est_se_g, gaps)
        if cell["trend"] == "T2":
            m["step_trend_rmse"] = rmse(r["trend"], tr, np.abs(t - (t0 + 11.4)) < 1.0)
        row.update({f"{cfg}_{k}": v for k, v in m.items()})
    return row


def aggregate(rows):
    keys = sorted({k for r in rows for k in r if not k.endswith("_error")})
    out = {}
    for k in keys:
        v = np.array([r.get(k, np.nan) for r in rows], dtype=float)
        out[k] = float(np.nanmean(v)) if np.any(np.isfinite(v)) else float("nan")
    errs = {}
    for r in rows:
        for k, v in r.items():
            if k.endswith("_error"):
                errs.setdefault(k, {}).setdefault(v, 0)
                errs[k][v] += 1
    if errs:
        out["errors"] = errs
    return out


# ----------------------------------------------------------------------------- real record
def real_record_cv(routine_only):
    d = choptank()
    if routine_only:
        d = d[d["event"] == "Routine sample"].reset_index(drop=True)
    t = d["t"].to_numpy()
    y = np.log(d["value"].to_numpy())
    cens = d["censored"].to_numpy().astype(bool)
    y_sub = np.where(cens, np.log(0.5 * d["value"].to_numpy()), y)  # DL/2 for non-tspec methods
    res = {k: [] for k in TSPEC_CFGS + OTHER_CFGS}
    miss = {k: [] for k in TSPEC_CFGS}
    for split in range(20):
        rng = np.random.default_rng(920_000 + split)
        test = rng.random(t.size) < 0.20
        train = ~test
        tt, ty = t[train], y[train]
        # every method scores the same held-out points: detected values inside the training span
        score = test & ~cens & (t >= tt[0]) & (t <= tt[-1])
        cs = np.where(cens[train], -1, 0)
        for cfg in TSPEC_CFGS:
            try:
                dd, _ = run_tspec(tt, ty, cfg, censor_status=cs if cs.any() else None, eval_times=t[score])
                pred = dd.trend_reg + dd.seasonal_reg
                v, frac_nan = rmse_finite(pred, y[score])
                res[cfg].append(v)
                miss[cfg].append(frac_nan)
            except Exception:  # noqa: BLE001
                res[cfg].append(np.nan)
                miss[cfg].append(1.0)
        for cfg in OTHER_CFGS:
            ys = y_sub[train]
            if cfg == "harm_lowess":
                r = run_harm_lowess(tt, ys)
                beta_seas = harm_design(t[score])[:, 1:]
                # seasonal at test times: refit harmonics to the final de-trended training data
                coef = np.linalg.lstsq(harm_design(tt), ys - r["trend"], rcond=None)[0]
                pred = np.interp(t[score], tt, r["trend"]) + beta_seas @ coef[1:]
            else:
                r = run_stl(tt, ys, "bin" if cfg == "stl_bin" else "lin", robust=cfg == "stl_lin_robust")
                pred = np.interp(t[score], r["grid"], r["trend_grid"]) + np.interp(t[score], r["grid"], r["seas_grid"])
            res[cfg].append(rmse(pred, y[score]))
    return {k: dict(mean=float(np.nanmean(v)), sd=float(np.nanstd(v)), n_fail=int(np.sum(~np.isfinite(v))),
                    frac_unpredicted=float(np.mean(miss[k])) if k in miss else 0.0) for k, v in res.items()}


def real_record_figure():
    d = choptank()
    t = d["t"].to_numpy()
    y = np.log(d["value"].to_numpy())
    cens = d["censored"].to_numpy().astype(bool)
    cs = np.where(cens, -1, 0)
    grid = eval_grid(t)
    dd = tspec.just_decompose(t, y, season="ols", window_mode="duration", censor_status=cs, eval_times=grid)
    stl = run_stl(t, np.where(cens, np.log(0.5 * d["value"].to_numpy()), y), "bin")
    fig, ax = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    storm = (d["event"] != "Routine sample").to_numpy()
    ax[0].plot(t[~storm], y[~storm], ".", ms=3, color="0.35", label="routine sample")
    ax[0].plot(t[storm], y[storm], ".", ms=3, color="tab:orange", label="storm / event sample")
    ax[0].plot(grid, dd.trend_reg + dd.seasonal_reg, lw=0.8, color="tab:blue", label="tspec trend + seasonal")
    ax[0].set_ylabel("ln(NO3+NO2, mg/L N)")
    ax[0].legend(fontsize=8, ncol=3)
    ax[1].fill_between(grid, dd.trend_reg_ci_low, dd.trend_reg_ci_high, color="tab:blue", alpha=0.2, lw=0)
    ax[1].plot(grid, dd.trend_reg, color="tab:blue", label="tspec (season='ols', 3-yr duration windows) + 95% band")
    ax[1].plot(stl["grid"], stl["trend_grid"], color="tab:red", lw=1, label="STL on monthly bin means")
    ax[1].set_ylabel("trend")
    ax[1].legend(fontsize=8)
    ax[2].plot(grid, dd.seasonal_reg, color="tab:blue", lw=0.8, label="tspec seasonal")
    ax[2].plot(stl["grid"], stl["seas_grid"], color="tab:red", lw=0.8, alpha=0.7, label="STL seasonal")
    ax[2].set_ylabel("seasonal")
    ax[2].legend(fontsize=8)
    ax[2].set_xlabel("year")
    fig.suptitle("Choptank River near Greensboro, MD (USGS 01491000): nitrate + nitrite, 1985-2023")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "choptank_decomposition.png", dpi=130)
    plt.close(fig)
    yrs = np.floor(grid)
    ann = {int(k): float(np.mean(dd.trend_reg[yrs == k])) for k in np.unique(yrs)}
    return dict(annual_trend_ln=ann, n=int(t.size))


def example_figures():
    """One illustrative replicate per irregular regime (figures only; not part of the hypotheses)."""
    FIG.mkdir(parents=True, exist_ok=True)
    regimes = ["R1", "R4", "R5", "R6", "R7"]
    fig, axes = plt.subplots(len(regimes), 2, figsize=(12, 2.4 * len(regimes)))
    for i, reg in enumerate(regimes):
        rng = np.random.default_rng(930_000 + i)
        t = sample_times(reg, rng)
        t0, span = float(t[0]), float(t[-1] - t[0])
        tr, se = trend_t1(t, t0), seas_s3(t, t0, span)
        y = tr + se + noise("N1", t, rng)
        d, _ = run_tspec(t, y, "tspec_ols")
        s = run_stl(t, y, "lin")
        o = np.argsort(t)
        a = axes[i, 0]
        a.plot(t, y, ".", ms=2, color="0.6")
        a.plot(t, tr, "k", lw=1.2, label="truth")
        a.plot(t, d.trend, color="tab:blue", lw=1, label="tspec_ols")
        a.plot(t, s["trend"], color="tab:red", lw=1, label="stl_lin")
        a.set_ylabel(reg)
        a = axes[i, 1]
        ph = np.mod(t, 1.0)
        oo = np.argsort(ph)
        a.plot(ph[oo], se[oo], "k", lw=1.2)
        a.plot(ph[oo], d.seasonal[oo], ".", ms=2, color="tab:blue")
        a.plot(ph[oo], s["seasonal"][oo], ".", ms=2, color="tab:red", alpha=0.5)
        del o
        if i == 0:
            axes[i, 0].set_title("trend (T1) at observation times")
            axes[i, 1].set_title("seasonal (S3, peaked) vs phase of year")
            axes[i, 0].legend(fontsize=7)
    axes[-1, 0].set_xlabel("year")
    axes[-1, 1].set_xlabel("phase of year")
    fig.tight_layout()
    fig.savefig(FIG / "regimes_example.png", dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------- hypotheses
def hypotheses(R):
    def g(label, key):
        return R[label][key]

    core = [c for c in R if R[c]["group"] == "core"]
    res = {}
    # H1
    ok = []
    for s in ["S1", "S2", "S3"]:
        lab = f"R1_T1{s}_N1"
        best_tr = min(g(lab, "tspec_default_trend_rmse"), g(lab, "tspec_ols_trend_rmse"))
        best_se = min(g(lab, "tspec_default_seas_rmse"), g(lab, "tspec_ols_seas_rmse"))
        ok.append((lab, best_tr / g(lab, "stl_lin_trend_rmse"), best_se / g(lab, "stl_lin_seas_rmse")))
    res["H1"] = dict(values=ok, verdict="PASS" if all(a <= 1.10 and b <= 1.10 for _, a, b in ok) else "FAIL")
    # H2
    cnt, vals = 0, []
    for lab in core:
        if R[lab]["regime"] not in ("R3", "R4", "R5", "R6", "R7"):
            continue
        stl_tr = min(g(lab, "stl_lin_trend_rmse"), g(lab, "stl_bin_trend_rmse"))
        stl_se = min(g(lab, "stl_lin_seas_rmse"), g(lab, "stl_bin_seas_rmse"))
        win = g(lab, "tspec_ols_trend_rmse") < stl_tr and g(lab, "tspec_ols_seas_rmse") < stl_se
        cnt += win
        vals.append((lab, g(lab, "tspec_ols_trend_rmse"), stl_tr, g(lab, "tspec_ols_seas_rmse"), stl_se, bool(win)))
    res["H2"] = dict(values=vals, wins=cnt, verdict="PASS" if cnt >= 12 else "FAIL")
    # H3
    cnt, vals = 0, []
    for lab in core:
        if R[lab]["regime"] == "R1":
            continue
        best = np.nanmin([g(lab, "tspec_ols_total_rmse"), g(lab, "tspec_ols_dur_total_rmse")])
        win = best <= g(lab, "harm_lowess_total_rmse")
        cnt += win
        vals.append((lab, float(best), g(lab, "harm_lowess_total_rmse"), bool(win)))
    res["H3"] = dict(values=vals, wins=cnt, verdict="PASS" if cnt >= 12 else "FAIL")
    # H4
    vals = [(lab, g(lab, "tspec_ols_trend_cov")) for lab in core if R[lab]["season"] == "S1"]
    res["H4"] = dict(values=vals, verdict="PASS" if all(0.90 <= v <= 0.98 for _, v in vals) else "FAIL")
    # H5
    vals = [(lab, g(lab, "tspec_ols_trend_cov")) for lab in R if R[lab]["noise"] == "N2"]
    res["H5"] = dict(values=vals, verdict="PASS" if all(0.90 <= v <= 0.98 for _, v in vals) else "FAIL")
    # H6
    vals = [(lab, g(lab, "tspec_ols_trend_cov"), g(lab, "tspec_ols_seas_cov"), g(lab, "tspec_default_trend_cov"))
            for lab in R if R[lab]["noise"] in ("N3", "N4")]
    res["H6"] = dict(values=vals, verdict="DESCRIPTIVE")
    # H7
    cnt = sum(g(lab, "tspec_ols_seas_rmse") < g(lab, "tspec_default_seas_rmse") for lab in core)
    res["H7"] = dict(ols_better=int(cnt), of=len(core), verdict="DESCRIPTIVE",
                     docs_recommend_ols=bool(cnt >= 14))
    # H8
    vals = []
    for lab in R:
        if R[lab]["group"] != "step":
            continue
        vals.append((lab, g(lab, "tspec_jump_jump_hit"), g(lab, "tspec_jump_false_jumps"),
                     g(lab, "tspec_jump_step_trend_rmse"), g(lab, "tspec_ols_step_trend_rmse"),
                     g(lab, "stl_lin_step_trend_rmse"), g(lab, "harm_lowess_step_trend_rmse")))
    res["H8"] = dict(values=vals, verdict="DESCRIPTIVE")
    return res


# ----------------------------------------------------------------------------- report
METHODS = TSPEC_CFGS + OTHER_CFGS


def fmt(x, nd=3):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def write_report(R, H, real, meta):
    L = ["# Held-out decomposition validation: `just_decompose` vs STL\n",
         "Pre-registered in [`validation/decomposition_heldout_plan.md`](../decomposition_heldout_plan.md) before any design was run. "
         f"Generated by `validation/bench_decomp_heldout.py`. Code under test: commit `{meta['commit']}` "
         f"(`src/` clean: {meta['src_clean']}). {NREP} replicates per cell, `OMP_NUM_THREADS=1`. "
         f"Versions: tspec {meta['tspec']}, numpy {meta['numpy']}, scipy {meta['scipy']}, statsmodels {meta['statsmodels']}, "
         f"pandas {meta['pandas']}, python {meta['python']}. Wall time {meta['wall']:.0f} s.\n",
         "RMSE values are replicate means at the observation times. `cov` is pointwise 95% band coverage of the truth (tspec only). "
         "Failures (exceptions) are counted, and excluded from the means.\n",
         "## Deviations from the registered plan\n",
         "Made while debugging the harness, before the registered run, with metric output suppressed (a 2-replicate smoke test of 8 cells and of the real-record code checked only for crashes):\n",
         "1. The monthly evaluation grid is clipped to the data span (the plan's wording); unclipped, its first or last point could fall a few days outside the data, which `eval_times` rejects. STL still fits on the full monthly grid.",
         "2. Primary (observation-time) metrics for `tspec` come from the registered call *without* `eval_times`; grid metrics come from a second call with `eval_times`. The harness first used one call with `eval_times` for both, but that changes the observation-time results (new finding **AUD-19b**, `audit/independent_2026_11/repro_aud19_eval_times.py`), so it would not have tested the registered call.",
         "3. Real-record cross-validation: every method scores the same held-out points, the detected values inside the training span. `eval_times` returned NaN at some held-out times (AUD-19b); RMSE is computed over finite predictions, and the share of unpredicted points is reported.",
         "4. The smoke test also exposed **AUD-19a** (OLS count windows over a storm-sampling burst are ill-conditioned). No code was changed in response; it appears in the R7 results below as it is.\n",
         "## Registered hypotheses\n", "| ID | Verdict | Summary |", "|---|---|---|"]
    h = H
    L.append(f"| H1 regular data: tspec within 10% of STL | **{h['H1']['verdict']}** | trend / seasonal RMSE ratio (best tspec vs `stl_lin`): "
             + "; ".join(f"{lab.split('_')[1]} {a:.2f} / {b:.2f}" for lab, a, b in h["H1"]["values"]) + " |")
    L.append(f"| H2 irregular: `tspec_ols` beats interpolate-then-STL on trend and seasonal | **{h['H2']['verdict']}** | {h['H2']['wins']} of 15 cells (need ≥ 12) |")
    L.append(f"| H3 irregular: tspec ≤ harmonics + LOWESS baseline (total signal) | **{h['H3']['verdict']}** | {h['H3']['wins']} of 18 cells (need ≥ 12) |")
    L.append(f"| H4 trend band 90–98%, white noise, S1, all regimes | **{h['H4']['verdict']}** | "
             + ", ".join(f"{lab.split('_')[0]} {v:.3f}" for lab, v in h["H4"]["values"]) + " |")
    L.append(f"| H5 trend band 90–98% under OU noise | **{h['H5']['verdict']}** | "
             + ", ".join(f"{lab.split('_')[0]} {v:.3f}" for lab, v in h["H5"]["values"]) + " |")
    L.append("| H6 misspecified correlation / heavy tails | DESCRIPTIVE | `tspec_ols` trend / seasonal coverage: "
             + "; ".join(f"{lab.split('_')[0]} {lab.split('_')[2]} {a:.3f} / {b:.3f}" for lab, a, b, _ in h["H6"]["values"]) + " |")
    L.append(f"| H7 `season='ols'` vs default ALLSSA, seasonal RMSE | DESCRIPTIVE | `ols` better in {h['H7']['ols_better']} of {h['H7']['of']} core cells"
             + (" → docs will recommend `season='ols'` for decomposition" if h["H7"]["docs_recommend_ols"] else "") + " |")
    L.append("| H8 steps | DESCRIPTIVE | see the step table below |")
    L.append("")
    L.append("### H2 detail (`tspec_ols` vs the better of `stl_lin` / `stl_bin`)\n")
    L.append("| cell | tspec trend | STL trend | tspec seasonal | STL seasonal | win |")
    L.append("|---|---|---|---|---|---|")
    for lab, a, b, c, d, w in h["H2"]["values"]:
        L.append(f"| {lab} | {a:.3f} | {b:.3f} | {c:.3f} | {d:.3f} | {'yes' if w else 'no'} |")
    L.append("\n### H3 detail (best of `tspec_ols` / `tspec_ols_dur` vs `harm_lowess`, total-signal RMSE)\n")
    L.append("| cell | tspec | harm_lowess | win |")
    L.append("|---|---|---|---|")
    for lab, a, b, w in h["H3"]["values"]:
        L.append(f"| {lab} | {a:.3f} | {b:.3f} | {'yes' if w else 'no'} |")
    L.append("\n## All cells: trend / seasonal / remainder RMSE at observation times\n")
    L.append("| cell | n | " + " | ".join(METHODS) + " |")
    L.append("|---|---|" + "---|" * len(METHODS))
    for lab, r in R.items():
        cells = []
        for m in METHODS:
            fail = r.get(f"{m}_fail", 0.0)
            s = f"{fmt(r.get(f'{m}_trend_rmse'))} / {fmt(r.get(f'{m}_seas_rmse'))} / {fmt(r.get(f'{m}_rem_rmse'))}"
            if fail and fail > 0:
                s += f" (fail {fail:.0%})"
            cells.append(s)
        L.append(f"| {lab} | {r['n']:.0f} | " + " | ".join(cells) + " |")
    L.append("\n## Bands, leakage, gaps and runtime (`tspec_ols` unless noted)\n")
    L.append("Grid NaN share (`tspec_ols`, `eval_times` call): " + ", ".join(
        f"{lab} {100 * r.get('tspec_ols_grid_nan', 0):.1f}%" for lab, r in R.items() if r.get("tspec_ols_grid_nan", 0) > 0) + "\n")
    L.append("| cell | trend cov | seasonal cov | default trend cov | leak seas (tspec_ols / stl_lin) | leak trend (tspec_ols / stl_lin) | gap trend RMSE (tspec_ols / stl_lin / harm_lowess) | sec (tspec_ols / default / stl_lin) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for lab, r in R.items():
        L.append(f"| {lab} | {fmt(r.get('tspec_ols_trend_cov'))} | {fmt(r.get('tspec_ols_seas_cov'))} | {fmt(r.get('tspec_default_trend_cov'))} | "
                 f"{fmt(r.get('tspec_ols_leak_seas'), 2)} / {fmt(r.get('stl_lin_leak_seas'), 2)} | "
                 f"{fmt(r.get('tspec_ols_leak_trend'), 2)} / {fmt(r.get('stl_lin_leak_trend'), 2)} | "
                 f"{fmt(r.get('tspec_ols_gap_trend_rmse'))} / {fmt(r.get('stl_lin_gap_trend_rmse'))} / {fmt(r.get('harm_lowess_gap_trend_rmse'))} | "
                 f"{fmt(r.get('tspec_ols_sec'), 2)} / {fmt(r.get('tspec_default_sec'), 2)} / {fmt(r.get('stl_lin_sec'), 3)} |")
    L.append("\n## H8: steps (trend RMSE within ±1 yr of the +0.9 step)\n")
    L.append("| cell | jump found | false jumps / series | tspec_jump | tspec_ols (no jump) | stl_lin | harm_lowess |")
    L.append("|---|---|---|---|---|---|---|")
    for lab, a, b, c, d, e, f in h["H8"]["values"]:
        L.append(f"| {lab} | {a:.2f} | {b:.2f} | {c:.3f} | {d:.3f} | {e:.3f} | {f:.3f} |")
    L.append("\n## Real record: Choptank River nitrate (USGS 01491000), ln(concentration), 1985–2023\n")
    L.append("Held-out prediction RMSE (20 random 80/20 splits; mean ± SD). Lower is better.\n")
    L.append("| method | all samples (n = 804) | routine samples only |")
    L.append("|---|---|---|")
    for m in METHODS:
        a, b = real["all"][m], real["routine"][m]
        def cv(x):
            s_ = f"{x['mean']:.4f} ± {x['sd']:.4f}"
            if x.get("frac_unpredicted", 0) > 0:
                s_ += f" ({100 * x['frac_unpredicted']:.1f}% of held-out points unpredicted)"
            if x["n_fail"]:
                s_ += f" (failed splits: {x['n_fail']})"
            return s_
        L.append(f"| `{m}` | {cv(a)} | {cv(b)} |")
    L.append("\n![Choptank](figures_decomp/choptank_decomposition.png)\n")
    L.append("Illustrative replicate per sampling regime (not part of the hypotheses):\n")
    L.append("![regimes](figures_decomp/regimes_example.png)\n")
    (REPORTS / "decomp_heldout_report.md").write_text("\n".join(L) + "\n")


def main():
    t_start = time.time()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    src_clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "src/"], cwd=ROOT).returncode == 0
    if not src_clean and not os.environ.get("DECOMP_ALLOW_DIRTY"):
        raise SystemExit("src/ differs from HEAD: the registered validation requires frozen code.")
    cells = build_cells()
    only = os.environ.get("DECOMP_ONLY")
    if only:
        cells = [c for c in cells if c["label"] in only.split(",")]
    R = {}
    for c in cells:
        t1 = time.time()
        with mp.get_context("fork").Pool(PROCS) as p:
            rows = p.map(cell_job, [(c, r) for r in range(NREP)])
        agg = aggregate(rows)
        agg.update({k: c[k] for k in ("group", "regime", "trend", "season", "noise", "id")})
        R[c["label"]] = agg
        print(f"{c['id']:2d} {c['label']:<16} n={agg['n']:.0f} ols tr {agg.get('tspec_ols_trend_rmse', np.nan):.3f} "
              f"se {agg.get('tspec_ols_seas_rmse', np.nan):.3f} cov {agg.get('tspec_ols_trend_cov', np.nan):.3f} | "
              f"stl_lin tr {agg['stl_lin_trend_rmse']:.3f} se {agg['stl_lin_seas_rmse']:.3f} | "
              f"hl tr {agg['harm_lowess_trend_rmse']:.3f} se {agg['harm_lowess_seas_rmse']:.3f} ({time.time() - t1:.0f}s)", flush=True)
    real = {}
    if not only:
        real = {"all": real_record_cv(False), "routine": real_record_cv(True)}
        print("real", json.dumps({k: {m: round(v["mean"], 4) for m, v in d.items()} for k, d in real.items()}), flush=True)
        real["figure"] = real_record_figure()
        example_figures()
    from importlib.metadata import version

    meta = dict(commit=commit, src_clean=src_clean, tspec=version("tspec"), numpy=np.__version__, scipy=scipy.__version__,
                statsmodels=statsmodels.__version__, pandas=pd.__version__, python=platform.python_version(),
                nrep=NREP, wall=time.time() - t_start)
    out = dict(meta=meta, cells=R, real=real)
    if not only:
        H = hypotheses(R)
        out["hypotheses"] = H
        write_report(R, H, real, meta)
        print(json.dumps({k: v["verdict"] for k, v in H.items()}), flush=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    name = "decomp_heldout_results.json" if not only else "decomp_heldout_debug.json"
    (REPORTS / name).write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
