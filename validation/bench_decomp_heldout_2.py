"""Pre-registered held-out validation 2 of ``just_decompose`` (decoupled seasonal window).

Implements ``validation/decomposition_heldout_plan_2.md``. Reuses the comparators and metric
helpers of validation 1 (``bench_decomp_heldout.py``) unchanged.

    OMP_NUM_THREADS=1 MPLBACKEND=Agg python validation/bench_decomp_heldout_2.py
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import i0

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_decomp_heldout as B  # noqa: E402

import tspec  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = B.ROOT
REPORTS = B.REPORTS
NREP = int(os.environ.get("DECOMP_NREP", "100"))
PROCS = B.PROCS
SIG = 0.25
YD = B.YEAR_DAYS
CONOWINGO = ROOT / "validation" / "data" / "conowingo_01578310_nitrate.csv"
_CONO = None


def conowingo():
    global _CONO
    if _CONO is None:
        d = pd.read_csv(CONOWINGO, parse_dates=["date"]).reset_index(drop=True)
        d["t"] = B.to_decimal_year(d["date"])
        _CONO = d
    return _CONO


# ----------------------------------------------------------------------------- designs
def sample_times(q, rng):
    t0 = 2000.0
    if q == "Q1":
        return t0 + np.arange(0, 15, 14 / YD)
    if q == "Q2":
        return np.sort(t0 + rng.uniform(0, 25, rng.poisson(250)))
    if q == "Q3":
        routine = t0 + (np.arange(240) + 0.5) / 12
        ev = []
        for yr in range(20):
            for _ in range(6):
                start = t0 + yr + rng.uniform(0, 1)
                dur = rng.uniform(2, 5) / YD
                ev.append(start + rng.uniform(0, dur, rng.integers(3, 7)))
        t = np.unique(np.concatenate([routine, *ev]))
        return t[t < t0 + 20]
    if q == "Q4":
        m = np.arange(240)
        t = t0 + (m + 0.5) / 12
        winter = np.isin(m % 12, [11, 0, 1])
        keep = rng.random(240) < np.where(winter, 0.25, 0.95)
        return t[keep]
    if q == "Q5":
        return np.concatenate([t0 + (np.arange(120) + 0.5) / 12, np.sort(t0 + 10 + rng.uniform(0, 10, 40))])
    if q == "Q6":
        return conowingo()["t"].to_numpy().copy()
    raise ValueError(q)


def trend_u1(t, t0, L):
    return 1.5 + 0.02 * (t - t0) - 0.4 * np.tanh((t - t0 - L / 2) / 2.5)


def trend_u2(t, t0, L):
    return 1.5 + 0.01 * (t - t0) + 0.3 * np.sin(2 * np.pi * (t - t0) / 8.0)


def seas_v1(t, t0, L):
    return 0.6 * np.cos(2 * np.pi * (t - 0.1)) + 0.25 * np.cos(4 * np.pi * (t - 0.3)) + 0.1 * np.cos(6 * np.pi * t)


def seas_v2(t, t0, L):
    u = (t - t0) / L
    a, phi = 1.0 - 0.6 * u, 0.8 * u
    return a * (np.cos(2 * np.pi * t - phi) + 0.3 * np.cos(4 * np.pi * t - 2 * phi))


def seas_v3(t, t0, L):
    return 0.9 * (np.exp(6 * np.cos(2 * np.pi * (t - 0.6))) - i0(6)) / (np.exp(6) - i0(6))


TR = {"U1": trend_u1, "U2": trend_u2}
SE = {"V1": seas_v1, "V2": seas_v2, "V3": seas_v3}


def noise(kind, t, rng):
    return SIG * rng.normal(size=t.size) if kind == "W" else B.ou(t, 60 / YD, SIG, rng)


def build_cells():
    cells = []
    for q in ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]:
        for tr, se in [("U1", "V1"), ("U1", "V2"), ("U2", "V3")]:
            cells.append(dict(group="core", q=q, tr=tr, se=se, noise="W"))
    for q in ["Q2", "Q3", "Q6"]:
        cells.append(dict(group="noise", q=q, tr="U1", se="V1", noise="O"))
    for i, c in enumerate(cells, start=1):
        c["id"] = i
        c["label"] = f"{c['q']}_{c['tr']}{c['se']}_{c['noise']}"
    return cells


TSPEC = {
    "tspec_default": {},
    "tspec_ols": {"season": "ols"},
    "tspec_legacy_ols": {"season": "ols", "seasonal_window": None},
}
OTHER = ["stl_lin", "stl_bin", "harm_lowess"]


def cell_job(args):
    c, rep = args
    rng = np.random.default_rng(940_000 + 1_000 * c["id"] + rep)
    t = sample_times(c["q"], rng)
    t0, L = float(t[0]), float(t[-1] - t[0])
    tr, se = TR[c["tr"]](t, t0, L), SE[c["se"]](t, t0, L)
    e = noise(c["noise"], t, rng)
    y = tr + se + e
    row = {"n": int(t.size)}
    for cfg, kw in TSPEC.items():
        t1 = time.perf_counter()
        try:
            d = tspec.just_decompose(t, y, **kw)
        except Exception as exc:  # noqa: BLE001
            row[f"{cfg}_fail"] = 1.0
            row[f"{cfg}_error"] = type(exc).__name__
            continue
        row[f"{cfg}_sec"] = time.perf_counter() - t1
        row[f"{cfg}_fail"] = 0.0
        ok = np.isfinite(d.trend) & np.isfinite(d.seasonal)
        m = B.metrics(t[ok], tr[ok], se[ok], e[ok], d.trend[ok], d.seasonal[ok])
        m["nan_rate"] = float(np.mean(~ok))
        m["blowup_rate"] = float(np.mean(np.abs(d.trend[ok] - tr[ok]) > 1.0)) if ok.any() else 1.0
        m["trend_cov"] = float(np.mean((d.trend_ci_low <= tr) & (tr <= d.trend_ci_high)))
        m["seas_cov"] = float(np.mean((d.seasonal_ci_low <= se) & (se <= d.seasonal_ci_high)))
        if cfg == "tspec_ols":
            try:
                dg = tspec.just_decompose(t, y, eval_times=B.eval_grid(t), **kw)
                m["eval_maxdiff"] = float(np.nanmax(np.abs(dg.trend - d.trend)))
            except Exception:  # noqa: BLE001
                m["eval_maxdiff"] = float("inf")
        row.update({f"{cfg}_{k}": v for k, v in m.items()})
    for cfg in OTHER:
        r = B.run_harm_lowess(t, y) if cfg == "harm_lowess" else B.run_stl(t, y, "bin" if cfg == "stl_bin" else "lin")
        m = B.metrics(t, tr, se, e, r["trend"], r["seasonal"])
        row.update({f"{cfg}_{k}": v for k, v in m.items()})
    return row


# ----------------------------------------------------------------------------- real records
def cv(d, seed0, routine_only):
    if routine_only:
        d = d[d["event"] == "Routine sample"].reset_index(drop=True)
    t = d["t"].to_numpy()
    y = np.log(d["value"].to_numpy())
    cens = d["censored"].to_numpy().astype(bool)
    y_sub = np.where(cens, np.log(0.5 * d["value"].to_numpy()), y)
    res = {k: [] for k in [*TSPEC, *OTHER]}
    miss = {k: [] for k in TSPEC}
    for split in range(20):
        rng = np.random.default_rng(seed0 + split)
        test = rng.random(t.size) < 0.20
        train = ~test
        tt, ty = t[train], y[train]
        score = test & ~cens & (t >= tt[0]) & (t <= tt[-1])
        cs = np.where(cens[train], -1, 0)
        for cfg, kw in TSPEC.items():
            try:
                kk = dict(kw)
                if cs.any():
                    kk["censor_status"] = cs
                dd = tspec.just_decompose(tt, ty, eval_times=t[score], **kk)
                v, fr = B.rmse_finite(dd.trend_reg + dd.seasonal_reg, y[score])
            except Exception:  # noqa: BLE001
                v, fr = np.nan, 1.0
            res[cfg].append(v)
            miss[cfg].append(fr)
        for cfg in OTHER:
            ys = y_sub[train]
            if cfg == "harm_lowess":
                r = B.run_harm_lowess(tt, ys)
                coef = np.linalg.lstsq(B.harm_design(tt), ys - r["trend"], rcond=None)[0]
                pred = np.interp(t[score], tt, r["trend"]) + B.harm_design(t[score])[:, 1:] @ coef[1:]
            else:
                r = B.run_stl(tt, ys, "bin" if cfg == "stl_bin" else "lin")
                pred = np.interp(t[score], r["grid"], r["trend_grid"]) + np.interp(t[score], r["grid"], r["seas_grid"])
            res[cfg].append(B.rmse(pred, y[score]))
    return {k: dict(mean=float(np.nanmean(v)), sd=float(np.nanstd(v)), n_fail=int(np.sum(~np.isfinite(v))),
                    frac_unpredicted=float(np.mean(miss[k])) if k in miss else 0.0) for k, v in res.items()}


# ----------------------------------------------------------------------------- hypotheses
def hypotheses(R):
    core = [k for k in R if R[k]["group"] == "core"]
    H = {}

    def stl(k, m):
        return min(R[k][f"stl_lin_{m}"], R[k][f"stl_bin_{m}"])

    for hid, metric, need in [("G1", "seas_rmse", 15), ("G2", "rem_rmse", 15), ("G5", "trend_rmse", 12)]:
        vals = [(k, R[k][f"tspec_ols_{metric}"], stl(k, metric)) for k in core]
        wins = sum(a < b for _, a, b in vals)
        H[hid] = dict(values=vals, wins=wins, need=need, of=len(core), verdict="PASS" if wins >= need else "FAIL")
    vals = [(k, cfg, R[k].get(f"{cfg}_blowup_rate", 1.0), R[k].get(f"{cfg}_nan_rate", 1.0), R[k].get(f"{cfg}_fail", 1.0))
            for k in R for cfg in ("tspec_ols", "tspec_default")]
    H["G3"] = dict(values=vals, verdict="PASS" if all(b <= 0.005 and nr == 0 and f == 0 for _, _, b, nr, f in vals) else "FAIL")
    vals = [(k, R[k].get("tspec_ols_eval_maxdiff", np.inf)) for k in R]
    H["G4"] = dict(values=vals, verdict="PASS" if all(v < 1e-8 for _, v in vals) else "FAIL")
    vals = [(k, R[k]["tspec_ols_trend_cov"], R[k]["tspec_ols_seas_cov"]) for k in core if R[k]["se"] == "V1"]
    H["G6"] = dict(values=vals, verdict="PASS" if all(0.90 <= a <= 0.98 and 0.90 <= b <= 0.98 for _, a, b in vals) else "FAIL")
    vals = [(k, R[k]["tspec_ols_seas_rmse"], R[k]["harm_lowess_seas_rmse"]) for k in core if R[k]["se"] == "V2"]
    wins = sum(a < b for _, a, b in vals)
    H["G7"] = dict(values=vals, wins=wins, verdict="PASS" if wins >= 5 else "FAIL")
    H["D1"] = dict(values=[(k, R[k]["tspec_ols_seas_rmse"] / R[k]["harm_lowess_seas_rmse"]) for k in core if R[k]["se"] != "V2"],
                   verdict="DESCRIPTIVE")
    H["overall"] = "PASS" if all(H[g]["verdict"] == "PASS" for g in ("G1", "G2", "G3", "G4")) else "FAIL"
    return H


def f3(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.3f}"


def write_report(R, H, real, meta):
    L = ["# Held-out decomposition validation 2: decoupled seasonal window\n",
         "Pre-registered in [`validation/decomposition_heldout_plan_2.md`](../decomposition_heldout_plan_2.md) before the change was "
         f"implemented. Code under test: commit `{meta['commit']}` (`src/` clean: {meta['src_clean']}). {NREP} replicates per cell. "
         f"Versions: tspec {meta['tspec']}, numpy {np.__version__}, statsmodels {meta['statsmodels']}, python {platform.python_version()}. "
         f"Wall time {meta['wall']:.0f} s.\n",
         f"## Overall verdict for decomposing unevenly sampled data: **{H['overall']}** (requires G1–G4)\n",
         "| ID | Verdict | Result |", "|---|---|---|"]
    for g, name in [("G1", "seasonal RMSE < best STL"), ("G2", "remainder RMSE < best STL"), ("G5", "trend RMSE < best STL (secondary)")]:
        L.append(f"| {g} {name} | **{H[g]['verdict']}** | {H[g]['wins']} of {H[g]['of']} core cells (need ≥ {H[g]['need']}) |")
    bad = [f"{k} {c}: blow-up {b:.2%}, NaN {nr:.2%}, fail {f:.0%}" for k, c, b, nr, f in H["G3"]["values"] if not (b <= 0.005 and nr == 0 and f == 0)]
    L.append(f"| G3 no blow-ups / NaN | **{H['G3']['verdict']}** | {'; '.join(bad) if bad else 'all 21 cells clean for both configurations'} |")
    L.append(f"| G4 eval_times consistency | **{H['G4']['verdict']}** | max difference {max(v for _, v in H['G4']['values']):.2e} |")
    L.append(f"| G6 bands 90–98% (V1, white) | **{H['G6']['verdict']}** | " + ", ".join(f"{k.split('_')[0]} {a:.3f}/{b:.3f}" for k, a, b in H["G6"]["values"]) + " |")
    L.append(f"| G7 evolving seasonality vs pooled baseline | **{H['G7']['verdict']}** | {H['G7']['wins']} of 6 |")
    L.append("| D1 fixed shapes vs pooled baseline | DESCRIPTIVE | seasonal RMSE ratio tspec_ols / harm_lowess: "
             + ", ".join(f"{k.split('_')[0]}{k.split('_')[1][2:]} {v:.2f}" for k, v in H["D1"]["values"]) + " |")
    L.append("")
    meths = [*TSPEC, *OTHER]
    L.append("## Trend / seasonal / remainder RMSE at observation times\n")
    L.append("| cell | n | " + " | ".join(meths) + " |")
    L.append("|---|---|" + "---|" * len(meths))
    for k, r in R.items():
        L.append(f"| {k} | {r['n']:.0f} | " + " | ".join(
            f"{f3(r.get(f'{m}_trend_rmse'))} / {f3(r.get(f'{m}_seas_rmse'))} / {f3(r.get(f'{m}_rem_rmse'))}" for m in meths) + " |")
    L.append("\n## Bands, blow-ups and runtime\n")
    L.append("| cell | tspec_ols trend / seasonal cov | blow-up rate (ols / default / legacy) | sec (ols / legacy) |")
    L.append("|---|---|---|---|")
    for k, r in R.items():
        L.append(f"| {k} | {f3(r.get('tspec_ols_trend_cov'))} / {f3(r.get('tspec_ols_seas_cov'))} | "
                 f"{100 * r.get('tspec_ols_blowup_rate', np.nan):.2f}% / {100 * r.get('tspec_default_blowup_rate', np.nan):.2f}% / "
                 f"{100 * r.get('tspec_legacy_ols_blowup_rate', np.nan):.2f}% | {r.get('tspec_ols_sec', np.nan):.2f} / {r.get('tspec_legacy_ols_sec', np.nan):.2f} |")
    L.append("\n## Real records: held-out prediction RMSE of ln(nitrate), 20 random 80/20 splits\n")
    L.append("| method | Conowingo all (registered R) | Conowingo routine | Choptank all (descriptive) | Choptank routine (descriptive) |")
    L.append("|---|---|---|---|---|")
    for m in meths:
        def c(x):
            s = f"{x['mean']:.4f} ± {x['sd']:.4f}"
            if x.get("frac_unpredicted", 0) > 0:
                s += f" ({100 * x['frac_unpredicted']:.1f}% unpredicted)"
            return s
        L.append(f"| `{m}` | {c(real['cono_all'][m])} | {c(real['cono_routine'][m])} | {c(real['chop_all'][m])} | {c(real['chop_routine'][m])} |")
    ra = real["cono_all"]
    verdict = "PASS" if ra["tspec_ols"]["mean"] <= min(ra["stl_lin"]["mean"], ra["stl_bin"]["mean"]) else "FAIL"
    L.append(f"\n**R (Conowingo, `tspec_ols` ≤ best STL): {verdict}**\n")
    (REPORTS / "decomp_heldout_2_report.md").write_text("\n".join(L) + "\n")
    return verdict


def main():
    t_start = time.time()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    src_clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "src/"], cwd=ROOT).returncode == 0
    if not src_clean and not os.environ.get("DECOMP_ALLOW_DIRTY"):
        raise SystemExit("src/ differs from HEAD: the registered validation requires committed code.")
    cells = build_cells()
    only = os.environ.get("DECOMP_ONLY")
    if only:
        cells = [c for c in cells if c["label"] in only.split(",")]
    R = {}
    for c in cells:
        t1 = time.time()
        with mp.get_context("fork").Pool(PROCS) as p:
            rows = p.map(cell_job, [(c, r) for r in range(NREP)])
        agg = B.aggregate(rows)
        agg.update({k: c[k] for k in ("group", "q", "tr", "se", "noise", "id")})
        R[c["label"]] = agg
        print(f"{c['id']:2d} {c['label']:<12} n={agg['n']:.0f} done ({time.time() - t1:.0f}s)", flush=True)
    meta = dict(commit=commit, src_clean=src_clean, tspec=tspec.__version__ if hasattr(tspec, "__version__") else "?",
                statsmodels=B.statsmodels.__version__, wall=0.0)
    out = dict(meta=meta, cells=R)
    if not only:
        real = {"cono_all": cv(conowingo(), 950_000, False), "cono_routine": cv(conowingo(), 950_000, True),
                "chop_all": cv(B.choptank(), 920_000, False), "chop_routine": cv(B.choptank(), 920_000, True)}
        H = hypotheses(R)
        meta["wall"] = time.time() - t_start
        H["R"] = write_report(R, H, real, meta)
        out.update(hypotheses=H, real=real)
        print(json.dumps({k: (v["verdict"] if isinstance(v, dict) else v) for k, v in H.items()}), flush=True)
    name = "decomp_heldout_2_results.json" if not only else "decomp_heldout_2_debug.json"
    (REPORTS / name).write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
