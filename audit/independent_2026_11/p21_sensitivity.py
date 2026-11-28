"""P21: sensitivity of the hand-chosen constants of the remediation (each varied about +/-50%).

New designs (not used for tuning): 300 random times over 25 years, trend 0.03/yr plus an 11-year
wave of amplitude 0.4, annual + semi-annual seasonality, noise sd 0.4. Question: does any
conclusion (false-alarm control, power, step/bend separation, decomposition accuracy) depend on
the exact value chosen?
"""
import json
import sys

import numpy as np

import tspec
import tspec._serial as S
import tspec.just as J
import tspec.stpd as ST
from common import ar1, pmap, wilson

sys.path.insert(0, "../../validation")
import bench_decomp_heldout_2 as C  # noqa: E402

NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 150
SD = 0.4


def base(rng, n=300):
    t = np.sort(2000 + rng.uniform(0, 25, n))
    tr = 0.03 * (t - 2000) + 0.4 * np.sin(2 * np.pi * (t - 2000) / 11)
    se = 0.7 * np.cos(2 * np.pi * t) + 0.25 * np.sin(4 * np.pi * t + 0.4)
    return t, tr, se


def apply(patch):
    for mod, name, val in patch:
        setattr({"S": S, "J": J, "ST": ST}[mod], name, val)


def jj_job(a):
    patch, scen, seed = a
    apply(patch)
    rng = np.random.default_rng(seed)
    t, tr, se = base(rng)
    e = SD * (ar1(rng, t.size, 0.5) if scen == "null_ar0.5" else rng.normal(size=t.size))
    step = 3 * SD * (t >= 2012.3) if scen == "step3sd" else 0.0
    bend = -0.5 * np.maximum(t - 2012.3, 0) if scen == "bend_precise" else 0.0
    if scen == "bend_precise":
        e = e * 0.05
    r = tspec.just_jumps(t, tr + se + e + step + bend, season="ols")
    tj = [r.times_numeric[j.valid_index] for j in r.jumps]
    if scen == "step3sd":
        return any(abs(x - 2012.3) < 0.5 for x in tj)
    return len(tj) > 0


def stpd_job(a):
    patch, scen, seed = a
    apply(patch)
    rng = np.random.default_rng(seed)
    t = np.sort(2000 + rng.uniform(0, 20, 240))
    if scen == "step4sd":
        y = 0.02 * (t - 2000) + 4 * SD * (t >= 2010.3)
    else:
        y = 0.25 * (t - 2000) - 0.5 * np.maximum(t - 2010.3, 0)
    y = y + SD * rng.normal(size=t.size)
    r = tspec.stpd(t, y)
    tp = any(abs(p.time_year - 2010.3) < 1 for p in r.turning_points)
    fl = any(abs(f.time_year - 2010.3) < 1 for f in r.flagged_jumps)
    return {"tp": tp, "flag": fl}


def dec_job(a):
    sw, q, seed = a
    rng = np.random.default_rng(seed)
    t = C.sample_times(q, rng)
    t0, L = float(t[0]), float(t[-1] - t[0])
    u = (t - t0) / L
    tr = 1.5 + 0.02 * (t - t0) + 0.3 * np.sin(2 * np.pi * (t - t0) / 11)
    se = (0.6 + 0.3 * u) * np.cos(2 * np.pi * t - 0.5 * u) + 0.2 * np.cos(4 * np.pi * t)
    y = tr + se + 0.25 * rng.normal(size=t.size)
    d = tspec.just_decompose(t, y, season="ols", seasonal_window=sw)
    return float(np.sqrt(np.mean((d.seasonal - se) ** 2))), float(np.sqrt(np.mean((d.trend - tr) ** 2)))


out = {}
serial_consts = [("RHO_MARGIN_SE", 1.0, [0.5, 1.5]), ("RHO_SE0_COEF", 0.9, [0.45, 1.35]),
                 ("JUMP_Z", 6.0, [3.0, 9.0]), ("SMOOTH_KNOTS", 4, [2, 6])]
configs = [("baseline", [])] + [(f"{n}={v}", [("S", n, v)]) for n, _, vs in serial_consts for v in vs]
configs += [("STEP_NEIGHBOURHOOD=1", [("J", "STEP_NEIGHBOURHOOD", 1)]), ("STEP_NEIGHBOURHOOD=3", [("J", "STEP_NEIGHBOURHOOD", 3)])]
for name, patch in configs:
    scens = ["null_white", "null_ar0.5", "step3sd"] if not name.startswith("STEP_N") else ["bend_precise", "step3sd"]
    if name == "baseline":
        scens = ["null_white", "null_ar0.5", "step3sd", "bend_precise"]
    for scen in scens:
        res = pmap(jj_job, [(patch, scen, 960_000 + s) for s in range(NREP)])
        k = int(np.sum(res))
        out[f"just_jumps|{name}|{scen}"] = dict(rate=k / NREP, ci=wilson(k, NREP))
        print(f"just_jumps {name:<22} {scen:<13} {k / NREP:.3f} {np.round(wilson(k, NREP), 3).tolist()}", flush=True)
for name, patch in [("baseline", []), ("STEP_SHARE_MIN=0.125", [("ST", "STEP_SHARE_MIN", 0.125)]),
                    ("STEP_SHARE_MIN=0.375", [("ST", "STEP_SHARE_MIN", 0.375)])]:
    for scen in ["step4sd", "bend"]:
        res = pmap(stpd_job, [(patch, scen, 970_000 + s) for s in range(NREP)])
        d = {k: float(np.mean([r[k] for r in res])) for k in ("tp", "flag")}
        out[f"stpd|{name}|{scen}"] = d
        print(f"stpd {name:<22} {scen:<8} TP {d['tp']:.3f} flagged-as-step {d['flag']:.3f}", flush=True)
for sw in [5.0, 7.0, 10.0]:
    for q in ["Q2", "Q4", "Q6"]:
        res = pmap(dec_job, [(sw, q, 980_000 + s) for s in range(100)])
        a = np.array(res)
        out[f"decompose|seasonal_window={sw}|{q}"] = dict(seas_rmse=float(a[:, 0].mean()), trend_rmse=float(a[:, 1].mean()))
        print(f"decompose seasonal_window={sw:<5} {q} seasonal {a[:, 0].mean():.3f} trend {a[:, 1].mean():.3f}", flush=True)
json.dump(out, open("p21_results.json", "w"), indent=1)
