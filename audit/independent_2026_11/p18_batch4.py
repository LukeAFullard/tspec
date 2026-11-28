"""P18: batch-4 checks: AUD-08 bootstrap band coverage (uneven density, heteroscedastic),
AUD-07 heavy tails under the new just_jumps default, README-pipeline band coverage."""
import json, sys, time, warnings
import numpy as np, tspec
from common import pmap, wilson
warnings.filterwarnings("ignore")
NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 150
out = {}
def stpd_band(a):
    seed, design, weighted = a
    rng = np.random.default_rng(seed); n = 200
    if design == "uneven":
        t = np.sort(np.r_[rng.uniform(0, 10, 150), rng.uniform(10, 20, 50)]); sd = np.full(n, 0.5)
    elif design == "hetero":
        t = np.sort(rng.uniform(0, 20, n)); sd = np.where(t >= 10, 2.0, 0.5)
    else:
        t = np.sort(rng.uniform(0, 20, n)); sd = np.full(n, 0.5)
    tr = 0.5*t - 2.0*np.maximum(t - 10, 0)
    y = tr + sd*rng.normal(size=n)
    r = tspec.stpd(t, y, weights=1/sd**2 if weighted else None)
    if not any(abs(p.time_year - 10) < 1 for p in r.turning_points) or len(r.turning_points) != 1:
        return dict(ok=0, cov=np.nan, cov_near=np.nan)
    pt = r.piecewise_trend; ins = (pt.trend_ci_low <= tr) & (tr <= pt.trend_ci_high)
    return dict(ok=1, cov=float(ins.mean()), cov_near=float(ins[np.abs(t-10) < 1].mean()))
def jj_tails(a):
    seed, dist, robust = a
    rng = np.random.default_rng(seed); t = np.sort(rng.uniform(0, 20, 240))
    e = rng.standard_t(3, 240)/np.sqrt(3) if dist == "t3" else rng.standard_t(2, 240)/np.sqrt(2)
    y = 0.05*t + np.cos(2*np.pi*t) + 0.4*np.sin(4*np.pi*t) + 0.5*e
    return len(tspec.just_jumps(t, y, season="ols", robust=robust).jumps) > 0
for design, weighted in [("uniform", False), ("uneven", False), ("hetero", False), ("hetero", True)]:
    t1 = time.time(); res = pmap(stpd_band, [(s, design, weighted) for s in range(NREP)])
    ok = [r for r in res if r["ok"]]
    d = dict(detected_exactly=len(ok)/NREP, cov=float(np.nanmean([r["cov"] for r in ok])), cov_near=float(np.nanmean([r["cov_near"] for r in ok])), sec=round(time.time()-t1, 1))
    out[f"stpd_bootstrap_band_{design}_w{weighted}"] = d; print(design, weighted, d, flush=True)
for dist in ["t3", "t2"]:
    for robust in ["none", "mm"]:
        res = pmap(jj_tails, [(s, dist, robust) for s in range(NREP)]); k = int(np.sum(res))
        out[f"just_jumps_default_{dist}_{robust}"] = dict(fwer=k/NREP, ci=wilson(k, NREP)); print(dist, robust, out[f"just_jumps_default_{dist}_{robust}"], flush=True)
json.dump(out, open("p18_results.json", "w"), indent=1)
