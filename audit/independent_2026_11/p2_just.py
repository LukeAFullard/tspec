"""P2: JUST just_jumps FWER/power and just_decompose CI coverage (independent)."""
import json, sys, time, warnings
import numpy as np, tspec
from common import ar1, pmap, wilson
warnings.filterwarnings("ignore")
NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 300

def gen(seed, n=240, span=20.0, noise="gauss", rho=0.0, jump=0.0, bend=0.0, sd=0.5):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, span, n))
    if noise == "gauss": e = rng.normal(0, sd, n)
    elif noise == "ar1": e = sd * ar1(rng, n, rho)
    elif noise == "t3": e = sd * rng.standard_t(3, n) / np.sqrt(3)
    elif noise == "t2": e = sd * rng.standard_t(2, n)
    seas = 1.0*np.cos(2*np.pi*t) + 0.4*np.sin(4*np.pi*t)
    tr = 0.05*t + jump*(t >= span/2) + bend*np.maximum(t - span/2, 0)
    return t, tr + seas + e, tr, seas

def job_null(a):
    seed, kw, jk = a
    t, y, *_ = gen(seed, **kw)
    try:
        r = tspec.just_jumps(t, y, **jk)
        return len(r.jumps)
    except Exception as ex:
        return -1

def job_power(a):
    seed, kw, jk = a
    t, y, tr, _ = gen(seed, **kw)
    r = tspec.just_jumps(t, y, **jk)
    tj = kw.get("span", 20.0)/2
    hits = [j for j in r.jumps if abs(r.times_numeric[j.valid_index] - tj) < 0.5] if r.jumps else []
    if not hits: return dict(hit=0, n=len(r.jumps), mag=np.nan, loc=np.nan)
    j = hits[0]
    return dict(hit=1, n=len(r.jumps), mag=float(j.magnitude), loc=float(r.times_numeric[j.valid_index]-tj))

out = {"NREP": NREP}
null_cases = {
    "white_allssa": (dict(), dict()),
    "white_ols": (dict(), dict(season="ols")),
    "white_long_n720": (dict(n=720, span=60.0), dict(season="ols")),
    "ar1_0.3_ols": (dict(noise="ar1", rho=0.3), dict(season="ols")),
    "ar1_0.6_ols": (dict(noise="ar1", rho=0.6), dict(season="ols")),
    "t3_ols": (dict(noise="t3"), dict(season="ols")),
    "t3_ols_mm": (dict(noise="t3"), dict(season="ols", robust="mm")),
    "t2_ols_mm": (dict(noise="t2"), dict(season="ols", robust="mm")),
    "bend_only_ols": (dict(bend=0.4), dict(season="ols")),
    "bend_only_ols_stepalpha": (dict(bend=0.4), dict(season="ols", step_alpha=0.05)),
}
for name, (kw, jk) in null_cases.items():
    t1 = time.time()
    res = np.array(pmap(job_null, [(s, kw, jk) for s in range(NREP)]))
    k = int(np.sum(res > 0)); err = int(np.sum(res < 0))
    out[f"fwer_{name}"] = dict(rate=k/NREP, ci95=wilson(k, NREP), errors=err, sec=round(time.time()-t1, 1))
    print(name, out[f"fwer_{name}"], flush=True)

for name, (kw, jk) in {"jump1.0_sd0.5_allssa": (dict(jump=1.0), dict()), "jump1.0_sd0.5_ols": (dict(jump=1.0), dict(season="ols")),
                        "jump0.5_sd0.5_ols": (dict(jump=0.5), dict(season="ols")), "jump1.0_t3_mm": (dict(jump=1.0, noise="t3"), dict(season="ols", robust="mm"))}.items():
    res = pmap(job_power, [(s, kw, jk) for s in range(NREP)])
    h = np.array([r["hit"] for r in res]); m = np.array([r["mag"] for r in res]); l = np.array([r["loc"] for r in res])
    out[f"power_{name}"] = dict(recall=float(h.mean()), mag_bias=float(np.nanmean(m) - kw["jump"]), mag_rmse=float(np.sqrt(np.nanmean((m-kw["jump"])**2))),
                                loc_mad_yr=float(np.nanmedian(np.abs(l))), mean_reported=float(np.mean([r["n"] for r in res])))
    print(name, out[f"power_{name}"], flush=True)

# ---- just_decompose trend-band coverage under regimes
def job_cov(a):
    seed, regime = a
    rng = np.random.default_rng(10_000 + seed)
    n, span = 240, 20.0
    t = np.sort(rng.uniform(0, span, n))
    if regime == "gapped":
        keep = ~((t > 8) & (t < 9.5)); t = t[keep]
    if regime == "linear": tr = 0.1*t
    elif regime == "curved": tr = 1.5*np.sin(2*np.pi*t/9.0)
    elif regime == "gapped": tr = 1.5*np.sin(2*np.pi*t/9.0)
    elif regime in ("break_known", "break_unknown"): tr = 0.1*t + 1.5*(t >= 10)
    seas = np.cos(2*np.pi*t) + 0.4*np.sin(4*np.pi*t)
    y = tr + seas + rng.normal(0, 0.4, t.size)
    ji = [int(np.searchsorted(t, 10.0))] if regime == "break_known" else None
    d = tspec.just_decompose(t, y, jump_indices=ji, season="ols")
    inside = (d.trend_ci_low <= tr) & (tr <= d.trend_ci_high)
    near = np.abs(t - 10.0) < 0.75
    seas_in = (d.seasonal_ci_low <= seas) & (seas <= d.seasonal_ci_high)
    return dict(cov=float(inside.mean()), cov_near=float(inside[near].mean()) if near.any() else np.nan,
                rmse=float(np.sqrt(np.mean((d.trend - tr)**2))), rmse_bc=float(np.sqrt(np.mean((d.trend_bc - tr)**2))),
                scov=float(seas_in.mean()), width=float(np.mean(d.trend_ci_high - d.trend_ci_low)),
                smear_near=float(np.sqrt(np.mean((d.trend[near] - tr[near])**2))) if near.any() else np.nan)
for regime in ["linear", "curved", "gapped", "break_known", "break_unknown"]:
    res = pmap(job_cov, [(s, regime) for s in range(200)])
    agg = {k: float(np.nanmean([r[k] for r in res])) for k in res[0]}
    out[f"decomp_cov_{regime}"] = agg
    print(regime, agg, flush=True)
json.dump(out, open("p2_results.json", "w"), indent=1, default=float)
