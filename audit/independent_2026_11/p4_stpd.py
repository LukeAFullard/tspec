"""P4: STPD null FPR, bend power, NDRI jump-vs-bend discrimination, band coverage."""
import json, sys, time, warnings
import numpy as np, tspec
from common import ar1, pmap, wilson
warnings.filterwarnings("ignore")
NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 200

def gen(seed, n=200, span=20.0, noise="gauss", rho=0.0, jump=0.0, bend=0.0, sd=0.5, density="uniform", hetero=1.0):
    rng = np.random.default_rng(seed)
    if density == "uniform":
        t = np.sort(rng.uniform(0, span, n))
    else:  # 3x denser first half
        n1 = int(0.75*n); t = np.sort(np.r_[rng.uniform(0, span/2, n1), rng.uniform(span/2, span, n-n1)])
    sdv = np.where(t >= span/2, sd*hetero, sd)
    if noise == "gauss": e = rng.normal(0, 1, n)
    elif noise == "ar1": e = ar1(rng, n, rho)
    elif noise == "t3": e = rng.standard_t(3, n)/np.sqrt(3)
    elif noise == "t2": e = rng.standard_t(2, n)/np.sqrt(2)
    tr = 0.5*t + jump*(t >= span/2) - bend*np.maximum(t - span/2, 0)
    return t, tr + sdv*e, tr, sdv

def job(a):
    seed, kw, sk = a
    t, y, tr, sdv = gen(seed, **kw)
    if sk.pop("use_weights", False): sk["weights"] = 1/sdv**2
    try:
        r = tspec.stpd(t, y, **sk)
    except Exception as ex:
        return dict(err=repr(ex)[:80])
    tps = [p.time_year for p in r.turning_points]; fl = [f.time_year for f in r.flagged_jumps]
    near = [x for x in tps if abs(x - 10) < 1.0]
    res = dict(ntp=len(tps), hit=int(bool(near)), loc=float(near[0]-10) if near else np.nan, nflag_near=int(any(abs(x-10) < 1 for x in fl)))
    pt = r.piecewise_trend
    if pt.trend_ci_low is not None:
        ins = (pt.trend_ci_low <= tr) & (tr <= pt.trend_ci_high)
        res["cov"] = float(ins.mean()); res["cov_near"] = float(ins[np.abs(t-10) < 1].mean())
    return res

cases = {
    "null_white": (dict(), dict()),
    "null_white_fastcond": (dict(), dict(band_method="conditional")),
    "null_ar1_0.5": (dict(noise="ar1", rho=0.5), dict(band_method="conditional")),
    "null_t3": (dict(noise="t3"), dict(band_method="conditional")),
    "null_t3_mm": (dict(noise="t3"), dict(band_method="conditional", robust="mm")),
    "null_t2": (dict(noise="t2"), dict(band_method="conditional")),
    "null_t2_mm": (dict(noise="t2"), dict(band_method="conditional", robust="mm")),
    "null_uneven_density": (dict(density="dense_first"), dict(band_method="conditional")),
    "null_hetero4x_unweighted": (dict(hetero=4.0), dict(band_method="conditional")),
    "null_hetero4x_weighted": (dict(hetero=4.0), dict(band_method="conditional", use_weights=True)),
    "bend1.0": (dict(bend=1.0), dict()),
    "bend1.0_uneven_density": (dict(bend=1.0, density="dense_first"), dict(band_method="conditional")),
    "bend1.0_hetero4x_unweighted": (dict(bend=1.0, hetero=4.0), dict(band_method="conditional")),
    "bend1.0_hetero4x_weighted": (dict(bend=1.0, hetero=4.0), dict(band_method="conditional", use_weights=True)),
    "bend1.0_rmse_vs_l2_l2": (dict(bend=1.0, density="dense_first"), dict(band_method="conditional", ndri_mode="l2")),
    "step2.0_only": (dict(jump=2.0), dict(band_method="conditional")),
    "step2.0_only_l2": (dict(jump=2.0), dict(band_method="conditional", ndri_mode="l2")),
    "step2.0_uneven_density": (dict(jump=2.0, density="dense_first"), dict(band_method="conditional")),
    "step2.0_uneven_density_l2": (dict(jump=2.0, density="dense_first"), dict(band_method="conditional", ndri_mode="l2")),
}
out = {"NREP": NREP}
for name, (kw, sk) in cases.items():
    t1 = time.time()
    nrep = NREP if "fastcond" not in name and sk.get("band_method") == "conditional" else max(100, NREP//2)
    res = pmap(job, [(s, kw, dict(sk)) for s in range(nrep)])
    errs = [r for r in res if "err" in r]; ok = [r for r in res if "err" not in r]
    any_tp = sum(r["ntp"] > 0 for r in ok); hit = sum(r["hit"] for r in ok)
    d = dict(nrep=nrep, errors=len(errs), err_example=errs[0]["err"] if errs else None,
             any_tp_rate=any_tp/len(ok), any_tp_ci=wilson(any_tp, len(ok)), tp_near_10_rate=hit/len(ok),
             flagged_jump_near_10=float(np.mean([r["nflag_near"] for r in ok])),
             loc_mad=float(np.nanmedian(np.abs([r["loc"] for r in ok]))) if hit else None,
             sec=round(time.time()-t1, 1))
    if "cov" in ok[0]:
        hits = [r for r in ok if r["hit"]] if "bend" in name else ok
        if hits:
            d["band_cov_given_detection"] = float(np.mean([r["cov"] for r in hits])); d["band_cov_near_tp"] = float(np.mean([r["cov_near"] for r in hits]))
    out[name] = d; print(name, d, flush=True)
json.dump(out, open("p4_results.json", "w"), indent=1, default=float)
