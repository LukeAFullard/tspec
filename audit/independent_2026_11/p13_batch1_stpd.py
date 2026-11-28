"""P13: post-fix validation of the STPD step-vs-bend gate (AUD-03/06) vs the NDRI gate."""
import json, sys, numpy as np, tspec, warnings
from common import pmap, wilson
from common import ar1
warnings.filterwarnings("ignore")

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

NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 200
def job(a):
    seed, kw, sk = a
    t, y, tr, sdv = gen(seed, **kw)
    if sk.pop("use_weights", False): sk["weights"] = 1/sdv**2
    r = tspec.stpd(t, y, band_method="conditional", **sk)
    tps = [p.time_year for p in r.turning_points]; fl = [f.time_year for f in r.flagged_jumps]
    return dict(any_tp=len(tps) > 0, tp_near=any(abs(x-10) < 1 for x in tps), flag_near=any(abs(x-10) < 1 for x in fl))
cases_all = {"null_white": dict(), "null_t3": dict(noise="t3"), "null_hetero4x_unweighted": dict(hetero=4.0),
         "bend1.0": dict(bend=1.0), "bend1.0_uneven": dict(bend=1.0, density="dense_first"), "bend2.0": dict(bend=2.0),
         "bend1.0_hetero4x_unweighted": dict(bend=1.0, hetero=4.0), "bend1.0_sd0.05": dict(bend=1.0, sd=0.05),
         "step1.0": dict(jump=1.0), "step2.0": dict(jump=2.0), "step2.0_uneven": dict(jump=2.0, density="dense_first"), "step3.0": dict(jump=3.0)}
import os
only = os.environ.get("P13_ONLY")
cases = {k: v for k, v in cases_all.items() if (only is None or any(o in k for o in only.split(",")))}
out = {}
for name, kw in cases.items():
    for jt in os.environ.get("P13_JT", "step,ndri").split(","):
        sk = dict(jump_test=jt)
        if os.environ.get("STEP_ALPHA") and jt == "step": sk["step_alpha"] = float(os.environ["STEP_ALPHA"])
        if "weighted" in name and "unweighted" not in name: sk["use_weights"] = True
        res = pmap(job, [(s, kw, dict(sk)) for s in range(NREP)])
        d = {k: float(np.mean([r[k] for r in res])) for k in res[0]}
        out[f"{name}|{jt}"] = d; print(name, jt, d, flush=True)
json.dump(out, open(os.environ.get("P13_OUT", "p13_results.json"), "w"), indent=1)
