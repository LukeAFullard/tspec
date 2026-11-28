"""P10: analyze_series on a pure step (no slope change) - is it reported as a jump or a turning point?"""
import json, sys, numpy as np, tspec, warnings
from common import pmap
warnings.filterwarnings("ignore")
def job(a):
    seed, jump, seas = a
    rng = np.random.default_rng(seed); t = np.sort(rng.uniform(0, 20, 240))
    y = 0.5*t + jump*(t >= 10) + (np.cos(2*np.pi*t) if seas else 0) + rng.normal(0, 0.5, t.size)
    r = tspec.analyze_series(t, y, band_method="conditional", remove_seasonality=seas)
    j = any(abs(e.time_year - 10) < 0.5 for e in r.jumps.jumps); tp = any(abs(p.time_year - 10) < 1.0 for p in r.turning_points.turning_points)
    return j, tp
out = {}
for jump in [1.0, 2.0, 3.0]:
    for seas in [True, False]:
        res = pmap(job, [(s, jump, seas) for s in range(100)], procs=2)
        out[f"step{jump}_sd0.5_seas{seas}"] = dict(jump_reported=float(np.mean([r[0] for r in res])), tp_reported_instead=float(np.mean([(not r[0]) and r[1] for r in res])))
        print(jump, seas, out[f"step{jump}_sd0.5_seas{seas}"], flush=True)
json.dump(out, open("p10_results.json", "w"), indent=1)
