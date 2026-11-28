"""P19 (exploratory, after the registered decomposition run): band width and NaN share for the
cells where tspec_ols / tspec_ols_dur failed, using the registered seeds."""
import sys, json
import numpy as np
sys.path.insert(0, "../../validation")
import bench_decomp_heldout as B
from common import pmap
cells = {c["label"]: c for c in B.build_cells()}
def job(a):
    lab, rep = a
    c = cells[lab]
    rng = np.random.default_rng(910_000 + 1_000 * c["id"] + rep)
    t = B.sample_times(c["regime"], rng); t0, sp = float(t[0]), float(t[-1] - t[0])
    tr = B.TRENDS[c["trend"]](t, t0); se = B.SEASONS[c["season"]](t, t0, sp)
    y = tr + se + B.noise(c["noise"], t, rng)
    out = {}
    for cfg in ["tspec_default", "tspec_ols", "tspec_ols_dur"]:
        d, _ = B.run_tspec(t, y, cfg)
        w = d.trend_ci_high - d.trend_ci_low
        e = d.trend - tr
        out[cfg] = dict(nan=float(np.mean(~np.isfinite(d.trend))), med_width=float(np.nanmedian(w)),
                        max_width=float(np.nanmax(w)), rmse_finite=float(np.sqrt(np.nanmean(e ** 2))),
                        med_abs_err=float(np.nanmedian(np.abs(e))), frac_err_gt_1=float(np.nanmean(np.abs(e) > 1)))
    return out
res = {}
for lab in ["R1_T1S1_N1", "R5_T1S1_N1", "R6_T1S1_N1", "R7_T1S1_N1"]:
    rows = pmap(job, [(lab, r) for r in range(100)])
    res[lab] = {cfg: {k: float(np.mean([r[cfg][k] for r in rows])) for k in rows[0][cfg]} for cfg in rows[0]}
    print(lab, json.dumps({c: {k: round(v, 3) for k, v in d.items()} for c, d in res[lab].items()}), flush=True)
json.dump(res, open("p19_results.json", "w"), indent=1)
