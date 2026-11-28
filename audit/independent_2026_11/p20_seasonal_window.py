"""P20 (exploratory): is tspec's seasonal error just the 3-year window? Seasonal / remainder RMSE
vs window length (duration windows), on registered designs and seeds (first 40 replicates)."""
import sys, json
import numpy as np
sys.path.insert(0, "../../validation")
import bench_decomp_heldout as B, tspec
from common import pmap
cells = {c["label"]: c for c in B.build_cells()}
def job(a):
    lab, rep = a
    c = cells[lab]
    rng = np.random.default_rng(910_000 + 1_000 * c["id"] + rep)
    t = B.sample_times(c["regime"], rng); t0, sp = float(t[0]), float(t[-1] - t[0])
    tr = B.TRENDS[c["trend"]](t, t0); se = B.SEASONS[c["season"]](t, t0, sp); e = B.noise(c["noise"], t, rng)
    y = tr + se + e
    out = {}
    for W in [3, 6, 10]:
        d = tspec.just_decompose(t, y, season="ols", window_mode="duration", window_duration=W)
        ok = np.isfinite(d.trend)
        out[f"W{W}"] = (float(np.sqrt(np.mean((d.seasonal - se)[ok] ** 2))), float(np.sqrt(np.mean((d.trend - tr)[ok] ** 2))),
                       float(np.sqrt(np.mean((d.remainder - e)[ok] ** 2))))
    return out
for lab in ["R3_T1S1_N1", "R4_T1S1_N1", "R7_T1S1_N1", "R3_T1S2_N1", "R4_T1S2_N1"]:
    rows = pmap(job, [(lab, r) for r in range(40)])
    print(lab, {W: tuple(round(float(np.mean([r[W][k] for r in rows])), 3) for k in range(3)) for W in rows[0]}, flush=True)
