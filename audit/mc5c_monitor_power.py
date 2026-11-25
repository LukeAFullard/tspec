"""MC5c: detection power and delay of just_monitor defaults for an abrupt level drop of k*sigma
starting at monitoring step 1 (history 96 obs, 48 monitoring obs, sigma=0.15). Seed 5700.
Run: OMP_NUM_THREADS=1 python audit/mc5c_monitor_power.py"""
import json, warnings
import numpy as np
import tspec
warnings.simplefilter("ignore")
R = 200; out = {}
for k in (1.0, 2.0, 4.0):
    rng = np.random.default_rng(5700 + int(k))
    res = {"pocock": [0, []], "consecutive_k2": [0, []], "obrien_fleming": [0, []]}
    for r in range(R):
        n = 144; t = np.sort(rng.uniform(2000, 2006, n))
        y = 2 + 0.1 * (t - 2000) + 0.5 * np.cos(2 * np.pi * t) + rng.normal(0, 0.15, n)
        y[96:] -= k * 0.15
        for name, kw in (("pocock", {}), ("consecutive_k2", dict(sequential_control="consecutive_k")),
                         ("obrien_fleming", dict(sequential_control="obrien_fleming"))):
            m = tspec.just_monitor(t, y, start=96, season="ols", alpha=0.05, **kw)
            if m.detected:
                res[name][0] += 1; res[name][1].append(m.disturbance_index - 96 + 1)
    out[f"drop={k}sigma"] = {nm: {"power": v[0] / R, "median_delay_steps": float(np.median(v[1])) if v[1] else None} for nm, v in res.items()}
    print(f"drop={k}sigma", json.dumps(out[f"drop={k}sigma"]), flush=True)
json.dump(out, open("audit/mc5c_results.json", "w"), indent=1)
