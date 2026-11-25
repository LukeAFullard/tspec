"""Post-fix MC5: just_monitor familywise false alarms with the new defaults (no horizon ->
open-ended spending) and with a planned horizon; PI coverage. Same DGP as audit/mc5_6_monitor.py.
Seed base 5000. Run: OMP_NUM_THREADS=1 python audit/post_mc5_monitor.py <R>"""
import sys, json, warnings
import numpy as np
import tspec
warnings.simplefilter("ignore")
R = int(sys.argv[1]); out = {}
for D in (24, 96):
    rng = np.random.default_rng(5000 + D + 50)
    a = {"pocock_no_horizon": 0, "pocock_horizon=D": 0, "obf_no_horizon": 0, "consecutive_k2": 0}; pi = []
    for r in range(R):
        n = 96 + D; t = np.sort(rng.uniform(2000, 2000 + n / 24, n))
        y = 2 + 0.1 * (t - 2000) + 0.5 * np.cos(2 * np.pi * t) + 0.2 * np.sin(4 * np.pi * t) + rng.normal(0, 0.15, n)
        m = tspec.just_monitor(t, y, start=96, season="ols", alpha=0.05); a["pocock_no_horizon"] += m.detected
        lo, hi = m.forecast_pi_low[96:], m.forecast_pi_high[96:]; pi.append(np.mean((y[96:] >= lo) & (y[96:] <= hi)))
        a["pocock_horizon=D"] += tspec.just_monitor(t, y, start=96, season="ols", alpha=0.05, horizon=D).detected
        a["obf_no_horizon"] += tspec.just_monitor(t, y, start=96, season="ols", alpha=0.05, sequential_control="obrien_fleming").detected
        a["consecutive_k2"] += tspec.just_monitor(t, y, start=96, season="ols", alpha=0.05, sequential_control="consecutive_k").detected
    out[f"D={D}"] = {"R": R, **{k: v / R for k, v in a.items()}, "PI95_cov": float(np.mean(pi)), "MCSE": float(np.sqrt(0.0475 / R))}
    print(f"D={D}", json.dumps(out[f"D={D}"]), flush=True)
json.dump(out, open("audit/post_mc5_results.json", "w"), indent=1)
