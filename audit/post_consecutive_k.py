"""Post-fix CRT-01: Monte Carlo-calibrated `consecutive_k` monitoring.
Design of MC5 (audit/mc5_6_monitor.py): stable truth 2 + 0.1 t + 0.5 cos(2 pi t) + 0.2 sin(4 pi t)
+ N(0, 0.15^2), irregular ~24 obs/yr (the calibration assumes regular spacing, so this also tests
robustness to that assumption), n_his = 96, horizon D. alpha = 0.05.
Reports the familywise false-alarm rate (null) and detection rate / median delay for an abrupt
step of -0.3 (2 sigma) and -0.45 (3 sigma) starting at monitoring step D//4, for
consecutive_k=2 (calibrated, horizon=D), consecutive_k=2 uncalibrated (no horizon) and pocock (horizon=D).
Seed base 15000. Run: OMP_NUM_THREADS=1 python audit/post_consecutive_k.py <season> <D> <R>
"""
import json
import sys
import warnings

import numpy as np

import tspec

warnings.simplefilter("ignore")
season = sys.argv[1] if len(sys.argv) > 1 else "ols"
D = int(sys.argv[2]) if len(sys.argv) > 2 else 24
R = int(sys.argv[3]) if len(sys.argv) > 3 else 500
alpha = 0.05
n_his = 96
controls = {
    "consecutive_k2_calibrated": dict(sequential_control="consecutive_k", consecutive_k=2, horizon=D),
    "consecutive_k3_calibrated": dict(sequential_control="consecutive_k", consecutive_k=3, horizon=D),
    "consecutive_k2_no_horizon": dict(sequential_control="consecutive_k", consecutive_k=2),
    "pocock_horizon": dict(sequential_control="pocock", horizon=D),
}
rng = np.random.default_rng(15000 + D + (1 if season == "allssa" else 0))
res = {k: {"null": 0, "det2": 0, "det3": 0, "delay2": [], "delay3": []} for k in controls}
onset = n_his + D // 4
for _ in range(R):
    n = n_his + D
    t = np.sort(rng.uniform(2000, 2000 + n / 24.0, n))
    mu = 2 + 0.1 * (t - 2000) + 0.5 * np.cos(2 * np.pi * t) + 0.2 * np.sin(4 * np.pi * t)
    e = rng.normal(0, 0.15, n)
    y0 = mu + e
    step = np.where(np.arange(n) >= onset, 1.0, 0.0)
    for k, kw in controls.items():
        r0 = tspec.just_monitor(t, y0, start=n_his, season=season, alpha=alpha, **kw)
        res[k]["null"] += r0.detected
        for mag, tag in ((0.3, "2"), (0.45, "3")):
            r1 = tspec.just_monitor(t, y0 - mag * step, start=n_his, season=season, alpha=alpha, **kw)
            if r1.detected and r1.disturbance_index is not None and r1.disturbance_index >= onset - 2:
                res[k]["det" + tag] += 1
                trig = next(s.step for s in r1.steps if s.exceeded and s.consecutive_count >= kw.get("consecutive_k", 1)) if kw["sequential_control"] == "consecutive_k" else next(s.step for s in r1.steps if s.exceeded)
                res[k]["delay" + tag].append(trig - (onset - n_his))
out = {}
for k, d in res.items():
    out[k] = {
        "R": R,
        "FWER": d["null"] / R,
        "detect_2sigma": d["det2"] / R,
        "detect_3sigma": d["det3"] / R,
        "median_delay_2sigma": float(np.median(d["delay2"])) if d["delay2"] else None,
        "median_delay_3sigma": float(np.median(d["delay3"])) if d["delay3"] else None,
    }
out["MCSE_at_alpha"] = float(np.sqrt(alpha * (1 - alpha) / R))
print(season, D, json.dumps(out), flush=True)
json.dump(out, open(f"audit/post_fix/post_consecutive_k_{season}_{D}.json", "w"), indent=1)
