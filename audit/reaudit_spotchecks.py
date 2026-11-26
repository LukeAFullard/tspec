"""Re-audit null-calibration spot checks for configurations of post-audit code not covered by
the main studies. Null data only; reports the familywise false-detection rate.
 a: just_monitor consecutive_k=2, horizon=48, sliding window, vector weights, season='allssa' (alpha=0.05)
 b: stpd with vector weights (sigma_i = 0.3 / sqrt(w_i)) and window_mode='duration' (alpha=0.01)
 c: just_jumps robust='huber', 30% left-censoring, detection_imputations=20 (jump_alpha=0.05)
 d: just_jumps robust='mm', season='ols' (jump_alpha=0.05)   [not covered by the original audit]
 e: stpd robust='mm' (alpha=0.01)                            [not covered by the original audit]
 f: just_jumps with 30% left-censoring, CensoredConfig(scope='windowed') (jump_alpha=0.05)
Seed base 19000. Run: OMP_NUM_THREADS=1 python audit/reaudit_spotchecks.py <a|b|c|d|e|f> <R>
"""
import json
import sys
import warnings

import numpy as np

import tspec

warnings.simplefilter("ignore")
case, R = sys.argv[1], int(sys.argv[2])
rng = np.random.default_rng(19000 + ord(case))
hits = 0
for _ in range(R):
    if case == "a":
        n = 96 + 48
        t = np.sort(rng.uniform(2000, 2000 + n / 24, n))
        w = rng.uniform(0.5, 2.0, n)
        y = 2 + 0.1 * (t - 2000) + 0.5 * np.cos(2 * np.pi * t) + rng.normal(0, 0.15, n) / np.sqrt(w)
        r = tspec.just_monitor(t, y, start=96, weights=w, window_mode="sliding", alpha=0.05,
                               sequential_control="consecutive_k", consecutive_k=2, horizon=48)
        hits += r.detected
    elif case == "b":
        n = 150
        t = np.sort(rng.uniform(2000, 2000 + n / 24, n))
        w = rng.uniform(0.5, 2.0, n)
        y = 3 + 0.2 * (t - 2000) + rng.normal(0, 0.3, n) / np.sqrt(w)
        hits += len(tspec.stpd(t, y, weights=w, window_mode="duration").turning_points) > 0
    elif case == "d":
        n = 150
        t = np.sort(rng.uniform(2000, 2000 + n / 24, n))
        y = 0.5 + 0.004 * (t - 2000) + 0.1 * np.cos(2 * np.pi * t) + rng.normal(0, 0.02, n)
        hits += len(tspec.just_jumps(t, y, season="ols", robust="mm").jumps) > 0
    elif case == "e":
        n = 150
        t = np.sort(rng.uniform(2000, 2000 + n / 24, n))
        y = 3 + 0.2 * (t - 2000) + rng.normal(0, 0.3, n)
        hits += len(tspec.stpd(t, y, robust="mm").turning_points) > 0
    elif case == "f":
        n = 150
        t = np.sort(rng.uniform(2000, 2000 + n / 24, n))
        y = 2 + 0.1 * (t - 2000) + rng.normal(0, 0.4, n)
        dl = float(np.quantile(y, 0.3))
        ys = np.array([f"<{dl:.6g}" if v < dl else float(v) for v in y], dtype=object)
        r = tspec.just_jumps(t, ys, season="none", censoring=tspec.CensoredConfig(scope="windowed"))
        hits += len(r.jumps) > 0
    else:
        n = 150
        t = np.sort(rng.uniform(2000, 2000 + n / 24, n))
        y = 2 + 0.1 * (t - 2000) + rng.normal(0, 0.4, n)
        dl = float(np.quantile(y, 0.3))
        ys = np.array([f"<{dl:.6g}" if v < dl else float(v) for v in y], dtype=object)
        r = tspec.just_jumps(t, ys, season="none", robust="huber",
                             censoring=tspec.CensoredConfig(detection_imputations=20))
        hits += len(r.jumps) > 0
out = {"case": case, "R": R, "false_detection_rate": hits / R}
print(json.dumps(out), flush=True)
json.dump(out, open(f"audit/post_fix/reaudit_spotcheck_{case}.json", "w"))
