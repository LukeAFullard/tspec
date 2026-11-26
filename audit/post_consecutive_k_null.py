"""Post-fix CRT-01 null check: familywise false-alarm rate of calibrated consecutive_k=2 only
(MC5 design, n_his = 96, horizon D, alpha = 0.05). Seed 16000 + shard.
Run: OMP_NUM_THREADS=1 python audit/post_consecutive_k_null.py <season> <D> <R> <shard>
"""
import json
import sys
import warnings

import numpy as np

import tspec

warnings.simplefilter("ignore")
season, D, R, shard = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
rng = np.random.default_rng(16000 + 97 * shard + D + (1 if season == "allssa" else 0))
n_his = 96
alarms = 0
for _ in range(R):
    n = n_his + D
    t = np.sort(rng.uniform(2000, 2000 + n / 24.0, n))
    mu = 2 + 0.1 * (t - 2000) + 0.5 * np.cos(2 * np.pi * t) + 0.2 * np.sin(4 * np.pi * t)
    y = mu + rng.normal(0, 0.15, n)
    alarms += tspec.just_monitor(
        t, y, start=n_his, season=season, alpha=0.05,
        sequential_control="consecutive_k", consecutive_k=2, horizon=D,
    ).detected
out = {"season": season, "D": D, "R": R, "shard": shard, "alarms": int(alarms)}
print(json.dumps(out), flush=True)
json.dump(out, open(f"audit/post_fix/post_consecutive_k_null_{season}_{D}_{shard}.json", "w"))
