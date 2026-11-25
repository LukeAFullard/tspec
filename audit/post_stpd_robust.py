"""Post-fix check: STPD robust='huber' (tp_calibration='bonferroni') null false-detection rate and
power under the MC7 design (n=150, ~24 obs/yr), Gaussian and t3 noise. Seed 17500.
Run: OMP_NUM_THREADS=1 python audit/post_stpd_robust.py <R>
"""
import json
import sys
import warnings

import numpy as np

import tspec

warnings.simplefilter("ignore")
R = int(sys.argv[1]) if len(sys.argv) > 1 else 500
rng = np.random.default_rng(17500)
out = {}
for noise in ("gauss", "t3"):
    for sig in (0.3, 1.0):
        n = 150; span = n / 24; ttp = 2000 + span / 2
        fp = det = 0
        for _ in range(R):
            t = np.sort(rng.uniform(2000, 2000 + span, n))
            e = rng.normal(0, sig, n) if noise == "gauss" else sig * rng.standard_t(3, n) / np.sqrt(3)
            e0 = rng.normal(0, sig, n) if noise == "gauss" else sig * rng.standard_t(3, n) / np.sqrt(3)
            y = 3 + 0.5 * (t - 2000) - np.maximum(0, t - ttp) + e
            y0 = 3 + 0.2 * (t - 2000) + e0
            s = tspec.stpd(t, y, robust="huber")
            det += len(s.tp_times) == 1 and abs(float(s.tp_times[0]) - ttp) < 0.5
            fp += len(tspec.stpd(t, y0, robust="huber").turning_points) > 0
        out[f"{noise},sigma={sig}"] = {"R": R, "null_FP_rate": fp / R, "single_TP_detect_rate": det / R}
        print(noise, sig, out[f"{noise},sigma={sig}"], flush=True)
json.dump(out, open("audit/post_fix/post_stpd_robust.json", "w"), indent=1)
