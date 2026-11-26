"""Re-audit MC9 on the shipped library path: Mann-Kendall Type-I error (true slope 0) and Sen
slope 95% CI coverage (true slope 0.1/yr) under AR(1) errors, via
analyze_series(slope_estimator='sen', default mk_correction='ar1'), single segment.
Errors: AR(1) with unit innovation variance scaled to sd 0.5; regular monthly sampling.
Seed base 21000. Run: OMP_NUM_THREADS=1 python audit/reaudit_mk.py <n> <R>
"""
import json
import sys
import warnings

import numpy as np

import tspec

warnings.simplefilter("ignore")
n, R = int(sys.argv[1]), int(sys.argv[2])
out = {}
for phi in (0.0, 0.3, 0.5, 0.7):
    rng = np.random.default_rng(21000 + n + int(100 * phi))
    rej = cov = 0
    for _ in range(R):
        e = rng.normal(size=n)
        x = np.empty(n)
        x[0] = e[0] / np.sqrt(1 - phi**2)
        for i in range(1, n):
            x[i] = phi * x[i - 1] + e[i]
        x = 0.5 * x * np.sqrt(1 - phi**2)  # marginal sd 0.5
        t = 2000 + np.arange(n) / 12.0
        kw = dict(detect_jumps=False, detect_turning_points=False, remove_seasonality=False, slope_estimator="sen")
        s0 = tspec.analyze_series(t, 2 + x, **kw).segments[0]
        rej += s0.pvalue < 0.05
        s1 = tspec.analyze_series(t, 2 + 0.1 * (t - 2000) + x, **kw).segments[0]
        cov += s1.slope_ci_low <= 0.1 <= s1.slope_ci_high
    out[f"phi={phi}"] = {"R": R, "typeI_at_0.05": rej / R, "sen_CI95_coverage": cov / R}
    print(n, phi, out[f"phi={phi}"], flush=True)
json.dump(out, open(f"audit/post_fix/reaudit_mk_n{n}.json", "w"), indent=1)
