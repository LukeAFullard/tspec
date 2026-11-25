"""Dose-response of just_decompose trend-CI coverage vs trend curvature (season='ols', defaults).
Trend 2 + A sin(2 pi (t-2000)/P); annual seasonal; sigma=0.3; ~24 obs/yr. Seed 4400.
Run: OMP_NUM_THREADS=1 python audit/repro_just_ci_curvature.py
"""
import warnings, json
import numpy as np
import tspec
warnings.simplefilter("ignore")
R = 200
rows = []
for n in (150, 500):
    for (A, P) in ((0.6, 6), (0.6, 9), (0.6, 15), (0.6, 25), (0.3, 15), (0.0, 1)):
        rng = np.random.default_rng(4400 + n + int(10 * A) + P)
        cov = []; maxbias = []
        for r in range(R):
            t = np.sort(rng.uniform(2000, 2000 + n / 24, n))
            T = 2 + A * np.sin(2 * np.pi * (t - 2000) / P)
            S = np.cos(2 * np.pi * t) + 0.4 * np.sin(2 * np.pi * t)
            y = T + S + rng.normal(0, 0.3, n)
            d = tspec.just_decompose(t, y, season="ols")
            cov.append(np.mean((T >= d.trend_ci_low) & (T <= d.trend_ci_high)))
        cov = np.array(cov)
        curv = A * (2 * np.pi / P) ** 2 if A else 0.0
        rows.append({"n": n, "A": A, "P_yr": P, "max|T''|": round(curv, 3),
                     "trend_cov": round(float(cov.mean()), 3), "MCSE": round(float(cov.std(ddof=1) / np.sqrt(R)), 3)})
        print(json.dumps(rows[-1]), flush=True)
json.dump(rows, open("audit/repro_just_ci_curvature.json", "w"), indent=1)
