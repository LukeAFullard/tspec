"""MC9: Theil-Sen CI coverage and Mann-Kendall Type-I under iid and AR(1) (phi=0.5) noise.
Also measures how often analyze_series(slope_estimator='sen') emits an autocorrelation warning.
Seed base 9000. Run: OMP_NUM_THREADS=1 python audit/mc9_sen_mk.py [R]
"""
import sys, json, warnings
import numpy as np
import tspec
from tspec._robust import fit_theil_sen_segment
warnings.simplefilter("ignore")
R = int(sys.argv[1]) if len(sys.argv) > 1 else 2000

def ar1(rng, n, phi):
    e = rng.normal(size=n)
    x = np.empty(n); x[0] = e[0] / np.sqrt(1 - phi ** 2) if phi else e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x

res = {}
for n in (50, 150, 500):
    for phi in (0.0, 0.5):
        rng = np.random.default_rng(9000 + n + int(phi * 10))
        t = np.sort(rng.uniform(2000, 2000 + n / 12, n))  # ~monthly irregular
        cov = mk_rej = 0; warn = 0; n_ap = 0; widths = []
        Rn = R if n < 500 else R // 2
        for r in range(Rn):
            e = ar1(rng, n, phi)
            y0 = e  # null: no trend
            f0 = fit_theil_sen_segment(t, y0, alpha=0.05)
            mk_rej += (f0.pvalue is not None and f0.pvalue < 0.05)
            beta = 0.3
            y1 = beta * (t - t[0]) + e
            f1 = fit_theil_sen_segment(t, y1, alpha=0.05)
            cov += f1.ci_low <= beta <= f1.ci_high
            widths.append(f1.ci_high - f1.ci_low)
            if r < 200:
                pr = tspec.analyze_series(t, y0, detect_jumps=False, detect_turning_points=False,
                                          remove_seasonality=False, slope_estimator="sen")
                codes = {w.code for w in pr.diagnostics.warnings}
                warn += bool(codes & {"MODERATE_AUTOCORRELATION", "STRONG_AUTOCORRELATION"})
                n_ap += 1
        key = f"n={n},phi={phi}"
        res[key] = {"R": Rn,
                    "MK_typeI@0.05": mk_rej / Rn, "MCSE_typeI": float(np.sqrt(0.05 * 0.95 / Rn)),
                    "Sen_CI95_coverage": cov / Rn, "MCSE_cov": float(np.sqrt(0.95 * 0.05 / Rn)),
                    "median_CI_width": float(np.median(widths)),
                    "autocorr_warning_rate(analyze_series)": warn / n_ap, "R_warn": n_ap}
        print(key, json.dumps(res[key]), flush=True)
json.dump(res, open("audit/mc9_results.json", "w"), indent=1)
