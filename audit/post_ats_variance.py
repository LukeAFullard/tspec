"""ATS (Akritas-Theil-Sen) calibration with the censoring-aware variance: slope CI coverage and
null Mann-Kendall Type-I error at 20% and 50% left-censoring, n=150. Seed 12000.
Run: OMP_NUM_THREADS=1 python audit/post_ats_variance.py <R>"""
import sys, warnings, json
import numpy as np
from tspec._robust import fit_theil_sen_segment
warnings.simplefilter("ignore")
R = int(sys.argv[1]) if len(sys.argv) > 1 else 500
out = {}
for frac in (0.2, 0.5):
    rng = np.random.default_rng(12000 + int(100 * frac))
    n = 150; t = np.sort(rng.uniform(0, 6.25, n))
    dl = float(np.quantile(2 + 0.1 * t[:, None] + rng.normal(0, 0.4, (n, 400)), frac))
    cov = rej = 0; widths = []
    for r in range(R):
        e = rng.normal(0, 0.4, n)
        for beta, kind in ((0.1, "cov"), (0.0, "null")):
            y = 2 + beta * t + e
            codes = np.where(y < dl, -1, 0); vals = np.where(y < dl, dl, y)
            f = fit_theil_sen_segment(t, np.where(codes == -1, dl / 2, y), status_codes=codes, original_values=vals)
            if kind == "cov":
                cov += f.ci_low <= beta <= f.ci_high; widths.append(f.ci_high - f.ci_low)
            else:
                rej += f.pvalue < 0.05
    out[f"censor~{frac}"] = {"R": R, "CI95_cov": cov / R, "MK_typeI@0.05": rej / R,
                             "median_width": float(np.median(widths)), "MCSE": float(np.sqrt(0.0475 / R))}
    print(json.dumps(out[f"censor~{frac}"]), flush=True)
json.dump(out, open("audit/post_fix/ats_variance_results.json", "w"), indent=1)
