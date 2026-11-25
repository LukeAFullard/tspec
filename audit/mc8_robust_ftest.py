"""MC8: Type-I error of the Schrader-Hettmansperger robust nested F-test (fixed, pre-specified
alternative; no split selection). Null: linear trend. Alternative: 2-piece discontinuous linear at
the midpoint (q=2). Noise: N(0,1), t3, 10% contaminated normal (sd 1 / 10).
Seed base 8000. Run: OMP_NUM_THREADS=1 python audit/mc8_robust_ftest.py [R]
"""
import sys, json, warnings
import numpy as np
from scipy import stats
from tspec._linalg import prepare_weights, solve_weighted_ols
from tspec._robust import huber_nested_ftest
from tspec.spectral import build_design_matrix
warnings.simplefilter("ignore")
R = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
res = {}
for n in (50, 150, 500):
    t0 = np.linspace(0, 10, n)
    Dn, cn = build_design_matrix(t0, "linear", np.empty(0, dtype=np.intp), False, np.empty(0))
    Df, cf = build_design_matrix(t0, "linear", np.array([n // 2]), False, np.empty(0))
    pw = prepare_weights(None, n)
    for noise in ("normal", "t3", "contam"):
        rng = np.random.default_rng(8000 + n + {"normal": 1, "t3": 2, "contam": 3}[noise])
        rej = {"huber": [0, 0], "mm": [0, 0], "ols": [0, 0]}
        for r in range(R):
            if noise == "normal":
                e = rng.normal(size=n)
            elif noise == "t3":
                e = stats.t.rvs(3, size=n, random_state=rng)
            else:
                e = np.where(rng.uniform(size=n) < 0.1, rng.normal(0, 10, n), rng.normal(0, 1, n))
            y = 2 + 0.3 * t0 + e
            for m in ("huber", "mm"):
                _, p, _, _ = huber_nested_ftest(Dn, Df, y, pw, method=m, col_scales_null=cn, col_scales_full=cf)
                rej[m][0] += p < 0.05; rej[m][1] += p < 0.01
            r0 = solve_weighted_ols(Dn, y, pw, col_scales=cn).norm_res
            r1 = solve_weighted_ols(Df, y, pw, col_scales=cf).norm_res
            F = ((r0 - r1) / 2) / (r1 / (n - 4))
            p = stats.f.sf(F, 2, n - 4)
            rej["ols"][0] += p < 0.05; rej["ols"][1] += p < 0.01
        key = f"n={n},{noise}"
        res[key] = {m: {"rej@0.05": v[0] / R, "rej@0.01": v[1] / R} for m, v in rej.items()}
        res[key]["R"] = R
        res[key]["MCSE@0.05"] = float(np.sqrt(0.05 * 0.95 / R))
        res[key]["MCSE@0.01"] = float(np.sqrt(0.01 * 0.99 / R))
        print(key, json.dumps(res[key]), flush=True)
json.dump(res, open("audit/mc8_results.json", "w"), indent=1)
