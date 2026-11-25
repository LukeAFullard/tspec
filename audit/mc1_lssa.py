"""MC1: LSSA / ALLSSA false-alarm calibration on Gaussian white noise, irregular sampling.
Seed base 1001. Run: python audit/mc1_lssa.py
"""
import sys, warnings, json
import numpy as np
import tspec
from tspec.spectral import critical_value_lss
warnings.simplefilter("ignore")
R = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
res = {}

def se(p, r): return float(np.sqrt(p * (1 - p) / r))

for n in (50, 150, 500):
    for gap in (False, True):
        rng = np.random.default_rng(1001 + n + 7 * gap)
        rej05 = rej01 = 0; fw05 = fw01 = 0; npts = 0; allssa_any = 0; huber05 = 0
        R_eff = R if n < 500 else max(500, R // 2)
        for r in range(R_eff):
            span = n / 24.0  # ~24 obs per year
            t = np.sort(rng.uniform(2000, 2000 + span, n))
            if gap:  # remove a block of 25% of span
                keep = ~((t > 2000 + 0.4 * span) & (t < 2000 + 0.65 * span))
                t = t[keep]
            y = rng.normal(0, 1, t.size)
            # pointwise: one pre-specified frequency
            f0 = np.array([1.37])
            p = tspec.lssa(t, y, omega=f0).pvalues[0]
            rej05 += p < 0.05; rej01 += p < 0.01
            # grid: 0.1 .. 12 cy/yr step 0.05 (~238 freqs)
            grid = np.arange(0.1, 12.0, 0.05)
            g = tspec.lssa(t, y, omega=grid, alpha=0.05)
            fw05 += bool(np.any(g.pvalues < 0.05)); fw01 += bool(np.any(g.pvalues < 0.01))
            npts += int(np.sum(g.pvalues < 0.05))
            if r < 600:
                a = tspec.allssa(t, y)  # defaults: alpha=0.01, integer grid 1..~11
                allssa_any += a.frequencies.size > 0
                ph = tspec.lssa(t, y, omega=f0, robust="huber").pvalues[0]
                huber05 += ph < 0.05
        Ra = min(600, R_eff)
        key = f"n={n},gap={gap}"
        res[key] = {
            "R": R_eff,
            "pointwise_FAR@0.05": (rej05 / R_eff, se(0.05, R_eff)),
            "pointwise_FAR@0.01": (rej01 / R_eff, se(0.01, R_eff)),
            "grid_any@0.05": (fw05 / R_eff, se(fw05 / R_eff, R_eff)),
            "grid_any@0.01": (fw01 / R_eff, se(fw01 / R_eff, R_eff)),
            "grid_mean_sig_points@0.05": npts / R_eff,
            "allssa_default_any_selected(alpha=0.01)": (allssa_any / Ra, se(max(allssa_any / Ra, 1e-3), Ra), Ra),
            "huber_pointwise_FAR@0.05": (huber05 / Ra, se(0.05, Ra), Ra),
        }
        print(key, json.dumps(res[key]), flush=True)
json.dump(res, open("audit/mc1_results.json", "w"), indent=1)
