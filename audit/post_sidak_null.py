"""Null calibration of ALLSSA significance='sidak' (after the effective-frequency fix) on
full records and on short JUST-window-sized series with the JUST default grid.
Gaussian white noise, linear trend model, alpha=0.01. Seed 13000.
Run: OMP_NUM_THREADS=1 python audit/post_sidak_null.py <R>"""
import sys, json, warnings
import numpy as np
import tspec
warnings.simplefilter("ignore")
R = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
cases = {
    "record n=50 (2.1 yr), default grid": (50, 50 / 24, None),
    "record n=150 (6.3 yr), default grid": (150, 150 / 24, None),
    "record n=500 (21 yr), default grid": (500, 500 / 24, None),
    "window n=40 (1.3 yr), JUST grid 0.8-3.8": (40, 40 / 30, np.arange(4, 20) / 5.0),
    "window n=72 (3 yr), JUST grid 0.8-3.8": (72, 3.0, np.arange(4, 20) / 5.0),
}
out = {}
for k, (n, span, om) in cases.items():
    rng = np.random.default_rng(13000 + n)
    hits = 0
    for r in range(R):
        t = np.sort(rng.uniform(0, span, n))
        hits += tspec.allssa(t, rng.normal(size=n), omega=om, significance="sidak").frequencies.size > 0
    out[k] = {"R": R, "rate": hits / R, "MCSE": float(np.sqrt(0.01 * 0.99 / R))}
    print(k, json.dumps(out[k]), flush=True)
json.dump(out, open("audit/post_fix/sidak_null_results.json", "w"), indent=1)
