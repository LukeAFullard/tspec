"""Post-fix MC7: STPD calibrated slope-change test (CRT-04).
Same data design as mc7_stpd.py (continuous piecewise-linear truth, one turning point at mid-record,
slope +0.5 -> -0.5 per yr, N(0, sigma^2), ~24 obs/yr irregular; null = straight line, slope 0.2).
Variants:
  legacy  : tp_calibration="none", snr_th=1.0 (previous defaults)
  cal     : defaults (tp_calibration="montecarlo", snr_th=0)
  cal_snr : tp_calibration="montecarlo", snr_th=1.0
Metrics: null false-detection rate (target <= alpha=0.01), single-TP detection within 0.5 yr,
location RMSE, pointwise curve coverage of the piecewise-trend band (all replicates and given detection).
Seed base 17000. Run: OMP_NUM_THREADS=1 python audit/post_mc7_stpd.py <R> <config-index>
"""
import json
import sys
import warnings

import numpy as np

import tspec

warnings.simplefilter("ignore")
R = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
CONFIGS = ((150, 0.3), (150, 1.0), (500, 0.6), (500, 1.5))
idx = [int(sys.argv[2])] if len(sys.argv) > 2 else range(len(CONFIGS))
VARIANTS = {
    "legacy": dict(tp_calibration="none", snr_th=1.0),
    "cal": dict(),
    "cal_snr": dict(tp_calibration="montecarlo", snr_th=1.0),
}
out = {}
for i in idx:
    n, sig = CONFIGS[i]
    rng = np.random.default_rng(17000 + n + int(10 * sig))
    span = n / 24.0
    ttp = 2000 + 0.5 * span
    res = {v: {"fp": 0, "det": 0, "loc": [], "cov": [], "cov_det": [], "ntp": []} for v in VARIANTS}
    for _r in range(R):
        t = np.sort(rng.uniform(2000, 2000 + span, n))
        truth = 3 + 0.5 * (t - 2000) - 1.0 * np.maximum(0, t - ttp)
        y = truth + rng.normal(0, sig, n)
        t0 = 3 + 0.2 * (t - 2000)
        y0 = t0 + rng.normal(0, sig, n)
        for v, kw in VARIANTS.items():
            s = tspec.stpd(t, y, **kw)
            pw = s.piecewise_trend
            c = float(np.mean((truth >= pw.trend_ci_low) & (truth <= pw.trend_ci_high)))
            res[v]["cov"].append(c)
            tps = s.tp_times
            res[v]["ntp"].append(len(tps))
            if len(tps) == 1 and abs(float(tps[0]) - ttp) < 0.5:
                res[v]["det"] += 1
                res[v]["loc"].append(float(tps[0]) - ttp)
                res[v]["cov_det"].append(c)
            s0 = tspec.stpd(t, y0, **kw)
            res[v]["fp"] += len(s0.turning_points) > 0
    key = f"n={n},sigma={sig}"
    out[key] = {}
    for v, d in res.items():
        out[key][v] = {
            "R": R,
            "null_FP_rate": d["fp"] / R,
            "single_TP_detect_rate": d["det"] / R,
            "mean_n_tp": float(np.mean(d["ntp"])),
            "TP_loc_rmse_yr": float(np.sqrt(np.mean(np.square(d["loc"])))) if d["loc"] else None,
            "curve_cov_all": float(np.mean(d["cov"])),
            "curve_cov_given_detect": float(np.mean(d["cov_det"])) if d["cov_det"] else None,
        }
    out[key]["MCSE_at_0.01"] = float(np.sqrt(0.01 * 0.99 / R))
    print(key, json.dumps(out[key]), flush=True)
tag = "_".join(str(i) for i in idx)
json.dump(out, open(f"audit/post_fix/post_mc7_stpd_{tag}.json", "w"), indent=1)
