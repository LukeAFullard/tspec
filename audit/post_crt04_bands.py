"""Post-fix CRT-04: coverage of post-detection trend bands, bootstrap vs. conditional.
 stpd: MC7 design (continuous piecewise-linear truth, one turning point at mid-record, slope
       +0.5 -> -0.5 /yr, N(0, sigma^2), ~24 obs/yr irregular); pointwise curve coverage of
       stpd().piecewise_trend bands given correct detection (single TP within 0.5 yr) and over
       all replicates.
 pipe: MC7b design (seasonal, one turning point at 2006, n = 288, sigma = 0.2); trend band and
       segment-slope CI coverage of analyze_series() given the correct structure.
Seed bases 17000 (stpd) and 7700 (pipe), as in the original studies.
Run: OMP_NUM_THREADS=1 python audit/post_crt04_bands.py <stpd|pipe> <R> [config-index]
"""
import json
import sys
import warnings

import numpy as np

import tspec

warnings.simplefilter("ignore")
mode, R = sys.argv[1], int(sys.argv[2])
out = {}
if mode == "stpd":
    configs = ((150, 0.3), (150, 1.0), (500, 0.6))
    idx = [int(sys.argv[3])] if len(sys.argv) > 3 else range(len(configs))
    for i in idx:
        n, sig = configs[i]
        rng = np.random.default_rng(17000 + n + int(10 * sig))
        span = n / 24.0
        ttp = 2000 + span / 2
        res = {b: {"cov_det": [], "cov_all": [], "width": []} for b in ("bootstrap", "conditional")}
        for _ in range(R):
            t = np.sort(rng.uniform(2000, 2000 + span, n))
            truth = 3 + 0.5 * (t - 2000) - 1.0 * np.maximum(0, t - ttp)
            y = truth + rng.normal(0, sig, n)
            for b in res:
                s = tspec.stpd(t, y, band_method=b)
                pw = s.piecewise_trend
                c = float(np.mean((truth >= pw.trend_ci_low) & (truth <= pw.trend_ci_high)))
                res[b]["cov_all"].append(c)
                res[b]["width"].append(float(np.mean(pw.trend_ci_high - pw.trend_ci_low)))
                if len(s.tp_times) == 1 and abs(float(s.tp_times[0]) - ttp) < 0.5:
                    res[b]["cov_det"].append(c)
        key = f"n={n},sigma={sig}"
        out[key] = {b: {"R": R, "n_detected": len(d["cov_det"]),
                        "curve_cov_given_detect": float(np.mean(d["cov_det"])) if d["cov_det"] else None,
                        "curve_cov_all": float(np.mean(d["cov_all"])),
                        "mean_width": float(np.mean(d["width"]))} for b, d in res.items()}
        print(key, json.dumps(out[key]), flush=True)
    tag = "_".join(str(i) for i in idx)
else:
    rng = np.random.default_rng(7700)
    res = {b: {"band": [], "slope_hit": 0, "slope_n": 0, "ok": 0} for b in ("bootstrap", "conditional")}
    for _ in range(R):
        t = np.sort(rng.uniform(2000, 2012, 288))
        truth = 3 + 0.4 * (t - 2000) - 0.7 * np.maximum(0, t - 2006)
        y = truth + 0.6 * np.cos(2 * np.pi * t) + 0.2 * np.sin(4 * np.pi * t) + rng.normal(0, 0.2, t.size)
        for b in res:
            a = tspec.analyze_series(t, y, band_method=b)
            tps = a.turning_points.turning_points if a.turning_points is not None else ()
            if len(tps) != 1 or len(a.jumps.jumps) > 0 or abs(tps[0].time_year - 2006) > 0.5:
                continue
            res[b]["ok"] += 1
            res[b]["band"].append(float(np.mean((truth >= a.trend_ci_low) & (truth <= a.trend_ci_high))))
            for seg, true_s in zip(a.segments, (0.4, -0.3), strict=False):
                res[b]["slope_n"] += 1
                res[b]["slope_hit"] += seg.slope_ci_low <= true_s <= seg.slope_ci_high
    out = {b: {"R": R, "correct_structure": d["ok"] / R,
               "trend_band_cov|correct": float(np.mean(d["band"])) if d["band"] else None,
               "segment_slope_cov|correct": d["slope_hit"] / d["slope_n"] if d["slope_n"] else None}
           for b, d in res.items()}
    print(json.dumps(out), flush=True)
    tag = "pipe"
json.dump(out, open(f"audit/post_fix/post_crt04_bands_{mode}_{tag}.json", "w"), indent=1)
