"""MC7: STPD / TPTR coverage. Truth: continuous piecewise-linear, one turning point at mid-record
(slope +0.5 -> -0.5 per yr), N(0, sigma^2), ~24 obs/yr irregular.
 (a) fixed (true) knot: fit_piecewise_trend modes sequential/joint/huber -> pointwise curve coverage
 (b) detected knots: stpd() default -> pointwise curve coverage of piecewise_trend CI
 (c) analyze_series(season none, no jumps) segment-slope CI coverage, ols and sen (when 1 TP found
     within 0.5 yr of truth)
 (d) null (straight line): rate of >=1 accepted turning point (alpha=0.01 default) and the
     post-selection curve coverage when spurious TPs are accepted.
Seed base 7000. Run: OMP_NUM_THREADS=1 python audit/mc7_stpd.py <R>
"""
import sys, json, warnings
import numpy as np
import tspec
warnings.simplefilter("ignore")
R = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
out = {}
for n, sig in ((150, 0.3), (150, 1.0), (500, 0.6)):
    rng = np.random.default_rng(7000 + n + int(10 * sig))
    span = n / 24.0; ttp = 2000 + 0.5 * span
    acc = {k: [] for k in ("fix_seq", "fix_joint", "fix_huber", "det_seq", "null_det_seq")}
    slope_cov = {"ols": [0, 0], "sen": [0, 0]}
    detect = 0; null_fp = 0; loc_err = []
    for r in range(R):
        t = np.sort(rng.uniform(2000, 2000 + span, n))
        T = 3 + 0.5 * (t - 2000) - 1.0 * np.maximum(0, t - ttp)
        y = T + rng.normal(0, sig, n)
        k = int(np.argmin(np.abs(t - ttp)))
        for mode, key in (("sequential", "fix_seq"), ("joint", "fix_joint"), ("huber", "fix_huber")):
            p = tspec.fit_piecewise_trend(t, y, [k], mode=mode)
            acc[key].append(np.mean((T >= p.trend_ci_low) & (T <= p.trend_ci_high)))
        s = tspec.stpd(t, y)
        pw = s.piecewise_trend
        acc["det_seq"].append(np.mean((T >= pw.trend_ci_low) & (T <= pw.trend_ci_high)))
        tps = s.tp_times
        if len(tps) == 1 and abs(float(tps[0]) - ttp) < 0.5:
            detect += 1; loc_err.append(float(tps[0]) - ttp)
            if r < 300:
                for est in ("ols", "sen"):
                    a = tspec.analyze_series(t, y, detect_jumps=False, remove_seasonality=False,
                                             slope_estimator=est)
                    segs = a.segments
                    if len(segs) == 2:
                        slope_cov[est][1] += 2
                        slope_cov[est][0] += (segs[0].slope_ci_low <= 0.5 <= segs[0].slope_ci_high) + \
                                             (segs[1].slope_ci_low <= -0.5 <= segs[1].slope_ci_high)
        # null
        T0 = 3 + 0.2 * (t - 2000)
        y0 = T0 + rng.normal(0, sig, n)
        s0 = tspec.stpd(t, y0)
        null_fp += len(s0.turning_points) > 0
        if len(s0.turning_points) > 0:
            pw0 = s0.piecewise_trend
            acc["null_det_seq"].append(np.mean((T0 >= pw0.trend_ci_low) & (T0 <= pw0.trend_ci_high)))
    key = f"n={n},sigma={sig}"
    out[key] = {"R": R}
    for kk, v in acc.items():
        v = np.array(v)
        out[key][f"curve_cov[{kk}]"] = (float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else None, int(v.size))
    out[key]["single_TP_detect_rate(within 0.5yr)"] = detect / R
    out[key]["TP_loc_bias_yr"] = float(np.mean(loc_err)) if loc_err else None
    out[key]["TP_loc_rmse_yr"] = float(np.sqrt(np.mean(np.square(loc_err)))) if loc_err else None
    out[key]["null_FP_rate(alpha=0.01)"] = null_fp / R
    out[key]["MCSE_null_at_0.01"] = float(np.sqrt(0.01 * 0.99 / R))
    for est, (h, tot) in slope_cov.items():
        out[key][f"segment_slope_cov[{est}]"] = (h / tot if tot else None, tot)
    print(key, json.dumps(out[key]), flush=True)
json.dump(out, open("audit/mc7_results.json", "w"), indent=1)
