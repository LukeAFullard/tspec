"""MC7b: analyze_series() defaults on a seasonal series with one turning point (no jump): coverage of
segment slope CIs and of the pointwise trend_ci band (truth = trend, i.e. series minus seasonal).
~24 obs/yr, 12 yr, sigma=0.2, slopes +0.4 -> -0.3 at 2006. Seed 7700.
Run: OMP_NUM_THREADS=1 python audit/mc7b_pipeline_two_stage.py <R>"""
import sys, json, warnings
import numpy as np
import tspec
warnings.simplefilter("ignore")
R = int(sys.argv[1]) if len(sys.argv) > 1 else 200
rng = np.random.default_rng(7700)
cov_seg = [0, 0]; cov_band = []; ok = 0; spurious_jump = 0; cov_band_all = []
for r in range(R):
    t = np.sort(rng.uniform(2000, 2012, 288))
    T = 3 + 0.4 * (t - 2000) - 0.7 * np.maximum(0, t - 2006)
    y = T + 0.6 * np.cos(2 * np.pi * t) + 0.2 * np.sin(4 * np.pi * t) + rng.normal(0, 0.2, t.size)
    p = tspec.analyze_series(t, y)
    spurious_jump += len(p.jumps.jumps) > 0
    band = (T >= p.trend_ci_low) & (T <= p.trend_ci_high) if p.trend_ci_low is not None else None
    if band is not None:
        cov_band_all.append(band.mean())
    tps = p.turning_points.tp_times if p.turning_points is not None else []
    if len(p.jumps.jumps) == 0 and len(tps) == 1 and abs(float(tps[0]) - 2006) < 0.5:
        ok += 1
        s0, s1 = p.segments
        cov_seg[0] += (s0.slope_ci_low <= 0.4 <= s0.slope_ci_high) + (s1.slope_ci_low <= -0.3 <= s1.slope_ci_high)
        cov_seg[1] += 2
        if band is not None:
            cov_band.append(band.mean())
res = {"R": R, "correct_structure_rate": ok / R, "spurious_jump_rate": spurious_jump / R,
       "segment_slope_CI95_cov|correct": cov_seg[0] / max(cov_seg[1], 1), "n_slopes": cov_seg[1],
       "trend_band_cov|correct": float(np.mean(cov_band)) if cov_band else None,
       "trend_band_cov_all": float(np.mean(cov_band_all)) if cov_band_all else None}
print(json.dumps(res))
json.dump(res, open("audit/mc7b_results.json", "w"), indent=1)
