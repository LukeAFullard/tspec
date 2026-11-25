"""Reporting / interpretation reproducers. Seeds as noted. Run: python audit/repro_reporting.py"""
import warnings
import numpy as np
import tspec
warnings.simplefilter("ignore")

print("=== R-A: lssa().explain() on pure white noise (seed 11) ===")
rng = np.random.default_rng(11)
t = np.sort(rng.uniform(2000, 2010, 240)); y = rng.normal(size=240)
r = tspec.lssa(t, y, omega=np.arange(0.1, 12, 0.05), alpha=0.05)
print(r.explain().splitlines()[0])

print("\n=== R-B: allssa().explain() on pure white noise (first seed with a selection) ===")
for s in range(200, 260):
    rng = np.random.default_rng(s)
    t = np.sort(rng.uniform(2000, 2010, 240)); y = rng.normal(size=240)
    a = tspec.allssa(t, y)
    if a.frequencies.size:
        print(f"seed={s}:", a.explain().splitlines()[0]); break

print("\n=== R-C: just_jumps defaults on NDVI-scale jump-free series (seed 12) ===")
rng = np.random.default_rng(12)
t = np.sort(rng.uniform(2000, 2006.25, 150))
y = 0.5 + 0.004 * (t - 2000) + 0.1 * np.cos(2 * np.pi * t) + rng.normal(0, 0.02, 150)
jj = tspec.just_jumps(t, y, season="ols")
print("\n".join(jj.explain().splitlines()[:3]))
print("warning codes:", [w.code for w in jj.diagnostics.warnings])
jj5 = tspec.just_jumps(t, y, season="ols", jump_alpha=0.05)
print("with jump_alpha=0.05 ->", len(jj5.jumps), "jump(s); p-values:", [f"{j.jump_pvalue:.3g}" for j in jj5.jumps])

print("\n=== R-D: Sen 'trend_ci' fan has zero width at the segment median time (seed 13) ===")
rng = np.random.default_rng(13)
t = np.sort(rng.uniform(2000, 2010, 200)); y = 1 + 0.2 * (t - 2000) + rng.normal(0, 0.5, 200)
p = tspec.analyze_series(t, y, detect_jumps=False, detect_turning_points=False,
                         remove_seasonality=False, slope_estimator="sen")
w = p.trend_ci_high - p.trend_ci_low
i = int(np.argmin(w))
print(f"min band width {w[i]:.3e} at t={t[i]:.3f} (segment median t={np.median(t):.3f}); max width {w.max():.3f}")
df = p.to_dataframe()
print("to_dataframe columns:", [c for c in df.columns if "ci" in c])
# coverage of the true trend line by this 'trend_ci' band, 1000 reps
cov = []
for s in range(1000):
    rr = np.random.default_rng(10_000 + s)
    yy = 1 + 0.2 * (t - 2000) + rr.normal(0, 0.5, 200)
    pp = tspec.analyze_series(t, yy, detect_jumps=False, detect_turning_points=False,
                              remove_seasonality=False, slope_estimator="sen")
    T = 1 + 0.2 * (t - 2000)
    cov.append(np.mean((T >= pp.trend_ci_low) & (T <= pp.trend_ci_high)))
cov = np.array(cov)
print(f"pointwise coverage of true trend by Sen 'trend_ci' band: mean {cov.mean():.3f} (MCSE {cov.std(ddof=1)/np.sqrt(cov.size):.3f})")
# coverage near the median
mid = np.abs(t - np.median(t)) < 0.5
cov_mid = []
for s in range(1000):
    rr = np.random.default_rng(10_000 + s)
    yy = 1 + 0.2 * (t - 2000) + rr.normal(0, 0.5, 200)
    pp = tspec.analyze_series(t, yy, detect_jumps=False, detect_turning_points=False,
                              remove_seasonality=False, slope_estimator="sen")
    T = 1 + 0.2 * (t - 2000)
    cov_mid.append(np.mean(((T >= pp.trend_ci_low) & (T <= pp.trend_ci_high))[mid]))
print(f"coverage within +-0.5 yr of the median time: {np.mean(cov_mid):.3f}")

print("\n=== R-E: two different slope numbers for the same segment in one PipelineResult (seed 14) ===")
rng = np.random.default_rng(14)
t = np.sort(rng.uniform(2000, 2012, 288)); ttp = 2006.0
y = 3 + 0.5 * (t - 2000) - 1.0 * np.maximum(0, t - ttp) + 0.6 * np.cos(2 * np.pi * t) + rng.normal(0, 0.15, t.size)
p = tspec.analyze_series(t, y)
for tp in p.turning_points.turning_points:
    print(f"STPD TP at {tp.time_year:.2f}: slope_before={tp.slope_before:+.4f} slope_after={tp.slope_after:+.4f}")
for s in p.segments:
    print(f"segment {s.segment_index}: slope={s.slope:+.4f} CI=[{s.slope_ci_low:+.4f},{s.slope_ci_high:+.4f}]")
d = p.to_dataframe()
print("combined_trend continuous at TP? jump in combined_trend at TP index:",
      None if not p.turning_points.turning_points else
      f"{d.combined_trend.iloc[p.turning_points.turning_points[0].index + 1] - d.combined_trend.iloc[p.turning_points.turning_points[0].index]:+.4f}")
