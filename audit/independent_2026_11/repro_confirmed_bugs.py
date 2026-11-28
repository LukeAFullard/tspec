"""Minimal reproducers for the confirmed defects of the independent 2026-11 audit.
IDs match audit/INDEPENDENT_AUDIT_2026-11.md (AUD-01, -02, -05, -06, -09, -10).

Each block prints OBSERVED vs EXPECTED and ends with `BUG CONFIRMED` / `ok`.
Runtime < 1 min single-threaded:  OMP_NUM_THREADS=1 python repro_confirmed_bugs.py
"""
import warnings
import numpy as np
import tspec
from common import ar1
warnings.filterwarnings("ignore")

def verdict(cond):
    return "BUG CONFIRMED" if cond else "ok"

# ---------------------------------------------------------------- AUD-01 (Critical)
# datetime -> decimal-year conversion uses each calendar year's own length (365 or 366 d),
# so the numeric time axis is non-uniform: a pure noiseless M2 tide can no longer be fitted.
dt = np.arange("2023-01-01T00", "2025-01-01T00", 3, dtype="datetime64[h]")
hours = (dt - dt[0]).astype(float)
y = np.cos(2 * np.pi * hours / 12.4206012)
grid = np.linspace(24 * 365 / 12.4206012 * 0.997, 24 * 366 / 12.4206012 * 1.003, 6001)
best = tspec.lssa(dt, y, omega=grid, trend="constant").spectrum.max()
ctrl = tspec.lssa(hours, y, omega=[1 / 12.4206012], trend="constant").spectrum[0]
print(f"AUD-01 datetime M2 best normalized LSS = {best:.3f} (numeric-hours control = {ctrl:.3f}; expected ~1.0): {verdict(best < 0.9)}")
tn = tspec.lssa(dt, y, omega=[700.0]).times_numeric
print(f"       decimal-year step ratio across 2023/2024 boundary = {np.diff(tn).max() / np.diff(tn).min():.5f} (expected 1.0)")

# ---------------------------------------------------------------- AUD-05 (Major)
# ALLSSA keeps at most ONE frequency per unit band [k-0.5, k+0.5): two real cycles at
# 1.00 and 1.37 cy/yr (or K1/O1, M2/S2 in cycles/day) can never both be selected.
rng = np.random.default_rng(0)
t = np.sort(rng.uniform(0, 12, 300))
y = 1.0 * np.cos(2 * np.pi * 1.37 * t) + 0.8 * np.sin(2 * np.pi * 1.0 * t) + rng.normal(0, 0.2, t.size)
a = tspec.allssa(t, y, decimal=2, n_mc=499)
both = np.any(np.abs(a.frequencies - 1.0) < 0.02) and np.any(np.abs(a.frequencies - 1.37) < 0.02)
print(f"AUD-05 allssa selected {a.frequencies.tolist()} for true {{1.00, 1.37}} at SNR~25: {verdict(not both)}")
# default grid silently ignores cycles slower than 0.5 cy/yr (e.g. 11.2-yr solar)
tt = np.arange(0, 60, 1 / 12); yy = 2 * np.sin(2 * np.pi * tt / 11.2) + np.cos(2 * np.pi * tt) + rng.normal(0, .3, tt.size)
a2 = tspec.allssa(tt, yy, n_mc=499)
print(f"       default-grid allssa on 11.2-yr + annual cycle selected {a2.frequencies.tolist()} (11.2-yr cycle missing, no warning: "
      f"{[w.code for w in a2.diagnostics.warnings]}): {verdict(not np.any(np.abs(a2.frequencies - 1/11.2) < 0.02))}")

# ---------------------------------------------------------------- AUD-06 (Major)
# High-SNR continuous bend is rejected as a "jump" by the NDRI screen (scale-free
# residual-asymmetry ratio degenerates when noise -> 0).
rng = np.random.default_rng(7); t = np.sort(rng.uniform(0, 20, 200))
for sd in (0.0, 1e-3):
    yb = 0.5 * t - np.maximum(t - 10, 0) + (rng.normal(0, sd, t.size) if sd else 0)
    s = tspec.stpd(t, yb, band_method="conditional")
    print(f"AUD-06 hinge (slope +0.5 -> -0.5 at t=10), noise sd={sd}: turning points {[round(p.time_year, 2) for p in s.turning_points]}, "
          f"flagged as jumps {[round(f.time_year, 2) for f in s.flagged_jumps]}: {verdict(len(s.turning_points) == 0)}")

# ---------------------------------------------------------------- AUD-09 (Minor)
# Zero-variance data: a perfectly straight line yields a "significant" jump of 6e-17 and
# analyze_series splits it into two segments (no zero-residual / effect-size guard).
yl = 3 + 0.2 * t
j = tspec.just_jumps(t, yl, season="none")
r = tspec.analyze_series(t, yl)
print(f"AUD-09 noiseless line: jumps {[(e.index, f'{e.magnitude:.1e}', f'{e.jump_pvalue:.1e}') for e in j.jumps]}, "
      f"analyze_series segments = {len(r.segments)}: {verdict(len(j.jumps) > 0)}")

# ---------------------------------------------------------------- AUD-10 (Minor)
# just_monitor accepts a 5-observation history for a trend + ALLSSA-seasonal model and
# raises an alarm, with no InsufficientData error or minimum-history warning.
y = 0.1 * t + np.cos(2 * np.pi * t) + rng.normal(0, .3, t.size) + 2.0 * (t > 10)
m = tspec.just_monitor(t, y, start=5)
print(f"AUD-10 just_monitor(start=5): detected={m.detected} at t={m.disturbance_time} (true break t=10); warnings "
      f"{[w.code for w in m.diagnostics.warnings]}: {verdict(m.detected and m.disturbance_time < 9)}")

# ---------------------------------------------------------------- AUD-02 (Critical, statistical)
# just_jumps family-wise false-jump rate under mild AR(1) serial correlation; does the
# STRONG_AUTOCORRELATION diagnostic fire on the false-positive series?
fp = warned = 0
N = 60
for s in range(N):
    rng = np.random.default_rng(s); t = np.sort(rng.uniform(0, 20, 240))
    yy = 0.05 * t + np.cos(2 * np.pi * t) + 0.4 * np.sin(4 * np.pi * t) + 0.5 * ar1(rng, 240, 0.3)
    r = tspec.just_jumps(t, yy, season="ols")
    if r.jumps:
        fp += 1
        warned += any("AUTOCORR" in w.code for w in r.diagnostics.warnings)
print(f"AUD-02 AR(1) rho=0.3 null: false-jump series {fp}/{N} (nominal <= 5%), of which {warned} carried an autocorrelation warning: "
      f"{verdict(fp / N > 0.15)}")
