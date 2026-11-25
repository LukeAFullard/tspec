"""Breakdown of 'l1' (claimed 50% vertical breakdown) and robust='mm' (claimed Yohai 1987 MM) with
vertical outliers only (no x-leverage manipulation): 35% of observations (the last 35% in time)
shifted by +M. A 50%-breakdown estimator's slope error must stay bounded as M -> inf.
Reference high-breakdown fit: simple LTS (h = floor(n/2)+2) via random elemental starts + C-steps.
Seed 21. Run: python audit/repro_breakdown.py
"""
import warnings
import numpy as np
import tspec
from tspec._linalg import prepare_weights
from tspec._robust import fit_l1_piecewise, solve_robust_irls
warnings.simplefilter("ignore")
rng = np.random.default_rng(21)
n = 100
t = np.linspace(0.0, 10.0, n)
base = 1.0 + 0.5 * t + rng.normal(0, 0.3, n)
D = np.column_stack([np.ones(n), t])

def lts(X, y, h, starts=500, rs=np.random.default_rng(0)):
    best = (np.inf, None)
    for _ in range(starts):
        idx = rs.choice(len(y), X.shape[1], replace=False)
        try:
            b = np.linalg.solve(X[idx], y[idx])
        except np.linalg.LinAlgError:
            continue
        for _ in range(20):
            r2 = (y - X @ b) ** 2
            sub = np.argsort(r2)[:h]
            b_new = np.linalg.lstsq(X[sub], y[sub], rcond=None)[0]
            if np.allclose(b_new, b):
                break
            b = b_new
        obj = np.sort((y - X @ b) ** 2)[:h].sum()
        if obj < best[0]:
            best = (obj, b)
    return best[1]

print(" contamination 35% (last 35 points), true slope 0.5")
print(f" {'M':>6} | {'L1 slope':>9} | {'huber':>8} | {'mm':>8} | {'LTS':>8}")
for M in (5, 20, 100, 1000):
    y = base.copy(); y[-35:] += M
    s_l1 = fit_l1_piecewise(t, y, np.empty(0, dtype=np.intp))[0][0]
    hu = solve_robust_irls(D, y, prepare_weights(None, n), method="huber").ols.coeff[1]
    mm = solve_robust_irls(D, y, prepare_weights(None, n), method="mm").ols.coeff[1]
    lt = lts(D, y, n // 2 + 2)[1]
    print(f" {M:>6} | {s_l1:9.3f} | {hu:8.3f} | {mm:8.3f} | {lt:8.3f}")
print("\n 20% isolated random spikes of +M (the case the docs target)")
for M in (20, 1000):
    y = base.copy(); idx = rng.choice(n, 20, replace=False); y[idx] += M
    s_l1 = fit_l1_piecewise(t, y, np.empty(0, dtype=np.intp))[0][0]
    hu = solve_robust_irls(D, y, prepare_weights(None, n), method="huber").ols.coeff[1]
    mm = solve_robust_irls(D, y, prepare_weights(None, n), method="mm").ols.coeff[1]
    print(f" {M:>6} | {s_l1:9.3f} | {hu:8.3f} | {mm:8.3f}")
# public API path: fit_piecewise_trend(mode='l1') with no TP
y = base.copy(); y[-35:] += 1000
print("\n fit_piecewise_trend(mode='l1') slope with 35% shifted by +1000:",
      float(tspec.fit_piecewise_trend(t, y, [], mode="l1").slopes[0]))
