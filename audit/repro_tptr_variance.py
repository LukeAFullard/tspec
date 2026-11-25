"""Brute-force check of the 'exact O(n) sequential TPTR variance' and joint hinge covariance.
The fitted curve is linear in y for fixed knots, so H (y_hat = H y) is obtained column-by-column
by fitting unit vectors; Var(y_hat_i)/sigma^2 = sum_j H_ij^2 / w_j. Seed 7.
Run: python audit/repro_tptr_variance.py
"""
import numpy as np
from tspec.stpd import _chained_forward_fit, _stpd_pvalue
from tspec._linalg import prepare_weights, solve_weighted_ols
rng = np.random.default_rng(7)
n = 60
t = np.sort(rng.uniform(0, 5, n)); t -= t[0]
w = rng.uniform(0.5, 2.0, n)
tps = np.array([18, 41])
for weights in (None, w):
    _, _, _, vfac = _chained_forward_fit(t, rng.normal(size=n), tps, weights)
    H = np.empty((n, n))
    for j in range(n):
        e = np.zeros(n); e[j] = 1.0
        H[:, j] = _chained_forward_fit(t, e, tps, weights)[2]
    wv = np.ones(n) if weights is None else weights
    v_bf = np.sum(H ** 2 / wv[None, :], axis=1)
    print("sequential: weights=", "none" if weights is None else "vector",
          " max rel diff v_fac vs brute force =", float(np.max(np.abs(vfac - v_bf) / v_bf)))
# a1 - a2 variance in _stpd_pvalue: brute-force via linear functional on unit vectors
ind = 25
def seq_slopes(y):
    t1, y1 = t[:ind + 1], y[:ind + 1]
    a1 = np.polyfit(t1, y1, 1)[0]; b1 = np.mean(y1) - a1 * np.mean(t1)
    d2 = t[ind:] - t[ind]; y2 = y[ind:] - (b1 + a1 * t[ind])
    a2 = np.dot(d2, y2) / np.dot(d2, d2)
    return a1, a2, b1
v = np.array([seq_slopes(np.eye(n)[j])[0] - seq_slopes(np.eye(n)[j])[1] for j in range(n)])
y = 1 + 0.5 * t + rng.normal(0, 0.3, n)
a1, a2, b1 = seq_slopes(y)
# reproduce tspec's SE from its p-value: invert t-stat
from scipy.stats import t as tdist
p = _stpd_pvalue(t, y, ind, b1, a1, a2, None)
tstat = tdist.isf(p / 2, n - 3); se_tspec = abs(a1 - a2) / tstat
r1 = y[:ind + 1] - (b1 + a1 * t[:ind + 1]); r2 = (y[ind + 1:] - (b1 + a1 * t[ind])) - a2 * (t[ind + 1:] - t[ind])
s2 = (r1 @ r1 + r2 @ r2) / (n - 3)
print("a1-a2 SE tspec =", se_tspec, " brute force =", float(np.sqrt(s2 * v @ v)))
