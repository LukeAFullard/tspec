import sys, warnings, numpy as np
from scipy.stats import norm, rankdata
from tspec._robust import fit_theil_sen_segment, hamed_rao_factor, _kendall_var_s_ties
warnings.simplefilter("ignore")
R = int(sys.argv[1])
def ar1(rng, n, phi):
    e = rng.normal(size=n); x = np.empty(n); x[0] = e[0] / np.sqrt(1 - phi**2)
    for i in range(1, n): x[i] = phi * x[i-1] + e[i]
    return x
def ar_factor(t, y, slope):
    n = y.size; r = y - slope * t; r = r - r.mean()
    rho = float(np.dot(r[:-1], r[1:]) / np.dot(r, r))
    rho = (n * rho + 1.0) / (n - 4.0)           # small-sample bias correction (mean + trend removed)
    rho = float(np.clip(rho, 0.0, 0.95))
    k = np.arange(1, n)
    rs = (6 / np.pi) * np.arcsin(rho ** k / 2)    # rank (Spearman) autocorrelation of Gaussian AR(1)
    return 1.0 + 2.0 * np.sum((n - k) * (n - k - 1) * (n - k - 2) * rs) / (n * (n - 1) * (n - 2))
def test(t, y, kind, beta=None):
    n = y.size; i, j = np.triu_indices(n, 1); dy = y[j] - y[i]; dt = t[j] - t[i]
    sl = np.sort(dy / dt); slope = np.median(sl); S = np.sum(np.sign(dy)); v = _kendall_var_s_ties(y, t)
    f = {"none": 1.0, "hr_trunc": max(1.0, hamed_rao_factor(t, y, slope)), "ar1": ar_factor(t, y, slope)}[kind]
    v *= f; z = (S - np.sign(S)) / np.sqrt(v) if S else 0.0; p = 2 * norm.sf(abs(z))
    C = norm.isf(0.025) * np.sqrt(v); N = sl.size
    lo = sl[max(0, int(round((N - C) / 2)) - 1)]; hi = sl[min(N - 1, int(round((N + C) / 2)))]
    return p, lo, hi
for n in (50, 150, 500):
    for phi in (0.0, 0.3, 0.5):
        rng = np.random.default_rng(9000 + n + int(phi * 10)); t = np.sort(rng.uniform(2000, 2000 + n / 12, n))
        Rn = R if n < 500 else R // 3; res = {k: [0, 0] for k in ("none", "hr_trunc", "ar1")}
        for r in range(Rn):
            e = ar1(rng, n, phi)
            for k in res:
                p, _, _ = test(t, e, k); res[k][0] += p < 0.05
                _, lo, hi = test(t, 0.3 * (t - t[0]) + e, k); res[k][1] += lo <= 0.3 <= hi
        print(n, phi, Rn, {k: (round(v[0] / Rn, 3), round(v[1] / Rn, 3)) for k, v in res.items()}, flush=True)
