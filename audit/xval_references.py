"""Independent cross-validation of tspec kernels against reference libraries.
Seed: 20261125. Run: python audit/xval_references.py
"""
import warnings
import numpy as np
import statsmodels.api as sm
from astropy.timeseries import LombScargle
import pymannkendall as mk
from scipy import stats
import tspec
from tspec._linalg import prepare_weights, solve_weighted_ols
from tspec._robust import solve_huber_irls, fit_theil_sen_segment, _kendall_var_s_ties
from tspec.spectral import build_design_matrix

warnings.simplefilter("ignore")
rng = np.random.default_rng(20261125)
out = {}

# 1. WLS vs statsmodels on decimal-year time axis with jumps + harmonics
n = 300
t = np.sort(rng.uniform(2000.0, 2015.0, n))
w = rng.uniform(0.2, 3.0, n)
y = 5 + 0.2 * (t - 2000) + 1.5 * (t > 2007.3) + np.cos(2 * np.pi * t) + rng.normal(0, 1 / np.sqrt(w))
j = int(np.searchsorted(t, 2007.3))
t0 = t - t[0]
D, cs = build_design_matrix(t0, "linear", np.array([j]), False, np.array([1.0, 2.0]))
ols = solve_weighted_ols(D, y, prepare_weights(w, n), col_scales=cs)
Du = D / cs
ref = sm.WLS(y, Du, weights=w).fit()
out["wls_coef_maxrel"] = float(np.max(np.abs(ols.coeff - ref.params) / np.maximum(1, np.abs(ref.params))))
out["wls_se_maxrel"] = float(np.max(np.abs(np.sqrt(np.diag(ols.cov)) - ref.bse) / ref.bse))
# same through public lssa (known_freqs) -> coeff & cov
r = tspec.lssa(t, y, weights=w, jump_indices=[j], known_freqs=[1.0, 2.0])
out["lssa_public_coef_maxrel"] = float(np.max(np.abs(r.coeff - ref.params) / np.maximum(1, np.abs(ref.params))))

# 1b. Unscaled raw decimal-year polynomial (cubic) conditioning
yc = 1 + 0.1 * (t - 2000) - 0.01 * (t - 2000) ** 2 + 0.001 * (t - 2000) ** 3 + rng.normal(0, 0.1, n)
rc = tspec.lssa(t, yc, trend="cubic")
X = np.column_stack([np.ones(n), t0, t0 ** 2, t0 ** 3])
refc = sm.OLS(yc, X).fit()
out["cubic_coef_maxrel"] = float(np.max(np.abs(rc.coeff - refc.params) / np.abs(refc.params)))
out["cubic_cond"] = rc.diagnostics.condition_number_max

# 2. LSS vs astropy LombScargle (constant trend, unweighted; and weighted)
n2 = 120
t2 = np.sort(rng.uniform(0, 10, n2))
y2 = 0.7 * np.sin(2 * np.pi * 1.3 * t2) + rng.normal(0, 1, n2)
f = np.linspace(0.05, 5, 400)
r2 = tspec.lssa(t2, y2, omega=f, trend="constant")
ls = LombScargle(t2, y2, fit_mean=True, center_data=True, normalization="standard")
p_ast = ls.power(f, method="cython", assume_regular_frequency=False)
out["lssa_vs_astropy_maxabs"] = float(np.max(np.abs(r2.spectrum - p_ast)))
dy = rng.uniform(0.5, 2.0, n2)
r2w = tspec.lssa(t2, y2, omega=f, trend="constant", weights=1 / dy ** 2)
lsw = LombScargle(t2, y2, dy, fit_mean=True, center_data=True, normalization="standard")
out["lssa_vs_astropy_weighted_maxabs"] = float(np.max(np.abs(r2w.spectrum - lsw.power(f, method="cython"))))
# trend='none' vs astropy fit_mean=False center_data=False
r2n = tspec.lssa(t2, y2, omega=f, trend="none")
lsn = LombScargle(t2, y2, fit_mean=False, center_data=False, normalization="standard")
out["lssa_vs_astropy_nomean_maxabs"] = float(np.max(np.abs(r2n.spectrum - lsn.power(f, method="cython"))))

# 2b. Beta critical value vs scipy.stats.beta
for (nn, nc, a) in [(50, 2, 0.05), (200, 4, 0.01), (30, 6, 0.001)]:
    from tspec.spectral import critical_value_lss
    cv = critical_value_lss(nn, nc, a)
    cv_ref = stats.beta.ppf(1 - a, 1, (nn - nc - 2) / 2)
    out[f"beta_cv_absdiff_n{nn}_nc{nc}_a{a}"] = abs(cv - cv_ref)

# 3. Huber IRLS vs statsmodels RLM(HuberT, scale MAD)
n3 = 200
x3 = np.linspace(0, 1, n3)
y3 = 1 + 2 * x3 + stats.t.rvs(3, size=n3, random_state=rng) * 0.5
D3 = np.column_stack([np.ones(n3), x3])
hr = solve_huber_irls(D3, y3, prepare_weights(None, n3), k=1.345)
rlm = sm.RLM(y3, D3, M=sm.robust.norms.HuberT(t=1.345)).fit(cov="H1")
out["huber_coef_absdiff"] = float(np.max(np.abs(hr.ols.coeff - rlm.params)))
out["huber_se_rel(tspec/sm_H1)"] = (np.sqrt(np.diag(hr.ols.cov)) / rlm.bse).tolist()
out["huber_scale_tspec_vs_sm"] = [hr.scale, float(rlm.scale)]

# 4. Theil-Sen & MK vs pymannkendall / scipy
n4 = 80
t4 = np.arange(n4, dtype=float)
y4 = 0.02 * t4 + rng.normal(0, 1, n4)
y4 = np.round(y4, 1)  # induce ties
fit = fit_theil_sen_segment(t4, y4, alpha=0.05)
pm = mk.original_test(y4)
ss = stats.theilslopes(y4, t4, alpha=0.95)
out["sen_slope_vs_pymk"] = [fit.slope, float(pm.slope)]
out["sen_slope_vs_scipy"] = [fit.slope, float(ss.slope)]
out["mk_p_tspec_vs_pymk"] = [fit.pvalue, float(pm.p)]
out["mk_varS_tspec_vs_pymk"] = [_kendall_var_s_ties(y4, t4), float(pm.var_s)]
out["sen_ci_tspec"] = [fit.ci_low, fit.ci_high]
out["sen_ci_scipy_theilslopes"] = [float(ss.low_slope), float(ss.high_slope)]
# R trend::sens.slope convention: rank.lo=round((N-C)/2), rank.up=round((N+C)/2+1)
i, jj = np.triu_indices(n4, 1)
sl = np.sort((y4[jj] - y4[i]) / (t4[jj] - t4[i]))
N = sl.size
C = stats.norm.isf(0.025) * np.sqrt(pm.var_s)
lo_R = sl[int(np.round((N - C) / 2)) - 1]
hi_R = sl[int(np.round((N + C) / 2 + 1)) - 1]
out["sen_ci_R_trend_convention"] = [float(lo_R), float(hi_R)]
for k, v in out.items():
    print(f"{k}: {v}")
