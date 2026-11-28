"""P1: independent spectral oracles + null calibration for lssa/allssa."""
import json, time, warnings
import numpy as np, scipy, statsmodels.api as sm
from scipy import stats, signal
import tspec
warnings.filterwarnings("ignore")
out = {}
rng = np.random.default_rng(1)

# ---- 1. exact LSS oracle vs statsmodels OLS/WLS/GLS (nested-model variance reduction)
def oracle(t, y, f, W=None, trend="linear"):
    t0 = t - t[0]
    X0 = np.column_stack([np.ones_like(t0), t0]) if trend == "linear" else np.ones((t.size, 1))
    X1 = np.column_stack([X0, np.cos(2*np.pi*f*t0), np.sin(2*np.pi*f*t0)])
    if W is None:
        r0 = sm.OLS(y, X0).fit().ssr; r1 = sm.OLS(y, X1).fit().ssr
    elif W.ndim == 1:
        r0 = sm.WLS(y, X0, weights=W).fit().ssr; r1 = sm.WLS(y, X1, weights=W).fit().ssr
    else:
        S = np.linalg.inv(W)
        r0 = sm.GLS(y, X0, sigma=S).fit().ssr; r1 = sm.GLS(y, X1, sigma=S).fit().ssr
    return (r0 - r1) / r0
n = 150
t = np.sort(rng.uniform(0, 10, n)); y = 0.3*t + np.sin(2*np.pi*1.3*t) + rng.normal(0, 1, n)
freqs = np.linspace(0.05, 6, 120)
w1 = rng.uniform(0.2, 3, n)
A = rng.normal(size=(n, n)); C = 0.5*np.exp(-np.abs(t[:, None]-t[None, :])/0.3) + np.eye(n)*0.5; P = np.linalg.inv(C)
errs = {}
for name, W in [("ols", None), ("wls_1d", w1), ("gls_2d", P)]:
    r = tspec.lssa(t, y, omega=freqs, weights=W)
    o = np.array([oracle(t, y, f, W) for f in freqs])
    errs[name] = float(np.max(np.abs(r.spectrum - o)))
r = tspec.lssa(t, y, omega=freqs, trend="constant")
# scipy GLS lombscargle with floating mean & normalize -> fraction of variance (Zechmeister-Kurster)
o = np.array([oracle(t, y, f, None, "constant") for f in freqs])
sp = signal.lombscargle(t, y, 2*np.pi*freqs, floating_mean=True, normalize=True)
errs["const_vs_statsmodels"] = float(np.max(np.abs(r.spectrum - o)))
errs["const_vs_scipy_lombscargle"] = float(np.max(np.abs(r.spectrum - sp)))
# coefficient & covariance vs statsmodels with known freq
r = tspec.lssa(t, y, known_freqs=[1.3])
t0 = t - t[0]; X = np.column_stack([np.ones(n), t0, np.cos(2*np.pi*1.3*t0), np.sin(2*np.pi*1.3*t0)])
f = sm.OLS(y, X).fit()
errs["coef_vs_statsmodels"] = float(np.max(np.abs(r.coeff - f.params)))
errs["cov_vs_statsmodels_rel"] = float(np.max(np.abs(r.cov - f.cov_params()) / np.abs(f.cov_params()).max()))
# large offsets / scales
for lab, tt, yy in [("t+1e7", t + 1e7, y), ("y+1e9", t, y + 1e9), ("y*1e-12", t, y*1e-12), ("y*1e12", t, y*1e12)]:
    rr = tspec.lssa(tt, yy, omega=freqs)
    errs[f"offset_{lab}"] = float(np.max(np.abs(rr.spectrum - tspec.lssa(t, y, omega=freqs).spectrum)))
out["oracle_max_abs_err"] = errs
print("oracle errs", errs)

# ---- 2. per-frequency Beta null: KS uniformity of p at a fixed freq, irregular+gapped, weights
def ks_null(nrep=3000, n=60, hetero=False):
    ps = []
    for i in range(nrep):
        tt = np.sort(rng.uniform(0, 5, n)); tt = tt[(tt < 2) | (tt > 3.2)]
        sd = np.where(tt > 2.5, 3.0, 1.0) if hetero else np.ones_like(tt)
        yy = rng.normal(0, sd)
        rr = tspec.lssa(tt, yy, omega=[1.7], weights=1/sd**2 if hetero else None)
        ps.append(rr.pvalues[0])
    ps = np.array(ps)
    return dict(ks_p=float(stats.kstest(ps, "uniform").pvalue), rej05=float(np.mean(ps < 0.05)), rej01=float(np.mean(ps < 0.01)))
out["beta_null_homo"] = ks_null()
out["beta_null_hetero_weighted"] = ks_null(hetero=True)
print(out["beta_null_homo"], out["beta_null_hetero_weighted"])

# ---- 3. peak FAP calibration (Sidak vs Monte Carlo), dense grid, white noise
def fap_cal(method, nrep, n=100, grid=np.linspace(0.1, 10, 600), irregular=True, noise="gauss"):
    faps = []
    for i in range(nrep):
        tt = np.sort(rng.uniform(0, 8, n)) if irregular else np.arange(n)*0.08
        e = rng.normal(size=n) if noise == "gauss" else rng.standard_t(3, n)
        rr = tspec.lssa(tt, e, omega=grid, peak_fap=method, n_mc=199)
        faps.append(rr.peak_fap)
    faps = np.array(faps)
    return dict(rej05=float(np.mean(faps <= 0.05)), rej10=float(np.mean(faps <= 0.10)), nrep=nrep)
t1 = time.time()
out["fap_sidak_irregular"] = fap_cal("sidak", 1500)
out["fap_sidak_regular"] = fap_cal("sidak", 1500, irregular=False)
out["fap_sidak_narrowband"] = fap_cal("sidak", 1500, grid=np.linspace(1.0, 1.5, 200))
out["fap_sidak_t3"] = fap_cal("sidak", 1500, noise="t3")
out["fap_mc_irregular"] = fap_cal("montecarlo", 300)
print({k: v for k, v in out.items() if k.startswith("fap")}, time.time()-t1)

# ---- 4. ALLSSA: null false-cycle rate (default significance) + non-integer recovery
def allssa_null(nrep, sig):
    k = 0
    for i in range(nrep):
        tt = np.sort(rng.uniform(0, 10, 120))
        rr = tspec.allssa(tt, rng.normal(size=120), significance=sig, n_mc=499)
        k += rr.frequencies.size > 0
    return k / nrep
out["allssa_null_fpr_alpha01_montecarlo"] = allssa_null(200, "montecarlo")
out["allssa_null_fpr_alpha01_sidak"] = allssa_null(300, "sidak")
out["allssa_null_fpr_alpha01_per_frequency"] = allssa_null(300, "per_frequency")
rec = []
for i in range(50):
    tt = np.sort(rng.uniform(0, 12, 200))
    yy = 1.0*np.cos(2*np.pi*1.37*tt+0.4) + 0.6*np.sin(2*np.pi*1.0*tt) + rng.normal(0, 0.5, 200)
    rr = tspec.allssa(tt, yy, decimal=2, n_mc=499)
    fr = rr.frequencies
    j = np.argmin(np.abs(fr-1.37)) if fr.size else None
    rec.append(dict(nf=int(fr.size), f137=float(fr[j]) if j is not None else np.nan, a137=float(rr.amplitudes[j]) if j is not None else np.nan,
                    has1=bool(np.any(np.abs(fr-1.0) < 0.02))))
f = np.array([r["f137"] for r in rec]); a = np.array([r["a137"] for r in rec])
out["allssa_noninteger"] = dict(freq_bias=float(np.nanmean(f-1.37)), freq_sd=float(np.nanstd(f)), amp_bias=float(np.nanmean(a-1.0)),
                               found_both=float(np.mean([(abs(r["f137"]-1.37) < 0.02) and r["has1"] for r in rec])),
                               mean_nfreq=float(np.mean([r["nf"] for r in rec])))
# low-frequency cycle (11.2-yr) outside default grid?
tt = np.arange(0, 60, 1/12); yy = 2*np.sin(2*np.pi*tt/11.2) + np.cos(2*np.pi*tt) + rng.normal(0, .3, tt.size)
rr = tspec.allssa(tt, yy, n_mc=499)
out["allssa_default_grid_11p2yr_cycle"] = dict(selected=rr.frequencies.tolist(), true=[1/11.2, 1.0])
rr = tspec.allssa(tt, yy, omega=np.arange(0.05, 5, 0.05), decimal=2, n_mc=499)
out["allssa_custom_grid_11p2yr_cycle"] = dict(selected=rr.frequencies.tolist())
print(out["allssa_noninteger"], out["allssa_default_grid_11p2yr_cycle"], out["allssa_custom_grid_11p2yr_cycle"])
print({k: v for k, v in out.items() if "null_fpr" in k})
json.dump(out, open("p1_results.json", "w"), indent=1, default=float)
