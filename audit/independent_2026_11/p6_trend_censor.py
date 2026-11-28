"""P6: Mann-Kendall/Sen Type-I under AR(1); censored slope bias/coverage (ols_mi, ats) vs DL/2."""
import json, sys, time, warnings
import numpy as np, tspec, statsmodels.api as sm
from common import ar1, pmap, wilson
warnings.filterwarnings("ignore")
NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 400
out = {"NREP": NREP}

def mk_job(a):
    seed, n, rho, corr, est = a
    rng = np.random.default_rng(seed)
    t = 2000 + np.arange(n) / 12.0
    y = ar1(rng, n, rho)
    r = tspec.analyze_series(t, y, detect_jumps=False, detect_turning_points=False, remove_seasonality=False,
                             slope_estimator=est, mk_correction=corr, band_method="conditional")
    s = r.segments[0]
    return (s.pvalue < 0.05, s.slope_ci_low <= 0.0 <= s.slope_ci_high)
for est in ["sen", "ols"]:
    for corr in (["ar1", "hamed_rao", "none"] if est == "sen" else ["ar1"]):
        for rho in [0.0, 0.5, 0.8]:
            for n in [60, 240]:
                res = pmap(mk_job, [(s, n, rho, corr, est) for s in range(NREP)])
                rej = sum(r[0] for r in res); cov = np.mean([r[1] for r in res])
                key = f"typeI_{est}_{corr}_rho{rho}_n{n}"
                out[key] = dict(reject=rej/NREP, ci95=wilson(rej, NREP), slope_ci_cov=float(cov)); print(key, out[key], flush=True)

def cens_job(a):
    seed, frac, two_dl, est = a
    rng = np.random.default_rng(50_000 + seed)
    n = 120; t = 2010 + np.sort(rng.uniform(0, 10, n)); b = 0.1
    y = 1.0 + b*(t - 2010) + rng.normal(0, 0.4, n)
    dl = np.full(n, np.quantile(y, frac))
    if two_dl:  # lab upgrade halfway: DL drops
        dl[t >= 2015] = np.quantile(y, frac * 0.6)
    cens = y < dl
    vals = np.where(cens, dl, y); status = np.where(cens, -1, 0)
    res = {}
    try:
        r = tspec.analyze_series(t, vals, censor_status=status, detect_jumps=False, detect_turning_points=False,
                                 remove_seasonality=False, slope_estimator=est, band_method="conditional",
                                 censoring=tspec.CensoredConfig(random_state=seed))
        s = r.segments[0]
        res["tspec"] = (s.slope - b, s.slope_ci_low <= b <= s.slope_ci_high, s.estimator)
    except Exception as ex:
        res["tspec"] = (np.nan, False, repr(ex)[:60])
    X = sm.add_constant(t - 2010)
    f = sm.OLS(np.where(cens, dl/2, y), X).fit(); ci = f.conf_int()[1]
    res["dl2_ols"] = (f.params[1] - b, ci[0] <= b <= ci[1], "dl/2")
    f = sm.OLS(y, X).fit(); ci = f.conf_int()[1]
    res["uncensored_oracle"] = (f.params[1] - b, ci[0] <= b <= ci[1], "oracle")
    return res
for frac in [0.3, 0.5]:
    for two_dl in [False, True]:
        for est in ["ols", "sen"]:
            res = pmap(cens_job, [(s, frac, two_dl, est) for s in range(200)])
            for k in ["tspec", "dl2_ols", "uncensored_oracle"]:
                b_ = np.array([r[k][0] for r in res], float)
                key = f"cens{frac}_{'twoDL' if two_dl else 'oneDL'}_{est}_{k}"
                out[key] = dict(bias=float(np.nanmean(b_)), rmse=float(np.sqrt(np.nanmean(b_**2))), ci_cov=float(np.mean([r[k][1] for r in res])),
                                estimator=res[0][k][2])
                print(key, out[key], flush=True)
json.dump(out, open("p6_results.json", "w"), indent=1, default=float)
