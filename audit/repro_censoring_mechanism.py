"""(1) Imputed values for censored points ignore time (marginal model) -> trend attenuation.
(2) DL-shift coincidence / recensor_check behaviour: real jump coinciding with a DL drop, and a DL
    drop without any real change. Seeds 61-63. Run: python audit/repro_censoring_mechanism.py
"""
import warnings
import numpy as np
import tspec
from tspec.censored import CensoredConfig, prepare_series_with_censoring
warnings.simplefilter("ignore")

# (1) mechanism
rng = np.random.default_rng(61)
n = 150; t = np.sort(rng.uniform(2000, 2006.25, n)); mu = 2 + 0.1 * (t - 2000)
corr_true = []; corr_imp = []; slope_true_cens = []; slope_imp_cens = []
for r in range(300):
    y = mu + rng.normal(0, 0.4, n)
    dl = np.quantile(y, 0.5); cen = y < dl
    ys = np.array([f"<{dl:.6g}" if c else float(v) for c, v in zip(cen, y)], dtype=object)
    prep = prepare_series_with_censoring(t, ys, censoring=CensoredConfig(random_state=r))
    yi = prep.series.values
    slope_true_cens.append(np.polyfit(t[cen], y[cen], 1)[0]); slope_imp_cens.append(np.polyfit(t[cen], yi[cen], 1)[0])
    corr_true.append(np.corrcoef(t[cen], y[cen])[0, 1]); corr_imp.append(np.corrcoef(t[cen], yi[cen])[0, 1])
print("(1) among censored points: mean corr(t, true y)=%.3f, corr(t, imputed y)=%.3f" % (np.mean(corr_true), np.mean(corr_imp)))
print("    OLS slope over censored points only: true %.4f vs imputed %.4f (overall truth 0.1)" % (np.mean(slope_true_cens), np.mean(slope_imp_cens)))
print("    -> imputation draws from a time-invariant fitted distribution truncated at the DL")

# (2) DL logic
def run(real_jump, R=100, seed=62):
    rng = np.random.default_rng(seed)
    flags = {"coincide": 0, "artifact": 0, "jump_found": 0, "dlshift_detected": 0}
    for r in range(R):
        n = 192; t = np.sort(rng.uniform(2000, 2008, n)); j = n // 2
        y = 1.0 + 0.15 * np.cos(2 * np.pi * t) + rng.normal(0, 0.15, n) + real_jump * (np.arange(n) >= j)
        y = np.exp(y - 1.0)  # positive, lognormal-ish
        dl = np.where(np.arange(n) < j, 0.95, 0.75)  # lab DL drops at the same time
        cen = y < dl
        ys = np.array([f"<{d:.3g}" if c else float(v) for c, v, d in zip(cen, y, dl)], dtype=object)
        cfg = CensoredConfig(recensor_check=True, random_state=r)
        try:
            res = tspec.just_jumps(t, ys, season="ols", jump_alpha=0.05, bonferroni_window=True,
                                   step_alpha=0.05, mag_th=0.0, dir_th=0.0, censoring=cfg)
        except Exception as e:
            continue
        flags["dlshift_detected"] += len(res.censoring.dl_shifts) > 0
        near = [q for q in res.jumps if abs(q.time_year - t[j]) < 0.5]
        if near:
            flags["jump_found"] += 1
            flags["coincide"] += near[0].coincides_with_dl_shift
            flags["artifact"] += bool(near[0].dl_artifact_likely)
    return {k: v / R for k, v in flags.items()}
print("(2a) REAL +0.4 (log-scale) jump coinciding with DL drop:", run(0.4))
print("(2b) NO real change, DL drop only:                     ", run(0.0, seed=63))
