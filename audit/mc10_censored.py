"""MC10: censored pipeline. Truth y = 2 + 0.1 (t-2000) + N(0, 0.4^2), ~24 obs/yr, n=150.
Left-censoring at a constant DL giving ~20% or ~50% censoring (strings '<DL' passed to tspec).
 (a) slope via analyze_series(detect_jumps=False, detect_turning_points=False, remove_seasonality=False)
     with slope_estimator 'ols' (on single stochastic ROS imputation) and 'sen' (auto ATS):
     bias and 95% CI coverage, vs. the same estimator on the uncensored data.
 (b) jump: + 1.0 step at mid-record; just_jumps(season='none') magnitude bias of the candidate
     nearest the true jump and analyze-style detection (Bonferroni + jump/step alpha 0.05).
Seed base 10000. Run: OMP_NUM_THREADS=1 python audit/mc10_censored.py <R>
"""
import sys, json, warnings
import numpy as np
import tspec
warnings.simplefilter("ignore")
R = int(sys.argv[1]) if len(sys.argv) > 1 else 500
BETA = 0.1
out = {}
for cfrac in (0.2, 0.5):
    rng = np.random.default_rng(10000 + int(100 * cfrac))
    n = 150
    stats = {k: [] for k in ("ols_unc", "ols_cen", "sen_unc", "ats_cen")}
    cov = {k: 0 for k in stats}; nok = {k: 0 for k in stats}
    jb = {"unc": [], "cen": []}; jdet = {"unc": 0, "cen": 0}; realized = []
    pop_rng = np.random.default_rng(424242)
    tp_ = np.linspace(2000, 2000 + n / 24, 200000)
    dl = float(np.quantile(2 + BETA * (tp_ - 2000) + pop_rng.normal(0, 0.4, tp_.size), cfrac))  # fixed DL per config
    for r in range(R):
        t = np.sort(rng.uniform(2000, 2000 + n / 24, n))
        mu = 2 + BETA * (t - 2000)
        y = mu + rng.normal(0, 0.4, n)
        cen = y < dl
        realized.append(cen.mean())
        ys = np.array([f"<{dl:.6g}" if c else float(v) for c, v in zip(cen, y)], dtype=object)
        for est in ("ols", "sen"):
            for which, yy in (("unc", y), ("cen", ys)):
                try:
                    a = tspec.analyze_series(t, yy, detect_jumps=False, detect_turning_points=False,
                                             remove_seasonality=False, slope_estimator=est)
                except Exception as e:
                    continue
                s = a.segments[0]
                k = f"{est}_{which}" if not (est == "sen" and which == "cen") else "ats_cen"
                stats[k].append(s.slope - BETA)
                if s.slope_ci_low is not None:
                    nok[k] += 1; cov[k] += s.slope_ci_low <= BETA <= s.slope_ci_high
        if r < min(R, 300):
            j = n // 2
            yj = y + 1.0 * (np.arange(n) >= j)
            cenj = yj < dl
            ysj = np.array([f"<{dl:.6g}" if c else float(v) for c, v in zip(cenj, yj)], dtype=object)
            for which, yy in (("unc", yj), ("cen", ysj)):
                try:
                    res = tspec.just_jumps(t, yy, season="none", mag_th=0.0, dir_th=0.0)
                except Exception:
                    continue
                if res.jumps:
                    tt = np.array([q.time_year for q in res.jumps])
                    q = res.jumps[int(np.argmin(np.abs(tt - t[j])))]
                    if abs(q.time_year - t[j]) < 0.5:
                        jb[which].append(q.magnitude - 1.0)
                    c0 = res.all_window_candidates[0]; mult = max(1, c0.window_stop - c0.window_start - 5)
                    jdet[which] += any(abs(q2.time_year - t[j]) < 0.5 and min(1, q2.jump_pvalue * mult) < 0.05
                                       and min(1, q2.step_pvalue * mult) < 0.05 for q2 in res.jumps)
    key = f"censor_frac~{cfrac}"
    out[key] = {"R": R, "realized_censor_frac": float(np.mean(realized))}
    for k, v in stats.items():
        v = np.array(v)
        out[key][k] = {"bias": float(v.mean()), "rmse": float(np.sqrt(np.mean(v ** 2))),
                       "CI95_cov": cov[k] / nok[k] if nok[k] else None, "n": int(v.size),
                       "MCSE_cov": float(np.sqrt(0.0475 / max(nok[k], 1)))}
    Rj = min(R, 300)
    for which in ("unc", "cen"):
        v = np.array(jb[which])
        out[key][f"jump_{which}"] = {"mag_bias": float(v.mean()) if v.size else None,
                                     "mag_rmse": float(np.sqrt(np.mean(v ** 2))) if v.size else None,
                                     "detect_rate(analyze-style)": jdet[which] / Rj, "R": Rj}
    print(key, json.dumps(out[key]), flush=True)
json.dump(out, open("audit/mc10_results.json", "w"), indent=1)
