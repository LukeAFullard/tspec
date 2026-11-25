"""Compare JUST seasonal-selection rules (per_frequency vs sidak) on:
 (a) MC4 decomposition coverage (season='allssa', linear/curved/jump truths, n=150, 24/yr)
 (b) jump magnitude bias/RMSE and detection (2 and 4 sigma steps, NDVI-like design, n=150)
 (c) jump false positives (no jump)
Seeds 14000+. Run: OMP_NUM_THREADS=1 python audit/post_season_sig.py <R>"""
import sys, json, warnings
import numpy as np
import tspec
warnings.simplefilter("ignore")
R = int(sys.argv[1]) if len(sys.argv) > 1 else 200
out = {}
for sig in ("per_frequency", "sidak"):
    rng = np.random.default_rng(14000)
    cov = {"A": [[], []], "B": [[], []]}
    for r in range(R):
        t = np.sort(rng.uniform(2000, 2006.25, 150))
        S = np.cos(2 * np.pi * t) + 0.4 * np.sin(2 * np.pi * t)
        for truth in ("A", "B"):
            T = 2 + 0.15 * (t - 2000) if truth == "A" else 2 + 0.6 * np.sin(2 * np.pi * (t - 2000) / 9)
            y = T + S + rng.normal(0, 0.3, t.size)
            d = tspec.just_decompose(t, y, season="allssa", season_significance=sig)
            cov[truth][0].append(np.mean((T >= d.trend_ci_low) & (T <= d.trend_ci_high)))
            cov[truth][1].append(np.mean((S >= d.seasonal_ci_low) & (S <= d.seasonal_ci_high)))
    res = {f"trend_cov_{k}": float(np.mean(v[0])) for k, v in cov.items()}
    res.update({f"seasonal_cov_{k}": float(np.mean(v[1])) for k, v in cov.items()})
    rng = np.random.default_rng(14500)
    fp = 0; mags = {2.0: [], 4.0: []}; det = {2.0: 0, 4.0: 0}
    for r in range(R):
        t = np.sort(rng.uniform(2000, 2006.25, 150)); tj = 2003.125
        base = 0.5 + 0.004 * (t - 2000) + 0.1 * np.cos(2 * np.pi * t) + 0.03 * np.sin(4 * np.pi * t)
        e = rng.normal(0, 0.02, t.size)
        fp += len(tspec.just_jumps(t, base + e, season="allssa", season_significance=sig).jumps) > 0
        for k in (2.0, 4.0):
            rj = tspec.just_jumps(t, base + e + k * 0.02 * (t >= tj), season="allssa", season_significance=sig)
            near = [j for j in rj.jumps if abs(j.time_year - tj) < 0.5]
            if near:
                det[k] += 1; mags[k].append(near[0].magnitude - k * 0.02)
    res["jump_FP"] = fp / R
    for k in (2.0, 4.0):
        m = np.array(mags[k])
        res[f"jump{k:g}sig_power"] = det[k] / R
        res[f"jump{k:g}sig_mag_bias"] = float(m.mean()) if m.size else None
        res[f"jump{k:g}sig_mag_rmse"] = float(np.sqrt(np.mean(m**2))) if m.size else None
    out[sig] = res
    print(sig, json.dumps(res), flush=True)
json.dump(out, open("audit/post_fix/season_significance_results.json", "w"), indent=1)
