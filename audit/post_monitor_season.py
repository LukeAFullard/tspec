"""Monitor forecast mean-CI coverage of the true mean and PI coverage with an ALLSSA
history fit, per_frequency vs sidak harmonic selection. Design as audit/mc5_6_monitor.py
(n_his=96, D=24). Seed 15000. Run: OMP_NUM_THREADS=1 python audit/post_monitor_season.py <R>"""
import sys, json, warnings
import numpy as np
import tspec
warnings.simplefilter("ignore")
R = int(sys.argv[1]) if len(sys.argv) > 1 else 300
out = {}
for sig in ("per_frequency", "sidak"):
    rng = np.random.default_rng(15000)
    ci = []; pi = []; fa = 0
    for r in range(R):
        n = 120; t = np.sort(rng.uniform(2000, 2005, n))
        mu = 2 + 0.1 * (t - 2000) + 0.5 * np.cos(2 * np.pi * t) + 0.2 * np.sin(4 * np.pi * t)
        y = mu + rng.normal(0, 0.15, n)
        m = tspec.just_monitor(t, y, start=96, season="allssa", alpha=0.05, horizon=24, season_significance=sig)
        ci.append(np.mean((mu[96:] >= m.forecast_ci_low[96:]) & (mu[96:] <= m.forecast_ci_high[96:])))
        pi.append(np.mean((y[96:] >= m.forecast_pi_low[96:]) & (y[96:] <= m.forecast_pi_high[96:])))
        fa += m.detected
    out[sig] = {"R": R, "meanCI95_cov": float(np.mean(ci)), "PI95_cov": float(np.mean(pi)), "FWER": fa / R}
    print(sig, json.dumps(out[sig]), flush=True)
json.dump(out, open("audit/post_fix/monitor_season_results.json", "w"), indent=1)
