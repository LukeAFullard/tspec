"""MC5 (family-wise false-alarm rate over the monitoring horizon) and MC6 (out-of-sample
prediction-interval coverage) for just_monitor on stable series.
Stable truth: 2 + 0.1 t + 0.5 cos(2 pi t) + 0.2 sin(4 pi t) + N(0, 0.15^2); irregular ~24 obs/yr.
History n_his = 96 (4 yr); monitoring D in {24, 96}. Seed base 5000.
Run: OMP_NUM_THREADS=1 python audit/mc5_6_monitor.py <season> <alpha> <R>
"""
import sys, json, warnings
import numpy as np
import tspec
warnings.simplefilter("ignore")
season = sys.argv[1] if len(sys.argv) > 1 else "ols"
alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
R = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
controls = {
    "sidak+pocock(default)": dict(freq_correction="sidak", sequential_control="pocock"),
    "sidak+obrien_fleming": dict(freq_correction="sidak", sequential_control="obrien_fleming"),
    "sidak+consecutive_k2": dict(freq_correction="sidak", sequential_control="consecutive_k", consecutive_k=2),
    "sidak+none": dict(freq_correction="sidak", sequential_control="none"),
    "parity_mode": dict(parity_mode=True),
}
out = {}
for n_his, D in ((96, 24), (96, 96)):
    rng = np.random.default_rng(5000 + D + (1 if season == "allssa" else 0) + int(alpha * 1000))
    alarms = {k: 0 for k in controls}; errs = {k: 0 for k in controls}
    pi_hits = 0; pi_tot = 0; ci_hits = 0; pi_hits_by_lead = np.zeros(D); pi_n_by_lead = np.zeros(D)
    for r in range(R):
        n = n_his + D
        t = np.sort(rng.uniform(2000, 2000 + n / 24.0, n))
        mu = 2 + 0.1 * (t - 2000) + 0.5 * np.cos(2 * np.pi * t) + 0.2 * np.sin(4 * np.pi * t)
        y = mu + rng.normal(0, 0.15, n)
        for k, kw in controls.items():
            try:
                res = tspec.just_monitor(t, y, start=n_his, season=season, alpha=alpha, **kw)
                alarms[k] += res.detected
            except Exception as e:
                errs[k] += 1
            if k == "sidak+pocock(default)" and res.forecast_pi_low is not None:
                lo = res.forecast_pi_low[n_his:]; hi = res.forecast_pi_high[n_his:]
                yo = y[n_his:]
                hit = (yo >= lo) & (yo <= hi)
                pi_hits += hit.sum(); pi_tot += hit.size
                pi_hits_by_lead += hit; pi_n_by_lead += 1
                ci_hits += ((mu[n_his:] >= res.forecast_ci_low[n_his:]) & (mu[n_his:] <= res.forecast_ci_high[n_his:])).sum()
    key = f"{season},alpha={alpha},n_his={n_his},D={D}"
    out[key] = {"R": R, "MCSE_at_alpha": float(np.sqrt(alpha * (1 - alpha) / R)),
                **{f"FWER[{k}]": alarms[k] / (R - errs[k]) for k in controls},
                "errors": errs,
                "PI95_coverage": pi_hits / pi_tot,
                "PI95_cov_first_quarter": float(pi_hits_by_lead[: D // 4].sum() / pi_n_by_lead[: D // 4].sum()),
                "PI95_cov_last_quarter": float(pi_hits_by_lead[-D // 4:].sum() / pi_n_by_lead[-D // 4:].sum()),
                "meanCI95_coverage_of_mu": ci_hits / pi_tot}
    print(key, json.dumps(out[key]), flush=True)
json.dump(out, open(f"audit/mc5_6_results_{season}_{alpha}.json", "w"), indent=1)
