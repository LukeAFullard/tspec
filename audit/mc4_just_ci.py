"""MC4: Pointwise 95% coverage of just_decompose trend and seasonal CIs.
Truth A: linear trend + annual sinusoid (window model correctly specified).
Truth B: smooth curved trend 0.6*sin(2*pi*(t-2000)/9) + annual sinusoid (local-linear bias).
Truth C: A plus a known jump passed via jump_indices.
Seed base 4000. Run: OMP_NUM_THREADS=1 python audit/mc4_just_ci.py <season> <R>
"""
import sys, json, warnings
import numpy as np
import tspec
warnings.simplefilter("ignore")
season = sys.argv[1] if len(sys.argv) > 1 else "ols"
R = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
SIG = 0.3

def gen(rng, n, rate, gap, truth):
    span = n / rate
    t = np.sort(rng.uniform(2000, 2000 + span, int(n * (1.33 if gap else 1))))
    if gap:
        t = t[~((t > 2000 + 0.4 * span) & (t < 2000 + 0.65 * span))]
    S = 1.0 * np.cos(2 * np.pi * t) + 0.4 * np.sin(2 * np.pi * t)
    if truth == "B":
        T = 2.0 + 0.6 * np.sin(2 * np.pi * (t - 2000) / 9.0)
    else:
        T = 2.0 + 0.15 * (t - 2000)
    jidx = None
    if truth == "C":
        j = int(np.searchsorted(t, 2000 + 0.5 * span)); T = T + 1.0 * (np.arange(t.size) >= j); jidx = [j]
    y = T + S + rng.normal(0, SIG, t.size)
    return t, y, T, S, jidx

out = {}
configs = [(50, 8, False), (150, 24, False), (150, 24, True), (500, 24, False)]
if season == "allssa":
    configs = [(150, 24, False), (500, 24, False)]
for (n, rate, gap) in configs:
    for truth in ("A", "B", "C"):
        rng = np.random.default_rng(4000 + n + 10 * gap + {"A": 1, "B": 2, "C": 3}[truth] + (50 if season == "allssa" else 0))
        Rn = R if n < 500 else R // 2
        cov_t = []; cov_s = []; cov_t_mid = []; w_t = []; recon = 0.0; nan_frac = []
        for r in range(Rn):
            t, y, T, S, jidx = gen(rng, n, rate, gap, truth)
            d = tspec.just_decompose(t, y, season=season, jump_indices=jidx)
            lo, hi = d.trend_ci_low, d.trend_ci_high
            ok = np.isfinite(lo) & np.isfinite(hi)
            nan_frac.append(1 - ok.mean())
            ct = (T >= lo) & (T <= hi)
            cs = (S >= d.seasonal_ci_low) & (S <= d.seasonal_ci_high)
            cov_t.append(ct[ok].mean()); cov_s.append(cs[ok].mean())
            m = ok & (t > t[0] + 0.25 * (t[-1] - t[0])) & (t < t[0] + 0.75 * (t[-1] - t[0]))
            cov_t_mid.append(ct[m].mean())
            w_t.append(np.nanmedian(hi - lo))
            recon = max(recon, float(np.nanmax(np.abs(d.trend + d.seasonal + d.remainder - y))))
        key = f"{season},n={n},rate={rate},gap={gap},truth={truth}"
        cov_t = np.array(cov_t); cov_s = np.array(cov_s)
        out[key] = {"R": Rn,
                    "trend_cov": float(cov_t.mean()), "trend_cov_MCSE": float(cov_t.std(ddof=1) / np.sqrt(Rn)),
                    "trend_cov_middle50%": float(np.mean(cov_t_mid)),
                    "seasonal_cov": float(cov_s.mean()), "seasonal_cov_MCSE": float(cov_s.std(ddof=1) / np.sqrt(Rn)),
                    "median_trend_CI_width": float(np.median(w_t)),
                    "max_reconstruction_err": recon, "mean_nan_frac": float(np.mean(nan_frac))}
        print(key, json.dumps(out[key]), flush=True)
json.dump(out, open(f"audit/mc4_results_{season}.json", "w"), indent=1)
