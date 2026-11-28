"""P12: follow-ups from the STL benchmark: ALLSSA coarse-grid blindness, false jumps near a true step, CO2 gap."""
import json, numpy as np, tspec, warnings
from statsmodels.tsa.seasonal import STL
warnings.filterwarnings("ignore")
out = {}
# (1) ALLSSA default integer coarse grid vs dense grid for a 2.37 cy/yr component (S4a design)
t = 2000 + (np.arange(240) + .5)/12
hit = {"default": 0, "dense": 0}
for s in range(40):
    rng = np.random.default_rng(300 + s)
    y = 3 + 0.05*(t - t[0]) + np.cos(2*np.pi*t) + 0.7*np.cos(2*np.pi*2.37*t + 0.5) + rng.normal(0, .25, t.size)
    for lab, om in [("default", None), ("dense", np.arange(0.5, 5.5, 0.05))]:
        a = tspec.allssa(t, y, omega=om, decimal=2, n_mc=499)
        hit[lab] += np.any(np.abs(a.frequencies - 2.37) < 0.02)
lss = tspec.lssa(t, y, omega=[1, 2, 3, 4, 5], known_freqs=[1.0]).spectrum
out["allssa_2.37_found_rate"] = {k: v/40 for k, v in hit.items()}
out["lss_at_integer_grid_given_annual"] = lss.tolist()
print(out, flush=True)
# (2) false jumps in presence of a real 6.7-sigma step (S2 design): where are they?
pos = []
for s in range(60):
    rng = np.random.default_rng(100 + s)
    y = 5 + 0.05*(t - t[0]) + 2.0*(t >= 2010) + np.cos(2*np.pi*t) + 0.4*np.sin(4*np.pi*t) + rng.normal(0, .3, t.size)
    r = tspec.just_jumps(t, y, season="ols")
    pos += [float(r.times_numeric[j.valid_index] - 2010) for j in r.jumps if abs(r.times_numeric[j.valid_index] - 2010) >= 0.25]
out["false_jumps_with_true_step"] = dict(n_false=len(pos), per_series=len(pos)/60, offsets_yr=sorted(round(p, 2) for p in pos))
# null comparator on identical design without the step
k = 0
for s in range(60):
    rng = np.random.default_rng(100 + s)
    y = 5 + 0.05*(t - t[0]) + np.cos(2*np.pi*t) + 0.4*np.sin(4*np.pi*t) + rng.normal(0, .3, t.size)
    k += len(tspec.just_jumps(t, y, season="ols").jumps)
out["false_jumps_null_same_design_per_series"] = k/60
print(out["false_jumps_with_true_step"], out["false_jumps_null_same_design_per_series"], flush=True)
# (3) CO2 blackout: tspec variants vs STL
a = np.genfromtxt("/home/user/tspec/validation/data/mauna_loa_co2_monthly.csv", delimiter=",", names=True)
tc, yc = a["decimal_date"], a["average"]; m = (tc >= 1959) & (tc < 2025) & (yc > 0); tc, yc = tc[m], yc[m]
rng = np.random.default_rng(11); keep = (rng.random(tc.size) > 0.35) & ~((tc > 1990.0) & (tc < 1990.5))
to, yo = tc[keep], yc[keep]; tg = tc[(tc >= to[0]) & (tc <= to[-1])]; gapg = (tg > 1990) & (tg < 1990.5)
S = STL(yc, period=12, robust=True).fit()
res = {}
for lab, kw in [("allssa_default", dict(season="allssa")), ("ols_1234", dict(season="ols")),
                ("ols_1234_duration4y", dict(season="ols", window_mode="duration", window_duration=4.0, step_duration=1.0))]:
    full = tspec.just_decompose(tc, yc, **kw); g = tspec.just_decompose(to, yo, eval_times=tg, **kw)
    ref_tr = np.interp(tg, tc, full.trend); ref_se = np.interp(tg, tc, full.seasonal)
    res[lab] = dict(trend_gap=float(np.sqrt(np.mean((g.trend_reg - ref_tr)[gapg]**2))), seas_gap=float(np.sqrt(np.mean((g.seasonal_reg - ref_se)[gapg]**2))),
                    vs_stl_seas_complete=float(np.sqrt(np.mean((full.seasonal - S.seasonal)**2))))
out["co2_gap_variants"] = res; print(res)
json.dump(out, open("p12_results.json", "w"), indent=1)
