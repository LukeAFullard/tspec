"""P9: analyze_series feature-composition fuzz (crash + coherence + truth recovery)."""
import itertools, json, time, warnings
import numpy as np, tspec
warnings.filterwarnings("ignore")
rng = np.random.default_rng(3)
n = 220
t = np.sort(rng.uniform(2015, 2025, n)); t = t[~((t > 2016.9) & (t < 2017.25))]
tr = np.where(t < 2018.5, 4.0, 6.4) + np.where(t > 2021.8, -1.4*(t - 2021.8), 0.0)
y = tr + 0.85*np.cos(2*np.pi*t) + 0.3*np.sin(4*np.pi*t) + rng.normal(0, 0.16, t.size)
y_sp = y.copy(); y_sp[rng.choice(t.size, 8, replace=False)] += 5.0
dl = np.quantile(y, 0.2); status = np.where(y < dl, -1, 0); y_c = np.where(status == -1, dl, y)
out = {}
grid = itertools.product(["offset", "segment"], ["count", "duration"], ["none", "mm"], ["ols", "sen"], [False, True], ["bootstrap", "conditional"])
for adj, wm, rob, est, cens, band in grid:
    key = f"{adj}|{wm}|{rob}|{est}|cens={cens}|{band}"
    kw = dict(jump_adjustment=adj, robust=rob, slope_estimator=est, band_method=band, n_boot=99)
    if wm == "duration":
        kw["just_kwargs"] = dict(window_mode="duration", window_duration=3.0, step_duration=1.0)
        kw["decompose_kwargs"] = dict(window_mode="duration", window_duration=3.0, step_duration=1.0)
    yy = y_sp if rob == "mm" else y
    if cens:
        yy = np.where(status == -1, dl, yy); kw["censor_status"] = status; kw["censoring"] = tspec.CensoredConfig(random_state=1)
    t1 = time.time()
    try:
        r = tspec.analyze_series(t, yy, **kw)
        jt = [round(float(j.time_year), 2) for j in r.jumps.jumps]
        tp = [round(float(p.time_year), 2) for p in r.turning_points.turning_points]
        ct = np.asarray(r.combined_trend)
        cov = float(np.mean((r.trend_ci_low <= tr) & (tr <= r.trend_ci_high))) if r.trend_ci_low is not None else None
        out[key] = dict(jumps=jt, tps=tp, trend_rmse=round(float(np.sqrt(np.mean((ct - tr)**2))), 3), band_cov=cov,
                        estimators=sorted({s.estimator for s in r.segments}), nseg=len(r.segments), sec=round(time.time()-t1, 1),
                        warn=sorted({w.code for w in r.diagnostics.warnings}))
    except Exception as ex:
        out[key] = f"RAISED {type(ex).__name__}: {str(ex)[:150]}"
    print(key, out[key], flush=True)
json.dump(out, open("p9_results.json", "w"), indent=1, default=str)
