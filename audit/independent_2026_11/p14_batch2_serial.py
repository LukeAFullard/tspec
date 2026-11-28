"""P14: batch-2 validation of AUD-02 (serial-correlation calibration) vs serial_correlation='none'."""
import json, os, sys, time, warnings
import numpy as np, tspec
from common import ar1, pmap, wilson
warnings.filterwarnings("ignore")
NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 300
ONLY = os.environ.get("P14_ONLY")

def ou(rng, t, tau):
    a = np.exp(-np.diff(t) / tau); x = np.empty(t.size); x[0] = rng.normal()
    for i in range(1, t.size): x[i] = a[i-1]*x[i-1] + np.sqrt(1 - a[i-1]**2)*rng.normal()
    return x

def noise(rng, t, kind, rho):
    if kind == "ar1": return ar1(rng, t.size, rho)
    if kind == "ou": return ou(rng, t, rho)          # rho = tau (years)
    if kind == "t3": return rng.standard_t(3, t.size) / np.sqrt(3)
    return rng.normal(size=t.size)

def j_job(a):
    seed, kind, rho, jump, kw = a
    rng = np.random.default_rng(seed); n = kw.pop("n", 240); span = n / 12
    t = np.sort(rng.uniform(0, span, n))
    y = 0.05*t + np.cos(2*np.pi*t) + 0.4*np.sin(4*np.pi*t) + 0.5*noise(rng, t, kind, rho) + jump*(t >= span/2)
    r = tspec.just_jumps(t, y, **kw)
    hit = any(abs(r.times_numeric[j.valid_index] - span/2) < 0.5 for j in r.jumps)
    return dict(any=len(r.jumps) > 0, hit=hit, rho=r.serial_rho)

def s_job(a):
    seed, kind, rho, bend, kw = a
    rng = np.random.default_rng(seed); t = np.sort(rng.uniform(0, 20, 200))
    y = 0.5*t - bend*np.maximum(t - 10, 0) + 0.5*noise(rng, t, kind, rho)
    r = tspec.stpd(t, y, band_method="conditional", **kw)
    return dict(any=len(r.turning_points) > 0, hit=any(abs(p.time_year - 10) < 1 for p in r.turning_points))

def p_job(a):
    seed, rho, n, kw = a
    rng = np.random.default_rng(seed); t = 2000 + np.arange(n)/12
    y = 0.03*(t - 2000) + 0.5*ar1(rng, n, rho)
    tr = 0.03*(t - 2000)
    r = tspec.analyze_series(t, y, detect_jumps=False, detect_turning_points=False, remove_seasonality=False,
                             band_method="conditional", **kw)
    s = r.segments[0]
    yb = y - 0.03*(t - 2000)  # trend-free version for type I
    r0 = tspec.analyze_series(t, yb, detect_jumps=False, detect_turning_points=False, remove_seasonality=False,
                              band_method="conditional", **kw)
    s0 = r0.segments[0]
    bc = float(np.mean((r.trend_ci_low <= tr) & (tr <= r.trend_ci_high))) if r.trend_ci_low is not None else np.nan
    return dict(rej=s0.pvalue < 0.05, cov=s.slope_ci_low <= 0.03 <= s.slope_ci_high, band=bc)

out = {"NREP": NREP}
def run(name, fn, args):
    if ONLY and not any(o in name for o in ONLY.split(",")): return
    t1 = time.time(); res = pmap(fn, args)
    d = {k: float(np.nanmean([r[k] for r in res])) for k in res[0]}
    for k in ("any", "rej"):
        if k in d: d[k + "_ci"] = wilson(int(round(d[k]*len(res))), len(res))
    d["sec"] = round(time.time() - t1, 1); out[name] = d; print(name, d, flush=True)

for sc in ["ar1", "none"]:
    for kind, rho in [("white", 0), ("ar1", 0.3), ("ar1", 0.6), ("ar1", 0.8)]:
        run(f"just_null_{kind}{rho}_ols|{sc}", j_job, [(s, kind, rho, 0.0, dict(season="ols", serial_correlation=sc)) for s in range(NREP)])
    run(f"just_null_ar1_0.6_allssa|{sc}", j_job, [(s, "ar1", 0.6, 0.0, dict(serial_correlation=sc)) for s in range(NREP // 2)])
    run(f"just_null_ou0.15_ols_mode_ou|{sc}", j_job, [(s, "ou", 0.15, 0.0, dict(season="ols", serial_correlation="ou" if sc == "ar1" else sc)) for s in range(NREP)])
    run(f"just_null_t3_mm|{sc}", j_job, [(s, "t3", 0, 0.0, dict(season="ols", robust="mm", serial_correlation=sc)) for s in range(NREP // 2)])
    run(f"just_null_white_long720|{sc}", j_job, [(s, "white", 0, 0.0, dict(season="ols", n=720, serial_correlation=sc)) for s in range(NREP // 2)])
    for jump in [1.0, 1.5, 2.0]:
        run(f"just_power_white_jump{jump}|{sc}", j_job, [(s, "white", 0, jump, dict(season="ols", serial_correlation=sc)) for s in range(NREP // 2)])
    run(f"just_power_ar0.5_jump2.0|{sc}", j_job, [(s, "ar1", 0.5, 2.0, dict(season="ols", serial_correlation=sc)) for s in range(NREP // 2)])
    for kind, rho in [("white", 0), ("ar1", 0.5), ("ar1", 0.8)]:
        run(f"stpd_null_{kind}{rho}|{sc}", s_job, [(s, kind, rho, 0.0, dict(serial_correlation=sc)) for s in range(NREP)])
    for kind, rho in [("white", 0), ("ar1", 0.5)]:
        run(f"stpd_bend1.0_{kind}{rho}|{sc}", s_job, [(s, kind, rho, 1.0, dict(serial_correlation=sc)) for s in range(NREP // 2)])
    for rho in [0.0, 0.5, 0.8]:
        for n in [60, 240]:
            run(f"pipe_ols_rho{rho}_n{n}|{sc}", p_job, [(s, rho, n, dict(serial_correlation=sc)) for s in range(NREP)])
json.dump(out, open(os.environ.get("P14_OUT", "p14_results.json"), "w"), indent=1)
