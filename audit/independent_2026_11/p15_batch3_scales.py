"""P15: batch-3 validation (AUD-04 multi-scale power, AUD-17 ghosts, AUD-18 curvature) for
analyze_series (new defaults) and standalone just_jumps."""
import json, os, sys, time, warnings
import numpy as np, tspec
from common import ar1, pmap, wilson
warnings.filterwarnings("ignore")
NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 200

def series(seed, kind, jump=0.0, bend=0.0, n=240, span=20.0):
    rng = np.random.default_rng(seed); t = np.sort(rng.uniform(0, span, n))
    tr = {"line": 0.05*t, "curved": 1.5*np.sin(2*np.pi*t/9.0), "curved_strong": 3.0*np.sin(2*np.pi*t/9.0),
          "ar0.5": 0.05*t, "ar0.8": 0.05*t}[kind]
    e = 0.5*ar1(rng, n, 0.5 if kind == "ar0.5" else 0.8) if kind.startswith("ar") else rng.normal(0, 0.5, n)
    tr = tr + jump*(t >= 10.3) - bend*np.maximum(t - 10.3, 0)
    y = tr + np.cos(2*np.pi*t) + 0.4*np.sin(4*np.pi*t) + e
    return t, y, tr

def pipe_job(a):
    seed, kind, jump, bend, kw = a
    t, y, tr = series(seed, kind, jump, bend)
    t1 = time.time()
    r = tspec.analyze_series(t, y, band_method="conditional", **kw)
    J = [e.time_year for e in r.jumps.jumps]; P = [p.time_year for p in r.turning_points.turning_points]
    return dict(any_jump=len(J) > 0, any_tp=len(P) > 0, jump_hit=any(abs(x - 10.3) < 0.5 for x in J),
                tp_hit=any(abs(x - 10.3) < 1.0 for x in P), extra_jumps=sum(abs(x - 10.3) >= 0.5 for x in J), sec=time.time() - t1)

def jj_job(a):
    seed, kind, jump, kw = a
    t, y, tr = series(seed, kind, jump)
    r = tspec.just_jumps(t, y, season="ols", **kw)
    J = [r.times_numeric[j.valid_index] for j in r.jumps]
    return dict(any_jump=len(J) > 0, jump_hit=any(abs(x - 10.3) < 0.5 for x in J),
                extra_jumps=sum(abs(x - 10.3) >= 0.5 for x in J))

out = {"NREP": NREP}
def run(name, fn, args):
    t1 = time.time(); res = pmap(fn, args)
    d = {k: float(np.mean([r[k] for r in res])) for k in res[0]}
    d["wall"] = round(time.time() - t1, 1); out[name] = d; print(name, {k: round(v, 3) for k, v in d.items()}, flush=True)

OLD = dict(just_kwargs=dict(window_scales=(1.0,)))  # batch-2 pipeline (single scale)
for lab, kw in [("new", {}), ("old_single_scale", OLD)]:
    for kind in ["line", "curved", "curved_strong", "ar0.5", "ar0.8"]:
        run(f"pipe_null_{kind}|{lab}", pipe_job, [(s, kind, 0.0, 0.0, kw) for s in range(NREP)])
    for jump in [0.5, 1.0, 2.0]:
        run(f"pipe_step{jump}_line|{lab}", pipe_job, [(s, "line", jump, 0.0, kw) for s in range(NREP)])
    run(f"pipe_step1.0_curved|{lab}", pipe_job, [(s, "curved", 1.0, 0.0, kw) for s in range(NREP)])
    run(f"pipe_bend0.6_line|{lab}", pipe_job, [(s, "line", 0.0, 0.6, kw) for s in range(NREP)])
for lab, kw in [("default", {}), ("step_ms", dict(step_alpha=0.05, window_scales=(1, 2, 4)))]:
    for kind in ["line", "curved"]:
        run(f"jj_null_{kind}|{lab}", jj_job, [(s, kind, 0.0, kw) for s in range(NREP)])
    run(f"jj_ghosts_step3.35|{lab}", jj_job, [(s, "line", 3.35, kw) for s in range(NREP)])
json.dump(out, open("p15_results.json", "w"), indent=1)
