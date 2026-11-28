"""P11: README hero example (analyze_series defaults) over 100 seeds - event accuracy & band coverage."""
import json, numpy as np, tspec, warnings
from common import pmap, wilson
warnings.filterwarnings("ignore")
def job(a):
    seed, mode = a
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(2015.0, 2025.0, size=220)); t = t[~((t > 2016.9) & (t < 2017.25))]
    tr = np.where(t < 2018.5, 4.0, 6.4) + np.where(t > 2021.8, -1.4*(t - 2021.8), 0.0)
    y = tr + 0.85*np.cos(2*np.pi*t) + 0.30*np.sin(4*np.pi*t) + rng.normal(0.0, 0.16, size=t.size)
    kw = {} if mode == "count" else dict(just_kwargs=dict(window_mode="duration", window_duration=3.0, step_duration=1.0),
                                         decompose_kwargs=dict(window_mode="duration", window_duration=3.0, step_duration=1.0))
    r = tspec.analyze_series(t, y, **kw)
    J = [e.time_year for e in r.jumps.jumps]; P = [p.time_year for p in r.turning_points.turning_points]
    jok = any(abs(x - 2018.5) < 0.25 for x in J); pok = any(abs(x - 2021.8) < 0.5 for x in P)
    extra = sum(abs(x - 2018.5) >= 0.25 for x in J) + sum(abs(x - 2021.8) >= 0.5 for x in P)
    gap_tp = any(2016.6 < x < 2017.8 for x in P)
    cov = float(np.mean((r.trend_ci_low <= tr) & (tr <= r.trend_ci_high)))
    exact = jok and pok and extra == 0
    return dict(jok=jok, pok=pok, extra=extra, gap_tp=gap_tp, cov=cov, exact=exact)
out = {}
for mode in ["count", "duration"]:
    res = pmap(job, [(s, mode) for s in range(100)])
    d = {k: float(np.mean([r[k] for r in res])) for k in res[0]}
    d["cov_when_exact"] = float(np.mean([r["cov"] for r in res if r["exact"]])) if any(r["exact"] for r in res) else None
    out[mode] = d; print(mode, d, flush=True)
json.dump(out, open("p11_results.json", "w"), indent=1)
