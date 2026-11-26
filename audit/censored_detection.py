"""Item 4 study: detection calibration on censored series (single vs. multiple imputation).
Truth (n = 150, ~24 obs/yr, decimal-year time): y = 2 + 0.1 (t - 2000) [+ 0.3 cos 2 pi t]
+ N(0, 0.4^2), optionally a +1.0 (2.5 sigma) step at mid-record (jumps) or a slope change
+0.5 -> -0.5 /yr at mid-record (turning points). Left-censoring at a fixed DL chosen as the
population quantile giving ~20% / ~50% censoring; censored entries are passed as '<DL'.
Reports, for just_jumps(season=...) and stpd(): null false-detection rate, detection rate within
0.5 yr of the truth, and decision instability across two imputation seeds.
Seed base 18000. Run: OMP_NUM_THREADS=1 python audit/censored_detection.py <method> <seasonal 0/1> <R> [extra cfg kwargs as k=v]
"""
import json
import sys
import warnings

import numpy as np

import tspec

warnings.simplefilter("ignore")
method = sys.argv[1]
seasonal = bool(int(sys.argv[2]))
R = int(sys.argv[3])
extra = {}
nullimp = "nullimp" in sys.argv[4:]
if nullimp:
    # Study variant: impute each candidate window from its null model (drop the tested
    # jump / hinge columns from the congenial design).
    import tspec.just as _J
    import tspec.stpd as _S
    _wci = _J.window_congenial_imputations
    _J.window_congenial_imputations = lambda v, c, lft, rgt, X, *a, **k: _wci(v, c, lft, rgt, X[:, :-2], *a, **k)
    _S.window_congenial_imputations = lambda v, c, lft, rgt, X, *a, **k: _wci(v, c, lft, rgt, X[:, :-1], *a, **k)
for kv in [x for x in sys.argv[4:] if x != "nullimp"]:
    k, v = kv.split("=")
    extra[k] = int(v) if v.isdigit() else (tuple(float(x) for x in v.split(",")) if "," in v else v)
n = 150
span = n / 24.0
tmid = 2000 + span / 2
season = "ols" if seasonal else "none"


def truth(t, alt):
    mu = 2 + 0.1 * (t - 2000) + (0.3 * np.cos(2 * np.pi * t) if seasonal else 0.0)
    if alt and method == "just":
        mu = mu + 1.0 * (t >= tmid)
    if alt and method == "stpd":
        mu = 2 + 0.5 * (t - 2000) - 1.0 * np.maximum(0, t - tmid) + (0.3 * np.cos(2 * np.pi * t) if seasonal else 0.0)
    return mu


def run(t, y, seed):
    cfg = tspec.CensoredConfig(random_state=seed, allow_heavy_censoring=True, **extra)
    if method == "just":
        r = tspec.just_jumps(t, y, season=season, censoring=cfg)
        return [float(j.time_year) for j in r.jumps]
    r = tspec.stpd(t, y, censoring=cfg) if not seasonal else None
    if r is None:
        raise SystemExit("stpd study uses seasonal=0")
    return [float(x) for x in r.tp_times]


out = {}
pop = np.random.default_rng(424242)
tp = np.linspace(2000, 2000 + span, 200000)
for cfrac in (0.0, 0.2, 0.5):
    rng = np.random.default_rng(18000 + int(100 * cfrac) + (1 if seasonal else 0) + (7 if method == "stpd" else 0))
    acc = {"null_fp": 0, "det": 0, "flip_null": 0, "flip_alt": 0, "cfrac": []}
    for _ in range(R):
        t = np.sort(rng.uniform(2000, 2000 + span, n))
        e = rng.normal(0, 0.4, n)
        for alt in (False, True):
            y = truth(t, alt) + e
            if cfrac > 0:
                dl = float(np.quantile(truth(tp, alt) + pop.normal(0, 0.4, tp.size), cfrac))
                cen = y < dl
                acc["cfrac"].append(cen.mean())
                yy = np.array([f"<{dl:.6g}" if c else float(v) for c, v in zip(cen, y)], dtype=object)
            else:
                yy = y
            ev0 = run(t, yy, 0)
            if not alt:
                acc["null_fp"] += len(ev0) > 0
            else:
                acc["det"] += any(abs(x - tmid) < 0.5 for x in ev0)
            if cfrac > 0:
                ev1 = run(t, yy, 1)
                flip = (len(ev0) > 0) != (len(ev1) > 0) if not alt else (
                    any(abs(x - tmid) < 0.5 for x in ev0) != any(abs(x - tmid) < 0.5 for x in ev1))
                acc["flip_alt" if alt else "flip_null"] += flip
    key = f"{method},seasonal={int(seasonal)},cfrac={cfrac}"
    out[key] = {"R": R, "realized_cfrac": float(np.mean(acc["cfrac"])) if acc["cfrac"] else 0.0,
                "null_FP": acc["null_fp"] / R, "detect": acc["det"] / R,
                "decision_flip_null": acc["flip_null"] / R, "decision_flip_alt": acc["flip_alt"] / R}
    print(key, json.dumps(out[key]), flush=True)
tag = "_".join(f"{k}{'-'.join(map(str, v)) if isinstance(v, tuple) else v}" for k, v in extra.items())
tag = tag + ("_nullimp" if nullimp else "")
json.dump(out, open(f"audit/post_fix/censored_detection_{method}_{int(seasonal)}{'_' + tag if tag else ''}.json", "w"), indent=1)
