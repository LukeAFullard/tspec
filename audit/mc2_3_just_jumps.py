"""MC2 (false positives, no jump) and MC3 (power / location) for just_jumps.
One just_jumps call per replicate with mag_th=dir_th=0 and jump_alpha=None returns every merged
candidate with its raw jump/step p-values; documented filters are applied post hoc (exactly
equivalent because tspec filters after merging):
  A  defaults (mag_th=0.05, dir_th=0.01, no p filter)  [NDVI-like scale, sigma=0.02]
  B  jump_alpha=0.05                                    [just_jumps recommended screening]
  C  jump_alpha=0.05 + bonferroni_window=True
  D  C + step_alpha=0.05 (= analyze_series Stage-1 defaults)
Seed base 2000. Run: OMP_NUM_THREADS=1 python audit/mc2_3_just_jumps.py <season> <R>
"""
import sys, json, warnings
import numpy as np
import tspec
warnings.simplefilter("ignore")
season = sys.argv[1] if len(sys.argv) > 1 else "ols"
R = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
SIG = 0.02

def make(rng, n, rate, gap, jump):
    span = n / rate
    t = np.sort(rng.uniform(2000, 2000 + span, int(n * (1.33 if gap else 1.0))))
    if gap:
        t = t[~((t > 2000 + 0.3 * span) & (t < 2000 + 0.55 * span))]
    tj = 2000 + 0.7 * span if gap else 2000 + 0.5 * span
    y = 0.5 + 0.004 * (t - 2000) + 0.1 * np.cos(2 * np.pi * t) + 0.03 * np.sin(4 * np.pi * t)
    y = y + jump * SIG * (t >= tj) + rng.normal(0, SIG, t.size)
    return t, y, tj

def evaluate(res, tj, tol):
    js = res.jumps
    out = {}
    if res.all_window_candidates:
        c0 = res.all_window_candidates[0]
        wsize = c0.window_stop - c0.window_start
    mult = max(1, wsize - 5)
    pj = np.array([j.jump_pvalue for j in js]); ps = np.array([j.step_pvalue for j in js])
    mag = np.array([j.magnitude for j in js]); di = np.array([j.direction for j in js])
    tt = np.array([j.time_year for j in js])
    masks = {
        "A": (np.abs(mag) >= 0.05) | (np.abs(di) >= 0.01),
        "B": pj < 0.05,
        "C": np.minimum(1, pj * mult) < 0.05,
        "D": (np.minimum(1, pj * mult) < 0.05) & (np.minimum(1, ps * mult) < 0.05),
    }
    for k, m in masks.items():
        if tj is None:
            out[k] = (bool(np.any(m)), None)
        else:
            near = m & (np.abs(tt - tj) <= tol)
            loc = float(tt[near][np.argmin(np.abs(tt[near] - tj))] - tj) if np.any(near) else None
            out[k] = (bool(np.any(near)), loc, int(np.sum(m & ~(np.abs(tt - tj) <= tol))))
    return out

results = {}
configs = [(50, 8, False), (150, 24, False), (150, 24, True), (500, 24, False)]
if season == "allssa":
    configs = [(150, 24, False), (150, 24, True)]
for (n, rate, gap) in configs:
    rng = np.random.default_rng(2000 + n + 100 * gap + (7 if season == "allssa" else 0))
    Rn = R if n < 500 else R // 2
    # MC2 null
    fp = {k: 0 for k in "ABCD"}; nerr = 0
    for r in range(Rn):
        t, y, _ = make(rng, n, rate, gap, 0.0)
        try:
            res = tspec.just_jumps(t, y, season=season, mag_th=0.0, dir_th=0.0)
        except Exception as e:
            nerr += 1; continue
        ev = evaluate(res, None, None)
        for k in "ABCD":
            fp[k] += ev[k][0]
    Rv = Rn - nerr
    key = f"{season},n={n},rate={rate},gap={gap}"
    results[key] = {"MC2_null_R": Rv, "errors": nerr,
                    **{f"FP_{k}": fp[k] / Rv for k in "ABCD"},
                    "MCSE_at_0.05": float(np.sqrt(0.05 * 0.95 / Rv))}
    print(key, json.dumps(results[key]), flush=True)
    # MC3 power/location
    Rp = max(200, Rn // 2)
    for jmp in (1.0, 2.0, 4.0):
        det = {k: 0 for k in "ABCD"}; locs = {k: [] for k in "ABCD"}; extra = {k: 0 for k in "ABCD"}
        for r in range(Rp):
            t, y, tj = make(rng, n, rate, gap, jmp)
            res = tspec.just_jumps(t, y, season=season, mag_th=0.0, dir_th=0.0)
            ev = evaluate(res, tj, 0.5)
            for k in "ABCD":
                det[k] += ev[k][0]; extra[k] += ev[k][2] > 0
                if ev[k][1] is not None:
                    locs[k].append(ev[k][1])
        pk = f"{key},jump={jmp}sigma"
        results[pk] = {"R": Rp}
        for k in "ABCD":
            L = np.array(locs[k])
            results[pk][k] = {"power": det[k] / Rp, "extra_false_jumps_rate": extra[k] / Rp,
                              "loc_bias_yr": float(L.mean()) if L.size else None,
                              "loc_rmse_yr": float(np.sqrt(np.mean(L ** 2))) if L.size else None}
        print(pk, json.dumps(results[pk]), flush=True)
json.dump(results, open(f"audit/mc2_3_results_{season}.json", "w"), indent=1)
