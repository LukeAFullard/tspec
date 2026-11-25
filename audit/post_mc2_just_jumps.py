"""Post-fix MC2/MC3: false-jump probability and power of just_jumps() DEFAULTS (jump_alpha=0.05,
Monte Carlo split calibration + Holm), and of the analyze_series stage-1 filter (+ step_alpha=0.05).
Same data-generating process as audit/mc2_3_just_jumps.py. Seed base 2000.
Run: OMP_NUM_THREADS=1 python audit/post_mc2_just_jumps.py <season> <R>"""
import sys, json, warnings
import numpy as np
import tspec
warnings.simplefilter("ignore")
season = sys.argv[1]; R = int(sys.argv[2]); SIG = 0.02
def make(rng, n, rate, gap, jump):
    span = n / rate
    t = np.sort(rng.uniform(2000, 2000 + span, int(n * (1.33 if gap else 1.0))))
    if gap:
        t = t[~((t > 2000 + 0.3 * span) & (t < 2000 + 0.55 * span))]
    tj = 2000 + 0.7 * span if gap else 2000 + 0.5 * span
    y = 0.5 + 0.004 * (t - 2000) + 0.1 * np.cos(2 * np.pi * t) + 0.03 * np.sin(4 * np.pi * t)
    return t, y + jump * SIG * (t >= tj) + rng.normal(0, SIG, t.size), tj
configs = [(50, 8, False), (150, 24, False), (150, 24, True), (500, 24, False)] if season == "ols" else [(150, 24, False)]
out = {}
for (n, rate, gap) in configs:
    rng = np.random.default_rng(2000 + n + 100 * gap + (7 if season == "allssa" else 0))
    Rn = R if n < 500 else R // 2
    fp = fp_step = 0
    for r in range(Rn):
        t, y, _ = make(rng, n, rate, gap, 0.0)
        res = tspec.just_jumps(t, y, season=season)
        fp += len(res.jumps) > 0; fp_step += any(j.step_pvalue < 0.05 for j in res.jumps)
    key = f"{season},n={n},rate={rate},gap={gap}"
    out[key] = {"R": Rn, "FP_default": fp / Rn, "FP_analyze_filter": fp_step / Rn, "MCSE": float(np.sqrt(0.0475 / Rn))}
    print(key, json.dumps(out[key]), flush=True)
    Rp = max(150, Rn // 3)
    for jmp in (1.0, 2.0, 4.0):
        det = det_s = 0; locs = []
        for r in range(Rp):
            t, y, tj = make(rng, n, rate, gap, jmp)
            res = tspec.just_jumps(t, y, season=season)
            near = [j for j in res.jumps if abs(j.time_year - tj) <= 0.5]
            det += bool(near); det_s += any(j.step_pvalue < 0.05 for j in near)
            if near: locs.append(near[0].time_year - tj)
        L = np.array(locs)
        out[f"{key},jump={jmp}"] = {"R": Rp, "power_default": det / Rp, "power_analyze_filter": det_s / Rp,
                                    "loc_bias": float(L.mean()) if L.size else None,
                                    "loc_rmse": float(np.sqrt(np.mean(L**2))) if L.size else None}
        print(f"{key},jump={jmp}", json.dumps(out[f"{key},jump={jmp}"]), flush=True)
json.dump(out, open(f"audit/post_mc2_results_{season}.json", "w"), indent=1)
