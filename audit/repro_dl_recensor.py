"""recensor_check false-artifact rate: REAL downward jump (-0.35 log units) coinciding with a DL drop
0.95 -> 0.60, so censored values exist at both limits. Seed 64. Run: python audit/repro_dl_recensor.py
"""
import warnings
import numpy as np
import tspec
from tspec.censored import CensoredConfig
warnings.simplefilter("ignore")
rng = np.random.default_rng(64)
R = 150; stats = {"dlshift_detected": 0, "jump_found": 0, "coincide|found": 0, "artifact|found": 0}
for r in range(R):
    n = 192; t = np.sort(rng.uniform(2000, 2008, n)); j = n // 2
    ly = 0.15 * np.cos(2 * np.pi * t) + rng.normal(0, 0.15, n) - 0.35 * (np.arange(n) >= j)
    y = np.exp(ly)
    dl = np.where(np.arange(n) < j, 0.95, 0.60)
    cen = y < dl
    ys = np.array([f"<{d:.3g}" if c else float(v) for c, v, d in zip(cen, y, dl)], dtype=object)
    try:
        res = tspec.just_jumps(t, ys, season="ols", jump_alpha=0.05, bonferroni_window=True, step_alpha=0.05,
                               mag_th=0.0, dir_th=0.0, censoring=CensoredConfig(recensor_check=True, random_state=r))
    except Exception as e:
        print("err", e); continue
    stats["dlshift_detected"] += len(res.censoring.dl_shifts) > 0
    near = [q for q in res.jumps if abs(q.time_year - t[j]) < 0.5]
    if near:
        stats["jump_found"] += 1; stats["coincide|found"] += near[0].coincides_with_dl_shift
        stats["artifact|found"] += bool(near[0].dl_artifact_likely)
f = stats["jump_found"]
print({"R": R, "dlshift_detected": stats["dlshift_detected"] / R, "jump_found": f / R,
       "coincide|found": stats["coincide|found"] / max(f, 1), "REAL jump flagged dl_artifact_likely|found": stats["artifact|found"] / max(f, 1)})
