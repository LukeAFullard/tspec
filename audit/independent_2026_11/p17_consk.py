"""P17: AUD-11 consecutive_k false-alarm rate after the conservative calibration."""
import json, sys, numpy as np, tspec, warnings
from common import pmap, wilson
warnings.filterwarnings("ignore")
def series(seed, mon_steps, irregular=False):
    rng = np.random.default_rng(seed); n = 120 + mon_steps
    t = 2015 + np.arange(n)/24
    if irregular: t = t + rng.uniform(-0.3, 0.3, n)/24
    y = 0.55 + 0.18*np.cos(2*np.pi*t) + 0.05*np.sin(4*np.pi*t) + rng.normal(0, .03, n)
    return t, y
def job(a):
    seed, H, irr = a
    t, y = series(seed, H, irr)
    return tspec.just_monitor(t, y, start=120, alpha=0.05, horizon=H, sequential_control="consecutive_k").detected
out = {}
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
for H, irr in [(24, False), (120, False), (24, True)]:
    k = int(np.sum(pmap(job, [(s, H, irr) for s in range(N)])))
    out[f"H{H}_irr{irr}"] = dict(rate=k/N, ci=wilson(k, N)); print(H, irr, out[f"H{H}_irr{irr}"], flush=True)
json.dump(out, open("p17_results.json", "w"), indent=1)
