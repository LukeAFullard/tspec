"""P2b: just_jumps power vs a whole-record MC-calibrated sup-F step test (oracle-free baseline)."""
import json, sys, warnings
import numpy as np, tspec
from common import pmap
warnings.filterwarnings("ignore")
NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 200

def supf_pvalue(t, y, rng, nmc=500):
    X = np.column_stack([np.ones_like(t), t - t[0]] + [f(2*np.pi*h*t) for h in (1, 2, 3, 4) for f in (np.cos, np.sin)])
    Q, _ = np.linalg.qr(X)
    ks = np.arange(int(0.1*t.size), int(0.9*t.size))
    A = (np.arange(t.size)[:, None] >= ks[None, :]).astype(float)
    A -= Q @ (Q.T @ A); A /= np.linalg.norm(A, axis=0)
    def stat(Y):
        R = Y - Q @ (Q.T @ Y)
        proj = A.T @ R; s2 = (np.sum(R*R, 0)[None, :] - proj**2) / (t.size - X.shape[1] - 1)
        return np.max(proj**2 / s2, axis=0)
    obs = stat(y[:, None])[0]
    null = stat(rng.standard_normal((t.size, nmc)))
    return (1 + np.sum(null >= obs)) / (nmc + 1)

def job(a):
    seed, jump, kw = a
    rng = np.random.default_rng(seed)
    n, span = 240, 20.0
    t = np.sort(rng.uniform(0, span, n))
    y = 0.05*t + jump*(t >= 10) + np.cos(2*np.pi*t) + 0.4*np.sin(4*np.pi*t) + rng.normal(0, 0.5, n)
    r = tspec.just_jumps(t, y, season="ols", **kw)
    hit = any(abs(r.times_numeric[j.valid_index] - 10) < 0.5 for j in r.jumps)
    return hit, supf_pvalue(t, y, rng) < 0.05
out = {}
for jump in [0.0, 0.5, 1.0, 1.5, 2.0]:
    for lab, kw in [("default_size36", {}), ("size72", dict(size=72, step=12)), ("size120", dict(size=120, step=12))]:
        res = pmap(job, [(s, jump, kw) for s in range(NREP)])
        out[f"jump{jump}_{lab}"] = dict(just_recall_or_fwer=float(np.mean([r[0] for r in res])), global_supF_reject=float(np.mean([r[1] for r in res])))
        print(jump, lab, out[f"jump{jump}_{lab}"], flush=True)
json.dump(out, open("p2b_results.json", "w"), indent=1)
