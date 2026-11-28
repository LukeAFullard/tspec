"""P1b: high-replicate check of lssa(peak_fap='montecarlo') and 'sidak' at 5%."""
import json, numpy as np, tspec, warnings
from common import pmap, wilson
warnings.filterwarnings("ignore")
def job(a):
    seed, method = a
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 8, 100))
    return tspec.lssa(t, rng.normal(size=100), omega=np.linspace(0.1, 10, 600), peak_fap=method, n_mc=199).peak_fap
out = {}
for m, N in [("montecarlo", 2000), ("sidak", 4000)]:
    f = np.array(pmap(job, [(s, m) for s in range(N)]))
    k = int(np.sum(f <= 0.05)); out[m] = dict(rej05=k/N, ci95=wilson(k, N), rej01=float(np.mean(f <= 0.01)), n=N); print(m, out[m], flush=True)
json.dump(out, open("p1b_results.json", "w"), indent=1)
