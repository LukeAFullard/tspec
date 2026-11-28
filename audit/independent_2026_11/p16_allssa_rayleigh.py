"""P16: AUD-05 validation of allssa(search='rayleigh') vs the published integer-band search."""
import json, sys, time, warnings
import numpy as np, tspec
from common import pmap, wilson
warnings.filterwarnings("ignore")
NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 100
out = {}

def two_close(a):
    seed, search = a
    rng = np.random.default_rng(seed); t = np.sort(rng.uniform(0, 12, 300))
    y = np.cos(2*np.pi*1.37*t + 0.4) + 0.8*np.sin(2*np.pi*1.0*t) + rng.normal(0, 0.5, t.size)
    f = tspec.allssa(t, y, decimal=2, n_mc=499, search=search).frequencies
    return dict(both=bool(np.any(abs(f-1.0) < 0.02) and np.any(abs(f-1.37) < 0.02)), nf=f.size,
                spurious=int(np.sum((abs(f-1.0) >= 0.02) & (abs(f-1.37) >= 0.02))))

def solar(a):
    seed, search = a
    rng = np.random.default_rng(seed); t = np.arange(0, 60, 1/12)
    y = 2*np.sin(2*np.pi*t/11.2) + np.cos(2*np.pi*t) + rng.normal(0, .5, t.size)
    f = tspec.allssa(t, y, n_mc=499, search=search).frequencies
    return dict(found_11=bool(np.any(abs(f - 1/11.2) < 0.01)), found_1=bool(np.any(abs(f-1) < 0.02)),
                spurious=int(np.sum((abs(f-1/11.2) >= 0.01) & (abs(f-1) >= 0.02))))

def noninteger(a):
    seed, search = a
    rng = np.random.default_rng(300 + seed); t = 2000 + (np.arange(240) + .5)/12
    y = 3 + 0.05*(t-t[0]) + np.cos(2*np.pi*t) + 0.7*np.cos(2*np.pi*2.37*t + 0.5) + rng.normal(0, .25, t.size)
    f = tspec.allssa(t, y, decimal=2, n_mc=499, search=search).frequencies
    return dict(found_237=bool(np.any(abs(f-2.37) < 0.02)), spurious=int(np.sum((abs(f-2.37) >= 0.02) & (abs(f-1) >= 0.02))))

TIDE = {"M2": 24/12.4206012, "S2": 2.0, "K1": 24/23.9344696, "O1": 24/25.8193417}
def tides(a):
    seed, search = a
    rng = np.random.default_rng(500 + seed); t = np.arange(0, 60, 1/24)
    amp = {"M2": 1.0, "S2": 0.35, "K1": 0.45, "O1": 0.30}
    y = 0.002*t + sum(amp[k]*np.cos(2*np.pi*f*t + 0.3*i) for i, (k, f) in enumerate(TIDE.items())) + rng.normal(0, .1, t.size)
    keep = rng.random(t.size) > 0.3
    f = tspec.allssa(t[keep], y[keep], omega=np.arange(0.5, 4.0, 1/240), decimal=3, n_mc=499, search=search).frequencies
    return dict(n_found=sum(bool(np.any(abs(f - v) < 0.005)) for v in TIDE.values()), spurious=int(sum(np.min(abs(x - np.array(list(TIDE.values())))) >= 0.005 for x in f)))

def null(a):
    seed, search = a
    rng = np.random.default_rng(900 + seed); t = np.sort(rng.uniform(0, 10, 120))
    return dict(any=tspec.allssa(t, rng.normal(size=120), n_mc=499, search=search).frequencies.size > 0)

for name, fn, nrep in [("two_close_1.0_1.37", two_close, NREP), ("solar_11.2yr_default_grid", solar, NREP // 2),
                       ("noninteger_2.37", noninteger, NREP), ("tides_M2S2K1O1", tides, NREP // 4), ("null_alpha0.01", null, 2 * NREP)]:
    for search in ["rayleigh", "integer"]:
        t1 = time.time(); res = pmap(fn, [(s, search) for s in range(nrep)])
        d = {k: float(np.mean([r[k] for r in res])) for k in res[0]}
        if "any" in d: d["any_ci"] = wilson(int(round(d["any"]*nrep)), nrep)
        d["sec_per_call"] = round((time.time() - t1) * 4 / nrep, 2)
        out[f"{name}|{search}"] = d; print(name, search, d, flush=True)
json.dump(out, open("p16_results.json", "w"), indent=1)
