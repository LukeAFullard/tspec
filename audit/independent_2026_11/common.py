"""Shared helpers for the independent 2026-11 audit probes."""
import os, warnings, multiprocessing as mp
import numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

def ar1(rng, n, rho, sd=1.0):
    e = rng.normal(0, sd, n); x = np.empty(n); x[0] = e[0] / np.sqrt(1 - rho**2) if rho < 1 else e[0]
    for i in range(1, n):
        x[i] = rho * x[i-1] + e[i]
    return x * np.sqrt(1 - rho**2)  # unit marginal variance * sd

def pmap(fn, args, procs=4):
    with mp.get_context("fork").Pool(procs) as p:
        return p.map(fn, args, chunksize=max(1, len(args) // (procs * 8)))

def wilson(k, n):
    from scipy.stats import norm
    z = norm.ppf(0.975); ph = k / n; den = 1 + z*z/n
    c = (ph + z*z/(2*n)) / den; h = z*np.sqrt(ph*(1-ph)/n + z*z/(4*n*n)) / den
    return [float(c-h), float(c+h)]
