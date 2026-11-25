"""How often do Huber IRLS / MM refits hit max_iter silently? (converged flag is never surfaced).
Seed 81. Run: python audit/repro_irls_convergence.py"""
import numpy as np
from scipy import stats
from tspec._linalg import prepare_weights
from tspec._robust import solve_huber_irls, solve_mm_biweight
rng = np.random.default_rng(81)
for noise in ("normal", "t1(Cauchy)", "t3", "10%x20 contam"):
    nc_h = nc_m = 0; R = 500
    for r in range(R):
        n = 150; t = np.linspace(0, 1, n); D = np.column_stack([np.ones(n), t, np.maximum(0, t - 0.5)])
        e = {"normal": rng.normal(size=n), "t1(Cauchy)": stats.t.rvs(1, size=n, random_state=rng),
             "t3": stats.t.rvs(3, size=n, random_state=rng),
             "10%x20 contam": np.where(rng.uniform(size=n) < .1, 20, 1) * rng.normal(size=n)}[noise]
        y = 1 + t + e
        nc_h += not solve_huber_irls(D, y, prepare_weights(None, n)).converged
        nc_m += not solve_mm_biweight(D, y, prepare_weights(None, n)).converged
    print(f"{noise:>14}: Huber not converged {nc_h / R:.1%}, MM not converged {nc_m / R:.1%} (R={R})")
