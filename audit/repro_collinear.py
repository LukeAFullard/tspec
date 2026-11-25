"""Near-collinear known frequencies: does tspec detect ill-conditioning / rank deficiency?
Seed 91. Run: python audit/repro_collinear.py"""
import warnings
import numpy as np
import tspec
warnings.simplefilter("ignore")
rng = np.random.default_rng(91)
t = np.sort(rng.uniform(2000, 2004, 100)); y = np.cos(2 * np.pi * t) + rng.normal(0, 0.1, 100)
for df in (1e-2, 1e-4, 1e-6, 1e-9):
    r = tspec.lssa(t, y, known_freqs=[1.0, 1.0 + df])
    D = np.column_stack([np.ones(100), t - t[0], np.cos(2*np.pi*(t-t[0])), np.sin(2*np.pi*(t-t[0])),
                         np.cos(2*np.pi*(1+df)*(t-t[0])), np.sin(2*np.pi*(1+df)*(t-t[0]))])
    true_cond = np.linalg.cond(D / np.linalg.norm(D, axis=0))
    codes = [w.code for w in r.diagnostics.warnings]
    print(f"df={df:g}: reported cond={r.diagnostics.condition_number_max:.2e}, column-normalised true cond={true_cond:.2e}, "
          f"max|coef|={np.max(np.abs(r.coeff)):.3g}, warnings={codes}")
