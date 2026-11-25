"""Look-ahead via default horizon: the alarm threshold applied at monitoring step d depends on how
many observations exist AFTER step d (horizon defaults to n_valid - start). Seed 31.
Run: python audit/repro_monitor_lookahead.py
"""
import warnings
import numpy as np
import tspec
warnings.simplefilter("ignore")
rng = np.random.default_rng(31)
n_his = 96
t = np.sort(rng.uniform(2000, 2010, 240))
y = 2 + 0.1 * (t - 2000) + 0.5 * np.cos(2 * np.pi * t) + rng.normal(0, 0.15, t.size)
y[n_his + 5:] -= 0.25  # modest real disturbance starting at monitoring step 6
for n_total in (n_his + 12, n_his + 48, n_his + 144):
    r = tspec.just_monitor(t[:n_total], y[:n_total], start=n_his, season="ols", alpha=0.05,
                           stop_on_first=False)
    s1 = r.steps[0]
    first = next((s.step for s in r.steps if s.exceeded), None)
    print(f"data available up to step {n_total - n_his:4d}: horizon={r.horizon:4d} "
          f"step-1 alpha_step={s1.alpha_step:.2e} crit={s1.critical_value:.4f}; "
          f"first exceedance at step {first}; detected={r.detected}")
# identical data for steps 1..12, explicit horizon removes the dependence
for n_total in (n_his + 12, n_his + 144):
    r = tspec.just_monitor(t[:n_total], y[:n_total], start=n_his, season="ols", alpha=0.05,
                           horizon=144, stop_on_first=False)
    print(f"explicit horizon=144, data up to step {n_total - n_his}: step-1 crit={r.steps[0].critical_value:.4f}")
