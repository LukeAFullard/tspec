"""MC5b: FWER when just_monitor is used as an actual near-real-time loop (re-run each time a new
observation arrives, default horizon), vs a single retrospective call. Stable series as in MC5.
Seed 5500. Run: OMP_NUM_THREADS=1 python audit/mc5b_monitor_realtime_loop.py <R> <D>
"""
import sys, json, warnings
import numpy as np
import tspec
warnings.simplefilter("ignore")
R = int(sys.argv[1]) if len(sys.argv) > 1 else 300
D = int(sys.argv[2]) if len(sys.argv) > 2 else 24
alpha = 0.05
rng = np.random.default_rng(5500 + D)
n_his = 96
retro = loop = loop_fixedH = 0
for r in range(R):
    n = n_his + D
    t = np.sort(rng.uniform(2000, 2000 + n / 24.0, n))
    y = 2 + 0.1 * (t - 2000) + 0.5 * np.cos(2 * np.pi * t) + 0.2 * np.sin(4 * np.pi * t) + rng.normal(0, 0.15, n)
    retro += tspec.just_monitor(t, y, start=n_his, season="ols", alpha=alpha).detected
    hit = hitH = False
    for d in range(1, D + 1):
        m = n_his + d
        if m - n_his < 1:
            continue
        if not hit:
            hit = tspec.just_monitor(t[:m + 0], y[:m + 0], start=n_his, season="ols", alpha=alpha).detected if m > n_his else False
        if not hitH:
            hitH = tspec.just_monitor(t[:m], y[:m], start=n_his, season="ols", alpha=alpha, horizon=D).detected
        if hit and hitH:
            break
    loop += hit; loop_fixedH += hitH
res = {"R": R, "D": D, "alpha": alpha, "FWER_single_retrospective_call": retro / R,
       "FWER_realtime_loop_default_horizon": loop / R,
       "FWER_realtime_loop_fixed_horizon=D": loop_fixedH / R,
       "MCSE_at_alpha": float(np.sqrt(alpha * (1 - alpha) / R))}
print(json.dumps(res))
json.dump(res, open(f"audit/mc5b_results_D{D}.json", "w"), indent=1)
