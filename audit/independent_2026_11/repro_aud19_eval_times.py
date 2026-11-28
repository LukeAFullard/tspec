"""AUD-19, found by the pre-registered decomposition validation (real Choptank sampling times).

(a) season="ols" with count windows: a window that covers only a few days of a storm-sampling
    burst cannot separate four seasonal harmonics from the linear trend; the fit is
    ill-conditioned and the trend is off by 50+ units, without any warning.
(b) eval_times: windows are rebuilt on the evaluation grid, so the values at the observation
    times differ from a call without eval_times and can be NaN.
Run: OMP_NUM_THREADS=1 python repro_aud19_eval_times.py
"""
import numpy as np
import pandas as pd
import tspec

d = pd.read_csv("../../validation/data/choptank_01491000_nitrate.csv", parse_dates=["date"])
d = d[(d["date"] >= "1985-01-01") & (d["date"] <= "2023-09-30")]
t = 2000.0 + (d["date"] - pd.Timestamp("2000-01-01")).dt.total_seconds().to_numpy() / (365.2425 * 86400)
rng = np.random.default_rng(5)
y = 2.0 + 0.03 * (t - t[0]) + 0.8 * np.cos(2 * np.pi * t) + 0.3 * rng.normal(size=t.size)  # synthetic values on real times
grid = 2000.0 + (np.arange(int((t[0] - 2000) * 12) + 1, int((t[-1] - 2000) * 12)) + 0.5) / 12
a = tspec.just_decompose(t, y, season="ols")
print("(a) worst trend errors (no eval_times):", np.round(np.sort(np.abs(a.trend - (2.0 + 0.03 * (t - t[0]))))[-3:], 1),
      "at", np.round(t[np.argsort(np.abs(a.trend - (2.0 + 0.03 * (t - t[0]))))[-3:]], 3))
b = tspec.just_decompose(t, y, season="ols", eval_times=grid)
true_trend = 2.0 + 0.03 * (t - t[0])
print("trend RMSE at observations, no eval_times :", float(np.sqrt(np.mean((a.trend - true_trend) ** 2))))
print("trend RMSE at observations, with eval_times:", float(np.sqrt(np.mean((b.trend - true_trend) ** 2))))
print("(b) NaN trend values at observations with eval_times:", int(np.isnan(b.trend).sum()), "of", t.size)
g_true = 2.0 + 0.03 * (grid - t[0])
print("trend_reg RMSE on the monthly grid         :", float(np.sqrt(np.nanmean((b.trend_reg - g_true) ** 2))))
