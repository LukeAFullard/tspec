"""Run README/docs fragments that reference placeholder data, by supplying synthetic data. Seed 3."""
import warnings, numpy as np, pandas as pd
import tspec
from tspec import CensoredConfig, just_jumps, just_decompose, just_monitor, analyze_series
warnings.simplefilter("ignore")
rng = np.random.default_rng(3)
dates = pd.date_range("2012-01-01", periods=144, freq="MS") + pd.to_timedelta(rng.integers(0, 20, 144), "D")
conc = np.exp(0.2 * np.cos(2 * np.pi * np.arange(144) / 12) + rng.normal(0, 0.3, 144))
nitrate_strings = [f"<{0.65:.2f}" if v < 0.65 else float(v) for v in conc]
cfg = CensoredConfig(method="ros", dist="lognormal", impute_type="stochastic", detect_dl_shifts=True,
                     recensor_check=True, random_state=42)
jumps = just_jumps(dates, nitrate_strings, season="ols", mag_th=0.8, jump_alpha=0.01, censoring=cfg)
decomp = just_decompose(dates, nitrate_strings, jump_indices=[j.index for j in jumps.jumps], season="ols", censoring=cfg)
print("README censored fragment: OK", len(jumps.jumps), "jumps")
t = np.sort(rng.uniform(2018, 2024, 140)); evi = 0.4 + 0.2 * np.cos(2 * np.pi * t) + rng.normal(0, 0.02, 140)
res = just_monitor(t, evi, start=2022.5, season="allssa", alpha=0.01, freq_correction="sidak", sequential_control="pocock")
print("README monitor fragment: OK detected=", res.detected)
y_strings = ["<0.05", 0.12, "<0.05", 0.34, None, ">10.0"]
try:
    analyze_series(pd.date_range("2020", periods=6, freq="MS"), y_strings)
    print("censored-data.md fragment 1: OK")
except Exception as e:
    print("censored-data.md fragment 1 (6-point toy series):", type(e).__name__, str(e)[:120])
values = conc; is_left = conc < 0.65
r2 = analyze_series(dates, np.where(is_left, 0.65, values), censor_status=is_left)
print("censored-data.md fragment 2: OK, segments=", len(r2.segments))
