"""P3: datetime->decimal-year conversion distorts non-annual frequencies (leap years)."""
import json, warnings
import numpy as np, tspec
warnings.filterwarnings("ignore")
# hourly record 2023-01-01 .. 2025-01-01 (spans leap year 2024), pure M2 tide, no noise
dt = np.arange("2023-01-01T00", "2025-01-01T00", 3, dtype="datetime64[h]")
hours = (dt - dt[0]).astype(float)
f_m2_cph = 1 / 12.4206012
y = np.cos(2*np.pi*f_m2_cph*hours)
out = {}
# frequency in cycles per *mean* year: what the user must pass to tspec for datetime input
for lab, ypd in [("365.25", 365.25), ("365", 365.0), ("366", 366.0)]:
    f_cpy = f_m2_cph*24*ypd
    r = tspec.lssa(dt, y, omega=[f_cpy], trend="constant")
    out[f"datetime_input_spectrum_at_M2_cpy_{lab}"] = float(r.spectrum[0])
# best achievable over a fine grid
grid = np.linspace(f_m2_cph*24*364, f_m2_cph*24*367, 4001)
r = tspec.lssa(dt, y, omega=grid, trend="constant")
out["datetime_input_best_spectrum_fine_grid"] = float(r.spectrum.max())
# numeric hours input (uniform axis) - control
r = tspec.lssa(hours, y, omega=[f_m2_cph], trend="constant")
out["numeric_hours_input_spectrum"] = float(r.spectrum[0])
# daily annual cycle on datetime axis: fine (harmonic of calendar year)
d = np.arange("2000-01-01", "2010-01-01", dtype="datetime64[D]")
days = (d - d[0]).astype(float)
ya = np.cos(2*np.pi*days/365.2425)
r = tspec.lssa(d, ya, omega=[1.0], trend="constant")
out["annual_cycle_daily_datetime"] = float(r.spectrum[0])
# 7-day weekly cycle on datetime axis over 10 yr
yw = np.cos(2*np.pi*days/7.0)
best = tspec.lssa(d, yw, omega=np.linspace(52.0, 52.4, 2001), trend="constant")
out["weekly_cycle_daily_datetime_best_spectrum"] = float(best.spectrum.max())
out["weekly_cycle_numeric_days"] = float(tspec.lssa(days, yw, omega=[1/7.0], trend="constant").spectrum[0])
# timestep non-uniformity
tn = tspec.lssa(d, ya, omega=[1.0]).times_numeric
dd = np.diff(tn)
out["decimal_year_step_ratio_max_min"] = float(dd.max()/dd.min())
print(json.dumps(out, indent=1)); json.dump(out, open("p3_results.json", "w"), indent=1)
