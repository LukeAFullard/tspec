# Tutorial: Water Quality with `<DL` Non-Detects & Lab Upgrades

In this tutorial you will analyze an irregular 8-year river nitrate series containing:

- Irregular monthly-to-biweekly sampling with winter gaps (zero interpolation!)
- Strong annual runoff seasonality
- ~25% left-censored `<DL` lab reports
- A true management-driven concentration drop in 2019

## Step 1: Simulate a Realistic Water-Quality Record

```python
import numpy as np
import tspec
from tspec import CensoredConfig

rng = np.random.default_rng(42)
dates = np.arange("2016-01-15", "2024-01-15", 22, dtype="datetime64[D]")
# Drop 25% random samples + a 3-month flood-damaged gauge outage
keep = rng.uniform(size=dates.size) > 0.25
t_yr_all = (dates - dates[0]).astype("timedelta64[D]").astype(float) / 365.25
keep &= ~((t_yr_all > 2.2) & (t_yr_all < 2.45))
dates = dates[keep]
t_yr = t_yr_all[keep]

# Lognormal nitrate (mg/L) with seasonal runoff & true policy drop at year 4.0
log_mu = np.where(t_yr < 4.0, 0.7, -0.3) + 0.45 * np.cos(2 * np.pi * t_yr)
nitrate_true = np.exp(log_mu + rng.normal(0.0, 0.25, size=t_yr.size))

# Censor bottom ~25% at laboratory detection limit 0.80 mg/L
dl = 0.80
lab_strings = [f"<{dl:.2f}" if v < dl else f"{v:.3f}" for v in nitrate_true]
```

## Step 2: Run `analyze_series()` with `ndimpute` Safeguards

```python
result = tspec.analyze_series(
    dates,
    lab_strings,
    censoring=CensoredConfig(
        method="ros",              # Robust ROS (Helsel 2012)
        dist="lognormal",
        impute_type="stochastic",  # Time-order safe (no ROS quantile trend artifact)
        n_stability_draws=20,      # Checks breakpoint stability across 20 tail draws
        detect_dl_shifts=True,     # Warns if a jump coincides with a lab DL upgrade
        random_state=0,
    ),
)

print(result.explain())
print("\n--- Summary ---")
print(result.summary())
```

## Step 3: Checking for Laboratory Detection-Limit Artifacts

If a laboratory upgraded its analyzer mid-record (e.g., `<0.80` → `<0.10`), `result.explain()` automatically alerts you with `DL_SHIFT_COINCIDENCE`. Turn on `recensor_check=True` to verify whether the jump survives once all values below `0.80` are recensored to a common limit:

```python
res_check = tspec.just_jumps(
    dates,
    lab_strings,
    jump_alpha=0.05,
    bonferroni_window=True,
    mag_th=0.0,
    dir_th=0.0,
    censoring=CensoredConfig(
        method="ros",
        dist="lognormal",
        detect_dl_shifts=True,
        recensor_check=True,
        random_state=0,
    ),
)
print(res_check.to_dataframe())
```
