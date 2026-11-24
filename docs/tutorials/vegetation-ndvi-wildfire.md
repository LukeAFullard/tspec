# Tutorial: Satellite Vegetation (NDVI/EVI) with Wildfire Disturbance

Satellite vegetation indices (Landsat/MODIS/Sentinel-2) are heavily affected by clouds and snow, producing **unevenly spaced records with multi-month gaps** plus multi-harmonic phenology (annual + semi-annual cycles).

## Step 1: Build an Irregular Cloud-Gapped NDVI Record with a Wildfire Drop

```python
import numpy as np
import tspec

rng = np.random.default_rng(123)
dates = np.arange("2015-01-01", "2022-01-01", 16, dtype="datetime64[D]")
t_yr = (dates - dates[0]).astype("timedelta64[D]").astype(float) / 365.25

# Simulate 35% cloud dropout + 4-month winter snow blackout
keep = rng.uniform(size=dates.size) > 0.35
keep &= ~((t_yr > 1.8) & (t_yr < 2.15))
dates = dates[keep]
t_yr = t_yr[keep]

# Wildfire in summer 2018 (t_yr = 3.6): sharp drop (-0.32) followed by slow recovery (+0.07/yr)
trend = np.where(t_yr < 3.6, 0.58 - 0.01 * t_yr, 0.26 + 0.07 * (t_yr - 3.6))
phenology = 0.16 * np.cos(2 * np.pi * t_yr) + 0.06 * np.sin(2 * np.pi * 2.0 * t_yr)
ndvi = trend + phenology + rng.normal(0.0, 0.025, size=t_yr.size)
```

## Step 2: Detect Abrupt Jumps via JUST (`ALLSSA`)

Using `season="allssa"` automatically separates both the annual (`1 c/yr`) and semi-annual (`2 c/yr`) phenology cycles from the piecewise trend without interpolating cloud gaps.

```python
jumps = tspec.just_jumps(
    dates,
    ndvi,
    season="allssa",
    mag_th=0.10,
    dir_th=0.03,
    jump_alpha=0.01,
)
print(jumps.explain())
```

## Step 3: Decompose into Trend, Seasonal & Remainder (with Optional Regular Grid)

Passing `eval_times` evaluates the fitted parametric trend and seasonal harmonics onto an evenly spaced visualization grid **without ever interpolating the input observations**.

```python
eval_grid = np.arange("2015-02-01", "2021-12-01", 10, dtype="datetime64[D]")
decomp = tspec.just_decompose(
    dates,
    ndvi,
    jump_indices=jumps.jump_indices,
    eval_times=eval_grid,
    season="allssa",
)
print(decomp.summary())
df = decomp.to_dataframe()
```
