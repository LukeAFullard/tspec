# Tutorial: Irregular Spectral Analysis (`lssa` & `allssa`)

Classical FFT and Lomb–Scargle periodograms struggle when a record simultaneously has **irregular timestamps, multi-month blackout gaps, a linear trend or abrupt jump, and multiple closely spaced non-integer frequencies**.

`tspec.lssa()` and `tspec.allssa()` solve all of those jointly via weighted least-squares spectral analysis (Ghaderpour & Pagiatakis, 2017; Ghaderpour et al., 2018) with **exact family-wise false-alarm control** (`peak_fap="sidak" | "montecarlo"` and `significance="montecarlo" | "sidak"`).

---

## Step 1: Why Standard Periodograms Leak on Irregular Records

Consider an 8-year geophysical or hydroclimatic record with **35% random missingness + a 5-month blackout gap**, a linear trend, and **three non-integer sinusoids** (`0.73`, `1.85`, and `3.25` cy/yr — e.g., an inter-annual ~16.4-month oscillation plus two non-harmonic intra-annual cycles):

- `lssa()` scans a user-defined frequency grid `omega` while projecting out the trend (and any known `jump_indices` or `known_freqs`) via an orthogonal Schur complement, reporting both per-frequency Beta $p$-values and the family-wise `peak_fap`.
- `allssa()` iteratively extracts the strongest peak, refines its frequency around its rounded integer bin to `decimal` decimal places (e.g. `0.01` cy/yr), and re-estimates **all** harmonics and trend terms simultaneously — eliminating spectral leakage sidelobes from irregular sampling.

```python
import numpy as np
import tspec

rng = np.random.default_rng(2026)
t_full = np.sort(rng.uniform(2016.0, 2024.0, size=240))
# 5-month sensor blackout gap (zero interpolation!)
t = t_full[~((t_full > 2019.1) & (t_full < 2019.52))]

true_freqs = np.array([0.73, 1.85, 3.25])
trend = 10.0 + 0.35 * (t - 2016.0)
signal = (
    0.9 * np.cos(2 * np.pi * 0.73 * (t - 2016.0))
    + 1.2 * np.sin(2 * np.pi * 1.85 * (t - 2016.0))
    + 0.55 * np.cos(2 * np.pi * 3.25 * (t - 2016.0))
)
y = trend + signal + rng.normal(0.0, 0.25, size=t.size)

# 1. Exploratory spectrum with family-wise peak_fap
omega_grid = np.linspace(0.2, 4.0, 381)
spec = tspec.lssa(t, y, omega=omega_grid, trend="linear", peak_fap="sidak")
print(spec.summary())

# 2. Iterative Anti-Leakage LSSA (family-wise Monte Carlo calibrated by default)
# Tip: for non-integer frequencies on a multi-year record of span T, set the coarse
# grid spacing <= 1/(2T) (here T=8 yr -> step 0.1 cy/yr) so narrow peaks do not fall
# in the spectral null between integer grid points.
res_allssa = tspec.allssa(
    t,
    y,
    omega=np.arange(0.5, 4.05, 0.1),
    decimal=2,
    alpha=0.01,
    trend="linear",
    significance="montecarlo",
)
print(res_allssa.summary())
print("Recovered frequencies:", np.sort(res_allssa.frequencies))
```

---

## Step 2: Controlling Spurious Cycles on Pure Noise

The original published ALLSSA stopping rule (`significance="per_frequency"`, forced by `parity_mode=True`) checks each refined frequency against a single-frequency Beta critical value — ignoring the search across hundreds of frequencies and selecting spurious cycles in roughly **20% of pure-noise records** at $\alpha = 0.01$.

`tspec` defaults to `significance="montecarlo"` (exact Gaussian-null distribution of the searched spectral maximum) and also offers the fast analytical `significance="sidak"` (using effective independent frequencies $K_{\text{eff}} \approx 3\,T\,\Delta f$):

```python
rng_null = np.random.default_rng(101)
t_null = np.sort(rng_null.uniform(2018.0, 2024.0, size=140))
y_null = 5.0 + 0.1 * (t_null - 2018.0) + rng_null.normal(0.0, 0.4, size=t_null.size)

spec_null = tspec.lssa(t_null, y_null, omega=np.linspace(0.5, 4.0, 150), peak_fap="montecarlo")
allssa_null = tspec.allssa(t_null, y_null, omega=np.array([1.0, 2.0, 3.0]), alpha=0.01)
print(f"Pure-noise peak_fap={spec_null.peak_fap:.3f}, ALLSSA detected frequencies={allssa_null.frequencies}")
```

---

## Step 3: Joint Spectral Analysis Across Known Jumps & Heteroscedastic Weights

If your record contains a known abrupt offset (e.g., an instrument swap at index `j`), pass `jump_indices=[j]` directly into `lssa()` or `allssa()`. The step discontinuity is projected out simultaneously with the trend so step energy never leaks into low-frequency spectral peaks:

```python
j_idx = int(np.searchsorted(t, 2020.5))
y_jumped = y + np.where(np.arange(t.size) >= j_idx, 3.0, 0.0)
weights = np.where(t < 2020.5, 1.0, 2.0)  # higher precision after sensor upgrade

res_joint = tspec.allssa(
    t,
    y_jumped,
    weights=weights,
    jump_indices=[j_idx],
    omega=np.arange(0.5, 4.05, 0.1),
    decimal=2,
    alpha=0.01,
    trend="linear",
)
print("Recovered frequencies with jump projected out:", np.sort(res_joint.frequencies))
```
