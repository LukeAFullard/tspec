# Tutorial: Outliers, Heavy Tails & Correlated Red Noise

Environmental, hydrological, and geophysical sensors frequently produce **two distinct departures from i.i.d. Gaussian noise**:

1. **Isolated spikes & heavy tails** (stormflow turbidity surges, telemetry glitches, unwrapping outliers), which pull plain OLS slopes and inflate jump false alarms.
2. **Serial correlation (red noise) & changing noise variance (heteroscedasticity)**, which cause naive white-noise $p$-values to over-detect trends and jumps, or cause unweighted STPD to confuse a variance shift with a vertical jump.

This tutorial shows how `tspec` handles all three natively.

---

## Step 1: Neutralizing Extreme Spikes (`robust="mm"` + `slope_estimator="sen"`)

Here we inject isolated $30\sigma$–$40\sigma$ glitches into an irregular seasonal record with a genuine abrupt step jump at `2019.0` (`+2.2`) and a continuous turning point at `2022.0` (`slope +0.2 → -1.1/yr`).

- `robust="huber"` uses convex Huber IRLS ($k=1.345$) and is ideal for heavy-tailed ($t_3$/$t_5$) errors.
- `robust="mm"` uses Huber-started Tukey-biweight IRLS ($c=4.685$; not a 50%-breakdown S-started MM), assigning **zero weight** to extreme isolated spikes while preserving sustained jumps.
- `slope_estimator="sen"` computes per-segment Theil–Sen (or automatic Akritas–Theil–Sen when censored) slopes, distribution-free 95% CIs, and AR(1)-corrected Mann–Kendall $p$-values.

```python
import numpy as np
import tspec

rng = np.random.default_rng(314)
t = np.sort(rng.uniform(2015.0, 2025.0, size=200))
trend_true = (
    np.where(t < 2019.0, 3.0 + 0.2 * (t - 2015.0), 6.0 + 0.2 * (t - 2019.0))
    + np.where(t > 2022.0, -1.3 * (t - 2022.0), 0.0)
)
seasonal_true = 0.7 * np.cos(2 * np.pi * t)
y_clean = trend_true + seasonal_true + rng.normal(0.0, 0.16, size=t.size)

# Inject 6 isolated extreme spikes (30–40 sigma)
y_spiky = y_clean.copy()
spike_idx = [15, 48, 95, 125, 160, 185]
y_spiky[spike_idx] += np.array([6.0, -5.5, 6.5, -6.0, 5.8, -5.2])

res_rob = tspec.analyze_series(
    t,
    y_spiky,
    robust="mm",
    slope_estimator="sen",
    just_kwargs={"season": "ols", "jump_alpha": 0.01},
    decompose_kwargs={"season": "ols"},
    band_method="conditional",
)
print(res_rob.summary())
print(res_rob.segments_to_dataframe()[["segment_index", "estimator", "slope", "slope_ci_low", "slope_ci_high", "pvalue"]])
```

For continuous piecewise trends without seasonality, you can also fit an exact continuous **$L_1$ (LAD)** or **Huber** hinge-spline directly with `fit_piecewise_trend(..., mode="l1" | "huber")`:

```python
t_tp = np.linspace(2016.0, 2024.0, 120)
y_tp = 2.0 + 0.5 * (t_tp - 2016.0) - 1.4 * np.maximum(0.0, t_tp - 2020.0) + rng.normal(0.0, 0.15, size=120)
y_tp[[20, 45, 85, 105]] += np.array([8.0, -7.5, 7.0, -8.5])

pw_l1 = tspec.fit_piecewise_trend(t_tp, y_tp, tp_indices=[60], mode="l1")
pw_huber = tspec.fit_piecewise_trend(t_tp, y_tp, tp_indices=[60], mode="huber")
print("L1 slopes:   ", np.round(pw_l1.slopes, 3))
print("Huber slopes:", np.round(pw_huber.slopes, 3))
```

---

## Step 2: Correlated Red Noise (`weights = C⁻¹` and `mk_correction="ar1"`)

When observations are serially correlated (e.g., AR(1) with $\rho = 0.7$), white-noise $p$-values treat correlated samples as independent and inflate false-positive rates dramatically. `tspec` handles red noise in two complementary ways:

1. **Exact 2D inverse-covariance weighting (`weights = C⁻¹`)** in `lssa()`, `allssa()`, `just_jumps()`, `just_decompose()`, and `just_monitor()` — including exact whitened Huber/MM IRLS when combined with `robust="huber" | "mm"`.
2. **Rank-based AR(1) bootstrap (`mk_correction="ar1"`, default)** for Mann–Kendall $p$-values and Theil–Sen / Akritas–Theil–Sen CIs in `analyze_series(..., slope_estimator="sen")`, which stays outlier-resistant and calibrated up to $\rho = 0.9$.

```python
# Generate an AR(1) red-noise series (rho = 0.7) with NO jump and NO true trend
rng_rn = np.random.default_rng(77)
n_rn = 120
t_rn = np.linspace(2014.0, 2024.0, n_rn)
rho = 0.7
eps = np.zeros(n_rn)
eps[0] = rng_rn.normal(0.0, 1.0)
for i in range(1, n_rn):
    eps[i] = rho * eps[i - 1] + np.sqrt(1.0 - rho**2) * rng_rn.normal(0.0, 1.0)
y_rn = 5.0 + eps

# Build the exact AR(1) inverse covariance matrix P = C^{-1}
idx = np.arange(n_rn)
C_ar1 = rho ** np.abs(idx[:, None] - idx[None, :])
P_ar1 = np.linalg.inv(C_ar1)

# 1. Spectral screening: compare unweighted vs. exact 2D GLS weights
spec_unw = tspec.lssa(t_rn, y_rn, omega=np.linspace(0.5, 4.0, 40), trend="linear")
spec_gls = tspec.lssa(t_rn, y_rn, weights=P_ar1, omega=np.linspace(0.5, 4.0, 40), trend="linear")
print(f"Unweighted peak_fap: {spec_unw.peak_fap:.4f} | 2D GLS peak_fap: {spec_gls.peak_fap:.4f}")

# 2. Segment trend test: compare unadjusted MK vs. default rank-AR(1) bootstrap
pipe_raw = tspec.analyze_series(
    t_rn, y_rn, detect_jumps=False, detect_turning_points=False,
    remove_seasonality=False, slope_estimator="sen", mk_correction="none",
)
pipe_ar1 = tspec.analyze_series(
    t_rn, y_rn, detect_jumps=False, detect_turning_points=False,
    remove_seasonality=False, slope_estimator="sen", mk_correction="ar1",
)
print(
    f"MK p-value (none): {pipe_raw.segments[0].pvalue:.4f} | "
    f"MK p-value (ar1): {pipe_ar1.segments[0].pvalue:.4f}"
)
```

---

## Step 3: Heteroscedastic Variance Shifts with Weighted `NDRI_RMSE`

Suppose a sensor upgrade lowers measurement noise by $4\times$ right around a smooth turning point. Without weights, the side with larger noise produces a much larger residual RMSE, causing `NDRI_RMSE` to exceed `ndri_th` and misclassify the smooth bend as a jump candidate. Passing inverse-variance weights `weights = 1.0 / sigma**2` whitens the residuals so `NDRI_RMSE` stays near `0` and STPD recovers the bend cleanly:

```python
rng_het = np.random.default_rng(404)
t_het = np.linspace(2014.0, 2024.0, 140)
sigma_het = np.where(t_het < 2019.0, 0.8, 0.2)  # 4x noise reduction at 2019.0
y_het = (
    5.0 + 0.4 * (t_het - 2014.0) - 1.2 * np.maximum(0.0, t_het - 2019.0)
    + rng_het.normal(0.0, sigma_het)
)

tp_weighted = tspec.detect_turning_points(
    t_het,
    y_het,
    weights=1.0 / (sigma_het**2),
    size=80,
    step=15,
    margin=10,
    alpha=0.01,
    band_method="conditional",
)
print(tp_weighted.summary())
```
