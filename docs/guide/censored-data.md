# Working with Censored Lab Data (`<DL` / `>UL`)

Environmental and analytical time series frequently contain **censored values** — measurements below a laboratory Limit of Detection (e.g., `"<0.05"` mg/L), above an upper reporting limit (`">100"`), or known only to lie within an interval (`[0.2, 0.5]`).

`tspec` integrates [`ndimpute`](https://github.com/LukeAFullard/ndimpute) directly into `just_jumps()`, `just_decompose()`, `just_monitor()`, `detect_turning_points()`, and `analyze_series()` while adding time-series-specific safeguards.

![Censored water-quality change detection with ndimpute ROS and DL-shift defense](../assets/figures/tutorial_water_quality_censored.png)

---

## Why Replacing `<DL` with `DL/2` Fails

Substituting half the detection limit (`DL/2`) is still common in spreadsheets, yet it causes three severe distortions (Helsel, 2012):

1. **Destroys variance:** Assigning the exact same constant to every non-detect makes the data look artificially smooth, shrinking standard errors and producing falsely tiny $p$-values.
2. **Biases trends & jump magnitudes:** Especially on right-skewed (lognormal) water-quality data, `DL/2` shifts the lower tail upward or downward relative to the true conditional expectation.
3. **Turns lab upgrades into fake jumps:** When a lab upgrades from `<0.10` to `<0.02`, `DL/2` drops mechanically from `0.05` to `0.01` even if river concentrations never budged!

---

## Supported Input Formats

You can pass censored data in any of three idiomatic forms:

```python
import numpy as np
import pandas as pd
from tspec import CensoredConfig, analyze_series

rng = np.random.default_rng(0)
dates = pd.date_range("2012-01-01", periods=144, freq="MS")
conc = np.exp(0.3 * np.cos(2 * np.pi * np.arange(144) / 12) + rng.normal(0.0, 0.3, 144))

# 1. Mixed strings & floats (auto-detected!)
y_strings = [f"<{0.8}" if v < 0.8 else float(v) for v in conc]
res = analyze_series(dates, y_strings)

# 2. Numeric array + boolean/signed-integer/LWPTrends status array
#    status: boolean True = left-censored (default), or signed -1=left, 0=obs, +1=right, or 'lt'/'gt'/'not'
is_left_censored = conc < 0.8
values = np.where(is_left_censored, 0.8, conc)
res = analyze_series(dates, values, censor_status=is_left_censored)

# 3. Interval bounds: (N, 2) array of [lower, upper]
bounds_nx2 = np.column_stack([np.where(is_left_censored, 0.0, conc), np.where(is_left_censored, 0.8, conc)])
res = analyze_series(
    dates,
    bounds_nx2,
    censoring=CensoredConfig(method="parametric", censoring_type="interval"),
)
print(res.segments_to_dataframe())
```

---

## Six Built-In Statistical Safeguards

### 1. Eliminating the ROS Quantile Temporal-Trend Artifact
In standard Robust ROS (`NADA::ros` and `ndimpute`'s default `impute_type="quantile"`), plotting positions are assigned using a **stable sort** on values. On a chronologically ordered time series, all non-detects tied at a detection limit $L$ are ranked $1, 2, \dots, C$ **from earliest timestamp to latest timestamp** — deterministically injecting a fake upward trend!

`tspec` protects you automatically:
- Defaults to **`impute_type="stochastic"`** (seeded with `random_state=0`), which preserves residual spread and has zero temporal rank correlation.
- If you explicitly request `impute_type="quantile"`, `tspec` **randomly permutes tied detection-limit groups** before calling `ndimpute` and maps them back afterward, preserving exact NADA marginal quantiles with zero artificial slope.

### 2. Time-Conditional Imputation (default) and Multiple Imputation
By default (`CensoredConfig(imputation_model="trend", scope="global")`), censored values are imputed from a **censored-regression (Tobit-type) MLE on time**: a linear trend plus annual and semi-annual harmonics for datetime input (`trend_harmonics`), with lognormal or normal errors (`dist`). Each censored value is drawn from its *conditional* truncated distribution given its timestamp, so imputations keep the series' trend. The legacy `imputation_model="marginal"` draws from one time-invariant `ndimpute` distribution instead; this erases the trend among censored values and biased OLS slopes by about −27%, with 95% CIs covering only 63% at 50% censoring in simulation.

A single imputation still understates uncertainty. For `analyze_series(..., slope_estimator="ols")`, segments that contain censored values are therefore re-estimated from `n_imputations` (default 20) proper multiple imputations, under a segment-congenial censored regression, pooled with **Rubin's rules** (segment `estimator="ols_mi"`). `slope_estimator="sen"` uses the Akritas–Theil–Sen estimator directly on the censored ranks.

**Detection p-values (`just_jumps`, `stpd`).** By default, candidate p-values are computed on the single time-conditional imputation. In simulation this kept valid size: on jump-free and straight-line series with 20% and 50% censoring, false jumps occurred in 2–3% of series at `jump_alpha=0.05` (borderline 6.7% for seasonal series at 50% censoring) and false turning points in 1% at `alpha=0.01`. The drawback is Monte Carlo noise: a different `random_state` changed 2–13% of jump decisions at 20–50% censoring.

`CensoredConfig(detection_imputations=20)` removes most of that noise. Each candidate window is re-imputed `detection_imputations` times from a censored regression of its null model (trend plus harmonics for JUST, a straight line for STPD; harmonics are dropped when too few values are detected). Each imputation is tested at its own best split or knot with the calibrated test, and the p-values are pooled with the Licht–Rubin rule (z-transform plus Rubin's rules) before the Holm adjustment. This is conservative: false detections were 1–3% (JUST) and 0–1% (STPD), and seed changes altered 1–9% of jump decisions. Under heavy censoring it costs substantial jump power, though. At 50% censoring a 2.5σ step was detected 66% instead of 83% of the time (32% instead of 69% for seasonal series). There was no loss at 20% censoring or for STPD. Prefer it at moderate censoring, or when decisions must not depend on the seed. Imputing with the candidate jump in the model is *not* offered. The jump location is chosen from the data, so building it into every imputation inflated false jumps to 10–15%.

Use `n_stability_draws` to report how often each breakpoint recurs across imputations.

### 3. Imputation Scopes for Non-Stationary Series
Because `ndimpute` fits a single marginal distribution, a huge regime jump (e.g., 60% `<DL` before a factory opens, 5% `<DL` afterward) can distort global imputations. Choose from three explicit scopes:

- `scope="global"` *(default)*: Lowest variance; best when censoring is $<30\%$ and regimes are comparable. Automatically emits `NONSTATIONARY_CENSORING_PATTERN` if censoring differs by $>30\%$ between halves of the record.
- `scope="windowed"`: Re-fits `ndimpute` locally inside each sliding JUST/STPD window (falling back to global when a window has $<5$ detects or $>80\%$ censoring).
- `scope="seasonal_stratified"`: Imputes separately by calendar quarter so summer baseflow non-detects aren't distorted by winter stormflow peaks.

### 4. Laboratory Detection-Limit Shift Defense
With `detect_dl_shifts=True` (default), `tspec` scans for both explicit step changes in active censoring limits AND implicit DL upgrades (where censoring at $L_{\text{old}}$ stops and uncensored detections appear below $L_{\text{old}}$):

- Any jump or turning point near a detected DL shift is marked `coincides_with_dl_shift = True` with a prominent `DL_SHIFT_COINCIDENCE` warning in `.explain()` and a dotted vertical marker in `.plot()`.
- Set **`recensor_check=True`** to run Helsel's common-limit recensoring verification: temporarily recensors everything below $L_{\max}$ in the jump window and re-tests whether the jump survives. If it disappears, `dl_artifact_likely = True` is flagged explicitly.

### 5. Breakpoint Stability Across Imputation Uncertainty
Single imputations condition on fitted parameters $\hat\theta$ (see `ndimpute`'s statistical documentation). To measure whether a detected jump or turning point is robust rather than an artifact of one random tail draw:

- `analyze_series()` automatically runs **`n_stability_draws=20`** stochastic draws and reports `stability_rate` (e.g., *"Stable in 19/20 draws"*).
- For full parameter-uncertainty propagation (proper resampling of $\hat\theta$ as well as tail draws), pass **`CensoredConfig(n_stability_draws=50, stability_mode="bootstrap")`**.

### 6. Tiered Censoring Guardrails
- **0–20% censored:** Informational provenance note.
- **20–50% censored:** `MODERATE_CENSORING` caution recommending `n_stability_draws >= 20`.
- **50–80% censored:** `HEAVY_CENSORING` warning (results depend strongly on distributional shape).
- **>80% censored or <5 detects:** Blocked with `CensoringError` by default; pass `allow_heavy_censoring=True` only for exploratory inspection.
- Every sliding window's local censoring fraction is recorded (`window_censored_fraction`), flagging any breakpoint backed by a >50% censored window.
