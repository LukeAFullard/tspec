# Interpreting Results, Plots & Warnings

Every result returned by `tspec` (`LssaResult`, `AllssaResult`, `JustJumpResult`, `JustDecomposeResult`, `MonitorResult`, `StpdResult`, `PiecewiseTrendResult`, `PipelineResult`) shares four uniform methods:

- **`.explain()`** — Plain-English narrative written for domain scientists and reports. Start here!
- **`.summary()`** — Compact technical table of indices, timestamps, magnitudes/slopes, $p$-values, NDRI, and red-noise diagnostics.
- **`.to_dataframe()`** — Tidy `pandas.DataFrame` ready for downstream export or batch aggregation.
- **`.plot()`** — Diagnostic Matplotlib figure using colorblind-safe Okabe–Ito colors and distinct markers for censored observations.

---

## Understanding Screening $p$-Values & Repeated Testing

!!! important "Screening vs. Fixed-Date $p$-Values"
    Inside a sliding window, JUST and STPD search across many candidate split indices and select the index $j^*$ that minimizes the residual sum of squares *before* computing `jump_pvalue` (JUST) or Student-$t$ `pvalue` (STPD). Because $j^*$ was chosen by looking at the data, raw window $p$-values are **screening statistics** — they are optimistically small compared to testing a single pre-specified date!

How `tspec` handles this:
- **`just_jumps()`** (and therefore `analyze_series()`) uses `jump_calibration="montecarlo"` by default: each window's `jump_pvalue` is referred to the exact Monte Carlo null distribution of the *maximum* F statistic over all candidate splits, then Holm-adjusted across candidate locations, so the default `jump_alpha=0.05` bounds the probability of reporting any false jump (about 2–6% in simulation, versus 70–100% for the raw screening p-values). `jump_calibration="bonferroni"` is a faster, more conservative analytical alternative; `"none"` returns raw screening p-values (forced by `parity_mode=True`).
- **`analyze_series()`** additionally applies the 1-df **`step_pvalue`** filter, which compares the best discontinuous jump against the best continuous hinge (using break-adjusted harmonics under `season="allssa"` so strong bends do not distort null-model frequencies), ensuring smooth bends are handed cleanly to STPD.
- **`stpd()`** uses `tp_calibration="montecarlo"` by default: each window's slope-change t statistic is referred to the Monte Carlo null of the *maximum* t over all candidate knots and both scan directions (straight line plus Gaussian noise), combined within `tp_th` clusters via Simes and Holm-adjusted across clusters. On straight-line series the probability of reporting any turning point at `alpha=0.01` was 0.0–0.6% in simulation. Robust fits use the analytical `"bonferroni"` correction; `"none"` (forced by `parity_mode=True`) returns raw screening p-values.

---

## Reading Diagnostic Warnings

| Warning Code | Severity | What It Means & What To Do |
| :--- | :--- | :--- |
| `DL_SHIFT_COINCIDENCE` | Warning | A jump, turning point, or NRT alarm landed right next to a laboratory detection-limit change. Re-run with `CensoredConfig(recensor_check=True)` and check lab metadata! |
| `DL_ARTIFACT_LIKELY` | Warning | The detected jump vanished once data were recensored to a common detection limit — this suggests a lab artifact, but recensoring discards information, so a real but weak shift can also vanish; check lab metadata. |
| `SERIAL_CORRELATION_ADJUSTED` | Info | Jump / turning-point / OLS segment-slope tests were calibrated for AR(1) (or OU) errors with the reported `rho` (`serial_correlation="ar1"`, default). Power is lower than for independent errors. |
| `STRONG_AUTOCORRELATION` | Warning | Residual lag-1 or continuous-time OU correlation $\ge 0.60$. Nominal $p$-values are unreliable; tighten `alpha` or supply correlated inverse-covariance `weights = C_f^{-1}`. |
| `MODERATE_AUTOCORRELATION` | Caution | Residual autocorrelation $\in [0.30, 0.60)$. False-positive rates are modestly elevated; consider `alpha=0.005`. |
| `STEP_JUMP_CANDIDATE` | Caution | STPD rejected a significant break as a turning point because a vertical step explains it significantly better than any continuous bend (`jump_test="step"`, default). Run `just_jumps()` (possibly with a longer `size`) or `analyze_series()` to test for an abrupt vertical jump. |
| `HIGH_NDRI_JUMP_CANDIDATE` | Caution | Same, under `jump_test="ndri"` (parity mode): residuals were lop-sided (`NDRI >= ndri_th`). |
| `HORIZON_EXCEEDED` | Warning | `just_monitor()` ran past its planned `horizon` under `pocock`, `obrien_fleming`, or `consecutive_k`; familywise Type I error control is no longer guaranteed past `horizon`. Increase `horizon`, or omit it (for Pocock/O'Brien–Fleming) to use open-ended spending. |
| `OPEN_ENDED_SPENDING` | Caution | No `horizon` was given, so open-ended alpha spending was used (valid for any length, but strict at late steps). Pass a planned `horizon` for more power. |
| `UNCALIBRATED_JUMP_SELECTION` / `UNCALIBRATED_JUMP_PVALUES` | Warning | `just_jumps()` ran with `jump_alpha=None` or `jump_calibration="none"`: reported jumps have no false-positive control. |
| `TIME_UNIT_ASSUMPTION` | Warning | The numeric time axis does not look like decimal years, but default grids, windows and thresholds assume years. |
| `TREND_IMPUTATION_FALLBACK` | Warning | Time-conditional censored imputation failed and the time-invariant marginal model was used; trends among censored values are biased toward zero. |
| `MONITOR_MARGINAL_IMPUTATION` | Info | `just_monitor()` used causal marginal imputation on the historical baseline (overriding `imputation_model="trend"`) to guarantee zero future look-ahead. |
| `INTEGER_STATUS_INTERPRETED_AS_RIGHT` | Caution | Integer `censor_status` values equal to `+1` were interpreted as right-censored (`>UL`), per `tspec` convention (`-1` = left, `0` = detected, `+1` = right). |
| `LONG_MONITORING_HORIZON` | Caution | The monitoring span exceeds the stable historical span, so extrapolated trend uncertainty grows quadratically. Keep `account_for_forecast_uncertainty=True` (default) or re-baseline periodically. |
| `UNADJUSTED_REPEATED_TESTING` | Warning | Either `freq_correction="none"` or `sequential_control="none"` is disabled in `just_monitor()` outside `parity_mode=True`. |
| `MI_DETECTION_FALLBACK` | Caution | With `detection_imputations > 1`, some candidate windows could not be re-imputed (or `scope="windowed"`, 2D weights, or Weibull prevented null-model re-imputation); their p-values use the single primary imputation. |
| `UNCALIBRATED_CONSECUTIVE_K` | Caution | `sequential_control="consecutive_k"` was used without `horizon`, so each step is tested at the full `alpha` and the familywise false-alarm rate grows with monitoring length. Pass a planned `horizon` to calibrate the per-step level. |
| `PENDING_CONFIRMATION` | Info | Under `sequential_control="consecutive_k"`, the latest observation exceeded the threshold but has not yet reached `consecutive_k` consecutive hits. |
| `THRESHOLD_SCALE_MISMATCH` | Caution | Threshold-only selection (`jump_alpha=None`) with `mag_th` looks badly scaled for your data's IQR. Set a domain-meaningful `mag_th` or use the calibrated default `jump_alpha=0.05`. |
| `VARIABLE_SAMPLING_RATE` | Caution | Sampling density shifts $> 2\times$ across the series under `window_mode="count"`. Try `window_mode="duration"`. |
| `NONSTATIONARY_CENSORING_PATTERN` | Caution | Censoring fraction shifts by $> 30\%$ between halves under `scope="global"`. Consider `CensoredConfig(scope="windowed")`. |
| `HEAVY_CENSORING` / `WINDOW_HEAVY_CENSORING` | Warning | Over 50% of the series (or supporting window) is censored. Enable `n_stability_draws=20` (or `stability_mode="bootstrap"`) and treat estimates cautiously. |
| `SPARSE_WINDOW_SKIPPED` | Info | One or more duration windows (or evaluation gaps) held too few observations and were safely skipped. |
| `MODERATE_CENSORING` | Caution | 20–50% of observations are censored. Consider `n_stability_draws=20` to check breakpoint stability. |
| `CENSORED_IMPUTATION_APPLIED` | Info | Some (≤20%) observations were censored and imputed; records the actual engine used (`tspec` Tobit vs `ndimpute`). |
| `CONDITIONAL_MEAN_VARIANCE_SHRINKAGE` | Info | `impute_type="mean"` assigns identical values to tied limits, shrinking residual variance; prefer `"stochastic"`. |
| `EXTREME_IMPUTED_DOWNWEIGHTING` | Caution | `imputed_weight_factor < 0.5` heavily down-weights censored values, which can mimic dropping non-detects. |
| `WINDOW_IMPUTATION_FALLBACK` | Info | Windowed censoring imputation fell back to the global imputation in windows with too few detects. |
| `MI_SEGMENT_FALLBACK` | Warning | Multiple imputation failed for a censored segment, so its OLS CI ignores imputation uncertainty. |
| `ALLSSA_MAX_ITER` | Caution | ALLSSA hit `max_iter` before the frequency set stabilized; increase `max_iter` or use a stricter `alpha`. |
| `DEGENERATE_DEFAULT_GRID` | Warning | The default ALLSSA frequency grid has ≤ 1 frequency, usually because time is not in decimal years; pass `omega`. |
| `CALIBRATION_FALLBACK` | Info / Warning | Monte Carlo jump calibration is not available for robust fits, so Bonferroni was used (info); or `consecutive_k` could not be calibrated because a full weight matrix was supplied or the history spacing is degenerate, so each step uses the full `alpha` (warning). |
| `BIAS_CORRECTION_FALLBACK` | Info | Some decomposition windows were too small for the bias-corrected trend band and used the model-based band. |
| `RANK_DEFICIENT_DESIGN` | Warning | The design matrix was numerically singular (duplicate times, very narrow jump segments, or collinear frequencies); an SVD pseudoinverse was used and individual coefficients are not identifiable. |
| `HIGH_CONDITION_NUMBER` | Caution | The design is ill-conditioned (cond > 1e6); fitted values are accurate, but individual coefficients and amplitudes are unstable. |
| `MARGIN_CLAMPED` | Info | The requested STPD `margin` was raised to the minimum safe value of 2. |
| `NDRI_L2_DENSITY_IMBALANCE` | Caution | With `ndri_mode="l2"`, uneven sample counts left/right of a turning point bias NDRI upward; use `ndri_mode="rmse"`. |
| `RAW_SNR_LARGE_BASELINE` | Caution | `snr_mode="raw"` on a series with a large non-zero mean inflates SNR; use `snr_mode="centered"`. |
| `SHORT_SEASONAL_WINDOW` | Caution | JUST windows cover fewer than two cycles of the lowest searched frequency, so real harmonics may not pass the family-wise selection rule and seasonality can leak into trends and jumps. Use longer windows, `season="ols"` with known frequencies, or `season_significance="per_frequency"`. |
| `ROBUST_NOT_CONVERGED` | Caution | A robust (Huber/MM) IRLS fit hit its iteration limit; treat robust estimates and CIs as approximate. |
| `L1_SOLVER_FALLBACK` | Warning | HiGHS LP solver did not converge cleanly for `mode="l1"` and fell back to a reweighted IRLS approximation. |
| `TWO_D_WEIGHTS_DROPPED_FOR_STPD` | Caution | 2D inverse-covariance weights were applied in JUST stages, while STPD and per-segment fits (1D weights only) ran unweighted. |
| `STABILITY_DRAW_FALLBACK` | Caution | Some censoring stability draws could not refit the imputation model and reused fixed parameters, so stability rates understate uncertainty. |
| `SEN_CI_UNAVAILABLE` | Caution | A Theil–Sen segment has more than 2000 observations; the slope uses sampled pairs and no CI or p-value is reported. |
| `ATS_UNAVAILABLE` | Caution | A censored segment could not use Akritas–Theil–Sen (interval censoring or more than 2000 observations); Theil–Sen on imputed values was used. |
| `STPD_SEGMENT_SKIPPED` | Info | With `jump_adjustment="segment"`, an epoch was too short for turning-point detection and got a single linear trend. |
| `DECOMPOSE_SKIPPED` / `STPD_SKIPPED` | Caution | `analyze_series()` skipped a stage because the series was too short for its windows; the message gives the reason. |

---

## Confidence & Prediction Intervals (with Examples)

Every trend, seasonal, and forecast estimator in `tspec` computes pointwise confidence intervals controlled by `ci_level` (default `0.95`) — using detect-then-fit wild bootstrap by default on `stpd()` and `analyze_series()` (`band_method="bootstrap"`) and fast analytical intervals on `just_decompose()`, `fit_piecewise_trend()`, and `just_monitor()` — and renders them automatically as semi-transparent shaded ribbons in `.plot()` (toggle with `show_ci=False`):

![Pointwise confidence and prediction intervals across JUST decomposition, robust Theil–Sen pipeline, and near-real-time monitoring](../assets/figures/example_confidence_intervals.png)

| Function / Mode | Output Fields | Interval Formula |
| :--- | :--- | :--- |
| `just_decompose()` | `trend_ci_low/high` (centered on `trend_bc`), `seasonal_ci_low/high` (centered on `seasonal_bc` for ALLSSA), plus `*_reg_ci_*` for `eval_times` | Trend: **bias-corrected** band from local-quadratic window refits (`trend_ci_method="bias_corrected"`). Seasonal: post-selection bias-corrected band for ALLSSA (exact within-window Student-$t$ for OLS). Blended across overlapping Gaussian windows with the Cauchy–Schwarz half-width blend. |
| `fit_piecewise_trend()` & `detect_turning_points()` | `trend_ci_low/high` on `PiecewiseTrendResult` | `detect_turning_points()` defaults to `band_method="bootstrap"` (including `refit_mode="l1"` via unpinned wild bootstrap). Analytical `band_method="conditional"` / `fit_piecewise_trend()`: exact $O(n)$ recursion for `"sequential"`, exact $\mathbf{d}_i^T \widehat{\text{Cov}}\,\mathbf{d}_i$ for `"joint"` / `"huber"`, `None` for analytical `"l1"`. |
| `analyze_series()` | `trend_ci_low/high`, `decomposition.seasonal_ci_low/high`, per-segment `slope_ci_low/high`, and `slope_fan_low/high` for `"sen"` | Default `band_method="bootstrap"` runs detect-then-fit wild bootstrap across offset, segmented, jump-only, and seasonal-only series (with robust weights when `robust != "none"`). Censored OLS segments use Rubin-pooled Tobit MI (`estimator="ols_mi"`, normal or lognormal + HC3). For `"sen"` the median-pivoted slope fan is exported as `slope_fan_*` (not a pointwise trend CI). |
| `just_monitor()` | `forecast_ci_low/high` **and** `forecast_pi_low/high` | Inner band: mean forecast uncertainty (widens quadratically past `start`). Outer band: out-of-sample observation **prediction interval** adding $\hat{\sigma}_{\text{his}}^2 / w_i$. |

!!! note "Statistical Interpretation & Caveats"
    - **Post-detection bands:** By default (`band_method="bootstrap"`), `stpd()` and `analyze_series()` trend bands, and uncensored OLS segment-slope CIs, come from a detect-then-fit bootstrap. Each replicate re-locates the turning points with STPD's own estimator and, in the pipeline, re-runs the seasonal and jump-step estimators (across both `"offset"` and `"segment"` modes, as well as jump-only and seasonal-only series). After correct detection, the STPD and pipeline bands covered ~95–98% at nominal 95% in simulation (analytical conditional bands: 83–92%). All bands remain conditional on the *number* of breakpoints found, and a missed turning point still leaves the band wrong near it. `band_method="conditional"` returns the faster analytical bands. For heavily censored records, pair with `CensoredConfig(n_stability_draws=20)` to assess breakpoint stability.
    - **Autocorrelation:** Standard intervals assume independent residuals unless you supply an inverse-covariance weight matrix `weights = C_f^{-1}`. If `tspec` emits `MODERATE_AUTOCORRELATION` or `STRONG_AUTOCORRELATION`, analytical bands are optimistic (too narrow).
    - **Theil–Sen bowtie fan (`slope_estimator="sen"`):** The plotted median-pivoted fan (`slope_fan_low/high`) shows **slope uncertainty** (`seg.slope_ci_low/high`) around each segment's median point $(t_{\text{med}}, \hat{y}_{\text{med}})$. It pinches to zero width at $t_{\text{med}}$, so it is not a confidence band for the trend line.
    - **Why analytical `fit_piecewise_trend(..., mode="l1")` returns `None`:** Exact linear-programming $L_1$ (LAD) regression has no closed-form finite-sample covariance without resampling. `detect_turning_points(..., refit_mode="l1", band_method="bootstrap")` provides bootstrap trend bands; use `slope_estimator="sen"` or `"huber"` when you want per-segment robust CIs.

### Example 1: JUST Decomposition Trend & Seasonal 95% CIs

```python
import numpy as np
import tspec

rng = np.random.default_rng(42)
t = np.sort(rng.uniform(2016.0, 2023.0, size=170))
y = 8.0 + 0.45 * (t - 2016.0) + 1.1 * np.cos(2 * np.pi * t) + rng.normal(0.0, 0.35, t.size)

dec = tspec.just_decompose(t, y, season="ols", ci_level=0.95)
df = dec.to_dataframe()  # includes trend_ci_low/high & seasonal_ci_low/high
assert {"trend_ci_low", "trend_ci_high", "seasonal_ci_low", "seasonal_ci_high"} <= set(df.columns)

fig, axes = dec.plot(show_ci=True)  # shaded 95% ribbons on trend & seasonal panels
```

### Example 2: Robust Pipeline with Theil–Sen Slopes and Slope Fan (`robust="mm"`, `slope_estimator="sen"`)

```python
t_b = np.sort(rng.uniform(2015.0, 2024.0, size=190))
tr_b = np.where(t_b < 2019.2, 5.0 + 0.2 * (t_b - 2015.0), 8.2 - 0.55 * (t_b - 2019.2))
y_b = tr_b + 0.65 * np.cos(2 * np.pi * t_b) + rng.normal(0.0, 0.28, t_b.size)
y_b[[22, 65, 130, 165]] += np.array([5.5, -4.8, 5.2, -5.0])  # isolated spikes

pipe = tspec.analyze_series(
    t_b,
    y_b,
    robust="mm",
    slope_estimator="sen",
    ci_level=0.95,
    just_kwargs={"season": "ols", "jump_alpha": 0.01},
    decompose_kwargs={"season": "ols"},
)
for seg in pipe.segments:
    print(f"Seg {seg.segment_index}: slope={seg.slope:+.3f} 95% CI=[{seg.slope_ci_low:+.3f}, {seg.slope_ci_high:+.3f}]")

fig, axes = pipe.plot(show_ci=True)  # plots the Sen slope fan & seasonal 95% CI
```

### Example 3: Near-Real-Time Monitoring Forecast CI vs. Prediction Interval

```python
t_c = np.linspace(2017.0, 2024.0, 150)
y_c = 2.0 + 0.12 * (t_c - 2017.0) + 0.45 * np.cos(2 * np.pi * t_c) + rng.normal(0.0, 0.14, t_c.size)
y_c = np.where(t_c >= 2023.15, y_c - 0.85, y_c)

mon = tspec.just_monitor(t_c, y_c, start=2021.8, season="ols", ci_level=0.95, stop_on_first=False)
assert np.all(mon.forecast_pi_low < mon.forecast_ci_low)
assert np.all(mon.forecast_ci_high < mon.forecast_pi_high)

fig, axes = mon.plot(show_ci=True)  # inner forecast CI cone + outer observation PI
```

---

## Visual Legend for `.plot()`

- **Solid dark dots (`●`):** Measured (uncensored) observations.
- **Hollow downward orange triangles (`▽`):** Imputed left-censored observations (`<DL`).
- **Hollow upward purple triangles (`△`):** Imputed right-censored observations (`>UL`).
- **Hollow squares with whiskers (`□`):** Imputed interval-censored observations (`[L, R]`).
- **Semi-transparent shaded ribbons:** Pointwise confidence intervals (`ci_level=0.95` by default; toggle with `show_ci=False`). On `plot_monitor()`, the darker inner ribbon is the mean forecast CI and the lighter outer ribbon is the observation prediction interval.
- **Vertical dashed orange lines:** JUST abrupt jumps (with `Δ` magnitude badge and `(DL shift!)` flag if coinciding).
- **Vertical dotted gray lines:** Detected laboratory detection-limit shifts.
- **Green diamonds (`◆`):** STPD continuous turning points (with `Δslope` and `NDRI` badges).
