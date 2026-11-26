# Known Limitations

`tspec` has been audited and its default procedures were re-calibrated by simulation (see `audit/AUDIT_REPORT.md`, `audit/REAUDIT_REPORT.md` and `audit/FIXES.md` in the repository). The defaults now control their stated error rates in the designs tested. The limitations below remain. Read this page before relying on `tspec` results for publication-critical conclusions.

All numbers come from Monte Carlo simulations with synthetic data, mostly with Gaussian noise. Real data can behave differently.

## 1. Confidence intervals after change detection can be slightly too narrow

When `stpd()` or `analyze_series()` first *detect* turning points and then fit a trend, the uncertainty of the detection carries into the trend. By default (`band_method="bootstrap"`) the bands include the uncertainty in *where* each turning point lies, and in `analyze_series()` also the uncertainty in the seasonal and jump estimates. Some shortfall remains:

| Interval | Coverage at nominal 95%, after correct detection |
|---|---|
| `stpd()` trend band | about 95–97% |
| `analyze_series()` trend band | **about 93%** (slightly too narrow) |
| `analyze_series()` OLS segment-slope CIs | **about 93%** (slightly too narrow) |
| Any of the above with `band_method="conditional"` | about 83–95% (too narrow, most so near turning points) |

- All of these intervals are **conditional on the number of turning points found**. If a real turning point is missed, the trend and its band are wrong near it, and coverage there can be very low.
- Treat a band edge that only just excludes a value of interest as weak evidence.

## 2. Turning points can be missed even in long records

STPD tests each sliding window separately. Its power depends on how much information one window contains (by default about five seasonal cycles), not on the length of the whole record. For example, a change in slope from +0.5 to −0.5 per year with noise σ = 1.5 was detected in only about 6% of simulated series of 500 observations. A non-detection is not evidence that the trend is straight.

## 3. Trend tests assume a simple form of serial correlation

- The Mann–Kendall / Sen correction (`mk_correction="ar1"`, default) assumes first-order autoregressive (AR(1)) residuals. Under AR(1) with ρ ≤ 0.7, the false-positive rate at α = 0.05 is 3.8–6.2% for 60–250 observations and about 5% for 400. Other dependence structures (long memory, seasonal correlation, regime changes) are not covered and may give more false trends.
- `mk_correction="hamed_rao"` was liberal (7–10%) even for independent data in simulation. `"none"` is strongly anti-conservative for autocorrelated data (about 25% at ρ = 0.5).
- The OLS trend tests, jump tests and spectral tests assume independent residuals unless you supply a full inverse-covariance weight matrix. Check the red-noise diagnostics in `.explain()`.

## 4. Censored (`<DL`) data

- By default, jump and turning-point p-values are computed on a single random imputation of the censored values. The false-positive rate stayed near or below nominal in simulation. The exception was a borderline 6.7% at α = 0.05 for seasonal series with 50% censoring. But a borderline decision can change with `random_state`, for 2–27% of decisions at 20–50% censoring.
- `CensoredConfig(detection_imputations=20)` pools several imputations and removes most of that seed dependence. It is conservative and costs power under heavy censoring: at 50% censoring a clear step was detected in 66% rather than 83% of series (32% rather than 69% for seasonal series).
- `scope="windowed"` and `"seasonal_stratified"` use a time-invariant imputation model, which can flatten trends among censored values. Interval censoring has not been validated by simulation.

## 5. Near-real-time monitoring trades speed for safety

- The default `just_monitor()` control (Šidák + Pocock spending) is very conservative: about 0.2–0.6% false alarms at α = 0.05. Real disturbances are therefore detected later than necessary.
- `sequential_control="consecutive_k"` with a planned `horizon` is calibrated (3.3–5.3% at α = 0.05) and detects faster. Its calibration assumes Gaussian errors, future sampling similar to the history, and fixed seasonal harmonics. Without a `horizon` it does **not** control the false-alarm rate.

## 6. Monte Carlo calibrations assume Gaussian errors

The calibrated jump test, ALLSSA frequency selection, STPD turning-point test and `consecutive_k` monitoring simulate their null distributions under Gaussian errors. With heavy-tailed errors or outliers, use the robust options (`robust="huber"` or `"mm"`). These switch to conservative Bonferroni-type corrections. Note that `robust="mm"` and `mode="l1"` are **not** high-breakdown estimators: they can break down below 35% contamination.

## 7. Other assumptions

- Default frequency grids and windows assume time in **decimal years**. Supply `omega` and window settings explicitly for other units (watch for `TIME_UNIT_ASSUMPTION`).
- Robust (IRLS) fits that fail to converge inside individual sliding windows are not reported per window. This was rare in simulation (≤ 0.2%).
- The bootstrap bands add about 0.2–0.6 s per call, or about 6 s with `refit_mode="l1"`. Use `band_method="conditional"` for quick exploratory runs.

## 8. Validation status

- The audit, the fixes and the re-audit were performed by the same automated agent, so this is not an independent review.
- The repository's `validation/` tier predates the audit and is labelled historical.
- The high-replicate calibration studies live in `audit/`, not in the continuous-integration test suite.
- Independent verification of the key studies (jump false positives, post-detection bands, Mann–Kendall under autocorrelation) is recommended before high-stakes use.
