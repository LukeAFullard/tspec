# Known Limitations

`tspec` has been audited and its default procedures were re-calibrated by simulation (see `audit/AUDIT_REPORT.md`, `audit/REAUDIT_REPORT.md` and `audit/FIXES.md` in the repository). The defaults now control their stated error rates in the designs tested. The limitations below remain. Read this page before relying on `tspec` results for publication-critical conclusions.

All numbers come from Monte Carlo simulations with synthetic data, mostly with Gaussian noise. Real data can behave differently.

## 1. Confidence intervals after change detection can be slightly too narrow

When `stpd()` or `analyze_series()` first *detect* turning points and then fit a trend, the uncertainty of the detection carries into the trend. By default (`band_method="bootstrap"`) the bands include the uncertainty in *where* each turning point lies, and in `analyze_series()` also the uncertainty in the seasonal and jump estimates. Some shortfall remains:

| Interval | Coverage at nominal 95%, after correct detection |
|---|---|
| `stpd()` trend band (`sequential`/`joint`/`huber`) | about 94–97% (`refit_mode="l1"`: ~93%) |
| `analyze_series()` trend band (`offset`, `segment`, jump-only, seasonal-only) | about 95–98% |
| `analyze_series()` OLS segment-slope CIs | about 93–96% |
| Any of the above with `band_method="conditional"` | about 83–95% (too narrow, most so near turning points) |

- All of these intervals are **conditional on the number of turning points found**. If a real turning point is missed, the trend and its band are wrong near it, and coverage there can be very low.
- Under strong unweighted heteroscedasticity (e.g. a 4× variance shift without passing inverse-variance `weights`), `stpd()` may reject smooth bends via `NDRI_RMSE`; supply `weights = 1 / sigma**2` whenever variance shifts are known.
- Treat a band edge that only just excludes a value of interest as weak evidence.

## 2. Turning points can be missed even in long records

STPD tests each sliding window separately. Its power depends on how much information one window contains (by default about five seasonal cycles), not on the length of the whole record. For example, a change in slope from +0.5 to −0.5 per year with noise σ = 1.5 was detected in only about 6% of simulated series of 500 observations. A non-detection is not evidence that the trend is straight.

## 3. Trend tests assume a simple form of serial correlation

- The Mann–Kendall / Sen / Akritas–Theil–Sen correction (`mk_correction="ar1"`, default) assumes first-order autoregressive (AR(1)) residuals and uses a rank-based AR(1) bootstrap whenever $n \le 1000$ and $n_{\text{eff}} < 150$. Under AR(1) up to $\rho = 0.9$ (including isolated spikes and left/right censoring), the Type-I error stays near 3–7% at $\alpha = 0.05$. Other dependence structures (long memory, seasonal correlation, regime changes) are not covered and may give more false trends.
- `mk_correction="hamed_rao"` was liberal (7–10%) even for independent data in simulation. `"none"` is strongly anti-conservative for autocorrelated data (about 25% at $\rho = 0.5$).
- The OLS trend tests, jump tests and spectral tests assume independent residuals unless you supply a full inverse-covariance weight matrix. Check the red-noise diagnostics in `.explain()`.

## 4. Censored (`<DL`) data

- By default, jump and turning-point p-values are computed on a single random imputation of the censored values (`imputation_model="trend"` Tobit MLE). The false-positive rate stayed near or below nominal in simulation, though borderline decisions can vary with `random_state` (roughly 2–27% across 20–50% censoring depending on signal strength).
- `CensoredConfig(detection_imputations=20)` pools several imputations and removes most of that seed dependence. It is conservative and costs power under heavy censoring: at 50% censoring a clear step was detected in 66% rather than 83% of series (32% rather than 69% for seasonal series).
- `scope="windowed"` and `"seasonal_stratified"` use a time-invariant marginal model (`ndimpute`), which can flatten trends among censored values. Interval censoring has not been extensively validated by Monte Carlo simulation.

## 5. Near-real-time monitoring trades speed for safety

- The default `just_monitor()` control (Šidák + Pocock spending) is very conservative: about 0.2–0.6% false alarms at $\alpha = 0.05$. Real disturbances are therefore detected later than necessary.
- `sequential_control="consecutive_k"` with a planned `horizon` is calibrated (~4–6% at $\alpha = 0.05$, including with `season="allssa"` via Gauss–Newton frequency-tangent covariance) and detects faster. Its calibration assumes Gaussian errors and future sampling similar to the history. Without a `horizon` it does **not** control the false-alarm rate.

## 6. Monte Carlo calibrations assume Gaussian errors

The calibrated jump test, ALLSSA frequency selection, STPD turning-point test, and `consecutive_k` monitoring simulate their null distributions under Gaussian errors. With heavy-tailed errors or outliers, use the robust options (`robust="huber"` or `"mm"`). These switch to conservative Bonferroni-type corrections in JUST/STPD (note that `lssa`/`allssa` Monte Carlo draws remain Gaussian-null). Note that `robust="mm"` and `mode="l1"` are **not** high-breakdown estimators: they can break down below 35% contamination.

## 7. Other assumptions & runtime

- Default frequency grids and windows assume time in **decimal years**. Supply `omega` and window settings explicitly for other units (watch for `TIME_UNIT_ASSUMPTION`).
- Robust (`robust="mm"`) IRLS fits that hit their iteration limit inside individual sliding windows are not reported per window; under heavy tails ($t_3$) or strong outliers, non-convergence can occur in roughly 2–6% of fits (rare under Gaussian errors).
- **Single-threaded runtime (`OMP_NUM_THREADS=1`, $n = 200$):** Default Monte Carlo + bootstrap calibration (`n_mc=999`, `n_boot=499`) takes approximately 5 s for `just_jumps()`, 5–6 s for `stpd()` (~25–30 s with `refit_mode="l1"`), and ~11–12 s for `analyze_series()`. Pass `band_method="conditional"` (and/or `jump_calibration="bonferroni"`, `season="ols"`) for fast exploratory or large-batch runs.

## 8. Validation status

- All four validation tiers under `validation/reports/` — Tier 1 upstream parity (41/41), refreshed Tier 2 Monte Carlo benchmarks (26/26), Tier 2b literature & compositional stress tests (9/9), Tier 2c real-world public-domain scientific/economic datasets (6/6), and Tier 2d external oracles (`statsmodels`/`scipy`), cross-domain signals & adversarial edge cases (15/15) — have been executed and verified against the current hardened codebase.
- High-replicate Monte Carlo calibration sweeps live in `audit/`, while 182 unit, property-based, calibration-regression, and executable-documentation tests run on every push in GitHub Actions CI.
- Independent domain verification of key assumptions (serial correlation, censoring distribution, seasonal structure) is still recommended before publication-critical use.
