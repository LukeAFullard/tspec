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
- The default bootstrap STPD band held 95–96% coverage after correct detection with uneven sampling density and with weighted heteroscedastic noise (independent audit, 2026-11); the `band_method="conditional"` band was 84–90% (69–80% near the turning point) in the same designs.
- Under strong **unweighted** heteroscedasticity (a 4× noise-SD shift without inverse-variance `weights`), STPD recovered exactly the true turning point in only 16% of series (61% with `weights = 1 / sigma**2`). Supply weights whenever variance shifts are known.
- On the README example, the `analyze_series` trend band covered about 89–92% (not 95%) because of the seasonal-adjustment and jump-offset steps that precede STPD; treat it as a ~90% band.
- Treat a band edge that only just excludes a value of interest as weak evidence.

## 2. Turning points can be missed even in long records

STPD tests each sliding window separately. Its power depends on how much information one window contains (by default about five seasonal cycles), not on the length of the whole record. For example, a change in slope from +0.5 to −0.5 per year with noise σ = 1.5 was detected in only about 6% of simulated series of 500 observations. A non-detection is not evidence that the trend is straight.

## 2b. JUST jumps: what they are and how small a step can be found

- **Definition.** Since 2026-11, `just_jumps` (`jump_definition="step"`, default) and `analyze_series` report only abrupt vertical steps: a significant level-and-slope break that a continuous bend plus gentle curvature cannot explain, searched over window scales 1, 2 and 4. `jump_definition="level_or_slope"` (and `parity_mode=True`) gives the published JUST definition, which also reports slope changes and, with slowly curving trends, curvature: a 9-year cycle of amplitude 3σ gave false "jumps" in 37% of series. In that mode the warning `NOT_A_VERTICAL_STEP` flags these.
- **Power.** Single 3-cycle windows find only large steps: in 240-point records with σ = 0.5, a 2σ step was found in about 6% and a 4σ step in about 50% of series (published definition). The step definition with scales (1, 2, 4) finds substantially smaller steps (see the CHANGELOG for the validated rates) at about 3× the runtime. Small steps on strongly curved trends, or in short or heavily censored records, remain hard to detect, because a continuous hinge with a steep segment can imitate a modest step within a short window.
- **Aliases.** In a 3-cycle window a step one seasonal period away from the split is nearly indistinguishable from "ramp + seasonal sawtooth". Single-scale JUST can therefore report an extra jump about one cycle from a real one (in one design, 7 extra jumps in 60 series). Multi-scale windows and the conditional re-test reduce this to about 1 in 60.

## 3. Tests assume a simple form of serial correlation

- **Change-point and OLS trend tests (`serial_correlation="ar1"`, default since 2026-11).** `just_jumps`, `stpd` and `analyze_series` estimate a lag-1 autocorrelation of the residuals from a smooth global fit. The estimator is difference-based, so genuine jumps hardly affect it. The tests are then calibrated under AR(1) errors:
  - Monte Carlo nulls are simulated with AR(1) noise;
  - analytical F/t references are deflated by $(1+\rho)/(1-\rho)$;
  - OLS segment slopes and the trend band of `analyze_series` use an exact sandwich variance.
  - The calibration uses $\hat\rho$ plus one standard error ($2/\sqrt{n}$) once $\hat\rho$ exceeds twice its null standard error, because under-estimating $\rho$ is the costlier error; below that the errors are treated as independent.
  - Before this, the family-wise false-jump rate of `just_jumps` at a nominal 5% was 34% (AR(1) $\rho = 0.3$) and 95% ($\rho = 0.6$); STPD gave false turning points in 27.5% of straight-line series at $\rho = 0.5$; the OLS segment-slope test rejected 24–49% of trend-free series at $\rho = 0.5$–$0.8$. See the CHANGELOG for the rates after the change.
  - **Cost:** detection power drops when the noise is red, because correlated data carry less information (e.g. a 4σ step at $\rho = 0.5$ was found in 10% instead of 26% of series). Power on white noise is essentially unchanged.
  - **Misfit can look like correlation:** on very precise records (small noise) with sharp structure the smooth global fit cannot follow (a bend between its knots, gaps), the estimate can be inflated by model misfit. For example, one test record with σ = 0.15 gave $\hat\rho \approx 0.2$ on independent noise and lost a 16σ jump at `jump_alpha=0.01`. Compare `serial_rho` with the residual diagnostics, and use `serial_correlation="none"` only when you have independent evidence that the errors are uncorrelated. On the README example (σ = 0.16) the jump and turning point were still found in 100% of 100 simulated records.
  - **Not covered:** long memory, seasonal or regime-dependent correlation, and AR orders above one. `serial_correlation="ou"` uses a continuous-time (Ornstein–Uhlenbeck) model for strongly irregular sampling. `"none"` restores independent-error tests.
  - `just_decompose` bands are widened by the same model (trend: long-run factor; seasonal: AR(1) spectral factor). The seasonal band still covered only about 91% at $\rho = 0.8$.
  - **Not yet adjusted:** `just_monitor` and `lssa`/`allssa` p-values still assume independent errors unless you supply a full inverse-covariance `weights` matrix.

- The Mann–Kendall / Sen / Akritas–Theil–Sen correction (`mk_correction="ar1"`, default) assumes first-order autoregressive (AR(1)) residuals and uses a rank-based AR(1) bootstrap whenever $n \le 1000$ and $n_{\text{eff}} < 150$. Under AR(1) up to $\rho = 0.9$ (including isolated spikes and left/right censoring), the Type-I error stays near 3–7% at $\alpha = 0.05$. Other dependence structures (long memory, seasonal correlation, regime changes) are not covered and may give more false trends.
- `mk_correction="hamed_rao"` was liberal even for independent data (7–10%) and barely corrects AR(1) dependence: 15–17% false trends at $\rho = 0.5$ and 24–29% at $\rho = 0.8$ (nominal 5%). Use the default `"ar1"`. `"none"` is strongly anti-conservative for autocorrelated data (about 25% at $\rho = 0.5$, 48% at $\rho = 0.8$).
- Check the red-noise diagnostics in `.explain()`. A residual autocorrelation far above the calibrated `serial_rho` suggests a dependence structure the AR(1) model does not capture.

## 4. Censored (`<DL`) data

- By default, jump and turning-point p-values are computed on a single random imputation of the censored values (`imputation_model="trend"` Tobit MLE). The false-positive rate stayed near or below nominal in simulation, though borderline decisions can vary with `random_state` (roughly 2–27% across 20–50% censoring depending on signal strength).
- `CensoredConfig(detection_imputations=20)` pools several imputations and removes most of that seed dependence. It is conservative and costs power under heavy censoring: at 50% censoring a clear step was detected in 66% rather than 83% of series (32% rather than 69% for seasonal series).
- `scope="windowed"` and `"seasonal_stratified"` use a time-invariant marginal model (`ndimpute`), which can flatten trends among censored values. Interval censoring has not been extensively validated by Monte Carlo simulation.
- Segment slopes under censoring: Akritas–Theil–Sen (`slope_estimator="sen"`) CIs covered 95.2–95.5% at 30–50% censoring with one or two detection limits. Rubin-pooled OLS (`"ols"` → `ols_mi`) covered 92.8–93.3% at 30% and 90.5–91.3% at 50% censoring, and segments over 40% censored now carry the caution `HEAVY_CENSORING_OLS_MI`. DL/2 substitution covered only 57–87%.

## 5. Near-real-time monitoring trades speed for safety

- The default `just_monitor()` control (Šidák + Pocock spending) is very conservative: about 0.2–0.6% false alarms at $\alpha = 0.05$. Real disturbances are therefore detected later than necessary.
- `sequential_control="consecutive_k"` with a planned `horizon` is Monte Carlo calibrated and detects faster, but it runs slightly hot. In the independent 2026-11 audit, false alarms at $\alpha = 0.05$ were 7.3% (H = 24), 5.3% (H = 120) and 4.0% (irregular sampling), even after a more conservative calibration. The likely cause is that harmonics selected from the history slightly overfit it, which the simulation does not reproduce. Use `alpha=0.03` if a strict 5% bound matters. The calibration assumes Gaussian errors and future sampling similar to the history. Without a `horizon` it does **not** control the false-alarm rate.

## 6. Monte Carlo calibrations assume Gaussian errors

The calibrated jump test, ALLSSA frequency selection, STPD turning-point test, and `consecutive_k` monitoring simulate their null distributions under Gaussian errors. With heavy-tailed errors or outliers, use the robust options (`robust="huber"` or `"mm"`). These switch to conservative Bonferroni-type corrections in JUST/STPD (note that `lssa`/`allssa` Monte Carlo draws remain Gaussian-null). Note that `robust="mm"` and `mode="l1"` are **not** high-breakdown estimators: they can break down below 35% contamination.

- **Heavy tails.** With non-robust fits (`robust="none"`, the default), Student-t(3) / t(2) noise gave 16% / 24% family-wise false jumps at a nominal 5% in `just_jumps`; `robust="mm"` gave 0.7% for t(3). `just_jumps` and `stpd` emit `HEAVY_TAILS` when more than 0.5% of residuals lie beyond 4.5 robust SDs. In simulation the warning fired for 64% of t(3) series, 96% of t(2) series, and never on Gaussian noise. Switch to `robust="mm"` when you see it.

## 7. Other assumptions & runtime

- Default frequency grids and windows assume time in **decimal years**. Supply `omega` and window settings explicitly for other units (watch for `TIME_UNIT_ASSUMPTION`). Datetime inputs are converted to a **uniform** decimal-year axis: 1 year = 365.2425 days (the mean Gregorian year), with 2000-01-01T00:00 = 2000.0. It stays within about ±2 days of the calendar decimal year, so "2020.0" is not exactly midnight on 1 January 2020; the original timestamps are kept in `times_raw`. Before 2026-11 the conversion used each calendar year's own length, which compressed leap years by 0.27% and destroyed non-annual periodicities (tides, weekly cycles).
- STPD's step-vs-bend test (`jump_test="step"`) has limited power for small steps on its own: a 4σ step with no slope change was still reported as a turning point by `stpd()` in about 29% of series (18% at 6σ). Through `analyze_series`, whose multi-scale JUST stage now catches such steps first, this fell to 0.5% (4σ) and 2.5% (2σ). A turning point that coincides with a known instrument, datum or method change should still be checked with `just_jumps(step_alpha=0.05, window_scales=(1, 2, 4))`.
- Robust (`robust="mm"`) IRLS fits that hit their iteration limit inside individual sliding windows are not reported per window; under heavy tails ($t_3$) or strong outliers, non-convergence can occur in roughly 2–6% of fits (rare under Gaussian errors).
- **Single-threaded runtime (`OMP_NUM_THREADS=1`, $n \approx 240$, measured 2026-11):** about 0.3–1 s for `just_jumps(jump_definition="level_or_slope")`, about 3× that for the default step definition (three window scales), 0.1–1 s for `stpd(band_method="conditional")` (the default bootstrap band adds a few seconds; `refit_mode="l1"` ~25–30 s), and 3–4 s for `analyze_series(band_method="conditional")`. Pass `band_method="conditional"`, `jump_calibration="bonferroni"` or `season="ols"` for fast exploratory or large-batch runs.

## 8. Validation status

- All four validation tiers under `validation/reports/` — Tier 1 upstream parity (41/41), refreshed Tier 2 Monte Carlo benchmarks (26/26), Tier 2b literature & compositional stress tests (9/9), Tier 2c real-world public-domain scientific/economic datasets (6/6), and Tier 2d external oracles (`statsmodels`/`scipy`), cross-domain signals & adversarial edge cases (15/15) — have been executed and verified against the current hardened codebase.
- High-replicate Monte Carlo calibration sweeps live in `audit/` (including the independent 2026-11 audit, `audit/INDEPENDENT_AUDIT_2026-11.md`, and its probes), while the unit, property-based, calibration-regression and executable-documentation tests run on every push in GitHub Actions CI.
- Independent domain verification of key assumptions (serial correlation, censoring distribution, seasonal structure) is still recommended before publication-critical use.
