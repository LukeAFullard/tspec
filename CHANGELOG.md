# Changelog

## Unreleased

Statistical-calibration and hardening release. Several defaults change results. `parity_mode=True` keeps reproducing the upstream Ghaderpour implementations.

### Fixed (independent audit, 2026-11; see `audit/INDEPENDENT_AUDIT_2026-11.md`)
- **`just_decompose`: separate seasonal window for unevenly sampled data (AUD-19, held-out validations 1 and 2).**
  - New `seasonal_window` (default `"auto"` = `min(7, span)` years; `None` = published single-window JUST, forced by `parity_mode`). The seasonal component is estimated in long duration windows, and the trend in the usual windows on the de-seasonalized series (STL-like two-stage backfitting, one extra pass). **Default results of `just_decompose` change.**
  - Why: a seasonal cycle estimated from one 3-year window is noisy (about σ·√(2K/n)). On irregular sampling, `tspec` lost to interpolate-then-STL on the seasonal component in 14 of 15 designs (pre-registered validation 1). With the separate window it wins in 17 of 18 new designs (validation 2), and on the remainder in 18 of 18.
  - With `eval_times`, windows are now always built on the observations, so results at the observation times no longer change and are never NaN (AUD-19b).
  - New warning `UNCOVERED_OBSERVATIONS` (AUD-19c).
  - Storm-burst or seasonally clustered sampling no longer makes windows ill-conditioned (AUD-19a). The legacy single window blew up on up to 10% of observations with real sampling dates.
  - Censored data: values are imputed once with the usual global censoring imputation and then decomposed with the seasonal window. Windowed imputation (`scope="windowed"`) and full weight matrices fall back to the single window (info message `SEASONAL_WINDOW_UNAVAILABLE`). `analyze_series` keeps the single window until validated.
- **`allssa` resolves close and non-integer frequencies (AUD-05).**
  - New `search="rayleigh"` (default; `"integer"` = published algorithm, forced by `parity_mode`). The default grid runs from `1.5/T` to the mean-rate pseudo-Nyquist in steps of `1/(4T)` (it was integers only, starting at 0.5 cycles/unit).
  - Each peak is refined within `±1/T`, and only a selected frequency within `1/T` is re-fitted, so several components may lie within one cycle per unit. Monte Carlo acceptance uses a matching dense grid.
  - Effects in simulation:
    - 1.00 + 1.37 cy/yr both found in 100% of series (was 0%);
    - an 11.2-year cycle on the default grid: 100% (was 0%);
    - a 2.37 cy/yr component: 100% (was 37.5%);
    - blind tides M2/S2/K1/O1: 4/4 (was 2/4);
    - null false-cycle rate unchanged (0/80 at α = 0.01).
- **`just_decompose` bands under serial correlation (AUD-02, remainder).** New `serial_correlation` (default `"ar1"`). The trend band is widened by `sqrt((1+ρ)/(1−ρ))`, and the seasonal band by the AR(1) spectral factor at 1 cycle/unit (never narrowed). Trend-band coverage under AR(1) ρ = 0.5 / 0.8: 73% / 44% → 97% / 95%. Seasonal: 89% / 86% → 98% / 91%.
- **Sparse records no longer fail with the default seasonal model (AUD-13).** When the smallest window cannot fit the default four OLS harmonics, the highest are dropped, with the warning `HARMONICS_REDUCED`.
- **Bounded memory in the vectorized spectrum (AUD-12).** The frequency chunk adapts to the record length (about 8 MB per work matrix). Peak memory at n = 32 000: 443 MB → 49 MB.
- **`consecutive_k` monitoring calibration is more conservative (AUD-11, partly resolved).** At least 1 999 Monte Carlo draws, and the per-step level is a one-sided 80% lower confidence bound on the simulated α-quantile. False alarms at α = 0.05 remain 7.3% (H = 24), 5.3% (H = 120) and 4.0% (irregular), which points to a plug-in effect of harmonics selected on the history rather than Monte Carlo error; this is disclosed in `limitations.md` §5.
- **Heavy-tail warning (AUD-07).** Non-robust `just_jumps` and `stpd` fits emit `HEAVY_TAILS` when more than 0.5% of the residuals (and at least two) lie beyond 4.5 robust standard deviations. Under Student-t(3) / t(2) noise the non-robust jump test gave 16% / 24% false jumps at a nominal 5%; `robust="mm"` gives 0.7% / 1.3%.
- **ALLSSA with a coarse user grid.** In `search="rayleigh"` mode, a user `omega` coarser than half the Rayleigh resolution is treated as the band of interest and oversampled to `1/(4T)`, so off-grid peaks are reached (blind tides with `omega=0.5, 1.0, …, 4.0`: 4/4).
- **Heavy censoring (AUD-15).** OLS segments more than 40% censored carry the caution `HEAVY_CENSORING_OLS_MI` (Rubin-pooled CIs covered ~91% at 50% censoring; Akritas–Theil–Sen ~95%).
- **`just_jumps` default is the step detector.** New `jump_definition="step"` (default) or `"level_or_slope"` (published JUST, forced by `parity_mode`). `step_alpha` defaults to `"auto"` and `window_scales` to `None`; both resolve from the definition, and explicit values override. **Results of `just_jumps()` with default arguments change:** pure slope changes and curvature are no longer reported as jumps.
- **JUST step detection: curvature-robust step test, multi-scale windows and conditional re-test (AUD-04, AUD-17, AUD-18).**
  - **AUD-18.** `step_pvalue` now compares the discontinuous two-piece fit with the best continuous hinge, both with a global quadratic term and fixed integer harmonics. Before, harmonics were re-selected with the split in place, which favoured the step model. The statistic is evaluated at the split chosen by the jump scan; Holm across candidates is kept and parity mode is unchanged. Previously, a smooth 9-year cycle produced false "jumps" in 37% (amplitude 3σ) and 89% (6σ) of series. The step test reports 0–2%.
  - **AUD-04.** New `window_scales` on `just_jumps` (e.g. `(1, 2, 4)` multiples of the base window); all scales form one Holm family.
  - **AUD-17.** Accepted jumps are re-tested conditional on stronger jumps in their window, and duplicates across scales are merged. Root cause of the "ghost" jumps: in a 3-cycle window, a step one seasonal period from the split is nearly indistinguishable from ramp + seasonal sawtooth. Longer windows resolve it (ghosts in a 6.7σ-step design: 7 → 1 per 60 series).
  - **`analyze_series` defaults.** Stage 1 (JUST) now runs `window_scales=(1, 2, 4)` with `step_alpha=0.05`, so it detects abrupt steps and leaves bends to STPD. In simulation (200 series per row, monthly-like 240-point records with seasonality, σ = 0.5):

    | | before (single scale) | now |
    |---|---|---|
    | false jumps: line / curved / AR(1) 0.5–0.8 | 0–0.5% | 0% |
    | 2σ step found (reported as TP instead) | 0% (11%) | 41.5% (4%) |
    | 4σ step found (reported as TP instead) | 9.5% (42.5%) | 99% (0.5%) |

    Against the original release, a 4σ step had been reported as a turning point in 59% of series.
  - **Costs:** `analyze_series` takes about 3× longer (~3 s instead of ~1 s for 240 points). Steps on curved trends remain hard to detect (2σ on a 9-year cycle of amplitude 3σ: 4%).
  - **Standalone `just_jumps`** defaults to the same step detector (`jump_definition="step"`, see above). In simulation (200 series, 240 points over 20 years, seasonality, σ = 0.5): false jumps on a straight line 1% (level-or-slope: 5.5%), on a smooth 9-year cycle 0% (level-or-slope: 36.5%); a 6.7σ step is found in 100% of series with no extra jumps (level-or-slope: 93% with 0.15 extra jumps per series). `jump_definition="level_or_slope"` restores the published definition, with the new warning `NOT_A_VERTICAL_STEP` on reported jumps that are not significant vertical steps.
- **Change-point and OLS trend tests are calibrated for serially correlated errors (AUD-02).** New `serial_correlation="ar1"` (default; `"ou"` continuous-time; `"none"`, also forced by `parity_mode` and by a 2D `weights` covariance) on `just_jumps`, `stpd`/`detect_turning_points` and `analyze_series`.
  - A lag-1 residual autocorrelation is estimated robustly to jumps from difference variances (Chakar et al., 2017), using trimmed means after a smooth global fit (quadratic, four hinges, annual harmonics, and a level-shift column for every step larger than six robust SDs of the differences). It is treated as zero unless distinguishable from zero; otherwise one standard error is added.
  - Monte Carlo nulls (JUST max-F, STPD max-t) simulate AR(1)/OU errors. Analytical references are deflated by `(1+rho)/(1-rho)`.
  - `analyze_series` OLS/`ols_mi`/Huber segment-slope CIs, p-values and the trend band use an exact O(n) sandwich variance.
  - New result fields `serial_correlation`, `serial_rho` (estimate) and `serial_rho_used` (calibration value), and the info warning `SERIAL_CORRELATION_ADJUSTED`.
  - Effects in simulation (300 series per row; nominal level in brackets):

    | Test | white | AR(1) ρ = 0.3 | ρ = 0.6 | ρ = 0.8 |
    |---|---|---|---|---|
    | `just_jumps` family-wise false jumps [5%] | 5.7% (was 5.7%) | 4.3% (was 34%) | 4.0% (was 95%) | 7.0% (was 100%) |
    | `stpd` false turning points [1%] | 0.3% (was 0.3%) | — | 0.7% at ρ = 0.5 (was 28%) | 3.0% (was 88%) |
    | `analyze_series` OLS slope, Type-I [5%] | 3.7–4.0% (was 4.3–4.7%) | — | 1.3–6.7% at ρ = 0.5 (was 22–25%) | 2.0–8.7% (was 47–50%) |

    OU noise with `serial_correlation="ou"`: 2.3% false jumps (was 100%). Slope-CI coverage under ρ = 0.5–0.8 is 91–99% (was 50–78%).
  - **Cost:** on white noise, `just_jumps` power for a 4σ step is 46% (was 51%) and `stpd` bend recall 38% (was 43%), because ρ̂ occasionally looks significant by chance. On red noise, power drops by design, since correlated data carry less information. `just_decompose`, `just_monitor` and `lssa`/`allssa` are not yet adjusted.
- **Datetime inputs use a uniform time axis (AUD-01).** Datetimes were converted to *calendar* decimal years (each year divided by its own 365 or 366 days), compressing leap years by 0.27%. A noiseless M2 tide on hourly datetimes reached a normalized LSS of only 0.25 instead of 1.00. Datetimes now map to `2000 + (t − 2000-01-01) / 365.2425 days` (the mean Gregorian year), which is within about ±2 days of the calendar decimal year. **Decimal-year values of datetime inputs change slightly.**
- **STPD no longer reports abrupt steps as smooth turning points (AUD-03, AUD-06).** New `jump_test="step"` (default; `"ndri"` in parity mode, `"both"` available) and `step_alpha=0.05`. A significant break is classified as a jump when a discontinuous two-line fit beats the best *continuous* hinge, with the knot optimized between samples (nested F-test, Bonferroni over splits), and the step captures ≥ 25% of the break's explained variance. Turning points within `tp_th` of a flagged step are dropped. New fields `step_pvalue`/`step_share` on `TurningPointEvent` and `FlaggedJumpCandidate`, `StpdResult.jump_test`, and the warning `STEP_JUMP_CANDIDATE`. Effects in simulation:
  - a 4σ step was reported as a turning point in 29% of series (was 69%); a 6σ step in 18% (was 63%);
  - through `analyze_series` a 4σ step was reported as a turning point instead of a jump in 38% (seasonal) / 14% (non-seasonal) of series (was 59% / 23%), and a 6σ step in 17% / 1% (was 24% / 2%);
  - genuine bends keep their recall (46–99%), wrongly flagged as jumps 0.5–1.5% (NDRI: 1.5–58.5%);
  - noiseless or σ = 10⁻³ bends are detected (NDRI rejected them).
  - `analyze_series().explain()` warns when STPD flags a step that JUST missed.
- **Zero-variance series no longer produce "significant" jumps (AUD-09).** On a perfectly straight noiseless line `just_jumps` reported a −6e-17 jump with p = 6.7e-10, and `analyze_series` split the line in two.
- **`just_monitor` requires an adequate history for numeric time (AUD-10).** With `season != "none"` it now needs about one seasonal cycle of observations, and at least the model size + 5 points, for numeric as well as datetime time axes. A 5-point history was previously accepted and raised false alarms.

### Changed defaults (results may differ)
- **`just_jumps`**: jump selection is family-wise calibrated by default.
  - `jump_alpha` now defaults to `"auto"`, which resolves to `0.05` (or `None` with `parity_mode=True`).
  - `mag_th`/`dir_th` default to `0` (they were `0.05`/`0.01`, the NDVI-scale thresholds).
  - The new `jump_calibration="montecarlo"` refers each `jump_pvalue` to the exact Monte Carlo null of the maximum F statistic over the split search. `step_pvalue` is Bonferroni-adjusted over splits. Both are Holm-adjusted across candidate locations.
  - Effect: the false-jump probability on jump-free series falls from 70–100% to about 4–6%.
- **`allssa`**: new `significance="montecarlo"` (default) accepts a frequency only if it is significant after accounting for the frequency search. `"sidak"` is an analytical alternative and `"per_frequency"` is the published rule. Effect: the spurious-cycle probability at `alpha=0.01` falls from 18–25% to ≤ 1.3%.
- **`just_decompose`**: new `trend_ci_method="bias_corrected"` (default) builds the trend band from local-quadratic window refits, centred on the new `trend_bc`. Effect: coverage of curved trends rises from 37–77% to about 95%. The bands are about 1.2–1.35× wider.
- **Censored data**: new `CensoredConfig(imputation_model="trend")` (default with `scope="global"`) imputes from a censored-regression MLE conditional on time.
  - `analyze_series` OLS segments with censored values use Rubin-pooled multiple imputation (`n_imputations=20`, `estimator="ols_mi"`).
  - `imputation_model="marginal"` restores the previous behaviour.
- **`just_monitor`**: without `horizon`, Pocock/O'Brien–Fleming use open-ended spending (reported as `horizon=0`) instead of inferring the horizon from the number of observations, which made past thresholds depend on future data. Pass a planned `horizon` for more power.
- **`stpd` / `detect_turning_points`**: turning-point p-values are family-wise calibrated by default.
  - The new `tp_calibration="montecarlo"` refers each window's slope-change t statistic to the Monte Carlo null of the maximum over candidate knots and scan directions; `"bonferroni"` is the default for robust fits. Calibrated p-values are Holm-adjusted across candidate locations.
  - `snr_th` now defaults to `None`, which resolves to `0` (no SNR gate), or `1.0` with `parity_mode=True`.
  - Effect: false turning points on straight-line series fall from up to 2.2% to 0.2–0.6% at `alpha=0.01`, and single-turning-point detection rises from 0% to 52% (n=150, σ=1) and from 32% to 82% (n=500, σ=0.6).
- **`just_monitor(sequential_control="consecutive_k")`** with a `horizon`: the per-step level is now calibrated by Monte Carlo so that `k` consecutive exceedances within the horizon occur with probability ≤ `alpha` under the fitted history model (new `n_mc`, default 999). Future sampling gaps are resampled from the history, so the level uses no monitoring data.
  - Effect at `alpha=0.05`: the familywise false-alarm rate is 3.3–5.3% over 24–96 steps (Monte Carlo SE 0.6–1.1%) (uncalibrated: up to 15% over 96 steps), and 2σ steps are detected more often than with Pocock spending (82–87% vs. 48–49% within 24 steps).
  - Without `horizon` the per-step `alpha` is unchanged and the new warning `UNCALIBRATED_CONSECUTIVE_K` is reported.
- **`stpd` / `detect_turning_points` and `analyze_series`**: post-detection bands come from a detect-then-fit bootstrap by default (new `band_method="bootstrap"`, `n_boot=499`; `"conditional"` and `parity_mode=True` keep the analytical bands). The bootstrap is a wild residual bootstrap with leverage-adjusted residuals that re-locates every turning point with STPD's own knot estimator, refits, and calibrates the fixed-structure dispersion to the analytical standard error. In `analyze_series` it also re-runs the windowed seasonal and jump-step estimators and bootstraps the OLS segment-slope CIs.
  - Effect, after correct detection at nominal 95%: STPD bands 95.5–97.2% (analytical: 85–95%). The `analyze_series` trend band covers ~95–98% (was 83%) and segment slopes ~95% (was 90%).
  - `refit_mode="l1"` fits now get a band (about 6 s per call).
- **Mann–Kendall / Sen with `mk_correction="ar1"`**: segments with n ≤ 300 refer the statistic to a parametric AR(1) bootstrap (debiased autocorrelation, with its sampling uncertainty propagated) for the p-value and the Sen CI. Type-I error at α = 0.05 is now 3.8–6.2% for n = 60–250 and ρ = 0–0.7 (analytical factor: up to 8.6%).
- **`analyze_series`**:
  - New `mk_correction="ar1"` (default) corrects Mann–Kendall/Sen inference for serial correlation; `"hamed_rao"` and `"none"` are also available.
  - With `slope_estimator="sen"`, the median-pivoted fan moves from `trend_ci_low/high` to the new `slope_fan_low/high` (it is not a trend CI).
  - Stage 1 now inherits the calibrated `just_jumps` defaults (`bonferroni_window` is no longer set).

### Added
- `season_significance` on `just_jumps`, `just_decompose` and `just_monitor` (default `"sidak"`; `"per_frequency"` forced by `parity_mode`). Harmonics selected by ALLSSA inside JUST windows and monitoring histories now use a family-wise threshold. Effects:
  - Seasonal-band coverage rises from about 90–91% to 93.5–94.6%.
  - Monitoring forecast mean-CI coverage rises from 92.9% to 94.9%.
  - Jump power and magnitude accuracy improve (4σ magnitude RMSE 0.029 → 0.019).
- New warning `SHORT_SEASONAL_WINDOW` when windows cover fewer than two cycles of the lowest searched frequency.
- `allssa(significance="sidak")` now counts about three times the Rayleigh number (T × bandwidth) of independent frequencies, calibrated by simulation for short windows.
- `lssa(peak_fap=...)` and `LssaResult.peak_fap`: the family-wise false-alarm probability of the highest peak.
- `band_method` / `n_boot` on `stpd`, `detect_turning_points` and `analyze_series`. New fields: `StpdResult.band_method`, `PipelineResult.band_method`, `TurningPointEvent.detection_window`, `JustDecomposeResult.window_bounds` / `window_dilation`.
- `CensoredConfig.detection_imputations` (default `1`): with `> 1`, `just_jumps` and `stpd` re-impute each candidate window from its null model and pool the calibrated p-values over the imputations (Licht–Rubin) before the Holm adjustment. Decisions then no longer depend on the imputation seed (1–9% of decisions flip vs. 2–27% with a single imputation at 20–50% censoring). It is conservative and can cost jump power under heavy censoring. Reported as `JustJumpResult.n_imputations` / `StpdResult.n_imputations`, with warning `MI_DETECTION_FALLBACK`.
- `MonitorResult.consecutive_k` / `consecutive_k_calibrated`, `StpdResult.tp_calibration`, `JustJumpResult.jump_calibration`, `AllssaResult.significance`, `JustDecomposeResult.trend_bc`/`trend_reg_bc`/`trend_ci_method`, and `PipelineResult.slope_fan_low/high`.
- `CensoredConfig.imputation_model`, `trend_harmonics`, and `n_imputations`.
- New warnings: `UNCALIBRATED_JUMP_SELECTION`, `UNCALIBRATED_JUMP_PVALUES`, `CALIBRATION_FALLBACK`, `OPEN_ENDED_SPENDING`, `TIME_UNIT_ASSUMPTION`, `DEGENERATE_DEFAULT_GRID`, `BIAS_CORRECTION_FALLBACK`, `TREND_IMPUTATION_FALLBACK`, `MI_SEGMENT_FALLBACK`, `UNCALIBRATED_CONSECUTIVE_K`, and `INTEGER_STATUS_INTERPRETED_AS_RIGHT`.

### Fixed
- **Statistical & numerical hardening**:
  - `just_monitor`: corrected observation prediction interval (`forecast_pi_low`/`high`) under 2D covariance weighting (including residual variance), fixed `consecutive_k` Monte Carlo calibration to simulate correlated Gaussian history errors under 2D weights and use `omega` / `monitor_omega` consistently, subtracted biweight-rejected historical spikes from monitoring window DOF under `robust="mm"`, and anchored seasonal harmonic phase to the zero-shifted historical time origin.
  - Robust estimation & 2D SPD weights: fixed 2D SPD Huber/MM IRLS to optimize the true whitened objective $L\,\mathrm{diag}(w)\,L^\top$ (guaranteeing monotone nested loss); exposed `fit_piecewise_trend(..., mode="l1"|"huber")` with strict 1D weights and exact weighted L1 scaling; fixed `detect_turning_points(..., robust="mm")` to run robust post-detection refits cleanly.
  - Censored data & MI: fixed Rubin-pooled OLS segment slope bias by drawing from the exact censored posterior under the fitted Tobit model; calibrated `recensor_check` with Monte Carlo jump p-values; fixed `CensoringProvenance.window_fallback_count` accounting for `scope="windowed"` in `just_jumps`, `just_decompose`, and `detect_turning_points`; accurately attributed Tobit MLE vs `ndimpute` in summaries/warnings; added `INTEGER_STATUS_INTERPRETED_AS_RIGHT` caution for numeric `{0, 1}` status arrays.
  - Decomposition & pipeline uncertainty: aligned `just_decompose` bias-corrected trend CIs with Gaussian-calibrated standard errors; fixed `just_jumps(..., season="allssa")` `step_pvalue` to use the break-adjusted harmonic basis so strong continuous turning points are not misclassified as abrupt jumps; fixed `analyze_series` bootstrap variance scaling and jump-offset deseasonalization so nominal 95% trend and slope bands achieve ~95–98% coverage; eliminated floating-point drift in `detect_turning_points` knot recovery.
  - Core edge cases & hygiene: fixed non-monotonic/duplicate timestamp handling and empty-window guards, removed remaining optional-dependency leaks (`pandas`/`ndimpute`/`matplotlib` strictly lazy), ensured `tspec.__all__` and `docs/reference/api.md` are synchronized, and fixed `lssa`/`allssa` rank-deficient and degenerate-grid guards.
- Akritas–Theil–Sen p-values and CIs use the censoring-aware permutation variance of the censored Kendall statistic (Daniels, 1944). They were conservative before: 98% CI coverage at 50% censoring, now about 95%.
- `just_monitor` censoring guardrails (maximum censored fraction, minimum detects) are evaluated on the historical period only, so heavy censoring after the monitoring start no longer blocks the analysis.
- Silent fallbacks are now reported with new warnings:
  - `ROBUST_NOT_CONVERGED`: robust IRLS hit its iteration limit in `lssa` or `fit_piecewise_trend(mode="huber")`.
  - `STABILITY_DRAW_FALLBACK`: censoring stability draws reused fixed parameters.
  - `STPD_SEGMENT_SKIPPED`: an epoch was too short for turning-point detection.
  - `SEN_CI_UNAVAILABLE`: a Theil–Sen segment had more than 2000 observations.
  - `ATS_UNAVAILABLE`: a censored segment could not use Akritas–Theil–Sen.
- Theil–Sen CI order statistics now follow Gilbert (1987) / `scipy.stats.theilslopes`.
- Numeric string time axes are parsed as numbers, and datetimes outside 1678–2262 are rejected (they previously wrapped silently).
- The `HIGH_CONDITION_NUMBER` warning now triggers at cond > 1e6 (it was 1e10).
- Divide-by-zero runtime warnings in the Huber weights are removed.
- `mypy` configuration now works with numpy ≥ 2.3 stubs, and the lint errors in `validation/` are fixed.

### Documentation
- Corrected claims:
  - `robust="mm"` is a Huber-started MM-type estimator, not Yohai's high-breakdown MM.
  - L1 hinge fits do not have 50% breakdown.
  - `consecutive_k` monitoring without a `horizon` does not bound the familywise false-alarm rate.
  - Validation numbers replaced with audited ones.
- Every public function parameter, configuration field, result attribute and warning code is now documented, including the difference between separately fitted `analyze_series` segment slopes and the connected-refit turning-point slopes.
- Figures regenerated; references added.

### Known limitations
- Post-detection bands are conditional on the number of breakpoints found. After correct detection at nominal 95%, `analyze_series` trend bands cover ~95–98% and segment-slope CIs cover ~95%.
- STPD power depends on the information within one window (default 5 seasonal cycles), not on the record length.
- By default, JUST/STPD detection p-values on censored data use a single time-conditional imputation. This kept valid size in simulation (2–3% false jumps at `jump_alpha=0.05`, borderline 6.7% for seasonal series at 50% censoring), but borderline decisions depend on the imputation seed. `detection_imputations` removes the seed dependence at a cost in power.
