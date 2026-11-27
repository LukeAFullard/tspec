# Changelog

## Unreleased

Statistical-calibration and hardening release. Several defaults change results. `parity_mode=True` keeps reproducing the upstream Ghaderpour implementations.

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
  - `just_monitor`: corrected observation prediction interval (`forecast_pi_low`/`high`) under 2D covariance weighting (including residual variance), fixed `consecutive_k` Monte Carlo calibration to simulate correlated Gaussian history errors under 2D weights and use `omega` / `monitor_omega` consistently, and anchored seasonal harmonic phase to the zero-shifted historical time origin.
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
