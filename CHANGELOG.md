# Changelog

## Unreleased

Statistical-calibration release following an independent audit (`audit/AUDIT_REPORT.md`, fixes and before/after evidence in `audit/FIXES.md`). Several defaults change results. `parity_mode=True` keeps reproducing the upstream Ghaderpour implementations.

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
- `StpdResult.tp_calibration`, `JustJumpResult.jump_calibration`, `AllssaResult.significance`, `JustDecomposeResult.trend_bc`/`trend_reg_bc`/`trend_ci_method`, and `PipelineResult.slope_fan_low/high`.
- `CensoredConfig.imputation_model`, `trend_harmonics`, and `n_imputations`.
- New warnings: `UNCALIBRATED_JUMP_SELECTION`, `UNCALIBRATED_JUMP_PVALUES`, `CALIBRATION_FALLBACK`, `OPEN_ENDED_SPENDING`, `TIME_UNIT_ASSUMPTION`, `DEGENERATE_DEFAULT_GRID`, `BIAS_CORRECTION_FALLBACK`, `TREND_IMPUTATION_FALLBACK`, and `MI_SEGMENT_FALLBACK`.

### Fixed
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
  - `consecutive_k` monitoring does not bound the familywise false-alarm rate.
  - Validation numbers replaced with audited ones.
- Every public function parameter, configuration field, result attribute and warning code is now documented, including the difference between separately fitted `analyze_series` segment slopes and the connected-refit turning-point slopes.
- Figures regenerated; references added.

### Known limitations
- STPD / `analyze_series` trend bands remain conditional on the detected structure (coverage roughly 83–92%).
- STPD power depends on the information within one window (default 5 seasonal cycles), not on the record length.
- JUST/STPD detection p-values on censored data condition on a single imputation.
