# Interpreting Results, Plots & Warnings

Every result returned by `tspec` (`LssaResult`, `AllssaResult`, `JustJumpResult`, `JustDecomposeResult`, `MonitorResult`, `StpdResult`, `PiecewiseTrendResult`, `PipelineResult`) shares four uniform methods:

- **`.explain()`** — Plain-English narrative written for domain scientists and reports. Start here!
- **`.summary()`** — Compact technical table of indices, timestamps, magnitudes/slopes, $p$-values, NDRI, and red-noise diagnostics.
- **`.to_dataframe()`** — Tidy `pandas.DataFrame` ready for downstream export or batch aggregation.
- **`.plot()`** — Diagnostic Matplotlib figure using colorblind-safe Okabe–Ito colors and distinct markers for censored observations.

---

## Understanding Screening $p$-Values & Repeated Testing

!!! important "Screening vs. Fixed-Date $p$-Values"
    Inside a sliding window, JUST and STPD search across many candidate split indices and select the index $j^*$ that minimizes the residual sum of squares *before* computing `jump_pvalue` (JUST) or Student-$t$ `pval` (STPD). Because $j^*$ was chosen by looking at the data, raw window $p$-values are **screening statistics** — they are optimistically small compared to testing a single pre-specified date!

How `tspec` handles this transparently:
- In **`analyze_series()`**, Stage 1 enables **`bonferroni_window=True`** by default (multiplying raw window $p$-values by the number of tested interior splits) AND applies the 1-df **`step_pvalue`** test comparing the best discontinuous jump against the best continuous hinge — suppressing post-selection noise while cleanly handing smooth bends to STPD.
- In **`just_jumps()`**, `bonferroni_window=False` and `jump_alpha=None` by default for exact parity with Ghaderpour (2020), while always computing and storing both `jump_pvalue` and `step_pvalue` on every `JumpEvent`.

---

## Reading Diagnostic Warnings

| Warning Code | Severity | What It Means & What To Do |
| :--- | :--- | :--- |
| `DL_SHIFT_COINCIDENCE` | Warning | A jump, turning point, or NRT alarm landed right next to a laboratory detection-limit change. Re-run with `CensoredConfig(recensor_check=True)` and check lab metadata! |
| `DL_ARTIFACT_LIKELY` | Warning | The detected jump vanished once data were recensored to a common detection limit — almost certainly a lab artifact, not an environmental shift. |
| `STRONG_AUTOCORRELATION` | Warning | Residual lag-1 or continuous-time OU correlation $\ge 0.60$. Nominal $p$-values are unreliable; tighten `alpha` or supply correlated inverse-covariance `weights = C_f^{-1}`. |
| `MODERATE_AUTOCORRELATION` | Caution | Residual autocorrelation $\in [0.30, 0.60)$. False-positive rates are modestly elevated; consider `alpha=0.005`. |
| `HIGH_NDRI_JUMP_CANDIDATE` | Caution | STPD rejected a strong candidate because residuals were lop-sided (`NDRI >= ndri_th`). Run `just_jumps()` or `analyze_series()` to check for an abrupt vertical jump. |
| `HORIZON_EXCEEDED` | Warning | `just_monitor()` ran past its planned `horizon` under Pocock or O'Brien–Fleming spending; familywise Type I error control is no longer guaranteed past `horizon`. Increase `horizon` or switch to `sequential_control="consecutive_k"`. |
| `LONG_MONITORING_HORIZON` | Caution | The monitoring span exceeds the stable historical span, so extrapolated trend uncertainty grows quadratically. Keep `account_for_forecast_uncertainty=True` (default) or re-baseline periodically. |
| `UNADJUSTED_REPEATED_TESTING` | Warning | Multiplicity controls (`freq_correction`, `sequential_control`) are disabled in `just_monitor()` outside `parity_mode=True`. |
| `PENDING_CONFIRMATION` | Info | Under `sequential_control="consecutive_k"`, the latest observation exceeded the threshold but has not yet reached `consecutive_k` consecutive hits. |
| `THRESHOLD_SCALE_MISMATCH` | Caution | Default `mag_th=0.05` (NDVI scale) looks too small or too large for your data's IQR. Set a domain-meaningful `mag_th` or enable `jump_alpha=0.05`. |
| `VARIABLE_SAMPLING_RATE` | Caution | Sampling density shifts $> 2\times$ across the series under `window_mode="count"`. Try `window_mode="duration"`. |
| `NONSTATIONARY_CENSORING_PATTERN` | Caution | Censoring fraction shifts by $> 30\%$ between halves under `scope="global"`. Consider `CensoredConfig(scope="windowed")`. |
| `HEAVY_CENSORING` / `WINDOW_HEAVY_CENSORING` | Warning | Over 50% of the series (or supporting window) is censored. Enable `n_stability_draws=20` (or `stability_mode="bootstrap"`) and treat estimates cautiously. |
| `SPARSE_WINDOW_SKIPPED` | Info | One or more duration windows (or evaluation gaps) held too few observations and were safely skipped. |

---

## Visual Legend for `.plot()`

- **Solid dark dots (`●`):** Measured (uncensored) observations.
- **Hollow downward orange triangles (`▽`):** Imputed left-censored observations (`<DL`).
- **Hollow upward purple triangles (`△`):** Imputed right-censored observations (`>UL`).
- **Hollow squares with whiskers (`□`):** Imputed interval-censored observations (`[L, R]`).
- **Vertical dashed orange lines:** JUST abrupt jumps (with `Δ` magnitude badge and `(DL shift!)` flag if coinciding).
- **Vertical dotted gray lines:** Detected laboratory detection-limit shifts.
- **Green diamonds (`◆`):** STPD continuous turning points (with `Δslope` and `NDRI` badges).
