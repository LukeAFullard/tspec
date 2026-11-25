# Common Mistakes & Pitfalls

Avoid these eight classic traps when analyzing irregular or environmental time series:

---

## 1. Interpolating Missing Data to a Regular Grid First
- **What goes wrong:** Filling gaps with linear or spline interpolation invents fake observations, artificially lowers $p$-values, and smears abrupt jumps into gradual slopes.
- **What to do instead:** Pass your raw irregular timestamps and `NaN`s straight into `tspec`. Every function works natively on uneven timestamps with zero interpolation.

## 2. Feeding Strongly Seasonal Data Directly into STPD
- **What goes wrong:** STPD fits straight connected lines without seasonal waves. If your river or vegetation record swings strongly between summer and winter, STPD can mistake seasonal peaks/troughs for rapid trend turning points.
- **What to do instead:** Use **`analyze_series()`** (which removes seasonality and vertical jumps via JUST before running STPD) or deseasonalize with `just_decompose()` first.

## 3. Mistaking a Lab Detection-Limit Upgrade for an Environmental Improvement
- **What goes wrong:** Suppose a water-quality laboratory upgrades its instrument in 2018, lowering the reporting limit from `<0.10 mg/L` to `<0.02 mg/L`. Even when true river concentrations never changed, values imputed below `0.10` average higher than values imputed below `0.02` — creating an apparent downward jump right at 2018!
- **What to do instead:** Keep `detect_dl_shifts=True` (enabled by default). If `tspec` prints a `DL_SHIFT_COINCIDENCE` warning, re-run with `CensoredConfig(recensor_check=True)` to test whether the jump vanishes once all values below the higher limit are harmonized.

## 4. Replacing `<DL` with `DL/2` (or Dropping Non-Detects)
- **What goes wrong:** Deleting non-detects biases trend levels upward. Substituting `DL/2` or `0` destroys natural variance (artificially tightening $p$-values) and biases both means and slopes (Helsel, 2012).
- **What to do instead:** Pass raw `"<0.05"` strings or `censor_status` directly to `tspec`, letting `ndimpute` perform unbiased Robust ROS or Parametric MLE imputation.

## 5. Thinking STPD "Missed" a Change When `flagged_jumps` Is Non-Empty
- **What goes wrong:** A practitioner sees `len(result.turning_points) == 0` on a series with an obvious shift and assumes STPD failed.
- **What to do instead:** Read `result.explain()` and inspect `result.flagged_jumps`. If STPD rejected a candidate due to **high NDRI**, it is telling you that the change is a **sudden vertical jump** rather than a smooth turning point — switch to `just_jumps()` or `analyze_series()`.

## 6. Ignoring Red-Noise (Autocorrelation) Warnings
- **What goes wrong:** Daily streamflow, high-frequency groundwater, or closely spaced sensor readings are strongly correlated from one point to the next. Standard white-noise $p$-values treat 100 correlated points as 100 independent draws, inflating false positives dramatically.
- **What to do instead:** Check `result.diagnostics.red_noise` in `.explain()`. When lag-1 autocorrelation exceeds `0.3–0.6`, either tighten `alpha`, supply an inverse-covariance weight matrix `weights = C_f^{-1}` to `lssa`/`allssa`/`just_jumps`, or aggregate redundant sub-daily noise before screening.

## 7. Repeated Peeking & Unbounded Forecast Extrapolation in Near-Real-Time Monitoring
- **What goes wrong:** Running unadjusted `LSSA` across 35 frequencies on every incoming observation causes cumulative false alarms to skyrocket (in our Monte Carlo benchmark `M1`, unadjusted repeated testing over 25 steps at $\alpha = 0.05$ triggers false alarms in **72.5%** of stable series). In addition, extrapolating a short historical trend far into the future without accounting for historical slope estimation error can trigger false drift alarms.
- **What to do instead:** Keep `just_monitor()`'s defaults (`freq_correction="sidak"`, `sequential_control="pocock"`, `account_for_forecast_uncertainty=True`), set `horizon` to your planned number of monitoring steps in advance (or omit it for open-ended spending; note that `sequential_control="consecutive_k"` does not bound the familywise false-alarm rate), and re-baseline periodically if `LONG_MONITORING_HORIZON` appears.

## 8. Running Plain OLS on Records With Extreme Spikes or Heavy-Tailed Noise
- **What goes wrong:** Isolated sensor glitches, stormflow turbidity spikes, or transcription errors carry quadratic leverage under OLS — biasing segment slopes, inflating seasonal errors, or triggering spurious jumps.
- **What to do instead:** Pass `robust="huber"` (heavy-tailed noise) or `robust="mm"` (extreme isolated $>5\sigma$ spikes; Huber-started Tukey biweight, zero jump masking) to any core function, and use `slope_estimator="sen"` (exact Theil–Sen / auto Akritas–Theil–Sen when censored) or `fit_piecewise_trend(..., mode="l1")` for outlier-resistant trend slopes.
