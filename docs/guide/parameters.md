# Parameter Cheat Sheet

A plain-English translation of every key parameter in `tspec`, its default value, what happens when you raise or lower it, and practical rules of thumb.

## Windowing Parameters (`just_jumps`, `just_decompose`, `detect_turning_points`)

| Parameter | Plain-English Meaning | Default | Make It Larger | Make It Smaller | Rule of Thumb |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `size` | How many data points sit inside each sliding window (`window_mode="count"`). | JUST: `3 * M`<br>STPD: `5 * M` | Smoother trends, more statistical power, misses closely spaced changes. | Tracks rapid changes, noisier seasonal/slope estimates. | Pick **~3 years** of observations for JUST (3 full seasonal cycles) and **~4–5 years** for STPD. |
| `step` | How many points the window slides forward each move. | `M` (~1 year) | Faster runtime, fewer overlapping checks per breakpoint. | Slower runtime, higher occurrence counts and finer overlap. | `size / 3` (JUST) or `size / 5` (STPD) works well. |
| `window_mode` | Whether windows count *observations* (`"count"`) or *calendar years* (`"duration"`). | `"count"` | — | — | Use `"duration"` whenever sampling frequency changes $> 2\times$ over the record. |
| `window_duration` | Calendar span of each window when `window_mode="duration"`. | `min(3.0, span)` (JUST)<br>`min(5.0, span)` (STPD) | More stable seasonal/slope fit over sparse intervals. | Better resolution for closely spaced events. | Match 3.0 years for seasonal JUST, 4–5 years for STPD. |

## Spectral & Significance Parameters (`lssa`, `allssa`, `just_jumps`)

| Parameter | Plain-English Meaning | Default | Make It Larger | Make It Smaller | Rule of Thumb |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `alpha` | Significance threshold for accepting seasonal cycles (ALLSSA) or slope changes (STPD). | `0.01` | Detects weaker cycles/bends, higher false-positive risk. | Stricter; fewer false alarms, may miss subtle changes. | `0.01` is a safe default; try `0.001` under moderate red noise. |
| `omega` | Candidate seasonal frequencies to search (cycles per unit time; cycles/yr for dates). | `lssa`: `None` (no spectrum)<br>`allssa`: `1..ceil(M/2)-1`<br>JUST ALLSSA: `0.8..3.8`<br>JUST OLS: `[1, 2, 3, 4]` | Searches faster/shorter sub-annual cycles (risk of overfitting sparse data). | Restricts search to annual/semi-annual waves only. | Default covers annual (`1`), semi-annual (`2`), and tri-annual (`3`) waves cleanly. |
| `mag_th` / `dir_th` | Minimum jump size / slope change kept by `just_jumps()` (effect-size filters applied after Holm adjustment). | `0` / `0` (`0.05` / `0.01`, NDVI scale, with `parity_mode=True`) | Ignores small jumps; keeps only large domain-meaningful shifts. | Keeps smaller shifts (more sensitive to noise). | Always match your measurement units (e.g., mg/L or mm), OR use `jump_alpha=0.05` / `analyze_series()`. |
| `jump_alpha` | Family-wise significance level for reported jumps (scale-free). | `0.05` (`None` with `parity_mode=True`) | Allows weaker jumps through. | Keeps only strongly significant jumps. | Keep the default; `jump_calibration="montecarlo"` (default) corrects for the split search and Holm-adjusts across candidates. |
| `jump_calibration` | How `just_jumps()` corrects jump p-values for searching every split point (`"montecarlo"`, `"bonferroni"`, `"none"`). | `"montecarlo"` (`"bonferroni"` for robust fits, `"none"` in parity) | — | — | Keep the default; `"bonferroni"` is faster but more conservative; `"none"` gives raw screening p-values with no false-positive control. |
| `n_mc` | Monte Carlo draws for calibrated jump, STPD, monitoring, and ALLSSA tests. | `999` | More precise p-values, slower. | Faster, coarser p-values. | `999` is enough for α ≥ 0.01. |
| `significance` | How `allssa()` decides a cycle is real (`"montecarlo"`, `"sidak"`, `"per_frequency"`). | `"montecarlo"` (`"per_frequency"` in parity) | — | — | Keep the default; `"per_frequency"` reproduces the published rule but selects spurious cycles in ~20% of noise series. |
| `season_significance` | Rule for harmonics that ALLSSA selects inside `just_jumps()`, `just_decompose()` and `just_monitor()` windows (`"sidak"`, `"montecarlo"`, `"per_frequency"`). | `"sidak"` (`"per_frequency"` in parity) | — | — | Keep the default; use windows spanning at least two seasonal cycles, or `season="ols"` when the frequencies are known. |
| `peak_fap` | How `lssa()` reports the family-wise false-alarm probability of its highest peak (`"sidak"`, `"montecarlo"`, `"none"`). | `"sidak"` | — | — | Judge peaks by `result.peak_fap`, not by how many points exceed the per-frequency threshold. |
| `jump_th` / `tp_th` | Minimum time gap required between two separate jumps (`jump_th`) or turning points (`tp_th`). | `0.75` (JUST)<br>`1.0` (STPD) | Merges nearby events into a single stronger breakpoint. | Allows closely spaced consecutive changes. | `0.75–1.0` yr prevents a single disturbance from being double-counted. |

## Turning-Point Quality Filters (`detect_turning_points` / `stpd`)

| Parameter | Plain-English Meaning | Default | Make It Larger | Make It Smaller | Rule of Thumb |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `jump_test` | How `stpd()` decides a significant break is an abrupt jump rather than a bend: `"step"` (nested step-vs-continuous-hinge test), `"ndri"` (published NDRI threshold), `"both"`. | `"step"` (`"ndri"` in parity) | — | — | Keep `"step"`; NDRI misses most pure steps and rejects very precise bends. |
| `step_alpha` | Level of the step-vs-bend test. | `0.05` | Flags more breaks as jumps (fewer steps mislabelled as bends, slightly more bends lost). | Flags fewer breaks as jumps. | `0.01–0.05`. |
| `ndri_th` | Maximum residual imbalance allowed (used only with `jump_test="ndri"`/`"both"`). | `0.30` | Accepts slightly jump-like bends. | Rejects anything even mildly asymmetric as a jump. | `0.25–0.35` (Ghaderpour et al., 2024). |
| `ndri_mode` | How NDRI compares left vs. right residuals. | `"rmse"` (`"l2"` in parity) | — | — | Keep `"rmse"` for uneven sampling and weighted heteroscedastic series; use `parity_mode=True` for Ghaderpour (2024) exact parity. |
| `serial_correlation` | Error model used to calibrate `just_jumps`, `stpd` and `analyze_series` tests (`"ar1"`, `"ou"`, `"none"`). | `"ar1"` (`"none"` in parity or with a 2D `weights` covariance) | — | — | Keep `"ar1"`; use `"ou"` for strongly irregular sampling. `"none"` gives far too many false jumps/turning points on autocorrelated data. |
| `tp_calibration` | How `stpd()` corrects turning-point p-values for searching every knot in a window (`"montecarlo"`, `"bonferroni"`, `"none"`). | `"montecarlo"` (`"bonferroni"` for robust fits, `"none"` in parity) | — | — | Keep the default; `"none"` gives raw screening p-values with no false-positive control. |
| `band_method` / `n_boot` | How the post-detection trend band (`stpd`, `analyze_series`) and OLS segment-slope CIs (`analyze_series`) are computed. | `"bootstrap"` / `499` (`"conditional"` in parity) | — | — | Keep `"bootstrap"`: it includes turning-point location uncertainty and, in `analyze_series`, seasonal and jump-step estimation. `"conditional"` treats the detected structure as known and is faster. |
| `snr_th` / `snr_mode` | Optional minimum signal-to-noise ratio for a turning point. | `None` → `0` (no gate; `1.0` in parity) / `"centered"` | Requires stronger trend relative to scatter, whatever the sample size. | Allows subtler bends in noisy records. | Leave the gate off: the calibrated test already controls false detections, and `snr_th=1.0` cut detection from 82% to 26% in simulation. Keep `"centered"` for data with non-zero baselines. |
| `margin` | Points ignored near each window edge so slopes aren't fit to 1–2 points. | `count`: `max(2, min(M, size // 5))`<br>`duration`: `max(2, min_obs // 5)`<br>`parity_mode=True`: `2` | More stable slopes; cannot detect bends near record edges. | Detects bends closer to window edges, noisier slopes. | Always $\ge 2$ (`tspec` enforces this automatically). |

## Censored Data (`CensoredConfig`)

| Parameter | Plain-English Meaning | Default | Guidance |
| :--- | :--- | :--- | :--- |
| `method` | Marginal imputation engine inside `ndimpute` (`"ros"`, `"parametric"`, or `"substitution"`), used when `imputation_model="marginal"` or under windowed/seasonal/Weibull fallback. | `"ros"` | Under default `imputation_model="trend"` + `scope="global"`, `tspec`'s Tobit MLE runs instead. For marginal mode, use `"ros"` for left-censored `<DL` data or `"parametric"` for right/mixed/interval censoring (`"substitution"` is for sensitivity checks only). |
| `impute_type` | How censored points are filled (`"stochastic"`, `"mean"`, `"quantile"`). | `"stochastic"` | Keep `"stochastic"` (seeded) — avoids variance shrinkage (`"mean"`) and temporal rank bias (unshuffled `"quantile"`). |
| `scope` | Whether one distribution is fitted globally (`"global"`), per window (`"windowed"`), or per season (`"seasonal_stratified"`). | `"global"` | Switch to `"windowed"` if a huge jump shifts censoring from >50% to <5%, or `"seasonal_stratified"` for intense seasonal censoring. |
| `n_stability_draws` | How many extra imputation realizations are run to measure breakpoint stability. | `20` in `analyze_series()`<br>`0` in single calls | Set `>= 20` when censoring exceeds 20%; use `stability_mode="bootstrap"` for full parameter-uncertainty propagation. |
| `detect_dl_shifts` / `recensor_check` | Guard against fake jumps from laboratory detection-limit upgrades. | `True` / `False` | Keep `detect_dl_shifts=True`; turn on `recensor_check=True` whenever a `DL_SHIFT_COINCIDENCE` warning appears. |
| `imputation_model` | Whether censored values are imputed conditionally on time (`"trend"`, Tobit censored regression) or from one time-invariant distribution (`"marginal"`, `ndimpute`). | `"trend"` | Keep `"trend"`; `"marginal"` flattens trends among censored values and biases slopes toward zero. |
| `trend_harmonics` | Seasonal harmonics (cycles/time unit) included in the `"trend"` imputation model. | `(1, 2)` for dates, none for numeric time | Add harmonics that match your seasonality; numeric time must be in years for `(1, 2)` to mean annual/semi-annual. |
| `n_imputations` | Multiple imputations pooled with Rubin's rules for censored OLS segment slopes in `analyze_series()`. | `20` | 20–50 is typical; `1` disables pooling (CIs then ignore imputation uncertainty). |
| `detection_imputations` | Imputations over which `just_jumps()`/`stpd()` pool each candidate's calibrated p-value on censored data. | `1` (single imputation) | `20` makes detections independent of the imputation seed and conservative, at some jump power under heavy censoring. |

## Near-Real-Time Monitoring (`just_monitor`)

| Parameter | Plain-English Meaning | Default | Guidance |
| :--- | :--- | :--- | :--- |
| `start` | Where active monitoring begins (0-based index OR timestamp/year). | `n - 2` | Set to the date after which you want to detect new disturbances (e.g., `start="2022-01-01"` or `start=2022.0`). |
| `history` | Where the stable historical baseline begins (index OR timestamp/year). | `0` | If an earlier historical jump occurred, set `history` right after it (or find it first with `just_jumps()`). |
| `h` | Fraction of the historical period kept immediately before `start` as context during monitoring. | `0.25` | `0.25` (25% of history) balances responsiveness to new changes against baseline stability. |
| `freq_correction` | Per-step adjustment across the 35 scanned monitoring frequencies. | `"sidak"` | Keep `"sidak"` (or `"bonferroni"`). Use `"none"` only via `parity_mode=True`. |
| `sequential_control` | Repeated-testing control across incoming monitoring steps. | `"pocock"` | `"pocock"` with `horizon` for a fixed plan, or without `horizon` (open-ended spending) for live streams; `"obrien_fleming"` to guard early peeks; `"consecutive_k"` with `horizon` (Monte Carlo-calibrated) to require repeated exceedances with familywise control and more power than Pocock; without `horizon` it does **not** bound the familywise rate. |
| `horizon` | Planned number of monitoring steps $D_{\max}$ for alpha-spending rules and `consecutive_k` calibration, fixed in advance. | `None` (open-ended spending for Pocock/O'Brien–Fleming; uncalibrated for `consecutive_k`) | Set explicitly (e.g., `horizon=24` for 2 years of monthly updates) for more power and calibrated `consecutive_k`; never derive it from how much data you currently have. |
| `window_mode` | Whether the residual monitoring window grows (`"expanding"`) or slides (`"sliding"`). | `"expanding"` | `"expanding"` matches Ghaderpour & Vujadinovic (2020b); `"sliding"` keeps memory bounded over multi-year monitoring. |

## Robust Estimation & Segment Slopes (All Core Functions & `analyze_series`)

| Parameter | Plain-English Meaning | Default | Guidance |
| :--- | :--- | :--- | :--- |
| `robust` | Phase 1 outlier-resistance mode (`"none"`, `"huber"`, or `"mm"`). | `"none"` | Use `"huber"` for convex M-estimation under heavy tails ($t_3$/$t_5$); use `"mm"` (Huber-init Tukey biweight) for extreme isolated spikes ($>5\sigma$). Ignored in `parity_mode=True`. |
| `huber_k` | Huber tuning threshold in MAD-scaled residual units. | `1.345` | `1.345` gives 95% Gaussian efficiency while bounding outlier influence; also initializes `"mm"`. |
| `mode` / `refit_mode` | Piecewise linear trend refit mode in `fit_piecewise_trend()` / `detect_turning_points()`. | `"sequential"` (`"huber"` when `robust != "none"`) | `"sequential"` (asymmetric OLS), `"joint"` (continuous OLS hinge-spline), `"huber"` (continuous Huber hinge-spline), `"l1"` (exact continuous $L_1$/LAD LP hinge-spline, design-dependent breakdown — robust to isolated outliers, not 50%). |
| `slope_estimator` | Per-segment trend estimator on `analyze_series()` (`result.segments`). | `"ols"` | `"ols"` (WLS + $t$-CI), `"sen"` (exact Theil–Sen / auto **Akritas–Theil–Sen** when censored + distribution-free CI & Mann–Kendall $p$), `"huber"` (continuous Huber + asymptotic CI), `"l1"` (exact continuous LAD). |
| `mk_correction` | Serial-correlation correction for Mann–Kendall p-values and Sen/ATS CIs (`"ar1"`, `"hamed_rao"`, `"none"`). | `"ar1"` | Keep `"ar1"`; `"none"` is anti-conservative with autocorrelated residuals. Uses rank-based AR(1) bootstrap whenever $n \le 1000$ and $n_{\text{eff}} < 150$ (or $n \le 300$), including censored ATS segments. |
| `trend_ci_method` | `just_decompose()` trend/seasonal band: `"bias_corrected"` (local-quadratic trend + post-selection seasonal refit) or `"model"` (local-linear/selected-only band). | `"bias_corrected"` (`"model"` in parity) | Keep the default; `"model"` under-covers curved trends and multi-harmonic ALLSSA seasonality. |
| `ci_level` | Nominal pointwise confidence & prediction interval level (`just_decompose`, `fit_piecewise_trend`, `detect_turning_points`, `just_monitor`, `analyze_series`). | `0.95` | Set `0.90`, `0.95`, or `0.99`. Propagated automatically across decomposition, turning-point refits, segment slopes, and NRT forecasts. |
| `show_ci` | Toggle shaded confidence / prediction interval ribbons on `.plot()`. | `True` | Pass `result.plot(show_ci=False)` for clean line-only plots. |
