# Parameter Cheat Sheet

A plain-English translation of every key parameter in `tspec`, its default value, what happens when you raise or lower it, and practical rules of thumb.

## Windowing Parameters (`just_jumps`, `just_decompose`, `detect_turning_points`)

| Parameter | Plain-English Meaning | Default | Make It Larger | Make It Smaller | Rule of Thumb |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `size` | How many data points sit inside each sliding window (`window_mode="count"`). | JUST: `3 * M`<br>STPD: `5 * M` | Smoother trends, more statistical power, misses closely spaced changes. | Tracks rapid changes, noisier seasonal/slope estimates. | Pick **~3 years** of observations for JUST (3 full seasonal cycles) and **~4–5 years** for STPD. |
| `step` | How many points the window slides forward each move. | `M` (~1 year) | Faster runtime, fewer overlapping checks per breakpoint. | Slower runtime, higher occurrence counts and finer overlap. | `size / 3` (JUST) or `size / 5` (STPD) works well. |
| `window_mode` | Whether windows count *observations* (`"count"`) or *calendar years* (`"duration"`). | `"count"` | — | — | Use `"duration"` whenever sampling frequency changes $> 2\times$ over the record. |
| `window_duration` | Calendar span of each window when `window_mode="duration"`. | `3.0` (JUST)<br>`5.0` (STPD) | More stable seasonal/slope fit over sparse intervals. | Better resolution for closely spaced events. | Match 3.0 years for seasonal JUST, 4–5 years for STPD. |

## Spectral & Significance Parameters (`lssa`, `allssa`, `just_jumps`)

| Parameter | Plain-English Meaning | Default | Make It Larger | Make It Smaller | Rule of Thumb |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `alpha` | Significance threshold for accepting seasonal cycles (ALLSSA) or slope changes (STPD). | `0.01` | Detects weaker cycles/bends, higher false-positive risk. | Stricter; fewer false alarms, may miss subtle changes. | `0.01` is a safe default; try `0.001` under moderate red noise. |
| `omega` | Candidate seasonal frequencies to search (cycles per unit time; cycles/yr for dates). | ALLSSA: `0.8..3.8`<br>OLS: `[1, 2, 3, 4]` | Searches faster/shorter sub-annual cycles (risk of overfitting sparse data). | Restricts search to annual/semi-annual waves only. | Default covers annual (`1`), semi-annual (`2`), and tri-annual (`3`) waves cleanly. |
| `mag_th` / `dir_th` | Minimum jump size / slope change kept by `just_jumps()`. | `0.05` / `0.01` *(NDVI scale)* | Ignores small jumps; keeps only large domain-meaningful shifts. | Keeps smaller shifts (more sensitive to noise). | Always match your measurement units (e.g., mg/L or mm), OR use `jump_alpha=0.05` / `analyze_series()`. |
| `jump_alpha` | Statistical $F$-test cutoff for `just_jumps()` (scale-free!). | `None` in `just_jumps()`<br>`0.05` in `analyze_series()` | Allows weaker candidate jumps through. | Keeps only strongly significant jumps. | Set `jump_alpha=0.05` + `bonferroni_window=True` when you don't want to hand-tune `mag_th`. |
| `jump_th` / `tp_th` | Minimum time gap required between two separate jumps (`jump_th`) or turning points (`tp_th`). | `0.75` (JUST)<br>`1.0` (STPD) | Merges nearby events into a single stronger breakpoint. | Allows closely spaced consecutive changes. | `0.75–1.0` yr prevents a single disturbance from being double-counted. |

## Turning-Point Quality Filters (`detect_turning_points` / `stpd`)

| Parameter | Plain-English Meaning | Default | Make It Larger | Make It Smaller | Rule of Thumb |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `ndri_th` | Maximum residual imbalance allowed ("jump alarm bell"). | `0.30` | Accepts slightly jump-like bends. | Rejects anything even mildly asymmetric as a jump. | `0.25–0.35` is well calibrated (Ghaderpour et al., 2024). |
| `ndri_mode` | How NDRI compares left vs. right residuals. | `"rmse"` (`"l2"` in parity) | — | — | Keep `"rmse"` for uneven sampling; use `parity_mode=True` for Ghaderpour (2024) exact parity. |
| `snr_th` / `snr_mode` | Minimum signal-to-noise ratio for a turning point. | `1.0` / `"centered"` | Requires stronger trend relative to scatter. | Allows subtler bends in noisy records. | Keep `"centered"` for data with non-zero baselines (pH, temperature, elevation). |
| `margin` | Points ignored near each window edge so slopes aren't fit to 1–2 points. | `max(2, min(M, size // 5))` | More stable slopes; cannot detect bends near record edges. | Detects bends closer to window edges, noisier slopes. | Always $\ge 2$ (`tspec` enforces this automatically). |

## Censored Data (`CensoredConfig`)

| Parameter | Plain-English Meaning | Default | Guidance |
| :--- | :--- | :--- | :--- |
| `method` | Imputation engine inside `ndimpute` (`"ros"` or `"parametric"`). | `"ros"` | Use `"ros"` for left-censored environmental data (`<DL`); use `"parametric"` for right-, mixed-, or interval-censored data. |
| `impute_type` | How censored points are filled (`"stochastic"`, `"mean"`, `"quantile"`). | `"stochastic"` | Keep `"stochastic"` (seeded) — avoids variance shrinkage (`"mean"`) and temporal rank bias (unshuffled `"quantile"`). |
| `scope` | Whether one distribution is fitted globally (`"global"`), per window (`"windowed"`), or per season (`"seasonal_stratified"`). | `"global"` | Switch to `"windowed"` if a huge jump shifts censoring from >50% to <5%, or `"seasonal_stratified"` for intense seasonal censoring. |
| `n_stability_draws` | How many extra imputation realizations are run to measure breakpoint stability. | `20` in `analyze_series()`<br>`0` in single calls | Set `>= 20` when censoring exceeds 20%; use `stability_mode="bootstrap"` for full parameter-uncertainty propagation. |
| `detect_dl_shifts` / `recensor_check` | Guard against fake jumps from laboratory detection-limit upgrades. | `True` / `False` | Keep `detect_dl_shifts=True`; turn on `recensor_check=True` whenever a `DL_SHIFT_COINCIDENCE` warning appears. |
