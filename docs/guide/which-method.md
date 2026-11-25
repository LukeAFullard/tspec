# Which Method Should I Use?

Choosing the right method takes 30 seconds once you answer three plain-English questions about your data:

1. Are you looking for **one overall trend**, or **changes in behavior over time**?
2. Do changes look like **sudden cliffs/jumps**, or **smooth bends in speed**?
3. Does your data have **seasonal waves** (winter/summer cycles) or **censored lab values** (`<0.05`)?

![Abrupt step jump (JUST) vs continuous turning point (STPD)](../assets/figures/concept_jump_vs_tp.png)

## Decision Flowchart

```mermaid
flowchart TD
    Start["Start: Unevenly spaced time series"] --> Q0{"Only want ONE overall trend direction (no changepoints or jumps)?"}
    Q0 -- "Yes" --> MK["Use a monotonic trend test (Mann-Kendall / Sen's slope or Seasonal Kendall)"]
    Q0 -- "No, suspect changes over time" --> QC{"Contains censored lab values (e.g., <0.05 or >10)?"}
    QC -- "Yes" --> Cens["Enable CensoredConfig via ndimpute and check DL-shift warnings"]
    QC -- "No" --> Q1
    Cens --> Q1{"What kind of change are you looking for?"}

    Q1 -- "Incoming near-real-time alert vs. a stable historical baseline" --> NRT["Use JUSTmonitor (v1.1): just_monitor() with Sidak + Pocock / consecutive_k control"]
    Q1 -- "Sudden leaps or offsets (fires, sensor swap, spills, policy shock)" --> Q1S["Use JUST: just_jumps() + just_decompose()"]
    Q1 -- "Smooth change in speed/slope without sudden leaps" --> Q2{"Strong seasonal cycles (winter/summer waves)?"}
    Q1 -- "Both, or not sure" --> Pipe["Use Guided Two-Stage Pipeline: analyze_series() (JUST removes seasonality & jumps, then STPD finds turning points)"]

    Q2 -- "No (minimal seasonality)" --> STPD["Use STPD: detect_turning_points()"]
    Q2 -- "Yes (strong seasonality)" --> Pipe

    STPD --> NDRI{"Check NDRI in .explain(): Any candidates flagged with high NDRI (> 0.30)?"}
    NDRI -- "Yes — jump alarm bell!" --> Pipe
    NDRI -- "No — low NDRI (~0)" --> Done["Done! Inspect .explain() & .plot()"]
    Q1S --> Done
    NRT --> Done
    Pipe --> Done
```

## Plain-English Method Comparison

| Your Question / Scenario | Best Tool | Why (Zero Jargon) |
| :--- | :--- | :--- |
| *"Did a wildfire, flood, spill, or sensor swap cause a sudden drop or jump — even though my series has summer/winter waves and missing months?"* | **`just_jumps()` + `just_decompose()`** | Fits local straight trend lines and seasonal cycles simultaneously **without filling gaps**, so abrupt offsets stand out clearly. |
| *"Is my landslide, ground subsidence, or groundwater trend gradually accelerating or slowing down — with no sudden cliff and minimal seasonality?"* | **`detect_turning_points()` (`stpd`)** | Fits connected straight lines that bend smoothly at a turning point, and uses **NDRI** to make sure it isn't fooled by a hidden sudden jump. |
| *"I have a stable historical baseline and want to raise an alert as soon as new incoming data deviate from normal behavior."* | **`just_monitor()`** | Forecasts the historical trend + seasonal baseline forward and scans incoming forecast residuals sequentially with built-in repeated-testing control. |
| *"My river or satellite record has seasonal cycles, maybe a sudden disturbance, AND maybe a later change in recovery rate — where should I start?"* | **`analyze_series()`** | Runs the recommended two-stage sequence automatically: JUST extracts seasonality and vertical jumps first, then STPD finds smooth trend bends. |
| *"My water-quality data has `<0.05` lab non-detects or `>100` upper limits."* | **Any function above + `CensoredConfig`** | Delegates to `ndimpute` for unbiased censored imputation (avoiding `DL/2` bias) and automatically warns if a lab detection-limit upgrade looks like a fake jump. |
| *"My record has extreme spikes, stormflow glitches, or heavy-tailed noise — and I want robust segment slopes (or Sen / Akritas–Theil–Sen slopes with CIs)."* | **Add `robust="huber"` or `robust="mm"` (and `slope_estimator="sen"` / `"l1"` / `"huber"` in `analyze_series()`)** | Uses convex Huber IRLS (`"huber"`) or Huber-started Tukey biweight (`"mm"`, zero influence for extreme $>5\sigma$ spikes) + Schrader–Hettmansperger robust $F$-tests during detection (without masking genuine jumps) and exact Theil–Sen / Akritas–Theil–Sen, $L_1$, or Huber segment slopes afterward. |
| *"I just want to know whether concentrations went up or down overall over 10 years, without locating breakpoints."* | **Mann–Kendall / Sen's slope** *(e.g., `MannKS` or `analyze_series(..., detect_jumps=False, detect_turning_points=False, slope_estimator="sen")`)* | Simpler, nonparametric, and higher statistical power when you do not need to detect change dates or separate jumps from bends. |
