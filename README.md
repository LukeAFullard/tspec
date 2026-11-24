<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/tspec-logo-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/tspec-logo.png">
    <img alt="tspec — irregular time-series change detection & spectral analysis" src="docs/assets/tspec-logo.png" width="540">
  </picture>
</p>

<p align="center">
  <strong>Unevenly sampled time-series change detection, turning points, near-real-time monitoring, and spectral decomposition — with built-in censored-data support via <a href="https://github.com/LukeAFullard/ndimpute"><code>ndimpute</code></a>.</strong>
</p>

<p align="center">
  <a href="https://github.com/LukeAFullard/tspec/actions/workflows/ci.yml"><img src="https://github.com/LukeAFullard/tspec/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://lukeafullard.github.io/tspec/"><img src="https://img.shields.io/badge/docs-MkDocs%20Material-00897B?logo=materialformkdocs&logoColor=white" alt="Documentation"></a>
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-0072B2" alt="Python versions">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-009E73" alt="MIT License"></a>
</p>

---

`tspec` is a clean-room, numerically stable Python implementation of:

- **LSSA & ALLSSA** — Weighted Least-Squares Spectral Analysis and iterative Anti-Leakage LSSA (Ghaderpour & Pagiatakis, 2017; Ghaderpour et al., 2018)
- **JUST** — *Jumps Upon Spectrum and Trend* abrupt jump detection (`just_jumps`) and trend/seasonal/remainder decomposition (`just_decompose`) (Ghaderpour & Vujadinovic, 2020a; Ghaderpour, 2021)
- **JUSTmonitor (v1.1)** — Near-Real-Time (NRT) abrupt & phenological disturbance monitoring (`just_monitor`) with Šidák, Pocock, O'Brien–Fleming, and $k$-consecutive repeated-testing controls (Ghaderpour & Vujadinovic, 2020b)
- **STPD & TPTR** — *Sequential Turning Point Detection* (`detect_turning_points` / `stpd`) with RMSE-NDRI jump discrimination and connected piecewise linear trend fitting (`fit_piecewise_trend`) (Ghaderpour et al., 2024)
- **Censored Time-Series Workflows** — First-class handling of `<LOD`, `>UL`, mixed, and `[L, R]` interval observations via [`ndimpute`](https://github.com/LukeAFullard/ndimpute) with automated temporal-order protection, lab detection-limit shift warnings, and multi-draw stability reporting.

All algorithms work natively on **irregular, gappy timestamps with zero interpolation**.

---

## Guided Two-Stage Workflow (`analyze_series`)

Real environmental and geophysical records rarely have just one type of change. `tspec.analyze_series()` runs the recommended **JUST → STPD** workflow automatically on raw, gappy records: it separates seasonality and abrupt step-changes first, then searches the adjusted residual trend for smooth continuous turning points.

![Guided JUST -> STPD combined workflow result](docs/assets/figures/hero_combined_workflow.png)

```python
import numpy as np
import tspec

# 10-year irregular record with a 4-month blackout gap, seasonality,
# an abrupt jump at 2018.5, and a smooth slope bend at 2021.8
rng = np.random.default_rng(42)
t = np.sort(rng.uniform(2015.0, 2025.0, size=220))
t = t[~((t > 2016.9) & (t < 2017.25))]  # 4-month blackout (zero interpolation!)
trend = np.where(t < 2018.5, 4.0, 6.4) + np.where(t > 2021.8, -1.4 * (t - 2021.8), 0.0)
y = trend + 0.85 * np.cos(2 * np.pi * t) + 0.30 * np.sin(4 * np.pi * t) + rng.normal(0.0, 0.16, size=t.size)

result = tspec.analyze_series(t, y)

print(result.summary())   # Structured technical table
print(result.explain())   # Plain-English interpretation & diagnostic caveats
df = result.to_dataframe()
fig, axes = result.plot()
```

---

## Which Method Should I Use?

![Abrupt step jump (JUST) vs continuous turning point (STPD)](docs/assets/figures/concept_jump_vs_tp.png)

| Your Question | Best Function | Why |
| :--- | :--- | :--- |
| *"Did a fire, spill, flood, or sensor swap cause a sudden step-change in a seasonal, gappy record?"* | `just_jumps()` + `just_decompose()` | Separates seasonality and piecewise linear trends simultaneously without filling gaps. |
| *"Is my landslide, aquifer, or concentration trend gradually accelerating or slowing down (no sudden cliff)?"* | `detect_turning_points()` (`stpd`) | Fits connected bent lines and checks **NDRI** ("the jump alarm bell") to reject disguised sudden jumps. |
| *"I have a stable historical baseline and want to catch a new disturbance as incoming observations arrive."* | `just_monitor()` | Forecasts the historical model and runs sequential LSSA with built-in repeated-testing control (Šidák + Pocock / O'Brien–Fleming / consecutive-$k$). |
| *"My series has seasonality, maybe a sudden jump, AND maybe a smooth trend bend later — where do I start?"* | `analyze_series()` | Runs the guided two-stage JUST → STPD workflow automatically with scale-free defaults. |
| *"My lab data contains `<0.05`, `>10`, or `[L, R]` censored values."* | Pass strings or `censor_status` + `CensoredConfig` to any function above | Imputes via `ndimpute` without `DL/2` bias and warns if a lab detection-limit upgrade mimics a jump. |
| *"I only want one overall monotonic up/down trend (no breakpoints)."* | Mann–Kendall / Sen's slope (external) | Simpler and higher power when you don't need to locate breakpoints. |

---

## Near-Real-Time Disturbance Monitoring (`just_monitor`, v1.1)

Scanning 35 frequencies on every incoming observation without multiplicity control inflates cumulative false alarms rapidly (**72.5%** null false-alarm rate over 25 steps at $\alpha = 0.05$ in unadjusted parity mode). `tspec.just_monitor()` defaults to **Šidák multi-frequency + Pocock sequential alpha-spending** (**0.0%** empirical null FPR over 25 steps, **100%** recall on abrupt & phenological disturbances), while also offering Lan–DeMets **O'Brien–Fleming**, open-ended **`consecutive_k`**, and `parity_mode=True` for exact parity with Ghaderpour & Vujadinovic (2020b).

![Near-real-time monitoring with sequential multiplicity control](docs/assets/figures/tutorial_nrt_monitoring.png)

```python
from tspec import just_monitor

res = just_monitor(
    t,
    evi,
    start=2022.5,                  # Index or timestamp / decimal year
    season="allssa",               # Historical trend + anti-leakage seasonal model
    alpha=0.01,
    freq_correction="sidak",       # Per-step adjustment across scanned frequencies
    sequential_control="pocock",   # Or "obrien_fleming", "consecutive_k", "none"
)
print(res.summary())
print(res.explain())
fig, axes = res.plot()
```

---

## First-Class Censored Environmental Data (`<DL`, `>UL`, `[L, R]`)

Environmental monitoring records routinely contain left-censored non-detects (`"<0.65"`) and historical lab method upgrades where detection limits drop mid-record. `tspec` integrates [`ndimpute`](https://github.com/LukeAFullard/ndimpute) directly, protects against ROS quantile temporal trend injection, down-weights imputed points optionally, and automatically flags lab detection-limit upgrades so analytical changes are never mistaken for environmental jumps.

![Censored water quality record with ndimpute ROS imputation and DL-shift defense](docs/assets/figures/tutorial_water_quality_censored.png)

```python
from tspec import CensoredConfig, just_jumps, just_decompose

cfg = CensoredConfig(
    method="ros",              # Robust ROS via ndimpute (avoids DL/2 bias)
    dist="lognormal",
    impute_type="stochastic",  # Time-order safe (prevents ROS quantile trend artifact)
    detect_dl_shifts=True,     # Flags lab detection-limit upgrades automatically
    recensor_check=True,       # Tests whether coinciding jumps vanish at common DL
    random_state=42,
)

# Pass "<0.65" strings directly (or numeric arrays + censor_status)
jumps = just_jumps(dates, nitrate_strings, season="ols", mag_th=0.8, jump_alpha=0.01, censoring=cfg)
decomp = just_decompose(dates, nitrate_strings, jump_indices=[j.index for j in jumps.jumps], season="ols", censoring=cfg)
print(jumps.explain())
```

---

## Installation

```bash
# Standard install (includes numpy, scipy, ndimpute & pandas)
pip install tspec

# With diagnostic plotting support (matplotlib)
pip install "tspec[plot]"
```

### Lite Installation (`numpy` + `scipy` only)
`ndimpute`, `pandas`, and `matplotlib` are imported strictly lazily. If you only need the core numerical spectral / JUST / STPD / monitoring engine:

```bash
pip install numpy scipy
pip install --no-deps tspec
```

---

## Verification & Performance

`tspec` is verified across four automated validation tiers (full reports in [`validation/reports/`](validation/reports/)):

- **Tier 1 — Upstream Numerical Parity:** **41 / 41 checks PASSED** against `Ghaderpour/LSWAVE-SignalProcessing` (`LSSA`, `ALLSSA`, `JUSTjumps`, `JUSTdecompose`, `JUSTmonitor`, `STPD`, `TPTR`; continuous max diffs `~1e-15`, exact discrete indices).
- **Tier 2 — Monte Carlo Benchmarks:** **22 / 22 statistical gates PASSED** across JUST (including null-selected `season="allssa"` at `5.0%` FPR), STPD (exact negative-covariance SE, NDRI ROC AUC `1.000`; 3:1 density imbalance false-rejection rate dropped from `42.0%` → `0.0%`), censored data (`ndimpute` ROS vs `DL/2`, `100%` DL-shift defense, zero causal NRT leakage), AR(1) red-noise calibration, and `just_monitor` sequential-peeking & long-horizon forecast-drift control (`72.5%` unadjusted → `0.0%` Šidák+Pocock; `25.0%` → `0.0%` with predictive covariance).
- **Tier 3 — Unit, Property & Docs Tests:** 75 `pytest` + Hypothesis invariance & audit-regression tests plus automated execution of every tutorial code block and unpacked PyPI sdist verification.
- **Tier 4 — Speed & $O(n)$ Memory:** `allssa(decimal=4)` **7.4× faster** than upstream Python; **1,000-series STPD batch in 4.83 s**; 1D diagonal weights scale strictly in $O(n)$ memory (`19.4 MB` peak at $n = 10{,}000$).

---

## Documentation

Full practitioner guides, zero-math concepts, parameter cheat sheet, domain tutorials (water quality, satellite NDVI wildfire, ground subsidence, near-real-time monitoring, combined workflow), and mathematical/API references live in [`docs/`](docs/) (built with MkDocs Material):

```bash
pip install -e ".[docs]"
mkdocs serve
```

---

## License & Citation

Released under the **MIT License**. Written as a strict clean-room implementation directly from the peer-reviewed literature (see [`CITATION.cff`](CITATION.cff) for full BibTeX and DOI records for Ghaderpour et al., 2017–2024; Lan & DeMets, 1983; Helsel, 2012; Turnbull, 1976; and `ndimpute`).