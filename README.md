# tspec

**Unevenly sampled time-series change detection, turning points, and spectral decomposition — with built-in censored-data support via [`ndimpute`](https://github.com/LukeAFullard/ndimpute).**

`tspec` is a clean-room, numerically stable Python implementation of:

- **LSSA & ALLSSA** — Weighted Least-Squares Spectral Analysis and iterative Anti-Leakage LSSA (Ghaderpour & Pagiatakis, 2017; Ghaderpour et al., 2018)
- **JUST** — *Jumps Upon Spectrum and Trend* abrupt jump detection and trend/seasonal/remainder decomposition (Ghaderpour & Vujadinovic, 2020; Ghaderpour, 2021)
- **STPD & TPTR** — *Sequential Turning Point Detection* with NDRI jump discrimination and connected piecewise linear trend fitting (Ghaderpour et al., 2024)
- **Censored Time-Series Workflows** — First-class handling of `<LOD`, `>UL`, mixed, and `[L, R]` interval observations via `ndimpute` with automated temporal-order protection, detection-limit shift warnings, and multi-draw stability reporting.

All algorithms work directly on **irregular, gappy timestamps with zero interpolation**.

---

## Which Method Should I Use?

| Your Question | Best Function | Why |
| :--- | :--- | :--- |
| *"Did a fire, spill, flood, or sensor swap cause a sudden step-change in a seasonal, gappy record?"* | `just_jumps()` + `just_decompose()` | Separates seasonality and piecewise linear trends simultaneously without filling gaps. |
| *"Is my landslide, aquifer, or concentration trend gradually accelerating or slowing down (no sudden cliff, little seasonality)?"* | `detect_turning_points()` (`stpd`) | Fits connected bent lines and checks **NDRI** ("the jump alarm bell") to reject disguised sudden jumps. |
| *"My series has seasonality, maybe a sudden jump, AND maybe a smooth trend bend later — where do I start?"* | `analyze_series()` | Runs the recommended two-stage JUST → STPD workflow automatically with scale-free defaults. |
| *"My lab data contains `<0.05` or `>10` censored values."* | Pass strings or `censor_status` + `CensoredConfig` to any function above | Imputes via `ndimpute` without `DL/2` bias and warns if a lab detection-limit upgrade mimics a jump. |
| *"I only want one overall monotonic up/down trend (no breakpoints)."* | Mann–Kendall / Sen's slope (external) | Simpler and higher power when you don't need to locate breakpoints. |

---

## Installation

```bash
# Standard install (includes numpy, scipy, ndimpute & pandas)
pip install tspec

# With diagnostic plotting support (matplotlib)
pip install "tspec[plot]"
```

### Lite Installation (`numpy` + `scipy` only)
`ndimpute` and `pandas` are imported strictly lazily. If you do not need censored-data imputation or `.to_dataframe()`, you can install a lightweight core environment:

```bash
pip install numpy scipy
pip install --no-deps tspec
```

---

## Quickstart

```python
import numpy as np
import tspec

# Irregular 8-year record with seasonality, an abrupt jump at year 3.5, and a smooth bend at year 6.0
rng = np.random.default_rng(42)
t = np.sort(rng.uniform(2016.0, 2024.0, size=180))
trend = np.where(t < 2019.5, 4.0, 6.5) + np.where(t > 2022.0, -1.5 * (t - 2022.0), 0.0)
y = trend + 0.8 * np.cos(2 * np.pi * t) + rng.normal(0.0, 0.15, size=t.size)

# Guided two-stage JUST -> STPD workflow
result = tspec.analyze_series(t, y)

print(result.summary())   # Technical summary table
print(result.explain())   # Plain-English interpretation & diagnostic caveats
df = result.to_dataframe()
fig, axes = result.plot()
```

### Working with Censored Environmental Data (`<LOD`)

```python
from tspec import CensoredConfig, analyze_series

# String markers ("<0.05", ">10") or explicit numeric + censor_status arrays work directly
result = analyze_series(
    dates,
    lab_values,
    censoring=CensoredConfig(
        method="ros",              # Robust ROS via ndimpute
        impute_type="stochastic",  # Time-order safe (avoids ROS quantile trend artifact)
        n_stability_draws=20,      # Reports breakpoint stability across tail draws
        detect_dl_shifts=True,     # Flags jumps coinciding with lab DL upgrades
    ),
)
print(result.explain())
```

---

## License & Citation

Released under the **MIT License**. Built as a clean-room implementation directly from the peer-reviewed publications (see [`CITATION.cff`](CITATION.cff) for full BibTeX / DOI references to Ghaderpour et al., 2017–2024; Helsel, 2012; Turnbull, 1976; and `ndimpute`).