# Validation & Verification Summary

`tspec` is verified across four validation tiers spanning upstream numerical parity, refreshed Monte Carlo calibration benchmarks, literature & compositional stress tests, six real-world public-domain scientific/economic datasets, external library oracles (`statsmodels`/`scipy`), adversarial edge cases, and automated CI tests.

## Tier 1: Numerical Parity Against `Ghaderpour/LSWAVE-SignalProcessing`

- **Oracle commit:** `bd15923b1d4880b799b14fe016de4c1252259013` (cloned locally into gitignored `validation/upstream_ref/`, never committed or packaged).
- **Result:** **41 / 41 checks PASSED** (`rtol=1e-6, atol=1e-8` on continuous values; exact `==` on discrete indices and frequencies).
- **Headline accuracy:**
  - `LSSA` spectrum on `Sim_Geophysics.dat`: max absolute difference `8.19e-16`
  - `ALLSSA(decimal=4)` on `Sim_Geophysics.dat`: exact frequencies `[4.0744, 20.3718, 22.2818]`, residual max diff `4.42e-14`
  - `JUSTjumps` + `JUSTdecompose` on `EVIA_CA.dat` (Ghaderpour & Vujadinovic, 2020a Fig. 14): exact jump indices `[13, 111, 144]`, trend max diff `4.71e-14`
  - `JUSTmonitor` (`parity_mode=True`, both `ALLSSA` & `OLS`) on `EVIA_AB.dat`, `EVIB_AB.dat`, `EVIC_AB.dat`, and `EVIA_CA.dat` (Ghaderpour & Vujadinovic, 2020b): exact `Type`, `Loc`, `Mag`, and `Dir` match across all 8 cases
  - `STPD` + `TPTR` on `DESC-PSInSAR-MonthlyResampled.csv` (Ghaderpour et al., 2024 Fig. 10-D): exact turning points `[32, 55]`, trend max diff `8.88e-15`
  - 1D diagonal and full 2D SPD correlated weight matrices (`P = C_f^{-1}`): max diff `1.55e-15`

See `validation/reports/parity_report.md` in the repository for the full 41-row parity matrix and documented intentional bug fixes.

## Tier 2: Refreshed Synthetic Ground-Truth & Monte Carlo Benchmarks (`validation/bench_mc.py`)

Executed against the current post-hardening codebase (`validation/reports/benchmarks_report.md`, note that Tier 2 uses 25–50 replicates per gate for rapid verification; high-replicate calibration sweeps live in `audit/`):

- **Result:** **26 / 26 benchmark gates PASSED** (`273.3 s` parallelized runtime).
- **Headline results:**
  - **JUST (`J1–J5`, `J1b`):**
    - Null calibration (`J1`, `J1b`): single-frequency `lssa` FPR = `5.0%` ($\alpha=0.05$), calibrated `allssa` scan FPR = **`0.0%`** ($\alpha=0.01$), Bonferroni `just_jumps(season="none")` FPR = **`0.0%`** ($\alpha=0.05$), and Bonferroni `just_jumps(season="allssa")` FPR = **`2.5%`** ($\alpha=0.05$).
    - Jump recovery (`J2`): SNR = 3 recall = `77.5%` (mag RMSE `0.081`), SNR = 6 recall = `97.5%` (mag RMSE `0.066`).
    - Non-integer multi-harmonic anti-leakage (`J3`): `100.0%` 3-frequency recovery; slope RMSE `0.0019` (`allssa`) vs. `0.0211` (fixed integer OLS — 11.1× lower error).
    - Severe irregularity (`J4`, 45% random dropout + 4-month blackout, zero interpolation): zero crashes, `100.0%` jump recall.
    - Heteroscedastic weighting (`J5`): weighted magnitude RMSE `0.0314` (`100%` recall) vs. unweighted `0.1137` (`72%` recall).
  - **STPD (`S1–S5`):**
    - Continuous TP recovery (`S1`): white-noise recall = `100.0%` (timing MAE `0.082 yr`); heavy-tailed $t_5$ recall = `98.0%`.
    - Null calibration (`S2`): empirical FPR = **`2.0%`** ($\alpha=0.01$, calibrated MC, $\text{NDRI} < 0.3$).
    - Jump vs. TP NDRI separation (`S3`): median NDRI = `0.083` (TP) vs. `0.716` (jump); **ROC AUC = `1.000`**; `50/50` datum jumps flagged.
    - 3:1 sampling-density imbalance (`S4`): false-rejection rate `42.0%` (`ndri_mode="l2"`) vs. **`0.0%`** (`ndri_mode="rmse"`).
    - Head-to-head on noisy continuous TPs (`S5`): timing RMSE `0.208 yr` (STPD) vs. `0.426 yr` (JUST, with `1.38` median spurious vertical step).
  - **Censored data (`C1–C6`):**
    - ROS quantile tie-shuffling guard (`C1`): unshuffled raw slope `+0.0415` (spurious trend) vs. `tspec` tie-shuffled `-0.0064 ± 0.0187` (unbiased).
    - 30% left-censored jump magnitude MAE vs. uncensored oracle (`C2`): `tspec` = **`0.0750`** vs. `DL/2` substitution = `0.2755` (**3.7× lower**).
    - Shifting detection limit `<2.0 → <0.2` (`C3`): `30/30` (`100%`) DL shifts detected, **`0/30`** spurious coinciding jumps under default trend-aware Tobit MLE.
    - Non-stationary large jump (`C4`): default trend-aware `scope="global"` MAE = **`0.282`** vs. marginal `scope="windowed"` `1.205`; imputed down-weighting (`C5`) lowest MAE `0.153` at $\gamma=0.5$.
    - Mixed censoring & causal NRT (`C6`): **`0/30`** false DL shifts on constant `<DL`/`>UL` limits; **`0.00e+00`** historical look-ahead leakage in `just_monitor`.
  - **Red noise & combined pipeline (`R1`, `P1`):**
    - AR(1) sweep (`R1`, $\alpha=0.05$): unweighted FPR climbs from `8%` ($\phi=0$) to `52%` ($\phi=0.6$) and `40%` ($\phi=0.8$); supplying $P = C_f^{-1}$ restores calibration to `8% / 10% / 5% / 5%`.
    - End-to-end `analyze_series()` (`P1`, seasonality + jump + TP + gaps + 20% censoring): **`100.0%` jump recovery** and **`100.0%` turning-point recovery**.
  - **Near-Real-Time Monitoring (`M1–M3`):**
    - Sequential-peeking null calibration (`M1`, 25 steps, `horizon=25`, $\alpha=0.05$): unadjusted `parity_mode=True` FPR = **`72.5%`** vs. Šidák+Pocock **`0.0%`**, Šidák+O'Brien–Fleming **`0.0%`**, and calibrated Šidák+Consecutive-2 **`2.5%`**.
    - Disturbance recovery (`M2`, default Šidák+Pocock + predictive covariance, $\alpha=0.01$): **`100.0%` abrupt recall** (median delay `0.160 yr`) and **`100.0%` phenological recall**.
    - Long-horizon forecast-drift null calibration (`M3`, $n_{\text{his}}=48$, $D=96$, $\alpha=0.05$): without forecast covariance **`25.0%`** vs. with predictive covariance **`0.0%`**.
  - **Robust Estimation & Non-Parametric Segment Slopes (`RB1–RB4`):**
    - Spike-contaminated JUST (`RB1`, 5% $12\sigma$ spikes): null FPR drops from `32.0%` (OLS) to **`0.0%`** (`robust="mm"`) with **`100.0%` jump recall** (zero masking).
    - Heavy-tailed $t_3$ + spikes STPD (`RB2`): **`100.0%`** TP recall, continuous $L_1$ slope MAE = **`0.027`** vs. OLS `0.209` (**7.7× lower**), and **`100.0%` NDRI jump separation**.
    - Outlier-contaminated `analyze_series()` (`RB3`): segment slope MAE drops from `6.456` (OLS) to **`0.014`** (`MM+Sen`, **461× lower**) and **`0.022`** (`MM+Huber`), with `86.0%` 95% CI coverage.
    - 25% left-censored + outliers Akritas–Theil–Sen (`RB4`): slope MAE drops from `0.114` (OLS) to **`0.012`** (ATS, **9.5× lower**), with `25/25` automatic ATS selection.

Full quantitative tables are published in `validation/reports/benchmarks_report.md`.

## Tier 2b: Literature & Compositional Stress Tests (`L1–L6`, `SC1–SC3`)

Executed against the current hardened codebase (`validation/bench_literature_and_components.py`, report in `validation/reports/literature_and_components_report.md`): **9 / 9 gates PASSED** (`17.8 s` single-threaded).

| Gate | Benchmark / Scenario | Status | Measured Result |
| :--- | :--- | :---: | :--- |
| `L1` | Ghaderpour et al. (2018) non-integer multi-harmonic anti-leakage on gapped data + jump | **PASS** | `12/12` exact 3-frequency recovery (`[0.73, 1.85, 3.25]`), `0/12` pure-noise FP |
| `L2` | Verbesselt (2010) / Ghaderpour & Vujadinovic (2020a) NDVI wildfire + regrowth | **PASS** | Exact jump (`|Δmag| = 0.042`), bias-corrected trend CI `89%`, seasonal CI `94%` |
| `L3` | Ghaderpour et al. (2024) PS-InSAR multi-bend subsidence + disguised unwrapping jump | **PASS** | Both TPs recovered (`[2017.5, 2019.83]`), $L_1$ max `|Δslope| = 0.43`, unwrapping jump flagged via `NDRI_RMSE` |
| `L4` | Helsel (2012) multi-DL lognormal water quality + lab-upgrade artifact defense | **PASS** | Rubin `ols_mi` MAE `0.005` vs. `DL/2` `0.019` (**3.8× lower**), ATS verified, DL artifact guarded |
| `L5` | Yue et al. (2002) / Hamed & Rao (1998) AR(1) red-noise hydrology ($\rho=0.7$) + spikes | **PASS** | MK FPR `8/30` (raw) → `1/30` (`ar1`); LSSA FPR `14/30` (unweighted) → `1/30` (`weights=C⁻¹`) |
| `L6` | Ghaderpour & Vujadinovic (2020b) NRT monitoring (null / abrupt / phenological) | **PASS** | Calibrated `consecutive_k=2` null FP `0/20`, abrupt & phenological alarms accurately classified |
| `SC1` | Kitchen-sink compositional recovery (gaps + 2× rate shift + 2 harmonics + jump + TP + `<DL` + spikes) | **PASS** | Jump `2017.56`, TP `2020.99`, final Sen slope `-1.36/yr` recovered simultaneously |
| `SC2` | Compound interaction matrix (bend + ALLSSA `step_pvalue`, 4× variance shift + weighted NDRI, quadratic trend CI, 2D SPD robust monotonicity) | **PASS** | All 4 pairwise interaction gates passed (`quad_cov = 100%`) |
| `SC3` | Exact invariance laws & strict online NRT causality | **PASS** | Time-origin shift, affine $y$-scaling, weight scaling, and bit-for-bit NRT causality verified |

## Tier 2c: Real-World Public-Domain Datasets (`RW1–RW6`)

Executed offline against six classic public-domain scientific and economic records stored in `validation/data/` (`validation/bench_public_datasets.py`, report in `validation/reports/public_datasets_report.md`): **6 / 6 gates PASSED** (`0.6 s` single-threaded).

| Gate | Public-Domain Dataset & Domain | Status | Measured Result |
| :--- | :--- | :---: | :--- |
| `RW1` | Nile River annual flow at Aswan (1871–1970; Hydrology, Cobb 1978) | **PASS** | `just_jumps` recovers classic **1899** Aswan drop ($\Delta = -283.9$, $p = 0.0009$); `just_monitor(start=1895)` triggers `"abrupt"` alarm at **1902** |
| `RW2` | NOAA Mauna Loa monthly $\text{CO}_2$ (1958–2026; Atmosphere, 20% gapped + outages) | **PASS** | `allssa` recovers `[1.0, 2.0]` cy/yr; `just_decompose(trend_ci_method="bias_corrected")` matches NOAA to **RMSE = `0.327 ppm`** across 68 yr; STPD traces acceleration **0.72 → 3.41 ppm/yr** |
| `RW3` | Annual Sunspot Numbers (1700–2008; Astrophysics, 30% + 10-yr blackout) | **PASS** | `lssa` recovers exact **10.99-year** Schwabe solar cycle ($f = 0.0910$ cy/yr, `peak_fap = 1.97e-11`) |
| `RW4` | NASA GISTEMP v4 Global Temp Anomaly (1880–2025; Climate Science) | **PASS** | STPD identifies classic **`[1910, 1974]`** turning points and modern warming rate **`+0.208 °C/decade`** |
| `RW5` | US Quarterly Macroeconomics `log(CPI)` (1959–2009; Economics) | **PASS** | STPD nails Great Inflation (**1972.25**: `2.77 → 8.91%/yr`), Volcker disinflation (**1981.50**: `8.91 → 4.10%/yr`), and Great Moderation (**1991.75**: `4.10 → 2.56%/yr`) |
| `RW6` | NOAA/CPC El Niño Monthly SST (1950–2010; Oceanography, 25% gapped) | **PASS** | `lssa` (with `1 & 2` cy/yr projected out) recovers irregular **`5.05-year`** ENSO cycle (`f = 0.1980` cy/yr, `FAP = 1.13e-15`) |

## Tier 2d: External Oracles, Cross-Domain Science & Adversarial Edge Cases (`OR1–OR3`, `D1–D4`, `E1–E8`)

Executed against independent library oracles (`statsmodels`, `scipy`), synthetic multi-disciplinary signals, and adversarial numerical stress tests (`validation/bench_cross_domain_and_edge_cases.py`, report in `validation/reports/cross_domain_and_edge_cases_report.md`): **15 / 15 gates PASSED** (`6.6 s`).

| Gate | Category | Validation Scenario | Status | Measured Result |
| :--- | :--- | :--- | :---: | :--- |
| `OR1` | Oracle | `statsmodels.api.GLS` full 2D AR(1) covariance parity | **PASS** | `max|Δβ| = 8.88e-16`, `max|ΔCov| = 5.55e-17`, `max|Δresid| = 2.89e-15` |
| `OR2` | Oracle | `scipy.stats.theilslopes` slope & Gilbert 95% CI parity | **PASS** | `|Δslope| = 0.00e+00`, `|ΔCI_lo| = 0.00e+00`, `|ΔCI_hi| = 0.00e+00` |
| `OR3` | Oracle | `scipy.signal.lombscargle` peak + exact Beta(1, ν/2) KS test | **PASS** | Exact peak `1.37 vs 1.37`, Beta-null KS `p = 0.588` |
| `D1` | Epidemiology | Seasonal incidence + vaccination shock + decay slope | **PASS** | Rollout `2020.25` ($\Delta=-3.73$), post-slope `-0.42/yr` (true `-0.40/yr`) |
| `D2` | Economics | Bai–Perron level break + growth bend + 3× volatility shift | **PASS** | Jump `2004.1` (true `2004`), weighted-NDRI TP `2014.3` (true `2015`) |
| `D3` | Engineering | Bearing wear RUL onset + 8% 25σ telemetry glitches | **PASS** | STPD `robust="mm"` TP `6.48` (true `6.50`), NRT `consecutive_k=2` alarm `6.75` |
| `D4` | Pharmacology | 37% `<LLOQ` lognormal clearance decay + assay change | **PASS** | Rubin `ols_mi = -0.196` (pop `-0.197`), auto ATS `-0.205` |
| `E1` | Edge Case | Extreme dynamic range ($10^{-9}$ to $10^{+9}$ scale) | **PASS** | Identical index `[60, 60, 60]` and scaled magnitude `2.8274` |
| `E2` | Edge Case | Catastrophic cancellation offsets ($t + 10^6$, $y + 10^8$) | **PASS** | Identical TP index `60`, `|Δdir| = 3.86e-09` |
| `E3` | Edge Case | Exact noiseless step/TP & flat constant signals | **PASS** | Zero div-by-zero; const `(0, 0)`, exact `mag = 3.0000`, `dir = -1.5000` |
| `E4` | Edge Case | Unsorted/duplicate guards + `NaN` bursts + 70% dropout | **PASS** | Explicit `ValueError` guards, `8` NaNs dropped, jump `2020.01` recovered |
| `E5` | Edge Case | Small-$n$ (`n=16`) graceful degradation & guardrail | **PASS** | Default `InsufficientDataError`, explicit small-size finite output + `STPD_SKIPPED` |
| `E6` | Edge Case | Mixed `<DL`/`>UL` + `[L,R]` interval + >80% `CensoringError` | **PASS** | Mixed `0.344`, interval `0.331` (true `0.35`), 85% guardrail raised |
| `E7` | Edge Case | Heavy-tailed $t_2$ + 15% extreme spikes | **PASS** | Max slope error: OLS `0.361` vs $L_1$ **`0.017`** (**21× lower**), Huber **`0.031`** |
| `E8` | Edge Case | Closely spaced double breakpoints | **PASS** | 2 jumps `[2017.51, 2020.53]`; jump+TP `(2018.02, 2020.57)` resolved cleanly |

## Tier 3: Unit, Invariance, Edge-Case & Property-Based Tests

- **182 automated `pytest` tests** (unit, property, robust, audit-regression, and end-to-end executable documentation smoke & recovery tests across all guides and tutorials) including Hypothesis property-based verification of time-origin shift invariance ($t \mapsto t + c$), amplitude scaling equivariance ($y \mapsto k y$), scalar weight invariance ($P \mapsto \lambda P$), extreme baseline/scale numerical stability, Huber/MM-biweight/Theil–Sen/Akritas–Theil–Sen/$L_1$ robustness, causal NRT censoring, degenerate/constant/rank-deficient inputs, unpacked sdist verification, and headless plotting smoke tests.
- Dedicated lite-environment CI job verifying zero `ndimpute`/`pandas` imports for core numerical functions (`pip install --no-deps tspec` with `numpy` + `scipy` only).

## Tier 4: Speed & $O(n)$ Memory Profiling

- **Historical speedups vs. upstream Python reference (pre-MC/bootstrap defaults, see `validation/reports/perf_report.md`):**
  - `allssa(decimal=4)` ($n=150$): `68.6 ms` vs. `437.1 ms` (**6.4× speedup**, gate $\ge 5.0\times$) — **PASS**
  - `stpd` single series ($n=100$, analytical/parity): `4.74 ms` vs. `16.21 ms` (**3.4× speedup** via $O(n)$ cumulative prefix sums); **1,000-series batch in `4.74 s`** (gate $< 10.0\text{ s}$) — **PASS**
  - `just_jumps(ALLSSA)` ($n=150$, pre-MC): `836.6 ms` vs. `1109.6 ms` (**1.3× speedup**) — **PASS**
  - `lssa` (100 freqs, 1D $P$, $n=5{,}000$): `48.7 ms` vs. `39.9 ms` (`0.8×`, slight overhead from full SVD/QR conditioning and orthogonal Schur-complement projection)
- **Current calibrated default single-threaded runtimes (`OMP_NUM_THREADS=1`, $n=200$, `n_mc=999`, `n_boot=499`):**
  - `just_jumps()` (default `allssa` + MC): ~5 s
  - `stpd()` (default `sequential` + bootstrap): ~5–6 s (`refit_mode="l1"`: ~25–30 s)
  - `analyze_series()` (default `allssa` + bootstrap): ~11–12 s (use `band_method="conditional"` / `season="ols"` for fast batch runs)
- **Memory scaling (`tracemalloc`):**
  - 1D diagonal weights peak memory: `1.95 MB` at $n = 1{,}000$ and **`19.40 MB`** at $n = 10{,}000$ (gate $< 50.0\text{ MB}$, strictly $O(n)$ vs. $\ge 800\text{ MB}$ dense $O(n^2)$) — **PASS**

Full historical profiling numbers are published in `validation/reports/perf_report.md`.
