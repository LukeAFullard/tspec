# Validation & Verification Summary

!!! warning "Superseded calibration claims"
    The Tier 2 Monte Carlo gates below used 25–50 replicates and permissive thresholds, and several of the tested procedures have since changed. Subsequent calibration audits (`audit/` in the repository) found that the raw JUST screening p-values, the per-frequency ALLSSA rule, the model-based decomposition trend band, and marginal censored imputation were badly miscalibrated. Those defaults have been replaced (calibrated jump selection, family-wise ALLSSA significance, bias-corrected trend & seasonal bands, time-conditional Tobit imputation with Rubin pooling, rank-based AR(1)-corrected Mann–Kendall/ATS, and local+joint step-offset bootstrap). Treat the Tier 2 numbers below as historical.

`tspec` is validated across four verification tiers.

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

## Tier 2: Synthetic Ground-Truth & Monte Carlo Benchmarks

- **Result:** **26 / 26 benchmark gates PASSED** (368.4 s total runtime).
- **Headline results:**
  - **JUST (`J1–J5`, `J1b`):**
    - Null calibration (`J1`, `J1b`): single-frequency `lssa` FPR = `5.0%` ($\alpha=0.05$), `allssa` scan FPR = `7.5%` (default $\alpha=0.01$), Bonferroni `just_jumps(season="none")` FPR = `2.5%` ($\alpha=0.05$), and Bonferroni `just_jumps(season="allssa")` FPR = **`5.0%`** ($\alpha=0.05$, with null-selected harmonics).
    - Jump recovery (`J2`): SNR = 3 recall = `77.5%`, SNR = 6 recall = `97.5%`.
    - Non-integer multi-harmonic anti-leakage (`J3`): `100.0%` 3-frequency recovery; slope RMSE `0.0018` (`allssa`) vs. `0.0211` (fixed integer OLS — 11.7× lower error).
    - Severe irregularity (`J4`, 45% random dropout + 4-month blackout, zero interpolation): zero crashes, `100.0%` jump recall.
    - Heteroscedastic weighting (`J5`): weighted magnitude RMSE `0.0314` (`100%` recall) vs. unweighted `0.1131` (`75%` recall).
  - **STPD (`S1–S5`):**
    - Continuous TP recovery (`S1`): white-noise recall = `100.0%` (timing MAE `0.085 yr`); heavy-tailed $t_5$ recall = `98.0%`.
    - Null calibration (`S2`): empirical FPR = **`4.0%`** ($\alpha=0.01$, $\text{SNR}>1$, $\text{NDRI}<0.3$, exact negative-covariance fixed-knot SE).
    - Jump vs. TP NDRI separation (`S3`): median NDRI = `0.083` (TP) vs. `0.716` (jump); **ROC AUC = `1.000`**; `50/50` datum jumps flagged.
    - 3:1 sampling-density imbalance (`S4`): false-rejection rate `42.0%` (`ndri_mode="l2"`) vs. **`0.0%`** (`ndri_mode="rmse"`).
    - Head-to-head on noisy continuous TPs (`S5`): timing RMSE `0.210 yr` (STPD) vs. `0.426 yr` (JUST, with `1.38` median spurious vertical step).
  - **Censored data (`C1–C6`):**
    - ROS quantile tie-shuffling guard (`C1`): unshuffled raw slope `+0.0415` (spurious trend) vs. `tspec` tie-shuffled `-0.0065 ± 0.0184` (unbiased).
    - 30% left-censored jump magnitude MAE vs. uncensored oracle (`C2`): `ndimpute` ROS = `0.1000` vs. `DL/2` substitution = `0.2755`.
    - Shifting detection limit `<2.0 → <0.2` (`C3`): `30/30` (`100%`) DL shifts detected, `17/17` (`100%`) coinciding false jumps flagged.
    - Non-stationary large jump (`C4`): `scope="windowed"` MAE `1.242` vs. `scope="global"` `1.415`; imputed down-weighting (`C5`) lowers MAE monotonically from `0.606` ($\gamma=1.0$) to `0.206` ($\gamma=0.25$).
    - Mixed censoring & causal NRT (`C6`): **`0/30`** false DL shifts on constant `<DL`/`>UL` limits; **`0.00e+00`** historical look-ahead leakage in `just_monitor`.
  - **Red noise & combined pipeline (`R1`, `P1`):**
    - AR(1) sweep (`R1`, $\alpha=0.05$): unweighted FPR climbs from `8%` ($\phi=0$) to `52%` ($\phi=0.6$) and `40%` ($\phi=0.8$); supplying $P = C_f^{-1}$ restores calibration to `8% / 10% / 5% / 5%`.
    - End-to-end `analyze_series()` (`P1`, seasonality + jump + TP + gaps + 20% censoring): **`100.0%` jump recovery** and **`100.0%` turning-point recovery**.
  - **Near-Real-Time Monitoring (`M1–M3`, v1.1):**
    - Sequential-peeking null calibration (`M1`, 25 steps, $\alpha=0.05$): unadjusted `parity_mode=True` FPR = **`72.5%`** vs. Šidák+Pocock **`0.0%`**, Šidák+O'Brien–Fleming **`0.0%`**, and Šidák+Consecutive-2 **`2.5%`**.
    - Disturbance recovery (`M2`, default Šidák+Pocock + predictive covariance, $\alpha=0.01$): **`100.0%` abrupt recall** (median delay `0.160 yr`) and **`100.0%` phenological recall**.
    - Long-horizon forecast-drift null calibration (`M3`, $n_{\text{his}}=48$, $D=96$, $\alpha=0.05$): without forecast covariance **`25.0%`** vs. with predictive covariance **`0.0%`**.
  - **Robust Estimation & Non-Parametric Segment Slopes (`RB1–RB4`):**
    - Spike-contaminated JUST (`RB1`, 5% $12\sigma$ spikes): null FPR drops from `40.0%` (OLS) to **`12.0%`** (`robust="mm"`) with **`100.0%` jump recall** (zero masking).
    - Heavy-tailed $t_3$ + spikes STPD (`RB2`): `96.0%` TP recall, continuous $L_1$ slope MAE = **`0.047`** vs. OLS `0.209` (**4.4× lower**), and **`100.0%` NDRI jump separation**.
    - Outlier-contaminated `analyze_series()` (`RB3`): segment slope MAE drops from `8.105` (OLS) to **`0.017`** (`MM+Sen`, **477× lower**) and **`0.024`** (`MM+Huber`), with `88.0%` 95% CI coverage.
    - 25% left-censored + outliers Akritas–Theil–Sen (`RB4`): slope MAE drops from `0.104` (OLS) to **`0.012`** (ATS, **8.7× lower**), with `25/25` automatic ATS selection.

Full quantitative tables are published in `validation/reports/benchmarks_report.md`.

## Tier 3: Unit, Invariance, Edge-Case & Property-Based Tests

- Automated `pytest` tests (unit, property, robust, audit-regression and end-to-end documentation smoke tests) including Hypothesis property-based verification of time-origin shift invariance ($t \mapsto t + c$), amplitude scaling equivariance ($y \mapsto k y$), scalar weight invariance ($P \mapsto \lambda P$), extreme baseline/scale numerical stability, Huber/MM-biweight/Theil–Sen/Akritas–Theil–Sen/$L_1$ robustness, causal NRT censoring, degenerate/constant/rank-deficient inputs, unpacked sdist verification, and headless plotting smoke tests.
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
