# Validation & Verification Summary

`tspec` is validated across four independent verification tiers prior to release.

## Tier 1: Numerical Parity Against `Ghaderpour/LSWAVE-SignalProcessing`

- **Oracle commit:** `bd15923b1d4880b799b14fe016de4c1252259013` (cloned locally into gitignored `validation/upstream_ref/`, never committed or packaged).
- **Result:** **41 / 41 checks PASSED** (`rtol=1e-6, atol=1e-8` on continuous values; exact `==` on discrete indices and frequencies).
- **Headline accuracy:**
  - `LSSA` spectrum on `Sim_Geophysics.dat`: max absolute difference `7.91e-16`
  - `ALLSSA(decimal=4)` on `Sim_Geophysics.dat`: exact frequencies `[4.0744, 20.3718, 22.2818]`, residual max diff `4.53e-14`
  - `JUSTjumps` + `JUSTdecompose` on `EVIA_CA.dat` (Ghaderpour & Vujadinovic, 2020a Fig. 14): exact jump indices `[13, 111, 144]`, trend max diff `4.71e-14`
  - `JUSTmonitor` (`parity_mode=True`, both `ALLSSA` & `OLS`) on `EVIA_AB.dat`, `EVIB_AB.dat`, `EVIC_AB.dat`, and `EVIA_CA.dat` (Ghaderpour & Vujadinovic, 2020b): exact `Type`, `Loc`, `Mag`, and `Dir` match across all 8 cases
  - `STPD` + `TPTR` on `DESC-PSInSAR-MonthlyResampled.csv` (Ghaderpour et al., 2024 Fig. 10-D): exact turning points `[32, 55]`, trend max diff `8.88e-15`
  - 1D diagonal and full 2D SPD correlated weight matrices (`P = C_f^{-1}`): max diff `1.44e-15`

See `validation/reports/parity_report.md` in the repository for the full 41-row parity matrix and documented intentional bug fixes.

## Tier 2: Synthetic Ground-Truth & Monte Carlo Benchmarks

- **Result:** **19 / 19 benchmark gates PASSED** (73.1 s total runtime).
- **Headline results:**
  - **JUST (`J1–J5`):**
    - Null calibration (`J1`): single-frequency `lssa` FPR = `5.0%` ($\alpha=0.05$), `allssa` scan FPR = `7.5%` (default $\alpha=0.01$), Bonferroni `just_jumps` FPR = `2.5%` ($\alpha=0.05$).
    - Jump recovery (`J2`): SNR = 3 recall = `77.5%`, SNR = 6 recall = `97.5%`.
    - Non-integer multi-harmonic anti-leakage (`J3`): `100.0%` 3-frequency recovery; slope RMSE `0.0018` (`allssa`) vs. `0.0211` (fixed integer OLS — 11.7× lower error).
    - Severe irregularity (`J4`, 45% random dropout + 4-month blackout, zero interpolation): zero crashes, `100.0%` jump recall.
    - Heteroscedastic weighting (`J5`): weighted magnitude RMSE `0.0314` (`100%` recall) vs. unweighted `0.1131` (`75%` recall).
  - **STPD (`S1–S5`):**
    - Continuous TP recovery (`S1`): white-noise recall = `100.0%` (timing MAE `0.085 yr`); heavy-tailed $t_5$ recall = `98.0%`.
    - Null calibration (`S2`): empirical FPR = `8.0%` ($\alpha=0.01$, $\text{SNR}>1$, $\text{NDRI}<0.3$).
    - Jump vs. TP NDRI separation (`S3`): median NDRI = `0.083` (TP) vs. `0.716` (jump); **ROC AUC = `1.000`**; `50/50` datum jumps flagged.
    - 3:1 sampling-density imbalance (`S4`): false-rejection rate `42.0%` (`ndri_mode="l2"`) vs. **`0.0%`** (`ndri_mode="rmse"`).
    - Head-to-head on noisy continuous TPs (`S5`): timing RMSE `0.210 yr` (STPD) vs. `0.426 yr` (JUST, with `1.38` median spurious vertical step).
  - **Censored data (`C1–C5`):**
    - ROS quantile tie-shuffling guard (`C1`): unshuffled raw slope `+0.0415` (spurious trend) vs. `tspec` tie-shuffled `-0.0065 ± 0.0184` (unbiased).
    - 30% left-censored jump magnitude MAE vs. uncensored oracle (`C2`): `ndimpute` ROS = `0.1000` vs. `DL/2` substitution = `0.2755`.
    - Shifting detection limit `<2.0 → <0.2` (`C3`): `30/30` (`100%`) DL shifts detected, `17/17` (`100%`) coinciding false jumps flagged.
    - Non-stationary large jump (`C4`): `scope="windowed"` MAE `1.242` vs. `scope="global"` `1.415`; imputed down-weighting (`C5`) lowers MAE monotonically from `0.606` ($\gamma=1.0$) to `0.206` ($\gamma=0.25$).
  - **Red noise & combined pipeline (`R1`, `P1`):**
    - AR(1) sweep (`R1`, $\alpha=0.05$): unweighted FPR climbs from `8%` ($\phi=0$) to `52%` ($\phi=0.6$) and `40%` ($\phi=0.8$); supplying $P = C_f^{-1}$ restores calibration to `8% / 10% / 5% / 5%`.
    - End-to-end `analyze_series()` (`P1`, seasonality + jump + TP + gaps + 20% censoring): **`100.0%` jump recovery** and **`100.0%` turning-point recovery**.
  - **Near-Real-Time Monitoring (`M1–M2`, v1.1):**
    - Sequential-peeking null calibration (`M1`, 25 steps, $\alpha=0.05$): unadjusted `parity_mode=True` FPR = **`72.5%`** vs. Šidák+Pocock **`0.0%`**, Šidák+O'Brien–Fleming **`0.0%`**, and Šidák+Consecutive-2 **`7.5%`**.
    - Disturbance recovery (`M2`, default Šidák+Pocock, $\alpha=0.01$): **`100.0%` abrupt recall** (median delay `0.160 yr`) and **`100.0%` phenological recall**.

Full quantitative tables are published in `validation/reports/benchmarks_report.md`.

## Tier 3: Unit, Invariance, Edge-Case & Property-Based Tests

- 53 automated `pytest` tests (46 unit/property tests including `test_monitor.py` + 7 end-to-end documentation tutorial smoke tests) including Hypothesis property-based verification of time-origin shift invariance ($t \mapsto t + c$), amplitude scaling equivariance ($y \mapsto k y$), scalar weight invariance ($P \mapsto \lambda P$), degenerate/constant/rank-deficient inputs, and headless plotting smoke tests.
- Dedicated lite-environment CI job verifying zero `ndimpute`/`pandas` imports for core numerical functions (`pip install --no-deps tspec` with `numpy` + `scipy` only).

## Tier 4: Speed & $O(n)$ Memory Profiling

- **Speedups vs. upstream Python reference:**
  - `allssa(decimal=4)` ($n=150$): `77.0 ms` vs. `493.8 ms` (**6.4× speedup**, gate $\ge 5.0\times$) — **PASS**
  - `stpd` single series ($n=100$): `4.39 ms` vs. `19.34 ms` (**4.4× speedup** via $O(n)$ cumulative prefix sums); **1,000-series batch in `4.39 s`** (gate $< 10.0\text{ s}$) — **PASS**
  - `just_jumps(ALLSSA)` ($n=150$): `786.1 ms` vs. `1282.6 ms` (**1.6× speedup**) — **PASS**
- **Memory scaling (`tracemalloc`):**
  - 1D diagonal weights peak memory: `1.95 MB` at $n = 1{,}000$ and **`19.40 MB`** at $n = 10{,}000$ (gate $< 50.0\text{ MB}$, strictly $O(n)$ vs. $\ge 800\text{ MB}$ dense $O(n^2)$) — **PASS**

Full profiling numbers are published in `validation/reports/perf_report.md`.
