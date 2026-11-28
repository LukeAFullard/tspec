# Pre-registered held-out validation of `just_decompose` vs STL

**Status:** registered 2026-11-28, before any of the designs below were run.
**Code under test:** `src/tspec/` exactly as committed together with this file. No change to `src/tspec/` is allowed while this validation runs; the benchmark records `git rev-parse HEAD` and fails if `src/` differs from it.

## Why this exists

The remediation batches of the 2026-11 audit chose several constants while looking at a narrow family of simulations: 240 points over 20 years, σ = 0.5, a 9-year sine trend, Gaussian or AR(1) noise. Rates measured on those designs are in-sample. This file fixes **new** designs, configurations, metrics and success criteria *before* running them. Every result is reported, including failures.

## Rules

1. Only the configurations in §2 are "registered". Anything else run later is labelled **exploratory** in the report and cannot change a conclusion.
2. No tuning. If a registered criterion fails, the failure is reported as is. A fix made afterwards must be validated on designs that are new again, not on these.
3. Seeds are fixed: replicate `r` of cell `c` (cells numbered in the order of §3) uses `numpy.random.default_rng(910_000 + 1_000 * c + r)`. 100 replicates per cell.
4. Run with `OMP_NUM_THREADS=1 MPLBACKEND=Agg`; report library versions.

## 1. Truths (time unit: years)

Every design is `y = trend + seasonal + noise`. The seasonal component has zero mean over each year, so the trend and seasonal components are identifiable. `t0` is the first sampling time.

| Code | Definition |
|---|---|
| **T1** smooth trend | `2.0 + 0.03 (t − t0) + 0.5 sin(2π (t − t0) / 13)` (13-year cycle; slope and period differ from the tuning designs) |
| **T2** trend with a step | T1 plus `+0.9` for `t ≥ t0 + 11.4` (3σ for white noise below) |
| **S1** fixed shape | `0.8 cos(2πt) + 0.3 sin(4πt + 0.5)` |
| **S2** evolving shape | amplitude grows linearly from 0.5 to 1.2 and the annual phase drifts linearly from 0 to 0.6 rad over the record: `A(t) [cos(2πt − φ(t)) + 0.3 sin(4πt + 0.5 − 2φ(t))]` |
| **S3** peaked (non-sinusoidal) | spring-flush pulse: `1.2 [exp(κ cos(2π(t − 0.25))) − I0(κ)] / (exp(κ) − I0(κ))` with κ = 4 (von Mises bump; zero annual mean; needs more than two harmonics) |

## 2. Registered configurations

| Label | Call |
|---|---|
| `tspec_default` | `just_decompose(t, y)` (all defaults: `season="allssa"`, count windows, `serial_correlation="ar1"`) |
| `tspec_ols` | `just_decompose(t, y, season="ols")` (harmonics 1–4) |
| `tspec_ols_dur` | `just_decompose(t, y, season="ols", window_mode="duration")` (3-year windows) |
| `tspec_jump` (T2 cells only) | `just_jumps(t, y, season="ols")` (defaults) → `just_decompose(t, y, season="ols", jump_indices=...)` |
| `stl_lin` | statsmodels `STL(period=12)` (defaults) on a monthly mid-month grid spanning the data, filled by linear interpolation of the observations |
| `stl_bin` | `STL(period=12)` on monthly bin means; empty months filled by linear interpolation of the bin means |
| `stl_lin_robust` | as `stl_lin` with `robust=True` |
| `harm_lowess` | irregular-native baseline: backfitting (3 passes) of a global OLS fit of harmonics 1–4 and a LOWESS trend (statsmodels `lowess`, `frac = 3 years / record span`, `it=0`) |

STL components are evaluated at observation times by linear interpolation of the monthly components. `harm_lowess` and `tspec` are evaluated directly at observation times. On the monthly evaluation grid (secondary metrics), `tspec` uses `eval_times` (`trend_reg`, `seasonal_reg`), `harm_lowess` interpolates its trend linearly between observations, and STL uses its grid directly.

## 3. Cells

Sampling regimes (20-year records unless stated):

| Code | Sampling |
|---|---|
| **R1** | regular monthly (mid-month), n = 240 |
| **R2** | R1 with 30% of months removed at random and a 6-month blackout starting at year 12.3 |
| **R3** | jittered monthly: mid-month ± U(−10, 10) days |
| **R4** | 180 times uniformly at random over 20 years |
| **R5** | seasonally biased: 200 times drawn with density ∝ `1 + 0.8 cos(2π(t − 0.55))` (summer field season) |
| **R6** | rate change: weekly for years 0–6, monthly for years 6–16, quarterly for years 16–20 |
| **R7** | the real sampling times of the Choptank River nitrate record (USGS 01491000, 1985-01-01 to 2023-09-30, n = 804, `validation/data/choptank_01491000_nitrate.csv`), applied to synthetic truths |

Noise (σ = 0.3 marginal standard deviation unless stated):

| Code | Noise |
|---|---|
| **N1** | independent Gaussian |
| **N2** | Ornstein–Uhlenbeck, e-folding time 45 days (correlation ≈ 0.5 at one month), sampled at the observation times |
| **N3** | sum of two OU processes, 10 days (σ = 0.25) plus 1 year (σ = 0.17): not AR(1), deliberately misspecified for the AR(1) band adjustment |
| **N4** | independent Student-t₃ scaled to standard deviation 0.3 |

Cells (numbered for the seed rule):

1. **Core grid (21 cells):** R1…R7 × {T1+S1, T1+S2, T1+S3} × N1, ordered by regime, then shape.
2. **Noise robustness (9 cells):** {R1, R4, R7} × T1+S1 × {N2, N3, N4}.
3. **Step (3 cells):** {R1, R4, R7} × T2+S1 × N1.

## 4. Metrics

Primary (at observation times): RMSE of the trend, the seasonal component, and the remainder against the true noise.

Secondary:
- the same RMSEs on the monthly grid inside the data span, and separately inside gaps longer than 3 months;
- leakage: correlation of the remainder with the true seasonal component, and with the true trend minus its mean;
- `tspec` only: pointwise 95% coverage of the trend band (`trend_ci_*`, around `trend_bc` as documented) and of the seasonal band, at observation times;
- runtime per fit.

## 5. Registered hypotheses and decision rules

| ID | Statement | Pass if |
|---|---|---|
| H1 | On regular data, `tspec` is competitive with STL | R1 core cells: the better of `tspec_default`/`tspec_ols` has trend RMSE ≤ 1.10 × `stl_lin` and seasonal RMSE ≤ 1.10 × `stl_lin` in all 3 cells |
| H2 | On irregular sampling, `tspec` beats interpolate-then-STL | R3–R7 core cells (15): `tspec_ols` has lower trend RMSE *and* lower seasonal RMSE than the better of `stl_lin`/`stl_bin` in ≥ 12 of 15 cells |
| H3 | `tspec` is at least as good as the simple irregular-native baseline | R2–R7 core cells (18): the better of `tspec_ols`/`tspec_ols_dur` has total-signal RMSE (trend + seasonal) ≤ `harm_lowess` in ≥ 12 of 18 cells |
| H4 | Trend bands are calibrated when the model is adequate | white-noise S1 cells, `tspec_ols`: trend-band coverage in [0.90, 0.98] in all 7 regimes |
| H5 | The AR(1) band adjustment works for AR(1)-type noise | N2 cells, `tspec_ols`: trend-band coverage in [0.90, 0.98] in all 3 regimes |
| H6 | Misspecified correlation and heavy tails (descriptive) | N3 and N4 cells: coverage reported; no pass/fail |
| H7 | Default vs fixed harmonics (descriptive) | report in how many of the 21 core cells `tspec_ols` has lower seasonal RMSE than `tspec_default`. If ≥ 14, the docs will recommend `season="ols"` for decomposition; changing the default is a separate decision |
| H8 | Steps (descriptive) | T2 cells: trend RMSE within ±1 year of the step for `tspec_jump` vs STL, and the jump detection rate |

## 6. Real record (no ground truth)

Choptank River nitrate, ln(concentration), 1985-01-01 to 2023-09-30. The one non-detect is passed to `tspec` as censored and to STL/`harm_lowess` at half its detection limit.

- **Cross-validated prediction (registered).** 20 random 80/20 splits (`default_rng(920_000 + split)`). Each method decomposes the 80% and predicts the held-out 20% from trend + seasonal at their times. `tspec` uses `eval_times`; STL interpolates its monthly grid; `harm_lowess` evaluates harmonics plus interpolated trend. Reported: held-out RMSE, mean ± SD across splits, for all routine and storm samples together, and separately for routine-only samples. No pass/fail criterion; the ranking is reported.
- **Descriptive.** Figures of each decomposition, and the trend compared qualitatively with the published WRTDS description of this record (Hirsch, Moyer & Archfield, 2010).

## 7. Reporting

Script: `validation/bench_decomp_heldout.py`. Outputs: `validation/reports/decomp_heldout_report.md`, `validation/reports/decomp_heldout_results.json`, figures in `validation/reports/figures_decomp/`. Each hypothesis is stated with its measured values and PASS / FAIL / DESCRIPTIVE.
