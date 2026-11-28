# Pre-registered held-out validation 2: decoupled seasonal window

**Status:** registered 2026-11-28, before the change below was implemented and before any design was run.
**Context:** validation 1 (`decomposition_heldout_plan.md`) failed on irregular sampling. The exploratory follow-up (`audit/independent_2026_11/p20_out.txt`) showed that the seasonal error is driven by the 3-year window, which `just_decompose` shares between trend and seasonal; longer windows (6–10 years) gave a better seasonal component than STL but a worse trend. This validation tests the fix on **new** designs and a **new** real record. Its pass/fail result is the verdict on decomposition of unevenly sampled data.

## 1. Change under test (specified before implementation)

1. **`seasonal_window`** (new `just_decompose` argument, in years; `None` = legacy single-window behaviour). Default `"auto"` = `min(7, record span)` years whenever `season != "none"` and `parity_mode=False`. The value 7 is STL's default seasonal span in cycles; it was fixed here, not tuned (p20 looked at 3, 6 and 10 years only).
2. **Algorithm** (two-stage backfitting, STL-like):
   - (a) seasonal stage: duration windows of length `seasonal_window` with the requested seasonal model plus a local linear trend (and known jumps);
   - (b) trend stage: the requested trend windows (count or duration, as now) on `y − seasonal` with `season="none"`;
   - (c) one more pass of (a) on `y − trend` and (b) on `y − seasonal`.

   Seasonal and its band come from the last seasonal stage; trend and its band from the last trend stage; remainder = `y − trend − seasonal`.
3. **AUD-19b:** with `eval_times`, windows are built on the observations as without `eval_times`; each window's model is evaluated at the evaluation times inside its time span (`parity_mode` keeps the published behaviour).
4. **AUD-19c:** observations or evaluation times left without any covering window produce a warning.

No other constant is changed. Censored inputs with `seasonal_window` raise a clear error in this version (not tested here).

## 2. Configurations

| Label | Call |
|---|---|
| `tspec_default` | `just_decompose(t, y)` (new defaults) |
| `tspec_ols` | `just_decompose(t, y, season="ols")` (new defaults) |
| `tspec_legacy_ols` | `just_decompose(t, y, season="ols", seasonal_window=None)` (for reference) |
| `stl_lin`, `stl_bin`, `harm_lowess` | exactly as in validation 1 |

## 3. Designs (none used before)

Sampling:

| Code | Sampling |
|---|---|
| Q1 | regular every 14 days for 15 years |
| Q2 | uniform random times, 10 per year on average, over 25 years (count Poisson(250)) |
| Q3 | storm clusters: monthly routine samples plus 6 events per year at random times, each with 3–6 samples spread over 2–5 days, 20 years |
| Q4 | winter-sparse: monthly samples kept with probability 0.25 in Dec–Feb and 0.95 otherwise, 20 years |
| Q5 | rate drop: monthly for 10 years, then 4 per year (random times) for 10 years |
| Q6 | real sampling times of the Susquehanna River at Conowingo record (USGS 01578310, 1978-11 to 2023-09, n = 1222) |

Truths (time in years, `t0` = first sample, `L` = span):

| Code | Definition |
|---|---|
| U1 | `1.5 + 0.02 (t − t0) − 0.4 tanh((t − t0 − L/2) / 2.5)` (smooth regime shift) |
| U2 | `1.5 + 0.01 (t − t0) + 0.3 sin(2π (t − t0) / 8)` (8-year cycle) |
| V1 | fixed asymmetric shape: `0.6 cos(2π(t − 0.1)) + 0.25 cos(4π(t − 0.3)) + 0.1 cos(6πt)` |
| V2 | evolving: amplitude falls linearly from 1.0 to 0.4 and phase advances by 0.8 rad over the record, applied to `cos(2πt − φ) + 0.3 cos(4πt − 2φ)` |
| V3 | peaked: `0.9 [exp(6 cos(2π(t − 0.6))) − I0(6)] / (exp(6) − I0(6))` |

Noise: W = independent Gaussian, σ = 0.25; O = Ornstein–Uhlenbeck, e-folding 60 days, σ = 0.25.

Cells: **core (18)** = Q1…Q6 × {U1+V1, U1+V2, U2+V3} × W; **noise (3)** = {Q2, Q3, Q6} × U1+V1 × O. 100 replicates; replicate `r` of cell `c` uses `default_rng(940_000 + 1_000 c + r)`.

## 4. Metrics

As in validation 1 (observation times): trend, seasonal and remainder RMSE; band coverage; plus
- **blow-up rate:** fraction of observations with |trend error| > 1.0 (4σ);
- **NaN rate** at observation times;
- **eval consistency:** max |trend with `eval_times` (monthly grid) − trend without| at observations.

## 5. Registered hypotheses

| ID | Pass if |
|---|---|
| G1 seasonal vs STL | `tspec_ols` seasonal RMSE < min(`stl_lin`, `stl_bin`) in ≥ 15 of 18 core cells |
| G2 remainder vs STL | `tspec_ols` remainder RMSE < min(`stl_lin`, `stl_bin`) in ≥ 15 of 18 core cells |
| G3 no blow-ups | blow-up rate ≤ 0.5% and NaN rate = 0 for `tspec_ols` and `tspec_default` in all 21 cells |
| G4 eval consistency | max difference < 1e-8 for `tspec_ols` in all 21 cells |
| G5 trend vs STL (secondary) | `tspec_ols` trend RMSE < min(`stl_lin`, `stl_bin`) in ≥ 12 of 18 core cells |
| G6 bands (secondary) | `tspec_ols` trend and seasonal coverage in [0.90, 0.98] in the six V1 white-noise cells |
| G7 evolving seasonality vs pooled baseline | `tspec_ols` seasonal RMSE < `harm_lowess` in ≥ 5 of the 6 V2 cells |
| D1 fixed shapes vs pooled baseline | descriptive: ratio `tspec_ols` / `harm_lowess` seasonal RMSE in V1/V3 cells |
| R  real record (Conowingo, ln concentration) | 20 random 80/20 splits (`default_rng(950_000 + split)`), same scoring as validation 1: held-out RMSE of `tspec_ols` ≤ min(`stl_lin`, `stl_bin`) |

**Overall verdict for decomposing unevenly sampled data:** PASS only if G1, G2, G3 and G4 all pass. Otherwise FAIL, reported as such. Choptank results are reported again but are descriptive (that record was used in validation 1).
