# Fixes Applied After the Audit

This document records the changes made in response to `audit/AUDIT_REPORT.md` (audited commit `22e2d44`). For each fixed finding it gives before/after evidence from re-running the audit's own Monte Carlo designs on the new code.

**Scope:** the four Blockers plus the quick, high-confidence items.
- CRT-01 is disclosed but not fixed.
- CRT-05 was fixed in a follow-up (see below). CRT-04 is partly fixed: STPD detection is now calibrated, but trend bands remain conditional on the detected structure (see *Not fixed* below).

**Policy:** defaults are now statistically calibrated. `parity_mode=True` still reproduces the upstream (Ghaderpour) behaviour, so existing parity results are unchanged.

Post-fix raw outputs are in `audit/post_fix/`. Scripts: `audit/post_mc2_just_jumps.py`, `audit/post_mc5_monitor.py`, `audit/post_mk_variants.py`, `audit/post_mc7_stpd.py`, `audit/post_stpd_robust.py`, plus the original audit scripts re-run unchanged.

## Summary

| Finding | Status | Before | After (same design) |
|---|---|---|---|
| **BLK-01** `just_jumps` false positives | **Fixed** | 98.5–100% (defaults), 71–100% (`jump_alpha=0.05`) | **4.0–5.6%** at default `jump_alpha=0.05` (ols, n = 50/150/500, gap); 2.0% (allssa); R = 300–1000 |
| BLK-01 power (2σ / 4σ jump, n = 150) | improved | 25% / 91% (old calibrated analyze filter) | **53% / 99%** |
| **BLK-02** ALLSSA spurious cycles (α = 0.01) | **Fixed** | 18–25% | **0–1.3%** (default Monte Carlo); Šidák option 0–1.7% |
| **BLK-03** decomposition trend-band coverage | **Fixed** | 37–77% for curved trends | **94.0–95.8%** across all curvatures (6–25-yr cycles), 95% for linear/jump truths; bands 1.2–1.35× wider |
| **BLK-04** censored OLS slope (50% censoring) | **Fixed** | bias −27%, coverage 63.4% | bias **+0.5%**, coverage **96.0%** (20% censoring: −2%, 94.0%) |
| CRT-02 monitor look-ahead | **Fixed** | step thresholds changed with future data | identical thresholds regardless of data length; FWER 0.2% |
| CRT-03 Sen fan exported as trend CI | **Fixed** | `trend_ci_*` covered 63% | fan moved to `slope_fan_*`; `trend_ci_*` is `None` for `"sen"` |
| CRT-01 `consecutive_k` FWER | **Disclosed** | claimed "protection at nominal α" | `explain()`/docs now state it does not bound FWER (still ≈11% over 96 steps) |
| MAJ-01 MK/Sen under autocorrelation | **Fixed** | Type-I 23–25%, coverage 74–77% (AR(1) 0.5) | **4.8–7.3%**, 92.7–95.2%; iid unchanged (4.0–4.2%) |
| MAJ-02 LSSA "significant points" narrative | **Fixed** | "Detected k significant points" on noise | reports the family-wise FAP of the top peak (`peak_fap`) |
| MAJ-03 MM / L1 breakdown claims | **Fixed (docs)** | "Yohai MM", "50% breakdown" | described as Huber-started MM-type; breakdown design-dependent |
| MAJ-04 years-only defaults | **Fixed (guarded)** | silent empty grid; cryptic crash | `TIME_UNIT_ASSUMPTION` / `DEGENERATE_DEFAULT_GRID` warnings; explicit error message |
| MAJ-05 overstated validation claims | **Fixed (docs)** | 26/26 gates, ALLSSA 7.5% | README/validation summary rewritten with audited numbers and limitations |
| CRT-05 seasonal/forecast bands under ALLSSA selection | **Fixed (follow-up)** | seasonal coverage 91.3% / 89.8%; monitor mean CI 92.9% | **94.6% / 93.5%**; monitor mean CI **94.9%**; jump FP unchanged (5%), power up (2σ: 55.5% → 61%), 4σ magnitude RMSE 0.029 → 0.019 (`audit/post_season_sig.py`, `audit/post_monitor_season.py`, `audit/post_sidak_null.py`) |
| CRT-04 STPD detection (SNR gate, uncalibrated knot search) | **Partly fixed (follow-up)** | null false-TP rate up to 2.2% at α = 0.01; detection 0% (n = 150, σ = 1), 32% (n = 500, σ = 0.6), 0% (n = 500, σ = 1.5) | `tp_calibration="montecarlo"`, SNR gate off by default: null **0.2–0.6%**; detection **52% / 82% / 6.4%**; curve coverage given detection 85–96%; robust (Bonferroni) null 0–0.2%; MC7b correct structure 93% → **96.5%** (`audit/post_mc7_stpd.py`, R = 1000; `audit/post_stpd_robust.py`, R = 500). Bands still conditional (see below) |
| MIN-01 Sen CI rank | **Fixed** | upper limit one rank low | matches `scipy.stats.theilslopes` exactly |
| MIN-02 datetime overflow / numeric strings | **Fixed** | `"1"` → year 1754; silent wrap | numeric strings parsed as numbers; out-of-range dates rejected |
| MIN-04 conditioning warning | **Fixed** | threshold 1e10 | caution at cond > 1e6 with explanation |
| MIN-07 overstated wording | **Fixed** | "confirms", "almost certainly" | softened |
| MIN-08 tooling | **Fixed** | mypy aborted; 7 ruff errors | `mypy src/`: 0 issues; `ruff check .`: clean |
| INF-06 divide-by-zero warnings | **Fixed** | 2 RuntimeWarnings in the tests | none |
| MIN-03 silent fallbacks | **Fixed** | IRLS non-convergence, bootstrap and STPD-segment fallbacks, and Sen/ATS limits were silent | reported as `ROBUST_NOT_CONVERGED`, `STABILITY_DRAW_FALLBACK`, `STPD_SEGMENT_SKIPPED`, `SEN_CI_UNAVAILABLE`, `ATS_UNAVAILABLE` |
| MIN-06 two slopes per segment | **Documented** | unexplained difference | `SegmentSummary` / `TurningPointEvent` docs explain separate vs. connected fits |
| MIN-09 ATS variance ignores censoring | **Fixed** | CI coverage 95.2% / 98.0% at 20% / 50% censoring (conservative) | **95.4% / 94.8%**; null Type-I 4.0% / 3.2% (`audit/post_ats_variance.py`, R = 500) |
| MIN-10 monitor guardrails use future data | **Fixed** | censoring fraction over the whole series | evaluated on the historical period only |

## What changed (API and behaviour)

**`just_jumps`**
- New defaults: `jump_alpha="auto"` resolves to 0.05, or to `None` in `parity_mode`.
- `mag_th`/`dir_th` default to 0, or to 0.05/0.01 in `parity_mode`.
- New `jump_calibration="montecarlo"` (default for non-robust fits; `"bonferroni"` for robust fits; `"none"` in `parity_mode`) and `n_mc=999`.
- How the Monte Carlo calibration works:
  - Each window's jump F statistic is referred to the exact Monte Carlo null distribution of the **maximum** F over all candidate splits. The split-wise statistics are computed jointly by 2-column Schur complements.
  - Below Monte Carlo resolution, the Bonferroni bound is reported (capped at 1/(B+1)), so the test is valid at every α.
  - The step test (jump vs. continuous hinge) is Bonferroni-adjusted over splits.
  - Jump and step p-values are Holm-adjusted across candidate locations.
- `bonferroni_window` is kept as a legacy alias.
- `JustJumpResult.jump_calibration` is new.
- New warnings: `UNCALIBRATED_JUMP_SELECTION` and `UNCALIBRATED_JUMP_PVALUES`.
- **This changes results:** jumps that were reported only because of the raw screening p-values or NDVI-scale thresholds are no longer reported by default.

**`allssa` / `lssa`**
- `allssa(significance="montecarlo" | "sidak" | "per_frequency", n_mc=999)`.
  - The default Monte Carlo rule accepts a new frequency only if its family-wise p-value is ≤ α. That p-value is the exact Gaussian-null distribution of the spectral maximum over a dense grid covering every frequency the search can visit.
  - `parity_mode=True` forces `"per_frequency"`.
- `AllssaResult.significance` is new.
- `lssa(peak_fap="sidak" | "montecarlo" | "none")` adds `LssaResult.peak_fap`.
- `to_dataframe()` column `significant` was renamed `significant_per_frequency`.
- Internal ALLSSA calls inside JUST and the monitor use `season_significance="sidak"` (follow-up fix for CRT-05).

**`just_decompose`**
- `trend_ci_method="bias_corrected"` (default; `"model"` in `parity_mode`). Each window is refit with an added quadratic trend term, harmonics held fixed. The band is centred on the blended bias-corrected estimate `trend_bc` (robust bias correction, Calonico–Cattaneo–Farrell 2018).
- The point estimate `trend` is unchanged, so the band is no longer guaranteed to contain `trend`.
- New fields: `trend_bc`, `trend_reg_bc`, `trend_ci_method`.
- New warning: `BIAS_CORRECTION_FALLBACK`.

**Censored data**
- `CensoredConfig(imputation_model="trend")` (default with `scope="global"`) imputes from a censored-regression (Tobit-type) MLE on time. The design is a linear trend plus `trend_harmonics`, defaulting to (1, 2) cy/yr for datetime input. The model handles left, right and interval censoring, with lognormal or normal errors (new module `tspec._cenreg`).
- `imputation_model="marginal"` restores the previous ndimpute behaviour.
- Stability draws use proper parameter draws.
- `analyze_series` OLS segments containing censored values use `n_imputations` (default 20) proper multiple imputations pooled with Rubin's rules and Barnard–Rubin degrees of freedom (`SegmentSummary.estimator="ols_mi"`).
- New warnings: `TREND_IMPUTATION_FALLBACK` and `MI_SEGMENT_FALLBACK`.

**`just_monitor`**
- With `horizon=None` (default), Pocock/O'Brien–Fleming use open-ended spending α_d = α / (2.1097 (d+1) ln²(d+1)). This is reported as `horizon=0` with an `OPEN_ENDED_SPENDING` caution. A planned `horizon` gives more power.
- `parity_mode` keeps the legacy horizon.
- `explain()` no longer claims FWER control for `consecutive_k`.

**`analyze_series`**
- The Sen fan is exported as `slope_fan_low/high`; `trend_ci_*` is `None` for `"sen"`.
- New `mk_correction="ar1"` (default): Var(S) is inflated by a bias-corrected AR(1) effective-sample-size factor on ranks.
  - `"hamed_rao"` matches `pymannkendall` exactly. I didn't make it the default because it stays liberal in small samples (12–17% Type-I at ρ = 0.5).
  - `"none"` assumes independence.
- Stage 1 inherits the calibrated `just_jumps` defaults.

**Other**
- `TIME_UNIT_ASSUMPTION` warning.
- Datetime range checks.
- Sen CI order statistics follow the Gilbert/scipy convention.
- `HIGH_CONDITION_NUMBER` warning at cond > 1e6.

## Verification

- `pytest`: **138 passed** (tests/), including new `tests/test_calibration_fixes.py` plus new tests in `test_just.py` and `test_monitor.py`. `audit/test_hypothesis_audit.py`: 6/6.
- `ruff check .`: clean. `mypy src/`: no issues in 16 source files (target now Python 3.12 for numpy ≥ 2.3 stubs; runtime support unchanged).
- Monte Carlo re-runs, with the same designs and seeds as the audit:

  | Study | Result |
  |---|---|
  | MC1 (ALLSSA, R = 500–600/cell) | 0.0–1.3% spurious selection at α = 0.01; pointwise LSSA unchanged (3.7–5.5%) |
  | MC2/MC3 (`post_mc2_just_jumps.py`, R = 1000 null) | false jumps 5.6% (n = 50), 4.2% (n = 150), 4.0% (n = 150 gap), 4.0% (n = 500), 2.0% (allssa) |
  | MC4 (R = 500) | trend coverage 94.9–95.9% for linear, curved and jump truths |
  | Curvature sweep | 94.0–95.8% for every curvature |
  | MC5 (`post_mc5_monitor.py`, R = 500) | FWER 0.2% (no horizon), 0.4–0.6% (horizon = D); PI coverage 94.8–94.9% |
  | MC7b (R = 200) | correct structure 93% (was 90%), spurious jump 0.5% (was 3.5%); 96.5% after the STPD calibration |
  | MC10 (R = 300) | above table |
  | MK variants (`post_mk_variants.py`, R = 1000) | above table |

## Not fixed (remaining limitations)

- **CRT-01:** `consecutive_k` still does not bound the family-wise rate. It is now documented as such, and the warnings no longer recommend it for error control.
- **CRT-04 (bands):** STPD / `analyze_series` trend bands remain conditional on the detected structure: about 83% coverage in MC7b even with the correct structure. The docs quantify this. A full fix needs a bootstrap of the detect-then-fit procedure. The detection half of CRT-04 (SNR gate, uncalibrated knot search) is fixed. STPD power still depends on the information within one window, not on the record length.
- **Censored data beyond segment slopes:** JUST/STPD detection p-values on censored data condition on a single (now time-conditional) imputation. The windowed and seasonal-stratified scopes still use the marginal model.
- **MIN-03 (partial):** IRLS non-convergence inside JUST/STPD sliding windows is still not reported per window. Convergence failures were ≤ 0.2% in the audit's simulations.
- **Not re-audited:** the readiness score in `AUDIT_REPORT.md` refers to the audited commit and has not been recomputed. Provisionally, with all four Blockers and CRT-02/03/05 fixed, CRT-04 partly fixed and CRT-01 still open, the cap rule would limit the score to 80%.
