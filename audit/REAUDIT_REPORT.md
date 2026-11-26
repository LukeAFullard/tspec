# tspec Statistical & Code Correctness Audit — Re-audit

**Commit audited:** `65d0f87b8598e65295be1343652f5e6cb4d9f628` plus the re-audit commits on branch `claude/ecstatic-cori-d57odm` (documentation and `explain()` text only; no change to numerical results) | **Date:** 2026-11-26 | **Auditor:** Claude Code (Anthropic), automated audit agent
**Environment:** Python 3.13.18, numpy 2.5.3, scipy 1.18.1, pandas 3.0.5, ndimpute 0.1.1 (reference libs: statsmodels 0.15.0, astropy 8.0.3, pymannkendall 1.4.3, hypothesis 6.170.17), Ubuntu 24.04.5 LTS (Linux 6.18 x86_64), 4 CPU cores

> **Update (addendum at the end):** both Critical issues found by this re-audit were then fixed and re-verified. CRT-06 is fixed. CRT-04 is fixed for STPD, with a residual on the `analyze_series` trend band (about 93% coverage) re-graded Major. The updated readiness score is **82%** (no caps apply).

This is a re-audit of the fixed package against the original audit (`audit/AUDIT_REPORT.md`, commit `22e2d44`, score 40%). It uses the same rubric, severity table and caps. Fix-by-fix before/after evidence is in `audit/FIXES.md`. Raw outputs are in `audit/post_fix/`. Unlike the original audit, the re-audit also made small documentation and `explain()` corrections where it found inaccuracies; these are listed in §6 as fixed during the re-audit.

---

## 1. Executive Summary

- Overall scientific-use readiness: **77%**: **Usable with stated caveats; fix the remaining Critical issues before publication-critical use.** The weighted score is 77.0%. Two Confirmed Critical issues cap it at 80%, which does not bind.
- Findings (open): 🟥 0 Blockers | 🟧 2 Critical | 🟨 0 Major | 🟦 5 Minor | ⬜ 9 Info. A further 3 Minor documentation findings were found **and fixed** during the re-audit (§6, MIN-R1 to MIN-R3).

Headline conclusions:

- **All four Blockers are resolved, with verified calibration.**
  - `just_jumps` false-jump probability: 2–6% at `jump_alpha=0.05`, previously 71–100% (BLK-01).
  - ALLSSA spurious cycles: 0–1.3% at α = 0.01, previously 18–25% (BLK-02).
  - Decomposition trend bands: 94–96% coverage for curved trends, previously 37–77% (BLK-03).
  - Censored OLS slopes at 50% censoring: bias +0.5%, coverage 96%, previously −27% and 63% (BLK-04).
- **The change-detection and monitoring layers now control false positives at their stated levels:**
  - STPD: 0.2–0.6% false turning points at α = 0.01. Detection of a clear turning point rose from 0–32% to 52–82% in the hard designs.
  - Calibrated `consecutive_k`: 3.3–5.3% at α = 0.05 over 24–96 steps, with faster detection than Pocock.
  - Monitoring thresholds no longer use future data (CRT-01/02 fixed).
- **Remaining Critical issues:**
  - **CRT-04 (residual):** trend bands after detection in `stpd`/`analyze_series` are conditional on the detected structure. They cover about 83% (pipeline) to 85–96% (STPD) even when the structure is found, and far less when a turning point is missed.
  - **CRT-06 (new):** the default `mk_correction="ar1"` is accurate for long series but still liberal for short, strongly autocorrelated series (Type-I 7–9%, Sen CI coverage 91–93% at n ≤ 150, ρ ≥ 0.5).
- **The evidence chain was re-verified, not assumed.**
  - A 42-case output-equivalence battery shows that HEAD produces output identical to the commits where each post-fix Monte Carlo study was run. The only differences are intended changes with their own evidence.
  - 892 option combinations added since the original audit run cleanly in fuzzing.
  - Six new null spot checks cover configurations the earlier studies did not (sliding/weighted monitoring, weighted STPD, MM inside JUST/STPD, windowed censoring, robust multiple-imputation detection). All are at or below nominal.
- **The numerical core is unchanged and still agrees with the reference libraries to about 1e-14.**

---

## 2. Readiness Assessment

### 2.1 Category Scores

| Category | Score (0–10) | Weight | Weighted | Justification |
|---|---:|---:|---:|---|
| Statistical validity of estimators & tests (A–H) | 8 | 30% | 2.40 | The sup-F jump test, family-wise ALLSSA selection, bias-corrected bands, time-conditional imputation with Rubin pooling, Daniels ATS variance, the max-over-knots STPD test and run-length-calibrated monitoring are all correctly constructed and verified. Deductions: post-detection bands remain conditional (CRT-04). The small-sample AR(1) MK correction is imperfect (CRT-06). The MM/L1 estimators are not high-breakdown (documented). |
| Calibration of uncertainty (section K) | 7 | 20% | 1.40 | Passes: MC1 (0–1.3%), MC2 (2–6%), MC4 (94–96%), CRT-05 bands (93.5–94.9%), MC5 (Pocock 0.2–0.6%, `consecutive_k` 3.3–5.3%), MC6 (≈95%), STPD null (0.2–0.6%), MC8, MC10 (96%), ATS (94.8–95.4%), censored detection (≤ 3%, one borderline 6.7%), six spot checks. Fails: pipeline trend band 83% / slopes 90% after correct detection (CRT-04); MK/Sen at n ≤ 150, ρ ≥ 0.5 (7–9%, 91–93%; CRT-06). |
| Code correctness & robustness to edge cases (J) | 8 | 15% | 1.20 | 144 tests and 6 Hypothesis properties pass. 892/892 fuzz combinations of new options run with in-range outputs or clear `ValueError`s. Silent fallbacks are now reported. Remaining: IRLS non-convergence inside sliding windows is not reported per window (MIN-03r). |
| Numerical stability | 9 | 10% | 0.90 | Cross-checks are unchanged at about 1e-14 (§9). The conditioning warning is now at cond > 1e6 (MIN-04 fixed). |
| Test-suite adequacy & independent validation | 6 | 10% | 0.60 | New calibration regression tests cover every fixed finding with thresholds far from the pre-fix behaviour. The audit scripts provide ≥ 300–2000-replicate evidence. But the in-repo calibration tests use 20–60 replicates, no CI job runs the high-replicate studies, and the `validation/` tier was relabelled historical rather than regenerated (MIN-11). |
| Documentation accuracy & disclosure | 8 | 10% | 0.80 | Every public parameter, result field and warning is documented, with quantified limitations. Earlier false claims are corrected. The re-audit still found three understated calibration figures, a missing conditional-band caveat in `explain()` and an unqualified MK claim; all were fixed in this re-audit (MIN-R1 to MIN-R3). |
| Reproducibility (determinism, pinning, installability) | 8 | 5% | 0.40 | Deterministic by default, including all new Monte Carlo calibrations (fixed seeds). mypy and ruff are clean, and the strict docs build passes. Core dependencies have no upper bounds, and ndimpute is 0.x. |
| **Total** | | 100% | **7.70 / 10 = 77.0%** | |

### 2.2 Caps Applied

- 0 Confirmed Blockers, so the 40% cap does not apply.
- 2 Confirmed Critical issues (CRT-04 residual, CRT-06), so the ≥ 1-Critical cap of **80%** applies. It does not bind (77.0% < 80%).
- The 3-Critical cap (65%) does not apply.
- The test suite passes (144/144) and the package installs, so the 25% cap does not apply.
- **Final overall readiness: 77%.** Verdict band 75–89%: *Usable with stated caveats; fix Major issues before publication-critical use.* There are no open Majors; the two open Criticals are the items to fix.

### 2.3 Per-Module Readiness

| Module | Was | Now | Main reasons |
|---|---:|---:|---|
| LSSA (`lssa`) | 85% | 90% | Exact against astropy; `peak_fap` reports the family-wise false-alarm probability of the top peak (MAJ-02 fixed). |
| ALLSSA (`allssa`) | 35% | 85% | Monte Carlo family-wise selection, 0–1.3% spurious cycles. Šidák alternative 0–1.7%. Time-unit guards (MAJ-04). |
| JUST jumps (`just_jumps`) | 30% | 85% | Monte Carlo sup-F + Holm, 2–6% false jumps; power at 2σ/4σ is 53%/99% (n = 150). Censored detection has valid size, but borderline calls depend on the imputation seed (MIN-12). |
| JUST decomposition (`just_decompose`) | 45% | 85% | Bias-corrected trend band, 94–96% coverage. Family-wise seasonal selection, seasonal bands 93.5–94.6%. |
| JUSTmonitor (`just_monitor`) | 60% | 85% | No look-ahead. Pocock/OBF 0.2–0.6% (very conservative, INF-01). Calibrated `consecutive_k` 3.3–5.3% with a planned horizon. Uncalibrated without a horizon, with a warning (MIN-13). |
| STPD / TPTR | 60% | 75% | Calibrated knot-search test with no SNR gate: 0.2–0.6% false turning points, much higher power. Bands after detection remain conditional (CRT-04). Power is limited by the information within one window (INF-08). |
| Robust (Huber / MM / L1 / S–H test) | 65% | 75% | Claims corrected (MAJ-03). MM inside JUST/STPD is conservative (2.0% / 0.0% false detections at 5% / 1%). Not high-breakdown (INF-10). |
| Sen / ATS / MK | 65% | 75% | Sen CI ranks exact, Daniels ATS variance calibrated, AR(1) correction accurate for long series. Liberal for short, strongly autocorrelated series (CRT-06). |
| Censored (`ndimpute` integration) | 30% | 80% | Time-conditional imputation; Rubin-pooled slopes (96% coverage); valid-size detection. The opt-in `detection_imputations` makes decisions seed-independent at a power cost (INF-07). Windowed/stratified scopes use the marginal model (INF-09). |
| Pipeline (`analyze_series`) | 40% | 70% | Correct structure 96.5% (MC7b), spurious jumps 0.5%. Trend band 83% and slope CIs 90% after correct detection (CRT-04). |

### 2.4 Why It Is Ready

- **Every headline test now controls its stated error rate.** Each figure below is from Monte Carlo runs on code verified output-identical to HEAD:

  | Component | Error rate | Nominal | Replicates |
  |---|---|---|---|
  | JUST jumps | 2–6% | 5% | 300–1000 |
  | ALLSSA | 0–1.3% | 1% | 500–600 |
  | STPD | 0.2–0.6% | 1% | 1000 |
  | Robust STPD | 0–0.2% | 1% | 500 |
  | Monitoring, Pocock/OBF | 0.2–0.6% | 5% | 500 |
  | Monitoring, calibrated `consecutive_k` | 3.3–5.3% | 5% | 400–2000 |
  | Censored detection | 0.25–3% | 5% | 300–400 |

- **Bands that were badly wrong now cover nominally:**
  - Decomposition trend band: 94.0–95.8% for every curvature (BLK-03).
  - Seasonal bands: 93.5–94.6% (CRT-05).
  - Monitoring forecast-mean CI: 94.9%; prediction interval: ~95% (MC6).
  - Censored OLS slopes: 96.0% (BLK-04).
  - Censored ATS slopes: 94.8–95.4% (MIN-09).
  - Sen slope CIs under independent errors: 95–96% (§8).
- **The calibration machinery is exact rather than asymptotic where it matters.**
  - The JUST sup-F, ALLSSA spectral maximum and STPD max-t nulls are pivotal under Gaussian errors and are simulated exactly for each window design.
  - `consecutive_k` is calibrated by simulating the full monitoring residual process. It is exact under regular sampling (5.15% at 5%, R = 2000).
- **Power improved alongside calibration:**
  - JUST at 2σ: 25% → 53%.
  - STPD clear turning point: 0–32% → 52–82%.
  - Monitoring 2σ step: Pocock 48–49%, calibrated `consecutive_k` 82–87%.
- **Robust engineering.**
  - Strict input validation, exact NaN index mapping, no mutation of inputs, deterministic seeded Monte Carlo.
  - Fallbacks are surfaced as named warnings.
  - Fuzzing of 892 new-option combinations is clean.
- **Candid, quantified documentation of the remaining limitations**, now also in the `explain()` output (MIN-R2).

### 2.5 Why It Is Not Ready (for ≥ 90%)

- Trend and slope bands after detection in `stpd`/`analyze_series` exclude breakpoint-location uncertainty. They cover about 83–92% after correct detection, and much less when a turning point is missed (**CRT-04**, residual).
- The Mann–Kendall/Sen AR(1) correction is liberal for short, strongly autocorrelated series (**CRT-06**).
- High-replicate calibration studies are not part of CI, and the `validation/` tier was not regenerated (**MIN-11**).
- IRLS non-convergence inside JUST/STPD windows is not reported per window (**MIN-03r**).
- Censored detection decisions depend on the imputation seed for borderline cases. The seed-independent option costs power under heavy censoring (**MIN-12**, **INF-07**).
- `consecutive_k` without a planned horizon remains uncalibrated, with a warning (**MIN-13**).

### 2.6 Path to ≥ 90%

1. **CRT-04:** provide a bootstrap of the full detect-then-fit procedure for `stpd` and `analyze_series` bands, for example a parametric residual bootstrap that re-runs detection in each replicate. Add an MC7b-style coverage test (≥ 500 replicates; target ≥ 93% after correct detection).
2. **CRT-06:** replace the AR(1) rank variance factor with a small-sample-calibrated version, either a simulation-calibrated factor as a function of (n, ρ̂) or a block or sieve bootstrap of the MK statistic. Add a regression test at n = 60/150, ρ = 0.5/0.7 (target 5 ± 1.5%).
3. **MIN-11:** add a scheduled CI job that runs the `audit/post_*` studies at ≥ 1000 replicates with "within 2 MC-SE of nominal" gates, and regenerate `validation/` from it.
4. **MIN-03r:** collect per-window IRLS convergence flags and emit `ROBUST_NOT_CONVERGED` with the window count.
5. With items 1–2 resolved, the Critical cap is lifted and the projected category scores rise to about 8.5 (validity), 8.5 (calibration) and 7–8 (tests), which takes the weighted total past 90%.

---

## 3. Blockers

No open Blockers. Resolution of the original four:

| ID | Original finding | Resolution | Evidence |
|---|---|---|---|
| BLK-01 | `just_jumps` 98.5–100% / 71–100% false positives | Monte Carlo max-over-splits F null per window, Bonferroni step test, Holm across candidates; `jump_alpha="auto"` (0.05) | 2.0–5.6% (n = 50–500, ols/allssa), 5% with ALLSSA after CRT-05; `audit/post_mc2_just_jumps.py`, `audit/post_season_sig.py` |
| BLK-02 | ALLSSA 18–25% spurious cycles at α = 0.01 | Family-wise significance (Monte Carlo maximum of the spectrum; Šidák option) | 0–1.3%; `audit/mc1_lssa.py` re-run |
| BLK-03 | Decomposition trend CIs 37–77% for curved trends | Bias-corrected local-quadratic band (`trend_ci_method="bias_corrected"`) | 94.0–95.8% across curvatures; `audit/repro_just_ci_curvature.py`, `audit/mc4_just_ci.py` |
| BLK-04 | Censored OLS slope −27% bias, 63% coverage | Time-conditional Tobit imputation + Rubin-pooled multiple imputation | +0.5% bias, 96.0% coverage at 50% censoring; `audit/mc10_censored.py` |

---

## 4. Critical Issues

### CRT-04 (residual): Trend and slope bands after detection in `stpd` / `analyze_series` are conditional on the detected structure
- **Location:** `src/tspec/stpd.py` (`fit_piecewise_trend` band via `detect_turning_points`); `src/tspec/pipeline.py` (`trend_ci_low/high`, `SegmentSummary.slope_ci_*`)
- **Confidence:** Confirmed
- **Affected API:** `stpd(...).piecewise_trend.trend_ci_*`, `analyze_series(...).trend_ci_*`, `analyze_series(...).segments[i].slope_ci_*`
- **Description:** The detection half of the original CRT-04 is fixed: the SNR gate is off by default, and the knot search is calibrated. The bands, however, are still computed as if the detected knots were known in advance, so breakpoint-location uncertainty and the selection itself are not propagated.
- **Evidence / Reproducer:** `audit/post_mc7_stpd.py` (R = 1000) and `audit/mc7b_pipeline_two_stage.py` (R = 200), both on code output-identical to HEAD:
  ```
  STPD curve coverage | single TP detected correctly : 92.2% (n=150, σ=0.3), 95.7% (σ=1), 84.8% (n=500, σ=0.6), 92.1% (σ=1.5)
  STPD curve coverage | all replicates (incl. misses) : 92.0%, 74.0%, 77.9%, 16.8%
  analyze_series      | correct structure (96.5%)      : trend band 83.2%, segment slope CI 90.2%
  ```
  Expected: 95%.
- **Why expected is correct:** Valid post-selection intervals must account for the selection step (Berk et al. 2013; Leeb & Pötscher 2005). A fixed-knot band is valid only conditionally on the true knots.
- **Scientific impact:**
  - The bands are too narrow. Users may over-state precision of trend levels and segment slopes around detected breakpoints.
  - Severity: the 83% pipeline coverage lies at the Blocker threshold ("95% CI covering < 85%"). This is graded Critical, not Blocker, because the shortfall is post-selection and is now disclosed in the docstrings, the guides and (since this re-audit) in `explain()`.
- **Recommended fix:** an opt-in `band_method="bootstrap"` that resamples residuals from the fitted piecewise model, re-runs detection and refit per replicate, and takes pointwise percentiles. Use this as the default only once its cost is acceptable.
- **Suggested regression test:** the MC7b design with R ≥ 500; require trend-band coverage ≥ 93% under the bootstrap method.

### CRT-06 (new): The Mann–Kendall / Sen AR(1) correction is liberal for short, strongly autocorrelated series
- **Location:** `src/tspec/_robust.py` (`ar1_rank_factor`, `fit_theil_sen_segment(autocorr_correction="ar1")`)
- **Confidence:** Confirmed
- **Affected API:** `analyze_series(slope_estimator="sen", mk_correction="ar1")` p-values and slope CIs (the default for Sen segments)
- **Description:** The variance factor uses a bias-corrected lag-1 autocorrelation estimate mapped to a rank correlation. With n ≤ 150 and ρ ≥ 0.5, ρ̂ remains biased low, so the variance is under-inflated.
- **Evidence / Reproducer:** `audit/reaudit_mk.py` (library path, R = 1000 per cell, monthly sampling, marginal sd 0.5; true slope 0 for Type-I and 0.1/yr for coverage). MC-SE is about 0.7% at 5%.
  ```
  n=60 : Type-I 4.3% / 7.4% / 5.0% / 8.6%   coverage 95.7 / 92.6 / 95.0 / 91.4%   (ρ = 0, 0.3, 0.5, 0.7)
  n=150: Type-I 3.5% / 5.3% / 7.3% / 7.0%   coverage 96.5 / 94.7 / 92.7 / 93.0%
  n=400: Type-I 3.9% / 4.9% / 5.2% / 5.1%   coverage 96.1 / 95.1 / 94.8 / 94.9%
  ```
  Expected: 5% / 95%.
- **Why expected is correct:** These are nominal-level tests and intervals. The MK variance under serial dependence requires the full (unknown) autocorrelation inflation (Hamed & Rao 1998; Yue & Wang 2004); plug-in lag-1 estimates are biased downward in short series (Marriott & Pope 1954).
- **Scientific impact:** Inflated false trend detections, and CIs that are too narrow, for short, strongly autocorrelated records, which are common in monthly environmental data. Long records are fine.
- **Note on grading:** the original audit graded the uncorrected behaviour (23–25% Type-I, 74–77% coverage) as Major (MAJ-01). Applying the severity table literally ("under-coverage of 88–93%" → Critical), the residual is graded Critical here. The original grade was too lenient.
- **Recommended fix:** calibrate the variance factor by simulation over (n, ρ̂), or use a block or sieve bootstrap of S when red noise is moderate or worse. Alternatively prewhiten (Yue–Wang trend-free prewhitening) with a bias-corrected ρ̂.
- **Suggested regression test:** the cells above with R ≥ 1000; require Type-I in [3.5%, 6.5%] and coverage in [93.5%, 96.5%].

### Original Critical issues (resolution)

| ID | Original finding | Status | Evidence |
|---|---|---|---|
| CRT-01 | `consecutive_k` not family-wise controlled (10–11.5% over 96 steps) | **Fixed with a planned `horizon`**: Monte Carlo run-length calibration. Without a horizon the old behaviour is kept and `UNCALIBRATED_CONSECUTIVE_K` is reported (MIN-13) | 4.8–5.3% (24 steps), 3.3% / 5.1% (96 steps, ols/allssa), 4.5% (sliding window, weights, allssa); `audit/post_consecutive_k*.py`, `audit/reaudit_spotchecks.py` (a) |
| CRT-02 | Monitoring look-ahead | **Fixed**: open-ended spending without a horizon; no calibration input uses monitoring data | Identical thresholds regardless of data length (`tests/test_monitor.py`); same-level test in `tests/test_calibration_fixes.py` |
| CRT-03 | Sen fan exported as `trend_ci_*` (63%) | **Fixed**: moved to `slope_fan_*` | `tests/test_ci.py` |
| CRT-04 | STPD SNR gate / uncalibrated knot search; conditional bands | **Detection fixed; bands open** (see above) | `audit/post_mc7_stpd.py` |
| CRT-05 | Seasonal/forecast bands under ALLSSA selection | **Fixed** | 93.5–94.6%; monitor mean CI 94.9%; `audit/post_season_sig.py`, `audit/post_monitor_season.py` |

---

## 5. Major Issues

No open Major issues. Resolution of the original five:

| ID | Original finding | Status | Evidence |
|---|---|---|---|
| MAJ-01 | No serial-correlation correction for MK/Sen | **Fixed for long series**; residual for short series re-graded as **CRT-06** | `audit/reaudit_mk.py` |
| MAJ-02 | LSSA "significant points" narrative | **Fixed** (`peak_fap`) | `tests/test_calibration_fixes.py` |
| MAJ-03 | MM / L1 breakdown claims false | **Fixed (docs)**; limitation remains as INF-10 | `docs/reference/math.md` |
| MAJ-04 | Silent years-only defaults | **Fixed (guarded)**: `TIME_UNIT_ASSUMPTION`, `DEGENERATE_DEFAULT_GRID` | `tests/test_calibration_fixes.py` |
| MAJ-05 | Overstated validation claims | **Fixed (docs)**; regeneration still open as MIN-11 | README / `docs/reference/validation-summary.md` |

---

## 6. Minor Issues

### MIN-03r: IRLS non-convergence inside JUST/STPD sliding windows is not reported per window
- **Location:** `src/tspec/just.py` (`_detect_jump_in_window`, `_nested_jump_ftests`), `src/tspec/stpd.py` (`_tpd_window`)
- **Confidence:** Confirmed (by code reading); impact Suspected low
- **Affected API:** `just_jumps(robust=...)`, `stpd(robust=...)`
- **Description:** `ROBUST_NOT_CONVERGED` is raised for top-level fits (`lssa`, `fit_piecewise_trend(mode="huber")`), but window-level IRLS flags are discarded.
- **Evidence:** Convergence failures were ≤ 0.2% in the original audit's simulations; no failures surfaced in the 892-case fuzz.
- **Scientific impact:** a non-converged window could yield an unreliable candidate silently.
- **Recommended fix / test:** propagate `converged` flags and count them; add a test with a degenerate window (e.g. heavy ties) that asserts the warning.

### MIN-11: High-replicate calibration evidence is not part of CI; the validation tier was relabelled, not regenerated
- **Location:** `tests/test_calibration_fixes.py`, `validation/`, `.github/workflows/ci.yml`
- **Confidence:** Confirmed
- **Description:** The in-repo calibration tests use 20–60 replicates with thresholds far from pre-fix behaviour. That catches regressions to the old failures, but not drift of a few percentage points. The ≥ 300–2000-replicate studies exist only as audit scripts.
- **Recommended fix:** add a scheduled (e.g. weekly) CI job that runs the `audit/post_*` studies with 2-MC-SE gates and publishes the table into `docs/reference/validation-summary.md`.

### MIN-12: Censored detection p-values use one imputation by default; borderline decisions depend on the seed
- **Location:** `src/tspec/just.py`, `src/tspec/stpd.py`; `CensoredConfig.detection_imputations`
- **Confidence:** Confirmed
- **Evidence:** `audit/censored_detection.py`, R = 300 per cell:
  - Size is valid: 2.0–3.0% false jumps at 5%, except 6.7% for seasonal series at 50% censoring (1.3 MC-SE high, borderline); STPD 0.3–1.0% at 1%.
  - Changing `random_state` flips 2–13% of jump decisions (up to 27% for seasonal series at 50% censoring).
  - The opt-in `detection_imputations=20` reduces flips to 1–9% and is conservative (1–3%), but costs power at 50% censoring (see INF-07).
- **Scientific impact:** reproducible with the default fixed seed, but a conclusion near the threshold may change under a different seed.
- **Recommended fix:** report `stability_rate` for every censored detection by default (`n_stability_draws > 0` in `just_jumps`/`stpd`), or improve the power of the pooled test (e.g. calibrate the pooled statistic's null by simulation).

### MIN-13: `consecutive_k` without a planned `horizon` is uncalibrated
- **Location:** `src/tspec/monitor.py`
- **Confidence:** Confirmed
- **Description:** Without a planned `horizon`, each step is tested at the full α. Family-wise false alarms then reach 9.5–15.3% over 96 steps. The case is warned (`UNCALIBRATED_CONSECUTIVE_K`) and documented, and the default (`pocock`) is unaffected.
- **Recommended fix:** an open-ended run-length calibration (e.g. spending over blocks of steps), or require `horizon` for `consecutive_k`.

### MIN-14: Approximations in the `consecutive_k` calibration
- **Location:** `src/tspec/monitor.py` (`_calibrate_consecutive_k`)
- **Confidence:** Suspected (impact small in the designs checked)
- **Description:** The calibration makes these simplifications:
  - Errors are Gaussian.
  - ALLSSA-selected harmonics are held fixed.
  - Robust history fits are simulated as WLS with the robust weights.
  - Future sampling follows gaps resampled from the history.
  - Full weight matrices fall back to the uncalibrated rule, with `CALIBRATION_FALLBACK`.
- **Evidence:** the allssa and sliding/weighted designs gave 5.1% (R = 1200) and 4.5% (R = 400) at 5%. Irregular sampling gave 4.2% (R = 2000) after the gap-resampling fix (it was 5.5–5.9% with a regular grid). All approximations are now disclosed in the docstring.
- **Recommended fix:** an optional fully parametric bootstrap of the whole monitoring procedure for heavy-tailed or robust use.

### Fixed during this re-audit

| ID | Finding | Fix |
|---|---|---|
| MIN-R1 | README and guides quoted calibration figures more favourably than the evidence (direction: reassuring). JUST "3–5%" (measured 2–6%); ALLSSA "0–0.5%" (0–1.3%); Pocock "0.1–0.3%" (0.2–0.6%); `consecutive_k` "3–5%" (3.3–5.3%); MK "≈ 5–7% at ρ = 0.5" with no statement about short series | Figures corrected in README, `docs/guide/interpreting-results.md`, the monitoring tutorial, CHANGELOG and the parameter cheat sheet |
| MIN-R2 | `analyze_series(...).explain()` and `stpd(...).explain()` printed unqualified "95% CI" values after detection, without the conditional caveat that the docstrings contain | `explain()` now states that intervals are conditional on the detected breakpoints, with the simulated coverage |
| MIN-R3 | The `consecutive_k` calibration docs did not state the robust and ALLSSA approximations | Disclosed in the `just_monitor` docstring |

---

## 7. Info / Suggestions

- **INF-01:** The default monitor (Šidák + Pocock/OBF) remains very conservative (0.2–0.6% family-wise at α = 0.05). Calibrated `consecutive_k` with a planned horizon gives near-nominal error and much faster detection.
- **INF-02:** Weights are relative precisions (unchanged).
- **INF-03:** JUST location convention (unchanged).
- **INF-04:** L1 solutions can be non-unique (unchanged).
- **INF-05:** Core dependencies have no upper bounds; ndimpute is 0.x.
- **INF-07:** `detection_imputations > 1` is conservative and costs jump power under heavy censoring: at 50% censoring, 66% instead of 83% detection of a 2.5σ step (32% vs 69% for seasonal series). There is no loss at 20% censoring or for STPD. Imputing with the candidate jump in the model was rejected: it builds the data-selected location into every imputation and gave 10–15% false jumps.
- **INF-08:** STPD power depends on the information within one window (default 5 seasonal cycles), not on the record length. For example, 6.4% detection at n = 500, σ = 1.5 with the MC7 design.
- **INF-09:** `scope="windowed"` and `"seasonal_stratified"` still use the marginal `ndimpute` model. The windowed null check gave 3.5% at 5% (spot check f). Multiple-imputation detection is not applied to the windowed scope.
- **INF-10:** `robust="mm"` (Huber-started MM-type) and `mode="l1"` are not high-breakdown estimators (documented). Within JUST/STPD windows MM is conservative (2.0% / 0.0% false detections at 5% / 1%; spot checks d, e).

---

## 8. Monte Carlo Calibration Results

The "Code state" column says how each result is tied to HEAD (§11):

- **≡ HEAD:** output-identical at HEAD.
- **≡ HEAD (ols):** the CRT-05 change affects only `season="allssa"`.
- **≡ HEAD (pocock):** only `consecutive_k` changed since that run.
- **HEAD:** run on HEAD itself.
- **Fix commit:** run on the commit of that fix.

| Study | Script | Settings | R | Seed base | Result (± MC-SE) | Nominal | Pass | Code state |
|---|---|---|---:|---:|---|---|---|---|
| MC1 ALLSSA spurious cycles | `audit/mc1_lssa.py` | noise; n = 50–500; α = 0.01 | 500–600 | 1001 | 0.0–1.3% (±0.5); pointwise LSSA 3.7–5.5% at 5% | ≤ 1% / 5% | Pass | ≡ HEAD |
| MC2 JUST false jumps | `audit/post_mc2_just_jumps.py` | trend + 2 harmonics; n = 50/150/500; gap | 300–1000 | 2000 | 4.0–5.6% (ols), 2.0% (allssa) (±0.7–1.3) | ≤ 5% | Pass (5.6% at n = 50 within 1 MC-SE) | ≡ HEAD (ols) |
| MC2 after CRT-05 (allssa) | `audit/post_season_sig.py` | ALLSSA in windows, Šidák | 200 | 14000 | 5.0% (±1.5); 2σ power 61% | ≤ 5% | Pass | ≡ HEAD |
| MC3 JUST power | `audit/post_mc2_just_jumps.py` | 2σ / 4σ, n = 150 | 300 | 2000 | 53% / 99% | — | — | ≡ HEAD (ols) |
| MC4 decomposition bands | `audit/mc4_just_ci.py`, `audit/repro_just_ci_curvature.py` | linear / curved / jump truths | 500 (curvature sweep 200) | 4000 / 4400 | 94.0–95.9% | 95% | Pass | ≡ HEAD |
| CRT-05 seasonal bands | `audit/post_season_sig.py` | ALLSSA seasonal | 200 | 14000 | 93.5–94.6% | 95% | Pass (≈ 1 MC-SE low) | ≡ HEAD |
| MC5 monitor FWER | `audit/post_mc5_monitor.py` | n_his = 96; D = 24/96; α = 0.05 | 500 | 5000 | Pocock/OBF 0.2–0.6% (±1.0) | ≤ 5% | Pass (conservative) | ≡ HEAD (pocock) |
| MC5 `consecutive_k` (calibrated) | `audit/post_consecutive_k.py`, `audit/post_consecutive_k_null.py` | k = 2/3; D = 24/96; ols/allssa; irregular | 400–1200 | 15000, 16000 | 4.8–5.3% (D = 24, ±0.7); 3.3% / 5.1% (D = 96, ±1.1 / ±0.6) | ≤ 5% | Pass | HEAD |
| MC5 `consecutive_k` regular sampling | (inline; `FIXES.md`) | D = 24 | 2000 | 99 | 5.15% (±0.5) | 5% | Pass (exact) | HEAD |
| MC6 monitor PIs | `audit/post_mc5_monitor.py` | as MC5 | 500 | 5000 | 94.8–94.9%; mean CI 94.9% | 95% | Pass | ≡ HEAD |
| MC7 STPD null / power | `audit/post_mc7_stpd.py` | 1 TP; n = 150/500; σ = 0.3–1.5 | 1000 | 17000 | null 0.2–0.6% (±0.3); detect 99.5 / 52 / 82 / 6.4% | ≤ 1% | Pass | ≡ HEAD |
| MC7 STPD robust | `audit/post_stpd_robust.py` | Huber; Gaussian / t3 | 500 | 17500 | null 0–0.2%; detect 33–100% | ≤ 1% | Pass | ≡ HEAD |
| MC7 STPD bands | `audit/post_mc7_stpd.py` | given correct detection | 1000 | 17000 | 84.8–95.7% | 95% | **Fail** (CRT-04) | ≡ HEAD |
| MC7b pipeline | `audit/mc7b_pipeline_two_stage.py` | seasonal, 1 TP | 200 | 7700 | structure 96.5%; band 83.2%; slope CI 90.2% | 95% | **Fail** (CRT-04) | ≡ HEAD |
| MC8 robust F-test | `audit/mc8_robust_ftest.py` | Gaussian / t3 / contaminated | 2000 | 8000 | 4.3–6.6% | 5% | Pass | unchanged code |
| MC9 MK/Sen AR(1) | `audit/reaudit_mk.py` | n = 60/150/400; ρ = 0–0.7 | 1000 | 21000 | Type-I 3.5–8.6%; coverage 91.4–96.5% | 5% / 95% | **Fail** for n ≤ 150, ρ ≥ 0.5 (CRT-06); pass otherwise | HEAD |
| MC10 censored slopes | `audit/mc10_censored.py` | 20% / 50% censoring | 300 | 10000 | OLS-MI bias +0.5%, coverage 96.0%; ATS 94.8–95.4% | 95% | Pass | ≡ HEAD |
| Censored detection (single imputation) | `audit/censored_detection.py` | 0/20/50%; JUST ± season; STPD | 300 | 18000 | JUST 2.0–3.0% (6.7% seasonal 50%); STPD 0.3–1.0% | 5% / 1% | Pass (one borderline) | HEAD |
| Censored detection (`detection_imputations=20`) | `audit/censored_detection.py` | as above | 300 | 18000 | JUST 1.3–3.0%; STPD 0.0–1.0% | 5% / 1% | Pass | HEAD |
| Spot checks a–f | `audit/reaudit_spotchecks.py` | see §11 | 400 | 19000 | 4.5% / 1.0% / 0.25% / 2.0% / 0.0% / 3.5% | 5 / 1 / 5 / 5 / 1 / 5% | Pass | HEAD |

---

## 9. Independent Cross-Validation

Re-run on HEAD (`audit/xval_references.py`); unchanged from the original audit:

| Quantity | Reference | Max difference |
|---|---|---|
| WLS coefficients / SEs | `statsmodels.WLS` | 7.9e-15 / 8.9e-15 relative |
| Raw decimal-year cubic trend | `statsmodels` | 3.1e-14 relative (cond 50.6) |
| LSS, standard normalisation (unweighted / weighted / no mean) | `astropy.timeseries.LombScargle` | 1.43e-15 / 1.42e-15 / 1.58e-15 absolute |
| Beta critical values | `scipy.stats.beta.ppf` | ≤ 5.6e-17 |
| Huber coefficients / SE ratio | `statsmodels.RLM` (H1) | 1.8e-4 / 1.0025 (scale-convention difference) |
| Sen slope; MK p and Var(S) | `pymannkendall`, `scipy` | exact / 6e-17 |
| Sen CI | `scipy.stats.theilslopes`, R `trend` convention | exact |

The Licht–Rubin pooling of p-values was checked against Rubin's Wald test on a proper-imputation z-test. It agreed exactly, and both were calibrated (4.6–4.9% at 5% with 20–50% missing; inline check in this re-audit). The median-p rule was anti-conservative at 7.5–12%, so it was not used.

---

## 10. Tooling Results

- **pytest:** 144 passed, 0 failed, 0 skipped (`tests/`). 3 warnings are from ndimpute's ROS on deliberately poor fits.
- **Hypothesis audit properties:** 6/6 passed (`audit/test_hypothesis_audit.py`).
- **ruff:** `ruff check .` is clean. `ruff format --check` would reformat 23 files; this is not enforced in CI and is pre-existing.
- **mypy:** `mypy src/`: no issues in 16 source files.
- **Docs:** `mkdocs build --strict` passes. 21/21 executable documentation blocks pass (`audit/run_doc_examples.py`). The README fragments that use placeholder variables (`evi`, `dates`) run when given synthetic data (`audit/run_doc_fragments.py`). The one stale entry in that harness refers to a snippet no longer in the docs.
- **Fuzzing:** `audit/reaudit_fuzz.py`, 892/892 combinations of new options pass:
  - STPD: `tp_calibration` × `robust` × weights × window mode × censoring × datetime.
  - JUST: `detection_imputations` × season × `robust` × weights × left/right censoring × scope.
  - `just_monitor`: `consecutive_k` × season × window mode × k × horizon × weight kinds × `robust` × forecast covariance × frequency correction.
  - `pool_pvalues` edge cases.

---

## 11. Areas Reviewed and Found Clean

- **Evidence equivalence** (`audit/reaudit_equivalence.py`, 42 cases): HEAD outputs are identical to commit `72f7a77` (where the STPD calibration evidence was produced) for every case. Against `02e47b3` (the CRT-05 evidence), they are identical except for STPD and `analyze_series`; the STPD digests differ by exactly the intended `snr_th` default change. Against `144101f` (the first post-fix studies), ALLSSA, LSSA and OLS slopes are identical; censored Sen/ATS differs only by the separately verified MIN-09 variance fix. The refactors since then therefore did not change any default result:
  - batched max-LSS;
  - cacheable max-F null;
  - batched STPD Monte Carlo;
  - seed-offset parameter in the draw generator.
- **Null spot checks of configurations not otherwise covered** (`audit/reaudit_spotchecks.py`, R = 400):
  - (a) `consecutive_k`, sliding window, vector weights, ALLSSA: 4.5% at 5%.
  - (b) STPD, vector weights, duration windows: 1.0% at 1%.
  - (c) JUST Huber, 30% censoring, `detection_imputations=20`: 0.25% at 5%.
  - (d) JUST `robust="mm"`: 2.0% at 5%.
  - (e) STPD `robust="mm"`: 0.0% at 1%.
  - (f) windowed censoring scope: 3.5% at 5%.
- **New code reviewed line by line:**
  - `_calibrate_consecutive_k` (simulation matches the monitoring rule; the batched spectrum equals the monitor's step statistic to 1e-16);
  - `_stpd_mc_pvalues` (batched t reproduces `_tpd_window`);
  - the JUST multiple-imputation block (each imputation tested at its own Schur-complement best split, which matches brute force);
  - `pool_pvalues`;
  - `window_congenial_imputations`;
  - the `resolve_imputation_dist` fallbacks.
- **Previously clean areas** remain clean: input validation, NaN index mapping, determinism, datetime conversion.

---

## 12. Limitations of This Audit

- Same auditor and methodology as the original audit, and the auditor also implemented the fixes, so this is not an independent re-audit. An external reviewer should repeat at least MC2, MC7b and MC9.
- Original papers were not retrieved (as before). The Licht (2010) pooling rule was checked numerically against Rubin's Wald test, not against the thesis.
- **Reduced replicates:**
  - MC7b and the CRT-05 bands: R = 200.
  - Censored detection: R = 300.
  - Spot checks: R = 400.
  - `consecutive_k` at 96 steps: R = 400–1200.
- MC-SE is reported throughout. Several "pass" results lie within 1 MC-SE of nominal on the liberal side (MC2 n = 50: 5.6%; `consecutive_k` k = 3: 5.3%; ALLSSA `consecutive_k`: 5.1%).
- **Not covered:**
  - interval censoring;
  - full 2D weight matrices in Monte Carlo (the `consecutive_k` calibration falls back for these);
  - heavy-tailed errors in the `consecutive_k` and STPD Monte Carlo calibrations (both assume Gaussian errors; robust STPD was checked with t₃);
  - upstream LSWAVE numerical parity (still unverified);
  - performance;
  - Python < 3.13.
- All designs are synthetic; real-data behaviour was not studied.

---

## Appendix A: Reproduction Scripts Index

| Script | Purpose |
|---|---|
| `audit/reaudit_equivalence.py` | 42-case output digest for code-version equivalence |
| `audit/reaudit_fuzz.py` | Combination fuzzing of post-audit options |
| `audit/reaudit_spotchecks.py` | Null spot checks a–f |
| `audit/reaudit_mk.py` | MK/Sen calibration under AR(1) on the library path (MC9) |
| `audit/censored_detection.py` | Censored detection: single vs. multiple imputation, congenial vs. null-model imputation |
| `audit/post_mc7_stpd.py`, `audit/post_stpd_robust.py` | STPD calibration and power |
| `audit/post_consecutive_k.py`, `audit/post_consecutive_k_null.py` | `consecutive_k` calibration |
| `audit/post_mc2_just_jumps.py`, `audit/post_mc5_monitor.py`, `audit/post_season_sig.py`, `audit/post_monitor_season.py`, `audit/post_sidak_null.py`, `audit/post_ats_variance.py`, `audit/post_mk_variants.py` | Post-fix studies (see `FIXES.md`) |
| `audit/xval_references.py`, `audit/test_hypothesis_audit.py`, `audit/run_doc_examples.py`, `audit/run_doc_fragments.py` | Cross-checks, properties, documentation examples |
| Original audit scripts (`audit/mc*.py`, `audit/repro_*.py`) | Unchanged; see `AUDIT_REPORT.md` Appendix A |

## Appendix B: References (with equation numbers used)

- Andrews, D. W. K. (1993). Tests for parameter instability and structural change with unknown change point. *Econometrica* 61, 821–856 (sup-F statistic).
- Berk, R., Brown, L., Buja, A., Zhang, K., & Zhao, L. (2013). Valid post-selection inference. *Annals of Statistics* 41, 802–837.
- Calonico, S., Cattaneo, M. D., & Farrell, M. H. (2018). On the effect of bias estimation on coverage accuracy in nonparametric inference. *JASA* 113, 767–779.
- Daniels, H. E. (1944). The relation between measures of correlation in the universe of sample permutations. *Biometrika* 33, 129–135.
- Hamed, K. H., & Rao, A. R. (1998). A modified Mann–Kendall trend test for autocorrelated data. *Journal of Hydrology* 204, 182–196.
- Leeb, H., & Pötscher, B. M. (2005). Model selection and inference: facts and fiction. *Econometric Theory* 21, 21–59.
- Licht, C. (2010). *New Methods for Generating Significance Levels from Multiply-Imputed Data*. Doctoral dissertation, University of Bamberg (z-transform pooling).
- Marriott, F. H. C., & Pope, J. A. (1954). Bias in the estimation of autocorrelations. *Biometrika* 41, 390–402.
- Rubin, D. B. (1987). *Multiple Imputation for Nonresponse in Surveys*. Wiley (Rubin's rules, ν = (M−1)(1 + 1/r)²).
- Barnard, J., & Rubin, D. B. (1999). Small-sample degrees of freedom with multiple imputation. *Biometrika* 86, 948–955.
- van Buuren, S. (2018). *Flexible Imputation of Missing Data* (2nd ed.), §5.3. CRC Press.
- Yue, S., & Wang, C. Y. (2004). The Mann–Kendall test modified by effective sample size to detect trend in serially correlated hydrological series. *Water Resources Management* 18, 201–218.
- Further references as in `AUDIT_REPORT.md` Appendix B.


---

## Addendum: Follow-up Fixes of CRT-04 and CRT-06 (2026-11-26)

Both Critical findings above were addressed after this re-audit. The rubric, severity table and caps are unchanged. Evidence is from new Monte Carlo runs on the fixed code.

### CRT-06: fixed
For segments with n ≤ 300, `mk_correction="ar1"` now refers the Mann–Kendall statistic to a parametric AR(1) bootstrap, used for both the p-value and the Sen CI critical value:
- The lag-1 autocorrelation is debiased by simulation.
- Its sampling uncertainty is propagated, and each replicate is studentized exactly like the observed statistic.
- Bootstrap seeds are per data set, so Monte Carlo error averages out across data sets.

Evidence (`audit/reaudit_mk.py`, R = 1000 per cell; nominal 5% / 95%):

| n | Type-I, ρ = 0 / 0.3 / 0.5 / 0.7 | Sen CI coverage, same ρ |
|---|---|---|
| 60 | 4.7% / 6.1% / 3.8% / 6.2% | 95.4% / 93.3% / 95.9% / 93.5% |
| 150 | 3.9% / 5.1% / 6.2% / 5.3% | 96.2% / 94.8% / 93.7% / 94.3% |
| 250 | 4.6% / 6.2% / 5.0% / 6.2% | 94.7% / 93.4% / 94.6% / 93.7% |

Across cells the mean is about 5.3%. Before the fix the same cells reached 8.6% and 91.4%.
- **Cross-checks:** the library's decisions agree with an oracle reference at the true ρ for 99.6% of 1500 data sets. Independent replications of the highest cell pool to 5.3% (4800 data sets).

### CRT-04: fixed for STPD; residual on the pipeline band (re-graded Major)
`stpd()` and `analyze_series()` default to `band_method="bootstrap"`, a detect-then-fit wild bootstrap (documented in `docs/reference/math.md` §7):
- Residuals are leverage-adjusted.
- Every turning point is re-located with STPD's own knot estimator over its detecting window, and the trend is refitted.
- The fixed-structure dispersion is calibrated to the analytical standard error.
- In `analyze_series`, the windowed seasonal and jump-step estimators are re-run with exact vectorised replicas, verified to 1e-14, and the OLS segment slopes are re-estimated.

`band_method="conditional"` and `parity_mode=True` keep the analytical bands.

Evidence (`audit/post_crt04_bands.py`; coverage after correct detection, nominal 95%):

| Design | Bootstrap (default) | Conditional (previous) |
|---|---|---|
| STPD n = 150, σ = 0.3 (R = 300) | 95.5% | 92.1% |
| STPD n = 150, σ = 1.0 | 97.1% | 94.8% |
| STPD n = 500, σ = 0.6 | 97.2% | 85.2% |
| `analyze_series` trend band (MC7b, R = 200) | 93.3% (93.1% on 230 further data sets) | 83.2% |
| `analyze_series` OLS segment slopes (MC7b) | 93.3% (≈ 94% further) | 90.2% |

Development notes:
- **Rejected variants:**
  - Profile-least-squares knot re-estimation made coverage worse (80–85%), because it is more precise than STPD's estimator.
  - A global harmonic seasonal re-fit reached only 86%.
- **Diagnostics behind the final design:**
  - A knot-fixed wild bootstrap under-dispersed the sequential trend estimator (92.3% against 94.5% analytical), which led to the variance calibration.
  - Percentile intervals were about 1 point closer to nominal than basic intervals for the trend bands.
  - The confirmatory runs above used fresh seeds where possible. The canonical seeds match the original MC7/MC7b designs.

**MAJ-06 (residual):** the `analyze_series` trend band covers about 93.2% (pooled; MC-SE ≈ 0.7%) at nominal 95% after correct detection.
- *Location:* `src/tspec/pipeline.py`, `_pipeline_bootstrap`.
- *Confidence:* Confirmed.
- *Grading:* this lies at the boundary of the severity table's Critical example (88–93%). It is graded Major because the band now includes location, seasonal and step uncertainty, and the shortfall is small.
- *Cause:* the remaining shortfall is not reproduced by re-running the full decomposition in each replicate (88.7% with B = 99). It is consistent with selection effects of the windowed ALLSSA frequency choice, which is held fixed.
- *Recommended next step:* a double bootstrap or a calibrated level adjustment.
- All post-detection bands remain conditional on the number of breakpoints found. A missed turning point still leaves the band wrong near it.

### Updated findings and score

| Severity | Open after the addendum |
|---|---|
| 🟥 Blocker | 0 |
| 🟧 Critical | 0 |
| 🟨 Major | 1 (MAJ-06) |
| 🟦 Minor | 5 (MIN-03r, MIN-11, MIN-12, MIN-13, MIN-14) |
| ⬜ Info | 10 (INF-01–05, 07–10, plus INF-11: the bootstrap bands add about 0.2–0.6 s per call, and about 6 s with `refit_mode="l1"`) |

| Category | Score | Weight | Weighted | Change and reason |
|---|---:|---:|---:|---|
| Statistical validity (A–H) | 8.5 | 30% | 2.55 | +0.5: post-detection inference now propagates knot-location uncertainty; short-series MK corrected |
| Calibration of uncertainty (K) | 8.5 | 20% | 1.70 | +1.5: STPD bands 95–97%, MK 3.8–6.2%; residual pipeline band ≈ 93% |
| Code correctness & robustness (J) | 8 | 15% | 1.20 | unchanged (148 tests pass) |
| Numerical stability | 9 | 10% | 0.90 | unchanged |
| Test-suite adequacy & validation | 6.5 | 10% | 0.65 | +0.5: calibration regression tests for both fixes; the high-replicate studies are still not in CI (MIN-11) |
| Documentation accuracy & disclosure | 8 | 10% | 0.80 | unchanged (guides, math, CHANGELOG, README, FIXES updated) |
| Reproducibility | 8 | 5% | 0.40 | unchanged (all bootstraps deterministic) |
| **Total** | | | **8.20 / 10 = 82%** | |

- **Caps:** no Blockers and no Critical issues, so no cap applies. The tests pass (148/148).
- **Updated readiness: 82%**, verdict *usable with stated caveats; fix Major issues before publication-critical use*.
- **Path to ≥ 90%:**
  - MAJ-06: calibrate the pipeline band, or bootstrap the frequency selection.
  - MIN-11: a scheduled CI job running the high-replicate studies.
  - MIN-03r: per-window IRLS convergence reporting.
  - MIN-12: report censored-detection stability by default.
