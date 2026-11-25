# tspec Statistical & Code Correctness Audit

**Commit audited:** `22e2d446300504803efde418b3b11d09c0527659` | **Date:** 2026-11-25 | **Auditor:** Claude Code (Anthropic), automated audit agent
**Environment:** Python 3.13.18, numpy 2.5.3, scipy 1.18.1, pandas 3.0.5, ndimpute 0.1.1 (reference libs: statsmodels 0.15.0, astropy 8.0.3, pymannkendall 1.4.3, hypothesis 6.170.17), Ubuntu 24.04.5 LTS (Linux 6.18 x86_64), 4 CPU cores

All scripts, raw outputs (`*_out.txt`) and JSON results referenced below are in `audit/`. Nothing under `src/`, `tests/`, `docs/` or `validation/` was modified.

---

## 1. Executive Summary

- Overall scientific-use readiness: **40%**: **Not ready; known correctness problems.** The weighted score is 49.5%, capped at 40% because there are Confirmed Blockers.
- Findings: 🟥 4 Blockers | 🟧 5 Critical | 🟨 5 Major | 🟦 10 Minor | ⬜ 6 Info

Headline conclusions:

- **The numerical core is excellent and independently verified.** Weighted least squares, the least-squares spectrum, Beta critical values, the O(n) TPTR variance recursion, Theil–Sen and Mann–Kendall all match statsmodels, astropy, scipy and pymannkendall, or a brute-force computation, to about 1e-14. They are stable from 1e-12 to 1e12 scale, at a 1e13 baseline, and with t ≈ 1.7e9 s. Fixed-model inference is well calibrated: pointwise LSSA false-alarm rate, robust F-test Type-I error, fixed-knot hinge CIs, monitoring prediction intervals, and iid Sen/MK.
- **The change-detection and selection layers do not control false positives at the advertised levels.**
  - `just_jumps` reports a false jump in 98.5–100% of jump-free series with its defaults, and in 71–100% with `jump_alpha=0.05` (BLK-01).
  - ALLSSA declares a "statistically significant cycle" in 18–25% of pure-noise series at α = 0.01 (BLK-02).
- **Several headline confidence bands under-cover badly:**
  - `just_decompose` trend CIs: 37–77% coverage whenever the trend is curved on the window scale (BLK-03).
  - Default OLS slope CIs on censored data: 63% coverage at 50% censoring, with a −27% slope bias from time-agnostic single imputation (BLK-04).
  - Sen "trend_ci" fan: 63% coverage (CRT-03).
  - STPD and pipeline trend bands after detection: 32–92% (CRT-04).
- **Monitoring is mixed.** The default Šidák+Pocock control is valid but extremely conservative (family-wise rate 0.1–0.3% at α = 0.05). `consecutive_k`, recommended for open-ended streams, lets family-wise false alarms reach 10–11.5% over 96 steps (CRT-01). The default `horizon` makes past thresholds depend on how many future observations exist (CRT-02).
- **The package's own validation tier overstates calibration.** Its gates use 25–50 replicates and permissive thresholds (e.g., "ALLSSA FPR ≤ 25% at α = 0.01"; Sen CI coverage of 88% counted as PASS), so it could not detect the problems above (MAJ-05).
- **The documentation is unusually candid in places**: screening p-values, bands conditional on breakpoints, autocorrelation, and the zero-width Sen bowtie are all mentioned. That softens, but does not remove, the scientific risk, because outputs and `explain()` still use unqualified "significant" / "CI" language.

---

## 2. Readiness Assessment

### 2.1 Category Scores

| Category | Score (0–10) | Weight | Weighted | Justification |
|---|---:|---:|---:|---|
| Statistical validity of estimators & tests (A–H) | 4 | 30% | 1.20 | Core estimators (WLS, LSS, Huber, Sen/MK, TPTR, S–H F-test) are correct. However, the JUST jump test is a sup-F statistic referred to a fixed-split F distribution (BLK-01). ALLSSA selection uses per-frequency α (BLK-02). Censored imputation ignores time (BLK-04). MM/L1 breakdown claims are false (MAJ-03). |
| Calibration of uncertainty (section K) | 3 | 20% | 0.60 | Passes: MC1 pointwise, MC5 Pocock/OBF, MC6, MC8, MC9 (iid), fixed-knot MC7, linear-truth MC4. Fails, often severely: ALLSSA (MC1), MC2 (with `jump_alpha`), MC4 curved trends (37–77%), seasonal ALLSSA bands (90–92%), MC5 `consecutive_k` (10–11.5%), MC7 after detection (32–92%), MC9 AR(1) (23–25% Type-I), MC10 OLS (63%). |
| Code correctness & robustness to edge cases (J) | 7 | 15% | 1.05 | No crashes in fuzzing. Strict input validation (unsorted, duplicates, NaT, inf). Exact NaN index mapping, no mutation of inputs, deterministic by default, global RNG untouched. Minor silent issues: datetime overflow, years-only defaults, unsurfaced convergence flags. |
| Numerical stability | 8 | 10% | 0.80 | Whitened thin-QR with SVD fallback, column scaling, and shift-invariant centring. Verified to 1e-14 against statsmodels, including raw decimal-year cubic trends. The ill-conditioning warning threshold (1e10) is too lax (MIN-04). |
| Test-suite adequacy & independent validation | 4 | 10% | 0.40 | 119 tests pass, Hypothesis invariance tests, and 41/41 upstream parity checks. But no test checks calibration of any headline statistical claim at adequate replicates; validation MC gates use R = 25–50 with permissive thresholds, so none of the Blockers would be caught. |
| Documentation accuracy & disclosure | 5 | 10% | 0.50 | Good disclosures (screening p-values, conditional bands, autocorrelation, Sen bowtie). Against that: incorrect claims (Yohai MM, L1 50% breakdown, a "conservative" blend that ignores bias), overstated validation numbers, stale counts, and `explain()` text that overstates significance. |
| Reproducibility (determinism, pinning, installability) | 8 | 5% | 0.40 | Installs cleanly. Tests pass on numpy 2.5 / pandas 3 / scipy 1.18. Seeded RNG by default. The mypy config is broken with current numpy stubs, there are no upper bounds on core dependencies, and ndimpute is a 0.x dependency. |
| **Total** | | 100% | **4.95 / 10 = 49.5%** | |

### 2.2 Caps Applied

- 4 Confirmed BLOCKERS → **overall score capped at 40%**.
- The test suite passes (119/119) and the package installs, so the 25% cap does not apply.
- **Final overall readiness: 40%** (min(49.5%, 40%)). Verdict band 25–49%: *Not ready; known correctness problems.*

### 2.3 Per-Module Readiness

| Module | Readiness | Main reasons |
|---|---:|---|
| LSSA (`lssa`) | 85% | Matches astropy LS to 1.4e-15; Beta null exact (pointwise FAR 3.8–5.3% at 5%). Only issue: `explain()` counts per-frequency exceedances as "significant" (MAJ-02). |
| ALLSSA (`allssa`) | 35% | Spurious "significant cycles" 18–25% at α = 0.01 (BLK-02). Default grid assumes years (MAJ-04). |
| JUST jumps (`just_jumps`) | 30% | 98.5–100% false positives with defaults; 71–100% with `jump_alpha=0.05` (BLK-01). The analyze-style filter is calibrated (1.3–5.4%) but has low power (e.g., 25% at 2σ, n = 150). |
| JUST decomposition (`just_decompose`) | 45% | Exact reconstruction and well-behaved point estimates. Trend CIs fail for curved trends (BLK-03). Seasonal CIs 90–92% with ALLSSA (CRT-05). |
| JUSTmonitor (`just_monitor`) | 60% | Default family-wise control valid but extremely conservative. PIs well calibrated (94.4–95.0%). `consecutive_k` not controlled (CRT-01). Horizon look-ahead (CRT-02). |
| STPD / TPTR | 60% | Fixed-knot inference exact (coverage 94.4–95.6%). The SNR > 1 hard gate misses turning points regardless of n; post-detection bands under-cover (CRT-04). |
| Robust (Huber / MM / L1 / S–H test) | 65% | Huber matches statsmodels RLM. Robust F-test Type-I 4.3–6.6%. MM/L1 breakdown claims false (MAJ-03). |
| Sen / ATS / MK | 65% | Exact vs pymannkendall/scipy; iid coverage 94.6–95.6%. No serial-correlation correction (MAJ-01). Fan mislabelled as trend CI (CRT-03). ATS conservative (MIN-09). |
| Censored (`ndimpute` integration) | 30% | Time-agnostic single imputation: −27% slope bias, 63% coverage at 50% censoring with default OLS (BLK-04). Parsing, guardrails and DL-shift detection behaved sensibly in the cases tested. |
| Pipeline (`analyze_series`) | 40% | Stage-1 jump control is calibrated (1.3–5.4%). But stage-2 bands cover 82–84% and slopes 92% even when the structure is right (CRT-04). Sen fan (CRT-03). Censored OLS default (BLK-04). |

### 2.4 Why It Is Ready

- **Exact, stable linear algebra.** WLS coefficients and SEs match `statsmodels.WLS` to 7.9e-15 and 8.9e-15 relative (jumps + harmonics + 1D weights, decimal-year axis); `audit/xval_references.py`.
- **LSS is exactly the standard Lomb–Scargle/LSSA quantity.** Max abs difference from `astropy.timeseries.LombScargle` (standard normalisation) is 1.4e-15, weighted and unweighted. Beta critical values equal `scipy.stats.beta.ppf` to 5.6e-17. Pointwise false-alarm rates are 3.8–5.3% at α = 0.05 (MC1).
- **TPTR "exact O(n)" variance is exact.** Max relative difference from a brute-force smoother-matrix computation is 4e-16. Fixed-knot curve coverage is 94.4–95.6% for the sequential, joint and Huber modes (MC7); `audit/repro_tptr_variance.py`.
- **Robust nested F-test (Schrader–Hettmansperger) is calibrated.** Type-I error is 4.3–6.6% at 5% under Gaussian, t₃ and 10% contaminated noise for n = 50–500 (MC8).
- **Monitoring prediction intervals are correct.** Out-of-sample PI coverage is 94.4–95.0%, flat across the horizon (MC6). Default Šidák+Pocock/OBF family-wise rate is ≤ 0.3% at α = 0.05 (MC5).
- **Robust input handling.** No mutation of inputs. Exact NaN-drop index mapping (Hypothesis property). Determinism, and the global numpy RNG is untouched. Datetime → decimal-year conversion is leap-year correct and tz-aware (UTC). Unsorted, duplicate or NaT times are rejected loudly.
- **Candid documentation** of screening p-values, conditional bands, autocorrelation risk and the Sen bowtie pinch (docs/guide/interpreting-results.md).

### 2.5 Why It Is Not Ready

- Jump screening does not control false positives under defaults or `jump_alpha` (**BLK-01**).
- ALLSSA "significant cycles" are ~20× over nominal (**BLK-02**).
- Trend bands in the decomposition ignore window-scale bias (**BLK-03**).
- Censored-data trend inference is biased and falsely precise (**BLK-04**).
- `consecutive_k` monitoring is not family-wise controlled (**CRT-01**); monitoring thresholds depend on future data (**CRT-02**).
- The Sen fan is exported as `trend_ci_*` (**CRT-03**).
- Post-detection bands under-cover, and the STPD SNR gate is blind to significant turning points (**CRT-04**).
- Seasonal and forecast-mean bands under ALLSSA under-cover (**CRT-05**).
- No autocorrelation correction for MK/Sen (**MAJ-01**).
- Breakdown claims are false (**MAJ-03**).
- Defaults silently assume time in years (**MAJ-04**).
- The validation tier is too weak to support its claims (**MAJ-05**).

### 2.6 Path to ≥90%

Minimum set of fixes, each with a regression test of ≥ 1000 replicates:

1. **JUST jump inference (BLK-01).** Calibrate the window statistic as a sup-F (e.g., Andrews 1993 critical values, or a per-window parametric/permutation bootstrap of max-F). Control the family-wise rate across windows. Make the analyze-style filter (Bonferroni + step test) the `just_jumps` default. Never print "Detected N abrupt changes" without stating the error-rate control in force.
2. **ALLSSA (BLK-02).** Accept a frequency only if the max-statistic over the searched grid exceeds a family-wise threshold: an effective-number-of-frequencies Šidák bound (Horne & Baliunas 1986), Baluev (2008) FAP, or a bootstrap of max LSS. Otherwise rename the output from "significant" to "selected".
3. **Decomposition bands (BLK-03).** Use bias-aware intervals (local-quadratic windows, undersmoothing, or robust bias-corrected intervals, Calonico, Cattaneo & Farrell 2018), or relabel them as "variance-only bands". Add a curved-trend coverage test.
4. **Censored trends (BLK-04).** Impute conditionally on time (censored regression / Tobit MLE with the trend and seasonal design, iterated with the analysis model). Implement proper multiple imputation with Rubin's rules (T = W̄ + (1 + 1/m)B). Default to ATS for slopes when censoring is present.
5. **Monitoring (CRT-01, CRT-02).** Calibrate per-step α for `consecutive_k` by simulating the run-length distribution, or drop the "protection at nominal α" claim. Require an explicit `horizon` for spending rules (never infer it from the data length).
6. **Bands (CRT-03, CRT-04, CRT-05).** Stop exporting the Sen fan as `trend_ci_*`; provide a real pointwise band, e.g., a bootstrap of (intercept, slope). Offer a bootstrap of the full detect-then-fit procedure for STPD/pipeline/ALLSSA bands. Replace the hard SNR > 1 gate with an n-aware criterion.
7. **MK/Sen (MAJ-01).** Add a Hamed–Rao or Yue–Wang variance correction, or a block bootstrap.
8. **Documentation (MAJ-03, MAJ-05).** Correct the MM/L1 breakdown claims. Regenerate the validation tier with ≥ 1000 replicates and gates of the form "within ±2 MC-SE of nominal".

---

## 3. Blockers

### BLK-01: `just_jumps` does not control false positives under its defaults or with `jump_alpha` (98.5–100% and 71–100% false-positive rates)
- **Location:** `src/tspec/just.py#L444-L577` (scan selects the RSS-minimising split), `#L580-L767` (F-test computed *at the selected split*; `f_dist.sf(f_jump, 2, dof2)` at L762-L767), `#L877-L907` (post-merge filters), `#L926-L931` (defaults `mag_th=0.05, dir_th=0.01, jump_alpha=None, bonferroni_window=False`), `#L1031-L1052` (scale warning recommends `jump_alpha=0.05`).
- **Confidence:** Confirmed
- **Affected API:** `just_jumps(...)` defaults; `jump_alpha`, `step_alpha`, `bonferroni_window`, `mag_th`, `dir_th`; `JumpEvent.jump_pvalue`; `JustJumpResult.explain()`.
- **Description:** In each window, every interior split is scanned and the split that minimises the RSS is chosen. `jump_pvalue` is then the F(2, n_w − n_c) tail probability at that split. That is the classic sup-F statistic, whose null distribution is far heavier than F(2, ·) (Andrews 1993). The procedure is then repeated across ~n/M overlapping windows. With the default `jump_alpha=None`, only the non-statistical thresholds `mag_th=0.05` OR `dir_th=0.01`/unit time apply. The `dir_th` threshold is met by almost any noisy slope difference.
- **Evidence / Reproducer:** `audit/mc2_3_just_jumps.py` (MC2). Jump-free series: trend + 2 harmonics + N(0, 0.02²) (NDVI-like scale, so no `THRESHOLD_SCALE_MISMATCH` warning fires). The probability of at least one reported jump, by filter:

  | configuration | A: defaults | B: `jump_alpha=0.05` | C: B + `bonferroni_window=True` | D: C + `step_alpha=0.05` (= `analyze_series`) |
  |---|---:|---:|---:|---:|
  | ols, n=50 (R=1000) | 99.2% | 71.0% | 8.9% | 4.6% |
  | ols, n=150 (R=1000) | 98.5% | 90.6% | 5.7% | 1.4% |
  | ols, n=150, 25% gap | 99.0% | 86.9% | 4.9% | 1.9% |
  | ols, n=500 (R=500) | 100% | 100% | 16.4% | 5.4% |
  | allssa, n=150 (R=400) | 98.8% | 81.5% | 3.0% | 1.8% |
  | allssa, n=150, 25% gap (R=400) | 99.3% | 78.3% | 4.8% | 1.3% |

  MC-SE at 5% is 0.69 pp (R = 1000), 0.97 pp (R = 500) and 1.09 pp (R = 400). A single-series demonstration (`audit/repro_reporting.py`, R-C) prints: *"Detected 2 abrupt change(s) in the trend component … screening F-test p=3.44e-02 … p=2.38e-03"*, with an empty warning list, on a series that has no jump.
- **Why expected is correct:** For a pre-specified split, (RSS₀ − RSS₁)/2 ÷ RSS₁/(n − p) ~ F(2, n − p). For a data-chosen split, the statistic is max_j F_j, whose tail is heavier (Andrews 1993; Davies 1987). Across many windows the family-wise rate compounds further. The observed 71–100% against a nominal 5% is consistent with this theory.
- **Scientific impact:** Inflated false positives. Spurious "abrupt changes" are reported in essentially every series under defaults, and in most series under the `jump_alpha` cutoff that the docstring and the scale warning suggest. Only `analyze_series` (configuration D) is calibrated, and it has low power (2σ jump: 3–26%; 4σ: 16–94% depending on n; MC3).
- **Recommended fix:** Calibrate per-window p-values against the sup-F distribution (Andrews' asymptotic critical values, or a parametric bootstrap of the max-F under the null model). Control the family-wise rate across windows (e.g., Bonferroni over the number of non-overlapping windows, or a max-over-windows bootstrap). Make configuration D the default. In `explain()`, state the error rate in force, or say "candidate" instead of "detected".
- **Suggested regression test:** 1000 jump-free series (n = 150 and 500, white noise). Assert `mean(any jump reported) <= alpha + 2*MCSE` for the default call and for `jump_alpha=0.05`.

### BLK-02: ALLSSA declares "statistically significant cycles" in 18–25% of pure-noise series at α = 0.01
- **Location:** `src/tspec/spectral.py#L344-L407` (iterative selection; acceptance `res_fine.spectrum[fine_idx] > res_fine.critical_value` at L396-L403), `#L174-L179` (per-frequency Beta critical value), `#L640-L656` (`explain()`: "identified N statistically significant cycle(s)").
- **Confidence:** Confirmed
- **Affected API:** `allssa()`; `just_jumps/just_decompose/just_monitor(season="allssa")` (the default), which select harmonics this way.
- **Description:** At each iteration, the frequency maximising the LSS over the coarse grid is located, then refined over a fine grid (11 points at `decimal=1`). The maximum is compared with the per-frequency critical value 1 − α^{2/ν}. That threshold is correct for one pre-specified frequency, not for a maximum over ~10–100 effectively independent frequencies.
- **Evidence / Reproducer:** `audit/mc1_lssa.py` (MC1). Gaussian white noise, irregular sampling (~24 obs/yr), default `allssa(t, y)`, R = 600 per cell:

  | n | gap | P(≥1 frequency selected) | MC-SE | nominal |
  |---|---|---:|---:|---:|
  | 50 | no / yes | 24.7% / 20.0% | 1.8 / 1.6 pp | 1% |
  | 150 | no / yes | 21.8% / 21.2% | 1.7 pp | 1% |
  | 500 | no / yes | 18.2% / 20.5% | 1.6 pp | 1% |

  Example (`audit/repro_reporting.py`, R-B, seed 201): *"Anti-leakage spectral search (ALLSSA) identified 1 statistically significant cycle(s) simultaneously with the linear trend: 4.000 (amplitude 0.31)."* on pure noise. On a single-sinusoid series, `allssa` also returned a spurious 12 cy/yr component (`audit/repro_time_inputs.py`).
- **Why expected is correct:** Under H₀ the LSS at a fixed frequency is Beta(1, (n − q − 2)/2) (Pagiatakis 1999; Ghaderpour & Pagiatakis 2017, normalised LSS section), and tspec reproduces this exactly (MC1 pointwise 3.8–5.3%). The false-alarm probability of the *maximum* over K effective frequencies is 1 − (1 − α)^{K_eff} (Scargle 1982; Horne & Baliunas 1986; Baluev 2008). With K_eff ≈ 20–25, that gives ≈ 18–22%, matching the observation. The package's own validation measured 7.5% with R = 40, and the gate allowed up to 25% (`validation/bench_mc.py#L99`).
- **Scientific impact:** Spurious periodicities are reported as significant at 1%. In JUST/monitor (default `season="allssa"`), spurious harmonics enter the trend/seasonal model and its uncertainty (see CRT-05).
- **Recommended fix:** Use a family-wise acceptance threshold for the maximum (Šidák over an effective number of frequencies, Baluev's analytical FAP, or a parametric bootstrap of max LSS under the current model). Alternatively, rename the outcome "selected (per-frequency α)" in `explain()` and the docs.
- **Suggested regression test:** 1000 white-noise series at n ∈ {50, 150, 500}. Assert `mean(len(allssa(t,y).frequencies) > 0) <= 0.01 + 2*MCSE`.

### BLK-03: `just_decompose` trend CIs ignore window-scale (local-linear) bias, giving 37–77% coverage for smoothly curved trends
- **Location:** `src/tspec/just.py#L1440-L1509` (`_window_component_half_widths`: variance only), `#L1796-L1822` and `#L1887-L1896` (Gaussian-weighted average of half-widths); claim in the docstring `#L1544-L1547` ("conservative Cauchy–Schwarz … blend") and docs/reference/math.md §8.1.
- **Confidence:** Confirmed
- **Affected API:** `JustDecomposeResult.trend_ci_low/high`, `trend_reg_ci_*`; `analyze_series(...).trend_ci_*` when STPD is off; plotted ribbons.
- **Description:** Each 3-year window fits a 2-piece linear trend plus harmonics. The pointwise half-width is t·SE from that window's covariance, and half-widths are averaged with the blending weights. The averaging is a valid upper bound for the *standard deviation* of the blended estimator: SD(Σ aₖXₖ) ≤ Σ aₖ SD(Xₖ). That bound does nothing about the bias of a local-linear fit to a curved trend. The bias does not shrink with n, so coverage collapses as n grows.
- **Evidence / Reproducer:** `audit/mc4_just_ci.py` (MC4) and `audit/repro_just_ci_curvature.py`, σ = 0.3, annual seasonal, ~24 obs/yr. Pointwise 95% trend coverage (mean over points; MC-SE 0.3–0.8 pp):

  | truth | season | n=150 | n=150 gap | n=500 |
  |---|---|---:|---:|---:|
  | A: linear trend | ols | 94.8% | 95.6% | 95.5% |
  | B: 0.6·sin(2π t/9 yr) | ols | **75.4%** (middle half 66%) | **76.8%** (middle 46%) | **70.1%** |
  | B | allssa | **72.3%** | – | **66.0%** |
  | C: linear + known jump | ols | 95.7% | 95.4% | 95.4% |

  Dose-response (n = 500, A = 0.6): period 25 yr → 95.4%; 15 yr → 90.6%; 9 yr → **70.5%**; 6 yr → **37.6%**. Seasonal coverage stays at 95% with `season="ols"`, so the failure is specific to the trend.
- **Why expected is correct:** For a local-linear smoother with half-width h, E[ĝ(t)] − g(t) ≈ ½ h² μ₂ g″(t) (Fan & Gijbels 1996, Thm 3.1). An interval ĝ ± t·SD has asymptotic coverage → 0 whenever bias/SD does not vanish, and that holds for fixed windows as n → ∞. The observed monotone decline with curvature and n matches this.
- **Scientific impact:** CIs are too narrow. Users conclude that the true trend lies in a band it misses at 25–63% of time points, typically around maxima and minima, which are the features of scientific interest. The docs call the band "conservative" without qualification.
- **Recommended fix:** Use bias-aware intervals: local-quadratic windows; undersmoothing combined with a separate point estimate; robust bias-corrected intervals (Calonico, Cattaneo & Farrell 2018); or a sandwich that adds an estimated squared bias. At minimum, relabel the band as a variance-only band, document the local-linearity assumption, and warn when a quadratic term is significant within windows.
- **Suggested regression test:** Truth B at n = 500, R = 500. Assert mean pointwise coverage ≥ 0.93, or that a bias warning is emitted.

### BLK-04: Censored data: time-agnostic single imputation attenuates trends (−27% at 50% censoring) and makes default OLS CIs falsely precise (63% coverage)
- **Location:** `src/tspec/censored.py#L366-L451` (`_call_ndimpute_single`: marginal model of the values only, no time covariate), `#L1243-L1260` (global imputation used downstream), `#L869-L957` (stability draws re-run detection only; no pooling of estimates or SEs); `src/tspec/pipeline.py#L549-L583` (default `slope_estimator="ols"` CIs computed on the single imputed series), `#L656-L659`.
- **Confidence:** Confirmed
- **Affected API:** All functions given censored input; especially `analyze_series(..., slope_estimator="ols")` (the default) segment slopes, CIs and p-values; also JUST/STPD/decomposition fits on censored series.
- **Description:** `ndimpute` fits one stationary distribution to all values (ROS/parametric) and draws each censored value from its lower tail below the DL, independently of t. Two consequences follow:
  - **Bias.** Censored observations lose their relationship with time, which flattens trends.
  - **False precision.** The single imputed series is analysed as if fully observed. There is no multiple imputation and no Rubin's rules, so SEs exclude imputation variance.

  The `n_stability_draws` option reports only whether *detections* recur; it does not pool estimates.
- **Evidence / Reproducer:** `audit/mc10_censored.py` (MC10). Truth y = 2 + 0.1(t − 2000) + N(0, 0.4²), n = 150, constant DL; R = 500:

  | censoring | estimator | bias (truth 0.1) | 95% CI coverage | uncensored same-estimator coverage |
  |---|---|---:|---:|---:|
  | 20% | OLS (default) | −0.0069 (−7%) | 92.4% | 94.2% |
  | 20% | ATS (`"sen"`) | −0.00003 | 95.2% | 94.4% |
  | 50% | OLS (default) | **−0.0269 (−27%)** | **63.4%** | 94.6% |
  | 50% | ATS (`"sen"`) | +0.0009 | 98.0% | 94.4% |

  MC-SE of coverage is ≈ 1.0 pp. Mechanism (`audit/repro_censoring_mechanism.py`): among censored points, corr(t, true y) = 0.25, but corr(t, imputed y) = −0.007. An OLS slope through the censored points alone is 0.038 for the true values and −0.0007 for the imputed ones.
- **Why expected is correct:** Imputation must be congenial with the analysis model, i.e., condition on the covariates used in the analysis (here time) (Meng 1994; Rubin 1987; Helsel 2012, ch. 12, censored regression for trends). With single imputation, the total variance T = W̄ + (1 + 1/m)B is underestimated by omitting B (Rubin 1987). ATS, which uses the censored ranks directly, is unbiased in the same data, confirming that the bias comes from the imputation step.
- **Scientific impact:** Trends in censored environmental records are understated and declared with CIs that are too narrow. The default pipeline estimator gives a confidently wrong slope (63% coverage) at 50% censoring. The warnings ("results depend on distributional assumptions") do not convey this.
- **Recommended fix:**
  - Impute from a censored-regression model that includes the analysis design (trend + seasonality + known jumps), e.g., Tobit / `scipy.stats.CensoredData` MLE on residuals, iterated.
  - Provide multiple imputation (m ≥ 20) with Rubin-pooled estimates and SEs.
  - Default `slope_estimator` to ATS when censoring is present.
  - Warn explicitly that OLS on imputed values is biased toward zero.
- **Suggested regression test:** The MC10 design at 50% censoring, R = 500. Assert |bias| < 0.1·|β| and coverage ≥ 0.93 for the default `analyze_series` call.

---

## 4. Critical Issues

### CRT-01: `sequential_control="consecutive_k"` does not control the family-wise false-alarm rate (10–11.5% at α = 0.05 over 96 steps)
- **Location:** `src/tspec/monitor.py#L363-L364` (`consecutive_k` returns the unadjusted α per step), `#L1030-L1034` (trigger), `#L280-L284` (`explain()`: "Repeated-testing protection active … at nominal alpha"), `#L892-L895` (the `HORIZON_EXCEEDED` warning recommends switching to `consecutive_k`); docs: docstring L556-L557, docs/guide/interpreting-results.md table.
- **Confidence:** Confirmed
- **Affected API:** `just_monitor(sequential_control="consecutive_k", consecutive_k=k)`.
- **Description:** Each step is tested at α_step = α (Šidák across frequencies only), and an alarm requires k consecutive exceedances. Consecutive statistics share nearly the whole window, so they are strongly positively correlated. Runs therefore occur at a rate close to the single-exceedance rate, and the family-wise rate grows without bound with the horizon.
- **Evidence / Reproducer:** `audit/mc5_6_monitor.py` (MC5), stable series, α = 0.05, R = 1000 (MC-SE 0.69 pp):

  | history model | D = 24 steps | D = 96 steps |
  |---|---:|---:|
  | ols, consecutive_k=2 | 2.2% | **10.0%** |
  | allssa, consecutive_k=2 | 2.8% | **11.5%** |
  | (reference) ols, sidak+none | 3.0% | 12.5% |
- **Why expected is correct:** P(alarm within D steps) ≥ P(run of length k somewhere), which increases monotonically in D for any fixed per-step level (Balakrishnan & Koutras 2002, runs theory). No bound ≤ α exists without D-dependent adjustment.
- **Scientific impact:** Inflated false alarms in long or open-ended monitoring, which is exactly the setting where `consecutive_k` is recommended. `explain()` asserts protection at nominal α.
- **Recommended fix:** Calibrate the per-step level for given (k, D), or for a target average run length, by simulation under the fitted null. Or remove the "protection at nominal α" text and the recommendation in `HORIZON_EXCEEDED`.
- **Suggested regression test:** MC5 design with D = 96 and R = 1000. Assert FWER ≤ α + 2·MCSE.

### CRT-02: Monitoring look-ahead: by default the threshold at step d depends on how many observations exist after d
- **Location:** `src/tspec/monitor.py#L877-L878` (`eff_horizon = horizon if given else n_mon`, the count of all monitoring observations in the input), `#L355-L380` (per-step budget depends on H), docstring L562-L565.
- **Confidence:** Confirmed
- **Affected API:** `just_monitor(sequential_control in {"pocock","obrien_fleming"}, horizon=None)` (the default).
- **Description:** With the default `horizon=None`, the spending plan is set from the total number of monitoring observations present, including future ones. Re-running the monitor as data arrive, or analysing a longer record retrospectively, changes the thresholds applied to *past* steps, and therefore which alarms occur and when.
- **Evidence / Reproducer:** `audit/repro_monitor_lookahead.py`. Identical data up to each step, with a modest disturbance (−1.7σ) from step 6:

  ```
  data available up to step   12: horizon=  12 step-1 crit=0.5432; first exceedance at step None; detected=False
  data available up to step   48: horizon=  48 step-1 crit=0.5951; first exceedance at step 18;  detected=True
  data available up to step  144: horizon= 144 step-1 crit=0.6320; first exceedance at step None; detected=False
  explicit horizon=144 → step-1 crit=0.6320 regardless of data length
  ```

  The alarm at step 18 exists or vanishes depending on observations 49–144. Family-wise error was not inflated in a realistic re-run-every-step loop (0/400 alarms; `audit/mc5b_monitor_realtime_loop.py`), because the procedure is very conservative. That is why this is Critical rather than Blocker.
- **Why expected is correct:** Lan–DeMets spending requires the information fraction t_d = d/H to be defined by a maximum H fixed *before* monitoring (Lan & DeMets 1983; Jennison & Turnbull 2000, §7.2). Using H = (observed length) makes past decisions depend on future data.
- **Scientific impact:** Retrospective "detection dates/delays" reported from tspec are not what a real-time operator would have seen. Alarms are not reproducible as records grow.
- **Recommended fix:** Require an explicit `horizon` for spending rules (raise if None), or default to an open-ended rule. Never derive H from the input length.
- **Suggested regression test:** Assert `steps[0].critical_value` is identical for `t[:n_his+12]` and `t[:n_his+144]` under default arguments.

### CRT-03: The Sen/ATS "slope-CI fan" is exported as `trend_ci_low/high`, with 63% pointwise coverage (8.6% near the segment median)
- **Location:** `src/tspec/pipeline.py#L434-L442` (fan pivots at (t_med, ŷ_med): zero width at t_med), `#L1118-L1121` (fan copied into `PipelineResult.trend_ci_low/high`), `#L235-L237` (`to_dataframe()` columns `trend_ci_low`, `trend_ci_high`); `src/tspec/plotting.py#L488-L497`.
- **Confidence:** Confirmed
- **Affected API:** `analyze_series(..., slope_estimator="sen")`: `.trend_ci_low/high`, `.to_dataframe()`, `.plot(show_ci=True)`.
- **Description:** The band is ŷ_med + β_{lo/hi}(t − t_med). It encodes slope uncertainty only and has zero width at the pivot, yet it is stored in the same fields (and exported under the same column names) as genuine pointwise trend CIs from other estimators, alongside `ci_level=0.95`. A docs note (interpreting-results.md, "bowtie") explains the pinch, but the API presents it as a CI.
- **Evidence / Reproducer:** `audit/repro_reporting.py` (R-D). Linear truth, n = 200, R = 1000: minimum band width 3.1e-4 at t = 2005.331 (median t = 2005.337). Mean pointwise coverage of the true line is **63.1%** (MC-SE 0.7 pp), and **8.6%** within ±0.5 yr of the median.
- **Why expected is correct:** A pointwise 1 − α CI for g(t) must have positive width wherever Var(ĝ(t)) > 0. The intercept is estimated, so Var(ĝ(t_med)) > 0, and coverage at t_med → 0.
- **Scientific impact:** CIs are too narrow. Exported `trend_ci_*` columns will be used as trend CIs in downstream figures and tables.
- **Recommended fix:** Store the fan under a distinct name (e.g., `slope_fan_low/high`) and leave `trend_ci_*` as `None`. Or provide a genuine band, e.g., a pairs/block bootstrap of (intercept, slope), or Conover-type nonparametric intervals.
- **Suggested regression test:** Linear truth, R = 1000. Assert coverage of `trend_ci_*` ≥ 0.93, or that `trend_ci_low is None` when `slope_estimator="sen"`.

### CRT-04: Post-detection trend and slope bands from `stpd` / `analyze_series` under-cover (82–92% when the structure is detected correctly; 32–41% when STPD's SNR > 1 gate misses a real turning point)
- **Location:** `src/tspec/stpd.py#L1342-L1350` (acceptance requires `snr_val > snr_th` with `snr_th=1.0`, independent of n), `#L1087-L1093` (TPTR CI conditional on accepted TPs), `#L1729-L1743`; `src/tspec/pipeline.py#L800-L819` (stage-1 seasonal and step offsets subtracted as if known), `#L1076-L1085` (`trend_ci` + estimated offsets with no added variance), `#L549-L583` (segment OLS on the deseasonalised series).
- **Confidence:** Confirmed
- **Affected API:** `detect_turning_points/stpd(...).piecewise_trend.trend_ci_*`; `analyze_series(...).trend_ci_*`, `.segments[*].slope_ci_*`, `pvalue`.
- **Description:**
  1. Bands and slope CIs condition on detected knots, estimated seasonality and step offsets. There is no correction for selection or stage-1 estimation.
  2. The hard SNR gate (centred SNR = ‖fit − mean‖ / ‖residual‖ > 1) rejects turning points whose within-window trend range is comparable to σ, however small the p-value and however large n. The final band is then a straight-line band around a bent truth.

  The docs say bands are "conditional on the detected breakpoints" but do not disclose these failure modes.
- **Evidence / Reproducer:** `audit/mc7_stpd.py`, `audit/mc7b_pipeline_two_stage.py`, `audit/repro_stpd_power.py`:

  | setting (R=1000 unless noted) | fixed true knot (seq/joint/huber) | after `stpd()` detection | TP detection rate |
  |---|---|---:|---:|
  | n=150, σ=0.3 | 95.3 / 94.6 / 94.4% | 91.7% | 99.4% |
  | n=150, σ=1.0 | 95.4 / 95.2 / 95.3% | **31.8%** | 0.0% |
  | n=500, σ=0.6 | 95.6 / 94.5 / 94.6% | **41.4%** | 33.8% |
  | `analyze_series` defaults, seasonal + TP, n=288 (R=200) | – | trend band **83.5%**, segment slopes **91.9%** (correct structure, 90% of runs) | – |

  In `repro_stpd_power.py`, the window centred on a true TP had p = 4.9e-5 but centred SNR = 0.63. The SNR gate rejected it in 100/100 replicates.
- **Why expected is correct:** Post-selection and plug-in intervals are anti-conservative unless the selection or nuisance-estimation variability is included (Leeb & Pötscher 2005; Berk et al. 2013). For segmented regression, knot uncertainty must enter the variance (Hinkley 1971; Muggeo 2003). The fixed-knot results show that the formulas themselves are right; the losses come entirely from selection and the gate.
- **Scientific impact:** Turning-point trend bands and segment slope CIs are too narrow. When STPD misses a TP, the reported band confidently excludes the true trend at most time points.
- **Recommended fix:** Provide a parametric or block bootstrap of the whole detect-then-fit procedure. Replace the absolute SNR gate with an n-aware effect-size or information criterion, or make it optional. Propagate the stage-1 seasonal and offset covariance into stage-2 bands (or estimate them jointly).
- **Suggested regression test:** MC7 n = 500, σ = 0.6 (R = 500). Assert detection ≥ 0.8, and assert band coverage ≥ 0.9 under the default pipeline.

### CRT-05: Seasonal CIs (`just_decompose`) and forecast-mean CIs (`just_monitor`) with the default `season="allssa"` under-cover even when the model is correct (90–92% and 92–93%)
- **Location:** `src/tspec/just.py#L1721-L1743` (covariance conditional on ALLSSA-selected frequencies), `#L1796-L1806`; `src/tspec/monitor.py#L726-L748`, `#L829`.
- **Confidence:** Confirmed
- **Affected API:** `just_decompose(season="allssa")`: `seasonal_ci_*`; `just_monitor(season="allssa")`: `forecast_ci_*`.
- **Description:** Frequencies are selected adaptively (BLK-02), then the covariance is computed as if they were known. Missed weak harmonics and spurious extra ones both enter the band without being accounted for.
- **Evidence / Reproducer:** MC4 (`audit/mc4_just_ci.py allssa 400`): seasonal coverage 90.3% (n = 150, linear truth, MC-SE 0.5 pp), 91.6% (n = 500, MC-SE 0.35 pp), 91.1% / 91.7% (truth with known jump). The same truth with `season="ols"` gives 95.3–95.5%. MC5/MC6: forecast-mean CI coverage of μ is 92.0% (D = 24) and 92.9% (D = 96) with ALLSSA history, against 94.9% with OLS. The PI is unaffected (94.4–95.0%).
- **Why expected is correct:** As in CRT-04 (Leeb & Pötscher 2005). The OLS-season control isolates selection as the cause.
- **Scientific impact:** Seasonal amplitude and forecast uncertainty are understated by roughly 3–5 coverage points.
- **Recommended fix:** Bootstrap the ALLSSA selection, inflate by a selection-aware factor, or fix frequencies a priori (`season="ols"`) when CIs are needed. Document the choice.
- **Suggested regression test:** MC4 truth A with `season="allssa"`, R = 500. Assert seasonal coverage ≥ 0.93.

---

## 5. Major Issues

### MAJ-01: Mann–Kendall / Sen inference has no serial-correlation correction (Type-I 23–25%, CI coverage 74–77% at AR(1) φ = 0.5)
- **Location:** `src/tspec/_robust.py#L674-L688`, `#L845-L889`; `src/tspec/_diagnostics.py#L64-L204` (warning only).
- **Confidence:** Confirmed
- **Affected API:** `analyze_series(..., slope_estimator="sen")`: `segments[*].pvalue`, `slope_ci_*`.
- **Description:** Var(S) assumes independent observations. A residual-autocorrelation warning (lag-1 or OU ≥ 0.30) is emitted in 84–100% of AR(1) φ = 0.5 cases. No corrected test or interval is offered.
- **Evidence / Reproducer:** `audit/mc9_sen_mk.py` (R = 2000; 1000 at n = 500):

  | n | φ | MK Type-I @5% | Sen 95% coverage | autocorr warning rate |
  |---|---|---:|---:|---:|
  | 50 | 0 | 4.40% | 95.6% | 2% |
  | 150 | 0 | 4.45% | 95.6% | 0% |
  | 500 | 0 | 5.40% | 94.6% | 0% |
  | 50 | 0.5 | **25.5%** | **74.5%** | 84% |
  | 150 | 0.5 | **23.4%** | **76.6%** | 99.5% |
  | 500 | 0.5 | **24.8%** | **75.2%** | 100% |
- **Why expected is correct:** Positive autocorrelation inflates Var(S) by the factor n/n* (Hamed & Rao 1998; Yue & Wang 2004).
- **Scientific impact:** Inflated false trends and CIs that are too narrow when residuals are autocorrelated, as they commonly are for monthly environmental data. The warning mitigates this but does not fix it.
- **Recommended fix:** Implement the Hamed–Rao or Yue–Wang modified variance, or a block bootstrap, and switch to it automatically when the red-noise severity is ≥ moderate.
- **Suggested regression test:** AR(1) φ = 0.5, n = 150, R = 1000. Assert corrected Type-I ≤ 0.07.

### MAJ-02: `lssa().explain()` reports "significant spectral points" on pure noise in 89–100% of series
- **Location:** `src/tspec/spectral.py#L551-L576`, `#L527-L530`.
- **Confidence:** Confirmed
- **Affected API:** `LssaResult.explain()`, `.summary()`, `.significant_mask`.
- **Description:** Each frequency is correctly tested at α, but the narrative states "Detected k significant spectral point(s) above the 95% confidence threshold". No family-wise threshold or false-alarm probability for the peak is offered.
- **Evidence / Reproducer:** MC1, 238-frequency grid, α = 0.05: P(≥1 "significant") is 89.3% (n = 50), 99.7% (n = 150) and 100% (n = 500). The mean number of "significant" points is 11.7, matching 0.05 × 238 = 11.9. `audit/repro_reporting.py` R-A prints "Detected 3 significant spectral point(s)…" on white noise.
- **Why expected is correct:** As in BLK-02. The expected number of per-frequency exceedances under H₀ is αK.
- **Scientific impact:** Over-interpretation of noise peaks.
- **Recommended fix:** Report the global false-alarm probability of the highest peak (Baluev 2008, or a bootstrap) and reword the narrative.
- **Suggested regression test:** White noise, R = 500. Assert the narrative contains no "significant" claim unless the peak's global FAP is < α.

### MAJ-03: `robust="mm"` is not Yohai's MM-estimator, and `mode="l1"` does not have "50% vertical breakdown": both break down below 35% contamination
- **Location:** `src/tspec/_robust.py#L329-L349` (Huber start, MAD of Huber residuals, no S-estimator), `#L596-L607` (claim "50% vertical breakdown"); README; docs/reference/math.md §7; `fit_piecewise_trend` docstring.
- **Confidence:** Confirmed
- **Affected API:** `robust="mm"` (all functions), `fit_piecewise_trend(mode="l1")`, `slope_estimator="l1"`.
- **Description:** Yohai's (1987) MM-estimator requires a high-breakdown S-estimator start and scale (breakdown 0.5); the MM step then attains 95% efficiency. tspec starts from Huber IRLS instead, whose regression breakdown is design-dependent and low. LAD's breakdown for vertical outliers with an intercept-plus-slope design is ≈ 0.29 for uniform designs (He, Jurečková, Koenker & Portnoy 1990), not 0.5.
- **Evidence / Reproducer:** `audit/repro_breakdown.py`. n = 100 on a uniform grid, the last 35% of points shifted by +M, true slope 0.5:

  ```
       M |  L1 slope |   huber |      mm |  LTS(reference)
       5 |     1.173 |   1.178 |   1.178 |   0.421
     100 |    14.206 |  14.102 |  14.091 |   0.421
    1000 |   137.956 | 136.547 | 136.436 |   0.421
  ```

  For isolated random spikes (20%, M = 1000), `mm` (0.495) and `l1` (0.513) behave well; Huber gives 6.41.
- **Why expected is correct:** By definition, a 50%-breakdown estimator stays bounded for any contamination below 50% (Hampel 1971; Rousseeuw & Leroy 1987, ch. 1). The slope here grows linearly in M at 35%.
- **Scientific impact:** Users relying on the advertised high breakdown, e.g., for sustained sensor faults covering a third of a record, get arbitrarily wrong slopes.
- **Recommended fix:** Implement a proper S-estimator start (fast-S with random subsamples) for `mm`, or rename it "Huber-started biweight M-estimator". Remove the 50% claim for L1 hinge fits.
- **Suggested regression test:** The configuration above. Assert |slope − 0.5| < 0.2 for `mm` at M = 1000, or update the docs.

### MAJ-04: Defaults silently assume time is in decimal years
- **Location:** `src/tspec/spectral.py#L160-L171` (integer cycles/unit-time grid); `src/tspec/just.py#L326-L343` (`rate = int(n/span)`), `#L928` (`jump_th=0.75`); `src/tspec/stpd.py#L1400` (`tp_th=1.0`); `src/tspec/monitor.py#L702`, `#L940` (`monitor_omega` 0.1–3.5 and the 1.0 "cy/yr" abrupt/phenological split).
- **Confidence:** Confirmed
- **Affected API:** `allssa`, `just_*`, `stpd`, `just_monitor` with numeric `t` not in years.
- **Description / Evidence:** `audit/repro_time_inputs.py`. Ten years of data with t in days, containing a clear annual cycle:
  - `allssa(td, yd).frequencies` returns `[]` (the default grid is [1.0] cycle/day).
  - `just_jumps(td, …)` raises `InsufficientDataError: Window of length 3 is too short`.
  - The same data in years gives `[1., 12.]`.

  No warning states the unit assumption.
- **Why expected is correct:** Defaults that encode physical scales must either be unit-aware or validate the unit.
- **Scientific impact:** Silent "no periodicity" results, or confusing failures, for users with day- or second-based axes. Monitoring would classify every disturbance relative to a 1 cycle/day split.
- **Recommended fix:** Add a `time_unit` argument, or warn when numeric `t` has a span or sampling rate inconsistent with years. Scale default grids from the record span.
- **Suggested regression test:** Same data in days and years. Assert equivalent results after rescaling, or that a warning is emitted.

### MAJ-05: Validation tier and README overstate calibration evidence
- **Location:** `README.md` ("Verification & Performance"), `docs/reference/validation-summary.md`, `validation/bench_mc.py` (e.g., L66 `n_rep=40`, L99 gate `allssa_fp/n_rep <= 0.25`, L283 gate `fp/n_rep <= 0.08` at α = 0.01), `validation/reports/benchmarks_report.md`.
- **Confidence:** Confirmed
- **Affected API:** Documentation / claims.
- **Description:**
  - The "26/26 gates PASSED" result rests on R = 25–50 replicates (MC-SE 3–4 pp at 5%) and permissive thresholds: ALLSSA FPR ≤ 25% at α = 0.01; STPD FPR ≤ 8% at α = 0.01; JUST ≤ 15% at α = 0.05.
  - RB3 "Sen 95% CI coverage = 88.0%" and R1 "FPR 52% at φ = 0.6" are reported as PASS.
  - The README states "ALLSSA scan FPR = 7.5%"; this audit measures 18–25% (BLK-02).
  - Test count stated as 99 (actual: 119).
  - The decomposition band is described as "conservative" (BLK-03).
- **Why expected is correct:** Calibration claims need ≥ 1000 replicates and gates of the form |estimate − nominal| ≤ 2·MC-SE.
- **Scientific impact:** Users and reviewers may rely on "validated" claims that do not hold.
- **Recommended fix:** Regenerate the validation with ≥ 1000 replicates and nominal-centred gates. Report failures honestly.
- **Suggested regression test:** CI job running the MC1/MC2/MC4/MC5/MC10 designs of this audit at R ≥ 1000 (nightly).

---

## 6. Minor Issues

### MIN-01: Sen CI upper limit is one order statistic too low relative to Gilbert/R/scipy
- **Location:** `src/tspec/_robust.py#L879-L885`
- **Confidence:** Confirmed
- **Affected API:** Sen `slope_ci_high`.
- **Description:** The code uses the ⌈(N + C)/2⌉-th ordered slope. Gilbert (1987, §16.4) and `trend::sens.slope` use the (M₂ + 1)-th.
- **Evidence:** `audit/xval_references.py` gives 0.029730 vs 0.029787 (scipy `theilslopes` and the R convention agree with each other). Empirical coverage is still 94.6–95.6% (MC9).
- **Recommended fix:** `m2 = int(np.round((n_p + c_alpha) / 2 + 1))`.
- **Suggested regression test:** Compare with `scipy.stats.theilslopes(alpha=0.95)` bounds.

### MIN-02: Numeric strings and out-of-range datetimes are converted silently to wrong times
- **Location:** `src/tspec/_time.py#L131-L140`, `#L100-L104`
- **Confidence:** Confirmed
- **Affected API:** Any function taking `t`.
- **Description:** `np.array(["1","2","3"])` is parsed as datetime64 and overflows `[ns]` to the year **1754** (`audit/repro_time_inputs.py`). datetime64 values outside 1678–2262 wrap silently on `.astype("datetime64[ns]")`.
- **Recommended fix:** Try a float conversion first for string arrays. Check the representable range before casting to ns.
- **Suggested regression test:** String years and pre-1678 dates. Expect correct values or an error.

### MIN-03: Silent fallbacks and unsurfaced convergence flags
- **Location:** `_robust.py#L249-L269`, `#L316-L326` (`converged` never read by any caller); `censored.py#L832` (`except Exception` silently degrades `stability_mode="bootstrap"` to fixed-parameter draws); `pipeline.py#L930-L931` (per-segment STPD `InsufficientDataError` silently replaced by a no-TP fit); `_robust.py#L789-L795`, `#L846-L869` (ATS silently replaced by Theil–Sen on imputed values when n > 2000 or interval-censored; p/CI become `None` with no warning).
- **Confidence:** Confirmed by code inspection. IRLS non-convergence was rare in tests (≤ 0.2%, `audit/repro_irls_convergence.py`).
- **Affected API:** Robust fits, censored stability, pipeline.
- **Recommended fix:** Emit `WarningItem`s for each fallback and for non-convergence.

### MIN-04: The ill-conditioning warning threshold (cond > 1e10) is too lax
- **Location:** `src/tspec/spectral.py#L807-L814`
- **Confidence:** Confirmed
- **Affected API:** `lssa`, `allssa` coefficient outputs.
- **Description:** `audit/repro_collinear.py`: known frequencies 1.0 and 1.0 + 1e-9 over 4 years give cond 3.7e8 and coefficients of 3e6, with no warning. Fitted values are fine; individual amplitudes are meaningless.
- **Recommended fix:** Warn at cond > 1e6 and report coefficient-level instability.

### MIN-05: Docstring and documentation inconsistencies
- **Location:** `spectral.py#L876-L881` (omega default "1 .. floor(M/2)" and max_iter "max(20, …)", whereas the code uses ⌈M/2⌉ − 1 and 25); docs/reference/math.md §3 ("enforcing zero mean on each local seasonal component", which is true only in `parity_mode`); README/validation-summary "99 pytest" (actual 119); docs/guide/censored-data.md format example (6 points, 2 detects) raises `CensoringError`.
- **Confidence:** Confirmed
- **Recommended fix:** Synchronise the docs.

### MIN-06: Two different slopes are reported for the same segment in one `PipelineResult`
- **Location:** `pipeline.py#L549-L583` (separate discontinuous OLS per segment) vs `stpd.py#L1786-L1788` (continuous refit)
- **Confidence:** Confirmed
- **Description:** `audit/repro_reporting.py` R-E: TP `slope_before` = +0.4916 but `segments[0].slope` = +0.4946.
- **Recommended fix:** Derive segment slopes from the same continuous fit, or label the two quantities distinctly.

### MIN-07: Overstated wording and misleading method names
- **Location:** `stpd.py#L259` ("low NDRI confirms a smooth bend"); docs `DL_ARTIFACT_LIKELY` ("almost certainly a lab artifact": the check is a non-rejection at p ≥ 0.05 after information-losing recensoring, `just.py#L1258`); `monitor.py#L365-L380` ("Pocock" and "O'Brien–Fleming" are per-step Šidák/Bonferroni spending allocations, not group-sequential boundaries).
- **Confidence:** Confirmed
- **Recommended fix:** Soften the wording and rename the methods ("Šidák constant spending", "OBF-shaped Bonferroni spending").

### MIN-08: Tooling configuration
- **Location:** `pyproject.toml [tool.mypy] python_version="3.10"`; `validation/bench_perf.py#L72-L74`; `validation/compare_upstream.py#L29`
- **Confidence:** Confirmed
- **Description:** `mypy src/` aborts on numpy 2.5.3 stubs ("Type statement is only supported in Python 3.12"), so nothing is type-checked. With `--python-version 3.13` it reports 0 issues. Ruff reports 7 errors, all in `validation/` (6×B023, 1×F401).
- **Recommended fix:** Pin the stubs or bump the mypy target; fix the validation lint.

### MIN-09: ATS uses the uncensored Var(S), making it conservative and lowering power
- **Location:** `_robust.py#L819-L824`
- **Confidence:** Confirmed
- **Description:** ATS CI coverage is 98.0% at 50% censoring (MC10), against nominal 95%. Akritas et al. (1995) give a censoring-adjusted variance.
- **Recommended fix:** Use the Akritas–Murphy–LaValley asymptotic variance, or a bootstrap.

### MIN-10: `just_monitor` censoring guardrails use future data
- **Location:** `censored.py#L1099-L1117`, `#L1262-L1271` (called from `just_monitor`)
- **Confidence:** Confirmed by code inspection
- **Description:** The censored fraction (`max_censor_fraction` error) and DL-shift detection are computed on the whole series, including future monitoring observations. The imputation itself is causal (`_impute_causal_history_and_monitoring`).
- **Recommended fix:** Restrict guardrails to `[history:start]` plus the observations seen so far.

---

## 7. Info / Suggestions

### INF-01: The default monitor is very conservative
- **Location:** `monitor.py#L355-L399`
- **Confidence:** Confirmed
- **Description:** Family-wise rate is 0.1–0.3% at α = 0.05 (MC5). Power for a level drop is 4.5% (1σ), 50.5% (2σ) and 100% (4σ), with median delays of 21 / 17 / 8 steps (`audit/mc5c_monitor_power.py`, R = 200).
- **Suggestion:** Simulation-calibrated max-statistic thresholds would recover power at the same family-wise rate.

### INF-02: Weights are relative precisions
- **Description:** Covariances are scaled by the a-posteriori variance factor RSS/(n − p) (`_linalg.py#L303-L307`), so absolute inverse variances are not used as known. This is standard geodetic practice and is internally consistent, including in the monitor PI.
- **Suggestion:** Document it and offer `known_variance=True`.

### INF-03: Location convention for JUST jumps
- **Description:** The reported time is the first post-jump observation. MC3 shows a location bias of +0.02 to +0.09 yr, about half the sampling interval, and RMSE of 0.05–0.28 yr decreasing with jump size.
- **Suggestion:** Document the convention.

### INF-04: L1 non-uniqueness
- **Description:** LAD solutions can be non-unique; HiGHS returns a vertex. This was not observed as a practical problem.

### INF-05: Dependencies
- **Description:** Tested successfully with numpy 2.5.3, pandas 3.0.5, scipy 1.18.1 and ndimpute 0.1.1. There are no upper bounds on numpy, scipy or pandas. `ndimpute<1.0` is a pre-1.0 API, so pin it more tightly for archival reproducibility.

### INF-06: Lint noise
- **Description:** `RuntimeWarning: divide by zero` at `_robust.py#L255`/`#L273` (`k/u` evaluated inside `np.where` when u = 0) is harmless but noisy.
- **Suggestion:** Use `np.divide(..., where=u>0)`.

---

## 8. Monte Carlo Calibration Results

All runs use `OMP_NUM_THREADS=1` and fixed seeds (listed per script). Coverage MC-SE is computed from replicate-level means (between-replicate SD/√R). "Pass" uses the task's criteria.

| # | Study (script) | Settings | Reps | Seed base | Estimate ± MC-SE | Nominal | Result |
|---|---|---|---:|---:|---|---:|---|
| MC1 | LSSA pointwise FAR (`mc1_lssa.py`) | white noise, irregular (~24/yr), n=50/150/500, ±25% gap, 1 pre-specified freq | 2000 (1000 at n=500) | 1001 | α=0.05: 4.55, 4.35, 4.30, 4.65, 3.80, 5.30% (±0.49–0.69); α=0.01: 0.65, 0.95, 0.75, 0.55, 1.00, 1.60% (±0.22–0.31) | 5% / 1% | **Pass** (all within ~2 MC-SE; slightly conservative) |
| MC1b | LSSA grid "any significant" | 238 freqs, α=0.05 | as above | 1001 | 89.3–100% | (per-freq test) | Info → MAJ-02 |
| MC1c | ALLSSA spurious selection | defaults (α=0.01) | 600 | 1001 | 18.2–24.7% (±1.6–1.8) | 1% | **Fail** (BLK-02) |
| MC1d | LSSA `robust="huber"` pointwise | α=0.05 | 600 | 1001 | 2.2–4.2% (±0.9) | 5% | Pass (conservative) |
| MC2 | JUST false positives (`mc2_3_just_jumps.py`) | no jump, trend + 2 harmonics, σ=0.02 | 1000 (500 n=500; 400 allssa) | 2000 | defaults 98.5–100%; `jump_alpha=.05` 71–100%; +Bonferroni 3.0–16.4%; analyze-style 1.3–5.4% (±0.7–1.1) | ≤5% | **Fail** for defaults/`jump_alpha` (BLK-01); pass for analyze-style |
| MC3 | JUST power / location | 1σ/2σ/4σ step, n=50/150/500, gap | 500 (250 n=500; 200 allssa) | 2000 | analyze-style power: n=150 2.0/25.0/90.6%; n=500 2.4/26.4/93.6%; n=50 0.8/3.0/16.2%; gap 1.4/8.0/49.2%. `jump_alpha` only: 26–62/43–89/78–99%. Location bias +0.02…+0.09 yr, RMSE 0.05–0.28 yr | – | Reported; monotone ✓; low power with calibrated filter |
| MC4 | JUST trend/seasonal CI (`mc4_just_ci.py`) | linear (A), curved (B), jump (C); ols & allssa; n=50/150/500; gap | 1000 / 500 (ols), 400 / 200 (allssa) | 4000 | trend A: 94.6–95.6%; trend B: **66.0–91.7%** (n≥150: 66.0–76.8%); trend C: 94.9–95.7%; seasonal ols 94.8–95.6%; seasonal allssa **90.3–91.7%** (±0.3–0.8) | 95% | **Fail** (BLK-03, CRT-05) |
| MC4b | Curvature dose-response (`repro_just_ci_curvature.py`) | n=150/500, A∈{0,0.3,0.6}, P∈{6,9,15,25} yr | 200 | 4400 | 95.5% → 92.1% → 76.6% → 36.9% as curvature ↑ (±0.4–1.1) | 95% | **Fail** |
| MC5 | Monitor FWER (`mc5_6_monitor.py`) | stable; n_his=96; D=24/96; α=0.05; ols & allssa | 1000 | 5000 | Pocock 0.1–0.3%; OBF 0.1–0.3%; consecutive_k2 2.2–2.8% (D=24), **10.0–11.5%** (D=96); sidak+none 3.0–3.7% / 12.5–14.0%; parity 70.5–98.4% (±0.69) | ≤5% | Pass (Pocock/OBF); **Fail** (consecutive_k, CRT-01) |
| MC5b | Real-time re-run loop (`mc5b_monitor_realtime_loop.py`) | D=24, default horizon vs fixed | 400 | 5524 | 0.0% / 0.0% / 0.0% (single call / loop / loop fixed H) | ≤5% | Pass (see CRT-02 for reproducibility) |
| MC6 | Monitor PI coverage | as MC5 (Pocock run) | 1000 | 5000 | PI 94.9–95.0% (ols), 94.4–94.5% (allssa), flat across lead; mean-CI of μ 94.9% (ols), **92.0–92.9%** (allssa) (≈±0.2 approx.) | 95% | Pass (PI); mean CI with allssa marginal (CRT-05) |
| MC7 | STPD CIs (`mc7_stpd.py`) | 1 TP, n=150 σ=0.3/1.0; n=500 σ=0.6 | 1000 | 7000 | fixed knot: seq 95.3–95.6%, joint 94.5–95.2%, huber 94.4–95.3%; after detection: 91.7%, **31.8%**, **41.4%**; null FP @α=.01: 1.7%, 0.0%, 0.0% (±0.31); segment slope (correct detection): ols 94.3–95.7%, sen 94.6–95.4% | 95% / 1% | Pass (fixed knot); **Fail** (post-selection, CRT-04) |
| MC7b | Two-stage pipeline (`mc7b_pipeline_two_stage.py`) | seasonal + TP, n=288 | 200 | 7700 | trend band 83.5% (correct structure), 82.2% all; segment slopes 91.9% | 95% | **Fail** (CRT-04) |
| MC8 | Robust F-test Type-I (`mc8_robust_ftest.py`) | fixed alternative (q=2), N(0,1), t₃, 10%×10 contam, n=50/150/500 | 2000 | 8000 | huber 4.25–6.20%; mm 4.40–6.60%; OLS-F 4.30–6.00% at 5% (±0.49); at 1%: 0.45–1.5% (±0.22) | 5% / 1% | Pass (t₃ n=150 slightly liberal for all three tests: 5.95–6.6%) |
| MC9 | Sen CI & MK (`mc9_sen_mk.py`) | no trend / β=0.3; iid & AR(1) φ=0.5; n=50/150/500 | 2000 (1000 n=500) | 9000 | iid: Type-I 4.4–5.4%, coverage 94.6–95.6%; AR(1): Type-I **23.4–25.5%**, coverage **74.5–76.6%** (±0.49–0.69) | 5% / 95% | Pass (iid); **AR(1) inflation flagged** (MAJ-01) |
| MC10 | Censored pipeline (`mc10_censored.py`) | n=150, 20%/50% left-censoring; slope & 1.0 jump | 500 (300 jump) | 10000 | slope OLS: bias −7% / **−27%**, coverage 92.4% / **63.4%**; ATS: bias ≈0, coverage 95.2% / 98.0%; uncensored 94.2–94.6%; jump magnitude bias +0.059 / −0.039 (vs +0.018 / +0.002 uncensored); analyze-style jump detection 77% / 68% (vs 80% / 77%) | 95% | **Fail** for default OLS (BLK-04); ATS pass (conservative) |
| MC5c | Monitor power (`mc5c_monitor_power.py`) | level drop 1/2/4σ | 200 | 5700 | Pocock 4.5 / 50.5 / 100%; consecutive_k2 30 / 85 / 100%; OBF 3.5 / 51 / 100% | – | Info (INF-01) |

For the ALLSSA JUST gap configuration (n = 150, 25% gap), the null run (R = 400) completed: 99.3% / 78.3% / 4.8% / 1.3% for configurations A–D (see BLK-01). Power rows for this configuration (analyze-style / `jump_alpha` only): 4σ 36.0% / 86.5%; 1σ and 2σ rows are in `audit/mc2_3_allssa_out.txt`.

---

## 9. Independent Cross-Validation

| tspec quantity | Reference | Data | Max difference |
|---|---|---|---|
| WLS coefficients / SEs (`solve_weighted_ols`, `lssa(known_freqs, jump_indices, weights)`) | `statsmodels.WLS` | n=300, decimal years 2000–2015, jump + 2 harmonics, 1D weights | rel 7.9e-15 / 8.9e-15 |
| Cubic trend on raw decimal years | `statsmodels.OLS` (t − t₀ design) | n=300 | rel 3.1e-14 (cond 51 after scaling) |
| Normalised LSS (`trend="constant"`) | `astropy LombScargle(fit_mean=True, center_data=True, normalization="standard")` | n=120 irregular, 400 freqs | abs 1.4e-15 (unweighted and weighted by 1/dy²) |
| LSS (`trend="none"`) | astropy (`fit_mean=False, center_data=False`) | same | abs 1.6e-15 |
| Beta critical value 1 − α^{2/ν} | `scipy.stats.beta.ppf(1−α, 1, ν/2)` | (n, n_c, α) ∈ {(50,2,.05), (200,4,.01), (30,6,.001)} | ≤ 5.6e-17 |
| Huber IRLS (k = 1.345) | `statsmodels RLM(HuberT, cov="H1")` | n=200, t₃ noise | coef abs 1.8e-4; SE ratio 1.0025; scale 0.6301 vs 0.6269 (tspec freezes MAD after 3 iterations and applies √(n/(n−p))) |
| Theil–Sen slope | `pymannkendall.original_test`, `scipy.stats.theilslopes` | n=80 with ties | identical |
| MK p-value, Var(S) with ties | `pymannkendall` | same | p 4.7676e-4 both; Var(S) 57815 both |
| Sen CI | `scipy.theilslopes(alpha=.95)` / R `trend::sens.slope` rule | same | lower identical; upper 0.029730 vs 0.029787 (MIN-01) |
| Sen slope (property, 60 random cases) | `scipy.stats.theilslopes` | Hypothesis | identical (rtol 1e-12) |
| TPTR pointwise variance factors | brute-force smoother matrix H (ŷ = Hy) | n=60, 2 knots, ±weights | rel 4.0e-16 |
| Var(â₁ − â₂) in `_stpd_pvalue` | brute-force linear functional | n=60 | 1e-15 rel |
| Robust F-test, fixed-knot CIs, PIs | closed-form null via Monte Carlo | MC8, MC7, MC6 | see §8 |

---

## 10. Tooling Results

- **pytest** (`pytest -q -p no:cacheprovider`): **119 passed, 0 failed, 0 skipped**, 6 warnings, 29.2 s. The warnings are 4 ndimpute ROS-capping `UserWarning`s and 2 `RuntimeWarning: divide by zero` from `_robust.py:255` and `:273` (INF-06).
- **ruff** (`ruff check .`): 7 errors, all in `validation/` (6 × B023 in `bench_perf.py:72–74`, 1 × F401 in `compare_upstream.py:29`). `src/` and `tests/` are clean.
- **mypy** (`mypy src/`, configured `python_version=3.10`): aborts at `numpy/__init__.pyi:737` ("Type statement is only supported in Python 3.12 and greater"), so 0 files are checked. With `mypy --python-version 3.13 src/`: "Success: no issues found in 15 source files" (MIN-08).
- **Docs / README examples** (`audit/run_doc_examples.py`): 20 of 23 code blocks run as-is. The 3 failures are `NameError`s on intentional placeholders (README monitor fragment, README censored fragment, `guide/censored-data.md` input-formats fragment). With synthetic data (`audit/run_doc_fragments.py`), both README fragments and the `censor_status` fragment run. The 6-point toy example in `censored-data.md` raises `CensoringError` (only 2 detects) and cannot run as written.
- **Audit property tests** (`pytest audit/test_hypothesis_audit.py`): 6/6 pass (LSS ∈ [0, 1]; NaN-insertion equivalence; exact decomposition reconstruction and band ordering; STPD scale/shift equivariance; Sen = scipy; determinism and untouched global RNG).

---

## 11. Areas Reviewed and Found Clean

- **Weighted least squares / QR / SVD fallback** (`_linalg.py`): correct whitening for scalar, 1D and 2D SPD weights (Cholesky Lᵀ); shift-invariant centring; column scaling and unscaling of coefficients and covariance; residual orthogonality used correctly in the Schur-complement LSS (`g = Φᵀ r_w` without explicit projection is exact because r_w ⟂ col(Q)); cancellation fallback.
- **Beta critical value and p-values** (`spectral.py#L174-L187`): exact; effective n_c uses the rank under rank deficiency.
- **Amplitude/phase conversion**: A = hypot(c, s), φ = atan2(s, c), relative to t₀ = t[0].
- **Time adapter**: leap-year-exact decimal years (2020-07-02 → 2020.5); tz-aware Python datetimes and pandas tz-aware Series/DatetimeIndex converted to UTC; unsorted, duplicate, NaT and inf rejected; float32/int coerced to float64; pandas non-default index handled positionally; nullable `Float64` with `pd.NA` handled; inputs never mutated.
- **Robust F-test** (Schrader–Hettmansperger): Type-I calibrated (MC8). Reduces exactly to the classical F when all residuals are inliers.
- **Huber covariance** (Huber 1981, eq. 7.6.11 as cited): matches statsmodels H1 to 0.25%. Fixed-knot Huber hinge-spline CIs cover 94.4–95.3%.
- **TPTR sequential estimator and variance**: exact (§9). Forward/backward reversal maps slopes with the correct sign.
- **Censored-value parsing**: whitespace, `<`/`>` markers, custom markers, numeric/None/"NA" handled. Locale decimals ("0,05"), "≤", "ND" and booleans raise `CensoringError` (no silent coercion). Interval bounds validated (lo ≤ hi, not both infinite). Canonical signed status codes are never flipped.
- **ATS concordance rules** (`_censored_kendall_score`): definite-concordance and discordance conditions are correct for left/right censoring (Akritas et al. 1995). ATS is unbiased in MC10.
- **Causal imputation in `just_monitor`**: historical imputation uses only `[history:start]`, and each monitoring point is imputed against history alone.
- **Monitor PI formula**: Var(y_new − ŷ) = σ²(1/w + dᵀ(DᵀPD)⁻¹d) with t_{n_his − p}; calibrated (MC6). Predictive-covariance window weighting is algebraically correct (Cov(e_in, e_out) = 0).
- **Decomposition reconstruction**: trend + seasonal + remainder = y to ≤ 3.1e-15 (MC4, Hypothesis).
- **DL-shift detection**: a DL drop without a real change was detected in 100% of cases, with spurious jumps in 1% (`repro_censoring_mechanism.py`). A real jump coinciding with a DL drop was not mis-flagged as an artifact (0/14).
- **Plotting**: CI ribbons use arrays aligned with `times_raw`. Input order is never changed (sorted input is required), so no misalignment is possible.
- **Determinism**: default seeds give identical repeated results, and the global numpy RNG state is untouched.

---

## 12. Limitations of This Audit

- **Original papers were not retrieved.** Comparisons used the formulas as stated in the package's math docs and in standard texts (Pagiatakis 1999 / Ghaderpour & Pagiatakis 2017 Beta null; Huber 1981; Gilbert 1987; Akritas et al. 1995; Lan & DeMets 1983). Equation numbers are those cited by the package or by section/content; they were **not** verified against the PDFs. RMSE-NDRI thresholds were checked against the package's definitions only; I did not verify them against Ghaderpour et al. (2024).
- **Reduced replicates in places:**
  - ALLSSA-based JUST: R = 400 null / 200 power.
  - MC4 ALLSSA: 400/200.
  - MC5b: 400; MC5c: 200; MC7b: 200.
  - Curvature sweep: 200.
  - Censoring DL logic: 100–150.
  - Uncertainty statements were widened accordingly (MC-SE is reported).
- **Not covered:**
  - Full 2D SPD weight matrices in Monte Carlo (only the cross-check and unit paths).
  - `robust="mm"` calibration within JUST/STPD windows.
  - `scope="windowed"` and `"seasonal_stratified"` censoring.
  - Interval censoring.
  - `parity_mode` numerical parity: the upstream LSWAVE code was not fetched, so the 41/41 parity claim is **unverified** by this audit.
  - Performance claims.
  - Python 3.10–3.12 and older numpy/pandas versions.
- **Simulated designs:** All Monte Carlo designs are synthetic with Gaussian or specified heavy-tailed noise. Real-data behaviour, especially with seasonal heteroscedasticity, was not studied.
- **Power** was measured only where needed to interpret calibration (MC3, MC5c, MC7). No optimality claims are made.

---

## Appendix A: Reproduction Scripts Index

| Script | Purpose | Seed(s) |
|---|---|---|
| `audit/xval_references.py` | Cross-validation vs statsmodels, astropy, pymannkendall, scipy | 20261125 |
| `audit/repro_time_inputs.py` | Time adapter and input edge cases; years-only defaults | 0–2 |
| `audit/repro_tptr_variance.py` | Brute-force TPTR variance check | 7 |
| `audit/repro_monitor_lookahead.py` | CRT-02 horizon look-ahead | 31 |
| `audit/repro_reporting.py` | `explain()` overstatement (MAJ-02, BLK-02, BLK-01), Sen fan coverage (CRT-03), slope inconsistency (MIN-06) | 11–14, 200+, 10000+ |
| `audit/repro_breakdown.py` | MM/L1 breakdown (MAJ-03) | 21 |
| `audit/repro_just_ci_curvature.py` | BLK-03 dose-response | 4400 |
| `audit/repro_stpd_power.py` | STPD SNR gate (CRT-04) | 7500 |
| `audit/repro_censoring_mechanism.py` | Imputation ignores time (BLK-04); DL logic | 61–63 |
| `audit/repro_dl_recensor.py` | `recensor_check` on a real jump | 64 |
| `audit/repro_irls_convergence.py` | IRLS convergence rates (MIN-03) | 81 |
| `audit/repro_collinear.py` | Conditioning warning (MIN-04) | 91 |
| `audit/mc1_lssa.py` | MC1 | 1001+ |
| `audit/mc2_3_just_jumps.py` | MC2 / MC3 (`ols`, `allssa`) | 2000+ |
| `audit/mc4_just_ci.py` | MC4 | 4000+ |
| `audit/mc5_6_monitor.py` | MC5 / MC6 | 5000+ |
| `audit/mc5b_monitor_realtime_loop.py` | MC5b | 5500+D |
| `audit/mc5c_monitor_power.py` | MC5c | 5700+ |
| `audit/mc7_stpd.py` | MC7 | 7000+ |
| `audit/mc7b_pipeline_two_stage.py` | MC7b | 7700 |
| `audit/mc8_robust_ftest.py` | MC8 | 8000+ |
| `audit/mc9_sen_mk.py` | MC9 | 9000+ |
| `audit/mc10_censored.py` | MC10 | 10000+, DL pop. 424242 |
| `audit/test_hypothesis_audit.py` | Property-based fuzzing | Hypothesis |
| `audit/run_doc_examples.py`, `audit/run_doc_fragments.py` | Docs/README execution | 3 |
| `audit/*_out.txt`, `audit/*_results*.json` | Raw outputs of the above | – |

Typical invocation: `OMP_NUM_THREADS=1 python audit/<script>.py [args]` (arguments are documented in each script's docstring).

---

## Appendix B: References (with equation numbers used)

- Akritas, M. G., Murphy, S. A., & LaValley, M. P. (1995). The Theil–Sen estimator with doubly censored data and applications to astronomy. *JASA*, 90(429), 170–177. (Censored Kendall concordance rules; asymptotic variance of the censored S statistic, §3.)
- Andrews, D. W. K. (1993). Tests for parameter instability and structural change with unknown change point. *Econometrica*, 61(4), 821–856. (sup-F null distribution, used for BLK-01.)
- Balakrishnan, N., & Koutras, M. V. (2002). *Runs and Scans with Applications*. Wiley. (CRT-01.)
- Baluev, R. V. (2008). Assessing the statistical significance of periodogram peaks. *MNRAS*, 385, 1279–1285. (Global FAP of the maximum periodogram peak; BLK-02, MAJ-02.)
- Berk, R., Brown, L., Buja, A., Zhang, K., & Zhao, L. (2013). Valid post-selection inference. *Ann. Statist.*, 41(2), 802–837.
- Calonico, S., Cattaneo, M. D., & Farrell, M. H. (2018). On the effect of bias estimation on coverage accuracy in nonparametric inference. *JASA*, 113, 767–779.
- Davies, R. B. (1987). Hypothesis testing when a nuisance parameter is present only under the alternative. *Biometrika*, 74, 33–43.
- Fan, J., & Gijbels, I. (1996). *Local Polynomial Modelling and Its Applications*. Chapman & Hall. (Thm 3.1: local-linear bias ½h²μ₂g″; BLK-03.)
- Ghaderpour, E., & Pagiatakis, S. D. (2017). Least-squares wavelet analysis of unequally spaced and non-stationary time series and its applications. *Math. Geosci.*, 49(7), 819–844. (Normalised LSS and Beta(1, (n − q − 2)/2) null; per-frequency critical value 1 − α^{2/(n − q − 2)}.)
- Ghaderpour, E., Liao, W., & Lamoureux, M. P. (2018). Antileakage least-squares spectral analysis. *Geophysics*, 83(3), V157–V170. (ALLSSA iteration.)
- Ghaderpour, E., & Vujadinovic, T. (2020a, b). *Remote Sensing*, 12(23), 4001 and 12(15), 2446. (JUST, JUSTmonitor.)
- Ghaderpour, E. (2021). JUST: MATLAB and Python software for change detection and time series analysis. *GPS Solutions*, 25, 85.
- Ghaderpour, E., et al. (2024). A fast and robust method for detecting trend turning points in InSAR displacement time series. *Computers & Geosciences*, 185, 105546. (STPD/TPTR, SNR and NDRI gates.)
- Gilbert, R. O. (1987). *Statistical Methods for Environmental Pollution Monitoring*. Van Nostrand Reinhold. (Eq. 16.8, Var(S) with ties; §16.4 Sen CI ranks M₁ and M₂ + 1; MIN-01.)
- Hamed, K. H., & Rao, A. R. (1998). A modified Mann–Kendall trend test for autocorrelated data. *J. Hydrol.*, 204, 182–196. (MAJ-01.)
- Hampel, F. R. (1971). A general qualitative definition of robustness. *Ann. Math. Statist.*, 42, 1887–1896.
- He, X., Jurečková, J., Koenker, R., & Portnoy, S. (1990). Tail behavior of regression estimators and their breakdown points. *Econometrica*, 58, 1195–1214. (LAD breakdown ≈ 1 − 1/√2 for uniform designs; MAJ-03.)
- Helsel, D. R. (2012). *Statistics for Censored Environmental Data Using Minitab and R* (2nd ed.). Wiley. (ROS; censored regression for trends, ch. 12.)
- Hinkley, D. V. (1971). Inference in two-phase regression. *JASA*, 66, 736–743.
- Horne, J. H., & Baliunas, S. L. (1986). A prescription for period analysis of unevenly sampled time series. *ApJ*, 302, 757–763.
- Huber, P. J. (1981). *Robust Statistics*. Wiley. (Eq. 7.6.11 covariance with correction factor K, as cited by tspec.)
- Jennison, C., & Turnbull, B. W. (2000). *Group Sequential Methods with Applications to Clinical Trials*. Chapman & Hall. (§7.2, error-spending with pre-specified maximum information; CRT-02.)
- Lan, K. K. G., & DeMets, D. L. (1983). Discrete sequential boundaries for clinical trials. *Biometrika*, 70, 659–663. (OBF-type spending α*(u) = 2(1 − Φ(z_{α/2}/√u)).)
- Leeb, H., & Pötscher, B. M. (2005). Model selection and inference: facts and fiction. *Econometric Theory*, 21, 21–59.
- Meng, X.-L. (1994). Multiple-imputation inferences with uncongenial sources of input. *Statist. Sci.*, 9, 538–573. (BLK-04.)
- Muggeo, V. M. R. (2003). Estimating regression models with unknown break-points. *Statist. Med.*, 22, 3055–3071.
- Pagiatakis, S. D. (1999). Stochastic significance of peaks in the least-squares spectrum. *J. Geodesy*, 73, 67–78. (Beta null of the LSS.)
- Rousseeuw, P. J., & Leroy, A. M. (1987). *Robust Regression and Outlier Detection*. Wiley. (Breakdown definitions; LTS.)
- Rubin, D. B. (1987). *Multiple Imputation for Nonresponse in Surveys*. Wiley. (Total variance T = W̄ + (1 + 1/m)B.)
- Scargle, J. D. (1982). Studies in astronomical time series analysis II. *ApJ*, 263, 835–853.
- Schrader, R. M., & Hettmansperger, T. P. (1980). Robust analysis of variance based upon a likelihood ratio criterion. *Biometrika*, 67(1), 93–101.
- Sen, P. K. (1968). Estimates of the regression coefficient based on Kendall's tau. *JASA*, 63, 1379–1389.
- Yohai, V. J. (1987). High breakdown-point and high efficiency robust estimates for regression. *Ann. Statist.*, 15, 642–656. (MM requires an S-estimator start; MAJ-03.)
- Yue, S., & Wang, C. Y. (2004). The Mann–Kendall test modified by effective sample size. *Water Resour. Manag.*, 18, 201–218.
