# Tier 2d: External-Oracle, Cross-Domain & Adversarial Edge-Case Validation Report

- **tspec version:** `0.1.0`
- **Result:** **15 / 15 gates PASSED** (6.6 s single-threaded runtime)

| Gate | Category | Validation Scenario | Status | Measured Metrics | Runtime |
| :--- | :--- | :--- | :---: | :--- | ---: |
| `OR1` | Oracle | statsmodels.GLS 2D AR(1) exact parity | **PASS** | max|Δβ|=8.88e-16, max|ΔCov|=5.55e-17, max|Δresid|=2.89e-15 | 0.7 s |
| `OR2` | Oracle | scipy.stats.theilslopes exact parity | **PASS** | |Δslope|=0.00e+00, |ΔCI_lo|=0.00e+00, |ΔCI_hi|=0.00e+00 | 0.0 s |
| `OR3` | Oracle | scipy.signal.lombscargle + Beta null KS | **PASS** | f_tspec=1.37 vs f_scipy=1.37 (true 1.37), Beta-null KS p=0.588 | 0.3 s |
| `D1` | Epidemiol. | Seasonal incidence + vaccine shock + decay | **PASS** | rollout=2020.25 (Δ=-3.73), post_slope=-0.42/yr | 0.5 s |
| `D2` | Economics | Bai-Perron break + bend + 3x vol shift | **PASS** | jump=[2004.1] (true 2004), TP=[2014.3] (true 2015, weighted NDRI) | 0.3 s |
| `D3` | Engineer. | Bearing RUL onset + 8% 25-sigma spikes | **PASS** | RUL onset TP=6.48 (true 6.50), NRT alarm@6.75 | 2.8 s |
| `D4` | Pharmacol. | 40% <LLOQ lognormal clearance + assay shift | **PASS** | cens=37%, ols_mi=-0.196 (pop=-0.197), ats=-0.205 | 0.2 s |
| `E1` | Edge Case | Extreme dynamic range (1e-9 to 1e+9) | **PASS** | indices=[60, 60, 60], scaled_mags=[2.8274, 2.8274, 2.8274] | 0.3 s |
| `E2` | Edge Case | Catastrophic cancellation (t+1e6, y+1e8) | **PASS** | idx0=60, idx_shifted=60, |Δdir|=3.86e-09 | 0.1 s |
| `E3` | Edge Case | Noiseless & flat constant signals | **PASS** | const(j=0,tp=0), exact_step_mag=3.0000, exact_tp_dir=-1.5000 | 0.2 s |
| `E4` | Edge Case | Unsorted/duplicate times + NaNs + 70% drop | **PASS** | guards(unsorted=True,dup=True), n_valid=138, dropped_nan=8, jump=2020.01 | 0.2 s |
| `E5` | Edge Case | Small-n (n=16) graceful degradation | **PASS** | default_guard=True, n=16 finite, warnings=['STPD_SKIPPED'] | 0.0 s |
| `E6` | Edge Case | Mixed <DL/>UL + [L,R] interval + >80% guard | **PASS** | mixed_slope=0.344, interval_slope=0.331, guardrail=True | 0.1 s |
| `E7` | Edge Case | Heavy-tailed t2 + 15% extreme spikes | **PASS** | max|Δslope| OLS=0.361 vs L1=0.017, Huber=0.031 | 0.0 s |
| `E8` | Edge Case | Closely spaced double breakpoints | **PASS** | 2-jumps=[2017.51, 2020.53] (true 2017.5,2020.5); jump+TP=(2018.02,2020.57) | 1.0 s |
