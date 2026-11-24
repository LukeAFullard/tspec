# Tier 2 Synthetic Ground-Truth & Monte Carlo Benchmarks Report

- **Summary:** **17 / 17 benchmark gates PASSED** in 66.9s

| ID | Benchmark Scenario | Status | Measured Metrics |
| :--- | :--- | :---: | :--- |
| **J1** | LSSA/ALLSSA/JUST Null Calibration (0 jumps, white noise) | **PASS** | LSSA single-freq FPR=5.0% (alpha=0.05); ALLSSA scan FPR=7.5% (default alpha=0.01); JUST Bonferroni FPR=2.5% (alpha=0.05) |
| **J2** | JUST Single-Jump Recovery (SNR=3 vs SNR=6) | **PASS** | SNR=3 recall=77.5% (mag RMSE=0.081); SNR=6 recall=97.5% (mag RMSE=0.066) |
| **J3** | Non-Integer Multi-Harmonic Seasonality (ALLSSA vs Fixed OLS) | **PASS** | 3-freq recovery=100.0%; slope RMSE ALLSSA=0.0018 vs OLS=0.0211 |
| **J4** | Severe Irregularity (45% dropout + 4-mo blackout, zero interp) | **PASS** | Zero crashes; jump recall=100.0% |
| **J5** | Heteroscedastic Weighting (Weighted P vs Unweighted OLS) | **PASS** | Weighted mag RMSE=0.0314 (recall=100%) vs Unweighted RMSE=0.1131 (recall=75%) |
| **S1** | STPD Continuous TP Recovery (White & Heavy-Tail t5) | **PASS** | White recall=100.0% (timing MAE=0.085 yr); t5 recall=98.0% |
| **S2** | STPD Null Constant-Slope Calibration (0 TPs) | **PASS** | Empirical FPR=8.0% at alpha=0.01, SNR>1, NDRI<0.3 |
| **S3** | STPD NDRI Jump vs. Continuous TP Discrimination | **PASS** | Median NDRI TP=0.083 vs Jump=0.716; ROC AUC=1.000; flagged jumps=50/50 |
| **S4** | Density-Imbalance NDRI Stress Test (3:1 sampling ratio) | **PASS** | False-rejection rate: ndri_mode='l2'=42.0% vs ndri_mode='rmse'=0.0% |
| **S5** | Head-to-Head JUST vs. STPD on Noisy Continuous TPs | **PASS** | Timing RMSE STPD=0.210 yr vs JUST=0.426 yr (JUST median spurious vertical step=1.38) |
| **C1** | Censored ROS Quantile Temporal-Trend-Injection Guard | **PASS** | Unshuffled raw slope=+0.0415 (biased!) vs tspec tie-shuffled=-0.0065 ± 0.0184 |
| **C2** | 30% Left-Censored Jump Magnitude Error (ndimpute ROS vs DL/2) | **PASS** | MAE vs uncensored oracle: ndimpute ROS=0.1000 vs DL/2 substitution=0.2755 |
| **C3** | Shifting Detection-Limit Null Artifact Defense (<2.0 -> <0.2) | **PASS** | DL shift detected=30/30 (100%); coinciding fake jumps flagged=17/17 |
| **C4** | Non-Stationary Large Jump (scope='global' vs scope='windowed') | **PASS** | Mag MAE vs oracle: global=1.415 vs windowed=1.242 |
| **C5** | Imputed Down-Weighting Trade-Off Sweep (30% censored) | **PASS** | Mag MAE vs oracle: γ=1.0:0.606, γ=0.5:0.359, γ=0.25:0.206 |
| **R1** | AR(1) Red-Noise Sweep & P=C_f^{-1} Calibration Restoration (alpha=0.05) | **PASS** | Unweighted FPR: φ=0.0:8%, φ=0.3:22%, φ=0.6:52%, φ=0.8:40% | Weighted P=C_f^{-1} FPR: φ=0.0:8%, φ=0.3:10%, φ=0.6:5%, φ=0.8:5% |
| **P1** | End-to-End Pipeline (seasonality + jump + TP + gaps + 20% censored) | **PASS** | Jump recovery=100.0%; Turning-point recovery=100.0% |
