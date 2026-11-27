# Tier 2b: Literature-Inspired & Compositional Sub-Component Validation Report

- **tspec version:** `0.1.0`
- **Result:** **9 / 9 gates PASSED** (17.8 s single-threaded runtime)

| Gate | Benchmark / Scenario | Status | Measured Metrics | Runtime |
| :--- | :--- | :---: | :--- | ---: |
| `L1` | Ghaderpour (2018) non-integer anti-leakage | **PASS** | 3-freq recovery=12/12, null FP=0/12 | 1.5 s |
| `L2` | Verbesselt/Ghaderpour NDVI wildfire + regrowth | **PASS** | jumps=1, |dmag|=0.042, trend_cov=89%, seas_cov=94% | 1.2 s |
| `L3` | Ghaderpour (2024) PS-InSAR multi-TP + NDRI | **PASS** | TPs=2 ([2017.5, 2019.83]), max|dslope|=0.43, jump_flagged=2 | 0.1 s |
| `L4` | Helsel (2012) multi-DL water quality + artifact | **PASS** | ols_mi_MAE=0.005 vs dl/2_MAE=0.019, ats_ok=True, dl_artifact_guarded=True | 1.8 s |
| `L5` | Yue (2002) / Hamed-Rao AR(1) hydrology + spikes | **PASS** | MK FPR raw=8/30 vs ar1=1/30 | LSSA FPR unw=14/30 vs GLS=1/30 | 3.5 s |
| `L6` | Ghaderpour (2020b) NRT null/abrupt/phenological | **PASS** | null_k2_FP=0/20, abrupt=abrupt@2022.75, phen=phenological@2023.46 | 1.7 s |
| `SC1` | Kitchen-sink compositional recovery | **PASS** | jumps=[2017.56], TPs=[2020.99], last_slope=-1.36 | 5.2 s |
| `SC2` | Compound sub-component interaction matrix | **PASS** | bend+allssa=True, het_NDRI=True, quad_cov=100%, 2D_rob_mono=True | 2.3 s |
| `SC3` | Exact invariances & strict NRT causality | **PASS** | time_shift=True, weight_scale=True, affine_y=True, nrt_causal=True | 0.5 s |
