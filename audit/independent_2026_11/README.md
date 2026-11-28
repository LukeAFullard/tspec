# Independent adversarial audit probes (2026-11)

Supporting scripts and raw outputs for [`../INDEPENDENT_AUDIT_2026-11.md`](../INDEPENDENT_AUDIT_2026-11.md).
All scripts are standalone and run from this directory with `OMP_NUM_THREADS=1 MPLBACKEND=Agg python <script>`
(Monte Carlo replicates are spread across 4 worker processes, see `common.py`).

| Script | Output | Topic |
|---|---|---|
| `repro_confirmed_bugs.py` | `repro_out.txt` | < 1 min minimal reproducers (AUD-01, -02, -05, -06, -09, -10) |
| `p1_spectral.py`, `p1b_mcfap.py` | `p1_results.json`, `p1b_out.txt` | LSSA/ALLSSA oracles & calibration |
| `p2_just.py [NREP]`, `p2b_power_baseline.py` | `p2_out.txt`, `p2b_out.txt` | JUST FWER, power, decomposition CI coverage |
| `p3_datetime_leap.py` | `p3_results.json` | datetime decimal-year non-uniformity |
| `p4_stpd.py [NREP]` | `p4_out.txt` | STPD FPR, NDRI, bands |
| `p5_monitor.py [NREP]` | `p5_out.txt` | monitoring causality & sequential FWER |
| `p6_trend_censor.py`, `p6b_censor.py` | `p6_out.txt`, `p6b_out.txt` | MK/Sen/OLS Type-I under AR(1); censored slopes |
| `p7_edge.py` | `p7_results.json` | adversarial inputs, reproducibility, memory |
| `p8_imports.sh` | `p8_out.txt` | minimal NumPy+SciPy environment |
| `p9_pipeline_combos.py`, `p10_pipeline_step.py`, `p11_readme_hero.py` | `p9_*`, `p10_*`, `p11_*` | `analyze_series` composition |
| `p12_followups.py` | `p12_results.json` | follow-ups from the STL benchmark |
| `p13_batch1_stpd.py` | `p13_final_out.txt` | batch-1 validation: STPD step-vs-bend gate vs NDRI |
| `p10_pipeline_step.py` (re-run) | `p10_post_out.txt` | batch-1 validation through `analyze_series` |
| `repro_confirmed_bugs.py` (re-run) | `repro_post_batch1_out.txt` | reproducers after batch 1 |
| `p14_batch2_serial.py [NREP]` | `p14_final_ar1_out.txt` (final), `p14_none_out.txt` (independent-error comparison), `p14_pre_jumpfix_out.txt`, `p14_margin0_partial_out.txt` (earlier variants) | batch-2 validation of serial-correlation calibration (AUD-02) |
| `p11_readme_hero.py` (re-run) | `p11_post_batch2_results.json` | README example after batch 2 |
| `p15_batch3_scales.py [NREP]` | `p15_out.txt` | batch-3 validation: multi-scale JUST, curvature-robust step test, ghosts (AUD-04/17/18) |
| `p10_pipeline_step.py`, `p11_readme_hero.py` (re-runs) | `p10_batch3_results.json`, `p11_batch3_results.json` | pipeline step-vs-TP and README example after batch 3 |
