# Tier 2c: Real-World Public Dataset Validation Report

- **tspec version:** `0.1.0`
- **Result:** **3 / 3 gates PASSED** (0.4 s single-threaded runtime)

| Gate | Public Dataset & Literature Reference | Status | Measured Metrics | Runtime |
| :--- | :--- | :---: | :--- | ---: |
| `RW1` | Nile River at Aswan (1871-1970, Cobb 1978) | **PASS** | JUST jump=1899 (Δ=-283.9, p=0.0009); NRT alarm=abrupt@1902 | 0.1 s |
| `RW2` | NOAA Mauna Loa CO2 (1958-2026, 20% gapped) | **PASS** | ALLSSA=[1.0, 2.0], RMSE vs NOAA=0.327 ppm, growth 0.72→3.41 ppm/yr (6 TPs) | 0.3 s |
| `RW3` | Sunspots (1700-2008, 30% + 10-yr blackout) | **PASS** | n_kept=204/309, peak=0.0910 cy/yr (period=10.99 yr), peak_fap=1.97e-11 | 0.0 s |
