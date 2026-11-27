# Public-Domain Real-World Benchmark Datasets (`validation/data/`)

All datasets in this directory are classic public-domain scientific records used strictly for Tier 2c real-world validation (`validation/bench_public_datasets.py`). As part of `validation/`, they are excluded from built wheels and source distributions (`pyproject.toml`).

1. **`nile_aswan_1871_1970.csv`** ($n = 100$, 1871–1970)
   - Annual flow volume ($10^8\ \text{m}^3$) of the River Nile at Aswan.
   - Classic change-point benchmark with a known abrupt regime drop in **1898–1899** (Cobb, 1978, *Biometrika* 65:243–251; Durbin & Koopman, 2001). Public domain.
2. **`mauna_loa_co2_monthly.csv`** ($n = 824$, March 1958 – October 2026)
   - Monthly mean atmospheric $\text{CO}_2$ (ppm) at Mauna Loa Observatory, Hawaii.
   - Source: NOAA Global Monitoring Laboratory (`https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv`; Lan, Tans, & Thoning, NOAA GML). US Government public domain.
3. **`sunspots_annual_1700_2008.csv`** ($n = 309$, 1700–2008)
   - Yearly mean sunspot activity numbers.
   - Classic geophysical spectral benchmark for the ~11-year Schwabe solar cycle (`f ≈ 0.091` cy/yr). Public domain (Wolf / SILSO / `statsmodels.datasets.sunspots`).
