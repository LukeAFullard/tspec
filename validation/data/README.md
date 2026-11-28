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
4. **`nasa_gistemp_global_annual.csv`** ($n = 146$, 1880–2025)
   - Annual Land-Ocean Global Mean Surface Temperature Anomaly (°C, relative to 1951–1980).
   - Source: NASA Goddard Institute for Space Studies (`https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv`, GISTEMP Team / Lenssen et al., 2019). US Government public domain.
5. **`us_macrodata_quarterly.csv`** ($n = 203$, 1959Q1–2009Q3)
   - US quarterly macroeconomic indicators (`realgdp`, `cpi`, `infl`, `unemp`).
   - Classic econometrics benchmark capturing the 1972 Great Inflation onset, 1981 Volcker disinflation turning point, and 1991 Great Moderation. Public domain (US BEA / BLS / `statsmodels.datasets.macrodata`).
6. **`elnino_sst_monthly.csv`** ($n = 732$, Jan 1950 – Dec 2010)
   - Monthly Pacific Niño 1+2 Sea Surface Temperature (°C).
   - Classic oceanography / climate oscillation benchmark combining annual seasonality with the 3–6 year irregular ENSO cycle. Public domain (NOAA CPC / `statsmodels.datasets.elnino`).
