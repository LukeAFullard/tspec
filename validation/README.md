# tspec 4-Tier Validation Suite

This directory holds the automated validation and verification scripts for `tspec`.
It is **excluded** from built wheels and source distributions (`pyproject.toml`).

## Clean-Room Isolation Policy

`validation/upstream_ref/` is strictly `.gitignore`d. No source code or example
datasets from `Ghaderpour/LSWAVE-SignalProcessing` are ever committed to this
repository or bundled in release archives.

## Running Validation Locally

```bash
# 1. Fetch pinned upstream oracle (commit bd15923b) into gitignored validation/upstream_ref/
python validation/fetch_upstream.py

# 2. Tier 1: Numerical parity against upstream Ghaderpour/LSWAVE-SignalProcessing
python validation/compare_upstream.py

# 3. Tier 2: Synthetic ground-truth & Monte Carlo benchmarks (JUST, STPD, Censored, Red Noise, Pipeline)
python validation/bench_mc.py

# 3b. Tier 2b: Post-hardening literature-inspired (L1-L6) & compositional sub-component (SC1-SC3) benchmarks
OMP_NUM_THREADS=1 MPLBACKEND=Agg python3 validation/bench_literature_and_components.py

# 3c. Tier 2c: Real-world public-domain scientific & economic datasets (Nile, Mauna Loa CO2, Sunspots, NASA GISTEMP, US Macro CPI, El Nino)
OMP_NUM_THREADS=1 MPLBACKEND=Agg python3 validation/bench_public_datasets.py

# 3d. Tier 2d: External library oracles (statsmodels/scipy), cross-domain signals & adversarial edge cases
OMP_NUM_THREADS=1 MPLBACKEND=Agg python3 validation/bench_cross_domain_and_edge_cases.py

# 3e. Tier 2e: Independent known-solution & real-world head-to-head vs STL / MSTL / ruptures (with figures)
#     (BENCH_NREP=60 default 40; figures -> reports/figures_stl/, report -> reports/stl_headtohead_report.md)
OMP_NUM_THREADS=1 MPLBACKEND=Agg python3 validation/bench_stl_headtohead.py

# 4. Tier 3: Unit, invariance, edge-case & Hypothesis property tests
pytest -v

# 5. Tier 4: Speed & O(n) memory profiling
python validation/bench_perf.py
```

Generated reports are written to `validation/reports/`.
