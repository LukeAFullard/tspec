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

# 4. Tier 3: Unit, invariance, edge-case & Hypothesis property tests
pytest -v

# 5. Tier 4: Speed & O(n) memory profiling
python validation/bench_perf.py
```

Generated reports are written to `validation/reports/`.
