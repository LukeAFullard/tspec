# Tier 4 Performance & Memory Profiling Report

## Speed Comparison vs. Upstream Python Reference

| Benchmark | $n$ | `tspec` | Upstream Python | Speedup |
| :--- | ---: | ---: | ---: | ---: |
| `lssa` (100 freqs, 1D P) | 100 | 5.03 ms | 6.69 ms | **1.3x** |
| `lssa` (100 freqs, 1D P) | 1000 | 11.49 ms | 13.96 ms | **1.2x** |
| `lssa` (100 freqs, 1D P) | 5000 | 56.68 ms | 47.17 ms | **0.8x** |
| `allssa` (`decimal=4`) | 150 | 77.0 ms | 493.8 ms | **6.4x** |
| `just_jumps` (`ALLSSA`) | 150 | 786.1 ms | 1282.6 ms | **1.6x** |
| `stpd` (single series) | 100 | 4.39 ms | 19.34 ms | **4.4x** |

## Batch Throughput & O(n) Memory Gates

- **STPD 1,000-series batch screening ($n=100$):** `4.39s` (gate `< 10.0s`) — **PASS**
- **`allssa(decimal=4)` speedup:** `6.4x` (gate `>= 5.0x`) — **PASS**
- **1D diagonal weights peak memory ($n=1{,}000$):** `1.95 MB`
- **1D diagonal weights peak memory ($n=10{,}000$):** `19.40 MB` (gate `< 50.0 MB`, verifying $O(n)$ vs $800\text{ MB}$ dense $O(n^2)$) — **PASS**
