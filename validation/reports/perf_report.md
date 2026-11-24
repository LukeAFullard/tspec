# Tier 4 Performance & Memory Profiling Report

## Speed Comparison vs. Upstream Python Reference

| Benchmark | $n$ | `tspec` | Upstream Python | Speedup |
| :--- | ---: | ---: | ---: | ---: |
| `lssa` (100 freqs, 1D P) | 100 | 3.86 ms | 6.76 ms | **1.7x** |
| `lssa` (100 freqs, 1D P) | 1000 | 11.21 ms | 13.61 ms | **1.2x** |
| `lssa` (100 freqs, 1D P) | 5000 | 48.99 ms | 42.35 ms | **0.9x** |
| `allssa` (`decimal=4`) | 150 | 69.7 ms | 517.3 ms | **7.4x** |
| `just_jumps` (`ALLSSA`) | 150 | 911.3 ms | 1262.5 ms | **1.4x** |
| `stpd` (single series) | 100 | 4.83 ms | 15.39 ms | **3.2x** |

## Batch Throughput & O(n) Memory Gates

- **STPD 1,000-series batch screening ($n=100$):** `4.83s` (gate `< 10.0s`) — **PASS**
- **`allssa(decimal=4)` speedup:** `7.4x` (gate `>= 5.0x`) — **PASS**
- **1D diagonal weights peak memory ($n=1{,}000$):** `1.95 MB`
- **1D diagonal weights peak memory ($n=10{,}000$):** `19.40 MB` (gate `< 50.0 MB`, verifying $O(n)$ vs $800\text{ MB}$ dense $O(n^2)$) — **PASS**
