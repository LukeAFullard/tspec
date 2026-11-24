# Tier 4 Performance & Memory Profiling Report

## Speed Comparison vs. Upstream Python Reference

| Benchmark | $n$ | `tspec` | Upstream Python | Speedup |
| :--- | ---: | ---: | ---: | ---: |
| `lssa` (100 freqs, 1D P) | 100 | 3.70 ms | 5.86 ms | **1.6x** |
| `lssa` (100 freqs, 1D P) | 1000 | 10.72 ms | 12.48 ms | **1.2x** |
| `lssa` (100 freqs, 1D P) | 5000 | 48.73 ms | 39.87 ms | **0.8x** |
| `allssa` (`decimal=4`) | 150 | 68.6 ms | 437.1 ms | **6.4x** |
| `just_jumps` (`ALLSSA`) | 150 | 836.6 ms | 1109.6 ms | **1.3x** |
| `stpd` (single series) | 100 | 4.74 ms | 16.21 ms | **3.4x** |

## Batch Throughput & O(n) Memory Gates

- **STPD 1,000-series batch screening ($n=100$):** `4.74s` (gate `< 10.0s`) — **PASS**
- **`allssa(decimal=4)` speedup:** `6.4x` (gate `>= 5.0x`) — **PASS**
- **1D diagonal weights peak memory ($n=1{,}000$):** `1.95 MB`
- **1D diagonal weights peak memory ($n=10{,}000$):** `19.40 MB` (gate `< 50.0 MB`, verifying $O(n)$ vs $800\text{ MB}$ dense $O(n^2)$) — **PASS**
