# Tier 1 Numerical Parity & Divergence Report

- **Upstream Oracle:** `Ghaderpour/LSWAVE-SignalProcessing` @ `bd15923b1d4880b799b14fe016de4c1252259013`
- **Summary:** **41 / 41 checks PASSED** (0 failures)
- **Tolerances:** `rtol=1e-6, atol=1e-8` on continuous arrays; exact `==` on discrete indices and frequencies.

## Parity Verification Matrix

| Dataset | Method | Verified Metrics | Status | Error / Output Details |
| :--- | :--- | :--- | :---: | :--- |
| `Sim_Geophysics.dat` | `lssa` | spectrum | **PASS** | max_abs=8.19e-16, max_rel=3.30e-14 |
| `Sim_Geophysics.dat` | `lssa` | coeff & crit | **PASS** | max_abs=5.88e-15 |
| `Sim_Geophysics.dat` | `allssa(decimal=2)` | frequencies & residuals | **PASS** | freqs=[4.07, 20.37, 22.29] vs [np.float64(4.07), np.float64(20.37), np.float64(22.29)], res_abs=5.95e-14 |
| `Sim_Geophysics.dat` | `allssa(decimal=4)` | frequencies & residuals | **PASS** | freqs=[4.0744, 20.3718, 22.2818] vs [np.float64(4.0744), np.float64(20.3718), np.float64(22.2818)], res_abs=4.42e-14 |
| `EVIA_CA.dat` | `just_jumps(parity)` | indices, mag, dir | **PASS** | idx=[13, 111, 144] vs [13, 111, 144]; mag=[-0.06, 0.03, -0.06] vs [-0.06, 0.03, -0.06] |
| `EVIA_CA.dat` | `just_decompose(parity)` | trend/seasonal/rem | **PASS** | max_abs(T=4.71e-14, S=1.36e-15, R=4.68e-14) |
| `EVIB_CA.dat` | `just_jumps(parity)` | indices, mag, dir | **PASS** | idx=[22, 47, 141] vs [22, 47, 141]; mag=[-0.04, 0.03, -0.06] vs [-0.04, 0.03, -0.06] |
| `EVIB_CA.dat` | `just_decompose(parity)` | trend/seasonal/rem | **PASS** | max_abs(T=7.44e-15, S=2.16e-15, R=7.49e-15) |
| `EVIC_CA.dat` | `just_jumps(parity)` | indices, mag, dir | **PASS** | idx=[29, 62, 117, 169] vs [29, 62, 117, 169]; mag=[0.3, -0.14, 0.08, 0.07] vs [0.3, -0.14, 0.08, 0.07] |
| `EVIC_CA.dat` | `just_decompose(parity)` | trend/seasonal/rem | **PASS** | max_abs(T=1.41e-14, S=3.33e-15, R=1.42e-14) |
| `ExA_NDVI_Australia.dat` | `just_jumps(parity)` | indices, mag, dir | **PASS** | idx=[39, 104, 143, 182] vs [39, 104, 143, 182]; mag=[-0.05, -0.25, 0.04, 0.13] vs [-0.05, -0.25, 0.04, 0.13] |
| `ExA_NDVI_Australia.dat` | `just_decompose(parity)` | trend/seasonal/rem | **PASS** | max_abs(T=4.47e-13, S=1.28e-14, R=4.41e-13) |
| `ExB_NDVI_Australia.dat` | `just_jumps(parity)` | indices, mag, dir | **PASS** | idx=[28, 64, 95, 122, 148] vs [28, 64, 95, 122, 148]; mag=[0.03, 0.12, -0.03, 0.05, -0.08] vs [0.03, 0.12, -0.03, 0.05, -0.08] |
| `ExB_NDVI_Australia.dat` | `just_decompose(parity)` | trend/seasonal/rem | **PASS** | max_abs(T=7.54e-14, S=1.79e-14, R=7.57e-14) |
| `EVIA_AB.dat` | `just_jumps(parity)` | indices, mag, dir | **PASS** | idx=[14, 27, 46, 63] vs [14, 27, 46, 63]; mag=[0.27, -0.17, -0.18, -0.03] vs [0.27, -0.17, -0.18, -0.03] |
| `EVIA_AB.dat` | `just_decompose(parity)` | trend/seasonal/rem | **PASS** | max_abs(T=1.43e-13, S=8.75e-14, R=2.14e-13) |
| `EVIB_AB.dat` | `just_jumps(parity)` | indices, mag, dir | **PASS** | idx=[24, 61, 72] vs [24, 61, 72]; mag=[-0.1, 0.04, -0.06] vs [-0.1, 0.04, -0.06] |
| `EVIB_AB.dat` | `just_decompose(parity)` | trend/seasonal/rem | **PASS** | max_abs(T=2.69e-13, S=1.14e-14, R=2.62e-13) |
| `EVIC_AB.dat` | `just_jumps(parity)` | indices, mag, dir | **PASS** | idx=[22, 45] vs [22, 45]; mag=[0.19, -0.19] vs [0.19, -0.19] |
| `EVIC_AB.dat` | `just_decompose(parity)` | trend/seasonal/rem | **PASS** | max_abs(T=3.85e-14, S=1.40e-14, R=3.58e-14) |
| `Sim_JumpInd_48_104.dat` | `just_jumps(parity)` | indices, mag, dir | **PASS** | idx=[47, 103] vs [47, 103]; mag=[0.26, -0.12] vs [0.26, -0.12] |
| `Sim_JumpInd_48_104.dat` | `just_decompose(parity)` | trend/seasonal/rem | **PASS** | max_abs(T=6.75e-14, S=2.41e-15, R=6.93e-14) |
| `Sim_JumpInd_101_321.dat` | `just_jumps(parity)` | indices, mag, dir | **PASS** | idx=[242] vs [242]; mag=[-7.96] vs [-7.96] |
| `Sim_JumpInd_101_321.dat` | `just_decompose(parity)` | trend/seasonal/rem | **PASS** | max_abs(T=3.14e-13, S=3.42e-14, R=3.45e-13) |
| `EVIA_CA.dat` | `just_decompose(tt, parity)` | trend_reg & seasonal_reg | **PASS** | max_abs(T_reg=2.34e-14, S_reg=1.39e-15) |
| `DESC-PSInSAR-MonthlyResampled.csv` | `stpd + tptr (parity)` | TPs, stats, fitted y | **PASS** | TPs=[32, 55] vs [32, 55]; y_abs=8.88e-15 |
| `SimulatedExample.csv` | `stpd + tptr (parity)` | TPs, stats, fitted y | **PASS** | TPs=[33] vs [33]; y_abs=2.66e-15 |
| `Synthetic (scalar P=1)` | `lssa` | spec, coeff, cov | **PASS** | spec_abs=1.67e-15, cov_abs=3.12e-17 |
| `Synthetic (scalar P=1)` | `just_jumps(OLS, parity)` | jump indices | **PASS** | idx=[54, 69, 88] vs [54, 69, 88] |
| `Synthetic (1D vector P)` | `lssa` | spec, coeff, cov | **PASS** | spec_abs=6.66e-16, cov_abs=4.16e-17 |
| `Synthetic (1D vector P)` | `just_jumps(OLS, parity)` | jump indices | **PASS** | idx=[6, 70, 88] vs [6, 70, 88] |
| `Synthetic (2D SPD P)` | `lssa` | spec, coeff, cov | **PASS** | spec_abs=1.55e-15, cov_abs=1.63e-16 |
| `Synthetic (2D SPD P)` | `just_jumps(OLS, parity)` | jump indices | **PASS** | idx=[6, 33, 70, 88] vs [6, 33, 70, 88] |
| `EVIA_AB.dat` | `just_monitor(ALLSSA, parity)` | Type, Loc, Mag, Dir | **PASS** | Type=1 vs 1, Loc=2019.3900, Mag=-0.4886 vs -0.4886, Dir=1.0837 vs 1.0837 |
| `EVIA_AB.dat` | `just_monitor(OLS, parity)` | Type, Loc, Mag, Dir | **PASS** | Type=1 vs 1, Loc=2019.3900, Mag=-0.4872 vs -0.4872, Dir=1.1107 vs 1.1107 |
| `EVIB_AB.dat` | `just_monitor(ALLSSA, parity)` | Type, Loc, Mag, Dir | **PASS** | Type=1 vs 1, Loc=2019.6340, Mag=0.0000 vs 0.0000, Dir=0.0000 vs 0.0000 |
| `EVIB_AB.dat` | `just_monitor(OLS, parity)` | Type, Loc, Mag, Dir | **PASS** | Type=1 vs 1, Loc=2019.6340, Mag=0.0000 vs 0.0000, Dir=0.0000 vs 0.0000 |
| `EVIC_AB.dat` | `just_monitor(ALLSSA, parity)` | Type, Loc, Mag, Dir | **PASS** | Type=0 vs 0, Loc=2019.8280, Mag=0.0000 vs 0.0000, Dir=0.0000 vs 0.0000 |
| `EVIC_AB.dat` | `just_monitor(OLS, parity)` | Type, Loc, Mag, Dir | **PASS** | Type=0 vs 0, Loc=2019.8280, Mag=0.0000 vs 0.0000, Dir=0.0000 vs 0.0000 |
| `EVIA_CA.dat` | `just_monitor(ALLSSA, parity)` | Type, Loc, Mag, Dir | **PASS** | Type=1 vs 1, Loc=2017.8183, Mag=-0.0508 vs -0.0508, Dir=0.0083 vs 0.0083 |
| `EVIA_CA.dat` | `just_monitor(OLS, parity)` | Type, Loc, Mag, Dir | **PASS** | Type=1 vs 1, Loc=2017.8183, Mag=-0.0490 vs -0.0490, Dir=0.0074 vs 0.0074 |

## Documented Intentional Divergences (Default Mode vs. `parity_mode=True`)

| Upstream Quirk / Bug | Behavior in `parity_mode=True` | Intentional Fix in Default Mode |
| :--- | :--- | :--- |
| `LSSA.py:270` `UnboundLocalError` when `trend='none', freq=[]` | Fixed in all modes (QR projection) | Handled cleanly without branching crash |
| `LSSA.py:250` `ZeroDivisionError` when `n == nc + 2` | Enforces `n >= nc + 3` for spectrum | Raises `InsufficientDataError` instead of dividing by zero |
| `ALLSSA.py:117` singular matrix when frequency `< 0.5` rounds to `0` | Unclamped fine grid | Clamps fine search grid strictly positive (`>= 10**-decimal`) + `max_iter` cycle guard |
| `JUSTjumps.py:168` / `JUSTdecompose.py:194` duplicate last window when `(n - size) % step == 0` | Reproduces upstream `stop > n` loop | Stops cleanly at `stop >= n` so final window is never double-counted |
| `JUSTjumps.py:283` premature 2-decimal rounding of `Mag`/`Dir` before thresholding | Rounds `Mag`/`Dir` to 2 decimals | Preserves full `float64` precision + computes nested partial $F$-test `jump_pvalue` |
| `JUSTdecompose.py:42` off-by-one `+1` shift in `TrendComponent` regularization | Reproduces `+1` grid index shift | Uses exact `np.searchsorted(tt, t[j], side='left')` |
| `JUSTmonitor.py:215` unadjusted repeated LSS testing across 35 freqs × $D$ monitoring steps | Uses `freq_correction='none'`, `sequential_control='none'` | Defaults to `freq_correction='sidak'` + `sequential_control='pocock'` (also supports `obrien_fleming` & `consecutive_k`) to bound familywise false alarms |
| `STPD.py:58, 181` default `margin=1` causes `0/0 = NaN` in `SE1` | Clamps `margin >= 2` safely | Defaults to `max(2, min(M, size // 5))` and enforces `margin >= 2` |
| `STPD.py:161, 164` non-shift-invariant `SNR` & density-biased L2 `NDRI` | Uses `snr_mode='raw'`, `ndri_mode='l2'` | Defaults to `snr_mode='centered'`, `ndri_mode='rmse'` (both variants always stored on `StpdResult`) |
| `TPTR.py:53, 126` rounds stats to 2 decimals & drops initial segment slope | Preserves sequential TPTR ordering | Returns full-precision `PiecewiseTrendResult` with all `m + 1` slopes + optional `mode='joint'` |
