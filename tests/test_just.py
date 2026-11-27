"""Unit tests for JUST jump detection and decomposition."""

from __future__ import annotations

import numpy as np
import pytest

from tspec import just_decompose, just_jumps


def test_just_recovers_abrupt_jump_with_seasonality_and_gaps(rng: np.random.Generator) -> None:
    t = np.sort(rng.uniform(2015.0, 2021.0, size=180))
    # Insert a 3-month blackout gap without interpolation
    t = t[~((t > 2016.3) & (t < 2016.55))]
    jump_time = 2018.2
    jump_idx_true = int(np.searchsorted(t, jump_time))

    trend_true = np.where(t < t[jump_idx_true], 0.5 + 0.02 * (t - 2015.0), 0.15 + 0.08 * (t - 2015.0))
    season_true = 0.15 * np.cos(2 * np.pi * t) + 0.07 * np.sin(2 * np.pi * 2.0 * t)
    y = trend_true + season_true + rng.normal(0.0, 0.02, size=t.size)

    res = just_jumps(t, y, season="allssa", mag_th=0.1, dir_th=0.05, jump_alpha=0.01)
    assert len(res.jumps) == 1
    j = res.jumps[0]
    assert abs(j.index - jump_idx_true) <= 2
    assert j.jump_pvalue < 1e-4
    assert "JUST Jump Detection" in res.summary()
    assert "abrupt change" in res.explain()

    # Decompose with detected jump & evaluate onto regular grid
    tt = np.linspace(t[0], t[-1], 240)
    dec = just_decompose(t, y, jump_indices=res.jump_indices, eval_times=tt, season="allssa")
    assert dec.trend.shape == t.shape
    assert dec.trend_reg is not None and dec.trend_reg.shape == tt.shape
    assert dec.seasonal_reg is not None and dec.seasonal_reg.shape == tt.shape
    assert np.allclose(dec.trend + dec.seasonal + dec.remainder, y, atol=1e-10)
    assert float(np.std(dec.remainder)) < 0.05


def test_just_duration_window_mode_and_variable_rate_hint(rng: np.random.Generator) -> None:
    t1 = np.linspace(0.0, 4.0, 48)
    t2 = np.linspace(4.02, 8.0, 200)
    t = np.concatenate([t1, t2])
    y = np.where(t < 5.0, 1.0, 2.5) + 0.3 * np.sin(2 * np.pi * t) + rng.normal(0.0, 0.05, size=t.size)

    # Count mode emits VARIABLE_SAMPLING_RATE hint
    r_cnt = just_jumps(t, y, season="ols", mag_th=0.5, dir_th=0.2)
    codes = {w.code for w in r_cnt.diagnostics.warnings}
    assert "VARIABLE_SAMPLING_RATE" in codes

    # Duration mode handles shifting rate cleanly
    r_dur = just_jumps(
        t,
        y,
        window_mode="duration",
        window_duration=3.0,
        step_duration=1.0,
        season="ols",
        mag_th=0.5,
        dir_th=0.2,
    )
    assert len(r_dur.jumps) == 1
    assert abs(r_dur.jumps[0].time_year - 5.0) < 0.15


def test_just_scale_mismatch_warning_on_large_units(rng: np.random.Generator) -> None:
    t = np.linspace(0.0, 6.0, 120)
    y = 500.0 + 20.0 * t + rng.normal(0.0, 5.0, size=t.size)
    # Threshold-only (uncalibrated) selection with NDVI-scale mag_th on large units
    res = just_jumps(t, y, season="none", jump_alpha=None, mag_th=0.05)
    codes = {w.code for w in res.diagnostics.warnings}
    assert "THRESHOLD_SCALE_MISMATCH" in codes
    assert "UNCALIBRATED_JUMP_SELECTION" in codes


def test_just_default_calibrated_selection_controls_false_positives() -> None:
    """Audit BLK-01: default just_jumps must bound P(any false jump) near jump_alpha."""
    hits = 0
    R = 120
    for r in range(R):
        rng = np.random.default_rng(10_000 + r)
        t = np.sort(rng.uniform(2000.0, 2006.25, 150))
        y = 0.5 + 0.1 * np.cos(2 * np.pi * t) + rng.normal(0.0, 0.02, t.size)
        res = just_jumps(t, y, season="ols", n_mc=199)
        hits += len(res.jumps) > 0
    # nominal 5%; 120 reps -> MCSE ~2%; old behaviour was ~98%
    assert hits / R <= 0.11
    assert res.jump_calibration == "montecarlo"


def test_mc_maxf_pvalue_matches_bruteforce_null() -> None:
    """The Monte Carlo max-over-splits p-value equals a brute-force refit estimate."""
    from scipy import stats

    from tspec._linalg import prepare_weights
    from tspec.just import _mc_maxf_pvalue

    rng = np.random.default_rng(0)
    n = 60
    t0 = np.sort(rng.uniform(0, 3, n))
    t0 -= t0[0]
    u = t0 / t0[-1]
    pw = prepare_weights(None, n)

    def max_f(y: np.ndarray) -> float:
        x0 = np.column_stack([np.ones(n), u])
        r0 = np.linalg.lstsq(x0, y, rcond=None)[1][0]
        best = 0.0
        for j in range(3, n - 2):
            d = (np.arange(n) >= j).astype(float)
            r1 = np.linalg.lstsq(np.column_stack([x0, d, d * u]), y, rcond=None)[1][0]
            best = max(best, ((r0 - r1) / 2) / (r1 / (n - 4)))
        return best

    null = np.array([max_f(rng.normal(size=n)) for _ in range(300)])
    f_obs = float(np.quantile(null, 0.9))
    p = _mc_maxf_pvalue(t0, pw, np.empty(0), f_obs, 3, n - 2, 2999, 1)
    assert abs(p - 0.10) < 0.05
    assert p > 5 * stats.f.sf(f_obs, 2, n - 4)  # naive fixed-split p is far smaller


def test_just_nan_remapping_to_original_axis(rng: np.random.Generator) -> None:
    t = np.linspace(0.0, 6.0, 120)
    y = np.where(t < 3.0, 0.2, 0.9) + rng.normal(0.0, 0.02, size=t.size)
    y[[5, 10, 15, 20]] = np.nan
    res = just_jumps(t, y, season="none", mag_th=0.3, dir_th=0.2)
    assert res.diagnostics.n_dropped_nan == 4
    assert len(res.jumps) == 1
    # Mapped index must point to t ~ 3.0 on the ORIGINAL 120-length axis (index ~60)
    assert abs(t[res.jumps[0].index] - 3.0) < 0.15


def test_just_datetime_input_and_dataframe_export(rng: np.random.Generator) -> None:
    pytest.importorskip("pandas")
    t = np.arange("2018-01-01", "2023-01-01", 15, dtype="datetime64[D]")
    t_yr = (t - t[0]).astype("timedelta64[D]").astype(float) / 365.25
    y = np.where(t_yr < 2.5, 0.2, 0.8) + 0.1 * np.cos(2 * np.pi * t_yr) + rng.normal(0.0, 0.02, size=t.size)
    res = just_jumps(t, y, season="ols", mag_th=0.2, dir_th=0.1, jump_alpha=0.01)
    assert res.is_datetime is True
    assert len(res.jumps) == 1
    df = res.to_dataframe()
    assert len(df) == 1
    assert "jump_pvalue" in df.columns


def test_just_jumps_allssa_detects_large_15sigma_and_30sigma_jumps() -> None:
    """BLK-02: null_allssa must not absorb large step jumps into sub-annual 0.5 cy/yr waves."""
    rng = np.random.default_rng(41750)
    t = np.linspace(2015.0, 2025.0, 120)
    t_j = 2020.0
    sigma = 0.2
    for js in (10.0, 15.0, 30.0):
        y = np.where(t < t_j, 1.0, 1.0 + js * sigma) + 0.8 * np.cos(2.0 * np.pi * t) + rng.normal(0.0, sigma, 120)
        res = just_jumps(t, y, season="allssa", n_mc=199)
        assert any(abs(j.time_year - t_j) <= 0.5 for j in res.jumps), f"Missed {js}sigma jump!"


def test_just_decompose_allssa_seasonal_ci_edge_jump_and_nan_safe_summary() -> None:
    """BLK-07, MAJ-02, MAJ-04: ALLSSA seasonal CI covers 2-harmonic wave, edge jumps don't leak, summary is NaN-safe."""
    rng = np.random.default_rng(42002)
    n = 120
    t = np.linspace(2010.0, 2020.0, n)
    t0 = t - t[0]
    tr_true = 2.0 + 0.25 * t0
    seas_true = 1.0 * np.cos(2.0 * np.pi * t0) + 0.4 * np.sin(4.0 * np.pi * t0)
    y = tr_true + seas_true + rng.normal(0.0, 0.3, size=n)

    dec = just_decompose(t, y, season="allssa", size=36, step=12)
    assert dec.seasonal_bc is not None
    assert dec.seasonal_ci_low is not None and dec.seasonal_ci_high is not None
    cov_s = float(np.mean((dec.seasonal_ci_low <= seas_true) & (seas_true <= dec.seasonal_ci_high)))
    assert cov_s >= 0.85

    # Edge jump at idx=35 with size=36 must not bleed across the discontinuity
    tr_edge = np.where(np.arange(n) < 35, 2.0 + 0.1 * t0, 12.0 + 0.1 * t0)
    y_edge = tr_edge + seas_true + rng.normal(0.0, 0.3, size=n)
    dec_edge = just_decompose(t, y_edge, jump_indices=[35], season="ols", size=36, step=12)
    assert abs(float(dec_edge.trend[34] - tr_edge[34])) < 1.0
    assert abs(float(dec_edge.trend[35] - tr_edge[35])) < 1.0

    # NaN-safe summary() and explain() when trend/seasonal/remainder contain NaN entries
    y_gap = y.copy()
    y_gap[45:75] = np.nan
    dec_gap = just_decompose(t, y_gap, season="ols", size=30, step=15, eval_times=t)
    from dataclasses import replace

    t_with_nan = dec_gap.trend.copy()
    t_with_nan[0] = np.nan
    dec_nan = replace(dec_gap, trend=t_with_nan, seasonal=t_with_nan, remainder=t_with_nan)
    summ = dec_nan.summary()
    expl = dec_nan.explain()
    assert "Trend range  : nan" not in summ
    assert "Seasonal half-range: nan" not in summ
    assert "Remainder RMS: nan" not in summ
    assert "~nan" not in expl


def test_just_jumps_holm_before_mag_th_and_parity_n_mc() -> None:
    """MAJ-12 & MIN-05: mag_th must not shrink Holm denominator; parity_mode ignores n_mc."""
    rng = np.random.default_rng(912)
    t = np.linspace(2015.0, 2025.0, 120)
    y = np.where(t < 2020.0, 1.0, 2.2) + rng.normal(0.0, 0.25, size=120)
    r0 = just_jumps(t, y, season="none", mag_th=0.0, dir_th=0.0, n_mc=199)
    r1 = just_jumps(t, y, season="none", mag_th=0.8, dir_th=0.0, n_mc=199)
    assert len(r0.jumps) == 1 and len(r1.jumps) == 1
    np.testing.assert_allclose(r1.jumps[0].jump_pvalue, r0.jumps[0].jump_pvalue, rtol=1e-12)

    # parity_mode=True must not reject small n_mc
    rp = just_jumps(t, y, season="none", parity_mode=True, n_mc=10)
    assert rp is not None

