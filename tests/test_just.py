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
    res = just_jumps(t, y, season="none")
    codes = {w.code for w in res.diagnostics.warnings}
    assert "THRESHOLD_SCALE_MISMATCH" in codes


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
