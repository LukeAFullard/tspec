"""Regression tests for batch 4 of the independent audit (AUD-05, -11, -12, -13, -15, AUD-02 rest)."""

from __future__ import annotations

import numpy as np
import pytest

import tspec


def test_allssa_resolves_two_components_within_one_cycle_per_unit() -> None:
    """AUD-05: the published integer-band rule keeps at most one frequency per unit band."""
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 12, 300))
    y = np.cos(2 * np.pi * 1.37 * t + 0.4) + 0.8 * np.sin(2 * np.pi * 1.0 * t) + rng.normal(0, 0.5, t.size)
    new = tspec.allssa(t, y, decimal=2, n_mc=499).frequencies
    assert np.any(np.abs(new - 1.0) < 0.02) and np.any(np.abs(new - 1.37) < 0.02)
    old = tspec.allssa(t, y, decimal=2, n_mc=499, search="integer").frequencies
    assert old.size == 1


def test_allssa_default_grid_reaches_slow_cycles() -> None:
    rng = np.random.default_rng(1)
    t = np.arange(0, 60, 1 / 12)
    y = 2 * np.sin(2 * np.pi * t / 11.2) + np.cos(2 * np.pi * t) + rng.normal(0, 0.5, t.size)
    f = tspec.allssa(t, y, n_mc=499).frequencies
    assert np.any(np.abs(f - 1 / 11.2) < 0.01) and np.any(np.abs(f - 1.0) < 0.02)


def test_allssa_search_options() -> None:
    rng = np.random.default_rng(2)
    t = np.sort(rng.uniform(0, 10, 120))
    y = rng.normal(size=120)
    assert tspec.allssa(t, y, n_mc=499).frequencies.size == 0
    with pytest.raises(ValueError, match="search"):
        tspec.allssa(t, y, search="dense")  # type: ignore[arg-type]
    # parity mode keeps the published integer-band search
    assert isinstance(tspec.allssa(t, y, parity_mode=True).frequencies, np.ndarray)


def test_sparse_records_reduce_default_harmonics() -> None:
    """AUD-13: the default 4-harmonic OLS model no longer fits sparse windows."""
    rng = np.random.default_rng(7)
    t = np.sort(rng.uniform(0, 20, 200))
    y = 0.1 * t + np.cos(2 * np.pi * t) + rng.normal(0, 0.3, 200) + 2.0 * (t > 10)
    y = np.where(rng.random(200) < 0.5, np.nan, y)
    r = tspec.just_jumps(t, y, season="ols")
    assert any(w.code == "HARMONICS_REDUCED" for w in r.diagnostics.warnings)
    assert any(abs(r.times_numeric[j.valid_index] - 10) < 0.5 for j in r.jumps)


def test_lssa_memory_chunk_is_bounded() -> None:
    from tspec._linalg import _adaptive_chunk

    assert _adaptive_chunk(1000, 512) == 512
    assert _adaptive_chunk(1_000_000, 512) == 8
    assert _adaptive_chunk(32_000, 512) * 32_000 <= 1_000_000 + 32_000


def test_just_decompose_bands_widen_under_ar1() -> None:
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0, 20, 240))
    e = np.empty(240)
    e[0] = rng.normal()
    for i in range(1, 240):
        e[i] = 0.7 * e[i - 1] + np.sqrt(1 - 0.49) * rng.normal()
    y = 1.5 * np.sin(2 * np.pi * t / 9) + np.cos(2 * np.pi * t) + 0.4 * e
    a = tspec.just_decompose(t, y, season="ols")
    b = tspec.just_decompose(t, y, season="ols", serial_correlation="none")
    assert np.nanmean(a.trend_ci_high - a.trend_ci_low) > 1.5 * np.nanmean(b.trend_ci_high - b.trend_ci_low)
    assert any(w.code == "SERIAL_CORRELATION_ADJUSTED" for w in a.diagnostics.warnings)
    assert np.allclose(a.trend, b.trend)


def test_heavy_censoring_ols_mi_caution() -> None:
    rng = np.random.default_rng(4)
    t = 2010 + np.sort(rng.uniform(0, 10, 120))
    y = 1.0 + 0.1 * (t - 2010) + rng.normal(0, 0.4, 120)
    dl = float(np.quantile(y, 0.5))
    status = np.where(y < dl, -1, 0)
    r = tspec.analyze_series(
        t, np.where(status == -1, dl, y), censor_status=status, detect_jumps=False,
        detect_turning_points=False, remove_seasonality=False, band_method="conditional",
    )
    assert r.segments[0].estimator == "ols_mi"
    assert any(w.code == "HEAVY_CENSORING_OLS_MI" for w in r.diagnostics.warnings)
