"""Tests for time-coordinate adapter, leap-year decimal years, and NaN remapping."""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pytest

from tspec._time import (
    coerce_times,
    convert_eval_times,
    estimate_sampling_rate_ratio,
    prepare_numeric_series,
)
from tspec.exceptions import InsufficientDataError

# Calendar decimal years differ from the uniform mean-Gregorian-year scale by at most ~2 days.
TWO_DAYS_YR = 2.0 / 365.2425


def _uniform_year(iso: str) -> float:
    """Reference uniform decimal year: 2000 + days since 2000-01-01 / 365.2425."""
    days = (np.datetime64(iso, "ns") - np.datetime64("2000-01-01", "ns")) / np.timedelta64(1, "D")
    return 2000.0 + float(days) / 365.2425


def test_datetime64_uniform_decimal_years() -> None:
    """AUD-01: datetimes map to a *uniform* decimal-year axis (mean Gregorian year)."""
    t = np.array(["2000-01-01", "2020-01-01", "2020-07-02", "2021-01-01"], dtype="datetime64[D]")
    t_num, t_raw, is_dt = coerce_times(t)
    assert is_dt is True
    assert t_raw.dtype == np.dtype("datetime64[ns]")
    assert t_num[0] == 2000.0
    for k, iso in enumerate(["2000-01-01", "2020-01-01", "2020-07-02", "2021-01-01"]):
        assert np.isclose(t_num[k], _uniform_year(iso), rtol=0.0, atol=1e-12)
    # Stays close to the calendar decimal year
    assert np.allclose(t_num[1:], [2020.0, 2020.5, 2021.0], atol=TWO_DAYS_YR)


def test_datetime_axis_is_uniform_across_leap_years() -> None:
    """AUD-01 regression: equal durations map to equal decimal-year increments."""
    dt = np.arange("2023-01-01T00", "2025-01-01T00", 3, dtype="datetime64[h]")
    t_num, _, _ = coerce_times(dt)
    steps = np.diff(t_num)
    assert np.max(steps) / np.min(steps) < 1.0 + 1e-9


def test_datetime_non_annual_frequency_is_phase_coherent() -> None:
    """AUD-01 regression: a noiseless M2 tide on datetime input is fitted exactly."""
    from tspec import lssa

    dt = np.arange("2023-01-01T00", "2025-01-01T00", 3, dtype="datetime64[h]")
    hours = (dt - dt[0]).astype(float)
    y = np.cos(2 * np.pi * hours / 12.4206012)
    f_cpy = 365.2425 * 24.0 / 12.4206012  # cycles per (mean) year
    r = lssa(dt, y, omega=[f_cpy], trend="constant")
    assert r.spectrum[0] > 0.999999


def test_python_datetime_and_date_inputs() -> None:
    dates = [date(2021, 1, 1), date(2021, 7, 2), date(2022, 1, 1)]
    t_num, _, is_dt = coerce_times(dates)
    assert is_dt is True
    assert t_num[0] == pytest.approx(2021.0, abs=TWO_DAYS_YR)
    assert t_num[-1] == pytest.approx(2022.0, abs=TWO_DAYS_YR)

    dts = [datetime(2021, 1, 1, 0, 0), datetime(2021, 1, 1, 12, 0)]
    t_num2, _, is_dt2 = coerce_times(dts)
    assert is_dt2 is True
    assert t_num2[1] > t_num2[0]


def test_pandas_datetime_index_and_series() -> None:
    pd = pytest.importorskip("pandas")
    idx = pd.date_range("2020-01-01", periods=10, freq="30D")
    s = pd.Series(np.arange(10, dtype=float), index=idx)
    prep = prepare_numeric_series(idx, s)
    assert prep.is_datetime is True
    assert prep.n_valid == 10
    assert np.allclose(prep.values, np.arange(10, dtype=float))


def test_nan_dropping_preserves_original_index_mapping() -> None:
    t = np.linspace(0.0, 5.0, 10)
    y = np.array([1.0, np.nan, 2.0, 3.0, np.nan, np.nan, 4.0, 5.0, 6.0, np.nan])
    prep = prepare_numeric_series(t, y)
    assert prep.n_input == 10
    assert prep.n_valid == 6
    assert prep.n_dropped_nan == 4
    assert prep.valid_to_orig.tolist() == [0, 2, 3, 6, 7, 8]
    assert np.allclose(prep.values, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_rejects_unsorted_duplicate_and_invalid_times() -> None:
    with pytest.raises(ValueError, match="monotonically ascending"):
        coerce_times([1.0, 3.0, 2.0])
    with pytest.raises(ValueError, match="duplicate"):
        coerce_times([1.0, 2.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="NaN or infinite"):
        coerce_times([1.0, np.nan, 3.0])
    with pytest.raises(ValueError, match="infinite"):
        prepare_numeric_series([1.0, 2.0, 3.0], [1.0, np.inf, 3.0])
    with pytest.raises(InsufficientDataError):
        prepare_numeric_series([1.0, 2.0], [np.nan, np.nan])


def test_eval_times_range_validation() -> None:
    tt_num, _ = convert_eval_times(np.linspace(1.0, 5.0, 20), 1.0, 5.0, expect_datetime=False)
    assert tt_num.size == 20
    with pytest.raises(ValueError, match="within the time span"):
        convert_eval_times(np.linspace(0.5, 5.0, 10), 1.0, 5.0, expect_datetime=False)


def test_sampling_rate_ratio_detects_density_shift() -> None:
    t_uniform = np.linspace(0.0, 10.0, 100)
    assert estimate_sampling_rate_ratio(t_uniform) < 1.3

    t_shift = np.concatenate([np.linspace(0.0, 5.0, 25), np.linspace(5.05, 10.0, 200)])
    assert estimate_sampling_rate_ratio(t_shift) > 4.0


def test_datetime_edge_cases_range_2262_iso_and_tz() -> None:
    """CRT-06: Python datetimes enforce ns range, year 2262 does not overflow, object/Series ISO strings work."""
    from datetime import timezone

    # Python datetime outside 1678-2262 must raise ValueError instead of silently wrapping
    with pytest.raises(ValueError, match="1678-2262"):
        coerce_times([datetime(2500, 1, 1), datetime(2500, 2, 1)])
    with pytest.raises(ValueError, match="1678-2262"):
        coerce_times([date(1500, 1, 1), date(1500, 2, 1)])

    # Year 2262 (valid within ns range) must not overflow when computing year_length
    t_2262 = np.array(["2262-01-01", "2262-02-01", "2262-03-01"], dtype="datetime64[D]")
    t_num, _, is_dt = coerce_times(t_2262)
    assert is_dt is True
    assert np.isclose(t_num[0], _uniform_year("2262-01-01"), atol=1e-12)
    assert np.all(np.diff(t_num) > 0)

    # Tz-aware Python datetimes normalize cleanly to UTC
    tz_dts = [
        datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc),
        datetime(2020, 7, 2, 0, 0, tzinfo=timezone.utc),
    ]
    t_tz, _, is_dt_tz = coerce_times(tz_dts)
    assert is_dt_tz is True
    assert np.isclose(t_tz[0], _uniform_year("2020-01-01"), atol=1e-12)
    assert np.isclose(t_tz[1], _uniform_year("2020-07-02"), atol=1e-12)

    # Object array of ISO date strings and pandas Series of ISO date strings
    iso_obj = np.array(["2020-01-01", "2020-07-02", "2021-01-01"], dtype=object)
    t_obj, _, is_dt_obj = coerce_times(iso_obj)
    assert is_dt_obj is True
    assert np.allclose(t_obj, [_uniform_year(d) for d in ("2020-01-01", "2020-07-02", "2021-01-01")], atol=1e-12)

    pd = pytest.importorskip("pandas")
    s_iso = pd.Series(["2020-01-01", "2020-07-02", "2021-01-01"])
    t_s, _, is_dt_s = coerce_times(s_iso)
    assert is_dt_s is True
    assert np.allclose(t_s, [_uniform_year(d) for d in ("2020-01-01", "2020-07-02", "2021-01-01")], atol=1e-12)

