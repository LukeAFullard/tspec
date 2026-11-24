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


def test_datetime64_leap_year_decimal_years() -> None:
    t = np.array(["2020-01-01", "2020-07-02", "2021-01-01"], dtype="datetime64[D]")
    t_num, t_raw, is_dt = coerce_times(t)
    assert is_dt is True
    assert t_raw.dtype == np.dtype("datetime64[ns]")
    assert np.isclose(t_num[0], 2020.0, atol=1e-12)
    # 2020 is a leap year (366 days): Jan 1 + 183 days = July 2 -> exact midpoint 2020.5
    assert np.isclose(t_num[1], 2020.5, atol=1e-12)
    assert np.isclose(t_num[2], 2021.0, atol=1e-12)


def test_python_datetime_and_date_inputs() -> None:
    dates = [date(2021, 1, 1), date(2021, 7, 2), date(2022, 1, 1)]
    t_num, _, is_dt = coerce_times(dates)
    assert is_dt is True
    assert t_num[0] == pytest.approx(2021.0)
    assert t_num[-1] == pytest.approx(2022.0)

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
