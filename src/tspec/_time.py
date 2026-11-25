"""Time-coordinate adapter, validation, and transparent NaN-mask index mapping.

Accepts numeric arrays, ``numpy.datetime64``, Python ``datetime`` / ``date``
sequences, or pandas ``DatetimeIndex`` / ``Series`` (without importing pandas
unless a pandas object is actually passed). Datetime inputs are converted to
continuous decimal years accounting for actual leap-year lengths so cyclic
frequencies are naturally expressed in cycles/year.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np

from ._types import ArrayLike, BoolArray, FloatArray, IntArray, TimeLike
from .exceptions import InsufficientDataError


@dataclass(frozen=True)
class TimeSeriesInput:
    """Validated, NaN-filtered time-series representation with original-index mapping."""

    times_numeric: FloatArray
    times_raw: np.ndarray[Any, Any]
    is_datetime: bool
    values: FloatArray
    valid_mask: BoolArray
    valid_to_orig: IntArray
    n_input: int
    n_valid: int
    n_dropped_nan: int
    sampling_rate_ratio: float


def _to_array(obj: Any) -> np.ndarray[Any, Any]:
    if type(obj).__module__.startswith("pandas") and hasattr(obj, "to_numpy"):
        # Normalize timezone-aware pandas DatetimeIndex / Series cleanly to UTC-naive
        # to avoid NumPy's timezone representation UserWarning.
        tz = getattr(obj, "tz", None)
        if tz is None and hasattr(obj, "dt"):
            tz = getattr(obj.dt, "tz", None)
            if tz is not None:
                return np.asarray(obj.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy())
        elif tz is not None:
            return np.asarray(obj.tz_convert("UTC").tz_localize(None).to_numpy())
        return np.asarray(obj.to_numpy())
    return np.asarray(obj)


def _datetime64_to_decimal_year(dt64: np.ndarray[Any, Any]) -> FloatArray:
    """Convert a 1D numpy datetime64 array to exact continuous decimal years."""
    dt_ns = dt64.astype("datetime64[ns]")
    if np.any(np.isnat(dt_ns)):
        raise ValueError("Time vector contains NaT (Not-a-Time) values.")

    years = dt_ns.astype("datetime64[Y]")
    year_int = years.astype(np.int64) + 1970
    start_of_year = years.astype("datetime64[ns]")
    start_of_next_year = (years + np.timedelta64(1, "Y")).astype("datetime64[ns]")

    elapsed = (dt_ns - start_of_year).astype(np.float64)
    year_length = (start_of_next_year - start_of_year).astype(np.float64)
    return np.asarray(year_int.astype(np.float64) + elapsed / year_length, dtype=np.float64)


def _check_datetime_ns_range(dt: np.ndarray[Any, Any]) -> None:
    """Reject datetimes outside the datetime64[ns] range (1678-2262), which would wrap silently."""
    d = dt.astype("datetime64[us]")
    fin = ~np.isnat(d)
    if np.any(fin):
        lo = np.datetime64("1678-01-01", "us")
        hi = np.datetime64("2262-04-11", "us")
        if np.any(d[fin] < lo) or np.any(d[fin] > hi):
            raise ValueError(
                "Datetime values outside 1678-2262 cannot be represented at nanosecond precision; "
                "pass numeric decimal years instead."
            )


def coerce_times(
    t: TimeLike,
    *,
    allow_duplicates: bool = False,
) -> tuple[FloatArray, np.ndarray[Any, Any], bool]:
    """Validate and convert an input time axis into numeric decimal years + raw timestamps.

    Parameters
    ----------
    t:
        1D array-like of numeric timestamps or datetime objects.
    allow_duplicates:
        If ``False`` (default), raises ``ValueError`` on duplicate timestamps.

    Returns
    -------
    times_numeric:
        Monotonically ordered ``float64`` timestamps (decimal years if datetime).
    times_raw:
        1D array in the original representation (``datetime64[ns]`` or ``float64``).
    is_datetime:
        ``True`` if input carried calendar datetime semantics.
    """
    arr = _to_array(t)

    if arr.ndim != 1:
        raise ValueError(f"Time vector `t` must be 1-D; got shape {arr.shape}.")
    if arr.size == 0:
        raise InsufficientDataError("Time vector `t` must not be empty.")

    if np.issubdtype(arr.dtype, np.str_):
        # Plain numeric strings (e.g. "2001.5", "3") are numbers, not dates.
        with contextlib.suppress(ValueError):
            arr = arr.astype(np.float64)

    is_datetime = False
    if np.issubdtype(arr.dtype, np.datetime64):
        is_datetime = True
        _check_datetime_ns_range(arr)
        raw_dt = arr.astype("datetime64[ns]")
        t_num = _datetime64_to_decimal_year(raw_dt)
        t_raw: np.ndarray[Any, Any] = raw_dt
    elif arr.dtype == object:
        # Inspect first non-None element to detect datetime/date objects or ISO strings
        first_non_none = next((x for x in arr if x is not None), None)
        if isinstance(first_non_none, (datetime, date, np.datetime64)):
            is_datetime = True
            if any(x is None for x in arr):
                raise ValueError("Time vector contains None/NaT entries.")
            raw_dt = np.array(
                [
                    np.datetime64(x, "ns")
                    if isinstance(x, (datetime, np.datetime64))
                    else np.datetime64(datetime(x.year, x.month, x.day), "ns")
                    for x in arr
                ],
                dtype="datetime64[ns]",
            )
            t_num = _datetime64_to_decimal_year(raw_dt)
            t_raw = raw_dt
        else:
            try:
                t_num = np.asarray(arr, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Could not convert time vector `t` to float or datetime64."
                ) from exc
            t_raw = t_num.copy()
    elif np.issubdtype(arr.dtype, np.str_):
        try:
            parsed = arr.astype("datetime64[us]")
        except ValueError as exc:
            raise ValueError(
                "String time vector `t` could not be parsed as ISO-8601 datetime64."
            ) from exc
        _check_datetime_ns_range(parsed)
        raw_dt = parsed.astype("datetime64[ns]")
        is_datetime = True
        t_num = _datetime64_to_decimal_year(raw_dt)
        t_raw = raw_dt
    else:
        t_num = np.asarray(arr, dtype=np.float64)
        t_raw = t_num.copy()

    if not np.all(np.isfinite(t_num)):
        raise ValueError("Time vector `t` contains NaN or infinite values.")

    diffs = np.diff(t_num)
    if np.any(diffs < 0):
        raise ValueError(
            "Time vector `t` must be sorted in monotonically ascending order."
        )
    if not allow_duplicates and np.any(diffs == 0):
        raise ValueError(
            "Time vector `t` contains duplicate timestamps. Aggregate or remove "
            "duplicates before analysis."
        )

    return t_num, t_raw, is_datetime


def convert_eval_times(
    eval_times: TimeLike,
    t_min: float,
    t_max: float,
    *,
    expect_datetime: bool,
) -> tuple[FloatArray, np.ndarray[Any, Any]]:
    """Convert and validate an evaluation/regularization grid ``eval_times``."""
    tt_num, tt_raw, is_dt = coerce_times(eval_times, allow_duplicates=False)
    if expect_datetime and not is_dt:
        # Numeric decimal years are still allowed for datetime series if in range
        pass
    elif not expect_datetime and is_dt:
        raise ValueError(
            "`eval_times` has datetime type, but input time vector `t` is numeric."
        )
    tol = 1e-9 * max(1.0, abs(t_max - t_min))
    if tt_num[0] < t_min - tol or tt_num[-1] > t_max + tol:
        raise ValueError(
            "`eval_times` must lie within the time span of input `t` "
            f"([{t_min:.6g}, {t_max:.6g}])."
        )
    return tt_num, tt_raw


def estimate_sampling_rate_ratio(t_num: FloatArray) -> float:
    """Measure local sampling-rate variation across halves/tertiles of a series.

    Returns the ratio ``max_rate / min_rate`` (>= 1.0). Ratios above 2.0
    indicate a substantial sampling-frequency shift where duration-based
    windowing may be preferable to observation-count windowing.
    """
    n = t_num.size
    if n < 12 or t_num[-1] <= t_num[0]:
        return 1.0
    n_bins = 3 if n >= 30 else 2
    edges = np.linspace(t_num[0], t_num[-1], n_bins + 1)
    rates: list[float] = []
    for k in range(n_bins):
        lo, hi = edges[k], edges[k + 1]
        span = hi - lo
        if span <= 0:
            continue
        if k < n_bins - 1:
            cnt = int(np.sum((t_num >= lo) & (t_num < hi)))
        else:
            cnt = int(np.sum((t_num >= lo) & (t_num <= hi)))
        rates.append(cnt / span)
    if len(rates) < 2 or max(rates) <= 0:
        return 1.0
    if min(rates) <= 0.0:
        return float("inf")
    return float(max(rates) / min(rates))


def prepare_numeric_series(
    t: TimeLike,
    y: ArrayLike,
    *,
    allow_duplicates: bool = False,
) -> TimeSeriesInput:
    """Validate ``t`` and numeric ``y``, dropping ``NaN``s transparently without interpolation."""
    t_num, t_raw, is_dt = coerce_times(t, allow_duplicates=allow_duplicates)

    y_arr = np.asarray(_to_array(y), dtype=np.float64)

    if y_arr.ndim != 1:
        raise ValueError(f"Value vector `y` must be 1-D; got shape {y_arr.shape}.")
    if y_arr.size != t_num.size:
        raise ValueError(
            f"`t` and `y` must have the same length; got {t_num.size} and {y_arr.size}."
        )
    if np.any(np.isinf(y_arr)):
        raise ValueError("Value vector `y` contains infinite entries.")

    valid_mask = ~np.isnan(y_arr)
    n_input = int(y_arr.size)
    n_valid = int(np.sum(valid_mask))
    if n_valid == 0:
        raise InsufficientDataError("All observations in `y` are NaN.")

    valid_to_orig = np.flatnonzero(valid_mask)
    t_valid = t_num[valid_mask]
    t_raw_valid = t_raw[valid_mask]
    y_valid = y_arr[valid_mask]
    rate_ratio = estimate_sampling_rate_ratio(t_valid)

    return TimeSeriesInput(
        times_numeric=t_valid,
        times_raw=t_raw_valid,
        is_datetime=is_dt,
        values=y_valid,
        valid_mask=valid_mask,
        valid_to_orig=valid_to_orig,
        n_input=n_input,
        n_valid=n_valid,
        n_dropped_nan=n_input - n_valid,
        sampling_rate_ratio=rate_ratio,
    )


def format_timestamp(value: Any, is_datetime: bool) -> str:
    """Format a scalar timestamp cleanly for summary tables and narratives."""
    if is_datetime:
        dt = np.datetime64(value, "D")
        return str(dt)
    return f"{float(value):.4f}"
