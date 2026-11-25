"""Edge cases for time/value adapters. Run: python audit/repro_time_inputs.py"""
import warnings, datetime as dt
import numpy as np, pandas as pd
import tspec
from tspec._time import coerce_times, prepare_numeric_series

def tryit(label, f):
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = f()
            msg = f"OK -> {r}" + (f"  [warnings: {[str(x.message)[:80] for x in w]}]" if w else "")
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
    print(f"{label}: {msg}")

# tz-aware python datetimes in different zones representing the same instants
a = [dt.datetime(2020, 1, 1, 12, tzinfo=dt.timezone.utc), dt.datetime(2020, 1, 2, 12, tzinfo=dt.timezone.utc)]
b = [dt.datetime(2020, 1, 1, 12, tzinfo=dt.timezone(dt.timedelta(hours=-10))),
     dt.datetime(2020, 1, 2, 12, tzinfo=dt.timezone(dt.timedelta(hours=-10)))]
tryit("tz-aware datetime UTC", lambda: coerce_times(a)[0])
tryit("tz-aware datetime UTC-10 (should be +10h/8784h later)", lambda: coerce_times(b)[0])
# pandas tz-aware series
s = pd.Series(pd.date_range("2020-01-01", periods=3, freq="D", tz="Pacific/Auckland"))
tryit("pandas tz-aware Series", lambda: coerce_times(s)[0])
idx = pd.date_range("2020-01-01", periods=3, freq="D", tz="Pacific/Auckland")
tryit("pandas tz-aware DatetimeIndex", lambda: coerce_times(idx)[0])
# leap-year correctness
tryit("2020-07-02 (leap year mid) decimal", lambda: coerce_times(np.array(["2020-07-02T00:00"], dtype="datetime64[ns]"))[0])
# unsorted input
tryit("unsorted numeric", lambda: coerce_times([3.0, 1.0, 2.0]))
tryit("duplicate", lambda: coerce_times([1.0, 1.0, 2.0]))
tryit("NaT", lambda: coerce_times(np.array(["2020-01-01", "NaT"], dtype="datetime64[ns]")))
tryit("numeric strings ['1','2']", lambda: coerce_times(np.array(["1", "2", "3"])))
tryit("year strings ['2001','2002']", lambda: coerce_times(np.array(["2001", "2002", "2003"]))[0])
# float32 & int input
tryit("float32 y", lambda: prepare_numeric_series(np.arange(5.0), np.arange(5, dtype=np.float32)).values.dtype)
tryit("int t", lambda: prepare_numeric_series(np.arange(5), np.arange(5.0)).times_numeric.dtype)
# pandas Series with non-default index -> positional
ys = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=[50, 40, 30, 20, 10])
tryit("pandas y non-default index", lambda: prepare_numeric_series(np.arange(5.0), ys).values)
# pandas nullable Float64 with NA
yna = pd.Series([1.0, None, 3.0, 4.0, 5.0], dtype="Float64")
tryit("pandas Float64 with NA", lambda: prepare_numeric_series(np.arange(5.0), yna).values)
# empty
tryit("empty", lambda: tspec.lssa([], []))
# all NaN
tryit("all-NaN", lambda: tspec.lssa([1.0, 2, 3], [np.nan] * 3))
# n smaller than params
tryit("n<params lssa linear n=2", lambda: tspec.lssa([1.0, 2.0], [1.0, 2.0]).coeff)
tryit("n=3 lssa linear with omega", lambda: tspec.lssa([1.0, 2.0, 3.0], [1.0, 2.0, 0.5], omega=[0.3]).spectrum)
# constant y
tryit("constant y lssa", lambda: tspec.lssa(np.arange(20.0), np.ones(20), omega=[0.1, 0.2]).spectrum)
tryit("constant y allssa", lambda: tspec.allssa(np.arange(20.0), np.ones(20)).frequencies)
tryit("constant y stpd", lambda: len(tspec.stpd(np.linspace(0, 10, 100), np.ones(100)).turning_points))
tryit("constant y just_jumps", lambda: len(tspec.just_jumps(np.linspace(2000, 2010, 200), np.ones(200)).jumps))
tryit("constant y analyze_series", lambda: tspec.analyze_series(np.linspace(2000, 2010, 200), np.ones(200)).segments[0])
# single unique t
tryit("single t", lambda: tspec.lssa([1.0], [2.0]))
# extreme magnitudes
tt = np.linspace(2000, 2010, 120)
yy = np.sin(2 * np.pi * tt) + 0.1 * np.random.default_rng(0).normal(size=120)
for sc in (1e-12, 1e12):
    tryit(f"lssa scale {sc} spectrum max", lambda sc=sc: float(np.max(tspec.lssa(tt, sc * yy, omega=[1.0, 2.0]).spectrum)))
tryit("lssa base 1e13", lambda: float(np.max(tspec.lssa(tt, 1e13 + yy, omega=[1.0, 2.0]).spectrum)))
# time in seconds-since-epoch (1.7e9) with tiny span
ts = 1.7e9 + np.linspace(0, 1000, 120)
tryit("lssa t~1.7e9 s", lambda: float(np.max(tspec.lssa(ts, yy, omega=[1/100, 1/50]).spectrum)))
# default ALLSSA grid for numeric time in days
td = np.sort(np.random.default_rng(1).uniform(0, 3650, 300))
yd = np.sin(2 * np.pi * td / 365.25) + 0.3 * np.random.default_rng(2).normal(size=300)
tryit("allssa days default grid freqs", lambda: tspec.allssa(td, yd).frequencies)
tryit("allssa years same data", lambda: tspec.allssa(td / 365.25, yd).frequencies)
# just_jumps with numeric time in days
tryit("just_jumps days", lambda: len(tspec.just_jumps(td, yd + 2.0 * (td > 1800), season="ols").jumps))
# input mutation check
y_in = yy.copy(); t_in = tt.copy()
tspec.analyze_series(t_in, y_in)
print("inputs unmodified:", np.array_equal(y_in, yy) and np.array_equal(t_in, tt))
