"""Property-based fuzzing of tspec invariants (audit-owned; run: pytest -q audit/test_hypothesis_audit.py -p no:cacheprovider)."""
import warnings
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck
import tspec
from tspec.exceptions import TspecError
warnings.simplefilter("ignore")
SET = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])

def series(draw, nmin=30, nmax=120):
    n = draw(st.integers(nmin, nmax))
    seed = draw(st.integers(0, 2**31 - 1))
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(2000, 2000 + n / draw(st.sampled_from([6, 12, 24])), n))
    t = np.unique(t)
    scale = draw(st.sampled_from([1e-6, 1.0, 1e6]))
    y = scale * (rng.normal(size=t.size) + np.cos(2 * np.pi * t))
    return t, y, rng

@SET
@given(st.data())
def test_lss_bounded_and_pvalues_valid(data):
    t, y, rng = series(data.draw)
    trend = data.draw(st.sampled_from(["none", "constant", "linear", "quadratic"]))
    w = rng.uniform(0.1, 10, t.size) if data.draw(st.booleans()) else None
    r = tspec.lssa(t, y, omega=np.linspace(0.05, 6, 50), trend=trend, weights=w)
    assert np.all((r.spectrum >= 0) & (r.spectrum <= 1))
    assert np.all((r.pvalues >= 0) & (r.pvalues <= 1))

@SET
@given(st.data())
def test_nan_insertion_equivalence(data):
    t, y, rng = series(data.draw)
    r0 = tspec.lssa(t, y, omega=np.linspace(0.1, 5, 20))
    k = data.draw(st.integers(1, 5))
    pos = np.sort(rng.choice(np.arange(1, t.size), k, replace=False))
    t2 = np.insert(t, pos, (t[pos - 1] + t[pos]) / 2); y2 = np.insert(y, pos, np.nan)
    r1 = tspec.lssa(t2, y2, omega=np.linspace(0.1, 5, 20))
    assert np.allclose(r0.spectrum, r1.spectrum, atol=1e-10)
    # residuals align with valid points in original order
    assert np.allclose(r0.residuals, r1.residuals, atol=1e-8 * max(1, np.abs(y).max()))

@SET
@given(st.data())
def test_decompose_reconstruction(data):
    t, y, rng = series(data.draw, 80, 160)
    try:
        d = tspec.just_decompose(t, y, season=data.draw(st.sampled_from(["ols", "none"])))
    except TspecError:
        return
    ok = np.isfinite(d.trend)
    assert np.allclose((d.trend + d.seasonal + d.remainder)[ok], y[ok], atol=1e-9 * max(1, np.abs(y).max()))
    lo, hi = d.trend_ci_low, d.trend_ci_high
    f = np.isfinite(lo) & np.isfinite(hi)
    center = d.trend_bc if d.trend_bc is not None else d.trend
    assert np.all(lo[f] <= center[f] + 1e-12) and np.all(hi[f] >= center[f] - 1e-12)

@SET
@given(st.data())
def test_stpd_scale_shift_equivariance(data):
    t, y, rng = series(data.draw, 60, 150)
    y = y / max(1e-300, np.abs(y).max())
    y = y + 0.5 * (t - t[0])
    a, b = data.draw(st.sampled_from([(3.0, 10.0), (0.01, -5.0)]))
    try:
        s0 = tspec.stpd(t, y); s1 = tspec.stpd(t, a * y + b)
    except TspecError:
        return
    assert list(s0.tp_indices) == list(s1.tp_indices)
    assert np.allclose(a * s0.piecewise_trend.fitted_trend + b, s1.piecewise_trend.fitted_trend, rtol=1e-7, atol=1e-7 * abs(b) + 1e-9)

@SET
@given(st.data())
def test_sen_vs_scipy(data):
    from scipy.stats import theilslopes
    from tspec._robust import fit_theil_sen_segment
    t, y, rng = series(data.draw, 10, 80)
    y = np.round(y, data.draw(st.integers(0, 3)))
    f = fit_theil_sen_segment(t, y)
    assert np.isclose(f.slope, theilslopes(y, t).slope, rtol=1e-12, atol=1e-12)

def test_determinism_and_global_rng_untouched():
    rng = np.random.default_rng(5)
    t = np.sort(rng.uniform(2000, 2010, 200)); y = rng.normal(size=200) + 2.0 * (t > 2005)
    yc = np.array([f"<{-0.5}" if v < -0.5 else float(v) for v in y], dtype=object)
    np.random.seed(123); before = np.random.random(3); np.random.seed(123)
    a1 = tspec.analyze_series(t, yc, slope_estimator="sen")
    after = np.random.random(3)
    a2 = tspec.analyze_series(t, yc, slope_estimator="sen")
    assert np.array_equal(before, after), "global numpy RNG state was consumed"
    assert np.array_equal(a1.combined_trend, a2.combined_trend)
    assert [j.index for j in a1.jumps.jumps] == [j.index for j in a2.jumps.jumps]
