"""Calibration regression tests for the audit fixes (audit/AUDIT_REPORT.md).

Replicate counts are kept small for CI speed; thresholds sit far from the
pre-fix behaviour reported in the audit so the tests cannot pass by chance.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import tspec
from tspec._robust import ar1_rank_factor, fit_theil_sen_segment, hamed_rao_factor

warnings.simplefilter("ignore")


def test_allssa_default_controls_spurious_cycles() -> None:
    """BLK-02: default ALLSSA selected a spurious cycle in ~20% of noise series."""
    hits = 0
    R = 60
    for r in range(R):
        rng = np.random.default_rng(300 + r)
        t = np.sort(rng.uniform(2000.0, 2006.25, 150))
        res = tspec.allssa(t, rng.normal(size=t.size))
        hits += res.frequencies.size > 0
        assert res.significance == "montecarlo"
    assert hits / R <= 0.05
    # The published per-frequency rule is still available and visibly anti-conservative
    per_freq = 0
    for r in range(R):
        rng = np.random.default_rng(300 + r)
        t = np.sort(rng.uniform(2000.0, 2006.25, 150))
        per_freq += tspec.allssa(t, rng.normal(size=t.size), significance="per_frequency").frequencies.size > 0
    assert per_freq > hits


def test_allssa_still_recovers_strong_cycle() -> None:
    rng = np.random.default_rng(7)
    t = np.sort(rng.uniform(2000.0, 2008.0, 200))
    y = 0.1 * (t - 2000) + np.cos(2 * np.pi * t) + 0.5 * np.sin(2 * np.pi * 2.4 * t) + rng.normal(0, 0.3, t.size)
    for sig in ("montecarlo", "sidak"):
        f = tspec.allssa(t, y, significance=sig).frequencies  # type: ignore[arg-type]
        assert np.any(np.abs(f - 1.0) < 0.1) and np.any(np.abs(f - 2.4) < 0.1)


def test_lssa_peak_fap_reported() -> None:
    rng = np.random.default_rng(11)
    t = np.sort(rng.uniform(2000, 2010, 240))
    r = tspec.lssa(t, rng.normal(size=240), omega=np.arange(0.1, 12, 0.05), alpha=0.05)
    assert r.peak_fap > 0.05  # pure noise: highest peak is not family-wise significant
    assert "not evidence of cycles" in r.explain()


def test_decompose_bias_corrected_trend_band_covers_curved_trend() -> None:
    """BLK-03: model band covered ~75% of a 9-yr-cycle trend; bias-corrected ~95%."""
    cov_bc, cov_model = [], []
    for r in range(40):
        rng = np.random.default_rng(4400 + r)
        t = np.sort(rng.uniform(2000, 2006.25, 150))
        truth = 2 + 0.6 * np.sin(2 * np.pi * (t - 2000) / 9.0)
        y = truth + np.cos(2 * np.pi * t) + 0.4 * np.sin(2 * np.pi * t) + rng.normal(0, 0.3, t.size)
        d = tspec.just_decompose(t, y, season="ols")
        assert d.trend_ci_method == "bias_corrected" and d.trend_bc is not None
        cov_bc.append(np.mean((truth >= d.trend_ci_low) & (truth <= d.trend_ci_high)))
        m = tspec.just_decompose(t, y, season="ols", trend_ci_method="model")
        cov_model.append(np.mean((truth >= m.trend_ci_low) & (truth <= m.trend_ci_high)))
    assert np.mean(cov_bc) >= 0.92
    assert np.mean(cov_model) < 0.85


def test_censored_ols_segment_slope_is_unbiased_and_covers() -> None:
    """BLK-04: marginal single imputation gave -27% bias and 63% coverage at 50% censoring."""
    beta = 0.1
    errs, cover = [], []
    for r in range(25):
        rng = np.random.default_rng(10_050 + r)
        t = np.sort(rng.uniform(2000, 2006.25, 150))
        y = 2 + beta * (t - 2000) + rng.normal(0, 0.4, t.size)
        dl = 2.31  # ~50% censoring
        ys = np.array([f"<{dl}" if v < dl else float(v) for v in y], dtype=object)
        a = tspec.analyze_series(t, ys, detect_jumps=False, detect_turning_points=False,
                                 remove_seasonality=False, slope_estimator="ols")
        seg = a.segments[0]
        assert seg.estimator == "ols_mi"
        assert a.censoring is not None and a.censoring.ndimpute_attrs.get("imputation_model") == "trend"
        errs.append(seg.slope - beta)
        cover.append(seg.slope_ci_low <= beta <= seg.slope_ci_high)
    assert abs(np.mean(errs)) < 0.012
    assert np.mean(cover) >= 0.84


def test_mk_corrections() -> None:
    """MAJ-01: AR(1) correction keeps MK near nominal; Hamed-Rao matches pymannkendall."""
    def ar1(rng: np.random.Generator, n: int, phi: float) -> np.ndarray:
        e = rng.normal(size=n)
        x = np.empty(n)
        x[0] = e[0] / np.sqrt(1 - phi**2)
        for i in range(1, n):
            x[i] = phi * x[i - 1] + e[i]
        return x

    rej_none = rej_ar1 = 0
    R = 200
    for r in range(R):
        rng = np.random.default_rng(900 + r)
        t = np.arange(150.0)
        x = ar1(rng, 150, 0.5)
        rej_none += fit_theil_sen_segment(t, x).pvalue < 0.05
        rej_ar1 += fit_theil_sen_segment(t, x, autocorr_correction="ar1").pvalue < 0.05
    assert rej_none / R > 0.15
    assert rej_ar1 / R <= 0.10
    rng = np.random.default_rng(3)
    x = ar1(rng, 80, 0.5)
    t = np.arange(80.0)
    assert ar1_rank_factor(t, x, 0.0) > 1.0
    mk = pytest.importorskip("pymannkendall")
    ref = mk.hamed_rao_modification_test(x)
    from tspec._robust import _kendall_var_s_ties

    slope = fit_theil_sen_segment(t, x).slope
    assert np.isclose(_kendall_var_s_ties(x, t) * hamed_rao_factor(t, x, slope), ref.var_s, rtol=1e-10)


def test_sen_ci_matches_scipy() -> None:
    """MIN-01: Sen CI order statistics follow Gilbert (1987) / scipy."""
    from scipy.stats import theilslopes

    rng = np.random.default_rng(5)
    t = np.arange(80.0)
    y = np.round(0.02 * t + rng.normal(0, 1, 80), 1)
    f = fit_theil_sen_segment(t, y)
    ref = theilslopes(y, t, alpha=0.95)
    assert np.isclose(f.ci_low, ref.low_slope) and np.isclose(f.ci_high, ref.high_slope)


def test_time_inputs_numeric_strings_and_range() -> None:
    """MIN-02: numeric strings are numbers; out-of-range datetimes are rejected."""
    from tspec._time import coerce_times

    np.testing.assert_allclose(coerce_times(np.array(["1", "2", "3"]))[0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        coerce_times(np.array(["1500-01-01", "1500-02-01"], dtype="datetime64[D]"))


def test_time_unit_warning_for_days() -> None:
    """MAJ-04: numeric time in days triggers an explicit unit warning."""
    rng = np.random.default_rng(2)
    td = np.sort(rng.uniform(0, 3650, 300))
    yd = np.sin(2 * np.pi * td / 365.25) + 0.3 * rng.normal(size=300)
    codes = {w.code for w in tspec.allssa(td, yd).diagnostics.warnings}
    assert "TIME_UNIT_ASSUMPTION" in codes
    with pytest.raises(tspec.InsufficientDataError, match="decimal years"):
        tspec.just_jumps(td, yd, season="ols")
