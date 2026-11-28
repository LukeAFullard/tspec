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


def test_censored_kendall_variance_matches_permutation_and_uncensored_formula() -> None:
    """MIN-09: ATS uses the censoring-aware permutation variance (Daniels, 1944)."""
    from tspec._robust import _censored_kendall_score, _censored_kendall_var

    rng = np.random.default_rng(1)
    n = 40
    t = np.sort(rng.uniform(0, 10, n))
    y = rng.normal(size=n)
    no_cen = np.zeros(n, dtype=np.intp)
    assert np.isclose(_censored_kendall_var(t, y, no_cen, 0.0), n * (n - 1) * (2 * n + 5) / 18.0)
    dl = np.quantile(y, 0.4)
    codes = np.where(y < dl, -1, 0).astype(np.intp)
    vals = np.where(y < dl, dl, y)
    perm = [
        _censored_kendall_score(t, vals[p], codes[p], 0.0)
        for p in (rng.permutation(n) for _ in range(4000))
    ]
    v = _censored_kendall_var(t, vals, codes, 0.0)
    assert abs(np.var(perm) / v - 1.0) < 0.08
    assert v < n * (n - 1) * (2 * n + 5) / 18.0


def test_monitor_guardrails_ignore_future_censoring() -> None:
    """MIN-10: heavy censoring after the monitoring start must not block the analysis."""
    rng = np.random.default_rng(4)
    n = 200
    t = np.sort(rng.uniform(2000, 2008, n))
    ys = np.array(["<0.8"] * n, dtype=object)
    det = rng.choice(110, 30, replace=False)  # 30 detects in the 110-point history (73% censored)
    ys[det] = list(np.round(rng.uniform(0.9, 2.0, 30), 3))
    # Overall 170/200 = 85% censored (> max_censor_fraction=0.8) because the future is censored
    res = tspec.just_monitor(t, ys, start=110, season="none", horizon=90)
    assert res.censoring is not None and res.censoring.censor_fraction > 0.8
    import pytest as _pytest

    from tspec import CensoringError

    with _pytest.raises(CensoringError):  # the same data still trip the guardrail outside monitoring
        tspec.just_jumps(t, ys, season="none")


def test_ats_unavailable_warning_for_interval_censoring() -> None:
    """MIN-03: silent ATS fallback is now reported."""
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(2000, 2006, 120))
    y = 2 + 0.2 * (t - 2000) + rng.normal(0, 0.3, t.size)
    lo = np.where(y < 2.5, 0.0, y)
    hi = np.where(y < 2.5, 2.5, y)
    from tspec import CensoredConfig

    res = tspec.analyze_series(
        t, np.column_stack([lo, hi]), detect_jumps=False, detect_turning_points=False,
        remove_seasonality=False, slope_estimator="sen",
        censoring=CensoredConfig(method="parametric", censoring_type="interval"),
    )
    assert "ATS_UNAVAILABLE" in {w.code for w in res.diagnostics.warnings}


# ---------------------------------------------------------------------------
# CRT-04: STPD turning-point p-values calibrated for the knot search
# ---------------------------------------------------------------------------


def _tp_series(seed: int, n: int, sigma: float, bend: bool) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    span = n / 24.0
    ttp = 2000 + span / 2
    t = np.sort(rng.uniform(2000, 2000 + span, n))
    trend = 3 + 0.5 * (t - 2000) - (1.0 * np.maximum(0, t - ttp) if bend else 0.3 * (t - 2000))
    return t, trend + rng.normal(0, sigma, n), ttp


def test_stpd_calibrated_defaults_and_parity() -> None:
    t, y, _ = _tp_series(0, 150, 0.3, bend=False)
    res = tspec.stpd(t, y)
    assert res.tp_calibration == "montecarlo"
    assert res.snr_th == 0.0
    par = tspec.stpd(t, y, parity_mode=True)
    assert par.tp_calibration == "none"
    assert par.snr_th == 1.0
    rob = tspec.stpd(t, y, robust="huber")
    assert rob.tp_calibration == "bonferroni"
    with pytest.raises(ValueError):
        tspec.stpd(t, y, tp_calibration="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        tspec.stpd(t, y, n_mc=10)


def test_stpd_mc_pvalue_exceeds_raw_screening_pvalue() -> None:
    from tspec.stpd import _stpd_mc_pvalue, _tpd_window

    t, y, ttp = _tp_series(1, 120, 0.6, bend=True)
    raw = _tpd_window(t, y, 5).pval
    cal = _stpd_mc_pvalue(t, y, 5, np.ones_like(t), 499, 0)
    assert cal >= raw
    t0, y0, _ = _tp_series(2, 120, 0.6, bend=False)
    assert _stpd_mc_pvalue(t0, y0, 5, np.ones_like(t0), 499, 0) >= _tpd_window(t0, y0, 5).pval


def test_stpd_calibrated_null_rate_and_power() -> None:
    fp = 0
    for seed in range(40):
        t, y, _ = _tp_series(100 + seed, 150, 0.3, bend=False)
        fp += len(tspec.stpd(t, y).turning_points) > 0
    assert fp <= 2  # nominal 1% family-wise; Monte Carlo design gave 0.6%
    # The previous SNR > 1 gate missed this clear turning point; the calibrated test finds it.
    hits = hits_iid = legacy = 0
    for seed in range(10):
        t, y, ttp = _tp_series(200 + seed, 500, 0.6, bend=True)
        hits += any(abs(float(x) - ttp) < 0.5 for x in tspec.stpd(t, y).tp_times)
        # Same comparison as originally designed (independent-error calibration vs SNR gate);
        # the default AR(1) calibration (AUD-02) may give up a borderline detection.
        hits_iid += any(
            abs(float(x) - ttp) < 0.5
            for x in tspec.stpd(t, y, serial_correlation="none").tp_times
        )
        legacy += any(
            abs(float(x) - ttp) < 0.5
            for x in tspec.stpd(t, y, tp_calibration="none", snr_th=1.0).tp_times
        )
    assert hits >= 6
    assert hits_iid > legacy


# ---------------------------------------------------------------------------
# CRT-01: consecutive_k calibrated over a planned horizon
# ---------------------------------------------------------------------------


def _stable_monitor_series(seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(2000, 2000 + n / 24.0, n))
    mu = 2 + 0.1 * (t - 2000) + 0.5 * np.cos(2 * np.pi * t)
    return t, mu + rng.normal(0, 0.15, n)


def test_consecutive_k_calibrated_with_horizon() -> None:
    t, y = _stable_monitor_series(0, 96 + 24)
    res = tspec.just_monitor(
        t, y, start=96, season="ols", alpha=0.05, sequential_control="consecutive_k", horizon=24
    )
    assert res.consecutive_k_calibrated
    levels = {s.alpha_step for s in res.steps}
    assert len(levels) == 1
    pocock = tspec.just_monitor(t, y, start=96, season="ols", alpha=0.05, horizon=24)
    assert pocock.steps[0].alpha_step < levels.pop() < 1.0
    assert "calibrated" in res.explain()
    # Same history and horizon -> same level regardless of how much monitoring data exists
    short = tspec.just_monitor(
        t[:110], y[:110], start=96, season="ols", alpha=0.05,
        sequential_control="consecutive_k", horizon=24,
    )
    assert short.steps[0].alpha_step == res.steps[0].alpha_step


def test_consecutive_k_without_horizon_warns() -> None:
    t, y = _stable_monitor_series(1, 120)
    res = tspec.just_monitor(t, y, start=96, season="ols", sequential_control="consecutive_k")
    assert not res.consecutive_k_calibrated
    assert "UNCALIBRATED_CONSECUTIVE_K" in {w.code for w in res.diagnostics.warnings}


def test_consecutive_k_calibrated_null_rate() -> None:
    alarms = 0
    for seed in range(20):
        t, y = _stable_monitor_series(500 + seed, 96 + 48)
        alarms += tspec.just_monitor(
            t, y, start=96, season="ols", alpha=0.05,
            sequential_control="consecutive_k", horizon=48, n_mc=199,
        ).detected
    assert alarms <= 4  # nominal 5% familywise (about 1 alarm expected)


# ---------------------------------------------------------------------------
# Censored detection: candidate p-values pooled over multiple imputations
# ---------------------------------------------------------------------------


def test_pool_pvalues_properties() -> None:
    from tspec._cenreg import pool_pvalues

    assert pool_pvalues(np.array([0.03])) == pytest.approx(0.03)
    # Identical p-values (no between-imputation variance) reproduce the common value
    assert pool_pvalues(np.full(10, 0.02)) == pytest.approx(0.02, rel=1e-6)
    # Disagreement between imputations makes the pooled evidence weaker
    assert pool_pvalues(np.array([0.001, 0.2, 0.01, 0.5])) > 0.01
    assert np.isnan(pool_pvalues(np.array([np.nan])))


def _censored_jump_series(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 150
    t = np.sort(rng.uniform(2000, 2000 + n / 24, n))
    y = 2 + 0.1 * (t - 2000) + 1.0 * (t >= 2000 + n / 48) + rng.normal(0, 0.4, n)
    dl = float(np.quantile(y, 0.4))
    return t, np.array([f"<{dl:.6g}" if v < dl else float(v) for v in y], dtype=object)


def _censored_bend_series(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 150
    t = np.sort(rng.uniform(2000, 2000 + n / 24, n))
    y = 2 + 0.6 * (t - 2000) - 1.2 * np.maximum(t - (2000 + n / 48), 0) + rng.normal(0, 0.15, n)
    dl = float(np.quantile(y, 0.4))
    return t, np.array([f"<{dl:.6g}" if v < dl else float(v) for v in y], dtype=object)


def test_detection_pvalues_pooled_over_imputations() -> None:
    t, ys = _censored_jump_series(11)
    # Pooling mechanics under the published level-or-slope definition (a 2.5-sigma step in a
    # 40%-censored record is below the power of the default step test in 3-cycle windows).
    lv = "level_or_slope"
    assert tspec.just_jumps(t, ys, season="none", jump_definition=lv).n_imputations == 1
    mi = tspec.CensoredConfig(detection_imputations=20)
    res = tspec.just_jumps(t, ys, season="none", censoring=mi, jump_definition=lv)
    assert res.n_imputations == 20
    assert any(abs(j.time_year - (2000 + 150 / 48)) < 0.5 for j in res.jumps)
    assert "pooled over 20 imputations" in res.explain()
    unc = tspec.just_jumps(
        t, np.array([float(v) if not isinstance(v, str) else 0.0 for v in ys]), season="none", censoring=mi,
        jump_definition=lv,
    )
    assert unc.n_imputations == 1
    # STPD pools its turning-point p-values on a censored *bend* (a step is now correctly
    # rejected as a turning point by the step-vs-bend test, AUD-03, so nothing is pooled).
    tb, ysb = _censored_bend_series(11)
    st_mi = tspec.stpd(tb, ysb, censoring=mi)
    assert st_mi.n_imputations == 20
    assert any(abs(tp.time_year - (2000 + 150 / 48)) < 0.5 for tp in st_mi.turning_points)
    assert tspec.stpd(t, ys, censoring=mi).turning_points == ()
    with pytest.raises(ValueError):
        tspec.CensoredConfig(detection_imputations=0)


def test_pooled_decisions_do_not_depend_on_imputation_seed() -> None:
    flips = 0
    for seed in range(8):
        t, ys = _censored_jump_series(40 + seed)
        a = tspec.just_jumps(
            t, ys, season="none", censoring=tspec.CensoredConfig(random_state=0, detection_imputations=20)
        )
        b = tspec.just_jumps(
            t, ys, season="none", censoring=tspec.CensoredConfig(random_state=1, detection_imputations=20)
        )
        flips += (len(a.jumps) > 0) != (len(b.jumps) > 0)
    assert flips <= 1


# ---------------------------------------------------------------------------
# CRT-06: short-series Mann-Kendall/Sen under AR(1) uses a parametric bootstrap reference
# ---------------------------------------------------------------------------


def _ar1_path(rng: np.random.Generator, n: int, phi: float) -> np.ndarray:
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0] / np.sqrt(1 - phi**2)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def test_short_series_mk_ar1_calibrated() -> None:
    rng = np.random.default_rng(606)
    n, phi, R = 60, 0.7, 300
    t = 2000 + np.arange(n) / 12.0
    rej = 0
    for _ in range(R):
        rej += fit_theil_sen_segment(t, _ar1_path(rng, n, phi), autocorr_correction="ar1").pvalue < 0.05
    # analytic plug-in factor: ~8.6% (audit CRT-06); bootstrap reference ~5-6%
    assert rej / R <= 0.075


def test_mk_bootstrap_reference_is_deterministic() -> None:
    rng = np.random.default_rng(1)
    t = 2000 + np.arange(80) / 12.0
    y = _ar1_path(rng, 80, 0.5)
    a = fit_theil_sen_segment(t, y, autocorr_correction="ar1")
    b = fit_theil_sen_segment(t, y, autocorr_correction="ar1")
    assert a.pvalue == b.pvalue and a.ci_low == b.ci_low and a.ci_high == b.ci_high


# ---------------------------------------------------------------------------
# CRT-04: detect-then-fit bootstrap bands
# ---------------------------------------------------------------------------


def test_stpd_bootstrap_band_covers_after_detection() -> None:
    hits_b = hits_c = n_ok = 0.0
    for seed in range(12):
        rng = np.random.default_rng(4040 + seed)
        n = 500
        span = n / 24
        ttp = 2000 + span / 2
        t = np.sort(rng.uniform(2000, 2000 + span, n))
        truth = 3 + 0.5 * (t - 2000) - np.maximum(0, t - ttp)
        y = truth + rng.normal(0, 0.6, n)
        sb = tspec.stpd(t, y)
        if len(sb.tp_times) != 1 or abs(float(sb.tp_times[0]) - ttp) > 0.5:
            continue
        assert sb.band_method == "bootstrap"
        sc = tspec.stpd(t, y, band_method="conditional")
        pb, pc = sb.piecewise_trend, sc.piecewise_trend
        hits_b += float(np.mean((truth >= pb.trend_ci_low) & (truth <= pb.trend_ci_high)))
        hits_c += float(np.mean((truth >= pc.trend_ci_low) & (truth <= pc.trend_ci_high)))
        n_ok += 1
    assert n_ok >= 6
    assert hits_b / n_ok >= 0.90  # conditional band ~85% in this design
    assert hits_b > hits_c


def test_band_method_options_and_parity() -> None:
    t, y, _ = _tp_series(5, 150, 0.3, bend=True)
    assert tspec.stpd(t, y, parity_mode=True).band_method == "conditional"
    with pytest.raises(ValueError):
        tspec.stpd(t, y, band_method="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        tspec.stpd(t, y, n_boot=10)
    rng = np.random.default_rng(9)
    tt = np.sort(rng.uniform(2000, 2012, 288))
    yy = 3 + 0.4 * (tt - 2000) - 0.7 * np.maximum(0, tt - 2006) + 0.6 * np.cos(2 * np.pi * tt)
    yy = yy + rng.normal(0, 0.2, tt.size)
    a = tspec.analyze_series(tt, yy)
    assert a.band_method == "bootstrap"
    c = tspec.analyze_series(tt, yy, band_method="conditional")
    assert c.band_method == "conditional"
    assert np.mean(a.trend_ci_high - a.trend_ci_low) > np.mean(c.trend_ci_high - c.trend_ci_low)
    assert "bootstrap intervals" in a.explain()
