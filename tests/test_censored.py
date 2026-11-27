"""Tests for ndimpute censored-data integration and statistical safeguards."""

from __future__ import annotations

import numpy as np
import pytest

from tspec import (
    CensoredConfig,
    CensoringError,
    detect_turning_points,
    just_jumps,
)
from tspec.censored import prepare_series_with_censoring

pytest.importorskip("ndimpute")


def test_quantile_tie_shuffling_prevents_spurious_temporal_trend() -> None:
    """Regression test for ndimpute stable-sort temporal trend injection on tied <DL."""
    slopes_tspec: list[float] = []
    slopes_raw_unshuffled: list[float] = []
    import ndimpute

    for seed in range(25):
        rng = np.random.default_rng(1000 + seed)
        t = np.linspace(0.0, 5.0, 120)
        y_true = rng.lognormal(mean=0.0, sigma=0.8, size=120)
        dl = float(np.quantile(y_true, 0.45))
        is_cen = y_true < dl
        y_obs = np.where(is_cen, dl, y_true)

        # Unshuffled ndimpute ROS quantile injects deterministic positive slope!
        df_raw = ndimpute.impute(
            y_obs, status=is_cen, method="ros", dist="lognormal", impute_type="quantile"
        )
        y_raw = df_raw["imputed_value"].to_numpy()
        slopes_raw_unshuffled.append(float(np.polyfit(t, y_raw, 1)[0]))

        # tspec tie-shuffled quantile eliminates temporal rank bias (exercises ndimpute marginal path)
        prep = prepare_series_with_censoring(
            t,
            y_obs,
            censor_status=is_cen,
            censoring=CensoredConfig(
                method="ros",
                imputation_model="marginal",
                dist="lognormal",
                impute_type="quantile",
                random_state=seed,
            ),
        )
        assert prep.provenance is not None and prep.provenance.method == "ros"
        slopes_tspec.append(float(np.polyfit(t, prep.series.values, 1)[0]))

    mean_raw = float(np.mean(slopes_raw_unshuffled))
    mean_tspec = float(np.mean(slopes_tspec))
    se_tspec = float(np.std(slopes_tspec, ddof=1) / np.sqrt(len(slopes_tspec)))
    assert mean_raw > 0.02  # Confirm raw unshuffled artifact exists
    assert abs(mean_tspec) <= 2.5 * se_tspec  # Zero temporal bias in tspec


def test_string_markers_left_right_mixed_and_interval_inputs(rng: np.random.Generator) -> None:
    t = np.linspace(0.0, 5.0, 60)
    y_raw = rng.lognormal(1.0, 0.5, size=60)
    # Mixed string input with NaN
    y_str: list[object] = [
        "<1.2" if v < 1.2 else (">8.0" if v > 8.0 else float(v)) for v in y_raw
    ]
    y_str[10] = None
    prep = prepare_series_with_censoring(
        t,
        y_str,
        censoring=CensoredConfig(method="parametric", dist="lognormal", random_state=42),
    )
    assert prep.provenance is not None
    assert prep.series.n_dropped_nan == 1
    assert prep.provenance.censoring_type in ("left", "right", "mixed")
    assert np.all(np.isfinite(prep.series.values))

    # Interval (N, 2) input
    lo = np.where(y_raw < 1.5, 0.5, y_raw)
    hi = np.where(y_raw < 1.5, 1.5, y_raw)
    bounds = np.column_stack([lo, hi])
    prep_int = prepare_series_with_censoring(
        t,
        bounds,
        censoring=CensoredConfig(
            method="parametric", censoring_type="interval", dist="lognormal", random_state=42
        ),
    )
    assert prep_int.provenance is not None
    assert prep_int.provenance.censoring_type == "interval"
    assert np.all(prep_int.series.values >= lo - 1e-12)
    assert np.all(prep_int.series.values <= hi + 1e-12)


def test_shifting_detection_limit_triggers_coincidence_warning_and_recensor_check(
    rng: np.random.Generator,
) -> None:
    # Stationary lognormal data with NO true environmental jump, but lab DL drops from 2.0 to 0.2!
    t = np.linspace(2014.0, 2022.0, 160)
    y_true = rng.lognormal(mean=0.2, sigma=0.7, size=t.size)
    dls = np.where(t < 2018.0, 2.0, 0.2)
    is_cen = y_true < dls
    y_obs = np.where(is_cen, dls, y_true)

    prep = prepare_series_with_censoring(
        t,
        y_obs,
        censor_status=is_cen,
        censoring=CensoredConfig(
            method="ros",
            dist="lognormal",
            impute_type="mean",
            detect_dl_shifts=True,
            recensor_check=True,
            random_state=0,
        ),
    )
    assert prep.provenance is not None
    assert len(prep.provenance.dl_shifts) >= 1
    assert any(abs(s.time_year - 2018.0) < 0.6 for s in prep.provenance.dl_shifts)

    # Strong lab DL-upgrade artifact (substitution at DL creates a large artificial drop at 2018.0)
    y_low = rng.lognormal(mean=-0.3, sigma=0.35, size=t.size)
    dls2 = np.where(t < 2018.0, 2.0, 0.2)
    is_cen2 = y_low < dls2
    y_obs2 = np.where(is_cen2, dls2, y_low)
    res = just_jumps(
        t,
        y_obs2,
        season="none",
        mag_th=0.2,
        dir_th=0.2,
        jump_alpha=0.05,
        bonferroni_window=True,
        censor_status=is_cen2,
        censoring=CensoredConfig(
            method="substitution",
            ndimpute_kwargs={"strategy": "value"},
            detect_dl_shifts=True,
            recensor_check=True,
            allow_heavy_censoring=True,
            random_state=0,
        ),
    )
    assert res.censoring is not None
    assert len(res.censoring.dl_shifts) >= 1
    assert len(res.jumps) >= 1
    assert any(j.coincides_with_dl_shift for j in res.jumps)
    codes = {w.code for w in res.diagnostics.warnings}
    assert "DL_SHIFT_COINCIDENCE" in codes


def test_censoring_guardrails_block_extreme_and_100pct_censoring() -> None:
    t = np.linspace(0.0, 5.0, 50)
    # 100% censored -> always raises CensoringError
    y_all = ["<0.1"] * 50
    with pytest.raises(CensoringError, match="Too few uncensored"):
        prepare_series_with_censoring(t, y_all)

    # 86% censored -> blocked unless allow_heavy_censoring=True
    y_heavy: list[object] = ["<0.1"] * 43 + [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    with pytest.raises(CensoringError, match="exceeds max_censor_fraction"):
        prepare_series_with_censoring(t, y_heavy)

    # Allowed with explicit opt-in and emits HEAVY_CENSORING warning
    prep_ok = prepare_series_with_censoring(
        t, y_heavy, censoring=CensoredConfig(allow_heavy_censoring=True, random_state=0)
    )
    assert any(w.code == "HEAVY_CENSORING" for w in prep_ok.warnings)


def test_windowed_scope_and_stability_draws_in_just_and_stpd(rng: np.random.Generator) -> None:
    t = np.linspace(0.0, 6.0, 120)
    # Jump from low regime to high regime
    y_true = np.exp(np.where(t < 3.0, 0.2, 1.5) + rng.normal(0.0, 0.25, size=t.size))
    dl = 1.1
    is_cen = y_true < dl
    y_obs = np.where(is_cen, dl, y_true)

    res_j = just_jumps(
        t,
        y_obs,
        season="none",
        mag_th=0.5,
        dir_th=0.2,
        jump_alpha=0.01,
        censor_status=is_cen,
        censoring=CensoredConfig(
            scope="windowed",
            n_stability_draws=6,
            stability_mode="bootstrap",
            random_state=0,
        ),
    )
    assert len(res_j.jumps) == 1
    assert res_j.jumps[0].stability_rate is not None
    assert res_j.jumps[0].stability_rate >= 0.5

    # STPD with censored turning point
    y_tp_true = np.exp(
        np.where(t <= 3.0, 0.3 + 0.1 * t, 0.6 + 0.9 * (t - 3.0)) + rng.normal(0.0, 0.12, size=t.size)
    )
    dl_tp = float(np.quantile(y_tp_true, 0.20))
    is_cen_tp = y_tp_true < dl_tp
    y_tp_obs = np.where(is_cen_tp, dl_tp, y_tp_true)
    res_s = detect_turning_points(
        t,
        y_tp_obs,
        size=80,
        step=15,
        margin=10,
        censor_status=is_cen_tp,
        censoring=CensoredConfig(n_stability_draws=5, random_state=0),
    )
    assert res_s.censoring is not None
    assert len(res_s.turning_points) >= 1


def test_tobit_scale_invariance_extreme_tails_and_psd_cov() -> None:
    """CRT-05: fit_censored_regression is scale-invariant, tail-safe, and strictly PSD."""
    import warnings

    from tspec._cenreg import (
        _interval_mills_terms,
        _nll_grad,
        fit_censored_regression,
        impute_censored_regression,
        trend_design,
    )

    rng = np.random.default_rng(801)
    n = 80
    t = np.linspace(0.0, 5.0, n)
    X = trend_design(t)
    y_lat = 2.0 + 0.4 * t + rng.normal(0.0, 0.3, size=n)
    dl = float(np.quantile(y_lat, 0.35))
    codes = np.where(y_lat < dl, -1, 0).astype(np.intp)
    vals = np.where(codes == -1, dl, y_lat)

    fit1 = fit_censored_regression(vals, codes, X, dist="normal")
    fit_lo = fit_censored_regression(vals * 1e-8, codes, X, dist="normal")
    fit_hi = fit_censored_regression(vals * 1e8, codes, X, dist="normal")
    assert fit1.converged and fit_lo.converged and fit_hi.converged
    np.testing.assert_allclose(fit_lo.beta * 1e8, fit1.beta, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(fit_hi.beta * 1e-8, fit1.beta, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.exp(fit_lo.log_sigma) * 1e8, np.exp(fit1.log_sigma), rtol=1e-5)
    np.testing.assert_allclose(np.exp(fit_hi.log_sigma) * 1e-8, np.exp(fit1.log_sigma), rtol=1e-5)

    # Strictly SPD correlation matrix & zero RuntimeWarning on draw_params=True at all scales
    for fit, scale_val in ((fit1, 1.0), (fit_lo, 1e-8), (fit_hi, 1e8)):
        d_sd = np.sqrt(np.diag(fit.cov))
        assert np.all(d_sd > 0.0)
        corr = fit.cov / np.outer(d_sd, d_sd)
        evals = np.linalg.eigvalsh(0.5 * (corr + corr.T))
        assert float(np.min(evals)) > 0.0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            imp_draw = impute_censored_regression(
                fit, vals * scale_val, codes, X, rng, draw_params=True
            )
        assert np.all(np.isfinite(imp_draw))
        assert not any("positive-semidefinite" in str(w.message) for w in caught)

    # Extreme tail Mills ratio accuracy at b = -40 (previously underflowed to 0)
    a_tail = np.array([-np.inf, 40.0])
    b_tail = np.array([-40.0, np.inf])
    r1, r2 = _interval_mills_terms(a_tail, b_tail)
    expected_mills = 40.0 + 1.0 / 40.0 - 2.0 / (40.0**3)
    np.testing.assert_allclose(r1[0], expected_mills, rtol=1e-6)
    np.testing.assert_allclose(r1[1], -expected_mills, rtol=1e-6)
    np.testing.assert_allclose(r2[0], -40.0 * expected_mills, rtol=1e-6)
    np.testing.assert_allclose(r2[1], -40.0 * expected_mills, rtol=1e-6)

    # Analytic gradient vs finite-difference check with deep-tail censored point
    lo_t = np.array([1.0, 2.0, -np.inf])
    hi_t = np.array([1.0, 2.0, -38.0])
    X_t = np.array([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]])
    p_t = np.array([1.5, 0.0, 0.0])
    nll0, g0 = _nll_grad(p_t, X_t, lo_t, hi_t, np.ones(3), np.array([True, True, False]))
    assert np.isfinite(nll0) and np.all(np.isfinite(g0))
    eps = 1e-6
    g_fd = np.empty_like(g0)
    for k in range(p_t.size):
        dp = np.zeros_like(p_t)
        dp[k] = eps
        g_fd[k] = (
            _nll_grad(p_t + dp, X_t, lo_t, hi_t, np.ones(3), np.array([True, True, False]))[0]
            - _nll_grad(p_t - dp, X_t, lo_t, hi_t, np.ones(3), np.array([True, True, False]))[0]
        ) / (2.0 * eps)
    np.testing.assert_allclose(g0, g_fd, rtol=1e-5, atol=1e-5)


def test_pool_pvalues_high_precision_upper_clip() -> None:
    """MAJ-06: pool_pvalues handles p=1.0 cleanly without anti-conservative 0.999 bias."""
    from tspec._cenreg import pool_pvalues

    p_pooled = pool_pvalues(np.array([0.04, 1.0, 1.0, 1.0]))
    assert np.isfinite(p_pooled) and p_pooled > 0.5
    # Verify monotonicity when replacing 0.999 with 1.0
    p_999 = pool_pvalues(np.array([0.04, 0.999]))
    p_100 = pool_pvalues(np.array([0.04, 1.0]))
    assert p_100 >= p_999


def test_resolve_trend_dist_interval_zero_and_negative_right() -> None:
    """CRT-10: [0, u] positive intervals resolve to lognormal; negative right-limits resolve to normal."""
    from tspec.censored import _resolve_trend_dist

    # [0.0, 0.8] interval bounds on positive data -> lognormal
    vals_iv = np.array([0.4, 1.2, 2.5, 0.4])
    codes_iv = np.array([2, 0, 0, 2], dtype=np.intp)
    lo_iv = np.array([0.0, 1.2, 2.5, 0.0])
    hi_iv = np.array([0.8, 1.2, 2.5, 0.8])
    assert _resolve_trend_dist(vals_iv, codes_iv, lo_iv, hi_iv, "auto") == "lognormal"

    # Negative right-censored limit (>-1.0) with positive observes -> MUST be normal
    vals_r = np.array([-1.0, 0.5, 1.2, 2.0])
    codes_r = np.array([1, 0, 0, 0], dtype=np.intp)
    assert _resolve_trend_dist(vals_r, codes_r, None, None, "auto") == "normal"

    # Non-positive interval upper bound -> normal
    lo_neg = np.array([-2.0, 1.0, 2.0])
    hi_neg = np.array([-0.5, 1.0, 2.0])
    assert _resolve_trend_dist(np.array([-1.25, 1.0, 2.0]), np.array([2, 0, 0]), lo_neg, hi_neg, "auto") == "normal"


def test_lognormal_ols_mi_respects_lognormal_dist() -> None:
    """BLK-03: analyze_series OLS-MI segment slope must honor dist='lognormal'."""
    import tspec

    n = 120
    t = np.linspace(2014.0, 2024.0, n)
    t0 = t - t[0]
    ey = np.exp(-0.5 + 0.5 * 0.36 + 0.12 * t0)
    pop_slope = float(np.polyfit(t0, ey, 1)[0])

    biases: list[float] = []
    covered: list[bool] = []
    for r in range(25):
        rng = np.random.default_rng(9000 + r)
        y_lat = np.exp(-0.5 + 0.12 * t0 + rng.normal(0.0, 0.6, size=n))
        dl = float(np.quantile(y_lat, 0.50))
        is_c = y_lat < dl
        vals = np.where(is_c, dl, y_lat)
        cfg = CensoredConfig(dist="lognormal", n_imputations=20, n_stability_draws=0, random_state=r)
        res = tspec.analyze_series(
            t,
            vals,
            censor_status=is_c,
            censoring=cfg,
            detect_jumps=False,
            detect_turning_points=False,
            remove_seasonality=False,
            slope_estimator="ols",
        )
        seg = res.segments[0]
        assert seg.estimator == "ols_mi"
        biases.append((seg.slope - pop_slope) / pop_slope)
        assert seg.slope_ci_low is not None and seg.slope_ci_high is not None
        covered.append(seg.slope_ci_low <= pop_slope <= seg.slope_ci_high)

    # Pre-fix relative bias was +61% with 57% coverage; fixed should be near 0% and ~95%
    assert abs(float(np.mean(biases))) < 0.15
    assert float(np.mean(covered)) >= 0.80


def test_recensor_check_flags_pure_noise_dl_upgrade() -> None:
    """CRT-02: recensor_check must use calibrated p-value and flag pure-noise DL artifacts."""
    n = 120
    t = np.linspace(2014.0, 2024.0, n)
    t0 = t - t[0]
    rng = np.random.default_rng(11001)
    flagged = 0
    coinciding = 0
    for r in range(15):
        y_true = 4.0 + rng.normal(0.0, 0.35, size=n)
        dl = np.where(t0 < 5.0, 4.2, 0.5)
        is_c = y_true < dl
        vals = np.where(is_c, dl, y_true)
        cfg = CensoredConfig(
            method="substitution",
            imputation_model="marginal",
            dist="normal",
            detect_dl_shifts=True,
            recensor_check=True,
            allow_heavy_censoring=True,
            random_state=r,
        )
        res = just_jumps(t, vals, censor_status=is_c, censoring=cfg, season="none", n_mc=199)
        for j in res.jumps:
            if j.dl_artifact_likely is not None:
                coinciding += 1
                if j.dl_artifact_likely:
                    flagged += 1
    assert coinciding >= 2
    assert flagged / coinciding >= 0.70


def test_censoring_attribution_and_diagnostics() -> None:
    """MAJ-10, MAJ-08, CRT-07: accurate Tobit attribution, MI skip warning, and {0,1} int caution."""
    import tspec

    rng = np.random.default_rng(802)
    t = np.linspace(2015.0, 2021.0, 80)
    y_lat = np.exp(0.5 + 0.1 * (t - 2015.0) + rng.normal(0.0, 0.25, size=80))
    dl = float(np.quantile(y_lat, 0.15))
    is_c = y_lat < dl
    vals = np.where(is_c, dl, y_lat)

    # Default Tobit trend imputation must NOT attribute to ndimpute in summary/explain/warnings
    res = tspec.analyze_series(
        t, vals, censor_status=is_c, detect_jumps=False, detect_turning_points=False,
        remove_seasonality=False,
    )
    assert "Tobit MLE" in res.summary()
    assert "censored-regression MLE" in res.explain()
    applied = [w for w in res.diagnostics.warnings if w.code == "CENSORED_IMPUTATION_APPLIED"]
    assert applied and "censored-regression MLE" in applied[0].message

    # Numeric integer {0, 1} without censoring_type must emit INTEGER_STATUS_INTERPRETED_AS_RIGHT
    prep_int = prepare_series_with_censoring(t, vals, censor_status=is_c.astype(int))
    assert any(w.code == "INTEGER_STATUS_INTERPRETED_AS_RIGHT" for w in prep_int.warnings)

    # detection_imputations > 1 with scope='windowed' must emit MI_DETECTION_FALLBACK
    res_j = just_jumps(
        t, vals, censor_status=is_c, season="none",
        censoring=CensoredConfig(scope="windowed", detection_imputations=5, random_state=0),
    )
    assert any(w.code == "MI_DETECTION_FALLBACK" for w in res_j.diagnostics.warnings)

