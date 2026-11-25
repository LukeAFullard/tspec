"""Tests for Phase 1 & Phase 2 robust estimation across tspec."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

import tspec
from tspec._linalg import prepare_weights
from tspec._robust import (
    fit_l1_piecewise,
    fit_theil_sen_segment,
    huber_nested_ftest,
    solve_huber_irls,
)


class TestHuberIRLSCore:
    """Unit tests for Huber IRLS, invariance properties, and robust nested F-tests."""

    @pytest.mark.parametrize("w_kind", ["none", "scalar", "vector", "matrix"])
    def test_converges_to_wls_for_large_k(self, w_kind: str) -> None:
        rng = np.random.default_rng(101)
        n = 60
        t = np.linspace(0.0, 5.0, n)
        D = np.column_stack([np.ones(n), t, np.cos(2.0 * np.pi * t)])
        y = 2.0 - 0.7 * t + 1.2 * np.cos(2.0 * np.pi * t) + rng.normal(0.0, 0.25, n)

        if w_kind == "none":
            w = None
        elif w_kind == "scalar":
            w = 2.5
        elif w_kind == "vector":
            w = rng.uniform(0.5, 3.0, n)
        else:
            A = rng.normal(0.0, 0.2, (n, n))
            w = A @ A.T + 2.0 * np.eye(n)

        pw = prepare_weights(w, n)
        ols = tspec._linalg.solve_weighted_ols(D, y, pw, compute_cov=True)
        hr = solve_huber_irls(D, y, pw, k=1e6, compute_cov=True)

        np.testing.assert_allclose(hr.ols.coeff, ols.coeff, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(hr.huber_weights, np.ones(n), rtol=1e-12, atol=1e-12)
        assert hr.ols.cov is not None and ols.cov is not None
        np.testing.assert_allclose(hr.ols.cov, ols.cov, rtol=1e-2, atol=1e-6)

    def test_shift_and_scale_invariance(self) -> None:
        rng = np.random.default_rng(102)
        n = 80
        t = np.linspace(0.0, 8.0, n)
        D_null = np.column_stack([np.ones(n), t])
        D_full = np.column_stack([np.ones(n), t, np.maximum(0.0, t - 4.0)])
        y = 1.5 + 0.4 * t + 0.6 * np.maximum(0.0, t - 4.0) + rng.normal(0.0, 0.2, n)
        y[[10, 35, 62]] += np.array([8.0, -9.0, 7.5])
        pw = prepare_weights(None, n)

        f0, p0, fit0, _ = huber_nested_ftest(D_null, D_full, y, pw, k=1.345)
        f_shift, p_shift, fit_shift, _ = huber_nested_ftest(
            D_null, D_full, y + 1e9, pw, k=1.345
        )
        f_scale, p_scale, fit_scale, _ = huber_nested_ftest(
            D_null, D_full, y * 1e-8, pw, k=1.345
        )

        np.testing.assert_allclose(fit_shift.ols.coeff[1:], fit0.ols.coeff[1:], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(fit_scale.ols.coeff, fit0.ols.coeff * 1e-8, rtol=1e-6, atol=1e-14)
        np.testing.assert_allclose(f_shift, f0, rtol=1e-5)
        np.testing.assert_allclose(p_shift, p0, rtol=1e-5, atol=1e-12)
        np.testing.assert_allclose(f_scale, f0, rtol=1e-5)
        np.testing.assert_allclose(p_scale, p0, rtol=1e-5, atol=1e-12)

    def test_noiseless_exact_fit_and_outlier_rejection(self) -> None:
        n = 50
        t = np.linspace(0.0, 5.0, n)
        D = np.column_stack([np.ones(n), t])
        y_exact = 3.0 - 1.2 * t
        pw = prepare_weights(None, n)

        hr_exact = solve_huber_irls(D, y_exact, pw, k=1.345)
        np.testing.assert_allclose(hr_exact.ols.coeff, [3.0, -1.2], atol=1e-12)
        assert hr_exact.scale == 0.0

        rng = np.random.default_rng(103)
        y_spike = y_exact + rng.normal(0.0, 0.1, n)
        y_spike[[5, 18, 31, 44]] += np.array([15.0, -18.0, 20.0, -14.0])
        hr_rob = solve_huber_irls(D, y_spike, pw, k=1.345)
        ols_bad = tspec._linalg.solve_weighted_ols(D, y_spike, pw)

        assert abs(hr_rob.ols.coeff[1] - (-1.2)) < 0.08
        assert abs(hr_rob.ols.coeff[1] - (-1.2)) < 0.25 * abs(ols_bad.coeff[1] - (-1.2))

    def test_mm_biweight_invariance_and_spike_zeroing(self) -> None:
        from tspec._robust import solve_mm_biweight

        rng = np.random.default_rng(104)
        n = 80
        t = np.linspace(0.0, 8.0, n)
        D = np.column_stack([np.ones(n), t])
        y = 2.0 + 0.6 * t + rng.normal(0.0, 0.15, n)
        spikes = [10, 35, 62]
        y[spikes] += np.array([15.0, -18.0, 16.0])
        pw = prepare_weights(None, n)

        mm0 = solve_mm_biweight(D, y, pw)
        mm_shift = solve_mm_biweight(D, y + 1e8, pw)
        mm_scale = solve_mm_biweight(D, y * 1e-6, pw)

        # Extreme spikes receive exact zero Tukey weight
        assert float(np.max(mm0.huber_weights[spikes])) == 0.0
        assert abs(mm0.ols.coeff[1] - 0.6) < 0.03
        np.testing.assert_allclose(mm_shift.ols.coeff[1], mm0.ols.coeff[1], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(mm_scale.ols.coeff, mm0.ols.coeff * 1e-6, rtol=1e-5, atol=1e-12)

    def test_validation_errors(self) -> None:
        t = np.linspace(0.0, 3.0, 40)
        y = np.sin(2.0 * np.pi * t)
        with pytest.raises(ValueError, match="Unknown `robust`"):
            tspec.lssa(t, y, robust="biweight")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="huber_k"):
            tspec.lssa(t, y, robust="huber", huber_k=-1.0)
        with pytest.raises(ValueError, match="slope_estimator"):
            tspec.analyze_series(t, y, slope_estimator="median")  # type: ignore[arg-type]


class TestRobustSpectral:
    """LSSA and ALLSSA under spike contamination."""

    def test_lssa_allssa_huber_under_spikes(self) -> None:
        rng = np.random.default_rng(201)
        n = 140
        t = np.sort(rng.uniform(0.0, 5.0, n))
        y_clean = 1.5 * np.sin(2.0 * np.pi * 1.0 * t + 0.4) + 0.2 * t
        y = y_clean + rng.normal(0.0, 0.15, n)
        spike_idx = [15, 42, 77, 108, 125]
        y[spike_idx] += np.array([12.0, -14.0, 13.0, -11.0, 15.0])

        omega = np.linspace(0.2, 3.0, 60)
        res_ols = tspec.allssa(t, y, omega=omega, robust="none")
        res_rob = tspec.allssa(t, y, omega=omega, robust="huber")

        # Robust ALLSSA should detect 1.0 cpy and achieve far better fit to clean signal
        assert any(abs(f - 1.0) < 0.15 for f in res_rob.frequencies)
        mae_ols = float(np.mean(np.abs((y - res_ols.residuals) - y_clean)))
        mae_rob = float(np.mean(np.abs((y - res_rob.residuals) - y_clean)))
        assert mae_rob < 0.10
        assert mae_rob < 0.3 * mae_ols

        # lssa with robust='huber' isolates the 1.0 cpy peak cleanly
        lssa_rob = tspec.lssa(t, y, omega=omega, robust="huber")
        assert abs(float(omega[int(np.argmax(lssa_rob.spectrum))]) - 1.0) < 0.15

        # parity_mode forces non-robust path in allssa
        allssa_p0 = tspec.allssa(t, y, omega=omega, parity_mode=True, robust="none")
        allssa_p1 = tspec.allssa(t, y, omega=omega, parity_mode=True, robust="huber")
        np.testing.assert_array_equal(allssa_p0.residuals, allssa_p1.residuals)


class TestRobustJust:
    """JUST jump detection and decomposition under spikes and real jumps."""

    @pytest.mark.parametrize("robust_mode", ["huber", "mm"])
    def test_spike_immunity_and_no_jump_masking(self, robust_mode: str) -> None:
        rng = np.random.default_rng(301)
        t = np.linspace(0.0, 6.0, 180)
        seasonal = 0.8 * np.sin(2.0 * np.pi * t)
        base = 2.0 + 0.3 * t + seasonal + rng.normal(0.0, 0.18, t.size)

        # Case A: No true jump, 4 massive isolated spikes
        y_spikes = base.copy()
        y_spikes[[30, 75, 115, 150]] += np.array([9.0, -10.0, 11.0, -9.5])

        res_null_rob = tspec.just_jumps(
            t,
            y_spikes,
            size=40,
            step=20,
            season="allssa",
            jump_alpha=0.05,
            bonferroni_window=True,
            mag_th=0.0,
            dir_th=0.0,
            robust=robust_mode,  # type: ignore[arg-type]
        )
        assert len(res_null_rob.jumps) == 0
        assert "SHORT_SEASONAL_WINDOW" in {w.code for w in res_null_rob.diagnostics.warnings}

        # Case B: Genuine step jump of +3.5 at t=3.0 PLUS isolated spikes
        y_jump_spikes = y_spikes.copy()
        y_jump_spikes[t >= 3.0] += 3.5

        res_jump_rob = tspec.just_jumps(
            t,
            y_jump_spikes,
            size=40,
            step=20,
            season="allssa",
            jump_alpha=0.05,
            bonferroni_window=True,
            mag_th=0.0,
            dir_th=0.0,
            robust=robust_mode,  # type: ignore[arg-type]
            # 40-point windows span ~1.3 seasonal cycles: the annual harmonic is not
            # family-wise identifiable there, so keep the published per-frequency rule
            # (seasonality is known to be present in this design).
            season_significance="per_frequency",
        )
        assert len(res_jump_rob.jumps) >= 1
        closest = min(res_jump_rob.jumps, key=lambda j: abs(j.time_year - 3.0))
        assert abs(closest.time_year - 3.0) < 0.25
        assert abs(closest.magnitude - 3.5) < 0.7

    def test_just_decompose_huber_under_spikes(self) -> None:
        rng = np.random.default_rng(302)
        t = np.linspace(0.0, 5.0, 150)
        true_trend = 1.0 + 0.5 * t
        true_seas = 1.2 * np.cos(2.0 * np.pi * t)
        y = true_trend + true_seas + rng.normal(0.0, 0.15, t.size)
        y[[20, 60, 100, 130]] += np.array([10.0, -12.0, 11.0, -10.0])

        dec_ols = tspec.just_decompose(t, y, size=45, step=22, robust="none")
        dec_rob = tspec.just_decompose(t, y, size=45, step=22, robust="huber")

        err_ols = float(np.mean(np.abs(dec_ols.trend - true_trend)))
        err_rob = float(np.mean(np.abs(dec_rob.trend - true_trend)))
        assert err_rob < 0.25
        assert err_rob < 0.5 * err_ols


class TestRobustStpd:
    """Robust continuous hinge-splines (L1 & Huber) and STPD detection."""

    @pytest.mark.parametrize("mode", ["l1", "huber"])
    def test_piecewise_l1_and_huber_continuity_and_spike_resistance(
        self, mode: str
    ) -> None:
        rng = np.random.default_rng(401)
        t = np.linspace(0.0, 10.0, 160)
        tp = 80
        true_y = 2.0 + 0.5 * t - 1.3 * np.maximum(0.0, t - t[tp])
        y = true_y + rng.normal(0.0, 0.15, t.size)
        # 8% massive 20-sigma spikes
        spike_idx = rng.choice(t.size, size=12, replace=False)
        y[spike_idx] += rng.choice([-1.0, 1.0], size=12) * 8.0

        pw_ols = tspec.fit_piecewise_trend(t, y, [tp], mode="joint")
        pw_rob = tspec.fit_piecewise_trend(t, y, [tp], mode=mode)  # type: ignore[arg-type]

        # Strict continuity at turning point
        left_val = pw_rob.intercepts[0] + pw_rob.slopes[0] * t[tp]
        right_val = pw_rob.intercepts[1] + pw_rob.slopes[1] * t[tp]
        np.testing.assert_allclose(left_val, right_val, atol=1e-9)

        # Accurate slope recovery vs true slopes (+0.5 and -0.8)
        err_rob = abs(pw_rob.slopes[0] - 0.5) + abs(pw_rob.slopes[1] - (-0.8))
        err_ols = abs(pw_ols.slopes[0] - 0.5) + abs(pw_ols.slopes[1] - (-0.8))
        assert err_rob < 0.15
        assert err_rob < 0.4 * err_ols

        # Shift invariance (+1e8)
        pw_shift = tspec.fit_piecewise_trend(t, y + 1e8, [tp], mode=mode)  # type: ignore[arg-type]
        np.testing.assert_allclose(pw_shift.slopes, pw_rob.slopes, rtol=1e-5, atol=1e-5)

    def test_stpd_huber_detects_tp_under_spikes_and_flags_jumps(self) -> None:
        rng = np.random.default_rng(402)
        t = np.linspace(0.0, 10.0, 200)
        y_tp = 1.0 + 0.6 * t - 1.4 * np.maximum(0.0, t - 5.0) + rng.normal(0.0, 0.18, t.size)
        y_tp[[25, 65, 130, 175]] += np.array([9.0, -10.0, 9.5, -8.5])

        res_tp = tspec.detect_turning_points(
            t, y_tp, size=60, step=20, robust="huber", refit_mode="l1"
        )
        assert len(res_tp.turning_points) == 1
        assert abs(res_tp.turning_points[0].time_year - 5.0) < 0.45
        assert abs(res_tp.piecewise_trend.slopes[0] - 0.6) < 0.15
        assert abs(res_tp.piecewise_trend.slopes[1] - (-0.8)) < 0.15

        # NDRI jump discrimination must remain intact under robust="huber"
        y_step = 1.0 + 0.1 * t + np.where(t >= 5.0, 4.0, 0.0) + rng.normal(0.0, 0.15, t.size)
        res_step = tspec.detect_turning_points(t, y_step, size=60, step=20, robust="huber")
        assert len(res_step.flagged_jumps) >= 1
        assert all(abs(tp.time_year - 5.0) > 0.5 for tp in res_step.turning_points)


class TestRobustMonitor:
    """Near-real-time JUST monitoring under historical spikes & real disturbances."""

    @pytest.mark.parametrize("robust_mode", ["huber", "mm"])
    def test_monitor_huber_historical_spike_immunity_and_recall(self, robust_mode: str) -> None:
        rng = np.random.default_rng(501)
        t = np.linspace(0.0, 6.0, 180)
        y_clean = 1.0 + 0.2 * t + 0.7 * np.sin(2.0 * np.pi * t) + rng.normal(0.0, 0.15, t.size)

        # Contaminate historical period (first 4 years, indices < 120) with large spikes
        y_hist_spike = y_clean.copy()
        y_hist_spike[[15, 45, 80, 105]] += np.array([10.0, -11.0, 9.5, -10.5])

        # No disturbance in monitoring period -> robust baseline stays calm
        mon_null = tspec.just_monitor(
            t, y_hist_spike, start=120, history=0, alpha=0.01, h=0.25, robust=robust_mode  # type: ignore[arg-type]
        )
        assert not mon_null.detected

        # Real disturbance (-3.0 step) in monitoring period -> must detect promptly
        y_disturb = y_hist_spike.copy()
        y_disturb[135:] -= 3.0
        mon_det = tspec.just_monitor(
            t, y_disturb, start=120, history=0, alpha=0.01, h=0.25, robust=robust_mode  # type: ignore[arg-type]
        )
        assert mon_det.detected
        assert mon_det.disturbance_index is not None
        assert 135 <= mon_det.disturbance_index <= 145


class TestRobustPipelineAndSen:
    """End-to-end analyze_series with robust='huber', Theil-Sen, ATS, L1, and Huber slopes."""

    @pytest.mark.parametrize("est", ["ols", "sen", "huber", "l1"])
    def test_pipeline_all_slope_estimators(self, est: str) -> None:
        rng = np.random.default_rng(601)
        t = np.linspace(0.0, 8.0, 160)
        y = (
            1.0
            + 0.6 * t
            - 1.4 * np.maximum(0.0, t - 4.0)
            + 0.7 * np.sin(2.0 * np.pi * t)
            + rng.normal(0.0, 0.15, t.size)
        )
        y[[18, 55, 110, 145]] += np.array([9.0, -10.0, 11.0, -9.0])

        res = tspec.analyze_series(
            t,
            y,
            robust="huber",
            slope_estimator=est,  # type: ignore[arg-type]
            just_kwargs={"size": 40, "step": 20},
            decompose_kwargs={"size": 40, "step": 20},
            stpd_kwargs={"size": 50, "step": 15},
        )
        assert res.robust == "huber"
        assert res.slope_estimator == est
        assert len(res.segments) >= 2
        assert sum(seg.n_obs for seg in res.segments) == t.size
        assert "Segment Trend Estimates" in res.summary()
        assert "Segment trend slopes" in res.explain()

        if est in ("sen", "huber", "l1"):
            assert abs(res.segments[0].slope - 0.6) < 0.2
            assert abs(res.segments[-1].slope - (-0.8)) < 0.2
        if est in ("sen", "huber"):
            for seg in res.segments:
                assert seg.slope_ci_low is not None and seg.slope_ci_high is not None
                assert seg.slope_ci_low <= seg.slope <= seg.slope_ci_high
                assert seg.pvalue is not None and 0.0 <= seg.pvalue <= 1.0
        if est == "l1":
            for seg in res.segments:
                assert seg.slope_ci_low is None and seg.slope_ci_high is None
                assert seg.pvalue is None

        if importlib.util.find_spec("pandas") is not None:
            df_seg = res.segments_to_dataframe()
            assert len(df_seg) == len(res.segments)
            assert {"slope", "intercept", "slope_ci_low", "slope_ci_high", "pvalue"} <= set(
                df_seg.columns
            )

    def test_theil_sen_and_akritas_theil_sen_unit(self) -> None:
        rng = np.random.default_rng(602)
        t = np.linspace(0.0, 10.0, 80)
        y_true = 1.0 + 0.5 * t + rng.normal(0.0, 0.25, t.size)

        # Uncensored Theil-Sen with spikes
        y_sp = y_true.copy()
        y_sp[[5, 22, 48, 70]] += np.array([12.0, -14.0, 15.0, -13.0])
        ts_res = fit_theil_sen_segment(t, y_sp)
        assert ts_res.estimator == "sen"
        assert abs(ts_res.slope - 0.5) < 0.08
        assert ts_res.ci_low is not None and ts_res.ci_high is not None
        assert ts_res.ci_low <= 0.5 <= ts_res.ci_high
        assert ts_res.pvalue is not None and ts_res.pvalue < 1e-6

        # Left-censored Akritas-Theil-Sen (30% left-censored below DL=2.5 + spikes)
        dl = 2.5
        codes = np.where(y_sp < dl, -1, 0).astype(np.intp)
        orig_vals = np.where(codes == -1, dl, y_sp)
        ats_res = fit_theil_sen_segment(
            t, y_sp, status_codes=codes, original_values=orig_vals
        )
        assert ats_res.estimator == "ats"
        assert abs(ats_res.slope - 0.5) < 0.12
        assert ats_res.ci_low is not None and ats_res.ci_high is not None
        assert ats_res.ci_low <= ats_res.slope <= ats_res.ci_high
        assert ats_res.pvalue is not None and ats_res.pvalue < 1e-4

    def test_pipeline_censored_ats_integration(self) -> None:
        if importlib.util.find_spec("ndimpute") is None:
            pytest.skip("ndimpute not installed")
        rng = np.random.default_rng(603)
        t = np.linspace(0.0, 8.0, 100)
        y_lat = np.exp(0.3 + 0.25 * t + rng.normal(0.0, 0.15, t.size))
        dl = float(np.quantile(y_lat, 0.28))
        y_str = [f"<{dl:.4f}" if v < dl else f"{v:.4f}" for v in y_lat]

        res = tspec.analyze_series(
            t,
            y_str,
            remove_seasonality=False,
            detect_jumps=False,
            detect_turning_points=False,
            robust="huber",
            slope_estimator="sen",
            censoring=tspec.CensoredConfig(n_stability_draws=0, random_state=0),
        )
        assert len(res.segments) == 1
        seg = res.segments[0]
        assert seg.estimator == "ats"
        assert seg.n_censored > 15
        assert seg.slope > 0.0
        assert seg.slope_ci_low is not None and seg.slope_ci_high is not None
        assert seg.slope_ci_low <= seg.slope <= seg.slope_ci_high
        assert seg.pvalue is not None and seg.pvalue < 1e-4

    def test_fit_l1_piecewise_weighted(self) -> None:
        t = np.linspace(0.0, 6.0, 60)
        y = 1.0 + 0.8 * t - 1.6 * np.maximum(0.0, t - t[30])
        y[10] += 50.0
        w = np.ones(60, dtype=np.float64)
        w[10] = 0.01
        slopes, intercepts, y_fit, cond = fit_l1_piecewise(
            t, y, np.array([30], dtype=np.intp), w=w
        )
        np.testing.assert_allclose(slopes, [0.8, -0.8], atol=1e-6)
        assert np.isfinite(cond)
        assert y_fit.shape == (60,)
