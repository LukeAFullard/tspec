"""Tests for analytical pointwise confidence & prediction intervals and CI plotting."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tspec import (
    analyze_series,
    detect_turning_points,
    fit_piecewise_trend,
    just_decompose,
    just_monitor,
)


def _count_fill_collections(ax: plt.Axes) -> int:
    """Count filled ribbon collections (excluding axvspan patches) on an axis."""
    return len(ax.collections)


class TestJustDecomposeCI:
    def test_basic_ci_properties_and_monotonicity(self) -> None:
        rng = np.random.default_rng(101)
        t = np.sort(rng.uniform(2015.0, 2023.0, size=180))
        y = 10.0 + 0.6 * (t - 2015.0) + 1.2 * np.cos(2 * np.pi * t) + rng.normal(0.0, 0.25, t.size)

        res80 = just_decompose(t, y, season="ols", ci_level=0.80)
        res95 = just_decompose(t, y, season="ols", ci_level=0.95)
        res99 = just_decompose(t, y, season="ols", ci_level=0.99)

        for r in (res80, res95, res99):
            assert r.trend_ci_low is not None and r.trend_ci_high is not None
            assert r.seasonal_ci_low is not None and r.seasonal_ci_high is not None
            assert np.all(np.isfinite(r.trend_ci_low))
            assert np.all(np.isfinite(r.seasonal_ci_low))
            assert np.all(r.trend_ci_low <= r.trend + 1e-12)
            assert np.all(r.trend <= r.trend_ci_high + 1e-12)
            assert np.all(r.seasonal_ci_low <= r.seasonal + 1e-12)
            assert np.all(r.seasonal <= r.seasonal_ci_high + 1e-12)

        hw80 = res80.trend_ci_high - res80.trend_ci_low  # type: ignore[operator]
        hw95 = res95.trend_ci_high - res95.trend_ci_low  # type: ignore[operator]
        hw99 = res99.trend_ci_high - res99.trend_ci_low  # type: ignore[operator]
        assert np.all(hw80 < hw95)
        assert np.all(hw95 < hw99)

        df = res95.to_dataframe()
        for col in ("trend_ci_low", "trend_ci_high", "seasonal_ci_low", "seasonal_ci_high"):
            assert col in df.columns

    def test_eval_times_and_season_none(self) -> None:
        rng = np.random.default_rng(102)
        t = np.sort(rng.uniform(2016.0, 2022.0, size=140))
        y = 3.0 + 0.4 * (t - 2016.0) + 0.8 * np.sin(2 * np.pi * t) + rng.normal(0.0, 0.2, t.size)
        t_reg = np.linspace(float(t[0]), float(t[-1]), 120)

        res = just_decompose(t, y, eval_times=t_reg, season="allssa")
        assert res.trend_reg_ci_low is not None and res.trend_reg_ci_high is not None
        assert res.seasonal_reg_ci_low is not None and res.seasonal_reg_ci_high is not None
        assert res.trend_reg_ci_low.shape == t_reg.shape
        assert res.trend_reg is not None and res.seasonal_reg is not None
        assert np.all(res.trend_reg_ci_low <= res.trend_reg + 1e-12)
        assert np.all(res.trend_reg <= res.trend_reg_ci_high + 1e-12)

        res_ns = just_decompose(t, y, season="none")
        assert res_ns.seasonal_ci_low is not None and res_ns.seasonal_ci_high is not None
        assert np.allclose(res_ns.seasonal_ci_low, 0.0)
        assert np.allclose(res_ns.seasonal_ci_high, 0.0)
        assert res_ns.trend_ci_low is not None and res_ns.trend_ci_high is not None
        assert np.all(res_ns.trend_ci_high > res_ns.trend_ci_low)

    def test_shift_and_scale_invariance(self) -> None:
        rng = np.random.default_rng(103)
        t = np.linspace(2018.0, 2024.0, 120)
        y = 2.0 + 0.5 * (t - 2018.0) + 0.9 * np.cos(2 * np.pi * t) + rng.normal(0.0, 0.2, t.size)

        r0 = just_decompose(t, y, season="ols")
        r_shift = just_decompose(t, y + 1e6, season="ols")
        r_scale = just_decompose(t, 50.0 * y, season="ols")

        assert r0.trend_ci_low is not None and r0.trend_ci_high is not None
        assert r_shift.trend_ci_low is not None and r_shift.trend_ci_high is not None
        assert r_scale.trend_ci_low is not None and r_scale.trend_ci_high is not None
        hw0 = r0.trend_ci_high - r0.trend_ci_low
        hw_shift = r_shift.trend_ci_high - r_shift.trend_ci_low
        hw_scale = r_scale.trend_ci_high - r_scale.trend_ci_low

        np.testing.assert_allclose(hw_shift, hw0, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(hw_scale, 50.0 * hw0, rtol=1e-6, atol=1e-8)

    def test_robust_ci_resists_outliers(self) -> None:
        rng = np.random.default_rng(104)
        t = np.linspace(2018.0, 2024.0, 144)
        y_clean = 5.0 + 0.3 * (t - 2018.0) + np.cos(2 * np.pi * t) + rng.normal(0.0, 0.15, t.size)
        y_spk = y_clean.copy()
        y_spk[[25, 70, 110]] += 12.0

        r_ols = just_decompose(t, y_spk, season="ols", robust="none")
        r_mm = just_decompose(t, y_spk, season="ols", robust="mm")
        assert r_ols.trend_ci_low is not None and r_ols.trend_ci_high is not None
        assert r_mm.trend_ci_low is not None and r_mm.trend_ci_high is not None
        med_hw_ols = float(np.median(r_ols.trend_ci_high - r_ols.trend_ci_low))
        med_hw_mm = float(np.median(r_mm.trend_ci_high - r_mm.trend_ci_low))
        assert med_hw_mm < 0.4 * med_hw_ols

    def test_empirical_pointwise_coverage(self) -> None:
        rng = np.random.default_rng(2025)
        t = np.linspace(2015.0, 2021.0, 120)
        true_trend = 4.0 + 0.5 * (t - 2015.0)
        true_seas = 1.1 * np.cos(2 * np.pi * t) - 0.4 * np.sin(2 * np.pi * t)
        true_seas -= np.mean(true_seas)

        tr_hits = []
        s_hits = []
        for _ in range(80):
            y = true_trend + true_seas + rng.normal(0.0, 0.3, t.size)
            res = just_decompose(t, y, season="ols", omega=[1.0], ci_level=0.95)
            assert res.trend_ci_low is not None and res.trend_ci_high is not None
            assert res.seasonal_ci_low is not None and res.seasonal_ci_high is not None
            tr_hits.append(np.mean((res.trend_ci_low <= true_trend) & (true_trend <= res.trend_ci_high)))
            s_hits.append(np.mean((res.seasonal_ci_low <= true_seas) & (true_seas <= res.seasonal_ci_high)))

        assert float(np.mean(tr_hits)) >= 0.91
        assert float(np.mean(s_hits)) >= 0.91

    def test_invalid_ci_level(self) -> None:
        t = np.linspace(0.0, 5.0, 50)
        y = np.sin(2 * np.pi * t)
        for bad in (0.0, 1.0, -0.2, 1.5):
            with pytest.raises(ValueError, match="ci_level"):
                just_decompose(t, y, ci_level=bad)


class TestPiecewiseAndStpdCI:
    @pytest.mark.parametrize("weighted", [False, True])
    def test_sequential_variance_recursion_matches_exact_jacobian(self, weighted: bool) -> None:
        """Verify O(n) sequential TPTR variance equals exact linear-operator sum(H^2 / w)."""
        rng = np.random.default_rng(301)
        n = 48
        t = np.sort(rng.uniform(0.0, 10.0, size=n))
        tps = [15, 32]
        y_true = np.where(
            t <= t[15],
            2.0 + 0.5 * t,
            np.where(
                t <= t[32],
                2.0 + 0.5 * t[15] - 0.8 * (t - t[15]),
                2.0 + 0.5 * t[15] - 0.8 * (t[32] - t[15]) + 1.2 * (t - t[32]),
            ),
        )
        y = y_true + rng.normal(0.0, 0.35, size=n)
        w = rng.uniform(0.4, 3.0, size=n) if weighted else None

        res = fit_piecewise_trend(t, y, tps, weights=w, mode="sequential", ci_level=0.95)
        assert res.trend_ci_low is not None and res.trend_ci_high is not None

        # Compute exact hat matrix H_{ij} = d(y_hat_i) / d(y_j) for the selected direction
        from scipy.stats import t as t_dist

        from tspec.stpd import _chained_forward_fit

        t0 = t - t[0]
        tp_arr = np.array(tps, dtype=np.intp)
        H = np.empty((n, n), dtype=np.float64)
        for j in range(n):
            ej = np.zeros(n, dtype=np.float64)
            ej[j] = 1.0
            if res.selected_direction == "forward":
                _, _, yf_j, _ = _chained_forward_fit(t0, ej, tp_arr, w)
                H[:, j] = yf_j
            else:
                t_rev = t0[-1] - t0[::-1]
                w_rev = w[::-1] if w is not None else None
                tp_rev = np.sort(n - 1 - tp_arr)
                _, _, yb_j, _ = _chained_forward_fit(t_rev, ej[::-1], tp_rev, w_rev)
                H[:, j] = yb_j[::-1]

        w_eff = np.ones(n, dtype=np.float64) if w is None else w
        v_exact = np.sum((H * H) / w_eff[None, :], axis=1)
        dof = n - (len(tps) + 2)
        sigma2_hat = float(np.sum(w_eff * res.residuals * res.residuals)) / dof
        hw_exact = float(t_dist.ppf(0.975, df=dof)) * np.sqrt(sigma2_hat * v_exact)
        hw_fast = 0.5 * (res.trend_ci_high - res.trend_ci_low)

        np.testing.assert_allclose(hw_fast, hw_exact, rtol=1e-11, atol=1e-12)

    def test_modes_joint_huber_l1_and_coverage(self) -> None:
        rng = np.random.default_rng(302)
        t = np.linspace(0.0, 10.0, 100)
        y_true = np.where(t <= 5.0, 1.0 + 0.5 * t, 3.5 - 0.9 * (t - 5.0))
        y = y_true + rng.normal(0.0, 0.3, t.size)

        for mode in ("sequential", "joint", "huber"):
            res = fit_piecewise_trend(t, y, [50], mode=mode, ci_level=0.95)
            assert res.trend_ci_low is not None and res.trend_ci_high is not None
            assert np.all(res.trend_ci_low <= res.fitted_trend + 1e-12)
            assert np.all(res.fitted_trend <= res.trend_ci_high + 1e-12)
            df = res.to_dataframe()
            assert "trend_ci_low" in df.columns and "trend_ci_high" in df.columns

        res_l1 = fit_piecewise_trend(t, y, [50], mode="l1")
        assert res_l1.trend_ci_low is None and res_l1.trend_ci_high is None
        df_l1 = res_l1.to_dataframe()
        assert "trend_ci_low" not in df_l1.columns

        # Coverage check for joint OLS hinge-spline
        hits = []
        for _ in range(150):
            y_mc = y_true + rng.normal(0.0, 0.4, t.size)
            rj = fit_piecewise_trend(t, y_mc, [50], mode="joint", ci_level=0.95)
            assert rj.trend_ci_low is not None and rj.trend_ci_high is not None
            hits.append(np.mean((rj.trend_ci_low <= y_true) & (y_true <= rj.trend_ci_high)))
        assert 0.92 <= float(np.mean(hits)) <= 0.98

    def test_detect_turning_points_propagates_ci_level(self) -> None:
        rng = np.random.default_rng(303)
        t = np.linspace(2015.0, 2023.0, 120)
        y = np.where(t <= 2019.0, 5.0 + 0.3 * (t - 2015.0), 6.2 - 1.1 * (t - 2019.0)) + rng.normal(
            0.0, 0.15, t.size
        )
        res90 = detect_turning_points(t, y, size=60, step=10, margin=8, ci_level=0.90)
        res99 = detect_turning_points(t, y, size=60, step=10, margin=8, ci_level=0.99)
        assert res90.piecewise_trend.ci_level == 0.90
        assert res99.piecewise_trend.ci_level == 0.99
        assert res90.piecewise_trend.trend_ci_low is not None
        assert res90.piecewise_trend.trend_ci_high is not None
        assert res99.piecewise_trend.trend_ci_low is not None
        assert res99.piecewise_trend.trend_ci_high is not None
        assert np.all(
            (res99.piecewise_trend.trend_ci_high - res99.piecewise_trend.trend_ci_low)
            > (res90.piecewise_trend.trend_ci_high - res90.piecewise_trend.trend_ci_low)
        )


class TestMonitorCIAndPI:
    def test_forecast_ci_and_pi_ordering_and_extrapolation_growth(self) -> None:
        rng = np.random.default_rng(401)
        t = np.linspace(2016.0, 2024.0, 160)
        y = 5.0 + 0.4 * (t - 2016.0) + 0.8 * np.cos(2 * np.pi * t) + rng.normal(0.0, 0.2, t.size)

        res = just_monitor(t, y, start=120, season="ols", ci_level=0.95)
        assert res.forecast_ci_low is not None and res.forecast_ci_high is not None
        assert res.forecast_pi_low is not None and res.forecast_pi_high is not None

        assert np.all(res.forecast_pi_low < res.forecast_ci_low)
        assert np.all(res.forecast_ci_low <= res.forecast + 1e-12)
        assert np.all(res.forecast <= res.forecast_ci_high + 1e-12)
        assert np.all(res.forecast_ci_high < res.forecast_pi_high)

        # Extrapolation cone must widen at the far end of the monitoring horizon vs start
        hw_ci = res.forecast_ci_high - res.forecast_ci_low
        assert hw_ci[-1] > hw_ci[120]

    def test_empirical_out_of_sample_pi_and_ci_coverage(self) -> None:
        rng = np.random.default_rng(402)
        t = np.linspace(2016.0, 2022.0, 120)
        mu = 3.0 + 0.5 * (t - 2016.0) + 0.7 * np.cos(2 * np.pi * t)

        ci_hits = []
        pi_hits = []
        for _ in range(120):
            y = mu + rng.normal(0.0, 0.25, t.size)
            res = just_monitor(t, y, start=90, season="ols", omega=[1.0], ci_level=0.95)
            assert res.forecast_ci_low is not None and res.forecast_ci_high is not None
            assert res.forecast_pi_low is not None and res.forecast_pi_high is not None
            ci_hits.append(
                np.mean((res.forecast_ci_low[90:] <= mu[90:]) & (mu[90:] <= res.forecast_ci_high[90:]))
            )
            pi_hits.append(
                np.mean((res.forecast_pi_low[90:] <= y[90:]) & (y[90:] <= res.forecast_pi_high[90:]))
            )

        assert 0.91 <= float(np.mean(ci_hits)) <= 0.99
        assert 0.91 <= float(np.mean(pi_hits)) <= 0.99

    def test_monitor_ci_with_robust_and_matrix_weights(self) -> None:
        rng = np.random.default_rng(403)
        t = np.linspace(2016.0, 2022.0, 90)
        y = 2.0 + 0.3 * (t - 2016.0) + rng.normal(0.0, 0.2, t.size)
        y[20] += 8.0

        r_mm = just_monitor(t, y, start=70, season="none", robust="mm", ci_level=0.90)
        assert r_mm.forecast_ci_low is not None and r_mm.forecast_pi_low is not None
        assert r_mm.ci_level == 0.90

        # 2D SPD weights
        idx = np.arange(t.size)
        cov_mat = 0.2**2 * (0.4 ** np.abs(idx[:, None] - idx[None, :]))
        P = np.linalg.inv(cov_mat)
        r_mat = just_monitor(t, y, weights=P, start=70, season="none", ci_level=0.95)
        assert r_mat.forecast_ci_low is not None and r_mat.forecast_pi_low is not None
        assert np.all(np.isfinite(r_mat.forecast_pi_low))


class TestPipelineCIAndPlotting:
    @pytest.mark.parametrize("slope_estimator", ["ols", "huber", "sen", "l1"])
    def test_pipeline_ci_all_estimators(self, slope_estimator: str) -> None:
        rng = np.random.default_rng(501)
        t = np.sort(rng.uniform(2015.0, 2023.0, size=180))
        trend = np.where(t < 2019.0, 4.0 + 0.3 * (t - 2015.0), 7.2 - 0.6 * (t - 2019.0))
        y = trend + 0.7 * np.cos(2 * np.pi * t) + rng.normal(0.0, 0.18, t.size)

        res = analyze_series(
            t,
            y,
            slope_estimator=slope_estimator,  # type: ignore[arg-type]
            ci_level=0.90,
            just_kwargs={"season": "ols", "jump_alpha": 0.01},
            decompose_kwargs={"season": "ols"},
            stpd_kwargs={"size": 70, "step": 12, "margin": 10},
        )
        assert res.ci_level == 0.90
        assert res.decomposition is not None and res.decomposition.ci_level == 0.90

        df = res.to_dataframe()
        assert "seasonal_ci_low" in df.columns and "seasonal_ci_high" in df.columns
        if slope_estimator == "l1":
            assert res.trend_ci_low is None and res.trend_ci_high is None
            assert "trend_ci_low" not in df.columns
        else:
            assert res.trend_ci_low is not None and res.trend_ci_high is not None
            assert "trend_ci_low" in df.columns and "trend_ci_high" in df.columns
            assert np.all(res.trend_ci_low <= res.combined_trend + 1e-10)
            assert np.all(res.combined_trend <= res.trend_ci_high + 1e-10)
            assert "CI90=" in res.summary()
            assert "90% CI" in res.explain()

        # Verify plotting works with both show_ci=True and show_ci=False
        fig_on, axes_on = res.plot(show_ci=True)
        fig_off, axes_off = res.plot(show_ci=False)
        if slope_estimator != "l1":
            assert _count_fill_collections(axes_on[0]) >= 1
        assert _count_fill_collections(axes_off[0]) == 0
        assert _count_fill_collections(axes_on[2]) >= 1
        assert _count_fill_collections(axes_off[2]) == 0
        plt.close(fig_on)
        plt.close(fig_off)

    def test_pipeline_segment_jump_adjustment_ci(self) -> None:
        rng = np.random.default_rng(502)
        t = np.linspace(2015.0, 2023.0, 160)
        y = np.where(t < 2019.0, 3.0 + 0.4 * (t - 2015.0), 7.5 - 0.5 * (t - 2019.0)) + rng.normal(
            0.0, 0.15, t.size
        )
        res = analyze_series(
            t,
            y,
            remove_seasonality=False,
            jump_adjustment="segment",
            ci_level=0.95,
            just_kwargs={"season": "none", "jump_alpha": 0.01},
        )
        assert res.trend_ci_low is not None and res.trend_ci_high is not None
        assert np.all(np.isfinite(res.trend_ci_low))
        assert np.all(res.trend_ci_low <= res.combined_trend + 1e-10)

    def test_all_plotting_functions_show_ci_toggle(self) -> None:
        rng = np.random.default_rng(503)
        t = np.linspace(2016.0, 2023.0, 140)
        y = np.where(t <= 2019.5, 5.0 + 0.3 * (t - 2016.0), 6.05 - 0.8 * (t - 2019.5))
        y_seas = y + 0.8 * np.cos(2 * np.pi * t) + rng.normal(0.0, 0.18, t.size)

        dec = just_decompose(t, y_seas, season="ols")
        fig1, ax1 = dec.plot(show_ci=True)
        fig2, ax2 = dec.plot(show_ci=False)
        assert _count_fill_collections(ax1[1]) >= 1 and _count_fill_collections(ax1[2]) >= 1
        assert _count_fill_collections(ax2[1]) == 0 and _count_fill_collections(ax2[2]) == 0
        plt.close(fig1)
        plt.close(fig2)

        pw = fit_piecewise_trend(t, y + rng.normal(0.0, 0.15, t.size), [70], mode="joint")
        fig3, ax3 = pw.plot(show_ci=True)
        fig4, ax4 = pw.plot(show_ci=False)
        assert _count_fill_collections(ax3) >= 1 and _count_fill_collections(ax4) == 0
        plt.close(fig3)
        plt.close(fig4)

        st = detect_turning_points(t, y + rng.normal(0.0, 0.15, t.size), size=60, step=10, margin=8)
        fig5, ax5 = st.plot(show_ci=True)
        fig6, ax6 = st.plot(show_ci=False)
        assert _count_fill_collections(ax5) >= 1 and _count_fill_collections(ax6) == 0
        plt.close(fig5)
        plt.close(fig6)

        mon = just_monitor(t, y_seas, start=105, season="ols")
        fig7, ax7 = mon.plot(show_ci=True)
        fig8, ax8 = mon.plot(show_ci=False)
        # Two ribbons on top panel (PI + CI)
        assert _count_fill_collections(ax7[0]) >= 2 and _count_fill_collections(ax8[0]) == 0
        plt.close(fig7)
        plt.close(fig8)
