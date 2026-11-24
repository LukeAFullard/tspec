"""End-to-end tests for analyze_series(), result methods, and headless plotting."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from tspec import CensoredConfig, allssa, analyze_series, just_decompose, just_jumps, lssa, stpd


def test_analyze_series_recovers_seasonality_jump_and_turning_point(
    rng: np.random.Generator,
) -> None:
    t = np.sort(rng.uniform(2014.0, 2024.0, size=200))
    # Insert a 3-month blackout
    t = t[~((t > 2016.2) & (t < 2016.45))]

    # Abrupt step jump at 2017.5 (+2.5) and continuous turning point at 2021.0 (slope 0.0 -> -1.5/yr)
    trend = np.where(t < 2017.5, 4.0, 6.5) + np.where(t > 2021.0, -1.5 * (t - 2021.0), 0.0)
    seasonal = 0.8 * np.cos(2 * np.pi * t) + 0.35 * np.sin(2 * np.pi * 2.0 * t)
    y = trend + seasonal + rng.normal(0.0, 0.15, size=t.size)

    res = analyze_series(
        t,
        y,
        just_kwargs={"jump_alpha": 0.01},
        stpd_kwargs={"size": 80, "step": 15, "margin": 12, "alpha": 0.01},
    )
    assert res.jumps is not None and len(res.jumps.jumps) >= 1
    assert any(abs(j.time_year - 2017.5) < 0.3 for j in res.jumps.jumps)
    assert res.turning_points is not None and len(res.turning_points.turning_points) >= 1
    assert any(abs(tp.time_year - 2021.0) < 0.5 for tp in res.turning_points.turning_points)

    assert "Guided Pipeline Result" in res.summary()
    assert "Stage 1 (JUST)" in res.explain()
    assert "Stage 2 (STPD" in res.explain()

    if importlib.util.find_spec("pandas") is not None:
        df = res.to_dataframe()
        assert {"time", "value", "adjusted_value", "combined_trend", "seasonal"}.issubset(df.columns)


def test_analyze_series_with_censored_strings_and_plots(rng: np.random.Generator) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange("2016-01-01", "2023-01-01", 20, dtype="datetime64[D]")
    t_yr = (t - t[0]).astype("timedelta64[D]").astype(float) / 365.25
    y_true = np.exp(
        np.where(t_yr < 3.5, 0.4, 1.3) + 0.3 * np.cos(2 * np.pi * t_yr) + rng.normal(0.0, 0.15, size=t.size)
    )
    dl = float(np.quantile(y_true, 0.18))
    y_str = [f"<{dl:.4f}" if v < dl else float(v) for v in y_true]

    res = analyze_series(
        t,
        y_str,
        censoring=CensoredConfig(n_stability_draws=4, random_state=0),
        just_kwargs={"season": "ols", "jump_alpha": 0.01},
        decompose_kwargs={"season": "ols"},
        stpd_kwargs={"size": 60, "step": 12, "margin": 8},
    )
    assert res.censoring is not None
    assert res.censoring.n_censored > 0
    assert res.jumps is not None and len(res.jumps.jumps) >= 1

    # Smoke-test every .plot() method headlessly
    for obj in [
        res,
        res.jumps,
        res.decomposition,
        res.turning_points,
        res.turning_points.piecewise_trend if res.turning_points is not None else None,
        lssa(t, res.values, omega=[0.5, 1.0, 2.0]),
        allssa(t, res.values, omega=[1.0, 2.0, 3.0]),
    ]:
        if obj is not None:
            fig, _ax = obj.plot()
            plt.close(fig)


def test_red_noise_warning_triggered_on_ar1_residuals(rng: np.random.Generator) -> None:
    n = 150
    t = np.linspace(0.0, 6.0, n)
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.75 * e[i - 1] + rng.normal(0.0, 0.2)
    y = 2.0 + 0.1 * t + e

    r_lssa = lssa(t, y, omega=[0.5, 1.0, 2.0])
    assert r_lssa.diagnostics.red_noise is not None
    assert r_lssa.diagnostics.red_noise.lag1_autocorr > 0.5
    assert r_lssa.diagnostics.red_noise.severity in ("moderate", "strong")
    assert any(
        w.code in ("MODERATE_AUTOCORRELATION", "STRONG_AUTOCORRELATION")
        for w in r_lssa.diagnostics.warnings
    )

    r_stpd = stpd(t, y, size=80, step=15, margin=10)
    assert r_stpd.diagnostics.red_noise is not None
    assert r_stpd.diagnostics.red_noise.lag1_autocorr > 0.5

    # Ensure unused imports check clean
    _ = just_jumps, just_decompose, pytest
