"""Tests for v1.1 Near-Real-Time disturbance monitoring (just_monitor)."""

from __future__ import annotations

import numpy as np
import pytest

from tspec import CensoredConfig, just_monitor


def test_just_monitor_detects_abrupt_disturbance_with_multiplicity_control() -> None:
    rng = np.random.default_rng(10)
    t = np.sort(rng.uniform(2016.0, 2022.0, size=140))
    season = 0.25 * np.cos(2 * np.pi * t) + 0.1 * np.sin(4 * np.pi * t)
    y = 0.65 - 0.01 * (t - 2016.0) + season + rng.normal(0.0, 0.025, size=t.size)
    # Abrupt disturbance (e.g., wildfire) at 2021.3
    y = np.where(t >= 2021.3, y - 0.35, y)

    res = just_monitor(
        t,
        y,
        start=2021.0,
        season="ols",
        alpha=0.01,
        freq_correction="sidak",
        sequential_control="pocock",
    )
    assert res.detected
    assert res.disturbance_type == "abrupt"
    assert res.type_code == 1
    assert res.disturbance_time_year is not None
    assert abs(res.disturbance_time_year - 2021.3) <= 0.35
    assert res.magnitude is not None
    assert res.magnitude < -0.20
    assert "ABRUPT" in res.summary()
    assert "ABRUPT" in res.explain()
    df = res.to_dataframe()
    assert len(df) == len(res.steps)

    fig, axes = res.plot()
    assert len(axes) == 2
    fig.clf()


def test_just_monitor_detects_phenological_shift() -> None:
    rng = np.random.default_rng(22)
    t = np.linspace(2015.0, 2022.0, 180)
    base_season = 0.25 * np.cos(2 * np.pi * t)
    # After 2020.8, strong intra-annual (2 cy/yr) oscillation appears without a baseline drop
    pheno_shift = np.where(t >= 2020.8, 0.30 * np.sin(2 * np.pi * 2.0 * t), 0.0)
    y = 1.0 + base_season + pheno_shift + rng.normal(0.0, 0.025, size=t.size)

    res = just_monitor(
        t,
        y,
        start=2020.5,
        season="ols",
        alpha=0.01,
        freq_correction="sidak",
        sequential_control="pocock",
        horizon=60,
    )
    assert res.detected
    assert res.disturbance_type == "phenological"
    assert res.type_code == 2
    assert res.dominant_frequency is not None and res.dominant_frequency >= 1.0
    assert res.magnitude is None


def test_just_monitor_stable_null_and_sequential_modes() -> None:
    rng = np.random.default_rng(33)
    t = np.linspace(2016.0, 2022.0, 150)
    y = 2.0 + 0.05 * (t - 2016.0) + 0.3 * np.cos(2 * np.pi * t) + rng.normal(0.0, 0.05, size=t.size)

    for mode in ("pocock", "obrien_fleming", "consecutive_k"):
        res = just_monitor(
            t,
            y,
            start=120,
            season="ols",
            alpha=0.01,
            freq_correction="sidak",
            sequential_control=mode,
        )
        assert not res.detected
        assert res.type_code == 0
        assert len(res.steps) == 30

    # Unadjusted mode emits explicit repeated-testing warning when not in parity_mode
    res_unadj = just_monitor(
        t,
        y,
        start=120,
        season="ols",
        alpha=0.01,
        freq_correction="none",
        sequential_control="none",
        parity_mode=False,
    )
    codes = {w.code for w in res_unadj.diagnostics.warnings}
    assert "UNADJUSTED_REPEATED_TESTING" in codes


def test_just_monitor_consecutive_k_pending_confirmation_and_sliding() -> None:
    rng = np.random.default_rng(44)
    t = np.linspace(2016.0, 2022.0, 120)
    y_base = 1.0 + 0.2 * np.cos(2 * np.pi * t) + rng.normal(0.0, 0.02, size=t.size)

    # First check step-by-step where a step-drop first exceeds threshold, then stop right at that point
    y_step = y_base.copy()
    y_step[112:] -= 0.35
    r_full = just_monitor(
        t,
        y_step,
        start=105,
        season="ols",
        alpha=0.01,
        freq_correction="sidak",
        sequential_control="consecutive_k",
        consecutive_k=2,
        window_mode="sliding",
        stop_on_first=False,
    )
    assert r_full.detected
    assert len(r_full.steps) == 15
    first_exc = next(s for s in r_full.steps if s.exceeded)
    assert r_full.disturbance_index == first_exc.index

    # Truncate right at the first exceedance -> pending_confirmation must be True
    n_trunc = first_exc.index + 1
    res_pend = just_monitor(
        t[:n_trunc],
        y_step[:n_trunc],
        start=105,
        season="ols",
        alpha=0.01,
        freq_correction="sidak",
        sequential_control="consecutive_k",
        consecutive_k=2,
        window_mode="sliding",
    )
    assert not res_pend.detected
    assert res_pend.pending_confirmation
    assert "PENDING_CONFIRMATION" in {w.code for w in res_pend.diagnostics.warnings}


def test_just_monitor_datetime_nans_and_censored_input() -> None:
    rng = np.random.default_rng(55)
    t_dt = np.arange("2016-01-15", "2023-01-15", np.timedelta64(21, "D"), dtype="datetime64[D]")
    t_yr = (t_dt - np.datetime64("2016-01-01")) / np.timedelta64(1, "D") / 365.25 + 2016.0
    y_true = np.exp(0.3 + 0.25 * np.cos(2 * np.pi * t_yr) + np.where(t_yr >= 2021.8, 1.3, 0.0) + rng.normal(0.0, 0.12, size=t_yr.size))
    dl = float(np.quantile(y_true, 0.20))
    y_str = [f"<{dl:.3f}" if v < dl else f"{v:.4f}" for v in y_true]
    y_str[5] = "nan"

    res = just_monitor(
        t_dt,
        y_str,
        start=np.datetime64("2021-01-01"),
        season="ols",
        alpha=0.01,
        censoring=CensoredConfig(method="ros", dist="lognormal", impute_type="stochastic", random_state=7),
    )
    assert res.detected
    assert res.disturbance_type == "abrupt"
    assert res.censoring is not None
    assert res.censoring.n_censored > 0
    assert res.diagnostics.n_dropped_nan == 1

    fig, _ = res.plot()
    fig.clf()


def test_just_monitor_validation_errors() -> None:
    t = np.linspace(0.0, 5.0, 50)
    y = np.sin(2 * np.pi * t)
    with pytest.raises(ValueError, match="h must be"):
        just_monitor(t, y, h=1.5)
    with pytest.raises(ValueError, match="history start must precede"):
        just_monitor(t, y, history=30, start=20)


def test_just_monitor_thresholds_do_not_depend_on_future_data() -> None:
    """Audit CRT-02: without `horizon`, step thresholds must not change as data accrue."""
    rng = np.random.default_rng(31)
    t = np.sort(rng.uniform(2000, 2010, 240))
    y = 2 + 0.1 * (t - 2000) + 0.5 * np.cos(2 * np.pi * t) + rng.normal(0, 0.15, t.size)
    crit = []
    for n_total in (96 + 12, 96 + 144):
        r = just_monitor(t[:n_total], y[:n_total], start=96, season="ols", alpha=0.05, stop_on_first=False)
        crit.append([s.critical_value for s in r.steps[:12]])
        assert r.horizon == 0
        assert "OPEN_ENDED_SPENDING" in {w.code for w in r.diagnostics.warnings}
    np.testing.assert_allclose(crit[0], crit[1], rtol=0, atol=1e-12)


def test_monitor_explain_and_warnings_for_unadjusted_and_exceeded_horizon() -> None:
    """CRT-03 & MIN-05: honest FWER warnings/explain and parity_mode n_mc guard."""
    rng = np.random.default_rng(51106)
    t = np.linspace(2015.0, 2021.0, 144)
    y = 2.0 + 0.1 * (t - 2015.0) + 0.6 * np.cos(2.0 * np.pi * t) + rng.normal(0.0, 0.2, size=144)

    # seq='none' + freq='sidak' must emit UNADJUSTED_REPEATED_TESTING and NOT claim FWER control
    r_unadj = just_monitor(
        t, y, start=72, season="ols", alpha=0.05, sequential_control="none", freq_correction="sidak"
    )
    assert "UNADJUSTED_REPEATED_TESTING" in {w.code for w in r_unadj.diagnostics.warnings}
    exp_u = r_unadj.explain()
    assert "bound the familywise" not in exp_u
    assert "does not adjust across repeated monitoring steps" in exp_u

    # Exceeding planned horizon must NOT claim unconditional FWER control in explain()
    r_exc = just_monitor(
        t, y, start=72, season="ols", alpha=0.05, sequential_control="pocock", horizon=24
    )
    assert "HORIZON_EXCEEDED" in {w.code for w in r_exc.diagnostics.warnings}
    exp_e = r_exc.explain()
    assert "only guaranteed within the planned horizon" in exp_e

    # parity_mode=True must not reject small n_mc even when sequential_control='consecutive_k'
    r_par = just_monitor(
        t, y, start=72, season="ols", sequential_control="consecutive_k", horizon=24, n_mc=10, parity_mode=True
    )
    assert r_par is not None

