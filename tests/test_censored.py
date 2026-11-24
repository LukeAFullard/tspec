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

        # tspec tie-shuffled quantile eliminates temporal rank bias
        prep = prepare_series_with_censoring(
            t,
            y_obs,
            censor_status=is_cen,
            censoring=CensoredConfig(
                method="ros",
                dist="lognormal",
                impute_type="quantile",
                random_state=seed,
            ),
        )
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
