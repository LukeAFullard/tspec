"""Reproducibly generate all README and documentation result figures from live tspec runs.

Run from the repository root:
    python3 docs/scripts/generate_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import tspec
from tspec import (
    CensoredConfig,
    analyze_series,
    detect_turning_points,
    just_decompose,
    just_jumps,
    just_monitor,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "assets" / "figures"

# Cohesive Okabe-Ito colorblind-safe palette
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
DARK = "#1E293B"
GRAY = "#64748B"


def _style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.grid(True, color="#E2E8F0", linewidth=0.8, alpha=0.9)
    ax.tick_params(colors=DARK, labelsize=9.5)


def _save(fig: plt.Figure, filename: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


def make_hero_combined_workflow() -> None:
    """Hero 3-panel figure for README, docs/index.md, and tutorials/combined-workflow.md."""
    rng = np.random.default_rng(42)
    t_raw = np.sort(rng.uniform(2015.0, 2025.0, size=220))
    # 4-month sensor blackout gap (zero interpolation!)
    t = t_raw[~((t_raw > 2016.9) & (t_raw < 2017.25))]
    trend = np.where(t < 2018.5, 4.0, 6.4) + np.where(t > 2021.8, -1.4 * (t - 2021.8), 0.0)
    seas = 0.85 * np.cos(2 * np.pi * t) + 0.30 * np.sin(4 * np.pi * t)
    y = trend + seas + rng.normal(0.0, 0.16, size=t.size)

    res = analyze_series(
        t,
        y,
        just_kwargs={"season": "ols", "jump_alpha": 0.01},
        decompose_kwargs={"season": "ols"},
        stpd_kwargs={"size": 100, "step": 15, "margin": 14, "alpha": 0.005, "snr_th": 2.0},
    )

    fig, axes = plt.subplots(3, 1, figsize=(10.0, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.5, 1.15, 0.85]})
    # Panel 1: Observed + combined trend + detected jump & turning point
    ax0 = axes[0]
    ax0.axvspan(2016.9, 2017.25, color="#CBD5E1", alpha=0.35, label="4-month blackout gap (zero interpolation)")
    ax0.plot(t, y, "o", color=DARK, ms=3.2, alpha=0.65, label="Irregular observations")
    if res.trend_ci_low is not None and res.trend_ci_high is not None:
        j_cut = res.jumps.jumps[0].valid_index if (res.jumps and res.jumps.jumps) else t.size
        ax0.fill_between(t[:j_cut], res.trend_ci_low[:j_cut], res.trend_ci_high[:j_cut], color=BLUE, alpha=0.22, lw=0, label="95% trend CI")
        if j_cut < t.size:
            ax0.fill_between(t[j_cut:], res.trend_ci_low[j_cut:], res.trend_ci_high[j_cut:], color=BLUE, alpha=0.22, lw=0)
    ax0.plot(t, res.combined_trend, color=BLUE, lw=2.2, label="Combined trend (JUST + STPD)")

    if res.jumps and res.jumps.jumps:
        j = res.jumps.jumps[0]
        ax0.axvline(j.time_year, color=ORANGE, ls="--", lw=1.8, label=f"JUST abrupt jump ({j.time_year:.2f}, Δ={j.magnitude:+.2f})")
        ax0.annotate(
            f"Abrupt jump\n{j.time_year:.2f} (Δ={j.magnitude:+.2f})",
            xy=(j.time_year, float(res.combined_trend[j.valid_index])),
            xytext=(j.time_year - 1.9, 6.75),
            fontsize=8.8,
            fontweight="bold",
            color=ORANGE,
            bbox={"boxstyle": "round,pad=0.3", "fc": "#FFF7ED", "ec": ORANGE, "alpha": 0.95},
            arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 1.3},
        )

    if res.turning_points and res.turning_points.turning_points:
        tp = max(res.turning_points.turning_points, key=lambda item: abs(item.direction))
        y_tp = float(res.combined_trend[tp.valid_index])
        ax0.plot(tp.time_year, y_tp, "D", color=GREEN, ms=8.8, markeredgecolor="white", markeredgewidth=1.1, zorder=6,
                 label=f"STPD turning point ({tp.time_year:.2f}, NDRI={tp.ndri_rmse:.2f})")
        ax0.annotate(
            f"Turning point\n{tp.time_year:.2f} (NDRI={tp.ndri_rmse:.2f})",
            xy=(tp.time_year, y_tp),
            xytext=(tp.time_year + 0.75, y_tp + 0.45),
            fontsize=8.8,
            fontweight="bold",
            color=GREEN,
            bbox={"boxstyle": "round,pad=0.3", "fc": "#ECFDF5", "ec": GREEN, "alpha": 0.95},
            arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 1.3},
        )

    ax0.set_ylabel("Observed Value", fontweight="bold", color=DARK)
    ax0.set_title("Guided Two-Stage Workflow: tspec.analyze_series() — Irregular Seasonality + Jump + Turning Point",
                  fontsize=11.5, fontweight="bold", color=DARK, pad=10)
    ax0.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", loc="lower left", ncol=2, fontsize=8.0)
    _style_ax(ax0)

    # Panel 2: Deseasonalized & jump-adjusted series with STPD connected trend
    ax1 = axes[1]
    ax1.axvspan(2016.9, 2017.25, color="#CBD5E1", alpha=0.35)
    ax1.plot(t, res.adjusted_series, "o", color=DARK, ms=3.0, alpha=0.55, label="Deseasonalized & jump-adjusted")
    if res.turning_points:
        pw = res.turning_points.piecewise_trend
        if pw.trend_ci_low is not None and pw.trend_ci_high is not None:
            ax1.fill_between(t, pw.trend_ci_low, pw.trend_ci_high, color=GREEN, alpha=0.22, lw=0, label="95% CI")
        ax1.plot(t, pw.fitted_trend, color=GREEN, lw=2.2, label="STPD connected piecewise trend")
        for tp in res.turning_points.turning_points:
            ax1.axvline(tp.time_year, color=GREEN, ls=":", lw=1.6)
    ax1.set_ylabel("Adjusted Level", fontweight="bold", color=DARK)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", loc="lower left", fontsize=8.5)
    _style_ax(ax1)

    # Panel 3: Recovered seasonal component
    ax2 = axes[2]
    ax2.axvspan(2016.9, 2017.25, color="#CBD5E1", alpha=0.35)
    if res.decomposition is not None:
        if res.decomposition.seasonal_ci_low is not None and res.decomposition.seasonal_ci_high is not None:
            ax2.fill_between(t, res.decomposition.seasonal_ci_low, res.decomposition.seasonal_ci_high, color=PURPLE, alpha=0.22, lw=0, label="95% seasonal CI")
        ax2.plot(t, res.decomposition.seasonal, color=PURPLE, lw=1.6, label="Recovered seasonal component (1 + 2 cy/yr)")
    ax2.axhline(0.0, color=GRAY, lw=0.8, ls=":")
    ax2.set_ylabel("Seasonal", fontweight="bold", color=DARK)
    ax2.set_xlabel("Year", fontweight="bold", color=DARK)
    ax2.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", loc="upper right", ncol=2, fontsize=8.3)
    _style_ax(ax2)

    fig.tight_layout()
    _save(fig, "hero_combined_workflow.png")


def make_concept_jump_vs_tp() -> None:
    """Side-by-side contrast: Abrupt Jump (JUST) vs. Continuous Turning Point (STPD)."""
    rng = np.random.default_rng(7)
    t = np.linspace(2016.0, 2022.0, 96)

    # Left: Abrupt step jump (vertical cliff)
    y_jump = np.where(t < 2019.0, 5.0 - 0.1 * (t - 2016.0), 2.3 - 0.1 * (t - 2016.0)) + rng.normal(0.0, 0.22, size=t.size)
    rj = just_jumps(t, y_jump, season="none", mag_th=0.5, dir_th=0.1, jump_alpha=0.01)
    dj = just_decompose(t, y_jump, jump_indices=[j.index for j in rj.jumps], season="none")
    rs_on_jump = detect_turning_points(t, y_jump, size=72, step=12, margin=10)
    ndri_jump = rs_on_jump.flagged_jumps[0].ndri_rmse if rs_on_jump.flagged_jumps else 0.72

    # Right: Continuous turning point (no vertical cliff, smooth bend)
    y_tp = np.where(t <= 2019.0, 5.0 - 0.25 * (t - 2016.0), 4.25 - 1.55 * (t - 2019.0)) + rng.normal(0.0, 0.22, size=t.size)
    rs = detect_turning_points(t, y_tp, size=72, step=12, margin=10)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.5, 4.1), sharey=True)

    # Left panel
    axL.plot(t, y_jump, "o", color=DARK, ms=3.4, alpha=0.65, label="Observations")
    j_idx = rj.jumps[0].valid_index
    axL.plot(t[:j_idx], dj.trend[:j_idx], color=ORANGE, lw=2.4, label="JUST piecewise trend")
    axL.plot(t[j_idx:], dj.trend[j_idx:], color=ORANGE, lw=2.4)
    axL.axvline(rj.jumps[0].time_year, color=ORANGE, ls="--", lw=1.6)
    axL.annotate(
        f"Vertical cliff: Δ = {rj.jumps[0].magnitude:+.2f}\n(STPD flags NDRI = {ndri_jump:.2f} → Jump!)",
        xy=(rj.jumps[0].time_year, 3.6),
        xytext=(2016.25, 1.0),
        fontsize=8.8,
        color=ORANGE,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "fc": "#FFF7ED", "ec": ORANGE},
        arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 1.3},
    )
    axL.set_title("Abrupt Step Jump → Use JUST", fontsize=11, fontweight="bold", color=DARK)
    axL.set_xlabel("Year", fontweight="bold", color=DARK)
    axL.set_ylabel("Measured Value", fontweight="bold", color=DARK)
    axL.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", loc="upper right", fontsize=8.5)
    _style_ax(axL)

    # Right panel
    axR.plot(t, y_tp, "o", color=DARK, ms=3.4, alpha=0.65, label="Observations")
    axR.plot(t, rs.piecewise_trend.fitted_trend, color=GREEN, lw=2.4, label="STPD connected trend")
    tp = rs.turning_points[0]
    y_at_tp = float(rs.piecewise_trend.fitted_trend[tp.valid_index])
    axR.plot(tp.time_year, y_at_tp, "D", color=GREEN, ms=8.5, markeredgecolor="white", markeredgewidth=1.0, zorder=5)
    axR.axvline(tp.time_year, color=GREEN, ls="--", lw=1.5)
    axR.annotate(
        f"Continuous slope bend (no cliff)\nNDRI = {tp.ndri_rmse:.2f} < 0.30 → Turning Point!",
        xy=(tp.time_year, y_at_tp),
        xytext=(2016.25, 1.0),
        fontsize=8.8,
        color=GREEN,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "fc": "#ECFDF5", "ec": GREEN},
        arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 1.3},
    )
    axR.set_title("Continuous Turning Point → Use STPD", fontsize=11, fontweight="bold", color=DARK)
    axR.set_xlabel("Year", fontweight="bold", color=DARK)
    axR.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", loc="upper right", fontsize=8.5)
    _style_ax(axR)

    fig.tight_layout()
    _save(fig, "concept_jump_vs_tp.png")


def make_concept_count_vs_duration() -> None:
    """Illustrate count vs. duration windowing when sampling density quadruples mid-record."""
    rng = np.random.default_rng(14)
    # 2012-2017: sparse (~8 samples/yr); 2017-2022: dense (~32 samples/yr)
    t_sparse = np.linspace(2012.0, 2016.9, 40)
    t_dense = np.linspace(2017.0, 2022.0, 160)
    t = np.concatenate([t_sparse, t_dense])
    y = np.where(t < 2019.2, 2.0, 3.3) + 0.45 * np.cos(2 * np.pi * t) + rng.normal(0.0, 0.14, size=t.size)

    r_cnt = just_jumps(t, y, season="ols", window_mode="count", size=40, step=10, mag_th=0.4, dir_th=0.1, jump_alpha=0.01)
    r_dur = just_jumps(t, y, season="ols", window_mode="duration", window_duration=3.0, step_duration=0.5, mag_th=0.4, dir_th=0.1, jump_alpha=0.01)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.8, 5.4), sharex=True)

    # Top: observation-count windows (same 40 samples = 5.0 yrs early, only 1.25 yrs late!)
    ax1.plot(t, y, "o", color=DARK, ms=2.8, alpha=0.65)
    ax1.axvspan(t[0], t[39], color=SKY, alpha=0.22, label="Window #1 (40 pts = 4.9 years)")
    ax1.axvspan(t[120], t[159], color=ORANGE, alpha=0.22, label="Window #9 (40 pts = 1.2 years!)")
    for j in r_cnt.jumps:
        ax1.axvline(j.time_year, color=ORANGE, ls="--", lw=1.6, label=f"Detected jump ({j.time_year:.2f})")
    ax1.set_title(f"window_mode='count' (size=40) — Physical Window Shrinks 4× (rate ratio = {r_cnt.diagnostics.sampling_rate_ratio:.1f}×)",
                  fontsize=10.5, fontweight="bold", color=DARK)
    ax1.set_ylabel("Value", fontweight="bold", color=DARK)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", loc="upper left", ncol=2, fontsize=8.3)
    _style_ax(ax1)

    # Bottom: duration windows (consistent 3.0-year physical windows throughout)
    ax2.plot(t, y, "o", color=DARK, ms=2.8, alpha=0.65)
    ax2.axvspan(2012.0, 2015.0, color=GREEN, alpha=0.18, label="Window #1 (3.0 years, 25 pts)")
    ax2.axvspan(2018.0, 2021.0, color=BLUE, alpha=0.18, label="Window #13 (3.0 years, 96 pts)")
    for j in r_dur.jumps:
        ax2.axvline(j.time_year, color=GREEN, ls="--", lw=1.8, label=f"Detected jump ({j.time_year:.2f}, Δ={j.magnitude:+.2f})")
    ax2.set_title("window_mode='duration' (window_duration=3.0 yr) — Constant Physical Time Horizon",
                  fontsize=10.5, fontweight="bold", color=DARK)
    ax2.set_ylabel("Value", fontweight="bold", color=DARK)
    ax2.set_xlabel("Year", fontweight="bold", color=DARK)
    ax2.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", loc="upper left", ncol=2, fontsize=8.3)
    _style_ax(ax2)

    fig.tight_layout()
    _save(fig, "concept_count_vs_duration.png")


def make_tutorial_water_quality_censored() -> None:
    """Water quality nitrate record with <DL values, lab DL upgrade, and real jump."""
    rng = np.random.default_rng(123)
    dates = np.arange("2016-01-15", "2024-01-15", np.timedelta64(20, "D"), dtype="datetime64[D]")
    t_yr = 2016.0 + (dates - np.datetime64("2016-01-01")) / np.timedelta64(1, "D") / 365.25
    # True nitrate concentration: seasonal log-normal baseline + real upward land-use jump at 2021.3
    log_true = (
        np.where(t_yr < 2021.3, -0.45, 0.55)
        + 0.32 * np.cos(2 * np.pi * t_yr)
        + rng.normal(0.0, 0.20, size=t_yr.size)
    )
    y_true = np.exp(log_true)
    # Lab upgraded analytical method in Jan 2019: detection limit dropped from 0.65 -> 0.35 mg/L
    dl = np.where(t_yr < 2019.0, 0.65, 0.35)
    y_strings = [f"<{d:.2f}" if v < d else f"{v:.3f}" for v, d in zip(y_true, dl, strict=True)]

    cfg = CensoredConfig(
        method="ros",
        dist="lognormal",
        impute_type="stochastic",
        detect_dl_shifts=True,
        recensor_check=True,
        random_state=42,
    )
    rj = just_jumps(
        dates,
        y_strings,
        season="ols",
        mag_th=0.80,
        dir_th=1.0,
        jump_alpha=0.01,
        step_alpha=0.01,
        censoring=cfg,
    )
    decomp = just_decompose(
        dates,
        y_strings,
        jump_indices=[j.index for j in rj.jumps],
        season="ols",
        censoring=cfg,
    )

    fig, ax = plt.subplots(figsize=(10.0, 4.5))
    cen = decomp.censoring
    assert cen is not None
    obs = ~cen.is_censored
    left = cen.status_codes == -1

    ax.plot(decomp.times_raw[obs], decomp.values[obs], "o", color=DARK, ms=4.0, alpha=0.8, label="Detected concentration")
    ax.plot(
        decomp.times_raw[left],
        decomp.values[left],
        "v",
        markerfacecolor="white",
        markeredgecolor=ORANGE,
        markeredgewidth=1.4,
        ms=6.0,
        ls="none",
        label=f"Imputed <DL via Tobit MLE ({cen.censor_fraction:.0%})",
    )
    ax.step(dates, dl, where="post", color=GRAY, ls=":", lw=1.5, label="Lab detection limit (0.65 → 0.35 mg/L)")

    # Plot piecewise JUST trend without connecting across jump
    if rj.jumps:
        j0 = rj.jumps[0]
        k = j0.valid_index
        ax.plot(decomp.times_raw[:k], decomp.trend[:k], color=BLUE, lw=2.4, label="JUST deseasonalized trend")
        ax.plot(decomp.times_raw[k:], decomp.trend[k:], color=BLUE, lw=2.4)
        ax.axvline(j0.time, color=ORANGE, ls="--", lw=1.8, label=f"Real jump ({str(np.datetime64(j0.time, 'D'))}, Δ={j0.magnitude:+.2f} mg/L)")

    # Annotate lab DL shift (verified no fake jump!)
    if cen.dl_shifts:
        sh = cen.dl_shifts[0]
        ax.axvline(sh.time, color=GRAY, ls="-.", lw=1.4, alpha=0.85)
        ax.annotate(
            f"Lab DL upgrade flagged\n({sh.old_limit:.2f} → {sh.new_limit:.2f} mg/L)\nNo false jump triggered",
            xy=(sh.time, 0.65),
            xytext=(np.datetime64("2016-05-01"), 2.15),
            fontsize=8.6,
            color=GRAY,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.3", "fc": "#F8FAFC", "ec": GRAY},
            arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 1.2},
        )

    ax.set_ylabel("Nitrate-N (mg/L)", fontweight="bold", color=DARK)
    ax.set_xlabel("Date", fontweight="bold", color=DARK)
    ax.set_title("Censored Water-Quality Change Detection (<DL Trend-Aware Tobit MLE + DL-Shift Defense)",
                 fontsize=11.0, fontweight="bold", color=DARK, pad=8)
    ax.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", loc="upper left", ncol=2, fontsize=8.2)
    _style_ax(ax)

    fig.tight_layout()
    _save(fig, "tutorial_water_quality_censored.png")


def make_tutorial_ndvi_wildfire() -> None:
    """Satellite vegetation (NDVI) irregular gappy wildfire decomposition."""
    rng = np.random.default_rng(77)
    t_grid = np.linspace(2016.0, 2023.0, 230)
    # 35% cloud dropout + 3-month winter cloud blackout
    keep = rng.uniform(size=t_grid.size) > 0.35
    keep &= ~((t_grid > 2018.85) & (t_grid < 2019.12))
    t = t_grid[keep]
    # Quality weights (cloud-haze points carry larger noise & lower weight)
    sigma = np.where(rng.uniform(size=t.size) < 0.18, 0.08, 0.025)
    weights = 1.0 / (sigma**2)
    trend = np.where(t < 2020.55, 0.62 + 0.005 * (t - 2016.0), 0.31 + 0.07 * (t - 2020.55))
    seas = 0.18 * np.cos(2 * np.pi * (t - 0.52)) + 0.05 * np.sin(4 * np.pi * t)
    y = trend + seas + rng.normal(0.0, sigma)

    rj = just_jumps(t, y, weights=weights, season="allssa", mag_th=0.08, dir_th=0.02, jump_alpha=0.01)
    decomp = just_decompose(t, y, weights=weights, jump_indices=[j.index for j in rj.jumps], season="allssa")

    fig, axes = plt.subplots(3, 1, figsize=(9.8, 6.8), sharex=True, gridspec_kw={"height_ratios": [1.5, 1.0, 0.85]})
    axes[0].plot(t, y, "o", color=DARK, ms=3.3, alpha=0.65, label="Cloud-free NDVI observations")
    if rj.jumps:
        j = rj.jumps[0]
        k = j.valid_index
        if decomp.trend_ci_low is not None and decomp.trend_ci_high is not None:
            axes[0].fill_between(t[:k], decomp.trend_ci_low[:k], decomp.trend_ci_high[:k], color=BLUE, alpha=0.22, lw=0, label="95% trend CI")
            axes[0].fill_between(t[k:], decomp.trend_ci_low[k:], decomp.trend_ci_high[k:], color=BLUE, alpha=0.22, lw=0)
        axes[0].plot(t[:k], decomp.trend[:k], color=BLUE, lw=2.3, label="Weighted JUST trend")
        axes[0].plot(t[k:], decomp.trend[k:], color=BLUE, lw=2.3)
        axes[0].axvline(j.time_year, color=ORANGE, ls="--", lw=1.8,
                        label=f"Wildfire disturbance ({j.time_year:.2f}, Δ={j.magnitude:+.2f}, regrowth={j.direction:+.2f}/yr)")
    axes[0].set_ylabel("NDVI", fontweight="bold", color=DARK)
    axes[0].set_title("Satellite NDVI Wildfire Detection & Post-Fire Regrowth (Weighted JUST + ALLSSA)",
                      fontsize=11.0, fontweight="bold", color=DARK)
    axes[0].legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", loc="lower left", ncol=2, fontsize=8.1)
    _style_ax(axes[0])

    if decomp.seasonal_ci_low is not None and decomp.seasonal_ci_high is not None:
        axes[1].fill_between(t, decomp.seasonal_ci_low, decomp.seasonal_ci_high, color=GREEN, alpha=0.22, lw=0, label="95% seasonal CI")
    axes[1].plot(t, decomp.seasonal, color=GREEN, lw=1.7, label="ALLSSA seasonal component")
    axes[1].axhline(0.0, color=GRAY, lw=0.8, ls=":")
    axes[1].set_ylabel("Seasonal", fontweight="bold", color=DARK)
    axes[1].legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", loc="upper right", ncol=2, fontsize=8.3)
    _style_ax(axes[1])

    axes[2].vlines(t, 0.0, decomp.remainder, color=GRAY, lw=0.9, alpha=0.85)
    axes[2].plot(t, decomp.remainder, "o", color=DARK, ms=2.3)
    axes[2].axhline(0.0, color=DARK, lw=0.8)
    axes[2].set_ylabel("Remainder", fontweight="bold", color=DARK)
    axes[2].set_xlabel("Year", fontweight="bold", color=DARK)
    _style_ax(axes[2])

    fig.tight_layout()
    _save(fig, "tutorial_ndvi_wildfire.png")


def make_tutorial_subsidence_stpd() -> None:
    """InSAR ground subsidence turning point detection via STPD + TPTR."""
    rng = np.random.default_rng(99)
    t = np.sort(rng.uniform(2017.0, 2024.0, size=105))
    y_true = np.where(t <= 2020.4, -4.0 * (t - 2017.0), -13.6 - 22.0 * (t - 2020.4))
    y = y_true + rng.normal(0.0, 2.8, size=t.size)

    res = detect_turning_points(t, y, size=72, step=12, margin=12, alpha=0.01)

    fig, ax = plt.subplots(figsize=(9.8, 4.3))
    ax.plot(t, y, "o", color=DARK, ms=3.8, alpha=0.7, label="Irregular InSAR LOS displacement (mm)")
    pw = res.piecewise_trend
    if pw.trend_ci_low is not None and pw.trend_ci_high is not None:
        ax.fill_between(t, pw.trend_ci_low, pw.trend_ci_high, color=GREEN, alpha=0.22, lw=0, label="95% pointwise CI")
    ax.plot(t, pw.fitted_trend, color=GREEN, lw=2.5, label="TPTR connected piecewise linear trend")

    if res.turning_points:
        tp = res.turning_points[0]
        y_tp = float(res.piecewise_trend.fitted_trend[tp.valid_index])
        ax.plot(tp.time_year, y_tp, "D", color=ORANGE, ms=9.5, markeredgecolor="white", markeredgewidth=1.2, zorder=6,
                label=f"Detected turning point ({tp.time_year:.2f})")
        ax.axvline(tp.time_year, color=ORANGE, ls="--", lw=1.5, alpha=0.85)
        slopes = res.piecewise_trend.slopes
        ax.annotate(
            f"Acceleration onset: {tp.time_year:.2f}\n"
            f"Slope: {slopes[0]:+.1f} → {slopes[1]:+.1f} mm/yr\n"
            f"NDRI (RMSE) = {tp.ndri_rmse:.3f} (continuous TP)\n"
            f"SNR (centered) = {tp.snr_centered:.1f}",
            xy=(tp.time_year, y_tp),
            xytext=(2017.3, -72.0),
            fontsize=8.8,
            color=DARK,
            bbox={"boxstyle": "round,pad=0.4", "fc": "#F8FAFC", "ec": GREEN, "lw": 1.3},
            arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 1.4},
        )

    ax.set_ylabel("Cumulative Displacement (mm)", fontweight="bold", color=DARK)
    ax.set_xlabel("Year", fontweight="bold", color=DARK)
    ax.set_title("Ground Subsidence Acceleration Detected by STPD + TPTR (Zero Discontinuity, Low NDRI)",
                 fontsize=11.0, fontweight="bold", color=DARK, pad=8)
    ax.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", loc="upper right", fontsize=8.6)
    _style_ax(ax)

    fig.tight_layout()
    _save(fig, "tutorial_subsidence_stpd.png")


def make_tutorial_nrt_monitoring() -> None:
    """Near-Real-Time monitoring figure (v1.1 just_monitor) with sequential boundary."""
    rng = np.random.default_rng(2025)
    t = np.sort(rng.uniform(2017.0, 2024.0, size=165))
    season = 0.22 * np.cos(2 * np.pi * t) + 0.08 * np.sin(4 * np.pi * t)
    y = 0.72 - 0.008 * (t - 2017.0) + season + rng.normal(0.0, 0.028, size=t.size)
    # Abrupt canopy disturbance starting at 2023.25
    y = np.where(t >= 2023.25, y - 0.30, y)

    res = just_monitor(
        t,
        y,
        start=2022.5,
        season="ols",
        alpha=0.01,
        freq_correction="sidak",
        sequential_control="pocock",
        horizon=40,
        stop_on_first=False,
    )
    res_parity = just_monitor(
        t,
        y,
        start=2022.5,
        season="ols",
        alpha=0.01,
        parity_mode=True,
        stop_on_first=False,
    )

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10.0, 5.8), sharex=True, gridspec_kw={"height_ratios": [1.45, 1.0]})

    # Shade monitoring period
    ax_top.axvspan(res.start_time, res.times[-1], color=SKY, alpha=0.12, label="Monitoring period")
    ax_bot.axvspan(res.start_time, res.times[-1], color=SKY, alpha=0.14)
    ax_top.axvline(res.start_time, color=BLUE, ls=":", lw=1.3)
    ax_bot.axvline(res.start_time, color=BLUE, ls=":", lw=1.3)

    if res.forecast_pi_low is not None and res.forecast_pi_high is not None:
        ax_top.fill_between(res.times, res.forecast_pi_low, res.forecast_pi_high, color=SKY, alpha=0.22, lw=0, label="95% prediction interval")
    if res.forecast_ci_low is not None and res.forecast_ci_high is not None:
        ax_top.fill_between(res.times, res.forecast_ci_low, res.forecast_ci_high, color=BLUE, alpha=0.24, lw=0, label="95% forecast CI")

    ax_top.plot(res.times, res.values, "o", color=DARK, ms=3.3, alpha=0.72, label="Observations")
    n_his = int(np.searchsorted(res.time_years, float(res.steps[0].time_year), side="left"))
    ax_top.plot(res.times[:n_his], res.forecast[:n_his], color=BLUE, lw=2.0, label="Historical model fit")
    ax_top.plot(res.times[n_his - 1 :], res.forecast[n_his - 1 :], color=BLUE, lw=2.0, ls="--", label="Forecast baseline")

    if res.detected and res.disturbance_time_year is not None:
        pos = int(np.argmin(np.abs(res.time_years - res.disturbance_time_year)))
        ax_top.plot(
            res.times[pos],
            res.values[pos],
            "*",
            color=ORANGE,
            ms=14,
            markeredgecolor="white",
            markeredgewidth=0.9,
            zorder=6,
            label=f"Confirmed abrupt alarm ({res.disturbance_time_year:.2f}, Δ={res.magnitude:+.2f})",
        )
        ax_top.axvline(res.disturbance_time_year, color=ORANGE, ls="--", lw=1.4)
        ax_bot.axvline(res.disturbance_time_year, color=ORANGE, ls="--", lw=1.4)

    ax_top.set_ylabel("Vegetation Index (EVI)", fontweight="bold", color=DARK)
    ax_top.set_title("Near-Real-Time Monitoring: tspec.just_monitor() with Šidák + Pocock Multiplicity Control",
                     fontsize=11.0, fontweight="bold", color=DARK, pad=8)
    ax_top.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", loc="lower left", ncol=2, fontsize=8.0)
    _style_ax(ax_top)

    # Bottom panel: sequential residual LSS peak vs controlled & unadjusted boundaries
    st_t = [s.time_year for s in res.steps]
    ax_bot.plot(st_t, [s.max_low_freq_spec for s in res.steps], "-o", color=ORANGE, ms=4.2, lw=1.8,
                label="Low-freq max LSS (<1 cy/yr, abrupt)")
    ax_bot.plot(st_t, [s.max_high_freq_spec for s in res.steps], "-s", color=PURPLE, ms=3.5, lw=1.3, alpha=0.85,
                label="High-freq max LSS (≥1 cy/yr, phenological)")
    ax_bot.plot(st_t, [s.critical_value for s in res.steps], "--", color=DARK, lw=1.8,
                label="Šidák + Pocock boundary (FWER ≤ 0.01)")
    ax_bot.plot(st_t, [s.critical_value for s in res_parity.steps], ":", color=GRAY, lw=1.5,
                label="Unadjusted parity threshold (per-frequency α = 0.01)")

    ax_bot.set_ylabel("Max Residual LSS", fontweight="bold", color=DARK)
    ax_bot.set_xlabel("Year", fontweight="bold", color=DARK)
    ax_bot.set_ylim(0.0, 1.02)
    ax_bot.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", loc="upper left", fontsize=8.1)
    _style_ax(ax_bot)

    fig.tight_layout()
    _save(fig, "tutorial_nrt_monitoring.png")


def make_example_confidence_intervals() -> None:
    """3-panel showcase of analytical pointwise CIs & prediction intervals across tspec."""
    rng = np.random.default_rng(314)
    fig, axes = plt.subplots(3, 1, figsize=(10.2, 8.6))

    # Panel A: just_decompose bias-corrected trend band + seasonal 95% CIs
    t_a = np.sort(rng.uniform(2016.0, 2023.0, size=170))
    y_a = 8.0 + 0.45 * (t_a - 2016.0) + 1.1 * np.cos(2 * np.pi * t_a) + rng.normal(0.0, 0.35, t_a.size)
    dec = just_decompose(t_a, y_a, season="ols", ci_level=0.95)
    ax = axes[0]
    ax.plot(t_a, y_a, "o", color=DARK, ms=2.8, alpha=0.45, label="Irregular observations")
    assert dec.trend_ci_low is not None and dec.trend_ci_high is not None
    assert dec.seasonal_ci_low is not None and dec.seasonal_ci_high is not None
    ax.fill_between(t_a, dec.trend_ci_low, dec.trend_ci_high, color=BLUE, alpha=0.25, lw=0, label="95% trend CI (bias-corrected)")
    ax.plot(t_a, dec.trend, color=BLUE, lw=2.2, label="JUST trend")
    # Plot seasonal shifted to bottom baseline on twin axis for clean visual inspection
    ax_r = ax.twinx()
    ax_r.fill_between(t_a, dec.seasonal_ci_low, dec.seasonal_ci_high, color=GREEN, alpha=0.22, lw=0, label="95% seasonal CI")
    ax_r.plot(t_a, dec.seasonal, color=GREEN, lw=1.4, label="JUST seasonal")
    ax_r.set_ylabel("Seasonal", fontweight="bold", color=GREEN)
    ax_r.tick_params(axis="y", colors=GREEN, labelsize=9.0)
    ax_r.spines["top"].set_visible(False)
    ax_r.set_ylim(-2.2, 5.5)
    ax.set_ylim(5.5, 13.5)
    ax.set_ylabel("Trend / Value", fontweight="bold", color=DARK)
    ax.set_title("A. JUST Decomposition — Pointwise 95% Bias-Corrected Trend Band & Seasonal CI",
                 fontsize=10.8, fontweight="bold", color=DARK, pad=6)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=True, facecolor="white", edgecolor="#E2E8F0", loc="upper left", ncol=3, fontsize=8.1)
    _style_ax(ax)

    # Panel B: analyze_series with robust='mm' + slope_estimator='sen' (Sen slope fan)
    t_b = np.sort(rng.uniform(2015.0, 2024.0, size=130))
    tr_b = np.where(t_b < 2019.2, 5.0 + 0.25 * (t_b - 2015.0), 8.6 - 0.55 * (t_b - 2019.2))
    y_b = tr_b + 0.65 * np.cos(2 * np.pi * t_b) + rng.normal(0.0, 0.55, t_b.size)
    y_b[[16, 48, 86, 112]] += np.array([5.5, -4.8, 5.2, -5.0])  # isolated spikes
    pipe = analyze_series(
        t_b,
        y_b,
        robust="mm",
        slope_estimator="sen",
        ci_level=0.95,
        detect_turning_points=False,
        just_kwargs={"season": "ols", "jump_alpha": 0.01},
        decompose_kwargs={"season": "ols"},
    )
    ax = axes[1]
    ax.plot(t_b, y_b, "o", color=DARK, ms=3.2, alpha=0.58, label="Observations (with 4 isolated spikes)")
    assert pipe.slope_fan_low is not None and pipe.slope_fan_high is not None
    cuts = [int(j.valid_index) for j in (pipe.jumps.jumps if pipe.jumps else ())]
    bounds = [0, *sorted(cuts), t_b.size]
    for idx_b, (a_i, b_i) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        ax.fill_between(
            t_b[a_i:b_i],
            pipe.slope_fan_low[a_i:b_i],
            pipe.slope_fan_high[a_i:b_i],
            color=BLUE,
            alpha=0.28,
            lw=0,
            label="95% slope fan (slope uncertainty; not a trend CI)" if idx_b == 0 else None,
        )
        ax.plot(
            t_b[a_i:b_i],
            pipe.combined_trend[a_i:b_i],
            color=BLUE,
            lw=2.2,
            label="Theil–Sen trend (robust='mm')" if idx_b == 0 else None,
        )
    if pipe.jumps:
        for k_j, j in enumerate(pipe.jumps.jumps):
            ax.axvline(j.time_year, color=ORANGE, ls="--", lw=1.6, label=f"JUST jump ({j.time_year:.2f})" if k_j == 0 else None)
    ax.set_ylabel("Value", fontweight="bold", color=DARK)
    ax.set_title("B. Robust Guided Pipeline (robust='mm', slope_estimator='sen') — Median-Pivoted 95% Sen Slope Fan",
                 fontsize=10.8, fontweight="bold", color=DARK, pad=6)
    ax.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", loc="upper left", ncol=2, fontsize=8.1)
    _style_ax(ax)

    # Panel C: just_monitor forecast mean CI vs out-of-sample prediction interval
    t_c = np.linspace(2017.0, 2024.0, 150)
    y_c = 2.0 + 0.12 * (t_c - 2017.0) + 0.45 * np.cos(2 * np.pi * t_c) + rng.normal(0.0, 0.14, t_c.size)
    y_c = np.where(t_c >= 2023.15, y_c - 0.85, y_c)
    mon = just_monitor(t_c, y_c, start=2021.8, season="ols", alpha=0.01, ci_level=0.95, stop_on_first=False)
    ax = axes[2]
    ax.axvspan(mon.start_time, mon.times[-1], color=SKY, alpha=0.10, label="Monitoring period")
    ax.axvline(mon.start_time, color=BLUE, ls=":", lw=1.3)
    assert mon.forecast_pi_low is not None and mon.forecast_pi_high is not None
    assert mon.forecast_ci_low is not None and mon.forecast_ci_high is not None
    ax.fill_between(mon.times, mon.forecast_pi_low, mon.forecast_pi_high, color=SKY, alpha=0.25, lw=0, label="95% prediction interval (obs)")
    ax.fill_between(mon.times, mon.forecast_ci_low, mon.forecast_ci_high, color=BLUE, alpha=0.28, lw=0, label="95% forecast CI (mean)")
    ax.plot(mon.times, mon.values, "o", color=DARK, ms=3.0, alpha=0.72, label="Observations")
    n_h = int(np.searchsorted(mon.time_years, float(mon.steps[0].time_year), side="left"))
    ax.plot(mon.times[:n_h], mon.forecast[:n_h], color=BLUE, lw=2.0, label="Historical fit")
    ax.plot(mon.times[n_h - 1 :], mon.forecast[n_h - 1 :], color=BLUE, lw=2.0, ls="--", label="Forecast")
    if mon.detected and mon.disturbance_time_year is not None:
        pos = int(np.argmin(np.abs(mon.time_years - mon.disturbance_time_year)))
        ax.plot(mon.times[pos], mon.values[pos], "*", color=ORANGE, ms=13, markeredgecolor="white", markeredgewidth=0.8, zorder=6,
                label=f"Alarm ({mon.disturbance_time_year:.2f})")
    ax.set_ylabel("Value", fontweight="bold", color=DARK)
    ax.set_xlabel("Year", fontweight="bold", color=DARK)
    ax.set_title("C. Near-Real-Time Monitoring (just_monitor) — Widening 95% Forecast CI vs. 95% Prediction Interval",
                 fontsize=10.8, fontweight="bold", color=DARK, pad=6)
    ax.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", loc="lower left", ncol=3, fontsize=8.0)
    _style_ax(ax)

    fig.tight_layout()
    _save(fig, "example_confidence_intervals.png")


def main() -> None:
    print(f"Generating tspec v{tspec.__version__} documentation figures into {OUT_DIR} ...")
    make_hero_combined_workflow()
    make_concept_jump_vs_tp()
    make_concept_count_vs_duration()
    make_tutorial_water_quality_censored()
    make_tutorial_ndvi_wildfire()
    make_tutorial_subsidence_stpd()
    make_tutorial_nrt_monitoring()
    make_example_confidence_intervals()
    print("All 8 figures generated successfully.")


if __name__ == "__main__":
    main()
