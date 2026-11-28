"""Figures for docs/tutorials/decomposition-uneven-sampling.md.

    OMP_NUM_THREADS=1 MPLBACKEND=Agg python docs/scripts/tutorial_decomposition.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from statsmodels.tsa.seasonal import STL  # noqa: E402

import tspec  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "figures"
INK, INK2 = "#191919", "#666666"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.edgecolor": INK2})


def load():
    d = pd.read_csv(ROOT / "validation" / "data" / "choptank_01491000_nitrate.csv", parse_dates=["date"])
    d = d[d["date"] >= "1985-01-01"].reset_index(drop=True)
    return d


def fig_choptank(d):
    y = np.log(d["value"].to_numpy())
    status = np.where(d["censored"], -1, 0)
    weekly = pd.date_range(d["date"].iloc[0].ceil("D"), d["date"].iloc[-1].floor("D"), freq="7D").to_numpy()
    res = tspec.just_decompose(d["date"].to_numpy(), y, season="ols", censor_status=status, eval_times=weekly)
    t = res.times_numeric
    g = res.eval_times_numeric
    storm = (d["event"] != "Routine sample").to_numpy()
    fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    ax[0].plot(t[~storm], y[~storm], ".", ms=2.5, color="#9e9e9e", label="routine sample")
    ax[0].plot(t[storm], y[storm], ".", ms=2.5, color=INK, label="storm / event sample")
    ax[0].plot(g, res.trend_reg + res.seasonal_reg, color=BLUE, lw=0.7, label="trend + seasonal (weekly grid)")
    ax[0].set_ylabel("ln(NO3+NO2, mg/L)")
    ax[0].legend(fontsize=7, frameon=False, ncol=3)
    ax[1].fill_between(g, res.trend_reg_ci_low, res.trend_reg_ci_high, color=BLUE, alpha=0.2, lw=0, label="95% band")
    ax[1].plot(g, res.trend_reg, color=BLUE, lw=1.2, label="trend")
    ax[1].set_ylabel("trend")
    ax[1].legend(fontsize=7, frameon=False)
    ax[2].fill_between(g, res.seasonal_reg_ci_low, res.seasonal_reg_ci_high, color=BLUE, alpha=0.2, lw=0)
    ax[2].plot(g, res.seasonal_reg, color=BLUE, lw=0.8)
    ax[2].set_ylabel("seasonal")
    ax[3].plot(t, res.remainder, ".", ms=2, color=INK2)
    ax[3].axhline(0, color=INK, lw=0.6)
    ax[3].set_ylabel("remainder")
    ax[3].set_xlabel("year")
    fig.suptitle("Choptank River near Greensboro, MD (USGS 01491000): just_decompose(season='ols')", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "tutorial_decomp_choptank.png", dpi=140)
    plt.close(fig)
    return res


def fig_seasonal_window(d):
    y = np.log(d["value"].to_numpy())
    status = np.where(d["censored"], -1, 0)
    t_dates = d["date"].to_numpy()
    runs = {"seasonal_window=None (published single 3-yr window)": (None, MAGENTA),
            "seasonal_window='auto' (7 years, default)": ("auto", BLUE),
            "seasonal_window=40 (one shape for the whole record)": (40.0, YELLOW)}
    days = {yr: pd.date_range(f"{yr}-01-01", f"{yr}-12-31", freq="3D").to_numpy() for yr in (1995, 2018)}
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    for name, (sw, col) in runs.items():
        r = tspec.just_decompose(t_dates, y, season="ols", censor_status=status, seasonal_window=sw)
        ax[0].plot(r.times_numeric, r.trend, color=col, lw=1.1, label=name)
        if sw is None:
            big = float(np.max(np.abs(r.trend)))
            ax[0].annotate(f"single window: excursion to ±{big:.0f}\nduring the 1989 storm burst (off-scale)",
                           (1989.5, 0.55), fontsize=7, color=INK2, ha="left", va="top")
        for k, yr in enumerate((1995, 2018)):
            e = tspec.just_decompose(t_dates, y, season="ols", censor_status=status, seasonal_window=sw,
                                     eval_times=days[yr])
            ax[1 + k].plot(np.mod(e.eval_times_numeric, 1.0), e.seasonal_reg, color=col, lw=1.2)
    ax[0].set_ylim(-0.6, 0.6)
    ax[0].set_title("trend", fontsize=9)
    ax[0].set_xlabel("year")
    ax[0].legend(fontsize=7, frameon=False, loc="lower right")
    for k, yr in enumerate((1995, 2018)):
        ax[1 + k].set_title(f"seasonal cycle in {yr}", fontsize=9)
        ax[1 + k].set_xlabel("fraction of year")
        ax[1 + k].set_ylim(-0.5, 0.5)
    fig.tight_layout()
    fig.savefig(OUT / "tutorial_decomp_seasonal_window.png", dpi=140)
    plt.close(fig)


def fig_vs_stl():
    rng = np.random.default_rng(3)
    m = np.arange(240)
    t = 2000 + (m + 0.5) / 12
    keep = rng.random(240) < np.where(np.isin(m % 12, [11, 0, 1]), 0.25, 0.95)  # winter-sparse
    storms = [2000 + yr + rng.uniform(0.2, 0.6) + rng.uniform(0, 4 / 365.25, 5) for yr in range(20)]
    t = np.unique(np.concatenate([t[keep], *storms]))
    trend = 1.5 + 0.02 * (t - 2000) - 0.4 * np.tanh((t - 2010) / 2.5)
    seas = 0.6 * np.cos(2 * np.pi * (t - 0.1)) + 0.25 * np.cos(4 * np.pi * (t - 0.3))
    y = trend + seas + rng.normal(0, 0.25, t.size)
    r = tspec.just_decompose(t, y, season="ols")
    grid = 2000 + (np.arange(240) + 0.5) / 12
    k = np.floor((t - 2000) * 12).astype(int)
    cnt = np.bincount(k, minlength=240)
    gy = np.bincount(k, weights=y, minlength=240) / np.maximum(cnt, 1)
    gy[cnt == 0] = np.interp(grid[cnt == 0], grid[cnt > 0], gy[cnt > 0])
    s = STL(gy, period=12).fit()
    st_tr, st_se = np.interp(t, grid, s.trend), np.interp(t, grid, s.seasonal)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
    ax[0].plot(t, y - seas, ".", ms=2, color="#bdbdbd", label="data minus true seasonal")
    ax[0].plot(t, trend, color=INK, lw=1.6, label="truth")
    ax[0].plot(t, r.trend, color=BLUE, lw=1.1, label=f"tspec (RMSE {np.sqrt(np.mean((r.trend - trend) ** 2)):.3f})")
    ax[0].plot(t, st_tr, color=ORANGE, lw=1.1, label=f"STL on monthly means (RMSE {np.sqrt(np.mean((st_tr - trend) ** 2)):.3f})")
    ax[0].legend(fontsize=7, frameon=False)
    ax[0].set_title("trend: winter-sparse sampling with storm bursts", fontsize=9)
    z = t >= 2016
    ax[1].plot(t[z], seas[z], color=INK, lw=1.6, label="truth")
    ax[1].plot(t[z], r.seasonal[z], color=BLUE, lw=1, label=f"tspec (RMSE {np.sqrt(np.mean((r.seasonal - seas) ** 2)):.3f})")
    ax[1].plot(t[z], st_se[z], color=ORANGE, lw=1, label=f"STL (RMSE {np.sqrt(np.mean((st_se - seas) ** 2)):.3f})")
    ax[1].legend(fontsize=7, frameon=False)
    ax[1].set_title("seasonal, 2016-2020", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "tutorial_decomp_vs_stl.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    d = load()
    res = fig_choptank(d)
    fig_seasonal_window(d)
    fig_vs_stl()
    print([w.code for w in res.diagnostics.warnings])
