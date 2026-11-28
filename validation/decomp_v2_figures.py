"""Figures and per-replicate metrics for decomposition validation 2, with a WRTDS-style
comparator added after the registered run (exploratory; it does not change tspec).

    OMP_NUM_THREADS=1 MPLBACKEND=Agg python validation/decomp_v2_figures.py synthetic
    OMP_NUM_THREADS=1 MPLBACKEND=Agg python validation/decomp_v2_figures.py real

Same designs and seeds as ``bench_decomp_heldout_2.py``. Outputs in
validation/reports/figures_decomp_v2/ and validation/reports/decomp_v2_metrics.json.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_decomp_heldout as B  # noqa: E402
import bench_decomp_heldout_2 as C  # noqa: E402
import wrtds_like as W  # noqa: E402

import tspec  # noqa: E402

warnings.filterwarnings("ignore")
FIG = B.REPORTS / "figures_decomp_v2"
OUT = B.REPORTS / "decomp_v2_metrics.json"
NREP = 100
METHODS = ["tspec_ols", "tspec_legacy_ols", "stl_bin", "wrtds", "harm_lowess"]
LABEL = {"tspec_ols": "tspec (seasonal window, new)", "tspec_legacy_ols": "tspec (single window, old)",
         "stl_bin": "STL (monthly means)", "wrtds": "WRTDS-style (no discharge)", "harm_lowess": "global harmonics + LOWESS"}
COLOR = {"tspec_ols": "#257de0", "stl_bin": "#f47230", "wrtds": "#00b894", "harm_lowess": "#f2a600", "tspec_legacy_ols": "#ee7fac"}
MARK = {"tspec_ols": "o", "stl_bin": "s", "wrtds": "D", "harm_lowess": "^", "tspec_legacy_ols": "v"}
INK, INK2 = "#191919", "#666666"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6})


def series(c, rep):
    rng = np.random.default_rng(940_000 + 1_000 * c["id"] + rep)
    t = C.sample_times(c["q"], rng)
    t0, L = float(t[0]), float(t[-1] - t[0])
    tr, se = C.TR[c["tr"]](t, t0, L), C.SE[c["se"]](t, t0, L)
    e = C.noise(c["noise"], t, rng)
    return t, tr, se, e, tr + se + e


def fit(method, t, y):
    if method == "tspec_ols":
        d = tspec.just_decompose(t, y, season="ols")
        return d.trend, d.seasonal
    if method == "tspec_legacy_ols":
        d = tspec.just_decompose(t, y, season="ols", seasonal_window=None)
        return d.trend, d.seasonal
    if method == "wrtds":
        r = W.run_wrtds(t, y)
    elif method == "harm_lowess":
        r = B.run_harm_lowess(t, y)
    else:
        r = B.run_stl(t, y, "bin")
    return r["trend"], r["seasonal"]


def job(args):
    c, rep = args
    t, tr, se, e, y = series(c, rep)
    out = {}
    for m in METHODS:
        T, S = fit(m, t, y)
        ok = np.isfinite(T) & np.isfinite(S)
        out[m] = (B.rmse(T[ok], tr[ok]), B.rmse(S[ok], se[ok]), B.rmse((y - T - S)[ok], e[ok]))
    return out


def synthetic():
    cells = C.build_cells()
    per = {}
    for c in cells:
        with mp.get_context("fork").Pool(B.PROCS) as p:
            rows = p.map(job, [(c, r) for r in range(NREP)])
        per[c["label"]] = {m: np.array([r[m] for r in rows]).tolist() for m in METHODS}
        med = {m: np.round(np.median(np.array(per[c["label"]][m]), axis=0), 3).tolist() for m in METHODS}
        print(c["label"], med, flush=True)
    OUT.write_text(json.dumps({"per_rep": per}, indent=0))
    figures_synthetic(per, cells)


def figures_synthetic(per, cells):
    FIG.mkdir(parents=True, exist_ok=True)
    labels = [c["label"] for c in cells]
    names = ["trend", "seasonal", "remainder"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.36 * len(labels) + 1.6), sharey=True)
    xmax = 0.32
    for j, ax in enumerate(axes):
        for i, lab in enumerate(labels):
            for k, m in enumerate(METHODS):
                v = np.array(per[lab][m])[:, j]
                q1, md, q3 = np.percentile(v, [25, 50, 75])
                yv = i + (k - 2) * 0.15
                if md > xmax:
                    ax.annotate(f"{md:.1f} →", (xmax, yv), fontsize=6, color=INK2, ha="right", va="center")
                    continue
                ax.plot([q1, min(q3, xmax)], [yv, yv], color=COLOR[m], lw=1.2, solid_capstyle="round")
                ax.plot(md, yv, MARK[m], color=COLOR[m], ms=4.5, mec="white", mew=0.6)
        ax.set_xlim(0, xmax)
        ax.set_title(f"{names[j]} RMSE (median, IQR over 100 runs)", color=INK, fontsize=9)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
    handles = [plt.Line2D([], [], color=COLOR[m], marker=MARK[m], lw=1.2, label=LABEL[m]) for m in METHODS]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8)
    fig.suptitle("Decomposition error by sampling design (validation 2 designs; Q = sampling, U = trend, V = seasonal, W/O = noise)",
                 color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(FIG / "rmse_by_design.png", dpi=150)
    plt.close(fig)

    # example decompositions for one replicate in three hard regimes
    ex = [("Q3_U1V1_W", "monthly + storm clusters"), ("Q4_U1V1_W", "winter-sparse monthly"), ("Q6_U2V3_W", "real Conowingo sampling dates")]
    cmap = {c["label"]: c for c in cells}
    fig, axes = plt.subplots(len(ex), 2, figsize=(13, 3.0 * len(ex)), gridspec_kw={"width_ratios": [1.6, 1]})
    for i, (lab, title) in enumerate(ex):
        t, tr, se, e, y = series(cmap[lab], 0)
        a, b = axes[i]
        a.plot(t, y - se, ".", ms=1.8, color="#bdbdbd", label="data minus true seasonal")
        a.plot(t, tr, color=INK, lw=1.6, label="truth")
        zoom = (t >= t[-1] - 4)
        b.plot(t[zoom], se[zoom], color=INK, lw=1.6, label="truth")
        for m in ["tspec_ols", "stl_bin", "wrtds", "harm_lowess"]:
            T, S = fit(m, t, y)
            a.plot(t, T, color=COLOR[m], lw=1.1, label=LABEL[m])
            b.plot(t[zoom], S[zoom], color=COLOR[m], lw=1.0, label=LABEL[m])
        a.set_title(f"{title}: trend ({lab})", fontsize=9, color=INK)
        b.set_title("seasonal, last 4 years", fontsize=9, color=INK)
        if i == 0:
            a.legend(fontsize=7, ncol=3, frameon=False, loc="upper left")
    axes[-1, 0].set_xlabel("year")
    axes[-1, 1].set_xlabel("year")
    fig.tight_layout()
    fig.savefig(FIG / "example_decompositions.png", dpi=150)
    plt.close(fig)


def real():
    FIG.mkdir(parents=True, exist_ok=True)
    res2 = json.loads((B.REPORTS / "decomp_heldout_2_results.json").read_text())["real"]
    recs = {"Conowingo (USGS 01578310)": (C.conowingo(), 950_000, "cono"),
            "Choptank (USGS 01491000)": (B.choptank(), 920_000, "chop")}
    cv_rows = {}
    fig, axes = plt.subplots(3, 2, figsize=(13, 9))
    for j, (name, (d, seed0, key)) in enumerate(recs.items()):
        t = d["t"].to_numpy()
        y = np.log(d["value"].to_numpy())
        cens = d["censored"].to_numpy().astype(bool)
        y_sub = np.where(cens, np.log(0.5 * d["value"].to_numpy()), y)
        kw = {"season": "ols"}
        if cens.any():
            kw["censor_status"] = np.where(cens, -1, 0)
        dd = tspec.just_decompose(t, y, **kw)
        fits = {"tspec_ols": (dd.trend, dd.seasonal), "stl_bin": None, "wrtds": None, "harm_lowess": None}
        for m in ["stl_bin", "wrtds", "harm_lowess"]:
            fits[m] = fit(m, t, y_sub)
        storm = (d["event"] != "Routine sample").to_numpy()
        ax = axes[0, j]
        ax.plot(t[~storm], y[~storm], ".", ms=2, color="#9e9e9e", label="routine sample")
        ax.plot(t[storm], y[storm], ".", ms=2, color=INK, label="storm/event sample")
        ax.set_title(f"{name}: ln(nitrate + nitrite, mg/L N)", fontsize=9, color=INK)
        ax.legend(fontsize=7, frameon=False, ncol=2)
        ax = axes[1, j]
        ax.fill_between(t, dd.trend_ci_low, dd.trend_ci_high, color=COLOR["tspec_ols"], alpha=0.18, lw=0, label="tspec 95% band")
        for m in ["tspec_ols", "stl_bin", "wrtds", "harm_lowess"]:
            ax.plot(t, fits[m][0], color=COLOR[m], lw=1.1, label=LABEL[m])
        ax.set_title("trend", fontsize=9, color=INK)
        if j == 0:
            ax.legend(fontsize=7, frameon=False, ncol=2)
        ax = axes[2, j]
        z = t >= t[-1] - 6
        for m in ["tspec_ols", "stl_bin", "wrtds", "harm_lowess"]:
            ax.plot(t[z], fits[m][1][z], color=COLOR[m], lw=1.0, marker=MARK[m], ms=2.5, label=LABEL[m])
        ax.set_title("seasonal component, last 6 years", fontsize=9, color=INK)
        ax.set_xlabel("year")
        # WRTDS cross-validation with the registered splits
        for sub in ("all", "routine"):
            dsub = d if sub == "all" else d[d["event"] == "Routine sample"].reset_index(drop=True)
            ts = dsub["t"].to_numpy()
            ys = np.log(dsub["value"].to_numpy())
            cs = dsub["censored"].to_numpy().astype(bool)
            yss = np.where(cs, np.log(0.5 * dsub["value"].to_numpy()), ys)
            errs = []
            for split in range(20):
                rng = np.random.default_rng(seed0 + split)
                test = rng.random(ts.size) < 0.20
                tr_ = ~test
                score = test & ~cs & (ts >= ts[tr_][0]) & (ts <= ts[tr_][-1])
                f = W.wrtds_surface(ts[tr_], yss[tr_], ts[score])
                errs.append(B.rmse(f, ys[score]))
            row = {m: res2[f"{key}_{sub}"][m] for m in res2[f"{key}_{sub}"]}
            row["wrtds"] = dict(mean=float(np.mean(errs)), sd=float(np.std(errs)))
            cv_rows[f"{name} — {sub} samples"] = row
    fig.tight_layout()
    fig.savefig(FIG / "real_records.png", dpi=150)
    plt.close(fig)

    order = ["tspec_ols", "tspec_legacy_ols", "stl_bin", "wrtds", "harm_lowess"]
    fig, axes = plt.subplots(1, len(cv_rows), figsize=(13, 3.4), sharey=False)
    for ax, (name, row) in zip(axes, cv_rows.items(), strict=True):
        ms = [m for m in order if m in row]
        means = [row[m]["mean"] for m in ms]
        sds = [row[m]["sd"] for m in ms]
        ax.bar(range(len(ms)), means, yerr=sds, color=[COLOR[m] for m in ms], width=0.7, capsize=3,
               error_kw={"ecolor": INK2, "lw": 0.8}, edgecolor="white", linewidth=2)
        lo = min(means) - 1.5 * max(sds)
        ax.set_ylim(max(0, lo), max(m + s for m, s in zip(means, sds, strict=True)) * 1.05)
        ax.set_xticks(range(len(ms)))
        ax.set_xticklabels([LABEL[m].split(" (")[0].replace("global harmonics + LOWESS", "harm+LOWESS") for m in ms],
                           rotation=35, ha="right", fontsize=7)
        ax.set_title(name, fontsize=8, color=INK)
        for k, v in enumerate(means):
            ax.text(k, v, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    axes[0].set_ylabel("held-out RMSE, ln units (mean ± SD, 20 splits)")
    fig.tight_layout()
    fig.savefig(FIG / "real_cv.png", dpi=150)
    plt.close(fig)
    m = json.loads(OUT.read_text()) if OUT.exists() else {}
    m["real_cv"] = cv_rows
    OUT.write_text(json.dumps(m, indent=0))
    print(json.dumps({k: {mm: round(v["mean"], 4) for mm, v in r.items()} for k, r in cv_rows.items()}, indent=1))


if __name__ == "__main__":
    {"synthetic": synthetic, "real": real}[sys.argv[1]]()
