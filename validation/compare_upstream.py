"""Tier 1 Validation: Isolated numerical parity against Ghaderpour/LSWAVE-SignalProcessing.

Loads the unmodified upstream Python scripts from gitignored ``validation/upstream_ref/``
with only runtime Python 3.8+ shims (``time.clock = time.perf_counter``, headless
Matplotlib ``Agg`` backend), compares every bundled JUST and STPD dataset plus
synthetic irregular weighted series (scalar, 1D, and 2D SPD ``P``), and writes
``validation/reports/parity_report.md``.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
if not hasattr(time, "clock"):
    time.clock = time.perf_counter  # type: ignore[attr-defined]

from tspec import (  # noqa: E402
    allssa,
    detect_turning_points,
    fit_piecewise_trend,
    just_decompose,
    just_jumps,
    lssa,
)

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "upstream_ref"
JUST_DIR = UPSTREAM / "JUST_PythonPackage_EGhaderpour"
JUST_MAT_DIR = UPSTREAM / "JUST_MATLABPackage_EGhaderpour"
STPD_DIR = UPSTREAM / "STPD_PythonPackage_EGhaderpour"
REPORT_PATH = ROOT / "reports" / "parity_report.md"


def _load_upstream_module(directory: Path, name: str) -> Any:
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    path = directory / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_upstream_{name}", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_dat(path: Path) -> tuple[np.ndarray, np.ndarray, Any]:
    if path.suffix == ".csv":
        arr = np.loadtxt(path, delimiter=",")
    else:
        with path.open() as fh:
            arr = np.array([line.split() for line in fh if line.strip()], dtype=float)
    t = arr[:, 0]
    f = arr[:, 1]
    P: Any = arr[:, 2] if arr.shape[1] >= 3 else 1
    return t, f, P


def _load_stpd_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    t_list: list[float] = []
    f_list: list[float] = []
    with path.open() as fh:
        reader = csv.reader(fh, skipinitialspace=True)
        next(reader)
        for row in reader:
            if row:
                t_list.append(float(row[0]))
                f_list.append(float(row[1]))
    return np.array(t_list, dtype=float), np.array(f_list, dtype=float)


def _max_err(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.size == 0 and b_arr.size == 0:
        return 0.0, 0.0
    diff = np.abs(a_arr - b_arr)
    abs_err = float(np.max(diff))
    denom = np.maximum(np.abs(b_arr), 1e-12)
    rel_err = float(np.max(diff / denom))
    return abs_err, rel_err


def main() -> int:
    if not UPSTREAM.exists():
        print("Upstream oracle missing. Run `python validation/fetch_upstream.py` first.")
        return 1

    up_lssa = _load_upstream_module(JUST_DIR, "LSSA")
    up_allssa = _load_upstream_module(JUST_DIR, "ALLSSA")
    up_just = _load_upstream_module(JUST_DIR, "JUSTjumps")
    up_decomp = _load_upstream_module(JUST_DIR, "JUSTdecompose")
    up_stpd = _load_upstream_module(STPD_DIR, "STPD")
    up_tptr = _load_upstream_module(STPD_DIR, "TPTR")

    rows: list[dict[str, Any]] = []
    failures: list[str] = []

    def record(dataset: str, method: str, metric: str, ok: bool, detail: str) -> None:
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "metric": metric,
                "status": "PASS" if ok else "FAIL",
                "detail": detail,
            }
        )
        if not ok:
            failures.append(f"{dataset} / {method} / {metric}: {detail}")

    # ------------------------------------------------------------------
    # 1. LSSA & ALLSSA on Sim_Geophysics.dat (Ghaderpour et al., 2018)
    # ------------------------------------------------------------------
    t_g, f_g, _ = _load_dat(JUST_DIR / "Sim_Geophysics.dat")
    t0_g = t_g - t_g[0]
    om_g = np.linspace(1.0, 30.0, 120)
    u_spec, u_crit, _u_cs, u_res, u_nres, u_coeff, u_cov = up_lssa.LSSA(
        t0_g, f_g, P=1, Omega=om_g, level=0.01, trend="linear"
    )
    our_l = lssa(t_g, f_g, omega=om_g, alpha=0.01, trend="linear")
    ae, re = _max_err(our_l.spectrum, u_spec)
    record("Sim_Geophysics.dat", "lssa", "spectrum", np.allclose(our_l.spectrum, u_spec, rtol=1e-6, atol=1e-8), f"max_abs={ae:.2e}, max_rel={re:.2e}")
    ae, re = _max_err(our_l.coeff, u_coeff)
    record("Sim_Geophysics.dat", "lssa", "coeff & crit", np.allclose(our_l.coeff, u_coeff, rtol=1e-6, atol=1e-8) and np.isclose(our_l.critical_value, u_crit, rtol=1e-12), f"max_abs={ae:.2e}")

    for dec in (2, 4):
        u_tr, u_cs, u_r, u_nr, u_cv, u_fr = up_allssa.ALLSSA(t0_g, f_g, P=1, decimal=dec)
        our_a = allssa(t_g, f_g, decimal=dec, parity_mode=True)
        freq_ok = np.allclose(our_a.frequencies, u_fr, atol=1e-12)
        ae_r, _ = _max_err(our_a.residuals, u_r)
        record(
            "Sim_Geophysics.dat",
            f"allssa(decimal={dec})",
            "frequencies & residuals",
            freq_ok and np.allclose(our_a.residuals, u_r, rtol=1e-6, atol=1e-8),
            f"freqs={our_a.frequencies.tolist()} vs {u_fr}, res_abs={ae_r:.2e}",
        )

    # ------------------------------------------------------------------
    # 2. JUSTjumps & JUSTdecompose across all bundled JUST datasets
    # ------------------------------------------------------------------
    just_datasets = [
        ("EVIA_CA.dat", JUST_DIR / "EVIA_CA.dat"),
        ("EVIB_CA.dat", JUST_DIR / "EVIB_CA.dat"),
        ("EVIC_CA.dat", JUST_DIR / "EVIC_CA.dat"),
        ("ExA_NDVI_Australia.dat", JUST_DIR / "ExA_NDVI_Australia.dat"),
        ("ExB_NDVI_Australia.dat", JUST_DIR / "ExB_NDVI_Australia.dat"),
        ("EVIA_AB.dat", JUST_DIR / "EVIA_AB.dat"),
        ("EVIB_AB.dat", JUST_DIR / "EVIB_AB.dat"),
        ("EVIC_AB.dat", JUST_DIR / "EVIC_AB.dat"),
        ("Sim_JumpInd_48_104.dat", JUST_MAT_DIR / "Sim_JumpInd_48_104.dat"),
        ("Sim_JumpInd_101_321.dat", JUST_MAT_DIR / "Sim_JumpInd_101_321.dat"),
    ]

    for label, path in just_datasets:
        t_d, f_d, P_d = _load_dat(path)
        w_arg = None if np.ndim(P_d) == 0 else P_d
        u_jumps = up_just.JUSTjumps(t_d.copy(), f_d.copy(), P=P_d)
        our_j = just_jumps(t_d, f_d, weights=w_arg, parity_mode=True)

        u_idx = [int(r[1]) for r in u_jumps]
        u_mag = [float(r[2]) for r in u_jumps]
        u_dir = [float(r[3]) for r in u_jumps]
        o_idx = [j.index for j in our_j.jumps]
        o_mag = [round(j.magnitude, 2) for j in our_j.jumps]
        o_dir = [round(j.direction, 2) for j in our_j.jumps]

        idx_ok = o_idx == u_idx
        vals_ok = np.allclose(o_mag, u_mag, atol=1e-8) and np.allclose(o_dir, u_dir, atol=1e-8)
        record(
            label,
            "just_jumps(parity)",
            "indices, mag, dir",
            idx_ok and vals_ok,
            f"idx={o_idx} vs {u_idx}; mag={o_mag} vs {u_mag}",
        )

        # Decomposition parity on non-empty or empty jump sets
        u_tr, u_se, u_re, u_tr_reg, u_se_reg = up_decomp.JUSTdecompose(
            t_d.copy(), f_d.copy(), P=P_d, ind=u_idx
        )
        our_dec = just_decompose(t_d, f_d, weights=w_arg, jump_indices=o_idx, parity_mode=True)
        ae_t, _ = _max_err(our_dec.trend, u_tr)
        ae_s, _ = _max_err(our_dec.seasonal, u_se)
        ae_r, _ = _max_err(our_dec.remainder, u_re)
        dec_ok = (
            np.allclose(our_dec.trend, u_tr, rtol=1e-6, atol=1e-8)
            and np.allclose(our_dec.seasonal, u_se, rtol=1e-6, atol=1e-8)
            and np.allclose(our_dec.remainder, u_re, rtol=1e-6, atol=1e-8)
        )
        record(
            label,
            "just_decompose(parity)",
            "trend/seasonal/rem",
            dec_ok,
            f"max_abs(T={ae_t:.2e}, S={ae_s:.2e}, R={ae_r:.2e})",
        )

    # Regularized decomposition check on EVIA_CA.dat with eval_times
    t_e, f_e, P_e = _load_dat(JUST_DIR / "EVIA_CA.dat")
    tt_e = np.linspace(t_e[0], t_e[-1], 300)
    u_tr, u_se, u_re, u_tr_reg, u_se_reg = up_decomp.JUSTdecompose(
        t_e.copy(), f_e.copy(), P=P_e, tt=tt_e.copy(), ind=[13, 111, 144]
    )
    our_dec_reg = just_decompose(
        t_e, f_e, weights=P_e, eval_times=tt_e, jump_indices=[13, 111, 144], parity_mode=True
    )
    assert our_dec_reg.trend_reg is not None and our_dec_reg.seasonal_reg is not None
    ae_tr, _ = _max_err(our_dec_reg.trend_reg, u_tr_reg)
    ae_sr, _ = _max_err(our_dec_reg.seasonal_reg, u_se_reg)
    record(
        "EVIA_CA.dat",
        "just_decompose(tt, parity)",
        "trend_reg & seasonal_reg",
        np.allclose(our_dec_reg.trend_reg, u_tr_reg, rtol=1e-6, atol=1e-8)
        and np.allclose(our_dec_reg.seasonal_reg, u_se_reg, rtol=1e-6, atol=1e-8),
        f"max_abs(T_reg={ae_tr:.2e}, S_reg={ae_sr:.2e})",
    )

    # ------------------------------------------------------------------
    # 3. STPD & TPTR on bundled STPD datasets
    # ------------------------------------------------------------------
    stpd_cases = [
        ("DESC-PSInSAR-MonthlyResampled.csv", {"size": 60, "step": 12, "margin": 12}),
        ("SimulatedExample.csv", {"size": 60, "step": 12, "margin": 12}),
    ]
    for fname, kw in stpd_cases:
        t_s, f_s = _load_stpd_csv(STPD_DIR / fname)
        u_tps = up_stpd.STPD(
            t_s, f_s, size=kw["size"], step=kw["step"], SNR=1, NDRI=0.3, dir_th=0, tp_th=1, margin=kw["margin"], alpha=0.01
        )
        u_stats, u_y = up_tptr.TPTR(t_s, f_s, u_tps)

        our_s = detect_turning_points(
            t_s,
            f_s,
            size=kw["size"],
            step=kw["step"],
            margin=kw["margin"],
            snr_th=1.0,
            ndri_th=0.3,
            dir_th=0.0,
            tp_th=1.0,
            alpha=0.01,
            parity_mode=True,
        )
        tp_ok = our_s.tp_indices.tolist() == [int(x) for x in u_tps]
        ae_y, _ = _max_err(our_s.piecewise_trend.fitted_trend, u_y)
        y_ok = np.allclose(our_s.piecewise_trend.fitted_trend, u_y, rtol=1e-6, atol=1e-8)

        # Compare rounded TPTR stats
        stats_ok = len(our_s.turning_points) == len(u_stats)
        for k, row_u in enumerate(u_stats):
            tp_e = our_s.turning_points[k]
            o_row = np.round([tp_e.index, tp_e.slope_after, tp_e.direction, tp_e.snr_raw, tp_e.ndri_l2], 2)
            if not np.allclose(o_row, row_u, atol=1e-8):
                stats_ok = False
        record(
            fname,
            "stpd + tptr (parity)",
            "TPs, stats, fitted y",
            tp_ok and y_ok and stats_ok,
            f"TPs={our_s.tp_indices.tolist()} vs {[int(x) for x in u_tps]}; y_abs={ae_y:.2e}",
        )

    # ------------------------------------------------------------------
    # 4. Synthetic irregular stress series: scalar P, 1D P, 2D SPD P
    # ------------------------------------------------------------------
    rng = np.random.default_rng(12345)
    n_syn = 110
    t_syn = np.sort(rng.uniform(0.0, 5.5, size=n_syn))
    t0_syn = t_syn - t_syn[0]
    y_syn = (
        np.where(t_syn < 2.8, 0.4 + 0.05 * t_syn, 0.9 - 0.08 * t_syn)
        + 0.25 * np.cos(2 * np.pi * 1.0 * t_syn)
        + 0.12 * np.sin(2 * np.pi * 2.2 * t_syn)
        + rng.normal(0.0, 0.04, size=n_syn)
    )
    w_1d = rng.uniform(0.5, 4.0, size=n_syn)
    idx_grid = np.arange(n_syn)
    cov_ar = 0.4 ** np.abs(idx_grid[:, None] - idx_grid[None, :])
    P_2d = np.linalg.inv(cov_ar)

    for w_label, P_up, w_our in [
        ("scalar P=1", 1, None),
        ("1D vector P", w_1d, w_1d),
        ("2D SPD P", P_2d, P_2d),
    ]:
        om_syn = np.linspace(0.5, 4.0, 40)
        u_sp, u_cr, _, u_r, u_nr, u_c, u_cv = up_lssa.LSSA(
            t0_syn, y_syn, P=P_up, Omega=om_syn, ind=[55], level=0.01
        )
        o_l = lssa(t_syn, y_syn, weights=w_our, omega=om_syn, jump_indices=[55], alpha=0.01)
        ae_sp, _ = _max_err(o_l.spectrum, u_sp)
        ae_cv, _ = _max_err(o_l.cov, u_cv)
        ok_l = (
            np.allclose(o_l.spectrum, u_sp, rtol=1e-6, atol=1e-8)
            and np.allclose(o_l.coeff, u_c, rtol=1e-6, atol=1e-8)
            and np.allclose(o_l.cov, u_cv, rtol=1e-6, atol=1e-8)
        )
        record(f"Synthetic ({w_label})", "lssa", "spec, coeff, cov", ok_l, f"spec_abs={ae_sp:.2e}, cov_abs={ae_cv:.2e}")

        u_j = up_just.JUSTjumps(t_syn.copy(), y_syn.copy(), P=P_up, season="OLS")
        o_j = just_jumps(t_syn, y_syn, weights=w_our, season="ols", parity_mode=True)
        u_ji = [int(r[1]) for r in u_j]
        o_ji = [j.index for j in o_j.jumps]
        record(
            f"Synthetic ({w_label})",
            "just_jumps(OLS, parity)",
            "jump indices",
            u_ji == o_ji,
            f"idx={o_ji} vs {u_ji}",
        )

    # Write Parity & Divergence Report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_pass = sum(1 for r in rows if r["status"] == "PASS")
    lines = [
        "# Tier 1 Numerical Parity & Divergence Report",
        "",
        "- **Upstream Oracle:** `Ghaderpour/LSWAVE-SignalProcessing` @ `bd15923b1d4880b799b14fe016de4c1252259013`",
        f"- **Summary:** **{n_pass} / {len(rows)} checks PASSED** (0 failures)" if not failures else f"- **Summary:** {n_pass} / {len(rows)} passed ({len(failures)} FAILED)",
        "- **Tolerances:** `rtol=1e-6, atol=1e-8` on continuous arrays; exact `==` on discrete indices and frequencies.",
        "",
        "## Parity Verification Matrix",
        "",
        "| Dataset | Method | Verified Metrics | Status | Error / Output Details |",
        "| :--- | :--- | :--- | :---: | :--- |",
    ]
    for r in rows:
        lines.append(
            f"| `{r['dataset']}` | `{r['method']}` | {r['metric']} | **{r['status']}** | {r['detail']} |"
        )

    lines.extend(
        [
            "",
            "## Documented Intentional Divergences (Default Mode vs. `parity_mode=True`)",
            "",
            "| Upstream Quirk / Bug | Behavior in `parity_mode=True` | Intentional Fix in Default Mode |",
            "| :--- | :--- | :--- |",
            "| `LSSA.py:270` `UnboundLocalError` when `trend='none', freq=[]` | Fixed in all modes (QR projection) | Handled cleanly without branching crash |",
            "| `LSSA.py:250` `ZeroDivisionError` when `n == nc + 2` | Enforces `n >= nc + 3` for spectrum | Raises `InsufficientDataError` instead of dividing by zero |",
            "| `ALLSSA.py:117` singular matrix when frequency `< 0.5` rounds to `0` | Unclamped fine grid | Clamps fine search grid strictly positive (`>= 10**-decimal`) + `max_iter` cycle guard |",
            "| `JUSTjumps.py:168` / `JUSTdecompose.py:194` duplicate last window when `(n - size) % step == 0` | Reproduces upstream `stop > n` loop | Stops cleanly at `stop >= n` so final window is never double-counted |",
            "| `JUSTjumps.py:283` premature 2-decimal rounding of `Mag`/`Dir` before thresholding | Rounds `Mag`/`Dir` to 2 decimals | Preserves full `float64` precision + computes nested partial $F$-test `jump_pvalue` |",
            "| `JUSTdecompose.py:42` off-by-one `+1` shift in `TrendComponent` regularization | Reproduces `+1` grid index shift | Uses exact `np.searchsorted(tt, t[j], side='left')` |",
            "| `STPD.py:58, 181` default `margin=1` causes `0/0 = NaN` in `SE1` | Clamps `margin >= 2` safely | Defaults to `max(2, min(M, size // 5))` and enforces `margin >= 2` |",
            "| `STPD.py:161, 164` non-shift-invariant `SNR` & density-biased L2 `NDRI` | Uses `snr_mode='raw'`, `ndri_mode='l2'` | Defaults to `snr_mode='centered'`, `ndri_mode='rmse'` (both variants always stored on `StpdResult`) |",
            "| `TPTR.py:53, 126` rounds stats to 2 decimals & drops initial segment slope | Preserves sequential TPTR ordering | Returns full-precision `PiecewiseTrendResult` with all `m + 1` slopes + optional `mode='joint'` |",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT_PATH} ({n_pass}/{len(rows)} passed).")
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
