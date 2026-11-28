"""Tier 2c: Real-World Public-Domain Dataset Validation Suite.

Validates ``tspec`` on three classic public-domain scientific records stored in
``validation/data/`` (offline, reproducible, excluded from package builds):

- ``RW1``: Nile River annual flow at Aswan (1871-1970; Cobb, 1978) — abrupt 1899 shift
  recovery via ``just_jumps`` and near-real-time detection via ``just_monitor``.
- ``RW2``: NOAA Mauna Loa monthly CO2 (1958-2026) — zero-interpolation decomposition with
  20% random missing months + historical outage gaps, ``allssa`` multi-harmonic recovery
  (1.0 + 2.0 cy/yr), bias-corrected trend agreement with NOAA's deseasonalized curve, and
  multi-decadal growth-rate acceleration turning points via ``detect_turning_points``.
- ``RW3``: Annual Sunspot Numbers (1700-2008) with 30% random missing years + 10-year
  historical blackout gap — zero-interpolation ``lssa`` / ``allssa`` recovery of the
  ~11-year Schwabe solar cycle (f ~ 0.091 cy/yr).

Run with:
    OMP_NUM_THREADS=1 MPLBACKEND=Agg python3 validation/bench_public_datasets.py
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import tspec
from tspec import allssa, detect_turning_points, just_decompose, just_jumps, just_monitor, lssa

DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class RealWorldGate:
    code: str
    name: str
    passed: bool
    metric: str
    elapsed_s: float


def _load_csv_columns(path: Path, cols: list[str]) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        buf: dict[str, list[float]] = {c: [] for c in cols}
        for row in reader:
            for c in cols:
                buf[c].append(float(row[c]))
    return {c: np.asarray(buf[c], dtype=np.float64) for c in cols}


def _run_gate(code: str, name: str, fn) -> RealWorldGate:
    t0 = time.perf_counter()
    passed, metric = fn()
    dt = time.perf_counter() - t0
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {code:4s} {name:46s} | {metric} ({dt:.1f}s)")
    return RealWorldGate(code=code, name=name, passed=passed, metric=metric, elapsed_s=dt)


def bench_rw1_nile_aswan_1871_1970() -> tuple[bool, str]:
    """RW1: Cobb (1978) Nile River at Aswan (1871-1970) — famous 1898-1899 regime shift."""
    data = _load_csv_columns(DATA_DIR / "nile_aswan_1871_1970.csv", ["year", "volume"])
    t = data["year"]
    y = data["volume"]

    # Retrospective abrupt-jump recovery (full-record annual window size=80)
    rj = just_jumps(t, y, season="none", size=80, step=10, jump_alpha=0.05, n_mc=499)
    jump_ok = (
        len(rj.jumps) == 1
        and abs(rj.jumps[0].time_year - 1899.0) <= 1.0
        and rj.jumps[0].magnitude < -150.0
        and rj.jumps[0].step_pvalue < 0.05
    )

    # Near-real-time monitoring starting at 1895 (pre-dam stable baseline 1871-1894)
    mon = just_monitor(
        t,
        y,
        start=1895.0,
        season="none",
        alpha=0.05,
        sequential_control="pocock",
        horizon=40,
        monitor_omega=np.linspace(0.02, 0.45, 25),
    )
    mon_ok = (
        mon.detected
        and mon.disturbance_type == "abrupt"
        and mon.disturbance_time_year is not None
        and 1898.0 <= mon.disturbance_time_year <= 1903.0
    )

    j_str = (
        f"{rj.jumps[0].time_year:.0f} (Δ={rj.jumps[0].magnitude:+.1f}, p={rj.jumps[0].jump_pvalue:.4f})"
        if rj.jumps
        else "none"
    )
    m_str = f"{mon.disturbance_type}@{mon.disturbance_time_year:.0f}" if mon.detected else "none"
    return jump_ok and mon_ok, f"JUST jump={j_str}; NRT alarm={m_str}"


def bench_rw2_mauna_loa_co2() -> tuple[bool, str]:
    """RW2: NOAA Mauna Loa monthly CO2 (1958-2026) — irregular gaps, ALLSSA, curved trend & acceleration."""
    data = _load_csv_columns(
        DATA_DIR / "mauna_loa_co2_monthly.csv", ["decimal date", "average", "deseasonalized"]
    )
    t_all = data["decimal date"]
    y_all = data["average"]
    des_noaa = data["deseasonalized"]

    # Drop historical 1964/1984 outage windows + 20% random months (zero interpolation!)
    rng = np.random.default_rng(2026)
    keep = (
        (rng.uniform(size=t_all.size) > 0.20)
        & ~((t_all >= 1964.08) & (t_all <= 1964.35))
        & ~((t_all >= 1984.15) & (t_all <= 1984.35))
    )
    t = t_all[keep]
    y = y_all[keep]
    des_ref = des_noaa[keep]

    # 1. ALLSSA multi-harmonic recovery on gapped 2005-2020 slice -> exact annual (1.0) + semi-annual (2.0)
    sl = (t >= 2005.0) & (t <= 2020.0)
    res_a = allssa(
        t[sl],
        y[sl],
        omega=np.array([1.0, 2.0, 3.0, 4.0]),
        decimal=2,
        alpha=0.01,
        trend="quadratic",
        significance="montecarlo",
        n_mc=499,
    )
    freqs_sorted = np.sort(res_a.frequencies)
    freq_ok = (
        freqs_sorted.size >= 2
        and np.any(np.abs(freqs_sorted - 1.0) <= 0.03)
        and np.any(np.abs(freqs_sorted - 2.0) <= 0.03)
    )

    # 2. Bias-corrected JUST decomposition on gapped record vs NOAA official deseasonalized curve
    dec = just_decompose(
        t,
        y,
        season="ols",
        omega=np.array([1.0, 2.0]),
        size=48,
        step=12,
        trend_ci_method="bias_corrected",
    )
    rmse_noaa = float(np.sqrt(np.mean((dec.trend_bc - des_ref) ** 2)))
    decomp_ok = rmse_noaa < 0.45

    # 3. STPD continuous acceleration turning points on deseasonalized trend
    tp = detect_turning_points(
        t, dec.trend_bc, size=144, step=30, margin=20, alpha=0.01, n_mc=199, band_method="conditional"
    )
    slopes = tp.piecewise_trend.slopes
    accel_ok = len(tp.turning_points) >= 2 and float(slopes[-1]) > float(slopes[0]) + 1.0

    ok = freq_ok and decomp_ok and accel_ok
    return (
        ok,
        f"ALLSSA={np.round(freqs_sorted, 2).tolist()}, RMSE vs NOAA={rmse_noaa:.3f} ppm, "
        f"growth {slopes[0]:.2f}→{slopes[-1]:.2f} ppm/yr ({len(tp.turning_points)} TPs)",
    )


def bench_rw3_sunspots_gapped_solar_cycle() -> tuple[bool, str]:
    """RW3: Annual Sunspots (1700-2008) with 30% missing years + 10-yr blackout -> ~11-yr Schwabe cycle."""
    data = _load_csv_columns(DATA_DIR / "sunspots_annual_1700_2008.csv", ["year", "sunspots"])
    t_all = data["year"]
    y_all = data["sunspots"]

    rng = np.random.default_rng(42)
    keep = (rng.uniform(size=t_all.size) > 0.30) & ~((t_all >= 1820.0) & (t_all <= 1830.0))
    t = t_all[keep]
    y = y_all[keep]

    om = np.linspace(0.01, 0.45, 441)
    sp = lssa(t, y, omega=om, trend="linear", peak_fap="sidak")
    peak_f = float(sp.frequencies[int(np.argmax(sp.spectrum))])
    period_yr = 1.0 / peak_f

    ok = (10.5 <= period_yr <= 11.5) and sp.peak_fap < 1e-6
    return (
        ok,
        f"n_kept={t.size}/{t_all.size}, peak={peak_f:.4f} cy/yr (period={period_yr:.2f} yr), peak_fap={sp.peak_fap:.2e}",
    )


def bench_rw4_nasa_gistemp_global_temp() -> tuple[bool, str]:
    """RW4: NASA GISTEMP v4 Global Temperature Anomaly (1880-2025) — 1970s warming acceleration."""
    data = _load_csv_columns(
        DATA_DIR / "nasa_gistemp_global_annual.csv", ["year", "temp_anomaly_c"]
    )
    t = data["year"]
    y = data["temp_anomaly_c"]

    tp = detect_turning_points(
        t,
        y,
        size=80,
        step=15,
        margin=12,
        tp_th=10.0,
        alpha=0.01,
        n_mc=199,
        band_method="conditional",
    )
    years = [p.time_year for p in tp.turning_points]
    has_1970s = any(1965.0 <= yr <= 1980.0 for yr in years)
    modern_rate_decade = float(tp.piecewise_trend.slopes[-1]) * 10.0
    ok = has_1970s and (0.15 <= modern_rate_decade <= 0.28)
    return (
        ok,
        f"TPs={[int(round(x)) for x in years]}, modern_rate={modern_rate_decade:+.3f} °C/decade",
    )


def bench_rw5_us_macrodata_cpi_regimes() -> tuple[bool, str]:
    """RW5: US Macrodata log(CPI) (1959-2009) — Great Inflation (1972), Volcker (1981), Great Moderation (1991)."""
    data = _load_csv_columns(DATA_DIR / "us_macrodata_quarterly.csv", ["decimal_year", "cpi"])
    t = data["decimal_year"]
    log_cpi = np.log(data["cpi"])

    tp = detect_turning_points(
        t, log_cpi, size=80, step=12, margin=12, alpha=0.01, n_mc=199, band_method="conditional"
    )
    years = [p.time_year for p in tp.turning_points]
    slopes_pct = [float(s) * 100.0 for s in tp.piecewise_trend.slopes]
    has_70s = any(1971.0 <= yr <= 1974.0 for yr in years)
    has_volcker = any(1980.0 <= yr <= 1983.0 for yr in years)
    has_90s = any(1990.0 <= yr <= 1993.0 for yr in years)
    ok = has_70s and has_volcker and has_90s and max(slopes_pct) > 7.5 and slopes_pct[-1] < 3.5
    return (
        ok,
        f"TPs={[round(x, 2) for x in years]}, infl_regimes={[round(s, 2) for s in slopes_pct]} %/yr",
    )


def bench_rw6_elnino_enso_oceanography() -> tuple[bool, str]:
    """RW6: NOAA/CPC El Nino SST (1950-2010, 25% gapped) — irregular 3-6 yr ENSO cycle."""
    data = _load_csv_columns(DATA_DIR / "elnino_sst_monthly.csv", ["decimal_year", "sst_c"])
    rng = np.random.default_rng(7)
    keep = rng.uniform(size=data["decimal_year"].size) > 0.25
    t = data["decimal_year"][keep]
    y = data["sst_c"][keep]

    sp = lssa(
        t,
        y,
        known_freqs=np.array([1.0, 2.0]),
        omega=np.linspace(0.10, 0.50, 201),
        trend="linear",
        peak_fap="sidak",
    )
    pk_f = float(sp.frequencies[int(np.argmax(sp.spectrum))])
    period_yr = 1.0 / pk_f
    ok = (3.0 <= period_yr <= 6.0) and sp.peak_fap < 1e-8
    return (
        ok,
        f"n_kept={t.size}/{data['decimal_year'].size}, ENSO peak={pk_f:.4f} cy/yr (period={period_yr:.2f} yr), FAP={sp.peak_fap:.2e}",
    )


def main() -> int:
    print(f"Running tspec v{tspec.__version__} Tier 2c Real-World Public Dataset Validation Suite...")
    t_start = time.perf_counter()

    gates = [
        _run_gate("RW1", "Nile River at Aswan (1871-1970, Cobb 1978)", bench_rw1_nile_aswan_1871_1970),
        _run_gate("RW2", "NOAA Mauna Loa CO2 (1958-2026, 20% gapped)", bench_rw2_mauna_loa_co2),
        _run_gate("RW3", "Sunspots (1700-2008, 30% + 10-yr blackout)", bench_rw3_sunspots_gapped_solar_cycle),
        _run_gate("RW4", "NASA GISTEMP Global Temp (1880-2025)", bench_rw4_nasa_gistemp_global_temp),
        _run_gate("RW5", "US Macrodata log(CPI) (1959-2009, Economics)", bench_rw5_us_macrodata_cpi_regimes),
        _run_gate("RW6", "El Nino SST (1950-2010, 25% gapped, Ocean)", bench_rw6_elnino_enso_oceanography),
    ]

    total_s = time.perf_counter() - t_start
    n_pass = sum(g.passed for g in gates)
    print(f"\nSummary: {n_pass}/{len(gates)} real-world gates PASSED in {total_s:.1f}s.")

    report_path = Path(__file__).resolve().parent / "reports" / "public_datasets_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Tier 2c: Real-World Public Dataset Validation Report",
        "",
        f"- **tspec version:** `{tspec.__version__}`",
        f"- **Result:** **{n_pass} / {len(gates)} gates PASSED** ({total_s:.1f} s single-threaded runtime)",
        "",
        "| Gate | Public Dataset & Domain | Status | Measured Metrics | Runtime |",
        "| :--- | :--- | :---: | :--- | ---: |",
    ]
    for g in gates:
        lines.append(
            f"| `{g.code}` | {g.name} | **{'PASS' if g.passed else 'FAIL'}** | {g.metric} | {g.elapsed_s:.1f} s |"
        )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report to {report_path}")
    return 0 if n_pass == len(gates) else 1


if __name__ == "__main__":
    raise SystemExit(main())
