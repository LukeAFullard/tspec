"""Re-audit: combination fuzzing of options added after the original audit
(tp_calibration, snr_th, consecutive_k calibration, detection_imputations, pool_pvalues).
Each case must either run and return finite, in-range p-values / levels, or raise a clear
ValueError. Run: OMP_NUM_THREADS=1 python audit/reaudit_fuzz.py
"""
import itertools
import traceback
import warnings

import numpy as np
import pandas as pd

import tspec
from tspec._cenreg import pool_pvalues

warnings.simplefilter("ignore")
fails = []
ok = 0


def series(seed, n=150, cens=0.0, kind="left", dt=False, bend=0.0, jump=0.0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(2000, 2000 + n / 24, n))
    tm = 2000 + n / 48
    y = 2 + 0.1 * (t - 2000) + 0.3 * np.cos(2 * np.pi * t) + jump * (t >= tm) - bend * np.maximum(0, t - tm)
    y = y + rng.normal(0, 0.3, n)
    if cens > 0:
        dl = float(np.quantile(y, cens))
        if kind == "left":
            y = np.array([f"<{dl:.6g}" if v < dl else float(v) for v in y], dtype=object)
        else:
            dh = float(np.quantile(y, 1 - cens))
            y = np.array([f">{dh:.6g}" if v > dh else float(v) for v in y], dtype=object)
    if dt:
        t = pd.to_datetime("2000-01-01") + pd.to_timedelta((t - 2000) * 365.25, unit="D")
    return t, y


def check(name, fn):
    global ok
    try:
        r = fn()
        ps = []
        jumps = getattr(r, "jumps", ()) or ()
        jumps = getattr(jumps, "jumps", jumps)  # PipelineResult.jumps is a JustJumpResult
        tps = getattr(r, "turning_points", ()) or ()
        tps = getattr(tps, "turning_points", tps)
        for ev in jumps:
            ps += [ev.jump_pvalue, ev.step_pvalue]
        for ev in tps:
            ps.append(ev.pvalue)
        for st in getattr(r, "steps", ()) or ():
            ps.append(st.alpha_step)
        ps = np.asarray(ps, dtype=float)
        if ps.size and (np.any(~np.isfinite(ps)) or np.any(ps < 0) or np.any(ps > 1)):
            fails.append((name, f"bad p/level {ps}"))
        else:
            ok += 1
    except ValueError as e:
        ok += 1
        print(f"  ValueError (accepted) {name}: {e}")
    except Exception:
        fails.append((name, traceback.format_exc(limit=3)))


# STPD
for cal, rob, wts, mode, cens, dt in itertools.product(
    (None, "montecarlo", "bonferroni", "none"), ("none", "huber", "mm"), (False, True),
    ("count", "duration"), (0.0, 0.4), (False, True)
):
    t, y = series(1, cens=cens, dt=dt, bend=1.0)
    w = np.random.default_rng(2).uniform(0.5, 2, len(y)) if wts else None
    cfg = tspec.CensoredConfig(detection_imputations=5) if cens else None
    check(f"stpd cal={cal} rob={rob} w={wts} mode={mode} cens={cens} dt={dt}",
          lambda: tspec.stpd(t, y, tp_calibration=cal, robust=rob, weights=w, window_mode=mode,
                             censoring=cfg, n_mc=199))
# JUST with detection imputations
for season, rob, wts, kind, scope in itertools.product(
    ("none", "ols", "allssa"), ("none", "huber"), (False, True), ("left", "right"),
    ("global", "windowed", "seasonal_stratified")
):
    t, y = series(3, cens=0.3, kind=kind, jump=1.0)
    w = np.random.default_rng(4).uniform(0.5, 2, len(y)) if wts else None
    cfg = tspec.CensoredConfig(detection_imputations=5, scope=scope)
    check(f"just season={season} rob={rob} w={wts} {kind} scope={scope}",
          lambda: tspec.just_jumps(t, y, season=season, robust=rob, weights=w, censoring=cfg, n_mc=199))
# weibull / marginal imputation model with detection imputations
for model, dist in (("marginal", "auto"), ("trend", "normal"), ("marginal", "weibull")):
    t, y = series(5, cens=0.3, jump=1.0)
    check(f"just model={model} dist={dist}", lambda: tspec.just_jumps(
        t, y, season="none", censoring=tspec.CensoredConfig(imputation_model=model, dist=dist, detection_imputations=5)))
    check(f"stpd model={model} dist={dist}", lambda: tspec.stpd(
        t, y, censoring=tspec.CensoredConfig(imputation_model=model, dist=dist, detection_imputations=5)))
# Monitor consecutive_k calibration
n = 96 + 48
rng = np.random.default_rng(7)
tm = np.sort(rng.uniform(2000, 2000 + n / 24, n))
ym = 2 + 0.5 * np.cos(2 * np.pi * tm) + rng.normal(0, 0.15, n)
wv = rng.uniform(0.5, 2, n)
A = rng.normal(size=(n, n)) * 0.01
wm = np.eye(n) + A @ A.T
for season, wmode, k, hz, wts, rob, fc, freq in itertools.product(
    ("ols", "allssa", "none"), ("expanding", "sliding"), (1, 2, 3), (None, 10, 48, 200),
    (None, "vector", "matrix"), ("none", "huber"), (True, False), ("sidak", "bonferroni", "none")
):
    if rng.random() > 0.25:  # sample a quarter of the grid
        continue
    w = None if wts is None else (wv if wts == "vector" else wm)
    check(f"monitor {season} {wmode} k={k} H={hz} w={wts} rob={rob} fc={fc} freq={freq}",
          lambda: tspec.just_monitor(tm, ym, start=96, season=season, window_mode=wmode,
                                     sequential_control="consecutive_k", consecutive_k=k, horizon=hz,
                                     weights=w, robust=rob, account_for_forecast_uncertainty=fc,
                                     freq_correction=freq, alpha=0.05, n_mc=199, stop_on_first=False))
check("monitor datetime", lambda: tspec.just_monitor(
    pd.to_datetime("2000-01-01") + pd.to_timedelta((tm - 2000) * 365.25, unit="D"), ym, start=96,
    sequential_control="consecutive_k", horizon=48, alpha=0.05))
# analyze_series with detection imputations
t, y = series(9, cens=0.4, jump=1.0, bend=0.0)
check("analyze detection_imputations", lambda: tspec.analyze_series(
    t, y, censoring=tspec.CensoredConfig(detection_imputations=5)))
# pool_pvalues edge cases
for arr in ([0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [np.nan, 0.2], [1e-300] * 3, [0.5]):
    v = pool_pvalues(np.array(arr))
    if not (np.isnan(v) or 0.0 <= v <= 1.0):
        fails.append(("pool", f"{arr} -> {v}"))
    else:
        ok += 1
print(f"\nOK: {ok}  FAIL: {len(fails)}")
for f in fails:
    print("FAIL", f[0], "\n", f[1])
