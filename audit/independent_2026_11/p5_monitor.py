"""P5: just_monitor causality (no look-ahead), sequential FWER, robust spikes."""
import json, sys, time, warnings
import numpy as np, tspec
from common import pmap, wilson
warnings.filterwarnings("ignore")
NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 300
out = {"NREP": NREP}

def series(seed, hist_years=5.0, mon_steps=24, per_year=24, irregular=False, spikes_hist=0, spikes_mon=0, shift=0.0, sd=0.03):
    rng = np.random.default_rng(seed)
    n = int(hist_years*per_year) + mon_steps
    t = 2015 + np.arange(n)/per_year
    if irregular: t = t + rng.uniform(-0.3, 0.3, n)/per_year
    y = 0.55 + 0.18*np.cos(2*np.pi*t) + 0.05*np.sin(4*np.pi*t) + rng.normal(0, sd, n)
    nh = int(hist_years*per_year)
    if spikes_hist: y[rng.choice(nh, spikes_hist, replace=False)] += 25*sd
    if spikes_mon: y[nh + rng.choice(mon_steps, spikes_mon, replace=False)] -= 25*sd
    if shift: y[nh + mon_steps//2:] -= shift
    return t, y, nh

# --- 1. causality: perturb future after step k; earlier step records must be identical
def causality(cfg):
    t, y, nh = series(0, mon_steps=30, irregular=True)
    base = tspec.just_monitor(t, y, start=nh, stop_on_first=False, **cfg)
    worst = 0.0
    for k in [3, 10, 20]:
        y2 = y.copy(); y2[nh + k:] += np.random.default_rng(k).normal(0, 5, y2.size - nh - k)
        y2[nh + k + 1] = np.nan
        alt = tspec.just_monitor(t, y2, start=nh, stop_on_first=False, **cfg)
        for s_b, s_a in zip(base.steps[:k], alt.steps[:k]):
            for f in ("max_spec", "critical_value", "alpha_adjusted"):
                worst = max(worst, abs(getattr(s_b, f) - getattr(s_a, f)))
            if s_b.exceeded != s_a.exceeded: worst = max(worst, 1.0)
        fb = base.forecast[: nh + k] if base.forecast is not None and base.forecast.size == y.size else None
    return worst
cfgs = {"default_pocock_h30": dict(horizon=30), "allssa_obf_h30": dict(horizon=30, sequential_control="obrien_fleming"),
        "consecutive_k_h30": dict(horizon=30, sequential_control="consecutive_k"), "open_ended_pocock": dict(),
        "ols_moving": dict(season="ols", window_mode="moving", horizon=30) if True else {}, "mm_robust": dict(robust="mm", horizon=30)}
caus = {}
for k, c in cfgs.items():
    try: caus[k] = causality(c)
    except Exception as ex: caus[k] = repr(ex)[:120]
out["causality_max_abs_diff_past_steps"] = caus; print(caus, flush=True)

# history-imputation leakage with censoring: does a future censored value change past steps?
def censor_causal():
    t, y, nh = series(1, mon_steps=30)
    s = np.zeros(y.size, int); lim = np.quantile(y, 0.25); s[y < lim] = -1; yc = np.where(s == -1, lim, y)
    cfg = tspec.CensoredConfig(random_state=3)
    base = tspec.just_monitor(t, yc, start=nh, censor_status=s, censoring=cfg, stop_on_first=False, horizon=30)
    s2 = s.copy(); y2 = yc.copy(); s2[nh+20:] = -1; y2[nh+20:] = lim
    alt = tspec.just_monitor(t, y2, start=nh, censor_status=s2, censoring=cfg, stop_on_first=False, horizon=30)
    return max(abs(a.max_spec - b.max_spec) for a, b in zip(base.steps[:20], alt.steps[:20]))
try: out["censoring_causality_max_abs_diff"] = censor_causal()
except Exception as ex: out["censoring_causality_max_abs_diff"] = repr(ex)[:150]
print("cens causal", out["censoring_causality_max_abs_diff"], flush=True)

# --- 2. sequential FWER under null
def job(a):
    seed, sk, mk = a
    t, y, nh = series(seed, **sk)
    try:
        r = tspec.just_monitor(t, y, start=nh, alpha=0.05, **mk)
        return dict(det=int(r.detected), lag=(r.disturbance_index - (nh + sk.get("mon_steps", 24)//2)) if r.detected else None)
    except Exception as ex:
        return dict(det=-1, err=repr(ex)[:80])
cases = {
    "pocock_h24": (dict(), dict(horizon=24)),
    "pocock_open": (dict(), dict()),
    "obf_h24": (dict(), dict(horizon=24, sequential_control="obrien_fleming")),
    "consk_h24": (dict(), dict(horizon=24, sequential_control="consecutive_k")),
    "consk_noH": (dict(), dict(sequential_control="consecutive_k")),
    "none_h24": (dict(), dict(horizon=24, sequential_control="none")),
    "pocock_h120_long": (dict(mon_steps=120), dict(horizon=120)),
    "consk_h120_long": (dict(mon_steps=120), dict(horizon=120, sequential_control="consecutive_k")),
    "consk_h24_irregular": (dict(irregular=True), dict(horizon=24, sequential_control="consecutive_k")),
    "pocock_h24_hist_spikes": (dict(spikes_hist=6), dict(horizon=24)),
    "pocock_h24_hist_spikes_mm": (dict(spikes_hist=6), dict(horizon=24, robust="mm")),
    "pocock_h24_mon_spike_mm": (dict(spikes_mon=1), dict(horizon=24, robust="mm")),
    "consk_h24_mon_spike_mm": (dict(spikes_mon=1), dict(horizon=24, robust="mm", sequential_control="consecutive_k")),
}
for name, (sk, mk) in cases.items():
    t1 = time.time()
    res = pmap(job, [(s, sk, mk) for s in range(NREP)])
    k = sum(r["det"] == 1 for r in res); e = sum(r["det"] == -1 for r in res)
    out[f"fwer_{name}"] = dict(rate=k/NREP, ci95=wilson(k, NREP), errors=e, sec=round(time.time()-t1, 1),
                               err=next((r["err"] for r in res if r["det"] == -1), None))
    print(name, out[f"fwer_{name}"], flush=True)
# --- 3. power / delay for a 0.1 (3.3 sd) drop at mid-monitoring
for name, mk in {"pocock_h24": dict(horizon=24), "consk_h24": dict(horizon=24, sequential_control="consecutive_k")}.items():
    res = pmap(job, [(s, dict(shift=0.10), mk) for s in range(NREP)])
    det = [r for r in res if r["det"] == 1]
    lags = [r["lag"] for r in det if r["lag"] is not None]
    out[f"power_{name}"] = dict(rate=len(det)/NREP, median_delay_steps=float(np.median(lags)) if lags else None,
                                frac_before_change=float(np.mean(np.array(lags) < 0)) if lags else None)
    print(name, out[f"power_{name}"], flush=True)
json.dump(out, open("p5_results.json", "w"), indent=1, default=float)
