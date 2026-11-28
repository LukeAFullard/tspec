"""P7: adversarial numerical edge cases, reproducibility, memory scaling."""
import json, time, tracemalloc, warnings, traceback
import numpy as np, tspec
warnings.filterwarnings("ignore")
rng = np.random.default_rng(7)
out = {}
def run(label, fn):
    try:
        v = fn(); out[label] = v if isinstance(v, (str, int, float, bool, dict, list)) or v is None else str(v)
    except Exception as ex:
        out[label] = f"RAISED {type(ex).__name__}: {str(ex)[:140]}"
    print(label, "->", out[label], flush=True)

n = 200; t = np.sort(rng.uniform(0, 20, n)); y = 0.1*t + np.cos(2*np.pi*t) + rng.normal(0, .3, n) + 2.0*(t > 10)
base = tspec.just_jumps(t, y, season="ols")
bj = [round(float(base.times_numeric[j.valid_index]), 6) for j in base.jumps]
out["base_jumps"] = bj
# --- invariances (affine in y; shift in t)
run("jumps_y_affine_1e9_offset", lambda: [round(float(r.times_numeric[j.valid_index]), 6) for r in [tspec.just_jumps(t, y + 1e9, season="ols")] for j in r.jumps] == bj)
run("jumps_y_scale_1e-12", lambda: [round(float(r.times_numeric[j.valid_index]), 6) for r in [tspec.just_jumps(t, y*1e-12, season="ols")] for j in r.jumps] == bj)
run("jumps_y_scale_1e12", lambda: [round(float(r.times_numeric[j.valid_index]), 6) for r in [tspec.just_jumps(t, y*1e12, season="ols")] for j in r.jumps] == bj)
run("jumps_t_offset_1e7(seasonal phase preserved since integer)", lambda: [round(float(r.times_numeric[j.valid_index]) - 1e7, 4) for r in [tspec.just_jumps(t + 1e7, y, season="ols")] for j in r.jumps] == [round(x, 4) for x in bj])
run("decompose_y_offset_1e9_trend_err", lambda: float(np.max(np.abs(tspec.just_decompose(t, y + 1e9, season="ols").trend - 1e9 - tspec.just_decompose(t, y, season="ols").trend))))
run("stpd_y_scale_1e-12_same_tps", lambda: [p.index for p in tspec.stpd(t, 1e-12*(0.5*t - np.maximum(t-10, 0)) + 1e-12*rng.normal(0, .3, n), band_method="conditional").turning_points])
# --- input pathologies
run("unsorted_t", lambda: tspec.lssa(t[::-1], y[::-1], omega=[1.0]).spectrum.tolist())
run("duplicate_t", lambda: tspec.lssa(np.r_[t[:5], t[4:]], np.r_[y[:5], y[4:]], omega=[1.0]).spectrum.tolist())
run("nan_burst_y_50pct", lambda: len(tspec.just_jumps(t, np.where(rng.random(n) < .5, np.nan, y), season="ols").jumps))
run("inf_in_y", lambda: tspec.lssa(t, np.r_[y[:-1], np.inf], omega=[1.0]).spectrum.tolist())
run("nan_in_t", lambda: tspec.lssa(np.r_[t[:-1], np.nan], y, omega=[1.0]).spectrum.tolist())
run("tiny_n_5_lssa", lambda: tspec.lssa(t[:5], y[:5], omega=[1.0]).spectrum.tolist())
run("tiny_n_10_just", lambda: len(tspec.just_jumps(t[:10], y[:10]).jumps))
run("tiny_n_10_stpd", lambda: len(tspec.stpd(t[:10], y[:10]).turning_points))
run("tiny_n_10_analyze", lambda: len(tspec.analyze_series(t[:10], y[:10]).segments))
run("constant_y_just", lambda: len(tspec.just_jumps(t, np.ones(n)).jumps))
run("constant_y_stpd", lambda: len(tspec.stpd(t, np.ones(n)).turning_points))
run("constant_y_lssa", lambda: tspec.lssa(t, np.ones(n), omega=[1.0, 2.0]).pvalues.tolist())
run("noiseless_linear_just_jumps", lambda: len(tspec.just_jumps(t, 3 + 0.2*t, season="none").jumps))
run("noiseless_linear_stpd", lambda: len(tspec.stpd(t, 3 + 0.2*t).turning_points))
run("noiseless_linear_analyze_segments", lambda: len(tspec.analyze_series(t, 3 + 0.2*t).segments))
run("noiseless_hinge_stpd_tp_time", lambda: [round(p.time_year, 2) for p in tspec.stpd(t, 0.5*t - np.maximum(t - 10, 0)).turning_points])
run("noiseless_step_just_mag", lambda: [round(j.magnitude, 6) for j in tspec.just_jumps(t, 1 + 2.0*(t > 10), season="none").jumps])
run("collinear_known_freqs", lambda: [w.code for w in tspec.lssa(t, y, known_freqs=[1.0, 1.0 + 1e-9]).diagnostics.warnings])
run("clustered_timestamps_1e-9", lambda: tspec.lssa(np.r_[t[:100], t[100] + np.arange(100)*1e-9], y, omega=[1.0, 3.0]).spectrum.tolist())
run("gap_70pct_missing_just", lambda: len(tspec.just_jumps(t[(t < 3) | (t > 17)], y[(t < 3) | (t > 17)], season="ols").jumps))
run("gap_70pct_decompose_ci_finite", lambda: bool(np.all(np.isfinite(tspec.just_decompose(t[(t < 3) | (t > 17)], y[(t < 3) | (t > 17)], season="ols").trend_ci_high))))
run("integer_t", lambda: tspec.lssa(np.arange(100), rng.normal(size=100), omega=[0.1]).spectrum.tolist())
run("python_list_inputs", lambda: tspec.lssa(list(t), list(y), omega=[1.0]).spectrum.tolist())
import pandas as pd
idx = pd.date_range("2020-01-01", periods=120, freq="MS", tz="America/New_York")
yy = np.cos(2*np.pi*np.arange(120)/12) + rng.normal(0, .1, 120)
run("tz_aware_pandas_index", lambda: float(tspec.lssa(idx, yy, omega=[1.0]).spectrum[0]))
run("tz_aware_vs_naive_equal", lambda: float(abs(tspec.lssa(idx, yy, omega=[1.0]).spectrum[0] - tspec.lssa(idx.tz_convert('UTC').tz_localize(None), yy, omega=[1.0]).spectrum[0])))
run("pandas_series_y_with_index", lambda: float(tspec.lssa(idx, pd.Series(yy, index=idx), omega=[1.0]).spectrum[0]))
run("negative_weights", lambda: tspec.lssa(t, y, weights=-np.ones(n), omega=[1.0]).spectrum.tolist())
run("weights_wrong_len", lambda: tspec.lssa(t, y, weights=np.ones(n-1), omega=[1.0]).spectrum.tolist())
run("weights_with_nan_y_alignment", lambda: float(tspec.lssa(t, np.where(np.arange(n) == 5, np.nan, y), weights=np.r_[1e-12, np.ones(n-1)], omega=[1.0]).spectrum[0]))
# --- extreme multi-regime heteroscedasticity
sd = np.where(t < 7, 0.1, np.where(t < 14, 3.0, 0.3)); yh = 0.1*t + rng.normal(0, sd)
run("hetero_unweighted_jumps_null", lambda: len(tspec.just_jumps(t, yh, season="none").jumps))
run("hetero_weighted_jumps_null", lambda: len(tspec.just_jumps(t, yh, season="none", weights=1/sd**2).jumps))
run("hetero_unweighted_stpd_null", lambda: len(tspec.stpd(t, yh, band_method="conditional").turning_points))
run("hetero_weighted_stpd_null", lambda: len(tspec.stpd(t, yh, weights=1/sd**2, band_method="conditional").turning_points))
# --- parameter validation footguns
run("alpha_gt1", lambda: tspec.lssa(t, y, omega=[1.0], alpha=1.5).critical_value)
run("ci_level_95_not_frac", lambda: tspec.just_decompose(t, y, ci_level=95).ci_level)
run("negative_omega", lambda: tspec.lssa(t, y, omega=[-1.0]).spectrum.tolist())
run("size_too_small", lambda: len(tspec.just_jumps(t, y, size=4).jumps))
run("size_gt_n", lambda: len(tspec.just_jumps(t, y, size=500).jumps))
run("unknown_kwarg_analyze_just_kwargs", lambda: len(tspec.analyze_series(t, y, just_kwargs=dict(not_a_param=1)).segments))
run("horizon_zero", lambda: tspec.just_monitor(t, y, start=150, horizon=0).detected)
run("start_at_end", lambda: tspec.just_monitor(t, y, start=n-1).detected)
run("monitor_start_before_enough_history", lambda: tspec.just_monitor(t, y, start=5).detected)
run("censor_90pct_guard", lambda: len(tspec.just_jumps(t, np.where(rng.random(n) < .9, "<5", y.astype(object)).tolist()).jumps))
run("time_unit_days_warning_codes", lambda: [w.code for w in tspec.just_jumps(np.arange(0, 3650, 18.25), np.cos(2*np.pi*np.arange(0, 3650, 18.25)/365.25) + rng.normal(0, .1, 200)).diagnostics.warnings])

# --- reproducibility (all stochastic paths): identical results with fixed seeds, twice
ys = np.where(y < np.quantile(y, .25), "<%.3f" % np.quantile(y, .25), y.astype(object)).tolist()
def rep(fn):
    a, b = fn(), fn(); return a == b
run("repro_just_jumps_mc", lambda: rep(lambda: [(j.index, j.jump_pvalue) for j in tspec.just_jumps(t, y).jumps]))
run("repro_stpd_bootstrap", lambda: rep(lambda: tspec.stpd(t, 0.5*t - np.maximum(t-10, 0) + np.sin(t)).piecewise_trend.trend_ci_low.tolist()))
run("repro_analyze_bootstrap", lambda: rep(lambda: tspec.analyze_series(t, y).trend_ci_low.tolist()))
run("repro_censored_mi", lambda: rep(lambda: [(s.slope, s.slope_ci_low) for s in tspec.analyze_series(t, ys, detect_turning_points=False, censoring=tspec.CensoredConfig(random_state=5)).segments]))
run("repro_censored_random_state_None_differs", lambda: not rep(lambda: [s.slope_ci_low for s in tspec.analyze_series(t, ys, detect_turning_points=False, censoring=tspec.CensoredConfig(random_state=None)).segments]))
run("repro_monitor_consk", lambda: rep(lambda: [s.alpha_adjusted for s in tspec.just_monitor(t, y, start=150, horizon=30, sequential_control="consecutive_k", stop_on_first=False).steps]))
run("repro_lssa_mc", lambda: rep(lambda: tspec.lssa(t, y, omega=np.linspace(.1, 5, 200), peak_fap="montecarlo").peak_fap))
# global numpy RNG must not be consumed/perturbed
np.random.seed(123); a0 = np.random.random(); np.random.seed(123); tspec.analyze_series(t, y); a1 = np.random.random()
out["global_rng_untouched"] = bool(a0 == a1); print("global rng", out["global_rng_untouched"])

# --- memory scaling for 1D weights (should be O(n))
mem = {}
for nn in [2000, 8000, 32000]:
    tt = np.sort(rng.uniform(0, 100, nn)); yy = rng.normal(size=nn)
    tracemalloc.start(); t1 = time.time()
    tspec.lssa(tt, yy, weights=np.ones(nn)*2.0, omega=np.linspace(0.1, 10, 300)); el = time.time() - t1
    peak = tracemalloc.get_traced_memory()[1] / 2**20; tracemalloc.stop()
    mem[nn] = dict(peak_MB=round(peak, 1), sec=round(el, 2))
out["lssa_1d_weight_memory"] = mem; print(mem)
mem = {}
for nn in [1000, 4000, 16000]:
    tt = np.sort(rng.uniform(0, nn/12, nn)); yy = np.cos(2*np.pi*tt) + rng.normal(size=nn)
    tracemalloc.start(); t1 = time.time()
    tspec.just_jumps(tt, yy, weights=np.ones(nn)*2.0, season="ols"); el = time.time() - t1
    peak = tracemalloc.get_traced_memory()[1] / 2**20; tracemalloc.stop()
    mem[nn] = dict(peak_MB=round(peak, 1), sec=round(el, 2))
out["just_jumps_1d_weight_memory"] = mem; print(mem)
json.dump(out, open("p7_results.json", "w"), indent=1, default=str)
