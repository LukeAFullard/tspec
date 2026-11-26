"""Re-audit: regression-equivalence battery. Runs the public API on fixed seeded inputs and
writes a digest of every numeric output, so two code versions can be compared exactly.
Run: PYTHONPATH=<src> python audit/reaudit_equivalence.py <out.json>
"""
import json
import sys
import warnings

import numpy as np

import tspec

warnings.simplefilter("ignore")


def series(seed, n=150, jump=0.0, bend=0.0, cens=0.0, season=0.3):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(2000, 2000 + n / 24, n))
    tm = 2000 + n / 48
    y = 2 + 0.1 * (t - 2000) + season * np.cos(2 * np.pi * t) + jump * (t >= tm) - bend * np.maximum(0, t - tm)
    y = y + rng.normal(0, 0.3, n)
    if cens > 0:
        dl = float(np.quantile(y, cens))
        y = np.array([f"<{dl:.6g}" if v < dl else float(v) for v in y], dtype=object)
    return t, y


def digest(obj):
    out = []
    def walk(x):
        if isinstance(x, (float, int, np.floating, np.integer)) and not isinstance(x, bool):
            out.append(float(x))
        elif isinstance(x, np.ndarray) and x.dtype.kind in "fiu":
            out.extend(np.asarray(x, dtype=float).ravel().tolist())
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
        elif hasattr(x, "__dataclass_fields__"):
            for k in x.__dataclass_fields__:
                if k in ("diagnostics", "censoring", "times_raw", "times", "n_imputations",
                         "consecutive_k", "consecutive_k_calibrated", "tp_calibration"):
                    continue
                walk(getattr(x, k))
    walk(obj)
    a = np.array(out, dtype=float)
    return {"n": int(a.size), "nan": int(np.isnan(a).sum()), "sum": float(np.nansum(a)), "abs": float(np.nansum(np.abs(a)))}



def main() -> None:
    res = {}
    for s in range(3):
        t, y = series(s, jump=0.8 if s else 0.0)
        res[f"lssa{s}"] = digest(tspec.lssa(t, y))
        res[f"allssa{s}"] = digest(tspec.allssa(t, y))
        res[f"just_jumps_ols{s}"] = digest(tspec.just_jumps(t, y, season="ols"))
        res[f"just_jumps_allssa{s}"] = digest(tspec.just_jumps(t, y))
        res[f"just_jumps_huber{s}"] = digest(tspec.just_jumps(t, y, season="ols", robust="huber"))
        res[f"decompose{s}"] = digest(tspec.just_decompose(t, y))
        tc, yc = series(10 + s, jump=0.8, cens=0.3, season=0.0)
        res[f"just_jumps_cens{s}"] = digest(tspec.just_jumps(tc, yc, season="none"))
        res[f"analyze_cens{s}"] = digest(tspec.analyze_series(tc, yc).segments)
        res[f"monitor_pocock{s}"] = digest(tspec.just_monitor(t, y, start=96, season="ols", stop_on_first=False))
        res[f"monitor_obf{s}"] = digest(tspec.just_monitor(t, y, start=96, season="ols", sequential_control="obrien_fleming", horizon=54, stop_on_first=False))
        tb, yb = series(20 + s, bend=1.0, season=0.0)
        res[f"stpd{s}"] = digest(tspec.stpd(tb, yb))
        res[f"stpd_huber{s}"] = digest(tspec.stpd(tb, yb, robust="huber"))
        res[f"stpd_cens{s}"] = digest(tspec.stpd(*series(30 + s, bend=1.0, cens=0.3, season=0.0)))
        res[f"analyze{s}"] = digest(tspec.analyze_series(t, y).segments)
    json.dump(res, open(sys.argv[1], "w"), indent=0)
    print(len(res), "digests")


if __name__ == "__main__":
    main()
