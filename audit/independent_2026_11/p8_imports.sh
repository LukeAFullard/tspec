#!/bin/bash
# P8: are optional deps lazy? Fresh venv with numpy+scipy only (+ declared hard deps separately)
set -e
D=$(mktemp -d)
python -m venv $D/v >/dev/null
$D/v/bin/pip install -q numpy scipy >/dev/null 2>&1
$D/v/bin/pip install -q --no-deps -e /home/user/tspec >/dev/null 2>&1
echo "--- numpy+scipy only (no pandas/ndimpute/matplotlib/statsmodels):"
$D/v/bin/python - <<'PY'
import sys
try:
    import tspec, numpy as np
    print("import ok; heavy modules loaded:", sorted(m for m in ("pandas","ndimpute","matplotlib","statsmodels") if m in sys.modules))
    t=np.sort(np.random.default_rng(0).uniform(0,10,120)); y=np.cos(2*np.pi*t)+np.random.default_rng(1).normal(0,.3,120)
    for name,f in [("lssa",lambda: tspec.lssa(t,y,omega=[1.0])),("just_jumps",lambda: tspec.just_jumps(t,y)),("just_decompose",lambda: tspec.just_decompose(t,y)),
                   ("stpd",lambda: tspec.stpd(t,y)),("analyze_series",lambda: tspec.analyze_series(t,y)),("monitor",lambda: tspec.just_monitor(t,y,start=100)),
                   ("summary/explain",lambda: tspec.lssa(t,y,omega=[1.0]).explain()),("censored strings",lambda: tspec.just_jumps(t,["<0.1" if v<0.1 else v for v in y])),
                   ("to_dataframe",lambda: tspec.lssa(t,y,omega=[1.0]).to_dataframe()),("plot",lambda: tspec.lssa(t,y,omega=[1.0]).plot())]:
        try: f(); print(f"  {name}: ok")
        except Exception as e: print(f"  {name}: {type(e).__name__}: {str(e)[:110]}")
    print("heavy modules loaded after calls:", sorted(m for m in ("pandas","ndimpute","matplotlib","statsmodels") if m in sys.modules))
except Exception as e:
    print("IMPORT FAILED", type(e).__name__, e)
PY
rm -rf $D
