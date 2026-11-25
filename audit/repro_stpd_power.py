"""Why does STPD miss an obvious single turning point at n=500, sigma=0.6 (MC7)? Seed 7500.
Run: python audit/repro_stpd_power.py
"""
import warnings
import numpy as np
import tspec
from tspec.stpd import _tpd_window
warnings.simplefilter("ignore")
rng = np.random.default_rng(7500)
n = 500; span = n / 24; ttp = 2000 + span / 2
reasons = {"accepted": 0, "pval": 0, "snr": 0, "ndri": 0, "fb": 0}
det = 0
for r in range(100):
    t = np.sort(rng.uniform(2000, 2000 + span, n))
    T = 3 + 0.5 * (t - 2000) - 1.0 * np.maximum(0, t - ttp)
    y = T + rng.normal(0, 0.6, n)
    s = tspec.stpd(t, y)
    det += any(abs(float(x) - ttp) < 0.5 for x in s.tp_times)
    # diagnose the window centred on the TP
    rate = int(n / span); size = 5 * rate
    k = int(np.argmin(np.abs(t - ttp))); a = max(0, k - size // 2); b = a + size
    st = _tpd_window(t[a:b], y[a:b], max(2, min(rate, size // 5)))
    if st.pval >= 0.01: reasons["pval"] += 1
    elif st.snr_centered <= 1.0: reasons["snr"] += 1
    elif st.ndri_rmse >= 0.3: reasons["ndri"] += 1
    elif st.fb_diff > 3: reasons["fb"] += 1
    else: reasons["accepted"] += 1
print("detection rate (within 0.5 yr):", det / 100)
print("centred-window gate that rejected:", reasons)
print("example centred window: fb_diff=%d pval=%.2e snr=%.2f ndri=%.3f" % (st.fb_diff, st.pval, st.snr_centered, st.ndri_rmse))
