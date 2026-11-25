import time, numpy as np, tspec, warnings; warnings.simplefilter("ignore")
rng=np.random.default_rng(0); n=192; t=np.sort(rng.uniform(2000,2008,n)); y=np.cos(2*np.pi*t)+rng.normal(0,.15,n)
for s in ("ols","allssa"):
    t0=time.time(); r=tspec.just_monitor(t,y,start=96,season=s,alpha=0.05); print(s, time.time()-t0, r.detected, len(r.steps))
