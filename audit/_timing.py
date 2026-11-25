import time, numpy as np, tspec, warnings
warnings.simplefilter("ignore")
rng = np.random.default_rng(1)
for n in (50,150,500):
    t = np.sort(rng.uniform(2010, 2010 + n/30, n))  # ~30 obs/yr
    y = 0.3*(t-2010) + np.cos(2*np.pi*t) + rng.normal(0,0.3,n)
    for name, f in [
        ("lssa", lambda: tspec.lssa(t,y,omega=np.arange(0.1,15,0.1))),
        ("allssa", lambda: tspec.allssa(t,y)),
        ("jj_ols", lambda: tspec.just_jumps(t,y,season="ols")),
        ("jj_allssa", lambda: tspec.just_jumps(t,y,season="allssa")),
        ("jd_ols", lambda: tspec.just_decompose(t,y,season="ols")),
        ("jd_allssa", lambda: tspec.just_decompose(t,y,season="allssa")),
        ("stpd", lambda: tspec.stpd(t,y)),
        ("mon", lambda: tspec.just_monitor(t,y,start=int(0.6*n),season="ols",stop_on_first=False)),
        ("pipe", lambda: tspec.analyze_series(t,y)),
    ]:
        t0=time.time()
        try:
            f(); dt=time.time()-t0
            print(n,name,round(dt,3))
        except Exception as e:
            print(n,name,"ERR",type(e).__name__,e)
