import numpy as np, time, json
from fastpot import FastInternalField
from direct import direct_field
from models import model, a

def tbest(f, rep=3):
    best = 1e9
    for _ in range(rep):
        t=time.perf_counter(); out=f(); best=min(best,time.perf_counter()-t)
    return best, out

if __name__ == "__main__":
    rmin, rmax = a+250, a+800
    direct_field(*model(13), a, np.full(10,a+300), np.ones(10), np.ones(10))
    rng = np.random.default_rng(0)
    Mmax = 2*10**6
    r = rng.uniform(rmin, rmax, Mmax); th = np.arccos(rng.uniform(-1,1,Mmax)); ph = rng.uniform(0,2*np.pi,Mmax)
    results = []
    for L in [13, 20, 30, 45, 70, 100, 133, 185]:
        g, h = model(L)
        Md = min(Mmax, max(4000, int(3e7/(L+1)**2)))
        td_pp, do = tbest(lambda: direct_field(g,h,a,r[:Md],th[:Md],ph[:Md]))
        td_pp /= Md                                     # seconds per point (direct is exactly O(M))
        Bmax = np.abs(do[0]).max()
        for cfg, eps, prec in [("double 1e-10",1e-10,"double"), ("double 1e-6",1e-6,"double"), ("single 1e-6",1e-6,"single")]:
            t0=time.perf_counter(); T = FastInternalField(g, h, rmin=rmin, rmax=rmax, eps=eps, precision=prec); tpre=time.perf_counter()-t0
            for M in [10**4, 10**5, 10**6, 2*10**6]:
                tf, fo = tbest(lambda: T(r[:M],th[:M],ph[:M]), rep=3 if M<=10**5 else 2)
                k = min(M, Md)
                err = max(np.abs(x[:k]-y[:k]).max() for x,y in zip(fo,do))
                td = td_pp*M
                res = dict(L=L, cfg=cfg, K=T.K, M=M, t_pre=tpre, t_fast=tf, t_direct=td,
                           speedup=td/tf, speedup_incl_pre=td/(tf+tpre), maxerr_nT=err, relerr=err/Bmax)
                results.append(res)
                print(f"L={L:3d} {cfg:13s} K={T.K:2d} M={M:8d} pre={tpre:6.3f}s fast={tf:7.3f}s direct={td:8.2f}s  x{td/tf:6.1f} (x{td/(tf+tpre):6.1f} incl pre)  err={err:.1e} nT", flush=True)
    json.dump(results, open('bench_results.json','w'), indent=1)
