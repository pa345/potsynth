import numpy as np, time, json
from fastpot import FastInternalField, FastInternalField2D
from models import model, a
rng=np.random.default_rng(0); Mmax=2*10**6
r=rng.uniform(a+250,a+800,Mmax); th=np.arccos(rng.uniform(-1,1,Mmax)); ph=rng.uniform(0,2*np.pi,Mmax)
direct_pp = {}
for b in json.load(open('bench_results.json')):
    if b['M']==10**6 and b['cfg']=='double 1e-6': direct_pp[b['L']] = b['t_direct']/1e6
def tb(f,rep=2):
    best=1e9
    for _ in range(rep):
        t=time.perf_counter(); f(); best=min(best,time.perf_counter()-t)
    return best
res=[]
for L in (13,20,45,100,185):
    g,h=model(L)
    for eps,prec in ((1e-6,"double"),(1e-10,"double"),(1e-6,"single")):
        t0=time.perf_counter(); T3=FastInternalField(g,h,rmin=a+250,rmax=a+800,eps=eps,precision=prec); p3=time.perf_counter()-t0
        t0=time.perf_counter(); T2=FastInternalField2D(g,h,rmin=a+250,rmax=a+800,eps=eps,precision=prec); p2=time.perf_counter()-t0
        for M in (10**5, 10**6, 2*10**6):
            t3=tb(lambda: T3(r[:M],th[:M],ph[:M])); t2=tb(lambda: T2(r[:M],th[:M],ph[:M]))
            d=direct_pp[L]*M
            row=dict(L=L,eps=eps,prec=prec,M=M,K=T3.K,rho=T2.rho,t3=t3,t2=t2,pre3=p3,pre2=p2,direct=d)
            res.append(row)
            print(f"L={L:3d} {prec} eps={eps:g} M={M:7d} K={T3.K:2d} rho={T2.rho:2d}  3D {t3:6.3f}s  2D {t2:6.3f}s  2D gain x{t3/t2:4.2f} | vs direct 3D x{d/t3:6.1f} 2D x{d/t2:6.1f}", flush=True)
json.dump(res,open('bench2d.json','w'),indent=1)
