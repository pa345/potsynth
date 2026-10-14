import numpy as np, time, json
from fused2d import FusedField2D
from fastpot import FastInternalField
from models import model, a
rng=np.random.default_rng(0); Mmax=2*10**6
r=rng.uniform(a+250,a+800,Mmax); th=np.arccos(rng.uniform(-1,1,Mmax)); ph=rng.uniform(0,2*np.pi,Mmax)
direct_pp = {b['L']: b['t_direct']/b['M'] for b in json.load(open('bench_results.json')) if b['M']==10**6 and b['cfg']=='double 1e-6'}
def tb(f,rep=2):
    best=1e9
    for _ in range(rep):
        t=time.perf_counter(); f(); best=min(best,time.perf_counter()-t)
    return best
FusedField2D(*model(13),rmin=a+250,rmax=a+800,eps=1e-6)(r[:100],th[:100],ph[:100])  # jit warm-up
res=[]
for L in (13,20,45,100,185):
    g,h=model(L)
    for eps in (1e-6,1e-10):
        t0=time.perf_counter(); T3=FastInternalField(g,h,rmin=a+250,rmax=a+800,eps=eps); p3=time.perf_counter()-t0
        t0=time.perf_counter(); F2=FusedField2D(g,h,rmin=a+250,rmax=a+800,eps=eps); p2=time.perf_counter()-t0
        for M in (10**5,10**6,2*10**6):
            t3=tb(lambda: T3(r[:M],th[:M],ph[:M])); t2=tb(lambda: F2(r[:M],th[:M],ph[:M]))
            d=direct_pp[L]*M
            res.append(dict(L=L,eps=eps,M=M,K=T3.K,rho=F2.rho,w=F2.w,t3=t3,t2=t2,pre3=p3,pre2=p2,direct=d))
            print(f"L={L:3d} eps={eps:g} M={M:7d} K={T3.K:2d} rho={F2.rho:2d}  3D(FINUFFT) {t3:6.3f}s  fused-2D {t2:6.3f}s  ratio x{t3/t2:4.2f} | vs direct: 3D x{d/t3:6.1f}  2D x{d/t2:6.1f} | pre {p3:.2f}/{p2:.2f}s", flush=True)
json.dump(res,open('bench_fused.json','w'),indent=1)
