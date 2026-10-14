"""Old (complex, full doubled sphere) fused 2-D vs new real-valued half-sphere FastField,
at identical window width, oversampling and radial rank tolerance."""
import numpy as np, time, json
from fastfield import FastField
from fused2d import FusedField2D
from models import model, a
rng=np.random.default_rng(0); M=10**6
r=rng.uniform(a+250,a+800,M); th=np.arccos(rng.uniform(-1,1,M)); ph=rng.uniform(0,2*np.pi,M)
direct_pp = {b['L']: b['t_direct']/b['M'] for b in json.load(open('bench_results.json')) if b['M']==10**6 and b['cfg']=='double 1e-6'}
def tb(f,rep=3):
    best=1e9
    for _ in range(rep):
        t=time.perf_counter(); out=f(); best=min(best,time.perf_counter()-t)
    return best,out
FastField(*model(13),rmin=a+250,rmax=a+800,eps=1e-6)(r[:10],th[:10],ph[:10])
FusedField2D(*model(13),rmin=a+250,rmax=a+800,eps=1e-6)(r[:10],th[:10],ph[:10])
res=[]
for L in (13,20,45,100,185):
    g,h=model(L)
    for w,rt in ((8,1e-8),(12,1e-12),(14,1e-15)):
        t0=time.perf_counter(); N=FastField(g,h,rmin=a+250,rmax=a+800,w=w,sigma=2.0,rank_tol=0.3*rt); pn=time.perf_counter()-t0
        t0=time.perf_counter(); O=FusedField2D(g,h,rmin=a+250,rmax=a+800,eps=1e-6,w=w,sigma=2.0,eps_radial=rt); po=time.perf_counter()-t0
        tn,fn=tb(lambda: N(r,th,ph)); to,fo=tb(lambda: O(r,th,ph))
        d=direct_pp[L]*M
        diff=max(np.abs(x-y).max() for x,y in zip(fn,fo))
        row=dict(L=L,w=w,rho_new=N.rho,rho_old=O.rho,t_new=tn,t_old=to,pre_new=pn,pre_old=po,mem_new=N.nbytes,mem_old=O.grid.nbytes,direct=d,diff_nT=diff)
        res.append(row)
        print(f"L={L:3d} w={w:2d} rho {O.rho}->{N.rho}  old {to:5.2f}s  new {tn:5.2f}s  x{to/tn:4.2f}  | mem {O.grid.nbytes/1e6:5.0f} -> {N.nbytes/1e6:5.0f} MB | vs direct x{d/tn:6.1f} | pre {po:.2f}/{pn:.2f}s | |old-new| {diff:.1e} nT",flush=True)
json.dump(res,open('bench_fastfield.json','w'),indent=1)
