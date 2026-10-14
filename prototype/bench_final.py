"""Final benchmark of FastField (parameters chosen from eps) vs the direct sum.
Accuracy is measured against the long-double reference on 1500 points (incl. both poles)."""
import numpy as np, time, json
from fastfield import FastField
from direct import direct_field
from refld import reference_field
from models import model, a
rng=np.random.default_rng(0); Mmax=10**6
r=rng.uniform(a+250,a+800,Mmax); th=np.arccos(rng.uniform(-1,1,Mmax)); ph=rng.uniform(0,2*np.pi,Mmax)
P=np.load('ref_points.npz'); rr,tt,pp=P['r'],P['th'],P['ph']
def tb(f,rep=3):
    best=1e9
    for _ in range(rep):
        t=time.perf_counter(); f(); best=min(best,time.perf_counter()-t)
    return best
direct_field(*model(13),a,r[:10],th[:10],ph[:10]); FastField(*model(13),rmin=a+250,rmax=a+800)(r[:10],th[:10],ph[:10])
res=[]
for L in (13,20,30,45,70,100,133,185):
    g,h=model(L)
    ref=[np.asarray(x,float) for x in reference_field(g,h,a,rr,tt,pp)]; B=max(np.abs(x).max() for x in ref)
    Md=min(Mmax,max(4000,int(4e7/(L+1)**2)))
    td=tb(lambda: direct_field(g,h,a,r[:Md],th[:Md],ph[:Md]),rep=2)/Md
    derr=max(np.abs(x-y).max() for x,y in zip(direct_field(g,h,a,rr,tt,pp),ref))/B
    for eps in (1e-6,1e-10,1e-13):
        t0=time.perf_counter(); F=FastField(g,h,rmin=a+250,rmax=a+800,eps=eps); pre=time.perf_counter()-t0
        err=max(np.abs(x-y).max() for x,y in zip(F(rr,tt,pp),ref))/B
        for M in (10**4,10**5,10**6):
            tf=tb(lambda: F(r[:M],th[:M],ph[:M]))
            row=dict(L=L,eps=eps,w=F.w,K=F.K,rho=F.rho,M=M,t_fast=tf,t_pre=pre,t_direct=td*M,mem=F.nbytes,err=err,err_direct=derr)
            res.append(row)
            print(f"L={L:3d} eps={eps:g} w={F.w} rho={F.rho:2d} M={M:7d} fast {tf:7.3f}s direct {td*M:8.3f}s x{td*M/tf:6.1f} (incl pre x{td*M/(tf+pre):6.1f}) err {err:.1e} (direct {derr:.1e}) mem {F.nbytes/1e6:.0f}MB",flush=True)
json.dump(res,open('bench_final.json','w'),indent=1)
