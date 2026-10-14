"""FastField: FINUFFT backend (library interpolation) vs numba backend (own fused loop),
same requested accuracy. Accuracy measured against the long-double reference."""
import numpy as np, time, json
from fastfield import FastField
from refld import reference_field
from models import model, a
rng=np.random.default_rng(0); M=10**6
r=rng.uniform(a+250,a+800,M); th=np.arccos(rng.uniform(-1,1,M)); ph=rng.uniform(0,2*np.pi,M)
P=np.load('ref_points.npz'); rr,tt,pp=P['r'],P['th'],P['ph']
direct={x['L']:x['t_direct'] for x in json.load(open('bench_final.json')) if x['M']==M and x['eps']==1e-6}
def tb(f,rep=3):
    best=1e9
    for _ in range(rep):
        t=time.perf_counter(); f(); best=min(best,time.perf_counter()-t)
    return best
FastField(*model(13),rmin=a+250,rmax=a+800,backend="numba")(r[:10],th[:10],ph[:10])
res=[]
for L in (13,20,45,100,185):
    g,h=model(L); ref=[np.asarray(x,float) for x in reference_field(g,h,a,rr,tt,pp)]; B=max(np.abs(x).max() for x in ref)
    for eps in (1e-6,1e-10,1e-13):
        row=dict(L=L,eps=eps,direct=direct[L])
        for be in ("finufft","numba"):
            t0=time.perf_counter(); F=FastField(g,h,rmin=a+250,rmax=a+800,eps=eps,backend=be); pre=time.perf_counter()-t0
            err=max(np.abs(x-y).max() for x,y in zip(F(rr,tt,pp),ref))/B
            t=tb(lambda: F(r,th,ph))
            row[be]=dict(t=t,pre=pre,err=err,w=F.w,rho=F.rho,mem=F.nbytes)
        res.append(row)
        f,n=row["finufft"],row["numba"]
        print(f"L={L:3d} eps={eps:g}  FINUFFT {f['t']:.2f}s (w={f['w']}, err {f['err']:.1e}, x{direct[L]/f['t']:.1f})  numba {n['t']:.2f}s (w={n['w']}, err {n['err']:.1e}, x{direct[L]/n['t']:.1f})  numba/FINUFFT speed {f['t']/n['t']:.2f}",flush=True)
json.dump(res,open('bench_backends.json','w'),indent=1)
