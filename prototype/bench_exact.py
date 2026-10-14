import numpy as np, time, json
from fused2d import FusedField2D
from direct import direct_field
from models import model, a
rng=np.random.default_rng(0); M=10**6
r=rng.uniform(a+250,a+800,M); th=np.arccos(rng.uniform(-1,1,M)); ph=rng.uniform(0,2*np.pi,M)
direct_pp = {b['L']: b['t_direct']/b['M'] for b in json.load(open('bench_results.json')) if b['M']==10**6 and b['cfg']=='double 1e-6'}
def tb(f,rep=2):
    best=1e9
    for _ in range(rep):
        t=time.perf_counter(); f(); best=min(best,time.perf_counter()-t)
    return best
FusedField2D(*model(13),rmin=a+250,rmax=a+800,eps=1e-6)(r[:10],th[:10],ph[:10])
res=[]
cfgs=[("machine precision, sigma=2, w=16",dict(w=16,sigma=2.0,eps_radial=1e-15)),
      ("machine precision, sigma=3, w=13",dict(w=13,sigma=3.0,eps_radial=1e-15)),
      ("1e-10, sigma=2, w=12",dict(w=12,sigma=2.0,eps_radial=1e-11))]
for L in (13,20,45,100,185):
    g,h=model(L)
    for name,kw in cfgs:
        t0=time.perf_counter(); F=FusedField2D(g,h,rmin=a+250,rmax=a+800,eps=1e-6,**kw); tp=time.perf_counter()-t0
        t=tb(lambda: F(r,th,ph)); d=direct_pp[L]*M
        res.append(dict(L=L,cfg=name,rho=F.rho,w=F.w,t=t,pre=tp,direct=d,mem=F.grid.nbytes))
        print(f"L={L:3d} {name:34s} rho={F.rho:2d}  fast {t:6.2f}s  direct {d:7.2f}s  speedup x{d/t:6.1f}  (pre {tp:.2f}s, grid {F.grid.nbytes/1e6:.0f} MB)",flush=True)
json.dump(res,open('bench_exact.json','w'),indent=1)
