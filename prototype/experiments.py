import numpy as np, json, time
from fastpot import FastInternalField, _cheb_coeffs_powers, choose_K, gh_to_complex
from direct import direct_field
from models import model, a
rng = np.random.default_rng(7); out = {}
# 1. accuracy vs requested eps
M = 20000
r = rng.uniform(a+250, a+800, M); th = np.arccos(rng.uniform(-1,1,M)); ph = rng.uniform(0,2*np.pi,M)
th[:2] = [0.0, np.pi]
acc = {}
for L in (13, 185):
    g,h = model(L); d = direct_field(g,h,a,r,th,ph); B = np.sqrt(d[0]**2+d[1]**2+d[2]**2).max()
    rows=[]
    for eps in [1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-11,1e-12,1e-13]:
        T = FastInternalField(g,h,rmin=a+250,rmax=a+800,eps=eps); f=T(r,th,ph)
        e = max(np.abs(x-y).max() for x,y in zip(f,d))
        rows.append(dict(eps=eps,K=T.K,maxerr=e,rel=e/B))
    acc[L]=rows; print(L, [(x['eps'],x['K'],f"{x['rel']:.1e}") for x in rows])
out['accuracy']=acc
# 2. K vs shell thickness
alts = [50,100,200,400,800,1600,3200,6400,12800,25000]
Kt = {}
for L in (13,45,185):
    g,h = model(L); C = gh_to_complex(g,h)
    for eps in (1e-6,1e-10):
        Kt[f"{L}_{eps}"] = [choose_K(a/(a+dz), 1.0, L, C, eps) for dz in alts]
out['K_vs_thickness']=dict(alt_km=alts, K=Kt); print(Kt)
# 3. Chebyshev coefficient decay of u^q over a satellite shell
u0,u1 = a/(a+800), a/(a+250)
cc = np.abs(_cheb_coeffs_powers(u0,u1,187,187))
out['cheb_decay'] = {q: (cc[:,q]/ (u1**q)).tolist() for q in (15,50,100,187)}
# 4. timing breakdown at M=1e6
M=10**6; r = rng.uniform(a+250,a+800,M); th=np.arccos(rng.uniform(-1,1,M)); ph=rng.uniform(0,2*np.pi,M)
bd={}
for L in (13,185):
    g,h=model(L)
    for eps in (1e-6,1e-10):
        T=FastInternalField(g,h,rmin=a+250,rmax=a+800,eps=eps)
        t=time.perf_counter(); u=a/r; psi=np.arccos((u-T.uc)/T.ud); p2=np.mod(ph+np.pi,2*np.pi)-np.pi; t1=time.perf_counter()-t
        t=time.perf_counter(); T.plan.setpts(psi,th,p2); t2=time.perf_counter()-t
        t=time.perf_counter(); o=T.plan.execute(T._Fs); t3=time.perf_counter()-t
        t=time.perf_counter(); o=(o*np.exp(1j*T._m0*p2)[None]).real; t4=time.perf_counter()-t
        bd[f"{L}_{eps}"]=dict(K=T.K, map=t1,setpts=t2,execute=t3,phase=t4, grid=[int(x) for x in T._Fs.shape[1:]])
        print(L,eps,bd[f"{L}_{eps}"])
out['breakdown']=bd
json.dump(out, open('experiments.json','w'), indent=1, default=float)
