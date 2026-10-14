import numpy as np, ppigrf, datetime, time
from fastpot import FastInternalField
from direct import direct_field

date = datetime.datetime(2025,1,1)
G, H = ppigrf.ppigrf.read_shc()
row = G.index.get_indexer([date], method='nearest')[0]
print("epoch used:", G.index[row])
L = 13
g = np.zeros((L+1,L+1)); h = np.zeros((L+1,L+1))
for (n,m), v in G.iloc[row].items(): g[n,m] = v
for (n,m), v in H.iloc[row].items(): h[n,m] = v

rng = np.random.default_rng(0)
M = 2000
r = 6371.2 + rng.uniform(0, 1000, M)
th = np.arccos(rng.uniform(-1,1,M)); th[:4] = [0, np.pi, 1e-9, np.pi-1e-9]   # include poles
ph = rng.uniform(0, 2*np.pi, M)

Br_ref, Bt_ref, Bp_ref = ppigrf.igrf_gc(r, np.degrees(th), np.degrees(ph), G.index[row])
Br_ref, Bt_ref, Bp_ref = [np.ravel(x) for x in (Br_ref, Bt_ref, Bp_ref)]
Br, Bt, Bp = direct_field(g, h, 6371.2, r, th, ph)
ok = slice(4, None)  # ppigrf divides by sin(theta) at the poles
print("direct vs ppigrf  max|dB| nT:", [np.abs(a[ok]-b[ok]).max() for a,b in ((Br,Br_ref),(Bt,Bt_ref),(Bp,Bp_ref))])

for eps in (1e-6, 1e-9, 1e-12):
    T = FastInternalField(g, h, rmin=6371.2, rmax=6371.2+1000, eps=eps)
    fr, ft, fp = T(r, th, ph)
    err = max(np.abs(fr-Br).max(), np.abs(ft-Bt).max(), np.abs(fp-Bp).max())
    print(f"eps={eps:g}  K={T.K}  fast vs direct max|dB| = {err:.3e} nT  (|B|max={np.abs(Br).max():.0f})")
print("poles fast:", fr[:4], ft[:4], fp[:4]); print("poles direct:", Br[:4], Bt[:4], Bp[:4])
# exact radial representation (K = L+2) over a wide range
T = FastInternalField(g, h, rmin=6371.2, rmax=6.6*6371.2, eps=1e-12)
r2 = rng.uniform(6371.2, 6.6*6371.2, M)
d = direct_field(g, h, 6371.2, r2, th, ph); f = T(r2, th, ph)
print("wide range 1-6.6 a: K=",T.K," rel err", max(np.abs(a-b).max() for a,b in zip(d,f))/np.abs(d[0]).max())
