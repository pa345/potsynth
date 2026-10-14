# Time-dependent IGRF: g(t) = g(2020) + (t-2020)/5 * (g(2025)-g(2020)) -- exactly what ppigrf does.
import numpy as np, ppigrf, datetime
from fastpot import FastInternalField
G,H = ppigrf.ppigrf.read_shc(); L=13; a=6371.2
def gh(date):
    i = G.index.get_indexer([date])[0]; g=np.zeros((L+1,L+1)); h=np.zeros((L+1,L+1))
    for (n,m),v in G.iloc[i].items(): g[n,m]=v
    for (n,m),v in H.iloc[i].items(): h[n,m]=v
    return g,h
g0,h0 = gh(datetime.datetime(2020,1,1)); g1,h1 = gh(datetime.datetime(2025,1,1))
T = FastInternalField(np.stack([g0,(g1-g0)]), np.stack([h0,(h1-h0)]), rmin=a, rmax=a+1000, eps=1e-10)
rng=np.random.default_rng(3); M=300
r=rng.uniform(a,a+1000,M); th=np.arccos(rng.uniform(-.99,.99,M)); ph=rng.uniform(0,2*np.pi,M)
days = rng.uniform(0, 5*365.25-1, M)
dates = [datetime.datetime(2020,1,1)+datetime.timedelta(days=float(d)) for d in days]
frac = np.array([(d-datetime.datetime(2020,1,1)).total_seconds() for d in dates]) / (datetime.datetime(2025,1,1)-datetime.datetime(2020,1,1)).total_seconds()
W = np.stack([np.ones(M), frac], 1)
Br,Bt,Bp = T(r,th,ph,weights=W)
err=0
for i in range(M):
    ref = [np.ravel(x)[0] for x in ppigrf.igrf_gc(r[i], np.degrees(th[i]), np.degrees(ph[i]), dates[i])]
    err=max(err, abs(Br[i]-ref[0]), abs(Bt[i]-ref[1]), abs(Bp[i]-ref[2]))
print("time-dependent IGRF, 300 random dates 2020-2025: max |fast - ppigrf| =", err, "nT")
# single precision check
Ts = FastInternalField(g1,h1,rmin=a,rmax=a+1000,eps=1e-6,precision="single"); Td = FastInternalField(g1,h1,rmin=a,rmax=a+1000,eps=1e-12)
print("single-precision max err nT:", max(np.abs(x-y).max() for x,y in zip(Ts(r,th,ph),Td(r,th,ph))))
