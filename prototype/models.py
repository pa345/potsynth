import numpy as np, time, json, sys, ppigrf, datetime
a = 6371.2
G, H = ppigrf.ppigrf.read_shc(); row = G.index.get_indexer([datetime.datetime(2025,1,1)])[0]

def model(L, seed=1):
    """IGRF-2025 core (n<=13) + CHAOS-like lithospheric spectrum (flat ~0.5 nT^2/degree at surface)."""
    rng = np.random.default_rng(seed)
    g = np.zeros((L+1,L+1)); h = np.zeros((L+1,L+1))
    for (n,m),v in G.iloc[row].items():
        if n<=L: g[n,m]=v
    for (n,m),v in H.iloc[row].items():
        if n<=L: h[n,m]=v
    for n in range(14, L+1):
        Rn = 0.5 if n>=16 else 5.0
        sd = np.sqrt(Rn/((n+1)*(2*n+1)))
        g[n,:n+1] = rng.normal(0,sd,n+1); h[n,1:n+1] = rng.normal(0,sd,n)
    return g, h

