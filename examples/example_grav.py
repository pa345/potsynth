"""
example_grav.py -- evaluate a gravity field model (potential and gravitational
acceleration) at many scattered points with potsynth.

With an argument, the model is read from an ICGEM .gfc file (e.g. EGM2008, EIGEN-6C4
or XGM2019e from https://icgem.gfz-potsdam.de), optionally truncated at degree lmax:

    POTSYNTH_LIBRARY=/path/to/libpotsynth.so python example_grav.py EGM2008.gfc 720

Without an argument, a synthetic Earth-like model up to degree 360 is used.
"""
import sys
import time

import numpy as np

import potsynth

rmin, rmax = 6378137.0 + 200e3, 6378137.0 + 500e3     # radial range of the points (m)
rng = np.random.default_rng(1)

# 1. coefficients: from a .gfc file, or synthetic arrays C[l, m], S[l, m]
if len(sys.argv) > 1:
    lmax = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # 0 = all degrees
    gfc = potsynth.read_gfc(sys.argv[1], lmax)
    coef, GM, a = gfc.coef, gfc.GM, gfc.a
    print(f"model:       {gfc.modelname}, degrees 0-{gfc.lmax}, tide system {gfc.tide_system}")
else:
    L = 360
    C = np.zeros((L + 1, L + 1))
    S = np.zeros((L + 1, L + 1))
    for l in range(2, L + 1):
        C[l, : l + 1] = rng.normal(0.0, 1e-5 / l**2, l + 1)
        S[l, 1 : l + 1] = rng.normal(0.0, 1e-5 / l**2, l)
    C[0, 0], C[2, 0] = 1.0, -0.484165143790e-03
    coef, GM, a = (C, S), 3.986004415e14, 6378136.3
    print(f"model:       synthetic, degrees 0-{L}")

# 2. evaluation points: random in the 200-500 km shell
npts = 1_000_000
r = rng.uniform(rmin, rmax, npts)
theta = np.arccos(rng.uniform(-1.0, 1.0, npts))     # colatitude in [0, pi]
phi = rng.uniform(0.0, 2.0 * np.pi, npts)

# 3. set-up: done once per model and radial range
t0 = time.perf_counter()
model = potsynth.GravModel(coef, GM, a, quantities="V,g", rmin=rmin, rmax=rmax, eps=1e-10,
                           npts_hint=npts)
print(f"set-up:      {time.perf_counter() - t0:.3f} s")
info = model.info
print(f"method:      {info['method']}, window w = {info['w']}, radial rank = {info['rho']}, "
      f"grid {info['grid_bytes'] / 2**20:.1f} MB")

# 4. evaluation (r in m): a named tuple (V, g_r, g_theta, g_phi)
t0 = time.perf_counter()
field = model(r, theta, phi)
print(f"evaluation:  {npts} points in {time.perf_counter() - t0:.3f} s")

# 5. compare the first 1000 points with the direct sum, quantity by quantity
direct = potsynth.GravModel(coef, GM, a, quantities="V,g", method="direct")
fd = direct(r[:1000], theta[:1000], phi[:1000])
for name in model.quantities:
    x, xd = getattr(field, name)[:1000], getattr(fd, name)
    err = np.max(np.abs(x - xd))
    print(f"fast vs direct (1000 points), {name:7s}: max difference {err:.2e} "
          f"({err / np.max(np.abs(xd)):.1e} relative)")

# 6. a 1-degree map of the radial acceleration at 250 km altitude (broadcasting), in mGal
colat = np.radians(np.arange(0.0, 180.5, 1.0))[:, None]
lon = np.radians(np.arange(0.0, 360.0, 1.0))[None, :]
gr_map = model(6378137.0 + 250e3, colat, lon).g_r * 1e5
print(f"\nmap of g_r at 250 km: shape {gr_map.shape}, "
      f"range [{gr_map.min():.1f}, {gr_map.max():.1f}] mGal")
