"""
example_mag.py -- evaluate a lithospheric-type magnetic field model at many
scattered points with potsynth.

The model is synthetic (a dipole-dominated core field plus a flat lithospheric
spectrum up to degree 185, similar in size to CHAOS); replace g and h with real
Gauss coefficients.

Run:  POTSYNTH_LIBRARY=/path/to/libpotsynth.so python example_mag.py
"""
import time

import numpy as np

import potsynth

L = 185                       # maximum degree
a = 6371.2                    # reference radius (km)
rmin, rmax = a + 250.0, a + 800.0
rng = np.random.default_rng(1)

# 1. Gauss coefficients g[n, m], h[n, m] in nT
g = np.zeros((L + 1, L + 1))
h = np.zeros((L + 1, L + 1))
for n in range(1, L + 1):
    Rn = 9e8 * (50.0 / 9e8) ** ((n - 1) / 12.0) if n <= 13 else 0.5
    sd = np.sqrt(Rn / ((n + 1) * (2 * n + 1)))
    g[n, : n + 1] = rng.normal(0.0, sd, n + 1)
    h[n, 1 : n + 1] = rng.normal(0.0, sd, n)
g[1, 0] = -29400.0

# 2. evaluation points: random in the 250-800 km shell
npts = 1_000_000
r = rng.uniform(rmin, rmax, npts)
theta = np.arccos(rng.uniform(-1.0, 1.0, npts))     # colatitude in [0, pi]
phi = rng.uniform(0.0, 2.0 * np.pi, npts)

# 3. set-up: done once per model and radial range
t0 = time.perf_counter()
model = potsynth.MagModel((g, h), a=a, rmin=rmin, rmax=rmax, eps=1e-10)
print(f"set-up:      {time.perf_counter() - t0:.3f} s")
info = model.info
print(f"method:      {info['method']}, window w = {info['w']}, radial rank = {info['rho']}, "
      f"grid {info['grid_bytes'] / 2**20:.1f} MB")

# 4. evaluation: arrays of any (broadcastable) shape
t0 = time.perf_counter()
Br, Btheta, Bphi = model(r, theta, phi)
print(f"evaluation:  {npts} points in {time.perf_counter() - t0:.3f} s")

# 5. compare the first 1000 points with the direct sum
direct = potsynth.MagModel((g, h), a=a, method="direct")
Bd = np.array(direct(r[:1000], theta[:1000], phi[:1000]))
err = np.max(np.abs(np.array([Br[:1000], Btheta[:1000], Bphi[:1000]]) - Bd))
print(f"fast vs direct (1000 points): max difference {err:.2e} nT "
      f"({err / np.max(np.abs(Bd)):.1e} relative)")

# 6. NEC components, and a 1-degree map at 450 km altitude (broadcasting)
BN, BE, BC = potsynth.rtp_to_nec(Br, Btheta, Bphi)
print("\n      r [km]   colat [deg]  lon [deg]      B_N [nT]      B_E [nT]      B_C [nT]")
for i in range(4):
    print(f"{r[i]:12.3f} {np.degrees(theta[i]):12.4f} {np.degrees(phi[i]):10.4f} "
          f"{BN[i]:13.4f} {BE[i]:13.4f} {BC[i]:13.4f}")

colat = np.radians(np.arange(0.0, 180.5, 1.0))[:, None]
lon = np.radians(np.arange(0.0, 360.0, 1.0))[None, :]
Br_map, _, _ = model(a + 450.0, colat, lon)
print(f"\nmap of B_r at 450 km: shape {Br_map.shape}, "
      f"range [{Br_map.min():.1f}, {Br_map.max():.1f}] nT")
