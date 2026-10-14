"""
Generate reference field values for the potsynth tests, using the prototype's
extended-precision (x87 long double, ~19 digits) synthesis (prototype/refld.py),
an implementation independent of the C library.

Usage: python gen_reference.py OUTDIR

File format (little endian):
  int64  lmax, npts
  float64 a, rmin, rmax
  float64 gh[ncoef]          Schmidt Gauss coefficients, green2 layout, lmin=1, mmax=lmax
  float64 r[npts], theta[npts], phi[npts]
  float64 Br[npts], Btheta[npts], Bphi[npts]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prototype"))
from refld import reference_field  # noqa: E402

A = 6371.2


def model(L, seed):
    """Core-like spectrum for l <= 13 (dipole ~ -29000 nT), flat lithospheric spectrum above."""
    rng = np.random.default_rng(seed)
    g = np.zeros((L + 1, L + 1))
    h = np.zeros((L + 1, L + 1))
    for l in range(1, L + 1):
        if l <= 13:
            Rl = 9e8 * (50.0 / 9e8) ** ((l - 1) / 12.0)
        elif l <= 15:
            Rl = 5.0
        else:
            Rl = 0.5
        sd = np.sqrt(Rl / ((l + 1) * (2 * l + 1)))
        g[l, : l + 1] = rng.normal(0, sd, l + 1)
        h[l, 1 : l + 1] = rng.normal(0, sd, l)
    g[1, 0] = -29000.0
    return g, h


def green2_vector(g, h):
    L = g.shape[0] - 1
    out = []
    for l in range(1, L + 1):
        for m in range(-l, l + 1):
            out.append(h[l, -m] if m < 0 else g[l, m])
    return np.array(out)


def points(rmin, rmax, n, seed):
    rng = np.random.default_rng(seed + 1000)
    r = rng.uniform(rmin, rmax, n)
    theta = np.arccos(rng.uniform(-1.0, 1.0, n))
    phi = rng.uniform(-np.pi, 3.0 * np.pi, n)      # also outside [0, 2pi)
    special = []
    for ph in (0.0, 1.0, np.pi, 2.0 * np.pi - 1e-15, -2.5, 100.0):
        for th in (0.0, 1e-300, 1e-10, 1e-6, np.pi / 2, np.pi - 1e-10, np.pi):
            for rr in (rmin, rmax, 0.5 * (rmin + rmax)):
                special.append((rr, th, ph))
    s = np.array(special)
    return (np.concatenate([s[:, 0], r]), np.concatenate([s[:, 1], theta]),
            np.concatenate([s[:, 2], phi]))


def write(path, L, rmin, rmax, g, h, r, theta, phi, B):
    with open(path, "wb") as f:
        np.array([L, r.size], dtype="<i8").tofile(f)
        np.array([A, rmin, rmax], dtype="<f8").tofile(f)
        green2_vector(g, h).astype("<f8").tofile(f)
        for v in (r, theta, phi):
            np.asarray(v, dtype="<f8").tofile(f)
        for v in B:
            np.asarray(v, dtype=np.float64).astype("<f8").tofile(f)


def main():
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)
    cases = [  # (L, rmin, rmax, npts, seed)
        (13, A, A + 1000.0, 1500, 1),
        (60, A + 250.0, A + 800.0, 1500, 2),
        (185, A + 250.0, A + 800.0, 1500, 3),
        (185, A, 2.0 * A, 1000, 4),          # thick shell
    ]
    for L, rmin, rmax, n, seed in cases:
        g, h = model(L, seed)
        r, theta, phi = points(rmin, rmax, n, seed)
        B = reference_field(g, h, A, r, theta, phi)
        name = os.path.join(outdir, f"ref_L{L}_s{seed}.bin")
        write(name, L, rmin, rmax, g, h, r, theta, phi, B)
        print(f"wrote {name}: {r.size} points")


if __name__ == "__main__":
    main()
