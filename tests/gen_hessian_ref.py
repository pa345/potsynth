"""
Generate reference values of the gradient tensor (Hessian) of a potential for
the potsynth tests, with 40-digit arithmetic (mpmath), independently of the
library's spherical formulas: the Hessian is computed in Cartesian
coordinates and rotated to the spherical basis (r, theta, phi),

  T_s = R T_c R^T,  R = [[ s cp, s sp,  c ],
                         [ c cp, c sp, -s ],
                         [  -sp,   cp,  0 ]]

At the poles the spherical components refer to the frame defined by the
longitude phi of the point.

Usage: python gen_hessian_ref.py OUTDIR

File format (little endian):
  int64   lmax, npts
  float64 a
  float64 coef[ncoef]        4-pi coefficients, potsynth_idx layout, lmin=0, mmax=lmax
  float64 r[npts], theta[npts], phi[npts]
  float64 T[6][npts]         T_rr, T_rt, T_rp, T_tt, T_tp, T_pp

for the potential V = sum_{l,m} (a/r)^(l+1) Pbar_l^m(cos theta) (C cos m phi + S sin m phi).
"""
import os
import sys

import mpmath as mp
import numpy as np

mp.mp.dps = 40
A = 1.0


def coef_model(L, seed):
    """Random 4-pi coefficients with rms 1/(l+1)^2 per coefficient."""
    rng = np.random.default_rng(seed)
    C = np.zeros((L + 1, L + 1))
    S = np.zeros((L + 1, L + 1))
    for l in range(L + 1):
        sd = 1.0 / (l + 1) ** 2
        C[l, : l + 1] = rng.normal(0, sd, l + 1)
        S[l, 1 : l + 1] = rng.normal(0, sd, l)
    return C, S


def coef_vector(C, S):
    L = C.shape[0] - 1
    return np.array([S[l, -m] if m < 0 else C[l, m] for l in range(L + 1) for m in range(-l, l + 1)])


def pbar(L, x, s):
    """4-pi normalized Pbar_l^m(x), x = cos(theta), s = sin(theta), no
    Condon-Shortley phase, by the standard (unscaled) recursion in mpmath
    arithmetic. s is passed separately: sqrt(1 - x^2) would lose all precision
    next to the axis, where the numerical derivatives sample the potential."""
    P = [[mp.mpf(0)] * (L + 1) for _ in range(L + 1)]
    P[0][0] = mp.mpf(1)
    for m in range(1, L + 1):
        f = mp.sqrt(3) if m == 1 else mp.sqrt(mp.mpf(2 * m + 1) / (2 * m))
        P[m][m] = f * s * P[m - 1][m - 1]
    for m in range(0, L):
        P[m + 1][m] = mp.sqrt(2 * m + 3) * x * P[m][m]
    for m in range(0, L + 1):
        for l in range(m + 2, L + 1):
            a = mp.sqrt(mp.mpf((2 * l + 1) * (2 * l - 1)) / ((l + m) * (l - m)))
            b = mp.sqrt(mp.mpf((l + m - 1) * (l - m - 1) * (2 * l + 1)) / ((l + m) * (l - m) * (2 * l - 3)))
            P[l][m] = a * x * P[l - 1][m] - b * P[l - 2][m]
    return P


def potential(C, S, X, Y, Z):
    L = C.shape[0] - 1
    rho = mp.sqrt(X * X + Y * Y)
    r = mp.sqrt(rho * rho + Z * Z)
    ph = mp.atan2(Y, X)
    P = pbar(L, Z / r, rho / r)
    u = A / r
    V = mp.mpf(0)
    for l in range(L + 1):
        ul = u ** (l + 1)
        for m in range(l + 1):
            V += ul * P[l][m] * (C[l, m] * mp.cos(m * ph) + S[l, m] * mp.sin(m * ph))
    return V


def hessian(C, S, r, th, ph):
    r, th, ph = mp.mpf(r), mp.mpf(th), mp.mpf(ph)
    st, ct, sp, cp = mp.sin(th), mp.cos(th), mp.sin(ph), mp.cos(ph)
    p = [r * st * cp, r * st * sp, r * ct]
    H = mp.matrix(3, 3)
    for i in range(3):
        for j in range(i, 3):
            o = [0, 0, 0]
            o[i] += 1
            o[j] += 1
            H[i, j] = H[j, i] = mp.diff(lambda X, Y, Z: potential(C, S, X, Y, Z), p, tuple(o))
    R = mp.matrix([[st * cp, st * sp, ct], [ct * cp, ct * sp, -st], [-sp, cp, 0]])
    T = R * H * R.T
    return [T[0, 0], T[0, 1], T[0, 2], T[1, 1], T[1, 2], T[2, 2]]


def points(rmin, rmax, n, seed):
    rng = np.random.default_rng(seed + 1000)
    pts = []
    for ph in (0.0, 1.0, 2.5, -2.0):
        for th in (0.0, 1e-12, 1e-6, np.pi / 2, np.pi - 1e-12, np.pi):
            pts.append((rmin if ph < 1.5 else rmax, th, ph))
    for _ in range(n):
        pts.append((rng.uniform(rmin, rmax), np.arccos(rng.uniform(-1, 1)), rng.uniform(0, 2 * np.pi)))
    return np.array(pts)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    for L, seed, n in ((8, 11, 20), (24, 12, 10)):
        C, S = coef_model(L, seed)
        p = points(1.0, 1.3, n, seed)
        T = np.array([[float(v) for v in hessian(C, S, *q)] for q in p]).T
        path = os.path.join(outdir, f"hess_L{L}_s{seed}.bin")
        with open(path, "wb") as f:
            np.array([L, p.shape[0]], dtype="<i8").tofile(f)
            np.array([A], dtype="<f8").tofile(f)
            coef_vector(C, S).astype("<f8").tofile(f)
            for k in range(3):
                p[:, k].astype("<f8").tofile(f)
            T.astype("<f8").tofile(f)
        print(f"{path}: L = {L}, {p.shape[0]} points, max |T| = {np.abs(T).max():.3e}")


if __name__ == "__main__":
    main()
