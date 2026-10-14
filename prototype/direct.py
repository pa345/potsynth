"""Direct point-by-point synthesis of an internal field model (baseline).
Optimised: numba-compiled, multi-threaded, O(L^2) work per point with
recursions for P_n^m, dP_n^m/dtheta, P_n^m/sin(theta), cos/sin(m phi), (a/r)^n."""
import numpy as np
import numba as nb


@nb.njit(parallel=True, fastmath=False, cache=True)
def direct_field(g, h, a, r, theta, phi):
    L = g.shape[0] - 1
    M = r.shape[0]
    Br = np.empty(M); Bt = np.empty(M); Bp = np.empty(M)
    # recursion constants
    A = np.zeros((L + 1, L + 1)); Bc = np.zeros((L + 1, L + 1))
    for m in range(L + 1):
        for n in range(m + 1, L + 1):
            A[n, m] = (2 * n - 1) / np.sqrt(n * n - m * m)
            if n >= m + 2:
                Bc[n, m] = np.sqrt(((n - 1) ** 2 - m * m) / (n * n - m * m))
    F = np.zeros(L + 1)
    for m in range(2, L + 1):
        F[m] = np.sqrt((2 * m - 1) / (2 * m))
    nchunk = 256
    nblk = (M + nchunk - 1) // nchunk
    for blk in nb.prange(nblk):
        P = np.empty(L + 1); dP = np.empty(L + 1); Q = np.empty(L + 1)
        un = np.empty(L + 3)
        for i in range(blk * nchunk, min(M, (blk + 1) * nchunk)):
            x = np.cos(theta[i]); s = np.sin(theta[i])
            u = a / r[i]
            un[0] = 1.0
            for k in range(1, L + 3):
                un[k] = un[k - 1] * u
            c1 = np.cos(phi[i]); s1 = np.sin(phi[i])
            cm = 1.0; sm = 0.0
            Pmm = 1.0; dPmm = 0.0; Qmm = 0.0
            br = 0.0; bt = 0.0; bp = 0.0
            for m in range(L + 1):
                # sectoral seeds
                if m == 1:
                    Pmm, dPmm, Qmm = s, x, 1.0
                elif m >= 2:
                    f = F[m]
                    dPmm = f * (s * dPmm + x * Pmm)
                    Pmm = f * s * Pmm
                    Qmm = f * s * Qmm
                if m >= 1:
                    cm, sm = cm * c1 - sm * s1, sm * c1 + cm * s1
                P[m] = Pmm; dP[m] = dPmm; Q[m] = Qmm
                for n in range(m + 1, L + 1):
                    aa = A[n, m]; bb = Bc[n, m]
                    if n >= m + 2:
                        P[n] = aa * x * P[n - 1] - bb * P[n - 2]
                        dP[n] = aa * (x * dP[n - 1] - s * P[n - 1]) - bb * dP[n - 2]
                        Q[n] = aa * x * Q[n - 1] - bb * Q[n - 2]
                    else:
                        P[n] = aa * x * P[n - 1]
                        dP[n] = aa * (x * dP[n - 1] - s * P[n - 1])
                        Q[n] = aa * x * Q[n - 1]
                for n in range(max(m, 1), L + 1):
                    gc = g[n, m] * cm + h[n, m] * sm
                    w = un[n + 2]
                    br += (n + 1) * w * gc * P[n]
                    bt -= w * gc * dP[n]
                    if m > 0:
                        bp += m * w * (g[n, m] * sm - h[n, m] * cm) * Q[n]
            Br[i] = br; Bt[i] = bt; Bp[i] = bp
    return Br, Bt, Bp
