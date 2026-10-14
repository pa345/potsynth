"""
fastfield -- fast evaluation of internal potential-field models (IGRF / CHAOS type)
at many scattered points, using the fused 2-D transform with real-valued,
half-sphere grids.

Model (Schmidt semi-normalised Gauss coefficients g_n^m, h_n^m in nT, reference radius a):

    V(r,th,ph) = a sum_{n=1}^{L} sum_{m=0}^{n} (a/r)^{n+1} [g cos(m ph) + h sin(m ph)] P_n^m(cos th)
    B = -grad V  ->  (B_r, B_theta, B_phi)

Method (see the accompanying PDF for a step-by-step derivation):

  1. Radial separation: with u = a/r, every field component is
         B_c = sum_{i<rho} b_i(u) G_{c,i}(th, ph)
     where b_i are rho optimal radial functions (weighted SVD of u^{n+2}).
  2. Each G_{c,i} is a REAL 2-D trigonometric polynomial on the doubled sphere
     (th in [0,2pi), ph in [0,2pi)), of degree <= L in each variable.
  3. Its Fourier coefficients are divided by the Fourier transform of a short window
     ("pre-sharpening"), and transformed once by FFT to an oversampled real grid.
     Only the rows with th in [0, pi] (plus a margin of w/2 rows) are stored.
  4. Evaluation at a point: a w x w window sum over the stored grid, accumulated for
     all 3*rho channels at once, followed by the rho-term radial sum.

The only approximation is the finite window (error falls exponentially in w) and the
SVD truncation of the radial basis; both are controlled by eps. At eps ~ 1e-14 the
result agrees with the direct double-precision sum to its own rounding error.
"""
import numpy as np
import numba as nb
import scipy.fft as sfft

# ----------------------------------------------------------------------------
# Coefficients
# ----------------------------------------------------------------------------

def vector_to_gh(gh, L):
    """IGRF/CHAOS coefficient vector g10 g11 h11 g20 g21 h21 ... -> arrays g[n,m], h[n,m]."""
    g = np.zeros((L + 1, L + 1)); h = np.zeros((L + 1, L + 1))
    i = 0
    for n in range(1, L + 1):
        g[n, 0] = gh[i]; i += 1
        for m in range(1, n + 1):
            g[n, m] = gh[i]; h[n, m] = gh[i + 1]; i += 2
    return g, h


# ----------------------------------------------------------------------------
# Schmidt semi-normalised Legendre functions on the doubled sphere (pole-safe).
# For each order m we return P_n^m, dP_n^m/dtheta and Q_n^m = P_n^m / sin(theta),
# n = m..L, using recursions that never divide by sin(theta). sin(theta) may be
# negative (theta in (pi, 2pi) on the doubled sphere).
# ----------------------------------------------------------------------------

def _sectoral(L, x, s):
    P = np.zeros((L + 1, x.size)); dP = np.zeros_like(P); Q = np.zeros_like(P)
    P[0] = 1.0
    if L >= 1:
        P[1], dP[1], Q[1] = s, x, 1.0
    for m in range(2, L + 1):
        f = np.sqrt((2 * m - 1) / (2 * m))
        P[m] = f * s * P[m - 1]
        dP[m] = f * (s * dP[m - 1] + x * P[m - 1])
        Q[m] = f * s * Q[m - 1]
    return P, dP, Q


def _legendre_column(m, L, x, s, Pmm, dPmm, Qmm):
    nn = L - m + 1
    P = np.zeros((nn, x.size)); dP = np.zeros_like(P); Q = np.zeros_like(P)
    P[0], dP[0], Q[0] = Pmm[m], dPmm[m], Qmm[m]
    for j in range(1, nn):
        n = m + j
        A = (2 * n - 1) / np.sqrt(n * n - m * m)
        if j >= 2:
            B = np.sqrt(((n - 1) ** 2 - m * m) / (n * n - m * m))
            P[j] = A * x * P[j - 1] - B * P[j - 2]
            dP[j] = A * (x * dP[j - 1] - s * P[j - 1]) - B * dP[j - 2]
            Q[j] = A * x * Q[j - 1] - B * Q[j - 2]
        else:
            P[j] = A * x * P[j - 1]
            dP[j] = A * (x * dP[j - 1] - s * P[j - 1])
            Q[j] = A * x * Q[j - 1]
    return P, dP, Q


# ----------------------------------------------------------------------------
# Step 1: radial basis
# ----------------------------------------------------------------------------

def _cheb_coeffs(f_nodes):
    """Chebyshev coefficients from values at the K+1 Lobatto nodes t_j = cos(pi j / K)."""
    K = f_nodes.shape[0] - 1
    j = np.arange(K + 1)
    C = np.cos(np.pi * np.outer(j, j) / K)
    c = np.ones(K + 1); c[0] = c[-1] = 0.5
    a = (2.0 / K) * C @ (c[:, None] * f_nodes)
    a[0] *= 0.5; a[K] *= 0.5
    return a


def radial_basis(u0, u1, L, wn, tol):
    """Optimal radial basis for the functions w_n u^(n+2), n = 1..L, on [u0, u1].

    Returns (K, basis, beta):
      basis[i, k]  Chebyshev coefficients of b_i(t), t = (u - uc)/ud in [-1, 1]
      beta[i, n]   such that  u^(n+2) ~= sum_i beta[i, n] b_i(u)   (beta[:, 0] = 0)
    """
    uc, ud = 0.5 * (u1 + u0), 0.5 * (u1 - u0)
    n = np.arange(1, L + 1)
    # Chebyshev degree K: smallest K for which the weighted functions are resolved to ~tol
    for K in range(2, L + 3):
        t = np.cos(np.pi * np.arange(K + 1) / K)
        A = (uc + ud * t)[:, None] ** (n[None, :] + 2) * wn[None, :]
        a = _cheb_coeffs(A)
        tail = np.abs(a[-2:]).sum(axis=0).sum()
        if tail <= tol * np.abs(A).max(axis=0).sum() or K == L + 2:
            break
    t = np.cos(np.pi * np.arange(K + 1) / K)
    A = (uc + ud * t)[:, None] ** (n[None, :] + 2) * wn[None, :]
    cw = np.ones(K + 1); cw[0] = cw[-1] = 0.5               # Chebyshev-Lobatto weights
    U, S, Wt = np.linalg.svd(np.sqrt(cw)[:, None] * A, full_matrices=False)
    rho = max(1, int(np.sum(S > tol * S[0])))
    U = U[:, :rho] / np.sqrt(cw)[:, None]                    # b_i at the nodes
    basis = _cheb_coeffs(U).T                                # (rho, K+1)
    # drop trailing Chebyshev terms that are below the tolerance (rounding noise)
    big = np.nonzero(np.abs(basis).max(axis=0) > 0.1 * tol * np.abs(basis).max())[0]
    K = int(big[-1]) if big.size else 0
    basis = np.ascontiguousarray(basis[:, :K + 1])
    beta = np.zeros((rho, L + 1))
    nz = wn > 0
    beta[:, n[nz]] = (S[:rho, None] * Wt[:rho][:, nz]) / wn[nz][None, :]
    return K, basis, beta


# ----------------------------------------------------------------------------
# Step 3: window (exponential of semicircle) and its Fourier transform
# ----------------------------------------------------------------------------

def window_ft(k, w, beta, h):
    """phi_hat(k) = integral of phi((x)/alpha) e^{-ikx} dx,  alpha = w h / 2,
    phi(z) = exp(beta (sqrt(1 - z^2) - 1)) on |z| <= 1 (Gauss-Legendre quadrature)."""
    z, wq = np.polynomial.legendre.leggauss(4 * w + 60)
    alpha = 0.5 * w * h
    phi = np.exp(beta * (np.sqrt(1 - z * z) - 1))
    return alpha * (np.cos(np.outer(k, alpha * z)) @ (wq * phi))


# (numba backend) Kernel width needed for a target accuracy. Measured maximum relative error of the whole
# transform against a 19-digit (long double) reference, worst case over L = 13 and L = 185
# (fastfield_calibration.json). The error is essentially independent of L.
_ERR_SIGMA2 = {4: 7e-4, 5: 2e-5, 6: 4e-6, 7: 6e-7, 8: 3e-8, 9: 4e-9, 10: 6e-10,
               11: 3e-11, 12: 3e-12, 13: 5e-13, 14: 4e-14, 15: 2e-14, 16: 2e-14}
_ERR_SIGMA3 = {4: 1e-4, 5: 3e-5, 6: 7e-7, 7: 1e-7, 8: 8e-9, 9: 6e-10, 10: 4e-11,
               11: 3e-12, 12: 3e-13, 13: 1e-14, 14: 1e-14, 15: 1e-14, 16: 1e-14}


def choose_window(eps, sigma=None):
    """Smallest window width w whose measured error is <= eps (default sigma = 2)."""
    sigma = 2.0 if sigma is None else float(sigma)
    table = _ERR_SIGMA3 if sigma >= 3 else _ERR_SIGMA2
    for w in sorted(table):
        if table[w] <= eps:
            return w, sigma
    return max(table), sigma


# ----------------------------------------------------------------------------
# FINUFFT backend: the library's own window, sampled through the library itself
# ----------------------------------------------------------------------------

def finufft_window_width(eps, upsampfac=2.0):
    """Window width (grid points) that FINUFFT uses at tolerance eps: found by interpolating
    a unit spike with the library's interpolation-only mode and measuring the support."""
    import finufft
    N = 64
    plan = finufft.Plan(2, (N,), eps=eps, spreadinterponly=1, upsampfac=upsampfac, dtype="complex128")
    spike = np.zeros(N, complex); spike[N // 2] = 1.0
    s = np.linspace(-9.0, 9.0, 3601)
    plan.setpts(s * 2 * np.pi / N)
    support = s[plan.execute(spike).real != 0]
    return int(round(support.max() - support.min()))


def finufft_presharpen(eps, upsampfac, N, ks, npts=64):
    """Pre-sharpening factors 1/r_k for FINUFFT's window on a periodic grid of N points.
    The pure Fourier mode e^{ikx} is placed on the grid and interpolated by FINUFFT's
    interpolation-only mode; the result is r_k e^{ikx} (plus aliases of size ~eps), so
    dividing the Fourier coefficients by r_k makes the window sum reproduce them exactly."""
    import finufft
    ks = np.asarray(ks)
    xl = -np.pi + 2 * np.pi * np.arange(N) / N                      # FINUFFT grid positions
    modes = np.exp(1j * np.outer(ks, xl))                           # (nk, N)
    x = np.random.default_rng(12345).uniform(-np.pi, np.pi, npts)
    plan = finufft.Plan(2, (N,), n_trans=ks.size, eps=eps, spreadinterponly=1,
                        upsampfac=upsampfac, dtype="complex128")
    plan.setpts(x)
    out = plan.execute(modes)                                       # (nk, npts)
    r = (out * np.exp(-1j * np.outer(ks, x))).mean(axis=1).real
    return 1.0 / r


# ----------------------------------------------------------------------------
# Step 5: fused evaluation kernel
# ----------------------------------------------------------------------------

@nb.njit(cache=True)
def _bin_sort(l1, l2, nrow, ncol, s):
    """Counting sort of the points by (l1 // s, l2 // s) so neighbouring points touch
    neighbouring grid cells (better cache use)."""
    M = l1.shape[0]
    nb2 = ncol // s + 1
    nbins = (nrow // s + 1) * nb2
    counts = np.zeros(nbins + 1, np.int64)
    key = np.empty(M, np.int64)
    for j in range(M):
        k = (l1[j] // s) * nb2 + (l2[j] // s)
        key[j] = k
        counts[k + 1] += 1
    for k in range(nbins):
        counts[k + 1] += counts[k]
    order = np.empty(M, np.int64)
    for j in range(M):
        order[counts[key[j]]] = j
        counts[key[j]] += 1
    return order


@nb.njit(parallel=True, fastmath=True, cache=True)
def _evaluate(grid, row0, th, ph, t, basis, w, beta, h1, h2, nc, rho, order):
    """grid[row, col, c*rho + i]: stored real grid (rows = theta, cols = phi).
    For each point: window weights in theta and phi, w x w weighted sum of all
    3*rho channels, then the radial sum with b_i(t)."""
    nrow, ncol, nch = grid.shape
    K1 = basis.shape[1]
    M = th.shape[0]
    out = np.empty((nc, M))
    half = 0.5 * w
    nblk = (M + 255) // 256
    for blk in nb.prange(nblk):
        k1 = np.empty(w); k2 = np.empty(w)
        acc = np.empty(nch); row = np.empty(nch)
        T = np.empty(K1); b = np.empty(rho)
        for jj in range(blk * 256, min(M, (blk + 1) * 256)):
            i = order[jj]
            # window weights (exponential of semicircle)
            l1 = int(np.ceil(th[i] / h1 - half))
            l2 = int(np.ceil(ph[i] / h2 - half))
            for q in range(w):
                z = (th[i] - (l1 + q) * h1) / (half * h1)
                k1[q] = np.exp(beta * (np.sqrt(max(0.0, 1.0 - z * z)) - 1.0))
                z = (ph[i] - (l2 + q) * h2) / (half * h2)
                k2[q] = np.exp(beta * (np.sqrt(max(0.0, 1.0 - z * z)) - 1.0))
            # w x w window sum, all channels at once
            for q in range(nch):
                acc[q] = 0.0
            r0 = l1 - row0
            for a in range(w):
                for q in range(nch):
                    row[q] = 0.0
                for bb in range(w):
                    col = (l2 + bb) % ncol
                    wt = k2[bb]
                    for q in range(nch):
                        row[q] += wt * grid[r0 + a, col, q]
                wa = k1[a]
                for q in range(nch):
                    acc[q] += wa * row[q]
            # radial basis values b_i(t) from Chebyshev series
            tt = t[i]
            T[0] = 1.0
            if K1 > 1:
                T[1] = tt
            for k in range(2, K1):
                T[k] = 2.0 * tt * T[k - 1] - T[k - 2]
            for r in range(rho):
                s = 0.0
                for k in range(K1):
                    s += basis[r, k] * T[k]
                b[r] = s
            for c in range(nc):
                s = 0.0
                for r in range(rho):
                    s += b[r] * acc[c * rho + r]
                out[c, i] = s
    return out


# ----------------------------------------------------------------------------
# The transform
# ----------------------------------------------------------------------------

class FastField:
    """
    Fast evaluation of an internal field model at scattered points.

    Parameters
    ----------
    g, h : arrays (L+1, L+1), Schmidt semi-normalised Gauss coefficients [n, m] in nT
    rmin, rmax : radial range (km) that the evaluation points will lie in
    a : reference radius (km)
    eps : target relative accuracy (default 1e-10; ~1e-14 reaches the rounding
          error of the direct double-precision sum)
    sigma, w, beta : optional overrides of the oversampling factor, the window width
          and the window shape (normally derived from eps)
    rank_tol : optional override of the radial SVD tolerance (default 0.1*eps)
    backend : "numba" (default) -- our compiled fused loop, which computes the window weights once
          per point and sums all 3*rho channels in one pass; "finufft" -- the window sums are done
          by FINUFFT's interpolation-only mode with FINUFFT's own window (about 1.4x slower,
          slightly more accurate at eps < 1e-13; needs the finufft package)
    nthreads : threads for the FINUFFT backend (default: all)

    Call with (r [km], theta [rad, colatitude], phi [rad]) -> (B_r, B_theta, B_phi) in nT.
    North/East/Centre components: B_N = -B_theta, B_E = B_phi, B_C = -B_r.
    """

    def __init__(self, g, h, rmin, rmax, a=6371.2, eps=1e-10, sigma=None, w=None,
                 beta=None, rank_tol=None, backend="numba", nthreads=None, chunk=500_000):
        g = np.asarray(g, float); h = np.asarray(h, float)
        L = g.shape[0] - 1
        C = g - 1j * h                                       # gamma_n^m = g - i h
        self.L, self.a, self.eps, self.backend, self.chunk = L, a, eps, backend, chunk
        self.u0, self.u1 = a / rmax, a / rmin
        self.uc, self.ud = 0.5 * (self.u0 + self.u1), 0.5 * (self.u1 - self.u0)

        # ---- step 1: radial basis ------------------------------------------------
        n = np.arange(1, L + 1)
        wn = (n + 1) * np.sqrt(np.sum(np.abs(C[1:]) ** 2, axis=1))  # per-degree field amplitude
        tol = max(rank_tol if rank_tol is not None else 0.1 * eps, 1e-16)
        self.K, self.basis, beta_r = radial_basis(self.u0, self.u1, L, wn, tol)
        self.rho = rho = self.basis.shape[0]

        # ---- step 2: angular functions as 2-D Fourier series ----------------------
        # H[c, i, p, m]: complex coefficients, p = -L..L, m = 0..L, such that
        #   G_{c,i}(th, ph) = Re sum_{p,m} H[c,i,p,m] e^{i(p th + m ph)}
        Nth = 2 * L + 2
        thg = (np.arange(Nth) + 0.5) * 2 * np.pi / Nth          # doubled sphere, off the poles
        x, s = np.cos(thg), np.sin(thg)
        p = np.arange(-L, L + 1)
        shift = np.exp(-1j * p * np.pi / Nth) / Nth
        pidx = np.mod(p, Nth)
        fourier = lambda f: np.fft.fft(f, axis=1)[:, pidx] * shift   # exact for degree <= L
        Pmm, dPmm, Qmm = _sectoral(L, x, s)
        H = np.zeros((3, rho, 2 * L + 1, L + 1), complex)
        for m in range(L + 1):
            P, dP, Q = _legendre_column(m, L, x, s, Pmm, dPmm, Qmm)
            nn = np.arange(max(m, 1), L + 1)
            off = nn[0] - m if nn.size else 0
            if nn.size == 0:
                continue
            cnm = C[nn, m]
            Bm = beta_r[:, nn]                                     # (rho, nn)
            H[0, :, :, m] = Bm @ (((nn + 1) * cnm)[:, None] * fourier(P[off:]))   # B_r
            H[1, :, :, m] = Bm @ ((-cnm)[:, None] * fourier(dP[off:]))            # B_theta
            if m > 0:
                H[2, :, :, m] = Bm @ ((-1j * m * cnm)[:, None] * fourier(Q[off:]))  # B_phi

        # ---- step 3: pre-sharpen and transform to an oversampled real grid ---------
        sigma = 2.0 if sigma is None else float(sigma)
        nband = 2 * L + 1                                         # modes -L..L in each direction
        if backend == "finufft":
            self.w, self.sigma, self.beta = finufft_window_width(eps, sigma), sigma, None
            margin = 2                                            # safety rows for the library's rounding
        elif backend == "numba":
            if w is None:
                w = choose_window(eps, sigma)[0]
            self.w, self.sigma = int(w), sigma
            self.beta = beta if beta is not None else np.pi * (1 - 1 / (2 * sigma)) * self.w
            margin = 0
        else:
            raise ValueError("backend must be 'finufft' or 'numba'")
        nf1 = sfft.next_fast_len(max(int(np.ceil(sigma * nband)), 2 * self.w))
        nf2 = sfft.next_fast_len(max(int(np.ceil(sigma * nband)), 2 * self.w))
        nf1 += nf1 % 2; nf2 += nf2 % 2
        self.nf1, self.nf2 = nf1, nf2
        self.h1, self.h2 = 2 * np.pi / nf1, 2 * np.pi / nf2
        mm = np.arange(L + 1)
        if backend == "finufft":                                  # measured through the library
            c1 = finufft_presharpen(eps, sigma, nf1, p)
            c2 = finufft_presharpen(eps, sigma, nf2, mm)
        else:
            c1 = self.h1 / window_ft(p, self.w, self.beta, self.h1)
            c2 = self.h2 / window_ft(mm, self.w, self.beta, self.h2)

        # ---- step 4: keep only the rows needed for theta in [0, pi] ----------------
        half = 0.5 * self.w
        self.row0 = int(np.ceil(-half)) - margin
        rowmax = int(np.ceil(np.pi / self.h1 - half)) + self.w - 1 + margin
        rows = np.mod(np.arange(self.row0, rowmax + 1), nf1)
        self.nrows = rows.size
        nch = 3 * rho
        if backend == "finufft":
            # two real channels share one complex grid (the window weights are real)
            npair = (nch + 1) // 2
            self.grid = np.zeros((npair, rows.size, nf2), complex)
        else:
            self.grid = np.empty((rows.size, nf2, nch))
        for c in range(3):
            for i in range(rho):
                # real two-sided coefficients F[p, m] for m >= 0 (Hermitian in (p, m)):
                #   m > 0: F = H/2 ;  m = 0: F[p,0] = (H[p,0] + conj(H[-p,0]))/2
                F = 0.5 * H[c, i].copy()
                F[:, 0] = 0.5 * (H[c, i, :, 0] + np.conj(H[c, i, ::-1, 0]))
                F *= c1[:, None] * c2[None, :]
                spec = np.zeros((nf1, nf2 // 2 + 1), complex)
                spec[np.mod(p, nf1)[:, None], mm[None, :]] = F
                full = sfft.irfft2(spec, s=(nf1, nf2), workers=-1) * (nf1 * nf2)
                q = c * rho + i
                if backend == "finufft":
                    if q % 2 == 0:
                        self.grid[q // 2].real = full[rows]
                    else:
                        self.grid[q // 2].imag = full[rows]
                else:
                    self.grid[:, :, q] = full[rows]
        self.nbytes = self.grid.nbytes
        if backend == "finufft":
            import finufft
            self._plan = finufft.Plan(2, (rows.size, nf2), n_trans=self.grid.shape[0], eps=eps,
                                      spreadinterponly=1, upsampfac=sigma, dtype="complex128",
                                      nthreads=nthreads or 0)

    # ---- step 5: evaluation -------------------------------------------------------
    def __call__(self, r, theta, phi):
        r = np.asarray(r, float)
        th = np.clip(np.asarray(theta, float), 0.0, np.pi)
        ph = np.mod(np.asarray(phi, float), 2 * np.pi)
        t = (self.a / r - self.uc) / self.ud if self.ud > 0 else np.zeros_like(r)
        if np.any(np.abs(t) > 1 + 1e-12):
            raise ValueError("points outside the radial range [rmin, rmax] of the transform")
        t = np.clip(t, -1.0, 1.0)
        if self.backend == "numba":
            half = 0.5 * self.w
            l1 = (np.ceil(th / self.h1 - half).astype(np.int64) - self.row0)
            l2 = np.mod(np.ceil(ph / self.h2 - half).astype(np.int64), self.nf2)
            order = _bin_sort(l1, l2, self.grid.shape[0], self.nf2, 16)
            out = _evaluate(self.grid, self.row0, th, ph, t, self.basis, self.w, self.beta,
                            self.h1, self.h2, 3, self.rho, order)
            return out[0], out[1], out[2]
        # FINUFFT backend: library coordinates put grid index j at -pi + j * (2 pi / N)
        M = r.shape[0]
        out = np.empty((3, M))
        nch = 3 * self.rho
        for s0 in range(0, M, self.chunk):
            sl = slice(s0, min(M, s0 + self.chunk))
            x1 = -np.pi + (th[sl] / self.h1 - self.row0) * (2 * np.pi / self.nrows)
            x2 = ph[sl] - np.pi
            self._plan.setpts(x1, x2)
            v = self._plan.execute(self.grid)                         # (npair, n) complex
            vals = np.empty((2 * v.shape[0], v.shape[1]))
            vals[0::2] = v.real; vals[1::2] = v.imag
            vals = vals[:nch].reshape(3, self.rho, -1)
            tt = t[sl]
            T = np.empty((self.basis.shape[1], tt.size)); T[0] = 1.0
            if T.shape[0] > 1:
                T[1] = tt
            for k in range(2, T.shape[0]):
                T[k] = 2 * tt * T[k - 1] - T[k - 2]
            b = self.basis @ T                                         # (rho, n) radial values
            out[:, sl] = np.einsum("cin,in->cn", vals, b)
        return out[0], out[1], out[2]
