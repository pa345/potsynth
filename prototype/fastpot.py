"""
fastpot -- fast evaluation of internal potential-field models (IGRF / CHAOS type)
at N scattered points (r_i, theta_i, phi_i).

Model (Schmidt semi-normalised, geocentric spherical coordinates):

    V(r,th,ph) = a * sum_{n=1}^{L} sum_{m=0}^{n} (a/r)^{n+1}
                     [g_n^m cos(m ph) + h_n^m sin(m ph)] P_n^m(cos th)

    B = -grad V   ->   (B_r, B_theta, B_phi)

Key observation (the basis of the method)
-----------------------------------------
With u = a/r every field component is

    B_c(r,th,ph) = sum_{n,m} u^{n+2} * w_c(n,m) * Theta_c^{nm}(th) * e^{i m ph}   (real part)

  * u^{n+2} is a POLYNOMIAL of degree <= L+2 in u. Map the radial range of
    the data, u in [u0,u1], to u = uc + ud*cos(psi): the radial factor is then
    a trigonometric polynomial of degree <= L+2 in psi (exact Chebyshev series).
  * P_n^m(cos th), dP_n^m/dth and P_n^m(cos th)/sin th (m>=1) are trigonometric
    polynomials of degree <= n in th (on the doubled sphere th in [0,2pi)).
  * e^{i m ph} is already Fourier.

Hence every field component is EXACTLY a band-limited 3-D Fourier series

    B_c = Re sum_{|k|<=K} sum_{|p|<=L} sum_{m=0}^{L} F_c[k,p,m] e^{i(k psi + p th + m ph)}

and evaluating it at M scattered points (psi_i, th_i, ph_i) is a single
type-2 3-D non-uniform FFT (NUFFT).  Cost:

    precompute  O(K L^3)        (dense contraction, once per model + radial range)
    evaluate    O(K L^2 log L + M * w^3)   with w ~ log10(1/eps)+1 the spreading width

versus O(M L^2) for direct point-by-point synthesis.  The only approximation is
the NUFFT tolerance eps (and, optionally, truncating the Chebyshev/psi degree
K below L+2 when the radial range is thin -- controlled by the same eps).
"""
import numpy as np
import finufft

# ----------------------------------------------------------------------------
# coefficient handling
# ----------------------------------------------------------------------------

def gh_to_complex(g, h):
    """g,h: arrays (L+1, L+1) indexed [n,m]. Returns C[n,m] = g - i h."""
    return np.asarray(g, float) - 1j * np.asarray(h, float)


def vector_to_gh(gh, L):
    """Standard IGRF/CHAOS coefficient vector ordering
    g10 g11 h11 g20 g21 h21 g22 h22 ... -> (g, h) arrays (L+1, L+1)."""
    g = np.zeros((L + 1, L + 1)); h = np.zeros((L + 1, L + 1))
    i = 0
    for n in range(1, L + 1):
        g[n, 0] = gh[i]; i += 1
        for m in range(1, n + 1):
            g[n, m] = gh[i]; h[n, m] = gh[i + 1]; i += 2
    return g, h


# ----------------------------------------------------------------------------
# Schmidt semi-normalised Legendre functions, pole-safe recursions.
# Returns P, dP/dtheta, and Q = P/sin(theta) (for m>=1), for one m at a time.
# Works for signed sin(theta) (needed on the doubled sphere theta in [0,2pi)).
# ----------------------------------------------------------------------------

def _sectoral_seeds(L, x, s):
    """Seeds P_m^m, dP_m^m, Q_m^m for m=0..L on vector of points."""
    npt = x.shape[0]
    Pmm = np.zeros((L + 1, npt)); dPmm = np.zeros((L + 1, npt)); Qmm = np.zeros((L + 1, npt))
    Pmm[0] = 1.0
    if L >= 1:
        Pmm[1] = s; dPmm[1] = x; Qmm[1] = 1.0
    for m in range(2, L + 1):
        f = np.sqrt((2 * m - 1) / (2 * m))
        Pmm[m] = f * s * Pmm[m - 1]
        dPmm[m] = f * (s * dPmm[m - 1] + x * Pmm[m - 1])
        Qmm[m] = f * s * Qmm[m - 1]
    return Pmm, dPmm, Qmm


def _column(m, L, x, s, Pmm, dPmm, Qmm):
    """P_n^m, dP_n^m, Q_n^m for n=m..L (rows), given seeds."""
    nn = L - m + 1
    npt = x.shape[0]
    P = np.zeros((nn, npt)); dP = np.zeros((nn, npt)); Q = np.zeros((nn, npt))
    P[0] = Pmm[m]; dP[0] = dPmm[m]; Q[0] = Qmm[m]
    for j in range(1, nn):
        n = m + j
        a = (2 * n - 1) / np.sqrt(n * n - m * m)
        b = np.sqrt(((n - 1) ** 2 - m * m) / (n * n - m * m)) if j >= 2 else 0.0
        Pm2 = P[j - 2] if j >= 2 else 0.0
        dPm2 = dP[j - 2] if j >= 2 else 0.0
        Qm2 = Q[j - 2] if j >= 2 else 0.0
        P[j] = a * x * P[j - 1] - b * Pm2
        dP[j] = a * (x * dP[j - 1] - s * P[j - 1]) - b * dPm2
        Q[j] = a * x * Q[j - 1] - b * Qm2
    return P, dP, Q


# ----------------------------------------------------------------------------
# radial (Chebyshev in u = a/r) part
# ----------------------------------------------------------------------------

def _cheb_coeffs_powers(u0, u1, pmax, K):
    """Chebyshev coefficients a[k, q] of u^q, q=0..pmax, on [u0,u1], via
    interpolation at K+1 Chebyshev-Lobatto nodes (exact when K >= pmax)."""
    j = np.arange(K + 1)
    uc, ud = 0.5 * (u1 + u0), 0.5 * (u1 - u0)
    unodes = uc + ud * np.cos(np.pi * j / K)
    vals = unodes[:, None] ** np.arange(pmax + 1)[None, :]          # (K+1, pmax+1)
    C = np.cos(np.pi * np.outer(np.arange(K + 1), j) / K)            # (k, j)
    wts = np.ones(K + 1); wts[0] = wts[-1] = 0.5
    a = (2.0 / K) * C @ (wts[:, None] * vals)
    a[0] *= 0.5; a[K] *= 0.5
    return a


def choose_K(u0, u1, L, Cnm, eps):
    """Smallest psi-degree K such that truncating the exact Chebyshev series of
    the radial factors costs less than ~eps relative to the field scale."""
    Kmax = L + 2
    if u1 - u0 < 1e-15:
        return 1
    a = np.abs(_cheb_coeffs_powers(u0, u1, L + 2, Kmax))            # exact
    # coefficients below the double-precision roundoff floor of each column
    # (relative to max_u u^q = u1^q) are noise: never chase them by raising K
    q = np.arange(L + 3)
    a[a < 50 * np.finfo(float).eps * u1 ** q[None, :]] = 0.0
    n = np.arange(L + 1)
    w = (n + 1) * np.sqrt(np.sum(np.abs(Cnm) ** 2, axis=1))           # per-degree weight
    scale = np.sum(w * u1 ** (n + 2))
    tails = np.cumsum(a[::-1], axis=0)[::-1]                          # tails[k] = sum_{k'>=k}
    for K in range(2, Kmax + 1):
        err = 2 * np.sum(w * tails[min(K + 1, Kmax)][n + 2]) if K < Kmax else 0.0
        if err <= eps * scale:
            return K
    return Kmax


# ----------------------------------------------------------------------------
# the transform
# ----------------------------------------------------------------------------

def _contract(C, R, L, a, potential):
    """F[c, k, p, m] = sum_n R[k, n+2] * s_c(n,m) * gamma_n^m * FourierTheta_c^{nm}[p]
    for an arbitrary radial basis matrix R (rows k, columns = power q)."""
    nrad = R.shape[0]
    if True:  # (body indented for historical reasons)
        # theta: doubled-sphere grid, shifted off the poles
        Nth = 2 * L + 2
        th = (np.arange(Nth) + 0.5) * 2 * np.pi / Nth
        x, s = np.cos(th), np.sin(th)
        Pmm, dPmm, Qmm = _sectoral_seeds(L, x, s)
        p = np.arange(-L, L + 1)
        # Fourier modes -L..L from samples on the shifted grid th_j = (j+1/2) h:
        #   c_p = 1/Nth sum_j f(th_j) e^{-i p th_j} = e^{-i p h/2} FFT(f)[p mod Nth] / Nth
        # (exact: the functions are trig polynomials of degree <= L < Nth/2)
        shift = np.exp(-1j * p * np.pi / Nth) / Nth
        pidx = np.mod(p, Nth)

        def E(f):
            return np.fft.fft(f, axis=1)[:, pidx] * shift

        ncomp = 4 if potential else 3
        F = np.zeros((ncomp, nrad, 2 * L + 1, L + 1), complex)
        n_all = np.arange(L + 1)
        for m in range(L + 1):
            P, dP, Q = _column(m, L, x, s, Pmm, dPmm, Qmm)
            n = n_all[m:]
            n = n[n >= 1]
            if n.size == 0:
                continue
            off = n[0] - m
            Ph, dPh, Qh = (E(P[off:]), E(dP[off:]), E(Q[off:]))           # (nn, 2L+1)
            c = C[n, m]
            Rn = R[:, n + 2]                                              # (2K+1, nn)
            F[0, :, :, m] = Rn @ (((n + 1) * c)[:, None] * Ph)            # B_r
            F[1, :, :, m] = Rn @ ((-c)[:, None] * dPh)                    # B_theta
            if m > 0:
                F[2, :, :, m] = Rn @ ((-1j * m * c)[:, None] * Qh)        # B_phi
            if potential:
                F[3, :, :, m] = a * (R[:, n + 1] @ (c[:, None] * Ph))
        return F


class FastInternalField:
    """
    Precompute the 3-D Fourier representation of an internal field model valid
    for r in [rmin, rmax]; then evaluate at arbitrary points with one NUFFT.

    Parameters
    ----------
    g, h   : (L+1, L+1) Schmidt coefficients [n, m] (nT)
    a      : reference radius (km), 6371.2 for IGRF/CHAOS
    rmin, rmax : radial range of the evaluation points (km)
    eps    : requested relative accuracy (NUFFT tolerance and radial truncation)
    K      : psi (radial) degree; default chosen automatically from eps.
             K = L+2 makes the radial representation exact.
    nthreads : threads for finufft
    precision : 'double' or 'single' (single is ~1.3-2x faster, fine for eps >= 1e-6)
    upsampfac : NUFFT oversampling sigma. Default 2.0: evaluation cost is
             dominated by interpolation (~w^3 per point), and sigma=2 gives the
             narrowest kernel (w ~ log10(1/eps)+1). finufft's automatic choice
             (upsampfac=None) may pick 1.25 for large grids, which is up to 2x slower here.

    Time-dependent models: pass g, h with a leading axis J (J coefficient sets,
    e.g. [main field, secular variation] for IGRF, or the local polynomial /
    active B-spline coefficients of CHAOS in a time window) and call with
    weights W of shape (M, J): B_i = sum_j W[i,j] B_j(x_i).  All J*3
    transforms share one set of nonuniform points (one setpts).
    """

    def __init__(self, g, h, a=6371.2, rmin=None, rmax=None, eps=1e-10, K=None,
                 nthreads=None, upsampfac=2.0, potential=False, precision="double"):
        C = gh_to_complex(g, h)
        if C.ndim == 2:
            C = C[None]
        self.J = C.shape[0]
        L = C.shape[1] - 1
        self.L, self.a, self.eps = L, a, eps
        self.u0, self.u1 = a / rmax, a / rmin
        self.uc, self.ud = 0.5 * (self.u0 + self.u1), 0.5 * (self.u1 - self.u0)
        if K is None:
            K = max(choose_K(self.u0, self.u1, L, c, eps) for c in C)
        self.K = K
        self.potential = potential
        self.F = np.concatenate([self._build_coeffs(c) for c in C], axis=0)
        self.cdtype = np.complex64 if precision == "single" else np.complex128
        self.rdtype = np.float32 if precision == "single" else np.float64
        if precision == "single":
            eps = max(eps, 2e-7)
        opts = dict(isign=1, eps=eps, nthreads=nthreads or 0, dtype=self.cdtype)
        if upsampfac:
            opts["upsampfac"] = upsampfac
        ncomp = self.F.shape[0]
        self.plan = finufft.Plan(2, self.F.shape[1:], n_trans=ncomp, **opts)
        self._m0 = (self.L + 1) // 2
        self._Fs = np.ascontiguousarray(self.F, dtype=self.cdtype)

    # -- precomputation -------------------------------------------------------
    def _build_coeffs(self, C):
        L, K = self.L, self.K
        # radial: Fourier modes (k=-K..K) of u^{q}(psi), q = n+2 (field) [n+1 for V]
        cheb = _cheb_coeffs_powers(self.u0, self.u1, L + 2, K)       # (K+1, L+3)
        R = np.zeros((2 * K + 1, L + 3))
        R[K] = cheb[0]
        R[K + 1:] = 0.5 * cheb[1:]
        R[:K][::-1] = 0.5 * cheb[1:]
        return _contract(C, R, L, self.a, self.potential)


    # -- evaluation -----------------------------------------------------------
    def __call__(self, r, theta, phi, weights=None):
        """r (km), theta colatitude (rad), phi longitude (rad).
        Returns (Br, Btheta, Bphi[, V]) in nT (geocentric spherical components;
        B_north = -Btheta, B_east = Bphi, B_down = -Br).
        With J>1 coefficient sets: weights (M, J) combine them per point;
        weights=None returns an array (J, ncomp, M)."""
        u = self.a / np.asarray(r, float)
        t = (u - self.uc) / self.ud if self.ud > 0 else np.zeros_like(u)
        if np.any(np.abs(t) > 1 + 1e-12):
            raise ValueError("points outside the radial range of the transform")
        psi = np.arccos(np.clip(t, -1.0, 1.0))
        th = np.asarray(theta, float)
        ph = np.mod(np.asarray(phi, float) + np.pi, 2 * np.pi) - np.pi   # NUFFT wants [-pi,pi)
        rd = self.rdtype
        self.plan.setpts(psi.astype(rd), th.astype(rd), ph.astype(rd))
        out = self.plan.execute(self._Fs)
        # phi modes are m = 0..L, stored as finufft modes m - m0 -> undo the shift
        out = (out * np.exp(1j * self._m0 * ph)[None, :]).real
        nc = out.shape[0] // self.J
        out = out.reshape(self.J, nc, -1)
        if self.J == 1 and weights is None:
            return tuple(out[0])
        if weights is None:
            return out
        return tuple(np.einsum("jcm,mj->cm", out, np.asarray(weights)))


# ----------------------------------------------------------------------------
# 2-D variant: radial direction summed explicitly (low-rank Chebyshev split)
# ----------------------------------------------------------------------------

class FastInternalField2D(FastInternalField):
    """Same model, but the radial factor is NOT part of the NUFFT (low-rank split):

        B_c(r,th,ph) = Re sum_{i=1}^{rho} b_i(u) G_{c,i}(th,ph),     u = a/r

    G_{c,i} are 2-D Fourier series on the doubled sphere -> 3*rho batched
    type-2 2-D NUFFTs at the same (th, ph), then a per-point sum over i.

    radial="svd" (default): b_i are the optimal (weighted-SVD) radial basis
        functions of {u^{n+2} w_n}, n=1..L, on [u0,u1]; rho = numerical rank at eps.
        Each b_i is stored as a Chebyshev series in t=(u-uc)/ud, so evaluating all
        b_i(u_j) costs O((K+1) rho) per point.
    radial="cheb": b_k = T_k(t), rho = K+1.
    The potential is not supported in this variant."""

    def __init__(self, g, h, a=6371.2, rmin=None, rmax=None, eps=1e-10, K=None,
                 nthreads=None, upsampfac=2.0, precision="double", radial="svd",
                 mem_budget=6e8, svd_tol=0.3):
        C = gh_to_complex(g, h)
        if C.ndim == 3:
            if C.shape[0] != 1:
                raise NotImplementedError("2-D variant: single coefficient set only")
            C = C[0]
        L = C.shape[0] - 1
        self.L, self.a, self.eps, self.J, self.potential = L, a, eps, 1, False
        self.u0, self.u1 = a / rmax, a / rmin
        self.uc, self.ud = 0.5 * (self.u0 + self.u1), 0.5 * (self.u1 - self.u0)
        self.K = K if K is not None else choose_K(self.u0, self.u1, L, C, eps)
        K = self.K
        cheb = _cheb_coeffs_powers(self.u0, self.u1, L + 2, K)          # (K+1, L+3): u^q = sum_k cheb[k,q] T_k
        if radial == "cheb":
            R = cheb                                                     # basis T_k, coefficient matrix
            self.basis = np.eye(K + 1)                                   # b_i = sum_k basis[i,k] T_k
        elif radial == "svd":
            n = np.arange(1, L + 1)
            wn = (n + 1) * np.sqrt(np.sum(np.abs(C[1:]) ** 2, axis=1))
            nz = wn > 0
            # values of the weighted radial factors at K+1 Chebyshev-Lobatto nodes
            j = np.arange(K + 1)
            tn = np.cos(np.pi * j / K) if K > 0 else np.zeros(1)
            un = self.uc + self.ud * tn
            A = (un[:, None] ** (n[None, :] + 2)) * wn[None, :]           # (K+1, L)
            # Clenshaw-Curtis-like row weights so the SVD approximates the L2(t) norm
            cw = np.ones(K + 1); cw[0] = cw[-1] = 0.5
            sw = np.sqrt(cw)
            U, S, Wt = np.linalg.svd(sw[:, None] * A, full_matrices=False)
            rho = max(1, int(np.sum(S > svd_tol * eps * S[0])))
            U, S, Wt = U[:, :rho] / sw[:, None], S[:rho], Wt[:rho]
            # b_i(t) = U[:, i] interpolated -> Chebyshev coefficients via DCT-I
            Cm = np.cos(np.pi * np.outer(j, j) / K)
            coef = (2.0 / K) * Cm @ (cw[:, None] * U)
            coef[0] *= 0.5; coef[K] *= 0.5
            self.basis = coef.T                                          # (rho, K+1)
            R = np.zeros((rho, L + 3))
            R[:, n[nz] + 2] = (S[:, None] * Wt[:, nz]) / wn[nz][None, :]
        else:
            raise ValueError(radial)
        G = _contract(C, R, L, a, False)                                 # (3, rho, 2L+1, L+1)
        self.rho = G.shape[1]
        self.nc = G.shape[0]
        self.cdtype = np.complex64 if precision == "single" else np.complex128
        self.rdtype = np.float32 if precision == "single" else np.float64
        self._G = np.ascontiguousarray(G.reshape(self.nc * self.rho, 2 * L + 1, L + 1),
                                       dtype=self.cdtype)
        opts = dict(isign=1, eps=max(eps, 2e-7) if precision == "single" else eps,
                    nthreads=nthreads or 0, dtype=self.cdtype)
        if upsampfac:
            opts["upsampfac"] = upsampfac
        self.plan = finufft.Plan(2, self._G.shape[1:], n_trans=self._G.shape[0], **opts)
        self._m0 = (L + 1) // 2
        # chunk so the (3*rho, chunk) complex output stays within mem_budget bytes
        self.chunk = max(10_000, int(mem_budget / (16 * self._G.shape[0])))

    def __call__(self, r, theta, phi):
        u = self.a / np.asarray(r, float)
        t = (u - self.uc) / self.ud if self.ud > 0 else np.zeros_like(u)
        if np.any(np.abs(t) > 1 + 1e-12):
            raise ValueError("points outside the radial range of the transform")
        th = np.asarray(theta, float)
        ph = np.mod(np.asarray(phi, float) + np.pi, 2 * np.pi) - np.pi
        M, K1 = t.shape[0], self.K + 1
        res = np.empty((self.nc, M))
        for s0 in range(0, M, self.chunk):
            sl = slice(s0, min(M, s0 + self.chunk)); n = sl.stop - sl.start
            tt = t[sl]
            T = np.empty((K1, n)); T[0] = 1.0
            if K1 > 1:
                T[1] = tt
            for k in range(2, K1):
                T[k] = 2 * tt * T[k - 1] - T[k - 2]
            b = self.basis @ T                                            # (rho, n) radial basis values
            self.plan.setpts(th[sl].astype(self.rdtype), ph[sl].astype(self.rdtype))
            out = self.plan.execute(self._G).reshape(self.nc, self.rho, n)
            acc = np.einsum("cim,im->cm", out, b)
            res[:, sl] = (acc * np.exp(1j * self._m0 * ph[sl])[None, :]).real
        return tuple(res)
