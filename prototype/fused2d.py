"""
Fused 2-D variant of the fast internal-field transform.

    B_c(r,th,ph) = Re sum_{i<rho} b_i(u) G_{c,i}(th,ph)          (low-rank radial split)

Differences from running FINUFFT on the 3*rho channels:
  * the oversampled fine grids (deconvolved + FFT'd) depend only on the model,
    so they are computed ONCE at precompute and cached;
  * per call there is no FFT at all: points are bin-sorted, then one fused
    interpolation pass evaluates the w x w exponential-of-semicircle kernel once
    per point and accumulates all 3*rho channels (channel-contiguous layout),
    followed by the rho-term radial sum.
Per-point cost ~ 3*rho*w^2 multiply-adds (vs 3*w^3 for the 3-D NUFFT).
"""
import numpy as np
import numba as nb
import scipy.fft as sfft
from fastpot import FastInternalField2D


def _es_params(eps):
    w = int(np.ceil(np.log10(1.0 / eps))) + 1
    w = max(2, min(16, w))
    return w, 2.30 * w


def _es_ft(k, w, beta, h):
    """psi_hat(k) = int phi((x)/alpha) e^{-ikx} dx, alpha = w h / 2 (Gauss-Legendre)."""
    z, wq = np.polynomial.legendre.leggauss(4 * w + 40)
    alpha = 0.5 * w * h
    phi = np.exp(beta * (np.sqrt(1 - z * z) - 1))
    return alpha * (np.cos(np.outer(k, alpha * z)) @ (wq * phi))


@nb.njit(cache=True)
def _bin_sort(i1, i2, nb1, nb2, s1, s2):
    M = i1.shape[0]
    nbins = nb1 * nb2
    counts = np.zeros(nbins + 1, np.int64)
    key = np.empty(M, np.int64)
    for j in range(M):
        k = (i1[j] // s1) * nb2 + (i2[j] // s2)
        key[j] = k
        counts[k + 1] += 1
    for k in range(nbins):
        counts[k + 1] += counts[k]
    order = np.empty(M, np.int64)
    for j in range(M):
        k = key[j]
        order[counts[k]] = j
        counts[k] += 1
    return order


@nb.njit(parallel=True, fastmath=True, cache=True)
def _fused_interp(grid, x1, x2, l1s, l2s, bvals, order, w, beta, h1, h2, nc, rho):
    """grid: (nf1, nf2, 2*nc*rho) float64 view of complex channels (re,im interleaved).
    Returns complex-valued channel sums contracted with the radial basis: (nc, M)."""
    nf1, nf2, nch2 = grid.shape
    M = x1.shape[0]
    out_re = np.empty((nc, M)); out_im = np.empty((nc, M))
    nblk = (M + 511) // 512
    for blk in nb.prange(nblk):
        k1 = np.empty(w); k2 = np.empty(w)
        acc = np.empty(nch2); row = np.empty(nch2)
        for jj in range(blk * 512, min(M, (blk + 1) * 512)):
            i = order[jj]
            l1 = l1s[i]; l2 = l2s[i]
            for a in range(w):
                z = (x1[i] - (l1 + a) * h1) / (0.5 * w * h1)
                k1[a] = np.exp(beta * (np.sqrt(max(0.0, 1 - z * z)) - 1)) if abs(z) < 1 else 0.0
                z = (x2[i] - (l2 + a) * h2) / (0.5 * w * h2)
                k2[a] = np.exp(beta * (np.sqrt(max(0.0, 1 - z * z)) - 1)) if abs(z) < 1 else 0.0
            for q in range(nch2):
                acc[q] = 0.0
            for a in range(w):
                g1 = (l1 + a) % nf1
                for q in range(nch2):
                    row[q] = 0.0
                for b in range(w):
                    g2 = (l2 + b) % nf2
                    wt = k2[b]
                    for q in range(nch2):
                        row[q] += wt * grid[g1, g2, q]
                wa = k1[a]
                for q in range(nch2):
                    acc[q] += wa * row[q]
            for c in range(nc):
                sr = 0.0; si = 0.0
                for r in range(rho):
                    bv = bvals[r, i]
                    sr += bv * acc[2 * (c * rho + r)]
                    si += bv * acc[2 * (c * rho + r) + 1]
                out_re[c, i] = sr; out_im[c, i] = si
    return out_re, out_im


class FusedField2D:
    """Fused, cached-grid 2-D transform. Same arguments as FastInternalField2D."""

    def __init__(self, g, h, a=6371.2, rmin=None, rmax=None, eps=1e-10, radial="svd",
                 svd_tol=0.3, nthreads=None, w=None, beta_per_w=None, eps_radial=None, sigma=2.0):
        base = FastInternalField2D(g, h, a=a, rmin=rmin, rmax=rmax,
                                   eps=eps if eps_radial is None else eps_radial,
                                   radial=radial, svd_tol=svd_tol)
        self.base = base
        L = base.L
        self.a, self.uc, self.ud, self.K, self.rho = a, base.uc, base.ud, base.K, base.rho
        self.basis, self.m0, self.nc = base.basis, base._m0, base.nc
        self.w, self.beta = _es_params(eps)
        if w is not None:
            self.w = int(w)
        if beta_per_w is None:
            beta_per_w = np.pi * (1 - 1 / (2 * sigma)) * 0.98    # ES shape parameter for oversampling sigma
        self.beta = beta_per_w * self.w
        G = base._G                                              # (nc*rho, 2L+1, L+1), modes p=-L..L, j=m-m0
        n1, n2 = G.shape[1], G.shape[2]
        nf1 = sfft.next_fast_len(max(int(np.ceil(sigma * n1)), 2 * self.w))
        nf2 = sfft.next_fast_len(max(int(np.ceil(sigma * n2)), 2 * self.w))
        nf1 += nf1 % 2; nf2 += nf2 % 2
        self.h1, self.h2 = 2 * np.pi / nf1, 2 * np.pi / nf2
        p = np.arange(-L, L + 1)
        j = np.arange(L + 1) - self.m0
        c1 = self.h1 / _es_ft(p, self.w, self.beta, self.h1)
        c2 = self.h2 / _es_ft(j, self.w, self.beta, self.h2)
        pad = np.zeros((G.shape[0], nf1, nf2), complex)
        pad[:, np.mod(p, nf1)[:, None], np.mod(j, nf2)[None, :]] = G * c1[None, :, None] * c2[None, None, :]
        fine = sfft.ifft2(pad, axes=(1, 2), workers=-1) * (nf1 * nf2)   # b_l = sum_k f'_k e^{i k x_l}
        self.grid = np.ascontiguousarray(np.moveaxis(fine, 0, -1)).view(np.float64)  # (nf1, nf2, 2*nch)
        self.nf1, self.nf2 = nf1, nf2

    def __call__(self, r, theta, phi):
        u = self.a / np.asarray(r, float)
        t = (u - self.uc) / self.ud if self.ud > 0 else np.zeros_like(u)
        if np.any(np.abs(t) > 1 + 1e-12):
            raise ValueError("points outside the radial range of the transform")
        th = np.asarray(theta, float)
        ph = np.mod(np.asarray(phi, float) + np.pi, 2 * np.pi) - np.pi
        K1 = self.K + 1
        T = np.empty((K1, t.shape[0])); T[0] = 1.0
        if K1 > 1:
            T[1] = t
        for k in range(2, K1):
            T[k] = 2 * t * T[k - 1] - T[k - 2]
        bvals = self.basis @ T
        w = self.w
        l1 = np.ceil(th / self.h1 - 0.5 * w).astype(np.int64)
        l2 = np.ceil(ph / self.h2 - 0.5 * w).astype(np.int64)
        order = _bin_sort(np.mod(l1, self.nf1), np.mod(l2, self.nf2),
                          (self.nf1 + 15) // 16, (self.nf2 + 15) // 16, 16, 16)
        ore, oim = _fused_interp(self.grid, th, ph, l1, l2, bvals, order, w, self.beta,
                                 self.h1, self.h2, self.nc, self.rho)
        c, s = np.cos(self.m0 * ph), np.sin(self.m0 * ph)
        return tuple(ore * c[None, :] - oim * s[None, :])
