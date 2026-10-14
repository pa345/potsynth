"""
Magnetic front-end: B = -grad V for Schmidt semi-normalised Gauss coefficients.
"""
import ctypes

import numpy as np

from . import _lib
from ._lib import lib
from .core import _Evaluator, _params, _raise, coef_vector, ncoef


class MagModel(_Evaluator):
    """Internal magnetic field model (IGRF, CHAOS, lithospheric models...).

    Parameters
    ----------
    gh : Gauss coefficients in nT, either a 1-D vector in the layout of idx()
         (g at m >= 0, h at m < 0), or a tuple (g, h) of 2-D arrays g[n, m], h[n, m]
    lmax : maximum degree (default: from gh)
    lmin : minimum degree (default 1)
    mmax : maximum order (default lmax)
    a : reference radius in km (default 6371.2)
    rmin, rmax : radial range in km of the evaluation points (required by the fast
                 method; default: none, i.e. direct method only)
    eps, method, nthreads, sigma, w, rank_tol, chunk : as for potsynth.Model

    Calling the model with (r [km], theta [rad, colatitude], phi [rad]) returns
    (B_r, B_theta, B_phi) in nT, each with the broadcast shape of the inputs.
    """

    def __init__(self, gh, lmax=None, lmin=1, mmax=None, a=6371.2, rmin=0.0, rmax=0.0,
                 eps=1e-10, method="auto", nthreads=0, sigma=0.0, w=0, rank_tol=0.0, chunk=0):
        if isinstance(gh, tuple):
            g, h = gh
            lmax = np.asarray(g).shape[0] - 1 if lmax is None else lmax
            mmax = lmax if mmax is None else mmax
            gh = coef_vector(g, h, lmin=lmin, lmax=lmax, mmax=mmax)
        else:
            gh = np.asarray(gh, dtype=float)
            if lmax is None:
                lmax = lmin
                while ncoef(lmin, lmax, lmax) < gh.size:
                    lmax += 1
            mmax = lmax if mmax is None else mmax
        gh = np.ascontiguousarray(gh, dtype=float)
        if gh.size != ncoef(lmin, lmax, mmax):
            raise ValueError(f"gh has length {gh.size}, expected {ncoef(lmin, lmax, mmax)}")
        if lmax > _lib.LMAX:
            raise ValueError(f"lmax > {_lib.LMAX} is not supported")

        p = _params(eps, method, nthreads, sigma, w, rank_tol)
        status = ctypes.c_int(0)
        ptr = lib.potsynth_mag_alloc(lmin, lmax, mmax, gh.ctypes.data_as(_lib.c_double_p), a,
                                     rmin, rmax, ctypes.byref(p), ctypes.byref(status))
        if not ptr:
            _raise(status.value, "potsynth_mag_alloc")
        super().__init__(ptr, 3, chunk)

    def __call__(self, r, theta, phi):
        B = self._evaluate(r, theta, phi)
        return B[0], B[1], B[2]


def rtp_to_nec(Br, Btheta, Bphi):
    """Convert (B_r, B_theta, B_phi) to (B_north, B_east, B_center)."""
    return -np.asarray(Btheta), np.asarray(Bphi), -np.asarray(Br)
