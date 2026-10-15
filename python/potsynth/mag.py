"""
Magnetic front-end: B = -grad V and its gradient tensor grad B for Schmidt
semi-normalised Gauss coefficients.
"""
import collections
import ctypes
import os
from dataclasses import dataclass

import numpy as np

from . import _lib
from ._lib import lib
from .core import _Evaluator, _params, _raise, coef_vector, nlm

MAG_B = 0x1
MAG_GRADB = 0x2

_MAG_QUANTITIES = {"B": MAG_B, "gradB": MAG_GRADB}


def _mag_flags(quantities):
    if isinstance(quantities, str):
        quantities = [q.strip() for q in quantities.split(",")]
    flags = 0
    for q in quantities:
        if q not in _MAG_QUANTITIES:
            raise ValueError(f"unknown quantity {q!r}; use 'B' or 'gradB'")
        flags |= _MAG_QUANTITIES[q]
    if flags == 0:
        raise ValueError("no quantities requested")
    names = (["B_r", "B_theta", "B_phi"] if flags & MAG_B else []) + \
            (["dB_rr", "dB_rt", "dB_rp", "dB_tt", "dB_tp", "dB_pp"] if flags & MAG_GRADB else [])
    return flags, names


@dataclass
class ShcModel:
    """Static model read from a Swarm SHC file.

    gh holds the Schmidt semi-normalized Gauss coefficients in nT, indexed by
    idx(l, m, lmin, mmax) (g at m >= 0, h at m < 0)."""
    lmin: int
    lmax: int
    mmax: int
    epoch: float
    gh: np.ndarray


def read_shc(filename):
    """Read a static model (one epoch, e.g. a lithospheric model) from a Swarm SHC file.

    Time-dependent SHC files are not supported (core field models are of low degree
    and are best evaluated with a direct sum). Returns a ShcModel."""
    _lib.clear_error()
    ptr = lib.potsynth_shc_read(os.fsencode(filename))
    if not ptr:
        _raise(_lib.POTSYNTH_EINVAL, "potsynth_shc_read")
    try:
        s = ptr.contents
        n = nlm(s.lmin, s.lmax, s.mmax)
        gh = np.ctypeslib.as_array(s.gh, shape=(n,)).copy()
        return ShcModel(int(s.lmin), int(s.lmax), int(s.mmax), float(s.epoch), gh)
    finally:
        lib.potsynth_shc_free(ptr)


class MagModel(_Evaluator):
    """Internal magnetic field model (IGRF, CHAOS, lithospheric models...).

    Parameters
    ----------
    gh : Gauss coefficients in nT, either a 1-D vector in the layout of idx()
         (g at m >= 0, h at m < 0), or a tuple (g, h) of 2-D arrays g[l, m], h[l, m]
    lmax : maximum degree (default: from gh)
    lmin : minimum degree (default 1)
    mmax : maximum order (default lmax)
    a : reference radius in km (default 6371.2)
    quantities : "B" (field, nT) and/or "gradB" (gradient tensor grad B = -grad grad V,
                 nT/km), as a list or a comma-separated string (default "B")
    rmin, rmax : radial range in km of the evaluation points (required by the fast
                 method; default: none, i.e. direct method only)
    eps, method, npts_hint, nthreads, sigma, w, rank_tol, chunk : as for potsynth.Model

    Calling the model with (r [km], theta [rad, colatitude], phi [rad]) returns a named
    tuple with the requested quantities, in the order B_r, B_theta, B_phi (nT), then
    dB_rr, dB_rt, dB_rp, dB_tt, dB_tp, dB_pp (nT/km), each with the broadcast shape of
    the inputs.
    """

    def __init__(self, gh, lmax=None, lmin=1, mmax=None, a=6371.2, rmin=0.0, rmax=0.0,
                 eps=1e-10, method="auto", nthreads=0, sigma=0.0, w=0, rank_tol=0.0, chunk=0,
                 npts_hint=0, quantities="B"):
        flags, names = _mag_flags(quantities)
        if isinstance(gh, tuple):
            g, h = gh
            lmax = np.asarray(g).shape[0] - 1 if lmax is None else lmax
            mmax = lmax if mmax is None else mmax
            gh = coef_vector(g, h, lmin=lmin, lmax=lmax, mmax=mmax)
        else:
            gh = np.asarray(gh, dtype=float)
            if lmax is None:
                lmax = lmin
                while nlm(lmin, lmax, lmax) < gh.size:
                    lmax += 1
            mmax = lmax if mmax is None else mmax
        gh = np.ascontiguousarray(gh, dtype=float)
        if gh.size != nlm(lmin, lmax, mmax):
            raise ValueError(f"gh has length {gh.size}, expected {nlm(lmin, lmax, mmax)}")
        if lmax > _lib.LMAX:
            raise ValueError(f"lmax > {_lib.LMAX} is not supported")

        p = _params(eps, method, nthreads, sigma, w, rank_tol, npts_hint)
        _lib.clear_error()
        ptr = lib.potsynth_mag_alloc(lmin, lmax, mmax, gh.ctypes.data_as(_lib.c_double_p), a,
                                     rmin, rmax, flags, ctypes.byref(p))
        if not ptr:
            _raise(_lib.POTSYNTH_EINVAL, "potsynth_mag_alloc")
        super().__init__(ptr, len(names), chunk)
        self.quantities = tuple(names)
        self._result = collections.namedtuple("MagField", names)

    @classmethod
    def from_shc(cls, filename, a=6371.2, rmin=0.0, rmax=0.0, **kwargs):
        """Build a model from a static Swarm SHC file (see read_shc()); the other
        arguments are as for MagModel."""
        shc = read_shc(filename)
        return cls(shc.gh, lmax=shc.lmax, lmin=shc.lmin, mmax=shc.mmax, a=a, rmin=rmin, rmax=rmax,
                   **kwargs)

    def __call__(self, r, theta, phi):
        return self._result(*self._evaluate(r, theta, phi))


def rtp_to_nec(Br, Btheta, Bphi):
    """Convert (B_r, B_theta, B_phi) to (B_north, B_east, B_center)."""
    return -np.asarray(Btheta), np.asarray(Bphi), -np.asarray(Br)
