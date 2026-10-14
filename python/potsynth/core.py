"""
Generic potential-field synthesis: channels of an internal potential.
"""
import ctypes
import threading
from dataclasses import dataclass

import numpy as np

from . import _lib
from ._lib import lib

_METHODS = {"auto": _lib.METHOD_AUTO, "fast": _lib.METHOD_FAST, "direct": _lib.METHOD_DIRECT}
_METHOD_NAMES = {v: k for k, v in _METHODS.items()}
_THETA_OPS = {"P": _lib.THETA_P, "dP": _lib.THETA_DP, "Psin": _lib.THETA_PSIN}


class PotsynthError(RuntimeError):
    """Error reported by the potsynth library."""

    def __init__(self, status, what="", reason=""):
        self.status = status
        detail = f"{reason} ({_lib.strerror(status)})" if reason else _lib.strerror(status)
        super().__init__(f"{what}{': ' if what else ''}{detail}")


class DomainError(PotsynthError, ValueError):
    """An evaluation point lies outside the valid domain (theta outside [0, pi],
    r <= 0, or, for the fast method, r outside [rmin, rmax])."""


def _raise(status, what):
    """Raise the first error recorded by the library error handler (or status)."""
    status, reason = _lib.last_error(status)
    if status == _lib.POTSYNTH_EDOM:
        raise DomainError(status, what, reason)
    raise PotsynthError(status, what, reason)


def nlm(lmin, lmax, mmax=None):
    """Length of a coefficient vector for degrees lmin..lmax and orders up to mmax."""
    mmax = lmax if mmax is None else mmax
    return int(lib.potsynth_nlm(lmin, lmax, mmax))


def idx(n, m, lmin, mmax):
    """Index of coefficient (n, m) in a coefficient vector (mainlib green2 layout):
    m >= 0 indexes the cosine coefficient, m < 0 the sine coefficient of order |m|."""
    if lmin <= mmax:
        if n <= mmax:
            return n * (n + 1) + m - lmin * lmin
        return n * (2 * mmax + 1) + m - mmax * mmax - lmin * lmin
    return (n - lmin) * (2 * mmax + 1) + m + mmax


def coef_vector(C, S, lmin=None, lmax=None, mmax=None):
    """Build a coefficient vector from arrays C[n, m] (cosine) and S[n, m] (sine),
    for example Gauss coefficients g[n, m], h[n, m].

    lmin defaults to 0 if C[0, 0] != 0 and to 1 otherwise; lmax to the largest
    degree in the arrays; mmax to lmax.
    """
    C = np.asarray(C, dtype=float)
    S = np.zeros_like(C) if S is None else np.asarray(S, dtype=float)
    L = C.shape[0] - 1
    lmax = L if lmax is None else lmax
    lmin = (0 if C[0, 0] != 0.0 else 1) if lmin is None else lmin
    mmax = lmax if mmax is None else mmax
    out = np.zeros(nlm(lmin, lmax, mmax))
    for n in range(lmin, lmax + 1):
        M = min(n, mmax)
        out[idx(n, 0, lmin, mmax)] = C[n, 0]
        for m in range(1, M + 1):
            out[idx(n, m, lmin, mmax)] = C[n, m]
            out[idx(n, -m, lmin, mmax)] = S[n, m]
    return out


@dataclass
class Channel:
    """One output quantity of the core model,

        f(r, theta, phi) = sum_{n,m} rfac[n] (a/r)^(n + rpow) T[Pbar_n^m(cos theta)]
                                     (d/dphi)^dphi [C_nm cos(m phi) + S_nm sin(m phi)]

    theta: "P" (Pbar), "dP" (d Pbar / d theta) or "Psin" (Pbar / sin theta, m >= 1).
    rfac: per-degree factors, length lmax + 1, or None for 1.
    """
    rpow: int
    theta: str = "P"
    dphi: int = 0
    rfac: object = None


def _params(eps, method, nthreads, sigma, w, rank_tol):
    p = lib.potsynth_default_parameters()
    p.eps = eps
    p.method = _METHODS[method]
    p.nthreads = nthreads
    p.sigma = sigma
    p.w = w
    p.rank_tol = rank_tol
    return p


class _Evaluator:
    """Owns a C workspace and an evaluation workspace; evaluates its channels."""

    def __init__(self, ptr, nchan, chunk):
        self._ptr = ptr
        self._nchan = nchan
        _lib.clear_error()
        self._ew = lib.potsynth_eval_alloc(chunk, ptr)
        self._lock = threading.Lock()      # one eval workspace per object
        if not self._ew:
            lib.potsynth_free(ptr)
            self._ptr = None
            raise MemoryError("potsynth_eval_alloc failed")

    def __del__(self):
        if getattr(self, "_ew", None):
            lib.potsynth_eval_free(self._ew)
            self._ew = None
        if getattr(self, "_ptr", None):
            lib.potsynth_free(self._ptr)
            self._ptr = None

    def _evaluate(self, r, theta, phi):
        r, theta, phi = np.broadcast_arrays(np.asarray(r, float), np.asarray(theta, float),
                                            np.asarray(phi, float))
        shape = r.shape
        r = np.ascontiguousarray(r.ravel())
        theta = np.ascontiguousarray(theta.ravel())
        phi = np.ascontiguousarray(phi.ravel())
        out = np.empty((self._nchan, r.size))
        ptr = lambda a: a.ctypes.data_as(_lib.c_double_p)
        with self._lock:
            _lib.clear_error()
            status = lib.potsynth_eval(r.size, ptr(r), ptr(theta), ptr(phi), ptr(out), self._ew)
        if status:
            _raise(status, "potsynth_eval")
        return out.reshape((self._nchan,) + shape)

    @property
    def info(self):
        """Parameters chosen at set-up (method, window width, radial rank, grid size...)."""
        i = _lib.Info_t()
        lib.potsynth_get_info(ctypes.byref(i), self._ptr)
        d = {name: getattr(i, name) for name, _ in _lib.Info_t._fields_}
        d["method"] = _METHOD_NAMES.get(d["method"], d["method"])
        return d


class Model(_Evaluator):
    """Generic internal potential-field model with user-defined channels.

    Parameters
    ----------
    coef : 1-D array, 4-pi fully normalised coefficients in the layout of idx()
           (see coef_vector()), length nlm(lmin, lmax, mmax)
    lmin, lmax, mmax : degree range and maximum order (mmax defaults to lmax)
    channels : list of Channel
    a : reference radius (same unit as r)
    rmin, rmax : radial range of the evaluation points (required by the fast method)
    eps : requested accuracy relative to the largest output value
    method : "auto", "fast" or "direct"
    nthreads : OpenMP threads (0 = all)
    sigma, w, rank_tol : expert overrides (0 = automatic)
    chunk : points per internal chunk (0 = default)

    Calling the model with (r, theta, phi) returns an array of shape
    (len(channels),) + broadcast shape of the inputs.
    """

    def __init__(self, coef, lmin, lmax, channels, a, rmin=0.0, rmax=0.0, mmax=None, eps=1e-10,
                 method="auto", nthreads=0, sigma=0.0, w=0, rank_tol=0.0, chunk=0):
        mmax = lmax if mmax is None else mmax
        coef = np.ascontiguousarray(coef, dtype=float)
        if coef.size != nlm(lmin, lmax, mmax):
            raise ValueError(f"coef has length {coef.size}, expected {nlm(lmin, lmax, mmax)}")
        if lmax > _lib.LMAX:
            raise ValueError(f"lmax > {_lib.LMAX} is not supported")

        chans = (_lib.Channel_t * len(channels))()
        self._rfac = []
        for i, ch in enumerate(channels):
            chans[i].rpow = ch.rpow
            chans[i].theta = _THETA_OPS[ch.theta]
            chans[i].dphi = ch.dphi
            if ch.rfac is not None:
                rf = np.ascontiguousarray(ch.rfac, dtype=float)
                if rf.size < lmax + 1:
                    raise ValueError("Channel.rfac must have length lmax + 1")
                self._rfac.append(rf)
                chans[i].rfac = rf.ctypes.data_as(_lib.c_double_p)

        p = _params(eps, method, nthreads, sigma, w, rank_tol)
        _lib.clear_error()
        ptr = lib.potsynth_alloc(lmin, lmax, mmax, coef.ctypes.data_as(_lib.c_double_p), a,
                                 rmin, rmax, len(channels), chans, ctypes.byref(p))
        if not ptr:
            _raise(_lib.POTSYNTH_EINVAL, "potsynth_alloc")
        super().__init__(ptr, len(channels), chunk)

    def __call__(self, r, theta, phi):
        return self._evaluate(r, theta, phi)
