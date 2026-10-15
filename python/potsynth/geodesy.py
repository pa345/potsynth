"""
Geodetic helpers for gravity models: reference ellipsoids, geodetic to geocentric
coordinates, the normal field and normal gravity on the ellipsoid (see the
"Gravity recipes" chapter of the documentation).
"""
import ctypes
from dataclasses import dataclass

import numpy as np

from . import _lib
from ._lib import lib
from .core import _raise, nlm


@dataclass(frozen=True)
class Ellipsoid:
    """Reference ellipsoid by its defining parameters: semi-major axis a (m), flattening
    f, geocentric gravitational constant GM (m^3/s^2) and angular velocity omega (rad/s)."""
    a: float
    f: float
    GM: float
    omega: float

    def _c(self):
        return _lib.Ellipsoid_t(self.a, self.f, self.GM, self.omega)


def _builtin(name):
    e = _lib.Ellipsoid_t.in_dll(lib, name)
    return Ellipsoid(e.a, e.f, e.GM, e.omega)


WGS84 = _builtin("potsynth_wgs84")
GRS80 = _builtin("potsynth_grs80")


def _check(status, what):
    if status:
        _raise(status, what)


def geodetic_to_geocentric(lat, h, ellipsoid=WGS84):
    """Convert geodetic latitude lat (radians) and ellipsoidal height h (m) to
    geocentric radius r (m) and colatitude theta (radians); the longitude is
    unchanged. Returns (r, theta) with the broadcast shape of the inputs."""
    lat, h = np.broadcast_arrays(np.asarray(lat, float), np.asarray(h, float))
    shape = lat.shape
    lat = np.ascontiguousarray(lat.ravel())
    h = np.ascontiguousarray(h.ravel())
    r = np.empty(lat.size)
    theta = np.empty(lat.size)
    ptr = lambda x: x.ctypes.data_as(_lib.c_double_p)
    e = ellipsoid._c()
    _lib.clear_error()
    _check(lib.potsynth_geodetic2geocentric(lat.size, ptr(lat), ptr(h), ctypes.byref(e), ptr(r),
                                            ptr(theta)), "potsynth_geodetic2geocentric")
    return r.reshape(shape), theta.reshape(shape)


def normal_zonals(lmax, ellipsoid=WGS84):
    """4-pi normalized zonal coefficients C[l], l = 0..lmax, of the normal gravitational
    potential of the ellipsoid, in its own scaling (its GM and a); C[0] = 1, odd degrees
    and degrees above 20 are 0."""
    C = np.empty(lmax + 1)
    e = ellipsoid._c()
    _lib.clear_error()
    _check(lib.potsynth_normal_zonals(ctypes.byref(e), lmax, C.ctypes.data_as(_lib.c_double_p)),
           "potsynth_normal_zonals")
    return C


def subtract_normal_field(coef, GM, a, lmin=0, lmax=None, mmax=None, ellipsoid=WGS84):
    """Return the coefficients of the disturbing potential T = V - U: a copy of the
    4-pi normalized coefficient vector coef (layout of idx(), model constants GM in
    m^3/s^2 and a in m) with the normal zonals of the ellipsoid subtracted, rescaled to
    GM and a. Degree 0 is included, so T contains the term (GM - GM_ell)/r."""
    coef = np.array(coef, dtype=float)
    if lmax is None:
        lmax = lmin
        while nlm(lmin, lmax, lmax) < coef.size:
            lmax += 1
    mmax = lmax if mmax is None else mmax
    if coef.size != nlm(lmin, lmax, mmax):
        raise ValueError(f"coef has length {coef.size}, expected {nlm(lmin, lmax, mmax)}")
    e = ellipsoid._c()
    _lib.clear_error()
    _check(lib.potsynth_grav_subtract_normal(lmin, lmax, mmax, coef.ctypes.data_as(_lib.c_double_p),
                                             GM, a, ctypes.byref(e)), "potsynth_grav_subtract_normal")
    return coef


def normal_gravity(lat, ellipsoid=WGS84):
    """Normal gravity (m/s^2) on the surface of the ellipsoid at geodetic latitude lat
    (radians), by Somigliana's formula."""
    lat = np.asarray(lat, float)
    shape = lat.shape
    x = np.ascontiguousarray(lat.ravel())
    g = np.empty(x.size)
    e = ellipsoid._c()
    _lib.clear_error()
    _check(lib.potsynth_normal_gravity(x.size, x.ctypes.data_as(_lib.c_double_p), ctypes.byref(e),
                                       g.ctypes.data_as(_lib.c_double_p)), "potsynth_normal_gravity")
    return g.reshape(shape)
