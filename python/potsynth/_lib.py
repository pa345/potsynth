"""
Low-level ctypes bindings to the potsynth C library.

The shared library is located, in order of preference, from
  1. the environment variable POTSYNTH_LIBRARY (full path),
  2. the package directory (a library installed alongside the package),
  3. the system library search path (ctypes.util.find_library).
"""
import ctypes
import ctypes.util
import os
import sys

c_double_p = ctypes.POINTER(ctypes.c_double)

POTSYNTH_SUCCESS = 0
POTSYNTH_EINVAL = 1
POTSYNTH_ENOMEM = 2
POTSYNTH_EDOM = 3
POTSYNTH_EUNSUP = 4

METHOD_AUTO, METHOD_FAST, METHOD_DIRECT = 0, 1, 2
THETA_P, THETA_DP, THETA_PSIN = 0, 1, 2

LMAX = 2800


class Channel_t(ctypes.Structure):
    _fields_ = [("rpow", ctypes.c_int),
                ("rfac", c_double_p),
                ("theta", ctypes.c_int),
                ("dphi", ctypes.c_int)]


class Params_t(ctypes.Structure):
    _fields_ = [("eps", ctypes.c_double),
                ("method", ctypes.c_int),
                ("sigma", ctypes.c_double),
                ("w", ctypes.c_int),
                ("rank_tol", ctypes.c_double),
                ("nthreads", ctypes.c_int)]


class Info_t(ctypes.Structure):
    _fields_ = [("lmin", ctypes.c_size_t),
                ("lmax", ctypes.c_size_t),
                ("mmax", ctypes.c_size_t),
                ("nchan", ctypes.c_size_t),
                ("method", ctypes.c_int),
                ("w", ctypes.c_int),
                ("sigma", ctypes.c_double),
                ("rho", ctypes.c_size_t),
                ("ncheb", ctypes.c_size_t),
                ("nf1", ctypes.c_size_t),
                ("nf2", ctypes.c_size_t),
                ("nrows", ctypes.c_size_t),
                ("grid_bytes", ctypes.c_size_t)]


def _candidates():
    env = os.environ.get("POTSYNTH_LIBRARY")
    if env:
        yield env
    here = os.path.dirname(os.path.abspath(__file__))
    names = {"win32": ["potsynth.dll", "libpotsynth.dll"],
             "darwin": ["libpotsynth.dylib"]}.get(sys.platform, ["libpotsynth.so"])
    for name in names:
        yield os.path.join(here, name)
    found = ctypes.util.find_library("potsynth")
    if found:
        yield found


def _load():
    errors = []
    for path in _candidates():
        try:
            return ctypes.CDLL(path)
        except OSError as exc:
            errors.append(f"{path}: {exc}")
    raise ImportError("cannot load the potsynth shared library; build it with "
                      "-DBUILD_SHARED_LIBS=ON and set POTSYNTH_LIBRARY to its path"
                      + ("\n  " + "\n  ".join(errors) if errors else ""))


lib = _load()

lib.potsynth_strerror.restype = ctypes.c_char_p
lib.potsynth_strerror.argtypes = [ctypes.c_int]

lib.potsynth_params_init.restype = None
lib.potsynth_params_init.argtypes = [ctypes.POINTER(Params_t)]

lib.potsynth_ncoef.restype = ctypes.c_size_t
lib.potsynth_ncoef.argtypes = [ctypes.c_size_t] * 3

lib.potsynth_alloc.restype = ctypes.c_void_p
lib.potsynth_alloc.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, c_double_p,
                               ctypes.c_double, ctypes.c_double, ctypes.c_double,
                               ctypes.c_size_t, ctypes.POINTER(Channel_t),
                               ctypes.POINTER(Params_t), ctypes.POINTER(ctypes.c_int)]

lib.potsynth_mag_alloc.restype = ctypes.c_void_p
lib.potsynth_mag_alloc.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, c_double_p,
                                   ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                   ctypes.POINTER(Params_t), ctypes.POINTER(ctypes.c_int)]

lib.potsynth_free.restype = None
lib.potsynth_free.argtypes = [ctypes.c_void_p]

lib.potsynth_eval_alloc.restype = ctypes.c_void_p
lib.potsynth_eval_alloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t]

lib.potsynth_eval_free.restype = None
lib.potsynth_eval_free.argtypes = [ctypes.c_void_p]

lib.potsynth_eval.restype = ctypes.c_int
lib.potsynth_eval.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                              c_double_p, c_double_p, c_double_p, c_double_p]

lib.potsynth_get_info.restype = ctypes.c_int
lib.potsynth_get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(Info_t)]


def strerror(status):
    return lib.potsynth_strerror(status).decode()
