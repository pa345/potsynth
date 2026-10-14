"""
potsynth -- fast synthesis of internal potential-field models at scattered points.

    import potsynth
    model = potsynth.MagModel((g, h), rmin=6371.2 + 250, rmax=6371.2 + 800, eps=1e-10)
    Br, Btheta, Bphi = model(r, theta, phi)

This module calls the C library through ctypes (no copies of the input or
output arrays; the GIL is released during evaluation).
"""
from .core import Channel, DomainError, Model, PotsynthError, coef_vector, idx, ncoef
from .mag import MagModel, rtp_to_nec

__version__ = "0.1.0"

__all__ = ["MagModel", "Model", "Channel", "rtp_to_nec", "coef_vector", "idx", "ncoef",
           "PotsynthError", "DomainError"]
