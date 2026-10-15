"""
potsynth -- fast synthesis of internal potential-field models at scattered points.

    import potsynth
    model = potsynth.MagModel((g, h), rmin=6371.2 + 250, rmax=6371.2 + 800, eps=1e-10)
    Br, Btheta, Bphi = model(r, theta, phi)

    grav = potsynth.GravModel.from_gfc("EGM2008.gfc", lmax=720, quantities="g_r",
                                       rmin=6.35e6, rmax=6.9e6)
    g_r = grav(r, theta, phi).g_r

This module calls the C library through ctypes (no copies of the input or
output arrays; the GIL is released during evaluation).
"""
from .core import Channel, DomainError, Model, PotsynthError, coef_vector, idx, nlm
from .mag import MagModel, ShcModel, read_shc, rtp_to_nec
from .grav import GfcModel, GravModel, read_gfc
from .tensor import tensor_matrix, tensor_rtp_to_nec, tensor_rtp_to_xyz

__version__ = "0.1.0"

__all__ = ["MagModel", "GravModel", "Model", "Channel", "ShcModel", "read_shc", "rtp_to_nec",
           "GfcModel", "read_gfc", "tensor_matrix", "tensor_rtp_to_nec", "tensor_rtp_to_xyz",
           "coef_vector",
           "idx", "nlm", "PotsynthError", "DomainError"]
