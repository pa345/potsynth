"""
Frame conversions of symmetric tensors (gradient tensors) given as six components
(11, 12, 13, 22, 23, 33), as returned by MagModel ("gradB") and GravModel ("T").
"""
import numpy as np


def tensor_rtp_to_nec(rr, rt, rp, tt, tp, pp):
    """Convert spherical components (rr, rt, rp, tt, tp, pp) to North-East-Center
    components (NN, NE, NC, EE, EC, CC), with e_N = -e_theta, e_E = e_phi, e_C = -e_r."""
    return (np.asarray(tt), -np.asarray(tp), np.asarray(rt), np.asarray(pp), -np.asarray(rp),
            np.asarray(rr))


def tensor_rtp_to_xyz(theta, phi, rr, rt, rp, tt, tp, pp):
    """Convert spherical components (rr, rt, rp, tt, tp, pp) at colatitude theta and
    longitude phi (radians) to Earth-fixed Cartesian components (xx, xy, xz, yy, yz, zz):
    T_xyz = R^T T_rtp R. Unlike the spherical components, these are unique at the poles."""
    S = tensor_matrix(rr, rt, rp, tt, tp, pp)
    theta, phi = np.broadcast_arrays(np.asarray(theta, float), np.asarray(phi, float))
    s, c, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
    R = np.empty(theta.shape + (3, 3))
    R[..., 0, :] = np.stack([s * cp, s * sp, c], axis=-1)
    R[..., 1, :] = np.stack([c * cp, c * sp, -s], axis=-1)
    R[..., 2, :] = np.stack([-sp, cp, np.zeros_like(s)], axis=-1)
    X = np.einsum("...ap,...ab,...bq->...pq", R, S, R)
    return X[..., 0, 0], X[..., 0, 1], X[..., 0, 2], X[..., 1, 1], X[..., 1, 2], X[..., 2, 2]


def tensor_matrix(t11, t12, t13, t22, t23, t33):
    """Assemble the six components into symmetric 3x3 matrices, shape (..., 3, 3)."""
    t = np.broadcast_arrays(*(np.asarray(v, float) for v in (t11, t12, t13, t22, t23, t33)))
    return np.stack([np.stack([t[0], t[1], t[2]], -1),
                     np.stack([t[1], t[3], t[4]], -1),
                     np.stack([t[2], t[4], t[5]], -1)], -2)
