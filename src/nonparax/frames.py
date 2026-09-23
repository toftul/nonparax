"""Beams with another focus or direction, built on the +z beam of :func:`nonparax.EH`.

The beam frame has the focus at the origin and the beam along +z. A beam in the
lab with focus r0 and rotation R (R maps the beam axes to lab axes) is

    r_b = R^T (r - r0),    E_lab = R E_b(r_b),    H_lab = R H_b(r_b).

The ``jones`` vector is always given in the beam frame.
"""

import numpy as np

from .core import _run

__all__ = ["to_beam_frame", "to_lab_frame", "rotation", "rotation_to", "EH_lab"]


def _matrix(R):
    if R is None:
        return np.eye(3)
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3) or not np.allclose(R @ R.T, np.eye(3), atol=1e-10) \
            or not np.isclose(np.linalg.det(R), 1.0):
        raise ValueError("R must be a 3x3 rotation matrix (orthogonal, det = +1)")
    return R


def to_beam_frame(x, y, z, r0=(0.0, 0.0, 0.0), R=None):
    """Lab points -> beam-frame points, r_b = R^T (r - r0). Returns (xb, yb, zb)."""
    R = _matrix(R)
    x, y, z = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (x, y, z)))
    r0 = np.asarray(r0, dtype=float)
    if r0.shape != (3,):
        raise ValueError("r0 must be a point (x0, y0, z0)")
    d = np.stack([x - r0[0], y - r0[1], z - r0[2]])
    xb, yb, zb = np.einsum("ji,j...->i...", R, d)
    return xb, yb, zb


def to_lab_frame(F, R=None):
    """Beam-frame vectors of shape (3, ...) -> lab vectors, F_lab = R F_b."""
    R = _matrix(R)
    F = np.asarray(F)
    if F.shape[:1] != (3,):
        raise ValueError("F must have shape (3, ...)")
    return np.einsum("ij,j...->i...", R, F)


def rotation(axis, angle):
    """Rotation by ``angle`` (rad) about ``axis``, right-hand rule (Rodrigues)."""
    n = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(n)
    if n.shape != (3,) or norm == 0:
        raise ValueError("axis must be a nonzero 3-vector")
    n = n / norm
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def rotation_to(u):
    """The smallest rotation that takes +z to the direction ``u``.

    For u = -z it is the rotation by pi about x.
    """
    u = np.asarray(u, dtype=float)
    norm = np.linalg.norm(u)
    if u.shape != (3,) or norm == 0:
        raise ValueError("u must be a nonzero 3-vector")
    u = u / norm
    zhat = np.array([0.0, 0.0, 1.0])
    axis = np.cross(zhat, u)
    s = np.linalg.norm(axis)
    if s < 1e-12:
        return np.eye(3) if u[2] > 0 else rotation((1, 0, 0), np.pi)
    return rotation(axis, np.arctan2(s, u[2]))


def EH_lab(kind, x, y, z, *, r0=(0.0, 0.0, 0.0), R=None, **kwargs):
    """E [V/m] and H [A/m] of a beam with focus ``r0`` and rotation ``R``.

    Equivalent to ``to_beam_frame`` -> :func:`nonparax.EH` -> ``to_lab_frame``.
    ``kwargs`` are the keyword arguments of :func:`nonparax.EH`; ``jones`` is in
    the beam frame, so x-polarized means along R x_hat.
    """
    xb, yb, zb = to_beam_frame(x, y, z, r0=r0, R=R)
    res = _run(kind, xb, yb, zb, ("E", "H"), **kwargs)
    return to_lab_frame(res["E"], R), to_lab_frame(res["H"], R)
