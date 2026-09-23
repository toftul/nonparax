"""Fields of an exact non-paraxial Gaussian beam along +z, focus at the origin.

The fields are the angular-spectrum integral of the white paper, Eq. (debye),
with the integral over the azimuth done analytically (Bessel form,
Eqs. (bessel), (bessel_H) and the vortex section). The remaining integral over
the polar angle theta uses fixed Gauss-Legendre quadrature.

Conventions: SI units, time dependence exp(-i omega t) unless
``time_convention="+j"``, circular basis e_sigma = (x + i sigma y)/sqrt(2) with
the handedness sigma = +-1.
"""

import numbers
import warnings

import numpy as np
from scipy.constants import epsilon_0, mu_0
from scipy.special import jv

from .beams import envelope_and_rule
from .power import amplitude_from_power

__all__ = ["EH", "E", "H", "convergence", "ott_w0", "ConvergenceWarning"]

# number of (point, node) pairs held in memory at once
_CHUNK = 2_000_000


class ConvergenceWarning(UserWarning):
    """The fields changed by more than ``rtol`` when n_theta was doubled."""


# ----------------------------------------------------------------------------- kernel
def _handedness_fields(a, c, theta, weights, k, sigma, charge, rho, vphi, z, which):
    """Fields for the input e_sigma of one handedness, per unit amplitude A, as sqrt(eps)E and sqrt(mu)H.

    Returns a dict with keys from ``which`` ("E", "H"); each value has shape (3, N).
    With E_inf = A a e^{i l phi} e^{i sigma phi} (u theta_hat + i sigma v phi_hat),
    the phi integral gives, in the basis (e_sigma, e_-sigma, z_hat),
        (u cos + v)/sqrt2 * i^l J_l e^{i l vphi},
        (u cos - v)/sqrt2 * i^(l+2s) J_(l+2s) e^{i(l+2s) vphi},
        -u sin * i^(l+s) J_(l+s) e^{i(l+s) vphi},
    times 2 pi int dtheta sin(theta) e^{ikz cos(theta)} (...).
    For E: u = c, v = 1. For H: u = 1, v = c and an overall factor -i sigma.
    Both are divided by sqrt(1 + c^2), which makes |p| = 1.
    """
    ct, st = np.cos(theta), np.sin(theta)
    norm = np.sqrt(1 + c**2)
    base = 2 * np.pi * weights * st * a / norm                      # (n,)
    orders = (charge, charge + sigma, charge + 2 * sigma)
    uv = {"E": (c, np.ones_like(c), 1.0), "H": (np.ones_like(c), c, -1j * sigma)}
    coef = {}
    for f in which:
        u, v, pref = uv[f]
        coef[f] = (pref * (u * ct + v) / np.sqrt(2), pref * (u * ct - v) / np.sqrt(2), -pref * u * st)

    N = rho.size
    out = {f: np.empty((3, N), dtype=complex) for f in which}
    e_s = np.array([1, 1j * sigma, 0]) / np.sqrt(2)
    e_m = np.array([1, -1j * sigma, 0]) / np.sqrt(2)
    e_z = np.array([0, 0, 1], dtype=complex)
    step = max(1, _CHUNK // max(1, theta.size))
    for i0 in range(0, N, step):
        sl = slice(i0, min(N, i0 + step))
        kr = k * rho[sl, None] * st[None, :]
        w = base[None, :] * np.exp(1j * k * z[sl, None] * ct[None, :])
        J = [jv(m, kr) for m in orders]
        ph = [(1j) ** m * np.exp(1j * m * vphi[sl]) for m in orders]
        for f in which:
            c_s, c_m, c_z = coef[f]
            s_s = np.sum(w * c_s * J[0], axis=1) * ph[0]
            s_z = np.sum(w * c_z * J[1], axis=1) * ph[1]
            s_m = np.sum(w * c_m * J[2], axis=1) * ph[2]
            out[f][:, sl] = np.outer(e_s, s_s) + np.outer(e_m, s_m) + np.outer(e_z, s_z)
    return out


# ----------------------------------------------------------------------------- inputs
def _real_positive(name, value):
    v = np.asarray(value)
    if v.ndim != 0 or np.iscomplexobj(v) or not np.isfinite(v) or not v > 0:
        raise ValueError(f"{name} must be a real positive number (lossless medium), got {value!r}")
    return float(v)


def _setup(kind, *, wavelength, w0, jones, eps_r, mu_r, P, theta_max, NA_stop, charge,
           time_convention, n_theta):
    wavelength = _real_positive("wavelength", wavelength)
    w0 = _real_positive("w0", w0)
    eps_r = _real_positive("eps_r", eps_r)
    mu_r = _real_positive("mu_r", mu_r)
    P = np.asarray(P)
    if P.ndim != 0 or np.iscomplexobj(P) or not np.isfinite(P) or P < 0:
        raise ValueError(f"P must be a real number >= 0 (watts), got {P!r}")
    P = float(P)
    if isinstance(charge, bool) or not isinstance(charge, numbers.Integral):
        raise ValueError(f"charge must be an integer, got {charge!r}")
    if time_convention not in ("-i", "+j"):
        raise ValueError('time_convention must be "-i" (exp(-i w t)) or "+j" (exp(+j w t))')
    if isinstance(n_theta, bool) or not isinstance(n_theta, numbers.Integral) or n_theta < 1:
        raise ValueError(f"n_theta must be a positive integer, got {n_theta!r}")

    jones = np.asarray(jones, dtype=complex)
    if jones.shape != (2,) or not np.all(np.isfinite(jones)):
        raise ValueError("jones must be a finite pair (Ex, Ey)")
    jn = np.linalg.norm(jones)
    if jn == 0:
        raise ValueError("jones must not be zero")
    ex, ey = jones / jn
    # e = alpha e_+ + beta e_-, white paper Eq. (arbitrary)
    handedness = {+1: (ex - 1j * ey) / np.sqrt(2), -1: (ex + 1j * ey) / np.sqrt(2)}

    n = np.sqrt(eps_r * mu_r)
    if theta_max is not None and NA_stop is not None:
        raise ValueError("give theta_max or NA_stop, not both")
    if NA_stop is not None:
        NA_stop = _real_positive("NA_stop", NA_stop)
        if NA_stop > n:
            raise ValueError(f"NA_stop = {NA_stop} exceeds the refractive index n = {n}")
        theta_max = np.arcsin(NA_stop / n)
    elif theta_max is None:
        theta_max = np.pi / 2
    else:
        theta_max = _real_positive("theta_max", theta_max)
        if theta_max > np.pi / 2:
            raise ValueError("theta_max must not exceed pi/2")

    k = 2 * np.pi * n / wavelength
    eps, mu = eps_r * epsilon_0, mu_r * mu_0
    return dict(kind=kind, k=k, kw0=k * w0, eps=eps, mu=mu, P=P, theta_max=theta_max,
                handedness=handedness, charge=int(charge), time_convention=time_convention)


def _compute(s, x, y, z, n_theta, which):
    x, y, z = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (x, y, z)))
    shape = x.shape
    rho = np.hypot(x, y).ravel()
    vphi = np.arctan2(y, x).ravel()
    zz = z.ravel()

    t, w = np.polynomial.legendre.leggauss(n_theta)
    theta = 0.5 * s["theta_max"] * (t + 1)
    weights = 0.5 * s["theta_max"] * w
    a, c = envelope_and_rule(s["kind"], theta, s["kw0"])
    A = amplitude_from_power(s["P"], s["k"], s["eps"], s["mu"], a, np.sin(theta), weights)

    total = {f: np.zeros((3, rho.size), dtype=complex) for f in which}
    for sigma, amp in s["handedness"].items():
        if amp == 0:
            continue
        part = _handedness_fields(a, c, theta, weights, s["k"], sigma, s["charge"], rho, vphi, zz, which)
        for f in which:
            total[f] += amp * part[f]
    scale = {"E": A / np.sqrt(s["eps"]), "H": A / np.sqrt(s["mu"])}
    result = {}
    for f in which:
        F = (scale[f] * total[f]).reshape((3,) + shape)
        result[f] = np.conj(F) if s["time_convention"] == "+j" else F
    return result


def _rel_change(F1, F2):
    ref = np.max(np.abs(F2))
    return 0.0 if ref == 0 else float(np.max(np.abs(F1 - F2)) / ref)


def _run(kind, x, y, z, which, *, wavelength, w0, jones, eps_r=1.0, mu_r=1.0, P=1.0,
         theta_max=None, NA_stop=None, charge=0, time_convention="-i", n_theta=200,
         check=False, rtol=1e-6):
    s = _setup(kind, wavelength=wavelength, w0=w0, jones=jones, eps_r=eps_r, mu_r=mu_r, P=P,
               theta_max=theta_max, NA_stop=NA_stop, charge=charge,
               time_convention=time_convention, n_theta=n_theta)
    res = _compute(s, x, y, z, n_theta, which)
    if check:
        res2 = _compute(s, x, y, z, 2 * n_theta, which)
        change = max(_rel_change(res[f], res2[f]) for f in which)
        if change > rtol:
            warnings.warn(
                f"fields changed by {change:.2e} (> rtol = {rtol:.0e}) when n_theta was doubled "
                f"from {n_theta}; increase n_theta", ConvergenceWarning, stacklevel=3)
    return res


# ----------------------------------------------------------------------------- public API
_DOC = """

    Parameters
    ----------
    kind : str or (callable, callable)
        One of ``nonparax.KINDS``: "spectrum" (A), "aplanatic" (B), "thin_lens" (C),
        "comsol" (D), "ott_tan" (E), "ott_sin" (F) -- letters as in Table I of the
        white paper -- or a pair ``(a, c)`` of functions of theta (rad) giving the
        envelope a(theta) (may be complex) and the real meridional weight c(theta).
    x, y, z : array_like
        Points in metres; broadcast against each other. The beam travels along +z
        and its focus is at the origin.
    wavelength : float
        Vacuum wavelength in metres.
    w0 : float
        Paraxial waist radius in metres (for OTT's NA use :func:`ott_w0`).
    jones : (complex, complex)
        Input polarization (Ex, Ey) of the paraxial beam; normalized internally.
        (1, 0) is x-polarized, (1, 1j) is circular with the handedness +1.
    eps_r, mu_r : float
        Relative permittivity and permeability of the lossless medium (real, > 0).
    P : float
        Beam power in watts, white paper Eq. (power).
    theta_max, NA_stop : float, optional
        A stop, as the largest polar angle in radians or as NA = n sin(theta_max).
        Give at most one; the default is theta_max = pi/2 (no stop).
    charge : int
        Vortex charge l; the plane-wave amplitude gains exp(i l phi).
    time_convention : {"-i", "+j"}
        "-i": exp(-i omega t) (white paper). "+j": exp(+j omega t) as in COMSOL;
        the result is the complex conjugate.
    n_theta : int
        Number of Gauss-Legendre nodes in theta.
    check : bool
        Also compute with 2*n_theta nodes and warn (ConvergenceWarning) if the
        largest relative change exceeds ``rtol``.
    rtol : float
        Tolerance for ``check``.
"""


def EH(kind, x, y, z, *, wavelength, w0, jones, eps_r=1.0, mu_r=1.0, P=1.0,
       theta_max=None, NA_stop=None, charge=0, time_convention="-i", n_theta=200,
       check=False, rtol=1e-6):
    """Electric field E [V/m] and magnetic field H [A/m], each of shape (3, *shape)."""
    res = _run(kind, x, y, z, ("E", "H"), wavelength=wavelength, w0=w0, jones=jones,
               eps_r=eps_r, mu_r=mu_r, P=P, theta_max=theta_max, NA_stop=NA_stop,
               charge=charge, time_convention=time_convention, n_theta=n_theta,
               check=check, rtol=rtol)
    return res["E"], res["H"]


def E(kind, x, y, z, *, wavelength, w0, jones, eps_r=1.0, mu_r=1.0, P=1.0,
      theta_max=None, NA_stop=None, charge=0, time_convention="-i", n_theta=200,
      check=False, rtol=1e-6):
    """Electric field E [V/m] of shape (3, *shape). Same arguments as :func:`EH`."""
    return _run(kind, x, y, z, ("E",), wavelength=wavelength, w0=w0, jones=jones,
                eps_r=eps_r, mu_r=mu_r, P=P, theta_max=theta_max, NA_stop=NA_stop,
                charge=charge, time_convention=time_convention, n_theta=n_theta,
                check=check, rtol=rtol)["E"]


def H(kind, x, y, z, *, wavelength, w0, jones, eps_r=1.0, mu_r=1.0, P=1.0,
      theta_max=None, NA_stop=None, charge=0, time_convention="-i", n_theta=200,
      check=False, rtol=1e-6):
    """Magnetic field H [A/m] of shape (3, *shape). Same arguments as :func:`EH`."""
    return _run(kind, x, y, z, ("H",), wavelength=wavelength, w0=w0, jones=jones,
                eps_r=eps_r, mu_r=mu_r, P=P, theta_max=theta_max, NA_stop=NA_stop,
                charge=charge, time_convention=time_convention, n_theta=n_theta,
                check=check, rtol=rtol)["H"]


EH.__doc__ += _DOC
E.__doc__ += _DOC
H.__doc__ += _DOC


def convergence(kind, x, y, z, **kwargs):
    """Largest relative change of E and H when n_theta is doubled.

    Same arguments as :func:`EH` (``check`` and ``rtol`` are ignored). A value
    well below the accuracy you need means that ``n_theta`` is large enough at
    these points.
    """
    kwargs.pop("check", None)
    kwargs.pop("rtol", None)
    n_theta = kwargs.pop("n_theta", 200)
    r1 = _run(kind, x, y, z, ("E", "H"), n_theta=n_theta, **kwargs)
    r2 = _run(kind, x, y, z, ("E", "H"), n_theta=2 * n_theta, **kwargs)
    return max(_rel_change(r1[f], r2[f]) for f in ("E", "H"))


def ott_w0(NA, wavelength, eps_r=1.0, mu_r=1.0, scaling="tan"):
    """Paraxial waist w0 [m] of the OTT beam with the numerical aperture ``NA``.

    In OTT the NA sets the angle theta_b = arcsin(NA/n) where the far-field
    amplitude falls to 1/e; it is not a stop. Then k w0 = 2/tan(theta_b) for
    ``scaling="tan"`` (kind "ott_tan") and k w0 = 2n/NA for ``scaling="sin"``
    (kind "ott_sin"), white paper Table I.
    """
    NA = _real_positive("NA", NA)
    n = np.sqrt(_real_positive("eps_r", eps_r) * _real_positive("mu_r", mu_r))
    if NA >= n:
        raise ValueError(f"NA = {NA} must be below the refractive index n = {n}")
    k = 2 * np.pi * n / _real_positive("wavelength", wavelength)
    theta_b = np.arcsin(NA / n)
    if scaling == "tan":
        return 2 / (k * np.tan(theta_b))
    if scaling == "sin":
        return 2 / (k * np.sin(theta_b))
    raise ValueError('scaling must be "tan" or "sin"')
