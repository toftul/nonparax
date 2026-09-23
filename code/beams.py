"""Focal fields of three finite-angle Gaussian beams (Debye-Wolf form).

Conventions (everything below depends on these)
------------------------------------------------
* Time dependence exp(-i omega t).  Wavenumber k = 1: all lengths are in
  units of 1/k, so the paraxial waist w0 enters only through k w0.
* Gaussian units in the symmetric form
      F = sqrt(eps) E,      G = sqrt(mu) H,
  so that for every plane wave  G = k_hat x F  (Faraday with k = n omega/c).
* Angular spectrum per solid angle (the field on the reference sphere):
      F(r) = A  int dOmega  a(theta) p(theta, phi)           e^{i k.r},
      G(r) = A  int dOmega  a(theta) [k_hat x p(theta, phi)] e^{i k.r},
  theta < thmax <= pi/2, a(0) = 1, p(0, phi) = x_hat.  The constant A is
  set to 1.  The Fourier amplitude per d^2k_rho is F_inf/(k^2 cos theta).
* Constructions, all labelled by the same paraxial waist w0
  (for the lens ones f/w_ap = k w0/2):
      'spectrum'  : a = cos theta        exp[-(k w0 sin theta/2)^2],
                    p = x_hat - tan theta cos phi z_hat      (E_y = 0 rule)
      'aplanatic' : a = cos^{1/2} theta  exp[-(k w0 sin theta/2)^2],
                    p = e_hat  (x_hat rotated in the meridional plane)
      'thinlens'  : a = cos^{-3/2} theta exp[-(k w0 tan theta/2)^2],
                    p = e_hat
  e_hat = (1 - (1 - cos)cos^2 phi, -(1 - cos) sin phi cos phi, -sin cos phi),
  k_hat x e_hat = (-(1 - cos) sin phi cos phi, 1 - (1 - cos) sin^2 phi, -sin sin phi).
* Power P = int dOmega a^2 <|p|^2>_phi  (far-field flux, same constant A).
  The paraxial beam with the same P has on-axis intensity
  |F_x(0)|^2/P = 8 pi/(k w0)^2  =: paraxial_peak(k w0).

Azimuthal integrals used (rho, varphi = polar coordinates of r):
  int_0^{2pi} dphi e^{i n phi} e^{i rho sin theta cos(phi - varphi)}
      = 2 pi i^n J_n(rho sin theta) e^{i n varphi}.
"""
import numpy as np
from scipy.special import j0, j1, jv

N_NODES = 1200                      # Gauss-Legendre nodes in theta; test_quadrature checks it
RULE = {'spectrum': 'xz', 'aplanatic': 'rw', 'thinlens': 'rw'}
LABEL = {'spectrum': 'Gaussian spectrum', 'aplanatic': 'aplanatic objective',
         'thinlens': 'thin lens'}
COLOR = {'spectrum': '#009E73', 'aplanatic': '#0072B2', 'thinlens': '#D55E00'}


def envelope(name, kw0, theta):
    """Angular envelope a(theta) of the field on the reference sphere, a(0) = 1."""
    th = np.asarray(theta, float)
    c, s = np.cos(th), np.sin(th)
    if name == 'spectrum':
        return c*np.exp(-(kw0*s/2)**2)
    if name == 'aplanatic':
        return np.sqrt(c)*np.exp(-(kw0*s/2)**2)
    if name == 'thinlens':
        return c**-1.5*np.exp(-(kw0*np.tan(th)/2)**2)
    raise ValueError(name)


def _nodes(thmax, n):
    xg, wg = np.polynomial.legendre.leggauss(n)
    return 0.5*thmax*(xg + 1), 0.5*thmax*wg


def fields(name, kw0, x, y, z=0.0, thmax=np.pi/2, n=N_NODES):
    """F = sqrt(eps) E and G = sqrt(mu) H at the points (x, y, z), k = 1.

    Returns two complex arrays of shape (3,) + broadcast shape of x, y, z.
    """
    x, y, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float),
                                  np.asarray(z, float))
    shape = x.shape
    x, y, z = x.ravel(), y.ravel(), z.ravel()
    rho, phi = np.hypot(x, y), np.arctan2(y, x)

    th, w = _nodes(thmax, n)
    c, s = np.cos(th), np.sin(th)
    W = w*s*envelope(name, kw0, th)                 # dOmega/dphi times the envelope
    arg = np.outer(rho, s)
    J0, J1, J2 = j0(arg), j1(arg), jv(2, arg)
    ph = np.exp(1j*np.outer(z, c))

    def integ(weight, J):                           # int dtheta W weight J e^{i z cos}
        return (ph*J) @ (W*weight)

    c2, s2, cp, sp = np.cos(2*phi), np.sin(2*phi), np.cos(phi), np.sin(phi)
    if RULE[name] == 'rw':
        I0, I1, I2 = integ(1 + c, J0), integ(s, J1), integ(1 - c, J2)
        Fx, Fy, Fz = np.pi*(I0 + c2*I2), np.pi*s2*I2, -2j*np.pi*cp*I1
        Gx, Gy, Gz = np.pi*s2*I2, np.pi*(I0 - c2*I2), -2j*np.pi*sp*I1
    else:                                           # p = x_hat - tan cos(phi) z_hat
        I0, I1 = integ(np.ones_like(c), J0), integ(s/c, J1)
        Fx, Fy, Fz = 2*np.pi*I0, np.zeros(len(rho), complex), -2j*np.pi*cp*I1
        K0, K1, K2 = integ((1 + c**2)/c, J0), integ(s, J1), integ(s**2/c, J2)
        Gx, Gy, Gz = np.pi*s2*K2, np.pi*(K0 - c2*K2), -2j*np.pi*sp*K1
    F = np.stack([Fx, Fy, Fz]).reshape((3,) + shape)
    G = np.stack([Gx, Gy, Gz]).reshape((3,) + shape)
    return F, G


def power(name, kw0, thmax=np.pi/2, n=N_NODES):
    """Far-field power P = int dOmega a^2 <|p|^2>_phi (same constant as fields)."""
    th, w = _nodes(thmax, n)
    a = envelope(name, kw0, th)
    p2 = 1.0 if RULE[name] == 'rw' else 1 + 0.5*np.tan(th)**2
    return 2*np.pi*np.sum(w*np.sin(th)*a**2*p2)


def paraxial_peak(kw0):
    """|F_x(0)|^2/P of the paraxial Gaussian beam: 8 pi/(k w0)^2."""
    return 8*np.pi/kw0**2


def gauss_law_ratio(kw0):
    """max|E_z|/max|E_x| from E_z = -2 i x/(k w0^2) E_x: sqrt(2/e)/(k w0)."""
    return np.sqrt(2/np.e)/kw0
