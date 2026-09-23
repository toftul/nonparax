"""Anchors for beams.py.  Each test is a physics statement that must hold
without the code; a failure reads as a false statement.

Run:  cd code && python -m pytest -q
"""
import numpy as np
import pytest
from scipy.integrate import quad

import beams

CONSTRUCTIONS = ('spectrum', 'aplanatic', 'thinlens')


# ---------------------------------------------------------------- envelopes

@pytest.mark.parametrize('name, g, dg', [
    ('aplanatic', np.sin, np.cos),
    ('thinlens', np.tan, lambda t: 1/np.cos(t)**2),
])
def test_apodization_conserves_power_between_pupil_and_reference_sphere(name, g, dg):
    """Power through the pupil zone r<r1 equals power through the sphere zone
    theta<theta1 for the mapping r = f g(theta), for every theta1."""
    kw0, f = 3.0, 1.0
    w_ap = 2*f/kw0                                  # f/w_ap = k w0/2
    for th1 in (0.3, 0.8, 1.2):
        sphere = quad(lambda t: beams.envelope(name, kw0, t)**2*np.sin(t), 0, th1)[0]
        pupil = quad(lambda r: np.exp(-2*r**2/w_ap**2)*r, 0, f*g(th1))[0]/f**2
        assert sphere == pytest.approx(pupil, rel=1e-9)


def test_all_envelopes_start_at_one_and_agree_to_order_theta_squared():
    kw0, th = 8.0, 0.05
    ref = np.exp(-(kw0*th/2)**2)
    for name in CONSTRUCTIONS:
        assert beams.envelope(name, kw0, 0.0) == pytest.approx(1.0)
        assert abs(beams.envelope(name, kw0, th)/ref - 1) < 1.0*th**2


# ------------------------------------------------------------- focal fields

def test_quadrature_is_converged_at_the_default_node_count():
    x = np.linspace(0, 3, 7)
    for name in CONSTRUCTIONS:
        F1, G1 = beams.fields(name, 3.0, x, 0.3, 0.2, n=beams.N_NODES//2)
        F2, G2 = beams.fields(name, 3.0, x, 0.3, 0.2)
        assert np.allclose(F1, F2, atol=1e-9*abs(F2).max())
        assert np.allclose(G1, G2, atol=1e-9*abs(G2).max())


@pytest.mark.parametrize('name', CONSTRUCTIONS)
def test_paraxial_limit_gives_gaussian_profile_and_gauss_law_longitudinal_field(name):
    """For k w0 >> 1: |E_x|^2 -> exp(-2 rho^2/w0^2) at the paraxial peak
    intensity, and E_z -> -2 i x/(k w0^2) E_x from div E = 0."""
    kw0 = 40.0
    w0 = kw0                                         # k = 1
    x = w0*np.array([0.0, 0.5, 1/np.sqrt(2), 1.0, 1.5])
    F, G = beams.fields(name, kw0, x, 0.0, 0.0)
    Ix = abs(F[0])**2/beams.power(name, kw0)/beams.paraxial_peak(kw0)
    assert np.allclose(Ix, np.exp(-2*(x/w0)**2), atol=1e-2)       # theta0^2 = 2.5e-3, 4x margin
    ratio = F[2][1:]/F[0][1:]
    assert np.allclose(ratio, -2j*x[1:]/w0**2, rtol=1e-2)
    # magnetic counterpart of the paraxial relation sqrt(mu) H = z x sqrt(eps) E
    assert np.allclose(G[1][:3], F[0][:3], rtol=1e-2)
    assert np.allclose(G[0], 0, atol=1e-2*abs(F[0][0]))


def test_spectrum_construction_is_gaussian_in_the_waist_plane_up_to_the_evanescent_weight():
    kw0 = 6.0
    rho = kw0*np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    F, _ = beams.fields('spectrum', kw0, rho, 0.0, 0.0)
    profile = F[0]/F[0][0]
    assert np.allclose(profile, np.exp(-(rho/kw0)**2), atol=5*np.exp(-(kw0/2)**2))


def _grad_fields(name, kw0, r0, h=1e-3):
    """Central-difference Jacobians dF_i/dx_j and dG_i/dx_j at r0."""
    dF = np.zeros((3, 3), complex)
    dG = np.zeros((3, 3), complex)
    for j in range(3):
        e = np.zeros(3); e[j] = h
        Fp, Gp = beams.fields(name, kw0, *(r0 + e))
        Fm, Gm = beams.fields(name, kw0, *(r0 - e))
        dF[:, j] = (Fp - Fm)/(2*h)
        dG[:, j] = (Gp - Gm)/(2*h)
    return dF, dG


def _curl(J):
    return np.array([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]])


@pytest.mark.parametrize('name', CONSTRUCTIONS)
def test_fields_satisfy_maxwell_equations_at_a_generic_point(name):
    """div F = 0, div G = 0, curl F = i k G, curl G = -i k F  (k = 1)."""
    kw0 = 3.0
    r0 = np.array([0.7, 0.4, 0.5])*kw0
    F, G = beams.fields(name, kw0, *r0)
    dF, dG = _grad_fields(name, kw0, r0)
    scale = abs(F).max()
    assert abs(np.trace(dF)) < 1e-5*scale
    assert abs(np.trace(dG)) < 1e-5*scale
    assert np.allclose(_curl(dF), 1j*G, atol=1e-5*scale)
    assert np.allclose(_curl(dG), -1j*F, atol=1e-5*scale)


def _rotz90(v):
    """Rotation by +90 degrees about z: x -> y."""
    return np.array([-v[1], v[0], v[2]])


def test_lens_rule_magnetic_field_is_the_electric_field_of_the_y_polarized_beam():
    """sqrt(mu) H_x-pol (r) = R sqrt(eps) E_x-pol (R^-1 r), R = rot_z(90 deg),
    for the Richards-Wolf polarization rule."""
    kw0 = 3.0
    for r in (np.array([0.5, 0.2, 0.0]), np.array([-0.3, 0.9, 0.4])):
        r = r*kw0
        _, G = beams.fields('aplanatic', kw0, *r)
        F_rot, _ = beams.fields('aplanatic', kw0, *_rotz90(_rotz90(_rotz90(r))))
        assert np.allclose(G, _rotz90(F_rot), atol=1e-8*abs(G).max())


def test_spectrum_rule_breaks_the_electric_magnetic_symmetry():
    """E_y = 0 exactly, but H_x != 0, for the Gaussian-spectrum construction."""
    kw0 = 3.0
    r = np.array([0.5, 0.5, 0.0])*kw0
    F, G = beams.fields('spectrum', kw0, *r)
    assert abs(F[1]) < 1e-12*abs(F[0])
    assert abs(G[0]) > 1e-2*abs(F[0])


# ------------------------------------------------- numbers quoted in the text

def test_longitudinal_field_ratio_quoted_in_the_manuscript():
    for kw0, expected in ((8.0, 0.110), (3.0, 0.327)):
        x = np.linspace(0, 1.5*kw0, 301)
        F, _ = beams.fields('aplanatic', kw0, x, 0.0, 0.0)
        ratio = abs(F[2]).max()/abs(F[0]).max()
        assert ratio == pytest.approx(expected, abs=2e-3)
        assert beams.gauss_law_ratio(kw0) == pytest.approx(np.sqrt(2/np.e)/kw0)


def test_spectrum_fields_agree_with_the_k_rho_integrals_of_the_manuscript():
    """E_x = 2 pi int dk k Ehat J0 e^{i kz z}, E_z = -2 pi i cos(phi) int dk k^2/kz Ehat J1 e^{i kz z}
    (Eqs. direct_Ex, direct_Ez), evaluated with scipy.quad, k = 1."""
    from scipy.special import j0 as J0, j1 as J1
    kw0 = 3.0
    Ehat = lambda kr: kw0**2/(4*np.pi)*np.exp(-(kr*kw0/2)**2)
    for (x, z) in ((0.8*kw0, 0.0), (0.5*kw0, 0.4*kw0)):
        kz = lambda kr: np.sqrt(1 - kr**2)
        fx = lambda kr: kr*Ehat(kr)*J0(kr*x)*np.exp(1j*kz(kr)*z)
        fz = lambda kr: kr**2/kz(kr)*Ehat(kr)*J1(kr*x)*np.exp(1j*kz(kr)*z)
        cquad = lambda f: quad(lambda k: f(k).real, 0, 1, limit=200)[0] + 1j*quad(lambda k: f(k).imag, 0, 1, limit=200)[0]
        Ex, Ez = 2*np.pi*cquad(fx), -2j*np.pi*cquad(fz)
        F, _ = beams.fields('spectrum', kw0, x, 0.0, z)
        # fields() drops the constant k^2 w0^2/(4 pi) of Ehat: compare the ratio E_z/E_x
        assert F[2]/F[0] == pytest.approx(Ez/Ex, rel=1e-6)
        assert F[0] == pytest.approx(Ex*4*np.pi/kw0**2, rel=1e-6)
