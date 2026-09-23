"""Tests of nonparax against independent references (white paper Eqs. in brackets)."""

import warnings

import numpy as np
import pytest
from scipy.constants import c as c0, epsilon_0, mu_0

import nonparax as npx

# medium and beam used in most tests: k = 1.5 per unit length, strongly non-paraxial
WL, EPS_R, MU_R = 2 * np.pi, 2.25, 1.0
N_MED = np.sqrt(EPS_R * MU_R)
K = 2 * np.pi * N_MED / WL
EPS, MU = EPS_R * epsilon_0, MU_R * mu_0
BASE = dict(wavelength=WL, eps_r=EPS_R, mu_r=MU_R, P=0.7)


def e_sigma(s):
    return np.array([1, 1j * s, 0]) / np.sqrt(2)


# ------------------------------------------------------------------ independent reference
def ref_far_field(kind, th, ph, kw0, sigma, charge):
    """sqrt(eps) E_inf, up to a constant, built from each beam's physical construction."""
    ct, st = np.cos(th), np.sin(th)
    G = lambda u: np.exp(-(kw0 * u / 2) ** 2)
    khat = np.stack([st * np.cos(ph), st * np.sin(ph), ct])
    that = np.stack([ct * np.cos(ph), ct * np.sin(ph), -st])
    phat = np.stack([-np.sin(ph), np.cos(ph), 0 * ph])
    rhat = np.stack([np.cos(ph), np.sin(ph), 0 * ph])
    e = e_sigma(sigma)[:, None, None] * np.exp(1j * charge * ph)
    if kind == "spectrum":   # transverse part cos(theta) G e_sigma, E_z from k.E = 0
        Et = ct * G(st) * e
        Ez = -(khat[0] * Et[0] + khat[1] * Et[1]) / ct
        return np.stack([Et[0], Et[1], Ez])
    if kind == "comsol":     # projection of e_sigma normal to k, Eq. (comsol)
        return ct * G(st) * (e - np.sum(e * khat, 0) * khat)
    # lenses and OTT: pupil field e_sigma G(g); radial part -> theta_hat, azimuthal part kept
    g, A = {"aplanatic": (st, np.sqrt(ct)), "thin_lens": (np.tan(th), ct ** -1.5),
            "ott_tan": (np.tan(th), 1.0), "ott_sin": (st, 1.0)}[kind]
    rot = np.sum(e * rhat, 0) * that + np.sum(e * phat, 0) * phat
    return A * G(g) * rot


def ref_fields(kind, pts, kw0, sigma, charge, P, n_th=300, n_ph=256):
    """E, H at pts (3, N) by direct 2D quadrature of Eq. (debye), normalized to power P."""
    t, w = np.polynomial.legendre.leggauss(n_th)
    th = np.pi / 4 * (t + 1)
    wt = np.pi / 4 * w
    ph = (np.arange(n_ph) + 0.5) * 2 * np.pi / n_ph
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    F = ref_far_field(kind, TH, PH, kw0, sigma, charge)
    dOm = (wt * np.sin(th))[:, None] * (2 * np.pi / n_ph)
    # power, Eq. (power) in the form P = 2 pi^2 / (k^2 sqrt(eps mu)) int |sqrt(eps) E_inf|^2 dOmega
    Pref = 2 * np.pi**2 / (K**2 * np.sqrt(EPS * MU)) * np.sum(np.sum(np.abs(F) ** 2, 0) * dOm)
    F = F * np.sqrt(P / Pref)
    khat = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)])
    G = np.cross(khat, F, axis=0)                     # sqrt(mu) H_inf, Eq. (pw)
    phase = np.exp(1j * K * np.einsum("iab,in->nab", khat, pts))
    E = np.einsum("iab,nab->in", F * dOm, phase) / np.sqrt(EPS)
    H = np.einsum("iab,nab->in", G * dOm, phase) / np.sqrt(MU)
    return E, H


PTS = np.array([[0.0, 0.0, 0.0], [1.3, -0.7, 0.4], [-2.0, 1.5, -0.8], [0.4, 3.1, 1.7]]).T


# ------------------------------------------------------------------ 1. Bessel form vs 2D quadrature
@pytest.mark.parametrize("kind", npx.KINDS)
@pytest.mark.parametrize("sigma", [1, -1])
@pytest.mark.parametrize("charge", [0, 1, -2])
def test_bessel_form_matches_2d_quadrature(kind, sigma, charge):
    kw0 = 2.5
    E, H = npx.EH(kind, *PTS, w0=kw0 / K, jones=e_sigma(sigma)[:2], charge=charge, **BASE)
    Er, Hr = ref_fields(kind, PTS, kw0, sigma, charge, BASE["P"])
    assert np.max(np.abs(E - Er)) < 1e-7 * np.max(np.abs(Er))
    assert np.max(np.abs(H - Hr)) < 1e-7 * np.max(np.abs(Hr))


# ------------------------------------------------------------------ 2. Maxwell's equations
@pytest.mark.parametrize("kind", ["spectrum", "thin_lens", "comsol"])
def test_maxwell_equations(kind):
    kw = dict(w0=2.0 / K, jones=(1, 0.3 + 0.5j), charge=1, **BASE)
    omega = 2 * np.pi * c0 / WL
    h = WL / 400
    r = np.array([0.6, -0.9, 0.5])
    d = {}
    for i in range(3):
        for s in (-2, -1, 1, 2):
            p = r.copy(); p[i] += s * h
            d[i, s] = npx.EH(kind, *p, **kw)
    def deriv(field, i):   # 4th-order central difference, field index 0 = E, 1 = H
        return (-d[i, 2][field] + 8 * d[i, 1][field] - 8 * d[i, -1][field] + d[i, -2][field]) / (12 * h)
    dE = [deriv(0, i) for i in range(3)]
    dH = [deriv(1, i) for i in range(3)]
    E0, H0 = npx.EH(kind, *r, **kw)
    curlE = np.array([dE[1][2] - dE[2][1], dE[2][0] - dE[0][2], dE[0][1] - dE[1][0]])
    curlH = np.array([dH[1][2] - dH[2][1], dH[2][0] - dH[0][2], dH[0][1] - dH[1][0]])
    divE = dE[0][0] + dE[1][1] + dE[2][2]
    scale_E = K * np.max(np.abs(E0))
    assert abs(divE) < 1e-6 * scale_E
    assert np.max(np.abs(curlE - 1j * omega * MU * H0)) < 1e-6 * scale_E
    assert np.max(np.abs(curlH + 1j * omega * EPS * E0)) < 1e-6 * K * np.max(np.abs(H0))


# ------------------------------------------------------------------ 3. power
@pytest.mark.parametrize("kind", ["aplanatic", "spectrum"])
def test_power_flux_equals_P(kind):
    w0 = 6.0 / K
    rho = np.linspace(0, 5 * w0, 801)
    vphi = (np.arange(48) + 0.5) * 2 * np.pi / 48
    R, V = np.meshgrid(rho, vphi, indexing="ij")
    E, H = npx.EH(kind, R * np.cos(V), R * np.sin(V), 0.0, w0=w0, jones=(1, 0), **BASE)
    Sz = 0.5 * np.real(E[0] * np.conj(H[1]) - E[1] * np.conj(H[0]))
    flux = np.trapezoid(np.mean(Sz, axis=1) * 2 * np.pi * rho, rho)
    assert abs(flux / BASE["P"] - 1) < 1e-4


# ------------------------------------------------------------------ 4. paraxial limit
def test_paraxial_limit():
    kw0 = 80.0
    w0 = kw0 / K
    z0 = K * w0**2 / 2
    x, y, z = 0.4 * w0, -0.3 * w0, 0.5 * z0
    for sigma in (1, -1):
        E = npx.E("aplanatic", x, y, z, w0=w0, jones=e_sigma(sigma)[:2], n_theta=400, **BASE)
        eta = np.sqrt(MU / EPS)
        E0 = np.sqrt(4 * eta * BASE["P"] / (np.pi * w0**2))       # focal amplitude A_0
        wz = w0 * np.sqrt(1 + (z / z0) ** 2)
        Rinv = z / (z**2 + z0**2)
        rho2 = x**2 + y**2
        Epar = E0 * e_sigma(sigma) * w0 / wz * np.exp(-rho2 / wz**2) * np.exp(
            1j * (K * z - np.arctan(z / z0) + K * rho2 * Rinv / 2))  # Eq. (paraxial)
        assert np.max(np.abs(E[:2] - Epar[:2])) < 5 / kw0**2 * abs(E0)


# ------------------------------------------------------------------ 5. c = 1 beams are helicity eigenstates
@pytest.mark.parametrize("kind", ["aplanatic", "thin_lens", "ott_tan", "ott_sin"])
@pytest.mark.parametrize("sigma", [1, -1])
def test_c1_helicity_relation(kind, sigma):
    E, H = npx.EH(kind, *PTS, w0=2.0 / K, jones=e_sigma(sigma)[:2], charge=2, **BASE)
    lhs = np.sqrt(MU) * H
    rhs = -1j * sigma * np.sqrt(EPS) * E
    assert np.max(np.abs(lhs - rhs)) < 1e-12 * np.max(np.abs(rhs))


# ------------------------------------------------------------------ 6. linearity in the Jones vector
@pytest.mark.parametrize("kind", ["spectrum", "comsol"])
def test_linear_input_is_sum_of_helicities(kind):
    kw = dict(w0=2.0 / K, **BASE)
    Ex, Hx = npx.EH(kind, *PTS, jones=(1, 0), **kw)
    Ep, Hp = npx.EH(kind, *PTS, jones=(1, 1j), **kw)
    Em, Hm = npx.EH(kind, *PTS, jones=(1, -1j), **kw)
    assert np.allclose(Ex, (Ep + Em) / np.sqrt(2), rtol=0, atol=1e-12 * np.max(np.abs(Ex)))
    assert np.allclose(Hx, (Hp + Hm) / np.sqrt(2), rtol=0, atol=1e-12 * np.max(np.abs(Hx)))


# ------------------------------------------------------------------ 7. time convention
def test_time_convention_conjugates():
    kw = dict(w0=2.0 / K, jones=(1, 0.2j), **BASE)
    E1, H1 = npx.EH("thin_lens", *PTS, **kw)
    E2, H2 = npx.EH("thin_lens", *PTS, time_convention="+j", **kw)
    assert np.array_equal(E2, np.conj(E1)) and np.array_equal(H2, np.conj(H1))


# ------------------------------------------------------------------ 8. frames
def test_rotation_pi_about_x_reverses_propagation():
    kw = dict(w0=3.0 / K, jones=(1, 0), **BASE)
    R = npx.rotation((1, 0, 0), np.pi)
    assert np.allclose(npx.rotation_to((0, 0, -1)), R)
    z = np.linspace(-2, 2, 5)
    E, H = npx.EH_lab("aplanatic", 0.0, 0.0, z, R=R, **kw)
    Sz = 0.5 * np.real(E[0] * np.conj(H[1]) - E[1] * np.conj(H[0]))
    assert np.all(Sz < 0)
    E0, H0 = npx.EH("aplanatic", 0.0, 0.0, z, **kw)
    Sz0 = 0.5 * np.real(E0[0] * np.conj(H0[1]) - E0[1] * np.conj(H0[0]))
    assert np.allclose(Sz, -Sz0[::-1])


def test_shift_moves_focus():
    kw = dict(w0=2.0 / K, jones=(1, 1j), **BASE)
    r0 = np.array([0.7, -0.4, 0.3])
    x = r0[0] + np.linspace(-1, 1, 41)
    E, H = npx.EH_lab("aplanatic", x, r0[1], r0[2], r0=r0, **kw)
    I = np.sum(np.abs(E) ** 2, 0)
    assert np.argmax(I) == 20
    Eb, Hb = npx.EH("aplanatic", x - r0[0], 0.0, 0.0, **kw)
    assert np.array_equal(E, Eb) and np.array_equal(H, Hb)


def test_rotation_to_and_frames_roundtrip():
    u = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    R = npx.rotation_to(u)
    assert np.allclose(R @ [0, 0, 1], u)
    xb, yb, zb = npx.to_beam_frame(*u, r0=(0, 0, 0), R=R)
    assert np.allclose([xb, yb, zb], [0, 0, 1])
    assert np.allclose(npx.to_lab_frame(np.array([0, 0, 1.0]), R), u)


# ------------------------------------------------------------------ 9. input checks
@pytest.mark.parametrize("bad", [
    dict(eps_r=2.0 + 0.1j), dict(eps_r=-1.0), dict(mu_r=0.0), dict(jones=(0, 0)),
    dict(theta_max=1.0, NA_stop=0.5), dict(NA_stop=2.0), dict(theta_max=2.0),
    dict(charge=0.5), dict(time_convention="+i"), dict(n_theta=0), dict(P=-1.0),
])
def test_bad_inputs_raise(bad):
    kw = dict(w0=1.0, jones=(1, 0), **BASE)
    kw.update(bad)
    with pytest.raises(ValueError):
        npx.EH("aplanatic", 0.0, 0.0, 0.0, **kw)


def test_bad_kind_raises():
    kw = dict(w0=1.0, jones=(1, 0), **BASE)
    with pytest.raises(ValueError):
        npx.EH("gaussian", 0.0, 0.0, 0.0, **kw)
    with pytest.raises(ValueError):
        npx.EH((lambda t: 1 + 0 * t, lambda t: 1j + 0 * t), 0.0, 0.0, 0.0, **kw)


# ------------------------------------------------------------------ helpers
def test_ott_w0():
    NA, n = 1.2, 1.33
    th_b = np.arcsin(NA / n)
    k = 2 * np.pi * n / 1064e-9
    assert np.isclose(npx.ott_w0(NA, 1064e-9, eps_r=n**2), 2 / (k * np.tan(th_b)))
    assert np.isclose(npx.ott_w0(NA, 1064e-9, eps_r=n**2, scaling="sin"), 2 * n / (k * NA))


def test_custom_kind_equals_named_kind():
    kw = dict(w0=2.0 / K, jones=(1, 0.4j), charge=-1, **BASE)
    kw0 = 2.0
    a = lambda t: np.sqrt(np.cos(t)) * np.exp(-(kw0 * np.sin(t) / 2) ** 2)
    c = lambda t: np.ones_like(t)
    E1, H1 = npx.EH("aplanatic", *PTS, **kw)
    E2, H2 = npx.EH((a, c), *PTS, **kw)
    assert np.allclose(E1, E2, rtol=1e-12) and np.allclose(H1, H2, rtol=1e-12)


def test_convergence_check():
    kw = dict(w0=2.0 / K, jones=(1, 0), **BASE)
    far = (np.array([40.0]), 0.0, 25.0)
    # sqrt(cos) endpoint at pi/2: algebraic, not exponential, convergence; still below rtol
    assert npx.convergence("aplanatic", *PTS, **kw) < 1e-6
    with pytest.warns(npx.ConvergenceWarning):
        npx.EH("aplanatic", *far, n_theta=8, check=True, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        npx.EH("aplanatic", *PTS, check=True, **kw)


def test_output_shape():
    kw = dict(w0=2.0 / K, jones=(1, 0), **BASE)
    x = np.linspace(-1, 1, 4)[:, None]
    y = np.linspace(-1, 1, 3)[None, :]
    E, H = npx.EH("aplanatic", x, y, 0.5, **kw)
    assert E.shape == H.shape == (3, 4, 3)
