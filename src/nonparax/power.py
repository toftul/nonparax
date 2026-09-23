"""Amplitude from the beam power, white paper Eq. (power)."""

import numpy as np


def amplitude_from_power(P, k, eps, mu, a, sin_theta, weights):
    """Plane-wave amplitude A (units of sqrt(eps) E) for the power P in watts.

    P = 4 pi^3 |A|^2 / (k^2 sqrt(eps mu)) * int_0^theta_max |a|^2 sin(theta) dtheta.
    ``eps`` and ``mu`` are absolute. The phase of A is set to zero.
    """
    integral = np.sum(weights * np.abs(a) ** 2 * sin_theta)
    if not integral > 0:
        raise ValueError("the envelope a(theta) carries no power on 0 < theta < theta_max")
    return np.sqrt(P * k**2 * np.sqrt(eps * mu) / (4 * np.pi**3 * integral))
