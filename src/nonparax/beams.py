"""The six beams of the white paper (Table I): envelope a(theta) and rule c(theta).

Every beam is fixed by its envelope a(theta) and its real meridional weight
c(theta), white paper Eqs. (ap) and (rule). All of them use the paraxial waist
w0 through kw0 = k w0, with k the wavenumber in the medium.
"""

import numpy as np

# name -> white-paper letter
LETTERS = {
    "spectrum": "A",
    "aplanatic": "B",
    "thin_lens": "C",
    "comsol": "D",
    "ott_sin": "E",
    "ott_tan": "F",
}

KINDS = tuple(LETTERS)


def _gauss(u, kw0):
    return np.exp(-((kw0 * u / 2) ** 2))


def _spectrum(theta, kw0):
    ct = np.cos(theta)
    return np.sqrt((1 + ct**2) / 2) * _gauss(np.sin(theta), kw0), 1 / ct


def _aplanatic(theta, kw0):
    return np.sqrt(np.cos(theta)) * _gauss(np.sin(theta), kw0), np.ones_like(theta)


def _thin_lens(theta, kw0):
    return np.cos(theta) ** -1.5 * _gauss(np.tan(theta), kw0), np.ones_like(theta)


def _comsol(theta, kw0):
    ct = np.cos(theta)
    return ct * np.sqrt((1 + ct**2) / 2) * _gauss(np.sin(theta), kw0), ct


def _ott_tan(theta, kw0):
    return _gauss(np.tan(theta), kw0), np.ones_like(theta)


def _ott_sin(theta, kw0):
    return _gauss(np.sin(theta), kw0), np.ones_like(theta)


_TABLE = {
    "spectrum": _spectrum,
    "aplanatic": _aplanatic,
    "thin_lens": _thin_lens,
    "comsol": _comsol,
    "ott_tan": _ott_tan,
    "ott_sin": _ott_sin,
}


def envelope_and_rule(kind, theta, kw0):
    """Return a(theta) (complex) and c(theta) (real) on the nodes ``theta``.

    ``kind`` is one of :data:`KINDS` or a pair of callables ``(a, c)`` of theta.
    """
    if isinstance(kind, str):
        if kind not in _TABLE:
            raise ValueError(f"unknown beam kind {kind!r}; use one of {KINDS} or a pair (a, c)")
        a, c = _TABLE[kind](theta, kw0)
    else:
        try:
            a_fun, c_fun = kind
        except (TypeError, ValueError):
            raise ValueError("kind must be a name from KINDS or a pair of callables (a, c)") from None
        if not (callable(a_fun) and callable(c_fun)):
            raise ValueError("a custom kind must be a pair of callables (a, c)")
        a = np.broadcast_to(np.asarray(a_fun(theta)), theta.shape)
        c = np.broadcast_to(np.asarray(c_fun(theta)), theta.shape)
    a = np.asarray(a, dtype=complex)
    c = np.asarray(c)
    if np.iscomplexobj(c):
        if np.any(np.abs(c.imag) > 1e-12 * np.maximum(np.abs(c.real), 1)):
            raise ValueError("c(theta) must be real (white paper, Eq. (rule))")
        c = c.real
    c = np.asarray(c, dtype=float)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(c))):
        raise ValueError("a(theta) and c(theta) must be finite on 0 < theta < theta_max")
    return a, c
