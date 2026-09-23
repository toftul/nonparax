# nonparax

Exact electromagnetic fields of non-paraxial Gaussian beams.

Beyond the paraxial limit, "Gaussian beam" does not name one field. The name
depends on where the Gaussian is: in the angular spectrum, or in the pupil of a
lens. `nonparax` computes six such beams as exact solutions of Maxwell's
equations. It includes the beams that COMSOL and the Optical Tweezers Toolbox
(OTT) use. The theory is in the [white paper](theory/nonparax.pdf).

## Install

```bash
pip install git+https://github.com/toftul/nonparaxial-gaussian
```

The package needs Python 3.10 or newer, NumPy 2 and SciPy.

## Quick start

```python
import numpy as np
import nonparax as npx

x = np.linspace(-1e-6, 1e-6, 201)          # m
E, H = npx.EH("aplanatic", x, 0.0, 0.0,
              wavelength=1064e-9,           # vacuum wavelength, m
              w0=0.5e-6,                    # paraxial waist, m
              jones=(1, 0),                 # x-polarized input
              eps_r=1.33**2,                # water
              P=0.1)                        # beam power, W
Ex, Ey, Ez = E                              # V/m, each of shape (201,)
intensity = np.sum(np.abs(E) ** 2, axis=0)
```

`E` is in V/m and `H` is in A/m. Both have the shape `(3, *shape)`, where
`shape` is the broadcast shape of `x`, `y` and `z`. The beam travels along `+z`,
and its focus is at the origin. `npx.E(...)` and `npx.H(...)` take the same
arguments and return one field.

## The six beams

| `kind`        | white paper | construction                                    | parameters        |
|---------------|-------------|-------------------------------------------------|-------------------|
| `"spectrum"`  | A           | Gaussian angular spectrum, no lens              | `w0`              |
| `"aplanatic"` | B           | Gaussian pupil, aplanatic objective (sine condition) | `w0` (+ stop) |
| `"thin_lens"` | C           | Gaussian pupil, thin lens (`r = f tan θ`)        | `w0` (+ stop)     |
| `"comsol"`    | D           | COMSOL "plane wave expansion"                   | `w0`              |
| `"ott_tan"`   | E           | OTT `BscPmGauss`, `angular_scaling="tantheta"`  | `w0` or NA        |
| `"ott_sin"`   | F           | OTT `BscPmGauss`, `angular_scaling="sintheta"`  | `w0` or NA        |

All six agree in the paraxial limit, `k w0 >> 2`, and differ when the beam is
tightly focused. To compare them, loop over `npx.KINDS`:

```python
for kind in npx.KINDS:
    E = npx.E(kind, x, 0.0, 0.0, wavelength=1064e-9, w0=0.5e-6, jones=(1, 0))
```

In OTT the NA sets the width of the beam, not a stop. Convert it to `w0` first:

```python
w0 = npx.ott_w0(NA=1.2, wavelength=1064e-9, eps_r=1.33**2, scaling="tan")
E, H = npx.EH("ott_tan", x, 0.0, 0.0, wavelength=1064e-9, w0=w0, jones=(1, 1j), eps_r=1.33**2)
```

A custom beam is a pair of functions of θ (rad): the envelope `a(θ)`, which can
be complex, and the real meridional weight `c(θ)` (white paper, Eqs. (ap) and (rule)):

```python
kw0 = 2 * np.pi * 1.33 / 1064e-9 * 0.5e-6
a = lambda t: np.sqrt(np.cos(t)) * np.exp(-(kw0 * np.sin(t) / 2) ** 2)
c = lambda t: np.ones_like(t)
E = npx.E((a, c), x, 0.0, 0.0, wavelength=1064e-9, w0=0.5e-6, jones=(1, 0), eps_r=1.33**2)
```

## Parameters

| argument          | meaning                                                         | default   |
|-------------------|-----------------------------------------------------------------|-----------|
| `wavelength`      | vacuum wavelength, m                                            | required  |
| `w0`              | paraxial waist, m                                               | required  |
| `jones`           | input polarization `(Ex, Ey)`, normalized internally             | required  |
| `eps_r`, `mu_r`   | relative permittivity and permeability (lossless: real, > 0)    | 1, 1      |
| `P`               | beam power, W                                                   | 1         |
| `theta_max`       | stop, as the largest polar angle in rad                          | π/2       |
| `NA_stop`         | stop, as `NA = n sin(theta_max)`; give at most one of the two   | none      |
| `charge`          | integer vortex charge ℓ, the amplitude gains `exp(iℓφ)`          | 0         |
| `time_convention` | `"-i"` for `exp(-iωt)`, `"+j"` for `exp(+jωt)` (COMSOL)          | `"-i"`    |
| `n_theta`         | number of Gauss–Legendre nodes in θ                              | 200       |
| `check`, `rtol`   | warn if doubling `n_theta` changes the fields by more than `rtol` | off, 1e-6 |

## Polarization

`jones=(Ex, Ey)` is any complex pair. For example, `(1, 0)` is x-polarized,
`(1, 1j)` has helicity +1 and `(1, -1j)` has helicity −1. The package splits the
input into the two helicities and adds the two beams (white paper, Eq. (arbitrary)).

A radially polarized beam is the sum of two vortex beams (white paper, Sec. "Vortex beams"):

```python
kw = dict(wavelength=1064e-9, w0=0.5e-6, eps_r=1.33**2)
Ep = npx.E("aplanatic", x, 0.0, 0.0, jones=(1, 1j), charge=-1, **kw)
Em = npx.E("aplanatic", x, 0.0, 0.0, jones=(1, -1j), charge=+1, **kw)
E_radial = (Ep + Em) / np.sqrt(2)
```

## Other focus positions and directions

The core computes a beam along `+z` with the focus at the origin. Use the
frame helpers for anything else. The `jones` vector stays in the beam frame.

```python
R = npx.rotation_to((0, 0, -1))                   # beam travelling along -z
E, H = npx.EH_lab("aplanatic", x, 0.0, 0.0, r0=(0, 0, 1e-6), R=R,
                  wavelength=1064e-9, w0=0.5e-6, jones=(1, 0))

# the same, step by step
xb, yb, zb = npx.to_beam_frame(x, 0.0, 0.0, r0=(0, 0, 1e-6), R=R)
Eb, Hb = npx.EH("aplanatic", xb, yb, zb, wavelength=1064e-9, w0=0.5e-6, jones=(1, 0))
E, H = npx.to_lab_frame(Eb, R), npx.to_lab_frame(Hb, R)
```

`npx.rotation(axis, angle)` gives a rotation about an axis.

## Accuracy

The azimuthal integral is exact (Bessel functions). The integral over θ uses
`n_theta` Gauss–Legendre nodes. Far from the focus, the integrand oscillates,
and more nodes are necessary. Check your points:

```python
npx.convergence("aplanatic", x, 0.0, 0.0, wavelength=1064e-9, w0=0.5e-6, jones=(1, 0))
# largest relative change of E and H when n_theta is doubled
```

`check=True` computes both and gives a `ConvergenceWarning` above `rtol`.

## Conventions

- SI units. The fields are complex amplitudes with `exp(-iωt)`, unless `time_convention="+j"`.
- Helicity basis `e_σ = (x̂ + iσ ŷ)/√2`.
- The medium is homogeneous and lossless.
- Evanescent waves are not included, because the beam comes from far away.

## Tests

```bash
uv sync
uv run pytest
```

The tests compare the fields with a direct 2D quadrature of the angular
spectrum, built from the physical construction of each beam. They also check
Maxwell's equations, the power, the paraxial limit, polarization and frames.

## License

MIT
