# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- The white paper is rewritten, shorter. Fig. 2 shows where the Gaussian of
  each beam lies. The section on vortex beams is gone; the package keeps
  `charge`.
- The white paper and `LETTERS` now call OTT `sintheta` beam E and OTT
  `tantheta` beam F, so that each shares its mapping with lens B or C.
  `KINDS` lists `"ott_sin"` before `"ott_tan"`.

## [0.1.0] - 2026-09-24

First release.

### Added

- Exact fields `E`, `H` and `EH` of six non-paraxial Gaussian beams
  (`KINDS`), including the beams that COMSOL and the Optical Tweezers Toolbox
  use.
- Vortex beams with an integer `charge`.
- Polarization set by a Jones vector, amplitude set by the beam power `P`, and
  a convergence check (`convergence`, `ConvergenceWarning`).
- Rotations between the beam frame and the lab frame (`EH_lab`,
  `to_beam_frame`, `to_lab_frame`, `rotation`, `rotation_to`).
- `time_convention` labels `"-iwt"` and `"+iwt"`.
- White paper `theory/nonparax.pdf`.

[Unreleased]: https://github.com/toftul/nonparax/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/toftul/nonparax/releases/tag/v0.1.0
