"""nonparax: exact electromagnetic fields of non-paraxial Gaussian beams.

Theory: white paper ``theory/nonparax.tex`` in
https://github.com/toftul/nonparaxial-gaussian

>>> import nonparax
>>> E, H = nonparax.EH("aplanatic", x, y, z, wavelength=1064e-9, w0=0.6e-6, jones=(1, 0))
"""

from .beams import KINDS, LETTERS
from .core import EH, E, H, ConvergenceWarning, convergence, ott_w0
from .frames import EH_lab, rotation, rotation_to, to_beam_frame, to_lab_frame

__version__ = "0.1.0"

__all__ = [
    "EH", "E", "H", "EH_lab", "convergence", "ott_w0",
    "to_beam_frame", "to_lab_frame", "rotation", "rotation_to",
    "KINDS", "LETTERS", "ConvergenceWarning",
]
