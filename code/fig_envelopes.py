"""Figure: angular envelopes a(theta) of the three constructions.

Panels: A  k w0 = 8,  B  k w0 = 3.  Dashed grey: paraxial exp[-(k w0 theta/2)^2].
Writes ../theory/fig/envelopes_raw.{png,pdf}.
"""
from pathlib import Path

import numpy as np

import beams
from figure_style import figure, letter, save_raw, use_paper_style

HERE = Path(__file__).resolve().parent
use_paper_style(HERE.parent)

KW0 = (8.0, 3.0)
theta = np.linspace(0, np.pi/2, 721)
deg = np.degrees(theta)

fig, axes = figure('double', 1, 2)
for ax, kw0, tag in zip(axes, KW0, 'AB'):
    ax.plot(deg, np.exp(-(kw0*theta/2)**2), color='0.5', ls='--', label='paraxial')
    for name in ('spectrum', 'aplanatic', 'thinlens'):
        a = beams.envelope(name, kw0, theta)
        ax.plot(deg, a, color=beams.COLOR[name], label=beams.LABEL[name])
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 30, 60, 90])
    ax.set_xlabel(r'Polar angle of the plane wave, $\theta$ (deg)')
    ax.set_title(rf'$kw_0 = {kw0:g}$')
    letter(ax, tag)
axes[0].set_ylabel(r'Angular envelope, $a(\theta)$')
axes[0].legend(loc='upper right')

save_raw(fig, HERE.parent/'theory'/'fig', 'envelopes')
