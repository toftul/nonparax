"""Figure: electric and magnetic energy densities in the focal plane of the
aplanatic construction, k w0 = 3, z = 0, normalized to the paraxial on-axis
intensity I0 at the same power.

Panels, one colour scale:
  A  eps |E|^2 / I0      B  mu |H|^2 / I0      C  (eps |E|^2 + mu |H|^2) / 2 I0
White dotted line: half of the panel maximum.
Writes ../theory/fig/energy_raw.{png,pdf}.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import beams
from figure_style import edges, figure, letter, save_raw, use_paper_style

HERE = Path(__file__).resolve().parent
use_paper_style(HERE.parent)
plt.rcParams['savefig.bbox'] = None                 # colour bar would be clipped otherwise

KW0, NAME = 3.0, 'aplanatic'
w0 = KW0                                            # k = 1
u = np.linspace(-2, 2, 161)                         # x/w0 and y/w0
X, Y = np.meshgrid(u*w0, u*w0)
F, G = beams.fields(NAME, KW0, X, Y, 0.0)
norm = beams.power(NAME, KW0)*beams.paraxial_peak(KW0)
wE = np.sum(abs(F)**2, axis=0)/norm
wH = np.sum(abs(G)**2, axis=0)/norm
maps = [(wE, r'$\varepsilon|\mathbf{E}|^2/I_0$'),
        (wH, r'$\mu|\mathbf{H}|^2/I_0$'),
        (0.5*(wE + wH), r'$(\varepsilon|\mathbf{E}|^2+\mu|\mathbf{H}|^2)/2I_0$')]
vmax = max(m.max() for m, _ in maps)

fig, axes = figure('double', 1, 3, height=0.34*6.2)
for ax, (m, title), tag in zip(axes, maps, 'ABC'):
    mesh = ax.pcolormesh(edges(u), edges(u), m, cmap='inferno', vmin=0, vmax=vmax,
                         shading='flat', rasterized=True)
    ax.contour(u, u, m, levels=[0.5*m.max()], colors='w', linestyles=':', linewidths=0.6)
    ax.set_aspect('equal')
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_xlabel(r'Distance along $x$, $x/w_0$')
    ax.set_title(title)
    letter(ax, tag, x=-0.2)
axes[0].set_ylabel(r'Distance along $y$, $y/w_0$')
for ax in axes[1:]:
    ax.set_yticklabels([])
bar = fig.colorbar(mesh, ax=axes, location='right', shrink=0.8, pad=0.02)
bar.outline.set_linewidth(0.5)
bar.set_label(r'Energy density / $I_0$')

save_raw(fig, HERE.parent/'theory'/'fig', 'energy')
