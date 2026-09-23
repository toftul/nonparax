"""Figure: focal-plane field components of the three constructions, k w0 = 3.

All fields are normalized to the same power.  I0 is the on-axis intensity of
the paraxial beam with that power, so the paraxial |E_x|^2 curve peaks at 1.
Panels (z = 0):
  A  eps |E_x|^2 along x               (three constructions + paraxial)
  B  eps |E_z|^2 along x               (three constructions + Gauss-law estimate)
  C  eps |E_y|^2 along the diagonal    (lens constructions; zero for the spectrum one)
Writes ../theory/fig/focal_raw.{png,pdf}.
"""
from pathlib import Path

import numpy as np

import beams
from figure_style import figure, letter, save_raw, use_paper_style

HERE = Path(__file__).resolve().parent
use_paper_style(HERE.parent)

KW0 = 3.0
w0 = KW0                                            # k = 1
u = np.linspace(0, 2.5, 251)                        # distance in units of w0
I0 = beams.paraxial_peak(KW0)

fig, axes = figure('double', 1, 3)
A, B, C = axes
for name in ('spectrum', 'aplanatic', 'thinlens'):
    norm = beams.power(name, KW0)*I0
    F_x, _ = beams.fields(name, KW0, u*w0, 0.0, 0.0)          # along x
    F_d, _ = beams.fields(name, KW0, u*w0/np.sqrt(2), u*w0/np.sqrt(2), 0.0)   # diagonal
    kw = dict(color=beams.COLOR[name], label=beams.LABEL[name])
    A.plot(u, abs(F_x[0])**2/norm, **kw)
    B.plot(u, abs(F_x[2])**2/norm, **kw)
    if beams.RULE[name] == 'rw':
        C.plot(u, abs(F_d[1])**2/norm, **kw)

A.plot(u, np.exp(-2*u**2), color='0.5', ls='--', label='paraxial')
B.plot(u, 4*u**2/KW0**2*np.exp(-2*u**2), color='0.5', ls='--',
       label=r'$E_z = -2ix\,E_x/kw_0^2$')

A.set_ylabel(r'Intensity, $\varepsilon|E_x|^2/I_0$')
B.set_ylabel(r'Intensity, $\varepsilon|E_z|^2/I_0$')
C.set_ylabel(r'Intensity, $\varepsilon|E_y|^2/I_0$')
A.set_xlabel(r'Distance along $x$, $x/w_0$')
B.set_xlabel(r'Distance along $x$, $x/w_0$')
C.set_xlabel(r'Diagonal distance, $\rho/w_0$')
for ax, tag in zip(axes, 'ABC'):
    ax.set_xlim(0, u[-1])
    ax.set_ylim(0, None)
    letter(ax, tag, x=-0.12)
handles, labels = [], []
for ax in axes:                                     # one figure legend, unique entries
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in labels:
            handles.append(h); labels.append(l)
fig.legend(handles, labels, loc='outside lower center', ncol=len(labels))

save_raw(fig, HERE.parent/'theory'/'fig', 'focal')
