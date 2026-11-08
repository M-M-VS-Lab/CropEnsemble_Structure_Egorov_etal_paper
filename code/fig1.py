"""Fig. 1: classification heat map of the 37 models from data/TableS1_scores.csv (Table S1).
Run: python code/fig1.py   -> figures/Fig1.png (and .tif)"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE) if os.path.basename(HERE) == 'code' else HERE
T = pd.read_csv(os.path.join(BASE, 'data', 'TableS1_scores.csv'))
COLS = ['CA', 'CS', '3D', 'PT', 'HS', 'SW', 'PH', 'RD', 'MC', 'AL', 'LR']
NAMES = ['Carbon assimilation', 'Canopy light scaling', '3D plant architecture', 'Phenology temperature response',
         'Reproductive heat stress', 'Soil water flow', 'Plant hydraulics', 'Residue decomposition',
         'Microbial C–N dynamics', 'Carbon allocation', 'Lateral water routing']
M = T[COLS].astype(float).values
cmap = ListedColormap(['#f5f2ea', '#5fb3c8', '#1b3a5c']); norm = BoundaryNorm([-0.25, 0.25, 0.75, 1.25], 3)

fig, ax = plt.subplots(figsize=(11, 14.5))
ax.imshow(M, cmap=cmap, norm=norm, aspect='auto')
ax.set_xticks(np.arange(-0.5, len(COLS)), minor=True); ax.set_yticks(np.arange(-0.5, len(T)), minor=True)
ax.grid(which='minor', color='white', linewidth=3); ax.tick_params(which='minor', length=0)
ax.set_xticks(range(len(COLS))); ax.set_xticklabels(NAMES, rotation=45, ha='right', fontsize=12)
ax.set_yticks(range(len(T))); ax.set_yticklabels(T.Model, fontsize=12)
for s in ax.spines.values(): s.set_visible(False)
# group separators and labels
groups = T.group.values; starts = [i for i in range(len(T)) if i == 0 or groups[i] != groups[i - 1]] + [len(T)]
for i in starts[1:-1]: ax.axhline(i - 0.5, color='black', linewidth=1.5)
for a, b in zip(starts[:-1], starts[1:]):
    ax.plot([len(COLS) - 0.3, len(COLS) - 0.3], [a - 0.3, b - 0.7], color='black', lw=1.2, clip_on=False)
    ax.text(len(COLS) - 0.05, (a + b - 1) / 2, groups[a].replace(' models', '\nmodels').replace(' crop', '\ncrop').replace(' and ', ' and\n'),
            va='center', ha='left', fontsize=12, clip_on=False)
ax.set_xlim(-0.5, len(COLS) - 0.5)
ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(cmap.colors, ['0  simplest form', '0.5  intermediate', '1  most detailed form'])],
          loc='lower center', bbox_to_anchor=(0.5, 1.005), ncol=3, frameon=False, fontsize=12)
plt.tight_layout()
out = os.path.join(BASE, 'figures'); os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, 'Fig1.png'), dpi=200); fig.savefig(os.path.join(out, 'Fig1.tif'), dpi=300, pil_kwargs={'compression': 'tiff_lzw'})
print('written', out)
