"""Fig. 3 (baseline test), Fig. S2 and Fig. S3 (1983 drought test) from results/ written by ca_pipeline.py.
Run: python code/make_figures.py   (needs matplotlib and basemap: pip install basemap basemap-data)"""
import os, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.basemap import Basemap
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE) if os.path.basename(HERE) == 'code' else HERE
R = os.path.join(BASE, 'results'); F = os.path.join(BASE, 'figures'); os.makedirs(F, exist_ok=True)
M19 = ['epic-boku', 'epic-boku-p', 'epic-boku-br', 'epic-boku-pt', 'epic-boku-hg', 'epic-iiasa', 'epic-tamu',
       'gepic', 'gepic-pm', 'gepic-br', 'gepic-pt', 'gepic-p', 'pdssat',
       'cgms-wofost', 'pegasus', 'lpjml', 'lpj-guess', 'orchidee-crop', 'clm-crop']
LAB = {m: m.upper() for m in M19}; LAB['pdssat'] = 'pDSSAT'; LAB['orchidee-crop'] = 'ORCHIDEE-Crop'; LAB['clm-crop'] = 'CLM-Crop'; LAB['lpjml'] = 'LPJmL'
# warm colors for RUE-type (EPIC/GEPIC/pDSSAT), cool for the others
warm = plt.get_cmap('YlOrRd')(np.linspace(0.95, 0.35, 13))
cool = ['#c9a227', '#2a9d8f', '#1f6fb2', '#1b3a5c', '#7b6fd0', '#9b59b6', '#1e8449']
COL = {m: warm[i] for i, m in enumerate(M19[:13])}; COL.update({m: cool[i] for i, m in enumerate(M19[13:])})
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})

def basemap(ax):
    m = Basemap(llcrnrlon=-125, llcrnrlat=24, urcrnrlon=-66, urcrnrlat=50, resolution='l', ax=ax)
    m.drawmapboundary(fill_color='#dfeaf5', zorder=0); m.fillcontinents(color='white', lake_color='#dfeaf5', zorder=1)
    m.drawstates(linewidth=0.3, color='#9aa4b0'); m.drawcountries(linewidth=0.5, color='#333'); m.drawcoastlines(linewidth=0.5, color='#333')
    m.drawparallels(range(25, 50, 5), labels=[1, 0, 0, 0], linewidth=0.2, color='#bbb', fontsize=8)
    m.drawmeridians(range(-120, -60, 15), labels=[0, 0, 0, 1], linewidth=0.2, color='#bbb', fontsize=8)
    return m

def grid(df, col):
    lats = np.arange(24.25, 50, 0.5); lons = np.arange(-124.75, -65.5, 0.5)
    Z = np.full((len(lats), len(lons)), np.nan)
    i = np.round((df.lat.values - 24.25) / 0.5).astype(int); j = np.round((df.lon.values + 124.75) / 0.5).astype(int)
    Z[i, j] = df[col].values
    return lons, lats, Z

def pcol(ax, m, df, col, cmap, norm):
    lons, lats, Z = grid(df, col)
    LO, LA = np.meshgrid(np.append(lons, lons[-1] + 0.5) - 0.25, np.append(lats, lats[-1] + 0.5) - 0.25)
    return m.pcolormesh(LO, LA, np.ma.masked_invalid(Z), cmap=cmap, norm=norm, ax=ax, zorder=3)

def main_fig(tag, out):
    P = pd.read_csv(os.path.join(R, f'predictions_{tag}.csv')); W = pd.read_csv(os.path.join(R, f'weights_lgb_{tag}.csv'))
    T = pd.read_csv(os.path.join(R, f'table_{tag}.csv'), index_col=0); S = pd.read_csv(os.path.join(R, f'single_models_{tag}.csv'), index_col=0).iloc[:, 0]
    rm = lambda c: P.groupby(['lat', 'lon']).apply(lambda g: np.sqrt(np.mean((g[c] - g.obs_d) ** 2))).rename('v').reset_index()
    dom = W.groupby(['lat', 'lon'])[M19].mean().idxmax(axis=1).rename('dom').reset_index(); dom['k'] = dom.dom.map({m: i for i, m in enumerate(M19)})
    fig = plt.figure(figsize=(13, 6.6)); gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.05], hspace=0.45, wspace=0.12)
    # (a) dominant model
    ax = fig.add_subplot(gs[0, 0]); m = basemap(ax)
    pcol(ax, m, dom, 'k', ListedColormap([COL[x] for x in M19]), BoundaryNorm(np.arange(-0.5, 19.5), 19))
    ax.text(0.04, 0.9, 'a', transform=ax.transAxes, fontsize=14)
    # (b) bars
    ax = fig.add_subplot(gs[0, 1])
    names = M19 + ['Best model per grid cell', 'Condition-aware (RF)', 'Condition-aware (LGB)', 'Condition-aware (XGB)']
    short = {'Best model per grid cell': 'Best model per cell', 'Condition-aware (RF)': 'Cond.-aware (RF)', 'Condition-aware (LGB)': 'Cond.-aware (LGB)', 'Condition-aware (XGB)': 'Cond.-aware (XGB)'}
    vals = [S[mm] for mm in M19] + [T.loc[n, 'RMSE'] for n in names[19:]]
    cols = [COL[mm] for mm in M19] + ['white', '#d9dde3', '#9aa3ad', '#4d5560']
    x = np.arange(len(names)); x[19:] += 1
    ax.bar(x, vals, color=cols, edgecolor='#222', linewidth=0.5, width=0.75)
    for xi, v in zip(x, vals): ax.text(xi, v + 0.04, f'{v:.2f}', rotation=90, ha='center', va='bottom', fontsize=7, fontweight='bold')
    med = T.loc['MME median', 'RMSE']; ax.axhline(med, color='k', lw=1.8, zorder=4); ax.axvline(19, color='#555', ls='--', lw=1)
    ax.text(1.01, med, 'MME', transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=12, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([LAB.get(n, short.get(n, n)) for n in names], rotation=45, ha='right', fontsize=7.5)
    ax.set_ylabel('RMSE (t/ha)', fontsize=11); ax.set_ylim(0, max(vals) * 1.25); ax.set_xlim(-0.8, x[-1] + 0.8)
    ax.spines[['top', 'right']].set_visible(False); ax.text(0.03, 0.9, 'b', transform=ax.transAxes, fontsize=14)
    # (c, d) rmse maps
    cmap = plt.get_cmap('YlOrRd'); norm = plt.Normalize(0, 4)
    for k, (col, lab) in enumerate([('MME mean', 'c'), ('Condition-aware (LGB)', 'd')]):
        ax = fig.add_subplot(gs[1, k]); m = basemap(ax); pc = pcol(ax, m, rm(col), 'v', cmap, norm)
        ax.text(0.04, 0.9, lab, transform=ax.transAxes, fontsize=14)
    cb = fig.colorbar(pc, ax=ax, fraction=0.035, pad=0.02, ticks=[0, 1, 2, 3, 4]); cb.ax.set_yticklabels(['0.0', '1.0', '2.0', '3.0', '≥4.0']); cb.set_label('RMSE (t/ha)', fontsize=10)
    fig.savefig(os.path.join(F, out + '.png'), dpi=300, bbox_inches='tight'); fig.savefig(os.path.join(F, out + '.tif'), dpi=300, bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
    print(out, 'dominant counts:', dom.dom.value_counts().head(6).to_dict())

def weight_maps(tag, out):
    W = pd.read_csv(os.path.join(R, f'weights_lgb_{tag}.csv')); Wm = W.groupby(['lat', 'lon'])[M19].mean().reset_index()
    fig, axs = plt.subplots(5, 4, figsize=(14, 10.2)); axs = axs.ravel()
    cmap = plt.get_cmap('viridis'); norm = plt.Normalize(0, 0.5)
    for i, mm in enumerate(M19):
        ax = axs[i]; m = basemap(ax); pc = pcol(ax, m, Wm, mm, cmap, norm)
        ax.set_title(f'({chr(97 + i)}) {LAB[mm]}  (mean {Wm[mm].mean():.2f}, max {Wm[mm].max():.2f})', fontsize=8.5, loc='left')
    axs[-1].axis('off')
    cb = fig.colorbar(pc, ax=axs[-1], fraction=0.6, pad=0.05, orientation='horizontal'); cb.set_label('Mean weight (capped at 0.5)')
    fig.tight_layout(); fig.savefig(os.path.join(F, out + '.png'), dpi=250, bbox_inches='tight'); fig.savefig(os.path.join(F, out + '.tif'), dpi=250, bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})

if __name__ == '__main__':
    main_fig('baseline', 'Fig3')
    if os.path.exists(os.path.join(R, 'predictions_d1983.csv')):
        main_fig('d1983', 'FigS2'); weight_maps('d1983', 'FigS3'); main_fig('d1988', 'Fig_1988_test_not_in_paper')
