import sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig2_hsc_ensemble import load, knutti, CODE_GROUP
obs, sim = load()
allw = pd.concat([sim["a"], sim["b"]]); w, sig = knutti(allw)
def means(S):
    g = S.T.groupby(lambda m: CODE_GROUP.get(m, m)).mean().mean()
    return {"Mean 1: unweighted mean": S.mean(axis=1), "Mean 2: group average": g, "Mean 3: similarity weighting": (S * w[S.columns]).sum(axis=1)}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True); lim = 26
for ax, key in zip(axes, means(sim["a"]).keys()):
    for p, c, mk, lab in [("a", "#2166ac", "o", "Winter, unheated"), ("b", "#d7301f", "s", "Spring, heated")]:
        o = obs[p].Biom.values; S = sim[p]; M = means(S)[key].values
        for m in S.columns: ax.scatter(o, S[m].values, s=9, color="0.75", alpha=0.6, linewidths=0, zorder=1)
        slope = M.sum() / o.sum(); xx = np.array([0, lim]); ax.plot(xx, slope * xx, color=c, lw=1, ls=":", zorder=2)
        ax.scatter(o, M, s=28, color=c, marker=mk, zorder=4, label="%s (y = %.2fx)" % (lab, slope))
    ax.plot([0, lim], [0, lim], "--", color="0.4", lw=0.8, zorder=1)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal"); ax.set_title(key, fontsize=8)
    ax.set_xlabel("Observed biomass (t/ha)"); ax.legend(fontsize=6, frameon=False, loc="upper left")
axes[0].set_ylabel("Simulated biomass (t/ha)")
fig.tight_layout(); fig.savefig("FigS1.png", dpi=400)
