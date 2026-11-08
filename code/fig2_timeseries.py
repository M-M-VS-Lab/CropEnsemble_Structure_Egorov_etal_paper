import sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig2_hsc_ensemble import ROOT, TRT, NAMES, CLASS, styles, LINES
ROOT = sys.argv[1] if len(sys.argv) > 1 else ROOT
m = pd.read_csv(os.path.join(ROOT, "4-Experimental data", "HSC_AgMIP_daily_measurments_ave_sd.txt"), sep="\t", encoding="latin1")
s = pd.read_csv(os.path.join(ROOT, "5-Model output", "HSC_AgMIP_daily_simulations.txt"), sep="\t", encoding="latin1", low_memory=False)
s = s[s.Version == "Original"]
st = styles(); CLR = {"RUE": "#d7301f", "light": "#1f5fbf", "FvCB": "#0a8f3c", "water": "#00a6a6"}
THIN = {"RUE": "#f9a870", "light": "#8fbce6", "FvCB": "#7fd49a", "water": "#7fdede"}
import matplotlib.patheffects as pe
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
for ax, p, title in zip(axes, ["a", "b"], ["Winter sowing, no heating (sown 1 Dec 2008)", "Spring sowing, heated (sown 13 Mar 2007)"]):
    pno, trt = TRT[p]
    o = m[(m.PNO == pno) & (m.Trt == trt) & m.Biom.notna()]
    sub = s[(s.PNO == pno) & (s.Trt == trt)]
    dmax = int(o.DAS.max()) + 3
    W = sub.pivot_table(index="DAS", columns="Model", values="Biom").reindex(range(0, dmax + 1)).ffill()  # hold final (maturity) biomass constant
    for mod in W.columns:
        ax.plot(W.index, W[mod], color=THIN[CLASS[mod]], lw=0.7, alpha=0.9, zorder=2 + (CLASS[mod] == "FvCB"))
    for k, col, lab in LINES:
        cols = list(W.columns) if k == "all" else [x for x in W.columns if CLASS[x] == k]
        mm = W[cols].mean(axis=1); ax.plot(W.index, mm, color=(col if k == "all" else CLR[k]), lw=2.4 if k == "all" else 2.0, zorder=5, path_effects=[pe.Stroke(linewidth=3.6, foreground="white"), pe.Normal()])
    ax.errorbar(o.DAS, o.Biom, yerr=o["Biom.sd"], fmt="o", color="black", mfc="white", ms=6, mew=1.3, capsize=3, zorder=6, label="Observed")
    ax.set_xlim(0, dmax); ax.set_ylim(0, 26); ax.set_xlabel("Days after sowing"); ax.grid(color="0.92", lw=0.5)
    ax.set_title(title, fontsize=8); ax.text(0.02, 0.96, p, transform=ax.transAxes, fontweight="bold", fontsize=11, va="top")
axes[0].set_ylabel("Aboveground biomass (t/ha)")
h = [Line2D([], [], marker="o", color="black", mfc="white", ls="", label="Observed (mean ± SD)")] + \
    [Line2D([], [], color=(c if k == "all" else CLR[k]), lw=2.4 if k == "all" else 2.0, label="Mean of " + ("all 29 models" if k == "all" else {"RUE": "RUE-based models", "light": "light-response models", "FvCB": "FvCB-type models"}[k])) for k, c, l in LINES] + \
    [Line2D([], [], color=THIN[k], lw=1.2, label=l + " (individual models)") for k, l in [("RUE", "RUE-based"), ("light", "Light-response"), ("FvCB", "FvCB-type"), ("water", "Water-driven")]]
fig.legend(handles=h, loc="lower center", ncol=3, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 0))
fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.30, wspace=0.15)
fig.savefig("Fig2_ts.png", dpi=400); fig.savefig("Fig2_ts.tif", dpi=400, pil_kwargs={"compression": "tiff_lzw"})
