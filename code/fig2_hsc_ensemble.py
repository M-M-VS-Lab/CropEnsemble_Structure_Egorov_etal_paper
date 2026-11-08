"""
Fig. 2, Table S3 and Fig. S1 of Egorov & Helman, computed directly from the public AgMIP-Wheat
Hot Serial Cereal (HSC) archive: Martre et al., Harvard Dataverse, doi:10.7910/DVN/ECSFZG (HSC-AgMIP_data.rar).

Treatments used (PNO / Trt in the archive):
  (a) PNO 14, Trt C – sown 1 Dec 2008, no heating  (sampling at 56, 126 and 162 days after sowing)
  (b) PNO 1,  Trt H – sown 13 Mar 2007, infrared heating (sampling at 43, 69 and 85 days after sowing)
Simulations: Version "Original" of the 29 AgMIP-Wheat models; aboveground biomass (Biom, t/ha).

Usage: python fig2_hsc_ensemble.py <path to unpacked HSC-AgMIP_data folder>
"""
import sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = sys.argv[1] if len(sys.argv) > 1 else "HSC-AgMIP_data"
TRT = {"a": (14, "C"), "b": (1, "H")}
NAMES = {"NG": "GECROS", "LP": "LPJmL", "WO": "WOFOST", "WG": "WHEATGROW", "MC": "MCWLA", "NS": "SUCROS",
         "NP": "SPASS", "DC": "DAISY", "MO": "MONICA", "AE": "APSIM-E", "AN": "APSIM-Nwheat", "AW": "APSIM-Wheat",
         "DS": "DSSAT-CERES", "DR": "DSSAT-CROPSIM", "NC": "CERES-Nwheat", "SQ": "SiriusQuality", "S2": "Sirius",
         "ST": "STICS", "LI": "LINTUL", "SP": "SIMPLACE", "FA": "FASSET", "HE": "HERMES", "IC": "InfoCrop",
         "GL": "GLAM", "EW": "EPIC", "SA": "SALUS", "AQ": "AquaCrop", "CS": "CropSyst", "OL": "O'Leary"}
CLASS = {**{m: "FvCB" for m in ["NG", "LP"]}, **{m: "light" for m in ["WO", "WG", "MC", "NS", "NP", "DC", "MO", "HE"]},
         **{m: "water" for m in ["AQ", "CS", "OL"]}}
CLASS.update({m: "RUE" for m in NAMES if m not in CLASS})
CODE_GROUP = {"AE": "APSIM", "AN": "APSIM", "AW": "APSIM", "DS": "DSSAT", "DR": "DSSAT", "NC": "DSSAT",
              "SQ": "Sirius", "S2": "Sirius", "WO": "Wageningen", "NS": "Wageningen", "LI": "Wageningen", "MO": "HERMES", "HE": "HERMES"}

def load():
    m = pd.read_csv(os.path.join(ROOT, "4-Experimental data", "HSC_AgMIP_daily_measurments_ave_sd.txt"), sep="\t", encoding="latin1")
    s = pd.read_csv(os.path.join(ROOT, "5-Model output", "HSC_AgMIP_daily_simulations.txt"), sep="\t", encoding="latin1", low_memory=False)
    s = s[s.Version == "Original"]
    obs, sim = {}, {}
    for p, (pno, trt) in TRT.items():
        o = m[(m.PNO == pno) & (m.Trt == trt) & m.Biom.notna()][["DAS", "Biom", "Biom.sd"]].sort_values("DAS")
        obs[p] = o.set_index("DAS")
        sim[p] = s[(s.PNO == pno) & (s.Trt == trt) & s.DAS.isin(o.DAS)].pivot_table(index="DAS", columns="Model", values="Biom")
    return obs, sim

def knutti(wide):
    X = wide.values; n = X.shape[1]
    D = np.sqrt(((X[:, :, None] - X[:, None, :]) ** 2).mean(axis=0)); off = ~np.eye(n, dtype=bool)
    sig = D[off].mean(); W = 1 / (1 + (np.exp(-(D / sig) ** 2) * off).sum(axis=1))
    return pd.Series(W / W.sum(), index=wide.columns), sig

def shades(c, n, lo, hi): return [plt.get_cmap(c)(v) for v in np.linspace(hi, lo, n)]
def styles():
    st = {}
    for m, c in zip(["NG", "LP"], ["#1a9850", "#66bd63"]): st[m] = (c, "*", 70)
    L = [m for m in NAMES if CLASS[m] == "light"]; Wt = [m for m in NAMES if CLASS[m] == "water"]; R = [m for m in NAMES if CLASS[m] == "RUE"]
    for m, c in zip(L, shades("Blues", len(L), 0.45, 0.95)): st[m] = (c, "s", 22)
    for m, c in zip(Wt, ["#01665e", "#35978f", "#5ab4ac", "#80cdc1"]): st[m] = (c, "D", 22)
    for i, (m, c) in enumerate(zip(R, shades("Reds", 9, 0.45, 0.95) + shades("Oranges", 9, 0.4, 0.9))): st[m] = (c, "ov^"[i % 3], 24)
    return st
LINES = [("all", "black", "All 29 models"), ("RUE", "#d7301f", "RUE-based models"),
         ("light", "#2166ac", "Light-response models"), ("FvCB", "#1a9850", "FvCB-type models")]

def main():
    obs, sim = load(); st = styles()
    means = {p: {"all": sim[p].mean(axis=1), **{c: sim[p][[m for m in sim[p] if CLASS[m] == c]].mean(axis=1) for c in ["RUE", "light", "FvCB", "water"]}} for p in sim}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.6)); lim = 26
    for ax, p in zip(axes, ["a", "b"]):
        o = obs[p]; S = sim[p]
        for m in S.columns:
            c, mk, s = st[m]; ax.scatter(o.Biom.values, S[m].values, s=s, color=c, marker=mk, alpha=0.85, linewidths=0, zorder=3)
        xx = np.array([0, lim])
        for k, col, _ in LINES:   # line through the origin with slope = total simulated / total observed biomass
            slope = means[p][k].sum() / o.Biom.sum(); ax.plot(xx, slope * xx, color=col, lw=1.6 if k == "all" else 1.2, zorder=4)
        ax.plot(xx, xx, "--", color="0.35", lw=0.9, zorder=1)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal"); ax.grid(color="0.9", lw=0.5)
        ax.set_xticks(range(0, lim + 1, 5)); ax.set_yticks(range(0, lim + 1, 5))
        ax.text(1, lim - 1, p, fontweight="bold", fontsize=11, va="top"); ax.set_xlabel("Observed aboveground biomass (t/ha)")
    axes[0].set_ylabel("Simulated aboveground biomass (t/ha)")
    h = [Line2D([], [], color=c, lw=1.5, label=l) for _, c, l in LINES] + [Line2D([], [], color="0.35", ls="--", label="1:1 line")]
    for cl in ["FvCB", "light", "RUE", "water"]:
        for m in NAMES:
            if CLASS[m] == cl:
                c, mk, s = st[m]; h.append(Line2D([], [], ls="", marker=mk, color=c, markersize=7 if mk == "*" else 5, label=NAMES[m]))
    fig.legend(handles=h, loc="lower center", ncol=5, fontsize=6.3, frameon=False, bbox_to_anchor=(0.5, 0))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.98, bottom=0.36, wspace=0.18)
    fig.savefig("Fig2.png", dpi=400); fig.savefig("Fig2.tif", dpi=400, pil_kwargs={"compression": "tiff_lzw"})

    # Table S3 and Fig. S1: three ways of averaging
    allw = pd.concat([sim["a"], sim["b"]]); w, sig = knutti(allw)
    rows = []
    for p in ["a", "b"]:
        S = sim[p]; g = S.T.groupby(lambda m: CODE_GROUP.get(m, m)).mean().mean()
        for d in S.index:
            rows.append(dict(panel=p, DAS=d, observed=obs[p].Biom[d], obs_sd=obs[p]["Biom.sd"][d], mean1=S.loc[d].mean(), mean2=g[d],
                             mean3=(S.loc[d] * w[S.columns]).sum(), RUE=means[p]["RUE"][d], light=means[p]["light"][d],
                             FvCB=means[p]["FvCB"][d], water=means[p]["water"][d], CV=100 * S.loc[d].std() / S.loc[d].mean()))
    T = pd.DataFrame(rows); T.round(2).to_csv("TableS3.csv", index=False); print(T.round(2).to_string(index=False))
    print("code-sharing groups:", len(set(CODE_GROUP.get(m, m) for m in allw.columns)), "| sigma_S = %.2f t/ha" % sig)
    for p in ["a", "b"]:
        r = T[T.panel == p].iloc[-1]
        sl = {k: means[p][k].sum() / obs[p].Biom.sum() for k in ["all", "RUE", "light", "FvCB", "water"]}
        print(p, "slopes (total sim/total obs):", {k: round(v, 2) for k, v in sl.items()})
        print(p, "last sampling (DAS %d), observed %.2f:" % (r.DAS, r.observed),
              ", ".join("%s %.2f (%+.0f%%)" % (k, r[k], 100 * (r[k] / r.observed - 1)) for k in ["mean1", "mean2", "mean3", "RUE", "light", "FvCB", "water"]))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
    for ax, k, t in zip(axes, ["mean1", "mean2", "mean3"], ["Mean 1: unweighted mean", "Mean 2: group average", "Mean 3: similarity weighting"]):
        for p, c, mk, l in [("a", "#2166ac", "o", "Winter sowing, no heating"), ("b", "#d7301f", "s", "Spring sowing, heated")]:
            r = T[T.panel == p]
            ax.errorbar(r.observed, r[k], xerr=r.obs_sd, color=c, marker=mk, ms=4, lw=1.2, capsize=2, label=l)
        ax.plot([0, lim], [0, lim], "--", color="0.4", lw=0.8); ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_title(t, fontsize=8)
        ax.set_xlabel("Observed biomass (t/ha)"); ax.set_aspect("equal")
    axes[0].set_ylabel("Ensemble mean (t/ha)"); axes[0].legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig("FigS1.png", dpi=400)

if __name__ == "__main__":
    main()
