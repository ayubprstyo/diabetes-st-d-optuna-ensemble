"""Reproduce the manuscript figures from the experiment outputs.
Usage: python src/make_figures.py results figures"""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
HATCH = ["", "////", "....", "xxxx"]; LS = ["-", "--", ":", "-."]
SEQ = ["#ffffff", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95"]
plt.rcParams.update({"hatch.color": "white", "hatch.linewidth": 0.6, "font.family": "Liberation Serif", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
import sys, os
R = sys.argv[1] if len(sys.argv) > 1 else "results"; FIG = sys.argv[2] if len(sys.argv) > 2 else "figures"; os.makedirs(FIG, exist_ok=True)
res = pd.read_csv(f"{R}/results.csv"); info = json.load(open(f"{R}/info.json")); P = np.load(f"{R}/probs.npz"); y = P["y"]
imp = pd.read_csv(f"{R}/importance.csv"); best = info["best_F"]; bm = best.split("_")[0]
MODELS = ["XGBoost", "LightGBM", "CatBoost", "RandomForest"]; CFG = list("ABGCDEF")
LAB = {"A": "A: Baseline", "B": "B: ST", "C": "C: Optuna", "D": "D: ST+Optuna", "E": "E: CW+Optuna", "F": "F: ST-D+Optuna", "G": "G: ST-D"}

# Gambar 1: alur penelitian
fig, ax = plt.subplots(figsize=(7.2, 3.3)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4.4)
steps = [("Business\nUnderstanding", "Early detection\nof diabetes risk"), ("Data\nUnderstanding", "BRFSS 2015\n253\u00a0680 × 22"),
         ("Data\nPreparation", "Deduplication,\n80:20 split,\nstandardization"), ("Modeling", "Resampling\n(ST / ST-D)\n+ Optuna TPE\n+ 4 ensemble\nmodels"),
         ("Evaluation", "Hold-out test,\nbootstrap CI,\npermutation\nimportance")]
w, gap = 1.72, 0.3
for i, (t, d) in enumerate(steps):
    x = 0.1 + i * (w + gap)
    ax.add_patch(FancyBboxPatch((x, 2.35), w, 1.3, boxstyle="round,pad=0.02,rounding_size=0.12", fc="#cde2fb", ec="#2a78d6", lw=1.2))
    ax.text(x + w / 2, 3.0, t, ha="center", va="center", fontsize=8, weight="bold", color=INK)
    ax.add_patch(FancyBboxPatch((x, 0.3), w, 1.65, boxstyle="round,pad=0.02,rounding_size=0.12", fc="white", ec=INK2, lw=0.8))
    ax.text(x + w / 2, 1.12, d, ha="center", va="center", fontsize=7.4, color=INK, linespacing=1.25)
    ax.annotate("", (x + w / 2, 1.97), (x + w / 2, 2.33), arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8, ls=":"))
    if i < 4: ax.annotate("", (x + w + gap - 0.02, 3.0), (x + w + 0.02, 3.0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1))
ax.annotate("", (0.1 + w / 2, 3.72), (0.1 + 4 * (w + gap) + w / 2, 3.72),
            arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8, connectionstyle="arc3,rad=0.12"))
ax.text(5, 4.25, "CRISP-DM iteration", ha="center", fontsize=7.5, color=INK2, style="italic")
fig.savefig(f"{FIG}/fig1_crisp_dm_workflow.png"); plt.close(fig)

# Gambar 2: F1 dan recall per konfigurasi
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=False)
for ax, met, tl in zip(axs, ["recall", "f1"], ["(a) Recall of the positive class", "(b) F1-score of the positive class"]):
    xs = np.arange(len(CFG)); bw = 0.19
    for j, m in enumerate(MODELS):
        v = [res[(res.model == m) & (res.config == c)][met].iloc[0] for c in CFG]
        ax.bar(xs + (j - 1.5) * bw, v, bw * 0.9, color=CAT[j], hatch=HATCH[j], edgecolor="white", linewidth=0, label=m.replace("RandomForest", "Random Forest"), zorder=3)
    ax.set_xticks(xs, CFG); ax.set_title(tl, fontsize=9.5, loc="left", color=INK)
    ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True); ax.set_xlabel("Configuration", fontsize=9); ax.set_ylabel("Recall" if met == "recall" else "F1-score", fontsize=9)
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(f"{FIG}/fig2_recall_f1.png"); plt.close(fig)

# Gambar 3: confusion matrix A vs F (model terbaik)
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.9))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", SEQ[:4])
for ax, c in zip(axs, ["A", "F"]):
    cm = confusion_matrix(y, P[f"{bm}_{c}"] >= 0.5); ax.imshow(cm / cm.sum(1, keepdims=True), cmap=cmap, vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(cm):
        frac = v / cm[i].sum(); ax.text(j, i, (f"{v:,}".replace(",", "\u00a0") if v >= 10000 else str(v)) + "\n(" + f"{frac:.1%}" + ")", ha="center", va="center",
                                        fontsize=8.5, color="white" if frac > 0.6 else INK)
    ax.set_xticks([0, 1], ["Non-diabetic", "Diabetic"], fontsize=8); ax.set_yticks([0, 1], ["Non-diabetic", "Diabetic"], fontsize=8)
    ax.set_xlabel("Predicted class", fontsize=8.5); ax.set_ylabel("Actual class", fontsize=8.5); ax.set_title(f"({'a' if c=='A' else 'b'}) {bm.replace('RandomForest', 'Random Forest')} – {LAB[c]}", fontsize=8.5, loc="left")
    for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(); fig.savefig(f"{FIG}/fig4_confusion_matrices.png"); plt.close(fig)

# Gambar 4: ROC & PR model terbaik untuk A, D, E, F
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1))
for k, c in enumerate(["A", "D", "E", "F"]):
    p = P[f"{bm}_{c}"]; r = res[(res.model == bm) & (res.config == c)].iloc[0]
    fpr, tpr, _ = roc_curve(y, p); pr, rc, _ = precision_recall_curve(y, p)
    axs[0].plot(fpr, tpr, color=CAT[k], lw=1.6, ls=LS[k], label=f"{LAB[c]} (AUC = {r.roc_auc:.3f})")
    axs[1].plot(rc, pr, color=CAT[k], lw=1.6, ls=LS[k], label=f"{LAB[c]} (AP = {r.pr_auc:.3f})")
axs[0].plot([0, 1], [0, 1], color=INK2, lw=0.8, ls="--"); axs[1].axhline(y.mean(), color=INK2, lw=0.8, ls="--")
axs[0].set(xlabel="False positive rate", ylabel="True positive rate"); axs[1].set(xlabel="Recall", ylabel="Precision")
axs[0].set_title(f"(a) ROC curve – {bm.replace('RandomForest', 'Random Forest')}", fontsize=9, loc="left"); axs[1].set_title(f"(b) Precision–recall curve – {bm.replace('RandomForest', 'Random Forest')}", fontsize=9, loc="left")
for ax in axs: ax.grid(True, color=GRID); ax.legend(frameon=False, fontsize=6.8, loc="lower right" if ax is axs[0] else "upper right")
fig.tight_layout(); fig.savefig(f"{FIG}/fig3_roc_pr_curves.png"); plt.close(fig)

# Gambar 5: importansi permutasi
d = imp.sort_values("perm_mean")
fig, ax = plt.subplots(figsize=(6.2, 4.0))
ax.barh(d.feature, d.perm_mean, xerr=d.perm_std, color="#2a78d6", height=0.65, error_kw=dict(ecolor=INK2, lw=0.8), zorder=3)
ax.xaxis.grid(True, color=GRID, zorder=0); ax.set_xlabel("Mean decrease in ROC-AUC when the feature is permuted (5 repeats)", fontsize=8.5)
ax.tick_params(axis="y", labelsize=8); ax.set_title(f"Permutation feature importance – {bm.replace('RandomForest', 'Random Forest')} (configuration F)", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(f"{FIG}/fig5_permutation_importance.png"); plt.close(fig)
print("figs ok")
