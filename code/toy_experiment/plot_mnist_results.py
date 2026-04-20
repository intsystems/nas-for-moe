"""Визуализация результатов двух MNIST SGEM запусков (K=5 и K=3)."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

BASE = Path(__file__).resolve().parent
RUNS = BASE / "runs"
OUT = RUNS / "mnist_results_comparison.png"

# ── данные ────────────────────────────────────────────────────────────────────

with open(RUNS / "results_mnist_sgem.json") as f:
    r5 = json.load(f)["mnist_sgem"]
with open(RUNS / "results_mnist_sgem_k3.json") as f:
    r3 = json.load(f)["mnist_sgem"]

M = 20

# K=5
hard5 = r5["hard_assignments"]         # list[M]
rmat5 = np.array(r5["r_matrix"])        # [M, 5]
hist5 = r5["history"]                   # [50]
obj5  = r5["objective_value"]

# K=3
hard3 = r3["hard_assignments"]         # list[M]
rmat3 = np.array(r3["r_matrix"])        # [M, 3]
hist3 = r3["history"]                   # [50]
obj3  = r3["objective_value"]

# per-expert accuracies (из финального S-шага)
acc5 = {
    0: (sorted([c for c, e in enumerate(hard5) if e == 0]), 0.890),
    1: (sorted([c for c, e in enumerate(hard5) if e == 1]), 0.970),
    2: (sorted([c for c, e in enumerate(hard5) if e == 2]), 0.697),
    3: (sorted([c for c, e in enumerate(hard5) if e == 3]), 0.740),
    4: (sorted([c for c, e in enumerate(hard5) if e == 4]), 0.969),
}
acc3 = {
    0: (sorted([c for c, e in enumerate(hard3) if e == 0]), 0.987),
    1: (sorted([c for c, e in enumerate(hard3) if e == 1]), 0.975),
    2: (sorted([c for c, e in enumerate(hard3) if e == 2]), 0.981),
}

# digit names per cluster (из визуальной инспекции)
cluster_digit = {
    0: "9", 1: "1", 2: "5", 3: "7", 4: "0",
    5: "6", 6: "7", 7: "8", 8: "3", 9: "0",
    10: "1", 11: "4", 12: "0", 13: "6", 14: "4",
    15: "2", 16: "2", 17: "7", 18: "3", 19: "5",
}

COLORS5 = ["#d62728", "#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd"]
COLORS3 = ["#d62728", "#2ca02c", "#1f77b4"]

# ── фигура ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 14))
gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35,
              height_ratios=[1.1, 1.3, 1.0])

# ── 1. Log-likelihood history ─────────────────────────────────────────────────
ax_hist = fig.add_subplot(gs[0, :])
iters = np.arange(1, 51)
ax_hist.plot(iters, hist5, color="#1f77b4", lw=1.5, label=f"K=5  (best={obj5:.3f})", zorder=3)
ax_hist.plot(iters, hist3, color="#d62728", lw=1.5, label=f"K=3  (best={obj3:.3f})", zorder=3)
ax_hist.axhline(0, color="gray", lw=0.7, ls="--", alpha=0.6)

# отметить лучшее значение
for hist, obj, color in [(hist5, obj5, "#1f77b4"), (hist3, obj3, "#d62728")]:
    best_iter = int(np.argmax(hist)) + 1
    ax_hist.scatter([best_iter], [obj], s=60, color=color, zorder=5)
    ax_hist.annotate(f"{obj:.3f}", xy=(best_iter, obj),
                     xytext=(4, 4), textcoords="offset points",
                     fontsize=8, color=color)

# S-шаг маркеры: каждые 2 итерации для K=3, каждую для K=5
for i in range(0, 50, 2):
    ax_hist.axvline(i + 1, color="#d62728", alpha=0.08, lw=1)
for i in range(0, 50, 1):
    ax_hist.axvline(i + 1, color="#1f77b4", alpha=0.04, lw=0.5)

ax_hist.set_xlabel("EM iteration")
ax_hist.set_ylabel("Log-likelihood")
ax_hist.set_title("Log-likelihood history")
ax_hist.legend(fontsize=9)
ax_hist.set_xlim(1, 50)

# ── 2. r_matrix heatmaps ─────────────────────────────────────────────────────
xtick_labels5 = [f"E{k}" for k in range(5)]
xtick_labels3 = [f"E{k}" for k in range(3)]
ytick_labels = [f"c{m}\n({cluster_digit[m]})" for m in range(M)]

ax_r5 = fig.add_subplot(gs[1, 0])
im5 = ax_r5.imshow(rmat5, aspect="auto", vmin=0, vmax=1, cmap="YlOrRd")
ax_r5.set_xticks(range(5)); ax_r5.set_xticklabels(xtick_labels5, fontsize=8)
ax_r5.set_yticks(range(M)); ax_r5.set_yticklabels(ytick_labels, fontsize=6)
ax_r5.set_title("r-matrix  K=5", fontsize=9)
plt.colorbar(im5, ax=ax_r5, fraction=0.046, pad=0.04)

ax_r3 = fig.add_subplot(gs[1, 1])
im3 = ax_r3.imshow(rmat3, aspect="auto", vmin=0, vmax=1, cmap="YlOrRd")
ax_r3.set_xticks(range(3)); ax_r3.set_xticklabels(xtick_labels3, fontsize=8)
ax_r3.set_yticks(range(M)); ax_r3.set_yticklabels(ytick_labels, fontsize=6)
ax_r3.set_title("r-matrix  K=3", fontsize=9)
plt.colorbar(im3, ax=ax_r3, fraction=0.046, pad=0.04)

# ── 3. Hard assignment (bar chart) ──────────────────────────────────────────
ax_ha = fig.add_subplot(gs[1, 2])
bar_colors5 = [COLORS5[hard5[m]] for m in range(M)]
ax_ha.barh(range(M), [1]*M, color=bar_colors5, edgecolor="none")
for m in range(M):
    ax_ha.text(0.5, m, f"c{m} ({cluster_digit[m]})",
               ha="center", va="center", fontsize=7, color="white", fontweight="bold")
ax_ha.set_xlim(0, 1); ax_ha.set_yticks([]); ax_ha.set_xticks([])
ax_ha.set_title("Hard assign  K=5", fontsize=9)
patches5 = [mpatches.Patch(color=COLORS5[k], label=f"E{k}") for k in range(5)]
ax_ha.legend(handles=patches5, fontsize=7, loc="lower right")

# ── 4. Per-expert accuracy bars ───────────────────────────────────────────────
ax_acc5 = fig.add_subplot(gs[2, 0])
ks5 = list(acc5.keys())
accs5 = [acc5[k][1] for k in ks5]
sizes5 = [len(acc5[k][0]) for k in ks5]
bars5 = ax_acc5.bar(ks5, accs5, color=COLORS5, edgecolor="white", width=0.6)
for bar, sz in zip(bars5, sizes5):
    ax_acc5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{bar.get_height():.3f}\n(n={sz})",
                 ha="center", va="bottom", fontsize=8)
ax_acc5.set_ylim(0.5, 1.05); ax_acc5.set_xticks(ks5)
ax_acc5.set_xticklabels([f"E{k}" for k in ks5])
ax_acc5.set_ylabel("Val accuracy"); ax_acc5.set_title("Expert accuracy  K=5", fontsize=9)
ax_acc5.axhline(np.mean(accs5), color="gray", ls="--", lw=1,
                label=f"mean={np.mean(accs5):.3f}")
ax_acc5.legend(fontsize=8)

ax_acc3 = fig.add_subplot(gs[2, 1])
ks3 = list(acc3.keys())
accs3 = [acc3[k][1] for k in ks3]
sizes3 = [len(acc3[k][0]) for k in ks3]
bars3 = ax_acc3.bar(ks3, accs3, color=COLORS3, edgecolor="white", width=0.4)
for bar, sz in zip(bars3, sizes3):
    ax_acc3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{bar.get_height():.3f}\n(n={sz})",
                 ha="center", va="bottom", fontsize=8)
ax_acc3.set_ylim(0.5, 1.05); ax_acc3.set_xticks(ks3)
ax_acc3.set_xticklabels([f"E{k}" for k in ks3])
ax_acc3.set_ylabel("Val accuracy"); ax_acc3.set_title("Expert accuracy  K=3", fontsize=9)
ax_acc3.axhline(np.mean(accs3), color="gray", ls="--", lw=1,
                label=f"mean={np.mean(accs3):.3f}")
ax_acc3.legend(fontsize=8)

# ── 5. Cluster sizes per expert ───────────────────────────────────────────────
ax_sz = fig.add_subplot(gs[2, 2])
all_sizes5 = [len(acc5[k][0]) for k in range(5)]
all_sizes3 = [len(acc3[k][0]) for k in range(3)]
ax_sz.bar(range(5), all_sizes5, color=COLORS5, alpha=0.85, label="K=5", edgecolor="white")
ax_sz.bar(range(3), all_sizes3, color=COLORS3, alpha=0.5, hatch="///",
           edgecolor="gray", label="K=3")
for i, s in enumerate(all_sizes5):
    ax_sz.text(i, s + 0.1, str(s), ha="center", va="bottom", fontsize=8)
for i, s in enumerate(all_sizes3):
    ax_sz.text(i, s + 0.1, str(s), ha="center", va="bottom", fontsize=8,
               color=COLORS3[i], fontweight="bold")
ax_sz.set_xticks(range(5))
ax_sz.set_xticklabels([f"E{k}" for k in range(5)])
ax_sz.set_ylabel("Clusters per expert")
ax_sz.set_title("Cluster count per expert", fontsize=9)
ax_sz.legend(fontsize=8)

fig.suptitle("MNIST SGEM: K=5 vs K=3", fontsize=13, fontweight="bold", y=1.01)
plt.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"saved: {OUT}")
