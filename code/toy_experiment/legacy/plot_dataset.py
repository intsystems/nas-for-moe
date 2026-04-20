"""Plot dataset for slides (no Linear/Ring in legend)."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent.parent / "code" / "toy_experiment" / "data"
OUT_DIR = Path(__file__).resolve().parent / "figures"

X = np.load(DATA_DIR / "data_X.npy")
y = np.load(DATA_DIR / "data_y.npy")

fig, ax = plt.subplots(figsize=(7, 5.5))

colors = {0: "blue", 1: "red", 2: "cyan", 3: "orange"}
markers = {0: "o", 1: "o", 2: "o", 3: "s"}

for cls in [0, 1, 2, 3]:
    ax.scatter(X[y == cls, 0], X[y == cls, 1],
               label=f"Класс {cls}", alpha=0.6, s=50,
               color=colors[cls], marker=markers[cls])

ax.set_xlabel("$x_1$", fontsize=16)
ax.set_ylabel("$x_2$", fontsize=16)
ax.set_title("Датасет (4 класса)", fontsize=18)
ax.legend(fontsize=14)
ax.tick_params(labelsize=13)
ax.grid(True, alpha=0.3)
plt.tight_layout()

out_path = OUT_DIR / "dataset.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
plt.close(fig)
