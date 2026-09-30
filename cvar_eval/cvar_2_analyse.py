import os, sys
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
import jax.numpy as jnp
from matplotlib import pyplot as plt

GROUPS = ["standard", "dr", "cnn", "hybrid_linear", "hybrid_soft_handoff",
          "hybrid_learnability_weighted", "hybrid_multiplicative"]
LABELS = {"standard": "Standard SFL", "dr": "DR", "cnn": "CNN (pure prior)",
          "hybrid_linear": "Linear Addition", "hybrid_soft_handoff": "Soft Handoff",
          "hybrid_learnability_weighted": "Learnability-Weighted",
          "hybrid_multiplicative": "Multiplicative"}
COLORS = {"standard": "#0077BB", "dr": "#BBBBBB", "cnn": "#EE7733",
          "hybrid_linear": "#009988", "hybrid_soft_handoff": "#33BBEE",
          "hybrid_learnability_weighted": "#EE3377",
          "hybrid_multiplicative": "#CC3311"}

SEEDS, TOTAL = 10, 10000
PERCENTAGES = [1, 2, 5, 10, 20, 50, 100]
OUT = "cvar_eval/results"
os.makedirs(OUT, exist_ok=True)

def get_vals(group, rollout_seed):
    wr = np.zeros((SEEDS, TOTAL), dtype=np.float32)
    for seed in range(SEEDS):
        f = (f"sfl/data/eval/results/jaxnav-single/"
             f"eval_{TOTAL}_envs_seed_{rollout_seed}/{group}/{seed}.csv")
        df = pd.read_csv(f)
        assert len(df) == TOTAL, f"{f}: {len(df)} rows"
        wr[seed] = df["win-rates"].to_numpy()
    return wr

cvar, lines = {}, []
for g in GROUPS:
    a, b = get_vals(g, 0), get_vals(g, 1)
    res = np.zeros((SEEDS, len(PERCENTAGES)))
    for seed in range(SEEDS):
        order = np.argsort(a[seed], kind="stable")  # sfls  jnp.argsort tie order
        unbiased = b[seed][order]
        biased = a[seed][order]
        for pi, pct in enumerate(PERCENTAGES):
            n = int(pct / 100 * TOTAL)
            res[seed, pi] = unbiased[:n].mean()
            lines.append(f"CVAR {g} seed{seed} alpha={pct}% "
                         f"biased={biased[:n].mean():.4f} unbiased={res[seed,pi]:.4f}")
    cvar[g] = res

with open(f"{OUT}/cvar_values.txt", "w") as f:
    f.write("\n".join(lines) + "\n\nSUMMARY (mean +- sem over seeds, unbiased)\n")
    for g in GROUPS:
        m, s = cvar[g].mean(0), cvar[g].std(0) / np.sqrt(SEEDS)
        f.write(f"{g:30s} " + "  ".join(
            f"a{p}%={mm:.4f}+-{ss:.4f}" for p, mm, ss in zip(PERCENTAGES, m, s)) + "\n")

fig, ax = plt.subplots(figsize=(6.4, 4.4))
for g in GROUPS:
    m = cvar[g].mean(0) * 100
    e = cvar[g].std(0) / np.sqrt(SEEDS) * 100
    ax.plot(PERCENTAGES, m, marker="o", lw=2.5, color=COLORS[g], label=LABELS[g])
    ax.fill_between(PERCENTAGES, m - e, m + e, color=COLORS[g], alpha=0.2, lw=0)
ax.set_xscale("log"); ax.set_xticks([1, 10, 100]); ax.set_xticklabels(["1%", "10%", "100%"])
ax.set_xlabel(r"$\alpha$"); ax.set_ylabel(r"Mean win rate % on worst-$\alpha$% levels")
ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/cvar_line.pdf", dpi=200, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.4, 3.6))
for i, g in enumerate(GROUPS):
    overall = get_vals(g, 1).mean(1)
    ax.bar(i, overall.mean() * 100, yerr=overall.std() / np.sqrt(SEEDS) * 100,
           color=COLORS[g], label=LABELS[g])
ax.set_xticks([]); ax.set_ylabel("Mean win rate %"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/winrates_bar.pdf", dpi=200, bbox_inches="tight")
plt.close(fig)

base = get_vals("standard", 1).mean(0)
others = [g for g in GROUPS if g != "standard"]
fig, axs = plt.subplots(1, len(others), figsize=(3.2 * len(others), 3.4))
for idx, g in enumerate(others):
    y = get_vals(g, 1).mean(0)
    h, xe, ye = np.histogram2d(base, y, bins=10, range=[[0, 1], [0, 1]])
    axs[idx].imshow(np.log(h.T + 1), extent=[0, 1, 0, 1], origin="lower")
    axs[idx].plot([0, 1], [0, 1], color="white", lw=0.8, ls="--")
    axs[idx].set_xlabel("Standard SFL"); axs[idx].set_ylabel(LABELS[g], fontsize=8)
    axs[idx].grid(False)
fig.tight_layout(); fig.savefig(f"{OUT}/compare_scatter.pdf", dpi=200, bbox_inches="tight")
print("wrote", OUT, "/ cvar_line.pdf winrates_bar.pdf compare_scatter.pdf cvar_values.txt")
