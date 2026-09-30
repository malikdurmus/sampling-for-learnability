"""Figure for the controller-variant comparison (E5): the difficulty target over training
and the solvable share of the training levels, unfiltered ("norm") generator, mean and
one-standard-deviation band over seeds 1-4 per variant.

Data: frontier_analysis_raw.json (written by frontier_analysis_pull.py). Refreshes are
numbered from 1 (cycle index + 1). V1 = the old-controller hybrid_linear runs of
sfl-jaxnav-campaign-fixed (hybrid_linear_fixed_seed1-4), V2 = norm_hybrid_pa, V3 = norm_hybrid_cf
(proximity) / norm_hybrid_cfp (learnability term), standard = norm_standard, all from
project sfl-jaxnav-frontier except V1.
Usage: python frontier_fig_mu.py [--gen norm|solv]  -> frontier_fig_mu_<gen>.png
"""
import os, sys, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
GEN = sys.argv[sys.argv.index("--gen") + 1] if "--gen" in sys.argv else "norm"
raw = json.load(open(os.path.join(HERE, "frontier_analysis_raw.json")))
K = "sampled-test-metrics.eval-sampled/overall_win_rate"; NC = 45
def cyc(run, key):
    a = np.full(NC, np.nan)
    for u, v in run["cycle"].get(key, {}).items():
        u = int(u)
        if u % 50 == 0 and 1 <= u // 50 <= NC: a[u // 50 - 1] = v
    return a
def updmean(run, key):
    acc = [[] for _ in range(NC)]
    for u, v in run["upd"].get(key, {}).items():
        c = (int(u) - 1) // 50
        if 0 <= c < NC: acc[c].append(v)
    return np.array([np.mean(x) if x else np.nan for x in acc])
runs = {}
for key, r in raw.items():
    label, name = key.split(":", 1); seed = int(name.rsplit("seed", 1)[1])
    if label == "new": gen, rest = name.split("_", 1); arm = rest.rsplit("_seed", 1)[0]
    elif name.startswith("solv_"): gen, arm = "solv", "old_" + name[5:].rsplit("_seed", 1)[0]
    else: gen, arm = "norm", "old_" + name.rsplit("_seed", 1)[0].replace("_fixed", "")
    runs.setdefault((gen, arm), {})[seed] = r
# ---- figure: mu (left) and passable share (right), norm generator, mean +- 1 std ----
COL = {"old_hybrid_linear": "#2a78d6", "hybrid_pa": "#eb6834", "hybrid_cf": "#1baf7a", "hybrid_cfp": "#eda100", "standard": "#52514e"}
LAB = {"old_hybrid_linear": "V1: all training envs, step rule", "hybrid_pa": "V2: curated levels, step rule",
       "hybrid_cf": "V3: calibrated frontier, proximity", "hybrid_cfp": "V3: calibrated frontier, learnability term", "standard": "standard SFL"}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))
x = np.arange(1, NC + 1)
for arm in ["old_hybrid_linear", "hybrid_pa", "hybrid_cf", "hybrid_cfp"]:
    M = np.array([cyc(runs[(GEN, arm)][s], "target_mu") for s in [1,2,3,4]]); m, sd = np.nanmean(M, 0), np.nanstd(M, 0)
    ax1.plot(x, m, color=COL[arm], lw=2, label=LAB[arm]); ax1.fill_between(x, m - sd, m + sd, color=COL[arm], alpha=0.15, lw=0)
for arm in ["standard", "old_hybrid_linear", "hybrid_pa", "hybrid_cf", "hybrid_cfp"]:
    P = np.array([updmean(runs[(GEN, arm)][s], "env-metrics/.passable") for s in [1,2,3,4]]); m, sd = np.nanmean(P, 0), np.nanstd(P, 0)
    ax2.plot(x, m, color=COL[arm], lw=2, label=LAB[arm], ls="--" if arm == "standard" else "-"); ax2.fill_between(x, m - sd, m + sd, color=COL[arm], alpha=0.15, lw=0)
for ax, yl, t in [(ax1, "difficulty target $\\mu$", "target over training (" + GEN + " generator)"), (ax2, "solvable share of levels trained on", "solvable share of the training levels")]:
    ax.set_xlabel("refresh"); ax.set_ylabel(yl); ax.set_title(t, loc="left", fontsize=11); ax.set_xlim(1, NC); ax.grid(True, color="#e6e5e1", lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax1.set_ylim(0, 1.02); ax2.set_ylim(0.5, 1.0)
ax1.legend(fontsize=8.5, frameon=False, loc="lower right"); ax2.legend(fontsize=8.5, frameon=False, loc="lower left")
fig.tight_layout(); out = os.path.join(HERE, f"frontier_fig_mu_{GEN}.png")
fig.savefig(out, dpi=150); print("\nsaved", out)
