"""Three-way comparison of CNN score normalization schemes on the SFL generation
distribution: raw sigmoid vs per-cycle min-max vs frozen-reference percentile.

Same deployment path and eval envs as cnn_score_distribution.py (seed 0, 5000 envs
= one SFL selection cycle). Min-max is computed over the pooled cycle, as the old
implementation did. A second independent cycle (seed 3000) shows how the min-max
mapping shifts between cycles while the frozen percentile mapping cannot.

Produces score_normalization_comparison.png / .pdf

Usage:  .venv/bin/python score_normalization_comparison.py
"""

import sys
import yaml

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "sfl/train")
from rlhf_utils import load_learnability_model, get_jaxnav_rasterizer
from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav
from flax import nnx

# Deployed-ensemble member init100 (aug set, min-raw-val-loss epoch); the
# old pre-reorganization path now lives under legacy/ and is not used.
CKPT = "/home/d/durmusy/Desktop/GIT/new/uedrlhf/outputs/checkpoints/ensemble/jaxnav/ens-jaxnav-aug-init100/epoch_26"
BATCH_SIZE = 1000
CYCLE_BATCHES = 5      # one SFL selection cycle = 5000 envs
REF_SIZE = 10_000

env_cfg = yaml.safe_load(open("sfl/train/config/env/jaxnav.yaml"))
env = JaxNav(num_agents=1, **env_cfg["env_params"])
ms = env_cfg["env_params"]["map_params"]["map_size"]
render_fn = get_jaxnav_rasterizer(img_height=64, img_width=64, map_height=ms[0], map_width=ms[1], cell_size=1.0)
graphdef, state = load_learnability_model(CKPT)
model = nnx.merge(graphdef, state)


@jax.jit
def score_batch(rng):
    reset_rng = jax.random.split(rng, BATCH_SIZE)
    _, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
    chunked = jax.tree.map(lambda x: x.reshape((BATCH_SIZE // 100, 100) + x.shape[1:]), env_state)
    _, logits = jax.lax.scan(
        lambda c, ch: (c, model(jax.vmap(render_fn)(ch), deterministic=True, return_logits=True)),
        None, chunked)
    return logits.reshape((BATCH_SIZE,))


def sample_logits(seed, n):
    rngs = jax.random.split(jax.random.PRNGKey(seed), n // BATCH_SIZE)
    return np.concatenate([np.asarray(score_batch(r)) for r in rngs])


cycle_a = sample_logits(0, CYCLE_BATCHES * BATCH_SIZE)       # same envs as prior analyses
cycle_b = sample_logits(3000, CYCLE_BATCHES * BATCH_SIZE)    # an independent later cycle
ref = np.sort(sample_logits(1000, REF_SIZE))                 # frozen percentile reference
n = len(cycle_a)

def minmax(x, pool):
    return (x - pool.min()) / (pool.max() - pool.min())

def percentile(x):
    return np.interp(x, ref, np.linspace(0.0, 1.0, len(ref)))

sig = np.asarray(jax.nn.sigmoid(jnp.asarray(cycle_a)))
mm = minmax(cycle_a, cycle_a)
pct = percentile(cycle_a)

schemes = {"sigmoid": sig, "min-max (per cycle)": mm, "percentile (frozen ref)": pct}
mus = np.linspace(0.0, 1.0, 101)
inner = (mus >= 0.1) & (mus <= 0.9)

print(f"n = {n} eval envs (one SFL selection cycle)")
print(f"{'scheme':>24} | {'distinct':>9} | {'% outside [.01,.99]':>19} | {'cov min':>7} | {'cov mean':>8} | (uniform ref: 500)")
coverages = {}
for name, s in schemes.items():
    cov = np.array([((s >= m - 0.05) & (s <= m + 0.05)).sum() for m in mus])
    coverages[name] = cov
    sat = 100 * ((s > 0.99) | (s < 0.01)).mean()
    print(f"{name:>24} | {len(np.unique(s)):>9} | {sat:>18.1f}% | {cov[inner].min():>7} | {cov[inner].mean():>8.0f} |")

# Cross-cycle comparability: how does a FIXED environment's score change between cycles?
probe = np.linspace(-20, 75, 200)  # logit range covered by both cycles
mm_a, mm_b = minmax(probe, cycle_a), minmax(probe, cycle_b)
pct_ab_diff = 0.0  # frozen reference: mapping identical by construction
print(f"\ncycle A logit range: [{cycle_a.min():.1f}, {cycle_a.max():.1f}]   "
      f"cycle B logit range: [{cycle_b.min():.1f}, {cycle_b.max():.1f}]")
print(f"fixed env, min-max score shift between cycles: mean |Δ|={np.abs(mm_a - mm_b).mean():.4f}, "
      f"max |Δ|={np.abs(mm_a - mm_b).max():.4f}")
print("fixed env, percentile score shift between cycles: 0 (frozen mapping, by construction)")

# ---- Plot ----
fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
colors = {"sigmoid": "#c0504d", "min-max (per cycle)": "#e8a33d", "percentile (frozen ref)": "#4878b0"}

ax = axes[0]
for name, s in schemes.items():
    ax.hist(s, bins=50, range=(0, 1), color=colors[name], alpha=0.55, label=name)
ax.axhline(n / 50, color="gray", linestyle="--", linewidth=1.2, label="uniform")
ax.set_xlabel("normalized CNN difficulty score")
ax.set_ylabel("environments")
ax.set_title(f"Score distribution ({n} DR-sampled environments)")
ax.legend(fontsize=8)

ax = axes[1]
for name, cov in coverages.items():
    ax.plot(mus, cov, color=colors[name], linewidth=1.8, label=name)
ax.axhline(n * 0.1, color="gray", linestyle="--", linewidth=1.2, label="uniform reference")
ax.set_xlabel(r"curriculum target $\mu$")
ax.set_ylabel(r"environments within $\mu \pm 0.05$")
ax.set_title("Curriculum coverage per target difficulty")
ax.legend(fontsize=8)

ax = axes[2]
ax.plot(probe, mm_a, color="#e8a33d", linewidth=1.8, label="min-max, cycle A")
ax.plot(probe, mm_b, color="#b0722a", linewidth=1.8, linestyle="--", label="min-max, cycle B")
ax.plot(probe, percentile(probe), color="#4878b0", linewidth=1.8, label="percentile (identical in A and B)")
ax.set_xlabel("raw CNN score (logit) of a fixed environment")
ax.set_ylabel("normalized score")
ax.set_title("Cross-cycle comparability: score assigned to the\nsame environment in two selection cycles")
ax.legend(fontsize=8)

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Normalization schemes for CNN difficulty scores on the SFL generation distribution "
             f"(checkpoint epoch_33, {n} envs)", y=1.02, fontsize=11)
plt.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"score_normalization_comparison.{ext}", dpi=200, bbox_inches="tight")
print("saved score_normalization_comparison.png / .pdf")
