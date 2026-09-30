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

CKPT = "/home/d/durmusy/Desktop/GIT/new/uedrlhf/outputs/checkpoints/ensemble/jaxnav/ens-jaxnav-aug-init100/epoch_26"
BATCH_SIZE = 1000
EVAL_BATCHES = 5          # same 5000 eval envs as cnn_score_distribution.py (seed 0)
REF_SIZE = 10_000
CNN_IMG_SIZE = 64

env_cfg = yaml.safe_load(open("sfl/train/config/env/jaxnav.yaml"))
env = JaxNav(num_agents=1, **env_cfg["env_params"])
ms = env_cfg["env_params"]["map_params"]["map_size"]
render_fn = get_jaxnav_rasterizer(img_height=CNN_IMG_SIZE, img_width=CNN_IMG_SIZE,
                                  map_height=ms[0], map_width=ms[1], cell_size=1.0)
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


def make_percentile_fn(ref_logits):
    ref_sorted = np.sort(ref_logits)
    grid = np.linspace(0.0, 1.0, len(ref_sorted))
    return lambda x: np.interp(x, ref_sorted, grid)


eval_logits = sample_logits(0, EVAL_BATCHES * BATCH_SIZE)
ref_a = sample_logits(1000, REF_SIZE)
ref_b = sample_logits(2000, REF_SIZE)

pct_a = make_percentile_fn(ref_a)
pct_b = make_percentile_fn(ref_b)
scores = pct_a(eval_logits)
scores_b = pct_b(eval_logits)
sig = np.asarray(jax.nn.sigmoid(jnp.asarray(eval_logits)))
n = len(scores)

# ---- 1. Distribution / utilization ----
sat_99 = int(((scores > 0.99) | (scores < 0.01)).sum())
uniq = len(np.unique(scores))
print(f"n = {n} eval envs, reference size = {REF_SIZE}")
print(f"percentile scores: mean={scores.mean():.4f} std={scores.std():.4f} median={np.median(scores):.4f}")
print(f"  (uniform target: mean 0.5, std {1/np.sqrt(12):.4f}, median 0.5)")
print(f"outside [0.01, 0.99]: {sat_99} ({100*sat_99/n:.1f}%)   [sigmoid was 76.1%]")
print(f"distinct values: {uniq}/{n}   [sigmoid was 2438/5000]")
deciles = [((scores >= i/10) & (scores < (i+1)/10)).sum() for i in range(10)]
print(f"envs per decile (uniform target {n//10}): {deciles}")

# ---- 2. Curriculum coverage ----
mus = np.linspace(0.0, 1.0, 101)
cov_pct = np.array([((scores >= m - 0.05) & (scores <= m + 0.05)).sum() for m in mus])
cov_sig = np.array([((sig >= m - 0.05) & (sig <= m + 0.05)).sum() for m in mus])
inner = (mus >= 0.1) & (mus <= 0.9)
print(f"coverage within mu±0.05, inner mu in [0.1,0.9]: "
      f"percentile min={cov_pct[inner].min()} mean={cov_pct[inner].mean():.0f} | "
      f"sigmoid min={cov_sig[inner].min()} mean={cov_sig[inner].mean():.0f} | uniform ref={n*0.1:.0f}")

# ---- 3. Reference  ----
diff = np.abs(scores - scores_b)
print(f"two independent {REF_SIZE}-env references: mean|Δscore|={diff.mean():.5f}, "
      f"p95={np.percentile(diff, 95):.5f}, max={diff.max():.5f}")

# Cross-rollout comparability: per-rollout score stats under ONE frozen reference
print("per-rollout stats under frozen reference A (should be ~identical):")
for i in range(EVAL_BATCHES):
    s = scores[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
    print(f"  rollout {i}: mean={s.mean():.3f}  10/50/90th pct={np.percentile(s,10):.3f}/{np.percentile(s,50):.3f}/{np.percentile(s,90):.3f}")

# ---- Plot ----
fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

ax = axes[0]
ax.hist(sig, bins=50, range=(0, 1), color="#c0504d", alpha=0.55, label="sigmoid (before)")
ax.hist(scores, bins=50, range=(0, 1), color="#4878b0", alpha=0.75, label="percentile (after)")
ax.axhline(n / 50, color="gray", linestyle="--", linewidth=1.2, label="uniform")
ax.set_xlabel("CNN difficulty score")
ax.set_ylabel("environments")
ax.set_title(f"Score distribution of {n} DR-sampled environments")
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(mus, cov_sig, color="#c0504d", linewidth=1.8, label="sigmoid (before)")
ax.plot(mus, cov_pct, color="#4878b0", linewidth=1.8, label="percentile (after)")
ax.axhline(n * 0.1, color="gray", linestyle="--", linewidth=1.2, label="uniform reference")
ax.set_xlabel(r"curriculum target $\mu$")
ax.set_ylabel(r"environments within $\mu \pm 0.05$")
ax.set_title("Curriculum coverage per target difficulty")
ax.legend(fontsize=8)

ax = axes[2]
ax.hist(diff, bins=50, color="#6aa66a", edgecolor="white", linewidth=0.4)
ax.set_xlabel("|score difference| between two independent references")
ax.set_ylabel("environments")
ax.set_title(f"Reference stability ({REF_SIZE}-env references)\n"
             f"mean {diff.mean():.4f}, 95th pct {np.percentile(diff, 95):.4f}")

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Frozen-reference percentile calibration on the SFL environment-generation distribution "
             f"(checkpoint epoch_33, {n} envs)", y=1.02, fontsize=11)
plt.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"percentile_score_distribution.{ext}", dpi=200, bbox_inches="tight")
print("saved percentile_score_distribution.png / .pdf")
