"""Empirical check: distribution of CNN difficulty scores on SFL-generated environments.

Replicates the exact deployment path of sfl/train/jaxnav_sfl.py with the
deployed scorer (the 5-member aug deep ensemble; checkpoint paths are read
from sfl/train/config/jaxnav-sfl.yaml CNN_CHECKPOINT_PATHS so this script can
never drift from the deployed configuration):
  - environments sampled from the same DR generator (env.reset)
  - rendered with the same 64x64 rasterizer
  - scored by the same ensemble loader; score = mean of member logits
  - same volume as one SFL selection cycle: BATCH_SIZE=1000 * NUM_BATCHES=5 = 5000
"""

import argparse
import sys
import yaml

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "sfl/train")
from rlhf_utils import (load_learnability_ensemble, make_member_logit_fn,
                        pairwise_rank_agreement, resolve_input_domain,
                        get_jaxnav_rasterizer)
from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav

_ap = argparse.ArgumentParser()
_ap.add_argument("--ckpt", nargs="+", default=None,
                 help="checkpoint dir(s); default: CNN_CHECKPOINT_PATHS from jaxnav-sfl.yaml")
_ap.add_argument("--out", default="cnn_score_distribution",
                 help="output basename (writes <out>.png and <out>.pdf)")
_ap.add_argument("--label", default=None,
                 help="scorer label for the figure title; default: deployed ensemble / ckpt name")
_ap.add_argument("--invert", choices=["auto", "on", "off"], default="auto",
                 help="input-domain handling: auto = look up input_domain in "
                      "CHECKPOINT_INDEX.json (fails for out-of-index checkpoints); "
                      "on/off = explicit override (use off for legacy true-domain "
                      "checkpoints that predate the index)")
_args = _ap.parse_args()

sfl_cfg = yaml.safe_load(open("sfl/train/config/jaxnav-sfl.yaml"))
CKPTS = _args.ckpt if _args.ckpt else sfl_cfg["CNN_CHECKPOINT_PATHS"]
BATCH_SIZE = 1000
NUM_BATCHES = 5
CNN_IMG_SIZE = 64
SEED = 0

env_cfg = yaml.safe_load(open("sfl/train/config/env/jaxnav.yaml"))
env = JaxNav(num_agents=1, **env_cfg["env_params"])  # num_agents=1 as in jaxnav-sfl.yaml

map_size = env_cfg["env_params"]["map_params"]["map_size"]
render_fn = get_jaxnav_rasterizer(
    img_height=CNN_IMG_SIZE, img_width=CNN_IMG_SIZE,
    map_height=map_size[0], map_width=map_size[1], cell_size=1.0,
)

graphdef, state, K = load_learnability_ensemble(CKPTS)
if _args.invert == "auto":
    _domain = resolve_input_domain(CKPTS)
    INVERT = _domain == "inverted"
else:
    INVERT = _args.invert == "on"
    _domain = "inverted" if INVERT else "true"
print(f"input domain: {_domain} (invert_input={INVERT})")
member_fn = make_member_logit_fn(graphdef, invert_input=INVERT)


@jax.jit
def score_batch(rng):
    reset_rng = jax.random.split(rng, BATCH_SIZE)
    _, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
    CHUNK = 100
    chunked = jax.tree.map(lambda x: x.reshape((BATCH_SIZE // CHUNK, CHUNK) + x.shape[1:]), env_state)

    def chunk_fn(carry, env_chunk):
        imgs = jax.vmap(render_fn)(env_chunk)
        return carry, member_fn(state, imgs)          # (K, CHUNK) raw member logits

    _, member_chunked = jax.lax.scan(chunk_fn, None, chunked)
    return jnp.moveaxis(member_chunked, 1, 0).reshape((K, BATCH_SIZE))


rng = jax.random.PRNGKey(SEED)
M = np.concatenate([np.asarray(score_batch(r)) for r in jax.random.split(rng, NUM_BATCHES)], axis=1)  # (K, n)
logits = M.mean(axis=0)                               # deployed ensemble score: mean of member logits
sig = np.asarray(jax.nn.sigmoid(jnp.asarray(logits)))
n = len(sig)

ref = np.sort(logits)
pct = np.interp(logits, ref, np.linspace(0.0, 1.0, n))

# ---- Stats ----
exactly_1 = int((sig >= 1.0).sum())
exactly_0 = int((sig <= 0.0).sum())
below_001 = int((sig < 0.01).sum())
above_099 = int((sig > 0.99).sum())
sat_99 = below_001 + above_099
mid = n - sat_99
rank_agree = float(pairwise_rank_agreement(jnp.asarray(M)))

print(f"n = {n} environments (DR-generated, SFL deployment path)")
print(f"scorer: K={K} (mean of member logits)" if K > 1 else "scorer: single checkpoint (K=1)")
for p, mu in zip(CKPTS, M.mean(axis=1)):
    print(f"  member {'/'.join(p.rstrip('/').split('/')[-2:])}: mean logit {mu:+.1f}")
if K > 1:  # member diagnostics are undefined for a single checkpoint
    print(f"mean pairwise Spearman rank agreement between members: {rank_agree:.4f}")
    print(f"mean per-level member std (raw logits): {M.std(axis=0).mean():.2f}")
print(f"sigmoid: mean={sig.mean():.4f} std={sig.std():.4f} median={np.median(sig):.4f}")
print(f"exactly 1.0 (float32): {exactly_1} ({100*exactly_1/n:.1f}%)")
print(f"exactly 0.0 (float32): {exactly_0} ({100*exactly_0/n:.1f}%)")
print(f"below 0.01:            {below_001} ({100*below_001/n:.1f}%)")
print(f"above 0.99:            {above_099} ({100*above_099/n:.1f}%)")
print(f"outside [0.01, 0.99]:  {sat_99} ({100*sat_99/n:.1f}%)")
print(f"usable mid-range [0.01, 0.99]: {mid} ({100*mid/n:.1f}%)")
print(f"ensemble logits: min={logits.min():.2f} max={logits.max():.2f} span={logits.max()-logits.min():.1f}")

# Curriculum coverage: candidates within +-0.05 of each target_mu
mus = np.linspace(0.0, 1.0, 101)
cov_sig = np.array([((sig >= m - 0.05) & (sig <= m + 0.05)).sum() for m in mus])
cov_pct = np.array([((pct >= m - 0.05) & (pct <= m + 0.05)).sum() for m in mus])
uniform_ref = n * 0.1  # a uniform score distribution puts ~10% in each +-0.05 band
print(f"coverage at target_mu=0.5: sigmoid {cov_sig[50]} vs percentile {cov_pct[50]} envs "
      f"(uniform reference: {uniform_ref:.0f})")

# ---- Plot ----
fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

ax = axes[0]
ax.hist(sig, bins=50, range=(0, 1), color="#4878b0", edgecolor="white", linewidth=0.4)
ax.set_xlabel("CNN difficulty score (sigmoid of ensemble-mean logit)")
ax.set_ylabel("environments")
ax.set_title(f"Sigmoid scores of {n} DR-sampled environments\n"
             f"{100*sat_99/n:.1f}% outside [0.01, 0.99] "
             f"({100*below_001/n:.1f}% below 0.01, {100*above_099/n:.1f}% above 0.99)")

ax = axes[1]
ax.hist(logits, bins=50, color="#8a6fb8", edgecolor="white", linewidth=0.4)
ax.axvspan(-5, 5, color="orange", alpha=0.18,
           label="sigmoid discriminative range (|logit| < 5)")
ax.set_xlabel("raw ensemble score (mean of member logits)")
ax.set_ylabel("environments")
ax.set_title(f"Raw scores span {logits.max()-logits.min():.0f} logits;\n"
             "sigmoid only discriminates in the shaded band")
ax.legend(fontsize=8, loc="upper right")

ax = axes[2]
ax.plot(mus, cov_sig, color="#c0504d", linewidth=1.8, label="sigmoid normalization")
ax.plot(mus, cov_pct, color="#4878b0", linewidth=1.8, label="percentile normalization")
ax.axhline(uniform_ref, color="gray", linestyle="--", linewidth=1.2,
           label="uniform score distribution (reference)")
ax.set_xlabel(r"curriculum target $\mu$")
ax.set_ylabel(r"environments with score within $\mu \pm 0.05$")
ax.set_title("Curriculum coverage: candidates available\nper target difficulty")
ax.legend(fontsize=8)

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)

scorer_label = _args.label or (f"deployed aug ensemble, K={K}" if len(CKPTS) > 1
               else "/".join(CKPTS[0].rstrip("/").split("/")[-2:]))
fig.suptitle("CNN difficulty-score distribution on the SFL environment-generation distribution "
             f"({scorer_label}, {n} envs)", y=1.02, fontsize=11)
plt.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{_args.out}.{ext}", dpi=200, bbox_inches="tight")
print(f"saved {_args.out}.png / .pdf")
