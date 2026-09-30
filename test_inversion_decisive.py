"""Decisive tests for the deployment-inversion proposal.

[A] jaxnav: is the scorer in its TRAINING (inverted) domain still ~a wall
    counter? If yes, Finding 4's characterization and Wave-1's conclusion
    survive the bug. If it encodes more, Wave 1 is confounded.
[B] xland: does the deployment render path + inversion land in the scorer's
    trained domain (logit range / member agreement)?
"""
import sys, os, yaml
import numpy as np
import jax, jax.numpy as jnp
from scipy.stats import spearmanr
REPO = "/home/d/durmusy/Desktop/GIT/updatedjnavwsfl/sampling-for-learnability"
os.chdir(REPO); sys.path.insert(0, REPO + "/sfl/train")
from rlhf_utils import (load_learnability_ensemble, make_member_logit_fn,
                        pairwise_rank_agreement, get_jaxnav_rasterizer)
from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav

def inv_f(imgs_f):
    """Apply the training-pipeline corruption to float [0,1] images."""
    u8 = np.clip(np.round(np.asarray(imgs_f) * 255.0), 0, 255).astype(np.int64)
    return (((256 - u8) % 256).astype(np.float32)) / 255.0

# ---------------- [A] jaxnav ----------------
env_cfg = yaml.safe_load(open("sfl/train/config/env/jaxnav.yaml"))
env = JaxNav(num_agents=1, **env_cfg["env_params"])
ms = env_cfg["env_params"]["map_params"]["map_size"]
render = get_jaxnav_rasterizer(img_height=64, img_width=64, map_height=ms[0], map_width=ms[1], cell_size=1.0)
paths = yaml.safe_load(open("sfl/train/config/jaxnav-sfl.yaml"))["CNN_CHECKPOINT_PATHS"]
gd, st, K = load_learnability_ensemble(paths)
fn = make_member_logit_fn(gd)

@jax.jit
def gen(rng):
    rr = jax.random.split(rng, 500)
    _, es = jax.vmap(env.reset)(rr)
    imgs = jax.vmap(render)(es)
    met = jax.vmap(env.get_env_metrics)(es)
    return imgs, met["passable"], es.map_data.sum(axis=(1, 2))

imgs, passable, walls = gen(jax.random.PRNGKey(11))
imgs = np.asarray(imgs); passable = np.asarray(passable); walls = np.asarray(walls)

def score_np(x_f, bs=100):
    out = []
    for s in range(0, len(x_f), bs):
        out.append(np.asarray(fn(st, jnp.asarray(x_f[s:s+bs]))))
    return np.concatenate(out, axis=1)

St, Si = score_np(imgs), score_np(inv_f(imgs))
print("[A] jaxnav, 500 DR levels — what does the scorer encode in each domain?")
for lab, S in (("TRUE (Wave-1 deployment)", St), ("INVERTED (training domain)", Si)):
    m = S.mean(0)
    print(f"  {lab:26s}: Spearman vs wall count {spearmanr(m, walls).statistic:+.4f}"
          f"   vs passable {spearmanr(m, passable.astype(float)).statistic:+.4f}"
          f"   member agreement {float(pairwise_rank_agreement(jnp.asarray(S))):.4f}")
print(f"  Spearman(TRUE score, INVERTED score) = {spearmanr(St.mean(0), Si.mean(0)).statistic:+.4f}")

# ---------------- [B] xland deployment path ----------------
import xminigrid
from xminigrid.wrappers import GymAutoResetWrapper
from xland_sfl import make_ruleset, build_xland_render_fn
xenv_cfg = yaml.safe_load(open("sfl/train/config/env/xland.yaml"))
xe, xp = xminigrid.make(xenv_cfg["env_id"]); xe = GymAutoResetWrapper(xe)
xp = xp.replace(ruleset=make_ruleset("difficult"))
xrender = build_xland_render_fn(tile_size=32, view_size=7, target_size=200)
xpaths = yaml.safe_load(open("sfl/train/config/xland-sfl.yaml"))["CNN_CHECKPOINT_PATHS"]
xgd, xst, xK = load_learnability_ensemble(xpaths)
xfn = make_member_logit_fn(xgd)

rr = jax.random.split(jax.random.PRNGKey(11), 150)
ts = jax.vmap(xe.reset, in_axes=(None, 0))(xp, rr)
ximgs = np.asarray(jax.vmap(xrender)(ts.state.grid, ts.state.agent))

def xscore(x_f, bs=25):
    out = []
    for s in range(0, len(x_f), bs):
        out.append(np.asarray(xfn(xst, jnp.asarray(x_f[s:s+bs]))))
    return np.concatenate(out, axis=1)

print("\n[B] xland, 150 DR levels through the deployment renderer")
for lab, X in (("TRUE (current deployment)", ximgs), ("INVERTED (proposed fix)", inv_f(ximgs))):
    S = xscore(X); m = S.mean(0)
    print(f"  {lab:26s}: ens-logit range [{m.min():.1f}, {m.max():.1f}]"
          f"   member agreement {float(pairwise_rank_agreement(jnp.asarray(S))):.4f}")
print("  reference (scorer's dataset domain): range about [-9, +2], agreement 0.97")
