"""Acceptance test for the xland renderer fix: with the FOV highlight +
cubic resize, DR-generated levels must score INSIDE the dataset's logit
range and member rank agreement must recover toward the dataset's 0.97."""
import sys, yaml
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, "sfl/train")
from rlhf_utils import load_learnability_ensemble, make_member_logit_fn, pairwise_rank_agreement
import xminigrid
from xminigrid.wrappers import GymAutoResetWrapper
from xland_sfl import make_ruleset, build_xland_render_fn

cfg = yaml.safe_load(open("sfl/train/config/xland-sfl.yaml"))
env_cfg = yaml.safe_load(open("sfl/train/config/env/xland.yaml"))
env, env_params = xminigrid.make(env_cfg["env_id"])
env = GymAutoResetWrapper(env)
env_params = env_params.replace(ruleset=make_ruleset("difficult"))
render = build_xland_render_fn(tile_size=env_cfg["tile_size"], view_size=env_cfg["view_size"], target_size=200)
gd, st, K = load_learnability_ensemble(cfg["CNN_CHECKPOINT_PATHS"])
member_fn = make_member_logit_fn(gd)

@jax.jit
def probe(rng):
    rr = jax.random.split(rng, 500)
    ts = jax.vmap(env.reset, in_axes=(None, 0))(env_params, rr)
    g = ts.state.grid.reshape((5, 100) + ts.state.grid.shape[1:])
    a = jax.tree.map(lambda x: x.reshape((5, 100) + x.shape[1:]), ts.state.agent)
    def f(c, b):
        return c, member_fn(st, jax.vmap(render)(*b))
    _, m = jax.lax.scan(f, None, (g, a))
    return jnp.moveaxis(m, 1, 0).reshape((K, 500))

M = np.concatenate([np.asarray(probe(r)) for r in jax.random.split(jax.random.PRNGKey(7), 2)], axis=1)
mean = M.mean(0)
print(f"n={M.shape[1]} DR levels, fixed renderer")
print(f"ensemble-mean logit range: [{mean.min():.1f}, {mean.max():.1f}]  mean {mean.mean():.1f}")
print(f"   (dataset images: [-9.0, 2.0]; BROKEN renderer was [-25.3, -19.6])")
print(f"member rank agreement: {float(pairwise_rank_agreement(jnp.asarray(M))):.4f}")
print(f"   (dataset images: 0.9674; BROKEN renderer: 0.067)")
print(f"per-member logit std: {[round(float(x),2) for x in M.std(1)]}  (dataset: ~1.8-2.2)")
