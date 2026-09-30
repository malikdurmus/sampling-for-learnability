"""Cold-start probe: does the CNN difficulty score predict the agent's success?

Replicates one SFL scoring cycle exactly (env.reset -> rollout -> p per level,
plus the 64x64 rasterizer -> ensemble -> percentile CNN score), for either an
UNTRAINED policy (the cold-start regime) or a saved trained policy.

Per level it records:
  c  : CNN difficulty percentile in [0,1]
  p  : empirical success rate over the episodes seen in the rollout
  n  : number of completed episodes that p is estimated from
  s  : SFL score p(1-p)

Answers: (Q1) is c predictive of p at cold start, (Q2) how badly is p
quantized by the episode count, (Q3) what would alternative selection rules
pick.  Usage:
  python coldstart_probe.py --levels 1000 --rollout 1000 [--policy PATH] --out FILE
"""
import argparse, os, sys, pickle
import numpy as np
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--levels", type=int, default=1000)
ap.add_argument("--rollout", type=int, default=1000)
ap.add_argument("--chunk", type=int, default=250, help="levels per rollout chunk")
ap.add_argument("--policy", default=None, help="model.safetensors of a trained policy")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="coldstart_probe.npz")
args = ap.parse_args()

import jax, jax.numpy as jnp
from flax.training.train_state import TrainState
import optax
sys.path.insert(0, "sfl/train")
from rlhf_utils import (load_learnability_ensemble, make_member_logit_fn,
                        get_jaxnav_rasterizer)
from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav
from sfl.train.common.network import ActorCriticRNN, ScannedRNN

cfg = yaml.safe_load(open("sfl/train/config/jaxnav-sfl.yaml"))
env_cfg = yaml.safe_load(open("sfl/train/config/env/jaxnav.yaml"))
lrn_cfg = yaml.safe_load(open("sfl/train/config/learning/ippo-jaxnav.yaml"))
lrn_cfg["NUM_ENVS"] = args.chunk

env = JaxNav(num_agents=1, **env_cfg["env_params"])
ms = env_cfg["env_params"]["map_params"]["map_size"]
render = get_jaxnav_rasterizer(img_height=64, img_width=64,
                               map_height=ms[0], map_width=ms[1], cell_size=1.0)
gd, st, K = load_learnability_ensemble(cfg["CNN_CHECKPOINT_PATHS"])
member_fn = make_member_logit_fn(gd)

network = ActorCriticRNN(env.agent_action_space().shape[0], config=lrn_cfg)
rng = jax.random.PRNGKey(args.seed)
rng, _rng = jax.random.split(rng)
init_x = (jnp.zeros((1, args.chunk, env.lidar_num_beams + 5)), jnp.zeros((1, args.chunk)))
init_h = ScannedRNN.initialize_carry(args.chunk, lrn_cfg["HIDDEN_SIZE"])
params = network.init(_rng, init_h, init_x)
if args.policy:
    from safetensors.flax import load_file
    from flax.traverse_util import unflatten_dict
    flat = load_file(args.policy)
    params = unflatten_dict({tuple(k.split(",")): v for k, v in flat.items()})
    print(f"loaded trained policy from {args.policy}")
else:
    print("using UNTRAINED (cold-start) policy")

def batchify(x, agents, n):
    return jnp.stack([x[a] for a in agents]).reshape((n, -1))
def unbatchify(x, agents, nenv, nact):
    x = x.reshape((nact, nenv, -1)); return {a: x[i] for i, a in enumerate(agents)}

@jax.jit
def rollout_chunk(rng):
    """One chunk: reset `chunk` levels, roll out, return (per-level p, n_eps, cnn c)."""
    B = args.chunk
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, B)
    obsv, env_state = jax.vmap(env.reset)(reset_rng)
    imgs = jax.vmap(render)(env_state)
    logits = member_fn(st, imgs).mean(axis=0)          # ensemble-mean raw logit

    def _step(carry, _):
        env_state, start_state, last_obs, last_done, hstate, rng = carry
        rng, _rng = jax.random.split(rng)
        obs_b = batchify(last_obs, env.agents, B)
        hstate, pi, value, _ = network.apply(params, hstate, (obs_b[np.newaxis, :], last_done[np.newaxis, :]))
        action = pi.sample(seed=_rng)
        env_act = {k: v.squeeze() for k, v in unbatchify(action, env.agents, B, env.num_agents).items()}
        rng, _rng = jax.random.split(rng)
        obsv, env_state, reward, done, info = jax.vmap(env.step)(
            jax.random.split(_rng, B), env_state, env_act, start_state)
        done_b = batchify(done, env.agents, B).squeeze()
        return (env_state, start_state, obsv, done_b, hstate, rng), (done_b, info)

    h0 = ScannedRNN.initialize_carry(B, lrn_cfg["HIDDEN_SIZE"])
    carry = (env_state, env_state, obsv, jnp.zeros(B, dtype=bool), h0, rng)
    _, (dones, info) = jax.lax.scan(_step, carry, None, args.rollout)

    # per level: count completed episodes and goal-reaching episodes
    goal = info["GoalR"].reshape(args.rollout, B)
    ep_end = dones.reshape(args.rollout, B)
    n_eps = ep_end.sum(axis=0)                     # completed episodes per level
    n_goal = (goal > 0).sum(axis=0)                # successful episodes per level
    return logits, n_goal, n_eps

n_chunks = args.levels // args.chunk
L, G, N = [], [], []
for i in range(n_chunks):
    rng, k = jax.random.split(rng)
    lo, g, n = rollout_chunk(k)
    L.append(np.asarray(lo)); G.append(np.asarray(g)); N.append(np.asarray(n))
    print(f"  chunk {i+1}/{n_chunks} done", flush=True)
logits = np.concatenate(L); n_goal = np.concatenate(G).astype(float); n_eps = np.concatenate(N).astype(float)

p = np.where(n_eps > 0, n_goal / np.maximum(n_eps, 1), 0.0)
s = p * (1 - p)
c = np.argsort(np.argsort(logits)) / (len(logits) - 1)     # percentile CNN score
np.savez(args.out, logits=logits, c=c, p=p, s=s, n_eps=n_eps, n_goal=n_goal)
print(f"\nsaved {args.out}  |  levels={len(p)}")
print(f"episodes per level: mean {n_eps.mean():.1f}  median {np.median(n_eps):.0f}"
      f"  min {n_eps.min():.0f}  max {n_eps.max():.0f}")
print(f"p: mean {p.mean():.4f}  frac(p==0) {np.mean(p==0):.3f}  frac(p==1) {np.mean(p==1):.3f}")
print(f"s: mean {s.mean():.4f}  frac(s==0) {np.mean(s==0):.3f}  max {s.max():.3f}")
print(f"corr(c, p)  Pearson {np.corrcoef(c,p)[0,1]:+.4f}")
from scipy.stats import spearmanr
print(f"corr(c, p)  Spearman {spearmanr(c,p).statistic:+.4f}")
