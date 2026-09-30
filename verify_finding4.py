"""Independent verification of Finding 4's claims (cross-check of a parallel
analysis). Fresh 4000 DR levels for the agent-independent claims; the
recorded probe_coldstart.npz for the cold-start claims.

Writes verify_finding4_output.txt.
"""
import sys, yaml
import numpy as np
import jax, jax.numpy as jnp
from scipy.stats import spearmanr

sys.path.insert(0, "sfl/train")
from rlhf_utils import load_learnability_ensemble, make_member_logit_fn, get_jaxnav_rasterizer
from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav

lines = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); lines.append(s)

cfg = yaml.safe_load(open("sfl/train/config/jaxnav-sfl.yaml"))
env_cfg = yaml.safe_load(open("sfl/train/config/env/jaxnav.yaml"))
env = JaxNav(num_agents=1, **env_cfg["env_params"])
ms = env_cfg["env_params"]["map_params"]["map_size"]
render = get_jaxnav_rasterizer(img_height=64, img_width=64, map_height=ms[0],
                               map_width=ms[1], cell_size=1.0)
gd, st, K = load_learnability_ensemble(cfg["CNN_CHECKPOINT_PATHS"])
member_fn = make_member_logit_fn(gd)

@jax.jit
def probe(rng):
    rr = jax.random.split(rng, 1000)
    _, es = jax.vmap(env.reset)(rr)
    ch = jax.tree.map(lambda x: x.reshape((10, 100) + x.shape[1:]), es)
    _, m = jax.lax.scan(lambda c, b: (c, member_fn(st, jax.vmap(render)(b))), None, ch)
    logits = jnp.moveaxis(m, 1, 0).reshape((K, 1000)).mean(0)
    met = jax.vmap(env.get_env_metrics)(es)
    walls = es.map_data.sum(axis=(1, 2)) if es.map_data.ndim == 3 else es.map_data.sum(axis=(1,2,3))
    return logits, met["passable"], met["shortest_path_length_mean"], walls

L, PA, SPL, W = [], [], [], []
for r in jax.random.split(jax.random.PRNGKey(123), 4):
    l, pa, spl, w = probe(r)
    L.append(np.asarray(l)); PA.append(np.asarray(pa)); SPL.append(np.asarray(spl)); W.append(np.asarray(w))
L, PA, SPL, W = map(np.concatenate, (L, PA, SPL, W))
c = np.argsort(np.argsort(L)) / (len(L) - 1)

p(f"[fresh probe] n={len(L)} levels, seed 123 (independent of Finding 4's sample)")
p(f"Spearman(CNN, wall count)        = {spearmanr(L, W).statistic:+.4f}   (claimed +0.9954)")
p(f"Spearman(CNN, passable)          = {spearmanr(L, PA.astype(float)).statistic:+.4f}   (claimed -0.5177)")
p(f"Spearman(CNN, shortest path len) = {spearmanr(L, SPL).statistic:+.4f}   (claimed -0.0969)")
# partial corr(CNN, passable | walls) via rank residuals
def ranks(x): return np.argsort(np.argsort(x)).astype(float)
rl, rw, rp = ranks(L), ranks(W), ranks(PA.astype(float) + 1e-9*np.arange(len(PA)))
def resid(y, x):
    A = np.vstack([x, np.ones_like(x)]).T
    return y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
pc = np.corrcoef(resid(rl, rw), resid(rp, rw))[0, 1]
p(f"partial corr(CNN, passable|walls)= {pc:+.4f}   (claimed -0.1139)")

dec = np.clip((c * 10).astype(int), 0, 9)
dec_pass = [PA[dec == i].mean() for i in range(10)]
p(f"passability by CNN decile: {[round(float(x),3) for x in dec_pass]}   overall {PA.mean():.3f}")
for mu in (0.3, 0.5, 0.7, 1.0):
    sel = np.argsort(np.abs(c - mu))[:800]  # 20% of 4000, matching deployment ratio
    p(f"passable share of mu={mu} selected buffer: {PA[sel].mean():.3f}")

p("\n[npz cold-start data] (the recorded untrained-policy probe itself)")
d = np.load("probe_coldstart.npz")
cp, pp, ss, ne = d["c"], d["p"], d["s"], d["n_eps"]
p(f"never solved: {np.mean(d['n_goal']==0):.1%}   (claimed 91.4%)")
p(f"pooled success rate: {(d['n_goal'].sum()/ne.sum()):.4f}   (claimed 0.0127)")
p(f"median episodes/level: {np.median(ne):.0f}, range {ne.min():.0f}-{ne.max():.0f}   (claimed 10, 2-68)")
p(f"Spearman(CNN percentile, p): {spearmanr(cp, pp).statistic:+.4f}   (claimed -0.077)")
p(f"Spearman(CNN percentile, n_eps): {spearmanr(cp, ne).statistic:+.4f}   (episode-difficulty confound)")
p(f"mean p by CNN decile: {[round(float(pp[(cp>=i/10)&(cp<(i+1)/10)].mean()),4) for i in range(10)]}")

p("\n[buffer-composition test of the '0.0% unsolvable SFL buffer' claim]")
# Cold-start SFL selection: top 20% by observed s (deployment ratio NUM_TO_SAVE/candidates).
# Passability of npz levels is unknown, so estimate via the percentile->passability
# curve measured on the fresh 4000 (smooth in percentile).
from numpy import interp
grid = (np.arange(10) + 0.5) / 10
pass_of_c = interp(cp, grid, dec_pass)
k = 200  # 20% of 1000
order = np.argsort(-ss, kind="stable")
sel = order[:k]
n_pos = int((ss > 0).sum())
p(f"levels with s>0 at cold start: {n_pos}/1000 -> only {n_pos}/{k} buffer slots have evidence")
p(f"expected passable share of top-{k} SFL buffer: {pass_of_c[sel].mean():.3f}")
p(f"expected passable share of the s>0 subset only: {pass_of_c[ss>0].mean():.3f}")
p(f"expected passable share of the zero-s tail portion: {pass_of_c[order[n_pos:k]].mean():.3f}")

open("verify_finding4_output.txt", "w").write("\n".join(lines) + "\n")
print("\nwrote verify_finding4_output.txt")
