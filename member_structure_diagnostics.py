"""Member-structure diagnostics for the deployed jaxnav scorer ensemble.

THE generating artifact for every member-level number cited in
THESIS_FINDINGS.md (Finding 1 mechanism, its corollaries, the z-scoring
justification, and the Finding-2 retraction). Consolidates the one-off
diagnostics run on 2026-08-28 into one reproducible script.

Scores 5000 DR-generated levels (BATCH_SIZE=1000 x NUM_BATCHES=5, PRNG seed
0 — one SFL selection cycle) through the exact deployment path
(env.reset -> 64x64 rasterizer -> ensemble members from
jaxnav-sfl.yaml CNN_CHECKPOINT_PATHS), then reports:
  1. per-member logit mean/std (offset & scale)
  2. pairwise Pearson vs Spearman (affine-model test) + OLS slopes
  3. offset-removal test (how much raw spread the offsets explain)
  4. variance share per member under raw-logit-mean combination
  5. combination rules (raw / z-scored / rank): ranking agreement,
     uncertainty-signal contamination R^2(spread ~ score)
  6. effect of z-scoring on sigmoid and percentile normalization
  7. Finding-2 re-test: disagreement of selected vs all levels, per rule/mu

Usage:  .venv/bin/python member_structure_diagnostics.py
        (writes member_structure_diagnostics_output.txt next to itself)
"""
import sys, yaml
import numpy as np
import jax, jax.numpy as jnp
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, "sfl/train")
from rlhf_utils import (load_learnability_ensemble, make_member_logit_fn,
                        pairwise_rank_agreement, get_jaxnav_rasterizer)
from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav

OUT = "member_structure_diagnostics_output.txt"
BATCH_SIZE, NUM_BATCHES, SEED = 1000, 5, 0

cfg = yaml.safe_load(open("sfl/train/config/jaxnav-sfl.yaml"))
env_cfg = yaml.safe_load(open("sfl/train/config/env/jaxnav.yaml"))
paths = cfg["CNN_CHECKPOINT_PATHS"]
names = [p.rstrip("/").split("/")[-2].replace("ens-jaxnav-aug-", "") for p in paths]

env = JaxNav(num_agents=1, **env_cfg["env_params"])
ms = env_cfg["env_params"]["map_params"]["map_size"]
render = get_jaxnav_rasterizer(img_height=64, img_width=64,
                               map_height=ms[0], map_width=ms[1], cell_size=1.0)
gd, st, K = load_learnability_ensemble(paths)
member_fn = make_member_logit_fn(gd)

@jax.jit
def score_batch(rng):
    rr = jax.random.split(rng, BATCH_SIZE)
    _, es = jax.vmap(env.reset)(rr)
    ch = jax.tree.map(lambda x: x.reshape((10, BATCH_SIZE // 10) + x.shape[1:]), es)
    _, m = jax.lax.scan(lambda c, b: (c, member_fn(st, jax.vmap(render)(b))), None, ch)
    return jnp.moveaxis(m, 1, 0).reshape((K, BATCH_SIZE))

M = np.asarray(jnp.concatenate(
    [score_batch(r) for r in jax.random.split(jax.random.PRNGKey(SEED), NUM_BATCHES)], axis=1))
n = M.shape[1]

lines = []
def p(*a):
    s = " ".join(str(x) for x in a)
    print(s); lines.append(s)

p(f"member_structure_diagnostics | n={n} levels | K={K} | seed={SEED}")
p(f"checkpoints: {paths}")

p("\n[1] per-member logit mean (offset) / std (scale)")
for i in range(K):
    p(f"  {names[i]:<8} mean {M[i].mean():+8.2f}   std {M[i].std():6.2f}")

p("\n[2] pairwise affine-model test: Pearson (linear) vs Spearman (monotone)")
ps, ss, slopes = [], [], []
for i in range(K):
    for j in range(i + 1, K):
        pr = pearsonr(M[i], M[j]).statistic
        sr = spearmanr(M[i], M[j]).statistic
        sl = np.polyfit(M[j], M[i], 1)[0]
        ps.append(pr); ss.append(sr); slopes.append(sl)
        p(f"  {names[i]}-{names[j]:<8} Pearson {pr:.4f}  Spearman {sr:.4f}  OLS slope {sl:8.3f}")
p(f"  MEAN Pearson {np.mean(ps):.4f}  MEAN Spearman {np.mean(ss):.4f}"
  f"  max |gap| {max(abs(np.array(ss) - np.array(ps))):.4f}")
p(f"  overall mean pairwise Spearman (pairwise_rank_agreement): "
  f"{float(pairwise_rank_agreement(jnp.asarray(M))):.4f}")

p("\n[3] offset-removal test")
Md = M - M.mean(1, keepdims=True)
p(f"  per-level member std: raw {M.std(0).mean():.2f} -> demeaned {Md.std(0).mean():.2f}"
  f"  (std reduction {100 * (1 - Md.std(0).mean() / M.std(0).mean()):.0f}%,"
  f" variance reduction {100 * (1 - (Md.std(0).mean() / M.std(0).mean()) ** 2):.0f}%)")

raw = M.mean(0)
p("\n[4] variance share per member under raw-logit-mean combination")
var = M.var(1)
p("  " + "  ".join(f"{names[k]}:{100 * var[k] / var.sum():5.1f}%" for k in range(K)))
p(f"  Spearman(raw-mean ensemble, widest member {names[int(np.argmax(var))]}): "
  f"{spearmanr(raw, M[int(np.argmax(var))]).statistic:.4f}")

def pctl(v):
    return np.argsort(np.argsort(v)) / (len(v) - 1)

Mz = (M - M.mean(1, keepdims=True)) / M.std(1, keepdims=True)
Mr = np.stack([pctl(M[i]) for i in range(K)])
rules = {"raw": M, "z": Mz, "rank": Mr}
scores = {k: X.mean(0) for k, X in rules.items()}
spreads = {k: X.std(0) for k, X in rules.items()}

p("\n[5] combination rules: ranking agreement & uncertainty contamination")
for a in rules:
    for b in rules:
        if a < b:
            p(f"  Spearman(score_{a}, score_{b}) = {spearmanr(scores[a], scores[b]).statistic:.5f}")
for k in rules:
    r2 = np.corrcoef(spreads[k], scores[k])[0, 1] ** 2
    p(f"  R^2(spread_{k} ~ score_{k}) = {r2:.3f}"
      + ("   <- raw spread is a restatement of the score" if k == "raw" else ""))

p("\n[6] normalization behaviour: raw-mean vs z-mean combination")
for lab, v in [("raw", raw), ("z", scores["z"])]:
    s = 1 / (1 + np.exp(-v))
    p(f"  sigmoid({lab}-mean): frac<0.01 {np.mean(s < 0.01):.3f}"
      f"  frac in [0.1,0.9] {np.mean((s >= 0.1) & (s <= 0.9)):.3f}  max {s.max():.3f}")
pr_, pz = pctl(raw), pctl(scores["z"])
p(f"  percentile: Spearman(raw, z) = {spearmanr(pr_, pz).statistic:.5f};"
  f"  frac |shift|>0.05 = {np.mean(np.abs(pr_ - pz) > 0.05):.4f}")
for mu in (0.0, 0.5):
    sr_ = set(np.argsort(np.abs(pr_ - mu))[:1000]); sz_ = set(np.argsort(np.abs(pz - mu))[:1000])
    p(f"  selected-set overlap (1000 closest to mu={mu}): {len(sr_ & sz_) / 1000:.3f}")

p("\n[7] Finding-2 re-test: spread(selected)/spread(all), selection in percentile space")
for k in rules:
    sc, sd = scores[k], spreads[k]
    pc = pctl(sc)
    for mu in (0.0, 0.5):
        sel = np.argsort(np.abs(pc - mu))[:1000]
        p(f"  rule={k:<5} mu={mu:<4} ratio = {sd[sel].mean() / sd.mean():5.2f}"
          f"   ({sd[sel].mean():.4g} vs {sd.mean():.4g})")

open(OUT, "w").write("\n".join(lines) + "\n")
print(f"\nwrote {OUT}")
