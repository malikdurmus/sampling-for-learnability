"""Would inverting images at deployment cancel the fast_data bug?

The corruption is g(x) = (-x) mod 256 (an involution), so feeding g(image)
to the scorer should put it back in its training domain. Tests:
  [1] jaxnav scorer: pair accuracy on TRUE vs INVERTED dataset images
      -> was the Wave-0/1 deployed scorer degraded by being fed true images?
  [2] xland deployment path: render DR levels -> g -> score; does the logit
      range / member agreement match the scorer's dataset (inverted) domain?
  [3] resize sensitivity: g amplifies near-black differences (0->0 but
      1->255). Compare g(PIL resize) vs g(jax resize) logits.
Run: JAX_PLATFORMS=cpu .venv/bin/python test_inversion_fix.py
"""
import sys, yaml
import numpy as np
import jax, jax.numpy as jnp
from scipy.stats import spearmanr
REPO = "/home/d/durmusy/Desktop/GIT/updatedjnavwsfl/sampling-for-learnability"
import os; os.chdir(REPO)
sys.path.insert(0, REPO + "/sfl/train")
from rlhf_utils import load_learnability_ensemble, make_member_logit_fn, pairwise_rank_agreement

def invert_u8(u8):
    """The exact corruption the training pipeline applied: 255*x mod 256."""
    return ((256 - u8.astype(np.int64)) % 256).astype(np.uint8)

d = np.load("xland_test_pairs_500.npz")   # TRUE dataset images (verified correct read path)
i0, i1, pref = d["i0"], d["i1"], d["pref"]

# sanity: g is an involution, and g(true) reproduces the corrupted read path
assert np.array_equal(invert_u8(invert_u8(i0[:5])), i0[:5]), "g must be an involution"
print("g(g(x)) == x  (involution): OK")

xcfg = yaml.safe_load(open("sfl/train/config/xland-sfl.yaml"))
jcfg = yaml.safe_load(open("sfl/train/config/jaxnav-sfl.yaml"))

def score_members(paths, imgs_u8, bs=50):
    gd, st, K = load_learnability_ensemble(paths)
    fn = make_member_logit_fn(gd)
    out = []
    for s in range(0, len(imgs_u8), bs):
        x = jnp.asarray(imgs_u8[s:s+bs], dtype=jnp.float32) / 255.0
        out.append(np.asarray(fn(st, x)))
    return np.concatenate(out, axis=1)   # (K, N)

def pair_acc(paths, a_u8, b_u8, prefs):
    S = score_members(paths, np.concatenate([a_u8, b_u8]))
    n = len(a_u8)
    s0, s1 = S[:, :n], S[:, n:]
    ens = ((s1.mean(0) - s0.mean(0)) > 0).astype(np.float32)
    return float((ens == prefs).mean())

N = 150
print(f"\n[2] XLAND scorer, {N} dataset pairs")
print(f"  TRUE images     : pair acc {pair_acc(xcfg['CNN_CHECKPOINT_PATHS'], i0[:N], i1[:N], pref[:N]):.4f}")
print(f"  INVERTED (g)    : pair acc {pair_acc(xcfg['CNN_CHECKPOINT_PATHS'], invert_u8(i0[:N]), invert_u8(i1[:N]), pref[:N]):.4f}"
      f"   (full-split reference: 0.8048)")
