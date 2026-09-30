"""Was the jaxnav Wave-0/1 deployed scorer degraded by the inversion bug?
It was TRAINED on inverted images but DEPLOYED on true ones."""
import sys, os, yaml
import numpy as np
import jax.numpy as jnp
from scipy.stats import spearmanr
REPO = "/home/d/durmusy/Desktop/GIT/updatedjnavwsfl/sampling-for-learnability"
os.chdir(REPO); sys.path.insert(0, REPO + "/sfl/train")
from rlhf_utils import load_learnability_ensemble, make_member_logit_fn, pairwise_rank_agreement

def inv(u8):
    return ((256 - u8.astype(np.int64)) % 256).astype(np.uint8)

d = np.load("jaxnav_test_pairs_800.npz")
i0, i1, pref = d["i0"], d["i1"], d["pref"]
paths = yaml.safe_load(open("sfl/train/config/jaxnav-sfl.yaml"))["CNN_CHECKPOINT_PATHS"]
gd, st, K = load_learnability_ensemble(paths)
fn = make_member_logit_fn(gd)

def score(imgs, bs=200):
    out = []
    for s in range(0, len(imgs), bs):
        out.append(np.asarray(fn(st, jnp.asarray(imgs[s:s+bs], dtype=jnp.float32) / 255.0)))
    return np.concatenate(out, axis=1)

for label, a, b in [("TRUE (what Wave 0/1 deployed)", i0, i1),
                    ("INVERTED (scorer's training domain)", inv(i0), inv(i1))]:
    S = score(np.concatenate([a, b]))
    n = len(a); s0, s1 = S[:, :n], S[:, n:]
    acc = float((((s1.mean(0) - s0.mean(0)) > 0).astype(np.float32) == pref).mean())
    M = np.concatenate([s0, s1], axis=1)
    print(f"jaxnav {label:36s}: pair acc {acc:.4f}  member agreement {float(pairwise_rank_agreement(jnp.asarray(M))):.4f}"
          f"  ens-logit range [{M.mean(0).min():.1f}, {M.mean(0).max():.1f}]")
print("  (full-split reference for the INVERTED/training domain: 0.9000)")

# do the two domains even rank levels the same way?
St = score(i0[:400]); Si = score(inv(i0[:400]))
print(f"\nSpearman(ensemble score on TRUE, on INVERTED) over 400 levels: "
      f"{spearmanr(St.mean(0), Si.mean(0)).statistic:+.4f}")
