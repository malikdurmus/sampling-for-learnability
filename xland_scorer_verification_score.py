"""Step 2 (SFL venv, deployment loader): score the exported labeled pairs
with the exact deployed ensemble; reproduce per-member + ensemble pair
accuracy, and measure member rank agreement ON DATASET IMAGES."""
import sys, yaml
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, "sfl/train")
from rlhf_utils import load_learnability_ensemble, make_member_logit_fn, pairwise_rank_agreement

d = np.load("xland_test_pairs_500.npz")
i0, i1, pref = d["i0"], d["i1"], d["pref"]
cfg = yaml.safe_load(open("sfl/train/config/xland-sfl.yaml"))
gd, st, K = load_learnability_ensemble(cfg["CNN_CHECKPOINT_PATHS"])
member_fn = make_member_logit_fn(gd)

def score_all(imgs_u8):
    out = []
    for s in range(0, len(imgs_u8), 100):
        x = jnp.asarray(imgs_u8[s:s+100], dtype=jnp.float32) / 255.0
        out.append(np.asarray(member_fn(st, x)))
    return np.concatenate(out, axis=1)  # (K, N)

S0, S1 = score_all(i0), score_all(i1)
print("per-member pair accuracy on 500 labeled test pairs (expect ~0.80):")
for k in range(K):
    acc = float((( (S1[k]-S0[k]) > 0).astype(np.float32) == pref).mean())
    print(f"  member {k}: {acc:.4f}")
ens = ((S1.mean(0) - S0.mean(0)) > 0).astype(np.float32)
print(f"ensemble accuracy: {float((ens==pref).mean()):.4f}  (report: 0.8048 on the full 2198-pair split)")

M = np.concatenate([S0, S1], axis=1)  # (K, 1000) dataset images
print(f"\nmember rank agreement on 1000 DATASET images: "
      f"{float(pairwise_rank_agreement(jnp.asarray(M))):.4f}   (DR-generated levels: 0.067)")
print(f"ensemble-mean logit range on dataset images: "
      f"[{M.mean(0).min():.1f}, {M.mean(0).max():.1f}] span {M.mean(0).max()-M.mean(0).min():.1f}"
      f"   (DR levels: span ~6)")
print(f"per-member logit std on dataset images: {[round(float(x),1) for x in M.std(1)]}")
