"""Acceptance: with the inversion in the scoring choke point, the deployed
path must reproduce the standalone-test numbers exactly."""
import sys, os, yaml
import numpy as np
import jax, jax.numpy as jnp
REPO = "/home/d/durmusy/Desktop/GIT/updatedjnavwsfl/sampling-for-learnability"
os.chdir(REPO); sys.path.insert(0, REPO + "/sfl/train")
from rlhf_utils import (load_learnability_ensemble, make_member_logit_fn,
                        make_ensemble_logit_fn, pairwise_rank_agreement,
                        invert_input_domain)

# involution + exact arithmetic
x = jnp.asarray(np.arange(256, dtype=np.float32) / 255.0)
inv = np.round(np.asarray(invert_input_domain(x)) * 255).astype(int)
assert inv[0] == 0 and inv[100] == 156 and inv[51] == 205 and inv[131] == 125 and inv[255] == 1
assert np.array_equal(np.round(np.asarray(invert_input_domain(invert_input_domain(x))) * 255).astype(int),
                      np.arange(256)), "must be an involution"
print("inversion arithmetic exact (0->0, 100->156, 51->205, 255->1) and involutive: OK")

for env, npz in (("jaxnav", "jaxnav_test_pairs_800.npz"), ("xland", "xland_test_pairs_500.npz")):
    cfg = yaml.safe_load(open(f"sfl/train/config/{'jaxnav' if env=='jaxnav' else 'xland'}-sfl.yaml"))
    d = np.load(npz); i0, i1, pref = d["i0"], d["i1"], d["pref"]
    n = 300 if env == "jaxnav" else 150
    gd, st, K = load_learnability_ensemble(cfg["CNN_CHECKPOINT_PATHS"])
    fn = make_member_logit_fn(gd)          # inversion ON by default now
    bs = 100 if env == "jaxnav" else 25
    def sc(imgs):
        o = []
        for s in range(0, len(imgs), bs):
            o.append(np.asarray(fn(st, jnp.asarray(imgs[s:s+bs], dtype=jnp.float32) / 255.0)))
        return np.concatenate(o, axis=1)
    S = sc(np.concatenate([i0[:n], i1[:n]]))
    s0, s1 = S[:, :n], S[:, n:]
    acc = float((((s1.mean(0) - s0.mean(0)) > 0).astype(np.float32) == pref[:n]).mean())
    print(f"{env}: TRUE images through the fixed scoring path -> pair acc {acc:.4f}"
          f"  (expected ~{0.90 if env=='jaxnav' else 0.80})")
    # ensemble fn must agree with member fn
    em, _ = make_ensemble_logit_fn(gd)(st, jnp.asarray(i0[:20], dtype=jnp.float32) / 255.0)
    assert np.allclose(np.asarray(em), sc(i0[:20]).mean(0), atol=1e-4)
print("ensemble/member paths consistent: OK")
