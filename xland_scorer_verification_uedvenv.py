"""Step 3 (uedrlhf venv, its own networks module — mini ensemble_eval):
score the same 500 exported pairs. Also dump restore fingerprints."""
import sys
import numpy as np
sys.path.insert(0, "/home/d/durmusy/Desktop/GIT/new/uedrlhf")
import jax.numpy as jnp
import numpy as onp
from flax import nnx
import orbax.checkpoint as ocp
from uedrlhf.networks import LearnabilityResNet

d = np.load("xland_test_pairs_500.npz")
i0, i1, pref = d["i0"], d["i1"], d["pref"]
import yaml
cfg = yaml.safe_load(open("sfl/train/config/xland-sfl.yaml"))
paths = cfg["CNN_CHECKPOINT_PATHS"]

def load(path):
    model = LearnabilityResNet(rngs=nnx.Rngs(0))
    gd, ab = nnx.split(model)
    st = ocp.StandardCheckpointer().restore(path, ab)
    return nnx.merge(gd, st)

accs, S0, S1 = [], [], []
for p in paths:
    m = load(p); m.eval()
    def score(imgs):
        out = []
        for s in range(0, len(imgs), 100):
            x = jnp.asarray(imgs[s:s+100], dtype=jnp.float32) / 255.0
            out.append(onp.asarray(m(x, deterministic=True, return_logits=True)))
        return onp.concatenate(out)
    s0, s1 = score(i0), score(i1)
    S0.append(s0); S1.append(s1)
    acc = float((((s1 - s0) > 0).astype(onp.float32) == pref).mean())
    accs.append(acc)
    # fingerprints: stem kernel + BN running stats
    st = nnx.state(m)
    k = onp.asarray(st["stem_conv"]["kernel"].value)
    bnm = onp.asarray(st["stem_bn"]["mean"].value)
    bnv = onp.asarray(st["stem_bn"]["var"].value)
    print(f"{p.split('/')[-2]}: acc={acc:.4f}  stem_kernel mean/std={k.mean():.5f}/{k.std():.5f}"
          f"  bn_mean[:3]={onp.round(bnm[:3],4)}  bn_var[:3]={onp.round(bnv[:3],4)}")
ens = ((onp.mean(S1,0) - onp.mean(S0,0)) > 0).astype(onp.float32)
print(f"ensemble acc (uedrlhf venv): {float((ens==pref).mean()):.4f}")
