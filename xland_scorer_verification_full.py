"""Full-split in-memory replication of uedrlhf ensemble_eval (READ-ONLY) for
xmg_difficult, plus a jaxnav control where the expected answer (0.90) is known."""
import sys
import numpy as np
sys.path.insert(0, "/home/d/durmusy/Desktop/GIT/new/uedrlhf")
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
from datasets import load_from_disk
from uedrlhf.networks import LearnabilityResNet
from uedrlhf.fast_data import three_way_split, to_device_uint8, to_float, batches
import yaml

def load(path):
    model = LearnabilityResNet(rngs=nnx.Rngs(0))
    gd, ab = nnx.split(model)
    return nnx.merge(gd, ocp.StandardCheckpointer().restore(path, ab))

def member_scores(model, test_data, batch_size=256):
    model.eval()
    n = test_data[0].shape[0]
    order = jnp.arange(n)
    s0p, s1p = [], []
    @nnx.jit
    def score(model, img0, img1):
        s = model(jnp.concatenate([to_float(img0), to_float(img1)], 0),
                  deterministic=True, return_logits=True)
        return jnp.split(s, 2, 0)
    for img0, img1, _ in batches(*test_data, order, batch_size):
        a, b = score(model, img0, img1)
        s0p.append(np.asarray(a).ravel()); s1p.append(np.asarray(b).ravel())
    return np.concatenate(s0p), np.concatenate(s1p)

def evaluate(ds_path, ckpts, label):
    ds = load_from_disk(ds_path)
    _, _, test = three_way_split(ds, 0)
    td = to_device_uint8(test)
    prefs = np.asarray(td[2])
    print(f"\n== {label}: {len(prefs)} test pairs")
    S0, S1 = [], []
    for p in ckpts:
        m = load(p)
        s0, s1 = member_scores(m, td)
        S0.append(s0); S1.append(s1)
        acc = float((((s1-s0)>0).astype(np.float32)==prefs).mean())
        print(f"  {p.split('/')[-2]}: acc={acc:.4f}")
    ens = ((np.mean(S1,0)-np.mean(S0,0))>0).astype(np.float32)
    print(f"  ENSEMBLE: {float((ens==prefs).mean()):.4f}")

xc = yaml.safe_load(open("sfl/train/config/xland-sfl.yaml"))["CNN_CHECKPOINT_PATHS"]
jc = yaml.safe_load(open("sfl/train/config/jaxnav-sfl.yaml"))["CNN_CHECKPOINT_PATHS"]
evaluate("/home/d/durmusy/Desktop/GIT/new/uedrlhf/data_v3/jaxnav/difficulty_feedback/"
         "compiled_gemini_3_5_flash_len_21990_arr", jc, "jaxnav CONTROL (report: 0.9000)")
evaluate("/home/d/durmusy/Desktop/GIT/new/uedrlhf/data_v3/xland_minigrid/ruleset_difficult/"
         "difficulty_feedback_low/compiled_gemini_3_5_flash_len_21973_arr", xc,
         "xmg_difficult (report: 0.8048)")
