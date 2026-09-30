"""Which jittable resize matches the dataset pipeline? Ground truth: render
416x416 (with FOV highlight, as now fixed), PIL-resize BICUBIC to 200 (the
dataset compile path), score. Compare jax resize variants against it."""
import sys, yaml
import numpy as np
import jax, jax.numpy as jnp
from PIL import Image
sys.path.insert(0, "sfl/train")
from rlhf_utils import load_learnability_ensemble, make_member_logit_fn, pairwise_rank_agreement
import xminigrid
from xminigrid.wrappers import GymAutoResetWrapper
from xland_sfl import make_ruleset, build_xland_render_fn

env_cfg = yaml.safe_load(open("sfl/train/config/env/xland.yaml"))
cfg = yaml.safe_load(open("sfl/train/config/xland-sfl.yaml"))
env, env_params = xminigrid.make(env_cfg["env_id"])
env = GymAutoResetWrapper(env)
env_params = env_params.replace(ruleset=make_ruleset("difficult"))
# native-resolution renderer (no resize): target = 13*32 = 416
render416 = build_xland_render_fn(tile_size=32, view_size=7, target_size=416)
gd, st, K = load_learnability_ensemble(cfg["CNN_CHECKPOINT_PATHS"])
member_fn = make_member_logit_fn(gd)

rr = jax.random.split(jax.random.PRNGKey(7), 300)
ts = jax.vmap(env.reset, in_axes=(None, 0))(env_params, rr)
imgs416 = np.asarray(jax.vmap(render416)(ts.state.grid, ts.state.agent))  # float [0,1], 416
u8 = (np.clip(imgs416, 0, 1) * 255).astype(np.uint8)

def score(imgs_f32):
    out = []
    for s in range(0, len(imgs_f32), 100):
        out.append(np.asarray(member_fn(st, jnp.asarray(imgs_f32[s:s+100]))))
    return np.concatenate(out, axis=1)

# ground truth: PIL BICUBIC on uint8 (dataset compile path)
pil = np.stack([np.asarray(Image.fromarray(x).resize((200, 200))) for x in u8]).astype(np.float32) / 255.0
Mp = score(pil)
mp = Mp.mean(0)
print(f"PIL-BICUBIC ground truth: ens-mean range [{mp.min():.1f}, {mp.max():.1f}] mean {mp.mean():.1f} "
      f"agree {float(pairwise_rank_agreement(jnp.asarray(Mp))):.3f}   (dataset images: [-9, 2], 0.967)")

for method, aa in [("cubic", True), ("cubic", False), ("bilinear", True), ("bilinear", False)]:
    r = np.asarray(jax.image.resize(jnp.asarray(imgs416), (len(u8), 200, 200, 3),
                                    method=method, antialias=aa))
    r = np.clip(r, 0, 1)
    Mv = score(r)
    mv = Mv.mean(0)
    pixdiff = float(np.abs(r - pil).max())
    print(f"jax {method:9s} antialias={aa!s:5s}: range [{mv.min():.1f}, {mv.max():.1f}] mean {mv.mean():.1f} "
          f"agree {float(pairwise_rank_agreement(jnp.asarray(Mv))):.3f}  "
          f"logit-delta-vs-PIL {float(np.abs(Mv-Mp).mean()):.2f}  maxpixdiff {pixdiff:.3f}")
