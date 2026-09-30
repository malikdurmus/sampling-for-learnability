import sys
import numpy as np
sys.path.insert(0, "/home/d/durmusy/Desktop/GIT/new/uedrlhf")
import jax.numpy as jnp
from uedrlhf.jax_renderer import build_tile_caches, jax_render
from xminigrid.types import AgentState

d = np.load("xrender_cross.npz")
tc, atc = build_tile_caches(32)
outs = []
for i in range(len(d["grid"])):
    ag = AgentState(position=jnp.asarray(d["pos"][i]), direction=jnp.asarray(d["dir"][i]))
    outs.append(np.asarray(jax_render(jnp.asarray(d["grid"][i]), ag, tc, atc, view_size=7, tile_size=32)))
theirs = np.stack(outs)
ours = d["ours"]
print("shapes:", theirs.shape, ours.shape)
diff = np.abs(theirs.astype(int) - ours.astype(int))
print(f"max pixel diff {diff.max()}  mean {diff.mean():.3f}  frac pixels differing >8: {(diff.max(-1)>8).mean():.4f}")
np.savez("xrender_theirs.npz", theirs=theirs)
