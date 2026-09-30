import sys, yaml
import numpy as np
import jax
sys.path.insert(0, "sfl/train")
import xminigrid
from xminigrid.wrappers import GymAutoResetWrapper
from xland_sfl import make_ruleset, build_xland_render_fn

env_cfg = yaml.safe_load(open("sfl/train/config/env/xland.yaml"))
env, env_params = xminigrid.make(env_cfg["env_id"])
env = GymAutoResetWrapper(env)
env_params = env_params.replace(ruleset=make_ruleset("difficult"))
rr = jax.random.split(jax.random.PRNGKey(7), 8)
ts = jax.vmap(env.reset, in_axes=(None, 0))(env_params, rr)
render416 = build_xland_render_fn(tile_size=32, view_size=7, target_size=416)
ours = np.asarray((np.clip(np.asarray(jax.vmap(render416)(ts.state.grid, ts.state.agent)),0,1)*255)).astype(np.uint8)
np.savez("xrender_cross.npz", grid=np.asarray(ts.state.grid), pos=np.asarray(ts.state.agent.position),
         dir=np.asarray(ts.state.agent.direction), ours=ours)
print("dumped 8 states + our renders")
