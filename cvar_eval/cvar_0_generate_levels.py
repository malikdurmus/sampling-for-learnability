"""CVaR step 0: sample the frozen 10k solvable eval set (10 chunks x 1000).

Mirrors sfl/deploy/eval_jaxnav_single_agent_0_generate_levels.py, with the env
config taken from the campaign itself (W&B: sfl-jaxnav-campaign-fixed) and
valid_path_check forced on, per the SFL paper's protocol (random but solvable).

Run once, from the repo root:  python cvar_eval/cvar_0_generate_levels.py
"""
import os, sys, pickle
sys.path.append(os.getcwd())
import jax
import wandb
from jaxmarl.environments.jaxnav import JaxNav

ENT = "malikdurmus-ludwig-maximilian-university-of-munich"
PROJECT = "sfl-jaxnav-campaign-fixed"
CONFIG_RUN = "standard_seed1"

SEED = 0
NUM_ENVS_PER_CHUNK = 1000
EVAL_ITERS = 10                      # 10 x 1000 = 10,000 levels
TOTAL = NUM_ENVS_PER_CHUNK * EVAL_ITERS
PATH = f"sfl/data/eval/jaxnav/cvar_single_agent_{TOTAL}e.pkl"

api = wandb.Api()
run = list(api.runs(f"{ENT}/{PROJECT}", {"display_name": CONFIG_RUN}))[0]
cfg = run.config
cfg["env"]["env_params"]["map_params"]["valid_path_check"] = True

env = JaxNav(num_agents=cfg["env"]["num_agents"], **cfg["env"]["env_params"])

rng = jax.random.PRNGKey(SEED)
state_set = []
for i in range(EVAL_ITERS):
    rng, _rng = jax.random.split(rng)
    reset_rngs = jax.random.split(_rng, NUM_ENVS_PER_CHUNK)
    _, states = jax.vmap(env.reset)(reset_rngs)
    state_set.append(states)
    print(f"chunk {i+1}/{EVAL_ITERS} sampled")

os.makedirs(os.path.dirname(PATH), exist_ok=True)
with open(PATH, "wb") as f:
    pickle.dump(state_set, f)
print("saved", PATH, f"({TOTAL} levels, valid_path_check=True)")
