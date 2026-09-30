
import os, sys, pickle, time
sys.path.append(os.getcwd())
import jax, jax.numpy as jnp
import pandas as pd
import wandb
from jaxmarl.environments.jaxnav import JaxNav
from sfl.runners.eval_runner import EvalSampledRunner
from sfl.train.train_utils import load_params
from sfl.train.common.network import ActorCriticRNN, ScannedRNN

ROLLOUT_SEED = int(sys.argv[1])
ENT = "malikdurmus-ludwig-maximilian-university-of-munich"
PROJECT = "sfl-jaxnav-campaign-fixed"
CONFIG_RUN = "standard_seed1"
CKPT_ROOT = "checkpoints/multi_robot_ued"
N_EPISODES = 10
NUM_ENVS_PER_CHUNK = 1000
TOTAL = 10000
LEVELS = f"sfl/data/eval/jaxnav/cvar_single_agent_{TOTAL}e.pkl"

METHOD_RUNS = {
    "standard":                     [f"standard_seed{s}" for s in range(1, 11)],
    "dr":                           [f"dr_seed{s}" for s in range(1, 11)],
    "cnn":                          [f"cnn_fixed_seed{s}" for s in range(1, 11)],
    "hybrid_linear":                [f"hybrid_linear_fixed_seed{s}" for s in range(1, 11)],
    "hybrid_soft_handoff":          [f"hybrid_soft_handoff_fixed_seed{s}" for s in range(1, 11)],
    "hybrid_learnability_weighted": [f"hybrid_learnability_weighted_fixed_seed{s}" for s in range(1, 11)],
    "hybrid_multiplicative":        [f"hybrid_multiplicative_fixed_seed{s}" for s in range(1, 11)],
}

api = wandb.Api()
cfg = list(api.runs(f"{ENT}/{PROJECT}", {"display_name": CONFIG_RUN}))[0].config
cfg["env"]["env_params"]["map_params"]["valid_path_check"] = True
t_config = cfg["learning"]
t_config["LOG_DORMANCY"] = True

env = JaxNav(num_agents=cfg["env"]["num_agents"], **cfg["env"]["env_params"])
network = ActorCriticRNN(action_dim=env.agent_action_space().shape[0], config=t_config)

with open(LEVELS, "rb") as f:
    state_set = pickle.load(f)

# builds the runners once

_rk = jax.random.PRNGKey(1234 + ROLLOUT_SEED)
RUNNERS = []
for env_states in state_set:
    _rk, _r = jax.random.split(_rk)
    RUNNERS.append(EvalSampledRunner(
        _r, env, network, ScannedRNN.initialize_carry,
        hidden_size=t_config["HIDDEN_SIZE"], greedy=False,
        env_init_states=env_states, n_episodes=N_EPISODES,
        n_envs=NUM_ENVS_PER_CHUNK))

def rollout_ckpt(run_name, rng):
    params = load_params(f"{CKPT_ROOT}/{run_name}/model.safetensors")
    outcomes = None
    for runner in RUNNERS:
        rng, _rng = jax.random.split(rng)
        o = runner.run(_rng, params)
        o = {k: o[k] for k in ("eval-sampled/win_rates", "eval-sampled/returns")}
        outcomes = o if outcomes is None else jax.tree.map(
            lambda a, b: jnp.concatenate([a, b]), outcomes, o)
    return outcomes

for method, runs in METHOD_RUNS.items():
    for s, run_name in enumerate(runs):
        out_path = (f"sfl/data/eval/results/jaxnav-single/"
                    f"eval_{TOTAL}_envs_seed_{ROLLOUT_SEED}/{method}/{s}.csv")
        if os.path.exists(out_path):
            print("skip (exists):", out_path); continue
        t0 = time.time()
        rng = jax.random.PRNGKey(ROLLOUT_SEED)
        o = rollout_ckpt(run_name, rng)
        wr = o["eval-sampled/win_rates"].squeeze()
        rt = o["eval-sampled/returns"].squeeze()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pd.DataFrame({"env-id": jnp.arange(len(wr)),
                      "win-rates": wr, "returns": rt}).to_csv(out_path, index=False)
        print(f"{run_name}: mean_win={float(wr.mean()):.4f}  "
              f"({time.time()-t0:.0f}s) -> {out_path}")
