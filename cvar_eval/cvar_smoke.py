import os, sys, pickle, time
sys.path.append(os.getcwd())
import jax, jax.numpy as jnp
import wandb
from jaxmarl.environments.jaxnav import JaxNav
from sfl.runners.eval_runner import EvalSampledRunner
from sfl.train.train_utils import load_params
from sfl.train.common.network import ActorCriticRNN, ScannedRNN

ENT="malikdurmus-ludwig-maximilian-university-of-munich"
cfg=list(wandb.Api().runs(f"{ENT}/sfl-jaxnav-campaign-fixed",{"display_name":"standard_seed1"}))[0].config
cfg["env"]["env_params"]["map_params"]["valid_path_check"]=True
t=cfg["learning"]; t["LOG_DORMANCY"]=True
env=JaxNav(num_agents=cfg["env"]["num_agents"], **cfg["env"]["env_params"])
net=ActorCriticRNN(action_dim=env.agent_action_space().shape[0], config=t)
with open("sfl/data/eval/jaxnav/cvar_single_agent_10000e.pkl","rb") as f:
    chunk=pickle.load(f)[0]
params=load_params("checkpoints/multi_robot_ued/standard_seed1/model.safetensors")
rng=jax.random.PRNGKey(0)
t0=time.time()
runner=EvalSampledRunner(rng, env, net, ScannedRNN.initialize_carry,
    hidden_size=t["HIDDEN_SIZE"], greedy=False, env_init_states=chunk,
    n_episodes=10, n_envs=1000)
o=runner.run(jax.random.PRNGKey(1), params)
wr=o["eval-sampled/win_rates"].squeeze()
print(f"chunk time (incl. compile): {time.time()-t0:.1f}s  mean_win={float(wr.mean()):.4f}  n={wr.shape}")
t0=time.time()
o=runner.run(jax.random.PRNGKey(2), params)
print(f"second call (compiled): {time.time()-t0:.1f}s")
