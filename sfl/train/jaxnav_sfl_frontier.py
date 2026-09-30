import os
#os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'
#os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import hydra
from omegaconf import OmegaConf
import os
from functools import partial
import pickle
import time 
from PIL import Image
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import wandb
from flax import nnx
import orbax.checkpoint as ocp
from functools import partial
from  rlhf_utils import (load_learnability_ensemble, make_ensemble_logit_fn,
                         make_member_logit_fn, pairwise_rank_agreement,
                         standardized_member_spread, resolve_input_domain,
                         get_jaxnav_rasterizer)



from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav, EnvInstance, NUM_REWARD_COMPONENTS, REWARD_COMPONENT_DENSE, REWARD_COMPONENT_SPARSE, listify_reward

from sfl.runners import EvalSingletonsRunner, EvalSampledRunner
from sfl.train.common.network import ActorCriticRNN, ScannedRNN
from sfl.train.train_utils import save_params

def isotonic_decreasing(y):
    n = y.shape[0]
    cs = jnp.concatenate([jnp.zeros((1,), y.dtype), jnp.cumsum(y)])
    jj = jnp.arange(n)[:, None]
    kk = jnp.arange(n)[None, :]
    valid = kk >= jj
    avg = jnp.where(valid, (cs[kk + 1] - cs[jj]) / jnp.maximum(kk - jj + 1, 1), 0.0)  # mean(y[j..k])
    out = []
    for i in range(n):
        inner = jnp.where((kk >= i) & valid, avg, -jnp.inf).max(axis=1)   # for each j: max over k>=i
        out.append(jnp.where(jnp.arange(n) <= i, inner, jnp.inf).min())   # min over j<=i
    return jnp.stack(out)


def frontier_fit(c, p, nbins, min_informative_bins, informative_p):
    N = c.shape[0]
    per = N // nbins
    order = jnp.argsort(c)
    cs = c[order][: per * nbins].reshape(nbins, per)
    ps = p[order][: per * nbins].reshape(nbins, per)
    bin_c = cs.mean(axis=1)
    bin_p = ps.mean(axis=1)
    iso = isotonic_decreasing(bin_p)
    n_above = (iso >= 0.5).sum()                      
    i_star = jnp.clip(n_above - 1, 0, nbins - 2)
    c0, c1 = bin_c[i_star], bin_c[i_star + 1]
    p0, p1 = iso[i_star], iso[i_star + 1]
    interp = c0 + (p0 - 0.5) / jnp.maximum(p0 - p1, 1e-6) * (c1 - c0)
    mu_star = jnp.where(n_above == 0, 0.0, jnp.where(n_above == nbins, 1.0, interp))
    mu_star = jnp.clip(mu_star, 0.0, 1.0)
    n_informative = (bin_p > informative_p).sum()
    use_frontier = n_informative >= min_informative_bins
    return {"bin_c": bin_c, "bin_p": bin_p, "iso_p": iso, "mu_star": mu_star,
            "n_informative": n_informative, "use_frontier": use_frontier}


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    mask: jnp.ndarray
    info: jnp.ndarray

class RolloutBatch(NamedTuple):
    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    log_probs: jnp.ndarray
    values: jnp.ndarray
    targets: jnp.ndarray
    advantages: jnp.ndarray
    # carry: jnp.ndarray
    mask: jnp.ndarray

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


@partial(jax.vmap, in_axes=(None, 1, 1, 1))
@partial(jax.jit, static_argnums=(0,))
def calc_progress_outcomes_by_agent(max_steps: int, dones, returns, info):
    
    idxs = jnp.arange(max_steps)

    @partial(jax.vmap, in_axes=(0, 0))
    def __ep_outcomes(start_idx, end_idx):
        mask = (idxs > start_idx) & (idxs <= end_idx) & (end_idx != max_steps)
        r = jnp.sum(returns * mask)
        success = jnp.sum(info["GoalR"] * mask)
        collision = jnp.sum((info["MapC"] + info["AgentC"]) * mask)
        timeo = jnp.sum(info["TimeO"] * mask)
        l = end_idx - start_idx
        #  gted by mask_done
        d_start = info["DPre"][jnp.minimum(start_idx + 1, max_steps - 1)]
        d_end = info["DPre"][jnp.minimum(end_idx, max_steps - 1)]
        d_min = jnp.min(jnp.where(mask, info["DPre"], jnp.inf))
        progress_end = jnp.clip(1.0 - d_end / jnp.maximum(d_start, 1e-6), 0.0, 1.0)
        progress_min = jnp.clip(1.0 - d_min / jnp.maximum(d_start, 1e-6), 0.0, 1.0)
        return r, success, collision, timeo, l, progress_end, progress_min, d_start

    done_idxs = jnp.argwhere(dones, size=10, fill_value=max_steps).squeeze()
    mask_done = jnp.where(done_idxs == max_steps, False, True)
    (ep_return, success, collision, timeo, length,
     progress_end, progress_min, d_start) = __ep_outcomes(
        jnp.concatenate([jnp.array([-1]), done_idxs[:-1]]), done_idxs)

    n_ep = mask_done.sum()
    has_ep = n_ep > 0

    def _masked_mean_var(x):
        m = jnp.where(has_ep, x.mean(where=mask_done), 0.0)
        v = jnp.where(has_ep, ((x - m) ** 2).mean(where=mask_done), 0.0)
        return m, v

    progress_end_mean, progress_end_var = _masked_mean_var(progress_end)
    progress_min_mean, progress_min_var = _masked_mean_var(progress_min)

    return {"ep_return": ep_return.mean(where=mask_done),
            "num_episodes": n_ep,
            "success_rate": success.mean(where=mask_done),
            "collision_rate": collision.mean(where=mask_done),
            "timeout_rate": timeo.mean(where=mask_done),
            "ep_len": length.mean(where=mask_done),
            "progress_end_mean": progress_end_mean,
            "progress_end_var": progress_end_var,
            "progress_min_mean": progress_min_mean,
            "progress_min_var": progress_min_var,
            "d_start_mean": jnp.where(has_ep, d_start.mean(where=mask_done), 0.0),
        }


@hydra.main(version_base=None, config_path="config", config_name="jaxnav-sfl-frontier")
def main(config):

    # WAND B QAND CONFIG
    config = OmegaConf.to_container(config)
    run = wandb.init(
        name= config["RUN_NAME"],
        group=config["GROUP_NAME"],
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "RNN", "DR", f"ts: {config['env']['test_set']}"],
        config=config,
        mode=config["WANDB_MODE"],
    )

    
    run.define_metric("update_count")
    run.define_metric("*", step_metric="update_count")

    def safe_wandb_log(payload):
        
        try:
            run.log(payload)
        except Exception as e:
            print(f"[wandb] run.log failed, skipping this cycle's payload: {e!r}")

    def safe_histogram(arr):
       
        arr = np.asarray(arr, dtype=np.float64).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        try:
            return wandb.Histogram(arr)
        except Exception as e:
            print(f"[wandb] Histogram construction failed, dropping key: {e!r}")
            return None

    # ---- Determine learn method from single config key ----
    learn_method_raw = config.get("LEARN_METHOD", "standard")
    VALID_LEARN_METHODS = (
        "standard", "random", "cnn", "dijkstra",
        "progress", "progress_mean", "progress_mindist",
        "hybrid_linear", "hybrid_soft_handoff",
        "hybrid_learnability_weighted", "hybrid_multiplicative",
    )
    # A misspelled hybrid mode must fail loudly, not silently fall back to pure SFL
    assert learn_method_raw in VALID_LEARN_METHODS, \
        f"Invalid LEARN_METHOD '{learn_method_raw}' (valid: {VALID_LEARN_METHODS})"
    if learn_method_raw.startswith("hybrid_"):
        learn_method = "hybrid"
        hybrid_mode = learn_method_raw.replace("hybrid_", "")  # e.g. "linear", "soft_handoff", etc.
        config["HYBRID_MODE"] = hybrid_mode
    else:
        learn_method = learn_method_raw  # "standard", "cnn", or "random"
        config["HYBRID_MODE"] = "linear"  # unused default
    needs_cnn = learn_method in ["cnn", "hybrid"]
    
    needs_dijkstra = learn_method == "dijkstra"
     
    needs_progress = learn_method.startswith("progress")
    print(f"--- USING LEARNABILITY METHOD: {learn_method} (raw: {learn_method_raw}) ---")

    # ---- CNN score normalization m-
    CNN_SCORE_METHOD = config.get("CNN_SCORE_METHOD", "sigmoid")
    assert CNN_SCORE_METHOD in ("sigmoid", "minmax", "percentile"), \
        f"Invalid CNN_SCORE_METHOD '{CNN_SCORE_METHOD}' (use sigmoid | minmax | percentile)"
    print(f"--- CNN SCORE METHOD: {CNN_SCORE_METHOD} ---")

    # ----- Frontier
    MU_MEASUREMENT = config.get("MU_MEASUREMENT", "buffer_env")
    assert MU_MEASUREMENT in ("buffer_env", "legacy"), f"Invalid MU_MEASUREMENT '{MU_MEASUREMENT}'"
    CURRICULUM_STRATEGY_CFG = config.get("CURRICULUM_STRATEGY", "performance_adaptive")
    assert CURRICULUM_STRATEGY_CFG in ("time_based", "performance_adaptive", "gaussian_frontier", "calibrated_frontier"), \
        f"Invalid CURRICULUM_STRATEGY '{CURRICULUM_STRATEGY_CFG}'"
    FRONTIER_MODE = config.get("FRONTIER_MODE", "crossing")
    assert FRONTIER_MODE in ("crossing", "priority"), f"Invalid FRONTIER_MODE '{FRONTIER_MODE}'"
    FRONTIER_NBINS = int(config.get("FRONTIER_NBINS", 20))
    FRONTIER_MIN_INFORMATIVE_BINS = int(config.get("FRONTIER_MIN_INFORMATIVE_BINS", 3))
    FRONTIER_INFORMATIVE_P = float(config.get("FRONTIER_INFORMATIVE_P", 0.05))
    if CURRICULUM_STRATEGY_CFG == "calibrated_frontier":
        assert learn_method == "hybrid", "calibrated_frontier needs per-candidate rollouts: hybrid arms only (this version)"
    if learn_method == "hybrid":
        assert (config["BATCH_SIZE"] * config["NUM_BATCHES"]) % FRONTIER_NBINS == 0, "candidate count must be a multiple of FRONTIER_NBINS"
    print(f"--- MU_MEASUREMENT: {MU_MEASUREMENT} | CURRICULUM_STRATEGY: {CURRICULUM_STRATEGY_CFG} | FRONTIER_MODE: {FRONTIER_MODE} ---")
    if wandb.run is not None:
        wandb.config.update({"MU_MEASUREMENT": MU_MEASUREMENT, "CURRICULUM_STRATEGY": CURRICULUM_STRATEGY_CFG,
                             "FRONTIER_MODE": FRONTIER_MODE, "FRONTIER_NBINS": FRONTIER_NBINS,
                             "FRONTIER_MIN_INFORMATIVE_BINS": FRONTIER_MIN_INFORMATIVE_BINS,
                             "FRONTIER_INFORMATIVE_P": FRONTIER_INFORMATIVE_P,
                             "TRAINER_VARIANT": "jaxnav_sfl_frontier.py"}, allow_val_change=True)

    if needs_cnn:
        
        cnn_checkpoint_paths = config.get("CNN_CHECKPOINT_PATHS")
        assert cnn_checkpoint_paths, \
            "LEARN_METHOD needs the CNN but CNN_CHECKPOINT_PATHS is not set in the config"
        print(f"Loading CNN learnability scorer ({len(cnn_checkpoint_paths)} member(s))...")

        cnn_graphdef, cnn_state, cnn_num_members = load_learnability_ensemble(cnn_checkpoint_paths)
        # Input domain comes from CHECKPOINT_INDEX.json we had a mismatch which was later fixed

        cnn_input_domain = resolve_input_domain(cnn_checkpoint_paths)
        _invert = cnn_input_domain == "inverted"
        print(f"--- CNN INPUT DOMAIN: {cnn_input_domain} (invert_input={_invert}) ---")
        cnn_logit_fn = make_ensemble_logit_fn(cnn_graphdef, invert_input=_invert)
        cnn_member_logit_fn = make_member_logit_fn(cnn_graphdef, invert_input=_invert)

        # Log the  cnn config to wandb so there is a record
        if wandb.run is not None:
            wandb.config.update({
                "CNN_CHECKPOINT_PATHS": list(cnn_checkpoint_paths),
                "CNN_ENSEMBLE_K": cnn_num_members,
                "CNN_ENSEMBLE_COMBINE": "mean_of_member_logits",
                "CNN_INPUT_DOMAIN": cnn_input_domain,
                "CNN_INVERT_INPUT": _invert,
                "CNN_SCORE_CONVENTION": "higher=more_difficult",
                "CNN_SCORE_METHOD": CNN_SCORE_METHOD,
                "CNN_POOLING": "mean",
                "CNN_IMG_SIZE": 64
            }, allow_val_change=True)
    else:
        # Create dummy variables
        cnn_graphdef, cnn_state, cnn_logit_fn = None, None, None
        cnn_member_logit_fn, cnn_num_members = None, 0


    rng = jax.random.PRNGKey(config["SEED"])
    
    assert (config["learning"]["NUM_ENVS_FROM_SAMPLED"] +  config["learning"]["NUM_ENVS_TO_GENERATE"]) == config["learning"]["NUM_ENVS"]
    
    
    env = JaxNav(num_agents=config["env"]["num_agents"],
                        **config["env"]["env_params"])  # use old config for env params to try reduce errors
    print('num agents', env.num_agents)
    t_config = config["learning"]
        
    t_config["NUM_ACTORS"] = env.num_agents * t_config["NUM_ENVS"]
    t_config["NUM_UPDATES"] = (
        t_config["TOTAL_TIMESTEPS"] // t_config["NUM_STEPS"] // t_config["NUM_ENVS"]
    )
    t_config["MINIBATCH_SIZE"] = (
        t_config["NUM_ACTORS"] * t_config["NUM_STEPS"] // t_config["NUM_MINIBATCHES"]
    )
    t_config["CLIP_EPS"] = (
        t_config["CLIP_EPS"] / env.num_agents
        if t_config["SCALE_CLIP_EPS"]
        else t_config["CLIP_EPS"]
    )
        
    network = ActorCriticRNN(env.agent_action_space().shape[0],
                            config=t_config)

    eval_singleton_runner = EvalSingletonsRunner(
        config["env"]["test_set"],
        network,
        init_carry=ScannedRNN.initialize_carry,
        hidden_size=t_config["HIDDEN_SIZE"],
        env_kwargs=config["env"]["env_params"]
    )
    # 100 instances # map size 11 x 11 (JaxNav Map size)
    with open(config["EVAL_SAMPLED_SET_PATH"], "rb") as f:
        eval_env_instances = pickle.load(f)
    _, eval_init_states = jax.vmap(env.set_env_instance, in_axes=(0))(eval_env_instances)
    
    eval_sampled_runner = EvalSampledRunner(
        None,
        env,
        network,
        ScannedRNN.initialize_carry,
        hidden_size=t_config["HIDDEN_SIZE"],
        greedy=False,
        env_init_states=eval_init_states,
        n_episodes=10,
    )
    
    def linear_schedule(count):
        count = count // (t_config["NUM_MINIBATCHES"] * t_config["UPDATE_EPOCHS"])
        frac = (
            1.0 - count / t_config["NUM_UPDATES"]
        )
        return t_config["LR"] * frac
    
    
    CNN_IMG_SIZE = 64  # CNN input resolution
    jaxnav_render_fn = get_jaxnav_rasterizer(
        img_height=CNN_IMG_SIZE,
        img_width=CNN_IMG_SIZE,
        map_height=config["env"]["env_params"]["map_params"]["map_size"][0],
        map_width=config["env"]["env_params"]["map_params"]["map_size"][1],
        cell_size=1.0,
    )

    if needs_dijkstra:
        _msz = config["env"]["env_params"]["map_params"]["map_size"]
        DIJKSTRA_SENTINEL = float(_msz[0] * _msz[1])  # unreachable = hardest

        def dijkstra_batch_scores(env_states):
            """Batched raw difficulty: shortest-path length per level; the
            per-level score is the mean over agents (1 agent = the length);
            unreachable -> DIJKSTRA_SENTINEL."""
            def _one(state):
                passable, plen = jax.vmap(
                    env.map_obj.dikstra_path, in_axes=(None, 0, 0)
                )(state.map_data, state.pos, state.goal)
                return jnp.where(passable, plen, DIJKSTRA_SENTINEL).mean()
            return jax.vmap(_one)(env_states)

        if wandb.run is not None:
            wandb.config.update({
                "SCORER": "dijkstra_shortest_path",
                "DIJKSTRA_SENTINEL": DIJKSTRA_SENTINEL,
                "SCORE_NOTE": ("integer lengths, heavy ties; percentile "
                               "handles ties, tied-band selection is random"),
                "CNN_SCORE_METHOD": CNN_SCORE_METHOD,
            }, allow_val_change=True)

    if needs_progress and wandb.run is not None:
        _progress_scorer_notes = {
            "progress": ("var over episodes of progress_end = "
                         "clip(1 - d_end/d_start, 0, 1) [registered formula]"),
            "progress_mindist": ("var over episodes of progress_min = "
                                 "clip(1 - d_min/d_start, 0, 1) "
                                 "[closest approach]"),
            "progress_mean": ("mp*(1-mp), mp = mean progress_end "
                              "[= ranking by |mp-0.5|, half-completion target]"),
        }
        wandb.config.update({
            "SCORER": f"goal_proximity_{learn_method}",
            "SCORE_NOTE": (_progress_scorer_notes[learn_method] +
                           "; distances from PRE-step state (post-step state "
                           "at a done index is the auto-reset state); binary "
                           "p(1-p) + all candidate scores logged alongside "
                           "(sfl/*, progress/*, agree/*)"),
        }, allow_val_change=True)

    PERCENTILE_REF_SIZE = int(config.get("PERCENTILE_REF_SIZE", 10000))
    PERCENTILE_REF_SEED = int(config.get("PERCENTILE_REF_SEED", 4242))

    if needs_cnn:
        if CNN_SCORE_METHOD == "percentile":
            
            assert PERCENTILE_REF_SIZE % 1000 == 0, "PERCENTILE_REF_SIZE must be a multiple of 1000"

            @jax.jit
            def _ref_batch_logits(ref_rng):
                ref_reset_rng = jax.random.split(ref_rng, 1000)
                _, ref_env_state = jax.vmap(env.reset, in_axes=(0,))(ref_reset_rng)
                ref_chunked = jax.tree.map(lambda x: x.reshape((10, 100) + x.shape[1:]), ref_env_state)
                def _score_chunk(carry, chunk):
                    imgs = jax.vmap(jaxnav_render_fn)(chunk)
                    return carry, cnn_logit_fn(cnn_state, imgs)[0]
                _, lg = jax.lax.scan(_score_chunk, None, ref_chunked)
                return lg.reshape((1000,))

            print(f"Building frozen percentile reference ({PERCENTILE_REF_SIZE} envs, seed {PERCENTILE_REF_SEED})...")
            _ref_rngs = jax.random.split(jax.random.PRNGKey(PERCENTILE_REF_SEED), PERCENTILE_REF_SIZE // 1000)
            ref_logits_sorted = jnp.sort(jnp.concatenate([_ref_batch_logits(r) for r in _ref_rngs]))
            ref_grid = jnp.linspace(0.0, 1.0, ref_logits_sorted.shape[0])
            print(f"Reference logit range: [{float(ref_logits_sorted[0]):.2f}, {float(ref_logits_sorted[-1]):.2f}]")

            def normalize_scores(logits):
                """Raw CNN logits -> difficulty percentile of the generation distribution."""
                return jnp.interp(logits, ref_logits_sorted, ref_grid)
        elif CNN_SCORE_METHOD == "minmax":
            def normalize_scores(logits):
                """Per-cycle min-max over raw logits (batch-dependent scale)."""
                return (logits - jnp.min(logits)) / (jnp.max(logits) - jnp.min(logits) + 1e-8)
        else:  # sigmoid
            def normalize_scores(logits):
                """Raw sigmoid of the logits (unanchored scale)."""
                return jax.nn.sigmoid(logits)

        
        _probe_rngs = jax.random.split(jax.random.PRNGKey(config["SEED"] + 999), 10)
        _, _probe_states = jax.vmap(env.reset)(_probe_rngs)
        _probe_imgs = jax.vmap(jaxnav_render_fn)(_probe_states)
        _probe_logits, _probe_std = cnn_logit_fn(cnn_state, _probe_imgs)
        _probe_scores = normalize_scores(_probe_logits)
        _fig, _axes = plt.subplots(1, 10, figsize=(20, 2.5))
        for _i, _ax in enumerate(_axes):
            _ax.imshow(np.asarray(_probe_imgs[_i]))
            _ax.set_title(f"cnn: {float(_probe_scores[_i]):.3f}\nlogit: {float(_probe_logits[_i]):.1f}±{float(_probe_std[_i]):.1f}", fontsize=8)
            _ax.axis("off")
        plt.tight_layout()
        _fig.canvas.draw()
        _probe_im = Image.fromarray(np.array(_fig.canvas.buffer_rgba())).convert("RGB")
        run.log({"cnn_input_check": wandb.Image(_probe_im), "update_count": 0})
        plt.close(_fig)
        print("CNN input check scores:", np.asarray(_probe_scores))
        print("CNN member-std over probes:", np.asarray(_probe_std))

        if wandb.run is not None and CNN_SCORE_METHOD == "percentile":
            wandb.config.update({
                "PERCENTILE_REF_SIZE": PERCENTILE_REF_SIZE,
                "PERCENTILE_REF_SEED": PERCENTILE_REF_SEED,
            }, allow_val_change=True)
    elif needs_dijkstra:
        if CNN_SCORE_METHOD == "percentile":
           
            assert PERCENTILE_REF_SIZE % 1000 == 0, "PERCENTILE_REF_SIZE must be a multiple of 1000"

            @jax.jit
            def _ref_batch_scores(ref_rng):
                ref_reset_rng = jax.random.split(ref_rng, 1000)
                _, ref_env_state = jax.vmap(env.reset, in_axes=(0,))(ref_reset_rng)
                return dijkstra_batch_scores(ref_env_state)

            print(f"Building frozen percentile reference ({PERCENTILE_REF_SIZE} envs, seed {PERCENTILE_REF_SEED}, dijkstra scores)...")
            _ref_rngs = jax.random.split(jax.random.PRNGKey(PERCENTILE_REF_SEED), PERCENTILE_REF_SIZE // 1000)
            ref_scores_sorted = jnp.sort(jnp.concatenate([_ref_batch_scores(r) for r in _ref_rngs]))
            ref_grid = jnp.linspace(0.0, 1.0, ref_scores_sorted.shape[0])
            print(f"Reference score range: [{float(ref_scores_sorted[0]):.1f}, {float(ref_scores_sorted[-1]):.1f}]  "
                  f"distinct values: {len(np.unique(np.asarray(ref_scores_sorted)))}")

            def normalize_scores(scores):
                """Raw dijkstra lengths -> difficulty percentile of the generation distribution."""
                return jnp.interp(scores, ref_scores_sorted, ref_grid)
        elif CNN_SCORE_METHOD == "minmax":
            def normalize_scores(scores):
                return (scores - jnp.min(scores)) / (jnp.max(scores) - jnp.min(scores) + 1e-8)
        else:  # sigmoid — defined for completeness; saturates on raw lengths
            def normalize_scores(scores):
                return jax.nn.sigmoid(scores)

        _probe_rngs = jax.random.split(jax.random.PRNGKey(config["SEED"] + 999), 10)
        _, _probe_states = jax.vmap(env.reset)(_probe_rngs)
        _probe_imgs = jax.vmap(jaxnav_render_fn)(_probe_states)
        _probe_raw = dijkstra_batch_scores(_probe_states)
        _probe_scores = normalize_scores(_probe_raw)
        _fig, _axes = plt.subplots(1, 10, figsize=(20, 2.5))
        for _i, _ax in enumerate(_axes):
            _ax.imshow(np.asarray(_probe_imgs[_i]))
            _ax.set_title(f"dij: {float(_probe_scores[_i]):.3f}\nlen: {float(_probe_raw[_i]):.0f}", fontsize=8)
            _ax.axis("off")
        plt.tight_layout()
        _fig.canvas.draw()
        _probe_im = Image.fromarray(np.array(_fig.canvas.buffer_rgba())).convert("RGB")
        run.log({"dijkstra_input_check": wandb.Image(_probe_im), "update_count": 0})
        plt.close(_fig)
        print("Dijkstra probe lengths:", np.asarray(_probe_raw))
        print("Dijkstra probe scores :", np.asarray(_probe_scores))

        if wandb.run is not None and CNN_SCORE_METHOD == "percentile":
            wandb.config.update({
                "PERCENTILE_REF_SIZE": PERCENTILE_REF_SIZE,
                "PERCENTILE_REF_SEED": PERCENTILE_REF_SEED,
            }, allow_val_change=True)
    else:
        normalize_scores = None


    # INIT NETWORK
    rng, _rng = jax.random.split(rng)
    init_x = (
        jnp.zeros(
            (1, t_config["NUM_ENVS"], env.lidar_num_beams+5)  # NOTE hardcoded (205)
        ),
        jnp.zeros((1, t_config["NUM_ENVS"])),
    )
    init_hstate = ScannedRNN.initialize_carry(t_config["NUM_ENVS"], t_config["HIDDEN_SIZE"])
    network_params = network.init(_rng, init_hstate, init_x)
    if t_config["ANNEAL_LR"]:
        tx = optax.chain(
            optax.clip_by_global_norm(t_config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(t_config["MAX_GRAD_NORM"]),
            optax.adam(t_config["LR"], eps=1e-5),
        )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

    rng, _rng = jax.random.split(rng)
    #initial_singleton_test_metrics = eval_singleton_runner.run(_rng, train_state.params)  #
    #initial_sampled_test_metrics = eval_sampled_runner.run(_rng, train_state.params)      #

    # INIT ENV
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, t_config["NUM_ENVS"])
    obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
    
    
    if t_config["LAMBDA_SCHEDULE"]:
        raise NotImplementedError("Lambda schedule not implemented for finetuning")
        rng, lambda_rng = jax.random.split(rng)
        env_state = env_state.replace(
            rew_lambda = sample_lambda_set(lambda_rng, 0),
        )
    start_state = env_state
    init_hstate = ScannedRNN.initialize_carry(t_config["NUM_ACTORS"], t_config["HIDDEN_SIZE"])
    

    @jax.jit
    def select_environments(rng, cnn_scores_normalized, target_mu):
        strategy = config.get("CURRICULUM_STRATEGY", "time_based")

        
        if strategy in ("time_based", "performance_adaptive", "calibrated_frontier"):
            distances = jnp.abs(cnn_scores_normalized - target_mu)
            top_indices = jnp.argsort(distances)[:config["NUM_TO_SAVE"]]
            top_indices = top_indices[::-1]  # closest to target_mu last
        elif strategy == "gaussian_frontier":
            var = config.get("CURRICULUM_GAUSSIAN_VAR", 0.1)
            weights = jnp.exp(-((cnn_scores_normalized - target_mu)**2) / (2 * var**2))
            probs = weights / jnp.sum(weights)
            top_indices = jax.random.choice(rng, cnn_scores_normalized.shape[0], shape=(config["NUM_TO_SAVE"],), p=probs, replace=False)
            sel_distances = jnp.abs(cnn_scores_normalized.at[top_indices].get() - target_mu)
            top_indices = top_indices.at[jnp.argsort(-sel_distances)].get()  # closest to target_mu last
        else:
            top_indices = jnp.argsort(cnn_scores_normalized)[-config["NUM_TO_SAVE"]:]  # highest score last

        return top_indices

    @partial(jax.jit, static_argnums=(1,)) # cnn_graphdef is static
    def get_learnability_set_cnn(rng, cnn_graphdef, cnn_state, target_mu):
        def _batch_step(unused, rng):
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["BATCH_SIZE"])
            obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
            
            env_instances = EnvInstance(
                agent_pos=env_state.pos,
                agent_theta=env_state.theta,
                goal_pos=env_state.goal,
                map_data=env_state.map_data,
                rew_lambda=env_state.rew_lambda,
            )
            
            # Chunk the rendering and evaluation to avoid OOM with large BATCH_SIZE mapped to 200x200 grids
            CHUNK_SIZE = min(config["BATCH_SIZE"], 100)
            NUM_CHUNKS = config["BATCH_SIZE"] // CHUNK_SIZE
            
            env_state_chunked = jax.tree.map(
                lambda x: x.reshape((NUM_CHUNKS, CHUNK_SIZE) + x.shape[1:]),
                env_state
            )

            def render_and_score_chunk_fn(carry, env_chunk):
                images_chunk = jax.vmap(jaxnav_render_fn)(env_chunk)

                member_logits_chunk = cnn_member_logit_fn(cnn_state, images_chunk)

                return carry, member_logits_chunk

            _, member_chunked = jax.lax.scan(render_and_score_chunk_fn, None, env_state_chunked)
            # (NUM_CHUNKS, K, CHUNK) -> (K, BATCH_SIZE)
            member_by_env = jnp.moveaxis(member_chunked, 1, 0).reshape((cnn_num_members, config["BATCH_SIZE"]))

            return None, (member_by_env, env_instances)

        rngs = jax.random.split(rng, config["NUM_BATCHES"])
        _, (member_logits, env_instances) = jax.lax.scan(_batch_step, None, rngs, config["NUM_BATCHES"])

        flat_env_instances = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), env_instances)
        member_logits = jnp.moveaxis(member_logits, 1, 0).reshape((cnn_num_members, -1))
        member_spread = member_logits.std(axis=0)     # per-level disagreement (raw logits)
        member_spread_z = standardized_member_spread(member_logits)
        learnability = member_logits.mean(axis=0)     # ensemble score
        learnability = normalize_scores(learnability)
        rng, select_rng = jax.random.split(rng)
        top_indices = select_environments(select_rng, learnability, target_mu)
        top_instances = jax.tree.map(lambda x: x.at[top_indices].get(), flat_env_instances)
        
        bottom_indices = jnp.argsort(learnability)[:20]
        bottom_instances = jax.tree.map(lambda x: x.at[bottom_indices].get(), flat_env_instances)

        # CNN method: the selection score IS the CNN difficulty score
        top_scores = learnability.at[top_indices].get()
        bottom_scores = learnability.at[bottom_indices].get()

        diag = {
            "cnn/all_mean": learnability.mean(),
            "cnn/selected_mean": top_scores.mean(),
            "cnn/selected_std": top_scores.std(),
            "cnn/selected_min": top_scores.min(),
            "cnn/selected_max": top_scores.max(),
            "cnn/tracking_error": jnp.abs(top_scores - target_mu).mean(),
            "hist/cnn_all": learnability,
            "hist/cnn_selected": top_scores,
        }
        
        if cnn_num_members > 1:
            diag.update({
                "cnn/member_spread_all": member_spread.mean(),
                "cnn/member_spread_selected": member_spread.at[top_indices].get().mean(),
                "cnn/member_spread_z_all": member_spread_z.mean(),
                "cnn/member_spread_z_selected": member_spread_z.at[top_indices].get().mean(),
                "cnn/member_rank_agreement": pairwise_rank_agreement(member_logits),
            })
        return top_scores, top_instances, bottom_scores, bottom_instances, jnp.zeros(20), bottom_instances, top_scores, bottom_scores, diag


    @jax.jit
    def get_learnability_set_dijkstra(rng, target_mu):
        """Heuristic-difficulty curriculum: identical to the cnn path, but
        the raw score is the shortest-path length (dijkstra_batch_scores) —
        no rendering, no images, no input-domain handling. Diagnostics
        reuse the cnn/* keys so the analysis tooling works unchanged."""
        def _batch_step(unused, rng):
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["BATCH_SIZE"])
            _, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
            env_instances = EnvInstance(
                agent_pos=env_state.pos,
                agent_theta=env_state.theta,
                goal_pos=env_state.goal,
                map_data=env_state.map_data,
                rew_lambda=env_state.rew_lambda,
            )
            scores_by_env = dijkstra_batch_scores(env_state)
            return None, (scores_by_env, env_instances)

        rngs = jax.random.split(rng, config["NUM_BATCHES"])
        _, (raw_scores, env_instances) = jax.lax.scan(_batch_step, None, rngs, config["NUM_BATCHES"])

        flat_env_instances = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), env_instances)
        raw_scores = raw_scores.flatten()
        learnability = normalize_scores(raw_scores)

        rng, select_rng = jax.random.split(rng)
        top_indices = select_environments(select_rng, learnability, target_mu)
        top_instances = jax.tree.map(lambda x: x.at[top_indices].get(), flat_env_instances)

        bottom_indices = jnp.argsort(learnability)[:20]
        bottom_instances = jax.tree.map(lambda x: x.at[bottom_indices].get(), flat_env_instances)

        top_scores = learnability.at[top_indices].get()
        bottom_scores = learnability.at[bottom_indices].get()

        diag = {
            "cnn/all_mean": learnability.mean(),
            "cnn/selected_mean": top_scores.mean(),
            "cnn/selected_std": top_scores.std(),
            "cnn/selected_min": top_scores.min(),
            "cnn/selected_max": top_scores.max(),
            "cnn/tracking_error": jnp.abs(top_scores - target_mu).mean(),
            "dijkstra/raw_mean": raw_scores.mean(),
            "dijkstra/raw_selected_mean": raw_scores.at[top_indices].get().mean(),
            "dijkstra/unreachable_frac": (raw_scores >= DIJKSTRA_SENTINEL).mean(),
            "hist/cnn_all": learnability,
            "hist/cnn_selected": top_scores,
        }
        return top_scores, top_instances, bottom_scores, bottom_instances, jnp.zeros(20), bottom_instances, top_scores, bottom_scores, diag

    @jax.jit
    def get_learnability_set_random(rng):
        def _batch_step(unused, rng):
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["BATCH_SIZE"])
            obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
            
            env_instances = EnvInstance(
                agent_pos=env_state.pos,
                agent_theta=env_state.theta,
                goal_pos=env_state.goal,
                map_data=env_state.map_data,
                rew_lambda=env_state.rew_lambda,
            )
            
            #  random scores
            rng, rand_rng = jax.random.split(rng)
            learnability_by_env = jax.random.uniform(rand_rng, (config["BATCH_SIZE"],))
            
            return None, (learnability_by_env, env_instances)
            
        rngs = jax.random.split(rng, config["NUM_BATCHES"])
        _, (learnability, env_instances) = jax.lax.scan(_batch_step, None, rngs, config["NUM_BATCHES"]) 
        
        flat_env_instances = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), env_instances)
        learnability = learnability.flatten()
        top_indices = jnp.argsort(learnability)[-config["NUM_TO_SAVE"]:]
        top_instances = jax.tree.map(lambda x: x.at[top_indices].get(), flat_env_instances)
        
        bottom_indices = jnp.argsort(learnability)[:20]
        bottom_instances = jax.tree.map(lambda x: x.at[bottom_indices].get(), flat_env_instances)

        return learnability.at[top_indices].get(), top_instances, learnability.at[bottom_indices].get(), bottom_instances, jnp.zeros(20), bottom_instances, jnp.zeros(config["NUM_TO_SAVE"]), jnp.zeros(20), {}

    @jax.jit
    def get_learnability_set_standard(rng, network_params): #
        
        
        BATCH_ACTORS = config["BATCH_SIZE"] * env.num_agents
        
        
        def _batch_step(unused, rng):
            def _env_step(runner_state, unused):
                env_state, start_state, last_obs, last_done, hstate, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, BATCH_ACTORS)
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                )
                hstate, pi, value, _ = network.apply(network_params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["BATCH_SIZE"], env.num_agents
                )
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["BATCH_SIZE"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, 0)
                )(rng_step, env_state, env_act, start_state)
                if env.do_sep_reward:
                    reward = listify_reward(reward, do_batchify=True)
                else:
                    reward = batchify(reward, env.agents, BATCH_ACTORS).squeeze()
                done_batch = batchify(done, env.agents, BATCH_ACTORS).squeeze()
                train_mask = info["terminated"].swapaxes(0, 1).reshape(-1)
                # train_mask = batchify(info["terminated"], env.agents, BATCH_ACTORS).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    reward,
                    log_prob.squeeze(),
                    obs_batch,
                    train_mask,
                    info,
                )
                runner_state = (env_state, start_state, obsv, done_batch, hstate, rng)
                return runner_state, transition
            
            @partial(jax.vmap, in_axes=(None, 1, 1, 1))
            @partial(jax.jit, static_argnums=(0,))
            def _calc_outcomes_by_agent(max_steps: int, dones, returns, info):
                idxs = jnp.arange(max_steps)
                @partial(jax.vmap, in_axes=(0, 0))
                def __ep_outcomes(start_idx, end_idx): 
                    mask = (idxs > start_idx) & (idxs <= end_idx) & (end_idx != max_steps)
                    r = jnp.sum(returns * mask)
                    success = jnp.sum(info["GoalR"] * mask)
                    collision = jnp.sum((info["MapC"] + info["AgentC"]) * mask)
                    timeo = jnp.sum(info["TimeO"] * mask)
                    l = end_idx - start_idx
                    #jax.debug.breakpoint()
                    return r, success, collision, timeo, l
                
                done_idxs = jnp.argwhere(dones, size=10, fill_value=max_steps).squeeze()
                mask_done = jnp.where(done_idxs == max_steps, False, True)
                #mask_done = jnp.nonzero(done_idxs != max_steps, size = max_steps)
                #jax.debug.breakpoint()
                ep_return, success, collision, timeo, length = __ep_outcomes(jnp.concatenate([jnp.array([-1]), done_idxs[:-1]]), done_idxs)        
                #jax.debug.breakpoint()

                return {"ep_return": ep_return.mean(where=mask_done),
                        "num_episodes": mask_done.sum(),
                        "success_rate": success.mean(where=mask_done),
                        "collision_rate": collision.mean(where=mask_done),
                        "timeout_rate": timeo.mean(where=mask_done),
                        "ep_len": length.mean(where=mask_done),
                    }
            
            # sample envs
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["BATCH_SIZE"])
            obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
            env_instances = EnvInstance(
                agent_pos=env_state.pos,
                agent_theta=env_state.theta,
                goal_pos=env_state.goal,
                map_data=env_state.map_data,
                rew_lambda=env_state.rew_lambda,
            )
            #### start here
            init_hstate = ScannedRNN.initialize_carry(BATCH_ACTORS, t_config["HIDDEN_SIZE"])
            #jax.debug.breakpoint()
            runner_state = (env_state, env_state, obsv, jnp.zeros((BATCH_ACTORS), dtype=bool), init_hstate, rng)
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["ROLLOUT_STEPS"]
            )
            print('traj batch done', traj_batch.done.shape)
            print('traj batch info', traj_batch.info["NumC"].shape)
            done_by_env = traj_batch.done.reshape((-1, env.num_agents, config["BATCH_SIZE"]))
            reward_by_env = traj_batch.reward.reshape((-1, env.num_agents, config["BATCH_SIZE"]))
            info_by_actor = jax.tree.map(lambda x: x.swapaxes(2, 1).reshape((-1, BATCH_ACTORS)), traj_batch.info)
            print('done_by_env', done_by_env.shape)
            print('reward_by_env', reward_by_env.shape)
            print('info_by_actor', info_by_actor)
            o = _calc_outcomes_by_agent(config["ROLLOUT_STEPS"], traj_batch.done, traj_batch.reward, info_by_actor)
            print('ooutcomes', o)
            #jax.debug.breakpoint()
            success_by_env = o["success_rate"].reshape((env.num_agents, config["BATCH_SIZE"]))
            solvability_by_env = success_by_env.mean(axis=0)
            learnability_by_env = (success_by_env * (1 - success_by_env)).sum(axis=0)

            print('learnability_by_env', learnability_by_env)
            return None, (learnability_by_env, solvability_by_env, env_instances)
            
        rngs = jax.random.split(rng, config["NUM_BATCHES"])
        #jax.debug.breakpoint()
        _, (learnability, solvability, env_instances) = jax.lax.scan(_batch_step, None, rngs, config["NUM_BATCHES"]) # # TODO learnability set has nan values FIX
        flat_env_instances = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), env_instances)
        learnability = learnability.flatten()
        ###jax.debug.breakpoint()
        top_1000 = jnp.argsort(learnability)[-config["NUM_TO_SAVE"]:]
        #jax.debug.print("top 1000 {}", top_1000)
        
        top_1000_instances = jax.tree.map(lambda x: x.at[top_1000].get(), flat_env_instances)
        #jax.debug.print('{}top 1000 instances', top_1000_instances)
        
        bottom_20 = jnp.argsort(learnability)[:20]
        bottom_20_instances = jax.tree.map(lambda x: x.at[bottom_20].get(), flat_env_instances)
        
        solvability = solvability.flatten()
        unsolvable_indices = jnp.argsort(solvability)[:20]
        unsolvable_instances = jax.tree.map(lambda x: x.at[unsolvable_indices].get(), flat_env_instances)
        
        diag = {
            "sfl/batch_mean": learnability.mean(),
            "sfl/batch_max": learnability.max(),
            "solvability/batch_mean": solvability.mean(),
            "hist/sfl_all": learnability,
        }
        return learnability.at[top_1000].get(), top_1000_instances, learnability.at[bottom_20].get(), bottom_20_instances, solvability.at[unsolvable_indices].get(), unsolvable_instances, jnp.zeros(config["NUM_TO_SAVE"]), jnp.zeros(20), diag
        
    
    def get_learnability_set_progress(rng, network_params):
        """Dense-learnability arms: identical rollout protocol to standard
        SFL; only the per-level selection score differs by learn_method
        ("progress" = var of end-based progress, "progress_mindist" = var of
        closest-approach progress, "progress_mean" = mp*(1-mp)). Episode
        outcome math lives in the module-level calc_progress_outcomes_by_agent
        (unit-tested by verify_progress_arms.py). ALL candidate scores plus
        binary p(1-p) are computed on the same rollouts every cycle and logged
        (sfl/*, progress/*, agree/*), so every run doubles as a score-level
        comparison of all four metrics regardless of which one selects."""

        BATCH_ACTORS = config["BATCH_SIZE"] * env.num_agents

        def _batch_step(unused, rng):
            def _env_step(runner_state, unused):
                env_state, start_state, last_obs, last_done, hstate, rng = runner_state

                # distance (the carry holds the freshly reset state)
                d_pre = jnp.linalg.norm(env_state.pos - env_state.goal, axis=-1)

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, BATCH_ACTORS)
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                )
                hstate, pi, value, _ = network.apply(network_params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["BATCH_SIZE"], env.num_agents
                )
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["BATCH_SIZE"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, 0)
                )(rng_step, env_state, env_act, start_state)
                info["DPre"] = d_pre
                if env.do_sep_reward:
                    reward = listify_reward(reward, do_batchify=True)
                else:
                    reward = batchify(reward, env.agents, BATCH_ACTORS).squeeze()
                done_batch = batchify(done, env.agents, BATCH_ACTORS).squeeze()
                train_mask = info["terminated"].swapaxes(0, 1).reshape(-1)
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    reward,
                    log_prob.squeeze(),
                    obs_batch,
                    train_mask,
                    info,
                )
                runner_state = (env_state, start_state, obsv, done_batch, hstate, rng)
                return runner_state, transition

            # sample envs
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["BATCH_SIZE"])
            obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
            env_instances = EnvInstance(
                agent_pos=env_state.pos,
                agent_theta=env_state.theta,
                goal_pos=env_state.goal,
                map_data=env_state.map_data,
                rew_lambda=env_state.rew_lambda,
            )
            init_hstate = ScannedRNN.initialize_carry(BATCH_ACTORS, t_config["HIDDEN_SIZE"])
            runner_state = (env_state, env_state, obsv, jnp.zeros((BATCH_ACTORS), dtype=bool), init_hstate, rng)
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["ROLLOUT_STEPS"]
            )
            info_by_actor = jax.tree.map(lambda x: x.swapaxes(2, 1).reshape((-1, BATCH_ACTORS)), traj_batch.info)
            o = calc_progress_outcomes_by_agent(config["ROLLOUT_STEPS"], traj_batch.global_done, traj_batch.reward, info_by_actor)

            def _by_env(key, reduce):
                arr = o[key].reshape((env.num_agents, config["BATCH_SIZE"]))
                return arr.sum(axis=0) if reduce == "sum" else arr.mean(axis=0)

            success_by_env = o["success_rate"].reshape((env.num_agents, config["BATCH_SIZE"]))
            solvability_by_env = success_by_env.mean(axis=0)
            binary_learnability_by_env = (success_by_env * (1 - success_by_env)).sum(axis=0)

            return None, (binary_learnability_by_env, solvability_by_env,
                          _by_env("progress_end_var", "sum"),
                          _by_env("progress_min_var", "sum"),
                          _by_env("progress_end_mean", "mean"),
                          _by_env("progress_min_mean", "mean"),
                          _by_env("d_start_mean", "mean"),
                          _by_env("num_episodes", "sum"),
                          env_instances)

        rngs = jax.random.split(rng, config["NUM_BATCHES"])
        _, (binary_learnability, solvability, end_var, min_var, end_mp, min_mp,
            d_start_mean, num_eps, env_instances) = jax.lax.scan(
            _batch_step, None, rngs, config["NUM_BATCHES"])
        flat_env_instances = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), env_instances)
        binary_learnability = binary_learnability.flatten()
        solvability = solvability.flatten()
        end_var = end_var.flatten()
        min_var = min_var.flatten()
        end_mp = end_mp.flatten()
        min_mp = min_mp.flatten()
        d_start_mean = d_start_mean.flatten()
        num_eps = num_eps.flatten()

        # All candidate scores
        mp_score = end_mp * (1.0 - end_mp)   # peaks at mp=0.5; = 0.25-(mp-0.5)^2
        candidates = {
            "binary": binary_learnability,
            "end_var": end_var,
            "min_var": min_var,
            "mp_score": mp_score,
        }
        selection = {"progress": end_var,
                     "progress_mindist": min_var,
                     "progress_mean": mp_score}[learn_method]

        top_1000 = jnp.argsort(selection)[-config["NUM_TO_SAVE"]:]
        top_1000_instances = jax.tree.map(lambda x: x.at[top_1000].get(), flat_env_instances)

        bottom_20 = jnp.argsort(selection)[:20]
        bottom_20_instances = jax.tree.map(lambda x: x.at[bottom_20].get(), flat_env_instances)

        unsolvable_indices = jnp.argsort(solvability)[:20]
        unsolvable_instances = jax.tree.map(lambda x: x.at[unsolvable_indices].get(), flat_env_instances)

        # ---- density / agreement diagnostics (all on the full 5000-level batch) ----
        def _safe_corr(a, b):
            ok = (jnp.std(a) > 1e-8) & (jnp.std(b) > 1e-8)
            c = jnp.corrcoef(a, b)[0, 1]
            return jnp.where(ok, c, 0.0)

        def _ranks(x):
            order = jnp.argsort(x)
            return jnp.zeros_like(x).at[order].set(jnp.arange(x.shape[0], dtype=x.dtype))

        diag = {
            # the arm's own selection score, under uniform keys for cross-arm plots
            "selection/score_mean": selection.mean(),
            "selection/score_std": selection.std(),
            "selection/score_max": selection.max(),
            "selection/frac_nonzero": (selection > 1e-8).mean(),
            "selection/selected_mean": selection.at[top_1000].get().mean(),
            "selection/selected_solvability": solvability.at[top_1000].get().mean(),
            "selection/selected_binary_learnability": binary_learnability.at[top_1000].get().mean(),
            "selection/selected_mp": end_mp.at[top_1000].get().mean(),
            "selection/selected_min_mp": min_mp.at[top_1000].get().mean(),
            # binary l on the same rollouts 
            "sfl/batch_mean": binary_learnability.mean(),
            "sfl/batch_max": binary_learnability.max(),
            "sfl/frac_nonzero": (binary_learnability > 1e-8).mean(),
            # every candidate's density + level
            "progress/end_var_mean": end_var.mean(),
            "progress/end_var_frac_nonzero": (end_var > 1e-8).mean(),
            "progress/min_var_mean": min_var.mean(),
            "progress/min_var_frac_nonzero": (min_var > 1e-8).mean(),
            "progress/mp_score_mean": mp_score.mean(),
            "progress/mp_score_frac_nonzero": (mp_score > 1e-8).mean(),
            "progress/end_mp_mean": end_mp.mean(),
            "progress/min_mp_mean": min_mp.mean(),
            # near-miss evidence: how far the two progress definitions diverge
            "progress/near_miss_gap": (min_mp - end_mp).mean(),
            "progress/d_start_mean": d_start_mean.mean(),
            "progress/num_episodes_mean": num_eps.mean(),
            "solvability/batch_mean": solvability.mean(),
            "hist/sfl_all": binary_learnability,
            "hist/progress_end_var": end_var,
            "hist/progress_min_var": min_var,
            "hist/progress_mp_score": mp_score,
        }
        # pairwise agreement among all four candidate scores
        _names = list(candidates)
        _tops = {n: jnp.argsort(candidates[n])[-config["NUM_TO_SAVE"]:] for n in _names}
        for i, a in enumerate(_names):
            for b in _names[i + 1:]:
                diag[f"agree/{a}_vs_{b}_pearson"] = _safe_corr(candidates[a], candidates[b])
                diag[f"agree/{a}_vs_{b}_spearman"] = _safe_corr(_ranks(candidates[a]), _ranks(candidates[b]))
                diag[f"agree/{a}_vs_{b}_top1000_overlap"] = jnp.isin(_tops[a], _tops[b]).mean()

        return (selection.at[top_1000].get(), top_1000_instances,
                selection.at[bottom_20].get(), bottom_20_instances,
                solvability.at[unsolvable_indices].get(), unsolvable_instances,
                jnp.zeros(config["NUM_TO_SAVE"]), jnp.zeros(20), diag)


    @partial(jax.jit, static_argnums=(2,))  # cnn_graphdef is static
    def get_learnability_set_hybrid(rng, network_params, cnn_graphdef, cnn_state, target_mu):
        """Hybrid: runs agent rollouts (SFL scores + solvability) AND CNN scoring,
        then combines them via config['HYBRID_MODE'] to select environments."""

        BATCH_ACTORS = config["BATCH_SIZE"] * env.num_agents
        
        def _batch_step(unused, rng):

            def _env_step(runner_state, unused):
                env_state, start_state, last_obs, last_done, hstate, rng = runner_state
                rng, _rng = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, BATCH_ACTORS)
                ac_in = (obs_batch[np.newaxis, :], last_done[np.newaxis, :])
                hstate, pi, value, _ = network.apply(network_params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action, env.agents, config["BATCH_SIZE"], env.num_agents)
                env_act = {k: v.squeeze() for k, v in env_act.items()}
                
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["BATCH_SIZE"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, 0)
                )(rng_step, env_state, env_act, start_state)
                if env.do_sep_reward:
                    reward = listify_reward(reward, do_batchify=True)
                else:
                    reward = batchify(reward, env.agents, BATCH_ACTORS).squeeze()
                done_batch = batchify(done, env.agents, BATCH_ACTORS).squeeze()
                train_mask = info["terminated"].swapaxes(0, 1).reshape(-1)
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    reward,
                    log_prob.squeeze(),
                    obs_batch,
                    train_mask,
                    info,
                )
                runner_state = (env_state, start_state, obsv, done_batch, hstate, rng)
                return runner_state, transition
            
            @partial(jax.vmap, in_axes=(None, 1, 1, 1))
            @partial(jax.jit, static_argnums=(0,))
            def _calc_outcomes_by_agent(max_steps, dones, returns, info):
                idxs = jnp.arange(max_steps)
                @partial(jax.vmap, in_axes=(0, 0))
                def __ep_outcomes(start_idx, end_idx):
                    mask = (idxs > start_idx) & (idxs <= end_idx) & (end_idx != max_steps)
                    r = jnp.sum(returns * mask)
                    success = jnp.sum(info["GoalR"] * mask)
                    collision = jnp.sum((info["MapC"] + info["AgentC"]) * mask)
                    timeo = jnp.sum(info["TimeO"] * mask)
                    l = end_idx - start_idx
                    return r, success, collision, timeo, l
                done_idxs = jnp.argwhere(dones, size=10, fill_value=max_steps).squeeze()
                mask_done = jnp.where(done_idxs == max_steps, False, True)
                ep_return, success, collision, timeo, length = __ep_outcomes(
                    jnp.concatenate([jnp.array([-1]), done_idxs[:-1]]), done_idxs
                )
                return {
                    "success_rate": success.mean(where=mask_done),
                }
            
            # Generate environments
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["BATCH_SIZE"])
            obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
            env_instances = EnvInstance(
                agent_pos=env_state.pos,
                agent_theta=env_state.theta,
                goal_pos=env_state.goal,
                map_data=env_state.map_data,
                rew_lambda=env_state.rew_lambda,
            )
            
            # ---- Agent rollout for SFL scores ----
            init_hstate_batch = ScannedRNN.initialize_carry(BATCH_ACTORS, t_config["HIDDEN_SIZE"])
            runner_state = (env_state, env_state, obsv, jnp.zeros((BATCH_ACTORS), dtype=bool), init_hstate_batch, rng)
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["ROLLOUT_STEPS"])
            
            info_by_actor = jax.tree.map(lambda x: x.swapaxes(2, 1).reshape((-1, BATCH_ACTORS)), traj_batch.info)
            o = _calc_outcomes_by_agent(config["ROLLOUT_STEPS"], traj_batch.done, traj_batch.reward, info_by_actor)
            success_by_env = o["success_rate"].reshape((env.num_agents, config["BATCH_SIZE"]))
            solvability_by_env = success_by_env.mean(axis=0)                          # (BATCH_SIZE,)
            sfl_scores = (success_by_env * (1 - success_by_env)).sum(axis=0)          # (BATCH_SIZE,) range [0, 0.25]
            
            # ---- CNN scoring ----
            CHUNK_SIZE = min(config["BATCH_SIZE"], 100)
            NUM_CHUNKS = config["BATCH_SIZE"] // CHUNK_SIZE
            env_state_chunked = jax.tree.map(
                lambda x: x.reshape((NUM_CHUNKS, CHUNK_SIZE) + x.shape[1:]), env_state
            )
            def render_and_score_chunk_fn(carry, env_chunk):
                images_chunk = jax.vmap(jaxnav_render_fn)(env_chunk)
                # Raw per-member logits (K, CHUNK); ensemble score = member mean
                member_logits_chunk = cnn_member_logit_fn(cnn_state, images_chunk)
                return carry, member_logits_chunk
            _, cnn_member_chunked = jax.lax.scan(render_and_score_chunk_fn, None, env_state_chunked)
            # (NUM_CHUNKS, K, CHUNK) -> (K, BATCH_SIZE) raw member logits

            cnn_member_by_env = jnp.moveaxis(cnn_member_chunked, 1, 0).reshape((cnn_num_members, config["BATCH_SIZE"]))

            return None, (sfl_scores, cnn_member_by_env, solvability_by_env, env_instances)
        
        # Run all batches
        rngs = jax.random.split(rng, config["NUM_BATCHES"])
        _, (sfl_scores, cnn_member_scores, solvability_p, env_instances) = jax.lax.scan(
            _batch_step, None, rngs, config["NUM_BATCHES"]
        )

        # Flatten across batches
        flat_env_instances = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), env_instances)
        sfl_scores = sfl_scores.flatten()           # (TOTAL,) range [0, 0.25]

        cnn_member_scores = jnp.moveaxis(cnn_member_scores, 1, 0).reshape((cnn_num_members, -1))
        cnn_member_spread = cnn_member_scores.std(axis=0)   
        cnn_member_spread_z = standardized_member_spread(cnn_member_scores)  
        cnn_scores = cnn_member_scores.mean(axis=0)         
        solvability_p = solvability_p.flatten()     

        cnn_norm = normalize_scores(cnn_scores)  
        
        # ---- frontier from THIS cycles candidate rollouts ----
        p_lvl = jnp.nan_to_num(solvability_p, nan=0.0)
        fr = frontier_fit(cnn_norm, p_lvl, FRONTIER_NBINS, FRONTIER_MIN_INFORMATIVE_BINS, FRONTIER_INFORMATIVE_P)
        if CURRICULUM_STRATEGY_CFG == "calibrated_frontier":
            mu_used = jnp.where(fr["use_frontier"], fr["mu_star"], target_mu)
        else:
            mu_used = target_mu

        # ---- CNN term: proximity to mu_usedor expected learnability (priority) ----
        cnn_proximity = 1.0 - jnp.abs(cnn_norm - mu_used)         # (TOTAL,) range [0, 1]
        if CURRICULUM_STRATEGY_CFG == "calibrated_frontier" and FRONTIER_MODE == "priority":
            p_at_c = jnp.interp(cnn_norm, fr["bin_c"], fr["iso_p"])
            cnn_term = jnp.where(fr["use_frontier"], 4.0 * p_at_c * (1.0 - p_at_c), cnn_proximity)
        else:
            cnn_term = cnn_proximity
        
        # ---- Compound scoring based on HYBRID_MODE ----
        hybrid_mode = config.get("HYBRID_MODE", "linear")
        
        if hybrid_mode == "linear":
            compound_scores = (4.0 * sfl_scores) + cnn_term
            
        elif hybrid_mode == "soft_handoff":
            batch_p = jnp.mean(solvability_p)
            alpha = jnp.clip(batch_p / 0.1, 0.0, 1.0)
            compound_scores = (alpha * (4.0 * sfl_scores)) + ((1.0 - alpha) * cnn_proximity)
            
        elif hybrid_mode == "learnability_weighted":
            cnn_weight = (0.25 - sfl_scores) * 4.0
            cnn_weight = jnp.where(solvability_p > 0.9, 0.0, cnn_weight)  # mastery safeguard
            compound_scores = (4.0 * sfl_scores) + (cnn_weight * cnn_proximity)

            
        elif hybrid_mode == "multiplicative":
            cnn_filter = jnp.exp(-((cnn_norm - mu_used)**2) / (2 * 0.1**2))
            compound_scores = (sfl_scores + 0.01) * cnn_filter
        
        else:
            # Fallback: pure SFL
            compound_scores = sfl_scores
        
        # ---- Select top environments ----
        top_indices = jnp.argsort(compound_scores)[-config["NUM_TO_SAVE"]:]
        top_instances = jax.tree.map(lambda x: x.at[top_indices].get(), flat_env_instances)
        
        # ---- Bottom 20 (global worst by compound score) ----
        bottom_indices = jnp.argsort(compound_scores)[:20]
        bottom_instances = jax.tree.map(lambda x: x.at[bottom_indices].get(), flat_env_instances)
        
        # ---- Unsolvable (lowest solvability) ----
        unsolvable_indices = jnp.argsort(solvability_p)[:20]
        unsolvable_instances = jax.tree.map(lambda x: x.at[unsolvable_indices].get(), flat_env_instances)
        
        # ---- Curriculum diagnostics (logged per eval cycle) ----
        cnn_sel = cnn_norm.at[top_indices].get()
        sfl_sel = sfl_scores.at[top_indices].get()
        solv_sel = solvability_p.at[top_indices].get()
        _sfl_c = sfl_scores - sfl_scores.mean()
        _cnn_c = cnn_norm - cnn_norm.mean()
        corr = (_sfl_c * _cnn_c).mean() / (sfl_scores.std() * cnn_norm.std() + 1e-8)
        diag = {
            "cnn/all_mean": cnn_norm.mean(),
            "cnn/selected_mean": cnn_sel.mean(),
            "cnn/selected_std": cnn_sel.std(),
            "cnn/selected_min": cnn_sel.min(),
            "cnn/selected_max": cnn_sel.max(),
            "cnn/tracking_error": jnp.abs(cnn_sel - mu_used).mean(),
            "sfl/batch_mean": sfl_scores.mean(),
            "sfl/batch_max": sfl_scores.max(),
            "sfl/selected_mean": sfl_sel.mean(),
            "cnn_proximity/batch_mean": cnn_proximity.mean(),
            "solvability/batch_mean": solvability_p.mean(),
            "solvability/selected_mean": solv_sel.mean(),
            "hybrid/alpha_soft_handoff": jnp.clip(solvability_p.mean() / 0.1, 0.0, 1.0),
            # calibrated-frontier diagnostics (logged for every hybrid arm)
            "frontier/mu_star": fr["mu_star"],
            "frontier/mu_used": mu_used,
            "frontier/use_frontier": fr["use_frontier"].astype(jnp.float32),
            "frontier/n_informative_bins": fr["n_informative"].astype(jnp.float32),
            "frontier/selected_frac_intermediate": ((solv_sel > 0.05) & (solv_sel < 0.95)).mean(),
            **{f"frontier/c_bin{_i:02d}": fr["bin_c"][_i] for _i in range(FRONTIER_NBINS)},
            **{f"frontier/p_bin{_i:02d}": fr["bin_p"][_i] for _i in range(FRONTIER_NBINS)},
            **{f"frontier/iso_bin{_i:02d}": fr["iso_p"][_i] for _i in range(FRONTIER_NBINS)},
            "corr/sfl_vs_cnn": corr,
            "hist/cnn_all": cnn_norm,
            "hist/cnn_selected": cnn_sel,
            "hist/sfl_all": sfl_scores,
        }
        # Ensemble disagreement (observation only, never fed into selection):
        if cnn_num_members > 1:
            diag.update({
                "cnn/member_spread_all": cnn_member_spread.mean(),
                "cnn/member_spread_selected": cnn_member_spread.at[top_indices].get().mean(),
                "cnn/member_spread_z_all": cnn_member_spread_z.mean(),
                "cnn/member_spread_z_selected": cnn_member_spread_z.at[top_indices].get().mean(),
                "cnn/member_rank_agreement": pairwise_rank_agreement(cnn_member_scores),
            })
        return compound_scores.at[top_indices].get(), top_instances, compound_scores.at[bottom_indices].get(), bottom_instances, solvability_p.at[unsolvable_indices].get(), unsolvable_instances, cnn_norm.at[top_indices].get(), cnn_norm.at[bottom_indices].get(), diag

    # TRAIN LOOP
    def train_step(runner_state_instances, unused):
        # COLLECT TRAJECTORIES
        runner_state, instances = runner_state_instances
        num_env_instances = instances.agent_pos.shape[0]

        def _env_step(runner_state, unused):
            train_state, env_state, start_state, last_obs, last_done, hstate, update_steps, rng = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            obs_batch = batchify(last_obs, env.agents, t_config["NUM_ACTORS"])
            ac_in = (
                obs_batch[np.newaxis, :],
                last_done[np.newaxis, :],
            )
            hstate, pi, value, dormancy = network.apply(train_state.params, hstate, ac_in)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)
            env_act = unbatchify(
                action, env.agents, t_config["NUM_ENVS"], env.num_agents
            )
            env_act = {k: v.squeeze() for k, v in env_act.items()}
            # jax.debug-print()
            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, t_config["NUM_ENVS"])
            obsv, env_state, reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, 0)
            )(rng_step, env_state, env_act, start_state)
            if env.do_sep_reward:
                reward = listify_reward(reward, do_batchify=True)
            else:
                reward = batchify(reward, env.agents, t_config["NUM_ACTORS"]).squeeze()
            done_batch = batchify(done, env.agents, t_config["NUM_ACTORS"]).squeeze()
            train_mask = info["terminated"].swapaxes(0, 1).reshape(-1)
            # train_mask = batchify(info["terminated"], env.agents, t_config["NUM_ACTORS"]).squeeze()
            transition = Transition(
                jnp.tile(done["__all__"], env.num_agents),
                last_done,
                action.squeeze(),
                value.squeeze(),
                reward,
                log_prob.squeeze(),
                obs_batch,
                train_mask,
                info,
            )
            runner_state = (train_state, env_state, start_state, obsv, done_batch, hstate, update_steps, rng)
            return runner_state, (transition, dormancy)

        initial_hstate = runner_state[-3]
        runner_state, traj_batch_dormancy = jax.lax.scan(
            _env_step, runner_state, None, t_config["NUM_STEPS"]
        )
        traj_batch, dormancy = traj_batch_dormancy
        dormancy = jax.tree.map(lambda x: x.mean(), dormancy)
        
        @partial(jax.vmap, in_axes=(1, 1))
        def _calc_ep_return_by_agent(dones, returns):
            idxs = jnp.arange(t_config["NUM_STEPS"])
            
            @partial(jax.vmap, in_axes=(None, 0, 0))
            def __ep_returns(rews, start_idx, end_idx): 
                mask = (idxs > start_idx) & (idxs <= end_idx) & (end_idx != t_config["NUM_STEPS"])
                r = jnp.sum(rews * mask, axis=0)
                l = end_idx - start_idx
                return r, l
            
            done_idxs = jnp.argwhere(dones, size=t_config["NUM_STEPS"]//4, fill_value=t_config["NUM_STEPS"]).squeeze()
            mask_done = jnp.where(done_idxs == t_config["NUM_STEPS"], False, True)
            r, l = __ep_returns(returns, jnp.concatenate([jnp.array([-1]), done_idxs[:-1]]), done_idxs)                
            return {"episodic_return_per_agent": r.mean(where=mask_done), "episodic_length_per_agent": l.mean(where=mask_done)}
        
        if env.do_sep_reward:
            reward_by_env = traj_batch.reward.sum(axis=-1)
        else:
            reward_by_env = traj_batch.reward
        episodic_return_length = _calc_ep_return_by_agent(traj_batch.done, reward_by_env)
        episodic_return_length = jax.tree.map(lambda x: x.mean(), episodic_return_length)
        # CALCULATE ADVANTAGE
        train_state, env_state, start_state, last_obs, last_done, hstate, update_steps, rng = runner_state
        last_obs_batch = batchify(last_obs, env.agents, t_config["NUM_ACTORS"])
        ac_in = (
            last_obs_batch[np.newaxis, :],
            last_done[np.newaxis, :],
        )
        _, _, last_val, _ = network.apply(train_state.params, hstate, ac_in)
        last_val = last_val.squeeze()
        print('last_val shape', last_val.shape)
        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition: Transition):
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.global_done, 
                    transition.value,
                    transition.reward,
                )
                delta = reward + t_config["GAMMA"] * next_value * (1 - done) - value
                gae = (
                    delta
                    + t_config["GAMMA"] * t_config["GAE_LAMBDA"] * (1 - done) * gae
                )
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
            )
            return advantages, advantages + traj_batch.value

        advantages, targets = _calculate_gae(traj_batch, last_val)

        # UPDATE NETWORK
        def _update_epoch(update_state, unused):
            def _update_minbatch(train_state, batch_info):
                init_hstate, traj_batch, advantages, targets = batch_info

                def _loss_fn_masked(params, init_hstate, traj_batch, gae, targets):
                                            
                    # RERUN NETWORK
                    _, pi, value, _ = network.apply(
                        params,
                        init_hstate.transpose(),
                        (traj_batch.obs, traj_batch.done),
                    )
                    log_prob = pi.log_prob(traj_batch.action)

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-t_config["CLIP_EPS"], t_config["CLIP_EPS"])
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = 0.5 * jnp.maximum(
                        value_losses, value_losses_clipped
                    )
                    if env.do_sep_reward:
                        value_loss_sparse = value_loss[..., REWARD_COMPONENT_SPARSE].mean(where=( jnp.logical_not(traj_batch.mask) ))
                        value_loss_dense  = value_loss[..., REWARD_COMPONENT_DENSE].mean(where=( jnp.logical_not(traj_batch.mask) ))
                        
                        critic_loss = t_config["VF_COEF"] * (value_loss_sparse + value_loss_dense)
                    else:
                        critic_loss = t_config["VF_COEF"] * value_loss.mean(where=(jnp.logical_not(traj_batch.mask)))
                    
                    # CALCULATE ACTOR LOSS
                    logratio = log_prob - traj_batch.log_prob
                    ratio = jnp.exp(logratio)
                    if env.do_sep_reward:
                        gae = gae.sum(axis=-1)
                    gae = (gae - gae.mean(where=(jnp.logical_not(traj_batch.mask)))) / (gae.std(where=(jnp.logical_not(traj_batch.mask))) + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - t_config["CLIP_EPS"],
                            1.0 + t_config["CLIP_EPS"],
                        )
                        * gae
                    )
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean(where=(jnp.logical_not(traj_batch.mask)))
                    entropy = pi.entropy().mean(where=(jnp.logical_not(traj_batch.mask)))
                    
                    # debug
                    approx_kl = jax.lax.stop_gradient(
                        ((ratio - 1) - logratio).mean()
                    )
                    clipfrac = jax.lax.stop_gradient(
                        (jnp.abs(ratio - 1) > t_config["CLIP_EPS"]).mean()
                    )

                    total_loss = (
                        loss_actor
                        + critic_loss
                        - t_config["ENT_COEF"] * entropy
                    )
                    return total_loss, (value_loss, loss_actor, entropy, ratio, approx_kl, clipfrac)

                grad_fn = jax.value_and_grad(_loss_fn_masked, has_aux=True)
                total_loss, grads = grad_fn(
                    train_state.params, init_hstate, traj_batch, advantages, targets
                )
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, total_loss

            (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            ) = update_state
            rng, _rng = jax.random.split(rng)

            init_hstate = jnp.reshape(
                init_hstate, (t_config["HIDDEN_SIZE"], t_config["NUM_ACTORS"])
            )
            batch = (
                init_hstate,
                traj_batch,
                advantages.squeeze(),
                targets.squeeze(),
            )
            permutation = jax.random.permutation(_rng, t_config["NUM_ACTORS"])

            shuffled_batch = jax.tree.map(
                lambda x: jnp.take(x, permutation, axis=1), batch
            )

            minibatches = jax.tree.map( #shape mismatch? maybe because of args?
                lambda x: jnp.swapaxes(
                    jnp.reshape(
                        x,
                        [x.shape[0], t_config["NUM_MINIBATCHES"], -1]
                        + list(x.shape[2:]),
                    ),
                    1,
                    0,
                ),
                shuffled_batch,
            )

            train_state, total_loss = jax.lax.scan(
                _update_minbatch, train_state, minibatches
            )
            # total_loss = jax.tree.map(lambda x: x.mean(), total_loss)
            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            return update_state, total_loss

        # init_hstate = initial_hstate[None, :].squeeze().transpose()
        init_hstate = jax.tree.map(lambda x: x[None, :].squeeze().transpose(), initial_hstate)
        update_state = (
            train_state,
            init_hstate,
            traj_batch,
            advantages,
            targets,
            rng,
        )
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, t_config["UPDATE_EPOCHS"]
        )
        train_state = update_state[0]
        metric = traj_batch.info
        metric = jax.tree.map(
            lambda x: x.sum(axis=-1).reshape(
                (t_config["NUM_STEPS"], t_config["NUM_ENVS"])  # , env.num_agents
            ),
            traj_batch.info,
        )
       
        _NUM_GEN = t_config["NUM_ENVS_TO_GENERATE"]
        _goal_env = metric["GoalR"].sum(axis=0)                       # (NUM_ENVS,) successes
        _eps_env = metric["NumC"].sum(axis=0)                         # (NUM_ENVS,) completed episodes
        _succ_env = jnp.where(_eps_env > 0, _goal_env / jnp.maximum(_eps_env, 1), 0.0)
        def _half_stats(sl):
            return {
                "success_env_weighted": _succ_env[sl].mean(),
                "success_term_weighted": _goal_env[sl].sum() / jnp.maximum(_eps_env[sl].sum(), 1),
                "episodes_per_env": _eps_env[sl].mean(),
            }
        metric["buffer"] = _half_stats(slice(_NUM_GEN, None))
        metric["generated"] = _half_stats(slice(0, _NUM_GEN))
        rng = update_state[-1]

        def callback(metric):
            safe_wandb_log(
                {
                    "train-term": metric["terminations"],
                    "train-buffer/": metric["buffer"],
                    "train-generated/": metric["generated"],
                    #"reward": metric["returned_episode_returns"],
                    
                    # "eval-collision": metric["test-metrics"]["collision-by-env"].mean(),
                    # "eval-timeout": metric["test-metrics"]["timeout-by-env"].mean(),
                    "env_step": metric["update_steps"]
                        * t_config["NUM_ENVS"]
                        * t_config["NUM_STEPS"],
                    "dormancy/": metric["dormancy"],
                    "env-metrics/": metric["env-metrics"],
                    # "mean_ued_score": metric["mean_ued_score"],
                    **metric["episodic_return_length"],
                    **metric["loss_info"],
                    "mean_lambda_val": metric["mean_lambda_val"],
                    "update_count": metric["update_steps"],
                }
            )

        dormancy_log = {
            "actor": dormancy.actor,
            "embedding": dormancy.embedding,
            "hidden": dormancy.hidden,
            "rnnout": dormancy.rnnout,
            "critic": dormancy.critic,
        }
        ratio0 = jnp.around(loss_info[1][3].at[0,0].get().mean(), decimals=6)
        loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
        metric["loss_info"] = {
            "total_loss": loss_info[0],
            "value_loss": loss_info[1][0],
            "actor_loss": loss_info[1][1],
            "entropy": loss_info[1][2],
            "ratio": loss_info[1][3],
            "ratio_0": ratio0,
            "approx_kl": loss_info[1][4],
            "clipfrac": loss_info[1][5],
            "mask_percentage": jnp.mean(traj_batch.mask),
        }
        metric["episodic_return_length"] = episodic_return_length
        metric["update_steps"] = update_steps
        metric["terminations"] = {k: traj_batch.info[k] for k in ["NumC", "GoalR", "AgentC", "MapC", "TimeO"]}
        metric["terminations"] = jax.tree.map(lambda x: x.sum(), metric["terminations"])
        metric["dormancy"] = dormancy_log
        metric["env-metrics"] = jax.tree.map(lambda x: x.mean(), jax.vmap(env.get_env_metrics)(start_state))
        metric["mean_lambda_val"] = env_state.rew_lambda.mean()
        jax.experimental.io_callback(callback, None, metric)
        
        # SAMPLE NEW ENVS
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, t_config["NUM_ENVS_TO_GENERATE"])
        obsv_gen, env_state_gen = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        
        rng, _rng = jax.random.split(rng)
        sampled_env_instances_idxs = jax.random.randint(_rng, (t_config["NUM_ENVS_FROM_SAMPLED"],), 0, num_env_instances)
        sampled_env_instances = jax.tree.map(lambda x: x.at[sampled_env_instances_idxs].get(), instances)
        obsv_sampled, env_state_sampled = jax.vmap(env.set_env_instance, in_axes=(0,))(sampled_env_instances)
        
        obsv = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=0), obsv_gen, obsv_sampled)
        env_state = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=0), env_state_gen, env_state_sampled)
        
        start_state = env_state
        hstate = ScannedRNN.initialize_carry(t_config["NUM_ACTORS"], t_config["HIDDEN_SIZE"])
        
        update_steps = update_steps + 1
        runner_state = (train_state, env_state, start_state, obsv, jnp.zeros((t_config["NUM_ACTORS"]), dtype=bool), hstate, update_steps, rng)
        return (runner_state, instances), metric
    
    def log_buffer(learnability, states, epoch, log_key="best_maps", cnn_scores=None):
        num_samples = states.pos.shape[0]
        rows = 2
        fig, axes = plt.subplots(rows, int(num_samples/rows), figsize=(20, 10))
        axes=axes.flatten()
        for i, ax in enumerate(axes):
            # ax.imshow(train_state.plr_buffer.get_sample(i))
            score = learnability[i]
            state = jax.tree.map(lambda x: x[i], states)

            env.init_render(ax, state, lidar=False, ticks_off=True)
            if cnn_scores is not None:
                ax.set_title(f'score: {score:.3f}\ncnn: {cnn_scores[i]:.3f}', fontsize=9)
            else:
                ax.set_title(f'score: {score:.3f}')
            ax.set_aspect('equal', 'box')

        plt.tight_layout()
        fig.canvas.draw()
        rgba_buffer = np.array(fig.canvas.buffer_rgba())
        im = Image.fromarray(rgba_buffer).convert("RGB")


        safe_wandb_log({log_key: wandb.Image(im), "update_count": int(epoch)})
        plt.close(fig)
    
    @partial(jax.jit, static_argnums=(2,)) # learn_method must be static!
    def train_and_eval_step(runner_state, eval_rng, learn_method, cnn_state, target_mu):
        
        learnability_rng, eval_singleton_rng, eval_sampled_rng, buffer_sample_rng = jax.random.split(eval_rng, 4)
        # -----------------------------------TRAIN---------------------------------------------
        if learn_method == "cnn":
            learnabilty_scores, instances, worst_scores, worst_instances, unsolvable_scores, unsolvable_instances, cnn_top_scores, cnn_worst_scores, curriculum_diag = get_learnability_set_cnn(
                learnability_rng,
                cnn_graphdef,
                cnn_state,
                target_mu
            )
        elif learn_method == "dijkstra":
            learnabilty_scores, instances, worst_scores, worst_instances, unsolvable_scores, unsolvable_instances, cnn_top_scores, cnn_worst_scores, curriculum_diag = get_learnability_set_dijkstra(
                learnability_rng,
                target_mu
            )
        elif learn_method == "standard":
            learnabilty_scores, instances, worst_scores, worst_instances, unsolvable_scores, unsolvable_instances, cnn_top_scores, cnn_worst_scores, curriculum_diag = get_learnability_set_standard(
                learnability_rng,
                runner_state[0].params
            )
        elif learn_method in ("progress", "progress_mean", "progress_mindist"):
            learnabilty_scores, instances, worst_scores, worst_instances, unsolvable_scores, unsolvable_instances, cnn_top_scores, cnn_worst_scores, curriculum_diag = get_learnability_set_progress(
                learnability_rng,
                runner_state[0].params
            )
        elif learn_method == "hybrid":
            learnabilty_scores, instances, worst_scores, worst_instances, unsolvable_scores, unsolvable_instances, cnn_top_scores, cnn_worst_scores, curriculum_diag = get_learnability_set_hybrid(
                learnability_rng,
                runner_state[0].params,
                cnn_graphdef,
                cnn_state,
                target_mu
            )
        else: # random
            learnabilty_scores, instances, worst_scores, worst_instances, unsolvable_scores, unsolvable_instances, cnn_top_scores, cnn_worst_scores, curriculum_diag = get_learnability_set_random(
                learnability_rng
            )
        #learnabilty_scores, instances = get_learnability_set(learnability_rng, runner_state[0].params)
        runner_state_instances = (runner_state, instances)
        ##jax.debug.print("learnabilityscores{}" , learnabilty_scores)
        runner_state_instances, metrics = jax.lax.scan(train_step, runner_state_instances, None, t_config["EVAL_FREQ"])
        
        goal_r = metrics["terminations"]["GoalR"].sum()
        agent_c = metrics["terminations"]["AgentC"].sum()
        map_c = metrics["terminations"]["MapC"].sum()
        time_o = metrics["terminations"]["TimeO"].sum()
        total_terms = goal_r + agent_c + map_c + time_o
        recent_success_rate = goal_r / (total_terms + 1e-8)        # LEGACY (all envs, termination-weighted)
        buffer_success_rate = metrics["buffer"]["success_env_weighted"].mean()       # FIXED measurement
        generated_success_rate = metrics["generated"]["success_env_weighted"].mean()
        buffer_success_term_weighted = metrics["buffer"]["success_term_weighted"].mean()



        # --------------------------------------EVAL-----------------------------------------
        #jax.debug.breakpoint() #np
        test_metrics = {
            "learnability_set_scores": learnabilty_scores,
            "learnability_set_mean_score": learnabilty_scores.mean(),
            "worst_learnability_scores": worst_scores,
            "worst_learnability_mean_score": worst_scores.mean(),
            "recent_success_rate": recent_success_rate,
            "buffer_success_rate": buffer_success_rate,
            "generated_success_rate": generated_success_rate,
            "buffer_success_term_weighted": buffer_success_term_weighted,
            "target_mu": target_mu,
        }
        test_metrics.update(curriculum_diag)
        #jax.debug.breakpoint() #np learnability scores healthy no nan
        test_metrics["singleton-test-metrics"] = eval_singleton_runner.run(eval_singleton_rng, runner_state[0].params)
        test_metrics["sampled-test-metrics"] = eval_sampled_runner.run(eval_sampled_rng, runner_state[0].params)
        #jax.debug.breakpoint()

        runner_state, _ = runner_state_instances
        test_metrics["update_count"] = runner_state[-2]
        #jax.debug.breakpoint()

        top_instances = jax.tree.map(lambda x: x.at[-20:].get(), instances)
        _, top_states = jax.vmap(env.set_env_instance)(top_instances)
        _, worst_states = jax.vmap(env.set_env_instance)(worst_instances)
        _, unsolvable_states = jax.vmap(env.set_env_instance)(unsolvable_instances)
        
        # Highest scores in the selected instances
        highest_in_top_idx = jnp.argsort(learnabilty_scores)[-20:]
        highest_in_top_scores = learnabilty_scores.at[highest_in_top_idx].get()
        highest_in_top_cnn = cnn_top_scores.at[highest_in_top_idx].get()
        highest_in_top_instances = jax.tree.map(lambda x: x.at[highest_in_top_idx].get(), instances)
        _, highest_in_top_states = jax.vmap(env.set_env_instance)(highest_in_top_instances)

        # Lowest scores in the selected instances
        lowest_in_top_idx = jnp.argsort(learnabilty_scores)[:20]
        lowest_in_top_scores = learnabilty_scores.at[lowest_in_top_idx].get()
        lowest_in_top_cnn = cnn_top_scores.at[lowest_in_top_idx].get()
        lowest_in_top_instances = jax.tree.map(lambda x: x.at[lowest_in_top_idx].get(), instances)
        _, lowest_in_top_states = jax.vmap(env.set_env_instance)(lowest_in_top_instances)


        sample_idx = jax.random.choice(buffer_sample_rng, learnabilty_scores.shape[0], shape=(20,), replace=False)
        sample_scores = learnabilty_scores.at[sample_idx].get()
        sample_cnn = cnn_top_scores.at[sample_idx].get()
        sample_instances = jax.tree.map(lambda x: x.at[sample_idx].get(), instances)
        _, sample_states = jax.vmap(env.set_env_instance)(sample_instances)

        print("train eval steps returns line reached")
        return runner_state, (learnabilty_scores.at[-20:].get(), cnn_top_scores.at[-20:].get(), top_states), (worst_scores, cnn_worst_scores, worst_states), (unsolvable_scores, unsolvable_states), (highest_in_top_scores, highest_in_top_cnn, highest_in_top_states), (lowest_in_top_scores, lowest_in_top_cnn, lowest_in_top_states), (sample_scores, sample_cnn, sample_states), test_metrics
    
    rng, _rng = jax.random.split(rng)
    runner_state = (
        train_state,
        env_state,
        start_state,
        obsv,
        jnp.zeros((t_config["NUM_ACTORS"]), dtype=bool),
        init_hstate,
        0,
        _rng,
    )
    checkpoint_steps = t_config["NUM_UPDATES"] // t_config["EVAL_FREQ"] // t_config["NUM_CHECKPOINTS"]
    print('eval freq', t_config["EVAL_FREQ"])

    target_mu = jnp.array(config.get("CURRICULUM_START_MU", 0.05))

    for eval_step in range(int(t_config["NUM_UPDATES"] // t_config["EVAL_FREQ"])):
        start_time = time.time()
        rng, eval_rng = jax.random.split(rng)
        
        curriculum_strategy = CURRICULUM_STRATEGY_CFG
        if curriculum_strategy == "time_based":
            current_update = runner_state[-2]
            target_mu = jnp.clip(current_update / t_config["NUM_UPDATES"], 0.0, 1.0)
            
        runner_state, top_instances_data, worst_instances_data, unsolvable_instances_data, highest_in_top_data, lowest_in_top_data, buffer_sample_data, metrics = train_and_eval_step(runner_state, eval_rng, learn_method, cnn_state, target_mu)
        #runner_state, instances, metrics = train_and_eval_step(runner_state, eval_rng) # TRAINING AND EVAL HAPPENS IN ONE STEP
        
        if "frontier/mu_used" in metrics:
            metrics["target_mu"] = metrics["frontier/mu_used"]
        if curriculum_strategy in ["performance_adaptive", "gaussian_frontier", "calibrated_frontier"]:

           
            s_drive = metrics["recent_success_rate"] if MU_MEASUREMENT == "legacy" else metrics["buffer_success_rate"]
            mu_base = metrics["target_mu"]
            step_size = config.get("CURRICULUM_STEP_SIZE", 0.05)
            target_mu = jnp.where(s_drive > 0.8, jnp.clip(mu_base + step_size, 0.0, 1.0), mu_base)
            target_mu = jnp.where(s_drive < 0.2, jnp.clip(mu_base - step_size, 0.0, 1.0), target_mu)
            
        curr_time = time.time()
        print('reached 716')
        #jax.debug.breakpoint()

        update_count = int(metrics["update_count"])
        top_scores20, top_cnn20, top_states20 = top_instances_data
        log_buffer(top_scores20, top_states20, update_count, log_key="best_maps",
                   cnn_scores=top_cnn20 if needs_cnn else None) # HERE THE LOGGING here no problem

        hi_s, hi_cnn, hi_st = highest_in_top_data
        log_buffer(hi_s, hi_st, update_count, log_key="highest_in_curriculum", cnn_scores=hi_cnn if needs_cnn else None)
        lo_s, lo_cnn, lo_st = lowest_in_top_data
        log_buffer(lo_s, lo_st, update_count, log_key="lowest_in_curriculum", cnn_scores=lo_cnn if needs_cnn else None)
        samp_s, samp_cnn, samp_st = buffer_sample_data
        log_buffer(samp_s, samp_st, update_count, log_key="buffer_sample", cnn_scores=samp_cnn if needs_cnn else None)
        if "frontier/mu_star" in metrics:

            _bc = np.array([float(metrics[f"frontier/c_bin{_i:02d}"]) for _i in range(FRONTIER_NBINS)])
            _bp = np.array([float(metrics[f"frontier/p_bin{_i:02d}"]) for _i in range(FRONTIER_NBINS)])
            _bi = np.array([float(metrics[f"frontier/iso_bin{_i:02d}"]) for _i in range(FRONTIER_NBINS)])
            _mu_used = float(metrics["frontier/mu_used"]); _mu_star = float(metrics["frontier/mu_star"])
            _fig, _ax = plt.subplots(figsize=(5.5, 3.4))
            _ax.plot(_bc, _bp, "o", ms=4, color="#86b6ef", label="bin mean p (level-weighted)")
            _ax.plot(_bc, _bi, "-", lw=2, color="#1c5cab", label="isotonic fit")
            _ax.axhline(0.5, color="#52514e", lw=1)
            _ax.axvline(_mu_used, color="#e34948", lw=1.5, label=f"mu used = {_mu_used:.2f}")
            _ax.axvline(_mu_star, color="#1baf7a", lw=1, ls=":", label=f"mu* = {_mu_star:.2f} (informative bins: {int(metrics['frontier/n_informative_bins'])})")
            _ax.set_xlim(0, 1); _ax.set_ylim(0, 1.02); _ax.set_xlabel("CNN difficulty percentile"); _ax.set_ylabel("success rate")
            _ax.set_title(f"calibrated frontier, update {update_count}", fontsize=9); _ax.legend(fontsize=7, frameon=False)
            plt.tight_layout(); _fig.canvas.draw()
            _im = Image.fromarray(np.array(_fig.canvas.buffer_rgba())).convert("RGB")
            safe_wandb_log({"frontier_curve": wandb.Image(_im), "update_count": update_count})
            plt.close(_fig)
        if learn_method in ["standard", "hybrid", "progress", "progress_mean", "progress_mindist"]:
            log_buffer(*unsolvable_instances_data, update_count, log_key="unsolvable_maps")
        metrics['time_delta'] = curr_time - start_time #ok
        metrics["steps_per_section"] = (t_config["EVAL_FREQ"] * t_config["NUM_STEPS"] * t_config["NUM_ENVS"]) / metrics['time_delta'] #ok

        for _hk in [k for k in metrics if isinstance(k, str) and k.startswith("hist/")]:
            _h = safe_histogram(metrics[_hk])
            if _h is None:
                del metrics[_hk]
            else:
                metrics[_hk] = _h
        metrics["update_count"] = update_count
        safe_wandb_log(metrics)
        print('reached 721')
        if (eval_step % checkpoint_steps == 0) & (eval_step > 0):    
            if config["SAVE_PATH"] is not None:
                params = runner_state[0].params
                
                save_dir = os.path.join(config["SAVE_PATH"], run.name)
                os.makedirs(save_dir, exist_ok=True)
                save_params(params, f'{save_dir}/model.safetensors')
                print(f'Parameters of saved in {save_dir}/model.safetensors')
                
                # upload this to wandb as an artifact   
                artifact = wandb.Artifact(f'{run.name}-checkpoint', type='checkpoint')
                artifact.add_file(f'{save_dir}/model.safetensors')
                artifact.save()

    #print('reached 736')
    #SAVE MODEL -----------------------------------
    if config["SAVE_PATH"] is not None:
        params = runner_state[0].params
        
        save_dir = os.path.join(config["SAVE_PATH"], run.name)
        os.makedirs(save_dir, exist_ok=True)
        save_params(params, f'{save_dir}/model.safetensors')
        print(f'Parameters of saved in {save_dir}/model.safetensors')
        
        # upload this to wandb as an artifact   
        artifact = wandb.Artifact(f'{run.name}-checkpoint', type='checkpoint')
        artifact.add_file(f'{save_dir}/model.safetensors')
        artifact.save()
    

if __name__ == "__main__":
    with jax.disable_jit(False):
        main()
