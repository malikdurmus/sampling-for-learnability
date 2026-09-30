
import os
import sys
import jax
import jax.experimental
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
from typing import NamedTuple
from flax.training.train_state import TrainState
import hydra
from omegaconf import OmegaConf
from functools import partial
import time
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb
from flax import nnx
import orbax.checkpoint as ocp

# XLand-MiniGrid imports
import xminigrid
from xminigrid.wrappers import GymAutoResetWrapper
from xminigrid.types import RuleSet
from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles
from xminigrid.core.goals import AgentHoldGoal
from xminigrid.core.rules import TileNearRule, AgentHoldRule, AgentNearRule

from sfl.train.train_utils import save_params

# Add xland/training to path for nn.py and utils.py imports
_xland_training_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'xland', 'training')
sys.path.insert(0, os.path.abspath(_xland_training_dir))
from nn import ActorCriticRNN as XLandActorCriticRNN
from utils import Transition as XLandTransition, calculate_gae, rollout, rollout_nsteps, RolloutStats

# CNN model loader
from rlhf_utils import (load_learnability_ensemble, make_ensemble_logit_fn,
                        make_member_logit_fn, pairwise_rank_agreement,
                        standardized_member_spread, resolve_input_domain)

# this will be default in new jax versions anyway
jax.config.update("jax_threefry_partitionable", True)


# ──────────────────────────────────────────────────────────────
# Rulesets (configurable via env.ruleset)
def make_ruleset(name: str):
    """Build the fixed training ruleset ("medium" or "difficult")."""
    yellow_ball    = TILES_REGISTRY[Tiles.BALL,    Colors.YELLOW]
    blue_key       = TILES_REGISTRY[Tiles.KEY,     Colors.BLUE]
    green_star     = TILES_REGISTRY[Tiles.STAR,    Colors.GREEN]
    purple_hex     = TILES_REGISTRY[Tiles.HEX,     Colors.PURPLE]
    pink_square    = TILES_REGISTRY[Tiles.SQUARE,  Colors.PINK]

    # Distractor objects (on the grid but not part of the rule chain)
    brown_pyramid  = TILES_REGISTRY[Tiles.PYRAMID, Colors.BROWN]
    orange_square  = TILES_REGISTRY[Tiles.SQUARE,  Colors.ORANGE]
    grey_hex       = TILES_REGISTRY[Tiles.HEX,     Colors.GREY]
    white_pyramid  = TILES_REGISTRY[Tiles.PYRAMID, Colors.WHITE]

    if name == "medium":
        med_goal = AgentHoldGoal(tile=purple_hex)
        med_rule1 = TileNearRule(tile_a=yellow_ball, tile_b=blue_key, prod_tile=green_star)
        med_rule2 = AgentHoldRule(tile=green_star, prod_tile=purple_hex)
        return RuleSet(
            goal=med_goal.encode(),
            rules=jnp.vstack([med_rule1.encode(), med_rule2.encode()]),
            init_tiles=jnp.array([
                yellow_ball,
                blue_key,
                brown_pyramid,
                orange_square,
            ]),
        )
    elif name == "difficult":
        diff_goal = AgentHoldGoal(tile=pink_square)
        # 1. Agent gets near white pyramid -> spawns blue key
        diff_rule1 = AgentNearRule(tile=white_pyramid, prod_tile=blue_key)
        # 2. Place blue key near yellow ball -> spawns green star
        diff_rule2 = TileNearRule(tile_a=blue_key, tile_b=yellow_ball, prod_tile=green_star)
        # 3. Place green star near brown pyramid -> spawns purple hex
        diff_rule3 = TileNearRule(tile_a=green_star, tile_b=brown_pyramid, prod_tile=purple_hex)
        # 4. Agent picks up purple hex -> transforms to pink square
        diff_rule4 = AgentHoldRule(tile=purple_hex, prod_tile=pink_square)
        return RuleSet(
            goal=diff_goal.encode(),
            rules=jnp.vstack([
                diff_rule1.encode(),
                diff_rule2.encode(),
                diff_rule3.encode(),
                diff_rule4.encode(),
            ]),
            init_tiles=jnp.array([
                white_pyramid,
                yellow_ball,
                brown_pyramid,
                orange_square,
                grey_hex,
            ]),
        )
    raise ValueError(f"Unknown ruleset '{name}' (use medium | difficult)")


# ──────────────────────────────────────────────────────────────
# XLand Tile-Cache Renderer for CNN scoring
# ──────────────────────────────────────────────────────────────
def build_xland_render_fn(tile_size=32, view_size=7, target_size=200):
    """Build a JAX-jittable render function that renders XLand grid state
    to a normalized float32 image of shape (target_size, target_size, 3).
    
    Returns a function: render_fn(grid, agent) -> image [0,1] float32
    """
    from xminigrid.rendering.rgb_render import render_tile
    from xminigrid.core.constants import NUM_COLORS, NUM_TILES
    from xminigrid.types import AgentState
    
    # Build tile caches (runs once, uses numpy internally)
    n = NUM_TILES * NUM_COLORS
    tile_cache = np.zeros((n, tile_size, tile_size, 3), dtype=np.uint8)
    agent_tile_cache = np.zeros((n, 4, tile_size, tile_size, 3), dtype=np.uint8)

    for tile_id in range(NUM_TILES):
        for color_id in range(NUM_COLORS):
            flat_idx = tile_id * NUM_COLORS + color_id
            tile_cache[flat_idx] = render_tile(
                tile=(tile_id, color_id), tile_size=tile_size
            )
            for direction in range(4):
                agent_tile_cache[flat_idx, direction] = render_tile(
                    tile=(tile_id, color_id), tile_size=tile_size,
                    agent_direction=direction
                )

    tile_cache_jax = jnp.array(tile_cache)
    agent_tile_cache_jax = jnp.array(agent_tile_cache)
    
    def _highlight_mask(grid_h, grid_w, agent_position, agent_direction):
        """Agent field-of-view mask, ported VERBATIM from
        uedrlhf/jax_renderer.py::_highlight_mask_jax. The scorer's training
        images carry this FOV highlight, so deployment must render it too —
        without it, rendered levels are out-of-distribution for the scorer
        (measured 2026-08-29: ensemble-mean logits -25..-20 vs the dataset's
        -9..+2, and member rank agreement 0.067 vs 0.97)."""
        padded_h = grid_h + 2 * view_size
        padded_w = grid_w + 2 * view_size
        agent_y = agent_position[0] + view_size
        agent_x = agent_position[1] + view_size
        half = view_size // 2

        def dir0(_):
            return agent_y - view_size + 1, agent_x - half
        def dir1(_):
            return agent_y - half, agent_x
        def dir2(_):
            return agent_y, agent_x - half
        def dir3(_):
            return agent_y - half, agent_x - view_size + 1

        vy, vx = jax.lax.switch(agent_direction, [dir0, dir1, dir2, dir3], None)
        ys = jnp.arange(padded_h)
        xs = jnp.arange(padded_w)
        yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
        mask = (yy >= vy) & (yy < vy + view_size) & (xx >= vx) & (xx < vx + view_size)
        return mask[view_size : view_size + grid_h, view_size : view_size + grid_w]

    def _render_single(grid, agent):
     
        grid_h, grid_w = grid.shape[0], grid.shape[1]

        # 1. Compute flat tile indices
        flat_idxs = grid[:, :, 0].astype(jnp.int32) * NUM_COLORS + grid[:, :, 1].astype(jnp.int32)

        # 2. Look up tiles from cache -> (H, W, tile_size, tile_size, 3)
        rendered = jnp.take(tile_cache_jax, flat_idxs, axis=0)

        # 3. Overlay agent tile
        agent_y = agent.position[0]
        agent_x = agent.position[1]
        agent_flat_idx = flat_idxs[agent_y, agent_x]
        agent_tile = agent_tile_cache_jax[agent_flat_idx, agent.direction.astype(jnp.int32)]
        rendered = rendered.at[agent_y, agent_x].set(agent_tile)

        # 4. FOV highlight (alpha-blend toward white inside the view cone),
        #    exactly as in the dataset renderer (jax_render, alpha=0.2)
        mask = _highlight_mask(grid_h, grid_w, agent.position,
                               agent.direction.astype(jnp.int32))
        mask_expanded = mask[:, :, None, None, None]
        alpha = 0.2
        blended = rendered.astype(jnp.float32) + alpha * (255.0 - rendered.astype(jnp.float32))
        blended = jnp.clip(blended, 0, 255).astype(jnp.uint8)
        rendered = jnp.where(mask_expanded, blended, rendered)

        # 5. Reshape to full image: (H*ts, W*ts, 3)
        img = rendered.transpose((0, 2, 1, 3, 4))  # (H, ts, W, ts, 3)
        img = img.reshape(grid_h * tile_size, grid_w * tile_size, 3)

        # 6. Convert to float32 [0, 1]
        img = img.astype(jnp.float32) / 255.0

        # 7. Resize to target_size; cubic to approximate the dataset's PIL
        #    bicubic 416->200 downscale (was bilinear before 2026-08-29)
        img = jax.image.resize(img, (target_size, target_size, 3), method='cubic')

        return jnp.clip(img, 0.0, 1.0)

    return _render_single


@hydra.main(version_base=None, config_path="config", config_name="xland-sfl")
def main(config):

    # WANDB AND CONFIG
    config = OmegaConf.to_container(config)
    run = wandb.init(
        name=config.get("RUN_NAME", "xland_sfl"),
        group=config["GROUP_NAME"],
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["PPO", "RNN", "SFL", "XLand"],
        config=config,
        mode=config["WANDB_MODE"],
    )

    # Key every metric to update_count instead of wandb's implicit step counter,
    # so per-update callback logs and per-eval-cycle logs can never misalign.
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

    # ---- Determine learn method ----
    learn_method_raw = config.get("LEARN_METHOD", "standard")
    VALID_LEARN_METHODS = (
        "standard", "random", "cnn",
        "hybrid_linear", "hybrid_soft_handoff",
        "hybrid_learnability_weighted", "hybrid_multiplicative",
    )
    # A misspelled hybrid mode must fail loudly, not silently fall back to pure SFL
    assert learn_method_raw in VALID_LEARN_METHODS, \
        f"Invalid LEARN_METHOD '{learn_method_raw}' (valid: {VALID_LEARN_METHODS})"
    if learn_method_raw.startswith("hybrid_"):
        learn_method = "hybrid"
        hybrid_mode = learn_method_raw.replace("hybrid_", "")
        config["HYBRID_MODE"] = hybrid_mode
    else:
        learn_method = learn_method_raw
        config["HYBRID_MODE"] = "linear"
    needs_cnn = learn_method in ["cnn", "hybrid"]
    print(f"--- USING LEARNABILITY METHOD: {learn_method} (raw: {learn_method_raw}) ---")

    # ---- CNN score normalization method (the ablation variable) ----
    CNN_SCORE_METHOD = config.get("CNN_SCORE_METHOD", "sigmoid")
    assert CNN_SCORE_METHOD in ("sigmoid", "minmax", "percentile"), \
        f"Invalid CNN_SCORE_METHOD '{CNN_SCORE_METHOD}' (use sigmoid | minmax | percentile)"
    print(f"--- CNN SCORE METHOD: {CNN_SCORE_METHOD} ---")

    if needs_cnn:
      
        cnn_checkpoint_paths = config.get("CNN_CHECKPOINT_PATHS")
        assert cnn_checkpoint_paths, \
            "LEARN_METHOD needs the CNN but CNN_CHECKPOINT_PATHS is not set in the config " \
            "(the xland ensemble checkpoints may still need to be pulled from the pods — " \
            "see uedrlhf outputs/checkpoints/ensemble/README.MD)"
        print(f"Loading XLand CNN learnability scorer ({len(cnn_checkpoint_paths)} member(s))...")

        
        cnn_input_domain = resolve_input_domain(cnn_checkpoint_paths)
        _invert = cnn_input_domain == "inverted"
        print(f"--- CNN INPUT DOMAIN: {cnn_input_domain} (invert_input={_invert}) ---")
        cnn_logit_fn = make_ensemble_logit_fn(cnn_graphdef, invert_input=_invert)
        cnn_member_logit_fn = make_member_logit_fn(cnn_graphdef, invert_input=_invert)

        if wandb.run is not None:
            wandb.config.update({
                "CNN_CHECKPOINT_PATHS": list(cnn_checkpoint_paths),
                "CNN_ENSEMBLE_K": cnn_num_members,
                "CNN_ENSEMBLE_COMBINE": "mean_of_member_logits",
                "CNN_INPUT_DOMAIN": cnn_input_domain,
                "CNN_INVERT_INPUT": _invert,
                "CNN_SCORE_CONVENTION": "higher=more_difficult",
                "CNN_SCORE_METHOD": CNN_SCORE_METHOD,
                "CNN_IMG_SIZE": 200
            }, allow_val_change=True)
    else:
        cnn_graphdef, cnn_state, cnn_logit_fn = None, None, None
        cnn_member_logit_fn, cnn_num_members = None, 0

    rng = jax.random.PRNGKey(config["SEED"])

    # ---- Setup Environment ----
    env_config = config["env"]
    t_config = config["learning"]
    
    assert (t_config["NUM_ENVS_FROM_SAMPLED"] + t_config["NUM_ENVS_TO_GENERATE"]) == t_config["NUM_ENVS"]
    
    env, env_params = xminigrid.make(env_config["env_id"])
    env = GymAutoResetWrapper(env)
    
    ruleset_name = env_config.get("ruleset", "difficult")
    env_params = env_params.replace(ruleset=make_ruleset(ruleset_name))
    print(f'--- RULESET: {ruleset_name} ---')

    print(f'Environment: {env_config["env_id"]}')
    print(f'Max steps per episode: {env_params.max_steps}')
    
    num_envs = t_config["NUM_ENVS"]
    num_steps_per_update = t_config["NUM_STEPS"]
    num_steps_per_env = t_config.get("NUM_STEPS_PER_ENV", num_steps_per_update * 128)
    
    t_config["NUM_ACTORS"] = num_envs  # single agent per env
    t_config["NUM_UPDATES"] = int(
        t_config["TOTAL_TIMESTEPS"] // num_steps_per_env // num_envs
    )
    t_config["NUM_INNER_UPDATES"] = num_steps_per_env // num_steps_per_update
    t_config["MINIBATCH_SIZE"] = (
        t_config["NUM_ACTORS"] * num_steps_per_update // t_config["NUM_MINIBATCHES"]
    )
    
    print(f"NUM_UPDATES (meta): {t_config['NUM_UPDATES']}")
    print(f"NUM_INNER_UPDATES: {t_config['NUM_INNER_UPDATES']}")
    print(f"MINIBATCH_SIZE: {t_config['MINIBATCH_SIZE']}")
    
    # ---- Network ----
    network = XLandActorCriticRNN(
        num_actions=env.num_actions(env_params),
        action_emb_dim=env_config["action_emb_dim"],
        rnn_hidden_dim=env_config["rnn_hidden_dim"],
        rnn_num_layers=env_config["rnn_num_layers"],
        head_hidden_dim=env_config["head_hidden_dim"],
        img_obs=env_config.get("img_obs", False),
    )
    
    # ---- LR schedule ----
    def linear_schedule(count):
        total_inner_updates = t_config["NUM_MINIBATCHES"] * t_config["UPDATE_EPOCHS"] * t_config["NUM_INNER_UPDATES"]
        frac = 1.0 - (count // total_inner_updates) / t_config["NUM_UPDATES"]
        return t_config["LR"] * frac
    
    rng, _rng = jax.random.split(rng)
    init_obs = {
        "observation": jnp.zeros((num_envs, 1, *env.observation_shape(env_params))),
        "prev_action": jnp.zeros((num_envs, 1), dtype=jnp.int32),
        "prev_reward": jnp.zeros((num_envs, 1)),
    }
    init_hstate = network.initialize_carry(batch_size=num_envs)
    network_params = network.init(_rng, init_obs, init_hstate)
    
    if t_config["ANNEAL_LR"]:
        tx = optax.chain(
            optax.clip_by_global_norm(t_config["MAX_GRAD_NORM"]),
            optax.inject_hyperparams(optax.adam)(learning_rate=linear_schedule, eps=1e-8),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(t_config["MAX_GRAD_NORM"]),
            optax.adam(t_config["LR"], eps=1e-8),
        )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

    # ---- Build XLand renderer for CNN ----
    CNN_IMG_SIZE = 200
    xland_render_fn = build_xland_render_fn(
        tile_size=env_config.get("tile_size", 32),
        view_size=env_config.get("view_size", 7),
        target_size=CNN_IMG_SIZE,
    )

    PERCENTILE_REF_SIZE = int(config.get("PERCENTILE_REF_SIZE", 10000))
    PERCENTILE_REF_SEED = int(config.get("PERCENTILE_REF_SEED", 4242))

    if needs_cnn:
        if CNN_SCORE_METHOD == "percentile":
           
            assert PERCENTILE_REF_SIZE % 1000 == 0, "PERCENTILE_REF_SIZE must be a multiple of 1000"

            @jax.jit
            def _ref_batch_logits(ref_rng):
                ref_reset_rng = jax.random.split(ref_rng, 1000)
                ref_timesteps = jax.vmap(env.reset, in_axes=(None, 0))(env_params, ref_reset_rng)
                grids_chunked = ref_timesteps.state.grid.reshape((10, 100) + ref_timesteps.state.grid.shape[1:])
                agents_chunked = jax.tree.map(lambda x: x.reshape((10, 100) + x.shape[1:]), ref_timesteps.state.agent)
                def _score_chunk(carry, chunk_data):
                    grid_chunk, agent_chunk = chunk_data
                    imgs = jax.vmap(xland_render_fn)(grid_chunk, agent_chunk)
                    return carry, cnn_logit_fn(cnn_state, imgs)[0]
                _, lg = jax.lax.scan(_score_chunk, None, (grids_chunked, agents_chunked))
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
        _probe_ts = jax.vmap(env.reset, in_axes=(None, 0))(env_params, _probe_rngs)
        _probe_imgs = jax.vmap(xland_render_fn)(_probe_ts.state.grid, _probe_ts.state.agent)
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
    else:
        normalize_scores = None

    # ---- Init Environment ----
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, num_envs)
    timestep = jax.vmap(env.reset, in_axes=(None, 0))(env_params, reset_rng)
    
    init_hstate_train = network.initialize_carry(batch_size=t_config["NUM_ACTORS"])
    prev_action = jnp.zeros(num_envs, dtype=jnp.int32)
    prev_reward = jnp.zeros(num_envs)
    
 
    # CNN-based learnability scoring
    # ──────────────────────────────────────────────────────────────
    @jax.jit
    def select_environments(rng, cnn_scores_normalized, target_mu):
        strategy = config.get("CURRICULUM_STRATEGY", "time_based")


        if strategy == "time_based" or strategy == "performance_adaptive":
            distances = jnp.abs(cnn_scores_normalized - target_mu)
            top_indices = jnp.argsort(distances)[:config["NUM_TO_SAVE"]]
            top_indices = top_indices[::-1]  # closest to target_mu last
        elif strategy == "gaussian_frontier":
            var = config.get("CURRICULUM_GAUSSIAN_VAR", 0.1)
            weights = jnp.exp(-((cnn_scores_normalized - target_mu)**2) / (2 * var**2))
            probs = weights / jnp.sum(weights)
            top_indices = jax.random.choice(rng, cnn_scores_normalized.shape[0], shape=(config["NUM_TO_SAVE"],), p=probs, replace=False)
            sel_distances = jnp.abs(cnn_scores_normalized.at[top_indices].get() - target_mu)
            top_indices = top_indices.at[jnp.argsort(-sel_distances)].get()  # closest to target mu last
        else:
            top_indices = jnp.argsort(cnn_scores_normalized)[-config["NUM_TO_SAVE"]:]  # highest score last

        return top_indices

    @partial(jax.jit, static_argnums=(1,))
    def get_learnability_set_cnn(rng, cnn_graphdef, cnn_state, target_mu):
        def _batch_step(unused, batch_rng):
            rng, _rng = jax.random.split(batch_rng)
            reset_rng = jax.random.split(_rng, config["BATCH_SIZE"])
            batch_timesteps = jax.vmap(env.reset, in_axes=(None, 0))(env_params, reset_rng)
            
            # Store the reset rngs as "level instances"
            env_instances = reset_rng  # shape: (BATCH_SIZE, 2)
            
            # Render and score with CNN
            CHUNK_SIZE = min(config["BATCH_SIZE"], 100)
            NUM_CHUNKS = config["BATCH_SIZE"] // CHUNK_SIZE
            
            grids_chunked = batch_timesteps.state.grid.reshape((NUM_CHUNKS, CHUNK_SIZE) + batch_timesteps.state.grid.shape[1:])
            agents_chunked = jax.tree.map(
                lambda x: x.reshape((NUM_CHUNKS, CHUNK_SIZE) + x.shape[1:]),
                batch_timesteps.state.agent
            )
            
            def render_and_score_chunk(carry, chunk_data):
                grid_chunk, agent_chunk = chunk_data
                images_chunk = jax.vmap(xland_render_fn)(grid_chunk, agent_chunk)
                # Raw per-member logits (K, CHUNK); the ensemble score is their
                # mean, and disagreement diagnostics need the full member axis
                member_logits_chunk = cnn_member_logit_fn(cnn_state, images_chunk)
                return carry, member_logits_chunk

            _, member_chunked = jax.lax.scan(
                render_and_score_chunk, None, (grids_chunked, agents_chunked)
            )
            # (NUM_CHUNKS, K, CHUNK) -> (K, BATCH_SIZE)
            member_by_env = jnp.moveaxis(member_chunked, 1, 0).reshape((cnn_num_members, config["BATCH_SIZE"]))

            return None, (member_by_env, env_instances)

        rngs = jax.random.split(rng, config["NUM_BATCHES"])
        _, (member_logits, env_instances) = jax.lax.scan(_batch_step, None, rngs, config["NUM_BATCHES"])

        flat_env_instances = env_instances.reshape((-1,) + env_instances.shape[2:])
        # (NUM_BATCHES, K, BATCH_SIZE) -> (K, TOTAL); raw logits, pre-normalization
        member_logits = jnp.moveaxis(member_logits, 1, 0).reshape((cnn_num_members, -1))
        member_spread = member_logits.std(axis=0)     # per-level disagreement (raw logits)
        member_spread_z = standardized_member_spread(member_logits)
        learnability = member_logits.mean(axis=0)     # ensemble score

        # Normalize raw logits according to CNN_SCORE_METHOD (higher = harder)
        learnability_norm = normalize_scores(learnability)

        rng, select_rng = jax.random.split(rng)
        top_indices = select_environments(select_rng, learnability_norm, target_mu)
        top_instances = flat_env_instances[top_indices]

        bottom_indices = jnp.argsort(learnability_norm)[:20]
        bottom_instances = flat_env_instances[bottom_indices]

        # CNN method: the selection score IS the CNN difficulty score
        top_scores = learnability_norm[top_indices]
        bottom_scores = learnability_norm[bottom_indices]

        diag = {
            "cnn/all_mean": learnability_norm.mean(),
            "cnn/selected_mean": top_scores.mean(),
            "cnn/selected_std": top_scores.std(),
            "cnn/selected_min": top_scores.min(),
            "cnn/selected_max": top_scores.max(),
            "cnn/tracking_error": jnp.abs(top_scores - target_mu).mean(),
            "hist/cnn_all": learnability_norm,
            "hist/cnn_selected": top_scores,
        }
        if cnn_num_members > 1:
            diag.update({
                "cnn/member_spread_all": member_spread.mean(),
                "cnn/member_spread_selected": member_spread[top_indices].mean(),
                "cnn/member_spread_z_all": member_spread_z.mean(),
                "cnn/member_spread_z_selected": member_spread_z[top_indices].mean(),
                "cnn/member_rank_agreement": pairwise_rank_agreement(member_logits),
            })
        return (top_scores, top_instances,
                bottom_scores, bottom_instances,
                jnp.zeros(20), bottom_instances,
                top_scores, bottom_scores, diag)

    @jax.jit
    def get_learnability_set_random(rng):
        def _batch_step(unused, batch_rng):
            rng, _rng = jax.random.split(batch_rng)
            reset_rng = jax.random.split(_rng, config["BATCH_SIZE"])
            _ = jax.vmap(env.reset, in_axes=(None, 0))(env_params, reset_rng)
            
            env_instances = reset_rng
            
            rng, rand_rng = jax.random.split(rng)
            learnability_by_env = jax.random.uniform(rand_rng, (config["BATCH_SIZE"],))
            
            return None, (learnability_by_env, env_instances)
        
        rngs = jax.random.split(rng, config["NUM_BATCHES"])
        _, (learnability, env_instances) = jax.lax.scan(_batch_step, None, rngs, config["NUM_BATCHES"])
        
        flat_env_instances = env_instances.reshape((-1,) + env_instances.shape[2:])
        learnability = learnability.flatten()
        top_indices = jnp.argsort(learnability)[-config["NUM_TO_SAVE"]:]
        top_instances = flat_env_instances[top_indices]

        bottom_indices = jnp.argsort(learnability)[:20]
        bottom_instances = flat_env_instances[bottom_indices]

        return (learnability[top_indices], top_instances,
                learnability[bottom_indices], bottom_instances,
                jnp.zeros(20), bottom_instances,
                jnp.zeros(config["NUM_TO_SAVE"]), jnp.zeros(20), {})

    @jax.jit
    def get_learnability_set_standard(rng, network_params):
        """Standard SFL: roll out the current policy, measure success rate,
        compute learnability = p(1-p)."""
        
        BATCH_ACTORS = config["BATCH_SIZE"]
        rnn_hidden_dim = env_config["rnn_hidden_dim"]
        rnn_num_layers = env_config["rnn_num_layers"]
        
        def _batch_step(unused, batch_rng):
            # Generate random levels
            rng, _rng = jax.random.split(batch_rng)
            reset_rng = jax.random.split(_rng, config["BATCH_SIZE"])
            batch_timesteps = jax.vmap(env.reset, in_axes=(None, 0))(env_params, reset_rng)
            env_instances = reset_rng
            
            # Rollout the current policy on each level
            rollout_rng = jax.random.split(rng, config["BATCH_SIZE"])
            rollout_stats = jax.vmap(rollout_nsteps, in_axes=(0, None, None, None, None, None))(
                rollout_rng,
                env,
                env_params,
                TrainState.create(apply_fn=network.apply, params=network_params, tx=optax.identity()),
                jnp.zeros((1, rnn_num_layers, rnn_hidden_dim)),
                config["ROLLOUT_STEPS"],
            )
            
            # Compute learnability
            success_rate = rollout_stats.success / jnp.maximum(rollout_stats.episodes, 1)
            learnability_by_env = success_rate * (1 - success_rate)
            solvability_by_env = success_rate
            
            return None, (learnability_by_env, solvability_by_env, env_instances)
        
        rngs = jax.random.split(rng, config["NUM_BATCHES"])
        _, (learnability, solvability, env_instances) = jax.lax.scan(_batch_step, None, rngs, config["NUM_BATCHES"])
        
        flat_env_instances = env_instances.reshape((-1,) + env_instances.shape[2:])
        learnability = learnability.flatten()
        
        top_indices = jnp.argsort(learnability)[-config["NUM_TO_SAVE"]:]
        top_instances = flat_env_instances[top_indices]
        
        bottom_indices = jnp.argsort(learnability)[:20]
        bottom_instances = flat_env_instances[bottom_indices]
        
        solvability = solvability.flatten()
        unsolvable_indices = jnp.argsort(solvability)[:20]
        unsolvable_instances = flat_env_instances[unsolvable_indices]

        diag = {
            "sfl/batch_mean": learnability.mean(),
            "sfl/batch_max": learnability.max(),
            "solvability/batch_mean": solvability.mean(),
            "hist/sfl_all": learnability,
        }
        return (learnability[top_indices], top_instances,
                learnability[bottom_indices], bottom_instances,
                solvability[unsolvable_indices], unsolvable_instances,
                jnp.zeros(config["NUM_TO_SAVE"]), jnp.zeros(20), diag)

    @partial(jax.jit, static_argnums=(2,))
    def get_learnability_set_hybrid(rng, network_params, cnn_graphdef, cnn_state, target_mu):
        """Hybrid: runs agent rollouts (SFL scores + solvability) AND CNN scoring."""

        BATCH_ACTORS = config["BATCH_SIZE"]
        rnn_hidden_dim = env_config["rnn_hidden_dim"]
        rnn_num_layers = env_config["rnn_num_layers"]
        
        def _batch_step(unused, batch_rng):
            rng, _rng = jax.random.split(batch_rng)
            reset_rng = jax.random.split(_rng, config["BATCH_SIZE"])
            batch_timesteps = jax.vmap(env.reset, in_axes=(None, 0))(env_params, reset_rng)
            env_instances = reset_rng
            
            # ---- Agent rollout for SFL scores ----
            rollout_rng = jax.random.split(rng, config["BATCH_SIZE"])
            rollout_stats = jax.vmap(rollout_nsteps, in_axes=(0, None, None, None, None, None))(
                rollout_rng,
                env,
                env_params,
                TrainState.create(apply_fn=network.apply, params=network_params, tx=optax.identity()),
                jnp.zeros((1, rnn_num_layers, rnn_hidden_dim)),
                config["ROLLOUT_STEPS"],
            )
            
            success_rate = rollout_stats.success / jnp.maximum(rollout_stats.episodes, 1)
            sfl_scores = success_rate * (1 - success_rate)
            solvability_by_env = success_rate
            
            # ---- CNN scoring ----
            CHUNK_SIZE = min(config["BATCH_SIZE"], 100)
            NUM_CHUNKS = config["BATCH_SIZE"] // CHUNK_SIZE
            
            grids_chunked = batch_timesteps.state.grid.reshape((NUM_CHUNKS, CHUNK_SIZE) + batch_timesteps.state.grid.shape[1:])
            agents_chunked = jax.tree.map(
                lambda x: x.reshape((NUM_CHUNKS, CHUNK_SIZE) + x.shape[1:]),
                batch_timesteps.state.agent
            )
            
            def render_and_score_chunk(carry, chunk_data):
                grid_chunk, agent_chunk = chunk_data
                images_chunk = jax.vmap(xland_render_fn)(grid_chunk, agent_chunk)
                # Raw per-member logits (K, CHUNK); ensemble score = member mean
                member_logits_chunk = cnn_member_logit_fn(cnn_state, images_chunk)
                return carry, member_logits_chunk

            _, cnn_member_chunked = jax.lax.scan(
                render_and_score_chunk, None, (grids_chunked, agents_chunked)
            )
            # (NUM_CHUNKS, K, CHUNK) -> (K, BATCH_SIZE) raw member logits
            cnn_member_by_env = jnp.moveaxis(cnn_member_chunked, 1, 0).reshape((cnn_num_members, config["BATCH_SIZE"]))

            return None, (sfl_scores, cnn_member_by_env, solvability_by_env, env_instances)

        rngs = jax.random.split(rng, config["NUM_BATCHES"])
        _, (sfl_scores, cnn_member_scores, solvability_p, env_instances) = jax.lax.scan(
            _batch_step, None, rngs, config["NUM_BATCHES"]
        )

        flat_env_instances = env_instances.reshape((-1,) + env_instances.shape[2:])
        sfl_scores = sfl_scores.flatten()
        # (NUM_BATCHES, K, BATCH_SIZE) -> (K, TOTAL) raw member logits
        cnn_member_scores = jnp.moveaxis(cnn_member_scores, 1, 0).reshape((cnn_num_members, -1))
        cnn_member_spread = cnn_member_scores.std(axis=0)   # per-level disagreement (raw logits)
        cnn_member_spread_z = standardized_member_spread(cnn_member_scores)  # genuine signal
        cnn_scores = cnn_member_scores.mean(axis=0)         # (TOTAL,) ensemble-mean raw logits
        solvability_p = solvability_p.flatten()

        # Normalize raw logits 
        cnn_norm = normalize_scores(cnn_scores)

        # CNN proximity to target_mu
        cnn_proximity = 1.0 - jnp.abs(cnn_norm - target_mu)
        
        # Compound scoring
        hybrid_mode = config.get("HYBRID_MODE", "linear")
        
        if hybrid_mode == "linear":
            compound_scores = (4.0 * sfl_scores) + cnn_proximity
        elif hybrid_mode == "soft_handoff":
            batch_p = jnp.mean(solvability_p)
            alpha = jnp.clip(batch_p / 0.1, 0.0, 1.0)
            compound_scores = (alpha * (4.0 * sfl_scores)) + ((1.0 - alpha) * cnn_proximity)
        elif hybrid_mode == "learnability_weighted":
            cnn_weight = (0.25 - sfl_scores) * 4.0
            cnn_weight = jnp.where(solvability_p > 0.9, 0.0, cnn_weight)
            compound_scores = (4.0 * sfl_scores) + (cnn_weight * cnn_proximity)
        elif hybrid_mode == "multiplicative":
            cnn_filter = jnp.exp(-((cnn_norm - target_mu)**2) / (2 * 0.1**2))
            compound_scores = (sfl_scores + 0.01) * cnn_filter
        else:
            compound_scores = sfl_scores
        
        top_indices = jnp.argsort(compound_scores)[-config["NUM_TO_SAVE"]:]
        top_instances = flat_env_instances[top_indices]

        bottom_indices = jnp.argsort(compound_scores)[:20]
        bottom_instances = flat_env_instances[bottom_indices]

        unsolvable_indices = jnp.argsort(solvability_p)[:20]
        unsolvable_instances = flat_env_instances[unsolvable_indices]

        # ---- Curriculum diagnostics (logged per eval cycle) ----
        cnn_sel = cnn_norm[top_indices]
        sfl_sel = sfl_scores[top_indices]
        solv_sel = solvability_p[top_indices]
        _sfl_c = sfl_scores - sfl_scores.mean()
        _cnn_c = cnn_norm - cnn_norm.mean()
        corr = (_sfl_c * _cnn_c).mean() / (sfl_scores.std() * cnn_norm.std() + 1e-8)
        diag = {
            "cnn/all_mean": cnn_norm.mean(),
            "cnn/selected_mean": cnn_sel.mean(),
            "cnn/selected_std": cnn_sel.std(),
            "cnn/selected_min": cnn_sel.min(),
            "cnn/selected_max": cnn_sel.max(),
            "cnn/tracking_error": jnp.abs(cnn_sel - target_mu).mean(),
            "sfl/batch_mean": sfl_scores.mean(),
            "sfl/batch_max": sfl_scores.max(),
            "sfl/selected_mean": sfl_sel.mean(),
            "cnn_proximity/batch_mean": cnn_proximity.mean(),
            "solvability/batch_mean": solvability_p.mean(),
            "solvability/selected_mean": solv_sel.mean(),
            "hybrid/alpha_soft_handoff": jnp.clip(solvability_p.mean() / 0.1, 0.0, 1.0),
            "corr/sfl_vs_cnn": corr,
            "hist/cnn_all": cnn_norm,
            "hist/cnn_selected": cnn_sel,
            "hist/sfl_all": sfl_scores,
        }
        if cnn_num_members > 1:
            diag.update({
                "cnn/member_spread_all": cnn_member_spread.mean(),
                "cnn/member_spread_selected": cnn_member_spread[top_indices].mean(),
                "cnn/member_spread_z_all": cnn_member_spread_z.mean(),
                "cnn/member_spread_z_selected": cnn_member_spread_z[top_indices].mean(),
                "cnn/member_rank_agreement": pairwise_rank_agreement(cnn_member_scores),
            })
        return (compound_scores[top_indices], top_instances,
                compound_scores[bottom_indices], bottom_instances,
                solvability_p[unsolvable_indices], unsolvable_instances,
                cnn_norm[top_indices], cnn_norm[bottom_indices], diag)

    # Train step  inner loop  PPO 
    def train_step(runner_state_instances, unused):
        runner_state, instances = runner_state_instances
        num_env_instances = instances.shape[0]  # instances are RNG keys
        
        rng, train_state, timestep, prev_action, prev_reward, hstate, update_steps = runner_state
        
        outcomes = jnp.zeros((num_envs, 2))  # (episodes_count, success_count)
        
        def _update_step(inner_state, _):
            def _env_step(step_state, _):
                rng, train_state, prev_timestep, prev_action, prev_reward, outcomes, prev_hstate = step_state
                
                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                dist, value, hstate = train_state.apply_fn(
                    train_state.params,
                    {
                        "observation": prev_timestep.observation[:, None],
                        "prev_action": prev_action[:, None],
                        "prev_reward": prev_reward[:, None],
                    },
                    prev_hstate,
                )
                action, log_prob = dist.sample_and_log_prob(seed=_rng)
                action, value, log_prob = action.squeeze(1), value.squeeze(1), log_prob.squeeze(1)
                
                # STEP ENV
                next_timestep = jax.vmap(env.step, in_axes=(None, 0, 0))(env_params, prev_timestep, action)
                success = next_timestep.discount == 0.0
                outcomes_new = outcomes.at[:, 0].add(jnp.where(next_timestep.last(), 1, 0))
                outcomes_new = outcomes_new.at[:, 1].add(jnp.where(success, 1, 0))
                
                transition = XLandTransition(
                    done=jnp.zeros_like(next_timestep.last()),  # meta-RL: always 0
                    action=action,
                    value=value,
                    reward=next_timestep.reward,
                    log_prob=log_prob,
                    obs=prev_timestep.observation,
                    prev_action=prev_action,
                    prev_reward=prev_reward,
                )
                step_state = (rng, train_state, next_timestep, action, next_timestep.reward, outcomes_new, hstate)
                return step_state, transition
            
            initial_hstate = inner_state[-1]
            inner_state, transitions = jax.lax.scan(_env_step, inner_state, None, num_steps_per_update)
            
            # CALCULATE ADVANTAGE
            rng, train_state, cur_timestep, cur_prev_action, cur_prev_reward, outcomes_curr, cur_hstate = inner_state
            _, last_val, _ = train_state.apply_fn(
                train_state.params,
                {
                    "observation": cur_timestep.observation[:, None],
                    "prev_action": cur_prev_action[:, None],
                    "prev_reward": cur_prev_reward[:, None],
                },
                cur_hstate,
            )
            advantages, targets = calculate_gae(transitions, last_val.squeeze(1), t_config["GAMMA"], t_config["GAE_LAMBDA"])
            
            # UPDATE NETWORK
            def _update_epoch(update_state, _):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, transitions, advantages, targets = batch_info
                    
                    # NORMALIZE ADVANTAGES
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    
                    def _loss_fn(params):
                        dist, value, _ = train_state.apply_fn(
                            params,
                            {
                                "observation": transitions.obs,
                                "prev_action": transitions.prev_action,
                                "prev_reward": transitions.prev_reward,
                            },
                            init_hstate.squeeze(1),
                        )
                        log_prob = dist.log_prob(transitions.action)
                        
                        # VALUE LOSS
                        value_pred_clipped = transitions.value + (value - transitions.value).clip(-t_config["CLIP_EPS"], t_config["CLIP_EPS"])
                        value_loss = jnp.square(value - targets)
                        value_loss_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_loss, value_loss_clipped).mean()
                        
                        # ACTOR LOSS
                        ratio = jnp.exp(log_prob - transitions.log_prob)
                        actor_loss1 = advantages * ratio
                        actor_loss2 = advantages * jnp.clip(ratio, 1.0 - t_config["CLIP_EPS"], 1.0 + t_config["CLIP_EPS"])
                        actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()
                        entropy = dist.entropy().mean()
                        
                        total_loss = actor_loss + t_config["VF_COEF"] * value_loss - t_config["ENT_COEF"] * entropy
                        return total_loss, (value_loss, actor_loss, entropy)
                    
                    (loss, (vloss, aloss, entropy)), grads = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.params)
                    train_state = train_state.apply_gradients(grads=grads)
                    update_info = {
                        "total_loss": loss,
                        "value_loss": vloss,
                        "actor_loss": aloss,
                        "entropy": entropy,
                    }
                    return train_state, update_info
                
                rng_upd, train_state, init_hstate, transitions, advantages, targets = update_state
                
                rng_upd, _rng = jax.random.split(rng_upd)
                permutation = jax.random.permutation(_rng, num_envs)
                batch = (init_hstate, transitions, advantages, targets)
                batch = jtu.tree_map(lambda x: x.swapaxes(0, 1), batch)
                shuffled_batch = jtu.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
                minibatches = jtu.tree_map(
                    lambda x: jnp.reshape(x, (t_config["NUM_MINIBATCHES"], -1) + x.shape[1:]), shuffled_batch
                )
                train_state, update_info = jax.lax.scan(_update_minbatch, train_state, minibatches)
                
                update_state = (rng_upd, train_state, init_hstate, transitions, advantages, targets)
                return update_state, update_info
            
            update_state = (rng, train_state, initial_hstate[None, :], transitions, advantages, targets)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, t_config["UPDATE_EPOCHS"])
            rng, train_state = update_state[:2]
            
            loss_info = jtu.tree_map(lambda x: x.mean(-1).mean(-1), loss_info)
            inner_state = (rng, train_state, cur_timestep, cur_prev_action, cur_prev_reward, outcomes_curr, cur_hstate)
            return inner_state, loss_info
        
        inner_state = (rng, train_state, timestep, prev_action, prev_reward, outcomes, hstate)
        inner_state, loss_info = jax.lax.scan(_update_step, inner_state, None, t_config["NUM_INNER_UPDATES"])
        rng, train_state = inner_state[:2]
        final_timestep = inner_state[2]
        final_prev_action = inner_state[3]
        final_prev_reward = inner_state[4]
        final_outcomes = inner_state[5]
        final_hstate = inner_state[6]
        
        # Compute success rate from outcomes
        success_rate_by_env = final_outcomes[:, 1] / jnp.maximum(final_outcomes[:, 0], 1)
        
        loss_info = jtu.tree_map(lambda x: x.mean(-1), loss_info)
        
        # Log metrics
        def callback(metric):
            safe_wandb_log(
                {
                    "env_step": metric["update_steps"] * num_steps_per_env * num_envs,
                    **metric["loss_info"],
                    "train/success_rate_mean": metric["success_rate_mean"],
                    "lr": metric["lr"],
                    "update_count": metric["update_steps"],
                }
            )
        
        metric = {
            "update_steps": update_steps,
            "loss_info": loss_info,
            "success_rate_mean": success_rate_by_env.mean(),
            "lr": train_state.opt_state[-1].hyperparams["learning_rate"] if t_config["ANNEAL_LR"] else t_config["LR"],
        }
        jax.experimental.io_callback(callback, None, metric)
        
        #  mix of fresh + sampled from buffer
        rng, _rng1, _rng2, _rng3 = jax.random.split(rng, 4)
        
        # Generate fresh environments
        gen_reset_rng = jax.random.split(_rng1, t_config["NUM_ENVS_TO_GENERATE"])
        gen_timestep = jax.vmap(env.reset, in_axes=(None, 0))(env_params, gen_reset_rng)
        
        # Sample from learnability buffer
        sampled_idxs = jax.random.randint(_rng2, (t_config["NUM_ENVS_FROM_SAMPLED"],), 0, num_env_instances)
        sampled_rngs = instances[sampled_idxs]
        sampled_timestep = jax.vmap(env.reset, in_axes=(None, 0))(env_params, sampled_rngs)
        
        # Concatenate
        new_timestep = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=0), gen_timestep, sampled_timestep)
        new_prev_action = jnp.zeros(num_envs, dtype=jnp.int32)
        new_prev_reward = jnp.zeros(num_envs)
        new_hstate = network.initialize_carry(batch_size=t_config["NUM_ACTORS"])
        
        update_steps = update_steps + 1
        new_runner_state = (rng, train_state, new_timestep, new_prev_action, new_prev_reward, new_hstate, update_steps)
        return (new_runner_state, instances), metric

    # Log buffer visualization

    def log_buffer(learnability, level_rngs, epoch, log_key="best_maps", cnn_scores=None):
        num_samples = level_rngs.shape[0]
        rows = 2
        cols = num_samples // rows
        fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            score = learnability[i]
            single_rng = level_rngs[i]
            single_timestep = env.reset(env_params, single_rng)
            img = env.render(env_params, single_timestep)
            ax.imshow(np.array(img))
            ax.set_xticks([]); ax.set_yticks([])
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

    # Train and eval step

    @partial(jax.jit, static_argnums=(2,))
    def train_and_eval_step(runner_state, eval_rng, learn_method, cnn_state, target_mu):
        learnability_rng, eval_rng_fixed, eval_rng_bench, buffer_sample_rng = jax.random.split(eval_rng, 4)

        # ---- Generate learnability buffer ----
        if learn_method == "cnn":
            learnability_scores, instances, worst_scores, worst_instances, unsolvable_scores, unsolvable_instances, cnn_top_scores, cnn_worst_scores, curriculum_diag = get_learnability_set_cnn(
                learnability_rng, cnn_graphdef, cnn_state, target_mu
            )
        elif learn_method == "standard":
            learnability_scores, instances, worst_scores, worst_instances, unsolvable_scores, unsolvable_instances, cnn_top_scores, cnn_worst_scores, curriculum_diag = get_learnability_set_standard(
                learnability_rng, runner_state[1].params
            )
        elif learn_method == "hybrid":
            learnability_scores, instances, worst_scores, worst_instances, unsolvable_scores, unsolvable_instances, cnn_top_scores, cnn_worst_scores, curriculum_diag = get_learnability_set_hybrid(
                learnability_rng, runner_state[1].params, cnn_graphdef, cnn_state, target_mu
            )
        else:  # random
            learnability_scores, instances, worst_scores, worst_instances, unsolvable_scores, unsolvable_instances, cnn_top_scores, cnn_worst_scores, curriculum_diag = get_learnability_set_random(
                learnability_rng
            )
        
        # ---- Train ----
        runner_state_instances = (runner_state, instances)
        runner_state_instances, metrics = jax.lax.scan(train_step, runner_state_instances, None, t_config["EVAL_FREQ"])
        
        # Compute recent success rate from training outcomes
        goal_r_proxy = metrics["success_rate_mean"].mean()
        recent_success_rate = goal_r_proxy
        
        # ---- Eval on the fixed training ruleset ----
        rnn_hidden_dim = env_config["rnn_hidden_dim"]
        rnn_num_layers = env_config["rnn_num_layers"]
        
        eval_fixed_rng = jax.random.split(eval_rng_fixed, env_config["eval_num_envs"])
        eval_fixed_stats = jax.vmap(rollout, in_axes=(0, None, None, None, None, None))(
            eval_fixed_rng,
            env,
            env_params,  # uses the fixed ruleset
            runner_state_instances[0][1],  # train_state
            jnp.zeros((1, rnn_num_layers, rnn_hidden_dim)),
            env_config["eval_num_episodes"],
        )
        
        # ---- Eval on benchmark-sampled rulesets ----
        benchmark = xminigrid.load_benchmark(env_config["benchmark_id"])
        eval_bench_ruleset_rng, eval_bench_reset_rng = jax.random.split(eval_rng_bench)
        eval_bench_ruleset_rng = jax.random.split(eval_bench_ruleset_rng, env_config["eval_num_envs"])
        eval_bench_reset_rng = jax.random.split(eval_bench_reset_rng, env_config["eval_num_envs"])
        
        eval_bench_rulesets = jax.vmap(benchmark.sample_ruleset)(eval_bench_ruleset_rng)
        eval_bench_env_params = env_params.replace(ruleset=eval_bench_rulesets)
        
        eval_bench_stats = jax.vmap(rollout, in_axes=(0, None, 0, None, None, None))(
            eval_bench_reset_rng,
            env,
            eval_bench_env_params,
            runner_state_instances[0][1],  # train_state
            jnp.zeros((1, rnn_num_layers, rnn_hidden_dim)),
            env_config["eval_num_episodes"],
        )
        
        # ---- Collect metrics ----
        test_metrics = {
            "learnability_set_scores": learnability_scores,
            "learnability_set_mean_score": learnability_scores.mean(),
            "worst_learnability_scores": worst_scores,
            "worst_learnability_mean_score": worst_scores.mean(),
            "recent_success_rate": recent_success_rate,
            "target_mu": target_mu,
            # Fixed ruleset eval
            "eval_fixed/returns_mean": eval_fixed_stats.reward.mean(),
            "eval_fixed/returns_median": jnp.median(eval_fixed_stats.reward),
            "eval_fixed/lengths_mean": eval_fixed_stats.length.mean(),
            "eval_fixed/success_rate_mean": jnp.mean(eval_fixed_stats.success / jnp.maximum(eval_fixed_stats.episodes, 1)),
            # Benchmark eval
            "eval_bench/returns_mean": eval_bench_stats.reward.mean(),
            "eval_bench/returns_median": jnp.median(eval_bench_stats.reward),
            "eval_bench/lengths_mean": eval_bench_stats.length.mean(),
            "eval_bench/success_rate_mean": jnp.mean(eval_bench_stats.success / jnp.maximum(eval_bench_stats.episodes, 1)),
        }
        test_metrics.update(curriculum_diag)

        runner_state, _ = runner_state_instances
        test_metrics["update_count"] = runner_state[-1]  # update_steps

        # Highest / lowest scores within the selected buffer
        highest_idx = jnp.argsort(learnability_scores)[-20:]
        highest_scores = learnability_scores[highest_idx]
        highest_cnn = cnn_top_scores[highest_idx]
        highest_rngs = instances[highest_idx]

        lowest_idx = jnp.argsort(learnability_scores)[:20]
        lowest_scores = learnability_scores[lowest_idx]
        lowest_cnn = cnn_top_scores[lowest_idx]
        lowest_rngs = instances[lowest_idx]

        # random sample of the buffer shows of what the agent actually trains on, since train_step draws from the buffer uniformly
        sample_idx = jax.random.choice(buffer_sample_rng, learnability_scores.shape[0], shape=(20,), replace=False)
        sample_scores = learnability_scores[sample_idx]
        sample_cnn = cnn_top_scores[sample_idx]
        sample_rngs = instances[sample_idx]

        return (runner_state,
                (learnability_scores[-20:], cnn_top_scores[-20:], instances[-20:]),
                (worst_scores, cnn_worst_scores, worst_instances),
                (unsolvable_scores, unsolvable_instances),
                (highest_scores, highest_cnn, highest_rngs),
                (lowest_scores, lowest_cnn, lowest_rngs),
                (sample_scores, sample_cnn, sample_rngs),
                test_metrics)

    # Main training loop
    rng, _rng = jax.random.split(rng)
    runner_state = (
        _rng,
        train_state,
        timestep,
        prev_action,
        prev_reward,
        init_hstate_train,
        0,  # update_steps
    )
    
    checkpoint_steps = max(1, t_config["NUM_UPDATES"] // t_config["EVAL_FREQ"] // t_config["NUM_CHECKPOINTS"])
    print(f'eval freq: {t_config["EVAL_FREQ"]}')
    print(f'num updates: {t_config["NUM_UPDATES"]}')
    print(f'total eval steps: {t_config["NUM_UPDATES"] // t_config["EVAL_FREQ"]}')
    
    target_mu = jnp.array(config.get("CURRICULUM_START_MU", 0.05))
    
    for eval_step in range(int(t_config["NUM_UPDATES"] // t_config["EVAL_FREQ"])):
        start_time = time.time()
        rng, eval_rng = jax.random.split(rng)
        
        curriculum_strategy = config.get("CURRICULUM_STRATEGY", "time_based")
        if curriculum_strategy == "time_based":
            current_update = runner_state[-1]
            target_mu = jnp.clip(current_update / t_config["NUM_UPDATES"], 0.0, 1.0)
        
        (runner_state, top_instances_data, worst_instances_data,
         unsolvable_instances_data, highest_in_top_data,
         lowest_in_top_data, buffer_sample_data, metrics) = train_and_eval_step(
            runner_state, eval_rng, learn_method, cnn_state, target_mu
        )
        
        if curriculum_strategy in ["performance_adaptive", "gaussian_frontier"]:
            recent_success_rate = metrics["recent_success_rate"]
            step_size = config.get("CURRICULUM_STEP_SIZE", 0.05)
            target_mu = jnp.where(recent_success_rate > 0.8, jnp.clip(target_mu + step_size, 0.0, 1.0), target_mu)
            target_mu = jnp.where(recent_success_rate < 0.2, jnp.clip(target_mu - step_size, 0.0, 1.0), target_mu)
        
        curr_time = time.time()
        print(f'eval_step {eval_step} completed')
        
        # Each maps grid shows the compound/selection score plus (for cnn/hybrid) the raw CNN difficulty
        update_count = int(metrics["update_count"])
        top_scores20, top_cnn20, top_rngs20 = top_instances_data
        log_buffer(top_scores20, top_rngs20, update_count, log_key="best_maps",
                   cnn_scores=top_cnn20 if needs_cnn else None)
        hi_s, hi_cnn, hi_rngs = highest_in_top_data
        log_buffer(hi_s, hi_rngs, update_count, log_key="highest_in_curriculum", cnn_scores=hi_cnn if needs_cnn else None)
        lo_s, lo_cnn, lo_rngs = lowest_in_top_data
        log_buffer(lo_s, lo_rngs, update_count, log_key="lowest_in_curriculum", cnn_scores=lo_cnn if needs_cnn else None)
        samp_s, samp_cnn, samp_rngs = buffer_sample_data
        log_buffer(samp_s, samp_rngs, update_count, log_key="buffer_sample", cnn_scores=samp_cnn if needs_cnn else None)
        if learn_method in ["standard", "hybrid"]:
            log_buffer(*unsolvable_instances_data, update_count, log_key="unsolvable_maps")

        metrics['time_delta'] = curr_time - start_time
        metrics["steps_per_section"] = (t_config["EVAL_FREQ"] * num_steps_per_env * num_envs) / metrics['time_delta']
        # Wrap distribution arrays as wandb histograms; everything logs keyed by update_count
        for _hk in [k for k in metrics if isinstance(k, str) and k.startswith("hist/")]:
            _h = safe_histogram(metrics[_hk])
            if _h is None:
                del metrics[_hk]
            else:
                metrics[_hk] = _h
        metrics["update_count"] = update_count
        safe_wandb_log(metrics)
        print(f'eval_step {eval_step} logged')
        
        if (eval_step % checkpoint_steps == 0) & (eval_step > 0):
            if config["SAVE_PATH"] is not None:
                params = runner_state[1].params
                
                save_dir = os.path.join(config["SAVE_PATH"], run.name)
                os.makedirs(save_dir, exist_ok=True)
                save_params(params, f'{save_dir}/model.safetensors')
                print(f'Parameters saved in {save_dir}/model.safetensors')
                
                artifact = wandb.Artifact(f'{run.name}-checkpoint', type='checkpoint')
                artifact.add_file(f'{save_dir}/model.safetensors')
                artifact.save()

    print('Training complete')
    # Final save
    if config["SAVE_PATH"] is not None:
        params = runner_state[1].params
        
        save_dir = os.path.join(config["SAVE_PATH"], run.name)
        os.makedirs(save_dir, exist_ok=True)
        save_params(params, f'{save_dir}/model.safetensors')
        print(f'Parameters saved in {save_dir}/model.safetensors')
        
        artifact = wandb.Artifact(f'{run.name}-checkpoint', type='checkpoint')
        artifact.add_file(f'{save_dir}/model.safetensors')
        artifact.save()


if __name__ == "__main__":
    with jax.disable_jit(False):
        main()
