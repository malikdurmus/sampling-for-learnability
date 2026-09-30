import argparse
import os
import sys

def parse_args():
    p = argparse.ArgumentParser(description="Interactive script to test CNN on XLand states.")
    p.add_argument("--cnn_checkpoint", type=str,
                   default="/home/d/durmusy/Desktop/GIT/new/uedrlhf/outputs/checkpoints/legacy/old_direct_xland_medium/cnn_compiled_gemini_3_5_flash_len_21980_3ws74l1c/epoch_7",
                   help="Path to the CNN orbax checkpoint directory")
    p.add_argument("--out_dir", type=str, default="./interactive_test_outputs_xland",
                   help="Directory to save the generated images")
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    return p.parse_args()

_args = parse_args()
if _args.cpu:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
else:
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless mode for SSH!
import matplotlib.pyplot as plt
from flax import nnx

# Import project specifics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfl.train.rlhf_utils import load_xland_learnability_model
import xminigrid
from xminigrid.wrappers import GymAutoResetWrapper
from xminigrid.types import RuleSet
from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles
from xminigrid.core.goals import AgentHoldGoal
from xminigrid.core.rules import TileNearRule, AgentHoldRule

# Re-use the renderer builder from xland_sfl.py
def build_xland_render_fn(tile_size=32, target_size=200):
    from xminigrid.rendering.rgb_render import render_tile
    from xminigrid.core.constants import NUM_COLORS, NUM_TILES
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
    
    def _render_single(grid, agent):
        grid_h, grid_w = grid.shape[0], grid.shape[1]
        flat_idxs = grid[:, :, 0].astype(jnp.int32) * NUM_COLORS + grid[:, :, 1].astype(jnp.int32)
        rendered = jnp.take(tile_cache_jax, flat_idxs, axis=0)
        agent_y, agent_x = agent.position[0], agent.position[1]
        agent_flat_idx = flat_idxs[agent_y, agent_x]
        agent_tile = agent_tile_cache_jax[agent_flat_idx, agent.direction.astype(jnp.int32)]
        rendered = rendered.at[agent_y, agent_x].set(agent_tile)
        img = rendered.transpose((0, 2, 1, 3, 4))
        img = img.reshape(grid_h * tile_size, grid_w * tile_size, 3)
        img = img.astype(jnp.float32) / 255.0
        img = jax.image.resize(img, (target_size, target_size, 3), method='bilinear')
        return img
    return _render_single

def make_medium_ruleset():
    yellow_ball = TILES_REGISTRY[Tiles.BALL, Colors.YELLOW]
    blue_key = TILES_REGISTRY[Tiles.KEY, Colors.BLUE]
    green_star = TILES_REGISTRY[Tiles.STAR, Colors.GREEN]
    purple_hex = TILES_REGISTRY[Tiles.HEX, Colors.PURPLE]
    brown_pyramid = TILES_REGISTRY[Tiles.PYRAMID, Colors.BROWN]
    orange_square = TILES_REGISTRY[Tiles.SQUARE, Colors.ORANGE]
    med_goal = AgentHoldGoal(tile=purple_hex)
    med_rule1 = TileNearRule(tile_a=yellow_ball, tile_b=blue_key, prod_tile=green_star)
    med_rule2 = AgentHoldRule(tile=green_star, prod_tile=purple_hex)
    return RuleSet(
        goal=med_goal.encode(),
        rules=jnp.vstack([med_rule1.encode(), med_rule2.encode()]),
        init_tiles=jnp.array([yellow_ball, blue_key, brown_pyramid, orange_square]),
    )

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("\n[1] Initializing XLand-MiniGrid...")
    env, env_params = xminigrid.make("XLand-MiniGrid-R4-13x13")
    env = GymAutoResetWrapper(env)
    env_params = env_params.replace(ruleset=make_medium_ruleset())
    reset_fn = jax.jit(lambda rng: env.reset(env_params, rng))
    
    print(f"\n[2] Loading XLand CNN from: {args.cnn_checkpoint}")
    cnn_graphdef, cnn_state = load_xland_learnability_model(args.cnn_checkpoint)
    cnn_model = nnx.merge(cnn_graphdef, cnn_state)
    cnn_model.eval()
    
    print("\n[3] Building Rasterizers...")
    render_cnn_fn = jax.jit(build_xland_render_fn(target_size=200))
    
    @jax.jit
    def run_cnn_inference(img_cnn):
        img_batch = jnp.expand_dims(img_cnn, axis=0)
        # Model now scores higher = harder
        score = cnn_model(img_batch, deterministic=True)
        return score[0]

    rng = jax.random.PRNGKey(42)
    step = 0
    
    print("\n========================================================")
    print(" [4] Warming up to establish Min-Max normalization bounds...")
    print("========================================================")
    
    rng, _rng = jax.random.split(rng)
    warmup_rngs = jax.random.split(_rng, 32)
    
    @jax.jit
    def get_warmup_scores(rngs):
        batch_timesteps = jax.vmap(env.reset, in_axes=(None, 0))(env_params, rngs)
        grids = batch_timesteps.state.grid
        agents = batch_timesteps.state.agent
        imgs = jax.vmap(render_cnn_fn)(grids, agents)
        return cnn_model(imgs, deterministic=True)
        
    warmup_scores = get_warmup_scores(warmup_rngs)
    c_min = float(jnp.min(warmup_scores))
    c_max = float(jnp.max(warmup_scores))
    
    print(f"   Distribution bounds found: Min = {c_min:.4f}, Max = {c_max:.4f}")
    
    print("\n========================================================")
    print(" READY! Generating maps...")
    print("========================================================")
    
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    
    while True:
        rng, reset_rng = jax.random.split(rng)
        
        # 1. Generate environment
        env_state = reset_fn(reset_rng)
        
        # 2. Rasterize
        img_cnn = render_cnn_fn(env_state.state.grid, env_state.state.agent)
        
        # 3. Predict learnability 
        score = run_cnn_inference(img_cnn)
        raw_score = float(score)
        norm_score = (raw_score - c_min) / (c_max - c_min + 1e-8)
        
        # 4. Display to screen
        axes.clear()
        axes.imshow(np.array(img_cnn))
        axes.set_title(f"CNN Input (200x200)\nNorm Score: {norm_score:.4f} (Raw: {raw_score:.1f})")
        axes.axis('off')
        
        # 5. Save to disk
        out_path = os.path.join(args.out_dir, f"xland_env_{step:04d}_score_{norm_score:.4f}.png")
        plt.savefig(out_path, dpi=100)
        
        # 6. Prompt user
        print(f"\n[Env {step}] Evaluated and saved to: {out_path}")
        print(f"         Normalized Score: {norm_score:.4f}  (Raw: {raw_score:.4f})")
        
        # Auto-generate 5 items for the test run, then quit
        if step >= 4:
            print("Generated 5 images. Exiting interactive loop.")
            break
            
        step += 1

if __name__ == "__main__":
    main(_args)
