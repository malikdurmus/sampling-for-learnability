import argparse
import os
import sys

def parse_args():
    p = argparse.ArgumentParser(description="Interactive script to test CNN on JaxNav states.")
    p.add_argument("--cnn_checkpoint", type=str,
                   default="/home/d/durmusy/Desktop/GIT/new/uedrlhf/outputs/checkpoints/finetune_linear_checkpoints/nc5766sd/epoch_7",
                   help="Path to the CNN orbax checkpoint directory")
    p.add_argument("--out_dir", type=str, default="./interactive_test_outputs",
                   help="Directory to save the generated images")
    p.add_argument("--num_agents", type=int, default=1,
                   help="Number of agents in JaxNav")
    p.add_argument("--cell_size", type=float, default=1.0,
                   help="Rasterizer cell_size")
    p.add_argument("--native_size", type=int, default=  500,
                   help="High-res rasterizer resolution for display")
    p.add_argument("--cnn_size", type=int, default=200,
                   help="CNN input resolution")
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
import matplotlib.pyplot as plt
from flax import nnx

# Disable interactive mode so we can control when the window updates
plt.ioff()

# Import project specifics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfl.train.rlhf_utils import load_learnability_model, get_jaxnav_rasterizer
from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("\n[1] Initializing JaxNav...")
    env = JaxNav(num_agents=args.num_agents)
    
    print(f"\n[2] Loading Learnability CNN from: {args.cnn_checkpoint}")
    cnn_graphdef, cnn_state = load_learnability_model(args.cnn_checkpoint)
    cnn_model = nnx.merge(cnn_graphdef, cnn_state)
    cnn_model.eval()
    
    print("\n[3] Building Rasterizers...")
    render_native_state = get_jaxnav_rasterizer(
        img_height=args.native_size,
        img_width=args.native_size,
        map_height=11,
        map_width=11,
        cell_size=args.cell_size,
    )
    render_cnn_state = get_jaxnav_rasterizer(
        img_height=args.cnn_size,
        img_width=args.cnn_size,
        map_height=11,
        map_width=11,
        cell_size=args.cell_size,
    )
    # JIT the functions for speed
    render_native_fn = jax.jit(render_native_state)
    render_cnn_fn = jax.jit(render_cnn_state)
    reset_fn = jax.jit(env.reset)
    
    @jax.jit
    def run_cnn_inference(img_cnn):
        # Add batch dim
        img_batch = jnp.expand_dims(img_cnn, axis=0)
        score = cnn_model(img_batch, deterministic=True)
        return score[0]

    rng = jax.random.PRNGKey(42)
    step = 0
    
    print("\n========================================================")
    print(" READY! Showing initial state.")
    print(" Make sure the Matplotlib window is visible.")
    print("========================================================")
    
    # We will use a single figure and update its contents
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plt.show(block=False)
    
    while True:
        rng, reset_rng = jax.random.split(rng)
        
        # 1. Generate environment
        _, env_state = reset_fn(reset_rng)
        
        # 2. Rasterize
        img_native = render_native_fn(env_state)
        img_cnn = render_cnn_fn(env_state)
        
        # 3. Predict learnability 
        score = run_cnn_inference(img_cnn)
        score_val = float(score)
        
        # 4. Display to screen
        axes[0].clear()
        axes[0].imshow(np.array(img_native))
        axes[0].set_title(f"Native Rasterizer ({args.native_size}x{args.native_size})\nEnv {step}")
        axes[0].axis('off')
        
        axes[1].clear()
        axes[1].imshow(np.array(img_cnn))
        axes[1].set_title(f"CNN Input ({args.cnn_size}x{args.cnn_size})\nLearnability Score: {score_val:.4f}")
        axes[1].axis('off')
        
        plt.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events() # Update the window immediately
        
        # 5. Save to disk
        out_path = os.path.join(args.out_dir, f"env_{step:04d}_score_{score_val:.4f}.png")
        plt.savefig(out_path, dpi=100)
        
        # 6. Prompt user
        print(f"\n[Env {step}] Evaluated and saved to: {out_path}")
        print(f"         Predicted Score: {score_val:.4f}")
        
        user_input = input("Press 'n' to generate another environment, or 'q' to quit: ").strip().lower()
        if user_input == 'q':
            print("Exiting...")
            break
        elif user_input != 'n' and user_input != '':
            print("Continuing anyway...")
        
        step += 1

if __name__ == "__main__":
    main(_args)
