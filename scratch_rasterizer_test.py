import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav

def get_jaxnav_rasterizer_new(img_height=120, img_width=120, map_height=11, map_width=11, cell_size=1.0):
    px_idx = jnp.arange(img_width)
    py_idx = jnp.arange(img_height)
    px_grid, py_grid = jnp.meshgrid(px_idx, py_idx)
    
    world_x = (px_grid + 0.5) / img_width * map_width
    world_y = (img_height - 0.5 - py_grid) / img_height * map_height
    world_xy = jnp.stack([world_x, world_y], axis=-1)
    
    grid_col = jnp.clip(jnp.floor(world_x).astype(jnp.int32), 0, map_width - 1)
    grid_row = jnp.clip(jnp.floor(world_y).astype(jnp.int32), 0, map_height - 1)
    
    px_size = map_width / img_width

    def rotation_matrix(theta):
        cos_t = jnp.cos(theta)
        sin_t = jnp.sin(theta)
        return jnp.array([[cos_t, -sin_t], [sin_t,  cos_t]])
        
    def transform_coords(pos, theta, coords):
        r = rotation_matrix(theta)
        return jnp.matmul(coords, r.T) + pos

    def dist_to_segment(p, a, b):
        ab = b - a
        l2 = jnp.sum(ab ** 2)
        t = jnp.sum((p - a) * ab, axis=-1) / jnp.maximum(l2, 1e-10)
        t = jnp.clip(t, 0.0, 1.0)
        proj = a + t[..., None] * ab
        return jnp.sqrt(jnp.sum((p - proj) ** 2, axis=-1))

    def sdf_box(p, half_extents):
        d = jnp.abs(p) - half_extents
        out_dist = jnp.linalg.norm(jnp.maximum(d, 0.0), axis=-1)
        in_dist = jnp.minimum(jnp.maximum(d[..., 0], d[..., 1]), 0.0)
        return out_dist + in_dist

    dir_line = jnp.array([[0.0, 0.0], [0.254, 0.0]])

    def render_state(env_state):
        is_wall = env_state.map_data[grid_row, grid_col]
        img = jnp.where(is_wall[..., None], jnp.array([0.0, 0.0, 0.0]), jnp.array([1.0, 1.0, 1.0]))
        
        def render_agent(img_carry, agent_idx):
            pos = env_state.pos[agent_idx]
            theta = env_state.theta[agent_idx]
            goal = env_state.goal[agent_idx]
            done = env_state.done[agent_idx]
            
            pos_s = pos / cell_size
            goal_s = goal / cell_size
            
            # Agent-Goal Line
            line_half = 0.05
            d_goal_line = dist_to_segment(world_xy, pos_s, goal_s)
            alpha_line = jnp.clip(0.5 - (d_goal_line - line_half) / px_size, 0.0, 1.0) * 0.5
            img_curr = img_carry * (1.0 - alpha_line[..., None])
            
            # Goal marker
            marker_half = 0.2
            marker_thick = 0.05
            p_goal = world_xy - goal_s
            sdf_goal_h = sdf_box(p_goal, jnp.array([marker_half, marker_thick]))
            sdf_goal_v = sdf_box(p_goal, jnp.array([marker_thick, marker_half]))
            sdf_goal = jnp.minimum(sdf_goal_h, sdf_goal_v)
            alpha_goal = jnp.clip(0.5 - sdf_goal / px_size, 0.0, 1.0)
            img_curr = img_curr * (1.0 - alpha_goal[..., None]) + jnp.array([0.0, 0.5, 0.0]) * alpha_goal[..., None]
            
            # Agent body
            p_agent = world_xy - pos_s
            rot_inv = rotation_matrix(-theta)
            p_agent = jnp.matmul(p_agent, rot_inv.T)
            agent_half_extents = jnp.array([0.25, 0.25])
            sdf_agent = sdf_box(p_agent, agent_half_extents)
            alpha_agent = jnp.clip(0.5 - sdf_agent / px_size, 0.0, 1.0)
            agent_color = jnp.where(done, jnp.array([0.0, 0.0, 0.0]), jnp.array([1.0, 0.0, 0.0]))
            img_curr = img_curr * (1.0 - alpha_agent[..., None]) + agent_color * alpha_agent[..., None]
            
            # Direction line
            dir_half_width = 0.05
            transformed_dir = transform_coords(pos, theta, dir_line) / cell_size
            d_dir = dist_to_segment(world_xy, transformed_dir[0], transformed_dir[1])
            alpha_dir = jnp.clip(0.5 - (d_dir - dir_half_width) / px_size, 0.0, 1.0)
            img_curr = img_curr * (1.0 - alpha_dir[..., None])
            
            return img_curr, None
            
        num_agents = env_state.pos.shape[0]
        img, _ = jax.lax.scan(render_agent, img, jnp.arange(num_agents))
        return jnp.clip(img, 0.0, 1.0)
        
    return render_state

env = JaxNav(num_agents=2)
rng = jax.random.PRNGKey(42)
_, env_state = env.reset(rng)

render_fn = jax.jit(get_jaxnav_rasterizer_new(120, 120))
img_120 = render_fn(env_state)
img_120_u8 = (np.array(img_120) * 255).astype(np.uint8)
Image.fromarray(img_120_u8).save("test_120_fixed.png")
print("Saved test_120_fixed.png")
