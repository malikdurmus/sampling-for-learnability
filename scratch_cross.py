import jax
import jax.numpy as jnp
from PIL import Image
import numpy as np

def to_uint8(arr):
    return (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)

def render_cross(offset_x, offset_y, marker_half, marker_thick, px_size=11/120):
    img_width = 120
    img_height = 120
    map_width = 11
    map_height = 11

    px_idx = jnp.arange(img_width)
    py_idx = jnp.arange(img_height)
    px_grid, py_grid = jnp.meshgrid(px_idx, py_idx)
    
    world_x = (px_grid + 0.5) / img_width * map_width
    world_y = (img_height - 0.5 - py_grid) / img_height * map_height
    world_xy = jnp.stack([world_x, world_y], axis=-1)

    # Goal at center + offset
    goal_s = jnp.array([5.5 + offset_x, 5.5 + offset_y])

    def sd_box(p, b):
        d = jnp.abs(p) - b
        return jnp.sqrt(jnp.sum(jnp.maximum(d, 0.0)**2, axis=-1)) + jnp.minimum(jnp.maximum(d[..., 0], d[..., 1]), 0.0)

    p_goal = world_xy - goal_s
    sd_h = sd_box(p_goal, jnp.array([marker_half, marker_thick]))
    sd_v = sd_box(p_goal, jnp.array([marker_thick, marker_half]))
    goal_sd = jnp.minimum(sd_h, sd_v)
    
    alpha_goal = jnp.clip(0.5 - goal_sd / (1.5 * px_size), 0.0, 1.0)
    
    img = jnp.ones((img_height, img_width, 3))
    goal_color = jnp.array([0.0, 0.5, 0.0])
    img = img * (1.0 - alpha_goal[..., None]) + goal_color * alpha_goal[..., None]
    
    return img

px_size = 11/120

# 1. Very thin (my previous edit)
img1 = render_cross(0.0, 0.0, 2.0 * px_size, 0.5 * px_size)
img2 = render_cross(0.4 * px_size, 0.2 * px_size, 2.0 * px_size, 0.5 * px_size)

# 2. A bit thicker
img3 = render_cross(0.0, 0.0, 2.5 * px_size, 0.75 * px_size)
img4 = render_cross(0.4 * px_size, 0.2 * px_size, 2.5 * px_size, 0.75 * px_size)

import os
os.makedirs("scratch_outputs", exist_ok=True)
Image.fromarray(to_uint8(img1)).crop((50,50,70,70)).resize((200,200), Image.NEAREST).save("scratch_outputs/thin_center.png")
Image.fromarray(to_uint8(img2)).crop((50,50,70,70)).resize((200,200), Image.NEAREST).save("scratch_outputs/thin_offset.png")
Image.fromarray(to_uint8(img3)).crop((50,50,70,70)).resize((200,200), Image.NEAREST).save("scratch_outputs/thick_center.png")
Image.fromarray(to_uint8(img4)).crop((50,50,70,70)).resize((200,200), Image.NEAREST).save("scratch_outputs/thick_offset.png")
