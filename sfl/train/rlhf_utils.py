import jax
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
from functools import partial

# ==========================================
# 1. THE CNN MODEL
# ==========================================
class CNN(nnx.Module):
    def __init__(self, * , rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(in_features=3, out_features=32, kernel_size=(5,5), padding="VALID", rngs=rngs)
        self.batch_norm1 = nnx.BatchNorm(num_features=32, rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=0.025, rngs=rngs)
        self.conv2 = nnx.Conv(in_features=32, out_features=64, kernel_size=(3,3), padding="VALID", rngs=rngs)
        self.batch_norm2 = nnx.BatchNorm(num_features=64, rngs=rngs)
        self.avg_pool = partial(nnx.avg_pool, window_shape=(2,2), strides=(2,2))
        self.linear1 = nnx.Linear(in_features=12544, out_features=256, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=0.025, rngs=rngs)
        self.linear2 = nnx.Linear(in_features=256, out_features=1, rngs=rngs)

    def __call__(self, x, rngs: nnx.Rngs | None = None):
        x = self.conv1(x)
        x = self.dropout1(x, rngs=rngs)
        x = self.batch_norm1(x)
        x = nnx.relu(x)
        x = self.avg_pool(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = nnx.relu(x)
        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1) 
        x = nnx.relu(self.dropout2(self.linear1(x), rngs=rngs))
        x = self.linear2(x)
        return x.squeeze()



class ResNetBlock(nnx.Module):

    def __init__(self, in_features: int, out_features: int, stride: int = 1, * , rngs: nnx.Rngs):

        self.conv1 = nnx.Conv(in_features, out_features, kernel_size=(3, 3), 
                            strides=stride, padding='SAME', use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(out_features, rngs=rngs)

        self.conv2 = nnx.Conv(out_features, out_features, kernel_size=(3, 3), 
                            strides=1, padding='SAME', use_bias=False, rngs=rngs)        
        self.bn2 = nnx.BatchNorm(out_features, rngs=rngs)

        if in_features != out_features or stride != 1:
            self.shortcut_conv = nnx.Conv(in_features, out_features, kernel_size=(1, 1), strides=stride, padding='SAME', use_bias=False, rngs=rngs)
            self.shortcut_bn = nnx.BatchNorm(out_features, rngs=rngs)
        else:
            self.shortcut_conv = None
            self.shortcut_bn = None       
        
    def __call__(self, x, deterministic: bool):
        residual = x
    
        #
        y = self.conv1(x)
        y = self.bn1(y, use_running_average=deterministic)
        y = nnx.relu(y)
        
        y = self.conv2(y)
        y = self.bn2(y, use_running_average=deterministic)
        
        # 2. shortcut (resnet)
        if self.shortcut_conv is not None:
            residual = self.shortcut_conv(residual)
            residual = self.shortcut_bn(residual, use_running_average=deterministic)
            
        # 3. rejoin paths
        return nnx.relu(residual + y)




class LearnabilityResNet(nnx.Module):
    """A Small ResNet customized for grid mapping, written in flax.nnx."""
    
    def __init__(self, *, rngs: nnx.Rngs):
        # stem
        self.stem_conv = nnx.Conv(3, 32, kernel_size=(3, 3), strides=1, padding='SAME', use_bias=False, rngs=rngs)
        self.stem_bn = nnx.BatchNorm(32, rngs=rngs)
        
        # (~62x62)
        self.stage1_block1 = ResNetBlock(in_features=32, out_features=64, stride=2, rngs=rngs)
        self.stage1_block2 = ResNetBlock(in_features=64, out_features=64, stride=1, rngs=rngs)
        
        # (~31x31)
        self.stage2_block1 = ResNetBlock(in_features=64, out_features=128, stride=2, rngs=rngs)
        self.stage2_block2 = ResNetBlock(in_features=128, out_features=128, stride=1, rngs=rngs)
        
        # (15x15)
        self.stage3_block1 = ResNetBlock(in_features=128, out_features=256, stride=2, rngs=rngs)
        self.stage3_block2 = ResNetBlock(in_features=256, out_features=256, stride=1, rngs=rngs)
        
        # scoring Head
        self.head_linear1 = nnx.Linear(in_features=256, out_features=128, rngs=rngs)
        self.head_dropout = nnx.Dropout(rate=0.3, rngs=rngs)
        self.head_linear2 = nnx.Linear(in_features=128, out_features=1, rngs=rngs)

    def __call__(self, x, deterministic: bool = False, rngs: nnx.Rngs | None = None):
        
        # stem
        x = self.stem_conv(x)
        x = self.stem_bn(x, use_running_average=deterministic)
        x = nnx.relu(x)
        
        # stages
        x = self.stage1_block1(x, deterministic=deterministic)
        x = self.stage1_block2(x, deterministic=deterministic)
        
        x = self.stage2_block1(x, deterministic=deterministic)
        x = self.stage2_block2(x, deterministic=deterministic)
        
        x = self.stage3_block1(x, deterministic=deterministic)
        x = self.stage3_block2(x, deterministic=deterministic)
        
        # this collapses (batch , channel) into (bath, channel)
        x = jnp.max(x, axis=(1, 2))
        
        #  Scoring Head
        x = self.head_linear1(x)
        x = nnx.relu(x)
        x = self.head_dropout(x, deterministic=deterministic, rngs=rngs)
        x = self.head_linear2(x)
        
        return jnp.squeeze(x, axis=-1)
    


# ==========================================
# 2. MODEL LOADER
# ==========================================
def load_learnability_model(checkpoint_path: str):
    """
    Loads the trained CNN and splits it into a static graphdef 
    and dynamic state so it can safely pass through jax.jit.
    """
    print(f"Loading CNN learnability model from {checkpoint_path}...")
    rngs = nnx.Rngs(0)
    model = LearnabilityResNet(rngs=rngs)
    
    # We load the exact abstract state shape to tell Orbax what to expect
    graphdef, abstract_state = nnx.split(model)
    
    checkpointer = ocp.StandardCheckpointer()
    restored_state = checkpointer.restore(checkpoint_path, abstract_state)
    
    # Re-merge to set the model to eval mode (disables dropout)
    model = nnx.merge(graphdef, restored_state)
    model.eval()
    
    # Split one final time to return the pure JAX components
    cnn_graphdef, cnn_state = nnx.split(model)
    
    return cnn_graphdef, cnn_state







import jax
import jax.numpy as jnp

def get_jaxnav_rasterizer(
    img_height=120,
    img_width=120,
    map_height=11.0,
    map_width=11.0,
    cell_size=1.0,
):
    # Calculate world units per pixel
    px_w = map_width / img_width
    px_h = map_height / img_height
    
    # Precompute coordinate grids (centered on pixels)
    px_idx = jnp.arange(img_width)
    py_idx = jnp.arange(img_height)
    px_grid, py_grid = jnp.meshgrid(px_idx, py_idx)
    
    # Correct coordinate mapping: 
    # px=0 should be left edge (0.0), px=img_width should be right (map_width)
    world_x = (px_grid + 0.5) * px_w
    world_y = (img_height - 0.5 - py_grid) * px_h
    world_xy = jnp.stack([world_x, world_y], axis=-1)

    def sdf_box(p, half_extents):
        """
        p: (H, W, 2)
        half_extents: (2,)
        """
        # Standard AABB SDF
        d = jnp.abs(p) - half_extents
        external = jnp.linalg.norm(jnp.maximum(d, 0.0), axis=-1)
        internal = jnp.minimum(jnp.maximum(d[..., 0], d[..., 1]), 0.0)
        return external + internal

    def dist_to_segment(p, a, b):
        """
        p: (H, W, 2)
        a: (2,)
        b: (2,)
        """
        pa = p - a
        ba = b - a
        # We use jnp.sum(... axis=-1) instead of jnp.dot to handle the (H, W, 2) shape
        num = jnp.sum(pa * ba, axis=-1)
        den = jnp.sum(ba * ba)
        
        h = jnp.clip(num / jnp.maximum(den, 1e-7), 0.0, 1.0)
        
        # h is (H, W). We add a trailing dimension [..., None] to make it (H, W, 1)
        # This allows it to broadcast correctly with ba which is (2,)
        return jnp.linalg.norm(pa - ba * h[..., None], axis=-1)

    def render_aa(dist, thickness, blur=1.0):
        """
        Anti-aliased rendering helper.
        blur: width in pixels to smooth the edge.
        """
        # thickness is in world units. We convert blur to world units.
        edge_width = blur * px_w 
        return jnp.clip(0.5 - (dist - thickness) / edge_width, 0.0, 1.0)

    def render_state(env_state):
        # 1. Background / Walls
        # Standardize grid indices
        grid_col = jnp.clip(jnp.floor(world_x / cell_size).astype(jnp.int32), 0, env_state.map_data.shape[1] - 1)
        grid_row = jnp.clip(jnp.floor(world_y / cell_size).astype(jnp.int32), 0, env_state.map_data.shape[0] - 1)
        
        is_wall = env_state.map_data[grid_row, grid_col]
        # Map: 1 (wall) -> Black (0,0,0), 0 (free) -> White (1,1,1)
        wall_color = jnp.array([0.0, 0.0, 0.0])  # Black
        free_color = jnp.array([1.0, 1.0, 1.0])  # White    

        img = jnp.where(is_wall[..., None], wall_color, free_color)

        def render_agent(img_carry, agent_idx):
            pos = env_state.pos[agent_idx]
            theta = env_state.theta[agent_idx]
            goal = env_state.goal[agent_idx]
            done = env_state.done[agent_idx]
            
            # --- Layer: Goal Line (Transparent Black) ---
            # Increase thickness slightly for low-res (at least 1 pixel)
            line_thickness = jnp.maximum(0.02, px_w * 0.5) 
            d_line = dist_to_segment(world_xy, pos, goal)
            alpha_line = render_aa(d_line, line_thickness, blur=1.5) * 0.3 # 0.3 alpha
            img_curr = img_carry * (1.0 - alpha_line[..., None])

            # --- Layer: Goal Marker (Green +) ---
            p_goal = world_xy - goal
            m_size, m_thick = 0.25, jnp.maximum(0.06, px_w * 0.8)
            sdf_g = jnp.minimum(sdf_box(p_goal, jnp.array([m_size, m_thick])), 
                                sdf_box(p_goal, jnp.array([m_thick, m_size])))
            alpha_goal = render_aa(sdf_g, 0.0, blur=1.0)
            goal_color = jnp.array([0.0, 0.6, 0.0])
            img_curr = img_curr * (1.0 - alpha_goal[..., None]) + goal_color * alpha_goal[..., None]

            # --- Layer: Agent Body (Red/Black Box) ---
            p_agent = world_xy - pos
            # Rotate points opposite to theta to align with AABB SDF
            cos_t, sin_t = jnp.cos(-theta), jnp.sin(-theta)
            rot_inv = jnp.array([[cos_t, -sin_t], [sin_t, cos_t]])
            p_agent_rot = jnp.matmul(p_agent, rot_inv.T)
            
            sdf_a = sdf_box(p_agent_rot, jnp.array([0.25, 0.25]))
            alpha_agent = render_aa(sdf_a, 0.0, blur=1.0)
            agent_color = jnp.where(done, jnp.array([0.0, 0.0, 0.0]), jnp.array([1.0, 0.0, 0.0]))
            img_curr = img_curr * (1.0 - alpha_agent[..., None]) + agent_color * alpha_agent[..., None]

            # --- Layer: Direction Line (Black) ---
            # Middle line logic (usually from center to front edge)
            dir_end = pos + jnp.array([jnp.cos(theta), jnp.sin(theta)]) * 0.25
            d_dir = dist_to_segment(world_xy, pos, dir_end)
            alpha_dir = render_aa(d_dir, jnp.maximum(0.03, px_w * 0.6), blur=1.0)
            img_curr = img_curr * (1.0 - alpha_dir[..., None])

            return img_curr, None

        num_agents = env_state.pos.shape[0]
        img_final, _ = jax.lax.scan(render_agent, img, jnp.arange(num_agents))
        return img_final

    return render_state