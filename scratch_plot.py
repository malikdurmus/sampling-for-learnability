import jax
import jax.numpy as jnp
from sfl.train.rlhf_utils import get_jaxnav_rasterizer
from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav
from PIL import Image
import numpy as np

def to_uint8(arr):
    return (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)

env = JaxNav(num_agents=2)
rng = jax.random.PRNGKey(42)
_, state = env.reset(rng)

# Render at 120x120 natively
render_120 = get_jaxnav_rasterizer(img_height=120, img_width=120, cell_size=1.0)
img_120 = render_120(state)
Image.fromarray(to_uint8(img_120)).save('test_120_native.png')

# Render at 480x480 natively and resize
render_480 = get_jaxnav_rasterizer(img_height=480, img_width=480, cell_size=1.0)
img_480 = render_480(state)
img_480_resized = jax.image.resize(img_480, (120, 120, 3), method="lanczos3")
Image.fromarray(to_uint8(img_480_resized)).save('test_120_resized.png')
