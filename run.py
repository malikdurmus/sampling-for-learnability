from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav
import jax
import matplotlib.pyplot as plt
jax.config.update("jax_disable_jit", True)

import time
seed = int(time.time())
print(f"Using PRNG seed: {seed}")

env1 = JaxNav(num_agents=1,map_params=  {
    "map_size": (11, 11),
    "check_valid_path" : True
})


env = JaxNav(num_agents=1) 

rng = jax.random.PRNGKey(seed)  # Dynamic seed for unique envs across runs
reset_rng, rng = jax.random.split(rng)

obsv, env_state = env.reset(rng)


fig, ax = plt.subplots(1, 1, figsize=(5, 5))

env.init_render(ax, env_state, lidar= False, agent=True,goal=True)
    