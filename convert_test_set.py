import pickle
import jax
import jax.numpy as jnp
import numpy as np

def main():

    with open("sfl/data/test_sets/sampled_tc_100e_1a.pkl", "rb") as f:
        eval_env_instances = pickle.load(f)
        print(type(eval_env_instances))
        print(eval_env_instances)

        eval_env_instances_np = jax.tree_util.tree_map( lambda x : np.array(x)  , eval_env_instances)

    with open("sfl/data/test_sets/sampled_tc_100e_1a_np.pkl", "wb") as f:
        pickle.dump(eval_env_instances_np, f)


main()