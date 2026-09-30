"""Step 1 (run with the uedrlhf venv, READ-ONLY on uedrlhf): export the first
500 pairs of the shared SEED=0 test split of the xland difficult dataset to
an npz in the SFL repo, exactly as ensemble_eval.py builds them."""
import sys
import numpy as np
sys.path.insert(0, "/home/d/durmusy/Desktop/GIT/new/uedrlhf")
from datasets import load_from_disk
from uedrlhf.fast_data import three_way_split

ds = load_from_disk("/home/d/durmusy/Desktop/GIT/new/uedrlhf/data_v3/xland_minigrid/"
                    "ruleset_difficult/difficulty_feedback_low/compiled_gemini_3_5_flash_len_21973_arr")
_, _, test = three_way_split(ds, 0)
test = test.with_format("numpy", columns=["env_0_image", "env_1_image", "preference"])
b = test[:500]
np.savez_compressed(
    "/home/d/durmusy/Desktop/GIT/updatedjnavwsfl/sampling-for-learnability/xland_test_pairs_500.npz",
    i0=np.asarray(b["env_0_image"], dtype=np.uint8),
    i1=np.asarray(b["env_1_image"], dtype=np.uint8),
    pref=np.asarray(b["preference"], dtype=np.float32))
print("exported 500 test pairs; shapes:", np.asarray(b["env_0_image"]).shape)
