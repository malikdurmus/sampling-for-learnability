import sys
import numpy as np
from PIL import Image
sys.path.insert(0, "/home/d/durmusy/Desktop/GIT/new/uedrlhf")
from datasets import load_from_disk
from uedrlhf.fast_data import three_way_split, to_device_uint8

ds = load_from_disk("/home/d/durmusy/Desktop/GIT/new/uedrlhf/data_v3/xland_minigrid/ruleset_difficult/difficulty_feedback_low/compiled_gemini_3_5_flash_len_21973_arr")
_, _, test = three_way_split(ds, 0)
td = to_device_uint8(test)
dataset_imgs = np.asarray(td[0][:4])  # (4, 200, 200, 3) uint8

d = np.load("xrender_cross.npz")
ours416 = d["ours"][:4]
ours200 = np.stack([np.asarray(Image.fromarray(x).resize((200, 200))) for x in ours416])

print("dataset imgs: mean RGB", dataset_imgs.reshape(-1,3).mean(0).round(1),
      "distinct colors (img0):", len(np.unique(dataset_imgs[0].reshape(-1,3), axis=0)))
print("our renders : mean RGB", ours200.reshape(-1,3).mean(0).round(1),
      "distinct colors (img0):", len(np.unique(ours200[0].reshape(-1,3), axis=0)))

rows = [np.concatenate(list(dataset_imgs), axis=1), np.concatenate(list(ours200), axis=1)]
panel = np.concatenate(rows, axis=0)
Image.fromarray(panel).save("xrender_dataset_vs_ours.png")
print("saved xrender_dataset_vs_ours.png (top: dataset, bottom: our renders)")
