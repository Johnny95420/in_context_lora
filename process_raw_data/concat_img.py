# %%
import glob
import os
import random
import shutil
from dataclasses import dataclass

import numpy as np
import tqdm
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

config = OmegaConf.load("data_clean_config.yaml")


@dataclass
class ConcatConfig:
    pooled_data_path = config["paired_data_path"]
    concat_data_path = config["concat_data_path"]
    max_concat_n = config["max_concat_n"]


def concat_and_organize_images(
    pooled_data_path,
    concat_data_path,
    k_sample=3,
    random_seed=42,
):
    random.seed(random_seed)
    set_imgs = glob.glob(f"{pooled_data_path}/*")
    for folder in tqdm(set_imgs, desc="Concatenating images"):
        name = os.path.basename(folder)
        human_files = sorted(glob.glob(f"{folder}/human/*.jpg"))
        object_files = sorted(glob.glob(f"{folder}/object/*.jpg"))
        obj_len, h_len = len(object_files), len(human_files)
        if h_len == 0 or obj_len == 0:
            continue

        if obj_len >= 2:
            img_obj1 = Image.open(object_files[0])
            img_obj2 = Image.open(object_files[1])
        elif obj_len == 1:
            img_obj1 = Image.open(object_files[0])
            img_obj2 = Image.open(object_files[0])

        if h_len > k_sample:
            human_files = random.sample(human_files, k=k_sample)

        os.makedirs(concat_data_path, exist_ok=True)
        for idx, human_f in enumerate(human_files):
            img_human = Image.open(human_f)
            obj1_arr = np.array(img_obj1)
            obj2_arr = np.array(img_obj2)
            human_arr = np.array(img_human)
            shapes = [obj1_arr.shape[0], obj2_arr.shape[0], human_arr.shape[0]]
            imgs_arr = [obj1_arr, obj2_arr, human_arr]
            max_d = max(shapes)
            for j, s in enumerate(shapes):
                diff = max_d - s
                left, right = diff // 2, diff // 2
                if diff % 2 != 0:
                    right += 1
                imgs_arr[j] = np.pad(
                    imgs_arr[j], ((left, right), (0, 0), (0, 0)), mode="edge"
                )
            img = np.concatenate(imgs_arr, axis=1)
            Image.fromarray(img).save(
                os.path.join(concat_data_path, f"{name}_{idx}.jpg")
            )

    set_imgs = glob.glob(f"{concat_data_path}/*")
    for f in tqdm(set_imgs, desc="Sorting images by size"):
        with Image.open(f) as im:
            w, h = im.size
        size_dir = os.path.join(concat_data_path, f"{w}x{h}")
        os.makedirs(size_dir, exist_ok=True)
        shutil.move(f, os.path.join(size_dir, os.path.basename(f)))


# %%
if __name__ == "__main__":
    concat_and_organize_images(
        ConcatConfig.pooled_data_path,
        ConcatConfig.concat_data_path,
        ConcatConfig.max_concat_n,
    )
