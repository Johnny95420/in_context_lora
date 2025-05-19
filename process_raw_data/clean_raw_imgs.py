# %%
import glob
import os
import shutil
from typing import List

import sklearn
import torch
import tqdm
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
from omegaconf import OmegaConf
from dataclasses import dataclass


clean_config = OmegaConf.load("data_clean_config.yaml")


@dataclass
class CleanConfig:
    training_data_human = clean_config["training_data_human"]
    training_data_object = clean_config["training_data_object"]
    all_data = clean_config["all_data"]
    target_path = clean_config["paired_data_path"]


device = "cuda"
model, processor = AutoModel.from_pretrained(
    "google/siglip-so400m-patch14-384"
), AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
model = model.to(device)


# %%
def fit_image_classification_model(
    human_files: List, object_files: List, model: torch.Tensor, batch_size: int = 5
):
    files = human_files + object_files
    l = len(files)
    images, embeddings = [], []
    for i in tqdm.tqdm(range(l)):
        images.append(Image.open(files[i]))
        if len(images) >= batch_size or (i == l - 1 and images):
            with torch.no_grad():
                inputs = processor(
                    images=images,
                    return_tensors="pt",
                    padding=True,
                )
                for key in inputs:
                    inputs[key] = inputs[key].to(device)
                outputs = model.get_image_features(**inputs)
            embeddings.append(outputs.cpu())
            images = []
    x = torch.cat(embeddings).numpy()
    y = [1] * len(human_files) + [0] * len(object_files)
    svc = sklearn.svm.SVC(gamma="auto")
    svc.fit(x, y)
    return svc


def classify_and_copy_images(
    dirs,
    target_path,
    processor,
    model,
    svc,
    device="cuda",
):
    for dir in tqdm(dirs):
        folder_name = dir.split("/")[-1]
        human_path = os.path.join(target_path, folder_name, "human")
        obj_path = os.path.join(target_path, folder_name, "object")
        os.makedirs(human_path, exist_ok=True)
        os.makedirs(obj_path, exist_ok=True)

        files = glob.glob(f"{dir}/*.*")
        predict_images, predict_files = [], []
        for f in files:
            try:
                img = Image.open(f)
                predict_images.append(img)
                predict_files.append(f)
            except Exception as e:
                print(f"Error loading {f}: {e}")

        if not predict_images:
            print(f"No valid images in {dir}")
            continue

        with torch.no_grad():
            inputs = processor(
                images=predict_images,
                return_tensors="pt",
                padding=True,
            )
            for key in inputs:
                inputs[key] = inputs[key].to(device)
            outputs = model.get_image_features(**inputs).cpu().numpy()

        predict_output = svc.predict(outputs)
        assert len(predict_output) == len(predict_files)
        for cls, pred_f in zip(predict_output, predict_files):
            name = os.path.basename(pred_f)
            if cls == 0:
                shutil.copy(pred_f, obj_path)
            else:
                shutil.copy(pred_f, human_path)


# %%
if __name__ == "__main__":
    human_files = glob.glob(f"{CleanConfig.training_data_human}/*.*")
    object_files = glob.glob(f"{CleanConfig.training_data_object}/*.*")
    svc = fit_image_classification_model(human_files, object_files, model, 10)
    dirs = glob.glob(f"{CleanConfig.all_data}/*")
    classify_and_copy_images(dirs, CleanConfig.target_path, processor, model, svc)
