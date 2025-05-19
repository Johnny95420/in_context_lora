# %%
import glob
import json
import pickle as pkl
from typing import List, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class PreEmbedDataset(Dataset):
    def __init__(
        self,
        files: List,
        target_size: Tuple = (1125, 563),
        text_data_path: str = "./final_data/description.json",
    ):
        self.files = files
        self.target_size = target_size
        with open(text_data_path, "r") as f:
            self.text_data = json.load(f)

        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path)
        img = img.resize(self.target_size)
        name = path.split("/")[-1]
        return name, self.transforms(img), self.text_data[name]["caption"]


class EmbeddedDataset(Dataset):
    def __init__(self, files_path: List[str]):
        self.files_path = files_path

    def __len__(self):
        return len(self.files_path)

    def __getitem__(self, idx):
        path = self.files_path[idx]
        with open(path, "rb") as f:
            data = pkl.load(f)
        data["latent_dist"] = data["latent_dist"].sample()[0]
        return data
