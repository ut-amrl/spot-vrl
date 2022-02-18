import numpy as np
import os
from typing import Any, List, Tuple

from torch.utils.data import Dataset
import torch
import torchvision
import torchvision.transforms.functional as F


class ConcreteGrassTripletDataset(Dataset):
    def __init__(self, patches_dir: str, trajectory: str) -> None:
        self.concrete_dir = f"{patches_dir}/concrete/{trajectory}"
        self.grass_dir = f"{patches_dir}/grass/{trajectory}"

        self.concrete_ids = sorted(os.listdir(self.concrete_dir))
        self.grass_ids = sorted(os.listdir(self.grass_dir))

    def _read_img(self, filename: str) -> torch.Tensor:
        img = torchvision.io.read_image(filename, torchvision.io.ImageReadMode.RGB)
        img: torch.Tensor = F.convert_image_dtype(img, dtype=torch.float32)
        return img

    def __getitem__(
        self, index: int
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], List[Any]]:
        if index < len(self.concrete_ids):
            concrete_id = self.concrete_ids[index]
            concrete_views = sorted(os.listdir(f"{self.concrete_dir}/{concrete_id}"))
            anchor, similar = np.random.choice(concrete_views, size=2, replace=True)

            grass_id = np.random.choice(self.grass_ids)
            grass_views = sorted(os.listdir(f"{self.grass_dir}/{grass_id}"))
            neg = np.random.choice(grass_views)

            anchor_img = self._read_img(f"{self.concrete_dir}/{concrete_id}/{anchor}")
            similar_img = self._read_img(f"{self.concrete_dir}/{concrete_id}/{similar}")
            neg_img = self._read_img(f"{self.grass_dir}/{grass_id}/{neg}")
        else:
            index -= len(self.concrete_ids)
            grass_id = self.grass_ids[index]
            grass_views = sorted(os.listdir(f"{self.grass_dir}/{grass_id}"))
            anchor, similar = np.random.choice(grass_views, size=2, replace=True)

            concrete_id = np.random.choice(self.concrete_ids)
            concrete_views = sorted(os.listdir(f"{self.concrete_dir}/{concrete_id}"))
            neg = np.random.choice(concrete_views)

            anchor_img = self._read_img(f"{self.grass_dir}/{grass_id}/{anchor}")
            similar_img = self._read_img(f"{self.grass_dir}/{grass_id}/{similar}")
            neg_img = self._read_img(f"{self.concrete_dir}/{concrete_id}/{neg}")

        # TODO(eyang): why return the list?
        return (anchor_img, similar_img, neg_img), []

    def __len__(self) -> int:
        return len(self.concrete_ids) + len(self.grass_ids)
