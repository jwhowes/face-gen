import torch

from torch.utils.data import Dataset
from datasets import load_dataset
from torchvision import transforms
from typing import Tuple, Union


class FaceDataset(Dataset):
    def __init__(self, image_size: Tuple[int, int] = (160, 128)):
        self.ds = load_dataset("nielsr/CelebA-faces", split="train")
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                image_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True
            ),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5)
            )
        ])

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx) -> torch.FloatTensor:
        return self.transform(self.ds[idx]["image"])
