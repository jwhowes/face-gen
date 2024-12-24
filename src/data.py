from torch.utils.data import Dataset
from datasets import load_dataset
from torchvision import transforms

from . import accelerator


class FaceDataset(Dataset):
    mean = (
        0.4141770303249359,
        0.34720903635025024,
        0.3103456199169159
    )
    std = (
        0.3434087038040161,
        0.3099803924560547,
        0.300066202878952
    )

    @accelerator.main_process_first()
    def __init__(self, image_size=192):
        assert image_size <= 218

        self.ds = load_dataset("nielsr/CelebA-faces", split="train")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(image_size),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.transform(self.ds[idx]["image"])
