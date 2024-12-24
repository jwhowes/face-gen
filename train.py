from argparse import ArgumentParser
from torch.utils.data import DataLoader

from src.config import Config
from src.model import UNet, ViT
from src.data import FaceDataset
from src.train import train


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config", type=str)

    args = parser.parse_args()

    config = Config(args.config, save=True)

    if config.arch == "unet":
        model = UNet(
            image_channels=3,
            d_t=config.model.d_t,
            dims=config.model.dims,
            depths=config.model.depths,
            sigma_min=config.model.sigma_min
        )
    elif config.arch == "vit":
        model = ViT(
            image_channels=3,
            image_size=config.dataset.image_size,
            patch_size=config.model.patch_size,
            d_t=config.model.d_t,
            n_layers=config.model.n_layers,
            n_heads=config.model.n_heads,
            sigma_min=config.model.sigma_min
        )
    else:
        raise NotImplementedError

    dataset = FaceDataset(image_size=config.dataset.image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=config.dataset.batch_size,
        shuffle=True,
        pin_memory=True
    )

    train(model, dataloader, config)
