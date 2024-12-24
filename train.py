from argparse import ArgumentParser
from torch.utils.data import DataLoader

from src.config import Config
from src.model import FlowModel
from src.data import FaceDataset
from src.train import train


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config", type=str)

    args = parser.parse_args()

    config = Config(args.config, save=True)

    model = FlowModel(
        image_channels=3,
        d_t=config.model.d_t,
        dims=config.model.dims,
        depths=config.model.depths,
        sigma_min=config.model.sigma_min
    )

    dataset = FaceDataset(image_size=config.dataset.image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=config.dataset.batch_size,
        shuffle=True,
        pin_memory=True
    )

    train(model, dataloader, config)
