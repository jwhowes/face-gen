from argparse import ArgumentParser

from src.config import Config
from src.model import FlowModel
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

    train(model, config)
