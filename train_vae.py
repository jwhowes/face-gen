from argparse import ArgumentParser

from src.config import Config, VAEConfig
from src.model import VAE
from src.train import train


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config", type=str)

    args = parser.parse_args()

    config = Config(args.config, model_class=VAEConfig, save=True, metrics=("recon loss", "kl loss"))

    model = VAE(
        image_channels=3,
        d_latent=config.model.d_latent,
        dims=config.model.dims,
        depths=config.model.depths,
        kl_weight=config.model.kl_weight
    )

    train(model, config)
