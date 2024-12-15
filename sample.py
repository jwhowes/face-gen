import os
import torch
import matplotlib.pyplot as plt

from src.model.vae import VAEDecoder
from src.model.flow_match import FlowMatchModel, FlowMatchSampler
from src.config import FlowMatchConfig, VAEConfig


if __name__ == "__main__":
    config = FlowMatchConfig("configs/flow_match.yaml")
    vae_config = VAEConfig(os.path.join(config.latent_model.dir, "config.yaml"))

    decoder = VAEDecoder(
        in_channels=3,
        d_latent=vae_config.model.d_latent,
        dims=vae_config.model.dims,
        depths=vae_config.model.depths
    )
    vae_ckpt = torch.load(
        os.path.join(config.latent_model.dir, f"checkpoint_{config.latent_model.checkpoint:02}.pt"),
        weights_only=True, map_location="cpu"
    )
    decoder.load_state_dict(vae_ckpt["decoder"])
    del vae_ckpt

    flow_match = FlowMatchModel(
        in_channels=vae_config.model.d_latent,
        dims=config.model.dims,
        depths=config.model.depths,
        d_t=config.model.d_t,
        sigma_min=config.model.sigma_min
    )
    ckpt = torch.load(
        "experiments/flow_match/checkpoint_02.pt", weights_only=True, map_location="cpu"
    )
    flow_match.load_state_dict(ckpt)
    del ckpt

    sampler = FlowMatchSampler(
        decoder, flow_match, latent_scale=config.latent_model.latent_scale,
        latent_channels=vae_config.model.d_latent,
        latent_size=(
            config.dataset.image_size[0] // (2 ** (len(vae_config.model.dims) - 1)),
            config.dataset.image_size[1] // (2 ** (len(vae_config.model.dims) - 1))
        )
    )

    pred = sampler(num_steps=200, step="euler")
    plt.imsave("pred.png", pred[0].permute(1, 2, 0).numpy())
