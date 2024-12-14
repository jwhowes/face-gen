import os
import torch

from argparse import ArgumentParser
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup

from src.config import VAEConfig, FlowMatchConfig
from src.model import FlowMatchModel, VAEEncoder
from src.data import FaceDataset


accelerator = Accelerator()


def save_model(model, config):
    save_model.num += 1
    torch.save(
        accelerator.get_state_dict(model),
        f"experiments/{config.exp_name}/checkpoint_{save_model.num:02}.pt"
    )


save_model.num = 0


def train(
        model: FlowMatchModel,
        encoder: VAEEncoder,
        dataloader: DataLoader,
        config: FlowMatchConfig
):
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=0,
        num_training_steps=config.num_epochs * len(dataloader)
    )

    model, encoder, dataloader = accelerator.prepare(
        model, encoder, dataloader
    )

    model.train()
    for epoch in range(config.num_epochs):
        print(f"EPOCH {epoch + 1} / {config.num_epochs}")

        for i, image in enumerate(dataloader):
            opt.zero_grad()

            with accelerator.autocast():
                dist = encoder(image)
                z = dist.sample() * config.latent_model.latent_scale

                loss = model(z)

            accelerator.backward(loss)
            if config.clip_grad:
                accelerator.clip_grad_norm_(model.parameters(), config.clip_grad)

            opt.step()
            lr_scheduler.step()

            if i % config.log_interval == 0:
                print(f"{i} / {len(dataloader)} iters.\tLoss: {loss.item():.6f}")

            if i > 0 and i % config.save_interval == 0:
                save_model(model, config)

        save_model(model, config)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config", type=str)

    args = parser.parse_args()

    config = FlowMatchConfig(args.config)
    vae_config = VAEConfig(os.path.join(config.latent_model.dir, "config.yaml"))

    encoder = VAEEncoder(
        in_channels=3,
        d_latent=vae_config.model.d_latent,
        dims=vae_config.model.dims,
        depths=vae_config.model.depths
    )
    ckpt = torch.load(
        os.path.join(config.latent_model.dir, f"checkpoint_{config.latent_model.checkpoint:02}.pt"),
        weights_only=True, map_location="cpu"
    )
    encoder.load_state_dict(ckpt["encoder"])
    del ckpt

    encoder.eval()
    encoder.requires_grad_(False)

    model = FlowMatchModel(
        in_channels=vae_config.model.d_latent,
        dims=config.model.dims,
        depths=config.model.depths,
        d_t=config.model.d_t,
        sigma_min=config.model.sigma_min
    )

    dataset = FaceDataset(config.dataset.image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=config.dataset.batch_size,
        shuffle=True
    )

    config.save()

    train(model, encoder, dataloader, config)
