import torch
import torch.nn.functional as F

from argparse import ArgumentParser
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup

from src.model import VAEEncoder, VAEDecoder
from src.data import FaceDataset
from src.config import VAEConfig


accelerator = Accelerator()


def save_model(encoder, decoder, config):
    save_model.num += 1
    torch.save(
        {
            "encoder": accelerator.get_state_dict(encoder),
            "decoder": accelerator.get_state_dict(decoder)
        },
        f"experiments/{config.exp_name}/checkpoint_{save_model.num:02}.pt"
    )


save_model.num = 0


def train(
        encoder: VAEEncoder, decoder: VAEDecoder,
        dataloader: DataLoader,
        config: VAEConfig
):
    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=config.lr
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=0,
        num_training_steps=config.num_epochs * len(dataloader)
    )

    encoder, decoder, dataloader, opt, lr_scheduler = accelerator.prepare(
        encoder, decoder, dataloader, opt, lr_scheduler
    )

    encoder.train()
    decoder.train()
    for epoch in range(config.num_epochs):
        print(f"EPOCH {epoch + 1} / {config.num_epochs}")

        for i, image in enumerate(dataloader):
            opt.zero_grad()

            with accelerator.autocast():
                dist = encoder(image)
                z = dist.sample()
                pred = decoder(z)

                recon_loss = F.mse_loss(pred, image)
                kl_loss = dist.kl.mean()

                vae_loss = recon_loss + config.kl_weight * kl_loss

            accelerator.backward(vae_loss)
            if config.clip_grad:
                accelerator.clip_grad_norm_(
                    list(encoder.parameters()) + list(decoder.parameters()),
                    config.clip_grad
                )

            opt.step()
            lr_scheduler.step()

            if i % config.log_interval == 0:
                print(f"{i} / {len(dataloader)} iters.\t"
                      f"Recon Loss: {recon_loss.item():.4f}\t"
                      f"KL Loss: {kl_loss.item():.4f}\t")

            if i > 0 and i % config.save_interval == 0:
                save_model(encoder, decoder, config)

        save_model(encoder, decoder, config)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config", type=str)

    args = parser.parse_args()

    config = VAEConfig(args.config)

    dataset = FaceDataset(config.dataset.image_size)

    encoder = VAEEncoder(
        in_channels=3,
        d_latent=config.model.d_latent,
        dims=config.model.dims,
        depths=config.model.depths
    )
    decoder = VAEDecoder(
        in_channels=3,
        d_latent=config.model.d_latent,
        dims=config.model.dims,
        depths=config.model.depths
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.dataset.batch_size,
        shuffle=True,
        pin_memory=True
    )

    config.save()

    train(encoder, decoder, dataloader, config)
