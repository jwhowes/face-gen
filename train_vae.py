import torch
import torch.nn.functional as F

from argparse import ArgumentParser
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup

from src.model.ConvNeXt import VAEEncoder, VAEDecoder, Discriminator
from src.data import FaceDataset
from src.config.vae import VAEConfig


accelerator = Accelerator()


def save_model(encoder, decoder, discriminator):
    save_model.num += 1
    torch.save(
        {
            "encoder": accelerator.get_state_dict(encoder),
            "decoder": accelerator.get_state_dict(decoder),
            "discriminator": accelerator.get_state_dict(discriminator)
        },
        f"experiments/{config.exp_name}/checkpoint_{save_model.num:02}.pt"
    )


save_model.num = 0


def train(
        encoder: VAEEncoder, decoder: VAEDecoder, discriminator: Discriminator,
        dataloader: DataLoader,
        config: VAEConfig
):
    vae_opt = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=config.lr
    )
    vae_lr_scheduler = get_cosine_schedule_with_warmup(
        vae_opt,
        num_warmup_steps=0,
        num_training_steps=config.num_epochs * len(dataloader)
    )

    disc_opt = torch.optim.Adam(
        discriminator.parameters(), lr=config.lr
    )
    disc_lr_scheduler = get_cosine_schedule_with_warmup(
        disc_opt,
        num_warmup_steps=0,
        num_training_steps=config.num_epochs * len(dataloader)
    )

    accelerator = Accelerator()
    encoder, decoder, discriminator, dataloader, vae_opt, disc_opt, vae_lr_scheduler, disc_lr_scheduler = accelerator.prepare(
        encoder, decoder, discriminator, dataloader, vae_opt, disc_opt, vae_lr_scheduler, disc_lr_scheduler
    )

    encoder.train()
    decoder.train()
    discriminator.train()
    for epoch in range(config.num_epochs):
        print(f"EPOCH {epoch + 1} / {config.num_epochs}")

        for i, image in enumerate(dataloader):
            vae_opt.zero_grad()

            with accelerator.autocast():
                dist = encoder(image)
                z = dist.sample()
                pred = decoder(z)

                pred_fake = discriminator(pred)

                recon_loss = F.mse_loss(pred, image)
                kl_loss = dist.kl.mean()
                adv_loss = F.binary_cross_entropy_with_logits(
                    pred_fake, torch.ones_like(pred_fake)
                )

                vae_loss = recon_loss + config.kl_weight * kl_loss + config.adv_weight * adv_loss

            accelerator.backward(vae_loss)
            if config.clip_grad:
                accelerator.clip_grad_norm_(
                    list(encoder.parameters()) + list(decoder.parameters()),
                    config.clip_grad
                )

            vae_opt.step()
            vae_lr_scheduler.step()

            disc_opt.zero_grad()

            with accelerator.autocast():
                pred_fake = discriminator(pred.detach())
                pred_real = discriminator(image)

                disc_loss = (
                    F.binary_cross_entropy_with_logits(
                        pred_fake, torch.zeros_like(pred_fake)
                    ) +
                    F.binary_cross_entropy_with_logits(
                        pred_real, torch.ones_like(pred_real)
                    )
                )

            accelerator.backward(disc_loss)
            if config.clip_grad:
                accelerator.clip_grad_norm_(discriminator.parameters(), config.clip_grad)

            disc_opt.step()
            disc_lr_scheduler.step()

            if i % config.log_interval == 0:
                print(f"{i} / {len(dataloader)} iters.\t"
                      f"Recon Loss: {recon_loss.item():.4f}\t"
                      f"KL Loss: {kl_loss.item():.4f}\t"
                      f"Adv Loss: {adv_loss.item():4f}")

            if i > 0 and i % config.save_interval == 0:
                save_model(encoder, decoder, discriminator)

        save_model(encoder, decoder, discriminator)


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
    discriminator = Discriminator(
        in_channels=3,
        dims=config.discriminator.dims,
        depths=config.discriminator.depths,
        patch_size=config.discriminator.patch_size
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.dataset.batch_size,
        shuffle=True,
        pin_memory=True
    )

    config.save()

    train(encoder, decoder, discriminator, dataloader, config)
