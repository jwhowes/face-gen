import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm

from .util import FiLM2d, LayerNorm2d, GRN, DiagonalGaussian, SinusoidalPosEmb

from ..config import Config, VAEConfig
from ..data import FaceDataset


class Block(nn.Module):
    def __init__(self, d_model, hidden_size=None, norm_eps=1e-6):
        super(Block, self).__init__()
        if hidden_size is None:
            hidden_size = 4 * d_model

        self.module = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=7, padding=3, groups=d_model),
            LayerNorm2d(d_model, eps=norm_eps),
            nn.Conv2d(d_model, hidden_size, kernel_size=1),
            nn.GELU(),
            GRN(hidden_size, eps=norm_eps),
            nn.Conv2d(hidden_size, d_model, kernel_size=1)
        )

    def forward(self, x):
        return x + self.module(x)


class FiLMBlock(nn.Module):
    def __init__(self, d_model, d_t, hidden_size=None, norm_eps=1e-6):
        super(FiLMBlock, self).__init__()
        if hidden_size is None:
            hidden_size = 4 * d_model

        self.conv = nn.Conv2d(d_model, d_model, kernel_size=7, padding=3, groups=d_model)
        self.norm = FiLM2d(d_model, d_t, eps=norm_eps)

        self.ffn = nn.Sequential(
            nn.Conv2d(d_model, hidden_size, kernel_size=1),
            nn.GELU(),
            GRN(hidden_size, eps=norm_eps),
            nn.Conv2d(hidden_size, d_model, kernel_size=1)
        )

    def forward(self, x, t):
        return x + self.ffn(self.norm(
            self.conv(x), t
        ))


class FlowModel(nn.Module):
    def __init__(
            self, image_channels, d_t=384, dims=(96, 192, 384, 768), depths=(2, 2, 5, 3), patch_size=4,
            sigma_min=1e-4
    ):
        super(FlowModel, self).__init__()
        self.stem = nn.Conv2d(image_channels, dims[0], kernel_size=patch_size, stride=patch_size)

        self.down_path = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.down_path.append(nn.ModuleList([
                FiLMBlock(dims[i], d_t) for _ in range(depths[i])
            ]))
            self.down_samples.append(nn.Conv2d(
                dims[i], dims[i + 1], kernel_size=4, stride=2, padding=1
            ))

        self.mid_blocks = nn.ModuleList([
            FiLMBlock(dims[-1], d_t) for _ in range(depths[-1])
        ])

        self.up_path = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        self.up_combines = nn.ModuleList()
        for i in range(len(dims) - 2, -1, -1):
            self.up_samples.append(nn.ConvTranspose2d(
                dims[i + 1], dims[i], kernel_size=4, stride=2, padding=1
            ))
            self.up_combines.append(nn.Conv2d(
                2 * dims[i], dims[i], kernel_size=3, padding=1
            ))
            self.up_path.append(nn.ModuleList([
                FiLMBlock(dims[i], d_t) for _ in range(depths[i])
            ]))

        self.head = nn.ConvTranspose2d(dims[0], image_channels, kernel_size=patch_size, stride=patch_size)

        self.t_model = nn.Sequential(
            SinusoidalPosEmb(d_t),
            nn.Linear(d_t, 4 * d_t),
            nn.GELU(),
            nn.Linear(4 * d_t, d_t)
        )

        self.image_channels = image_channels
        self.sigma_min = sigma_min
        self.sigma_offset = 1 - sigma_min

        self.register_buffer(
            "mean",
            torch.tensor(FaceDataset.mean).view(1, -1, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor(FaceDataset.std).view(1, -1, 1, 1)
        )

    def pred_flow(self, x_t, t):
        t_emb = self.t_model(t)

        x_t = self.stem(x_t)

        acts = []
        for blocks, sample in zip(self.down_path, self.down_samples):
            for block in blocks:
                x_t = block(x_t, t_emb)
            acts.append(x_t)
            x_t = sample(x_t)

        for block in self.mid_blocks:
            x_t = block(x_t, t_emb)

        for blocks, sample, combine, act in zip(self.up_path, self.up_samples, self.up_combines, acts[::-1]):
            x_t = combine(torch.concatenate((
                sample(x_t),
                act
            ), dim=1))

            for block in blocks:
                x_t = block(x_t, t_emb)

        return self.head(x_t)

    @torch.inference_mode()
    def sample(self, num_samples=1, image_size=192, num_steps=200, step="euler"):
        dt = 1 / num_steps

        x_t = torch.randn(
            num_samples, self.image_channels, image_size, image_size
        )
        ts = torch.linspace(0, 1, num_steps).unsqueeze(1).expand(-1, num_samples)

        for i in tqdm(range(num_steps)):
            pred_flow = self.pred_flow(x_t, ts[i])

            if step == "euler":
                x_t = x_t + dt * pred_flow
            elif step == "midpoint":
                x_t = x_t + dt * self.pred_flow(x_t + 0.5 * dt * pred_flow, ts[i] + 0.5 * dt)
            elif step == "heun":
                next_x = x_t + dt * pred_flow
                if i == num_steps - 1:
                    x_t = next_x
                else:
                    x_t = x_t + dt * 0.5 * (pred_flow + self.pred_flow(next_x, ts[i + 1]))
            elif step == "stochastic":
                x_t = x_t + (1 - ts[i]).view(-1, 1, 1, 1) * pred_flow
                if i < num_steps - 1:
                    next_t = ts[i + 1].view(-1, 1, 1, 1)
                    z_0 = torch.randn_like(x_t)
                    x_t = (1 - self.sigma_offset * next_t) * z_0 + next_t * x_t
            else:
                raise NotImplementedError

        return (x_t * self.std + self.mean).clamp(0.0, 1.0)

    def forward(self, x_1):
        B = x_1.shape[0]

        t = torch.rand(B, device=x_1.device).view(B, 1, 1, 1)
        x_0 = torch.randn_like(x_1)
        x_t = (1 - self.sigma_offset * t) * x_0 + t * x_1

        pred = self.pred_flow(x_t, t)

        loss = F.mse_loss(pred, x_1 - self.sigma_offset * x_0)
        return {
            "loss": loss,
            "metrics": (loss.item(),)
        }


class VAEEncoder(nn.Module):
    def __init__(self, image_channels, d_latent=4, dims=(96, 192, 384), depths=(2, 2, 5)):
        super(VAEEncoder, self).__init__()

        layers = [nn.Conv2d(image_channels, dims[0], kernel_size=5, padding=2)]

        for i in range(len(dims) - 1):
            layers += [
                Block(dims[i]) for _ in range(depths[i])
            ]
            layers += [
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=4, stride=2, padding=1)
            ]

        layers += [
            Block(dims[-1]) for _ in range(depths[-1])
        ]
        layers += [
            nn.Conv2d(dims[-1], 2 * d_latent, kernel_size=5, padding=2)
        ]

        self.module = nn.Sequential(*layers)

    def forward(self, x):
        mean, log_var = self.module(x).chunk(2, 1)

        return DiagonalGaussian(mean=mean, log_var=log_var)


class VAEDecoder(nn.Module):
    def __init__(self, image_channels, d_latent=4, dims=(96, 192, 384), depths=(2, 2, 5)):
        super(VAEDecoder, self).__init__()
        layers = [nn.Conv2d(d_latent, dims[-1], kernel_size=5, padding=2)]

        layers += [
            Block(dims[-1]) for _ in range(depths[-1])
        ]

        for i in range(len(dims) - 2, -1, -1):
            layers += [
                nn.ConvTranspose2d(dims[i + 1], dims[i], kernel_size=4, stride=2, padding=1)
            ]
            layers += [
                Block(dims[i]) for _ in range(depths[i])
            ]

        layers += [
            nn.Conv2d(dims[0], image_channels, kernel_size=5, padding=2)
        ]

        self.module = nn.Sequential(*layers)

    def forward(self, z):
        return self.module(z)


class VAE(nn.Module):
    def __init__(self, image_channels, d_latent=4, dims=(96, 192, 384), depths=(2, 2, 5), kl_weight=1e-4):
        super(VAE, self).__init__()
        self.kl_weight = kl_weight

        self.encoder = VAEEncoder(image_channels, d_latent, dims, depths)
        self.decoder = VAEDecoder(image_channels, d_latent, dims, depths)

    def forward(self, x):
        dist = self.encoder(x)
        kl_loss = dist.kl.mean()

        z = dist.sample()
        pred = self.decoder(z)
        recon_loss = F.mse_loss(pred, x)

        return {
            "loss": recon_loss + self.kl_weight * kl_loss,
            "metrics": (
                recon_loss.item(),
                kl_loss.item()
            )
        }
