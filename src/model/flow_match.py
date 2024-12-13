import torch
import torch.nn.functional as F

from torch import nn
from typing import Tuple
from tqdm import tqdm

from .util import SinusoidalPosEmb
from .ConvNeXt import TimeDependentUNet
from .vae import VAEDecoder


class FlowMatchModel(nn.Module):
    def __init__(
            self, in_channels: int, dims: Tuple[int] = (64, 128, 256), depths: Tuple[int] = (2, 2, 3), d_t: int = 256,
            sigma_min: float = 1e-8
    ):
        super(FlowMatchModel, self).__init__()
        self.sigma_min = sigma_min
        self.sigma_offset = 1 - sigma_min

        self.t_model = nn.Sequential(
            SinusoidalPosEmb(d_t),
            nn.Linear(d_t, 4 * d_t),
            nn.GELU(),
            nn.Linear(4 * d_t, d_t)
        )

        self.unet = TimeDependentUNet(in_channels, dims, depths, d_t)

    def pred_flow(self, x: torch.FloatTensor, t: torch.FloatTensor):
        t_emb = self.t_model(t)

        return self.unet(x, t_emb)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        B = x.shape[0]

        t = torch.rand(B, device=x.device).view(B, 1, 1, 1)
        x_0 = torch.randn_like(x)
        x_t = (1 - self.sigma_offset * t) * x_0 + x * t

        pred = self.pred_flow(x_t, t)

        return F.mse_loss(pred, x - x_0 * self.sigma_offset)


class FlowMatchSampler(nn.Module):
    def __init__(
            self, decoder: VAEDecoder, flow_match: FlowMatchModel,
            image_size: Tuple[int, int] = (160, 128),
            latent_scale: float = 1.7232484817504883,
            image_mean: Tuple[float, float, float] = (
                    0.5043166875839233,
                    0.4233281910419464,
                    0.3812117874622345
                ),
            image_std: Tuple[float, float, float] = (
                    0.30935946106910706,
                    0.28855881094932556,
                    0.28814613819122314
                )
    ):
        super(FlowMatchSampler, self).__init__()
        self.decoder = decoder
        self.decoder.eval()
        self.decoder.requires_grad_(False)

        self.flow_match = flow_match
        self.flow_match.eval()
        self.flow_match.requires_grad_(False)

        self.latent_scale = latent_scale
        self.image_size = image_size

        self.register_buffer(
            "image_mean",
            torch.tensor(image_mean).view(1, -1, 1, 1)
        )
        self.register_buffer(
            "image_std",
            torch.tensor(image_std).view(1, -1, 1, 1)
        )

    @torch.inference_mode()
    def forward(self, num_samples: int = 1, num_steps: int = 50, step: str = "euler"):
        z = torch.randn(num_samples, self.image_mean.shape[1], *self.image_size, device=self.image_mean.device)
        ts = torch.linspace(0, 1, num_steps, device=self.image_mean.device).expand(num_samples)

        for i in tqdm(range(num_steps)):
            pred_flow = self.flow_match.pred_flow(z, ts[i])

            if step == "euler":
                z = z + (1 / num_steps) * pred_flow
            else:
                raise NotImplementedError

        image = self.decoder(z / self.latent_scale)

        return (
            image * self.image_std + self.image_mean
        ).clamp(0.0, 1.0)
