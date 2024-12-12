import torch
import torch.nn.functional as F

from torch import nn
from typing import Tuple

from .ConvNeXt import TimeDependentUNet
from .util import SinusoidalPosEmb


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

        t = torch.rand(B, device=x.device)
        x_0 = torch.randn_like(x)
        x_t = (1 - self.sigma_offset * t) * x_0 + x * t

        pred = self.pred_flow(x_t, t)

        return F.mse_loss(pred, x - x_0 * self.sigma_offset)
