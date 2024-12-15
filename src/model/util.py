import torch

from torch import nn
from dataclasses import dataclass


@dataclass
class DiagonalGaussian:
    mean: torch.FloatTensor
    log_var: torch.FloatTensor

    def sample(self):
        return torch.randn_like(self.log_var) * (0.5 * self.log_var).exp() + self.mean

    @property
    def kl(self):
        return 0.5 * (
            self.mean.pow(2) + self.log_var.exp() - 1.0 - self.log_var
        ).mean((1, 2, 3))


class FiLM(nn.Module):
    def __init__(self, d_model: int, d_t: int, *args, **kwargs):
        super(FiLM, self).__init__()
        self.norm = nn.LayerNorm(d_model, *args, elementwise_affine=False, **kwargs)

        self.gamma = nn.Linear(d_t, d_model)
        self.beta = nn.Linear(d_t, d_model)

    def forward(self, x, t):
        g = self.gamma(t).unsqueeze(1)
        b = self.beta(t).unsqueeze(1)

        return g * self.norm(x) + b


class FiLM2d(nn.Module):
    def __init__(self, d_model: int, d_t: int, *args, **kwargs):
        super(FiLM2d, self).__init__()
        self.norm = nn.LayerNorm(d_model, *args, elementwise_affine=False, **kwargs)

        self.gamma = nn.Linear(d_t, d_model)
        self.beta = nn.Linear(d_t, d_model)

    def forward(self, x: torch.FloatTensor, t: torch.FloatTensor) -> torch.FloatTensor:
        B = x.shape[0]
        g = self.gamma(t).view(B, -1, 1, 1)
        b = self.beta(t).view(B, -1, 1, 1)

        return g * self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) + b


class SinusoidalPosEmb(nn.Module):
    def __init__(self, d_model: int, base: float = 1e5):
        super(SinusoidalPosEmb, self).__init__()
        assert d_model % 2 == 0
        self.register_buffer(
            "theta",
            1.0 / (base ** (2 * torch.arange(d_model // 2) / d_model)),
            persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.FloatTensor:
        B = x.shape[0]
        x = x.view(-1, 1).float() * self.theta

        return torch.stack((
            x.cos(),
            x.sin()
        ), dim=-1).view(B, -1)
