import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from abc import ABC, abstractmethod
from tqdm import tqdm

from ..data import FaceDataset


class FlowModel(ABC, nn.Module):
    def __init__(self, image_channels, d_t=384, sigma_min=1e-4):
        super(FlowModel, self).__init__()
        self._t_model = nn.Sequential(
            SinusoidalPosEmb(d_t),
            nn.Linear(d_t, 4 * d_t),
            nn.GELU(),
            nn.Linear(4 * d_t, d_t)
        )

        self.image_channels = image_channels
        self.sigma_min = sigma_min
        self.sigma_offset = 1 - sigma_min

        self.log_t_mult = nn.Parameter(
            torch.tensor(np.log(1.0))
        )

        self.register_buffer(
            "mean",
            torch.tensor(FaceDataset.mean).view(1, -1, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor(FaceDataset.std).view(1, -1, 1, 1)
        )

    def t_model(self, t):
        return self._t_model(t * self.log_t_mult.exp().clamp(max=1000.0))

    @abstractmethod
    def pred_flow(self, x_t, t):
        ...

    @torch.inference_mode()
    def sample(self, num_samples=1, image_size=192, num_steps=200, step="euler"):
        dt = 1 / num_steps

        x_t = torch.randn(num_samples, self.image_channels, image_size, image_size)
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
                    x_0 = torch.randn_like(x_t)
                    x_t = (1 - self.sigma_offset * next_t) * x_0 + next_t * x_t
            else:
                raise NotImplementedError

        return (x_t * self.std + self.mean).clamp(0.0, 1.0)

    def forward(self, x_1):
        B = x_1.shape[0]

        t = torch.rand(B, device=x_1.device).view(B, 1, 1, 1)
        x_0 = torch.randn_like(x_1)
        x_t = (1 - self.sigma_offset * t) * x_0 + t * x_1

        pred = self.pred_flow(x_t, t)

        return F.mse_loss(pred, x_1 - self.sigma_offset * x_0)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, d_model, base=1e5):
        super(SinusoidalPosEmb, self).__init__()
        assert d_model % 2 == 0
        self.register_buffer(
            "theta",
            1.0 / (base ** (2 * torch.arange(d_model // 2) / d_model)),
            persistent=False
        )

    def forward(self, x):
        x = x.view(-1, 1) * self.theta

        return torch.stack((
            x.cos(),
            x.sin()
        ), dim=1).flatten(-2)


class FiLM2d(nn.Module):
    def __init__(self, d_model, d_t, *norm_args, **norm_kwargs):
        super(FiLM2d, self).__init__()
        self.norm = nn.LayerNorm(d_model, *norm_args, elementwise_affine=False, bias=False, **norm_kwargs)
        self.gamma = nn.Linear(d_t, d_model)
        self.beta = nn.Linear(d_t, d_model)

    def forward(self, x, t):
        B = x.shape[0]
        g = self.gamma(t).view(B, -1, 1, 1)
        b = self.beta(t).view(B, -1, 1, 1)
        return g * self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) + b


class RMSFiLM(nn.Module):
    def __init__(self, d_model, d_t, eps=1e-6):
        super(RMSFiLM, self).__init__()
        self.eps = eps

        self.gamma = nn.Linear(d_t, d_model)
        self.beta = nn.Linear(d_t, d_model)

    def forward(self, x, t):
        B = x.shape[0]
        g = self.gamma(t).view(B, -1, 1, 1)
        b = self.beta(t).view(B, -1, 1, 1)

        return g * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) + b


class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_size=None):
        super(SwiGLU, self).__init__()
        if hidden_size is None:
            hidden_size = 4 * d_model

        self.gate = nn.Linear(d_model, hidden_size, bias=False)
        self.hidden = nn.Linear(d_model, hidden_size, bias=False)

        self.out = nn.Linear(hidden_size, d_model)

    def forward(self, x):
        return self.out(
            F.silu(self.gate(x)) * self.hidden(x)
        )


class GRN(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(GRN, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.empty(1, d_model, 1, 1).normal_(std=0.02))
        self.beta = nn.Parameter(torch.empty(1, d_model, 1, 1).normal_(std=0.02))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + self.eps)

        return self.gamma * (x * Nx) + self.beta + x
