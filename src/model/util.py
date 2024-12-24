import torch

from torch import nn


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
