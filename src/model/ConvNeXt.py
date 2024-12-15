import torch

from torch import nn
from typing import Optional, Tuple

from .util import FiLM2d


class Block(nn.Module):
    def __init__(self, d_model: int, hidden_size: Optional[int] = None, norm_eps: float = 1e-6):
        super(Block, self).__init__()
        if hidden_size is None:
            hidden_size = 4 * d_model

        self.dwconv = nn.Conv2d(d_model, d_model, kernel_size=7, padding=3, groups=d_model)

        self.norm = nn.LayerNorm(d_model, eps=norm_eps)

        self.ffn = nn.Sequential(
            nn.Conv2d(d_model, hidden_size, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_size, d_model, kernel_size=1)
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return residual + self.ffn(x)


class TimeDependentBlock(nn.Module):
    def __init__(self, d_model: int, d_t: int, hidden_size: Optional[int] = None, norm_eps: float = 1e-6):
        super(TimeDependentBlock, self).__init__()
        if hidden_size is None:
            hidden_size = 4 * d_model

        self.dwconv = nn.Conv2d(d_model, d_model, kernel_size=7, padding=3, groups=d_model)

        self.norm = FiLM2d(d_model, d_t, eps=norm_eps)

        self.ffn = nn.Sequential(
            nn.Conv2d(d_model, hidden_size, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_size, d_model, kernel_size=1)
        )

    def forward(self, x: torch.FloatTensor, t: torch.FloatTensor) -> torch.FloatTensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x, t)

        return residual + self.ffn(x)


class TimeDependentUNet(nn.Module):
    def __init__(
            self, in_channels: int, dims: Tuple[int] = (64, 128, 256), depths: Tuple[int] = (2, 2, 3), d_t: int = 256
    ):
        super(TimeDependentUNet, self).__init__()
        self.stem = nn.Conv2d(in_channels, dims[0], kernel_size=7, padding=3)

        self.down_path = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.down_path.append(nn.ModuleList([
                TimeDependentBlock(dims[i], d_t) for _ in range(depths[i])
            ]))
            self.down_samples.append(nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2))

        self.mid_blocks = nn.ModuleList([
            TimeDependentBlock(dims[-1], d_t) for _ in range(depths[-1])
        ])

        self.up_path = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        self.up_combines = nn.ModuleList()
        for i in range(len(dims) - 2, -1, -1):
            self.up_samples.append(nn.ConvTranspose2d(dims[i + 1], dims[i], kernel_size=2, stride=2))
            self.up_combines.append(nn.Conv2d(2 * dims[i], dims[i], kernel_size=1))
            self.up_path.append(nn.ModuleList([
                TimeDependentBlock(dims[i], d_t) for _ in range(depths[i])
            ]))

        self.head = nn.Conv2d(dims[0], in_channels, kernel_size=1)

    def forward(self, x: torch.FloatTensor, t: torch.FloatTensor):
        x = self.stem(x)

        down_acts = []
        for down_blocks, down_sample in zip(self.down_path, self.down_samples):
            for block in down_blocks:
                x = block(x, t)

            down_acts.append(x)
            x = down_sample(x)

        for block in self.mid_blocks:
            x = block(x, t)

        for up_blocks, up_sample, up_combine, act in zip(
                self.up_path, self.up_samples, self.up_combines, down_acts[::-1]
        ):
            x = up_combine(torch.concatenate((
                up_sample(x),
                act
            ), dim=1))

            for block in up_blocks:
                x = block(x, t)

        return self.head(x)


class Encoder(nn.Module):
    def __init__(
            self, in_channels: int, dims: Tuple[int] = (64, 128, 256), depths: Tuple[int] = (2, 2, 1)
    ):
        super(Encoder, self).__init__()
        self.stem = nn.Conv2d(in_channels, dims[0], kernel_size=7, padding=3)

        self.down_path = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.down_path.append(nn.Sequential(*[
                Block(dims[i]) for _ in range(depths[i])
            ]))
            self.down_samples.append(nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2))

        self.mid_blocks = nn.Sequential(*[
            Block(dims[-1]) for _ in range(depths[-1])
        ])

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.stem(x)

        for blocks, down_sample in zip(self.down_path, self.down_samples):
            x = blocks(x)
            x = down_sample(x)

        return self.mid_blocks(x)
