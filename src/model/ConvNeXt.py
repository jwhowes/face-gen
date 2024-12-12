import torch

from torch import nn
from typing import Optional, Tuple
from einops import rearrange

from .util import FiLM2d, DiagonalGaussian


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


class VAEEncoder(Encoder):
    def __init__(
            self, in_channels: int, d_latent: int, dims: Tuple[int] = (64, 128, 256), depths: Tuple[int] = (2, 2, 1)
    ):
        super(VAEEncoder, self).__init__(in_channels, dims, depths)
        self.head = nn.Conv2d(dims[-1], 2 * d_latent, kernel_size=1)

    def forward(self, x: torch.FloatTensor) -> DiagonalGaussian:
        x = super().forward(x)
        mean, log_var = self.head(x).chunk(2, 1)

        return DiagonalGaussian(mean, log_var)


class Discriminator(Encoder):
    def __init__(
            self, in_channels: int, dims: Tuple[int] = (64, 128, 256), depths: Tuple[int] = (2, 2, 1),
            patch_size: int = 32
    ):
        super(Discriminator, self).__init__(in_channels, dims, depths)
        self.patch_size = patch_size
        self.classifier = nn.Sequential(
            nn.Linear(dims[-1], 4 * dims[-1]),
            nn.GELU(),
            nn.Linear(4 * dims[-1], 1)
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = rearrange(x, "b c (h hp) (w wp) -> (b h w) c hp wp", hp=self.patch_size, wp=self.patch_size)
        x = super().forward(x).mean((-1, -2))

        return self.classifier(x).squeeze(-1)


class VAEDecoder(nn.Module):
    def __init__(
            self, in_channels: int, d_latent: int, dims: Tuple[int] = (64, 128, 256), depths: Tuple[int] = (2, 2, 1)
    ):
        super(VAEDecoder, self).__init__()
        self.stem = nn.Conv2d(d_latent, dims[-1], kernel_size=7, padding=3)

        self.mid_blocks = nn.Sequential(*[
            Block(dims[-1]) for _ in range(depths[-1])
        ])

        self.up_path = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for i in range(len(dims) - 2, -1, -1):
            self.up_samples.append(nn.ConvTranspose2d(dims[i + 1], dims[i], kernel_size=2, stride=2))
            self.up_path.append(nn.Sequential(*[
                Block(dims[i]) for _ in range(depths[i])
            ]))

        self.head = nn.Conv2d(dims[0], in_channels, kernel_size=1)

    def forward(self, z: torch.FloatTensor) -> torch.FloatTensor:
        z = self.stem(z)

        z = self.mid_blocks(z)

        for blocks, sample in zip(self.up_path, self.up_samples):
            z = sample(z)
            z = blocks(z)

        return self.head(z)
