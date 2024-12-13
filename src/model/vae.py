import torch

from torch import nn
from typing import Tuple

from .ConvNeXt import Encoder
from .util import DiagonalGaussian


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
