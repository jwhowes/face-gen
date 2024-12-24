import torch
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm

from .util import FiLM2d, SinusoidalPosEmb, GRN
from ..data import FaceDataset


class Block(nn.Module):
    def __init__(self, d_model, d_t, hidden_size=None, norm_eps=1e-6):
        super(Block, self).__init__()
        if hidden_size is None:
            hidden_size = 4 * d_model

        self.conv = nn.Conv2d(d_model, d_model, kernel_size=7, padding=3, groups=d_model)
        self.norm = FiLM2d(d_model, d_t, eps=norm_eps)

        self.ffn = nn.Sequential(
            nn.Conv2d(d_model, hidden_size, kernel_size=1),
            nn.GELU(),
            GRN(4 * d_model, eps=norm_eps),
            nn.Conv2d(hidden_size, d_model, kernel_size=1)
        )

    def forward(self, x, t):
        return x + self.ffn(self.norm(
            self.conv(x), t
        ))


class FlowModel(nn.Module):
    def __init__(self, image_channels, d_t=384, dims=(96, 192, 384, 768), depths=(2, 2, 5, 3), sigma_min=1e-4):
        super(FlowModel, self).__init__()
        self.image_channels = image_channels

        self.sigma_min = sigma_min
        self.sigma_offset = 1 - sigma_min

        self.t_model = nn.Sequential(
            SinusoidalPosEmb(d_t),
            nn.Linear(d_t, 4 * d_t),
            nn.GELU(),
            nn.Linear(4 * d_t, d_t)
        )

        self.stem = nn.Conv2d(image_channels, dims[0], kernel_size=5, padding=2)

        self.down_path = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.down_path.append(nn.ModuleList([
                Block(dims[i], d_t) for _ in range(depths[i])
            ]))
            self.down_samples.append(nn.Conv2d(
                dims[i], dims[i + 1], kernel_size=4, stride=2, padding=1
            ))

        self.mid_blocks = nn.ModuleList([
            Block(dims[-1], d_t) for _ in range(depths[-1])
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
                Block(dims[i], d_t) for _ in range(depths[i])
            ]))

        self.head = nn.Conv2d(dims[0], image_channels, kernel_size=5, padding=2)

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
    def sample(self, num_samples=1, image_size=218, num_steps=200, step="euler"):
        dt = 1 / num_steps

        x_t = torch.randn(num_samples, self.image_channels, image_size, image_size)
        ts = torch.linspace(1, 0, num_steps).unsqueeze(1).expand(-1, num_samples)

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

        return (x_t * FaceDataset.std + FaceDataset.mean).clamp(0.0, 1.0)

    def forward(self, x_1):
        B = x_1.shape[0]

        t = torch.rand(B, device=x_1.device).view(B, 1, 1, 1)
        x_0 = torch.randn_like(x_1)
        x_t = (1 - self.sigma_offset * t) * x_0 + t * x_1

        pred = self.pred_flow(x_t, t)

        return F.mse_loss(pred, x_1 - self.sigma_offset * x_0)
