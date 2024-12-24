import torch
import torch.nn.functional as F

from torch import nn
from math import sqrt
from einops import rearrange

from .util import RMSFiLM, SwiGLU, SinusoidalPosEmb, FlowModel


class Attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(Attention, self).__init__()
        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.scale = sqrt(d_model // n_heads)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        q = rearrange(self.W_q(x), "b l (n d) -> b n l d", n=self.n_heads)
        k = rearrange(self.W_k(x), "b l (n d) -> b n l d", n=self.n_heads)
        v = rearrange(self.W_v(x), "b l (n d) -> b n l d", n=self.n_heads)

        attn = (q @ k.transpose(-2, -1)) / self.scale

        return self.W_o(
            F.softmax(attn, dim=-1) @ v, "b n l d -> b l (n d)"
        )


class Block(nn.Module):
    def __init__(self, d_model, d_t, n_heads, hidden_size=None, norm_eps=1e-6):
        super(Block, self).__init__()
        self.attn = Attention(d_model, n_heads)
        self.attn_norm = RMSFiLM(d_model, d_t, eps=norm_eps)

        self.ffn = SwiGLU(d_model, hidden_size)
        self.ffn_norm = RMSFiLM(d_model, d_t, eps=norm_eps)

    def forward(self, x, t):
        x = x + self.attn(self.attn_norm(x, t))

        return x + self.ffn(self.ffn_norm(x, t))


class ViT(FlowModel):
    def __init__(
            self, image_channels, image_size, patch_size=8, d_model=768, d_t=384, n_layers=12, n_heads=12,
            sigma_min=1e-4
    ):
        super(ViT, self).__init__(image_channels, d_t, sigma_min)
        assert image_size % patch_size == 0

        self.num_patches = image_size // patch_size
        self.patch_size = patch_size

        self.stem = nn.Linear(image_channels * patch_size * patch_size, d_model)
        self.pos_emb = nn.Parameter(
            torch.empty(1, self.num_patches * self.num_patches, d_model).normal_(std=(1 / sqrt(d_model)))
        )

        self.layers = nn.ModuleList([
            Block(d_model, d_t, n_heads) for _ in range(n_layers)
        ])

        self.head = nn.Linear(d_model, image_channels * patch_size * patch_size)

    def pred_flow(self, x_t, t):
        t_emb = self.t_model(t)

        x_t = self.stem(rearrange(
            x_t, "b c (h hp) (w wp) -> b (h w) (c hp wp)",
            hp=self.patch_size, wp=self.patch_size
        )) + self.pos_emb

        for layer in self.layers:
            x_t = layer(x_t, t_emb)

        return rearrange(
            self.head(x_t), "b (h w) (c hp wp) -> b c (h hp) (w wp)",
            hp=self.patch_size, wp=self.patch_size,
            h=self.num_patches, w=self.num_patches
        )
