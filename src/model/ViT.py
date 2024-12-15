import torch
import torch.nn.functional as F

from torch import nn
from einops import rearrange
from math import sqrt

from .util import FiLM


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

        return self.W_o(rearrange(
            F.softmax((q @ k.transpose(-2, -1)) / self.scale, dim=-1) @ v,
            "b n l d -> b l (n d)"
        ))


class ViTBlock(nn.Module):
    def __init__(self, d_model, n_heads, hidden_size=None, norm_eps=1e-6):
        super(ViTBlock, self).__init__()
        if hidden_size is None:
            hidden_size = 4 * d_model

        self.attn = Attention(d_model, n_heads)
        self.attn_norm = nn.LayerNorm(d_model, eps=norm_eps)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, d_model)
        )
        self.ffn_norm = nn.LayerNorm(d_model, eps=norm_eps)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))

        return x + self.ffn(self.ffn_norm(x))


class ViTFiLMBlock(nn.Module):
    def __init__(self, d_model, d_t, n_heads, hidden_size=None, norm_eps=1e-6):
        super(ViTFiLMBlock, self).__init__()
        if hidden_size is None:
            hidden_size = 4 * d_model

        self.attn = Attention(d_model, n_heads)
        self.attn_norm = FiLM(d_model, d_t, eps=norm_eps)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, d_model)
        )
        self.ffn_norm = FiLM(d_model, d_t, eps=norm_eps)

    def forward(self, x, t):
        x = x + self.attn(self.attn_norm(x, t))

        return x + self.ffn(self.ffn_norm(x, t))
