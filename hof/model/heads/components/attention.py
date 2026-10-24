from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_qkvpacked_func, flash_attn_kvpacked_func
from flash_attn.layers.rotary import RotaryEmbedding

class MultiHeadRMSNorm(nn.Module):
    """Multi-head RMS normalization layer.

    Ref:
        https://github.com/lucidrains/mmdit/blob/main/mmdit/mmdit_pytorch.py

    Args:
        dim: Feature dimension.
        heads: Number of attention heads.
    """

    def __init__(self, dim: int, heads: int = 1):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head RMS normalization."""
        orig = x.dtype
        x = F.normalize(x.float(), dim=-1, eps=1e-6)
        return (x * self.gamma * self.scale).to(orig)

class FlashSelfAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            inner_dim: int,
            heads: int,
            dropout_p: float = 0.0,
            qkv_proj_bias: bool = False,
            qk_norm: bool = True,
            attn_dtype: torch.dtype = torch.float16,
            use_rope: bool = True
        ):
        super().__init__()

        assert inner_dim % heads == 0, "inner_dim must be divisible by heads"

        self.dim = dim
        self.inner_dim = inner_dim
        self.heads = heads
        self.head_dim = inner_dim // heads
        self.dropout_p = dropout_p
        self.qk_norm = qk_norm
        self.attn_dtype = attn_dtype

        self.qkv_proj = nn.Linear(dim, inner_dim * 3, bias=qkv_proj_bias)
        self.out_proj = nn.Linear(inner_dim, dim)

        if qk_norm:
            self.q_norm = MultiHeadRMSNorm(self.head_dim, self.heads)
            self.k_norm = MultiHeadRMSNorm(self.head_dim, self.heads)

        self.use_rope = use_rope
        if use_rope:
            self.rotary_emb = RotaryEmbedding(dim=self.head_dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv_proj(x)

        if self.qk_norm:
            qkv = qkv.reshape(batch_size * seq_len, 3, self.heads, self.head_dim)
            q, k, v = qkv.unbind(dim=1)
            q = self.q_norm(q).to(v.dtype)
            k = self.k_norm(k).to(v.dtype)
            qkv = torch.stack([q, k, v], dim=1)

        qkv = qkv.view(batch_size, seq_len, 3, self.heads, self.head_dim)

        if self.use_rope:
            qkv = self.rotary_emb(qkv)

        out = flash_attn_qkvpacked_func(
            qkv=qkv.to(self.attn_dtype),
            dropout_p=self.dropout_p if self.training else 0.0,
        ).to(x.dtype)
        out = out.view(batch_size, seq_len, self.inner_dim)

        out = self.out_proj(out)
        return out

class FlashCrossAttention(nn.Module):
    def __init__(
            self,
            q_dim: int,
            kv_dim: int,
            inner_dim: int,
            heads: int,
            dropout_p: float = 0.0,
            qkv_proj_bias: bool = False,
            qk_norm: bool = True,
            attn_dtype: torch.dtype = torch.float16
        ):
        super().__init__()

        assert inner_dim % heads == 0, "inner_dim must be divisible by heads"

        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.inner_dim = inner_dim
        self.heads = heads
        self.head_dim = inner_dim // heads
        self.dropout_p = dropout_p
        self.qk_norm = qk_norm
        self.attn_dtype = attn_dtype

        self.q_proj = nn.Linear(q_dim, inner_dim, bias=qkv_proj_bias)
        self.kv_proj = nn.Linear(kv_dim, inner_dim * 2, bias=qkv_proj_bias)
        self.out_proj = nn.Linear(inner_dim, q_dim)

        if qk_norm:
            self.q_norm = MultiHeadRMSNorm(self.head_dim, self.heads)
            self.k_norm = MultiHeadRMSNorm(self.head_dim, self.heads)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch_size, q_seq_len, _ = x.shape
        kv_seq_len = context.shape[1]

        q = self.q_proj(x)
        kv = self.kv_proj(context)

        if self.qk_norm:
            q = q.reshape(batch_size * q_seq_len, self.heads, self.head_dim)
            kv = kv.reshape(batch_size * kv_seq_len, 2, self.heads, self.head_dim)
            k, v = kv.unbind(dim=1)
            q = self.q_norm(q).to(v.dtype)
            k = self.k_norm(k).to(v.dtype)
            kv = torch.stack([k, v], dim=1)

        q = q.view(batch_size, q_seq_len, self.heads, self.head_dim)
        kv = kv.view(batch_size, kv_seq_len, 2, self.heads, self.head_dim)

        out = flash_attn_kvpacked_func(
            q=q.to(self.attn_dtype),
            kv=kv.to(self.attn_dtype),
            dropout_p=self.dropout_p if self.training else 0.0,
        ).to(x.dtype)
        out = out.view(batch_size, q_seq_len, self.inner_dim)

        out = self.out_proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(
            self,
            dim: int,
            inner_dim: Optional[int] = None,
            mult: int = 4,
            dropout_p: float = 0.0,
        ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
