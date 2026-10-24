import torch
import torch.nn as nn

from .attention import FlashSelfAttention, FlashCrossAttention, FeedForward
from .time_embedding import Timesteps, TimestepEmbedding

class AdaptiveLayerNorm(nn.Module):
    """Adaptive layer normalization with timestep conditioning."""

    def __init__(
        self, dim: int, act_fn: nn.Module = nn.SiLU, num_channels: int = 256
    ):
        """Initialize the adaptive layer normalization.

        Args:
            dim (int): Dimension of embeddings.
            act_fn (nn.Module): Activation function. Default: nn.SiLU.
            num_channels (int): Number of channels for timestep projection. Default: 256.
        """
        super().__init__()
        self.timestep_proj = Timesteps(
            num_channels=num_channels, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=num_channels, time_embed_dim=dim
        )
        self.activation = act_fn()
        self.linear = nn.Linear(dim, dim * 2)                       # for scale and shift
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Apply adaptive layer normalization.

        Args:
            x (B, N, dim): Input tensor.
            timestep (B,): Timestep tensor.

        Returns:
            (B, N, dim): Normalized tensor.
        """
        emb = self.timestep_embedder(self.timestep_proj(timestep))    # (B, dim)
        emb = self.linear(self.activation(emb))                       # (B, dim * 2)
        scale, shift = emb.unsqueeze(1).chunk(2, dim=-1)              # (B, 1, dim) for both
        return self.norm(x) * (1 + scale) + shift


class GraspSeqRefineDiTLayer(nn.Module):
    def __init__(
        self,
        hoi_dim: int,
        visual_dim: int,
        obj_dim: int,
        pose_dim: int,
        heads: int = 8,
        inner_dim: int = 1024,
        dropout: float = 0.0,
        self_attn: bool = True,
        hoi_cross_attn: bool = True,
        visual_cross_attn: bool = True,
        obj_cross_attn: bool = False,
        qkv_proj_bias: bool = False,
        qk_norm: bool = True,
        attn_dtype: torch.dtype = torch.float16,
        use_rope: bool = True
    ):
        super().__init__()

        self.self_attn = self_attn
        self.hoi_cross_attn = hoi_cross_attn
        self.visual_cross_attn = visual_cross_attn
        self.obj_cross_attn = obj_cross_attn

        self.self_norm = AdaptiveLayerNorm(pose_dim) if self_attn else None
        self.self_attn_module = FlashSelfAttention(
            dim=pose_dim,
            inner_dim=inner_dim,
            heads=heads,
            dropout_p=dropout,
            qkv_proj_bias=qkv_proj_bias,
            qk_norm=qk_norm,
            attn_dtype=attn_dtype,
            use_rope=use_rope
        ) if self_attn else None

        self.hoi_norm = AdaptiveLayerNorm(pose_dim) if hoi_cross_attn else None
        self.hoi_cross_attn_module = FlashCrossAttention(
            q_dim=pose_dim,
            kv_dim=hoi_dim,
            inner_dim=inner_dim,
            heads=heads,
            dropout_p=dropout,
            qkv_proj_bias=qkv_proj_bias,
            qk_norm=qk_norm,
            attn_dtype=attn_dtype
        ) if hoi_cross_attn else None

        self.visual_norm = AdaptiveLayerNorm(pose_dim) if visual_cross_attn else None
        self.visual_cross_attn_module = FlashCrossAttention(
            q_dim=pose_dim,
            kv_dim=visual_dim,
            inner_dim=inner_dim,
            heads=heads,
            dropout_p=dropout,
            qkv_proj_bias=qkv_proj_bias,
            qk_norm=qk_norm,
            attn_dtype=attn_dtype
        ) if visual_cross_attn else None

        self.obj_norm = AdaptiveLayerNorm(pose_dim) if obj_cross_attn else None
        self.obj_cross_attn_module = FlashCrossAttention(
            q_dim=pose_dim,
            kv_dim=obj_dim,
            inner_dim=inner_dim,
            heads=heads,
            dropout_p=dropout,
            qkv_proj_bias=qkv_proj_bias,
            qk_norm=qk_norm,
            attn_dtype=attn_dtype
        ) if obj_cross_attn else None

        self.ffn_norm = nn.LayerNorm(pose_dim)
        self.ffn = FeedForward(dim=pose_dim, dropout_p=dropout)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, hoi_context: torch.Tensor, visual_context: torch.Tensor, obj_context: torch.Tensor = None) -> torch.Tensor:
        # x: (B, seq_len, pose_dim)
        # hoi_context: (B, seq_len, dim)
        # visual_context: (B, seq_len, num_patches + 1, dim)
        # obj_context: (B, seq_len, dim)
        if self.self_attn:
            x = x + self.self_attn_module(self.self_norm(x, timestep))
        if self.hoi_cross_attn:
            x = x + self.hoi_cross_attn_module(self.hoi_norm(x, timestep), context=hoi_context)
        if self.visual_cross_attn:
            batch_size, seq_len, _ = x.shape
            x_norm = self.visual_norm(x, timestep).view(batch_size * seq_len, 1, -1)
            num_tokens = visual_context.shape[2]
            visual_context_view = visual_context.view(batch_size * seq_len, num_tokens, -1)
            attn_out = self.visual_cross_attn_module(x_norm, context=visual_context_view)
            x = x + attn_out.view(batch_size, seq_len, -1)
        if self.obj_cross_attn:
            x = x + self.obj_cross_attn_module(self.obj_norm(x, timestep), context=obj_context)
        x = x + self.ffn(self.ffn_norm(x))

        return x

class GraspSeqRefineDiT(nn.Module):
    def __init__(
        self,
        hoi_dim: int,
        visual_dim: int,
        pose_dim: int,
        obj_dim: int = -1,
        depth: int = 6,
        heads: int = 8,
        inner_dim: int = 1024,
        dropout: float = 0.0,
        self_attn: bool = True,
        hoi_cross_attn: bool = True,
        visual_cross_attn: bool = True,
        obj_cross_attn: bool = False,
        qkv_proj_bias: bool = False,
        qk_norm: bool = True,
        attn_dtype: torch.dtype = torch.float16,
        use_rope: bool = True
    ):
        super().__init__()

        self.self_attn = self_attn
        self.hoi_cross_attn = hoi_cross_attn
        self.visual_cross_attn = visual_cross_attn
        self.obj_cross_attn = obj_cross_attn
        self.layers = nn.ModuleList(
            GraspSeqRefineDiTLayer(
                hoi_dim=hoi_dim,
                visual_dim=visual_dim,
                pose_dim=pose_dim,
                obj_dim=obj_dim,
                heads=heads,
                inner_dim=inner_dim,
                dropout=dropout,
                self_attn=self_attn,
                hoi_cross_attn=hoi_cross_attn,
                visual_cross_attn=visual_cross_attn,
                obj_cross_attn=obj_cross_attn,
                qkv_proj_bias=qkv_proj_bias,
                qk_norm=qk_norm,
                attn_dtype=attn_dtype,
                use_rope=use_rope
            ) for _ in range(depth)
        )

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, hoi_context: torch.Tensor, visual_context: torch.Tensor, obj_context: torch.Tensor = None) -> torch.Tensor:
        # x: (B, seq_len, hand_dim)
        # hoi_context: (B, seq_len, dim)
        # visual_context: (B, seq_len, num_patches + 1, dim)
        # obj_context: (B, seq_len, dim)
        for layer in self.layers:
            x = layer(x, timestep, hoi_context, visual_context, obj_context)

        return x
