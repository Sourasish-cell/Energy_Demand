"""
Cross-Attention Fusion: merges the four encoder streams into a shared
representation using multi-head cross-attention.

Encoding streams:
  1. Building encoder → (B, d_b)
  2. Weather encoder  → (B, d_w)
  3. Temporal encoder → (B, d_t)
  4. Spatial GAT      → (B, d_s)  [after GAT, select node embedding for each sample]

All streams are projected to a common fusion_d_model before attention.

Cross-attention pattern:
  - temporal + spatial serve as the *query* (they carry time-series structure)
  - building + weather serve as the *key/value* context
  - A second cross-attention step swaps the roles so building/weather can
    also attend over the temporal context
  - Final output: mean-pool over the four projected tokens → (B, fusion_d_model)
"""

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """
    Fuse four modality embeddings with cross-attention.

    Parameters
    ----------
    in_dims : tuple[int, int, int, int]
        Input dimensions for (building, weather, temporal, spatial).
        Defaults match the default config: (64, 64, 128, 128).
    fusion_d_model : int
        Common projection size for all streams.
    n_heads : int
        Attention heads.
    dropout : float
    shared_dim : int
        Output dimension after the final linear projection.
    """

    def __init__(
        self,
        in_dims: tuple = (64, 64, 128, 128),
        fusion_d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
        shared_dim: int = 256,
        no_cross_attn: bool = False,
    ) -> None:
        super().__init__()
        d = fusion_d_model
        # no_cross_attn ablation: replace both attention steps with a plain
        # concatenation + FFN so the four streams are fused WITHOUT attending
        # to one another. Isolates the contribution of cross-attention.
        self.no_cross_attn = no_cross_attn

        # Project each stream to d
        self.proj_b = nn.Sequential(nn.Linear(in_dims[0], d), nn.LayerNorm(d))
        self.proj_w = nn.Sequential(nn.Linear(in_dims[1], d), nn.LayerNorm(d))
        self.proj_t = nn.Sequential(nn.Linear(in_dims[2], d), nn.LayerNorm(d))
        self.proj_s = nn.Sequential(nn.Linear(in_dims[3], d), nn.LayerNorm(d))

        # Cross-attention 1: temporal+spatial query over building+weather context
        self.ca1 = nn.MultiheadAttention(
            embed_dim=d, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        # Cross-attention 2: building+weather query over temporal+spatial context
        self.ca2 = nn.MultiheadAttention(
            embed_dim=d, num_heads=n_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward after each cross-attention
        self.ff1 = _FFN(d, dropout=dropout)
        self.ff2 = _FFN(d, dropout=dropout)

        # Used only in the no_cross_attn path (self-attention-free fusion)
        self.ff_alt = _FFN(4 * d, dropout=dropout)
        self.norm_alt = nn.LayerNorm(4 * d)

        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

        # Final shared projection over the 4 fused tokens
        self.out = nn.Sequential(
            nn.Linear(4 * d, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        z_building: torch.Tensor,   # (B, d_b)
        z_weather: torch.Tensor,    # (B, d_w)
        z_temporal: torch.Tensor,   # (B, d_t)
        z_spatial: torch.Tensor,    # (B, d_s)
    ) -> torch.Tensor:
        """
        Returns (B, shared_dim).
        """
        # Project all streams to fusion_d_model
        b = self.proj_b(z_building).unsqueeze(1)   # (B, 1, d)
        w = self.proj_w(z_weather).unsqueeze(1)    # (B, 1, d)
        t = self.proj_t(z_temporal).unsqueeze(1)   # (B, 1, d)
        s = self.proj_s(z_spatial).unsqueeze(1)    # (B, 1, d)

        if self.no_cross_attn:
            # Ablation: no inter-stream attention — just concatenate the four
            # projected streams and pass through an FFN.
            fused = torch.cat([b, w, t, s], dim=1)          # (B, 4, d)
            fused = fused.reshape(fused.shape[0], -1)       # (B, 4*d)
            fused = self.ff_alt(self.norm_alt(fused))       # (B, 4*d)
            return self.out(fused)                           # (B, shared_dim)

        # Cross-attention 1: [temporal, spatial] attend over [building, weather]
        ts_query = torch.cat([t, s], dim=1)        # (B, 2, d)
        bw_ctx   = torch.cat([b, w], dim=1)        # (B, 2, d)
        ts_out, _ = self.ca1(ts_query, bw_ctx, bw_ctx)
        ts_out = self.norm1(ts_out + ts_query)
        ts_out = self.ff1(ts_out)                  # (B, 2, d)

        # Cross-attention 2: [building, weather] attend over updated [temporal, spatial]
        bw_out, _ = self.ca2(bw_ctx, ts_out, ts_out)
        bw_out = self.norm2(bw_out + bw_ctx)
        bw_out = self.ff2(bw_out)                  # (B, 2, d)

        # Concatenate all 4 tokens and project to shared_dim
        fused = torch.cat([bw_out, ts_out], dim=1)   # (B, 4, d)
        fused = fused.reshape(fused.shape[0], -1)    # (B, 4*d)
        return self.out(fused)                        # (B, shared_dim)


class _FFN(nn.Module):
    """Two-layer feed-forward block with residual + LayerNorm."""
    def __init__(self, d: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * expansion, d),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)
