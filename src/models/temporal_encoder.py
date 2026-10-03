"""
Temporal Encoder: PatchTST-style patch transformer over 1-week hourly history.

Input  : (B, T, C)  — time-series of T steps, C channels
Output : (B, d_model) — pooled temporal representation

Reference: Nie et al., "A Time Series is Worth 64 Words", ICLR 2023.
We implement a clean standalone version (no external dependency on
the original PatchTST repo) that is compatible with PyTorch >= 2.1.
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class PatchEmbedding(nn.Module):
    """
    Slice a multivariate time series into overlapping patches and project
    each patch to d_model.

    Input  : (B, T, C)
    Output : (B, n_patches * C, d_model)  — channel-independent patching
    """

    def __init__(self, patch_len: int, stride: int,
                 in_channels: int, d_model: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.in_channels = in_channels
        self.projection = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, C)
        Returns (B * C, n_patches, d_model)
        """
        B, T, C = x.shape
        # Pad if necessary so the last patch is complete
        pad_len = (self.patch_len - (T - self.patch_len) % self.stride) % self.stride
        if pad_len > 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_len))
        T_padded = x.shape[1]

        # Extract patches: (B, C, n_patches, patch_len)
        n_patches = (T_padded - self.patch_len) // self.stride + 1
        # unfold along the time dimension
        x_t = x.permute(0, 2, 1)  # (B, C, T_padded)
        patches = x_t.unfold(dimension=2, size=self.patch_len, step=self.stride)
        # patches: (B, C, n_patches, patch_len)

        # Channel-independent: treat each channel as a separate "batch" item
        patches = patches.reshape(B * C, n_patches, self.patch_len)
        emb = self.projection(patches)   # (B*C, n_patches, d_model)
        return self.dropout(emb)


class PatchTemporalEncoder(nn.Module):
    """
    Transformer encoder over temporal patches.

    Returns a (B, d_model) summary vector: mean-pool of the final
    encoder layer outputs, then a linear projection per channel,
    then mean over channels.

    Parameters
    ----------
    in_channels  : int   Number of input features (demand + gen + weather = 11).
    patch_len    : int   Hours per patch (default 24 = one day).
    patch_stride : int   Patch stride (default 12 = 50% overlap).
    d_model      : int   Transformer hidden dimension.
    n_heads      : int   Attention heads.
    n_layers     : int   Transformer encoder layers.
    dropout      : float Dropout probability.
    """

    def __init__(
        self,
        in_channels: int = 11,
        patch_len: int = 24,
        patch_stride: int = 12,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model

        self.patch_embed = PatchEmbedding(
            patch_len=patch_len,
            stride=patch_stride,
            in_channels=in_channels,
            d_model=d_model,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN for training stability
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Per-channel projection back to d_model, then mean over channels
        self.channel_proj = nn.Linear(d_model, d_model)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, C)   e.g. (32, 168, 11)

        Returns
        -------
        (B, d_model)
        """
        B, T, C = x.shape

        # Patch + embed: (B*C, n_patches, d_model)
        emb = self.patch_embed(x)
        n_patches = emb.shape[1]

        # Transformer: (B*C, n_patches, d_model)
        out = self.encoder(emb)

        # Pool over patches: (B*C, d_model)
        out = out.mean(dim=1)

        # Project: (B*C, d_model)
        out = self.channel_proj(out)

        # Reshape to (B, C, d_model) and average over channels: (B, d_model)
        out = out.reshape(B, C, self.d_model).mean(dim=1)
        return self.out_norm(out)
