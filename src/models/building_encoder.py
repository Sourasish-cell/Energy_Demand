"""
Building Encoder: MLP over tabular building / postcode attributes.

Input  : (B, building_dim)  — normalised building features
Output : (B, building_out)  — dense building embedding
"""

import torch
import torch.nn as nn
from typing import Sequence


class BuildingEncoder(nn.Module):
    """
    Two-layer MLP with LayerNorm and GELU activations.

    Architecture:
        building_dim → hidden[0] → hidden[1] → building_out

    Parameters
    ----------
    building_dim : int
        Number of input building features (default 16).
    hidden_dims  : Sequence[int]
        Hidden layer widths.  Default (64, 64).
    out_dim      : int
        Output embedding dimension.  Default 64.
    dropout      : float
        Dropout probability applied after each hidden activation.
    """

    def __init__(
        self,
        building_dim: int = 16,
        hidden_dims: Sequence[int] = (64, 64),
        out_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        dims = [building_dim, *hidden_dims, out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                # Hidden layers: norm → activation → dropout
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        # Final layer: just linear (caller may add norm at fusion stage)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, building_dim)

        Returns
        -------
        (B, out_dim)
        """
        return self.net(x)
