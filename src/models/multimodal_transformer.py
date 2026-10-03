"""
MultimodalEnergyTransformer — the full Paper 1 model.

Wires together:
  BuildingEncoder  (MLP over tabular attributes)
  WeatherEncoder   (MLP over current-window weather snapshot)
  PatchTemporalEncoder (PatchTST-style transformer)
  SpatialGAT       (pure-PyTorch multi-head GAT)
  CrossAttentionFusion
  Multi-task prediction heads (demand + generation × 4 horizons)

Input batch dict (produced by AusgridSubgraphDataset); forward() receives
these as positional tensors:
  x_building      : (B, 16)            normalised building features
  x_weather_now   : (B, 9)             weather snapshot at the anchor hour
  x_temporal      : (B, T, 19)         root time-series window (19 channels)
  sub_temporal    : (B, K+1, T, 19)    root (idx 0) + K neighbour windows,
                                       every window ending at the anchor hour
  sub_edge_index  : (2, E_local)       local star-subgraph topology (one graph)
  sub_edge_weight : (B, E_local)       per-item train-period edge weights
  sub_edge_bias   : (B, E_local)       0 for real edges, -1e9 for padded ones

Outputs:
  demand_pred   : (B, 4)   scaled predictions at 1h/6h/24h/168h
  gen_pred      : (B, 4)   scaled predictions at 1h/6h/24h/168h
"""

import torch
import torch.nn as nn

from .building_encoder import BuildingEncoder
from .temporal_encoder import PatchTemporalEncoder
from .spatial_gnn import SpatialGAT
from .fusion import CrossAttentionFusion


class WeatherEncoder(nn.Module):
    """Simple MLP over the 9-channel weather snapshot."""

    def __init__(self, weather_dim: int = 9,
                 hidden_dims: tuple = (64,),
                 out_dim: int = 64,
                 dropout: float = 0.1) -> None:
        super().__init__()
        dims = [weather_dim, *hidden_dims, out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultimodalEnergyTransformer(nn.Module):
    """
    Full architecture for joint demand + generation forecasting.

    Parameters (passed as a flat dict, matching configs/default.yaml):
        building_dim, building_hidden, building_out
        weather_dim, weather_hidden, weather_out
        temporal_features, patch_len, patch_stride,
        d_model, n_heads, n_layers, dropout
        gat_in, gat_hidden, gat_out, gat_heads, gat_dropout
        fusion_d_model, fusion_n_heads, fusion_dropout
        shared_dim
        n_horizons  (int, default 4)
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        m = cfg["model"]

        # ── encoders ──────────────────────────────────────────────────────────
        self.building_enc = BuildingEncoder(
            building_dim=m["building_dim"],
            hidden_dims=m["building_hidden"],
            out_dim=m["building_out"],
            dropout=m.get("dropout", 0.1),
        )
        self.weather_enc = WeatherEncoder(
            weather_dim=m["weather_dim"],
            hidden_dims=m["weather_hidden"],
            out_dim=m["weather_out"],
            dropout=m.get("dropout", 0.1),
        )
        self.temporal_enc = PatchTemporalEncoder(
            in_channels=m["temporal_features"],
            patch_len=m["patch_len"],
            patch_stride=m["patch_stride"],
            d_model=m["d_model"],
            n_heads=m["n_heads"],
            n_layers=m["n_layers"],
            dropout=m.get("dropout", 0.1),
        )
        self.spatial_gnn = SpatialGAT(
            in_features=m["gat_in"],
            hidden=m["gat_hidden"],
            out_features=m["gat_out"],
            n_heads=m["gat_heads"],
            dropout=m.get("gat_dropout", 0.1),
        )

        # ── fusion ────────────────────────────────────────────────────────────
        self.fusion = CrossAttentionFusion(
            in_dims=(
                m["building_out"],
                m["weather_out"],
                m["d_model"],
                m["gat_out"],
            ),
            fusion_d_model=m["fusion_d_model"],
            n_heads=m["fusion_n_heads"],
            dropout=m.get("fusion_dropout", 0.1),
            shared_dim=m["shared_dim"],
            no_cross_attn=bool(m.get("no_cross_attn", False)),
        )

        # ── multi-task prediction heads ───────────────────────────────────────
        # One linear layer per target per horizon.
        # We share the trunk and split only at the final projection.
        n_horizons = m.get("n_horizons", 4)
        self.demand_head = nn.Linear(m["shared_dim"], n_horizons)
        self.gen_head    = nn.Linear(m["shared_dim"], n_horizons)

        # Store for forward
        self._d_model = m["d_model"]
        self._gat_in  = m["gat_in"]

    # ── graph utilities (leakage-safe) ────────────────────────────────────────
    #
    # DESIGN NOTE — why the spatial embedding is computed per sample, not once:
    # A naive implementation encodes every graph node over the whole timeline
    # once, then indexes the embedding per sample.  That leaks future
    # information: a prediction anchored at hour t would receive a neighbour
    # embedding that summarises the neighbour's data at hours > t (and, across
    # the graph, test-period data).  Instead, every sample supplies a LOCAL
    # STAR SUBGRAPH whose node windows ALL END at that sample's anchor hour t.
    # The GAT then runs over just that subgraph and we read the root's output.
    # This makes the spatial embedding a function of data available at t only.

    def forward(
        self,
        x_building: torch.Tensor,       # (B, 16)
        x_weather_now: torch.Tensor,    # (B, 9)
        x_temporal: torch.Tensor,       # (B, T, 11)  root window
        sub_temporal: torch.Tensor,     # (B, K+1, T, 11) root at idx 0, + K neighbours
        sub_edge_index: torch.Tensor,   # (2, E_local) node ids in [0, K] (single graph)
        sub_edge_weight: torch.Tensor,  # (B, E_local) per-item train-period weights
        sub_edge_bias: torch.Tensor,    # (B, E_local) 0 real, -1e9 padded
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        demand_pred : (B, n_horizons)
        gen_pred    : (B, n_horizons)
        """
        B, Kp1, T, C = sub_temporal.shape
        K = Kp1 - 1

        # ── encode each stream for the root sample ────────────────────────────
        z_b = self.building_enc(x_building)           # (B, building_out)
        z_w = self.weather_enc(x_weather_now)         # (B, weather_out)

        # Root temporal embedding (used directly by the temporal stream)
        z_t = self.temporal_enc(x_temporal)           # (B, d_model)

        # ── spatial stream over the contemporaneous local star subgraph ───────
        # Encode every node (root + neighbours) in the subgraph. Each window ends
        # at the sample's anchor hour t, so no future information enters here.
        flat = sub_temporal.reshape(B * Kp1, T, C)
        node_emb = self.temporal_enc(flat)            # (B*Kp1, d_model)

        # Build a block-diagonal batched graph: item b owns nodes
        # [b*Kp1, (b+1)*Kp1).  A single GAT call then processes all B subgraphs.
        device = node_emb.device
        offsets = (torch.arange(B, device=device) * Kp1).unsqueeze(1)  # (B, 1)
        batched_ei = (sub_edge_index.to(device)
                      + offsets.unsqueeze(1))          # (2, B*E_local)
        batched_ei = batched_ei.reshape(2, -1)
        batched_ew = sub_edge_weight.reshape(-1)       # (B*E_local,)
        batched_eb = sub_edge_bias.reshape(-1)         # (B*E_local,)

        node_out = self.spatial_gnn(
            node_emb, batched_ei, batched_ew, edge_bias=batched_eb
        )                                             # (B*Kp1, gat_out)
        node_out = node_out.reshape(B, Kp1, -1)
        z_s = node_out[:, 0, :]                        # root node = index 0

        # ── cross-attention fusion ────────────────────────────────────────────
        z = self.fusion(z_b, z_w, z_t, z_s)           # (B, shared_dim)

        # ── prediction heads ──────────────────────────────────────────────────
        demand_pred = self.demand_head(z)             # (B, n_horizons)
        gen_pred    = self.gen_head(z)                # (B, n_horizons)

        return demand_pred, gen_pred
