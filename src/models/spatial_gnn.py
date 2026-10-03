"""
Spatial GAT: Graph Attention Network over the customer/postcode graph.

No external graph library required — implemented in pure PyTorch.
Supports batched node feature matrices and a shared edge_index / edge_weight
that is fixed for all samples (the graph topology is static within a run).

Equation (per layer):
    h_i' = σ( Σ_{j ∈ N(i)} α_ij · W · h_j )
    α_ij = softmax_j( LeakyReLU( a^T [W·h_i || W·h_j] ) )

Reference: Veličković et al., "Graph Attention Networks", ICLR 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """
    Single-head GAT layer (multi-head via GATLayer stacking or concat).

    Input  : (N, in_features)
    Output : (N, out_features)
    """

    def __init__(self, in_features: int, out_features: int,
                 dropout: float = 0.1,
                 negative_slope: float = 0.2,
                 residual: bool = True) -> None:
        super().__init__()
        self.out_features = out_features
        self.residual = residual
        self.W = nn.Linear(in_features, out_features, bias=False)
        # Attention vector: operates on concatenated source+dest projections
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leaky = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_features)

        if residual:
            if in_features != out_features:
                self.res_proj = nn.Linear(in_features, out_features, bias=False)
            else:
                self.res_proj = nn.Identity()

    def forward(self, h: torch.Tensor,
                edge_index: torch.Tensor,
                edge_weight: torch.Tensor,
                edge_bias: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        h            : (N, in_features)   node features
        edge_index   : (2, E)             [source_indices; dest_indices]
        edge_weight  : (E,)               scalar edge weights in [0, 1]
        edge_bias    : (E,) or None       additive logit bias.  Padded /
                       masked edges should carry a large negative value
                       (e.g. -1e9) so softmax assigns them ~0 weight.  This
                       is how variable-sized star subgraphs are batched
                       without diluting attention with padding.

        Returns
        -------
        (N, out_features)
        """
        N = h.shape[0]
        Wh = self.W(h)                          # (N, out_features)
        Wh = self.dropout(Wh)

        src, dst = edge_index[0], edge_index[1]  # (E,)

        # Concatenate source and destination projections for attention
        e_input = torch.cat([Wh[src], Wh[dst]], dim=-1)   # (E, 2*out)
        e = self.leaky(self.a(e_input)).squeeze(-1)         # (E,)

        # Incorporate structural edge weights (multiplicative)
        e = e * edge_weight
        # Apply the additive mask (removes padded neighbours before softmax)
        if edge_bias is not None:
            e = e + edge_bias

        # Softmax over neighbours for each destination node
        alpha = _softmax_by_dest(e, dst, N)   # (E,)
        alpha = self.dropout(alpha)

        # Aggregate: h'_i = Σ alpha_ij * Wh_j
        out = torch.zeros(N, self.out_features, device=h.device, dtype=h.dtype)
        out.scatter_add_(0, dst.unsqueeze(1).expand(-1, self.out_features),
                         alpha.unsqueeze(1) * Wh[src])

        if self.residual:
            out = out + self.res_proj(h)

        return self.norm(out)


def _softmax_by_dest(e: torch.Tensor, dst: torch.Tensor,
                     n_nodes: int) -> torch.Tensor:
    """
    Compute softmax of edge scores grouped by destination node.
    Uses the numerically stable scatter-based trick.
    """
    # Max per destination for numerical stability
    e_max = torch.full((n_nodes,), float("-inf"),
                       device=e.device, dtype=e.dtype)
    e_max.scatter_reduce_(0, dst, e, reduce="amax", include_self=True)
    e_shifted = e - e_max[dst]
    exp_e = torch.exp(e_shifted)

    exp_sum = torch.zeros(n_nodes, device=e.device, dtype=e.dtype)
    exp_sum.scatter_add_(0, dst, exp_e)
    return exp_e / (exp_sum[dst] + 1e-8)


class SpatialGAT(nn.Module):
    """
    Multi-head GAT with two layers.

    Architecture:
        in_features → (gat_hidden × gat_heads) → gat_out

    The multi-head outputs from layer 1 are concatenated then projected
    to gat_hidden before feeding layer 2 (following the original paper).

    Parameters
    ----------
    in_features  : int   Input node feature size (= d_model from temporal enc.).
    hidden       : int   Per-head hidden size.
    out_features : int   Output embedding dimension.
    n_heads      : int   Number of attention heads in layer 1.
    dropout      : float Dropout probability.
    """

    def __init__(self, in_features: int = 128,
                 hidden: int = 128,
                 out_features: int = 128,
                 n_heads: int = 4,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.heads = nn.ModuleList([
            GATLayer(in_features, hidden, dropout=dropout, residual=(i == 0))
            for i in range(n_heads)
        ])
        # Project concatenated multi-head output → hidden
        self.proj = nn.Linear(n_heads * hidden, hidden)
        self.act = nn.GELU()
        # Single-head second layer
        self.layer2 = GATLayer(hidden, out_features, dropout=dropout, residual=True)

    def forward(self, h: torch.Tensor,
                edge_index: torch.Tensor,
                edge_weight: torch.Tensor,
                edge_bias: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        h            : (N, in_features)
        edge_index   : (2, E)  long tensor
        edge_weight  : (E,)    float tensor
        edge_bias    : (E,) or None   additive softmax logit bias.  Padded
                       edges carry a large negative value (e.g. -1e9) so the
                       softmax assigns them ~0 weight; without this the
                       zero-weight padded neighbours would still contribute
                       a nonzero attention mass to the aggregation.

        Returns
        -------
        (N, out_features)
        """
        # Multi-head layer 1: concat head outputs
        head_outs = [head(h, edge_index, edge_weight, edge_bias)
                     for head in self.heads]
        h = self.act(self.proj(torch.cat(head_outs, dim=-1)))
        # Single-head layer 2
        h = self.layer2(h, edge_index, edge_weight, edge_bias)
        return h
