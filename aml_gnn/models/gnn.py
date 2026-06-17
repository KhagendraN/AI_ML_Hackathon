"""
models/gnn.py

AML-GNN: Graph Attention Network with edge features for node-level
anomaly scoring.

Architecture: GraphSAGE + GAT hybrid with:
  - Edge-conditioned message passing (edge features modulate messages)
  - Jumping Knowledge (JK) aggregation across layers
  - Residual connections
  - Layer normalisation
  - MLP classifier head

Mathematical Intuition
----------------------
For each node v at layer ℓ, we compute:

    α_{uv} = softmax_u( LeakyReLU( a^T [Wh_u || Wh_v || We_{uv}] ) )
    h_v^(ℓ+1) = σ( Σ_{u∈N(v)} α_{uv} · W_msg · (h_u^(ℓ) || e_{uv}) )

where e_{uv} is the edge feature vector. Stacking ℓ=1..L layers
allows the model to aggregate information from L-hop neighbourhoods,
surfacing multi-hop patterns (chains, fan-in/fan-out, circular flows)
that are invisible to single-transaction classifiers.

JK-Net concatenates representations from all layers:
    h_v^final = JK( h_v^(1), h_v^(2), ..., h_v^(L) )
giving the MLP head access to both local (shallow) and global (deep)
structural context.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import (
    GATConv,
    SAGEConv,
    JumpingKnowledge,
    BatchNorm,
    global_mean_pool,
)
from torch_geometric.utils import add_self_loops, softmax


# ──────────────────────────────────────────────
# Edge-feature augmented message passing
# ──────────────────────────────────────────────

class EdgeGATConv(nn.Module):
    """
    GAT-style layer where edge features are concatenated to the
    key/query/value computation, allowing the attention coefficient
    to be modulated by transaction-level signals (amount, cross-border,
    timing, etc.).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        heads: int = 4,
        dropout: float = 0.2,
        concat: bool = True,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.out_channels = out_channels
        self.concat = concat
        self.dropout = dropout

        # Project node features
        self.lin_src = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_dst = nn.Linear(in_channels, heads * out_channels, bias=False)

        # Project edge features
        self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)

        # Attention vector
        self.att = nn.Parameter(torch.empty(1, heads, out_channels))
        nn.init.xavier_uniform_(self.att.view(1, -1).unsqueeze(0))

        self.bias = nn.Parameter(torch.zeros(
            heads * out_channels if concat else out_channels
        ))
        self.norm = nn.LayerNorm(heads * out_channels if concat else out_channels)

    def forward(
        self,
        x: Tensor,                 # [N, in_channels]
        edge_index: Tensor,        # [2, E]
        edge_attr: Tensor,         # [E, edge_dim]
    ) -> Tensor:
        from torch_geometric.utils import add_self_loops
        N = x.size(0)
        H, C = self.heads, self.out_channels

        # Add self-loops (no edge features for them)
        num_edges = edge_index.size(1)
        edge_index_sl, _ = add_self_loops(edge_index, num_nodes=N)
        # Pad edge_attr with zeros for self-loop edges
        padding = torch.zeros(
            edge_index_sl.size(1) - num_edges,
            edge_attr.size(1),
            device=edge_attr.device,
        )
        edge_attr_sl = torch.cat([edge_attr, padding], dim=0)

        src, dst = edge_index_sl[0], edge_index_sl[1]

        # Linear projections → [N, H*C] → [N, H, C]
        x_src = self.lin_src(x[src]).view(-1, H, C)
        x_dst = self.lin_dst(x[dst]).view(-1, H, C)
        x_edge = self.lin_edge(edge_attr_sl).view(-1, H, C)

        # Attention score: element-wise sum then dot with att
        alpha = (x_src + x_dst + x_edge) * self.att
        alpha = alpha.sum(dim=-1)                         # [E, H]
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = softmax(alpha, dst)                       # [E, H]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Message: (x_src + x_edge) * alpha, aggregated at dst
        msg = (x_src + x_edge) * alpha.unsqueeze(-1)     # [E, H, C]
        out = torch.zeros(N, H, C, device=x.device)
        out.scatter_add_(0, dst.view(-1, 1, 1).expand_as(msg), msg)

        if self.concat:
            out = out.view(N, H * C) + self.bias
        else:
            out = out.mean(dim=1) + self.bias

        return self.norm(out)


# ──────────────────────────────────────────────
# Main GNN backbone
# ──────────────────────────────────────────────

class AMLGNN(nn.Module):
    """
    Node-level binary classifier for suspicious account detection.

    Layers: EdgeGATConv (L layers) → JK-concat → MLP head

    Parameters
    ----------
    in_channels    : dimensionality of input node features
    edge_dim       : dimensionality of edge features
    hidden_channels: width of hidden layers
    num_layers     : number of message-passing layers
    heads          : number of attention heads (per GAT layer)
    dropout        : dropout probability
    jk_mode        : JumpingKnowledge aggregation ('cat'|'max'|'lstm')
    """

    def __init__(
        self,
        in_channels: int,
        edge_dim: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.3,
        jk_mode: str = "cat",
    ) -> None:
        super().__init__()

        self.dropout = dropout
        self.num_layers = num_layers
        self.jk_mode = jk_mode

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Message-passing layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            cin = hidden_channels
            # For concat-heads, downstream layers see heads*hidden_channels
            # but we project back to hidden_channels via a linear inside conv
            self.convs.append(
                EdgeGATConv(
                    in_channels=cin,
                    out_channels=hidden_channels // heads,
                    edge_dim=edge_dim,
                    heads=heads,
                    dropout=dropout,
                    concat=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_channels))

        # Jumping Knowledge
        jk_in_channels = hidden_channels * num_layers if jk_mode == "cat" else hidden_channels
        self.jk = JumpingKnowledge(mode=jk_mode, channels=hidden_channels, num_layers=num_layers)

        # MLP classifier head
        self.classifier = nn.Sequential(
            nn.Linear(jk_in_channels, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        """
        Returns
        -------
        logits : [N, 1] raw logits (pass through sigmoid for probabilities)
        """
        h = self.input_proj(x)

        xs = []
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index, edge_attr)
            h_new = norm(h_new)
            h_new = F.elu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            # Residual: project h if same dim
            if h.shape == h_new.shape:
                h_new = h_new + h
            h = h_new
            xs.append(h)

        # JK aggregation
        h = self.jk(xs)

        # Classification
        logits = self.classifier(h)
        return logits.squeeze(-1)

    @torch.no_grad()
    def predict_proba(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        self.eval()
        logits = self.forward(x, edge_index, edge_attr)
        return torch.sigmoid(logits)


# ──────────────────────────────────────────────
# Autoencoder for unsupervised anomaly baseline
# ──────────────────────────────────────────────

class NodeAutoencoder(nn.Module):
    """
    Graph Autoencoder using SAGEConv for reconstruction-based
    anomaly scoring.  Used as an unsupervised baseline and also
    to pre-train node representations.
    """

    def __init__(self, in_channels: int, hidden: int = 64) -> None:
        super().__init__()
        self.enc1 = SAGEConv(in_channels, hidden * 2)
        self.enc2 = SAGEConv(hidden * 2, hidden)
        self.dec  = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, in_channels),
        )

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        z = F.relu(self.enc1(x, edge_index))
        z = self.enc2(z, edge_index)
        return z

    def forward(self, x: Tensor, edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.encode(x, edge_index)
        x_hat = self.dec(z)
        return x_hat, z

    def anomaly_score(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x_hat, _ = self.forward(x, edge_index)
        return ((x - x_hat) ** 2).mean(dim=1)


def build_model(
    in_channels: int,
    edge_dim: int,
    cfg: dict,
) -> AMLGNN:
    """Factory function — reads hyperparams from config dict."""
    m = cfg.get("model", {})
    return AMLGNN(
        in_channels=in_channels,
        edge_dim=edge_dim,
        hidden_channels=m.get("hidden_channels", 128),
        num_layers=m.get("num_layers", 3),
        heads=m.get("heads", 4),
        dropout=m.get("dropout", 0.3),
        jk_mode=m.get("jk_mode", "cat"),
    )
