from typing import Dict, Tuple, List, Optional, Literal
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GraphConv, global_mean_pool, global_add_pool
from torch_geometric.data import HeteroData
from torch_geometric.utils import dropout_edge

# An edge type in PyG hetero graphs is a 3-tuple: (src_node_type, relation_name, dst_node_type)
EdgeType = Tuple[str, str, str]
# Readout modes (how we aggregate node embeddings into a single graph embedding)
Readout = Literal["fact", "company", "concat", "gated"]


class HeteroGNN(nn.Module):
    r"""
    Heterogeneous GCN for document–company graphs with **type-specific encoders**,
    **learned edge gating**, and flexible **readout**.

    # Architecture Overview
    1) **Type-specific input encoders (per node type)**  
       - Each node type has its own small MLP: `Linear(d_type → H) → ReLU → LayerNorm(H)`.  
       - This projects very different raw spaces (text embeddings for `fact`, numeric features for `company`) into a shared hidden size `H`, improving stability and performance.

    2) **Learned edge gate on edge attributes**  
       - Raw edge_attr = `[sentiment ∈ [-1,1], decay ∈ [0,1]]`.  
       - A tiny linear → sigmoid maps this 2D vector to a **scalar edge weight w ∈ (0,1)**.  
       - This keeps GCNConv stable (non-negative weights) while letting the model learn how to mix signals.

    3) **HeteroConv stack (message passing)**  
       - A stack of `num_layers` HeteroConv blocks, each holding a per-relation `GCNConv`.  
       - Since inputs are already projected to `H`, each `GCNConv(-1, H)` just preserves width.  
       - After each block: `ReLU → LayerNorm(H) per node type → Feature Dropout`.

    4) **Readout (graph-level pooling)**  
       - `'fact'`    : mean over fact nodes.  
       - `'company'` : mean over company nodes (or **primary-ticker-aware** mean if a boolean `primary_mask` is provided on `data['company']`).  
       - `'concat'`  : concat of both pooled vectors → `2H`.  
       - `'gated'`   : learn α ∈ [0,1] from `[fact_pool, company_pool]` and return `α*fact + (1-α)*company`.

    5) **Classifier**  
       - A simple `Linear` head maps the pooled embedding to a single logit per graph (use with `BCEWithLogitsLoss`).

    # Notes
    - Edge weights are passed with the correct kwarg `edge_weight=` (PyG expects this).
    - Optional **edge dropout** randomly removes edges during training (regularization).
    - Encoders are **lazily initialized** on the first forward to infer input dims from data, so you don't need to pass d_fact/d_company explicitly.
    """

    def __init__(
        self,
        metadata: Tuple[List[str], List[EdgeType]],  # (node_types, edge_types) from any HeteroData instance
        hidden_channels: int = 128,
        num_layers: int = 2,
        feature_dropout: float = 0.2,
        edge_dropout: float = 0.0,
        final_dropout: float = 0.0,
        readout: Readout = "gated",
    ) -> None:
        super().__init__()

        # Unpack graph meta (kept so the module can operate on any batch with same schema)
        self.node_types, self.edge_types = metadata

        # Hyperparams
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.feature_dropout = feature_dropout
        self.edge_dropout = edge_dropout
        self.final_dropout = final_dropout
        self.readout: Readout = readout

        # 1) Per-edge learned gate: edge_attr (2,) -> (1,) then sigmoid to (0,1)
        self.edge_mixer = nn.Linear(2, 1)
        self.edge_gate = nn.Sigmoid()
        # Stable init: bias to 0, weight to emphasize decay initially
        nn.init.constant_(self.edge_mixer.bias, 0.0)
        with torch.no_grad():
            self.edge_mixer.weight[:] = torch.tensor([[0.5, 1.0]])  # [sentiment, decay]

        # 2) Type-specific input encoders (lazy: we create them on first forward when dims are known)
        self.in_proj = nn.ModuleDict()   # to be filled with {'fact': MLP, 'company': MLP, ...}
        self.post_norm = nn.ModuleDict() # per-type LayerNorm after each conv block

        # 3) HeteroConv stack: each relation gets a GCNConv; outputs stay at hidden_channels
        self.convs = nn.ModuleList([
        HeteroConv(
            {
                et: GraphConv(
                    (-1, -1),                  # bipartite: (in_src, in_dst)
                    hidden_channels,
                    aggr='add',                # default; fine to keep
                )
                for et in self.edge_types
            },
            aggr="sum",
        )
        for _ in range(num_layers)
    ])

        # 4) Readout-specific params
        if self.readout == "gated":
            self.readout_gate = nn.Linear(2 * hidden_channels, 1)

        # 5) Classifier head size depends on readout
        cls_in = hidden_channels if self.readout in ("fact", "company", "gated") else 2 * hidden_channels
        self.classifier = nn.Linear(cls_in, 1)

    # -------------------------------
    # Helpers
    # -------------------------------
    def _maybe_build_type_encoders(self, data: HeteroData) -> None:
        """
        Lazily create per-type encoders and per-type post LayerNorms based on input dims.
        """
        for nt in self.node_types:
            if nt in self.in_proj:
                continue
            d_in = int(data[nt].x.size(-1))
            mlp = nn.Sequential(
                nn.Linear(d_in, self.hidden_channels),
                nn.ReLU(),
                nn.LayerNorm(self.hidden_channels),
            )
            self.in_proj[nt] = mlp
            self.post_norm[nt] = nn.LayerNorm(self.hidden_channels)

    def _edge_dicts(
        self, data: HeteroData
    ) -> Tuple[Dict[EdgeType, Tensor], Dict[EdgeType, Tensor]]:
        """
        For each edge type:
          - Grab edge_index.
          - Map edge_attr [sentiment, decay] → learned weight w ∈ (0,1).
          - Optionally apply edge dropout during training.
        """
        edge_index_dict: Dict[EdgeType, Tensor] = {}
        edge_weight_dict: Dict[EdgeType, Tensor] = {}

        for et in self.edge_types:
            ei: Tensor = data[et].edge_index
            ea: Optional[Tensor] = getattr(data[et], "edge_attr", None)

            if ea is None:
                w = torch.ones(ei.size(1), device=ei.device)
            else:
                w = self.edge_gate(self.edge_mixer(ea)).squeeze(-1)  # [E]

            if self.training and self.edge_dropout > 0.0 and ei.numel() > 0:
                ei, edge_mask = dropout_edge(ei, p=self.edge_dropout)
                # Apply the same mask to edge weights
                w = w[edge_mask] if w is not None else None

            edge_index_dict[et] = ei
            edge_weight_dict[et] = w

        return edge_index_dict, edge_weight_dict

    def _primary_company_pool(self, x: Tensor, batch: Tensor, mask: Optional[Tensor]) -> Tensor:
        """
        Primary-ticker-aware pooling:
          - If a boolean mask `mask` is provided (same length as x) and any True per graph,
            compute masked mean per graph.
          - Else fallback to global mean pool.
        """
        if mask is None or mask.dtype != torch.bool or mask.numel() != x.size(0):
            return global_mean_pool(x, batch)

        # Sum embeddings and counts per graph for masked nodes
        weights = mask.float().unsqueeze(-1)          # [N,1]
        sum_x = global_add_pool(x * weights, batch)   # [B, H]
        cnt = global_add_pool(weights.squeeze(-1), batch).clamp_min(1.0).unsqueeze(-1)  # [B,1]
        # If a graph has zero masked nodes, this reduces to mean of zeros / 1 → zeros.
        # Fallback: for those graphs, replace with unmasked mean.
        mean_masked = sum_x / cnt

        # Detect zero-count graphs and fill with global mean for those graphs
        has_mask = (cnt.squeeze(-1) > 1e-6)  # [B]
        fallback = global_mean_pool(x, batch)
        return torch.where(has_mask.unsqueeze(-1), mean_masked, fallback)

    def _readout(self, x_dict: Dict[str, Tensor], data: HeteroData) -> Tensor:
        """
        Pools node embeddings to a single per-graph vector according to self.readout.
        """
        fact_pool = global_mean_pool(x_dict["fact"], data["fact"].batch)

        # Company pool can be primary-aware
        comp_batch = data["company"].batch
        primary_mask = getattr(data["company"], "primary_mask", None)  # optional: torch.bool [N_company]
        company_pool = self._primary_company_pool(x_dict["company"], comp_batch, primary_mask)

        if self.readout == "fact":
            return fact_pool
        if self.readout == "company":
            return company_pool
        if self.readout == "concat":
            return torch.cat([fact_pool, company_pool], dim=-1)

        # 'gated' readout
        both = torch.cat([fact_pool, company_pool], dim=-1)  # [B, 2H]
        alpha = torch.sigmoid(self.readout_gate(both))       # [B, 1]
        return alpha * fact_pool + (1 - alpha) * company_pool

    # -------------------------------
    # Forward
    # -------------------------------
    def forward(self, data: HeteroData) -> Tensor:
        """
        Args:
          data: a (possibly batched) HeteroData with:
            - data[nt].x            : node features per type (shape varies by type)
            - data[et].edge_index   : edges per relation, shape [2, E]
            - data[et].edge_attr    : edge features per relation, shape [E, 2] (sentiment, decay)
            - data[nt].batch        : graph id per node type (added by PyG DataLoader)
            - optional: data['company'].primary_mask (torch.bool) to bias readout to primary tickers
        Returns:
          logits: Tensor of shape [batch_size], one logit per graph (use with BCEWithLogitsLoss).
        """
        # Lazily build encoders when we see the actual dims
        self._maybe_build_type_encoders(data)

        # Type-specific projection to hidden space H
        x_dict = {nt: self.in_proj[nt](data[nt].x) for nt in self.node_types}

        # Build per-relation connectivity and edge weights
        edge_index_dict, edge_weight_dict = self._edge_dicts(data)

        # HeteroConv stack
        for conv in self.convs:
            # RIGHT
            x_dict = conv(x_dict, edge_index_dict, edge_weight_dict=edge_weight_dict)

            # ReLU → per-type LayerNorm → feature dropout
            x_dict = {
                nt: F.dropout(
                    self.post_norm[nt](F.relu(x)),
                    p=self.feature_dropout,
                    training=self.training
                )
                for nt, x in x_dict.items()
            }

        # Readout to a per-graph embedding
        pooled = self._readout(x_dict, data)
        pooled = F.dropout(pooled, p=self.final_dropout, training=self.training)

        # Classifier: [B, H] or [B, 2H] → [B]
        return self.classifier(pooled).squeeze(-1)
