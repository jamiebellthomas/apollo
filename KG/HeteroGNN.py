from typing import Dict, Tuple, List, Optional, Literal
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, global_mean_pool
from torch_geometric.data import HeteroData
from torch_geometric.utils import dropout_adj

# An edge type in PyG hetero graphs is a 3-tuple: (src_node_type, relation_name, dst_node_type)
EdgeType = Tuple[str, str, str]

# Readout modes we allow (how we aggregate node embeddings into a single graph embedding)
Readout = Literal["fact", "company", "concat", "gated"]


class HeteroGNN(nn.Module):
    """
    Heterogeneous GCN with:
      - A tiny learned gate that turns each edge's [sentiment, decay] into a safe edge weight in [0,1].
      - HeteroConv stack using GCNConv per edge relation (supports scalar edge_weight).
      - Dropout on node features after each layer (feature_dropout).
      - Optional edge dropout (randomly remove edges during training).
      - Optional final dropout before the classifier.
      - Flexible readout strategy over node types: 'fact' | 'company' | 'concat' | 'gated'.

    High-level picture of the forward pass:
      1) Build per-relation edge_index and a learned per-edge weight w ∈ [0,1].
      2) Run message passing with HeteroConv (which applies GCNConv per relation).
      3) After each layer: ReLU + dropout on node features.
      4) Readout (pool) node embeddings to a per-graph vector (based on the chosen strategy).
      5) Optional final dropout, then a Linear layer → 1 logit per graph (for BCEWithLogitsLoss).

    Why the learned gate?
      - Your raw edge_attr = [sentiment ∈ [-1,1], decay ∈ [0,1]].
      - Passing negative weights directly into GCNConv can be unstable.
      - We learn a small mapping [sentiment, decay] -> sigmoid -> [0,1].
      - This keeps GCN stable (non-negative weights) and lets the model learn how to mix both signals.
    """

    def __init__(
        self,
        metadata: Tuple[List[str], List[EdgeType]],  # (node_types, edge_types) from any HeteroData instance
        hidden_channels: int = 64,                   # size of hidden node embeddings produced by each GCNConv
        num_layers: int = 2,                         # how many HeteroConv( GCNConv per relation ) blocks to stack
        feature_dropout: float = 0.3,                # dropout prob applied to node features after each layer
        edge_dropout: float = 0.0,                   # dropout prob for randomly removing edges during training
        final_dropout: float = 0.0,                  # dropout prob applied to pooled graph embedding before classifier
        readout: Readout = "fact",                   # which node types to pool: 'fact', 'company', 'concat', or 'gated'
    ) -> None:
        super().__init__()

        # Unpack graph meta (kept so the module can operate on any batch with same schema)
        self.node_types, self.edge_types = metadata

        # Store dropout settings and readout mode
        self.feature_dropout = feature_dropout
        self.edge_dropout = edge_dropout
        self.final_dropout = final_dropout
        self.readout: Readout = readout

        # -----------------------------
        # 1) Per-edge learned gate
        # -----------------------------
        # edge_mixer: a tiny linear layer that maps the 2D edge_attr [sentiment, decay] -> R
        # edge_gate: sigmoid squashes that scalar into [0,1] so weights are safe for GCNConv
        self.edge_mixer = nn.Linear(2, 1)
        self.edge_gate  = nn.Sigmoid()

        # -----------------------------
        # 2) Heterogeneous message passing stack
        # -----------------------------
        # We create `num_layers` HeteroConv blocks.
        # Each HeteroConv holds a dict {edge_type: GCNConv}, and aggregates their results (sum).
        #
        # GCNConv((-1, -1), hidden_channels):
        #   - (-1, -1) tells PyG to infer input dimensions for (src, dst) node types at first forward.
        #   - This is critical in hetero graphs because different node types can have different feature sizes.
        self.convs = nn.ModuleList([
            HeteroConv(
                {et: GCNConv((-1, -1), hidden_channels) for et in self.edge_types},
                aggr="sum",  # sum relation-specific messages into each node's representation
            )
            for _ in range(num_layers)
        ])

        # -----------------------------
        # 3) Readout-specific parameters
        # -----------------------------
        # 'gated' readout learns a scalar α ∈ [0,1] to blend fact/company pooled vectors.
        if self.readout == "gated":
            # Input size is fact_pool (H) + company_pool (H) -> 2H, output is 1 scalar
            self.readout_gate = nn.Linear(2 * hidden_channels, 1)

        # Classifier input size depends on readout:
        #   - 'fact'/'company'/'gated' produce an H-dim vector
        #   - 'concat' produces a 2H-dim vector
        cls_in = (
            hidden_channels if self.readout in ("fact", "company", "gated")
            else 2 * hidden_channels  # 'concat'
        )

        # Final graph-level classifier: maps pooled embedding -> 1 logit
        # We keep it simple on purpose; for richer heads you could add MLP layers.
        self.classifier = nn.Linear(cls_in, 1)

    # ----------------------------------------------------
    # Helper: build per-relation edge_index and edge_weight
    # ----------------------------------------------------
    def _edge_dicts(
        self, data: HeteroData
    ) -> Tuple[Dict[EdgeType, Tensor], Dict[EdgeType, Tensor]]:
        """
        For each edge type:
          - Grab edge_index (connectivity).
          - Turn edge_attr [sentiment, decay] into a learned weight w ∈ [0,1].
          - Optionally apply edge dropout (randomly drop edges during training).
        Returns:
          edge_index_dict: mapping edge_type -> (2, E') tensor of edges (maybe fewer after dropout)
          edge_weight_dict: mapping edge_type -> (E',) scalar weights aligned with edge_index
        """
        edge_index_dict: Dict[EdgeType, Tensor] = {}
        edge_weight_dict: Dict[EdgeType, Tensor] = {}

        for et in self.edge_types:
            # Connectivity tensor: shape [2, num_edges]
            ei: Tensor = data[et].edge_index

            # Edge attributes: expected shape [num_edges, 2] = [sentiment, decay]
            # Could be None if you ever omitted them; we handle that gracefully.
            ea: Optional[Tensor] = data[et].edge_attr

            if ea is None:
                # No edge features → default to weight 1.0 for all edges
                w = torch.ones(ei.size(1), device=ei.device)
            else:
                # Learned per-edge weight in [0,1]:
                #   w_raw = edge_mixer([sentiment, decay]) ∈ R
                #   w = sigmoid(w_raw) ∈ (0,1), stable for GCNConv
                w = self.edge_gate(self.edge_mixer(ea)).squeeze(-1)  # shape [num_edges]

            # Edge dropout: randomly removes edges & their weights during training only.
            # This is a regularizer; it forces the model not to rely too heavily on any single edge.
            if self.training and self.edge_dropout > 0.0 and ei.numel() > 0:
                # dropout_adj returns a pruned edge_index and corresponding (pruned) edge weights
                ei, w = dropout_adj(
                    ei,
                    w,
                    p=self.edge_dropout,
                    training=True,
                    num_nodes=None  # PyG can infer num nodes; you can set explicitly if you like
                )

            # Store results for this relation
            edge_index_dict[et] = ei
            edge_weight_dict[et] = w

        return edge_index_dict, edge_weight_dict

    # ----------------------------------------------------
    # Helper: readout (pooling over nodes → one vector per graph)
    # ----------------------------------------------------
    def _readout(self, x_dict: Dict[str, Tensor], data: HeteroData) -> Tensor:
        """
        Pools node embeddings to a single per-graph vector according to self.readout:
          - 'fact'   : use only 'fact' nodes (evidence-centric)
          - 'company': use only 'company' nodes
          - 'concat' : concatenate both pools [fact_pool, company_pool]
          - 'gated'  : learn α in [0,1] and return α*fact_pool + (1-α)*company_pool
        Shapes:
          fact_pool, comp_pool: [batch_size, hidden_channels]
          return:
            - 'fact'/'company'/'gated' -> [batch_size, hidden_channels]
            - 'concat'                  -> [batch_size, 2*hidden_channels]
        """
        # Global mean pool groups nodes by their graph membership (batch vector auto-added by DataLoader)
        fact_pool = global_mean_pool(x_dict["fact"],    data["fact"].batch)
        comp_pool = global_mean_pool(x_dict["company"], data["company"].batch)

        if self.readout == "fact":
            return fact_pool
        if self.readout == "company":
            return comp_pool
        if self.readout == "concat":
            # Concatenate along feature dimension
            return torch.cat([fact_pool, comp_pool], dim=-1)

        # 'gated' readout: learn a scalar gate α ∈ [0,1] from both pools,
        # then blend them: α * fact + (1-α) * company. Lets the model decide the mix.
        both = torch.cat([fact_pool, comp_pool], dim=-1)  # [B, 2H]
        alpha = torch.sigmoid(self.readout_gate(both))    # [B, 1] in (0,1)
        return alpha * fact_pool + (1 - alpha) * comp_pool  # [B, H]

    # ----------------------------------------------------
    # Forward pass: the whole pipeline
    # ----------------------------------------------------
    def forward(self, data: HeteroData) -> Tensor:
        """
        Args:
          data: a (possibly batched) HeteroData with:
            - data[nt].x            : node features per type (shape varies by type)
            - data[et].edge_index   : edges per relation, shape [2, E]
            - data[et].edge_attr    : edge features per relation, shape [E, 2] (sentiment, decay)
            - data[nt].batch        : graph id per node type (added automatically by PyG DataLoader)
        Returns:
          logits: Tensor of shape [batch_size], one logit per graph.
                  Use with BCEWithLogitsLoss for binary classification.
        """
        # 1) Gather input node features per node type into a dict
        #    e.g., x_dict["fact"] is [num_fact_nodes_in_batch, d_fact]
        #          x_dict["company"] is [num_company_nodes_in_batch, d_company]
        x_dict = {nt: data[nt].x for nt in self.node_types}

        # 2) Build per-relation edge_index and a per-edge weight in [0,1]
        edge_index_dict, edge_weight_dict = self._edge_dicts(data)

        # 3) Message passing: for each HeteroConv layer
        for conv in self.convs:
            # HeteroConv applies each relation's GCNConv, then sums the results into each node type.
            # We pass the same edge_index_dict and edge_weight_dict we just prepared.
            x_dict = conv(x_dict, edge_index_dict, edge_weight_dict=edge_weight_dict)

            # Non-linearity + feature dropout per node type (regularization)
            x_dict = {
                nt: F.dropout(F.relu(x), p=self.feature_dropout, training=self.training)
                for nt, x in x_dict.items()
            }

        # 4) Readout: turn node embeddings into a single graph embedding per example in the batch
        pooled = self._readout(x_dict, data)

        # 5) Optional final dropout before classification (regularization on the graph embedding)
        pooled = F.dropout(pooled, p=self.final_dropout, training=self.training)

        # 6) Classifier: map pooled embedding -> single logit per graph
        #    (We squeeze the trailing dim so the shape is [B] instead of [B,1].)
        return self.classifier(pooled).squeeze(-1)
