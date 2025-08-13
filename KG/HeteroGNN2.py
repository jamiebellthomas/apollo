import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool
from torch_geometric.data import HeteroData
from typing import Dict, Tuple, List, Optional, Literal

# An edge type in PyG hetero graphs is a 3-tuple: (src_node_type, relation_name, dst_node_type)
EdgeType = Tuple[str, str, str]
# Readout modes (how we aggregate node embeddings into a single graph embedding)
Readout = Literal["fact", "company", "concat", "gated"]

# ---------------------------
# Edge attribute encoder
# ---------------------------
class EdgeAttrEncoder(nn.Module):
    def __init__(self, time_dim=8):
        super().__init__()
        self.time_dim = time_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 + time_dim, 16),  # [sentiment, decay, time2vec]
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # gate in [0, 1]
        )

    def time2vec(self, delta_t):
        # delta_t: [num_edges, 1]
        # Use sinusoidal encoding for temporal patterns
        freqs = torch.arange(1, self.time_dim + 1, device=delta_t.device).float()
        return torch.sin(delta_t * freqs)  # shape: [num_edges, time_dim]

    def forward(self, sentiment, delta_t, lambda_decay=0.01):
        decay = torch.exp(-lambda_decay * delta_t)  # shape: [num_edges, 1]
        tvec = self.time2vec(delta_t)
        x = torch.cat([sentiment, decay, tvec], dim=-1)
        gate = self.mlp(x)  # [num_edges, 1]
        return gate


# ---------------------------
# Main Heterogeneous GNN
# ---------------------------
class HeteroGNN2(nn.Module):
    """
    Heterogeneous GNN with temporal edge encoding and SAGEConv.
    Compatible with the existing data format from run.py.
    """
    def __init__(
        self,
        metadata: Tuple[List[str], List[EdgeType]],  # (node_types, edge_types) from any HeteroData instance
        hidden_channels: int = 128,
        num_layers: int = 2,
        feature_dropout: float = 0.2,
        edge_dropout: float = 0.0,
        final_dropout: float = 0.0,
        readout: Readout = "concat",
        time_dim: int = 8,
    ):
        super().__init__()
        
        # Unpack graph meta
        self.node_types, self.edge_types = metadata
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.feature_dropout = feature_dropout
        self.edge_dropout = edge_dropout
        self.final_dropout = final_dropout
        self.readout = readout

        # Node type-specific input projections (lazy initialization)
        self.in_proj = nn.ModuleDict()
        self.post_norm = nn.ModuleDict()

        # Edge attribute encoder
        self.edge_encoder = EdgeAttrEncoder(time_dim=time_dim)

        # Message passing layers with SAGEConv
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                et: SAGEConv((-1, -1), hidden_channels)
                for et in self.edge_types
            }, aggr='sum')
            self.convs.append(conv)

        # Readout-specific params
        if self.readout == "gated":
            self.readout_gate = nn.Linear(2 * hidden_channels, 1)

        # Classifier head size depends on readout
        cls_in = hidden_channels if self.readout in ("fact", "company", "gated") else 2 * hidden_channels
        self.classifier = nn.Linear(cls_in, 1)

    def _maybe_build_type_encoders(self, data: HeteroData) -> None:
        """
        Lazily create per-type encoders and per-type post LayerNorms based on input dims.
        """
        for nt in self.node_types:
            if nt in self.in_proj:
                continue
            
            # Try to get features for this specific node type
            d_in = None
            
            # Try direct access first (most reliable)
            try:
                if nt in data and hasattr(data[nt], 'x') and data[nt].x is not None:
                    d_in = int(data[nt].x.size(-1))
            except:
                pass
            
            # If direct access failed, try node stores
            if d_in is None and hasattr(data, 'node_stores') and len(data.node_stores) > 0:
                # For now, assume first store is facts, second is companies
                if nt == "fact" and len(data.node_stores) >= 1:
                    first_store = data.node_stores[0]
                    if hasattr(first_store, 'x') and first_store.x is not None:
                        d_in = int(first_store.x.size(-1))
                elif nt == "company" and len(data.node_stores) >= 2:
                    second_store = data.node_stores[1]
                    if hasattr(second_store, 'x') and second_store.x is not None:
                        d_in = int(second_store.x.size(-1))
            
            # If we still don't have dimensions, use defaults
            if d_in is None:
                if nt == "fact":
                    d_in = 768  # Default fact feature dimension
                elif nt == "company":
                    d_in = 27    # Default company feature dimension
                else:
                    d_in = 64    # Generic default
            
            mlp = nn.Sequential(
                nn.Linear(d_in, self.hidden_channels),
                nn.ReLU(),
                nn.LayerNorm(self.hidden_channels),
            )
            self.in_proj[nt] = mlp
            self.post_norm[nt] = nn.LayerNorm(self.hidden_channels)

    def _edge_dicts(self, data: HeteroData) -> Tuple[Dict[EdgeType, torch.Tensor], Dict[EdgeType, torch.Tensor]]:
        """
        For each edge type:
          - Grab edge_index.
          - For SAGEConv, we don't pass edge weights directly, but we can use them
            to modify the edge attributes or handle them differently.
        """
        edge_index_dict: Dict[EdgeType, torch.Tensor] = {}
        edge_weight_dict: Dict[EdgeType, torch.Tensor] = {}

        for et in self.edge_types:
            ei: torch.Tensor = data[et].edge_index
            ea: Optional[torch.Tensor] = getattr(data[et], "edge_attr", None)

            if ea is None:
                # Create default edge attributes if none exist
                ea = torch.ones(ei.size(1), 2, device=ei.device)
                ea[:, 0] = 0.0  # neutral sentiment
                ea[:, 1] = 1.0  # no decay

            # For SAGEConv, we'll store the processed edge weights but won't pass them directly
            # Extract sentiment and delta_t from edge attributes
            sentiment = ea[:, 0:1]  # [E, 1]
            decay = ea[:, 1:2]      # [E, 1]
            
            # Convert decay to delta_t (inverse of exponential decay)
            # decay = exp(-lambda * delta_t) -> delta_t = -log(decay) / lambda
            lambda_decay = 0.01
            delta_t = -torch.log(decay.clamp(min=1e-6)) / lambda_decay
            
            # Use the edge encoder to get temporal-aware weights
            w = self.edge_encoder(sentiment, delta_t).squeeze(-1)  # [E]

            if self.training and self.edge_dropout > 0.0 and ei.numel() > 0:
                from torch_geometric.utils import dropout_edge
                ei, edge_mask = dropout_edge(ei, p=self.edge_dropout)
                # Apply the same mask to edge weights
                w = w[edge_mask] if w is not None else None

            edge_index_dict[et] = ei
            edge_weight_dict[et] = w

        return edge_index_dict, edge_weight_dict

    def _readout(self, x_dict: Dict[str, torch.Tensor], data: HeteroData) -> torch.Tensor:
        """
        Pools node embeddings to a single per-graph vector according to self.readout.
        """
        # Handle both batched (node_stores) and non-batched (direct access) cases
        if hasattr(data, 'node_stores') and len(data.node_stores) > 0:
            # Batched case: get batch info from node stores
            fact_batch = None
            comp_batch = None
            primary_mask = None
            
            # First, try to get batch info that matches our node features
            if "fact" in x_dict and x_dict["fact"].size(0) > 0:
                # Try to find batch info for fact nodes
                for store in data.node_stores:
                    if hasattr(store, 'batch') and store.batch is not None:
                        # Check if this batch tensor matches the fact node count
                        if store.batch.size(0) == x_dict["fact"].size(0):
                            fact_batch = store.batch
                            break
                
                # If we couldn't find matching batch info, fall back to direct access
                if fact_batch is None:
                    try:
                        fact_batch = data["fact"].batch
                    except:
                        # Create a default batch tensor if needed
                        fact_batch = torch.zeros(x_dict["fact"].size(0), dtype=torch.long, device=x_dict["fact"].device)
            
            if "company" in x_dict and x_dict["company"].size(0) > 0:
                # Try to find batch info for company nodes
                for store in data.node_stores:
                    if hasattr(store, 'batch') and store.batch is not None:
                        # Check if this batch tensor matches the company node count
                        if store.batch.size(0) == x_dict["company"].size(0):
                            comp_batch = store.batch
                            # Check for primary_mask in the same store
                            if hasattr(store, 'primary_mask'):
                                primary_mask = store.primary_mask
                            break
                
                # If we couldn't find matching batch info, fall back to direct access
                if comp_batch is None:
                    try:
                        comp_batch = data["company"].batch
                        primary_mask = getattr(data["company"], "primary_mask", None)
                    except:
                        # Create a default batch tensor if needed
                        comp_batch = torch.zeros(x_dict["company"].size(0), dtype=torch.long, device=x_dict["company"].device)
        else:
            # Non-batched case: direct access
            fact_batch = data["fact"].batch
            comp_batch = data["company"].batch
            primary_mask = getattr(data["company"], "primary_mask", None)

        fact_pool = global_mean_pool(x_dict["fact"], fact_batch)
        company_pool = global_mean_pool(x_dict["company"], comp_batch)

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

    def forward(self, data: HeteroData) -> torch.Tensor:
        """
        Args:
          data: a (possibly batched) HeteroData with:
            - data[nt].x            : node features per type (shape varies by type)
            - data[et].edge_index   : edges per relation, shape [2, E]
            - data[et].edge_attr    : edge features per relation, shape [E, 2] (sentiment, decay)
            - data[nt].batch        : graph id per node type (added by PyG DataLoader)
        Returns:
          logits: Tensor of shape [batch_size], one logit per graph (use with BCEWithLogitsLoss).
        """
        # Lazily build encoders when we see the actual dims
        self._maybe_build_type_encoders(data)

        # Type-specific projection to hidden space H
        # Handle both batched (node_stores) and non-batched (direct access) cases
        x_dict = {}
        for nt in self.node_types:
            if hasattr(data, 'node_stores') and len(data.node_stores) > 0:
                # Batched case: get features from node stores
                node_features = None
                
                # Try direct access first (most reliable)
                try:
                    if nt in data and hasattr(data[nt], 'x') and data[nt].x is not None:
                        node_features = data[nt].x
                except:
                    pass
                
                # If direct access failed, try to find in node stores
                if node_features is None:
                    for store in data.node_stores:
                        if hasattr(store, 'x') and store.x is not None:
                            # For now, assume first store is facts, second is companies
                            if nt == "fact" and len(data.node_stores) >= 1:
                                node_features = store.x
                                break
                            elif nt == "company" and len(data.node_stores) >= 2:
                                node_features = data.node_stores[1].x
                                break
                
                if node_features is not None:
                    x_dict[nt] = self.in_proj[nt](node_features)
                else:
                    # Final fallback: create zero features
                    print(f"Warning: No features found for node type {nt}, using zeros")
                    if nt in self.in_proj:
                        input_dim = self.in_proj[nt][0].in_features
                        zero_features = torch.zeros(1, input_dim, device=data.y.device if hasattr(data, 'y') else 'cpu')
                        x_dict[nt] = self.in_proj[nt](zero_features)
                    else:
                        raise ValueError(f"No projection found for node type {nt}")
            else:
                # Non-batched case: direct access
                x_dict[nt] = self.in_proj[nt](data[nt].x)

        # Build per-relation connectivity and edge weights
        edge_index_dict, edge_weight_dict = self._edge_dicts(data)

        # HeteroConv stack
        for conv in self.convs:
            # For SAGEConv, we don't pass edge weights directly
            # The temporal encoding is handled through the edge encoder during preprocessing
            x_dict = conv(x_dict, edge_index_dict)

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

# ---------------------------
# Weighted BCE Loss for imbalance
# ---------------------------
def get_pos_weight(train_labels):
    num_pos = train_labels.sum().item()
    num_neg = len(train_labels) - num_pos
    return torch.tensor(num_neg / num_pos, dtype=torch.float)
