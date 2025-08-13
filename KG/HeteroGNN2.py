import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GraphConv, global_mean_pool
from torch_geometric.data import HeteroData
from torch_geometric.utils import dropout_edge
from typing import Dict, Tuple, List, Optional, Literal

# An edge type in PyG hetero graphs is a 3-tuple: (src_node_type, relation_name, dst_node_type)
EdgeType = Tuple[str, str, str]
# Readout modes (how we aggregate node embeddings into a single graph embedding)
Readout = Literal["fact", "company", "concat", "gated"]

# ---------------------------
# Improved Edge Attribute Encoder
# ---------------------------
class EdgeAttrEncoder(nn.Module):
    def __init__(self, time_dim=8, lambda_decay=0.01):
        super().__init__()
        self.time_dim = time_dim
        self.lambda_decay = lambda_decay
        
        # Calculate MLP input dimension: [sentiment, log1p(delta_t), time2vec_with_linear]
        # time2vec includes: sin_terms + cos_terms + linear_term
        mlp_input_dim = 2 + (2 * (time_dim // 2) + 1)  # sentiment + log1p(delta_t) + time2vec
        
        # Improved MLP with better architecture
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 32),  # [sentiment, delta_t, time2vec]
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # gate in [0, 1]
        )

    def time2vec(self, delta_t):
        # Improved Time2Vec with scaling and linear term
        # Scale delta_t to prevent wild oscillations
        delta_t_scaled = torch.clamp(delta_t / 30.0, 0, 12)  # Scale to reasonable range
        
        # Classic Time2Vec: [sin(ωt), cos(ωt), sin(2ωt), cos(2ωt), ..., t]
        freqs = torch.arange(1, self.time_dim // 2 + 1, device=delta_t.device).float()
        sin_terms = torch.sin(delta_t_scaled * freqs)
        cos_terms = torch.cos(delta_t_scaled * freqs)
        
        # Always include linear term for better temporal modeling
        linear_term = delta_t_scaled
        
        # Combine all terms: sin_terms + cos_terms + linear_term
        # Ensure all tensors have the same number of dimensions
        tvec = torch.cat([sin_terms, cos_terms, linear_term], dim=-1)
        
        return tvec

    def forward(self, sentiment, delta_t):
        # Standardize input: expect [sentiment, delta_t] format
        # sentiment: [num_edges, 1] in [-1, 1]
        # delta_t: [num_edges, 1] in [0, inf)
        
        # Defensive clamping on sentiment
        sentiment = sentiment.clamp(min=-1.0, max=1.0)
        
        # Apply Time2Vec encoding
        tvec = self.time2vec(delta_t)
        
        # Normalize delta_t for MLP input (use same scaling as time2vec)
        delta_t_n = torch.log1p(delta_t)  # log(1 + delta_t) for bounded/monotone transform
        
        # Concatenate all features: [sentiment, log1p(delta_t), time2vec]
        x = torch.cat([sentiment, delta_t_n, tvec], dim=-1)
        
        # Generate edge weight gate
        gate = self.mlp(x)  # [num_edges, 1]
        return gate


# ---------------------------
# Improved Heterogeneous GNN
# ---------------------------
class HeteroGNN2(nn.Module):
    """
    Improved Heterogeneous GNN with proper edge weight handling and temporal encoding.
    Addresses all CPT-5 feedback issues.
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
        lambda_decay: float = 0.01,
        use_residual: bool = True,
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
        self.use_residual = use_residual

        # Node type-specific input projections (lazy initialization)
        self.in_proj = nn.ModuleDict()
        
        # Edge attribute encoder with configurable lambda_decay
        self.edge_encoder = EdgeAttrEncoder(time_dim=time_dim, lambda_decay=lambda_decay)

        # Message passing layers with GraphConv (supports edge weights)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                et: GraphConv((-1, -1), hidden_channels, aggr='add')
                for et in self.edge_types
            }, aggr='sum')
            self.convs.append(conv)

        # Post-conv normalization (single normalization as recommended)
        self.post_norm = nn.ModuleDict()

        # Readout-specific params
        if self.readout == "gated":
            self.readout_gate = nn.Linear(2 * hidden_channels, 1)

        # Improved classifier head: 2-layer MLP
        cls_in = hidden_channels if self.readout in ("fact", "company", "gated") else 2 * hidden_channels
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, 1)
        )

    def _maybe_build_type_encoders(self, data: HeteroData) -> None:
        """
        Lazily create per-type encoders and post LayerNorms based on input dims.
        Uses direct data access (standard PyG DataLoader).
        """
        for nt in self.node_types:
            if nt in self.in_proj:
                continue
            
            # Direct access to node features (standard PyG DataLoader)
            d_in = int(data[nt].x.size(-1))
            
            # Input projection (removed LayerNorm as recommended)
            mlp = nn.Sequential(
                nn.Linear(d_in, self.hidden_channels),
                nn.ReLU(),
            )
            self.in_proj[nt] = mlp
            
            # Post-conv LayerNorm (single normalization)
            self.post_norm[nt] = nn.LayerNorm(self.hidden_channels)

    def _edge_dicts(self, data: HeteroData) -> Tuple[Dict[EdgeType, torch.Tensor], Dict[EdgeType, torch.Tensor]]:
        """
        Build edge indices and weights for each edge type.
        Fixed to properly handle edge weights with GraphConv.
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

            # Extract sentiment and decay from edge attributes
            sentiment = ea[:, 0:1]  # [E, 1]
            decay = ea[:, 1:2]      # [E, 1]
            
            # Handle both decay and delta_t formats
            # Check if second column is decay [0,1] or raw delta_t
            if ea.size(0) > 0:  # Only check if there are edges
                if decay.max() <= 1.0 and decay.min() >= 0.0:
                    # Input is decay: convert to delta_t
                    delta_t = -torch.log(decay.clamp(min=1e-6)) / self.edge_encoder.lambda_decay
                else:
                    # Input is already delta_t
                    delta_t = decay
            else:
                # No edges, use default delta_t
                delta_t = decay
            
            # Ensure delta_t is nonnegative before Time2Vec
            delta_t = delta_t.clamp(min=0.0)
            
            # Generate edge weights using the encoder
            w = self.edge_encoder(sentiment, delta_t).squeeze(-1)  # [E]

            # Apply edge dropout if enabled
            if self.training and self.edge_dropout > 0.0 and ei.numel() > 0:
                ei, edge_mask = dropout_edge(ei, p=self.edge_dropout)
                # Apply the same mask to edge weights
                w = w[edge_mask]

            edge_index_dict[et] = ei
            edge_weight_dict[et] = w

        return edge_index_dict, edge_weight_dict

    def _readout(self, x_dict: Dict[str, torch.Tensor], data: HeteroData) -> torch.Tensor:
        """
        Pools node embeddings to a single per-graph vector according to self.readout.
        Simplified to use direct data access with fallbacks.
        """
        # Get batch information using direct access (standard PyG DataLoader)
        try:
            fact_batch = data['fact'].batch
            comp_batch = data['company'].batch
        except AttributeError:
            # Fallback for test data or single graphs
            fact_batch = torch.zeros(x_dict["fact"].size(0), dtype=torch.long, device=x_dict["fact"].device)
            comp_batch = torch.zeros(x_dict["company"].size(0), dtype=torch.long, device=x_dict["company"].device)

        # Pool node embeddings
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
            - data[et].edge_attr    : edge features per relation, shape [E, 2] (sentiment, decay/delta_t)
            - data[nt].batch        : graph id per node type (added by PyG DataLoader)
        Returns:
          logits: Tensor of shape [batch_size], one logit per graph (use with BCEWithLogitsLoss).
        """
        # Lazily build encoders when we see the actual dims
        self._maybe_build_type_encoders(data)

        # Type-specific projection to hidden space H
        x_dict = {}
        for nt in self.node_types:
            # Direct access to node features (standard PyG DataLoader)
            x_dict[nt] = self.in_proj[nt](data[nt].x)

        # Build per-relation connectivity and edge weights
        edge_index_dict, edge_weight_dict = self._edge_dicts(data)

        # HeteroConv stack with proper edge weight passing
        for i, conv in enumerate(self.convs):
            # Store previous features for residual connections
            if self.use_residual:
                prev = {nt: x.clone() for nt, x in x_dict.items()}
            
            # Pass edge weights correctly with GraphConv using edge_weight_dict
            x_dict = conv(x_dict, edge_index_dict, edge_weight_dict=edge_weight_dict)

            # ReLU first
            x_dict = {nt: F.relu(x) for nt, x in x_dict.items()}
            
            # Add residual connections if enabled (before LayerNorm for stability)
            if self.use_residual:
                x_dict = {
                    nt: x + prev[nt] if nt in prev else x
                    for nt, x in x_dict.items()
                }
            
            # LayerNorm after residual
            x_dict = {
                nt: self.post_norm[nt](x)
                for nt, x in x_dict.items()
            }
            
            # Feature dropout
            x_dict = {
                nt: F.dropout(x, p=self.feature_dropout, training=self.training)
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
