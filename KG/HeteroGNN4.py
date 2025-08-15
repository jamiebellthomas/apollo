from typing import Dict, Tuple, List, Optional, Literal
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, global_mean_pool, global_add_pool
from torch_geometric.data import HeteroData
from torch_geometric.utils import dropout_edge, softmax as pyg_softmax

EdgeType = Tuple[str, str, str]
Readout = Literal["fact", "company", "concat", "gated"]

class TimeEdgeBuilder(nn.Module):
    """
    Builds edge_attr for attention:
      input columns can be [sentiment, decay] or [sentiment, delta_t]
      output edge_attr = [sentiment, decay, time2vec(log1p(delta_t))] with dimension (2 + time_dim)
    """
    def __init__(self, time_dim: int = 8, lambda_decay: float = 0.01, t_scale: float = 30.0):
        super().__init__()
        self.time_dim = time_dim
        self.lambda_decay = nn.Parameter(torch.tensor(lambda_decay), requires_grad=True)  # learnable λ
        self.t_scale = t_scale

    def _time2vec(self, delta_t: Tensor) -> Tensor:
        # delta_t: [E, 1]
        t = torch.log1p(delta_t.clamp(min=0.0)) / self.t_scale  # stabilise and scale
        # Ensure freqs is on the same device as delta_t
        freqs = torch.arange(1, self.time_dim, device=delta_t.device, dtype=delta_t.dtype)  # [D-1]
        sin_terms = torch.sin(t * freqs)  # [E, time_dim-1]
        return torch.cat([sin_terms, t], dim=-1)  # [E, time_dim] (last column = linear term)

    def forward(self, ea: Optional[Tensor]) -> Tensor:
        # Expect ea shape [E, 2]: [:,0]=sentiment, [:,1]=decay OR delta_t
        if ea is None:
            raise ValueError("edge_attr required for attention with edge_dim. Provide sentiment and decay or delta_t.")
        
        # Ensure lambda_decay is on the same device as input
        lambda_decay = self.lambda_decay.to(ea.device)
        
        s = ea[:, 0:1].clamp(-1.0, 1.0)  # sentiment in [-1,1], clamp defensively
        second = ea[:, 1:2]

        looks_like_decay = ((second >= 0.0) & (second <= 1.0)).all()
        if looks_like_decay:
            decay = second.clamp(0.0, 1.0)
            # recover approx delta_t from decay using current λ (ok for consistency within training)
            delta_t = (-torch.log(decay.clamp(min=1e-6))) / lambda_decay.clamp(min=1e-6)
        else:
            delta_t = second.clamp(min=0.0)
            decay = torch.exp(-lambda_decay.clamp(min=1e-6) * delta_t)

        tvec = self._time2vec(delta_t)  # [E, time_dim]
        return torch.cat([s, decay, tvec], dim=-1)  # [E, 2 + time_dim]


class HeteroAttnGNN(nn.Module):
    """
    Heterogeneous attention GNN with edge-feature-aware attention (GATv2Conv with edge_dim),
    optional funnel mode (fact->company only), and optional top-k pre-gating per primary node.
    """
    def __init__(
        self,
        metadata: Tuple[List[str], List[EdgeType]],
        hidden_channels: int = 128,
        num_layers: int = 2,
        heads: int = 4,
        time_dim: int = 8,
        feature_dropout: float = 0.2,
        edge_dropout: float = 0.0,
        final_dropout: float = 0.1,
        readout: Readout = "gated",
        funnel_to_primary: bool = False,   # if True: only ('fact','mentions','company') relation is used
        topk_per_primary: Optional[int] = None  # if set, keep top-k incoming fact edges per primary before attention
    ):
        super().__init__()
        self.node_types, self.edge_types_full = metadata
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.heads = heads
        self.out_per_head = hidden_channels // heads
        assert hidden_channels % heads == 0, "hidden_channels must be divisible by heads"
        self.feature_dropout = feature_dropout
        self.edge_dropout = edge_dropout
        self.final_dropout = final_dropout
        self.readout: Readout = readout
        self.funnel_to_primary = funnel_to_primary
        self.topk_per_primary = topk_per_primary

        # restrict edge types if funnel mode
        if funnel_to_primary:
            self.edge_types = [et for et in self.edge_types_full if et == ('fact', 'mentions', 'company')]
            if len(self.edge_types) == 0:
                raise ValueError("Funnel mode expects ('fact','mentions','company') relation in metadata.")
            # Note: In funnel mode, facts won't receive messages (no company->fact edges)
            # This is intentional for pure "funnel" to company readout
        else:
            self.edge_types = self.edge_types_full

        # per-type input projections (lazy init on first forward)
        self.in_proj = nn.ModuleDict()
        self.post_norm = nn.ModuleDict()

        # temporal edge builder (produces edge_attr for attention)
        self.edge_builder = TimeEdgeBuilder(time_dim=time_dim)

        # build per-relation attention convs for each layer
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            convs: Dict[EdgeType, nn.Module] = {}
            for et in self.edge_types:
                # add self-loops only on homogeneous relations; here all are bipartite, so set add_self_loops=False
                convs[et] = GATv2Conv(
                    in_channels=(-1, -1),     # PyG will infer from x_dict
                    out_channels=self.out_per_head,
                    heads=self.heads,
                    edge_dim=2 + time_dim,    # [sentiment, decay] + time2vec
                    add_self_loops=False,
                    share_weights=False,      # safer for bipartite
                    dropout=0.0               # we’ll use feature-level dropout externally
                )
            self.layers.append(HeteroConv(convs, aggr="sum"))

        # readout
        if self.readout == "gated":
            self.readout_gate = nn.Linear(2 * hidden_channels, 1)

        # classifier
        cls_in = hidden_channels if self.readout in ("fact", "company", "gated") else 2 * hidden_channels
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, hidden_channels),
            nn.ReLU(),
            nn.Dropout(final_dropout),
            nn.Linear(hidden_channels, 1),
        )

    def _maybe_build_type_encoders(self, data: HeteroData) -> None:
        """
        Lazily create per-type encoders and per-type post LayerNorms based on input dims.
        """
        # Get device from model parameters
        device = next(self.parameters()).device
        
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
                # This is a temporary fix - we need a more robust way to identify node types
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
                    d_in = 768  # Default fact feature dimension (sentence transformer)
                elif nt == "company":
                    d_in = 27    # Default company feature dimension
                else:
                    d_in = 64    # Generic default
            
            mlp = nn.Sequential(
                nn.Linear(d_in, self.hidden_channels),
                nn.ReLU(),
            )
            self.in_proj[nt] = mlp
            self.post_norm[nt] = nn.LayerNorm(self.hidden_channels)
            
            # Move the newly created layers to the correct device
            self.in_proj[nt].to(device)
            self.post_norm[nt].to(device)

    def _build_edge_inputs(
        self, data: HeteroData
    ) -> Tuple[Dict[EdgeType, Tensor], Dict[EdgeType, Tensor], Dict[EdgeType, Tensor]]:
        edge_index_dict: Dict[EdgeType, Tensor] = {}
        edge_attr_for_attn: Dict[EdgeType, Tensor] = {}
        kept_mask_dict: Dict[EdgeType, Tensor] = {}

        # Get device from model parameters
        device = next(self.parameters()).device

        for et in self.edge_types:
            store = data[et]
            ei: Tensor = store.edge_index
            ea: Optional[Tensor] = getattr(store, "edge_attr", None)
            
            # Ensure tensors are on the correct device
            if ei.device != device:
                ei = ei.to(device)
            if ea is not None and ea.device != device:
                ea = ea.to(device)

            # build attention edge_attr
            ea_attn = self.edge_builder(ea)  # [E, 2+time_dim]

            # optional edge dropout (structure-level)
            if self.training and self.edge_dropout > 0.0 and ei.numel() > 0:
                ei, drop_mask = dropout_edge(ei, p=self.edge_dropout, force_undirected=False)
                ea_attn = ea_attn[drop_mask]
                kept_mask = drop_mask
            else:
                kept_mask = torch.ones(ei.size(1), dtype=torch.bool, device=device)

            # optional top-k pre-gating per primary for fact->company only
            if self.topk_per_primary is not None and et == ('fact', 'mentions', 'company') and ei.numel() > 0:
                # score edges by a simple scalar from edge_attr to rank (you can replace with a small MLP)
                # here we use decay * sigmoid(α * sentiment) as a heuristic score
                s = ea_attn[:, 0:1]              # sentiment
                decay = ea_attn[:, 1:2]          # decay
                score = decay * torch.sigmoid(2.0 * s)  # [E,1]
                score = score.squeeze(-1)

                # select top-k edges per destination (company) node
                dst = ei[1]  # indices into company nodes
                # compute rank per edge within its dst group
                _, sort_idx = torch.sort(score, descending=True)  # rank directly on score, not softmax
                # accumulate counts per dst in descending order
                keep = torch.zeros_like(score, dtype=torch.bool, device=device)
                counts = {}
                for idx in sort_idx.tolist():
                    d = int(dst[idx].item())
                    c = counts.get(d, 0)
                    if c < self.topk_per_primary:
                        keep[idx] = True
                        counts[d] = c + 1
                # apply keep
                ei = ei[:, keep]
                ea_attn = ea_attn[keep]
                # combine masks: keep indexes the post-dropout sequence, so map back to original indices
                km = torch.zeros_like(kept_mask, device=device)
                km[kept_mask.nonzero().view(-1)[keep]] = True
                kept_mask = km

            edge_index_dict[et] = ei
            edge_attr_for_attn[et] = ea_attn
            kept_mask_dict[et] = kept_mask

        return edge_index_dict, edge_attr_for_attn, kept_mask_dict



    def _primary_company_pool(self, x: Tensor, batch: Tensor, mask: Optional[Tensor]) -> Tensor:
        if mask is None or mask.dtype != torch.bool or mask.numel() != x.size(0):
            return global_mean_pool(x, batch)
        weights = mask.float().unsqueeze(-1)
        sum_x = global_add_pool(x * weights, batch)
        cnt = global_add_pool(weights.squeeze(-1), batch).clamp_min(1.0).unsqueeze(-1)
        mean_masked = sum_x / cnt
        has_mask = (cnt.squeeze(-1) > 1e-6)
        fallback = global_mean_pool(x, batch)
        return torch.where(has_mask.unsqueeze(-1), mean_masked, fallback)

    def _readout(self, x_dict: Dict[str, Tensor], data: HeteroData) -> Tensor:
        fact_pool = global_mean_pool(x_dict["fact"], data["fact"].batch)
        company_pool = self._primary_company_pool(
            x_dict["company"], data["company"].batch, getattr(data["company"], "primary_mask", None)
        )
        if self.readout == "fact":
            return fact_pool
        if self.readout == "company":
            return company_pool
        if self.readout == "concat":
            return torch.cat([fact_pool, company_pool], dim=-1)
        both = torch.cat([fact_pool, company_pool], dim=-1)
        alpha = torch.sigmoid(self.readout_gate(both))
        return alpha * fact_pool + (1 - alpha) * company_pool

    def forward(self, data: HeteroData) -> Tensor:
        self._maybe_build_type_encoders(data)

        # Ensure all node features are on the same device/dtype as the model
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        
        x_dict = {nt: self.in_proj[nt](data[nt].x.to(device=device, dtype=dtype)) for nt in self.node_types}

        edge_index_dict, edge_attr_for_attn, _ = self._build_edge_inputs(data)

        for layer in self.layers:
            x_new = layer(
                x_dict,
                edge_index_dict,
                edge_attr_dict=edge_attr_for_attn  # HeteroConv forwards by kw name to each GATv2Conv
            )
            # light residual + norm + dropout
            x_dict = {
                nt: F.dropout(self.post_norm[nt](F.relu(x_new[nt]) + x_dict[nt]),
                              p=self.feature_dropout, training=self.training)
                for nt in x_new
            }

        pooled = self._readout(x_dict, data)
        pooled = F.dropout(pooled, p=self.final_dropout, training=self.training)
        return self.classifier(pooled).squeeze(-1)
