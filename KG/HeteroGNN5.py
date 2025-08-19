from typing import Dict, Tuple, List, Optional, Literal
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_add_pool
from torch_geometric.data import HeteroData
from torch_geometric.utils import dropout_edge, softmax as pyg_softmax

EdgeType = Tuple[str, str, str]
Readout = Literal["fact", "company", "concat", "gated"]


class TimeEdgeBuilder(nn.Module):
    """
    Builds edge_attr for attention:
      input columns can be [sentiment, decay] or [sentiment, delta_t]
      output edge_attr = [sentiment, (optional |s|), (optional polarity), decay, time2vec(log1p(delta_t)), (optional time_bucket_emb)]
      Final dim = 2 + time_dim + (add_abs_sent ? 1:0) + (add_polarity_bit ? 1:0) + (time_bucket_emb_dim or 0)
    """
    def __init__(
        self,
        time_dim: int = 8,
        lambda_decay: float = 0.01,
        t_scale: float = 30.0,
        time_bucket_edges: Optional[List[float]] = None,  # e.g., [0,7,30,90,9999]
        time_bucket_emb_dim: int = 0,                     # 0 disables bucket embeddings
        add_abs_sent: bool = False,
        add_polarity_bit: bool = False,
        sentiment_jitter_std: float = 0.0,                # train-time jitter on s
        delta_t_jitter_frac: float = 0.0,                 # train-time jitter on Δt (fractional)
    ):
        super().__init__()
        self.time_dim = time_dim
        self.lambda_decay = nn.Parameter(torch.tensor(lambda_decay), requires_grad=True)  # learnable λ
        self.t_scale = t_scale
        self.add_abs_sent = add_abs_sent
        self.add_polarity_bit = add_polarity_bit
        self.sentiment_jitter_std = sentiment_jitter_std
        self.delta_t_jitter_frac = delta_t_jitter_frac

        # time bucket embeddings (coarse recency regimes)
        self.bucket_edges = None
        self.time_bucket_emb = None
        if time_bucket_edges is not None and time_bucket_emb_dim > 0:
            self.register_buffer("bucket_edges_buf", torch.tensor(time_bucket_edges, dtype=torch.float), persistent=False)
            self.time_bucket_emb = nn.Embedding(len(time_bucket_edges) - 1, time_bucket_emb_dim)
            self.bucket_edges = "bucket_edges_buf"  # name of buffer to fetch in forward

    def _time2vec(self, delta_t: Tensor) -> Tensor:
        # Δt' = log1p(Δt) / t_scale for stability; final T2V has (time_dim-1) sine terms + 1 linear term
        t = torch.log1p(delta_t.clamp(min=0.0)) / self.t_scale
        freqs = torch.arange(1, self.time_dim, device=delta_t.device, dtype=delta_t.dtype)  # [time_dim-1]
        sin_terms = torch.sin(t * freqs)  # [E, time_dim-1]
        return torch.cat([sin_terms, t], dim=-1)  # [E, time_dim]

    @staticmethod
    def _bin_indices(vals: Tensor, edges: Tensor) -> Tensor:
        # vals: [E], edges: [B+1] ascending; returns bin index in [0..B-1]
        idx = torch.bucketize(vals, edges, right=False) - 1
        return idx.clamp(min=0, max=edges.numel() - 2)

    def forward(self, ea: Optional[Tensor], training: bool = False) -> Tensor:
        if ea is None:
            raise ValueError("edge_attr required for attention with edge_dim. Provide sentiment and decay or delta_t.")

        s = ea[:, 0:1].clamp(-1.0, 1.0)  # sentiment in [-1,1]
        second = ea[:, 1:2]              # decay in [0,1] OR raw Δt

        # train-time jitter
        if training and self.sentiment_jitter_std > 0.0:
            s = s + torch.randn_like(s) * self.sentiment_jitter_std

        looks_like_decay = ((second >= 0.0) & (second <= 1.0)).all()
        lam = self.lambda_decay.clamp(min=1e-6)
        if looks_like_decay:
            decay = second.clamp(0.0, 1.0)
            delta_t = (-torch.log(decay.clamp(min=1e-6))) / lam
        else:
            delta_t = second.clamp(min=0.0)
            if training and self.delta_t_jitter_frac > 0.0:
                delta_t = delta_t * (1.0 + self.delta_t_jitter_frac * torch.randn_like(delta_t))
                delta_t = delta_t.clamp(min=0.0)
            decay = torch.exp(-lam * delta_t)

        comps = [s]
        if self.add_abs_sent:
            comps.append(torch.abs(s))
        if self.add_polarity_bit:
            comps.append((s >= 0).float())
        comps.append(decay)
        comps.append(self._time2vec(delta_t))

        # optional bucket embedding
        if self.time_bucket_emb is not None:
            edges = getattr(self, self.bucket_edges)  # fetch buffer
            bins = self._bin_indices(delta_t.view(-1), edges.to(delta_t.device, delta_t.dtype))
            comps.append(self.time_bucket_emb(bins))

        return torch.cat(comps, dim=-1)  # [E, 2 + time_dim + extras]


class HeteroAttnGNN(nn.Module):
    """
    Heterogeneous attention GNN with edge-feature-aware attention (GATv2Conv with edge_dim),
    optional funnel mode (fact->company only), top-k pre-gating, attention temperature, entropy regularisation,
    time-bucket embeddings, polarity/|s| features, jitter, and MC-Dropout support.
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
        funnel_to_primary: bool = False,         # if True: only ('fact','mentions','company') relation is used
        topk_per_primary: Optional[int] = None,  # keep top-k incoming fact edges per primary before attention
        # NEW: fine-tuning knobs (all default to no-op)
        attn_temperature: float = 1.0,           # <1 sharpens, >1 softens attention (scales edge features)
        entropy_reg_weight: float = 0.0,         # >0 enables attention sparsity penalty; add model.extra_loss to your loss
        time_bucket_edges: Optional[List[float]] = None,
        time_bucket_emb_dim: int = 0,
        add_abs_sent: bool = False,
        add_polarity_bit: bool = False,
        sentiment_jitter_std: float = 0.0,
        delta_t_jitter_frac: float = 0.0,
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

        # sparsity/control
        self.attn_temperature = float(attn_temperature)
        self.entropy_reg_weight = float(entropy_reg_weight)
        self.register_buffer("attn_scale", torch.tensor(1.0 / max(1e-6, attn_temperature)), persistent=False)
        self.extra_loss: Tensor = torch.tensor(0.0)  # updated each forward

        # restrict edge types if funnel mode
        if funnel_to_primary:
            self.edge_types = [et for et in self.edge_types_full if et == ('fact', 'mentions', 'company')]
            if len(self.edge_types) == 0:
                raise ValueError("Funnel mode expects ('fact','mentions','company') relation in metadata.")
        else:
            self.edge_types = self.edge_types_full

        # per-type input projections (lazy init on first forward)
        self.in_proj = nn.ModuleDict()
        self.post_norm = nn.ModuleDict()

        # temporal/edge builder (produces edge_attr for attention)
        self.edge_builder = TimeEdgeBuilder(
            time_dim=time_dim,
            lambda_decay=0.01,
            t_scale=30.0,
            time_bucket_edges=time_bucket_edges,
            time_bucket_emb_dim=time_bucket_emb_dim,
            add_abs_sent=add_abs_sent,
            add_polarity_bit=add_polarity_bit,
            sentiment_jitter_std=sentiment_jitter_std,
            delta_t_jitter_frac=delta_t_jitter_frac,
        )

        # build per-relation attention convs for each layer (manual hetero loop so we can read attention weights)
        edge_dim = 2 + time_dim + (1 if add_abs_sent else 0) + (1 if add_polarity_bit else 0) + (time_bucket_emb_dim or 0)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            convs = nn.ModuleDict()
            for et in self.edge_types:
                convs[str(et)] = GATv2Conv(
                    in_channels=(-1, -1),            # PyG infers from x_dict
                    out_channels=self.out_per_head,
                    heads=self.heads,
                    edge_dim=edge_dim,
                    add_self_loops=False,
                    share_weights=False,
                    dropout=0.0                      # use feature-level dropout externally
                )
            self.layers.append(convs)

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

    # -------------------------------
    # Helpers
    # -------------------------------
    def _maybe_build_type_encoders(self, data: HeteroData) -> None:
        device = next(self.parameters()).device
        for nt in self.node_types:
            if nt in self.in_proj:
                continue

            d_in = None
            try:
                if nt in data and hasattr(data[nt], 'x') and data[nt].x is not None:
                    d_in = int(data[nt].x.size(-1))
                    print(f"[HeteroGNN5] Detected {nt} input dimension: {d_in}")
            except Exception as e:
                print(f"[HeteroGNN5] Error detecting {nt} input dimension: {e}")
                pass

            if d_in is None:
                # Try to get dimensions from the first batch
                try:
                    if hasattr(data, 'node_types') and nt in data.node_types:
                        # This is a batched HeteroData, try to access the first graph
                        if hasattr(data, 'num_graphs') and data.num_graphs > 0:
                            # For batched data, we need to check the actual dimensions
                            # Let's try to get a sample from the batch
                            sample_data = data[0] if hasattr(data, '__getitem__') else data
                            if hasattr(sample_data, nt) and hasattr(sample_data[nt], 'x'):
                                d_in = int(sample_data[nt].x.size(-1))
                                print(f"[HeteroGNN5] Detected {nt} input dimension from sample: {d_in}")
                except Exception as e:
                    print(f"[HeteroGNN5] Error detecting {nt} input dimension from sample: {e}")

            if d_in is None:
                if nt == "fact":
                    d_in = 1536  # Updated to match the actual data dimensions
                elif nt == "company":
                    d_in = 27
                else:
                    d_in = 64
                print(f"[HeteroGNN5] Using default {nt} input dimension: {d_in}")

            mlp = nn.Sequential(nn.Linear(d_in, self.hidden_channels), nn.ReLU())
            self.in_proj[nt] = mlp.to(device)
            self.post_norm[nt] = nn.LayerNorm(self.hidden_channels).to(device)
            print(f"[HeteroGNN5] Created {nt} encoder: {d_in} -> {self.hidden_channels}")

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

    def _build_edge_inputs(
        self, data: HeteroData
    ) -> Tuple[Dict[EdgeType, Tensor], Dict[EdgeType, Tensor]]:
        edge_index_dict: Dict[EdgeType, Tensor] = {}
        edge_attr_for_attn: Dict[EdgeType, Tensor] = {}
        device = next(self.parameters()).device

        for et in self.edge_types:
            store = data[et]
            ei: Tensor = store.edge_index.to(device)
            ea: Optional[Tensor] = getattr(store, "edge_attr", None)
            if ea is not None and ea.device != device:
                ea = ea.to(device)

            # structure-level edge dropout
            kept_mask = None
            if self.training and self.edge_dropout > 0.0 and ei.numel() > 0:
                ei, kept_mask = dropout_edge(ei, p=self.edge_dropout, force_undirected=False)

            # enriched edge features for attention (with jitter if training)
            ea_attn_full = self.edge_builder(ea, training=self.training).to(device)
            if kept_mask is not None:
                ea_attn_full = ea_attn_full[kept_mask]

            # attention temperature (features scaled by 1/T)
            ea_attn = ea_attn_full * self.attn_scale

            # optional top-k pre-gating (only for fact->company)
            if self.topk_per_primary is not None and et == ('fact', 'mentions', 'company') and ei.numel() > 0:
                # Ensure we have the same number of edges and edge features
                assert ei.size(1) == ea_attn_full.size(0), f"Edge index count {ei.size(1)} != edge attr count {ea_attn_full.size(0)}"
                
                # score: recent & positive sentiment get preference (heuristic)
                s = ea_attn_full[:, 0:1]
                decay = ea_attn_full[:, 1:2]  # Use the same edge features for scoring
                score = decay * torch.sigmoid(2.0 * s)
                score = score.squeeze(-1)  # [E]

                dst = ei[1]
                # keep top-k per dst
                order = torch.argsort(score, descending=True)
                keep = torch.zeros_like(score, dtype=torch.bool, device=device)
                counts: Dict[int, int] = {}
                for idx in order.tolist():
                    d = int(dst[idx])
                    c = counts.get(d, 0)
                    if c < self.topk_per_primary:
                        keep[idx] = True
                        counts[d] = c + 1
                
                # Apply filtering consistently
                ei = ei[:, keep]
                ea_attn = ea_attn_full[keep] * self.attn_scale  # Re-apply temperature scaling
                
                # Final consistency check
                assert ei.size(1) == ea_attn.size(0), f"After filtering: Edge index count {ei.size(1)} != edge attr count {ea_attn.size(0)}"

            edge_index_dict[et] = ei
            edge_attr_for_attn[et] = ea_attn

        return edge_index_dict, edge_attr_for_attn

    def _attn_layer(
        self,
        convs: nn.ModuleDict,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
        edge_attr_dict: Dict[EdgeType, Tensor],
        collect_entropy_on: Optional[EdgeType] = ('fact', 'mentions', 'company'),
    ) -> Dict[str, Tensor]:
        """
        Apply one hetero attention layer manually so we can capture attention weights for entropy regularisation.
        """
        out: Dict[str, List[Tensor]] = {nt: [] for nt in x_dict.keys()}
        entropy_term = x_dict[next(iter(x_dict))].new_tensor(0.0)

        for et, ei in edge_index_dict.items():
            conv: GATv2Conv = convs[str(et)]
            ea = edge_attr_dict[et]
            src_type, _, dst_type = et
            x_src, x_dst = x_dict[src_type], x_dict[dst_type]

            y, (edge_index_used, alpha) = conv(
                (x_src, x_dst),
                ei,
                edge_attr=ea,
                return_attention_weights=True
            )
            out[dst_type].append(y)

            # entropy regularisation on incoming edges (per head)
            if self.entropy_reg_weight > 0.0 and collect_entropy_on is not None and et == collect_entropy_on:
                dst = edge_index_used[1]
                eps = 1e-9
                # mean entropy across heads, then average over nodes
                h_sum = 0.0
                for h in range(alpha.size(1)):
                    a = alpha[:, h].clamp(min=eps)
                    neg_a_log_a = -(a * torch.log(a))
                    h_node = global_add_pool(neg_a_log_a, dst)  # [num_dst_in_batch]
                    h_sum = h_sum + h_node.mean()
                entropy_term = entropy_term + (h_sum / alpha.size(1))

        # aggregate per destination node type
        x_new: Dict[str, Tensor] = {}
        for nt, parts in out.items():
            if parts:
                x_new[nt] = torch.stack(parts, dim=0).sum(dim=0)  # sum over incoming relations
            else:
                x_new[nt] = x_dict[nt]

        # store auxiliary regulariser (weighted); no-op if weight=0
        self.extra_loss = self.entropy_reg_weight * entropy_term
        return x_new

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

    # -------------------------------
    # Forward
    # -------------------------------
    def forward(self, data: HeteroData, mc_dropout: bool = False) -> Tensor:
        """
        Args:
          data: HeteroData with:
            - data[nt].x
            - data[et].edge_index
            - data[et].edge_attr  ([:,0]=sentiment, [:,1]=decay or delta_t)
            - data[nt].batch
            - optional: data['company'].primary_mask (torch.bool)
          mc_dropout: if True, keeps Dropout active in eval mode (Monte Carlo Dropout).
        Returns:
          logits: Tensor [batch_size]
        """
        self._maybe_build_type_encoders(data)

        # MC-Dropout handling (does nothing in training mode)
        if not self.training and mc_dropout:
            def enable_dropout(m):
                if isinstance(m, nn.Dropout):
                    m.train()
            self.apply(enable_dropout)

        # ensure features on model device/dtype
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        x_dict = {nt: self.in_proj[nt](data[nt].x.to(device=device, dtype=dtype)) for nt in self.node_types}

        # build per-relation connectivity and enriched edge features
        edge_index_dict, edge_attr_for_attn = self._build_edge_inputs(data)

        # manual hetero attention stack (so we can regularise attention)
        for convs in self.layers:
            x_updated = self._attn_layer(
                convs, x_dict, edge_index_dict, edge_attr_for_attn,
                collect_entropy_on=('fact', 'mentions', 'company')
            )
            # residual + norm + dropout
            x_dict = {
                nt: F.dropout(self.post_norm[nt](F.relu(x_updated[nt]) + x_dict[nt]),
                              p=self.feature_dropout, training=self.training or mc_dropout)
                for nt in x_updated
            }

        pooled = self._readout(x_dict, data)
        pooled = F.dropout(pooled, p=self.final_dropout, training=self.training or mc_dropout)
        return self.classifier(pooled).squeeze(-1)
