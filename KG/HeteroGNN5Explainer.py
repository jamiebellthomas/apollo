#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Explainability for HeteroGNN5 (HeteroAttnGNN) on the test set.
- No CLI args: configure in CONFIG below.
- Captures final-layer attention on (fact, mentions, company).
- Ranks top-k facts per positive test graph by attention influence.
- Optional removal-impact sanity check.
- Exports per-graph CSV/JSON and an aggregate CSV.

Adjust the import: `from model_gnn5 import HeteroAttnGNN` to match your codebase.
"""

import os
import json
from typing import Dict, Tuple, List, Optional
from collections import defaultdict

import torch
from torch import nn, Tensor
from torch_geometric.data import HeteroData, InMemoryDataset
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import global_add_pool

# ========= USER CONFIG =========
CONFIG = {
    "TEST_PATH": "/path/to/test.pt",                # .pt list of HeteroData, directory of .pt graphs, or dict with 'data_list'
    "CHECKPOINT_PATH": "/path/to/gnn5_state.pt",    # model state_dict or checkpoint dict
    "OUT_DIR": "./gnn5_explanations",               # output folder
    "DEVICE": "cuda:0",                             # "cuda:0" or "cpu"
    "BATCH_SIZE": 1,                                # recommend 1 for clean per-graph capture
    "NUM_WORKERS": 0,
    "TOPK": 5,                                      # top-k facts to report per positive graph
    "ONLY_POSITIVE_LABELS": True,                   # only produce outputs for label==1 graphs
    "THRESHOLD_LOGIT": None,                        # e.g., 0.0 to require logit >= 0; None disables
    "DO_IMPACT_CHECK": True,                        # set False to disable removal-impact sanity check
    "MC_DROPOUT_EVAL": False,                       # set True to keep dropout active at eval for uncertainty
    "PRIMARY_ONLY": True                            # restrict attribution edges to the primary company node
}
# ========= /USER CONFIG =========

# === IMPORT YOUR MODEL HERE ===
# Change this import to match your repository structure.
from HeteroGNN5 import HeteroAttnGNN  # <-- UPDATE THIS IMPORT
# ==============================


# --------- IO helpers ---------
def load_testset(path: str):
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith(".pt")]
        if not files:
            raise FileNotFoundError(f"No .pt files found under directory: {path}")
        graphs = [torch.load(fp) for fp in files]
        return graphs

    obj = torch.load(path)
    if isinstance(obj, (list, tuple)):
        return list(obj)
    if isinstance(obj, InMemoryDataset):
        return obj
    if isinstance(obj, dict):
        if 'data_list' in obj:
            return obj['data_list']
        if 'dataset' in obj:
            return obj['dataset']
    if isinstance(obj, HeteroData):
        return [obj]
    raise ValueError(f"Unsupported test set format at {path}")

def infer_metadata_from_graph(g: HeteroData) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    return list(g.node_types), list(g.edge_types)

def load_model(checkpoint_path: str, metadata, device: str) -> HeteroAttnGNN:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and 'state_dict' not in ckpt:
        state_dict = ckpt
        hparams = {}
    else:
        state_dict = ckpt.get('state_dict', ckpt)
        hparams = ckpt.get('hparams', {})

    model = HeteroAttnGNN(
        metadata=metadata,
        hidden_channels=int(hparams.get('hidden_channels', 128)),
        num_layers=int(hparams.get('num_layers', 2)),
        heads=int(hparams.get('heads', 4)),
        time_dim=int(hparams.get('time_dim', 8)),
        feature_dropout=float(hparams.get('feature_dropout', 0.2)),
        edge_dropout=float(hparams.get('edge_dropout', 0.0)),
        final_dropout=float(hparams.get('final_dropout', 0.1)),
        readout=hparams.get('readout', 'gated'),
        funnel_to_primary=bool(hparams.get('funnel_to_primary', False)),
        topk_per_primary=hparams.get('topk_per_primary', None),
        attn_temperature=float(hparams.get('attn_temperature', 1.0)),
        entropy_reg_weight=float(hparams.get('entropy_reg_weight', 0.0)),
        time_bucket_edges=hparams.get('time_bucket_edges', None),
        time_bucket_emb_dim=int(hparams.get('time_bucket_emb_dim', 0)),
        add_abs_sent=bool(hparams.get('add_abs_sent', False)),
        add_polarity_bit=bool(hparams.get('add_polarity_bit', False)),
        sentiment_jitter_std=float(hparams.get('sentiment_jitter_std', 0.0)),
        delta_t_jitter_frac=float(hparams.get('delta_t_jitter_frac', 0.0)),
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


# --------- Attention capture (monkey patch) ---------
class AttnCapture:
    """
    Recomputes attention on the LAST layer for ('fact','mentions','company') and stores (edge_index, alpha).
    """
    def __init__(self, model: HeteroAttnGNN):
        self.model = model
        self.orig_attn_layer = model._attn_layer
        self.buffer: List[Dict[str, Tensor]] = []
        self.layer_count = 0
        self.total_layers = len(model.layers)

    def __call__(self, convs, x_dict, edge_index_dict, edge_attr_dict, collect_entropy_on=('fact','mentions','company')):
        out = self.orig_attn_layer(convs, x_dict, edge_index_dict, edge_attr_dict, collect_entropy_on=collect_entropy_on)
        self.layer_count += 1
        if self.layer_count == self.total_layers:
            key = ('fact','mentions','company')
            if key in edge_index_dict:
                conv = convs[str(key)]
                src, dst = key[0], key[2]
                x_src, x_dst = x_dict[src], x_dict[dst]
                ei = edge_index_dict[key]
                ea = edge_attr_dict[key]
                with torch.no_grad():
                    _, (edge_index_used, alpha) = conv((x_src, x_dst), ei, edge_attr=ea, return_attention_weights=True)
                self.buffer.append({
                    "edge_index": edge_index_used.detach().cpu(),
                    "alpha": alpha.detach().cpu()
                })
        return out

    def install(self):
        self.model._attn_layer = self.__call__
        self.layer_count = 0
        self.buffer.clear()

    def uninstall(self):
        self.model._attn_layer = self.orig_attn_layer

    def pop(self):
        out = list(self.buffer)
        self.buffer.clear()
        self.layer_count = 0
        return out


# --------- Attribution utilities ---------
def primary_company_index(data: HeteroData) -> Optional[int]:
    mask = getattr(data['company'], 'primary_mask', None)
    if mask is None:
        return None
    if mask.dtype != torch.bool:
        mask = mask.bool()
    idx = mask.nonzero(as_tuple=False).view(-1)
    if idx.numel() == 0:
        return None
    return int(idx[0].item())

def compute_influence_for_graph(
    data: HeteroData,
    edge_index_used: Tensor,  # [2,E]
    alpha: Tensor,            # [E,heads]
    restrict_to_primary: bool = True,
) -> Dict[int, float]:
    if edge_index_used.numel() == 0:
        return {}
    a = alpha.mean(dim=1)  # [E]
    src = edge_index_used[0]
    dst = edge_index_used[1]
    if restrict_to_primary:
        pidx = primary_company_index(data)
        if pidx is not None:
            mask = (dst == pidx)
            src = src[mask]
            a = a[mask]
    scores: Dict[int, float] = defaultdict(float)
    for s, w in zip(src.tolist(), a.tolist()):
        scores[s] += float(w)
    return dict(scores)

def mask_edges_by_fact(
    data: HeteroData,
    facts_to_remove: List[int],
    relation: Tuple[str,str,str] = ('fact','mentions','company')
) -> HeteroData:
    d = data.clone()
    ei = d[relation].edge_index
    src = ei[0]
    keep = ~torch.isin(src, torch.tensor(facts_to_remove, device=src.device, dtype=src.dtype))
    d[relation].edge_index = ei[:, keep]
    if hasattr(d[relation], "edge_attr") and d[relation].edge_attr is not None:
        d[relation].edge_attr = d[relation].edge_attr[keep]
    return d

def sigmoid(x: Tensor) -> Tensor:
    return torch.sigmoid(x)

def gather_edge_metadata_for_fact(
    data: HeteroData,
    edge_index_used: Tensor,
    alpha: Tensor,
    fact_idx: int,
    top_only_to_primary: bool = True
) -> List[Dict]:
    src = edge_index_used[0]; dst = edge_index_used[1]
    heads_mean = alpha.mean(dim=1)
    rows = []
    pidx = primary_company_index(data) if top_only_to_primary else None
    rel = ('fact','mentions','company')
    ea = getattr(data[rel], "edge_attr", None)

    for e in range(edge_index_used.size(1)):
        if int(src[e].item()) != fact_idx:
            continue
        if top_only_to_primary and pidx is not None and int(dst[e].item()) != pidx:
            continue
        row = {
            "edge_eid": e,
            "alpha_mean": float(heads_mean[e].item()),
            "dst_company_index": int(dst[e].item())
        }
        if ea is not None and ea.size(0) > e:
            s = float(ea[e,0].item())
            second = float(ea[e,1].item())
            row.update({"sentiment": s, "second_attr": second})
        rows.append(row)
    return rows


# --------- Main pipeline ---------
def main():
    TEST_PATH = CONFIG["TEST_PATH"]
    CHECKPOINT_PATH = CONFIG["CHECKPOINT_PATH"]
    OUT_DIR = CONFIG["OUT_DIR"]
    DEVICE = CONFIG["DEVICE"]
    BATCH_SIZE = CONFIG["BATCH_SIZE"]
    NUM_WORKERS = CONFIG["NUM_WORKERS"]
    TOPK = CONFIG["TOPK"]
    ONLY_POS = CONFIG["ONLY_POSITIVE_LABELS"]
    THRESH_LOGIT = CONFIG["THRESHOLD_LOGIT"]
    DO_IMPACT = CONFIG["DO_IMPACT_CHECK"]
    MC_DROPOUT = CONFIG["MC_DROPOUT_EVAL"]
    PRIMARY_ONLY = CONFIG["PRIMARY_ONLY"]

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load test data
    testset = load_testset(TEST_PATH)
    loader = GeoDataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Infer metadata + load model
    sample_batch = next(iter(loader))
    meta = infer_metadata_from_graph(sample_batch if hasattr(sample_batch, 'node_types') else testset[0])

    device = torch.device(DEVICE if torch.cuda.is_available() or "cpu" not in DEVICE else "cpu")
    model = load_model(CHECKPOINT_PATH, meta, str(device))

    # Monkey-patch attention capture
    capturer = AttnCapture(model)
    capturer.install()

    agg_rows = []

    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            data = data.to(device)

            # MC-Dropout during eval if requested
            if not model.training and MC_DROPOUT:
                def enable_dropout(m):
                    if isinstance(m, nn.Dropout):
                        m.train()
                model.apply(enable_dropout)

            logits = model(data, mc_dropout=MC_DROPOUT)
            probs = sigmoid(logits)

            comp_batch = data['company'].batch
            B = int(comp_batch.max().item()) + 1 if comp_batch.numel() > 0 else 1

            captured = capturer.pop()
            if not captured:
                continue
            cap = captured[-1]
            edge_index_used = cap["edge_index"].to(device)
            alpha = cap["alpha"].to(device)

            dst = edge_index_used[1]
            dst_batch = comp_batch[dst] if dst.numel() > 0 else dst

            for g in range(B):
                mask_e = (dst_batch == g) if dst.numel() > 0 else torch.tensor([], dtype=torch.bool, device=device)
                if dst.numel() > 0 and mask_e.sum().item() == 0:
                    continue

                if dst.numel() > 0:
                    ei_g = edge_index_used[:, mask_e]
                    alpha_g = alpha[mask_e, :]
                else:
                    ei_g = edge_index_used
                    alpha_g = alpha

                fact_batch = data['fact'].batch
                if (fact_batch == g).sum().item() == 0:
                    continue

                influence = compute_influence_for_graph(data, ei_g, alpha_g, restrict_to_primary=PRIMARY_ONLY)
                if len(influence) == 0:
                    continue

                y_true = int(data.y[g].item()) if hasattr(data, 'y') else None
                y_logit = float(logits[g].item())
                y_prob = float(probs[g].item())

                if ONLY_POS and (y_true is not None) and (y_true != 1):
                    continue
                if THRESH_LOGIT is not None and y_logit < THRESH_LOGIT:
                    continue

                topk = sorted(influence.items(), key=lambda kv: kv[1], reverse=True)[:TOPK]

                # Optional removal-impact check
                impact_rows = []
                if DO_IMPACT:
                    for f_idx, score in topk:
                        data_masked = mask_edges_by_fact(data, [f_idx])
                        logit_masked = model(data_masked, mc_dropout=False)[g]
                        prob_masked = float(sigmoid(logit_masked).item())
                        delta_prob = y_prob - prob_masked
                        rows = gather_edge_metadata_for_fact(data, ei_g, alpha_g, f_idx, top_only_to_primary=PRIMARY_ONLY)
                        impact_rows.append({
                            "fact_index": int(f_idx),
                            "influence": float(score),
                            "prob_base": y_prob,
                            "prob_masked": prob_masked,
                            "delta_prob": float(delta_prob),
                            "edges": rows
                        })
                else:
                    for f_idx, score in topk:
                        rows = gather_edge_metadata_for_fact(data, ei_g, alpha_g, f_idx, top_only_to_primary=PRIMARY_ONLY)
                        impact_rows.append({
                            "fact_index": int(f_idx),
                            "influence": float(score),
                            "edges": rows
                        })

                # Optional raw_text extraction
                raw_texts = getattr(data['fact'], 'raw_text', None)
                def maybe_text(i):
                    try:
                        if raw_texts is None:
                            return None
                        if isinstance(raw_texts, list):
                            return raw_texts[i]
                        return None
                    except Exception:
                        return None

                # Write per-graph CSV + JSON
                out_prefix = os.path.join(OUT_DIR, f"graph_{batch_idx}_{g}")

                import csv
                csv_path = out_prefix + "_topk.csv"
                with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
                    wr = csv.writer(fcsv)
                    wr.writerow(["graph_batch", "g_index", "true_label", "logit", "prob",
                                 "rank", "fact_index", "influence", "alpha_edge_max", "sentiment", "second_attr", "snippet"])
                    rank = 1
                    for row in impact_rows:
                        fact_i = row["fact_index"]
                        infl = row["influence"]
                        alpha_edges = [e["alpha_mean"] for e in row.get("edges", [])]
                        sents = [e.get("sentiment", None) for e in row.get("edges", [])]
                        seconds = [e.get("second_attr", None) for e in row.get("edges", [])]
                        snippet = maybe_text(fact_i)
                        wr.writerow([batch_idx, g, y_true, y_logit, y_prob,
                                     rank, fact_i, infl,
                                     max(alpha_edges) if alpha_edges else None,
                                     sents[0] if sents else None,
                                     seconds[0] if seconds else None,
                                     snippet])
                        rank += 1

                json_path = out_prefix + "_detail.json"
                with open(json_path, "w", encoding="utf-8") as fj:
                    json.dump({
                        "graph_batch": batch_idx,
                        "g_index": g,
                        "true_label": y_true,
                        "logit": y_logit,
                        "prob": y_prob,
                        "topk": impact_rows
                    }, fj, indent=2)

                for rank, row in enumerate(impact_rows, start=1):
                    agg_rows.append({
                        "graph_batch": batch_idx,
                        "g_index": g,
                        "true_label": y_true,
                        "logit": y_logit,
                        "prob": y_prob,
                        "rank": rank,
                        "fact_index": row["fact_index"],
                        "influence": row["influence"],
                        "delta_prob": row.get("delta_prob", None)
                    })

    # Aggregate CSV
    agg_csv = os.path.join(OUT_DIR, "aggregate_topk.csv")
    import csv
    with open(agg_csv, "w", newline="", encoding="utf-8") as fcsv:
        wr = csv.DictWriter(fcsv, fieldnames=[
            "graph_batch", "g_index", "true_label", "logit", "prob",
            "rank", "fact_index", "influence", "delta_prob"
        ])
        wr.writeheader()
        for r in agg_rows:
            wr.writerow(r)

    print(f"[OK] Explanations written to: {OUT_DIR}")
    print(f"     Aggregate CSV: {agg_csv}")

if __name__ == "__main__":
    main()
