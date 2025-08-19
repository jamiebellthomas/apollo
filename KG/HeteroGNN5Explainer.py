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
import pickle
import time
import glob
from typing import Dict, Tuple, List, Optional
from collections import defaultdict

import torch
from torch import nn, Tensor
from torch_geometric.data import HeteroData, InMemoryDataset
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import global_add_pool

# Import the safe_read_pickle function
from cache_dataset import safe_read_pickle

# ========= USER CONFIG =========
CONFIG = {
    "USE_CURRENT_DATA": False,  # Use cached dataset
    "SUBGRAPHS_PATH": "Data/subgraphs.jsonl",  # Path to current subgraphs data (from project root)
    "TEST_PATH": "KG/dataset_cache/testing_cached_dataset_nf35_limall.pkl",  # Test data from cache
    "CHECKPOINT_PATH": "Results/heterognn5/20250818_171525/model.pt",     # Will be overridden for each model
    "OUT_DIR": "KG/heterognn5_explanations",                               # output folder
    "DEVICE": "cpu",                                                      # Use CPU since CUDA not available
    "BATCH_SIZE": 1,                                                      # recommend 1 for clean per-graph capture
    "NUM_WORKERS": 0,
    "TOPK": 15,                                                            # top-15 facts as requested
    "ONLY_POSITIVE_LABELS": False,                                         # Process all predictions, filter by positive predictions
    "THRESHOLD_LOGIT": 0.0,                                              # Only process when logit >= 0 (positive prediction)
    "DO_IMPACT_CHECK": True,                                              # set False to disable removal-impact sanity check
    "MC_DROPOUT_EVAL": False,                                             # set True to keep dropout active at eval for uncertainty
    "PRIMARY_ONLY": True,                                                 # restrict attribution edges to the primary company node
    "N_FACTS": 25,                                                        # Number of facts per subgraph for processing
    "LIMIT": None                                                          # No limit - use all available data
}
# ========= /USER CONFIG =========

# === IMPORT YOUR MODEL HERE ===
# Change this import to match your repository structure.
from HeteroGNN5 import HeteroAttnGNN  # <-- Updated to match repository structure
# ==============================


# --------- IO helpers ---------
def load_testset(path: str):
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith(".pt")]
        if not files:
            raise FileNotFoundError(f"No .pt files found under directory: {path}")
        graphs = [torch.load(fp) for fp in files]
        return graphs

    # Use safe_read_pickle for cache files
    obj = safe_read_pickle(path)
    if obj is None:
        raise ValueError(f"Failed to load cache file: {path}")
    
    if isinstance(obj, (list, tuple)):
        return list(obj)
    if isinstance(obj, InMemoryDataset):
        return obj
    if isinstance(obj, dict):
        if 'data_list' in obj:
            return obj['data_list']
        if 'dataset' in obj:
            return obj['dataset']
        if 'graphs' in obj:  # Handle the cache format from run.py
            return obj['graphs']
    if isinstance(obj, HeteroData):
        return [obj]
    raise ValueError(f"Unsupported test set format at {path}")

def load_current_testset(subgraphs_path: str, n_facts: int, limit: int):
    """Load and process current subgraphs data for testing"""
    print(f"Loading current subgraphs data from {subgraphs_path}")
    print(f"Processing with n_facts={n_facts}, limit={limit}")
    
    # Import required modules
    import sys
    # Add the project root to the path so we can import from KG
    project_root = os.path.dirname(os.path.dirname(__file__))
    sys.path.append(project_root)
    sys.path.append(os.path.join(project_root, 'KG'))
    
    from KG.SubGraphDataLoader import SubGraphDataLoader
    from KG.run import encode_all_to_heterodata, attach_y_and_meta
    
    # Load subgraphs with the same parameters as the cached dataset
    loader = SubGraphDataLoader(
        min_facts=n_facts, 
        limit=limit, 
        jsonl_path=subgraphs_path,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42,
        split_data=True
    )
    
    # Encode to HeteroData
    training_graphs, testing_graphs, training_raw_sg, testing_raw_sg = encode_all_to_heterodata(loader)
    
    # Attach labels and metadata
    attach_y_and_meta(testing_graphs, testing_raw_sg)
    
    print(f"✅ Loaded {len(testing_graphs)} test graphs from current data")
    return testing_graphs, testing_raw_sg

def infer_metadata_from_graph(g: HeteroData) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    return list(g.node_types), list(g.edge_types)

def load_model(checkpoint_path: str, metadata, device: str) -> HeteroAttnGNN:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and 'state_dict' not in ckpt:
        state_dict = ckpt
        hparams = {}
    else:
        state_dict = ckpt['state_dict']
        hparams = ckpt.get('hyper_parameters', {})
    
    # Extract model parameters from state dict
    # Look for key patterns to determine dimensions
    hidden_channels = 128  # Default
    num_layers = 2  # Default
    heads = 4  # Default
    time_dim = 8  # Default
    
    # Try to extract hidden_channels from classifier
    for key in state_dict.keys():
        if 'classifier.0.weight' in key:
            shape = state_dict[key].shape
            if len(shape) == 2:
                hidden_channels = shape[0]
                break
    
    # Try to extract num_layers from layer keys
    max_layer = 0
    for key in state_dict.keys():
        if 'layers.' in key:
            parts = key.split('.')
            if len(parts) > 1:
                try:
                    layer_num = int(parts[1])
                    max_layer = max(max_layer, layer_num)
                except:
                    pass
    if max_layer > 0:
        num_layers = max_layer + 1
    
    # Extract heads from attention weights
    att_keys = [k for k in state_dict.keys() if 'att' in k and k.endswith('.att') and 'layers' in k]
    if att_keys:
        shape = state_dict[att_keys[0]].shape
        heads = shape[1]  # This should be 8
    else:
        heads = 4  # Default
    
    # Also check if we need to adjust hidden_channels to be divisible by heads
    if hidden_channels % heads != 0:
        # Round down to nearest multiple of heads
        hidden_channels = (hidden_channels // heads) * heads
        print(f"Adjusted hidden_channels to {hidden_channels} to be divisible by {heads} heads")
    
    # Try to extract edge_dim from lin_edge weights
    edge_dim = 10  # Default
    for key in state_dict.keys():
        if 'lin_edge.weight' in key:
            shape = state_dict[key].shape
            if len(shape) == 2:
                edge_dim = shape[1]
                break
    
    # Calculate time_dim from edge_dim
    # edge_dim = 2 + time_dim + (add_abs_sent ? 1:0) + (add_polarity_bit ? 1:0) + (time_bucket_emb_dim or 0)
    # Assuming no extras for now
    time_dim = edge_dim - 2
    
    print(f"Model parameters: hidden_channels={hidden_channels}, num_layers={num_layers}, heads={heads}, edge_dim={edge_dim}, time_dim={time_dim}")
    
    # Create model with extracted parameters
    model = HeteroAttnGNN(
        metadata=metadata,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        heads=heads,
        time_dim=time_dim,
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
    
    # Load state dict
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
                
                # Check if we have valid edges and nodes
                if ei.numel() > 0 and x_src.size(0) > 0 and x_dst.size(0) > 0:
                    # Ensure edge indices are within bounds
                    max_src_idx = x_src.size(0) - 1
                    max_dst_idx = x_dst.size(0) - 1
                    
                    # Filter edges that are within bounds
                    valid_src = (ei[0] >= 0) & (ei[0] <= max_src_idx)
                    valid_dst = (ei[1] >= 0) & (ei[1] <= max_dst_idx)
                    valid_edges = valid_src & valid_dst
                    
                    if valid_edges.any():
                        ei_valid = ei[:, valid_edges]
                        ea_valid = ea[valid_edges] if ea is not None else None
                        
                        with torch.no_grad():
                            _, (edge_index_used, alpha) = conv((x_src, x_dst), ei_valid, edge_attr=ea_valid, return_attention_weights=True)
                        
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

def find_all_models():
    """Find all model.pt files in Results/heterognn5/"""
    pattern = "Results/heterognn5/*/model.pt"
    model_paths = glob.glob(pattern)
    model_paths.sort()  # Sort for consistent ordering
    print(f"Found {len(model_paths)} models")
    return model_paths

def main():
    """Main function to process all models and get explanations"""
    print("🚀 Starting Comprehensive HeteroGNN5 Explainer")
    print("=" * 60)
    
    # Extract config
    USE_CURRENT_DATA = CONFIG["USE_CURRENT_DATA"]
    TEST_PATH = CONFIG["TEST_PATH"]
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
    
    # Create output directory
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Load test data
    if USE_CURRENT_DATA:
        testset, raw_sg = load_current_testset(CONFIG["SUBGRAPHS_PATH"], CONFIG["N_FACTS"], CONFIG["LIMIT"])
    else:
        testset = load_testset(TEST_PATH)
        # Load original subgraphs for mapping back to original data
        print("Loading original subgraphs for fact mapping...")
        raw_sg = []
        with open(CONFIG["SUBGRAPHS_PATH"], 'r') as f:
            for line in f:
                if line.strip():
                    raw_sg.append(json.loads(line))
        print(f"✅ Loaded {len(raw_sg)} original subgraphs for mapping")
    
    # Find all models
    model_paths = find_all_models()
    
    # Process each model
    all_results = []
    all_agg_rows = []
    
    for model_idx, model_path in enumerate(model_paths):
        print(f"\n📊 Processing model {model_idx+1}/{len(model_paths)}: {os.path.basename(os.path.dirname(model_path))}")
        
        # Load model
        meta = infer_metadata_from_graph(testset[0])
        model = load_model(model_path, meta, DEVICE)
        
        # Monkey-patch attention capture for proper influence computation
        capturer = AttnCapture(model)
        capturer.install()
        
        # Create data loader
        loader = GeoDataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        
        model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(loader):
                data = data.to(DEVICE)
                
                # Get model predictions with attention capture
                logits = model(data, mc_dropout=MC_DROPOUT)
                probs = torch.sigmoid(logits)
                
                # Get captured attention weights
                captured = capturer.pop()
                if not captured:
                    continue
                    
                cap = captured[-1]
                edge_index_used = cap["edge_index"].to(DEVICE)
                alpha = cap["alpha"].to(DEVICE)
                
                # Process each graph in the batch
                for g in range(data.num_graphs):
                    y_logit = float(logits[g].item())
                    y_prob = float(probs[g].item())
                    
                    # Check if this is a positive prediction (logit >= threshold)
                    if THRESH_LOGIT is not None and y_logit < THRESH_LOGIT:
                        continue
                    
                    try:
                        # Compute real influence using attention weights
                        dst = edge_index_used[1]
                        comp_batch = data['company'].batch if hasattr(data, 'company') and hasattr(data['company'], 'batch') else None
                        
                        if comp_batch is not None:
                            # Filter edges for this graph
                            mask_e = (comp_batch[dst] == g) if dst.numel() > 0 else torch.tensor([], dtype=torch.bool, device=DEVICE)
                            
                            if dst.numel() > 0 and mask_e.sum().item() > 0:
                                ei_g = edge_index_used[:, mask_e]
                                alpha_g = alpha[mask_e, :]
                                
                                # Compute influence for this graph
                                influence = compute_influence_for_graph(data, ei_g, alpha_g, restrict_to_primary=PRIMARY_ONLY)
                            else:
                                # Fallback: uniform influence
                                influence = {i: 1.0 for i in range(data['fact'].num_nodes)}
                        else:
                            # Fallback: uniform influence
                            influence = {i: 1.0 for i in range(data['fact'].num_nodes)}
                        
                        # Get top-k facts
                        topk = sorted(influence.items(), key=lambda kv: kv[1], reverse=True)[:TOPK]
                        
                        # Process each top fact
                        for rank, (fact_idx, score) in enumerate(topk, 1):
                            fact_data = {
                                "model_name": os.path.basename(os.path.dirname(model_path)),
                                "graph_batch": batch_idx,
                                "g_index": g,
                                "logit": y_logit,
                                "prob": y_prob,
                                "rank": rank,
                                "fact_index": int(fact_idx),
                                "influence": float(score)
                            }
                            
                            # Add original fact data from HeteroData attributes
                            if hasattr(data, 'fact_ids') and fact_idx < len(data.fact_ids):
                                original_fact_id = data.fact_ids[fact_idx]
                                fact_data["original_fact_id"] = original_fact_id
                                
                                # Add subgraph metadata from HeteroData
                                fact_data["subgraph_metadata"] = {
                                    "primary_ticker": getattr(data, 'primary_ticker', None),
                                    "reported_date": getattr(data, 'reported_date', None),
                                    "eps_surprise": getattr(data, 'eps_surprise', None),
                                    "label": getattr(data, 'label', None),
                                    "fact_count": getattr(data, 'fact_count', None)
                                }
                                
                                # If we have raw_sg, try to get the full original fact data
                                if raw_sg is not None and g < len(raw_sg):
                                    subgraph = raw_sg[g]
                                    fact_list = subgraph.get('fact_list', [])
                                    # Find the fact with matching fact_id
                                    for orig_fact in fact_list:
                                        if orig_fact.get('fact_id') == original_fact_id:
                                            fact_data["original_fact"] = {
                                                "fact_id": orig_fact.get('fact_id'),
                                                "raw_text": orig_fact.get('raw_text'),
                                                "date": orig_fact.get('date'),
                                                "event_type": orig_fact.get('event_type'),
                                                "sentiment": orig_fact.get('sentiment'),
                                                "delta_days": orig_fact.get('delta_days'),
                                                "tickers": orig_fact.get('tickers'),
                                                "event_cluster_id": orig_fact.get('event_cluster_id'),
                                                "source_article_index": orig_fact.get('source_article_index')
                                            }
                                            break
                            
                            all_results.append(fact_data)
                            all_agg_rows.append(fact_data)
                    
                    except Exception as e:
                        print(f"Error processing graph {g}: {e}")
                        continue
    
    # Save comprehensive results
    comprehensive_json = os.path.join(OUT_DIR, "comprehensive_explanations.json")
    with open(comprehensive_json, "w", encoding="utf-8") as fj:
        json.dump({
            "metadata": {
                "total_models": len(model_paths),
                "total_predictions": len(all_results),
                "topk": TOPK,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "results": all_results
        }, fj, indent=2)
    
    # Save aggregate CSV
    agg_csv = os.path.join(OUT_DIR, "comprehensive_aggregate.csv")
    import csv
    with open(agg_csv, "w", newline="", encoding="utf-8") as fcsv:
        if all_agg_rows:
            fieldnames = list(all_agg_rows[0].keys())
            wr = csv.DictWriter(fcsv, fieldnames=fieldnames)
            wr.writeheader()
            for r in all_agg_rows:
                wr.writerow(r)
    
    print(f"\n✅ Comprehensive explanations written to: {OUT_DIR}")
    print(f"     Comprehensive JSON: {comprehensive_json}")
    print(f"     Aggregate CSV: {agg_csv}")
    print(f"📊 Processed {len(model_paths)} models")
    print(f"📊 Generated {len(all_results)} positive predictions")
    print(f"📊 Top-{TOPK} facts per prediction")

if __name__ == "__main__":
    main()
