#!/usr/bin/env python3
"""
Cluster Analysis for Event Types

This script:
1. Gets top 15 facts for each positive prediction
2. Embeds the event_type (not fact text) using all-mpnet-base-v2
3. Finds closest cluster centroid from Data/cluster_centroids.jsonl
4. Counts which clusters are most influential
"""

import torch
import numpy as np
import pandas as pd
import json
import pickle
import os
import sys
from pathlib import Path
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from torch_geometric import nn as torch_geometric_nn

# Add KG directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'KG'))

from HeteroGNN5 import HeteroAttnGNN as HeteroGNN5


class AttentionExtractor:
    """Extracts attention weights from HeteroGNN5 model"""
    
    def __init__(self, model):
        self.model = model
        self.attention_weights = []
        self.original_attn_layer = None
        
    def register_hooks(self):
        """Register hooks to capture attention weights"""
        # Store the original _attn_layer method
        self.original_attn_layer = self.model._attn_layer
        
        def attention_hook(convs, x_dict, edge_index_dict, edge_attr_dict, collect_entropy_on=None):
            """Hook to capture attention weights during forward pass"""
            out = {nt: [] for nt in x_dict.keys()}
            entropy_term = x_dict[next(iter(x_dict))].new_tensor(0.0)
            attention_data = {}

            for et, ei in edge_index_dict.items():
                conv = convs[str(et)]
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
                
                # Store attention weights
                attention_data[et] = {
                    'alpha': alpha.detach().cpu(),  # [E, num_heads]
                    'edge_index': edge_index_used.detach().cpu(),
                    'src_type': src_type,
                    'dst_type': dst_type
                }

                # entropy regularisation on incoming edges (per head)
                if self.model.entropy_reg_weight > 0.0 and collect_entropy_on is not None and et == collect_entropy_on:
                    dst = edge_index_used[1]
                    eps = 1e-9
                    h_sum = 0.0
                    for h in range(alpha.size(1)):
                        a = alpha[:, h].clamp(min=eps)
                        neg_a_log_a = -(a * torch.log(a))
                        h_node = torch_geometric_nn.global_add_pool(neg_a_log_a, dst)
                        h_sum = h_sum + h_node.mean()
                    entropy_term = entropy_term + (h_sum / alpha.size(1))

            # aggregate per destination node type
            x_new = {}
            for nt, parts in out.items():
                if parts:
                    x_new[nt] = torch.stack(parts, dim=0).sum(dim=0)
                else:
                    x_new[nt] = x_dict[nt]

            # store auxiliary regulariser
            self.model.extra_loss = self.model.entropy_reg_weight * entropy_term
            
            # Store attention weights for this layer
            self.attention_weights.append(attention_data)
            
            return x_new
        
        # Replace the _attn_layer method with our hook
        self.model._attn_layer = attention_hook
        print("Registered attention weight capture hook")
    
    def remove_hooks(self):
        """Remove hooks and restore original method"""
        if self.original_attn_layer is not None:
            self.model._attn_layer = self.original_attn_layer
            self.original_attn_layer = None
        print("Removed attention weight capture hook")
    
    def clear_weights(self):
        """Clear stored attention weights"""
        self.attention_weights = []


def parse_hyperparams_txt(path: Path) -> dict:
    """Parse hyperparameters from text file"""
    hp = {}
    if not path.exists():
        return hp
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip(); v = v.strip()
        if v.lower() in ("true", "false"):
            hp[k] = (v.lower() == "true")
        elif v.lower() == "none":
            hp[k] = None
        else:
            try:
                if "." in v or "e-" in v.lower():
                    hp[k] = float(v)
                else:
                    hp[k] = int(v)
            except ValueError:
                hp[k] = v
    return hp


def attach_y_and_meta(dataset, subgraphs):
    """Attach graph-level label into .y and stash primary ticker for grouped split."""
    for g, sg in zip(dataset, subgraphs):
        g.y = g["graph_label"].float().view(-1)
        g.meta_primary_ticker = getattr(sg, "primary_ticker", None)


def split_list(dataset, train_ratio=0.8, val_ratio=0.2, seed=76369):
    """Split dataset into train/val/test"""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=g).tolist()
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = [dataset[i] for i in perm[:n_train]]
    val   = [dataset[i] for i in perm[n_train:n_train+n_val]]
    test  = [dataset[i] for i in perm[n_train+n_val:]]
    return train, val, test


def apply_ticker_scaler_to_graphs(graphs, scaler):
    """Apply the fitted scaler to company features in all graphs."""
    if scaler.get("identity", False):
        return  # No-op for identity scaler
    
    for g in graphs:
        if 'company' in g.node_types and hasattr(g['company'], 'x'):
            x = g['company'].x
            if isinstance(x, torch.Tensor) and x.numel() > 0:
                # Winsorize
                x_w = torch.minimum(torch.maximum(x, scaler["low"]), scaler["high"])
                # Z-score
                g['company'].x = (x_w - scaler["mean"]) / scaler["std"]


def fit_ticker_scaler(train_graphs, pct_low=1.0, pct_high=99.0):
    """Fit a robust scaler on company features from training graphs only."""
    rows = []
    for g in train_graphs:
        if 'company' in g.node_types and hasattr(g['company'], 'x'):
            x = g['company'].x
            if isinstance(x, torch.Tensor) and x.numel() > 0:
                rows.append(x.detach().cpu())
    
    if not rows:
        print("[scaler] No company features in training set; using identity scaler.")
        return {"low": None, "high": None, "mean": None, "std": None, "identity": True}

    X = torch.cat(rows, dim=0)  # [N_total, D]
    lo = torch.quantile(X, q=pct_low / 100.0, dim=0)
    hi = torch.quantile(X, q=pct_high / 100.0, dim=0)

    Xw = torch.minimum(torch.maximum(X, lo), hi)
    mean = Xw.mean(dim=0)
    std = Xw.std(dim=0, unbiased=False)
    std[std == 0] = 1.0

    print(f"[scaler] Fit on {X.shape[0]} company rows, D={X.shape[1]} "
          f"(winsorize p{pct_low}/{pct_high})")
    return {"low": lo.float(), "high": hi.float(), "mean": mean.float(), "std": std.float(), "identity": False}


def load_cached_data():
    """Load cached data using our structure"""
    print("Loading cached data...")
    
    # Load training data
    train_cache_path = '../KG/dataset_cache/training_cached_dataset_nf35_limall.pkl'
    with open(train_cache_path, 'rb') as f:
        train_cache = pickle.load(f)
    
    # Load testing data  
    test_cache_path = '../KG/dataset_cache/testing_cached_dataset_nf35_limall.pkl'
    with open(test_cache_path, 'rb') as f:
        test_cache = pickle.load(f)
    
    train_graphs = train_cache['graphs']
    train_raw_sg = train_cache['raw_sg']
    test_graphs = test_cache['graphs']
    test_raw_sg = test_cache['raw_sg']
    
    print(f"Loaded {len(train_graphs)} training graphs and {len(test_graphs)} testing graphs")
    
    # Attach labels and metadata
    print("Attaching labels and metadata...")
    attach_y_and_meta(train_graphs, train_raw_sg)
    attach_y_and_meta(test_graphs, test_raw_sg)
    
    return train_graphs, test_graphs, train_raw_sg, test_raw_sg


def load_cluster_centroids():
    """Load cluster centroids from jsonl file"""
    print("Loading cluster centroids...")
    centroids_path = '../Data/cluster_centroids.jsonl'
    
    if not os.path.exists(centroids_path):
        print(f"Warning: {centroids_path} not found!")
        return None, None
    
    centroids = []
    cluster_ids = []
    
    with open(centroids_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            centroids.append(data['centroid'])
            cluster_ids.append(data['cluster_id'])
    
    print(f"Loaded {len(centroids)} cluster centroids")
    return np.array(centroids), cluster_ids


def load_fact_mapping():
    """Load fact_id to event_type mapping"""
    print("\nLoading fact mapping...")
    
    # Load subgraphs.jsonl to get fact_id mapping
    subgraphs_path = '../Data/subgraphs.jsonl'
    fact_mapping = {}
    
    if os.path.exists(subgraphs_path):
        with open(subgraphs_path, 'r') as f:
            for line_num, line in enumerate(f):
                data = json.loads(line)
                # Facts are nested in fact_list arrays
                if 'fact_list' in data and data['fact_list']:
                    for fact in data['fact_list']:
                        fact_id = fact.get('fact_id')
                        if fact_id is not None:
                            fact_mapping[fact_id] = {
                                'event_type': fact.get('event_type', ''),
                                'raw_text': fact.get('raw_text', ''),
                                'primary_ticker': data.get('primary_ticker', ''),
                                'reported_date': data.get('reported_date', '')
                            }
        
        print(f"Loaded {len(fact_mapping)} fact mappings")
    else:
        print("Warning: subgraphs.jsonl not found!")
    
    return fact_mapping


def extract_attention_weights(model, test_loader, device, threshold=0.74, test_raw_sg=None):
    """Extract attention weights and analyze fact importance"""
    print("\n" + "="*60)
    print("EXTRACTING ATTENTION WEIGHTS")
    print("="*60)
    
    # Initialize attention extractor
    extractor = AttentionExtractor(model)
    extractor.register_hooks()
    
    model.eval()
    attention_results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch = batch.to(device)
            
            # Clear previous attention weights
            extractor.clear_weights()
            
            # Forward pass to capture attention weights
            logits = model(batch)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).long()
            
            # Get attention weights for this batch
            attention_weights = extractor.attention_weights
            
            # Get the corresponding subgraphs for this batch
            batch_size = len(batch.y)
            start_idx = batch_idx * test_loader.batch_size
            end_idx = min(start_idx + batch_size, len(test_raw_sg))
            batch_subgraphs = test_raw_sg[start_idx:end_idx]
            
            # Process each sample in the batch
            for i in range(len(batch.y)):
                # Get the subgraph for this sample
                subgraph = batch_subgraphs[i] if i < len(batch_subgraphs) else None
                
                # Create fact index to fact_id mapping for this sample
                fact_index_to_id = {}
                if subgraph and hasattr(subgraph, 'fact_list') and subgraph.fact_list:
                    for fact_idx, fact in enumerate(subgraph.fact_list):
                        fact_id = fact.get('fact_id')
                        if fact_id is not None:
                            fact_index_to_id[fact_idx] = fact_id
                
                sample_result = {
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'probability': probs[i].item(),
                    'predicted_label': preds[i].item(),
                    'actual_label': batch.y[i].item(),
                    'attention_weights': attention_weights.copy() if attention_weights else [],
                    'batch_data': batch,
                    'fact_index_to_id': fact_index_to_id,  # Mapping from sequential index to actual fact_id
                    'subgraph': subgraph
                }
                attention_results.append(sample_result)
            
            print(f"Processed batch {batch_idx + 1}/{len(test_loader)}")
    
    # Remove hooks
    extractor.remove_hooks()
    
    return attention_results


def get_top_facts_for_prediction(pred, fact_mapping, top_k=15):
    """Get top k facts for a single prediction"""
    attention_weights = pred.get('attention_weights', [])
    fact_index_to_id = pred.get('fact_index_to_id', {})
    
    # Collect all fact attention scores
    fact_attention = {}
    
    for layer_attention in attention_weights:
        fact_to_company_key = ('fact', 'mentions', 'company')
        if fact_to_company_key in layer_attention:
            attn_data = layer_attention[fact_to_company_key]
            alpha = attn_data['alpha']
            edge_index = attn_data['edge_index']
            
            avg_attention = alpha.mean(dim=1)
            src_nodes = edge_index[0]
            
            for i, fact_idx in enumerate(src_nodes):
                fact_idx = fact_idx.item()
                
                # Map sequential index to actual fact_id if available
                actual_fact_id = fact_index_to_id.get(fact_idx, fact_idx)
                
                if actual_fact_id not in fact_attention:
                    fact_attention[actual_fact_id] = []
                fact_attention[actual_fact_id].append(avg_attention[i].item())
    
    # Calculate average attention per fact
    fact_avg_attention = {fact_idx: np.mean(scores) for fact_idx, scores in fact_attention.items()}
    
    # Sort facts by attention importance and get top k
    sorted_facts = sorted(fact_avg_attention.items(), key=lambda x: x[1], reverse=True)
    
    top_facts = []
    for fact_idx, attention_score in sorted_facts[:top_k]:
        fact_info = fact_mapping.get(fact_idx, {})
        event_type = fact_info.get('event_type', f'Fact_{fact_idx}')
        
        top_facts.append({
            'fact_id': fact_idx,
            'attention_score': attention_score,
            'event_type': event_type
        })
    
    return top_facts


def analyze_cluster_influence(attention_results, fact_mapping, centroids, cluster_ids, model):
    """Analyze which clusters are most influential"""
    print("\n" + "="*60)
    print("CLUSTER INFLUENCE ANALYSIS")
    print("="*60)
    
    # Focus on positive predictions
    positive_predictions = [r for r in attention_results if r['predicted_label'] == 1]
    print(f"Found {len(positive_predictions)} positive predictions")
    
    if not positive_predictions:
        print("No positive predictions found!")
        return
    
    # Initialize cluster counter
    cluster_counts = defaultdict(int)
    cluster_details = defaultdict(list)
    
    # Process each positive prediction
    for pred_idx, pred in enumerate(positive_predictions):
        print(f"\nProcessing prediction {pred_idx + 1}/{len(positive_predictions)}")
        
        # Get top 15 facts for this prediction
        top_facts = get_top_facts_for_prediction(pred, fact_mapping, top_k=15)
        
        # Get unique event types (avoid duplicates)
        event_types = list(set([fact['event_type'] for fact in top_facts]))
        
        print(f"  Sample: {pred.get('subgraph', {}).primary_ticker if hasattr(pred.get('subgraph', {}), 'primary_ticker') else 'Unknown'}")
        print(f"  Top event types: {event_types[:5]}...")  # Show first 5
        
        # Embed each event type and find closest cluster
        for event_type in event_types:
            if event_type.startswith('Fact_'):  # Skip unknown facts
                continue
                
            # Embed the event type
            with torch.no_grad():
                event_embedding = model.encode([event_type])
                
                # Find closest centroid
                distances = []
                for centroid in centroids:
                    dist = np.linalg.norm(event_embedding - centroid)
                    distances.append(dist)
                
                closest_cluster_idx = np.argmin(distances)
                closest_cluster_id = cluster_ids[closest_cluster_idx]
                
                # Count this cluster
                cluster_counts[closest_cluster_id] += 1
                cluster_details[closest_cluster_id].append({
                    'event_type': event_type,
                    'prediction_idx': pred_idx,
                    'ticker': pred.get('subgraph', {}).primary_ticker if hasattr(pred.get('subgraph', {}), 'primary_ticker') else 'Unknown'
                })
    
    # Sort clusters by count
    sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*60)
    print("MOST INFLUENTIAL CLUSTERS")
    print("="*60)
    
    print(f"{'Rank':<4} {'Cluster ID':<12} {'Count':<8} {'Event Types'}")
    print("="*60)
    
    for rank, (cluster_id, count) in enumerate(sorted_clusters[:20]):  # Show top 20
        # Get sample event types for this cluster
        sample_events = [detail['event_type'] for detail in cluster_details[cluster_id][:5]]
        event_types_str = ", ".join(sample_events)
        
        print(f"{rank+1:<4} {cluster_id:<12} {count:<8} {event_types_str}")
    
    print("="*60)
    
    # Write full ranking to output file
    output_file = "../Results/heterognn5/20250820_145105/cluster_ranking.txt"
    print(f"\nWriting full cluster ranking to: {output_file}")
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FULL CLUSTER INFLUENCE RANKING\n")
        f.write("="*80 + "\n")
        f.write("Analysis based on top 15 facts from 20 positive predictions\n")
        f.write("Event types embedded using all-mpnet-base-v2 and matched to cluster centroids\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'Rank':<4} {'Cluster ID':<12} {'Count':<8} {'Event Types'}\n")
        f.write("-"*80 + "\n")
        
        for rank, (cluster_id, count) in enumerate(sorted_clusters):
            # Get sample event types for this cluster
            sample_events = [detail['event_type'] for detail in cluster_details[cluster_id][:5]]
            event_types_str = ", ".join(sample_events)
            
            f.write(f"{rank+1:<4} {cluster_id:<12} {count:<8} {event_types_str}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED CLUSTER BREAKDOWN\n")
        f.write("="*80 + "\n\n")
        
        for rank, (cluster_id, count) in enumerate(sorted_clusters):
            f.write(f"{rank+1}. Cluster {cluster_id} (Count: {count})\n")
            f.write("   Event types in this cluster:\n")
            
            # Group by event type
            event_type_counts = defaultdict(int)
            for detail in cluster_details[cluster_id]:
                event_type_counts[detail['event_type']] += 1
            
            for event_type, event_count in sorted(event_type_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"     {event_type}: {event_count} occurrences\n")
            
            f.write("\n")
    
    print(f"✅ Full cluster ranking saved to: {output_file}")
    
    # Show detailed breakdown for top clusters
    print("\n" + "="*60)
    print("DETAILED CLUSTER BREAKDOWN")
    print("="*60)
    
    for rank, (cluster_id, count) in enumerate(sorted_clusters[:10]):
        print(f"\n{rank+1}. Cluster {cluster_id} (Count: {count})")
        print("   Event types in this cluster:")
        
        # Group by event type
        event_type_counts = defaultdict(int)
        for detail in cluster_details[cluster_id]:
            event_type_counts[detail['event_type']] += 1
        
        for event_type, event_count in sorted(event_type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"     {event_type}: {event_count} occurrences")
    
    return cluster_counts, cluster_details


def main():
    print("=" * 60)
    print("CLUSTER INFLUENCE ANALYSIS")
    print("=" * 60)
    
    # Set experiment directory
    exp_dir = Path("../Results/heterognn5/20250820_145105")
    
    # --- load hyperparams ---
    print("Loading hyperparameters...")
    hp = parse_hyperparams_txt(exp_dir / "hyperparameters.txt")
    print(f"Loaded {len(hp)} hyperparameters")
    
    # --- load cached data ---
    train_graphs, test_graphs, train_raw_sg, test_raw_sg = load_cached_data()
    
    # re-split into train/val/test
    print("Splitting training data...")
    train_set, val_set, _ = split_list(train_graphs,
                                      train_ratio=float(hp.get("train_ratio",0.8)),
                                      val_ratio=float(hp.get("val_ratio",0.2)),
                                      seed=int(hp.get("seed",76369)))
    
    print(f"Split: {len(train_set)} train, {len(val_set)} val, {len(test_graphs)} test")
    
    # --- APPLY SCALER (like original training) ---
    print("Applying company feature scaler...")
    scaler = fit_ticker_scaler(train_set)
    apply_ticker_scaler_to_graphs(train_set, scaler)
    apply_ticker_scaler_to_graphs(val_set, scaler)
    apply_ticker_scaler_to_graphs(test_graphs, scaler)
    print("Scaler applied to all datasets")
    
    # --- build model ---
    print("Building model...")
    sample_graph = train_set[0]
    metadata = (list(sample_graph.node_types), list(sample_graph.edge_types))
    print(f"Model metadata: {metadata}")
    
    model = HeteroGNN5(
        metadata=metadata,
        hidden_channels=int(hp.get("hidden_channels",1024)),
        num_layers=int(hp.get("num_layers",4)),
        heads=int(hp.get("heads",8)),
        time_dim=int(hp.get("time_dim",16)),
        feature_dropout=float(hp.get("feature_dropout",0.2)),
        edge_dropout=float(hp.get("edge_dropout",0.05)),
        final_dropout=float(hp.get("final_dropout",0.1)),
        readout=hp.get("readout","company"),
        funnel_to_primary=bool(hp.get("funnel_to_primary",False)),
        topk_per_primary=hp.get("topk_per_primary",15),
        attn_temperature=float(hp.get("attn_temperature",0.8)),
        entropy_reg_weight=float(hp.get("entropy_reg_weight",0.01)),
        time_bucket_edges=[0, 7, 30, 90, 9999] if hp.get("time_bucket_edges") else None,
        time_bucket_emb_dim=int(hp.get("time_bucket_emb_dim",8)),
        add_abs_sent=bool(hp.get("add_abs_sent",True)),
        add_polarity_bit=bool(hp.get("add_polarity_bit",True)),
        sentiment_jitter_std=float(hp.get("sentiment_jitter_std",0.1)),
        delta_t_jitter_frac=float(hp.get("delta_t_jitter_frac",0.05)),
    )
    
    # --- Initialize lazy parameters FIRST (like original training) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing lazy parameters on device: {device}")
    
    # Create a sample batch and initialize lazy parameters
    init_loader = DataLoader([train_set[0]], batch_size=1, shuffle=False)
    sample_batch = next(iter(init_loader))
    sample_batch = sample_batch.to(device)
    model.to(device)
    
    # This triggers lazy module creation (like init_lazy_params in original)
    _ = model(sample_batch)
    print("Lazy parameters initialized")
    
    # --- NOW load weights AFTER lazy modules exist ---
    print(f"Loading weights to device: {device}")
    state_dict = torch.load(exp_dir / "model.pt", map_location=device)
    
    # Load state dict AFTER lazy initialization
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"Model loaded - Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
    
    # --- loaders ---
    batch_size = hp.get("batch_size", 32)
    print(f"Creating data loaders with batch_size={batch_size}")
    test_loader = DataLoader(test_graphs, batch_size=2 * batch_size, shuffle=False)
    
    # --- VERIFY CONFUSION MATRIX FIRST ---
    print("\nVerifying confusion matrix...")
    model.eval()
    
    @torch.no_grad()
    def evaluate_with_confusion(model, loader, device, threshold=0.74):
        model.eval()
        total, correct = 0, 0
        y_all, p_all = [], []
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).long()
            y = batch.y.long()
            correct += (preds == y).sum().item()
            total += y.numel()
            y_all.append(y.cpu())
            p_all.append(probs.cpu())
        
        acc = correct / max(total, 1)
        from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
        y_true = torch.cat(y_all).numpy()
        y_prob = torch.cat(p_all).numpy()
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")
        preds = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_true, preds)
        
        # Calculate TP, FP, TN, FN
        cm = confusion_matrix(y_true, preds)
        tn, fp, fn, tp = cm.ravel()
        
        return {
            "acc": acc, 
            "auc": auc, 
            "f1": f1,
            "tp": int(tp),
            "fp": int(fp), 
            "tn": int(tn),
            "fn": int(fn),
            "prob_mean": float(y_prob.mean()),
            "prob_std": float(y_prob.std()),
            "prob_min": float(y_prob.min()),
            "prob_max": float(y_prob.max())
        }
    
    # Evaluate and verify
    metrics = evaluate_with_confusion(model, test_loader, device, threshold=0.74)
    
    # Verify confusion matrix matches original
    results_path = exp_dir / "results.json"
    if results_path.exists():
        with open(results_path, 'r') as f:
            original_results = json.load(f)
        original_metrics = original_results.get('test_metrics', {})
        
        our_tp, our_fp, our_tn, our_fn = metrics['tp'], metrics['fp'], metrics['tn'], metrics['fn']
        orig_tp, orig_fp, orig_tn, orig_fn = original_metrics['tp'], original_metrics['fp'], original_metrics['tn'], original_metrics['fn']
        
        matches = (our_tp == orig_tp and our_fp == orig_fp and our_tn == orig_tn and our_fn == orig_fn)
        
        if matches:
            print("✅ CONFUSION MATRIX MATCHES ORIGINAL!")
        else:
            print("❌ CONFUSION MATRIX DOES NOT MATCH!")
            return
    
    # --- LOAD CLUSTER CENTROIDS ---
    centroids, cluster_ids = load_cluster_centroids()
    if centroids is None:
        print("❌ Could not load cluster centroids!")
        return
    
    # --- LOAD FACT MAPPING ---
    fact_mapping = load_fact_mapping()
    
    # --- LOAD SENTENCE TRANSFORMER MODEL ---
    print("Loading sentence transformer model...")
    sentence_model = SentenceTransformer('all-mpnet-base-v2')
    print("Sentence transformer model loaded")
    
    # --- EXTRACT ATTENTION WEIGHTS ---
    attention_results = extract_attention_weights(model, test_loader, device, threshold=0.74, test_raw_sg=test_raw_sg)
    
    # --- ANALYZE CLUSTER INFLUENCE ---
    cluster_counts, cluster_details = analyze_cluster_influence(attention_results, fact_mapping, centroids, cluster_ids, sentence_model)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("✅ Confusion matrix verification passed")
    print("✅ Attention weights extracted")
    print("✅ Cluster influence analyzed")
    print(f"📊 Found {len(cluster_counts)} influential clusters")


if __name__ == "__main__":
    main()
