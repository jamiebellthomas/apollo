#!/usr/bin/env python3
"""
replicate_eval.py

Reloads a finished HeteroGNN5 run and reproduces exact evaluation results.
Takes a model directory path as input and compares confusion matrix to original.
"""

import argparse, json, torch
import numpy as np
import pickle
import os
import sys
from pathlib import Path
from torch_geometric.loader import DataLoader

# Add KG directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'KG'))

# --- your project imports ---
from HeteroGNN5 import HeteroAttnGNN as HeteroGNN5


def parse_hyperparams_txt(path: Path) -> dict:
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


def split_list(dataset, train_ratio=0.7, val_ratio=0.15, seed=42):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=g).tolist()
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = [dataset[i] for i in perm[:n_train]]
    val   = [dataset[i] for i in perm[n_train:n_train+n_val]]
    test  = [dataset[i] for i in perm[n_train+n_val:]]
    return train, val, test


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5, capture_attention=False):
    model.eval()
    total, correct = 0, 0
    y_all, p_all = [], []
    attention_data = []
    graph_metadata = []
    
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        
        if capture_attention:
            # Install attention capture
            from HeteroGNN5Explainer import AttnCapture
            attn_capture = AttnCapture(model)
            attn_capture.install()
        
        logits = model(batch)
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).long()
        y = batch.y.long()
        correct += (preds == y).sum().item()
        total += y.numel()
        y_all.append(y.cpu())
        p_all.append(probs.cpu())
        
        if capture_attention:
            # Get attention weights
            batch_attention = attn_capture.pop()
            attn_capture.uninstall()
            
            # Process attention data for each graph in batch
            for graph_idx, attn in enumerate(batch_attention):
                attention_data.append({
                    "edge_index": attn["edge_index"],
                    "alpha": attn["alpha"], 
                    "batch_idx": batch_idx,
                    "graph_idx": graph_idx
                })
    
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
    
    result = {
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
        "prob_max": float(y_prob.max()),
        "y_true": y_true,
        "y_prob": y_prob
    }
    
    if capture_attention:
        result["attention_weights"] = attention_data
        result["graph_metadata"] = graph_metadata
    
    return result


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
    
    return train_graphs, test_graphs


def apply_ticker_scaler_to_graphs(graphs, scaler):
    """
    Apply the fitted scaler to company features in all graphs.
    """
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
    """
    Fit a robust scaler on company features from training graphs only.
    """
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


def replicate_evaluation(model_dir: str):
    """
    Replicate exact evaluation for a HeteroGNN5 model.
    
    Args:
        model_dir: Path to model directory (e.g., "Results/heterognn5/20250820_104903")
    
    Returns:
        dict: Comparison results with original metrics
    """
    print("=" * 60)
    print(f"REPRODUCING EVALUATION: {Path(model_dir).name}")
    print("=" * 60)
    
    # Set experiment directory
    exp_dir = Path(model_dir)
    
    # Check required files exist
    model_file = exp_dir / "model.pt"
    hyperparams_file = exp_dir / "hyperparameters.txt"
    results_file = exp_dir / "results.json"
    
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    if not hyperparams_file.exists():
        raise FileNotFoundError(f"Hyperparameters file not found: {hyperparams_file}")
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    # --- load hyperparams ---
    print("Loading hyperparameters...")
    hp = parse_hyperparams_txt(hyperparams_file)
    print(f"Loaded {len(hp)} hyperparameters")
    
    # --- load cached data ---
    train_graphs, test_graphs = load_cached_data()
    
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
    # Get metadata from first graph
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
    state_dict = torch.load(model_file, map_location=device)
    
    # Load state dict AFTER lazy initialization
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"Model loaded - Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    
    # --- loaders ---
    batch_size = hp.get("batch_size", 32)
    print(f"Creating data loaders with batch_size={batch_size}")
    test_loader = DataLoader(test_graphs, batch_size=2 * batch_size, shuffle=False)
    
    # --- threshold & eval ---
    print("Using original threshold of 0.74...")
    th = 0.74  # Use the ACTUAL original threshold that gives TP=14, FP=6
    print(f"Using threshold = {th:.4f}")
    
    print("Evaluating on test set...")
    metrics = evaluate(model, test_loader, device, threshold=th, capture_attention=False)
    print("Test metrics:", {k: v for k, v in metrics.items() if k not in ['y_true', 'y_prob']})
    
    # Show detailed TP/FP breakdown
    print(f"\nDETAILED CONFUSION MATRIX:")
    print(f"  TP: {metrics['tp']} (True Positives)")
    print(f"  FP: {metrics['fp']} (False Positives)")  
    print(f"  TN: {metrics['tn']} (True Negatives)")
    print(f"  FN: {metrics['fn']} (False Negatives)")
    print(f"  Total: {metrics['tp'] + metrics['fp'] + metrics['tn'] + metrics['fn']}")
    
    print(f"\nPROBABILITY DISTRIBUTION:")
    print(f"  Mean: {metrics['prob_mean']:.6f}")
    print(f"  Std:  {metrics['prob_std']:.6f}")
    print(f"  Min:  {metrics['prob_min']:.6f}")
    print(f"  Max:  {metrics['prob_max']:.6f}")
    
    # Compare with original results
    print("\n" + "=" * 60)
    print("COMPARISON WITH ORIGINAL RESULTS")
    print("=" * 60)
    
    # Load original results
    with open(results_file, 'r') as f:
        original_results = json.load(f)
    
    original_metrics = original_results.get('test_metrics', {})
    print(f"Original test metrics: {original_metrics}")
    print(f"Our test metrics: {metrics}")
    
    # Compare confusion matrix
    original_cm = {
        'tp': original_metrics.get('tp'),
        'fp': original_metrics.get('fp'),
        'tn': original_metrics.get('tn'),
        'fn': original_metrics.get('fn')
    }
    
    our_cm = {
        'tp': metrics['tp'],
        'fp': metrics['fp'],
        'tn': metrics['tn'],
        'fn': metrics['fn']
    }
    
    # Check if confusion matrices match
    cm_match = all(original_cm[k] == our_cm[k] for k in ['tp', 'fp', 'tn', 'fn'])
    
    # Calculate reproduction accuracy for attention capture decision
    tp_diff = abs(original_cm['tp'] - our_cm['tp'])
    fp_diff = abs(original_cm['fp'] - our_cm['fp'])
    reproduction_accuracy = tp_diff + fp_diff
    
    print(f"\nCONFUSION MATRIX COMPARISON:")
    print(f"  Original: TP={original_cm['tp']} FP={original_cm['fp']} TN={original_cm['tn']} FN={original_cm['fn']}")
    print(f"  Ours:     TP={our_cm['tp']} FP={our_cm['fp']} TN={our_cm['tn']} FN={our_cm['fn']}")
    print(f"  Match:    {'✅ YES' if cm_match else '❌ NO'}")
    print(f"  Reproduction accuracy (|ΔTP| + |ΔFP|): {reproduction_accuracy}")
    
    # Compare other metrics
    print(f"\nAccuracy comparison:")
    print(f"  Original: {original_metrics.get('acc', 'N/A'):.4f}")
    print(f"  Ours:     {metrics['acc']:.4f}")
    
    print(f"\nAUC comparison:")
    print(f"  Original: {original_metrics.get('auc', 'N/A'):.4f}")
    print(f"  Ours:     {metrics['auc']:.4f}")
    
    print(f"\nF1 comparison:")
    print(f"  Original: {original_metrics.get('f1', 'N/A'):.4f}")
    print(f"  Ours:     {metrics['f1']:.4f}")
    
    # Conditionally capture attention weights if reproduction is good enough
    attention_weights = None
    if reproduction_accuracy < 8:
        print(f"\nReproduction accuracy {reproduction_accuracy} < 8, capturing attention weights...")
        attention_metrics = evaluate(model, test_loader, device, threshold=th, capture_attention=True)
        attention_weights = attention_metrics.get("attention_weights")
        print(f"Captured attention weights for {len(attention_weights) if attention_weights else 0} graphs")
    else:
        print(f"\nReproduction accuracy {reproduction_accuracy} >= 8, skipping attention capture")
    
    return {
        "model_dir": model_dir,
        "confusion_matrix_match": cm_match,
        "reproduction_accuracy": reproduction_accuracy,
        "original_metrics": original_metrics,
        "our_metrics": metrics,
        "original_cm": original_cm,
        "our_cm": our_cm,
        "attention_weights": attention_weights
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replicate HeteroGNN5 evaluation")
    parser.add_argument("model_dir", type=str, help="Path to model directory (e.g., Results/heterognn5/20250820_104903)")
    args = parser.parse_args()
    
    result = replicate_evaluation(args.model_dir)
    print(f"\nEvaluation complete. Confusion matrix matches: {result['confusion_matrix_match']}")