#!/usr/bin/env python3
"""
Simple baseline test to diagnose if the issue is with the GNN or the data.
Uses a basic MLP on company features only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
import numpy as np
import os
import sys
from pathlib import Path
import datetime
import json
import logging
import warnings
import time
from sentence_transformers import SentenceTransformer

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# Import only what we need
from SubGraphDataLoader import SubGraphDataLoader
from SubGraph import SubGraph

# Set environment variable to disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="transformers")


class SimpleBaseline(nn.Module):
    """
    Simple baseline model that uses features from node stores.
    Since PyTorch Geometric creates heterogeneous graphs with node stores, we'll access features from there.
    """
    def __init__(self, input_dim=1536):  # Default to fact feature dimension
        super().__init__()
        self.input_dim = input_dim
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
    
    def forward(self, data):
        # Try to get features from node stores (heterogeneous graph structure)
        if hasattr(data, 'node_stores') and len(data.node_stores) > 0:
            # Get features from the first node store (fact features)
            first_store = data.node_stores[0]
            if hasattr(first_store, 'x') and first_store.x is not None:
                features = first_store.x
                # Get batch information to do per-graph pooling
                if hasattr(first_store, 'batch') and first_store.batch is not None:
                    batch = first_store.batch
                    num_graphs = batch.max().item() + 1
                    outputs = []
                    
                    for i in range(num_graphs):
                        # Get nodes belonging to graph i
                        mask = (batch == i)
                        if mask.sum() > 0:
                            graph_features = features[mask]
                            # Mean pooling over nodes in this graph
                            graph_features = graph_features.mean(dim=0, keepdim=True)
                            outputs.append(self.classifier(graph_features).squeeze(-1))
                        else:
                            # Empty graph - use zero features
                            zero_features = torch.zeros(1, self.input_dim, device=features.device)
                            outputs.append(self.classifier(zero_features).squeeze(-1))
                    
                    return torch.stack(outputs).squeeze(-1)
                else:
                    # No batch info - treat as single graph
                    if features.dim() > 1 and features.size(0) > 1:
                        features = features.mean(dim=0, keepdim=True)
                    return self.classifier(features).squeeze(-1)
        
        # Fallback: try direct 'x' attribute (homogeneous graph)
        if hasattr(data, 'x') and data.x is not None:
            features = data.x
            if hasattr(data, 'batch') and data.batch is not None:
                batch = data.batch
                num_graphs = batch.max().item() + 1
                outputs = []
                
                for i in range(num_graphs):
                    mask = (batch == i)
                    if mask.sum() > 0:
                        graph_features = features[mask]
                        graph_features = graph_features.mean(dim=0, keepdim=True)
                        outputs.append(self.classifier(graph_features).squeeze(-1))
                    else:
                        zero_features = torch.zeros(1, self.input_dim, device=features.device)
                        outputs.append(self.classifier(zero_features).squeeze(-1))
                
                return torch.stack(outputs).squeeze(-1)
            else:
                if features.dim() > 1 and features.size(0) > 1:
                    features = features.mean(dim=0, keepdim=True)
                return self.classifier(features).squeeze(-1)
        
        # Final fallback: use constant input
        batch_size = data.y.size(0) if hasattr(data, 'y') else 1
        device = data.y.device if hasattr(data, 'y') else 'cpu'
        constant_input = torch.ones(batch_size, self.input_dim, device=device)
        return self.classifier(constant_input).squeeze(-1)


# Simple logger class
class SimpleLogger:
    def __init__(self, base_dir="baseline_results"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Create timestamped experiment folder
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.base_dir / timestamp
        self.experiment_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.log_file = self.experiment_dir / "output_log.txt"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_experiment_start(self, **kwargs):
        self.logger.info("=" * 60)
        self.logger.info("BASELINE EXPERIMENT START")
        self.logger.info("=" * 60)
        for key, value in kwargs.items():
            self.logger.info(f"{key}: {value}")
    
    def log_experiment_end(self, test_metrics, history, best_epoch):
        self.logger.info("=" * 60)
        self.logger.info("BASELINE EXPERIMENT END")
        self.logger.info("=" * 60)
        self.logger.info(f"Best epoch: {best_epoch}")
        self.logger.info(f"Final test metrics: {test_metrics}")


# Global cached sentence transformer
_CACHED_TRANSFORMER = None

def get_cached_transformer():
    """Get or create a cached sentence transformer."""
    global _CACHED_TRANSFORMER
    if _CACHED_TRANSFORMER is None:
        print("[transformer] Loading sentence transformer model...")
        try:
            cache_dir = os.path.join(os.path.dirname(__file__), "model_cache")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _CACHED_TRANSFORMER = SentenceTransformer(
                    "all-mpnet-base-v2",
                    cache_folder=cache_dir
                )
            print("[transformer] Model loaded successfully")
        except Exception as e:
            print(f"[transformer] Error loading model: {e}")
            raise
    return _CACHED_TRANSFORMER


def encode_all_to_heterodata(loader, batch_size=20):
    """Encode every SubGraph to HeteroData."""
    graphs = []
    total_subgraphs = len(loader.items)
    
    print(f"[encode] Processing {total_subgraphs} subgraphs in batches of {batch_size}")
    
    transformer = get_cached_transformer()
    
    for i in range(0, total_subgraphs, batch_size):
        batch_end = min(i + batch_size, total_subgraphs)
        batch_items = loader.items[i:batch_end]
        
        print(f"[encode] Processing batch {i//batch_size + 1}/{(total_subgraphs + batch_size - 1)//batch_size}")
        
        for j, sg in enumerate(batch_items):
            try:
                if j % 5 == 0:
                    print(f"  [encode] Processed {j+1}/{len(batch_items)} in current batch")
                
                data = sg.to_pyg_data(text_encoder=transformer)
                graphs.append(data)
                
                if j % 3 == 0:
                    time.sleep(0.05)
                    
            except Exception as e:
                print(f"[encode] Error processing subgraph {i+j+1}: {e}")
                continue
        
        if batch_end < total_subgraphs:
            time.sleep(1.0)
    
    print(f"[encode] Successfully processed {len(graphs)}/{total_subgraphs} subgraphs")
    return graphs, loader.items


def attach_y_and_meta(dataset, subgraphs):
    """Attach graph-level label into .y and stash primary ticker."""
    for g, sg in zip(dataset, subgraphs):
        g.y = g["graph_label"].float().view(-1)
        g.meta_primary_ticker = getattr(sg, "primary_ticker", None)


def split_list(dataset, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Simple randomized split."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=g).tolist()
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    idx_train = perm[:n_train]
    idx_val = perm[n_train:n_train+n_val]
    idx_test = perm[n_train+n_val:]

    def class_stats(split, name):
        y = torch.cat([g.y for g in split]).cpu().numpy()
        pos = (y > 0.5).sum()
        neg = len(y) - pos
        print(f"[split] {name}: total={len(y)}, pos={pos} ({pos/len(y):.2%}), neg={neg} ({neg/len(y):.2%})")

    train_set = [dataset[i] for i in idx_train]
    val_set = [dataset[i] for i in idx_val]
    test_set = [dataset[i] for i in idx_test]

    class_stats(train_set, "train")
    class_stats(val_set, "val")
    class_stats(test_set, "test")

    return train_set, val_set, test_set, idx_train, idx_val, idx_test


def make_loader(ds, batch_size, shuffle):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )


def compute_pos_weight(dataset):
    """Compute positive class weight."""
    y = torch.cat([g.y for g in dataset]).float()
    pos = (y > 0.5).sum().item()
    neg = len(y) - pos
    
    if pos == 0:
        return torch.tensor(1.0, dtype=torch.float)
    
    ratio = neg / pos
    print(f"[class_balance] Positive: {pos}, Negative: {neg}, Ratio: {ratio:.2f}")
    
    return torch.tensor(ratio, dtype=torch.float)


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        y = batch.y
        loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, report_loss=True, detailed_analysis=False, threshold=0.5, criterion=None, return_raw_predictions=False):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    if criterion is None:
        eval_criterion = nn.BCEWithLogitsLoss(reduction="sum")
    else:
        eval_criterion = criterion

    all_probs = []
    all_y = []
    all_preds = []
    raw_predictions = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        y = batch.y

        if report_loss:
            loss = eval_criterion(logits, y)
            total_loss += loss.item()

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).long()
        correct += (preds == y.long()).sum().item()
        total += y.numel()

        all_probs.append(probs.detach().cpu())
        all_y.append(y.detach().cpu())
        all_preds.append(preds.detach().cpu())
        
        if return_raw_predictions:
            batch_probs = probs.detach().cpu().numpy()
            batch_y = y.detach().cpu().numpy()
            batch_preds = preds.detach().cpu().numpy()
            
            for i in range(len(batch_probs)):
                raw_predictions.append({
                    'probability': float(batch_probs[i]),
                    'actual_label': int(batch_y[i]),
                    'predicted_label': int(batch_preds[i])
                })

    metrics = {"acc": correct / max(total, 1)}
    if report_loss:
        metrics["loss"] = total_loss / max(total, 1)

    # AUC
    try:
        from sklearn.metrics import roc_auc_score
        y_true = torch.cat(all_y).numpy()
        y_prob = torch.cat(all_probs).numpy()
        if len(np.unique(y_true)) == 2:
            metrics["auc"] = float(roc_auc_score(y_true, y_prob))
        else:
            metrics["auc"] = float("nan")
    except Exception:
        metrics["auc"] = float("nan")

    # Detailed analysis
    if detailed_analysis:
        y_true = torch.cat(all_y).numpy()
        y_pred = torch.cat(all_preds).numpy()
        y_prob = torch.cat(all_probs).numpy()
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        detailed_metrics = {
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "pred_0": int(np.sum(y_pred == 0)), "pred_1": int(np.sum(y_pred == 1)),
            "true_0": int(np.sum(y_true == 0)), "true_1": int(np.sum(y_true == 1)),
            "prob_mean": float(np.mean(y_prob)), "prob_std": float(np.std(y_prob)),
            "prob_min": float(np.min(y_prob)), "prob_max": float(np.max(y_prob)),
        }
        
        metrics.update(detailed_metrics)

    if return_raw_predictions:
        metrics["raw_predictions"] = raw_predictions

    return metrics


def find_optimal_threshold(model, val_loader, device, criterion=None):
    """Find optimal threshold."""
    model.eval()
    all_probs = []
    all_y = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.detach().cpu())
            all_y.append(batch.y.detach().cpu())
    
    all_probs = torch.cat(all_probs).numpy()
    all_y = torch.cat(all_y).numpy()
    
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in np.arange(0.1, 0.9, 0.05):
        preds = (all_probs >= threshold).astype(int)
        
        tp = np.sum((all_y == 1) & (preds == 1))
        fp = np.sum((all_y == 0) & (preds == 1))
        fn = np.sum((all_y == 1) & (preds == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"[threshold] Optimal threshold: {best_threshold:.3f} (F1: {best_f1:.3f})")
    return best_threshold


def run_baseline_test(
    n_facts=35,
    limit=None,
    use_cache=True,
    batch_size=32,
    epochs=50,
    lr=1e-3,
    weight_decay=1e-4,
    seed=42,
    loss_type="weighted_bce",
    early_stopping=True,
    patience=10
):
    """Run the simple baseline test."""
    
    # Initialize logger
    logger = SimpleLogger()
    
    # Log experiment start
    logger.log_experiment_start(
        n_facts=n_facts,
        limit=limit,
        use_cache=use_cache,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        loss_type=loss_type,
        early_stopping=early_stopping,
        patience=patience,
        model_type="simple_baseline"
    )
    
    print(f"[baseline] Starting simple baseline test with n_facts={n_facts}")
    
    # Load data
    print("[baseline] Loading data...")
    graphs = None
    raw_sg = None
    
    if use_cache:
        try:
            from cache_dataset import load_cached_dataset
            print("[baseline] Attempting to load from cache...")
            graphs, raw_sg = load_cached_dataset(n_facts=n_facts, limit=limit)
            if graphs is not None and raw_sg is not None:
                print(f"[baseline] ✅ Loaded {len(graphs)} graphs from cache")
            else:
                print("[baseline] ❌ Cache loading failed")
                graphs = None
                raw_sg = None
        except Exception as e:
            print(f"[baseline] Cache error: {e}")
            graphs = None
            raw_sg = None
    
    if graphs is None or raw_sg is None:
        print("[baseline] Processing from scratch...")
        subgraphs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), config.SUBGRAPHS_JSONL)
        loader = SubGraphDataLoader(min_facts=n_facts, limit=limit, jsonl_path=subgraphs_path)
        graphs, raw_sg = encode_all_to_heterodata(loader)
    
    # Attach labels
    attach_y_and_meta(graphs, raw_sg)
    
    # Split data
    train_set, val_set, test_set, train_indices, val_indices, test_indices = split_list(
        dataset=graphs, train_ratio=0.7, val_ratio=0.15, seed=seed
    )
    
    # Create loaders
    train_loader = make_loader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(val_set, batch_size=batch_size*2, shuffle=False)
    test_loader = make_loader(test_set, batch_size=batch_size*2, shuffle=False)
    
    # Create model with fact features
    print("[baseline] Creating simple baseline model with fact features...")
    
    # Get sample batch to determine input dimension
    sample_batch = next(iter(train_loader))
    
    # Check if fact features are available
    if hasattr(sample_batch, 'node_stores') and len(sample_batch.node_stores) > 0:
        first_store = sample_batch.node_stores[0]
        if hasattr(first_store, 'x') and first_store.x is not None:
            input_dim = first_store.x.size(-1)
            print(f"[baseline] Using node store features, dimension: {input_dim}")
        else:
            input_dim = 1536  # Default fact feature dimension
            print(f"[baseline] No features in first node store, using default dimension: {input_dim}")
    elif hasattr(sample_batch, 'x') and sample_batch.x is not None:
        input_dim = sample_batch.x.size(-1)
        print(f"[baseline] Using homogeneous graph features, dimension: {input_dim}")
    else:
        input_dim = 1536  # Default fact feature dimension
        print(f"[baseline] No features found, using default dimension: {input_dim}")
    
    model = SimpleBaseline(input_dim)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[baseline] Using device: {device}")
    
    model.to(device)
    
    # Debug initial outputs
    sample_batch = next(iter(train_loader))
    model.eval()
    with torch.no_grad():
        sample_outputs = model(sample_batch.to(device))
        sample_probs = torch.sigmoid(sample_outputs)
        print(f"[baseline] Sample outputs: {sample_outputs[:5]}")
        print(f"[baseline] Sample probs: {sample_probs[:5]}")
        print(f"[baseline] Sample targets: {sample_batch.y[:5]}")
    
    model.train()
    
    # Setup loss and optimizer
    pos_weight = compute_pos_weight(train_set).to(device)
    print(f"[baseline] Class weight: {pos_weight:.3f}")
    
    if loss_type == "weighted_bce":
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_type == "bce":
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Training loop
    print("[baseline] Starting training...")
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_auc": []}
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    best_epoch = 0
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer,
            criterion=criterion, device=device, grad_clip=1.0
        )
        
        # Validate
        val_metrics = evaluate(
            model=model, loader=val_loader, device=device,
            detailed_analysis=True, threshold=0.5, criterion=criterion
        )
        
        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics.get("loss", float("nan")))
        history["val_acc"].append(val_metrics["acc"])
        history["val_auc"].append(val_metrics.get("auc", float("nan")))
        
        current_val_loss = val_metrics.get("loss", float("inf"))
        
        # Check for improvement
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            best_epoch = epoch
        else:
            patience_counter += 1
        
        # Print progress
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
              f"val_loss={current_val_loss:.4f} | val_acc={val_metrics['acc']:.3f} | "
              f"val_auc={val_metrics.get('auc', float('nan')):.3f} | patience={patience_counter}")
        
        # Early stopping
        if early_stopping and patience_counter >= patience:
            print(f"[baseline] Early stopping at epoch {epoch}")
            break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Find optimal threshold
    print("[baseline] Finding optimal threshold...")
    optimal_threshold = find_optimal_threshold(model, val_loader, device, criterion)
    
    # Final test
    test_metrics = evaluate(
        model=model, loader=test_loader, device=device,
        detailed_analysis=True, threshold=optimal_threshold, criterion=criterion,
        return_raw_predictions=True
    )
    
    print(f"[baseline] FINAL TEST | loss={test_metrics.get('loss', float('nan')):.4f} | "
          f"acc={test_metrics['acc']:.3f} | auc={test_metrics.get('auc', float('nan')):.3f} | "
          f"threshold={optimal_threshold:.3f} | best_epoch={best_epoch}")
    
    # Log experiment end
    logger.log_experiment_end(test_metrics, history, best_epoch)
    
    return model, test_metrics, history


if __name__ == "__main__":
    # Run baseline test with a limit for faster processing
    model, test_metrics, history = run_baseline_test(
        n_facts=35,
        limit=100,  # Limit to 100 subgraphs for faster processing
        use_cache=True,
        batch_size=32,
        epochs=50,
        lr=1e-3,
        weight_decay=1e-4,
        seed=42,
        loss_type="weighted_bce",
        early_stopping=True,
        patience=10
    )
    
    print("\n=== BASELINE TEST RESULTS ===")
    print(f"Test Accuracy: {test_metrics['acc']:.4f}")
    print(f"Test AUC: {test_metrics.get('auc', float('nan')):.4f}")
    
    if "tp" in test_metrics:
        print(f"Confusion Matrix:")
        print(f"TP: {test_metrics['tp']}, TN: {test_metrics['tn']}")
        print(f"FP: {test_metrics['fp']}, FN: {test_metrics['fn']}")
        print(f"Precision: {test_metrics['precision']:.4f}")
        print(f"Recall: {test_metrics['recall']:.4f}")
        print(f"F1: {test_metrics['f1']:.4f}") 