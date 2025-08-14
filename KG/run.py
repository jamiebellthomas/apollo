# train_hetero_gnn.py
from __future__ import annotations

"""
HeteroGNN Training Script with Run Scraping Feature

This script includes a feature to automatically scrape runs that end too early after the patience threshold.
If training ends less than MIN_EPOCHS_AFTER_PATIENCE epochs after the patience threshold would be hit,
the entire experiment directory and all logs are removed.

Example:
- If patience=20 and min_epochs_before_stopping=10, patience threshold is at epoch 30
- If MIN_EPOCHS_AFTER_PATIENCE=10, training should continue until at least epoch 40
- If training ends at epoch 35, the run will be scraped
- If training ends at epoch 45, the run will be kept

Configuration:
- ENABLE_RUN_SCRAPING: Set to True to enable the feature, False to disable
- MIN_EPOCHS_AFTER_PATIENCE: Minimum epochs that should occur after patience threshold
"""

from typing import List, Tuple, Dict
import math
import os
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
import time
import logging
import warnings
from sentence_transformers import SentenceTransformer
import json
import datetime
import sys
from pathlib import Path

# Add logging setup
import logging
from contextlib import redirect_stdout, redirect_stderr
import io

class ComprehensiveLogger:
    """Comprehensive logging system for experiment tracking."""
    
    def __init__(self, base_dir="results", model_type="heterognn"):
        self.base_dir = Path(base_dir)
        self.model_type = model_type.lower()
        
        # Create model-specific directory
        self.model_dir = self.base_dir / self.model_type
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped experiment folder within model directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.model_dir / timestamp
        self.experiment_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.log_file = self.experiment_dir / "output_log.txt"
        self.setup_logging()
        
        # Capture stdout/stderr
        self.stdout_capture = io.StringIO()
        self.stderr_capture = io.StringIO()
        
    def setup_logging(self):
        """Setup logging to both console and file."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def log_hyperparameters(self, **kwargs):
        """Save hyperparameters to file."""
        hyperparams_file = self.experiment_dir / "hyperparameters.txt"
        
        with open(hyperparams_file, 'w') as f:
            f.write("=== EXPERIMENT HYPERPARAMETERS ===\n\n")
            f.write(f"Model Type: {self.model_type}\n\n")
            for key, value in kwargs.items():
                f.write(f"{key}: {value}\n")
                
        self.logger.info(f"Hyperparameters saved to {hyperparams_file}")
        
    def log_data_split(self, train_set, val_set, test_set, raw_sg, train_indices, val_indices, test_indices):
        """Log data split information with ticker and date."""
        split_info = {
            "train": [],
            "val": [],
            "test": []
        }
        
        # Helper function to extract ticker and date
        def extract_info(indices, split_name):
            info_list = []
            for idx in indices:
                if idx < len(raw_sg):
                    sg = raw_sg[idx]
                    ticker = getattr(sg, "primary_ticker", "unknown")
                    date = getattr(sg, "reported_date", "unknown")
                    info_list.append(f"{ticker},{date}")
                else:
                    info_list.append("unknown,unknown")
            return info_list
        
        # Extract info for each split
        split_info["train"] = extract_info(train_indices, "train")
        split_info["val"] = extract_info(val_indices, "val")
        split_info["test"] = extract_info(test_indices, "test")
        
        # Save to file
        split_file = self.experiment_dir / "data_split.json"
        with open(split_file, 'w') as f:
            json.dump(split_info, f, indent=2)
            
        self.logger.info(f"Data split info saved to {split_file}")
        
        # Also log summary
        self.logger.info(f"Data split: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
        
    def save_model(self, model, filename="model.pt"):
        """Save model to experiment directory."""
        model_path = self.experiment_dir / filename
        torch.save(model.state_dict(), model_path)
        self.logger.info(f"Model saved to {model_path}")
        
    def save_test_predictions_csv(self, test_metrics, test_indices, raw_sg):
        """Save test predictions to CSV with ticker and date information."""
        if "raw_predictions" not in test_metrics:
            self.logger.warning("No raw predictions found in test metrics")
            return
            
        # Create CSV data
        csv_data = []
        for i, pred in enumerate(test_metrics["raw_predictions"]):
            if i < len(test_indices) and test_indices[i] < len(raw_sg):
                sg = raw_sg[test_indices[i]]
                ticker = getattr(sg, "primary_ticker", "unknown")
                date = getattr(sg, "reported_date", "unknown")
                
                csv_data.append({
                    "Ticker": ticker,
                    "Reported_Date": date,
                    "Actual_Label": pred["actual_label"],
                    "Predicted_Label": pred["predicted_label"],
                    "Probability": pred["probability"]
                })
            else:
                csv_data.append({
                    "Ticker": "unknown",
                    "Reported_Date": "unknown",
                    "Actual_Label": pred["actual_label"],
                    "Predicted_Label": pred["predicted_label"],
                    "Probability": pred["probability"]
                })
        
        # Save to CSV
        import pandas as pd
        df = pd.DataFrame(csv_data)
        csv_file = self.experiment_dir / "test_predictions.csv"
        df.to_csv(csv_file, index=False)
        
        self.logger.info(f"Test predictions saved to {csv_file}")
        self.logger.info(f"CSV contains {len(csv_data)} predictions")
        
    def save_results(self, test_metrics, history, best_epoch):
        """Save results to JSON file."""
        results = {
            "model_type": self.model_type,
            "test_metrics": test_metrics,
            "training_history": history,
            "best_epoch": best_epoch,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        results_file = self.experiment_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        self.logger.info(f"Results saved to {results_file}")
        
    def get_experiment_dir(self):
        """Get the experiment directory path."""
        return str(self.experiment_dir)
        
    def log_experiment_start(self, **kwargs):
        """Log experiment start with hyperparameters."""
        self.logger.info("=" * 60)
        self.logger.info("EXPERIMENT START")
        self.logger.info("=" * 60)
        self.logger.info(f"Model Type: {self.model_type}")
        self.logger.info(f"Experiment directory: {self.experiment_dir}")
        self.log_hyperparameters(**kwargs)
        
    def log_experiment_end(self, test_metrics, history, best_epoch):
        """Log experiment end and save results."""
        self.logger.info("=" * 60)
        self.logger.info("EXPERIMENT END")
        self.logger.info("=" * 60)
        self.logger.info(f"Model Type: {self.model_type}")
        self.logger.info(f"Best epoch: {best_epoch}")
        self.logger.info(f"Final test metrics: {test_metrics}")
        self.save_results(test_metrics, history, best_epoch)

# Set environment variable to disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress Hugging Face warnings and errors
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="transformers")

# --- your loader + model ---
from SubGraphDataLoader import SubGraphDataLoader
from HeteroGNN import HeteroGNN  # ensure this matches your file/module name
from HeteroGNN2 import HeteroGNN2  # add HeteroGNN2 as an option
from HeteroGNN3 import HeteroGNN3  # add HeteroGNN3 as an option

# Import config
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# Global cached sentence transformer to avoid repeated downloads
_CACHED_TRANSFORMER = None

def get_cached_transformer():
    """Get or create a cached sentence transformer to avoid repeated downloads."""
    global _CACHED_TRANSFORMER
    if _CACHED_TRANSFORMER is None:
        print("[transformer] Loading sentence transformer model from local cache...")
        try:
            # Use local cache directory
            cache_dir = os.path.join(os.path.dirname(__file__), "model_cache")
            
            # Suppress all warnings during model loading
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _CACHED_TRANSFORMER = SentenceTransformer(
                    "all-mpnet-base-v2",  # Changed to better encoder
                    cache_folder=cache_dir
                )
            print("[transformer] Model loaded successfully from local cache")
        except Exception as e:
            print(f"[transformer] Error loading model: {e}")
            raise
    return _CACHED_TRANSFORMER

# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def attach_y_and_meta(dataset: List[HeteroData], subgraphs) -> None:
    """
    Attach graph-level label into .y and stash primary ticker for grouped split.
    """
    for g, sg in zip(dataset, subgraphs):
        g.y = g["graph_label"].float().view(-1)
        # Keep primary ticker as a simple python string attribute for grouping
        g.meta_primary_ticker = getattr(sg, "primary_ticker", None)

def encode_all_to_heterodata(loader: SubGraphDataLoader, batch_size: int = 20) -> tuple[list[HeteroData], list[HeteroData], list, list]:
    """
    Encodes training and testing SubGraphs to HeteroData separately and returns 
    (training_graphs, testing_graphs, training_subgraphs, testing_subgraphs).
    Uses batch processing to avoid rate limiting.
    """
    training_graphs: List[HeteroData] = []
    testing_graphs: List[HeteroData] = []
    
    # Get cached transformer once
    transformer = get_cached_transformer()
    
    # Process training items
    training_items = loader.get_training_items()
    training_subgraphs = training_items.copy()  # Keep original subgraphs
    total_training = len(training_items)
    
    print(f"[encode] Processing {total_training} training subgraphs in batches of {batch_size}")
    
    for i in range(0, total_training, batch_size):
        batch_end = min(i + batch_size, total_training)
        batch_items = training_items[i:batch_end]
        
        print(f"[encode] Processing training batch {i//batch_size + 1}/{(total_training + batch_size - 1)//batch_size} "
              f"(subgraphs {i+1}-{batch_end}/{total_training})")
        
        batch_success = 0
        for j, sg in enumerate(batch_items):
            max_retries = 3
            retry_delay = 2.0  # Start with 2 seconds
            
            for attempt in range(max_retries):
                try:
                    # Progress indicator within batch
                    if j % 5 == 0:
                        print(f"  [encode] Processed {j+1}/{len(batch_items)} in current training batch")
                    
                    # Encode SubGraph to HeteroData using cached transformer
                    data = sg.to_pyg_data(text_encoder=transformer)
                    training_graphs.append(data)
                    batch_success += 1
                    
                    # Small delay to avoid overwhelming the system
                    if j % 3 == 0:  # Every 3 subgraphs
                        time.sleep(0.05)  # 50ms delay
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    if "429" in str(e) or "rate limit" in str(e).lower():
                        if attempt < max_retries - 1:
                            print(f"[encode] Rate limit hit, retrying in {retry_delay:.1f}s... (attempt {attempt + 1}/{max_retries})")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                            continue
                        else:
                            print(f"[encode] Rate limit exceeded after {max_retries} attempts, skipping training subgraph {i+j+1}")
                            break
                    else:
                        print(f"[encode] Error processing training subgraph {i+j+1} ({sg.primary_ticker}): {e}")
                        break
        
        # Longer delay between batches to avoid rate limiting
        if batch_end < total_training:
            delay = 3.0 + (batch_success / len(batch_items)) * 2.0  # 3-5 seconds based on success rate
            print(f"[encode] Training batch completed ({batch_success}/{len(batch_items)} successful). Waiting {delay:.1f}s before next batch...")
            time.sleep(delay)
    
    print(f"[encode] Successfully processed {len(training_graphs)}/{total_training} training subgraphs")
    
    # Process testing items
    testing_items = loader.get_testing_items()
    testing_subgraphs = testing_items.copy()  # Keep original subgraphs
    total_testing = len(testing_items)
    
    print(f"[encode] Processing {total_testing} testing subgraphs in batches of {batch_size}")
    
    for i in range(0, total_testing, batch_size):
        batch_end = min(i + batch_size, total_testing)
        batch_items = testing_items[i:batch_end]
        
        print(f"[encode] Processing testing batch {i//batch_size + 1}/{(total_testing + batch_size - 1)//batch_size} "
              f"(subgraphs {i+1}-{batch_end}/{total_testing})")
        
        batch_success = 0
        for j, sg in enumerate(batch_items):
            max_retries = 3
            retry_delay = 2.0  # Start with 2 seconds
            
            for attempt in range(max_retries):
                try:
                    # Progress indicator within batch
                    if j % 5 == 0:
                        print(f"  [encode] Processed {j+1}/{len(batch_items)} in current testing batch")
                    
                    # Encode SubGraph to HeteroData using cached transformer
                    data = sg.to_pyg_data(text_encoder=transformer)
                    testing_graphs.append(data)
                    batch_success += 1
                    
                    # Small delay to avoid overwhelming the system
                    if j % 3 == 0:  # Every 3 subgraphs
                        time.sleep(0.05)  # 50ms delay
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    if "429" in str(e) or "rate limit" in str(e).lower():
                        if attempt < max_retries - 1:
                            print(f"[encode] Rate limit hit, retrying in {retry_delay:.1f}s... (attempt {attempt + 1}/{max_retries})")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                            continue
                        else:
                            print(f"[encode] Rate limit exceeded after {max_retries} attempts, skipping testing subgraph {i+j+1}")
                            break
                    else:
                        print(f"[encode] Error processing testing subgraph {i+j+1} ({sg.primary_ticker}): {e}")
                        break
        
        # Longer delay between batches to avoid rate limiting
        if batch_end < total_testing:
            delay = 3.0 + (batch_success / len(batch_items)) * 2.0  # 3-5 seconds based on success rate
            print(f"[encode] Testing batch completed ({batch_success}/{len(batch_items)} successful). Waiting {delay:.1f}s before next batch...")
            time.sleep(delay)
    
    print(f"[encode] Successfully processed {len(testing_graphs)}/{total_testing} testing subgraphs")
    print(f"[encode] Total: {len(training_graphs)} training + {len(testing_graphs)} testing = {len(training_graphs) + len(testing_graphs)} total graphs")
    
    return training_graphs, testing_graphs, training_subgraphs, testing_subgraphs

def split_list(dataset: List[HeteroData], train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Simple randomized split for a list.
    Prints class distribution for each split.
    Returns (train_set, val_set, test_set, train_indices, val_indices, test_indices)
    """
    g = torch.Generator().manual_seed(42)
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
    val_set   = [dataset[i] for i in idx_val]
    test_set  = [dataset[i] for i in idx_test]

    # Print class distributions
    class_stats(train_set, "train")
    class_stats(val_set, "val")
    class_stats(test_set, "test")

    return train_set, val_set, test_set, idx_train, idx_val, idx_test


def make_loader(ds: List[HeteroData], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Set to 0 to avoid tokenizer forking issues
        pin_memory=torch.cuda.is_available()
    )

def compute_pos_weight(dataset: List[HeteroData]) -> Tensor:
    """
    Very aggressive pos_weight computation for severe class imbalance.
    Uses 3 * (neg/pos) for extremely aggressive weighting.
    """
    y = torch.cat([g.y for g in dataset]).float()
    pos = (y > 0.5).sum().item()
    neg = len(y) - pos
    
    if pos == 0:
        return torch.tensor(1.0, dtype=torch.float)
    
    # MODIFY SCALE FACTOR TO ADJUST WEIGHTING AGRESSION
    ratio = 1.0 * (neg / pos)
    
    print(f"[class_balance] Positive: {pos}, Negative: {neg}, Ratio: {neg/pos:.2f}, Aggressive ratio: {ratio:.2f}")
    
    return torch.tensor(ratio, dtype=torch.float)

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Reduces the relative loss for well-classified examples and focuses on hard examples.
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

@torch.no_grad()
def init_lazy_params(model: nn.Module, sample_batch: HeteroData, device: torch.device) -> None:
    """
    Trigger lazy layer initialization (e.g., GCNConv with in_channels=-1) before creating the optimizer.
    """
    model.to(device)
    _ = model(sample_batch.to(device))

# ----------------------------
# Ticker feature scaler (train-only)
# ----------------------------
def winsorize(X: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    return torch.minimum(torch.maximum(X, low), high)

# ---------- TICKER FEATURE SCALER (train-only, robust, no-op safe) ----------

def fit_ticker_scaler(train_graphs: List[HeteroData], pct_low=1.0, pct_high=99.0) -> dict:
    """
    Fit a robust scaler on company features from training graphs only:
      - winsorize to [p1, p99] per feature
      - z-score from the winsorized stats
    Returns an identity scaler if no company features are found (no crash).
    """
    rows = []
    for g in train_graphs:
        # Check if 'company' is a node type in the HeteroData
        node_types = g.metadata()[0] if hasattr(g, 'metadata') else []
        if 'company' in node_types and hasattr(g['company'], 'x'):
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
    std  = Xw.std(dim=0, unbiased=False)
    std[std == 0] = 1.0

    print(f"[scaler] Fit on {X.shape[0]} company rows, D={X.shape[1]} "
          f"(winsorize p{pct_low}/{pct_high})")
    return {"low": lo.float(), "high": hi.float(), "mean": mean.float(), "std": std.float(), "identity": False}

def apply_ticker_scaler_to_graphs(graphs: List[HeteroData], scaler: dict) -> None:
    """
    In-place transform of company features on graphs: winsorize then z-score.
    No-op if scaler['identity'] is True.
    """
    if scaler.get("identity", False):
        return
    low, high, mean, std = scaler["low"], scaler["high"], scaler["mean"], scaler["std"]
    for g in graphs:
        # Check if 'company' is a node type in the HeteroData
        node_types = g.metadata()[0] if hasattr(g, 'metadata') else []
        if 'company' in node_types and hasattr(g['company'], 'x'):
            X = g['company'].x
            if not isinstance(X, torch.Tensor) or X.numel() == 0:
                continue
            low_d, high_d = low.to(X.device), high.to(X.device)
            mean_d, std_d = mean.to(X.device), std.to(X.device)
            X = torch.minimum(torch.maximum(X, low_d), high_d)
            X = (X - mean_d) / std_d
            g['company'].x = X


# ----------------------------
# Train / Eval loops
# ----------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float | None = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        logits: Tensor = model(batch)              # [B]
        y: Tensor = batch.y                        # [B]
        loss: Tensor = criterion(logits, y)

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
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    report_loss: bool = True,
    detailed_analysis: bool = False,
    threshold: float = 0.5,  # Prediction threshold
    criterion: nn.Module = None,  # Use the same criterion as training
    return_raw_predictions: bool = False,  # New parameter to return raw predictions
) -> Dict[str, float]:
    """
    Returns avg loss (if report_loss), accuracy, AUC, and detailed prediction analysis.
    """
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    # Use provided criterion or fallback to standard BCEWithLogitsLoss
    if criterion is None:
        eval_criterion = nn.BCEWithLogitsLoss(reduction="sum")
    else:
        # Use the same criterion as training for consistent loss calculation
        eval_criterion = criterion

    all_probs = []
    all_y = []
    all_preds = []
    
    # For raw predictions
    raw_predictions = []

    for batch in loader:
        batch = batch.to(device)
        logits: Tensor = model(batch)    # [B]
        y: Tensor = batch.y              # [B]

        if report_loss:
            loss = eval_criterion(logits, y)
            total_loss += loss.item()
            
            # Debug: print some sample values
            if total == 0:  # Only print for first batch
                # Get probabilities for debugging
                debug_probs = torch.sigmoid(logits)
                
                print(f"[DEBUG] Sample logits: {logits[:5]}")
                print(f"[DEBUG] Sample targets: {y[:5]}")
                print(f"[DEBUG] Sample probs: {debug_probs[:5]}")
                print(f"[DEBUG] Batch loss: {loss.item():.6f}")
                print(f"[DEBUG] Loss criterion type: {type(eval_criterion).__name__}")

        # Get probabilities - handle both raw logits and pre-sigmoided outputs
        if isinstance(criterion, nn.BCEWithLogitsLoss) and hasattr(criterion, 'pos_weight') and criterion.pos_weight != 1.0:
            # Weighted BCE - still need to apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)
        else:
            # Standard case: apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)
            
        preds = (probs >= threshold).long()  # Use custom threshold
        correct += (preds == y.long()).sum().item()
        total += y.numel()

        all_probs.append(probs.detach().cpu())
        all_y.append(y.detach().cpu())
        all_preds.append(preds.detach().cpu())
        
        # Store raw predictions for CSV export
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

    # Optional AUC
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

    # Detailed prediction analysis
    if detailed_analysis:
        y_true = torch.cat(all_y).numpy()
        y_pred = torch.cat(all_preds).numpy()
        y_prob = torch.cat(all_probs).numpy()
        
        # Confusion matrix components
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        # Additional metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Prediction distribution
        pred_0_count = np.sum(y_pred == 0)
        pred_1_count = np.sum(y_pred == 1)
        true_0_count = np.sum(y_true == 0)
        true_1_count = np.sum(y_true == 1)
        
        # Probability distribution
        prob_mean = np.mean(y_prob)
        prob_std = np.std(y_prob)
        prob_min = np.min(y_prob)
        prob_max = np.max(y_prob)
        
        detailed_metrics = {
            # Confusion matrix
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            
            # Derived metrics
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            
            # Prediction counts
            "pred_0": int(pred_0_count),
            "pred_1": int(pred_1_count),
            "true_0": int(true_0_count),
            "true_1": int(true_1_count),
            
            # Probability stats
            "prob_mean": float(prob_mean),
            "prob_std": float(prob_std),
            "prob_min": float(prob_min),
            "prob_max": float(prob_max),
        }
        
        metrics.update(detailed_metrics)

    # Add raw predictions if requested
    if return_raw_predictions:
        metrics["raw_predictions"] = raw_predictions

    return metrics

def find_optimal_threshold(model, val_loader, device, criterion=None):
    """
    Find the optimal threshold that targets a reasonable number of positive predictions.
    Aims for 10-20 positive predictions with good precision (>50%).
    """
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
    
    # Try different thresholds and find the one that gives reasonable positive predictions
    best_f1 = 0
    best_threshold = 0.7  # Start with moderate threshold
    best_precision = 0
    best_recall = 0
    target_positives = 15  # Target around 15 positive predictions
    
    for threshold in np.arange(0.6, 0.95, 0.02):  # Try moderate to high thresholds
        preds = (all_probs >= threshold).astype(int)
        
        # Calculate metrics
        tp = np.sum((all_y == 1) & (preds == 1))
        fp = np.sum((all_y == 0) & (preds == 1))
        fn = np.sum((all_y == 1) & (preds == 0))
        total_positives = tp + fp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Prioritize thresholds that give reasonable number of positive predictions
        # and have good precision
        if (total_positives >= 5 and  # At least 5 positive predictions
            total_positives <= 25 and  # No more than 25 positive predictions
            precision >= 0.4 and  # Require reasonable precision
            f1 > best_f1):
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
    
    print(f"[threshold] Balanced threshold: {best_threshold:.3f} (F1: {best_f1:.3f}, Precision: {best_precision:.3f}, Recall: {best_recall:.3f})")
    return best_threshold

# ----------------------------
# Main runner
# ----------------------------
def run_training(
    # --- data / encoding ---
    n_facts: int = 10,
    limit: int | None = None,
    use_cache: bool = True,  # New parameter to enable/disable caching
    # --- model ---
    model_type: str = "heterognn",  # "heterognn" or "heterognn2"
    hidden_channels: int = 64,
    num_layers: int = 2,
    feature_dropout: float = 0.3,
    edge_dropout: float = 0.0,
    final_dropout: float = 0.1,
    readout: str = "concat",          # 'fact' | 'company' | 'concat' | 'gated'
    time_dim: int = 8,  # For HeteroGNN2 temporal encoding
    # --- training ---
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    grad_clip: float | None = 1.0,
    ckpt_path: str = "best_model.pt",
    loss_type: str = "bce",  # Use standard BCE loss for reliable training
    early_stopping: bool = True,  # Enable/disable early stopping
    patience: int = 10,  # Early stopping patience
    lr_scheduler: str = "none",  # 'none', 'step', 'cosine', 'plateau'
    lr_step_size: int = 10,  # For step scheduler
    lr_gamma: float = 0.5,  # For step scheduler
    time_aware_split: bool = False,  # DEPRECATED: Fixed data splits are now used instead
    optimizer_type: str = "adam", # "adam", "adamw", "sgd", "rmsprop"
    # --- run scraping ---
    enable_run_scraping: bool = True,  # Enable/disable run scraping feature
    min_epochs_after_patience: int = 10,  # Minimum epochs that should occur after patience threshold
) -> Tuple[nn.Module, Dict[str, float], Dict[str, list]]:
    
    # Initialize comprehensive logger
    logger = ComprehensiveLogger(model_type=model_type)
    
    # Log experiment start with all hyperparameters
    logger.log_experiment_start(
        n_facts=n_facts,
        limit=limit,
        use_cache=use_cache,
        model_type=model_type,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        feature_dropout=feature_dropout,
        edge_dropout=edge_dropout,
        final_dropout=final_dropout,
        readout=readout,
        time_dim=time_dim,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        grad_clip=grad_clip,
        loss_type=loss_type,
        early_stopping=early_stopping,
        patience=patience,
        lr_scheduler=lr_scheduler,
        lr_step_size=lr_step_size,
        lr_gamma=lr_gamma,
        time_aware_split=time_aware_split,
        optimizer_type=optimizer_type,
        enable_run_scraping=enable_run_scraping,
        min_epochs_after_patience=min_epochs_after_patience
    )
    
    set_seed(seed)

    # 1) Load & encode graphs -------------------------------------------------
    print(f"[data] Loading subgraphs with min_facts={n_facts}, limit={limit}")
    
    # Try to load from cache first
    training_graphs = None
    testing_graphs = None
    training_raw_sg = None
    testing_raw_sg = None
    
    if use_cache:
        try:
            from cache_dataset import get_cache_dir, get_cache_filename, safe_read_pickle, validate_cache_data
            import time
            
            print("[data] Attempting to load from separate training/testing cache...")
            
            # Get cache directory and filenames for training and testing
            cache_dir = get_cache_dir()
            base_cache_filename = get_cache_filename(n_facts, limit)
            
            # Create separate filenames for training and testing
            training_cache_filename = f"training_{base_cache_filename}"
            testing_cache_filename = f"testing_{base_cache_filename}"
            
            training_cache_path = os.path.join(cache_dir, training_cache_filename)
            testing_cache_path = os.path.join(cache_dir, testing_cache_filename)
            
            # Check if both cache files exist
            if os.path.exists(training_cache_path) and os.path.exists(testing_cache_path):
                try:
                    # Load training cache
                    training_cache_data = safe_read_pickle(training_cache_path)
                    if training_cache_data and validate_cache_data(training_cache_data):
                        training_graphs = training_cache_data['graphs']
                        training_raw_sg = training_cache_data['raw_sg']
                        print(f"[data] ✅ Loaded {len(training_graphs)} training graphs from cache")
                    else:
                        print("[data] ❌ Training cache validation failed")
                        
                    # Load testing cache
                    testing_cache_data = safe_read_pickle(testing_cache_path)
                    if testing_cache_data and validate_cache_data(testing_cache_data):
                        testing_graphs = testing_cache_data['graphs']
                        testing_raw_sg = testing_cache_data['raw_sg']
                        print(f"[data] ✅ Loaded {len(testing_graphs)} testing graphs from cache")
                    else:
                        print("[data] ❌ Testing cache validation failed")
                        
                    # Check if both loads were successful
                    if training_graphs is not None and testing_graphs is not None:
                        print(f"[data] ✅ Successfully loaded {len(training_graphs)} training + {len(testing_graphs)} testing graphs from cache")
                    else:
                        print("[data] ❌ Cache loading failed, will process from scratch")
                        training_graphs = None
                        testing_graphs = None
                        training_raw_sg = None
                        testing_raw_sg = None
                        
                except Exception as e:
                    print(f"[data] Error loading cache: {e}, processing from scratch")
                    training_graphs = None
                    testing_graphs = None
                    training_raw_sg = None
                    testing_raw_sg = None
            else:
                print(f"[data] Cache files not found:")
                print(f"   Training cache: {training_cache_path}")
                print(f"   Testing cache: {testing_cache_path}")
                print("[data] Will process from scratch")
                
        except ImportError:
            print("[data] Cache module not found, processing from scratch")
        except Exception as e:
            print(f"[data] Error loading cache: {e}, processing from scratch")
    
    # If cache failed or disabled, process from scratch
    if training_graphs is None or testing_graphs is None:
        print("[data] Processing dataset from scratch...")
        # Use absolute path for subgraphs file
        import os
        subgraphs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), config.SUBGRAPHS_JSONL)
        
        # Create loader with fixed splits
        loader = SubGraphDataLoader(
            min_facts=n_facts, 
            limit=limit, 
            jsonl_path=subgraphs_path,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            split_data=True
        )
        
        # Encode training and testing data separately
        training_graphs, testing_graphs, training_raw_sg, testing_raw_sg = encode_all_to_heterodata(loader)
        
        # Optionally save to separate cache files for future use
        if use_cache:
            try:
                from cache_dataset import get_cache_dir, get_cache_filename, atomic_write_pickle
                import time
                
                print("[data] Saving processed dataset to separate training/testing cache files...")
                
                # Get cache directory and filenames
                cache_dir = get_cache_dir()
                base_cache_filename = get_cache_filename(n_facts, limit)
                
                # Create separate filenames for training and testing
                training_cache_filename = f"training_{base_cache_filename}"
                testing_cache_filename = f"testing_{base_cache_filename}"
                
                training_cache_path = os.path.join(cache_dir, training_cache_filename)
                testing_cache_path = os.path.join(cache_dir, testing_cache_filename)
                
                # Create cache directory if it doesn't exist
                os.makedirs(cache_dir, exist_ok=True)
                
                # Prepare training cache data
                training_cache_data = {
                    'graphs': training_graphs,
                    'raw_sg': training_raw_sg,
                    'n_facts': n_facts,
                    'limit': limit,
                    'timestamp': time.time(),
                    'graph_count': len(training_graphs),
                    'split_type': 'training'
                }
                
                # Prepare testing cache data
                testing_cache_data = {
                    'graphs': testing_graphs,
                    'raw_sg': testing_raw_sg,
                    'n_facts': n_facts,
                    'limit': limit,
                    'timestamp': time.time(),
                    'graph_count': len(testing_graphs),
                    'split_type': 'testing'
                }
                
                # Save the separate cache files
                atomic_write_pickle(training_cache_data, training_cache_path)
                atomic_write_pickle(testing_cache_data, testing_cache_path)
                
                print(f"[data] ✅ Dataset cached successfully to separate files:")
                print(f"   Training: {training_cache_path}")
                print(f"   Testing: {testing_cache_path}")
                
            except Exception as e:
                print(f"[data] Warning: Failed to save cache: {e}")
    
    # Attach labels and metadata for grouping (always needed, whether from cache or scratch)
    print("[data] Attaching labels and metadata to graphs")
    attach_y_and_meta(training_graphs, training_raw_sg)
    attach_y_and_meta(testing_graphs, testing_raw_sg)

    if len(training_graphs) < 3:
        raise RuntimeError(f"Need at least 3 training graphs to split; got {len(training_graphs)}")

    print(f"[data] Loaded {len(training_graphs)} training graphs and {len(testing_graphs)} testing graphs")
    if len(training_graphs) != len(training_raw_sg) or len(testing_graphs) != len(testing_raw_sg):
        raise RuntimeError("Mismatch between encoded graphs and raw subgraphs.")

    # 2) Split training data into train and validation ---------------------------
    print("[data] Splitting training data into train and validation sets...")
    
    # Split training graphs into train and validation
    train_set, val_set, _, train_indices, val_indices, _ = split_list(
        dataset=training_graphs,
        train_ratio=train_ratio/(train_ratio + val_ratio),  # Adjust ratio for train/val split
        val_ratio=val_ratio/(train_ratio + val_ratio),      # Adjust ratio for train/val split
        seed=seed
    )
    
    # Test set is already separate
    test_set = testing_graphs
    
    # Combine raw subgraphs for logging
    all_raw_sg = training_raw_sg + testing_raw_sg
    
    # Create indices for logging (train and val indices are relative to training_graphs)
    # Test indices start after training_graphs
    test_indices = list(range(len(training_graphs), len(training_graphs) + len(testing_graphs)))
    
    print(f"[data] Final split: {len(train_set)} train, {len(val_set)} val, {len(test_set)} test")
    
    # Log data split information
    logger.log_data_split(train_set, val_set, test_set, all_raw_sg, train_indices, val_indices, test_indices)

    # 3) Fit train-only scaler on company features, then apply to all splits --
    # After you create train_set, val_set, test_set
    print(f"[data] company-feature presence: "
        f"train={sum((('company' in g.metadata()[0] if hasattr(g, 'metadata') else False) and hasattr(g['company'],'x') and g['company'].x.numel()>0) for g in train_set)}, "
        f"val={sum((('company' in g.metadata()[0] if hasattr(g, 'metadata') else False) and hasattr(g['company'],'x') and g['company'].x.numel()>0) for g in val_set)}, "
        f"test={sum((('company' in g.metadata()[0] if hasattr(g, 'metadata') else False) and hasattr(g['company'],'x') and g['company'].x.numel()>0) for g in test_set)}")

    scaler = fit_ticker_scaler(train_set)   # never crashes; returns identity if empty
    apply_ticker_scaler_to_graphs(train_set, scaler)
    apply_ticker_scaler_to_graphs(val_set,   scaler)
    apply_ticker_scaler_to_graphs(test_set,  scaler)

    # 4) DataLoaders ----------------------------------------------------------
    train_loader = make_loader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = make_loader(val_set,   batch_size=2 * batch_size, shuffle=False)
    test_loader  = make_loader(test_set,  batch_size=2 * batch_size, shuffle=False)

    # 5) Model / Loss / Optimizer --------------------------------------------
    metadata = train_set[0].metadata()
    
    # Choose model type
    if model_type.lower() == "heterognn":
        model = HeteroGNN(
            metadata=metadata,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            feature_dropout=feature_dropout,
            edge_dropout=edge_dropout,
            final_dropout=final_dropout,
            readout=readout,
        )
        print(f"[model] Using HeteroGNN with {hidden_channels} hidden channels, {num_layers} layers")
    elif model_type.lower() == "heterognn2":
        model = HeteroGNN2(
            metadata=metadata,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            feature_dropout=feature_dropout,
            edge_dropout=edge_dropout,
            final_dropout=final_dropout,
            readout=readout,
            time_dim=time_dim,
        )
        print(f"[model] Using HeteroGNN2 with {hidden_channels} hidden channels, {num_layers} layers, time_dim={time_dim}")
    elif model_type.lower() == "heterognn3":
        model = HeteroGNN3(
            metadata=metadata,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            feature_dropout=feature_dropout,
            edge_dropout=edge_dropout,
            final_dropout=final_dropout,
            readout=readout,
        )
        print(f"[model] Using HeteroGNN3 with {hidden_channels} hidden channels, {num_layers} layers (NO TEMPORAL ENCODING)")
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'heterognn', 'heterognn2', or 'heterognn3'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] Using {device}")

    # Initialize lazy params BEFORE creating optimizer
    sample_batch = next(iter(train_loader))
    init_lazy_params(model, sample_batch, device)
    
    # Debug: Check initial model outputs
    model.eval()
    with torch.no_grad():
        sample_outputs = model(sample_batch.to(device))
        sample_probs = torch.sigmoid(sample_outputs)
        print(f"[DEBUG] Initial model outputs (first 5): {sample_outputs[:5]}")
        print(f"[DEBUG] Initial probabilities (first 5): {sample_probs[:5]}")
        print(f"[DEBUG] Initial targets (first 5): {sample_batch.y[:5]}")
        print(f"[DEBUG] Model output range: [{sample_outputs.min():.3f}, {sample_outputs.max():.3f}]")
        print(f"[DEBUG] Probability range: [{sample_probs.min():.3f}, {sample_probs.max():.3f}]")
    
    model.train()  # Set back to training mode

    pos_weight = compute_pos_weight(train_set).to(device)
    
    # Choose loss function based on loss_type parameter
    if loss_type == "bce":
        criterion = nn.BCEWithLogitsLoss()
    elif loss_type == "weighted_bce":
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_type == "bce_label_smooth":
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, label_smoothing=0.1)
    elif loss_type == "focal":
        # Use conservative Focal Loss parameters to prioritize precision
        criterion = FocalLoss(alpha=0.25, gamma=3.0)  # Lower alpha, higher gamma for harder examples
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    
    print(f"[loss] Using {loss_type} loss function with pos_weight={pos_weight:.3f}")

    # Choose optimizer
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_type == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type}")
    
    print(f"[optimizer] Using {optimizer_type} optimizer")

    # Setup learning rate scheduler
    scheduler = None
    if lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
        print(f"[scheduler] Using StepLR with step_size={lr_step_size}, gamma={lr_gamma}")
    elif lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        print(f"[scheduler] Using CosineAnnealingLR with T_max={epochs}")
    elif lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_gamma, patience=5)
        print(f"[scheduler] Using ReduceLROnPlateau with factor={lr_gamma}, patience=5")
    elif lr_scheduler == "one_cycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr*10, epochs=epochs, steps_per_epoch=len(train_loader))
        print(f"[scheduler] Using OneCycleLR with max_lr={lr*10}")
    elif lr_scheduler == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
        print(f"[scheduler] Using ExponentialLR with gamma={lr_gamma}")
    elif lr_scheduler == "none":
        print("[scheduler] No learning rate scheduler")
    else:
        raise ValueError(f"Unknown lr_scheduler: {lr_scheduler}")

    # ---- training history ----
    history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_acc": [], "val_auc": []}

    # 6) Train loop with best checkpointing ----------------------------------
    best_val = math.inf
    best_state: Dict[str, Tensor] | None = None
    patience_counter = 0
    min_epochs_before_stopping = 10  # Don't stop before epoch 10

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip=grad_clip,
        )
        val_metrics = evaluate(model, val_loader, device, detailed_analysis=True, threshold=0.5, criterion=criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics.get("loss", float("nan")))
        history["val_acc"].append(val_metrics["acc"])
        history["val_auc"].append(val_metrics.get("auc", float("nan")))

        current_val_loss = val_metrics.get("loss", float("inf"))
        
        # Check if this is the best validation loss so far (with small tolerance)
        improvement_threshold = 1e-4  # Very small improvement threshold
        if current_val_loss < (best_val - improvement_threshold):
            best_epoch = epoch
            best_val = current_val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        # Enhanced printing with detailed metrics
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={current_val_loss:.4f} | "
            f"val_acc={val_metrics['acc']:.3f} | "
            f"val_auc={val_metrics.get('auc', float('nan')):.3f} | "
            f"patience={patience_counter}/{patience}"
        )
        
        # Print detailed prediction analysis
        if "tp" in val_metrics:
            print(f"  [pred] TP:{val_metrics['tp']:3d} TN:{val_metrics['tn']:3d} "
                  f"FP:{val_metrics['fp']:3d} FN:{val_metrics['fn']:3d} | "
                  f"P:{val_metrics['precision']:.3f} R:{val_metrics['recall']:.3f} "
                  f"F1:{val_metrics['f1']:.3f}")
            print(f"  [dist] Pred:0={val_metrics['pred_0']:3d} 1={val_metrics['pred_1']:3d} | "
                  f"True:0={val_metrics['true_0']:3d} 1={val_metrics['true_1']:3d} | "
                  f"Prob:μ={val_metrics['prob_mean']:.3f} σ={val_metrics['prob_std']:.3f}")
            
            # Additional debugging for class imbalance issues
            if epoch == 1:  # Only print this detailed info for first epoch
                print(f"  [DEBUG] Class imbalance analysis:")
                print(f"    - True class distribution: {val_metrics['true_0']} neg, {val_metrics['true_1']} pos")
                print(f"    - Predicted distribution: {val_metrics['pred_0']} neg, {val_metrics['pred_1']} pos")
                print(f"    - Probability stats: mean={val_metrics['prob_mean']:.3f}, std={val_metrics['prob_std']:.3f}")
                print(f"    - Probability range: [{val_metrics['prob_min']:.3f}, {val_metrics['prob_max']:.3f}]")
                if val_metrics['pred_1'] == 0:
                    print(f"    - ⚠️  WARNING: Model predicted ALL samples as negative!")
                    print(f"    - This suggests the model is biased toward the majority class")
                    print(f"    - Consider lowering threshold or increasing pos_weight")
        
        # Update learning rate scheduler
        if scheduler is not None:
            if lr_scheduler == "plateau":
                scheduler.step(current_val_loss)
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  [lr] Current learning rate: {current_lr:.2e}")
        
        # Early stopping - only after minimum epochs and with patience
        if early_stopping and epoch >= min_epochs_before_stopping and patience_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs (no improvement for {patience} epochs)")
            break

    # Check if training ended too early after patience threshold
    def should_scrape_run(final_epoch: int, patience: int, min_epochs_after_patience: int = 10) -> bool:
        """
        Check if training ended too early after the patience threshold.
        If training ends less than min_epochs_after_patience epochs after patience threshold,
        the run should be scraped.
        
        Args:
            final_epoch: The epoch where training ended
            patience: The patience value used for early stopping
            min_epochs_after_patience: Minimum epochs that should occur after patience threshold
            
        Returns:
            bool: True if run should be scraped, False otherwise
        """
        # Calculate when patience threshold would be hit
        patience_threshold_epoch = patience + min_epochs_before_stopping
        
        # Check if training ended too early after patience threshold
        if final_epoch <= (patience_threshold_epoch + min_epochs_after_patience):
            return True
        return False
    
    def scrape_run(logger):
        """
        Scrape the run by removing the experiment directory and its contents.
        """
        import shutil
        experiment_dir = Path(logger.get_experiment_dir())
        if experiment_dir.exists():
            try:
                shutil.rmtree(experiment_dir)
                print(f"[SCRAPE] Successfully removed experiment directory: {experiment_dir}")
                return True
            except Exception as e:
                print(f"[SCRAPE] Error removing experiment directory: {e}")
                return False
        else:
            print(f"[SCRAPE] Experiment directory not found: {experiment_dir}")
            return False
    
    # Check if run should be scraped
    final_epoch = epoch
    if enable_run_scraping and should_scrape_run(final_epoch, patience, min_epochs_after_patience):
        print(f"[SCRAPE] Training ended too early! Final epoch: {final_epoch}")
        print(f"[SCRAPE] Patience threshold would be at epoch: {patience + min_epochs_before_stopping}")
        print(f"[SCRAPE] Training should have continued for at least {patience + min_epochs_before_stopping + min_epochs_after_patience} epochs")
        print(f"[SCRAPE] Scraping run and removing all logs...")
        
        # Scrape the run
        scrape_success = scrape_run(logger)
        if scrape_success:
            print(f"[SCRAPE] Run successfully scraped. Exiting...")
            # Return None values to indicate the run was scraped
            return None, None, None
        else:
            print(f"[SCRAPE] Failed to scrape run. Continuing with results...")

    if best_state is not None:
        model.load_state_dict(best_state)
        if ckpt_path:
            # Save to experiment directory instead of current directory
            experiment_ckpt_path = Path(logger.get_experiment_dir()) / "model.pt"
            torch.save(best_state, experiment_ckpt_path)
            print(f"[ckpt] Best model saved to {experiment_ckpt_path}")
            logger.save_model(model, "model.pt")
        
        # Find which epoch had the best validation loss
        best_epoch = history["val_loss"].index(min(history["val_loss"])) + 1
        print(f"[best] Best model found at epoch {best_epoch} with val_loss={min(history['val_loss']):.4f}")

    # 7) Final test -----------------------------------------------------------
    # Find optimal threshold on validation set
    print("[test] Finding optimal threshold on validation set...")
    optimal_threshold = find_optimal_threshold(model, val_loader, device, criterion)
    
    # Use optimal threshold for final evaluation
    test_metrics = evaluate(model, test_loader, device, detailed_analysis=True, threshold=optimal_threshold, criterion=criterion, return_raw_predictions=True)
    print(
        f"TEST | loss={test_metrics.get('loss', float('nan')):.4f} | "
        f"acc={test_metrics['acc']:.3f} | "
        f"auc={test_metrics.get('auc', float('nan')):.3f} | "
        f"threshold={optimal_threshold:.3f} | "
        f"best_epoch={best_epoch}"
    )
    
    # Save test predictions to CSV
    logger.save_test_predictions_csv(test_metrics, test_indices, all_raw_sg)
    
    # Print detailed test metrics
    if "tp" in test_metrics:
        print(f"\n=== FINAL TEST SET ANALYSIS ===")
        print(f"Confusion Matrix:")
        print(f"                Predicted")
        print(f"               0     1")
        print(f"Actual 0    {test_metrics['tn']:4d}  {test_metrics['fp']:4d}")
        print(f"       1    {test_metrics['fn']:4d}  {test_metrics['tp']:4d}")
        print()
        
        print(f"Metrics:")
        print(f"  Accuracy:  {test_metrics['acc']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall:    {test_metrics['recall']:.4f}")
        print(f"  F1-Score:  {test_metrics['f1']:.4f}")
        print()
        
        print(f"Counts:")
        print(f"  True Positives (TP):  {test_metrics['tp']:4d}")
        print(f"  True Negatives (TN):  {test_metrics['tn']:4d}")
        print(f"  False Positives (FP): {test_metrics['fp']:4d}")
        print(f"  False Negatives (FN): {test_metrics['fn']:4d}")
        print()

    # Log experiment end with final results
    logger.log_experiment_end(test_metrics, history, best_epoch)

    return model, test_metrics, history


# ----------------------------
# Script entrypoint
# ----------------------------



if __name__ == "__main__":
    # ============================================================
    # HYPERPARAMETERS - Edit these to change model behavior
    # ============================================================
    
    # Choose which model to run
    MODEL_TYPE = "heterognn3"  # "heterognn", "heterognn2", or "heterognn3"
    
    # Data configuration
    N_FACTS = 35  # Minimum number of facts per subgraph
    LIMIT = None    # Limit number of subgraphs (None for all)
    USE_CACHE = True  # Enable caching for faster loading
    
    # Model architecture
    HIDDEN_CHANNELS = 128
    NUM_LAYERS = 4
    FEATURE_DROPOUT = 0.3
    EDGE_DROPOUT = 0.1
    FINAL_DROPOUT = 0.2
    READOUT = "company"  # 'fact' | 'company' | 'concat' | 'gated'
    TIME_DIM = 16  # For HeteroGNN2 temporal encoding
    
    # Training configuration
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 1e-4
    import random
    SEED = random.randint(0, 1000000)   
    # SEED = 157071
    GRAD_CLIP = 1.0
    LOSS_TYPE = "weighted_bce"  # "bce", "weighted_bce", "bce_label_smooth", "focal"
    EARLY_STOPPING = True
    PATIENCE = 20
    LR_SCHEDULER = "cosine"  # "none", "step", "cosine", "plateau", "one_cycle", "exponential"
    LR_STEP_SIZE = 10
    LR_GAMMA = 0.5
    TIME_AWARE_SPLIT = False
    OPTIMIZER_TYPE = "adam"  # "adam", "adamw", "sgd", "rmsprop"
    
    # Run scraping configuration
    ENABLE_RUN_SCRAPING = False  # Enable/disable the run scraping feature
    MIN_EPOCHS_AFTER_PATIENCE = 10  # Minimum epochs that should occur after patience threshold
    
    # Data splitting
    TRAIN_RATIO = 0.75
    VAL_RATIO = 0.1
    
    # ============================================================
    # RUN TRAINING
    # ============================================================
    
    print("=" * 60)
    if MODEL_TYPE == "heterognn2":
        print("RUNNING HETEROGNN2 (Temporal-Aware Model)")
    elif MODEL_TYPE == "heterognn3":
        print("RUNNING HETEROGNN3 (No Temporal Encoding)")
    else:
        print("RUNNING HETEROGNN (Original Model)")
    print("=" * 60)
    
    model, test_metrics, history = run_training(
        # Model configuration
        model_type=MODEL_TYPE,
        time_dim=TIME_DIM if MODEL_TYPE == "heterognn2" else None,
        
        # Data configuration
        n_facts=N_FACTS,
        limit=LIMIT,
        use_cache=USE_CACHE,
        
        # Model architecture
        hidden_channels=HIDDEN_CHANNELS,
        num_layers=NUM_LAYERS,
        feature_dropout=FEATURE_DROPOUT,
        edge_dropout=EDGE_DROPOUT,
        final_dropout=FINAL_DROPOUT,
        readout=READOUT,
        
        # Training configuration
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        seed=SEED,
        grad_clip=GRAD_CLIP,
        ckpt_path="best_model.pt",
        loss_type=LOSS_TYPE,
        early_stopping=EARLY_STOPPING,
        patience=PATIENCE,
        lr_scheduler=LR_SCHEDULER,
        lr_step_size=LR_STEP_SIZE,
        lr_gamma=LR_GAMMA,
        time_aware_split=TIME_AWARE_SPLIT,
        optimizer_type=OPTIMIZER_TYPE,
        
        # Run scraping configuration
        enable_run_scraping=ENABLE_RUN_SCRAPING,
        min_epochs_after_patience=MIN_EPOCHS_AFTER_PATIENCE,
        
        # Data splitting
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
    )

    # Check if run was scraped
    if model is None or test_metrics is None or history is None:
        print("\n" + "=" * 60)
        print("RUN WAS SCRAPED - Training ended too early after patience threshold")
        print("=" * 60)
        print("The experiment directory and all logs have been removed.")
        print("This run did not meet the minimum training duration requirements.")
        print("=" * 60)
        sys.exit(0)  # Use sys.exit() for more reliable exit

    # Additional safety check - if any of the returned values are None, exit
    if not all([model, test_metrics, history]):
        print("\n" + "=" * 60)
        print("ERROR: Incomplete results returned from training")
        print("=" * 60)
        sys.exit(1)

    model_name = "HeteroGNN2" if MODEL_TYPE == "heterognn2" else "HeteroGNN" if MODEL_TYPE == "heterognn" else "HeteroGNN3"
    print(f"\n{model_name} Results:")
    print(f"Test Accuracy: {test_metrics['acc']:.4f}")
    print(f"Test AUC: {test_metrics.get('auc', float('nan')):.4f}")
    if 'precision' in test_metrics:
        print(f"Test Precision: {test_metrics['precision']:.4f}")
        print(f"Test Recall: {test_metrics['recall']:.4f}")
        print(f"Test F1: {test_metrics['f1']:.4f}")

    # Example of how to run the other model for comparison
    if MODEL_TYPE == "heterognn":
        other_model = "heterognn2"
    elif MODEL_TYPE == "heterognn2":
        other_model = "heterognn3"
    else:
        other_model = "heterognn"
    print(f"\n" + "=" * 60)
    print(f"EXAMPLE: To run {other_model.upper()}, change MODEL_TYPE = '{other_model}' at the top of the script")
    print("=" * 60)

    # Plot loss curves
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(history["train_loss"], label="train loss")
        plt.plot(history["val_loss"], label="val loss")
        plt.xlabel("epoch")
        plt.ylabel("BCE loss")
        plt.title(f"{model_name} Training/Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[plot] Skipped plotting due to: {e}")
