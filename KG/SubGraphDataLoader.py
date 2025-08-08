# train_hetero_gnn.py
from __future__ import annotations

from typing import List, Tuple, Dict
import math
import os
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

# --- bring your loader + model ---
from SubGraphDataLoader import SubGraphDataLoader
# If your HeteroGNN is in another file, import it here:
# from hetero_gnn import HeteroGNN
from HeteroGNN import HeteroGNN  # <- adjust to your actual path/name


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def attach_y(dataset: List[HeteroData]) -> None:
    """
    Mirror your graph-level label to the canonical .y field (float),
    shape [1] per graph. DataLoader will stack to [B].
    """
    for g in dataset:
        g.y = g["graph_label"].float().view(-1)

def split_list(dataset: List[HeteroData], train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Simple randomized split for a list.
    """
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=g).tolist()
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    idx_train = perm[:n_train]
    idx_val = perm[n_train:n_train+n_val]
    idx_test = perm[n_train+n_val:]
    return [dataset[i] for i in idx_train], [dataset[i] for i in idx_val], [dataset[i] for i in idx_test]

def compute_pos_weight(dataset: List[HeteroData]) -> Tensor:
    """
    pos_weight for BCEWithLogitsLoss = N_neg / N_pos (helps with class imbalance).
    """
    y = torch.cat([g.y for g in dataset]).float()
    pos = (y > 0.5).sum().item()
    neg = len(y) - pos
    if pos == 0:
        return torch.tensor(1.0, dtype=torch.float)
    return torch.tensor(neg / pos, dtype=torch.float)

def make_loader(ds: List[HeteroData], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

@torch.no_grad()
def init_lazy_params(model: nn.Module, sample_batch: HeteroData, device: torch.device) -> None:
    """
    GCNConv with in_channels=(-1, -1) lazily initializes on first forward.
    We do a dry forward BEFORE creating the optimizer so those params get included.
    """
    model.to(device)
    _ = model(sample_batch.to(device))

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
) -> Dict[str, float]:
    """
    Returns average loss (if report_loss) and accuracy.
    """
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    # Loss just for reporting; use reduction="sum" to average later.
    criterion = nn.BCEWithLogitsLoss(reduction="sum")

    for batch in loader:
        batch = batch.to(device)
        logits: Tensor = model(batch)    # [B]
        y: Tensor = batch.y              # [B]

        if report_loss:
            loss = criterion(logits, y)
            total_loss += loss.item()

        preds = (torch.sigmoid(logits) >= 0.5).long()
        correct += (preds == y.long()).sum().item()
        total += y.numel()

    metrics = {"acc": correct / max(total, 1)}
    if report_loss:
        metrics["loss"] = total_loss / max(total, 1)
    return metrics


# ----------------------------
# Main runner
# ----------------------------
def run_training(
    # --- data / encoding ---
    n_facts: int = 50,
    limit: int | None = None,
    # --- model ---
    hidden_channels: int = 64,
    num_layers: int = 2,
    feature_dropout: float = 0.3,
    edge_dropout: float = 0.0,
    final_dropout: float = 0.1,
    readout: str = "fact",          # 'fact' | 'company' | 'concat' | 'gated'
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
) -> Tuple[nn.Module, Dict[str, float], Dict[str, list]]:
    set_seed(seed)

    # 1) Load & encode graphs -------------------------------------------------
    loader = SubGraphDataLoader(n_facts=n_facts, limit=limit)
    graphs: List[HeteroData] = loader.encode_all_to_heterodata()

    if len(graphs) < 3:
        raise RuntimeError(f"Need at least 3 graphs to split; got {len(graphs)}")

    attach_y(graphs)

    # 2) Split & DataLoaders --------------------------------------------------
    train_set, val_set, test_set = split_list(graphs, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    train_loader = make_loader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = make_loader(val_set,   batch_size=2 * batch_size, shuffle=False)
    test_loader  = make_loader(test_set,  batch_size=2 * batch_size, shuffle=False)

    # 3) Model / Loss / Optimizer --------------------------------------------
    metadata = train_set[0].metadata()
    model = HeteroGNN(
        metadata=metadata,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        feature_dropout=feature_dropout,
        edge_dropout=edge_dropout,
        final_dropout=final_dropout,
        readout=readout,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize lazy params BEFORE creating optimizer
    sample_batch = next(iter(train_loader))
    init_lazy_params(model, sample_batch, device)

    pos_weight = compute_pos_weight(train_set).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ---- NEW: training history we’ll fill every epoch ----
    history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_acc": []}

    # 4) Train loop with best checkpointing ----------------------------------
    best_val = math.inf
    best_state: Dict[str, Tensor] | None = None

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip=grad_clip,
        )
        val_metrics = evaluate(model, val_loader, device)

        # ---- NEW: record into history ----
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['acc']:.3f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        if ckpt_path:
            torch.save(best_state, ckpt_path)
            print(f"[ckpt] Best model saved to {ckpt_path}")

    # 5) Final test -----------------------------------------------------------
    test_metrics = evaluate(model, test_loader, device)
    print(f"TEST | loss={test_metrics['loss']:.4f} | acc={test_metrics['acc']:.3f}")

    # ---- NEW: return history as third item ----
    return model, test_metrics, history



# ----------------------------
# Script entrypoint
# ----------------------------
if __name__ == "__main__":
    model, test_metrics, history = run_training(
        n_facts=50,
        limit=None,
        hidden_channels=64,
        num_layers=2,
        feature_dropout=0.3,
        edge_dropout=0.1,
        final_dropout=0.1,
        readout="fact",
        batch_size=32,
        epochs=20,
        lr=1e-3,
        weight_decay=1e-4,
        seed=42,
        ckpt_path="best_model.pt",
    )

    print(test_metrics)

    # --- NEW: Plot losses ---
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["val_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("BCE loss")
    plt.title("Training/Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()  # or: plt.savefig("loss_curve.png")

