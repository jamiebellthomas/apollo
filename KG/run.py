from __future__ import annotations

from typing import List, Tuple, Dict
import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from HeteroGNN import HeteroGNN  
# ----------------------------
# 1) small helpers
# ----------------------------

def attach_y(dataset: List[HeteroData], dtype: torch.dtype = torch.float) -> None:
    """Mirror your 'graph_label' to the canonical .y field as float."""
    for g in dataset:
        # Ensure shape [1] per-graph; DataLoader will stack to [B]
        g.y = g["graph_label"].to(dtype=dtype).view(-1)

def split_dataset(dataset: List[HeteroData], train_ratio=0.7, val_ratio=0.15, seed=42):
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
    """pos_weight for BCEWithLogitsLoss = N_neg / N_pos."""
    y = torch.cat([g.y for g in dataset]).float()
    pos = (y > 0.5).sum().item()
    neg = len(y) - pos
    # Avoid div-by-zero
    pw = (neg / max(pos, 1)) if pos > 0 else 1.0
    return torch.tensor(pw, dtype=torch.float)

# ----------------------------
# 2) build dataset (example)
# ----------------------------
# Suppose you have a list of your SubGraph objects -> convert to HeteroData:
# graphs: List[HeteroData] = [sg.to_pyg_data() for sg in subgraphs]

# Make sure .y is present and float:
# attach_y(graphs)

# ----------------------------
# 3) data splits & loaders
# ----------------------------
# train_set, val_set, test_set = split_dataset(graphs, train_ratio=0.7, val_ratio=0.15, seed=42)
# attach_y(train_set); attach_y(val_set); attach_y(test_set)  # no-op if already done

def make_loader(ds: List[HeteroData], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

# train_loader = make_loader(train_set, batch_size=32, shuffle=True)
# val_loader   = make_loader(val_set,   batch_size=64, shuffle=False)
# test_loader  = make_loader(test_set,  batch_size=64, shuffle=False)

# ----------------------------
# 4) model / loss / optimizer
# ----------------------------
# Instantiate the model with dataset metadata:
# metadata = train_set[0].metadata()
# model = HeteroGNN(metadata=metadata, hidden_channels=64, num_layers=2, edge_weight_index=1, dropout=0.3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# Because the model uses LazyLinear for per-type projections,
# initialize parameters with a dry forward on one batch *before* creating the optimizer.
@torch.no_grad()
def init_lazy_params(m: nn.Module, sample_batch: HeteroData, device: torch.device) -> None:
    m.to(device)
    _ = m(sample_batch.to(device))

# sample_batch = next(iter(train_loader))
# init_lazy_params(model, sample_batch, device)

# Handle class imbalance if present:
# pos_weight = compute_pos_weight(train_set).to(device)
# criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # or nn.BCEWithLogitsLoss()

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# ----------------------------
# 5) train / eval loops
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
        logits: Tensor = model(batch)                 # [B]
        y: Tensor = batch.y.float().to(device)        # [B]
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
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    all_logits, all_targets = [], []

    # Use the same loss as training just for reporting
    dummy_crit = nn.BCEWithLogitsLoss(reduction="sum")

    for batch in loader:
        batch = batch.to(device)
        logits: Tensor = model(batch)                 # [B]
        y: Tensor = batch.y.float().to(device)        # [B]
        loss = dummy_crit(logits, y)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()
        correct += (preds == y.long()).sum().item()
        total += y.numel()

        total_loss += loss.item()
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return {"loss": avg_loss, "acc": acc}

# ----------------------------
# 6) full training script (wire it all together)
# ----------------------------
def run_training(
    train_set: List[HeteroData],
    val_set: List[HeteroData],
    test_set: List[HeteroData],
    hidden_channels: int = 64,
    num_layers: int = 2,
    edge_weight_index: int = 1,   # 1 = decayed_weight, 0 = sentiment
    dropout: float = 0.3,
    batch_size: int = 32,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> Tuple[nn.Module, Dict[str, float]]:
    torch.manual_seed(seed)

    attach_y(train_set); attach_y(val_set); attach_y(test_set)
    train_loader = make_loader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = make_loader(val_set,   batch_size=2*batch_size, shuffle=False)
    test_loader  = make_loader(test_set,  batch_size=2*batch_size, shuffle=False)

    metadata = train_set[0].metadata()
    model = HeteroGNN(
        metadata=metadata,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        edge_weight_index=edge_weight_index,
        dropout=dropout,
    )
    model.to(device)

    # Initialize LazyLinear params
    sample_batch = next(iter(train_loader))
    init_lazy_params(model, sample_batch, device)

    pos_weight = compute_pos_weight(train_set).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = math.inf
    best_state: Dict[str, Tensor] | None = None

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} "
              f"| val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['acc']:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, device)
    print(f"TEST | loss={test_metrics['loss']:.4f} | acc={test_metrics['acc']:.3f}")

    return model, test_metrics

# ----------------------------
# 7) usage example
# ----------------------------
# graphs = [sg.to_pyg_data() for sg in subgraphs]   # your construction
# train_set, val_set, test_set = split_dataset(graphs)
# model, test_metrics = run_training(train_set, val_set, test_set)
