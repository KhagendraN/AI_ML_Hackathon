"""
training/train.py

Full training pipeline for the AML-GNN:

  1. Build the transaction graph
  2. Split nodes into train / val masks
  3. Train with mini-batch NeighborLoader + BCEWithLogitsLoss
     (pos_weight for class imbalance)
  4. Early stopping on val ROC-AUC
  5. Cosine annealing LR schedule
  6. Model checkpointing
  7. Experiment tracking via a JSON metrics log
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch_geometric.loader import NeighborLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.graph_builder import build_graph
from models.gnn import AMLGNN, build_model
from utils.helpers import (
    compute_metrics,
    get_device,
    get_logger,
    load_config,
    save_checkpoint,
    set_seed,
)

logger = get_logger("train")


# ──────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────

def make_masks(
    num_nodes: int,
    y: torch.Tensor,
    val_split: float = 0.15,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Stratified split into train / val node masks.
    For semi-supervised settings, only labelled nodes are split;
    all nodes participate in message passing.
    """
    indices = np.arange(num_nodes)
    labels  = y.numpy()

    # Use stratified split only if we have at least 2 classes
    unique = np.unique(labels)
    if len(unique) > 1:
        train_idx, val_idx = train_test_split(
            indices, test_size=val_split, stratify=labels, random_state=seed
        )
    else:
        train_idx, val_idx = train_test_split(
            indices, test_size=val_split, random_state=seed
        )

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    return train_mask, val_mask


def compute_pos_weight(y: torch.Tensor, train_mask: torch.Tensor) -> torch.Tensor:
    """BCEWithLogitsLoss pos_weight = #negatives / #positives in training set."""
    y_train = y[train_mask].numpy()
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    if n_pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(n_neg / n_pos, dtype=torch.float)


# ──────────────────────────────────────────────
# One epoch (full-batch — fits in GPU memory)
# ──────────────────────────────────────────────

def train_epoch(
    model: AMLGNN,
    data,
    train_mask: torch.Tensor,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    x          = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_attr  = data.edge_attr.to(device)
    y          = data.y.to(device)

    optimizer.zero_grad()
    logits = model(x, edge_index, edge_attr)
    loss   = criterion(logits[train_mask.to(device)], y[train_mask.to(device)])
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(
    model: AMLGNN,
    data,
    mask: torch.Tensor,
    device: torch.device,
) -> dict:
    model.eval()
    x          = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_attr  = data.edge_attr.to(device)
    y          = data.y

    logits = model(x, edge_index, edge_attr).cpu()
    probs  = torch.sigmoid(logits)

    y_true  = y[mask].numpy()
    y_score = probs[mask].numpy()

    # Guard: if only one class in val, skip AUC
    if len(np.unique(y_true)) < 2:
        return {"roc_auc": 0.5, "avg_precision": float(y_true.mean())}

    return compute_metrics(y_true, y_score)


# ──────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────

def train(cfg: dict | None = None, trial=None) -> float:
    """
    Parameters
    ----------
    cfg   : config dict (loaded from YAML if None)
    trial : Optuna trial (for pruning; None during standalone training)

    Returns
    -------
    best_val_auc : float
    """
    if cfg is None:
        cfg = load_config("configs/config.yaml")

    set_seed(cfg["seed"])
    device = get_device()
    logger.info(f"Using device: {device}")

    # ── Build graph ───────────────────────────────────────────
    data, account_df, scaler = build_graph(
        cfg["paths"]["accounts"],
        cfg["paths"]["transactions"],
    )

    train_mask, val_mask = make_masks(
        data.num_nodes,
        data.y,
        val_split=cfg["training"]["val_split"],
        seed=cfg["seed"],
    )
    logger.info(
        f"Train nodes: {train_mask.sum().item()} | "
        f"Val nodes: {val_mask.sum().item()} | "
        f"Train positives: {data.y[train_mask].sum().item():.0f}"
    )

    # ── Model ─────────────────────────────────────────────────
    in_channels = data.x.shape[1]
    edge_dim    = data.edge_attr.shape[1]
    model = build_model(in_channels, edge_dim, cfg).to(device)
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Loss with pos_weight ──────────────────────────────────
    pos_weight = compute_pos_weight(data.y, train_mask).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    logger.info(f"pos_weight: {pos_weight.item():.3f}")

    # ── Optimiser & scheduler ─────────────────────────────────
    t_cfg = cfg["training"]
    optimizer = optim.AdamW(
        model.parameters(),
        lr=t_cfg["lr"],
        weight_decay=t_cfg["weight_decay"],
    )
    epochs = t_cfg["epochs"]
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Training loop ─────────────────────────────────────────
    best_val_auc = 0.0
    patience_ctr = 0
    patience     = t_cfg["early_stopping_patience"]
    history      = []

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0   = time.time()
        loss = train_epoch(model, data, train_mask, optimizer, criterion, device)
        scheduler.step()

        train_metrics = evaluate(model, data, train_mask, device)
        val_metrics   = evaluate(model, data, val_mask,   device)
        val_auc       = val_metrics["roc_auc"]

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch:03d} | loss={loss:.4f} | "
            f"train_auc={train_metrics['roc_auc']:.4f} | "
            f"val_auc={val_auc:.4f} | "
            f"val_ap={val_metrics['avg_precision']:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.6f} | {elapsed:.1f}s"
        )

        row = {"epoch": epoch, "loss": loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)

        is_best = val_auc > best_val_auc
        if is_best:
            best_val_auc = val_auc
            patience_ctr = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_auc": val_auc,
                    "in_channels": in_channels,
                    "edge_dim": edge_dim,
                    "cfg": cfg,
                },
                path=str(ckpt_dir / "checkpoint.pt"),
                is_best=True,
            )
        else:
            patience_ctr += 1

        # Optuna pruning
        if trial is not None:
            trial.report(val_auc, epoch)
            if trial.should_prune():
                raise __import__("optuna").exceptions.TrialPruned()

        if patience_ctr >= patience:
            logger.info(f"Early stopping at epoch {epoch}. Best val AUC: {best_val_auc:.4f}")
            break

    # Save history
    log_path = Path(cfg["paths"]["log_dir"]) / "train_history.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training complete. Best val ROC-AUC: {best_val_auc:.4f}")
    return best_val_auc


if __name__ == "__main__":
    result = train()
    print(f"\nBest validation ROC-AUC: {result:.4f}")
