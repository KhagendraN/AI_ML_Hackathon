"""
utils/helpers.py — Shared utilities for the AML-GNN project.
"""
from __future__ import annotations

import os
import random
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

def get_logger(name: str, log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        fh = logging.FileHandler(Path(log_dir) / f"{name}.log")
        fh.setFormatter(fmt)
        logger.addHandler(sh)
        logger.addHandler(fh)
    return logger


# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ──────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────

def load_config(path: str = "configs/config.yaml") -> Dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


# ──────────────────────────────────────────────
# Device handling
# ──────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    return dev


# ──────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────

def save_checkpoint(
    state: Dict[str, Any],
    path: str,
    is_best: bool = False,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    if is_best:
        best_path = str(path).replace(".pt", "_best.pt")
        torch.save(state, best_path)


def load_checkpoint(path: str, model: torch.nn.Module, optimizer: Optional[Any] = None) -> Dict:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt


# ──────────────────────────────────────────────
# Metric helpers
# ──────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
    )

    y_pred = (y_score >= threshold).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(y_true, y_score),
        "avg_precision": average_precision_score(y_true, y_score),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }
    return metrics


# ──────────────────────────────────────────────
# Graph-level structural metrics (no labels needed)
# ──────────────────────────────────────────────

def structural_suspicion_score(
    out_degree: int,
    in_degree: int,
    unique_receivers: int,
    unique_senders: int,
    pep_flag: int,
    sanctions_hit: int,
    cross_border_ratio: float,
    amount_zscore_max: float,
) -> float:
    """
    Heuristic suspicion score for unsupervised ranking.
    Combines structural and attribute signals.
    """
    score = 0.0
    # Fan-out / smurfing signal
    if unique_receivers > 10:
        score += min(unique_receivers / 25.0, 1.0) * 2.0
    # Fan-in / aggregation signal
    if unique_senders > 10:
        score += min(unique_senders / 25.0, 1.0) * 2.0
    # KYC risk flags
    score += pep_flag * 1.5
    score += sanctions_hit * 3.0
    # Cross-border intensity
    score += cross_border_ratio * 2.0
    # Abnormal amounts
    score += min(max(amount_zscore_max, 0.0) / 5.0, 1.0) * 1.5
    return score
