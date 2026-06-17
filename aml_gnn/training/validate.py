"""
training/validate.py

Standalone validation script.
Loads the best checkpoint, runs inference on the validation split,
and produces:
  - Full metrics report (AUC, AP, F1, precision, recall)
  - Precision-Recall curve data (saved as JSON)
  - ROC curve data (saved as JSON)
  - Per-node score CSV for inspection
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve, roc_curve

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.graph_builder import build_graph
from models.gnn import AMLGNN, build_model
from training.train import make_masks
from utils.helpers import (
    compute_metrics,
    get_device,
    get_logger,
    load_checkpoint,
    load_config,
    set_seed,
)

logger = get_logger("validate")


def validate(cfg: dict | None = None) -> dict:
    if cfg is None:
        cfg = load_config("configs/config.yaml")

    set_seed(cfg["seed"])
    device = get_device()

    # ── Data ─────────────────────────────────────────────────
    data, account_df, scaler = build_graph(
        cfg["paths"]["accounts"],
        cfg["paths"]["transactions"],
    )
    _, val_mask = make_masks(
        data.num_nodes,
        data.y,
        val_split=cfg["training"]["val_split"],
        seed=cfg["seed"],
    )

    # ── Load model ───────────────────────────────────────────
    ckpt_path = str(Path(cfg["paths"]["checkpoint_dir"]) / "checkpoint_best.pt")
    in_channels = data.x.shape[1]
    edge_dim    = data.edge_attr.shape[1]
    model = build_model(in_channels, edge_dim, cfg).to(device)
    ckpt  = load_checkpoint(ckpt_path, model)
    logger.info(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} (val_auc={ckpt.get('val_auc', '?'):.4f})")

    # ── Inference ────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        logits = model(
            data.x.to(device),
            data.edge_index.to(device),
            data.edge_attr.to(device),
        ).cpu()
    probs = torch.sigmoid(logits).numpy()

    y_true  = data.y.numpy()
    y_val   = y_true[val_mask.numpy()]
    p_val   = probs[val_mask.numpy()]

    # ── Metrics ──────────────────────────────────────────────
    metrics = compute_metrics(y_val, p_val, threshold=cfg["inference"]["threshold"])
    logger.info("Validation metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # ── Curves ──────────────────────────────────────────────
    fpr, tpr, roc_thresh = roc_curve(y_val, p_val)
    prec, rec, pr_thresh = precision_recall_curve(y_val, p_val)

    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "roc_curve.json", "w") as f:
        json.dump({"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": metrics["roc_auc"]}, f)

    with open(out_dir / "pr_curve.json", "w") as f:
        json.dump({"precision": prec.tolist(), "recall": rec.tolist(), "ap": metrics["avg_precision"]}, f)

    # ── Per-node scores ──────────────────────────────────────
    score_df = account_df[["account_id", "pep_flag", "sanctions_hit", "risk_grade_enc"]].copy()
    score_df["gnn_score"] = probs
    score_df["label"]     = y_true
    score_df["in_val"]    = val_mask.numpy()
    score_df = score_df.sort_values("gnn_score", ascending=False)
    score_df.to_csv(out_dir / "node_scores.csv", index=False)

    logger.info(f"Results saved to {out_dir}")
    return metrics


if __name__ == "__main__":
    m = validate()
    print(json.dumps(m, indent=2))
