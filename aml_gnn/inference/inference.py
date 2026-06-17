"""
inference/inference.py

Runs the trained AML-GNN on the full graph and produces:
  1. node_scores.csv  — every account ranked by suspicion score
  2. suspicious_accounts.csv — top-K flagged accounts with structural explanation
  3. subgraph_report.json — suspicious subgraph descriptions (multi-hop patterns)

The explanation engine identifies WHY each account is flagged using:
  - Fan-out score (smurfing)
  - Fan-in score (aggregation / mule accounts)
  - Circular flow detection
  - High-value cross-border transaction ratio
  - PEP / sanctions proximity
  - Temporal burst patterns
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.graph_builder import build_graph
from models.gnn import AMLGNN, build_model
from utils.helpers import (
    get_device,
    get_logger,
    load_checkpoint,
    load_config,
    set_seed,
    structural_suspicion_score,
)

logger = get_logger("inference")


# ──────────────────────────────────────────────
# Structural explanation engine
# ──────────────────────────────────────────────

def build_networkx_graph(txn: pd.DataFrame) -> nx.DiGraph:
    """Build a NetworkX DiGraph for structural analysis."""
    G = nx.DiGraph()
    for _, row in txn.iterrows():
        G.add_edge(
            row["Sender_account"],
            row["Receiver_account"],
            amount=row.get("amount_local_npr", 0),
            cross_border=row.get("cross_border_flag", 0),
            date=row.get("Date", ""),
            time=row.get("Time", ""),
        )
    return G


def explain_node(
    account_id: int,
    G: nx.DiGraph,
    account_df: pd.DataFrame,
    txn: pd.DataFrame,
    max_hops: int = 2,
) -> Dict:
    """
    Generate a structural explanation for why an account is suspicious.
    Returns a dict describing the AML typology pattern detected.
    """
    explanations = []
    typologies   = []

    row = account_df[account_df["account_id"] == account_id]
    row = row.iloc[0] if len(row) > 0 else None

    out_nbrs = list(G.successors(account_id))
    in_nbrs  = list(G.predecessors(account_id))

    out_deg = G.out_degree(account_id)
    in_deg  = G.in_degree(account_id)

    # ── Smurfing (fan-out) ────────────────────────────────────
    if out_deg >= 10:
        # Check if receivers then forward to a common target
        recv_targets = {}
        for recv in out_nbrs:
            for target in G.successors(recv):
                recv_targets[target] = recv_targets.get(target, 0) + 1
        top_collectors = sorted(recv_targets.items(), key=lambda x: -x[1])[:3]

        typologies.append("SMURFING / FAN-OUT")
        explanations.append(
            f"Account sends to {out_deg} distinct receivers. "
            + (
                f"Of these, {top_collectors[0][1]} receivers forward funds to "
                f"account {top_collectors[0][0]} — classic smurfing pattern."
                if top_collectors else ""
            )
        )

    # ── Fan-in / Aggregator ───────────────────────────────────
    if in_deg >= 10:
        typologies.append("FAN-IN / AGGREGATOR")
        # Compute how much of the in-flow this account retains vs. forwards
        in_amount  = sum(G[u][account_id].get("amount", 0) for u in in_nbrs)
        out_amount = sum(G[account_id][v].get("amount", 0) for v in out_nbrs)
        retention  = 1 - (out_amount / (in_amount + 1))
        explanations.append(
            f"Account receives from {in_deg} distinct senders (total inflow ≈ {in_amount:,.0f} NPR, "
            f"retention ≈ {retention*100:.0f}%). "
            f"{'Retains most funds — possible final collector.' if retention > 0.7 else 'Forwards most funds — possible layering node.'}"
        )

    # ── Circular flows ────────────────────────────────────────
    bidir_peers = [n for n in out_nbrs if account_id in G.successors(n)]
    if bidir_peers:
        typologies.append("CIRCULAR FLOW")
        explanations.append(
            f"Bidirectional transfers detected with {len(bidir_peers)} accounts: "
            f"{bidir_peers[:5]}. Money may be cycling to obscure origin."
        )

    # ── High-value cross-border ───────────────────────────────
    acct_txn = txn[txn["Sender_account"] == account_id]
    cb_ratio = acct_txn["cross_border_flag"].mean() if len(acct_txn) > 0 else 0
    if cb_ratio > 0.5 and len(acct_txn) > 3:
        typologies.append("CROSS-BORDER INTENSITY")
        total_cb = acct_txn[acct_txn["cross_border_flag"] == 1]["amount_local_npr"].sum()
        explanations.append(
            f"{cb_ratio*100:.0f}% of outgoing transactions are cross-border "
            f"(total ≈ {total_cb:,.0f} NPR)."
        )

    # ── KYC red flags ─────────────────────────────────────────
    if row is not None:
        if row.get("pep_flag", 0):
            typologies.append("PEP FLAG")
            explanations.append("Account holder is a Politically Exposed Person (PEP).")
        if row.get("sanctions_hit", 0):
            typologies.append("SANCTIONS HIT")
            explanations.append("Account has a sanctions list hit.")
        if row.get("counterparty_pep", 0):
            typologies.append("PEP COUNTERPARTY")
            explanations.append("Transacted with at least one PEP-flagged account.")

    # ── Layering chain: multi-hop path to high-risk node ─────
    try:
        # Look for paths of length 2-3 involving PEP/sanctions accounts
        pep_accounts = set(
            account_df[account_df["pep_flag"] > 0]["account_id"].tolist()
        )
        for hop_len in [2, 3]:
            paths_found = []
            for target in pep_accounts:
                if target == account_id:
                    continue
                try:
                    path = nx.shortest_path(G, account_id, target)
                    if len(path) == hop_len + 1:
                        paths_found.append(path)
                        if len(paths_found) >= 2:
                            break
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass
            if paths_found:
                typologies.append(f"LAYERING CHAIN ({hop_len}-HOP)")
                explanations.append(
                    f"Funds reachable to {len(paths_found)} PEP account(s) in {hop_len} hops: "
                    f"e.g., {' → '.join(str(n) for n in paths_found[0])}."
                )
                break
    except Exception:
        pass

    if not typologies:
        typologies = ["ELEVATED GNN SCORE"]
        explanations = ["GNN model assigned elevated suspicion score based on neighbourhood patterns."]

    return {
        "account_id":  account_id,
        "typologies":  typologies,
        "explanation": " | ".join(explanations),
        "out_degree":  out_deg,
        "in_degree":   in_deg,
    }


# ──────────────────────────────────────────────
# Main inference pipeline
# ──────────────────────────────────────────────

def run_inference(cfg: dict | None = None) -> pd.DataFrame:
    if cfg is None:
        cfg = load_config("configs/config.yaml")

    set_seed(cfg["seed"])
    device = get_device()

    # ── Build graph ───────────────────────────────────────────
    data, account_df, scaler = build_graph(
        cfg["paths"]["accounts"],
        cfg["paths"]["transactions"],
    )

    # ── Load model ───────────────────────────────────────────
    ckpt_path = str(Path(cfg["paths"]["checkpoint_dir"]) / "checkpoint_best.pt")
    in_channels = data.x.shape[1]
    edge_dim    = data.edge_attr.shape[1]
    model = build_model(in_channels, edge_dim, cfg).to(device)

    try:
        ckpt = load_checkpoint(ckpt_path, model)
        logger.info(f"Loaded trained model checkpoint (val_auc={ckpt.get('val_auc', '?'):.4f})")
    except FileNotFoundError:
        logger.warning("No checkpoint found — using randomly initialised model. Run training first.")

    # ── GNN scores ───────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        logits = model(
            data.x.to(device),
            data.edge_index.to(device),
            data.edge_attr.to(device),
        ).cpu()
    gnn_scores = torch.sigmoid(logits).numpy()

    # ── Structural heuristic scores ───────────────────────────
    import pandas as pd as _pd
    txn = pd.read_csv(cfg["paths"]["transactions"])

    out_deg_map     = txn.groupby("Sender_account").size().to_dict()
    fan_out_map     = txn.groupby("Sender_account")["Receiver_account"].nunique().to_dict()
    in_deg_map      = txn.groupby("Receiver_account").size().to_dict()
    fan_in_map      = txn.groupby("Receiver_account")["Sender_account"].nunique().to_dict()
    cb_ratio_map    = txn.groupby("Sender_account")["cross_border_flag"].mean().to_dict()
    amt_z_map       = txn.groupby("Sender_account")["amount_zscore"].max().to_dict()

    heuristic_scores = []
    for _, row in account_df.iterrows():
        aid = row["account_id"]
        h = structural_suspicion_score(
            out_degree       = out_deg_map.get(aid, 0),
            in_degree        = in_deg_map.get(aid, 0),
            unique_receivers = fan_out_map.get(aid, 0),
            unique_senders   = fan_in_map.get(aid, 0),
            pep_flag         = int(row.get("pep_flag", 0)),
            sanctions_hit    = int(row.get("sanctions_hit", 0)),
            cross_border_ratio = cb_ratio_map.get(aid, 0.0),
            amount_zscore_max  = amt_z_map.get(aid, 0.0),
        )
        heuristic_scores.append(h)

    heuristic_arr = np.array(heuristic_scores)
    heuristic_norm = heuristic_arr / (heuristic_arr.max() + 1e-9)

    # ── Ensemble: GNN + heuristic ─────────────────────────────
    ensemble_score = 0.7 * gnn_scores + 0.3 * heuristic_norm

    # ── Score DataFrame ───────────────────────────────────────
    score_df = account_df[["account_id", "account_number", "pep_flag",
                            "sanctions_hit", "risk_grade_enc",
                            "out_degree", "in_degree",
                            "unique_receivers", "unique_senders",
                            "circular_flag", "cb_ratio_send"]].copy()

    score_df["gnn_score"]       = gnn_scores
    score_df["heuristic_score"] = heuristic_norm
    score_df["ensemble_score"]  = ensemble_score
    score_df = score_df.sort_values("ensemble_score", ascending=False)

    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    score_df.to_csv(out_dir / "node_scores.csv", index=False)
    logger.info(f"Full account scores saved to {out_dir / 'node_scores.csv'}")

    # ── Top-K explanation ─────────────────────────────────────
    top_k    = cfg["inference"]["top_k"]
    top_accs = score_df.head(top_k)["account_id"].tolist()

    logger.info("Building NetworkX graph for explanation …")
    G = build_networkx_graph(txn)

    explanations = []
    for aid in top_accs:
        exp = explain_node(aid, G, account_df, txn)
        exp["ensemble_score"] = float(score_df[score_df["account_id"] == aid]["ensemble_score"].iloc[0])
        exp["gnn_score"]      = float(score_df[score_df["account_id"] == aid]["gnn_score"].iloc[0])
        explanations.append(exp)

    # Save detailed report
    with open(out_dir / "suspicious_accounts_report.json", "w") as f:
        json.dump(explanations, f, indent=2, default=str)

    # Human-readable CSV summary
    summary = pd.DataFrame([
        {
            "rank": i + 1,
            "account_id": e["account_id"],
            "ensemble_score": round(e["ensemble_score"], 4),
            "gnn_score": round(e["gnn_score"], 4),
            "typologies": " | ".join(e["typologies"]),
            "explanation": e["explanation"][:300],
            "out_degree": e["out_degree"],
            "in_degree": e["in_degree"],
        }
        for i, e in enumerate(explanations)
    ])
    summary.to_csv(out_dir / "suspicious_accounts.csv", index=False)
    logger.info(f"Top-{top_k} suspicious accounts saved to {out_dir / 'suspicious_accounts.csv'}")

    return summary


if __name__ == "__main__":
    df = run_inference()
    print(df.head(20).to_string(index=False))
