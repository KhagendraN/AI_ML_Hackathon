"""
data/graph_builder.py

Constructs a PyTorch Geometric HeteroData graph from raw CSVs.
Nodes  = accounts.
Edges  = directed transactions (Sender → Receiver).
Node features are enriched with structural graph metrics computed
over the full transaction graph.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, RobustScaler
from torch_geometric.data import Data, HeteroData

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 1.  Graph-structural feature engineering
# ──────────────────────────────────────────────────────────────

def compute_graph_features(txn: pd.DataFrame) -> pd.DataFrame:
    """
    For every account that appears in the transaction table compute
    rich structural features that capture known money-laundering
    typologies:

        Smurfing  → high fan-out (one account → many receivers)
        Layering  → long chains; high betweenness-like scores
        Fan-in    → many senders feeding one receiver (aggregator)
        Circular  → bidirectional pairs
    """
    # Out-degree / in-degree
    out_deg = txn.groupby("Sender_account").size().rename("out_degree")
    in_deg  = txn.groupby("Receiver_account").size().rename("in_degree")

    # Unique counterparties
    unique_recv = txn.groupby("Sender_account")["Receiver_account"].nunique().rename("unique_receivers")
    unique_send = txn.groupby("Receiver_account")["Sender_account"].nunique().rename("unique_senders")

    # Amount statistics per sender/receiver
    send_amt = txn.groupby("Sender_account")["amount_local_npr"].agg(
        send_total="sum", send_mean="mean", send_max="max", send_std="std"
    )
    recv_amt = txn.groupby("Receiver_account")["amount_local_npr"].agg(
        recv_total="sum", recv_mean="mean", recv_max="max", recv_std="std"
    )

    # Cross-border ratio
    cb_send = txn.groupby("Sender_account")["cross_border_flag"].mean().rename("cb_ratio_send")
    cb_recv = txn.groupby("Receiver_account")["cross_border_flag"].mean().rename("cb_ratio_recv")

    # Temporal burst: std of inter-transaction gap (seconds) as sender
    txn2 = txn.copy()
    txn2["datetime"] = pd.to_datetime(txn2["Date"] + " " + txn2["Time"])
    txn2 = txn2.sort_values(["Sender_account", "datetime"])
    txn2["gap"] = txn2.groupby("Sender_account")["datetime"].diff().dt.total_seconds()
    burst = txn2.groupby("Sender_account")["gap"].agg(
        gap_mean="mean", gap_std="std", gap_min="min"
    )
    burst["burst_score"] = burst["gap_mean"] / (burst["gap_std"] + 1.0)

    # Circular flow flag: accounts that both send and receive from the same peer
    pairs = set(zip(txn["Sender_account"], txn["Receiver_account"]))
    bidir_senders = set(s for s, r in pairs if (r, s) in pairs)
    bidir_series = pd.Series(
        {acc: int(acc in bidir_senders) for acc in
         pd.concat([txn["Sender_account"], txn["Receiver_account"]]).unique()},
        name="circular_flag"
    )

    # Amount z-score maximum (abnormal single transactions)
    amt_z_max = txn.groupby("Sender_account")["amount_zscore"].max().rename("amt_zscore_max_send")

    # Velocity: transactions in first 10 and 30 rows (pre-computed in dataset)
    vel = txn.groupby("Sender_account")["velocity_sum_10tx"].mean().rename("mean_velocity")

    # PEP / sanctions propagation (any counterparty with PEP/sanction)
    pep_recv = txn.groupby("Sender_account")["receiver_pep"].max().rename("counterparty_pep")
    sanc_recv = txn.groupby("Sender_account")["receiver_sanctions"].max().rename("counterparty_sanctions")

    # Merge all into one account-level DataFrame
    all_accounts = pd.concat([txn["Sender_account"], txn["Receiver_account"]]).unique()
    gf = pd.DataFrame(index=all_accounts)
    gf.index.name = "account_id"

    for s in [out_deg, in_deg, unique_recv, unique_send,
              send_amt, recv_amt, cb_send, cb_recv,
              burst[["burst_score", "gap_std"]],
              bidir_series, amt_z_max, vel,
              pep_recv, sanc_recv]:
        gf = gf.join(s, how="left")

    gf = gf.fillna(0.0)

    # Derived ratios
    gf["fan_out_ratio"] = gf["unique_receivers"] / (gf["out_degree"] + 1)
    gf["fan_in_ratio"]  = gf["unique_senders"]   / (gf["in_degree"]  + 1)
    gf["send_recv_ratio"] = gf["send_total"] / (gf["recv_total"] + 1.0)
    gf["log_out_degree"] = np.log1p(gf["out_degree"])
    gf["log_in_degree"]  = np.log1p(gf["in_degree"])

    logger.info(f"Graph features computed for {len(gf)} accounts.")
    return gf.reset_index()


# ──────────────────────────────────────────────────────────────
# 2.  Node feature assembly
# ──────────────────────────────────────────────────────────────

RISK_GRADE_MAP = {"RISK-LOW": 0, "RISK-MED": 1, "RISK-HIGH": 2}
ACCT_TYPE_MAP  = {}  # filled at runtime

def encode_accounts(acc: pd.DataFrame, gf: pd.DataFrame) -> pd.DataFrame:
    """Merge KYC attributes with structural graph features."""
    acc = acc.copy()
    acc["risk_grade_enc"] = acc["risk_grade"].map(RISK_GRADE_MAP).fillna(1).astype(int)

    le_type = LabelEncoder()
    acc["acct_type_enc"] = le_type.fit_transform(acc["acct_type"].fillna("Unknown"))

    le_inst = LabelEncoder()
    acc["institution_enc"] = le_inst.fit_transform(acc["institution"].fillna("Unknown"))

    le_city = LabelEncoder()
    acc["city_enc"] = le_city.fit_transform(acc["city"].fillna("Unknown"))

    acc["opened_dt"] = pd.to_datetime(acc["opened"], errors="coerce")
    ref_date = pd.Timestamp("2022-11-06")
    acc["account_age_days"] = (ref_date - acc["opened_dt"]).dt.days.fillna(0).clip(lower=0)
    acc["is_person_int"] = acc["is_person"].astype(int)

    merged = acc.merge(gf, on="account_id", how="left")
    merged = merged.fillna(0.0)
    return merged


# ──────────────────────────────────────────────────────────────
# 3.  Edge feature assembly
# ──────────────────────────────────────────────────────────────

EDGE_FEATURE_COLS = [
    "log_amount", "fx_rate_to_npr", "cross_border_flag", "currency_mismatch",
    "hour_of_day", "day_of_week", "is_weekend",
    "sender_country_risk", "receiver_country_risk",
    "amount_zscore", "above_1M_NPR", "above_10M_NPR",
    "transmode_A", "transmode_B", "transmode_E", "transmode_F",
    "transmode_J", "transmode_P", "transmode_Z",
    "sender_pep", "sender_sanctions", "receiver_pep", "receiver_sanctions",
]


# ──────────────────────────────────────────────────────────────
# 4.  Main graph builder
# ──────────────────────────────────────────────────────────────

def build_graph(
    accounts_path: str,
    transactions_path: str,
    label_col: Optional[str] = None,
    scaler: Optional[RobustScaler] = None,
    fit_scaler: bool = True,
) -> Tuple[Data, pd.DataFrame, RobustScaler]:
    """
    Returns
    -------
    data       : PyG homogeneous Data object (accounts as nodes)
    account_df : merged account DataFrame (account_id aligned to node indices)
    scaler     : fitted RobustScaler (reuse for test/inference)
    """
    logger.info("Loading raw data …")
    acc = pd.read_csv(accounts_path)
    txn = pd.read_csv(transactions_path)

    # ── Structural graph features ──────────────────────────────
    logger.info("Computing structural graph features …")
    gf = compute_graph_features(txn)

    # ── Node feature table ────────────────────────────────────
    account_df = encode_accounts(acc, gf)

    # Node index mapping: account_id → integer index
    node_ids = account_df["account_id"].values
    id_to_idx = {aid: i for i, aid in enumerate(node_ids)}
    N = len(node_ids)
    logger.info(f"Total nodes: {N}")

    # ── Node feature matrix ───────────────────────────────────
    NODE_FEAT_COLS = [
        "is_person_int", "pep_flag", "sanctions_hit",
        "risk_grade_enc", "acct_type_enc", "institution_enc", "city_enc",
        "account_age_days",
        # structural
        "out_degree", "in_degree", "unique_receivers", "unique_senders",
        "log_out_degree", "log_in_degree",
        "fan_out_ratio", "fan_in_ratio", "send_recv_ratio",
        "send_total", "send_mean", "send_max",
        "recv_total", "recv_mean", "recv_max",
        "cb_ratio_send", "cb_ratio_recv",
        "burst_score", "gap_std",
        "circular_flag", "amt_zscore_max_send",
        "mean_velocity", "counterparty_pep", "counterparty_sanctions",
    ]
    # Keep only columns that exist
    NODE_FEAT_COLS = [c for c in NODE_FEAT_COLS if c in account_df.columns]
    x_raw = account_df[NODE_FEAT_COLS].values.astype(np.float32)

    if fit_scaler:
        scaler = RobustScaler()
        x = scaler.fit_transform(x_raw)
    else:
        x = scaler.transform(x_raw)

    x_tensor = torch.tensor(x, dtype=torch.float)

    # ── Edge index + edge features ────────────────────────────
    logger.info("Building edge index …")
    valid_txn = txn[
        txn["Sender_account"].isin(id_to_idx) &
        txn["Receiver_account"].isin(id_to_idx)
    ].copy()

    src = valid_txn["Sender_account"].map(id_to_idx).values
    dst = valid_txn["Receiver_account"].map(id_to_idx).values
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

    edge_feat_cols = [c for c in EDGE_FEATURE_COLS if c in valid_txn.columns]
    edge_attr = torch.tensor(
        valid_txn[edge_feat_cols].fillna(0.0).values.astype(np.float32),
        dtype=torch.float,
    )
    logger.info(f"Edges: {edge_index.shape[1]}, edge features: {edge_attr.shape[1]}")

    # ── Labels (optional) ─────────────────────────────────────
    if label_col and label_col in account_df.columns:
        y = torch.tensor(account_df[label_col].values.astype(np.float32), dtype=torch.float)
    else:
        # Pseudo-labels: flag accounts with PEP or sanctions as seed positives
        # for semi-supervised warm-start
        pseudo = (
            (account_df["pep_flag"] > 0) | (account_df["sanctions_hit"] > 0)
        ).astype(float).values
        y = torch.tensor(pseudo, dtype=torch.float)
        logger.warning("No ground-truth labels found. Using PEP/sanctions as pseudo-labels.")

    data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.num_nodes = N
    data.node_feature_names = NODE_FEAT_COLS
    data.edge_feature_names = edge_feat_cols

    logger.info(f"Graph built: {data}")
    return data, account_df, scaler


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data, df, scaler = build_graph(
        "data/raw/accounts.csv",
        "data/raw/transactions.csv",
    )
    print(data)
    print(f"Positive labels: {data.y.sum().item():.0f} / {data.num_nodes}")
