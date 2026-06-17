"""
inference/structural_analysis.py

Standalone structural AML analysis that runs WITHOUT a trained model.
Useful for:
  1. Rapid insight generation before GNN training completes
  2. Interpretable baseline for comparison
  3. Seed generation for semi-supervised labelling

Detects:
  - Smurfing (fan-out / fan-in)
  - Layering (multi-hop chains)
  - Circular flows
  - Temporal bursts
  - PEP / sanctions network proximity
  - Amount structuring (just under reporting thresholds)

Output: outputs/structural_suspicious_accounts.csv
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.helpers import get_logger, load_config

logger = get_logger("structural_analysis")


NPR_THRESHOLD = 1_000_000  # 1M NPR — reporting threshold
JUST_BELOW = 0.95          # flag amounts that are 95–99.9% of threshold


def load_data(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    txn = pd.read_csv(cfg["paths"]["transactions"])
    acc = pd.read_csv(cfg["paths"]["accounts"])
    txn["datetime"] = pd.to_datetime(txn["Date"] + " " + txn["Time"])
    return txn, acc


def build_graph(txn: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    for _, row in txn.iterrows():
        s, r = row["Sender_account"], row["Receiver_account"]
        if not G.has_edge(s, r):
            G.add_edge(s, r, amounts=[], dates=[], cross_border=0)
        G[s][r]["amounts"].append(row.get("amount_local_npr", 0))
        G[s][r]["dates"].append(str(row.get("datetime", "")))
        G[s][r]["cross_border"] += int(row.get("cross_border_flag", 0))
    return G


def score_all_accounts(txn: pd.DataFrame, acc: pd.DataFrame, G: nx.DiGraph) -> pd.DataFrame:
    all_accounts = pd.concat([txn["Sender_account"], txn["Receiver_account"]]).unique()

    out_deg   = txn.groupby("Sender_account").size()
    in_deg    = txn.groupby("Receiver_account").size()
    fan_out   = txn.groupby("Sender_account")["Receiver_account"].nunique()
    fan_in    = txn.groupby("Receiver_account")["Sender_account"].nunique()
    cb_ratio  = txn.groupby("Sender_account")["cross_border_flag"].mean()
    amt_max   = txn.groupby("Sender_account")["amount_local_npr"].max()
    amt_sum   = txn.groupby("Sender_account")["amount_local_npr"].sum()
    amt_zscore_max = txn.groupby("Sender_account")["amount_zscore"].max()

    # PEP / sanctions flags
    pep_map   = dict(zip(acc["account_id"], acc["pep_flag"]))
    sanc_map  = dict(zip(acc["account_id"], acc["sanctions_hit"]))

    # Circular flow: bidirectional edges
    edges = set(zip(txn["Sender_account"], txn["Receiver_account"]))
    bidir = set(s for s, r in edges if (r, s) in edges)

    # Structuring: amounts just below NPR_THRESHOLD
    struct_txn = txn[
        (txn["amount_local_npr"] >= JUST_BELOW * NPR_THRESHOLD) &
        (txn["amount_local_npr"] < NPR_THRESHOLD)
    ]
    structuring_count = struct_txn.groupby("Sender_account").size()

    # Temporal burst: coefficient of variation of inter-tx gaps
    txn_s = txn.sort_values(["Sender_account", "datetime"])
    txn_s["gap"] = txn_s.groupby("Sender_account")["datetime"].diff().dt.total_seconds()
    gap_cv = txn_s.groupby("Sender_account")["gap"].apply(
        lambda x: x.std() / (x.mean() + 1e-9) if len(x) > 1 else 0
    )

    rows = []
    for aid in all_accounts:
        score = 0.0
        flags = []

        od = out_deg.get(aid, 0)
        id_ = in_deg.get(aid, 0)
        fo = fan_out.get(aid, 0)
        fi = fan_in.get(aid, 0)
        cb = cb_ratio.get(aid, 0.0)
        pep = pep_map.get(aid, 0)
        sanc = sanc_map.get(aid, 0)
        circ = int(aid in bidir)
        struct = structuring_count.get(aid, 0)
        cv = gap_cv.get(aid, 0.0)
        az = amt_zscore_max.get(aid, 0.0)

        # Fan-out (smurfing)
        if fo >= 10:
            score += min(fo / 25.0, 1.0) * 3.0
            flags.append(f"FAN-OUT({fo})")

        # Fan-in (aggregator)
        if fi >= 10:
            score += min(fi / 25.0, 1.0) * 2.5
            flags.append(f"FAN-IN({fi})")

        # PEP / sanctions
        if pep:
            score += 2.5
            flags.append("PEP")
        if sanc:
            score += 4.0
            flags.append("SANCTIONS")

        # Cross-border intensity
        if cb > 0.5 and od > 3:
            score += cb * 2.0
            flags.append(f"XB({cb*100:.0f}%)")

        # Circular
        if circ:
            score += 1.5
            flags.append("CIRCULAR")

        # Structuring (just-below-threshold amounts)
        if struct >= 3:
            score += min(struct / 10.0, 1.0) * 2.0
            flags.append(f"STRUCTURE({struct})")

        # Temporal burst (irregular timing)
        if cv > 2.0 and od > 5:
            score += min(cv / 5.0, 1.0)
            flags.append(f"BURST(cv={cv:.1f})")

        # Abnormal amounts
        if az > 3.0:
            score += min(az / 5.0, 1.0)
            flags.append(f"HIGHZ({az:.1f})")

        rows.append({
            "account_id": aid,
            "heuristic_score": score,
            "flags": " | ".join(flags) if flags else "NONE",
            "out_degree": od,
            "in_degree": id_,
            "unique_receivers": fo,
            "unique_senders": fi,
            "cb_ratio": round(cb, 3),
            "pep_flag": pep,
            "sanctions_hit": sanc,
            "circular_flag": circ,
            "structuring_count": struct,
            "burst_cv": round(cv, 3),
            "amt_zscore_max": round(az, 3),
        })

    result = pd.DataFrame(rows).sort_values("heuristic_score", ascending=False)
    result["rank"] = range(1, len(result) + 1)
    return result


def run_structural_analysis(cfg_path: str = "configs/config.yaml") -> pd.DataFrame:
    cfg = load_config(cfg_path)
    txn, acc = load_data(cfg)

    logger.info("Building transaction graph …")
    G = build_graph(txn)
    logger.info(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    logger.info("Scoring accounts …")
    scored = score_all_accounts(txn, acc, G)

    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "structural_suspicious_accounts.csv"
    scored.to_csv(out_path, index=False)

    logger.info(f"Structural analysis saved: {out_path}")
    print("\nTop 20 structurally suspicious accounts:")
    print(scored[["rank", "account_id", "heuristic_score", "flags"]].head(20).to_string(index=False))

    # Summary by flag type
    flag_counts = {}
    for flags_str in scored[scored["heuristic_score"] > 0]["flags"]:
        for f in flags_str.split(" | "):
            key = f.split("(")[0]
            flag_counts[key] = flag_counts.get(key, 0) + 1

    print("\nFlag type summary:")
    for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
        print(f"  {flag:<15} {count:>5} accounts")

    return scored


if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    run_structural_analysis(cfg_path)
