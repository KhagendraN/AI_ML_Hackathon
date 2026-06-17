"""
inference/generate_submission.py

Generates the final hackathon submission:
  1. suspicious_accounts.csv  — ranked list of accounts with explanations
  2. subgraph_patterns.csv    — multi-hop subgraph descriptions

Run AFTER inference/inference.py has produced node_scores.csv.

Usage:
    python inference/generate_submission.py
    python inference/generate_submission.py --top_k 200 --threshold 0.4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.helpers import get_logger, load_config

logger = get_logger("submission")


# ──────────────────────────────────────────────
# Multi-hop pattern finder
# ──────────────────────────────────────────────

def find_fan_out_fan_in(G: nx.DiGraph, suspicious_ids: set, min_fanout: int = 5) -> list:
    """
    Find accounts that fan-out to many intermediaries that then
    fan-in to a common collector — the classic smurfing pattern.
    Returns list of {source, intermediaries, collector, description}.
    """
    patterns = []
    for src in suspicious_ids:
        if src not in G:
            continue
        out_nbrs = list(G.successors(src))
        if len(out_nbrs) < min_fanout:
            continue
        # Count where the intermediaries forward to
        collector_votes: dict = {}
        for mid in out_nbrs:
            for dst in G.successors(mid):
                collector_votes[dst] = collector_votes.get(dst, 0) + 1
        # Find a collector that receives from >= min_fanout intermediaries
        top_collectors = [(c, n) for c, n in collector_votes.items() if n >= min_fanout]
        for collector, n_paths in top_collectors[:3]:
            patterns.append({
                "source_account": src,
                "num_intermediaries": len(out_nbrs),
                "collector_account": collector,
                "paths_through_collector": n_paths,
                "pattern_type": "SMURFING (fan-out → fan-in)",
                "description": (
                    f"Account {src} sends to {len(out_nbrs)} intermediaries, "
                    f"of which {n_paths} forward to collector {collector}. "
                    f"Consistent with smurfing / structuring typology."
                ),
            })
    return patterns


def find_circular_chains(G: nx.DiGraph, suspicious_ids: set, max_len: int = 4) -> list:
    """
    Detect short cycles involving suspicious nodes.
    """
    patterns = []
    for node in suspicious_ids:
        if node not in G:
            continue
        try:
            # DFS-based cycle detection limited to max_len
            for cycle in nx.simple_cycles(G.subgraph(
                list(nx.ego_graph(G, node, radius=max_len).nodes())
            )):
                if node in cycle and 2 <= len(cycle) <= max_len:
                    patterns.append({
                        "cycle": cycle,
                        "length": len(cycle),
                        "pattern_type": "CIRCULAR FLOW",
                        "description": (
                            f"Money cycle detected: {' → '.join(str(n) for n in cycle)} → {cycle[0]}. "
                            f"Funds return to origin after {len(cycle)} hops — possible layering."
                        ),
                    })
                    if len(patterns) > 50:
                        break
        except Exception:
            pass
        if len(patterns) > 50:
            break
    return patterns


def find_layering_chains(
    G: nx.DiGraph,
    suspicious_ids: set,
    account_df: pd.DataFrame,
    chain_len: int = 3,
) -> list:
    """
    Identify linear chains of length >= chain_len where funds flow
    through a sequence of accounts — layering typology.
    """
    patterns = []
    # Nodes with low in-degree and high out-degree are chain starters
    for src in suspicious_ids:
        if src not in G:
            continue
        if G.in_degree(src) > 3:
            continue  # Skip aggregators, focus on chain initiators
        try:
            # Find all simple paths of exactly chain_len hops
            all_paths = list(nx.all_simple_paths(
                G, src, cutoff=chain_len,
                # target: any node reachable in chain_len steps
            ))
            long_paths = [p for p in all_paths if len(p) == chain_len + 1]
            if long_paths:
                ex_path = long_paths[0]
                # Compute total amount along path
                total = sum(
                    G[ex_path[i]][ex_path[i + 1]].get("amount", 0)
                    for i in range(len(ex_path) - 1)
                )
                patterns.append({
                    "source": src,
                    "chain": ex_path,
                    "chain_length": chain_len,
                    "approx_amount_npr": total,
                    "pattern_type": f"LAYERING CHAIN ({chain_len}-HOP)",
                    "description": (
                        f"Funds flow: {' → '.join(str(n) for n in ex_path)} "
                        f"(≈ {total:,.0f} NPR over {chain_len} hops). "
                        f"Sequential transfers consistent with layering."
                    ),
                })
                if len(patterns) >= 30:
                    break
        except Exception:
            pass
    return patterns


# ──────────────────────────────────────────────
# Main submission generator
# ──────────────────────────────────────────────

def generate_submission(
    top_k: int = 200,
    threshold: float = 0.4,
    cfg_path: str = "configs/config.yaml",
) -> None:
    cfg = load_config(cfg_path)
    out_dir = Path(cfg["paths"]["output_dir"])

    # ── Load scores ───────────────────────────────────────────
    scores_path = out_dir / "node_scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(
            f"{scores_path} not found. Run `python inference/inference.py` first."
        )
    score_df = pd.read_csv(scores_path)

    # ── Load transactions for graph construction ───────────────
    txn = pd.read_csv(cfg["paths"]["transactions"])
    acc = pd.read_csv(cfg["paths"]["accounts"])

    logger.info("Building NetworkX graph …")
    G = nx.DiGraph()
    for _, row in txn.iterrows():
        G.add_edge(
            row["Sender_account"],
            row["Receiver_account"],
            amount=row.get("amount_local_npr", 0),
            cross_border=int(row.get("cross_border_flag", 0)),
        )

    # ── Flag suspicious accounts ──────────────────────────────
    flagged = score_df[score_df["ensemble_score"] >= threshold].head(top_k)
    suspicious_ids = set(flagged["account_id"].tolist())

    logger.info(f"Flagged accounts: {len(suspicious_ids)} (threshold={threshold})")

    # ── Primary submission: ranked account list ────────────────
    # Load explanations from inference run
    exp_path = out_dir / "suspicious_accounts_report.json"
    if exp_path.exists():
        with open(exp_path) as f:
            explanations = {e["account_id"]: e for e in json.load(f)}
    else:
        explanations = {}

    submission_rows = []
    for rank, (_, row) in enumerate(
        flagged.sort_values("ensemble_score", ascending=False).iterrows(), start=1
    ):
        aid = row["account_id"]
        exp = explanations.get(aid, {})

        # Look up KYC info
        acc_row = acc[acc["account_id"] == aid]
        acct_num  = acc_row["account_number"].iloc[0] if len(acc_row) > 0 else "Unknown"
        city      = acc_row["city"].iloc[0]           if len(acc_row) > 0 else "Unknown"
        risk_grade = acc_row["risk_grade"].iloc[0]    if len(acc_row) > 0 else "Unknown"
        is_person  = acc_row["is_person"].iloc[0]     if len(acc_row) > 0 else "Unknown"

        submission_rows.append({
            "rank":               rank,
            "account_id":         aid,
            "account_number":     acct_num,
            "city":               city,
            "risk_grade":         risk_grade,
            "is_person":          is_person,
            "ensemble_score":     round(float(row["ensemble_score"]), 4),
            "gnn_score":          round(float(row["gnn_score"]), 4),
            "heuristic_score":    round(float(row["heuristic_score"]), 4),
            "typologies":         " | ".join(exp.get("typologies", ["ELEVATED SCORE"])),
            "explanation":        exp.get("explanation", "Elevated GNN score based on neighbourhood.")[:500],
            "out_degree":         int(G.out_degree(aid)) if aid in G else 0,
            "in_degree":          int(G.in_degree(aid))  if aid in G else 0,
        })

    submission_df = pd.DataFrame(submission_rows)
    sub_path = out_dir / "submission_ranked_accounts.csv"
    submission_df.to_csv(sub_path, index=False)
    logger.info(f"Primary submission saved: {sub_path}")

    # ── Secondary submission: multi-hop subgraph patterns ─────
    logger.info("Mining multi-hop structural patterns …")
    fo_fi_patterns = find_fan_out_fan_in(G, suspicious_ids, min_fanout=5)
    circ_patterns  = find_circular_chains(G, suspicious_ids, max_len=4)
    lay_patterns   = find_layering_chains(G, suspicious_ids, acc, chain_len=3)

    all_patterns = (
        [{"category": "SMURFING", **p} for p in fo_fi_patterns] +
        [{"category": "CIRCULAR", **p} for p in circ_patterns]  +
        [{"category": "LAYERING", **p} for p in lay_patterns]
    )

    patterns_df = pd.DataFrame([
        {
            "category": p["category"],
            "pattern_type": p.get("pattern_type", p["category"]),
            "description": p.get("description", ""),
        }
        for p in all_patterns
    ])
    pat_path = out_dir / "submission_subgraph_patterns.csv"
    patterns_df.to_csv(pat_path, index=False)
    logger.info(f"Subgraph patterns saved: {pat_path}")

    # ── Summary stats ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("   SUBMISSION SUMMARY")
    print("=" * 60)
    print(f"  Flagged accounts:      {len(submission_rows)}")
    print(f"  Smurfing patterns:     {len(fo_fi_patterns)}")
    print(f"  Circular flows:        {len(circ_patterns)}")
    print(f"  Layering chains:       {len(lay_patterns)}")
    print(f"\nTop 10 suspicious accounts:")
    print(submission_df[["rank", "account_id", "ensemble_score", "typologies"]].head(10).to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k",     type=int,   default=200)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--config",    type=str,   default="configs/config.yaml")
    args = parser.parse_args()
    generate_submission(args.top_k, args.threshold, args.config)
