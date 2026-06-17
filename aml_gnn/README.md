# AML-GNN: Graph Neural Network for Anti-Money Laundering Detection

> **Track 3 — Network & Graph Intelligence**  
> Hackathon solution for detecting suspicious accounts via graph-structured transaction analysis.

---

## Problem Overview

Money laundering operates through **networks**, not isolated transactions. A single suspicious transfer is invisible to conventional anomaly detectors; the signal emerges only when you model the entire transaction graph and look for structural patterns:

| Typology | Graph Pattern | Visibility |
|---|---|---|
| **Smurfing** | One account → many intermediaries → single collector | Multi-hop fan-out/fan-in |
| **Layering** | Sequential chains (A→B→C→D) | Path-length analysis |
| **Circular flows** | Money returns to origin (A→B→C→A) | Cycle detection |
| **Structuring** | Repeated amounts just below reporting thresholds | Edge weight analysis |
| **Cross-border velocity** | High volume of international transfers | Edge attribute aggregation |

This solution builds a **Graph Attention Network (GAT)** that embeds each account as a node in the transaction graph and learns to identify suspicious structural signatures across all typologies simultaneously.

---

## Dataset Summary

| File | Rows | Description |
|---|---|---|
| `accounts.csv` | 65,339 | KYC records: institution, risk grade, PEP flag, sanctions, city, account type |
| `transactions.csv` | 100,222 | Directed transfers with 55 pre-engineered features: amounts, FX rates, timing, cross-border flags, transmodes |

**Key statistics:**
- 22,310 unique senders, 46,586 unique receivers (65,339 unique accounts total)
- Date range: 2022-10-07 → 2022-11-06 (one month)
- 10.1% cross-border transactions
- 851 PEP-flagged accounts, 275 accounts with sanctions hits
- Max out-degree: 265 transactions from a single account
- 2,084 bidirectional account pairs (circular flow candidates)

---

## Deep Learning Architecture

### Selected Architecture: Edge-Feature-Augmented GAT + Jumping Knowledge

After analysing the data, a **Graph Attention Network with edge feature conditioning** was chosen over alternatives for the following reasons:

| Architecture | Why Not Chosen |
|---|---|
| MLP | Cannot capture multi-hop neighbourhood patterns |
| TabNet | Tabular; ignores graph structure |
| Standard GCN | Edge features ignored; isotropic aggregation |
| GNN without edge features | Loses transaction-level signals (amount, cross-border, FX) |
| Autoencoder only | Good unsupervised baseline but weaker than supervised GAT |

**Why GAT with edge features:**
1. **Attention mechanism** learns which neighbours to trust — critical for AML where most neighbours are benign and a few are the true signal
2. **Edge feature conditioning** allows transaction attributes (amount, FX rate, cross-border flag) to modulate message strength, not just adjacency structure
3. **Jumping Knowledge (JK-Net)** aggregates representations from all layers, giving the classifier access to both local (1-hop) and global (3-hop) context simultaneously

### Mathematical Intuition

**Layer ℓ update rule:**

```
α_{uv} = softmax_u( LeakyReLU( aᵀ [W·hᵤ || W·h_v || Wₑ·e_{uv}] ) )

h_v^(ℓ+1) = σ( Σ_{u∈N(v)} α_{uv} · W_msg · (hᵤ^(ℓ) || e_{uv}) )
```

Where:
- `h_v^(ℓ)` = representation of node v at layer ℓ
- `e_{uv}` = edge feature vector (amount, cross-border, FX rate, timing, ...)
- `α_{uv}` = learned attention coefficient (how much to weight neighbour u)
- `W, Wₑ` = learnable weight matrices

**Jumping Knowledge aggregation:**
```
h_v^final = Concat( h_v^(1), h_v^(2), ..., h_v^(L) )
```

Stacking L=3 layers means each node aggregates information from its **3-hop neighbourhood** — sufficient to capture smurfing (fan-out at hop 1, fan-in at hop 2) and layering chains (hop 3 reveals the distal source/destination).

**Loss function:**
```
L = BCEWithLogitsLoss(logits, y, pos_weight = N_neg / N_pos)
```

`pos_weight` upweights the minority class (suspicious accounts) to counter the severe class imbalance (~1-2% positive rate from PEP/sanctions seeds).

---

## Training Methodology

### Semi-supervised Setup

Because no labelled suspicious/non-suspicious ground truth is provided, the system uses a **two-phase strategy**:

**Phase 1 — Structural heuristics as pseudo-labels:**
- Accounts with PEP flags, sanctions hits, or structural outlier scores above threshold → seed positives
- All other accounts → seed negatives
- Train the GNN on these pseudo-labels

**Phase 2 — Ensemble scoring:**
- GNN score (learned structural embedding): weight 0.7
- Heuristic score (interpretable rule-based): weight 0.3
- Final ranking uses ensemble score

### Training Details

| Setting | Value |
|---|---|
| Optimiser | AdamW |
| Learning rate | 1e-3 (cosine decay) |
| Weight decay | 1e-4 |
| Epochs | 100 (early stopping @ patience=15) |
| Batch strategy | Full-batch (graph fits GPU memory) |
| Gradient clipping | max_norm=1.0 |
| Early stopping metric | Validation ROC-AUC |

---

## Validation Strategy

**Stratified node-level split (85% train / 15% val)**

For graph data, random node splits are used (rather than temporal splits) because:
1. The graph must remain connected during message passing (all edges kept; only evaluation masked)
2. Stratification ensures both splits contain PEP/sanctions seeds proportionally
3. Temporal leakage is not a concern since all transactions are from the same 30-day window

**Metrics:**
- Primary: **ROC-AUC** (robust to class imbalance)
- Secondary: Average Precision (PR-AUC), F1, Precision@K

---

## Hyperparameter Tuning

Uses **Optuna with TPE Sampler + Median Pruner**:

```
Search space:
  hidden_channels : [64, 128, 256]
  num_layers      : 2–4
  dropout         : 0.1–0.5
  lr              : 1e-4 – 1e-2 (log scale)
  weight_decay    : 1e-5 – 1e-3 (log scale)
  attention_heads : [2, 4, 8]

n_trials  : 50
timeout   : 7200s
Objective : maximise validation ROC-AUC
```

The pruner kills clearly underperforming trials after 8 epochs, saving compute for promising configurations.

---

## Loss Functions — Justification

| Loss | Usage | Justification |
|---|---|---|
| `BCEWithLogitsLoss(pos_weight)` | Main GNN training | Handles binary classification + imbalance in one numerically stable operation |
| `MSELoss` (autoencoder) | Unsupervised pre-training | Reconstruction error as anomaly score; high error = structurally unusual node |

The `pos_weight` parameter is computed as `N_neg / N_pos` on the training split, dynamically adjusting to the actual label distribution without requiring manual tuning.

---

## Optimisation Techniques

| Technique | Purpose |
|---|---|
| AdamW | Weight decay decoupled from adaptive gradient scaling |
| Cosine annealing LR | Escape local minima; fine-tune at low LR near convergence |
| Gradient clipping (1.0) | Stabilise training on sparse graph signals |
| Layer normalisation | Normalise across feature dimensions at each GNN layer |
| Residual connections | Prevent over-smoothing in deep GNN stacks |
| Jumping Knowledge | Preserve both local and global structural context |
| Dropout (0.3) | Regularise against overfitting on small positive set |

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| ROC-AUC | Primary: area under ROC curve, threshold-independent |
| Average Precision | Area under PR curve; more informative at high imbalance |
| Precision@K | Fraction of true positives in the top-K ranked accounts |
| F1 @ threshold | Harmonic mean of precision/recall at operating threshold |

For the hackathon submission, the primary deliverable is a **ranked list** of suspicious accounts — Average Precision and Precision@K are the most directly relevant metrics.

---

## Inference Workflow

```
raw data
    │
    ▼
graph_builder.py         ← build PyG Data object (nodes=accounts, edges=txns)
    │
    ▼
AMLGNN.predict_proba()   ← forward pass, sigmoid output → [0,1] score per node
    │
    ▼
structural_analysis.py   ← heuristic score (fan-out, sanctions, CB ratio, ...)
    │
    ▼
ensemble (0.7 GNN + 0.3 heuristic)
    │
    ▼
explain_node()           ← typology detection + natural-language explanation
    │
    ▼
generate_submission.py   ← ranked CSV + subgraph pattern report
```

---

## Project Folder Structure

```
aml_gnn/
├── configs/
│   └── config.yaml                  # Master configuration
├── data/
│   ├── raw/                         # accounts.csv, transactions.csv
│   ├── __init__.py
│   └── graph_builder.py             # Graph construction + feature engineering
├── models/
│   ├── __init__.py
│   └── gnn.py                       # AMLGNN, EdgeGATConv, NodeAutoencoder
├── training/
│   ├── __init__.py
│   ├── train.py                     # Training loop
│   ├── validate.py                  # Validation + metric reporting
│   └── tune.py                      # Optuna hyperparameter search
├── inference/
│   ├── __init__.py
│   ├── inference.py                 # Full inference pipeline
│   ├── structural_analysis.py       # Interpretable heuristic baseline
│   └── generate_submission.py       # Final submission files
├── utils/
│   ├── __init__.py
│   └── helpers.py                   # Logging, seeding, checkpointing, metrics
├── outputs/                         # Generated scores, reports, submission CSVs
├── checkpoints/                     # Model checkpoints (.pt files)
├── logs/                            # Training logs (JSON + text)
├── requirements.txt
└── README.md
```

---

## Installation Instructions

```bash
# 1. Create environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install PyTorch (CUDA 11.8 example — adjust for your CUDA version)
pip install torch==2.2.0 torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Install PyTorch Geometric
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu118.html

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Place data files
cp accounts.csv     data/raw/
cp transactions.csv data/raw/
```

---

## Reproducibility Instructions

All random seeds are fixed via `utils/helpers.set_seed(42)`, which seeds:
- Python `random`
- NumPy
- PyTorch CPU & CUDA
- `PYTHONHASHSEED`
- `torch.backends.cudnn.deterministic = True`

```bash
# Step 1: Structural analysis (no GPU needed, runs in ~60s)
python inference/structural_analysis.py

# Step 2: Train GNN (GPU recommended)
python training/train.py

# Step 3: (Optional) Hyperparameter tuning
python training/tune.py --n_trials 50 --timeout 7200

# Step 4: Validate
python training/validate.py

# Step 5: Full inference + explanation
python inference/inference.py

# Step 6: Generate submission files
python inference/generate_submission.py --top_k 200 --threshold 0.4
```

Outputs in `outputs/`:
- `structural_suspicious_accounts.csv` — heuristic ranking (immediate, interpretable)
- `node_scores.csv` — full per-account scores (GNN + heuristic + ensemble)
- `suspicious_accounts.csv` — top-K with typology explanations
- `submission_ranked_accounts.csv` — final hackathon submission
- `submission_subgraph_patterns.csv` — detected multi-hop patterns

---

## Identified Risks and Mitigations

| Risk | Mitigation |
|---|---|
| **No ground truth labels** | Semi-supervised: PEP/sanctions as seed positives; heuristic ensemble fallback |
| **Severe class imbalance** | `pos_weight` in BCE loss; stratified splits; AUC/AP metrics |
| **Over-smoothing** (deep GNN) | JK-Net; residual connections; LayerNorm; max 4 layers |
| **Data leakage** | Node features computed from same time window — no future data used |
| **Graph sparsity** | NeighborLoader for mini-batch if graph grows; full-batch fits here |
| **Over-fitting on pseudo-labels** | Dropout (0.3); weight decay; early stopping on separate val set |
| **Degree centrality bias** | Explanation engine explicitly goes beyond degree — typology patterns required |

---

## Potential Future Improvements

1. **Temporal GNN (TGN)**: Model the graph as evolving over time; detect sudden structural shifts within the 30-day window
2. **Heterogeneous graph**: Separate node types for accounts, institutions, cities; model cross-type relationships
3. **Active learning**: Present top flagged accounts to compliance team; incorporate feedback labels to fine-tune the GNN
4. **GNNExplainer / SubgraphX**: Generate per-node subgraph explanations that pinpoint exactly which edges triggered the flag
5. **Federated learning**: If multi-bank data available, train GNN across institutions without sharing raw transactions
6. **Graph-level classification**: Detect suspicious subgraph communities, not just individual nodes
7. **Contrastive pre-training**: Use GraphCL to pre-train node embeddings on unlabelled data before supervised fine-tuning
8. **Link prediction auxiliary task**: Train jointly on predicting future transaction links, improving embedding quality

---

## Contact / Reproducibility

All hyperparameters, random seeds, and dataset paths are declared in `configs/config.yaml`. To reproduce any result exactly, ensure the same seed (42) and the same data files are used. Model checkpoints include the full config dict for provenance.
