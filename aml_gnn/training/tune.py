"""
training/tune.py

Optuna hyperparameter optimisation for the AML-GNN.

Objective: maximise validation ROC-AUC.
Search space: hidden_channels, num_layers, dropout, lr, weight_decay, heads.

Usage:
    python training/tune.py --n_trials 50 --timeout 7200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

sys.path.insert(0, str(Path(__file__).parent.parent))
from training.train import train
from utils.helpers import get_logger, load_config, set_seed

logger = get_logger("tune")


def objective(trial: optuna.Trial, base_cfg: dict) -> float:
    import copy
    cfg = copy.deepcopy(base_cfg)

    sp = cfg["optuna"]["search_space"]

    # Sample hyperparameters
    cfg["model"]["hidden_channels"] = trial.suggest_categorical(
        "hidden_channels", sp["hidden_channels"]
    )
    cfg["model"]["num_layers"] = trial.suggest_int(
        "num_layers", sp["num_layers"][0], sp["num_layers"][1]
    )
    cfg["model"]["dropout"] = trial.suggest_float(
        "dropout", sp["dropout"][0], sp["dropout"][1]
    )
    cfg["model"]["heads"] = trial.suggest_categorical(
        "heads", sp["heads"]
    )
    cfg["training"]["lr"] = trial.suggest_float(
        "lr", sp["lr"][0], sp["lr"][1], log=True
    )
    cfg["training"]["weight_decay"] = trial.suggest_float(
        "weight_decay", sp["weight_decay"][0], sp["weight_decay"][1], log=True
    )

    # Short epochs for tuning
    cfg["training"]["epochs"] = 40
    cfg["training"]["early_stopping_patience"] = 8

    # Unique checkpoint per trial to avoid collisions
    cfg["paths"]["checkpoint_dir"] = f"checkpoints/trial_{trial.number}"

    val_auc = train(cfg=cfg, trial=trial)
    return val_auc


def run_study(n_trials: int = 50, timeout: int = 7200, cfg_path: str = "configs/config.yaml"):
    cfg = load_config(cfg_path)
    set_seed(cfg["seed"])

    optuna.logging.get_logger("optuna").propagate = True
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=cfg["seed"]),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        study_name="aml_gnn_tuning",
        storage=None,  # In-memory; swap for "sqlite:///optuna.db" for persistence
    )

    study.optimize(
        lambda trial: objective(trial, cfg),
        n_trials=n_trials,
        timeout=timeout,
        catch=(Exception,),
    )

    best = study.best_trial
    logger.info(f"Best trial: #{best.number}")
    logger.info(f"  Val ROC-AUC: {best.value:.4f}")
    logger.info(f"  Params: {best.params}")

    # Save best params
    out_path = Path(cfg["paths"]["output_dir"]) / "best_params.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"val_auc": best.value, "params": best.params}, f, indent=2)

    logger.info(f"Best params saved to {out_path}")
    return best.params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--timeout",  type=int, default=7200)
    parser.add_argument("--config",   type=str, default="configs/config.yaml")
    args = parser.parse_args()
    run_study(args.n_trials, args.timeout, args.config)
