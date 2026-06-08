"""
Shared evaluation module for v1, v2, and v3 xT models.

Computes a standardised set of metrics so results are directly comparable
across all three model generations:

  - ROC AUC    — ranking quality (most meaningful with binary labels in v1;
                 still valid with soft labels in v2/v3 where 1.0 = goal chain)
  - Brier      — mean squared error between predicted prob and true label
  - Log Loss   — cross-entropy; penalises confident wrong predictions heavily
  - Spearman ρ — rank correlation; the primary metric for v2/v3 training

Results are printed to stdout and saved as JSON for cross-version comparison.
"""
import json
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def compute_metrics(y_true, y_prob) -> dict:
    y_true = np.array(y_true, dtype=np.float64)
    y_prob = np.clip(np.array(y_prob, dtype=np.float64), 1e-7, 1 - 1e-7)

    # Manual binary cross-entropy — handles soft labels that sklearn's log_loss rejects
    bce = -np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))

    metrics = {
        "spearman": float(spearmanr(y_true, y_prob).statistic),
        "brier":    float(np.mean((y_true - y_prob) ** 2)),
        "log_loss": float(bce),
    }
    # For soft labels, binarise: any event in a goal chain (label > 0) counts as positive.
    # This lets ROC AUC measure whether the model ranks goal-chain events above background.
    y_binary = (y_true > 0).astype(np.float64)
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_binary, y_prob))
    except ValueError:
        metrics["roc_auc"] = float("nan")

    return metrics


def print_metrics(metrics: dict, label: str = "") -> None:
    header = f"--- {label} ---" if label else "--- Metrics ---"
    print(f"\n{header}")
    print(f"  ROC AUC:   {metrics.get('roc_auc', float('nan')):.4f}")
    print(f"  Brier:     {metrics['brier']:.4f}")
    print(f"  Log Loss:  {metrics['log_loss']:.4f}")
    print(f"  Spearman:  {metrics['spearman']:.4f}")


def save_metrics(metrics: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved → {path}")
