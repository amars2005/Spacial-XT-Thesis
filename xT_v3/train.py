import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))

# Load v3 config by file path to avoid v2/config shadowing it on sys.path
_spec  = importlib.util.spec_from_file_location("_v3config", os.path.join(_HERE, "config.py"))
_v3cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v3cfg)

NUM_EPOCHS_V3             = _v3cfg.NUM_EPOCHS_V3
LEARNING_RATE_V3          = _v3cfg.LEARNING_RATE_V3
LEARNING_RATE_V3_FINETUNE = _v3cfg.LEARNING_RATE_V3_FINETUNE
WEIGHT_DECAY_V3           = _v3cfg.WEIGHT_DECAY_V3
BEST_MODEL_PATH_V3        = _v3cfg.BEST_MODEL_PATH_V3
CHECKPOINT_DIR_V3         = _v3cfg.CHECKPOINT_DIR_V3
METRICS_PATH_V3           = _v3cfg.METRICS_PATH_V3

_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_ROOT, 'scripts')
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from evaluate import compute_metrics, print_metrics, save_metrics


def _log(msg: str) -> None:
    print(msg, flush=True)


def train_v3(
    model:            nn.Module,
    train_loader:     DataLoader,
    val_loader:       DataLoader,
    device:           torch.device,
    test_loader:      DataLoader | None = None,
    metrics_path:     str | None = None,
    best_model_path:  str | None = None,
) -> nn.Module:
    """
    Train Stage 2 of xT_v3.  Stage 1 (xT_v2 CNN) is frozen inside the model.

    Uses BCEWithLogitsLoss with a capped pos_weight to handle class imbalance,
    Adam with L2 regularisation, and ReduceLROnPlateau monitoring ROC AUC.

    Each epoch logs a ΔAUC = (v3 AUC) − (Stage 1 alone AUC) on the
    same val events, so you can see exactly how much Stage 2 adds.

    Returns
    -------
    model with best-checkpoint weights loaded.
    """
    if metrics_path is None:
        metrics_path = METRICS_PATH_V3
    if best_model_path is None:
        best_model_path = BEST_MODEL_PATH_V3

    os.makedirs(CHECKPOINT_DIR_V3, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(best_model_path)), exist_ok=True)

    # Positive class weight (same capping strategy as v2)
    train_labels = train_loader.dataset.labels
    pos_rate     = train_labels.mean().item()
    pos_weight   = torch.tensor(
        [min((1.0 - pos_rate) / (pos_rate + 1e-8), 10.0)]
    ).to(device)
    _log(f"Class balance — goal rate: {pos_rate:.4f}  |  pos_weight: {pos_weight.item():.1f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    stage2_params = [p for p in model.parameters() if p.requires_grad]
    _log(f"Trainable Stage 2 parameters: {sum(p.numel() for p in stage2_params):,}")

    optimizer = optim.Adam(stage2_params, lr=LEARNING_RATE_V3, weight_decay=WEIGHT_DECAY_V3)
    all_params = stage2_params
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5,
    )

    # Compute Stage 1 baseline once on the val set before any Stage 2 training.
    # This is the reference AUC we want Stage 2 to beat.
    _log("\nComputing Stage 1 baseline on val set...")
    stage1_auc, stage1_mse, _ = _evaluate_stage1(model.stage1, val_loader, device)
    _log(
        f"Stage 1 baseline  |  "
        f"Val AUC: {stage1_auc:.4f}  |  Val MSE: {stage1_mse:.4f}\n"
    )

    best_val_auc = 0.0

    for epoch in range(1, NUM_EPOCHS_V3 + 1):

        # ---- Training ------------------------------------------------
        model.train()
        train_loss = torch.tensor(0.0, device=device)

        for spatial, scalar, player_tokens, player_mask, labels in train_loader:
            spatial       = spatial.to(device,       non_blocking=True)
            scalar        = scalar.to(device,        non_blocking=True)
            player_tokens = player_tokens.to(device, non_blocking=True)
            player_mask   = player_mask.to(device,   non_blocking=True)
            labels        = labels.to(device,        non_blocking=True)

            optimizer.zero_grad()
            logits = model(spatial, scalar, player_tokens, player_mask)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.detach() * len(labels)

        torch.cuda.synchronize()
        train_loss = train_loss.item() / len(train_loader.dataset)

        # ---- Validation ----------------------------------------------
        val_auc, val_mse, val_loss, mean_adj, grad_norm = _evaluate(
            model, val_loader, criterion, device, all_params
        )
        delta_auc = val_auc - stage1_auc
        scheduler.step(val_auc)

        _log(
            f"Epoch {epoch:>3}/{NUM_EPOCHS_V3}  |  "
            f"Train Loss: {train_loss:.4f}  |  "
            f"Val Loss: {val_loss:.4f}  |  "
            f"Val AUC: {val_auc:.4f}  |  "
            f"ΔAUC vs Stage1: {delta_auc:+.4f}  |  "
            f"Val MSE: {val_mse:.4f}  |  "
            f"Mean |adj|: {mean_adj:.5f}  |  "
            f"GradNorm: {grad_norm:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            _log(f"  -> Saved best model  (AUC {val_auc:.4f})")

    _log(
        f"\nTraining complete."
        f"  Best Val AUC: {best_val_auc:.4f}"
        f"  (Stage 1 baseline: {stage1_auc:.4f},"
        f"  ΔAUC: {best_val_auc - stage1_auc:+.4f})"
    )

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    if test_loader is not None:
        _log("\nRunning final evaluation on held-out test set...")
        probs, labels = _get_predictions_v3(model, test_loader, device)
        metrics = compute_metrics(labels, probs)
        print_metrics(metrics, "Test Set (v3 Two-Stage Transformer)")
        save_metrics(metrics, metrics_path)

    return model


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def _evaluate(
    model:        nn.Module,
    loader:       DataLoader,
    criterion:    nn.Module,
    device:       torch.device,
    stage2_params: list,
) -> tuple[float, float, float, float, float]:
    """Return (ROC AUC, MSE, mean loss, mean |adj|, grad norm)."""
    model.eval()
    all_probs:  list[float] = []
    all_labels: list[float] = []
    all_adjs:   list[float] = []
    total_loss = 0.0

    with torch.no_grad():
        for spatial, scalar, player_tokens, player_mask, labels in loader:
            spatial       = spatial.to(device,       non_blocking=True)
            scalar        = scalar.to(device,        non_blocking=True)
            player_tokens = player_tokens.to(device, non_blocking=True)
            player_mask   = player_mask.to(device,   non_blocking=True)
            labels        = labels.to(device,        non_blocking=True)

            # Get Stage 1 logit and Stage 2 adjustment separately
            with torch.no_grad():
                stage1_logit = model.stage1(spatial, scalar)           # (B,1)
            logits = model(spatial, scalar, player_tokens, player_mask) # (B,1)
            adj    = logits - stage1_logit                              # (B,1)

            loss   = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_adjs.extend(adj.abs().cpu().numpy().tolist())

    all_probs  = np.array(all_probs,  dtype=np.float64)
    all_labels = np.array(all_labels, dtype=np.float64)

    auc       = roc_auc_score(all_labels, all_probs)
    mse       = np.mean((all_labels - all_probs) ** 2)
    mean_loss = total_loss / len(loader.dataset)
    mean_adj  = float(np.mean(all_adjs))

    # Gradient norm of Stage 2 params (computed from last backward pass)
    grad_norm = float(np.sqrt(sum(
        p.grad.norm().item() ** 2
        for p in stage2_params if p.grad is not None
    )))

    return auc, mse, mean_loss, mean_adj, grad_norm


def _evaluate_stage1(
    stage1: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    """
    Run Stage 1 alone on the val loader (spatial + scalar inputs only).
    Returns (ROC AUC, MSE, 0.0) as the like-for-like baseline on exactly
    the same freeze-frame events Stage 2 trains on.
    """
    stage1.eval()
    all_probs:  list[float] = []
    all_labels: list[float] = []

    with torch.no_grad():
        for spatial, scalar, _player_tokens, _player_mask, labels in loader:
            spatial = spatial.to(device, non_blocking=True)
            scalar  = scalar.to(device,  non_blocking=True)

            logits = stage1(spatial, scalar)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_probs  = np.array(all_probs,  dtype=np.float64)
    all_labels = np.array(all_labels, dtype=np.float64)

    auc = roc_auc_score(all_labels, all_probs)
    mse = np.mean((all_labels - all_probs) ** 2)

    return auc, mse, 0.0


def _get_predictions_v3(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (probs, labels) arrays for the full loader — used for final test evaluation."""
    model.eval()
    all_probs:  list[float] = []
    all_labels: list[float] = []

    with torch.no_grad():
        for spatial, scalar, player_tokens, player_mask, labels in loader:
            spatial       = spatial.to(device,       non_blocking=True)
            scalar        = scalar.to(device,        non_blocking=True)
            player_tokens = player_tokens.to(device, non_blocking=True)
            player_mask   = player_mask.to(device,   non_blocking=True)

            logits = model(spatial, scalar, player_tokens, player_mask)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    return (
        np.array(all_probs,  dtype=np.float64),
        np.array(all_labels, dtype=np.float64),
    )
