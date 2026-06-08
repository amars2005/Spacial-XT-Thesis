import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from config import NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, BEST_MODEL_PATH, CHECKPOINT_DIR, METRICS_PATH

_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_ROOT, 'scripts')
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from evaluate import compute_metrics, print_metrics, save_metrics

LOG_PATH = os.path.join(os.path.dirname(CHECKPOINT_DIR), 'train.log')


def _log(msg: str) -> None:
    """Print to stdout, flushed immediately. tee handles the log file."""
    print(msg, flush=True)


def train(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    device:       torch.device,
    test_loader:  DataLoader | None = None,
) -> nn.Module:
    """
    Full training loop with:
      - BCEWithLogitsLoss
      - Adam optimiser with L2 weight decay
      - ReduceLROnPlateau scheduler (monitors val AUC)
      - Checkpoint saving on best validation AUC

    Parameters
    ----------
    model        : XTModel instance (already moved to device)
    train_loader : DataLoader for training set
    val_loader   : DataLoader for validation set
    device       : torch.device

    Returns
    -------
    model with weights from the best checkpoint loaded.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Compute positive class weight from training labels to handle severe class imbalance.
    # Goal rate is ~0.5%, so without this the model collapses to predicting all zeros.
    # The raw ratio (~315) over-amplifies goal-chain gradients, so cap at 10.
    train_labels = train_loader.dataset.labels
    pos_rate = train_labels.mean().item()
    pos_weight = torch.tensor([min((1.0 - pos_rate) / (pos_rate + 1e-8), 10.0)]).to(device)
    _log(f"Class balance — goal rate: {pos_rate:.4f}  |  pos_weight: {pos_weight.item():.1f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5
    )

    best_val_auc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):

        # ---- Training ------------------------------------------------
        model.train()
        t0 = time.perf_counter()
        train_loss = torch.tensor(0.0, device=device)

        for spatial, scalar, labels in train_loader:
            spatial = spatial.to(device, non_blocking=True)
            scalar  = scalar.to(device,  non_blocking=True)
            labels  = labels.to(device,  non_blocking=True)

            optimizer.zero_grad()
            logits = model(spatial, scalar)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.detach() * len(labels)

        torch.cuda.synchronize()
        epoch_secs = time.perf_counter() - t0
        train_loss = train_loss.item() / len(train_loader.dataset)

        # ---- Validation ----------------------------------------------
        val_auc, val_mse, val_loss = _evaluate(model, val_loader, criterion, device)
        scheduler.step(val_auc)

        _log(
            f"Epoch {epoch:>3}/{NUM_EPOCHS}  |  "
            f"Train Loss: {train_loss:.4f}  |  "
            f"Val Loss: {val_loss:.4f}  |  "
            f"Val AUC: {val_auc:.4f}  |  "
            f"Val MSE: {val_mse:.4f}  |  "
            f"Time: {epoch_secs:.1f}s"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            _log(f"  -> Saved best model  (AUC {val_auc:.4f})")

    _log(f"\nTraining complete.  Best Val AUC: {best_val_auc:.4f}")

    # Reload best weights before returning
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

    if test_loader is not None:
        _log("\nRunning final evaluation on held-out test set...")
        probs, labels = _get_predictions(model, test_loader, device)
        metrics = compute_metrics(labels, probs)
        print_metrics(metrics, "Test Set (v2 CNN)")
        save_metrics(metrics, METRICS_PATH)

    return model


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def _evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float, float]:
    """Return (ROC AUC, MSE, mean loss) over the loader."""
    model.eval()
    all_probs:  list[float] = []
    all_labels: list[float] = []
    total_loss = 0.0

    with torch.no_grad():
        for spatial, scalar, labels in loader:
            spatial = spatial.to(device, non_blocking=True)
            scalar  = scalar.to(device,  non_blocking=True)
            labels  = labels.to(device,  non_blocking=True)

            logits = model(spatial, scalar)
            loss   = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    all_probs  = np.array(all_probs,  dtype=np.float64)
    all_labels = np.array(all_labels, dtype=np.float64)

    auc       = roc_auc_score(all_labels, all_probs)
    mse       = np.mean((all_labels - all_probs) ** 2)
    mean_loss = total_loss / len(loader.dataset)

    return auc, mse, mean_loss


def _get_predictions(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (probs, labels) arrays for the full loader — used for final test evaluation."""
    model.eval()
    all_probs:  list[float] = []
    all_labels: list[float] = []

    with torch.no_grad():
        for spatial, scalar, labels in loader:
            spatial = spatial.to(device, non_blocking=True)
            scalar  = scalar.to(device,  non_blocking=True)
            logits  = model(spatial, scalar)
            probs   = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    return (
        np.array(all_probs,  dtype=np.float64),
        np.array(all_labels, dtype=np.float64),
    )
