"""
xT_v2 — Spatial Expected Threat (SxT) model with StatsBomb 360 data and CNN.

Usage
-----
  python main.py --build              # Encode all 360-compatible matches to disk
  python main.py --build --limit 20   # Quick test: encode only 20 matches
  python main.py --build --force      # Re-encode even if .npz already exists
  python main.py --train              # Train the CNN on the built dataset
  python main.py --evaluate           # Evaluate checkpoint (aligned to v3 test IDs)
  python main.py --save-preds         # Save test-set predictions for bootstrap CI
  python main.py --visualize          # Generate xT heatmap from best checkpoint
  python main.py --all                # Build → Train → Visualize in sequence
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch

from config import BEST_MODEL_PATH


def _get_device(gpu: int | None = None) -> torch.device:
    if not torch.cuda.is_available():
        print("Using device: cpu")
        return torch.device("cpu")
    idx = gpu if gpu is not None else 0
    torch.cuda.set_device(idx)
    device = torch.device(f"cuda:{idx}")
    props = torch.cuda.get_device_properties(idx)
    free_gb = (props.total_memory - torch.cuda.memory_reserved(idx)) / 1e9
    print(f"Using device: cuda:{idx}  ({props.name},  {free_gb:.1f} GB free)")
    return device


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")


def run_build(limit: int | None, force: bool) -> None:
    from builder import DatasetBuilder
    builder = DatasetBuilder()
    builder.build(limit=limit, force_rebuild=force)


def run_train(device: torch.device, seed: int = 42) -> None:
    from builder import DatasetBuilder
    from dataset import make_dataloaders_from_disk
    from model import XTModel
    from train import train

    _set_seed(seed)

    builder = DatasetBuilder()

    print("\n--- Loading dataset (parallel, split-aware) ---")
    train_loader, val_loader, test_loader = make_dataloaders_from_disk(builder, seed=seed)

    goal_rate = train_loader.dataset.labels.mean().item()
    print(f"Train goal rate: {goal_rate:.4f}")

    print("\n--- Building model ---")
    model = XTModel().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    print("\n--- Training ---")
    train(model, train_loader, val_loader, device, test_loader=test_loader)


def run_evaluate(device: torch.device) -> None:
    from builder import DatasetBuilder
    from dataset import make_test_loader
    from model import XTModel
    from train import _get_predictions
    _scripts = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts')
    if _scripts not in sys.path:
        sys.path.insert(0, _scripts)
    from evaluate import compute_metrics, print_metrics, save_metrics
    from config import METRICS_PATH

    _v3_ids_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "xT_v3", "checkpoints", "test_match_ids.json",
    )
    if not os.path.exists(_v3_ids_path):
        raise FileNotFoundError(
            "xT_v3/checkpoints/test_match_ids.json not found — train v3 first."
        )
    with open(_v3_ids_path) as f:
        v3_test_ids = set(json.load(f))
    print(f"Loaded {len(v3_test_ids)} v3 test match IDs.")

    print("\n--- Loading v2 dataset ---")
    builder = DatasetBuilder()
    spatial, scalar, labels, match_ids = builder.load_all()

    test_loader = make_test_loader(spatial, scalar, labels, match_ids, v3_test_ids)

    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"No checkpoint at {BEST_MODEL_PATH}.")
    model = XTModel().to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()

    probs, true_labels = _get_predictions(model, test_loader, device)
    metrics = compute_metrics(true_labels, probs)
    print_metrics(metrics, "Test Set (v2 CNN — aligned to v3 test IDs)")
    save_metrics(metrics, METRICS_PATH)


def run_save_preds(device: torch.device) -> None:
    import numpy as np
    from builder import DatasetBuilder
    from dataset import make_test_loader
    from model import XTModel
    from train import _get_predictions
    from config import BEST_MODEL_PATH

    _v3_ids_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "xT_v3", "checkpoints", "test_match_ids.json",
    )
    if not os.path.exists(_v3_ids_path):
        raise FileNotFoundError(
            "xT_v3/checkpoints/test_match_ids.json not found — train v3 first."
        )
    with open(_v3_ids_path) as f:
        v3_test_ids = set(json.load(f))

    builder = DatasetBuilder()
    spatial, scalar, labels, match_ids = builder.load_all()

    mask = np.array([m in v3_test_ids for m in match_ids])
    match_ids_test = match_ids[mask]

    test_loader = make_test_loader(spatial, scalar, labels, match_ids, v3_test_ids)

    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"No checkpoint at {BEST_MODEL_PATH}.")
    model = XTModel().to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()

    probs, true_labels = _get_predictions(model, test_loader, device)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "preds.npz")
    np.savez(out_path, probs=probs, labels=true_labels, match_ids=match_ids_test)
    print(f"Saved → {out_path}  ({len(probs):,} events, {len(v3_test_ids)} matches)")


def run_visualize(device: torch.device) -> None:
    import os
    from model import XTModel
    from visualize import generate_heatmap_weighted, generate_all_scenarios_weighted

    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"No checkpoint found at {BEST_MODEL_PATH}.\n"
            "Run  python main.py --train  first."
        )

    model = XTModel().to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()

    print("\n--- Generating weighted heatmap ---")
    generate_heatmap_weighted(model, device)
    print("\n--- Generating weighted scenario comparison ---")
    generate_all_scenarios_weighted(model, device)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="xT_v2 pipeline")
    parser.add_argument("--build",      action="store_true", help="Encode dataset from StatsBomb API")
    parser.add_argument("--train",      action="store_true", help="Train the CNN model")
    parser.add_argument("--evaluate",   action="store_true", help="Evaluate saved checkpoint on v3 test IDs (no retraining)")
    parser.add_argument("--save-preds", action="store_true", help="Save test-set predictions to predictions/preds.npz for bootstrap CI")
    parser.add_argument("--visualize",  action="store_true", help="Generate xT heatmap from checkpoint")
    parser.add_argument("--all",        action="store_true", help="Run full pipeline: build → train → visualize")
    parser.add_argument("--limit",      type=int, default=None, help="Cap number of matches for --build")
    parser.add_argument("--force",      action="store_true",    help="Force re-encode in --build")
    parser.add_argument("--seed",       type=int, default=42,   help="Random seed for training (default 42)")
    parser.add_argument("--gpu",        type=int, default=None,
                        help="GPU index to use (e.g. --gpu 2). Defaults to CUDA_VISIBLE_DEVICES or GPU 0.")
    args = parser.parse_args()

    # Default to --all if no flags given
    if not any([args.build, args.train, args.evaluate, args.save_preds, args.visualize, args.all]):
        args.all = True

    device = _get_device(gpu=args.gpu)

    if args.build or args.all:
        print("\n=== STEP 1: BUILD DATASET ===")
        run_build(limit=args.limit, force=args.force)

    if args.train or args.all:
        print("\n=== STEP 2: TRAIN MODEL ===")
        run_train(device, seed=args.seed)

    if args.evaluate:
        print("\n=== EVALUATE (aligned to v3 test IDs) ===")
        run_evaluate(device)

    if args.save_preds:
        print("\n=== SAVE PREDICTIONS (for bootstrap CI) ===")
        run_save_preds(device)

    if args.visualize or args.all:
        print("\n=== STEP 3: VISUALIZE ===")
        run_visualize(device)
