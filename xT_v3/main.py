"""
xT_v3 — Two-stage Spatial Expected Threat (SxT) model.

Stage 1: Frozen xT_v2 CNN  (position-based threat estimate)
Stage 2: Transformer cross-attention over 360 freeze-frame player tokens

Usage
-----
  python main.py --build              # Encode 360-compatible matches (freeze-frame events only)
  python main.py --build --limit 20   # Quick test: encode only 20 matches
  python main.py --build --force      # Re-encode even if .npz already exists
  python main.py --train              # Train Stage 2 (Stage 1 weights must exist)
  python main.py --train --seed 0     # Train with a specific random seed (default 42)
  python main.py --ablation mlp       # Train ablation: mean-pool MLP instead of cross-attention
  python main.py --ablation ball-only # Train ablation: ball features only (no player tokens)
  python main.py --calibrate          # Fit temperature scaling on val set, eval calibrated V3
  python main.py --save-preds         # Save test-set predictions for bootstrap CI
  python main.py --visualize          # Generate heatmaps from best_model.pt
  python main.py --all                # Build → Train in sequence
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from config import (
    BEST_MODEL_PATH_V3,
    BEST_MODEL_PATH_V3_BALL,
    BEST_MODEL_PATH_V3_MLP,
    METRICS_PATH_V3,
    METRICS_PATH_V3_BALL,
    METRICS_PATH_V3_CALIBRATED,
    METRICS_PATH_V3_MLP,
    TEMP_PATH_V3,
    V2_BEST_MODEL_PATH,
)


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


def _load_stage1(device: torch.device):
    import importlib.util
    _V2_DIR = os.path.join(_HERE, '..', 'xT_v2')
    _spec = importlib.util.spec_from_file_location("_v2model", os.path.join(_V2_DIR, "model.py"))
    _mod  = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
    XTModel = _mod.XTModel
    if not os.path.exists(V2_BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Stage 1 checkpoint not found at {V2_BEST_MODEL_PATH}.\n"
            "Train xT_v2 first:  cd ../xT_v2 && python main.py --train"
        )
    stage1 = XTModel().to(device)
    stage1.load_state_dict(torch.load(V2_BEST_MODEL_PATH, map_location=device))
    stage1.eval()
    print(f"Stage 1 loaded from {V2_BEST_MODEL_PATH}")
    return stage1


def _get_v3builder():
    """Load DatasetBuilderV3 via explicit file path to avoid sys.path collisions with xT_v2."""
    import importlib.util
    _spec_b = importlib.util.spec_from_file_location("_v3builder", os.path.join(_HERE, "builder.py"))
    _v3b    = importlib.util.module_from_spec(_spec_b); _spec_b.loader.exec_module(_v3b)
    return _v3b.DatasetBuilderV3()


def run_build(limit: int | None, force: bool) -> None:
    builder = _get_v3builder()
    builder.build(limit=limit, force_rebuild=force)


def run_train(device: torch.device, seed: int = 42) -> None:
    import importlib.util

    _set_seed(seed)

    # Load v3 model by file path
    _spec_v3m = importlib.util.spec_from_file_location("_v3model", os.path.join(_HERE, "model.py"))
    _v3m      = importlib.util.module_from_spec(_spec_v3m); _spec_v3m.loader.exec_module(_v3m)
    XTModelV3 = _v3m.XTModelV3

    _spec_d = importlib.util.spec_from_file_location("_v3dataset", os.path.join(_HERE, "dataset.py"))
    _v3d    = importlib.util.module_from_spec(_spec_d); _spec_d.loader.exec_module(_v3d)
    make_dataloaders_v3_from_disk = _v3d.make_dataloaders_v3_from_disk

    _spec_t = importlib.util.spec_from_file_location("_v3train", os.path.join(_HERE, "train.py"))
    _v3t    = importlib.util.module_from_spec(_spec_t); _spec_t.loader.exec_module(_v3t)
    train_v3 = _v3t.train_v3

    stage1 = _load_stage1(device)
    model  = XTModelV3(stage1).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Stage 2 trainable parameters: {trainable:,}")

    builder = _get_v3builder()

    print("\n--- Loading v3 dataset (split-aware) ---")
    train_loader, val_loader, test_loader = make_dataloaders_v3_from_disk(builder, seed=seed)

    # Determine metrics output path: use seed-specific path if non-default seed
    metrics_path = METRICS_PATH_V3 if seed == 42 else os.path.join(
        os.path.dirname(_HERE), "metrics", f"metrics_v3_seed{seed}.json"
    )
    best_ckpt = BEST_MODEL_PATH_V3 if seed == 42 else os.path.join(
        os.path.dirname(_HERE), "xT_v3", "checkpoints", f"best_model_seed{seed}.pt"
    )

    print("\n--- Training Stage 2 ---")
    train_v3(
        model, train_loader, val_loader, device,
        test_loader=test_loader,
        metrics_path=metrics_path,
        best_model_path=best_ckpt,
    )


def run_ablation_train(device: torch.device, ablation: str) -> None:
    """Train an ablation variant of Stage 2 and evaluate on the same test set."""
    import importlib.util

    _set_seed(42)

    _spec_v3m = importlib.util.spec_from_file_location("_v3model", os.path.join(_HERE, "model.py"))
    _v3m      = importlib.util.module_from_spec(_spec_v3m); _spec_v3m.loader.exec_module(_v3m)

    _spec_d = importlib.util.spec_from_file_location("_v3dataset", os.path.join(_HERE, "dataset.py"))
    _v3d    = importlib.util.module_from_spec(_spec_d); _spec_d.loader.exec_module(_v3d)
    make_dataloaders_v3_from_disk = _v3d.make_dataloaders_v3_from_disk

    _spec_t = importlib.util.spec_from_file_location("_v3train", os.path.join(_HERE, "train.py"))
    _v3t    = importlib.util.module_from_spec(_spec_t); _spec_t.loader.exec_module(_v3t)
    train_v3 = _v3t.train_v3

    stage1 = _load_stage1(device)

    if ablation == "mlp":
        model        = _v3m.XTModelV3MeanPool(stage1).to(device)
        best_ckpt    = BEST_MODEL_PATH_V3_MLP
        metrics_path = METRICS_PATH_V3_MLP
        label        = "V3 Ablation: Mean-Pool MLP (no cross-attention)"
    elif ablation == "ball-only":
        model        = _v3m.XTModelV3BallOnly(stage1).to(device)
        best_ckpt    = BEST_MODEL_PATH_V3_BALL
        metrics_path = METRICS_PATH_V3_BALL
        label        = "V3 Ablation: Ball-Only Stage 2 (no player tokens)"
    else:
        raise ValueError(f"Unknown ablation type: {ablation!r}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== {label} ===")
    print(f"Stage 2 trainable parameters: {trainable:,}")

    builder = _get_v3builder()

    print("\n--- Loading v3 dataset (split-aware) ---")
    train_loader, val_loader, test_loader = make_dataloaders_v3_from_disk(builder)

    train_v3(
        model, train_loader, val_loader, device,
        test_loader=test_loader,
        metrics_path=metrics_path,
        best_model_path=best_ckpt,
    )


def run_calibrate(device: torch.device) -> None:
    """
    Fit a temperature scalar T on the validation set by minimising NLL, then
    evaluate calibrated V3 on the test set.  Saves T to TEMP_PATH_V3 and
    calibrated metrics to METRICS_PATH_V3_CALIBRATED.
    """
    import gc
    import importlib.util
    from scipy.optimize import minimize_scalar
    from torch.utils.data import DataLoader
    from config import BATCH_SIZE_V3, NUM_WORKERS, CHECKPOINT_DIR_V3 as _CKPT_DIR

    _spec_v3m = importlib.util.spec_from_file_location("_v3model", os.path.join(_HERE, "model.py"))
    _v3m      = importlib.util.module_from_spec(_spec_v3m); _spec_v3m.loader.exec_module(_v3m)

    _spec_d = importlib.util.spec_from_file_location("_v3dataset", os.path.join(_HERE, "dataset.py"))
    _v3d    = importlib.util.module_from_spec(_spec_d); _spec_d.loader.exec_module(_v3d)
    XTDatasetV3 = _v3d.XTDatasetV3

    stage1 = _load_stage1(device)
    model  = _v3m.XTModelV3(stage1).to(device)

    if not os.path.exists(BEST_MODEL_PATH_V3):
        raise FileNotFoundError(
            f"No V3 checkpoint at {BEST_MODEL_PATH_V3}. Train V3 first."
        )
    model.load_state_dict(torch.load(BEST_MODEL_PATH_V3, map_location=device))
    model.eval()

    split_path = os.path.join(_CKPT_DIR, "split_match_ids.json")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"No split file at {split_path}. Train V3 first.")
    with open(split_path) as f:
        split = json.load(f)

    builder = _get_v3builder()

    loader_kw = dict(batch_size=BATCH_SIZE_V3, num_workers=NUM_WORKERS, pin_memory=True,
                     persistent_workers=True, prefetch_factor=4)

    print("\n--- Loading val split for calibration ---")
    vl = builder.load_split(split["val"], desc="  val")
    val_loader = DataLoader(XTDatasetV3(*vl[:5]), shuffle=False, **loader_kw)
    del vl; gc.collect()

    print("--- Loading test split ---")
    te = builder.load_split(split["test"], desc="  test")
    test_loader = DataLoader(XTDatasetV3(*te[:5]), shuffle=False, **loader_kw)
    del te; gc.collect()

    def _collect_logits(loader):
        all_logits, all_labels = [], []
        with torch.no_grad():
            for spatial_b, scalar_b, pt_b, pm_b, lbl_b in loader:
                spatial_b = spatial_b.to(device, non_blocking=True)
                scalar_b  = scalar_b.to(device,  non_blocking=True)
                pt_b      = pt_b.to(device,       non_blocking=True)
                pm_b      = pm_b.to(device,       non_blocking=True)
                logits = model(spatial_b, scalar_b, pt_b, pm_b)
                all_logits.append(logits.cpu().numpy())
                all_labels.append(lbl_b.numpy())
        return np.concatenate(all_logits), np.concatenate(all_labels)

    print("Collecting validation logits...")
    val_logits, val_labels = _collect_logits(val_loader)

    def nll(log_t):
        t = np.exp(log_t)
        p = 1.0 / (1.0 + np.exp(-val_logits / t))
        p = np.clip(p, 1e-7, 1.0 - 1e-7)
        return -np.mean(val_labels * np.log(p) + (1.0 - val_labels) * np.log(1.0 - p))

    result = minimize_scalar(nll, bounds=(-3.0, 3.0), method='bounded')
    T = float(np.exp(result.x))
    print(f"Optimal temperature T = {T:.4f}  (val NLL before: {nll(0):.6f}, after: {result.fun:.6f})")

    with open(TEMP_PATH_V3, 'w') as f:
        json.dump({"temperature": T}, f, indent=2)
    print(f"Temperature saved → {TEMP_PATH_V3}")

    # Evaluate calibrated model on test set
    _scripts = os.path.join(os.path.dirname(_HERE), 'scripts')
    if _scripts not in sys.path:
        sys.path.insert(0, _scripts)
    from evaluate import compute_metrics, print_metrics, save_metrics

    print("Collecting test logits...")
    test_logits, test_labels = _collect_logits(test_loader)
    calibrated_probs = 1.0 / (1.0 + np.exp(-test_logits / T))

    metrics = compute_metrics(test_labels, calibrated_probs)
    print_metrics(metrics, "Test Set (V3 Calibrated — temperature scaling)")
    save_metrics(metrics, METRICS_PATH_V3_CALIBRATED)


def run_save_preds(device: torch.device) -> None:
    import gc
    import importlib.util as _ilu
    from torch.utils.data import DataLoader
    from config import BATCH_SIZE_V3, NUM_WORKERS, TEST_MATCH_IDS_PATH

    if not os.path.exists(TEST_MATCH_IDS_PATH):
        raise FileNotFoundError(
            f"{TEST_MATCH_IDS_PATH} not found — train v3 first."
        )
    with open(TEST_MATCH_IDS_PATH) as f:
        test_ids = json.load(f)

    # Load v2 model (Stage 1)
    _sv2m = _ilu.spec_from_file_location("_v2model", os.path.join(_HERE, '..', 'xT_v2', 'model.py'))
    _mv2m = _ilu.module_from_spec(_sv2m); _sv2m.loader.exec_module(_mv2m)
    stage1 = _mv2m.XTModel().to(device)
    stage1.load_state_dict(torch.load(V2_BEST_MODEL_PATH, map_location=device))

    # Load v3 model (Stage 2)
    _sv3m = _ilu.spec_from_file_location("_v3model", os.path.join(_HERE, 'model.py'))
    _mv3m = _ilu.module_from_spec(_sv3m); _sv3m.loader.exec_module(_mv3m)
    model = _mv3m.XTModelV3(stage1).to(device)
    if not os.path.exists(BEST_MODEL_PATH_V3):
        raise FileNotFoundError(f"No checkpoint at {BEST_MODEL_PATH_V3}.")
    model.load_state_dict(torch.load(BEST_MODEL_PATH_V3, map_location=device))
    model.eval()

    # Load only the test split from disk (no need to touch train/val data)
    builder = _get_v3builder()

    _svd = _ilu.spec_from_file_location("_v3dataset", os.path.join(_HERE, 'dataset.py'))
    _mvd = _ilu.module_from_spec(_svd); _svd.loader.exec_module(_mvd)
    XTDatasetV3 = _mvd.XTDatasetV3

    print("\n--- Loading test split ---")
    te = builder.load_split(test_ids, desc="  test")
    spatial, scalar, player_tokens, player_mask, labels, match_ids_test = te
    test_dataset = XTDatasetV3(spatial, scalar, player_tokens, player_mask, labels)
    test_loader  = DataLoader(test_dataset, shuffle=False,
                              batch_size=BATCH_SIZE_V3, num_workers=NUM_WORKERS, pin_memory=True)
    del te; gc.collect()

    _svt = _ilu.spec_from_file_location("_v3train", os.path.join(_HERE, 'train.py'))
    _mvt = _ilu.module_from_spec(_svt); _svt.loader.exec_module(_mvt)
    probs, true_labels = _mvt._get_predictions_v3(model, test_loader, device)

    out_dir = os.path.join(_HERE, "predictions")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "preds.npz")
    np.savez(out_path, probs=probs, labels=true_labels, match_ids=match_ids_test)
    print(f"Saved → {out_path}  ({len(probs):,} events, {len(test_ids)} matches)")


def run_ablation_save_preds(device: torch.device, ablation: str) -> None:
    """Save test-set predictions for an ablation variant without retraining."""
    import gc
    import importlib.util as _ilu
    from torch.utils.data import DataLoader
    from config import BATCH_SIZE_V3, NUM_WORKERS, TEST_MATCH_IDS_PATH

    if not os.path.exists(TEST_MATCH_IDS_PATH):
        raise FileNotFoundError(
            f"{TEST_MATCH_IDS_PATH} not found — train v3 first."
        )
    with open(TEST_MATCH_IDS_PATH) as f:
        test_ids = json.load(f)

    if ablation == "mlp":
        ckpt_path = BEST_MODEL_PATH_V3_MLP
        out_name  = "preds_meanpool.npz"
    elif ablation == "ball-only":
        ckpt_path = BEST_MODEL_PATH_V3_BALL
        out_name  = "preds_ballonly.npz"
    else:
        raise ValueError(f"Unknown ablation: {ablation!r}")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}.\n"
            f"Train the ablation first:  python main.py --ablation {ablation}"
        )

    # Load Stage 1
    _sv2m = _ilu.spec_from_file_location("_v2model", os.path.join(_HERE, '..', 'xT_v2', 'model.py'))
    _mv2m = _ilu.module_from_spec(_sv2m); _sv2m.loader.exec_module(_mv2m)
    stage1 = _mv2m.XTModel().to(device)
    stage1.load_state_dict(torch.load(V2_BEST_MODEL_PATH, map_location=device))

    # Load ablation model
    _sv3m = _ilu.spec_from_file_location("_v3model", os.path.join(_HERE, 'model.py'))
    _mv3m = _ilu.module_from_spec(_sv3m); _sv3m.loader.exec_module(_mv3m)
    if ablation == "mlp":
        model = _mv3m.XTModelV3MeanPool(stage1).to(device)
    else:
        model = _mv3m.XTModelV3BallOnly(stage1).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"Loaded {ablation} checkpoint from {ckpt_path}")

    # Load test split
    builder = _get_v3builder()
    _svd = _ilu.spec_from_file_location("_v3dataset", os.path.join(_HERE, 'dataset.py'))
    _mvd = _ilu.module_from_spec(_svd); _svd.loader.exec_module(_mvd)
    XTDatasetV3 = _mvd.XTDatasetV3

    print("\n--- Loading test split ---")
    te = builder.load_split(test_ids, desc="  test")
    spatial, scalar, player_tokens, player_mask, labels, match_ids_test = te
    test_dataset = XTDatasetV3(spatial, scalar, player_tokens, player_mask, labels)
    test_loader  = DataLoader(test_dataset, shuffle=False,
                              batch_size=BATCH_SIZE_V3, num_workers=NUM_WORKERS, pin_memory=True)
    del te; gc.collect()

    _svt = _ilu.spec_from_file_location("_v3train", os.path.join(_HERE, 'train.py'))
    _mvt = _ilu.module_from_spec(_svt); _svt.loader.exec_module(_mvt)
    probs, true_labels = _mvt._get_predictions_v3(model, test_loader, device)

    out_dir  = os.path.join(_HERE, "predictions")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    np.savez(out_path, probs=probs, labels=true_labels, match_ids=match_ids_test)
    print(f"Saved → {out_path}  ({len(probs):,} events, {len(test_ids)} matches)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="xT_v3 pipeline")
    parser.add_argument("--build",      action="store_true", help="Encode dataset (freeze-frame events)")
    parser.add_argument("--train",      action="store_true", help="Train Stage 2 Transformer")
    parser.add_argument("--seed",       type=int, default=42, help="Random seed for training (default 42)")
    parser.add_argument("--ablation",   choices=["mlp", "ball-only"],
                        help="Train an ablation variant: 'mlp' (mean-pool) or 'ball-only'")
    parser.add_argument("--calibrate",  action="store_true",
                        help="Fit temperature scaling on val set; eval calibrated V3 on test set")
    parser.add_argument("--save-preds", action="store_true", help="Save test-set predictions to predictions/preds.npz for bootstrap CI")
    parser.add_argument("--visualize",  action="store_true", help="Generate heatmaps from best checkpoint")
    parser.add_argument("--all",        action="store_true", help="Run full pipeline: build → train")
    parser.add_argument("--limit",      type=int, default=None, help="Cap number of matches for --build")
    parser.add_argument("--force",      action="store_true",    help="Force re-encode in --build")
    parser.add_argument("--gpu",        type=int, default=None,
                        help="GPU index to use (e.g. --gpu 2). Defaults to CUDA_VISIBLE_DEVICES or GPU 0.")
    args = parser.parse_args()

    if not any([args.build, args.train, args.ablation, args.calibrate,
                args.save_preds, args.visualize, args.all]):
        args.all = True

    device = _get_device(gpu=args.gpu)

    if args.build or args.all:
        print("\n=== STEP 1: BUILD DATASET (v3) ===")
        run_build(limit=args.limit, force=args.force)

    if args.train or args.all:
        print("\n=== STEP 2: TRAIN STAGE 2 ===")
        run_train(device, seed=args.seed)

    if args.ablation and (args.train or args.all or not args.save_preds):
        # Skip retraining when --save-preds is the sole goal (use existing checkpoint)
        print(f"\n=== ABLATION: {args.ablation.upper()} ===")
        run_ablation_train(device, args.ablation)

    if args.calibrate:
        print("\n=== TEMPERATURE CALIBRATION ===")
        run_calibrate(device)

    if args.save_preds:
        if args.ablation:
            print(f"\n=== SAVE ABLATION PREDICTIONS ({args.ablation}) ===")
            run_ablation_save_preds(device, args.ablation)
        else:
            print("\n=== SAVE PREDICTIONS (for bootstrap CI) ===")
            run_save_preds(device)

    if args.visualize:
        print("\n=== VISUALIZE (v3) ===")
        import importlib.util as _ilu
        _sv2 = _ilu.spec_from_file_location("_v2model", os.path.join(os.path.dirname(__file__), '..', 'xT_v2', 'model.py'))
        _mv2 = _ilu.module_from_spec(_sv2); _sv2.loader.exec_module(_mv2)
        _sv3 = _ilu.spec_from_file_location("_v3model", os.path.join(os.path.dirname(__file__), 'model.py'))
        _mv3 = _ilu.module_from_spec(_sv3); _sv3.loader.exec_module(_mv3)

        stage1 = _mv2.XTModel().to(device)
        stage1.load_state_dict(torch.load(V2_BEST_MODEL_PATH, map_location=device))
        model  = _mv3.XTModelV3(stage1).to(device)
        model.load_state_dict(torch.load(BEST_MODEL_PATH_V3, map_location=device))
        model.eval()

        _sv = _ilu.spec_from_file_location("_v3viz", os.path.join(os.path.dirname(__file__), 'visualize.py'))
        _vz = _ilu.module_from_spec(_sv); _sv.loader.exec_module(_vz)
        _vz.generate_heatmap(model, device)
        _vz.generate_all_scenarios(model, device)
