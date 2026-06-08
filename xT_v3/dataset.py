import gc
import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))

_spec  = importlib.util.spec_from_file_location("_v3config", os.path.join(_HERE, "config.py"))
_v3cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v3cfg)

TRAIN_SPLIT         = _v3cfg.TRAIN_SPLIT
VAL_SPLIT           = _v3cfg.VAL_SPLIT
BATCH_SIZE_V3       = _v3cfg.BATCH_SIZE_V3
NUM_WORKERS         = _v3cfg.NUM_WORKERS
MAX_PLAYERS         = _v3cfg.MAX_PLAYERS
PLAYER_DIM          = _v3cfg.PLAYER_DIM
DATA_DIR_V3         = _v3cfg.DATA_DIR_V3
CHECKPOINT_DIR_V3   = _v3cfg.CHECKPOINT_DIR_V3
TEST_MATCH_IDS_PATH = _v3cfg.TEST_MATCH_IDS_PATH


class XTDatasetV3(Dataset):
    """
    PyTorch Dataset for the two-stage xT_v3 model.

    Each item is a tuple of:
        spatial        : (NUM_CHANNELS, GRID_H, GRID_W)    float32
        scalar         : (SCALAR_DIM,)                     float32
        player_tokens  : (MAX_PLAYERS, PLAYER_DIM)          float32
        player_mask    : (MAX_PLAYERS,)                     bool
        label          : ()                                float32
    """

    def __init__(
        self,
        spatial:       np.ndarray,   # (N, C, H, W)
        scalar:        np.ndarray,   # (N, SCALAR_DIM)
        player_tokens: np.ndarray,   # (N, MAX_PLAYERS, PLAYER_DIM)
        player_mask:   np.ndarray,   # (N, MAX_PLAYERS) bool
        labels:        np.ndarray,   # (N,)
    ):
        self.spatial       = torch.from_numpy(spatial).float()
        self.scalar        = torch.from_numpy(scalar).float()
        self.player_tokens = torch.from_numpy(player_tokens).float()
        self.player_mask   = torch.from_numpy(player_mask.astype(bool))
        self.labels        = torch.from_numpy(labels).float()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            self.spatial[idx],
            self.scalar[idx],
            self.player_tokens[idx],
            self.player_mask[idx],
            self.labels[idx],
        )


def make_dataloaders_v3(
    spatial:       np.ndarray,
    scalar:        np.ndarray,
    player_tokens: np.ndarray,
    player_mask:   np.ndarray,
    labels:        np.ndarray,
    match_ids:     np.ndarray,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split by match ID (no leakage) and return train / val / test DataLoaders.

    Uses the same seed=42 shuffle and 70/15/15 split as v1 and v2 for
    consistent cross-version comparison.  Saves test match IDs to
    TEST_MATCH_IDS_PATH so xT_v2 can re-scope its own test evaluation to
    the same freeze-frame event population.
    """
    unique_matches = np.unique(match_ids)
    rng = np.random.default_rng(seed=42)
    rng.shuffle(unique_matches)

    train_end = int(len(unique_matches) * TRAIN_SPLIT)
    val_end   = train_end + int(len(unique_matches) * VAL_SPLIT)

    train_set = set(unique_matches[:train_end].tolist())
    val_set   = set(unique_matches[train_end:val_end].tolist())
    test_set  = set(unique_matches[val_end:].tolist())

    train_mask = np.array([m in train_set for m in match_ids])
    val_mask   = np.array([m in val_set   for m in match_ids])
    test_mask  = np.array([m in test_set  for m in match_ids])

    print(
        f"Train: {train_mask.sum():,} events ({len(train_set)} matches)  |  "
        f"Val: {val_mask.sum():,} events ({len(val_set)} matches)  |  "
        f"Test: {test_mask.sum():,} events ({len(test_set)} matches)"
    )

    # Persist the full split so v2 can use identical train/val/test matches.
    os.makedirs(CHECKPOINT_DIR_V3, exist_ok=True)
    with open(TEST_MATCH_IDS_PATH, "w") as f:
        json.dump(sorted(test_set), f)
    print(f"Test match IDs saved → {TEST_MATCH_IDS_PATH}")

    split_path = os.path.join(CHECKPOINT_DIR_V3, "split_match_ids.json")
    with open(split_path, "w") as f:
        json.dump({"train": sorted(train_set), "val": sorted(val_set), "test": sorted(test_set)}, f)
    print(f"Full split saved → {split_path}")

    loader_kwargs = dict(
        batch_size=BATCH_SIZE_V3,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    def _make(mask):
        return XTDatasetV3(
            spatial[mask], scalar[mask],
            player_tokens[mask], player_mask[mask], labels[mask],
        )

    train_loader = DataLoader(_make(train_mask), shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(_make(val_mask),   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(_make(test_mask),  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


def make_dataloaders_v3_from_disk(
    builder,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Memory-efficient alternative to make_dataloaders_v3.

    Loads each split from disk sequentially so peak RAM ≈ size of the train
    split (~50 GB) rather than the full dataset (~140 GB from np.concatenate).
    The split is derived from filenames on first call (no data loading) then
    saved; subsequent calls reuse it for reproducibility.
    """
    split_path = os.path.join(CHECKPOINT_DIR_V3, "split_match_ids.json")

    if os.path.exists(split_path):
        with open(split_path) as f:
            split = json.load(f)
        print(f"Split loaded from {split_path}")
    else:
        files   = sorted(f for f in os.listdir(DATA_DIR_V3) if f.endswith('.npz'))
        all_ids = np.unique([int(f.split('.')[0]) for f in files])
        rng     = np.random.default_rng(seed=seed)
        rng.shuffle(all_ids)
        train_end = int(len(all_ids) * TRAIN_SPLIT)
        val_end   = train_end + int(len(all_ids) * VAL_SPLIT)
        split = {
            "train": sorted(int(x) for x in all_ids[:train_end]),
            "val":   sorted(int(x) for x in all_ids[train_end:val_end]),
            "test":  sorted(int(x) for x in all_ids[val_end:]),
        }
        os.makedirs(CHECKPOINT_DIR_V3, exist_ok=True)
        with open(split_path, "w") as f:
            json.dump(split, f)
        with open(TEST_MATCH_IDS_PATH, "w") as f:
            json.dump(split["test"], f)
        print(f"Split saved → {split_path}")

    print(
        f"Split: {len(split['train'])} train  |  "
        f"{len(split['val'])} val  |  "
        f"{len(split['test'])} test  matches"
    )

    loader_kwargs = dict(
        batch_size=BATCH_SIZE_V3,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    print("Loading train split...")
    tr = builder.load_split(split["train"], desc="  train")
    train_loader = DataLoader(XTDatasetV3(*tr[:5]), shuffle=True,  **loader_kwargs)
    del tr; gc.collect()

    print("Loading val split...")
    vl = builder.load_split(split["val"], desc="  val")
    val_loader   = DataLoader(XTDatasetV3(*vl[:5]), shuffle=False, **loader_kwargs)
    del vl; gc.collect()

    print("Loading test split...")
    te = builder.load_split(split["test"], desc="  test")
    test_loader  = DataLoader(XTDatasetV3(*te[:5]), shuffle=False, **loader_kwargs)
    del te; gc.collect()

    print(
        f"Events: {len(train_loader.dataset):,} train  |  "
        f"{len(val_loader.dataset):,} val  |  "
        f"{len(test_loader.dataset):,} test"
    )
    return train_loader, val_loader, test_loader
