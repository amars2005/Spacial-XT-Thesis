import gc
import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import TRAIN_SPLIT, VAL_SPLIT, BATCH_SIZE, NUM_WORKERS


class XTDataset(Dataset):
    """
    PyTorch Dataset wrapping the pre-encoded spatial tensors, scalar
    feature vectors, and binary chain-goal labels.
    """

    def __init__(self, spatial: np.ndarray, scalar: np.ndarray, labels: np.ndarray):
        self.spatial = torch.from_numpy(spatial).float()
        self.scalar  = torch.from_numpy(scalar).float()
        self.labels  = torch.from_numpy(labels).float()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.spatial[idx], self.scalar[idx], self.labels[idx]


def _loader_kwargs(shuffle: bool) -> dict:
    return dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
        shuffle=shuffle,
    )


def make_dataloaders(
    spatial:   np.ndarray,
    scalar:    np.ndarray,
    labels:    np.ndarray,
    match_ids: np.ndarray,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split the dataset by match ID (to prevent data leakage) and return
    DataLoaders for train, val, and held-out test splits.

    Ratios are controlled by TRAIN_SPLIT and VAL_SPLIT in config.py
    (default 70 / 15 / 15).  The split is deterministic (seed=42).
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

    train_loader = DataLoader(
        XTDataset(spatial[train_mask], scalar[train_mask], labels[train_mask]),
        **_loader_kwargs(shuffle=True),
    )
    val_loader = DataLoader(
        XTDataset(spatial[val_mask], scalar[val_mask], labels[val_mask]),
        **_loader_kwargs(shuffle=False),
    )
    test_loader = DataLoader(
        XTDataset(spatial[test_mask], scalar[test_mask], labels[test_mask]),
        **_loader_kwargs(shuffle=False),
    )

    return train_loader, val_loader, test_loader


def make_dataloaders_fixed(
    spatial:   np.ndarray,
    scalar:    np.ndarray,
    labels:    np.ndarray,
    match_ids: np.ndarray,
    train_ids: set,
    val_ids:   set,
    test_ids:  set,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/val/test DataLoaders using pre-defined match ID sets.

    Used when a shared split file exists (xT_v3/checkpoints/split_match_ids.json)
    so that v2 and v3 train, validate, and test on exactly the same matches.
    """
    train_mask = np.array([m in train_ids for m in match_ids])
    val_mask   = np.array([m in val_ids   for m in match_ids])
    test_mask  = np.array([m in test_ids  for m in match_ids])

    print(
        f"Train: {train_mask.sum():,} events ({len(train_ids)} matches)  |  "
        f"Val: {val_mask.sum():,} events ({len(val_ids)} matches)  |  "
        f"Test: {test_mask.sum():,} events ({len(test_ids)} matches)"
    )

    train_loader = DataLoader(
        XTDataset(spatial[train_mask], scalar[train_mask], labels[train_mask]),
        **_loader_kwargs(shuffle=True),
    )
    val_loader = DataLoader(
        XTDataset(spatial[val_mask], scalar[val_mask], labels[val_mask]),
        **_loader_kwargs(shuffle=False),
    )
    test_loader = DataLoader(
        XTDataset(spatial[test_mask], scalar[test_mask], labels[test_mask]),
        **_loader_kwargs(shuffle=False),
    )

    return train_loader, val_loader, test_loader


def make_test_loader(
    spatial:          np.ndarray,
    scalar:           np.ndarray,
    labels:           np.ndarray,
    match_ids:        np.ndarray,
    target_match_ids: set,
) -> DataLoader:
    """
    Build a test DataLoader restricted to a specific set of match IDs.
    """
    mask = np.array([m in target_match_ids for m in match_ids])
    print(
        f"Test (freeze-frame scope): {mask.sum():,} events "
        f"({len(target_match_ids)} matches)"
    )
    return DataLoader(
        XTDataset(spatial[mask], scalar[mask], labels[mask]),
        **_loader_kwargs(shuffle=False),
    )


def make_dataloaders_from_disk(
    builder,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Memory-efficient fast loading that mirrors xT_v3's make_dataloaders_v3_from_disk.

    Derives the train/val/test split from filenames (no data loaded in phase 1),
    saves it to xT_v3/checkpoints/split_match_ids.json for cross-version alignment,
    then loads each split in parallel using two-phase pre-allocated filling.

    Workers = NUM_WORKERS (from config); DataLoaders use persistent_workers and
    prefetch_factor=4 for maximum GPU utilisation.
    """
    from config import DATA_DIR
    _v3_ckpt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'xT_v3', 'checkpoints')
    split_path = os.path.join(_v3_ckpt, 'split_match_ids.json')

    if os.path.exists(split_path):
        with open(split_path) as f:
            split = json.load(f)
        print(f"Split loaded from {split_path}")
    else:
        files   = sorted(f for f in os.listdir(DATA_DIR) if f.endswith('.npz'))
        all_ids = np.unique([int(f[:-4]) for f in files])
        rng     = np.random.default_rng(seed=seed)
        rng.shuffle(all_ids)
        train_end = int(len(all_ids) * TRAIN_SPLIT)
        val_end   = train_end + int(len(all_ids) * VAL_SPLIT)
        split = {
            "train": sorted(int(x) for x in all_ids[:train_end]),
            "val":   sorted(int(x) for x in all_ids[train_end:val_end]),
            "test":  sorted(int(x) for x in all_ids[val_end:]),
        }
        os.makedirs(_v3_ckpt, exist_ok=True)
        with open(split_path, "w") as f:
            json.dump(split, f)
        print(f"Split saved → {split_path}")

    print(
        f"Split: {len(split['train'])} train  |  "
        f"{len(split['val'])} val  |  "
        f"{len(split['test'])} test  matches"
    )

    print("Loading train split...")
    tr = builder.load_split(split["train"], desc="  train")
    train_loader = DataLoader(XTDataset(*tr[:3]), **_loader_kwargs(shuffle=True))
    del tr; gc.collect()

    print("Loading val split...")
    vl = builder.load_split(split["val"], desc="  val")
    val_loader = DataLoader(XTDataset(*vl[:3]), **_loader_kwargs(shuffle=False))
    del vl; gc.collect()

    print("Loading test split...")
    te = builder.load_split(split["test"], desc="  test")
    test_loader = DataLoader(XTDataset(*te[:3]), **_loader_kwargs(shuffle=False))
    del te; gc.collect()

    print(
        f"Events: {len(train_loader.dataset):,} train  |  "
        f"{len(val_loader.dataset):,} val  |  "
        f"{len(test_loader.dataset):,} test"
    )
    return train_loader, val_loader, test_loader
