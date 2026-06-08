"""
Run this once after V3 training finishes to save the split_match_ids.json
that V2 will use to align its train/val/test to V3's exact splits.
"""
import json, os
import numpy as np

_HERE        = os.path.dirname(os.path.abspath(__file__))
V3_DATA_DIR  = os.path.join(_HERE, "..", "xT_v3", "data", "matches")
SPLIT_PATH   = os.path.join(_HERE, "..", "xT_v3", "checkpoints", "split_match_ids.json")
TRAIN_SPLIT  = 0.70
VAL_SPLIT    = 0.15

match_ids = []
for fname in os.listdir(V3_DATA_DIR):
    if fname.endswith(".npz"):
        d = np.load(os.path.join(V3_DATA_DIR, fname))
        match_ids.append(int(d["match_id"][0]))

unique_matches = np.array(sorted(set(match_ids)))
rng = np.random.default_rng(seed=42)
rng.shuffle(unique_matches)

train_end = int(len(unique_matches) * TRAIN_SPLIT)
val_end   = train_end + int(len(unique_matches) * VAL_SPLIT)

train_set = sorted(unique_matches[:train_end].tolist())
val_set   = sorted(unique_matches[train_end:val_end].tolist())
test_set  = sorted(unique_matches[val_end:].tolist())

os.makedirs(os.path.dirname(SPLIT_PATH), exist_ok=True)
with open(SPLIT_PATH, "w") as f:
    json.dump({"train": train_set, "val": val_set, "test": test_set}, f, indent=2)

print(f"Saved split: {len(train_set)} train / {len(val_set)} val / {len(test_set)} test matches")
print(f"→ {SPLIT_PATH}")
