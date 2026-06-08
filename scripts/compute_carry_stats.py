"""
Compute empirical carry-distance statistics from the 292 training matches.

Reads carry events from the StatsBomb open dataset (via StatsBombPy), filters
to the training split only, and saves summary statistics to
metrics/carry_stats.json.

Run this script once before starting app.py or xT_v2/visualize.py so that the
carry end-position inference uses a data-driven median instead of a hardcoded value.

Usage:
    python scripts/compute_carry_stats.py
"""

import json
import os
import warnings

import numpy as np

# Suppress StatsBombPy's "open data access only" credential warning
warnings.filterwarnings("ignore", message="credentials were not supplied")

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(HERE, "..")

SPLIT_FILE  = os.path.join(REPO_ROOT, "xT_v3", "checkpoints", "split_match_ids.json")
OUTPUT_FILE = os.path.join(REPO_ROOT, "metrics", "carry_stats.json")


def main():
    # Load training match IDs (use only train split to avoid data leakage)
    with open(SPLIT_FILE) as f:
        split = json.load(f)
    train_ids = sorted(int(m) for m in split["train"])
    print(f"Training matches: {len(train_ids)}")

    try:
        from statsbombpy import sb
    except ImportError:
        raise ImportError(
            "statsbombpy is required. Install with:  pip install statsbombpy"
        )

    distances = []
    skipped   = 0

    for i, mid in enumerate(train_ids):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Match {i+1}/{len(train_ids)} (id={mid}) …")
        try:
            events = sb.events(match_id=mid, fmt="dataframe")
        except Exception as e:
            print(f"    [SKIP] {e}")
            skipped += 1
            continue

        if "carry_end_location" not in events.columns:
            continue

        carry_mask = (events["type"] == "Carry") & events["carry_end_location"].notna()
        carries    = events.loc[carry_mask]

        for _, row in carries.iterrows():
            start = row["location"]
            end   = row["carry_end_location"]
            if start is None or end is None:
                continue
            dx   = end[0] - start[0]
            dy   = end[1] - start[1]
            dist = float(np.sqrt(dx**2 + dy**2))
            if dist > 0:
                distances.append(dist)

    if not distances:
        raise ValueError(
            "No carry distances found — check that training matches are accessible "
            "via StatsBombPy."
        )

    distances = np.array(distances)
    stats = {
        "n_carries":       int(len(distances)),
        "n_matches_used":  int(len(train_ids) - skipped),
        "n_matches_skipped": int(skipped),
        "median_distance": float(np.median(distances)),
        "mean_distance":   float(np.mean(distances)),
        "p25_distance":    float(np.percentile(distances, 25)),
        "p75_distance":    float(np.percentile(distances, 75)),
        "unit":            "StatsBomb coordinate units (~1 unit per metre on a 120x80 pitch)",
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nCarry distance statistics from {stats['n_matches_used']} matches "
          f"({stats['n_carries']:,} carry events):")
    print(f"  Median : {stats['median_distance']:.2f} units")
    print(f"  Mean   : {stats['mean_distance']:.2f} units")
    print(f"  P25–P75: {stats['p25_distance']:.2f} – {stats['p75_distance']:.2f} units")
    print(f"\nSaved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
