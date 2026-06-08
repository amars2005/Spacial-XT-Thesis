#!/usr/bin/env python3
"""
VAEP baseline comparison for the SxT project.

Trains a VAEP model (Decroos et al. 2019) on the same training match split
used by xT v1/v2/v3, then evaluates it on the same 64 held-out test matches.

Predictions (scores_prob) are scored against our chain_goal labels using the
identical metrics from evaluate.py, so numbers are directly comparable to
metrics/metrics_v1/v2/v3.json.

Design notes
------------
- VAEP training uses its own label (goal within next k=10 SPADL actions).
  Evaluation uses our chain_goal label (any action in a possession chain
  ending in a goal) — identical ranking task to our other models.
- Trained on split_match_ids.json "train" key only (same split as xT models).
- Evaluated on all ball-progressing SPADL actions in the 64 test matches:
  pass / cross / throw-in / freekick / corner / take-on / dribble / shot.

Usage
-----
  python scripts/baseline_vaep.py

Output
------
  metrics/metrics_vaep.json  — ROC AUC / Brier / Log Loss / Spearman
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

from socceraction.data.statsbomb import StatsBombLoader
from socceraction.spadl.statsbomb import convert_to_actions
import socceraction.spadl.config as spadlcfg
import socceraction.vaep.features as vaepfeat
import socceraction.vaep.labels as vaeplabels

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT        = os.path.dirname(_SCRIPTS_DIR)

sys.path.insert(0, _SCRIPTS_DIR)
from evaluate import compute_metrics, print_metrics, save_metrics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NB_PREV_ACTIONS  = 3   # feature context window (socceraction default)
NR_ACTIONS_LABEL = 10  # label look-ahead (standard VAEP, Decroos 2019)

SHOT_TYPE_IDS = frozenset(
    i for i, t in enumerate(spadlcfg.actiontypes)
    if t in {"shot", "shot_penalty", "shot_freekick"}
)

BALL_PROG_TYPE_IDS = frozenset(
    i for i, t in enumerate(spadlcfg.actiontypes)
    if t in {
        "pass", "cross", "throw_in",
        "freekick_crossed", "freekick_short",
        "corner_crossed", "corner_short",
        "take_on", "dribble",
        "shot", "shot_penalty", "shot_freekick",
    }
)

FEATURE_FNS = [
    vaepfeat.actiontype_onehot,
    vaepfeat.bodypart_onehot,
    vaepfeat.result_onehot,
    vaepfeat.goalscore,
    vaepfeat.startlocation,
    vaepfeat.endlocation,
    vaepfeat.movement,
    vaepfeat.time_delta,
]

METRICS_PATH = os.path.join(_ROOT, "metrics", "metrics_vaep.json")
SPLIT_PATH   = os.path.join(_ROOT, "xT_v3", "checkpoints", "split_match_ids.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_name_columns(actions: pd.DataFrame) -> pd.DataFrame:
    """Add type_name / result_name / bodypart_name (required by vaepfeat)."""
    a = actions.copy()
    a["type_name"]     = a["type_id"].apply(lambda i: spadlcfg.actiontypes[i])
    a["result_name"]   = a["result_id"].apply(lambda i: spadlcfg.results[i])
    a["bodypart_name"] = a["bodypart_id"].apply(lambda i: spadlcfg.bodyparts[i])
    return a


def assign_chain_goal(actions: pd.DataFrame) -> np.ndarray:
    """
    Assign chain_goal labels to a match's SPADL actions, mirroring the
    possession-chain logic in xT_v3/builder.py.

    A new chain starts on: period change | team change | previous action was a shot.
    A chain is labelled 1 if it contains a successful shot (goal).
    """
    period_change = actions["period_id"] != actions["period_id"].shift(1)
    team_change   = actions["team_id"]   != actions["team_id"].shift(1)
    prev_was_shot = actions["type_id"].shift(1).isin(SHOT_TYPE_IDS)

    is_new_chain = (period_change | team_change | prev_was_shot).copy()
    is_new_chain.iloc[0] = True
    chain_id = is_new_chain.cumsum()

    is_goal     = actions["type_id"].isin(SHOT_TYPE_IDS) & (actions["result_id"] == 1)
    goal_chains = set(chain_id[is_goal].tolist())

    return chain_id.isin(goal_chains).values.astype(np.float32)


def compute_vaep_features(actions: pd.DataFrame, home_team_id: int) -> pd.DataFrame:
    """Return VAEP feature matrix (one row per action) for a single match."""
    gamestates = vaepfeat.gamestates(actions, nb_prev_actions=NB_PREV_ACTIONS)
    gamestates = vaepfeat.play_left_to_right(gamestates, home_team_id)
    X = pd.concat([fn(gamestates) for fn in FEATURE_FNS], axis=1)
    return X.fillna(0)


# ---------------------------------------------------------------------------
# Game index
# ---------------------------------------------------------------------------

def build_game_index(loader: StatsBombLoader) -> pd.DataFrame:
    """
    Iterate over all StatsBomb open-data competitions, collect game metadata
    (home_team_id, competition_id, season_id), and return a DataFrame
    indexed by game_id.
    """
    comps = loader.competitions()
    rows = []
    for _, row in tqdm(comps.iterrows(), total=len(comps), desc="Indexing competitions"):
        try:
            games = loader.games(
                competition_id=int(row["competition_id"]),
                season_id=int(row["season_id"]),
            )
            for _, g in games.iterrows():
                rows.append({
                    "game_id":        int(g["game_id"]),
                    "competition_id": int(row["competition_id"]),
                    "season_id":      int(row["season_id"]),
                    "home_team_id":   int(g["home_team_id"]),
                })
        except Exception:
            pass

    return (
        pd.DataFrame(rows)
        .drop_duplicates("game_id")
        .set_index("game_id")
    )


# ---------------------------------------------------------------------------
# Per-match processing
# ---------------------------------------------------------------------------

def process_match(
    loader: StatsBombLoader,
    mid: int,
    meta: pd.Series,
) -> tuple | None:
    """
    Load one match and return:
      X            — VAEP feature matrix  (N_actions × F)
      y_scores     — VAEP training labels  (N_actions,)
      chain_goal   — our evaluation labels  (N_actions,)
      is_ball_prog — bool mask for ball-progressing actions  (N_actions,)

    Returns None on failure.
    """
    try:
        home_team_id = int(meta["home_team_id"])
        events  = loader.events(game_id=mid)
        actions = convert_to_actions(events, home_team_id=home_team_id)
        actions = _add_name_columns(actions.reset_index(drop=True))

        X          = compute_vaep_features(actions, home_team_id)
        y_scores   = vaeplabels.scores(actions, nr_actions=NR_ACTIONS_LABEL)["scores"].values
        chain_goal = assign_chain_goal(actions)
        is_ball_prog = actions["type_id"].isin(BALL_PROG_TYPE_IDS).values

        return X, y_scores, chain_goal, is_ball_prog

    except Exception as exc:
        tqdm.write(f"  [skip] match {mid}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    with open(SPLIT_PATH) as f:
        split = json.load(f)

    train_ids = split["train"]
    test_ids  = split["test"]

    loader = StatsBombLoader(getter="remote")

    print("Building game index across all StatsBomb open-data competitions...")
    game_index = build_game_index(loader)
    print(f"  {len(game_index):,} games indexed")

    known_train = [m for m in train_ids if m in game_index.index]
    known_test  = [m for m in test_ids  if m in game_index.index]
    print(f"  Train matches found: {len(known_train)}/{len(train_ids)}")
    print(f"  Test  matches found: {len(known_test)}/{len(test_ids)}")

    # -------------------------------------------------------------------
    # Build training set
    # -------------------------------------------------------------------
    print(f"\nProcessing {len(known_train)} training matches...")
    X_parts, ys_parts = [], []

    for mid in tqdm(known_train, desc="Train"):
        result = process_match(loader, mid, game_index.loc[mid])
        if result is None:
            continue
        X, y_scores, _, _ = result
        X_parts.append(X)
        ys_parts.append(y_scores)

    if not X_parts:
        print("No training data loaded. Exiting.")
        return

    X_train = pd.concat(X_parts, ignore_index=True).fillna(0)
    y_train = np.concatenate(ys_parts)
    print(f"  Training on {len(X_train):,} actions  (goal rate: {y_train.mean():.4f})")

    del X_parts, ys_parts

    # -------------------------------------------------------------------
    # Train XGBoost scoring classifier (VAEP)
    # -------------------------------------------------------------------
    print("\nTraining VAEP scoring classifier...")
    clf = XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.75,
        colsample_bytree=0.75,
        n_jobs=-1,
        eval_metric="logloss",
        verbosity=0,
    )
    clf.fit(X_train, y_train)
    print("  Done.")

    del X_train, y_train

    # -------------------------------------------------------------------
    # Evaluate on test set
    # -------------------------------------------------------------------
    print(f"\nEvaluating on {len(known_test)} test matches...")
    all_probs  = []
    all_labels = []

    for mid in tqdm(known_test, desc="Test"):
        result = process_match(loader, mid, game_index.loc[mid])
        if result is None:
            continue
        X, _, chain_goal, is_ball_prog = result

        if not is_ball_prog.any():
            continue

        X_bp          = X[is_ball_prog].fillna(0)
        chain_goal_bp = chain_goal[is_ball_prog]

        probs = clf.predict_proba(X_bp)[:, 1]
        all_probs.extend(probs.tolist())
        all_labels.extend(chain_goal_bp.tolist())

    y_true = np.array(all_labels, dtype=np.float64)
    y_prob = np.array(all_probs,  dtype=np.float64)

    print(f"\n  Evaluated on {len(y_true):,} ball-progressing test actions")
    print(f"  Chain-goal rate: {y_true.mean():.4f}")

    metrics = compute_metrics(y_true, y_prob)
    print_metrics(metrics, "VAEP Baseline (scores_prob vs chain_goal labels)")
    save_metrics(metrics, METRICS_PATH)

    print(f"\nDone.  Results → {METRICS_PATH}")


if __name__ == "__main__":
    main()
