"""
Bootstrap confidence intervals and significance test for v2 vs v3.

PREREQUISITES
-------------
This script requires saved per-event prediction files (probabilities and labels)
for Versions 2 and 3, indexed by match.  They are NOT committed to the repo by
default.  Generate them by running the respective evaluate scripts with the
--save-preds flag:

    cd xT_v2 && python evaluate.py --save-preds
    cd xT_v3 && python evaluate.py --save-preds

Each script should write:
    xT_v2/predictions/preds.npz   — keys: 'probs' (N,), 'labels' (N,), 'match_ids' (N,)
    xT_v3/predictions/preds.npz   — same format

'match_ids' must contain the integer match ID for each event row so that
bootstrap resampling can be done at the match level (not the event level).

HOW IT WORKS
------------
Bootstrap at the match level (resample 64 test matches with replacement, 10 000
iterations).  For each bootstrap sample we compute ROC AUC, Brier Score, and
Log Loss for both v2 and v3, giving empirical 95 % CIs and a paired p-value for
the v2→v3 AUC difference.
"""

import argparse
import os
import sys
import json
import pathlib
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
V1_PREDS_PATH = os.path.join(HERE, "..", "xT_v1", "predictions", "preds.npz")
V2_PREDS_PATH = os.path.join(HERE, "..", "xT_v2", "predictions", "preds.npz")
V3_PREDS_PATH = os.path.join(HERE, "..", "xT_v3", "predictions", "preds.npz")

N_BOOTSTRAP = 2_000
RNG_SEED     = 42
CI_LEVEL     = 0.95


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_preds(path: str):
    """Load predictions npz; return (probs, labels, match_ids) as numpy arrays."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Prediction file not found: {path}\n"
            "Run the relevant evaluate.py --save-preds to generate it."
        )
    data = np.load(path)
    return data["probs"], data["labels"], data["match_ids"]


def compute_metrics(probs: np.ndarray, labels: np.ndarray):
    """Return (roc_auc, brier, log_loss) — fast numpy-only implementations."""
    y = (labels > 0).astype(np.float64)
    p = probs.astype(np.float64)

    # ROC AUC via trapezoid rule (avoids sklearn sort overhead)
    order = np.argsort(-p)
    y_s   = y[order]
    npos  = y_s.sum()
    nneg  = len(y_s) - npos
    tp    = np.cumsum(y_s)
    fp    = np.cumsum(1.0 - y_s)
    tpr   = tp / npos
    fpr   = fp / nneg
    auc   = float(np.trapz(tpr, fpr))

    brier = float(np.mean((p - y) ** 2))

    pc = np.clip(p, 1e-7, 1.0 - 1e-7)
    ll = float(-np.mean(y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc)))

    return auc, brier, ll


def _build_index(match_ids: np.ndarray):
    """Pre-compute {match_id: array_of_row_indices} so bootstrap loops are O(1) per match."""
    index = {}
    for m in np.unique(match_ids):
        index[m] = np.where(match_ids == m)[0]
    return index


def bootstrap(probs_v2, labels_v2, match_ids_v2,
              probs_v3, labels_v3, match_ids_v3,
              n_iter=N_BOOTSTRAP, seed=RNG_SEED):
    """
    Paired bootstrap over test matches.

    Returns dict with arrays of length n_iter for each metric and version,
    plus an array of paired AUC differences (v3 - v2).
    """
    rng = np.random.default_rng(seed)

    matches = np.unique(match_ids_v2)
    assert np.array_equal(np.sort(matches), np.sort(np.unique(match_ids_v3))), \
        "v2 and v3 prediction files have different match sets."

    # Pre-build index to avoid 64 full-array scans per bootstrap iteration
    idx_v2_by_match = _build_index(match_ids_v2)
    idx_v3_by_match = _build_index(match_ids_v3)

    results = {
        "auc_v2": [], "brier_v2": [], "ll_v2": [],
        "auc_v3": [], "brier_v3": [], "ll_v3": [],
        "auc_diff": [],
    }

    for _ in range(n_iter):
        sampled = rng.choice(matches, size=len(matches), replace=True)

        idx_v2 = np.concatenate([idx_v2_by_match[m] for m in sampled])
        idx_v3 = np.concatenate([idx_v3_by_match[m] for m in sampled])

        a2, b2, l2 = compute_metrics(probs_v2[idx_v2], labels_v2[idx_v2])
        a3, b3, l3 = compute_metrics(probs_v3[idx_v3], labels_v3[idx_v3])

        results["auc_v2"].append(a2);   results["brier_v2"].append(b2); results["ll_v2"].append(l2)
        results["auc_v3"].append(a3);   results["brier_v3"].append(b3); results["ll_v3"].append(l3)
        results["auc_diff"].append(a3 - a2)

    return {k: np.array(v) for k, v in results.items()}


def bootstrap_single(probs, labels, match_ids, n_iter=N_BOOTSTRAP, seed=RNG_SEED):
    """Unpaired match-level bootstrap CIs for a single model."""
    rng = np.random.default_rng(seed)
    matches = np.unique(match_ids)
    idx_by_match = _build_index(match_ids)
    results = {"auc": [], "brier": [], "ll": []}
    for _ in range(n_iter):
        sampled = rng.choice(matches, size=len(matches), replace=True)
        idx = np.concatenate([idx_by_match[m] for m in sampled])
        a, b, l = compute_metrics(probs[idx], labels[idx])
        results["auc"].append(a)
        results["brier"].append(b)
        results["ll"].append(l)
    return {k: np.array(v) for k, v in results.items()}


def ci(arr, level=CI_LEVEL):
    """Return (lower, upper) percentile CI."""
    alpha = (1 - level) / 2
    return np.quantile(arr, alpha), np.quantile(arr, 1 - alpha)


def pvalue_greater(diff_arr: np.ndarray) -> float:
    """One-sided bootstrap p-value: P(v3 AUC > v2 AUC) under H0: diff <= 0."""
    # Shift distribution to be centred at zero under H0
    shifted = diff_arr - diff_arr.mean()
    observed = diff_arr.mean()
    return float(np.mean(shifted >= observed))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap CIs and paired significance test for v2 vs v3."
    )
    parser.add_argument(
        "--save-json", metavar="PATH", default=None,
        help="Persist all CIs and test results to this JSON file."
    )
    parser.add_argument(
        "--include-v1", action="store_true",
        help="Also compute standalone 95%% CIs for V1 (360-subset) predictions."
    )
    args = parser.parse_args()

    print("Loading predictions …")
    probs_v2, labels_v2, mids_v2 = load_preds(V2_PREDS_PATH)
    probs_v3, labels_v3, mids_v3 = load_preds(V3_PREDS_PATH)

    # Point estimates
    auc_v2, brier_v2, ll_v2 = compute_metrics(probs_v2, labels_v2)
    auc_v3, brier_v3, ll_v3 = compute_metrics(probs_v3, labels_v3)

    print(f"Running {N_BOOTSTRAP:,} bootstrap iterations (match-level) …")
    bs = bootstrap(probs_v2, labels_v2, mids_v2,
                   probs_v3, labels_v3, mids_v3)

    # CIs
    ci_auc_v2   = ci(bs["auc_v2"])
    ci_brier_v2 = ci(bs["brier_v2"])
    ci_ll_v2    = ci(bs["ll_v2"])

    ci_auc_v3   = ci(bs["auc_v3"])
    ci_brier_v3 = ci(bs["brier_v3"])
    ci_ll_v3    = ci(bs["ll_v3"])

    ci_diff = ci(bs["auc_diff"])
    p_val   = pvalue_greater(bs["auc_diff"])

    # Print table
    pct = int(CI_LEVEL * 100)
    print()
    print(f"{'':22s}  {'ROC AUC':>22s}  {'Brier Score':>22s}  {'Log Loss':>22s}")
    print("-" * 92)
    print(f"{'v2: CNN + MLP':22s}  "
          f"{auc_v2:.4f} [{ci_auc_v2[0]:.4f}, {ci_auc_v2[1]:.4f}]  "
          f"{brier_v2:.4f} [{ci_brier_v2[0]:.4f}, {ci_brier_v2[1]:.4f}]  "
          f"{ll_v2:.4f} [{ci_ll_v2[0]:.4f}, {ci_ll_v2[1]:.4f}]")
    print(f"{'v3: Transformer':22s}  "
          f"{auc_v3:.4f} [{ci_auc_v3[0]:.4f}, {ci_auc_v3[1]:.4f}]  "
          f"{brier_v3:.4f} [{ci_brier_v3[0]:.4f}, {ci_brier_v3[1]:.4f}]  "
          f"{ll_v3:.4f} [{ci_ll_v3[0]:.4f}, {ci_ll_v3[1]:.4f}]")
    print()
    print(f"v3 - v2 AUC difference: {auc_v3 - auc_v2:+.4f}")
    print(f"{pct}% CI for difference: [{ci_diff[0]:+.4f}, {ci_diff[1]:+.4f}]")
    print(f"One-sided p-value (H0: v3 AUC ≤ v2 AUC): {p_val:.4f}")
    print()
    if ci_diff[0] > 0:
        print("The CI excludes zero: the v3 AUC improvement is statistically significant.")
    else:
        print("The CI includes zero: the v3 AUC improvement is not statistically significant "
              "at the chosen level — treat the gap as indicative rather than definitive.")

    # Optional V1 standalone CIs
    v1_output = None
    if args.include_v1:
        print("\nLoading V1 predictions …")
        probs_v1, labels_v1, mids_v1 = load_preds(V1_PREDS_PATH)
        auc_v1, brier_v1, ll_v1 = compute_metrics(probs_v1, labels_v1)
        print(f"Running {N_BOOTSTRAP:,} bootstrap iterations for V1 …")
        bs_v1 = bootstrap_single(probs_v1, labels_v1, mids_v1)
        ci_auc_v1   = ci(bs_v1["auc"])
        ci_brier_v1 = ci(bs_v1["brier"])
        ci_ll_v1    = ci(bs_v1["ll"])
        print()
        print(f"{'v1: RF (360 subset)':22s}  "
              f"{auc_v1:.4f} [{ci_auc_v1[0]:.4f}, {ci_auc_v1[1]:.4f}]  "
              f"{brier_v1:.4f} [{ci_brier_v1[0]:.4f}, {ci_brier_v1[1]:.4f}]  "
              f"{ll_v1:.4f} [{ci_ll_v1[0]:.4f}, {ci_ll_v1[1]:.4f}]")
        v1_output = {
            "roc_auc":     float(auc_v1),
            "ci_auc":      [float(ci_auc_v1[0]),   float(ci_auc_v1[1])],
            "brier":       float(brier_v1),
            "ci_brier":    [float(ci_brier_v1[0]), float(ci_brier_v1[1])],
            "log_loss":    float(ll_v1),
            "ci_log_loss": [float(ci_ll_v1[0]),    float(ci_ll_v1[1])],
        }

    if args.save_json:
        output = {
            "v2": {
                "roc_auc":    float(auc_v2),
                "ci_auc":     [float(ci_auc_v2[0]),   float(ci_auc_v2[1])],
                "brier":      float(brier_v2),
                "ci_brier":   [float(ci_brier_v2[0]), float(ci_brier_v2[1])],
                "log_loss":   float(ll_v2),
                "ci_log_loss":[float(ci_ll_v2[0]),    float(ci_ll_v2[1])],
            },
            "v3": {
                "roc_auc":    float(auc_v3),
                "ci_auc":     [float(ci_auc_v3[0]),   float(ci_auc_v3[1])],
                "brier":      float(brier_v3),
                "ci_brier":   [float(ci_brier_v3[0]), float(ci_brier_v3[1])],
                "log_loss":   float(ll_v3),
                "ci_log_loss":[float(ci_ll_v3[0]),    float(ci_ll_v3[1])],
            },
            "v2_vs_v3_auc_diff": {
                "observed": float(auc_v3 - auc_v2),
                "ci_diff":  [float(ci_diff[0]), float(ci_diff[1])],
                "p_value":  float(p_val),
            },
            "n_bootstrap": N_BOOTSTRAP,
            "ci_level":    CI_LEVEL,
        }
        if v1_output is not None:
            output["v1"] = v1_output
        out_path = pathlib.Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nCIs saved → {out_path}")


if __name__ == "__main__":
    main()
