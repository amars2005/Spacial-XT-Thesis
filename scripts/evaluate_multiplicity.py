"""
Paired bootstrap AUC tests for architectural variants vs a baseline,
with Holm-Bonferroni multiple-comparisons correction.

USAGE
-----
python scripts/evaluate_multiplicity.py \
    --baseline  xT_v3/predictions/preds.npz \
    --variants  mean_pool:xT_v3/predictions/preds_meanpool.npz \
                ball_only:xT_v3/predictions/preds_ballonly.npz \
                multi_query:xT_v3/predictions/preds_multiquery.npz \
                rich_feat:xT_v3/predictions/preds_richfeat.npz \
                unfreeze:xT_v3/predictions/preds_unfreeze.npz \
    [--save-json metrics/multiplicity.json]

Each --variants argument is a colon-separated  name:path  pair.
One-sided tests check H_0: variant AUC <= baseline AUC.
Holm-Bonferroni correction is applied over all m variants simultaneously.

PREREQUISITES
-------------
Generate prediction .npz files with --save-preds:
    cd xT_v3 && python evaluate.py --save-preds        # preds.npz (V3 full)
    cd xT_v3 && python experiments_p4.py               # preds_multiquery/richfeat/unfreeze
For ball-only and mean-pool, add --save-preds to main.py ablation runs:
    cd xT_v3 && python main.py --train --save-preds

See Section 5.3 of the report (eq:holm) for the mathematical details.
"""

import argparse
import json
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(HERE, "..")

# Import shared helpers from evaluate_ci (sibling script in same directory)
sys.path.insert(0, HERE)
from evaluate_ci import (
    load_preds,
    compute_metrics,
    _build_index,
    ci,
    pvalue_greater,
    N_BOOTSTRAP,
    RNG_SEED,
    CI_LEVEL,
)


# ---------------------------------------------------------------------------
# Single-variant paired bootstrap
# ---------------------------------------------------------------------------

def bootstrap_pair(probs_base, labels_base, mids_base,
                   probs_var,  labels_var,  mids_var,
                   n_iter=N_BOOTSTRAP, seed=RNG_SEED):
    """
    Paired match-level bootstrap comparing one variant against the baseline.

    Returns a dict:
      auc_diff  — array(n_iter): AUC(variant) - AUC(baseline) per sample
      auc_base  — array(n_iter): AUC(baseline) per sample
      auc_var   — array(n_iter): AUC(variant)  per sample
    """
    rng = np.random.default_rng(seed)
    matches = np.unique(mids_base)
    if not np.array_equal(np.sort(matches), np.sort(np.unique(mids_var))):
        raise ValueError(
            "Baseline and variant prediction files cover different test-match sets."
        )

    idx_base = _build_index(mids_base)
    idx_var  = _build_index(mids_var)

    auc_diffs, auc_bases, auc_vars = [], [], []
    for _ in range(n_iter):
        sampled  = rng.choice(matches, size=len(matches), replace=True)
        rows_b   = np.concatenate([idx_base[m] for m in sampled])
        rows_v   = np.concatenate([idx_var[m]  for m in sampled])
        ab, _, _ = compute_metrics(probs_base[rows_b], labels_base[rows_b])
        av, _, _ = compute_metrics(probs_var[rows_v],  labels_var[rows_v])
        auc_bases.append(ab)
        auc_vars.append(av)
        auc_diffs.append(av - ab)

    return {
        "auc_diff": np.array(auc_diffs),
        "auc_base": np.array(auc_bases),
        "auc_var":  np.array(auc_vars),
    }


# ---------------------------------------------------------------------------
# Holm-Bonferroni step-down correction
# ---------------------------------------------------------------------------

def holm_bonferroni(pvalues):
    """
    Apply Holm-Bonferroni correction to a list of p-values.

    Returns adjusted p-values in the *same order* as the input.
    Formula: p~_(i) = min(1, max_{j<=i} [(m - j + 1) * p_(j)])
    where indices are rank-ordered smallest-first.

    Holm (1979), Scand. J. Statist. 6(2):65-70.
    """
    m = len(pvalues)
    order = np.argsort(pvalues)
    adj = np.zeros(m)
    running_max = 0.0
    for rank, orig_idx in enumerate(order):
        corrected   = (m - rank) * pvalues[orig_idx]
        running_max = max(running_max, corrected)
        adj[orig_idx] = min(1.0, running_max)
    return adj.tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve(path):
    """Resolve a path relative to repo root if not already absolute."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(REPO_ROOT, path))


def main():
    parser = argparse.ArgumentParser(
        description="Paired bootstrap AUC tests with Holm-Bonferroni correction."
    )
    parser.add_argument(
        "--baseline", required=True, metavar="PATH",
        help="Path to baseline .npz file (relative to repo root or absolute)."
    )
    parser.add_argument(
        "--variants", nargs="+", metavar="NAME:PATH", required=True,
        help="Space-separated name:path pairs for variant prediction files."
    )
    parser.add_argument(
        "--save-json", metavar="PATH", default=None,
        help="Write full results to this JSON file (relative to repo root or absolute)."
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance level (default: 0.05)."
    )
    args = parser.parse_args()

    # parse variant name:path pairs
    variants = {}
    for item in args.variants:
        parts = item.split(":", 1)
        if len(parts) != 2:
            sys.exit(f"ERROR: each --variants entry must be  name:path  — got: {item!r}")
        variants[parts[0]] = _resolve(parts[1])

    alpha = args.alpha

    # load baseline
    baseline_path = _resolve(args.baseline)
    print(f"Loading baseline: {baseline_path}")
    probs_base, labels_base, mids_base = load_preds(baseline_path)
    auc_base, _, _ = compute_metrics(probs_base, labels_base)
    print(f"  Baseline AUC (point): {auc_base:.4f}\n")

    # bootstrap each variant
    results = {}
    for name, path in variants.items():
        print(f"Loading variant '{name}': {path}")
        probs_v, labels_v, mids_v = load_preds(path)
        auc_v, _, _ = compute_metrics(probs_v, labels_v)
        print(f"  Variant AUC (point): {auc_v:.4f}  (Δ = {auc_v - auc_base:+.4f})")

        print(f"  Running {N_BOOTSTRAP:,} bootstrap iterations …")
        bs = bootstrap_pair(
            probs_base, labels_base, mids_base,
            probs_v,    labels_v,    mids_v,
        )

        obs_diff = float(auc_v - auc_base)
        ci_diff  = ci(bs["auc_diff"])
        p_raw    = pvalue_greater(bs["auc_diff"])

        results[name] = {
            "auc_variant":  float(auc_v),
            "auc_baseline": float(auc_base),
            "delta_auc":    obs_diff,
            "ci_diff":      [float(ci_diff[0]), float(ci_diff[1])],
            "p_raw":        p_raw,
        }

    # Holm-Bonferroni correction
    names  = list(results.keys())
    p_raws = [results[n]["p_raw"] for n in names]
    p_adjs = holm_bonferroni(p_raws)

    for name, p_adj in zip(names, p_adjs):
        results[name]["p_holm"]              = p_adj
        results[name]["significant_holm"]    = bool(p_adj <= alpha)

    # print table
    pct  = int(CI_LEVEL * 100)
    m    = len(names)
    FWER = 1.0 - (1.0 - alpha) ** m
    print()
    print(f"Holm-Bonferroni correction  (m={m}, α={alpha})")
    print(f"Uncorrected FWER = 1-(1-α)^m ≈ {FWER:.3f}")
    print()
    print(
        f"{'Variant':<18s}  {'AUC':>6s}  {'ΔAUC':>7s}  "
        f"{pct}% CI (diff)            {'p raw':>8s}  {'p Holm':>8s}  Sig?"
    )
    print("-" * 80)
    for name in names:
        r = results[name]
        sig = "YES" if r["significant_holm"] else "no"
        lo, hi = r["ci_diff"]
        print(
            f"{name:<18s}  {r['auc_variant']:.4f}  {r['delta_auc']:+.4f}  "
            f"[{lo:+.4f}, {hi:+.4f}]  "
            f"{r['p_raw']:8.4f}  {r['p_holm']:8.4f}  {sig}"
        )

    print(f"\nBaseline (V3 Full) AUC: {auc_base:.4f}")

    # save JSON
    if args.save_json:
        out_path = _resolve(args.save_json)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        output = {
            "baseline_auc": float(auc_base),
            "n_variants":   m,
            "alpha":        alpha,
            "n_bootstrap":  N_BOOTSTRAP,
            "ci_level":     CI_LEVEL,
            "results":      results,
        }
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
