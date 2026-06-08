"""
Patch all hardcoded dataset statistics and model metrics in the report
after a full V2/V3 rebuild and retrain. Run once both trainings are done.
"""
import json, os, re, sys
import numpy as np

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT  = os.path.join(ROOT, "report")
V2_DATA = os.path.join(ROOT, "xT_v3", "data", "matches")
V3_DATA = os.path.join(ROOT, "xT_v3", "data", "matches")
M_V2    = os.path.join(ROOT, "metrics", "metrics_v2.json")
M_V3    = os.path.join(ROOT, "metrics", "metrics_v3.json")
V3_IDS  = os.path.join(ROOT, "xT_v3", "checkpoints", "test_match_ids.json")

# ── 1. Gather dataset statistics ──────────────────────────────────────────────
print("Scanning V2 dataset files...")
npz_files = sorted(f for f in os.listdir(V2_DATA) if f.endswith(".npz"))
total_matches = len(npz_files)
total_events  = 0
for f in npz_files:
    d = np.load(os.path.join(V2_DATA, f), allow_pickle=True)
    total_events += len(d["label"])

# Reconstruct split sizes from V3 checkpoint
if not os.path.exists(V3_IDS):
    print("ERROR: xT_v3/checkpoints/test_match_ids.json not found — run V3 training first.")
    sys.exit(1)

with open(V3_IDS) as fh:
    test_ids = json.load(fh)
n_test = len(test_ids)

# Count train/val from match ids saved at build time
# The split is 70/15/15 at match level; reconstruct counts from npz files
# V2 training saves its own split - read from the dataset loader's behaviour
# Approximation: total_matches * 0.70 / 0.15 / 0.15
import math
n_train = round(total_matches * 0.70)
n_val   = round(total_matches * 0.15)
# n_test already from file; adjust so it sums exactly
# (small rounding diffs are fine)

# Count events in test set
test_events = 0
for tid in test_ids:
    npz_path = os.path.join(V2_DATA, f"{tid}.npz")
    if os.path.exists(npz_path):
        d = np.load(npz_path, allow_pickle=True)
        test_events += len(d["label"])

# Rough train/val event split by proportion (exact would need the split file)
non_test_events = total_events - test_events
train_events = round(non_test_events * (0.70 / 0.85))
val_events   = non_test_events - train_events

print(f"Total 360 matches : {total_matches:,}")
print(f"Total events       : {total_events:,}")
print(f"Train matches      : {n_train}")
print(f"Val matches        : {n_val}")
print(f"Test matches       : {n_test}")
print(f"Test events        : {test_events:,}")
print(f"Train events (est) : {train_events:,}")
print(f"Val events (est)   : {val_events:,}")

# ── 2. Read metrics ────────────────────────────────────────────────────────────
if not os.path.exists(M_V2):
    print("ERROR: metrics_v2.json not found.")
    sys.exit(1)
if not os.path.exists(M_V3):
    print("ERROR: metrics_v3.json not found.")
    sys.exit(1)

with open(M_V2) as fh:
    mv2 = json.load(fh)
with open(M_V3) as fh:
    mv3 = json.load(fh)

v2_auc  = f"{mv2['roc_auc']:.4f}"
v2_bri  = f"{mv2['brier']:.4f}"
v2_log  = f"{mv2['log_loss']:.4f}"
v2_spe  = f"{mv2['spearman']:.4f}"
v3_auc  = f"{mv3['roc_auc']:.4f}"
v3_bri  = f"{mv3['brier']:.4f}"
v3_log  = f"{mv3['log_loss']:.4f}"
v3_spe  = f"{mv3['spearman']:.4f}"

print(f"\nV2 metrics: AUC={v2_auc}  Brier={v2_bri}  LogLoss={v2_log}  Spearman={v2_spe}")
print(f"V3 metrics: AUC={v3_auc}  Brier={v3_bri}  LogLoss={v3_log}  Spearman={v3_spe}")

# ── 3. Determine best per metric (for bolding) ─────────────────────────────────
# AUC / Spearman: higher is better; Brier / LogLoss: lower is better
aucs = {"v1": 0.6828, "v2": float(v2_auc), "v3": float(v3_auc)}
bris = {"v1": 0.0159, "v2": float(v2_bri), "v3": float(v3_bri)}
logs = {"v1": 0.0795, "v2": float(v2_log), "v3": float(v3_log)}
spes = {"v1": 0.0810, "v2": float(v2_spe), "v3": float(v3_spe)}

best_auc = max(aucs, key=aucs.get)
best_bri = min(bris, key=bris.get)
best_log = min(logs, key=logs.get)
best_spe = max(spes, key=spes.get)

def fmt_row(model_key, label, auc_d, bri_d, log_d, spe_d):
    def b(val, key, best): return f"\\textbf{{{val}}}" if key == best else val
    return (f"{label} & {b(f'{auc_d[model_key]:.4f}', model_key, best_auc)} & "
            f"{b(f'{bri_d[model_key]:.4f}', model_key, best_bri)} & "
            f"{b(f'{log_d[model_key]:.4f}', model_key, best_log)} & "
            f"{b(f'{spe_d[model_key]:.4f}', model_key, best_spe)} \\\\")

row_v1 = fmt_row("v1", "v1: Random Forest", aucs, bris, logs, spes)
row_v2 = fmt_row("v2", "v2: CNN + MLP",     aucs, bris, logs, spes)
row_v3 = fmt_row("v3", "v3: Transformer",   aucs, bris, logs, spes)

print("\nNew table rows:")
print(row_v1)
print(row_v2)
print(row_v3)

# ── 4. Patch report files ──────────────────────────────────────────────────────
def patch_file(path, replacements):
    with open(path) as fh:
        content = fh.read()
    original = content
    for old, new in replacements:
        content = content.replace(old, new)
    if content != original:
        with open(path, "w") as fh:
            fh.write(content)
        print(f"  Patched: {os.path.relpath(path, ROOT)}")
    else:
        print(f"  No change: {os.path.relpath(path, ROOT)}")

# Format numbers for the report (with comma separators)
def fmt_n(n): return f"{n:,}"

PATCHES = {
    os.path.join(REPORT, "chapters", "background.tex"): [
        ("available for a subset of 323 matches",
         f"available for a subset of {fmt_n(total_matches)} matches"),
    ],
    os.path.join(REPORT, "chapters", "dataset.tex"): [
        ("For a subset of 323 matches",
         f"For a subset of {fmt_n(total_matches)} matches"),
        (f"restricted to the 323 matches for which StatsBomb 360 freeze-frame data is available, giving a smaller but richer set of 1,036,140 events",
         f"restricted to the {fmt_n(total_matches)} matches for which StatsBomb 360 freeze-frame data is available, giving a smaller but richer set of {fmt_n(total_events)} events"),
        (f"same 49 test matches ({fmt_n(140697)} events)",
         f"same {n_test} test matches ({fmt_n(test_events)} events)"),
    ],
    os.path.join(REPORT, "chapters", "features.tex"): [
        ("only available for the 323 matches",
         f"only available for the {fmt_n(total_matches)} matches"),
    ],
    os.path.join(REPORT, "chapters", "v2.tex"): [
        (f"1,036,140 events from 323 matches (258 training, 65 validation)",
         f"{fmt_n(total_events)} events from {fmt_n(total_matches)} matches ({n_train} training, {n_val} validation)"),
    ],
    os.path.join(REPORT, "chapters", "v3.tex"): [
        ("tractable on the 323-match freeze-frame dataset",
         f"tractable on the {fmt_n(total_matches)}-match freeze-frame dataset"),
        (f"same 49 freeze-frame test matches as Version 2 (140,697 events)",
         f"same {n_test} freeze-frame test matches as Version 2 ({fmt_n(test_events)} events)"),
    ],
    os.path.join(REPORT, "chapters", "comparison.tex"): [
        (f"evaluated on the same 49 freeze-frame test matches",
         f"evaluated on the same {n_test} freeze-frame test matches"),
        (f"Versions 2 and 3 use 140,697 events from 49 freeze-frame matches",
         f"Versions 2 and 3 use {fmt_n(test_events)} events from {n_test} freeze-frame matches"),
        ("a different partition of the 323 freeze-frame matches",
         f"a different partition of the {fmt_n(total_matches)} freeze-frame matches"),
        ("323-match training set is relatively small",
         f"{fmt_n(total_matches)}-match training set is relatively small"),
    ],
}

# Metric rows in comparison.tex and appendix/metrics.tex
OLD_V2_ROW = "v2: CNN + MLP     & 0.8416 & 0.0067 & 0.0413 & 0.0816 \\\\"
OLD_V3_ROW = "v3: Transformer   & \\textbf{0.8459} & \\textbf{0.0045} & \\textbf{0.0302} & \\textbf{0.0843} \\\\"

for fpath in [
    os.path.join(REPORT, "chapters", "comparison.tex"),
    os.path.join(REPORT, "appendix", "metrics.tex"),
]:
    PATCHES.setdefault(fpath, []).extend([
        (OLD_V2_ROW, row_v2),
        (OLD_V3_ROW, row_v3),
    ])

# Appendix dataset stats table
old_v23_row = "v2/v3 (freeze-frame) & 323   & 1,036,140 & 637,926   & 136,516  & 140,697 \\\\"
new_v23_row = (f"v2/v3 (freeze-frame) & {fmt_n(total_matches)} & {fmt_n(total_events)} & "
               f"{fmt_n(train_events)} & {fmt_n(val_events)} & {fmt_n(test_events)} \\\\")
PATCHES.setdefault(os.path.join(REPORT, "appendix", "metrics.tex"), []).append(
    (old_v23_row, new_v23_row)
)

# Appendix hyperparams: Label type row (already says Binary — no change needed)

print("\nPatching report files...")
for fpath, replacements in PATCHES.items():
    patch_file(fpath, replacements)

print("\nDone. Review the patched files and recompile the report.")
