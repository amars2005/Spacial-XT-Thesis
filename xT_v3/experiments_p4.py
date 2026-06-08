"""
xT_v3 — Priority-4 experiments: can the player-positional signal be made to land?

Three variants of Stage 2, each trained on the SAME 360 dataset, the SAME
match-level split (read from checkpoints/split_match_ids.json), the SAME frozen
Stage 1 backbone, seed 42, and 40 epochs as the headline V3 and its ablations —
so every number is directly comparable to Table (comparison.tex).

    richfeat    Cross-attention with RICHER per-player features. Each 8-dim
                player token is augmented on the fly with 5 geometric features
                derived from the freeze frame (distance + direction to goal,
                and the player's position relative to the ball->goal shooting
                lane). Tests whether the null is a feature-poverty problem.

    multiquery  Multi-query cross-attention. Instead of a single scalar-logit
                query, K learned queries (conditioned on the Stage 1 logit and
                the 4 ball features) each attend over the player context, and
                the K context vectors are fused. Tests whether the single
                scalar query is an information bottleneck.

    unfreeze    Partial unfreezing. Full cross-attention Stage 2 PLUS gentle
                fine-tuning of Stage 1's last conv block and classifier at a
                low LR (5e-5), while Stage 2 trains at 5e-4.
                Frozen layers (incl. their BatchNorm running stats) are kept in
                eval mode. Tests whether holding the backbone fixed is the
                limiting factor.

Usage
-----
    python3 experiments_p4.py --variant richfeat   --gpu 2
    python3 experiments_p4.py --variant multiquery --gpu 3
    python3 experiments_p4.py --variant unfreeze   --gpu 5
    python3 experiments_p4.py --variant richfeat   --gpu 2 --quick   # smoke test

Each run writes metrics to ../metrics_v3_<variant>.json and a checkpoint to
checkpoints/best_model_<variant>.pt, and (for direct CI testing) test-set
predictions to predictions/preds_<variant>.npz.
"""
import argparse
import importlib.util
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_V2_DIR = os.path.join(_HERE, "..", "xT_v2")
_SCRIPTS = os.path.join(_ROOT, 'scripts')
for _p in (_HERE, _ROOT, _SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# v3 config / dataset / builder, loaded by path to dodge v2/config name clashes
_cfg = _load_by_path("_v3cfg", os.path.join(_HERE, "config.py"))
_ds = _load_by_path("_v3ds", os.path.join(_HERE, "dataset.py"))

D_MODEL = _cfg.D_MODEL
N_HEADS = _cfg.N_HEADS
N_LAYERS = _cfg.N_LAYERS
PLAYER_DIM = _cfg.PLAYER_DIM          # 8
BALL_DIM = _cfg.BALL_DIM              # 4
BALL_FEAT_INDICES = _cfg.BALL_FEAT_INDICES
NUM_EPOCHS = _cfg.NUM_EPOCHS_V3       # 40
LR_STAGE2 = _cfg.LEARNING_RATE_V3     # 5e-4
LR_FINETUNE = _cfg.LEARNING_RATE_V3_FINETUNE  # 5e-5
WEIGHT_DECAY = _cfg.WEIGHT_DECAY_V3
V2_BEST = _cfg.V2_BEST_MODEL_PATH
CKPT_DIR = _cfg.CHECKPOINT_DIR_V3

RICH_EXTRA = 5
RICH_DIM = PLAYER_DIM + RICH_EXTRA    # 13

from evaluate import compute_metrics, print_metrics, save_metrics  # noqa: E402


def _log(m):
    print(m, flush=True)


# ---------------------------------------------------------------------------
# Richer per-player geometric features (derived, no re-encoding needed)
# ---------------------------------------------------------------------------
def augment_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """
    tokens: (B, P, 8) = [x_norm, y_norm, dx, dy, dist, is_teammate, is_keeper, is_actor]
    Returns (B, P, 13): appends [p_dist_goal, cos_to_goal, sin_to_goal,
                                 lane_along, lane_perp].
    Goal is at normalised (1.0, 0.5); ball position is recovered per event as
    (x_norm - dx, y_norm - dy). Padded rows produce finite junk but are masked
    out in all attention, exactly as the original zero-padded tokens are.
    """
    x = tokens[..., 0]
    y = tokens[..., 1]
    dx = tokens[..., 2]
    dy = tokens[..., 3]

    ball_x = x - dx
    ball_y = y - dy

    gx = 1.0 - x
    gy = 0.5 - y
    p_dist_goal = torch.sqrt(gx * gx + gy * gy) + 1e-6
    cos_to_goal = gx / p_dist_goal
    sin_to_goal = gy / p_dist_goal

    # Player position relative to the ball -> goal shooting lane
    vx = 1.0 - ball_x
    vy = 0.5 - ball_y
    vnorm2 = vx * vx + vy * vy + 1e-6
    wx = x - ball_x
    wy = y - ball_y
    lane_along = (wx * vx + wy * vy) / vnorm2          # 0..1 ⇒ between ball & goal
    lane_perp = torch.abs(vx * wy - vy * wx) / torch.sqrt(vnorm2)

    extra = torch.stack(
        [p_dist_goal, cos_to_goal, sin_to_goal, lane_along, lane_perp], dim=-1
    )
    return torch.cat([tokens, extra], dim=-1)


# ---------------------------------------------------------------------------
# Stage 2 variants
# ---------------------------------------------------------------------------
class Stage2CrossAttn(nn.Module):
    """Full cross-attention Stage 2 (optionally with richer token features)."""

    def __init__(self, token_dim=PLAYER_DIM, rich=False):
        super().__init__()
        self.rich = rich
        self.player_embed = nn.Linear(token_dim, D_MODEL)
        self.query_embed = nn.Linear(1, D_MODEL)
        enc = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=D_MODEL * 4,
            dropout=0.1, batch_first=True,
        )
        self.player_encoder = nn.TransformerEncoder(enc, num_layers=N_LAYERS)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=D_MODEL, num_heads=N_HEADS, dropout=0.1, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(D_MODEL + 1, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )
        nn.init.normal_(self.mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, stage1_logit, ball_feats, player_tokens, player_mask):
        if stage1_logit.dim() == 1:
            stage1_logit = stage1_logit.unsqueeze(1)
        if self.rich:
            player_tokens = augment_tokens(player_tokens)
        pe = self.player_embed(player_tokens)
        ctx = self.player_encoder(pe, src_key_padding_mask=player_mask)
        q = self.query_embed(stage1_logit).unsqueeze(1)
        attn, _ = self.cross_attn(q, ctx, ctx, key_padding_mask=player_mask)
        attn = attn.squeeze(1)
        out = self.mlp(torch.cat([attn, stage1_logit], dim=-1))
        return (stage1_logit + out).squeeze(1)


class Stage2MultiQuery(nn.Module):
    """K learned queries (conditioned on logit + ball features) over players."""

    def __init__(self, n_queries=8):
        super().__init__()
        self.n_queries = n_queries
        self.player_embed = nn.Linear(PLAYER_DIM, D_MODEL)
        self.query_base = nn.Linear(1 + BALL_DIM, D_MODEL)
        self.query_table = nn.Parameter(torch.randn(n_queries, D_MODEL) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=D_MODEL * 4,
            dropout=0.1, batch_first=True,
        )
        self.player_encoder = nn.TransformerEncoder(enc, num_layers=N_LAYERS)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=D_MODEL, num_heads=N_HEADS, dropout=0.1, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(D_MODEL * n_queries + 1, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 1),
        )
        nn.init.normal_(self.mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, stage1_logit, ball_feats, player_tokens, player_mask):
        if stage1_logit.dim() == 1:
            stage1_logit = stage1_logit.unsqueeze(1)
        B = stage1_logit.size(0)
        pe = self.player_embed(player_tokens)
        ctx = self.player_encoder(pe, src_key_padding_mask=player_mask)
        base = self.query_base(torch.cat([stage1_logit, ball_feats], dim=-1))  # (B,D)
        queries = base.unsqueeze(1) + self.query_table.unsqueeze(0)            # (B,K,D)
        attn, _ = self.cross_attn(queries, ctx, ctx, key_padding_mask=player_mask)
        attn = attn.reshape(B, -1)                                            # (B,K*D)
        out = self.mlp(torch.cat([attn, stage1_logit], dim=-1))
        return (stage1_logit + out).squeeze(1)


class V3Experiment(nn.Module):
    """
    Wraps frozen (or partially unfrozen) Stage 1 with one Stage 2 variant.

    unfreeze_layers: list of attribute paths on stage1 to make trainable, e.g.
        ['cnn.block3', 'cnn.head', 'classifier']. When non-empty, Stage 1 is run
        WITH gradients (no torch.no_grad) but kept in eval() mode so frozen
        BatchNorm running stats do not drift.
    """

    def __init__(self, stage1, stage2, unfreeze_layers=None):
        super().__init__()
        self.stage1 = stage1
        self.stage2 = stage2
        self.register_buffer("ball_idx", torch.tensor(BALL_FEAT_INDICES, dtype=torch.long))
        self.unfreeze_layers = unfreeze_layers or []

        for p in self.stage1.parameters():
            p.requires_grad = False
        for path in self.unfreeze_layers:
            mod = self.stage1
            for attr in path.split("."):
                mod = getattr(mod, attr)
            for p in mod.parameters():
                p.requires_grad = True

    @property
    def finetune(self):
        return len(self.unfreeze_layers) > 0

    def forward(self, spatial, scalar, player_tokens, player_mask):
        if self.finetune:
            stage1_logit = self.stage1(spatial, scalar)        # grads flow
        else:
            with torch.no_grad():
                stage1_logit = self.stage1(spatial, scalar)
        ball_feats = scalar[:, self.ball_idx]
        return self.stage2(stage1_logit, ball_feats, player_tokens, player_mask)


# ---------------------------------------------------------------------------
# Training loop (mirrors train.py; adds param groups for fine-tuning)
# ---------------------------------------------------------------------------
def _get_probs_labels(model, loader, device):
    """Run full-pass inference; return (probs, labels) as float64 arrays."""
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for sp, sc, pt, pm, lb in loader:
            sp, sc = sp.to(device, non_blocking=True), sc.to(device, non_blocking=True)
            pt, pm = pt.to(device, non_blocking=True), pm.to(device, non_blocking=True)
            probs.append(torch.sigmoid(model(sp, sc, pt, pm)).cpu().numpy())
            labels.append(lb.numpy())
    return (np.concatenate(probs).astype(np.float64),
            np.concatenate(labels).astype(np.float64))


def _spearman(model, loader, device):
    """Run inference and return (spearman_rho, probs, labels)."""
    from scipy.stats import spearmanr
    probs, labels = _get_probs_labels(model, loader, device)
    rho = float(spearmanr(labels, probs).statistic)
    return rho, probs, labels


def _stage1_auc(stage1, loader, device):
    stage1.eval()
    probs, labels = [], []
    with torch.no_grad():
        for sp, sc, _pt, _pm, lb in loader:
            sp, sc = sp.to(device, non_blocking=True), sc.to(device, non_blocking=True)
            probs.append(torch.sigmoid(stage1(sp, sc)).cpu().numpy())
            labels.append(lb.numpy())
    return roc_auc_score(np.concatenate(labels).astype(np.float64),
                         np.concatenate(probs).astype(np.float64))


def train_variant(model, train_loader, val_loader, test_loader, device,
                  epochs, best_path, metrics_path, preds_path, test_match_ids):
    """Train a Stage 2 variant and evaluate on the test set.

    Checkpoint saving and LR scheduling are driven by validation ROC AUC.
    """
    model.to(device)

    pos_rate = train_loader.dataset.labels.mean().item()
    pos_weight = torch.tensor([min((1.0 - pos_rate) / (pos_rate + 1e-8), 10.0)]).to(device)
    _log(f"goal rate {pos_rate:.4f} | pos_weight {pos_weight.item():.1f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    stage2_params = list(model.stage2.parameters())
    stage1_params = [p for p in model.stage1.parameters() if p.requires_grad]
    groups = [{"params": stage2_params, "lr": LR_STAGE2}]
    if stage1_params:
        groups.append({"params": stage1_params, "lr": LR_FINETUNE})
        _log(f"Fine-tuning {sum(p.numel() for p in stage1_params):,} Stage-1 params @ {LR_FINETUNE}")
    _log(f"Stage-2 trainable params: {sum(p.numel() for p in stage2_params):,}")

    optimizer = optim.Adam(groups, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)

    base_auc = _stage1_auc(model.stage1, val_loader, device)
    _log(f"Stage 1 baseline val AUC: {base_auc:.4f}")

    best = 0.0
    for epoch in range(1, epochs + 1):
        model.stage2.train()
        # Keep Stage 1 in eval mode always (frozen BN stats), even when fine-tuning.
        model.stage1.eval()
        tot = 0.0
        for sp, sc, pt, pm, lb in train_loader:
            sp, sc = sp.to(device, non_blocking=True), sc.to(device, non_blocking=True)
            pt, pm = pt.to(device, non_blocking=True), pm.to(device, non_blocking=True)
            lb = lb.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(sp, sc, pt, pm), lb)
            loss.backward()
            optimizer.step()
            tot += loss.item() * len(lb)
        tot /= len(train_loader.dataset)

        val_probs, val_labels = _get_probs_labels(model, val_loader, device)
        val_auc = roc_auc_score(val_labels, val_probs)
        scheduler.step(val_auc)
        _log(f"Epoch {epoch:>3}/{epochs} | train {tot:.4f} | val AUC {val_auc:.4f} "
             f"| ΔAUC {val_auc - base_auc:+.4f}")
        if val_auc > best:
            best = val_auc
            torch.save(model.state_dict(), best_path)
            _log(f"  -> saved best (AUC {val_auc:.4f})")

    model.load_state_dict(torch.load(best_path, map_location=device))
    _log(f"\nBest val AUC: {best:.4f} (Stage 1 baseline: {base_auc:.4f}, ΔAUC {best - base_auc:+.4f})")

    test_rho, probs, labels = _spearman(model, test_loader, device)
    metrics = compute_metrics(labels, probs)
    print_metrics(metrics, f"Test Set ({os.path.basename(metrics_path)})")
    save_metrics(metrics, metrics_path)

    # also persist test match ids alongside probs for match-level CI
    os.makedirs(os.path.dirname(preds_path), exist_ok=True)
    np.savez(preds_path, probs=probs, labels=labels, match_ids=test_match_ids)
    _log(f"Saved predictions -> {preds_path}  ({len(probs):,} events)")
    return metrics


# ---------------------------------------------------------------------------
def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_stage1(device):
    v2model = _load_by_path("_v2model", os.path.join(_V2_DIR, "model.py"))
    s1 = v2model.XTModel().to(device)
    s1.load_state_dict(torch.load(V2_BEST, map_location=device))
    s1.eval()
    return s1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["richfeat", "multiquery", "multiquery_k16", "unfreeze"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--quick", action="store_true", help="few epochs, smoke test")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training; load saved checkpoint and run test evaluation only")
    args = ap.parse_args()

    epochs = 3 if args.quick else args.epochs
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu)
    _log(f"variant={args.variant} device={device} epochs={epochs} seed={args.seed}")
    _set_seed(args.seed)

    stage1 = _load_stage1(device)
    if args.variant == "richfeat":
        stage2 = Stage2CrossAttn(token_dim=RICH_DIM, rich=True)
        model = V3Experiment(stage1, stage2)
    elif args.variant == "multiquery":
        stage2 = Stage2MultiQuery(n_queries=8)
        model = V3Experiment(stage1, stage2)
    elif args.variant == "multiquery_k16":
        stage2 = Stage2MultiQuery(n_queries=16)
        model = V3Experiment(stage1, stage2)
    else:  # unfreeze
        stage2 = Stage2CrossAttn(token_dim=PLAYER_DIM, rich=False)
        model = V3Experiment(stage1, stage2, unfreeze_layers=["cnn.block3", "classifier"])

    builder = _load_by_path("_v3builder", os.path.join(_HERE, "builder.py")).DatasetBuilderV3()
    train_loader, val_loader, test_loader = _ds.make_dataloaders_v3_from_disk(builder, seed=args.seed)

    # Reload test split (deterministic, same order as the shuffle=False test_loader)
    # purely to recover per-event match IDs for match-level bootstrap CIs.
    with open(os.path.join(CKPT_DIR, "split_match_ids.json")) as f:
        split = json.load(f)
    te = builder.load_split(split["test"], desc="  test(ids)")
    test_match_ids = te[5]
    assert len(test_match_ids) == len(test_loader.dataset), \
        f"match-id misalignment: {len(test_match_ids)} vs {len(test_loader.dataset)}"
    del te

    suffix = args.variant
    best_path = os.path.join(CKPT_DIR, f"best_model_{suffix}.pt")
    metrics_path = os.path.join(_ROOT, "metrics", f"metrics_v3_{suffix}.json")
    preds_path = os.path.join(_HERE, "predictions", f"preds_{suffix}.npz")

    if args.eval_only:
        _log(f"--eval-only: loading checkpoint {best_path}")
        model.to(device)
        model.load_state_dict(torch.load(best_path, map_location=device))
        test_rho, probs, labels = _spearman(model, test_loader, device)
        metrics = compute_metrics(labels, probs)
        print_metrics(metrics, f"Test Set ({os.path.basename(metrics_path)})")
        save_metrics(metrics, metrics_path)
        os.makedirs(os.path.dirname(preds_path), exist_ok=True)
        np.savez(preds_path, probs=probs, labels=labels, match_ids=test_match_ids)
        _log(f"Saved predictions -> {preds_path}  ({len(probs):,} events)")
    else:
        train_variant(model, train_loader, val_loader, test_loader, device,
                      epochs, best_path, metrics_path, preds_path, test_match_ids)


if __name__ == "__main__":
    main()
