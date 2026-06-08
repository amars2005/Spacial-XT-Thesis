"""
xT v3 visualizer — weighted action combination.

For every grid cell the model is run 4 times (Shot / Pass / Carry / Dribble),
each with an action-appropriate end position inferred from the player layout.
The displayed value is the probability-weighted average:

    xT(cell) = Σ P(action | position, players) × model(action, end_pos)

Action weights and end positions are inferred from player positions using the
same heuristics as the frontend (visualisations/frontend/app.py).
"""
import math
import os
import sys
import importlib.util

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_HERE   = os.path.dirname(os.path.abspath(__file__))
_V2DIR  = os.path.join(_HERE, '..', 'xT_v2')
_VISDIR = os.path.join(_HERE, '..', 'visualisations')

# Load v3 config
_spec = importlib.util.spec_from_file_location("_v3cfg", os.path.join(_HERE, "config.py"))
_cfg  = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_cfg)

GRID_W         = _cfg.GRID_W
GRID_H         = _cfg.GRID_H
PITCH_W        = _cfg.PITCH_W
PITCH_H        = _cfg.PITCH_H
NUM_CHANNELS   = _cfg.NUM_CHANNELS
SCALAR_DIM     = _cfg.SCALAR_DIM
MAX_PLAYERS    = _cfg.MAX_PLAYERS
PLAYER_DIM     = _cfg.PLAYER_DIM
HEATMAP_PATH   = _cfg.HEATMAP_PATH_V3
ACTION_TYPES   = _cfg.ACTION_TYPES
SCALAR_COLS    = _cfg.SCALAR_COLS
GAUSSIAN_SIGMA = _cfg.GAUSSIAN_SIGMA

if _V2DIR not in sys.path:
    sys.path.insert(0, _V2DIR)
from encoder import encode_event

_GOAL_X   = 120.0
_GOAL_Y   = 40.0
_POST_L   = 36.0
_POST_R   = 44.0
_MAX_DIST = math.sqrt(_GOAL_X**2 + _GOAL_Y**2)

ACTIONS = ['Shot', 'Pass', 'Carry', 'Dribble']


# ---------------------------------------------------------------------------
# Player format helpers
# ---------------------------------------------------------------------------

def _p(x, y, teammate=False, keeper=False):
    return {'location': [x, y], 'teammate': teammate, 'actor': False, 'keeper': keeper}


def _build_player_tokens(players):
    tokens = np.zeros((MAX_PLAYERS, PLAYER_DIM), dtype=np.float32)
    mask   = np.zeros(MAX_PLAYERS, dtype=bool)
    if not players:
        return tokens, mask
    for i, p in enumerate(players[:MAX_PLAYERS]):
        x, y = p['location']
        # dx/dy/dist are relative to the ball carrier; set to 0 in the batch sweep
        # because ball position varies per grid cell (actor token is not in this list)
        tokens[i] = [x / PITCH_W, y / PITCH_H,
                     0.0, 0.0, 0.0,
                     float(p.get('teammate', False)),
                     float(p.get('keeper',   False)),
                     float(p.get('actor',    False))]
        mask[i] = True
    return tokens, mask


# ---------------------------------------------------------------------------
# Action weight / end-position inference
# ---------------------------------------------------------------------------

def _dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def _in_shooting_lane(ox, oy, bx, by):
    def cross(ax, ay, bx2, by2, px, py):
        return (bx2 - ax) * (py - ay) - (by2 - ay) * (px - ax)
    c1 = cross(bx, by, _GOAL_X, _POST_L, ox, oy)
    c2 = cross(_GOAL_X, _POST_L, _GOAL_X, _POST_R, ox, oy)
    c3 = cross(_GOAL_X, _POST_R, bx, by, ox, oy)
    return (c1 >= 0 and c2 >= 0 and c3 >= 0) or (c1 <= 0 and c2 <= 0 and c3 <= 0)


def _action_context(sx, sy, players):
    """
    Infer action weights, end positions, and under_pressure for one ball location.

    Returns
    -------
    weights       : dict {action: float}  — normalised, sum to 1
    end_positions : dict {action: (ex, ey) or None}
    under_pressure: bool
    """
    opps_all  = [p for p in (players or []) if not p.get('teammate', True)]
    tms_out   = [p for p in (players or []) if p.get('teammate') and not p.get('keeper')]

    goalside          = [p for p in opps_all if p['location'][0] >= sx]
    gs_within_5       = [p for p in goalside if _dist(*p['location'], sx, sy) < 5]
    gs_within_10      = [p for p in goalside if _dist(*p['location'], sx, sy) < 10]
    lane_def          = sum(1 for p in opps_all if _in_shooting_lane(*p['location'], sx, sy))

    under_pressure = len(gs_within_5) > 0
    dist_to_goal   = _dist(sx, sy, _GOAL_X, _GOAL_Y)

    # Weights
    shot_w    = math.exp(-dist_to_goal / 25.0) * max(0.0, 1.0 - lane_def / 3.0)
    pass_w    = min(len(tms_out), 5) / 5.0 if tms_out else 0.0
    carry_w   = max(0.0, 1.0 - len(gs_within_10) / 4.0)
    dribble_w = len(gs_within_5) * max(0.0, 1.0 - (len(gs_within_5) - 1) / 3.0)

    total = max(shot_w + pass_w + carry_w + dribble_w, 0.01)
    weights = {
        'Shot':    shot_w    / total,
        'Pass':    pass_w    / total,
        'Carry':   carry_w   / total,
        'Dribble': dribble_w / total,
    }

    # Carry end: 8 yards toward goal
    dx = _GOAL_X - sx;  dy = _GOAL_Y - sy
    cd = math.sqrt(dx**2 + dy**2)
    if cd > 0:
        carry_ex = min(sx + 8.0 * dx / cd, PITCH_W - 0.1)
        carry_ey = max(0.1, min(sy + 8.0 * dy / cd, PITCH_H - 0.1))
    else:
        carry_ex, carry_ey = sx, sy

    end_positions = {
        'Shot':    (_GOAL_X, _GOAL_Y),
        'Carry':   (carry_ex, carry_ey),
        'Dribble': (sx, sy),
        # All outfield teammates — sweep will run model per target and take max xT
        'Pass':    [(p['location'][0], p['location'][1]) for p in tms_out],
    }

    return weights, end_positions, under_pressure


# ---------------------------------------------------------------------------
# Scalar builder (mirrors visualisations/frontend/app.py build_scalar)
# ---------------------------------------------------------------------------

def _build_scalar(sx, sy, action_type, ex, ey, under_pressure):
    idx = {col: i for i, col in enumerate(SCALAR_COLS)}
    vec = np.zeros(SCALAR_DIM, dtype=np.float32)

    if action_type in ACTION_TYPES:
        vec[ACTION_TYPES.index(action_type)] = 1.0

    vec[idx['start_x_norm']] = sx / PITCH_W
    vec[idx['start_y_norm']] = sy / PITCH_H

    dx = _GOAL_X - sx;  dy = _GOAL_Y - sy
    vec[idx['dist_to_goal_norm']]  = math.sqrt(dx**2 + dy**2) / _MAX_DIST
    vec[idx['angle_to_goal_norm']] = math.atan2(dy, dx) / math.pi

    vec[idx['under_pressure']] = float(under_pressure)
    vec[idx['period_norm']]    = 0.25
    vec[idx['score_diff_norm']]= 0.5

    if action_type == 'Pass' and ex is not None:
        pdx = ex - sx;  pdy = ey - sy
        pl = math.sqrt(pdx**2 + pdy**2)
        vec[idx['pass_length_norm']] = pl / math.sqrt(PITCH_W**2 + PITCH_H**2)
        vec[idx['pass_angle_norm']]  = math.atan2(pdy, pdx) / math.pi

    vec[idx['end_x_norm']] = (ex if ex is not None else sx) / PITCH_W
    vec[idx['end_y_norm']] = (ey if ey is not None else sy) / PITCH_H
    return vec


# ---------------------------------------------------------------------------
# Weighted pitch sweep
# ---------------------------------------------------------------------------

def _sweep_pitch_weighted(model, device, players):
    """
    Sweep ball position across the pitch grid.
    For each cell, compute the weighted xT using all 4 action types.
    Returns (xx, yy, z_weighted, z_per_action) where z_per_action is a
    dict of {action: (GRID_H, GRID_W)} arrays for the per-action subplot.
    """
    x_centres = np.linspace(PITCH_W / (2*GRID_W), PITCH_W - PITCH_W/(2*GRID_W), GRID_W)
    y_centres = np.linspace(PITCH_H / (2*GRID_H), PITCH_H - PITCH_H/(2*GRID_H), GRID_H)
    xx, yy    = np.meshgrid(x_centres, y_centres)

    N = GRID_H * GRID_W
    tokens, tok_mask = _build_player_tokens(players)
    frame_data = {'players': players, 'visible_area': []} if players else None

    # Pre-compute per-cell action context
    # Pass end_positions is a list of targets; stored separately for multi-target max.
    weights_arr     = np.zeros((N, len(ACTIONS)), dtype=np.float32)
    end_pos_arr     = np.full((N, len(ACTIONS), 2), np.nan, dtype=np.float32)
    under_press_arr = np.zeros(N, dtype=bool)
    # Pass targets are the same for every cell (players don't move); extract once.
    pass_targets = [(p['location'][0], p['location'][1])
                    for p in (players or [])
                    if p.get('teammate') and not p.get('keeper')]

    for idx in range(N):
        gy, gx = divmod(idx, GRID_W)
        sx, sy = float(xx[gy, gx]), float(yy[gy, gx])
        w, ep, up = _action_context(sx, sy, players)
        under_press_arr[idx] = up
        for ai, a in enumerate(ACTIONS):
            weights_arr[idx, ai] = w[a]
            if a == 'Pass':
                pass  # handled separately below
            elif ep[a] is not None:
                end_pos_arr[idx, ai] = ep[a]
            else:
                end_pos_arr[idx, ai] = [sx, sy]

    # Build base spatial (channels 0–3, 5 constant per grid cell; ch4 varies per action)
    base_spatial = np.zeros((N, NUM_CHANNELS, GRID_H, GRID_W), dtype=np.float32)
    scalar_base  = np.zeros((N, SCALAR_DIM), dtype=np.float32)

    for idx in range(N):
        gy, gx = divmod(idx, GRID_W)
        sx, sy = float(xx[gy, gx]), float(yy[gy, gx])
        ev = {'start_x': sx, 'start_y': sy, 'end_x': np.nan, 'end_y': np.nan}
        sp = encode_event(ev, frame_data)
        base_spatial[idx] = sp

    # Run model for each action type
    BATCH = 256
    action_xt = {}

    def _run_forward(spatial_a, scalar_a):
        """Batch forward pass; returns flat (N,) probability array."""
        all_probs = []
        with torch.no_grad():
            for start in range(0, N, BATCH):
                end_  = min(start + BATCH, N)
                b     = end_ - start
                sp_t  = torch.from_numpy(spatial_a[start:end_]).to(device)
                sc_t  = torch.from_numpy(scalar_a[start:end_]).to(device)
                tok_b = torch.from_numpy(tokens).unsqueeze(0).expand(b, -1, -1).to(device)
                msk_b = torch.from_numpy(tok_mask).unsqueeze(0).expand(b, -1).to(device)
                logits = model(sp_t, sc_t, tok_b, msk_b)
                probs  = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs.tolist())
        return np.array(all_probs, dtype=np.float32).reshape(N)

    for ai, action in enumerate(ACTIONS):
        if action == 'Pass':
            # Run once per teammate target, take per-cell max (mirrors app.py)
            if not pass_targets:
                action_xt['Pass'] = np.zeros(N, dtype=np.float32)
                continue
            candidate_probs = []
            for (tex, tey) in pass_targets:
                spatial_a = base_spatial.copy()
                scalar_a  = np.zeros((N, SCALAR_DIM), dtype=np.float32)
                gex = tex / PITCH_W * GRID_W
                gey = tey / PITCH_H * GRID_H
                xs  = np.arange(GRID_W, dtype=np.float32)
                ys  = np.arange(GRID_H, dtype=np.float32)
                XX, YY = np.meshgrid(xs, ys)
                end_ch = np.exp(-((XX - gex)**2 + (YY - gey)**2) / (2 * GAUSSIAN_SIGMA**2))
                for idx in range(N):
                    spatial_a[idx, 4] = end_ch
                    gy, gx = divmod(idx, GRID_W)
                    sx, sy = float(xx[gy, gx]), float(yy[gy, gx])
                    up = under_press_arr[idx]
                    scalar_a[idx] = _build_scalar(sx, sy, 'Pass', tex, tey, bool(up))
                candidate_probs.append(_run_forward(spatial_a, scalar_a))
            action_xt['Pass'] = np.max(np.stack(candidate_probs), axis=0)
            continue

        spatial_a = base_spatial.copy()
        scalar_a  = np.zeros((N, SCALAR_DIM), dtype=np.float32)

        for idx in range(N):
            gy, gx = divmod(idx, GRID_W)
            sx, sy = float(xx[gy, gx]), float(yy[gy, gx])
            ex, ey = end_pos_arr[idx, ai]
            up     = under_press_arr[idx]

            if action != 'Dribble':
                gex = ex / PITCH_W * GRID_W
                gey = ey / PITCH_H * GRID_H
                xs  = np.arange(GRID_W, dtype=np.float32)
                ys  = np.arange(GRID_H, dtype=np.float32)
                XX, YY = np.meshgrid(xs, ys)
                spatial_a[idx, 4] = np.exp(
                    -((XX - gex)**2 + (YY - gey)**2) / (2 * GAUSSIAN_SIGMA**2)
                )

            scalar_a[idx] = _build_scalar(sx, sy, action, ex, ey, bool(up))

        action_xt[action] = _run_forward(spatial_a, scalar_a)

    # Weighted combination
    z_flat = np.zeros(N, dtype=np.float32)
    for ai, action in enumerate(ACTIONS):
        z_flat += weights_arr[:, ai] * action_xt[action]

    z_weighted    = z_flat.reshape(GRID_H, GRID_W)
    z_per_action  = {a: action_xt[a].reshape(GRID_H, GRID_W) for a in ACTIONS}

    return xx, yy, z_weighted, z_per_action


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

SCENARIOS = {
    'No Players (Baseline)': None,

    'Compact Low Block': [
        _p(119, 40, keeper=True),
        _p(102, 20), _p(102, 33), _p(102, 47), _p(102, 60),
        _p(88, 22),  _p(88, 35),  _p(88, 45),  _p(88, 58),
    ],

    'High Press': [
        _p(119, 40, keeper=True),
        _p(62, 22), _p(62, 35), _p(62, 45), _p(62, 58),
        _p(72, 20), _p(72, 33), _p(72, 47), _p(72, 60),
    ],

    '2v1 Overload in Box': [
        _p(119, 40, keeper=True),
        _p(107, 40),
        _p(108, 28, teammate=True),
        _p(108, 52, teammate=True),
        _p(103, 18), _p(103, 62),
    ],

    'Counter-Attack 3v2': [
        _p(119, 40, keeper=True),
        _p(105, 30), _p(105, 50),
        _p(110, 22, teammate=True),
        _p(110, 58, teammate=True),
        _p(92,  40, teammate=True),
    ],

    'Goalkeeper Off Line': [
        _p(106, 40, keeper=True),
        _p(100, 22), _p(100, 35), _p(100, 45), _p(100, 58),
    ],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_heatmap(model, device):
    """Weighted xT heatmap with no players (baseline positional threat)."""
    print("[v3] Generating weighted baseline heatmap...")
    model.eval()
    xx, yy, z, _ = _sweep_pitch_weighted(model, device, players=None)
    out = HEATMAP_PATH.replace('_v3.png', '_weighted_v3.png')
    _plot_heatmap(xx, yy, z, 'Weighted xT — No Players (Baseline)', out)


def generate_per_action_heatmaps(model, device):
    """Per-action xT heatmaps (baseline, no players) for all four action types."""
    print("[v3] Generating per-action baseline heatmaps...")
    model.eval()
    xx, yy, _, z_per_action = _sweep_pitch_weighted(model, device, players=None)
    for action in ACTIONS:
        slug = action.lower()
        out  = os.path.join(_VISDIR, f'xt_heatmap_{slug}_v3.png')
        _plot_heatmap(xx, yy, z_per_action[action],
                      f'SxT v3 — {action} (No Players, Baseline)', out,
                      cbar_label=f'{action} xT')


def generate_per_action_scenarios(model, device):
    """Per-action scenario comparison grids for all four action types."""
    print("[v3] Generating per-action scenario heatmaps...")
    model.eval()

    n     = len(SCENARIOS)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    all_results = {}
    for name, players in SCENARIOS.items():
        print(f"  Scenario: {name}")
        xx, yy, _, z_per_action = _sweep_pitch_weighted(model, device, players)
        all_results[name] = (xx, yy, z_per_action, players)

    for action in ACTIONS:
        global_max = max(v[2][action].max() for v in all_results.values())
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 5))
        axes = axes.flatten()

        for ax, (name, (xx, yy, z_per_action, players)) in zip(axes, all_results.items()):
            cf = ax.contourf(xx, yy, z_per_action[action],
                             levels=30, cmap='magma', vmin=0, vmax=global_max)
            fig.colorbar(cf, ax=ax, fraction=0.03, pad=0.02)
            _draw_pitch(ax)
            if players:
                for p in players:
                    px, py = p['location']
                    if p.get('keeper'):
                        ax.scatter(px, py, c='cyan', s=80, zorder=6, marker='s')
                    elif p.get('teammate'):
                        ax.scatter(px, py, c='lime', s=60, zorder=6, marker='^')
                    else:
                        ax.scatter(px, py, c='red',  s=60, zorder=6, marker='o')
            ax.set_title(name, fontsize=11, pad=6)
            ax.set_xlim(0, PITCH_W); ax.set_ylim(PITCH_H, 0)
            ax.set_aspect('equal'); ax.axis('off')

        for ax in axes[n:]:
            ax.set_visible(False)

        slug = action.lower()
        fig.suptitle(f'SxT v3 — {action} Scenarios', fontsize=14, y=1.01)
        plt.tight_layout()
        out = os.path.join(_VISDIR, f'xt_scenarios_{slug}_v3.png')
        plt.savefig(out, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out}")


def generate_all_scenarios(model, device):
    """Scenario comparison grid using weighted action xT."""
    print("[v3] Generating weighted scenario heatmaps...")
    model.eval()

    n     = len(SCENARIOS)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 5))
    axes = axes.flatten()

    results    = {}
    global_max = 0.0
    for name, players in SCENARIOS.items():
        print(f"  Scenario: {name}")
        xx, yy, z, _ = _sweep_pitch_weighted(model, device, players)
        results[name] = (xx, yy, z, players)
        global_max = max(global_max, z.max())

    for ax, (name, (xx, yy, z, players)) in zip(axes, results.items()):
        cf = ax.contourf(xx, yy, z, levels=30, cmap='magma', vmin=0, vmax=global_max)
        fig.colorbar(cf, ax=ax, fraction=0.03, pad=0.02)
        _draw_pitch(ax)

        if players:
            for p in players:
                px, py = p['location']
                if p.get('keeper'):
                    ax.scatter(px, py, c='cyan',  s=80, zorder=6, marker='s')
                elif p.get('teammate'):
                    ax.scatter(px, py, c='lime',  s=60, zorder=6, marker='^')
                else:
                    ax.scatter(px, py, c='red',   s=60, zorder=6, marker='o')

        ax.set_title(name, fontsize=11, pad=6)
        ax.set_xlim(0, PITCH_W); ax.set_ylim(PITCH_H, 0)
        ax.set_aspect('equal'); ax.axis('off')

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle('SxT v3 — Weighted Action Scenarios', fontsize=14, y=1.01)
    plt.tight_layout()
    out = os.path.join(_VISDIR, 'xt_scenarios_weighted_v3.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_heatmap(xx, yy, z, title, out_path, cbar_label='xT'):
    fig, ax = plt.subplots(figsize=(12, 8))
    cf   = ax.contourf(xx, yy, z, levels=30, cmap='magma', vmin=0, vmax=z.max())
    cbar = fig.colorbar(cf, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(cbar_label, fontsize=11)
    _draw_pitch(ax)
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlim(0, PITCH_W); ax.set_ylim(PITCH_H, 0)
    ax.set_aspect('equal'); ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _draw_pitch(ax):
    lc, lw = 'white', 1.5
    ax.plot([0,0],[0,80],color=lc,linewidth=lw); ax.plot([120,120],[0,80],color=lc,linewidth=lw)
    ax.plot([0,120],[0,0],color=lc,linewidth=lw); ax.plot([0,120],[80,80],color=lc,linewidth=lw)
    ax.plot([60,60],[0,80],color=lc,linewidth=lw)
    ax.add_patch(mpatches.Circle((60,40),10,color=lc,fill=False,linewidth=lw))
    ax.scatter(60,40,color=lc,s=15,zorder=5)
    ax.plot([18,18],[18,62],color=lc,linewidth=lw); ax.plot([0,18],[18,18],color=lc,linewidth=lw)
    ax.plot([0,18],[62,62],color=lc,linewidth=lw); ax.plot([102,102],[18,62],color=lc,linewidth=lw)
    ax.plot([102,120],[18,18],color=lc,linewidth=lw); ax.plot([102,120],[62,62],color=lc,linewidth=lw)
    ax.plot([6,6],[30,50],color=lc,linewidth=lw); ax.plot([0,6],[30,30],color=lc,linewidth=lw)
    ax.plot([0,6],[50,50],color=lc,linewidth=lw); ax.plot([114,114],[30,50],color=lc,linewidth=lw)
    ax.plot([114,120],[30,30],color=lc,linewidth=lw); ax.plot([114,120],[50,50],color=lc,linewidth=lw)
    ax.scatter(12,40,color=lc,s=15,zorder=5); ax.scatter(108,40,color=lc,s=15,zorder=5)
