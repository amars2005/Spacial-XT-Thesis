import json
import math
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import GRID_W, GRID_H, PITCH_W, PITCH_H, NUM_CHANNELS, HEATMAP_PATH, GAUSSIAN_SIGMA
from encoder import encode_event, build_inference_scalar

_GOAL_X   = 120.0
_GOAL_Y   = 40.0
_POST_L   = 36.0
_POST_R   = 44.0
_MAX_DIST = math.sqrt(_GOAL_X**2 + _GOAL_Y**2)

# Empirical median carry distance — set by scripts/compute_carry_stats.py; falls back to 8.0
_carry_stats_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'metrics', 'carry_stats.json')
_CARRY_DISTANCE = 8.0
if os.path.exists(_carry_stats_path):
    with open(_carry_stats_path) as _cf:
        _CARRY_DISTANCE = json.load(_cf).get("median_distance", 8.0)

ACTIONS = ['Shot', 'Pass', 'Carry', 'Dribble']


# ---------------------------------------------------------------------------
# Scenario definitions
# All coordinates are in StatsBomb space: x ∈ [0,120], y ∈ [0,80]
# Attacking direction is left → right (goal at x=120, y=40)
# 'teammate' = same team as ball carrier (attackers)
# 'keeper'   = opponent goalkeeper
# ---------------------------------------------------------------------------

def _p(x, y, teammate=False, keeper=False):
    return {'location': [x, y], 'teammate': teammate, 'actor': False, 'keeper': keeper}

SCENARIOS = {
    'No Players (Baseline)': None,

    'Compact Low Block': {
        'players': [
            _p(119, 40, keeper=True),
            # Back four
            _p(102, 20), _p(102, 33), _p(102, 47), _p(102, 60),
            # Mid four
            _p(88, 22),  _p(88, 35),  _p(88, 45),  _p(88, 58),
        ],
        'visible_area': [],
    },

    'High Press': {
        'players': [
            _p(119, 40, keeper=True),
            # Mid four pressing high into opponent half
            _p(62, 22), _p(62, 35), _p(62, 45), _p(62, 58),
            # Back four pushed up to halfway
            _p(72, 20), _p(72, 33), _p(72, 47), _p(72, 60),
        ],
        'visible_area': [],
    },

    '2v1 Overload in Box': {
        'players': [
            _p(119, 40, keeper=True),
            # 1 defender in box
            _p(107, 40),
            # 2 attacking teammates making runs
            _p(108, 28, teammate=True),
            _p(108, 52, teammate=True),
            # Covering defenders wide
            _p(103, 18), _p(103, 62),
        ],
        'visible_area': [],
    },

    'Counter-Attack 3v2': {
        'players': [
            _p(119, 40, keeper=True),
            # 2 covering defenders caught between ball and runners
            _p(105, 30), _p(105, 50),
            # Wide runners already in behind the defence
            _p(110, 22, teammate=True),
            _p(110, 58, teammate=True),
            # Central runner linking play (ball carrier area)
            _p(92, 40, teammate=True),
        ],
        'visible_area': [],
    },

    'Goalkeeper Off Line': {
        'players': [
            # Keeper rushed out to edge of box (~18 yards from goal)
            _p(106, 40, keeper=True),
            # Back four sitting deeper than keeper, covering behind
            _p(100, 22), _p(100, 35), _p(100, 45), _p(100, 58),
        ],
        'visible_area': [],
    },
}


def _sweep_pitch(model, device, frame_data, action_type):
    """Run ball-position sweep across full pitch with fixed player layout."""
    x_centres = np.linspace(PITCH_W / (2 * GRID_W), PITCH_W - PITCH_W / (2 * GRID_W), GRID_W)
    y_centres = np.linspace(PITCH_H / (2 * GRID_H), PITCH_H - PITCH_H / (2 * GRID_H), GRID_H)
    xx, yy = np.meshgrid(x_centres, y_centres)

    from config import SCALAR_DIM
    n_points = GRID_H * GRID_W
    spatial_batch = np.zeros((n_points, NUM_CHANNELS, GRID_H, GRID_W), dtype=np.float32)
    scalar_batch  = np.zeros((n_points, SCALAR_DIM), dtype=np.float32)

    idx = 0
    for gy in range(GRID_H):
        for gx in range(GRID_W):
            sx, sy = float(xx[gy, gx]), float(yy[gy, gx])
            event_row = {'start_x': sx, 'start_y': sy, 'end_x': np.nan, 'end_y': np.nan}
            spatial_batch[idx] = encode_event(event_row, frame_data)
            scalar_batch[idx]  = build_inference_scalar(sx, sy, action_type)
            idx += 1

    all_probs = []
    with torch.no_grad():
        for start in range(0, n_points, 256):
            end    = min(start + 256, n_points)
            sp     = torch.from_numpy(spatial_batch[start:end]).to(device)
            sc     = torch.from_numpy(scalar_batch[start:end]).to(device)
            logits = model(sp, sc)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())

    return xx, yy, np.array(all_probs, dtype=np.float32).reshape(GRID_H, GRID_W)


def generate_heatmap(
    model:       nn.Module,
    device:      torch.device,
    action_type: str = 'Pass',
) -> None:
    """
    Sweeps ball position across every cell of the pitch grid, runs each
    through the model (with empty player channels and full visibility),
    and saves a contour heatmap of the predicted xT values.

    Parameters
    ----------
    model       : trained XTModel (weights already loaded)
    device      : torch.device
    action_type : the StatsBomb action type to simulate ('Pass' by default)
    """
    print(f"Generating xT heatmap for action type: '{action_type}'...")
    model.eval()
    xx, yy, z = _sweep_pitch(model, device, frame_data=None, action_type=action_type)
    _plot_heatmap(xx, yy, z, action_type)


def generate_all_scenarios(
    model:       nn.Module,
    device:      torch.device,
    action_type: str = 'Pass',
) -> None:
    """
    Generate a multi-panel heatmap comparing all defined player scenarios.
    Saved alongside the standard heatmap as xt_scenarios_v2.png.
    """
    print(f"Generating scenario heatmaps for action type: '{action_type}'...")
    model.eval()

    n = len(SCENARIOS)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 5))
    axes = axes.flatten()

    global_max = 0.0
    results = {}
    for name, frame_data in SCENARIOS.items():
        xx, yy, z = _sweep_pitch(model, device, frame_data, action_type)
        results[name] = (xx, yy, z)
        global_max = max(global_max, z.max())

    for ax, (name, (xx, yy, z)) in zip(axes, results.items()):
        cf = ax.contourf(xx, yy, z, levels=30, cmap='magma', vmin=0, vmax=global_max)
        fig.colorbar(cf, ax=ax, fraction=0.03, pad=0.02)
        _draw_pitch(ax)

        # Overlay player markers for non-baseline scenarios
        frame_data = SCENARIOS[name]
        if frame_data and 'players' in frame_data:
            for p in frame_data['players']:
                px, py = p['location']
                if p.get('keeper'):
                    ax.scatter(px, py, c='cyan',   s=80, zorder=6, marker='s', label='GK')
                elif p.get('teammate'):
                    ax.scatter(px, py, c='lime',   s=60, zorder=6, marker='^', label='Attacker')
                else:
                    ax.scatter(px, py, c='red',    s=60, zorder=6, marker='o', label='Defender')

        ax.set_title(name, fontsize=11, pad=6)
        ax.set_xlim(0, PITCH_W)
        ax.set_ylim(PITCH_H, 0)
        ax.set_aspect('equal')
        ax.axis('off')

    # Hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(
        f'SxT — Scenario Comparison (action={action_type})',
        fontsize=14, y=1.01,
    )
    plt.tight_layout()

    slug = action_type.replace(' ', '_').replace('*', '').lower()
    out_path = os.path.join(os.path.dirname(HEATMAP_PATH), f'xt_scenarios_{slug}_v2.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Scenario heatmaps saved to {out_path}")


# ---------------------------------------------------------------------------
# Weighted xT — action context helpers
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
    opps_all  = [p for p in (players or []) if not p.get('teammate', True)]
    tms_out   = [p for p in (players or []) if p.get('teammate') and not p.get('keeper')]

    goalside     = [p for p in opps_all if p['location'][0] >= sx]
    gs_within_5  = [p for p in goalside if _dist(*p['location'], sx, sy) < 5]
    gs_within_10 = [p for p in goalside if _dist(*p['location'], sx, sy) < 10]
    lane_def     = sum(1 for p in opps_all if _in_shooting_lane(*p['location'], sx, sy))

    under_pressure = len(gs_within_5) > 0
    dist_to_goal   = _dist(sx, sy, _GOAL_X, _GOAL_Y)

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

    dx = _GOAL_X - sx;  dy = _GOAL_Y - sy
    cd = math.sqrt(dx**2 + dy**2)
    if cd > 0:
        carry_ex = min(sx + _CARRY_DISTANCE * dx / cd, PITCH_W - 0.1)
        carry_ey = max(0.1, min(sy + _CARRY_DISTANCE * dy / cd, PITCH_H - 0.1))
    else:
        carry_ex, carry_ey = sx, sy

    end_positions = {
        'Shot':    (_GOAL_X, _GOAL_Y),
        'Carry':   (carry_ex, carry_ey),
        'Dribble': (sx, sy),
        'Pass':    [(p['location'][0], p['location'][1]) for p in tms_out],
    }

    return weights, end_positions, under_pressure


def _build_scalar_v2(sx, sy, action_type, ex, ey, under_pressure):
    from config import ACTION_TYPES, SCALAR_COLS, SCALAR_DIM
    idx = {col: i for i, col in enumerate(SCALAR_COLS)}
    vec = np.zeros(SCALAR_DIM, dtype=np.float32)

    if action_type in ACTION_TYPES:
        vec[ACTION_TYPES.index(action_type)] = 1.0

    vec[idx['start_x_norm']] = sx / PITCH_W
    vec[idx['start_y_norm']] = sy / PITCH_H

    dx = _GOAL_X - sx;  dy = _GOAL_Y - sy
    vec[idx['dist_to_goal_norm']]  = math.sqrt(dx**2 + dy**2) / _MAX_DIST
    vec[idx['angle_to_goal_norm']] = math.atan2(dy, dx) / math.pi

    vec[idx['under_pressure']]  = float(under_pressure)
    vec[idx['period_norm']]     = 0.25   # neutral: period 2
    vec[idx['score_diff_norm']] = 0.5    # neutral: level

    if action_type == 'Pass' and ex is not None:
        pdx = ex - sx;  pdy = ey - sy
        pl  = math.sqrt(pdx**2 + pdy**2)
        vec[idx['pass_length_norm']] = pl / math.sqrt(PITCH_W**2 + PITCH_H**2)
        vec[idx['pass_angle_norm']]  = math.atan2(pdy, pdx) / math.pi

    vec[idx['end_x_norm']] = (ex if ex is not None else sx) / PITCH_W
    vec[idx['end_y_norm']] = (ey if ey is not None else sy) / PITCH_H
    return vec


def _sweep_pitch_weighted(model, device, frame_data):
    """Weighted xT sweep: runs all 4 action types per cell, combines by action weights."""
    from config import SCALAR_DIM
    players = frame_data.get('players', []) if frame_data else []

    x_centres = np.linspace(PITCH_W / (2*GRID_W), PITCH_W - PITCH_W/(2*GRID_W), GRID_W)
    y_centres  = np.linspace(PITCH_H / (2*GRID_H), PITCH_H - PITCH_H/(2*GRID_H), GRID_H)
    xx, yy = np.meshgrid(x_centres, y_centres)
    N = GRID_H * GRID_W

    # Pass targets are fixed (players don't move per cell)
    pass_targets = [(p['location'][0], p['location'][1])
                    for p in players if p.get('teammate') and not p.get('keeper')]

    weights_arr     = np.zeros((N, len(ACTIONS)), dtype=np.float32)
    end_pos_arr     = np.full((N, len(ACTIONS), 2), np.nan, dtype=np.float32)
    under_press_arr = np.zeros(N, dtype=bool)

    for i in range(N):
        gy, gx = divmod(i, GRID_W)
        sx, sy = float(xx[gy, gx]), float(yy[gy, gx])
        w, ep, up = _action_context(sx, sy, players)
        under_press_arr[i] = up
        for ai, a in enumerate(ACTIONS):
            weights_arr[i, ai] = w[a]
            if a != 'Pass':
                ep_val = ep[a]
                end_pos_arr[i, ai] = ep_val if ep_val is not None else [sx, sy]

    # Base spatial: channels 0-3 and 5 are constant per cell; channel 4 varies per action
    base_spatial = np.zeros((N, NUM_CHANNELS, GRID_H, GRID_W), dtype=np.float32)
    for i in range(N):
        gy, gx = divmod(i, GRID_W)
        sx, sy = float(xx[gy, gx]), float(yy[gy, gx])
        ev = {'start_x': sx, 'start_y': sy, 'end_x': np.nan, 'end_y': np.nan}
        base_spatial[i] = encode_event(ev, frame_data)

    BATCH = 256
    action_xt = {}

    def _run_forward(spatial_a, scalar_a):
        all_probs = []
        with torch.no_grad():
            for start in range(0, N, BATCH):
                end_  = min(start + BATCH, N)
                sp_t  = torch.from_numpy(spatial_a[start:end_]).to(device)
                sc_t  = torch.from_numpy(scalar_a[start:end_]).to(device)
                logits = model(sp_t, sc_t)
                probs  = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs.tolist())
        return np.array(all_probs, dtype=np.float32).reshape(N)

    xs_grid = np.arange(GRID_W, dtype=np.float32)
    ys_grid = np.arange(GRID_H, dtype=np.float32)
    XX_g, YY_g = np.meshgrid(xs_grid, ys_grid)

    for ai, action in enumerate(ACTIONS):
        if action == 'Pass':
            if not pass_targets:
                action_xt['Pass'] = np.zeros(N, dtype=np.float32)
                continue
            candidate_probs = []
            for (tex, tey) in pass_targets:
                spatial_a = base_spatial.copy()
                scalar_a  = np.zeros((N, SCALAR_DIM), dtype=np.float32)
                gex = tex / PITCH_W * GRID_W
                gey = tey / PITCH_H * GRID_H
                end_ch = np.exp(-((XX_g - gex)**2 + (YY_g - gey)**2) / (2*GAUSSIAN_SIGMA**2))
                for i in range(N):
                    spatial_a[i, 4] = end_ch
                    gy, gx = divmod(i, GRID_W)
                    sx, sy = float(xx[gy, gx]), float(yy[gy, gx])
                    scalar_a[i] = _build_scalar_v2(sx, sy, 'Pass', tex, tey, bool(under_press_arr[i]))
                candidate_probs.append(_run_forward(spatial_a, scalar_a))
            action_xt['Pass'] = np.max(np.stack(candidate_probs), axis=0)
            continue

        spatial_a = base_spatial.copy()
        scalar_a  = np.zeros((N, SCALAR_DIM), dtype=np.float32)
        for i in range(N):
            gy, gx = divmod(i, GRID_W)
            sx, sy = float(xx[gy, gx]), float(yy[gy, gx])
            ex, ey = end_pos_arr[i, ai]
            if action != 'Dribble':
                gex = ex / PITCH_W * GRID_W
                gey = ey / PITCH_H * GRID_H
                spatial_a[i, 4] = np.exp(
                    -((XX_g - gex)**2 + (YY_g - gey)**2) / (2*GAUSSIAN_SIGMA**2)
                )
            scalar_a[i] = _build_scalar_v2(sx, sy, action, ex, ey, bool(under_press_arr[i]))
        action_xt[action] = _run_forward(spatial_a, scalar_a)

    z_flat = np.zeros(N, dtype=np.float32)
    for ai, action in enumerate(ACTIONS):
        z_flat += weights_arr[:, ai] * action_xt[action]

    return xx, yy, z_flat.reshape(GRID_H, GRID_W)


def generate_heatmap_weighted(model: nn.Module, device: torch.device) -> None:
    """Weighted xT heatmap with no players (baseline positional threat)."""
    print("[v2] Generating weighted baseline heatmap...")
    model.eval()
    xx, yy, z = _sweep_pitch_weighted(model, device, frame_data=None)
    out = os.path.join(os.path.dirname(HEATMAP_PATH), 'xt_heatmap_weighted_v2.png')
    _plot_heatmap_titled(xx, yy, z, 'SxT v2 — Weighted Action (No Players)', out)


def generate_all_scenarios_weighted(model: nn.Module, device: torch.device) -> None:
    """Scenario comparison grid using weighted action xT."""
    print("[v2] Generating weighted scenario heatmaps...")
    model.eval()

    n     = len(SCENARIOS)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 5))
    axes = axes.flatten()

    results    = {}
    global_max = 0.0
    for name, frame_data in SCENARIOS.items():
        print(f"  Scenario: {name}")
        xx, yy, z = _sweep_pitch_weighted(model, device, frame_data)
        results[name] = (xx, yy, z, frame_data)
        global_max = max(global_max, z.max())

    for ax, (name, (xx, yy, z, frame_data)) in zip(axes, results.items()):
        cf = ax.contourf(xx, yy, z, levels=30, cmap='magma', vmin=0, vmax=global_max)
        fig.colorbar(cf, ax=ax, fraction=0.03, pad=0.02)
        _draw_pitch(ax)

        if frame_data and 'players' in frame_data:
            for p in frame_data['players']:
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

    fig.suptitle('SxT v2 — Weighted Action Scenarios', fontsize=14, y=1.01)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(HEATMAP_PATH), 'xt_scenarios_weighted_v2.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_heatmap_titled(xx: np.ndarray, yy: np.ndarray, z: np.ndarray,
                         title: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    cf   = ax.contourf(xx, yy, z, levels=30, cmap='magma', vmin=0, vmax=z.max())
    cbar = fig.colorbar(cf, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Weighted xT', fontsize=11)
    _draw_pitch(ax)
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlim(0, PITCH_W)
    ax.set_ylim(PITCH_H, 0)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Heatmap saved to {out_path}")


def _plot_heatmap(xx: np.ndarray, yy: np.ndarray, z: np.ndarray, action_type: str) -> None:
    slug     = action_type.replace(' ', '_').replace('*', '').lower()
    out_path = HEATMAP_PATH.replace('_v2.png', f'_{slug}_v2.png')
    _plot_heatmap_titled(
        xx, yy, z,
        f'Spatial Expected Threat Map (CNN, action={action_type})',
        out_path,
    )


def _draw_pitch(ax: plt.Axes) -> None:
    """
    Draw standard football pitch markings on StatsBomb coordinates (120 × 80).
    All lines are white so they show clearly over the dark magma colormap.
    """
    lc = 'white'
    lw = 1.5

    # Outline & halfway line
    ax.plot([0,   0],   [0,  80], color=lc, linewidth=lw)
    ax.plot([120, 120], [0,  80], color=lc, linewidth=lw)
    ax.plot([0,  120],  [0,   0], color=lc, linewidth=lw)
    ax.plot([0,  120],  [80, 80], color=lc, linewidth=lw)
    ax.plot([60,  60],  [0,  80], color=lc, linewidth=lw)

    # Centre circle & spot
    circle = mpatches.Circle((60, 40), 10, color=lc, fill=False, linewidth=lw)
    ax.add_patch(circle)
    ax.scatter(60, 40, color=lc, s=15, zorder=5)

    # Penalty areas (18-yard boxes)
    ax.plot([18,  18],  [18, 62], color=lc, linewidth=lw)
    ax.plot([0,   18],  [18, 18], color=lc, linewidth=lw)
    ax.plot([0,   18],  [62, 62], color=lc, linewidth=lw)
    ax.plot([102, 102], [18, 62], color=lc, linewidth=lw)
    ax.plot([102, 120], [18, 18], color=lc, linewidth=lw)
    ax.plot([102, 120], [62, 62], color=lc, linewidth=lw)

    # 6-yard boxes
    ax.plot([6,   6],   [30, 50], color=lc, linewidth=lw)
    ax.plot([0,   6],   [30, 30], color=lc, linewidth=lw)
    ax.plot([0,   6],   [50, 50], color=lc, linewidth=lw)
    ax.plot([114, 114], [30, 50], color=lc, linewidth=lw)
    ax.plot([114, 120], [30, 30], color=lc, linewidth=lw)
    ax.plot([114, 120], [50, 50], color=lc, linewidth=lw)

    # Penalty spots
    ax.scatter(12,  40, color=lc, s=15, zorder=5)
    ax.scatter(108, 40, color=lc, s=15, zorder=5)
