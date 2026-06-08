"""
xT v3 Frontend — Flask backend.

Loads the trained v3 model (frozen v2 CNN + Stage2 Transformer) and exposes
a /predict endpoint. All action context (type, end position, pressure) is
inferred automatically from player positions — the user only places players.

Run with:  python3 app.py
Then open: http://localhost:5001
"""
import json
import os
import sys
import math
import importlib.util

_HERE   = os.path.dirname(os.path.abspath(__file__))
_V2_DIR = os.path.join(_HERE, '..', '..', 'xT_v2')
_V3_DIR = os.path.join(_HERE, '..', '..', 'xT_v3')

# v2 must be on sys.path so XTModel is importable as 'model' inside v3/model.py
sys.path.insert(0, _V2_DIR)

import torch
import numpy as np
from flask import Flask, request, jsonify, render_template

# Load v2 model class via importlib (avoids name conflict with v3/model.py)
_spec_v2m = importlib.util.spec_from_file_location("_v2model", os.path.join(_V2_DIR, "model.py"))
_v2model  = importlib.util.module_from_spec(_spec_v2m)
_spec_v2m.loader.exec_module(_v2model)
XTModel = _v2model.XTModel

# Load v3 model via importlib
_spec_v3m = importlib.util.spec_from_file_location("_v3model", os.path.join(_V3_DIR, "model.py"))
_v3model  = importlib.util.module_from_spec(_spec_v3m)
_spec_v3m.loader.exec_module(_v3model)
XTModelV3 = _v3model.XTModelV3

# Load v2 config
_spec_cfg = importlib.util.spec_from_file_location("_v2cfg", os.path.join(_V2_DIR, "config.py"))
_v2cfg    = importlib.util.module_from_spec(_spec_cfg)
_spec_cfg.loader.exec_module(_v2cfg)

PITCH_W        = _v2cfg.PITCH_W       # 120
PITCH_H        = _v2cfg.PITCH_H       # 80
GRID_W         = _v2cfg.GRID_W        # 60
GRID_H         = _v2cfg.GRID_H        # 40
NUM_CHANNELS   = _v2cfg.NUM_CHANNELS  # 6
GAUSSIAN_SIGMA = _v2cfg.GAUSSIAN_SIGMA
SCALAR_COLS    = _v2cfg.SCALAR_COLS
SCALAR_DIM     = _v2cfg.SCALAR_DIM    # 22
ACTION_TYPES   = _v2cfg.ACTION_TYPES

V2_BEST_MODEL_PATH = os.path.join(_V2_DIR, 'checkpoints', 'best_model.pt')
V3_BEST_MODEL_PATH = os.path.join(_V3_DIR, 'checkpoints', 'best_model.pt')

MAX_PLAYERS = 22
PLAYER_DIM  = 8  # [x_norm, y_norm, dx_norm, dy_norm, dist_norm, is_teammate, is_keeper, is_actor]

BALL_FEAT_INDICES = [
    SCALAR_COLS.index('start_x_norm'),
    SCALAR_COLS.index('start_y_norm'),
    SCALAR_COLS.index('dist_to_goal_norm'),
    SCALAR_COLS.index('angle_to_goal_norm'),
]

_GOAL_X   = 120.0
_GOAL_Y   = 40.0
_POST_L   = 36.0
_POST_R   = 44.0
_MAX_DIST = math.sqrt(_GOAL_X**2 + _GOAL_Y**2)

# Empirical median carry distance — set by scripts/compute_carry_stats.py; falls back to 8.0
_carry_stats_path = os.path.join(_HERE, '..', '..', 'metrics', 'carry_stats.json')
CARRY_DISTANCE = 8.0
if os.path.exists(_carry_stats_path):
    with open(_carry_stats_path) as _f:
        CARRY_DISTANCE = json.load(_f).get("median_distance", 8.0)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

stage1 = XTModel()
stage1.load_state_dict(torch.load(V2_BEST_MODEL_PATH, map_location=device))

model = XTModelV3(stage1=stage1).to(device)
model.load_state_dict(torch.load(V3_BEST_MODEL_PATH, map_location=device))
model.eval()
print(f"v3 model loaded  (device: {device})")

# ---------------------------------------------------------------------------
# Spatial encoding helpers
# ---------------------------------------------------------------------------
_xs = np.arange(GRID_W, dtype=np.float32)
_ys = np.arange(GRID_H, dtype=np.float32)
_XX, _YY = np.meshgrid(_xs, _ys)   # (GRID_H, GRID_W)


def _gaussian(px: float, py: float) -> np.ndarray:
    gx = px / PITCH_W * GRID_W
    gy = py / PITCH_H * GRID_H
    return np.exp(-((_XX - gx) ** 2 + (_YY - gy) ** 2) / (2 * GAUSSIAN_SIGMA ** 2))


def _dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def _in_shooting_lane(ox, oy, bx, by):
    """Return True if opponent (ox,oy) is inside the shooting triangle ball→goal."""
    def cross(ax, ay, bx2, by2, px, py):
        return (bx2 - ax) * (py - ay) - (by2 - ay) * (px - ax)
    c1 = cross(bx, by, _GOAL_X, _POST_L, ox, oy)
    c2 = cross(_GOAL_X, _POST_L, _GOAL_X, _POST_R, ox, oy)
    c3 = cross(_GOAL_X, _POST_R, bx, by, ox, oy)
    return (c1 >= 0 and c2 >= 0 and c3 >= 0) or (c1 <= 0 and c2 <= 0 and c3 <= 0)


# ---------------------------------------------------------------------------
# Action context inference from player positions
# ---------------------------------------------------------------------------

def compute_action_context(sx: float, sy: float, players: list) -> dict:
    """
    Infer action weights, end positions, and under-pressure flag from
    the player positions placed on the pitch.

    Returns:
        {
          'under_pressure': bool,
          'weights': {'Pass': float, 'Carry': float, 'Dribble': float, 'Shot': float},
          'end_positions': {'Pass': [(ex,ey),...], 'Carry': (ex,ey), 'Dribble': (ex,ey), 'Shot': (ex,ey)},
        }
    """
    opponents = [p for p in players if not p.get('teammate', True) and not p.get('keeper', False)]
    goalkeepers_opp = [p for p in players if not p.get('teammate', True) and p.get('keeper', False)]
    all_opps = opponents + goalkeepers_opp
    teammates_outfield = [p for p in players if p.get('teammate', False) and not p.get('keeper', False)]

    # Goalside opponents (x >= ball_x)
    goalside = [p for p in all_opps if p['x'] >= sx]
    goalside_within_5  = [p for p in goalside if _dist(p['x'], p['y'], sx, sy) < 5]
    goalside_within_10 = [p for p in goalside if _dist(p['x'], p['y'], sx, sy) < 10]

    under_pressure = len(goalside_within_5) > 0

    # Defenders in shooting lane
    lane_defenders = sum(1 for p in all_opps if _in_shooting_lane(p['x'], p['y'], sx, sy))

    # Distance to goal
    dist_to_goal = _dist(sx, sy, _GOAL_X, _GOAL_Y)

    # ---- Weights ----
    shot_w    = math.exp(-dist_to_goal / 25.0) * max(0.0, 1.0 - lane_defenders / 3.0)
    pass_w    = min(len(teammates_outfield), 5) / 5.0 if teammates_outfield else 0.0
    carry_w   = max(0.0, 1.0 - len(goalside_within_10) / 4.0)
    dribble_w = len(goalside_within_5) * max(0.0, 1.0 - (len(goalside_within_5) - 1) / 3.0)

    total = shot_w + pass_w + carry_w + dribble_w
    total = max(total, 0.01)
    weights = {
        'Shot':    shot_w    / total,
        'Pass':    pass_w    / total,
        'Carry':   carry_w   / total,
        'Dribble': dribble_w / total,
    }

    # ---- End positions ----
    # Carry: project toward goal by empirical median carry distance (from training data)
    dx = _GOAL_X - sx;  dy = _GOAL_Y - sy
    carry_dist = math.sqrt(dx**2 + dy**2)
    if carry_dist > 0:
        carry_ex = min(sx + CARRY_DISTANCE * dx / carry_dist, PITCH_W - 0.1)
        carry_ey = max(0.1, min(sy + CARRY_DISTANCE * dy / carry_dist, PITCH_H - 0.1))
    else:
        carry_ex, carry_ey = sx, sy

    end_positions = {
        'Shot':    (_GOAL_X, _GOAL_Y),
        'Carry':   (carry_ex, carry_ey),
        'Dribble': (sx, sy),                                     # no end pos in StatsBomb
        'Pass':    [(p['x'], p['y']) for p in teammates_outfield],  # one per teammate
    }

    return {
        'under_pressure': under_pressure,
        'weights': weights,
        'end_positions': end_positions,
        'lane_defenders': lane_defenders,
        'goalside_close': len(goalside_within_5),
        'teammates_count': len(teammates_outfield),
    }


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def build_spatial(sx, sy, ex, ey, has_end: bool, players: list) -> np.ndarray:
    """Build (NUM_CHANNELS, GRID_H, GRID_W) spatial tensor."""
    spatial = np.zeros((NUM_CHANNELS, GRID_H, GRID_W), dtype=np.float32)
    spatial[0] = _gaussian(sx, sy)   # ch0: ball start

    for p in players:
        if p.get('keeper') and not p.get('teammate'):
            spatial[3] = np.maximum(spatial[3], _gaussian(p['x'], p['y']))
        elif p.get('teammate') and not p.get('keeper'):
            spatial[1] = np.maximum(spatial[1], _gaussian(p['x'], p['y']))
        elif not p.get('teammate') and not p.get('keeper'):
            spatial[2] = np.maximum(spatial[2], _gaussian(p['x'], p['y']))

    if has_end:
        spatial[4] = _gaussian(ex, ey)   # ch4: action end

    spatial[5] = np.ones((GRID_H, GRID_W), dtype=np.float32)   # ch5: visible area
    return spatial


def build_scalar(sx, sy, action_type: str, ex, ey, under_pressure: bool) -> np.ndarray:
    """Build (SCALAR_DIM,) scalar feature vector."""
    scalar = np.zeros(SCALAR_DIM, dtype=np.float32)

    if action_type in ACTION_TYPES:
        scalar[ACTION_TYPES.index(action_type)] = 1.0

    scalar[SCALAR_COLS.index('start_x_norm')] = sx / PITCH_W
    scalar[SCALAR_COLS.index('start_y_norm')] = sy / PITCH_H

    dx = _GOAL_X - sx;  dy = _GOAL_Y - sy
    scalar[SCALAR_COLS.index('dist_to_goal_norm')]  = math.sqrt(dx**2 + dy**2) / _MAX_DIST
    scalar[SCALAR_COLS.index('angle_to_goal_norm')] = math.atan2(dy, dx) / math.pi

    scalar[SCALAR_COLS.index('under_pressure')] = float(under_pressure)
    scalar[SCALAR_COLS.index('period_norm')]     = 0.25
    scalar[SCALAR_COLS.index('minute_norm')]     = 0.0
    scalar[SCALAR_COLS.index('score_diff_norm')] = 0.5

    if action_type == 'Pass':
        pdx = ex - sx;  pdy = ey - sy
        pl = math.sqrt(pdx**2 + pdy**2)
        scalar[SCALAR_COLS.index('pass_length_norm')] = pl / math.sqrt(PITCH_W**2 + PITCH_H**2)
        scalar[SCALAR_COLS.index('pass_angle_norm')]  = math.atan2(pdy, pdx) / math.pi

    scalar[SCALAR_COLS.index('end_x_norm')] = ex / PITCH_W
    scalar[SCALAR_COLS.index('end_y_norm')] = ey / PITCH_H
    return scalar


def build_player_tokens(sx, sy, players: list):
    """Build (MAX_PLAYERS, PLAYER_DIM) tokens and (MAX_PLAYERS,) padding mask."""
    tokens = np.zeros((MAX_PLAYERS, PLAYER_DIM), dtype=np.float32)
    mask   = np.ones(MAX_PLAYERS, dtype=bool)

    ax = sx / PITCH_W
    ay = sy / PITCH_H
    tokens[0] = [ax, ay, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]  # ball carrier = actor
    mask[0]   = False

    for i, p in enumerate(players[:MAX_PLAYERS - 1], start=1):
        px = p['x'] / PITCH_W
        py = p['y'] / PITCH_H
        dx = px - ax
        dy = py - ay
        dist = math.sqrt(dx * dx + dy * dy)
        tokens[i] = [
            px, py, dx, dy, dist,
            float(p.get('teammate', False)),
            float(p.get('keeper', False)),
            0.0,
        ]
        mask[i] = False

    return tokens, mask


def run_model(sx, sy, action_type, ex, ey, has_end, under_pressure, players) -> float:
    """Run v3 model for one action configuration and return xT (sigmoid output)."""
    spatial = build_spatial(sx, sy, ex, ey, has_end, players)
    scalar  = build_scalar(sx, sy, action_type, ex, ey, under_pressure)
    tokens, pmask = build_player_tokens(sx, sy, players)

    s_t = torch.from_numpy(spatial).unsqueeze(0).to(device)
    c_t = torch.from_numpy(scalar).unsqueeze(0).to(device)
    p_t = torch.from_numpy(tokens).unsqueeze(0).to(device)
    m_t = torch.from_numpy(pmask).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(s_t, c_t, p_t, m_t)
        return float(torch.sigmoid(logit).item())


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data        = request.get_json()
    sx          = float(data.get('start_x', 60))
    sy          = float(data.get('start_y', 40))
    players     = data.get('players', [])
    action_type = data.get('action_type', 'Shot')
    if action_type not in ('Shot', 'Pass', 'Carry', 'Dribble'):
        action_type = 'Shot'

    ctx = compute_action_context(sx, sy, players)
    under_pressure = ctx['under_pressure']
    end_positions  = ctx['end_positions']

    results = {}

    ex, ey = end_positions['Shot']
    results['Shot'] = run_model(sx, sy, 'Shot', ex, ey, True, under_pressure, players)

    ex, ey = end_positions['Carry']
    results['Carry'] = run_model(sx, sy, 'Carry', ex, ey, True, under_pressure, players)

    # Dribble — no end position (matches StatsBomb training distribution)
    results['Dribble'] = run_model(sx, sy, 'Dribble', sx, sy, False, under_pressure, players)

    # Pass — run once per teammate, take max xT over targets
    best_pass_xt  = 0.0
    best_pass_pos = None
    for (tex, tey) in end_positions['Pass']:
        xt = run_model(sx, sy, 'Pass', tex, tey, True, under_pressure, players)
        if xt > best_pass_xt:
            best_pass_xt  = xt
            best_pass_pos = (round(tex, 1), round(tey, 1))
    results['Pass'] = best_pass_xt

    scores = [
        {'action': a, 'xt': round(results[a], 5), 'selected': a == action_type}
        for a in ['Shot', 'Pass', 'Carry', 'Dribble']
    ]

    return jsonify({
        'xt':             round(results[action_type], 6),
        'action_type':    action_type,
        'under_pressure': under_pressure,
        'best_pass_pos':  best_pass_pos if action_type == 'Pass' else None,
        'scores':         scores,
    })


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5001)
