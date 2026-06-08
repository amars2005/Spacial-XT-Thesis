import numpy as np
from matplotlib.path import Path

from config import GRID_W, GRID_H, PITCH_W, PITCH_H, GAUSSIAN_SIGMA, NUM_CHANNELS


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pitch_to_grid(x: float, y: float) -> tuple[float, float]:
    """
    Map continuous StatsBomb pitch coordinates (x in [0,120], y in [0,80])
    to continuous grid coordinates (gx in [0,GRID_W], gy in [0,GRID_H]).
    Returns floats so Gaussian placement is sub-pixel accurate.
    """
    gx = np.clip(x / PITCH_W * GRID_W, 0.0, GRID_W - 1e-6)
    gy = np.clip(y / PITCH_H * GRID_H, 0.0, GRID_H - 1e-6)
    return gx, gy


def _add_gaussian(channel: np.ndarray, x: float, y: float,
                  sigma: float = GAUSSIAN_SIGMA) -> None:
    """
    Accumulate a 2D Gaussian blob (peak = 1) at pitch coordinates (x, y)
    into a (GRID_H, GRID_W) channel array in-place.
    Multiple blobs (e.g. several teammates) are summed, then the channel
    is clipped to [0, 1] before saving — done at the encode_event level.
    """
    gx, gy = _pitch_to_grid(x, y)
    xs = np.arange(GRID_W, dtype=np.float32)
    ys = np.arange(GRID_H, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)  # (GRID_H, GRID_W)
    blob = np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2.0 * sigma ** 2))
    channel += blob


def _render_visible_area(visible_area: list) -> np.ndarray:
    """
    Rasterise the visible-area polygon into a binary (GRID_H, GRID_W) mask.

    If no polygon is provided (sparse / missing 360 data) we return all-ones
    so the model sees "unknown coverage" as fully visible rather than fully
    hidden. This prevents the model from learning to ignore missing frames.
    """
    mask = np.ones((GRID_H, GRID_W), dtype=np.float32)

    if not visible_area or len(visible_area) < 3:
        return mask   # Assume full visibility when data is absent

    # Scale polygon vertices from pitch coords to grid coords
    scaled = [
        (x / PITCH_W * GRID_W, y / PITCH_H * GRID_H)
        for x, y in visible_area
    ]
    path = Path(scaled)

    # Test cell centres (offset by 0.5 to hit cell midpoints)
    xs = np.arange(GRID_W, dtype=np.float32) + 0.5
    ys = np.arange(GRID_H, dtype=np.float32) + 0.5
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    inside = path.contains_points(points).reshape(GRID_H, GRID_W)
    mask = inside.astype(np.float32)
    return mask


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encode_event(event_row: dict, frame_data: dict | None) -> np.ndarray:
    """
    Encode a single event into a spatial tensor of shape
    (NUM_CHANNELS, GRID_H, GRID_W) = (6, 40, 60).

    Channel layout
    --------------
    0  Ball start position        — Gaussian at (start_x, start_y)
    1  Teammate positions          — Gaussian per teammate (excl. actor)
    2  Opponent positions          — Gaussian per opponent
    3  Goalkeeper position         — Gaussian at keeper location
    4  Action end position         — Gaussian at (end_x, end_y)
    5  Visible area mask           — Binary polygon rasterisation

    Parameters
    ----------
    event_row : dict
        Must contain 'start_x', 'start_y', 'end_x', 'end_y' (NaN where absent).
    frame_data : dict or None
        {'players': [{'location', 'teammate', 'actor', 'keeper'}, ...],
         'visible_area': [[x, y], ...]}
        Pass None when no 360 frame is available for this event.
    """
    channels = [np.zeros((GRID_H, GRID_W), dtype=np.float32)
                for _ in range(NUM_CHANNELS)]

    # --- Channel 0: Ball start ---
    sx = event_row.get('start_x', np.nan)
    sy = event_row.get('start_y', np.nan)
    if sx is not None and not np.isnan(float(sx)):
        _add_gaussian(channels[0], float(sx), float(sy))

    # --- Channel 4: Action end ---
    ex = event_row.get('end_x', np.nan)
    ey = event_row.get('end_y', np.nan)
    if ex is not None and not np.isnan(float(ex)):
        _add_gaussian(channels[4], float(ex), float(ey))

    # --- Channels 1-3: Players from 360 freeze frame ---
    players = frame_data.get('players', []) if frame_data else []
    for player in players:
        loc = player.get('location')
        if not isinstance(loc, list) or len(loc) < 2:
            continue
        px, py = float(loc[0]), float(loc[1])

        if player.get('keeper', False):
            _add_gaussian(channels[3], px, py)
        elif player.get('teammate', False) and not player.get('actor', False):
            _add_gaussian(channels[1], px, py)
        elif not player.get('teammate', False):
            _add_gaussian(channels[2], px, py)

    # --- Channel 5: Visible area ---
    visible_area = frame_data.get('visible_area', []) if frame_data else []
    channels[5] = _render_visible_area(visible_area)

    # Clip player-density channels to [0, 1] so overlapping blobs don't
    # dominate; the model sees presence/density, not raw blob sums.
    for i in range(1, 4):
        channels[i] = np.clip(channels[i], 0.0, 1.0)

    return np.stack(channels, axis=0)  # (NUM_CHANNELS, GRID_H, GRID_W)


def build_inference_scalar(start_x: float, start_y: float,
                           action_type: str = 'Pass') -> np.ndarray:
    """
    Build a minimal scalar feature vector for pitch-sweep inference
    (used by the visualiser to generate the xT heatmap).

    Only the features derivable from position and action type are set;
    all context features (under_pressure, score_diff, etc.) are left at 0.
    """
    from config import ACTION_TYPES, SCALAR_COLS, SCALAR_DIM

    vec = np.zeros(SCALAR_DIM, dtype=np.float32)

    # Action-type one-hot
    if action_type in ACTION_TYPES:
        vec[ACTION_TYPES.index(action_type)] = 1.0

    # Spatial
    idx = {col: i for i, col in enumerate(SCALAR_COLS)}

    vec[idx['start_x_norm']] = start_x / PITCH_W
    vec[idx['start_y_norm']] = start_y / PITCH_H

    goal_x, goal_y = 120.0, 40.0
    dx = goal_x - start_x
    dy = goal_y - start_y
    max_dist = np.sqrt(goal_x ** 2 + goal_y ** 2)
    vec[idx['dist_to_goal_norm']]  = np.sqrt(dx ** 2 + dy ** 2) / max_dist
    vec[idx['angle_to_goal_norm']] = np.arctan2(dy, dx) / np.pi

    # End position defaults to start for sweep inference
    vec[idx['end_x_norm']] = start_x / PITCH_W
    vec[idx['end_y_norm']] = start_y / PITCH_H

    return vec
