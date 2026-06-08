import os

# ---------------------------------------------------------------------------
# Pitch / Grid
# ---------------------------------------------------------------------------
PITCH_W = 120.0   # StatsBomb pitch width  (x-axis)
PITCH_H = 80.0    # StatsBomb pitch height (y-axis)
GRID_W  = 60      # Downsampled grid width
GRID_H  = 40      # Downsampled grid height

# ---------------------------------------------------------------------------
# Spatial Encoding
# ---------------------------------------------------------------------------
NUM_CHANNELS = 6        # ball_start | teammates | opponents | goalkeeper | action_end | visible_area
GAUSSIAN_SIGMA = 1.5    # Gaussian blob radius in grid units

# ---------------------------------------------------------------------------
# Action Types
# Order here defines the one-hot index in SCALAR_COLS — do not reorder.
# ---------------------------------------------------------------------------
ACTION_TYPES = [
    'Pass', 'Carry', 'Shot', 'Dribble',
    'Ball Receipt*', 'Pressure', 'Interception',
]

# ---------------------------------------------------------------------------
# Scalar Features
# Defines the fixed-size feature vector passed to the MLP branch.
# Order matters — indices are used directly in encoder and visualizer.
# ---------------------------------------------------------------------------
_action_cols = [f"is_{t.replace(' ', '_').replace('*', '').lower()}" for t in ACTION_TYPES]

SCALAR_COLS = _action_cols + [
    # Spatial context
    'start_x_norm', 'start_y_norm',
    'dist_to_goal_norm', 'angle_to_goal_norm',
    # Match context
    'under_pressure', 'period_norm', 'minute_norm', 'score_diff_norm',
    # Pass-specific (zero-filled for non-pass events)
    'pass_length_norm', 'pass_angle_norm', 'pass_height_norm',
    'is_pass_switch', 'is_through_ball',
    # End position
    'end_x_norm', 'end_y_norm',
]

SCALAR_DIM = len(SCALAR_COLS)  # 22

# ---------------------------------------------------------------------------
# Dataset Building
# ---------------------------------------------------------------------------
BUILD_WORKERS = 1     # Sequential by default — GitHub rate limits hurt parallel fetching

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE   = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
NUM_EPOCHS    = 50
TRAIN_SPLIT   = 0.70
VAL_SPLIT     = 0.15
# implied TEST_SPLIT = 0.15
# 0 on Windows (no fork support); increase on Linux GPU machine (e.g. 4-8)
NUM_WORKERS   = 16

# ---------------------------------------------------------------------------
# Paths  (all relative to this file's directory)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))

DATA_DIR        = os.path.join(_HERE, '..', 'xT_v3', 'data', 'matches')
CHECKPOINT_DIR  = os.path.join(_HERE, "checkpoints")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
HEATMAP_PATH    = os.path.join(_HERE, '..', 'visualisations', 'xt_heatmap_v2.png')
METRICS_PATH    = os.path.join(os.path.dirname(_HERE), 'metrics', 'metrics_v2.json')
