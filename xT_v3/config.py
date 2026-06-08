import os
import sys
import importlib.util

# ---------------------------------------------------------------------------
# Load xT_v2 config by file path to avoid circular-import via name collision
# ---------------------------------------------------------------------------
_V2_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'xT_v2')
_spec   = importlib.util.spec_from_file_location("_v2config", os.path.join(_V2_DIR, "config.py"))
_v2cfg  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v2cfg)

PITCH_W      = _v2cfg.PITCH_W
PITCH_H      = _v2cfg.PITCH_H
GRID_W       = _v2cfg.GRID_W
GRID_H       = _v2cfg.GRID_H
NUM_CHANNELS = _v2cfg.NUM_CHANNELS
GAUSSIAN_SIGMA = _v2cfg.GAUSSIAN_SIGMA
ACTION_TYPES = _v2cfg.ACTION_TYPES
SCALAR_COLS  = _v2cfg.SCALAR_COLS
SCALAR_DIM   = _v2cfg.SCALAR_DIM
TRAIN_SPLIT  = _v2cfg.TRAIN_SPLIT
VAL_SPLIT    = _v2cfg.VAL_SPLIT
BATCH_SIZE   = _v2cfg.BATCH_SIZE
NUM_WORKERS  = _v2cfg.NUM_WORKERS

# ---------------------------------------------------------------------------
# Pitch / Grid  (same as v2 — re-exported for convenience)
# ---------------------------------------------------------------------------
# PITCH_W, PITCH_H, GRID_W, GRID_H, NUM_CHANNELS imported above

# ---------------------------------------------------------------------------
# Player token encoding
# ---------------------------------------------------------------------------
# Each player is encoded as:
#   [x_norm, y_norm, dx, dy, dist, is_teammate, is_keeper, is_actor]
# where dx/dy/dist are relative to the action start position (= actor/ball location).
PLAYER_DIM  = 8
MAX_PLAYERS = 22    # Max players in a freeze frame; extras are clipped, missing are padded

# Indices into SCALAR_COLS that describe ball position/context for the Stage2 query.
# Layout: 7 action one-hots + start_x_norm(7) + start_y_norm(8) + dist(9) + angle(10)
BALL_FEAT_INDICES = [
    SCALAR_COLS.index('start_x_norm'),
    SCALAR_COLS.index('start_y_norm'),
    SCALAR_COLS.index('dist_to_goal_norm'),
    SCALAR_COLS.index('angle_to_goal_norm'),
]
BALL_DIM = len(BALL_FEAT_INDICES)   # 4

# ---------------------------------------------------------------------------
# Stage 2 Transformer hyper-parameters
# ---------------------------------------------------------------------------
D_MODEL  = 64    # Embedding dimension for player / ball tokens
N_HEADS  = 4     # Attention heads (D_MODEL must be divisible by N_HEADS)
N_LAYERS = 2     # TransformerEncoder layers for player self-attention

# ---------------------------------------------------------------------------
# Training  (Stage 2 only — Stage 1 weights are frozen)
# ---------------------------------------------------------------------------
LEARNING_RATE_V3          = 5e-4   # Stage 2 (Transformer) — learns from scratch
LEARNING_RATE_V3_FINETUNE = 5e-5   # Stage 1 (CNN) — gentle fine-tune
WEIGHT_DECAY_V3  = 1e-4
NUM_EPOCHS_V3    = 40
BATCH_SIZE_V3    = 256

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))

# v2 best model is the frozen Stage 1
V2_BEST_MODEL_PATH = os.path.join(_V2_DIR, 'checkpoints', 'best_model.pt')

DATA_DIR_V3          = os.path.join(_HERE, 'data', 'matches')
CHECKPOINT_DIR_V3    = os.path.join(_HERE, 'checkpoints')
BEST_MODEL_PATH_V3   = os.path.join(CHECKPOINT_DIR_V3, 'best_model.pt')
TEST_MATCH_IDS_PATH  = os.path.join(CHECKPOINT_DIR_V3, 'test_match_ids.json')
HEATMAP_PATH_V3      = os.path.join(_HERE, '..', 'visualisations', 'xt_heatmap_v3.png')
METRICS_PATH_V3      = os.path.join(os.path.dirname(_HERE), 'metrics', 'metrics_v3.json')

# Ablation checkpoints and metrics
BEST_MODEL_PATH_V3_MLP    = os.path.join(CHECKPOINT_DIR_V3, 'best_model_mlp.pt')
BEST_MODEL_PATH_V3_BALL   = os.path.join(CHECKPOINT_DIR_V3, 'best_model_ball.pt')
METRICS_PATH_V3_MLP       = os.path.join(os.path.dirname(_HERE), 'metrics', 'metrics_v3_mlp.json')
METRICS_PATH_V3_BALL      = os.path.join(os.path.dirname(_HERE), 'metrics', 'metrics_v3_ball.json')

# Temperature scaling (post-hoc calibration)
TEMP_PATH_V3              = os.path.join(CHECKPOINT_DIR_V3, 'temperature.json')
METRICS_PATH_V3_CALIBRATED = os.path.join(os.path.dirname(_HERE), 'metrics', 'metrics_v3_calibrated.json')
