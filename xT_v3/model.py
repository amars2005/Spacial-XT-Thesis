"""
xT_v3 two-stage model.

Stage 1  — Frozen xT_v2 CNN (position-based threat estimate).
Stage 2  — Small Transformer that reads player tokens from a 360 freeze frame
           and outputs a residual adjustment on top of Stage 1's logit.

Forward pass returns a single logit (pre-sigmoid), same as xT_v2, so the same
BCEWithLogitsLoss training loop applies.
"""
import sys
import os
import importlib.util

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Resolve imports
# ---------------------------------------------------------------------------
_HERE   = os.path.dirname(os.path.abspath(__file__))
_V2_DIR = os.path.join(_HERE, '..', 'xT_v2')
if _V2_DIR not in sys.path:
    sys.path.insert(0, _V2_DIR)

from model import XTModel  # noqa: E402 — v2 CNN

# Load v3 config by file path to avoid v2/config shadowing it on sys.path
_spec  = importlib.util.spec_from_file_location("_v3config", os.path.join(_HERE, "config.py"))
_v3cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v3cfg)

PLAYER_DIM        = _v3cfg.PLAYER_DIM
MAX_PLAYERS       = _v3cfg.MAX_PLAYERS
D_MODEL           = _v3cfg.D_MODEL
N_HEADS           = _v3cfg.N_HEADS
N_LAYERS          = _v3cfg.N_LAYERS
BALL_DIM          = _v3cfg.BALL_DIM
BALL_FEAT_INDICES = _v3cfg.BALL_FEAT_INDICES


# ---------------------------------------------------------------------------
# Stage 2: context-adjustment Transformer
# ---------------------------------------------------------------------------

class Stage2Model(nn.Module):
    """
    Reads player tokens from a 360 freeze frame and produces a scalar
    residual adjustment for Stage 1's threat logit.

    Inputs
    ------
    stage1_logit  : (B, 1)              — pre-sigmoid logit from frozen Stage 1
    ball_feats    : (B, BALL_DIM)       — [start_x_norm, start_y_norm, dist_to_goal, angle_to_goal]
    player_tokens : (B, MAX_PLAYERS, PLAYER_DIM)
                    — [x_norm, y_norm, is_teammate, is_keeper, is_actor] per player row
    player_mask   : (B, MAX_PLAYERS) bool
                    — True = padding position (player slot is empty); ignored in attention

    Output
    ------
    adjusted_logit : (B, 1)   — stage1_logit + learned context adjustment
    """

    def __init__(self):
        super().__init__()

        # Project player tokens into D_MODEL space
        # Query is Stage 1's logit (scalar) — genuinely novel vs ball position features
        self.player_embed = nn.Linear(PLAYER_DIM, D_MODEL)
        self.query_embed  = nn.Linear(1,           D_MODEL)

        # Player self-attention: players reason about each other's positions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=D_MODEL * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.player_encoder = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)

        # Cross-attention: ball position queries the player context
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=D_MODEL,
            num_heads=N_HEADS,
            dropout=0.1,
            batch_first=True,
        )

        # Final MLP: fuses attended player context with Stage 1 logit
        self.mlp = nn.Sequential(
            nn.Linear(D_MODEL + 1, 64),  # +1 for stage1_logit
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

        # Small init so Stage 2 starts as a near-zero correction
        nn.init.normal_(self.mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        stage1_logit:  torch.Tensor,   # (B,) or (B, 1)
        player_tokens: torch.Tensor,   # (B, MAX_PLAYERS, PLAYER_DIM)
        player_mask:   torch.Tensor,   # (B, MAX_PLAYERS) bool
    ) -> torch.Tensor:

        # Normalise stage1_logit to (B, 1)
        if stage1_logit.dim() == 1:
            stage1_logit = stage1_logit.unsqueeze(1)

        # --- Embed players ---
        player_embeds = self.player_embed(player_tokens)        # (B, P, D)

        # Self-attention among players
        player_context = self.player_encoder(
            player_embeds,
            src_key_padding_mask=player_mask,
        )                                                        # (B, P, D)

        # --- Stage 1 logit queries player context ---
        # Asks: "given this threat estimate, what do the player positions add?"
        query = self.query_embed(stage1_logit).unsqueeze(1)     # (B, 1, D)
        attn_out, _ = self.cross_attn(
            query=query,
            key=player_context,
            value=player_context,
            key_padding_mask=player_mask,
        )                                                        # (B, 1, D)
        attn_out = attn_out.squeeze(1)                          # (B, D)

        # --- Fuse with Stage 1 logit ---
        combined   = torch.cat([attn_out, stage1_logit], dim=-1)  # (B, D+1)
        adjustment = self.mlp(combined)                            # (B, 1)

        return (stage1_logit + adjustment).squeeze(1)             # (B,)


# ---------------------------------------------------------------------------
# Full xT_v3 model (Stage 1 frozen + Stage 2 trainable)
# ---------------------------------------------------------------------------

class XTModelV3(nn.Module):
    """
    Wraps a frozen xT_v2 CNN (Stage 1) with a trainable Stage 2 Transformer.

    Only Stage 2 parameters are updated during training.
    """

    def __init__(self, stage1: XTModel):
        super().__init__()
        self.stage1 = stage1
        self.stage2 = Stage2Model()

        for param in self.stage1.parameters():
            param.requires_grad = False

    def forward(
        self,
        spatial:       torch.Tensor,   # (B, NUM_CHANNELS, GRID_H, GRID_W)
        scalar:        torch.Tensor,   # (B, SCALAR_DIM)
        player_tokens: torch.Tensor,   # (B, MAX_PLAYERS, PLAYER_DIM)
        player_mask:   torch.Tensor,   # (B, MAX_PLAYERS) bool
    ) -> torch.Tensor:

        with torch.no_grad():
            stage1_logit = self.stage1(spatial, scalar)     # (B,)

        return self.stage2(stage1_logit, player_tokens, player_mask)


# ---------------------------------------------------------------------------
# Ablation 1: mean-pool MLP (cross-attention replaced by masked mean-pooling)
# ---------------------------------------------------------------------------

class Stage2ModelMeanPoolMLP(nn.Module):
    """
    Ablation variant: replaces the cross-attention block with masked mean-pooling
    over player embeddings, concatenated with the Stage 1 logit and passed
    through the same 2-layer fusion MLP.

    Tests whether the attentional weighting over players is necessary, or
    whether any player-context aggregation achieves the same result.
    """

    def __init__(self):
        super().__init__()
        self.player_embed = nn.Linear(PLAYER_DIM, D_MODEL)
        self.mlp = nn.Sequential(
            nn.Linear(D_MODEL + 1, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
        nn.init.normal_(self.mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        stage1_logit:  torch.Tensor,   # (B,) or (B, 1)
        player_tokens: torch.Tensor,   # (B, MAX_PLAYERS, PLAYER_DIM)
        player_mask:   torch.Tensor,   # (B, MAX_PLAYERS) bool — True = padding
    ) -> torch.Tensor:

        if stage1_logit.dim() == 1:
            stage1_logit = stage1_logit.unsqueeze(1)   # (B, 1)

        player_embeds = self.player_embed(player_tokens)    # (B, P, D)

        # Masked mean-pool: exclude padding slots
        real = (~player_mask).float().unsqueeze(-1)         # (B, P, 1)
        n_real = real.sum(dim=1).clamp(min=1.0)             # (B, 1)
        pooled = (player_embeds * real).sum(dim=1) / n_real  # (B, D)

        combined   = torch.cat([pooled, stage1_logit], dim=-1)  # (B, D+1)
        adjustment = self.mlp(combined)                          # (B, 1)
        return (stage1_logit + adjustment).squeeze(1)            # (B,)


class XTModelV3MeanPool(nn.Module):
    """
    Ablation: mean-pool MLP replacing cross-attention in Stage 2.
    Same interface as XTModelV3 so the same training loop applies.
    """

    def __init__(self, stage1: XTModel):
        super().__init__()
        self.stage1 = stage1
        self.stage2 = Stage2ModelMeanPoolMLP()
        for param in self.stage1.parameters():
            param.requires_grad = False

    def forward(
        self,
        spatial:       torch.Tensor,
        scalar:        torch.Tensor,
        player_tokens: torch.Tensor,
        player_mask:   torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            stage1_logit = self.stage1(spatial, scalar)
        return self.stage2(stage1_logit, player_tokens, player_mask)


# ---------------------------------------------------------------------------
# Ablation 2: ball-only Stage 2 (no player tokens)
# ---------------------------------------------------------------------------

class Stage2ModelBallOnly(nn.Module):
    """
    Ablation variant: Stage 2 receives only the Stage 1 logit and 4 ball
    features — no player tokens, no self-attention, no cross-attention.

    Tests whether any improvement in V3 over V2 comes from player positional
    information, or merely from adding an extra learned correction layer.
    """

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1 + BALL_DIM, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )
        nn.init.normal_(self.mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        stage1_logit: torch.Tensor,   # (B,) or (B, 1)
        ball_feats:   torch.Tensor,   # (B, BALL_DIM)
    ) -> torch.Tensor:
        if stage1_logit.dim() == 1:
            stage1_logit = stage1_logit.unsqueeze(1)   # (B, 1)
        combined   = torch.cat([stage1_logit, ball_feats], dim=-1)  # (B, 5)
        adjustment = self.mlp(combined)                              # (B, 1)
        return (stage1_logit + adjustment).squeeze(1)                # (B,)


class XTModelV3BallOnly(nn.Module):
    """
    Ablation: Stage 2 uses only Stage 1 logit + 4 ball features, no player tokens.
    Same interface as XTModelV3; extracts ball features from the scalar array.
    """

    def __init__(self, stage1: XTModel):
        super().__init__()
        self.stage1 = stage1
        self.stage2 = Stage2ModelBallOnly()
        self.register_buffer(
            'ball_idx',
            torch.tensor(BALL_FEAT_INDICES, dtype=torch.long),
        )
        for param in self.stage1.parameters():
            param.requires_grad = False

    def forward(
        self,
        spatial:       torch.Tensor,
        scalar:        torch.Tensor,
        player_tokens: torch.Tensor,   # ignored
        player_mask:   torch.Tensor,   # ignored
    ) -> torch.Tensor:
        with torch.no_grad():
            stage1_logit = self.stage1(spatial, scalar)
        ball_feats = scalar[:, self.ball_idx]   # (B, BALL_DIM)
        return self.stage2(stage1_logit, ball_feats)
