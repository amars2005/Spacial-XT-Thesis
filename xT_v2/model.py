import torch
import torch.nn as nn

from config import NUM_CHANNELS, GRID_H, GRID_W, SCALAR_DIM


class CNNBranch(nn.Module):
    """
    Processes the spatial pitch image tensor of shape
    (batch, NUM_CHANNELS, GRID_H, GRID_W) = (batch, 6, 40, 60).

    Three conv-blocks with BatchNorm and MaxPool halve the spatial
    dimensions at each stage:
        Input  : (6,  40, 60)
        Block1 : (32, 20, 30)
        Block2 : (64, 10, 15)
        Block3 : (128, 5,  7)   ← floor(15/2) = 7
        Flatten: 128 * 5 * 7 = 4480
        Head   : 4480 → 256 (+ Dropout)
    """

    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Compute flattened size from config — handles arbitrary GRID_H/W
        h_out = GRID_H // 2 // 2 // 2   # 40 → 20 → 10 → 5
        w_out = GRID_W // 2 // 2 // 2   # 60 → 30 → 15 → 7
        self.flat_dim = 128 * h_out * w_out  # 4480

        self.head = nn.Sequential(
            nn.Linear(self.flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        return self.head(x)           # (batch, 256)


class MLPBranch(nn.Module):
    """
    Processes the scalar feature vector of shape (batch, SCALAR_DIM).
    Two fully-connected layers: SCALAR_DIM → 64 → 32.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(SCALAR_DIM, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)            # (batch, 32)


class XTModel(nn.Module):
    """
    Dual-branch Expected Threat (xT) model.

    Architecture
    ------------
    CNN branch  : pitch image  (6, 40, 60)  → 256-d vector
    MLP branch  : scalar feats (22,)         →  32-d vector
                                                ─────────────
    Concatenate                              → 288-d vector
    Classifier  : 288 → 128 → 1  (raw logit, no sigmoid)

    The output is a raw logit.  Use torch.sigmoid() at inference time,
    or pair with BCEWithLogitsLoss during training for numerical stability.
    """

    def __init__(self):
        super().__init__()
        self.cnn = CNNBranch()
        self.mlp = MLPBranch()

        combined_dim = 256 + 32  # CNN head output + MLP output

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, spatial: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        spatial : (batch, NUM_CHANNELS, GRID_H, GRID_W)
        scalar  : (batch, SCALAR_DIM)

        Returns
        -------
        logits  : (batch,)  — raw (pre-sigmoid) scores
        """
        cnn_feat = self.cnn(spatial)                     # (batch, 256)
        mlp_feat = self.mlp(scalar)                      # (batch, 32)
        combined = torch.cat([cnn_feat, mlp_feat], dim=1)  # (batch, 288)
        return self.classifier(combined).squeeze(1)      # (batch,)
