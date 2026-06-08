"""Time a single training epoch at a given batch size."""
import os, sys, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from builder import DatasetBuilder
from dataset import XTDataset
from model import XTModel

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, required=True)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

builder = DatasetBuilder()
spatial, scalar, labels, match_ids = builder.load_all()

# Use same fixed split
split_path = os.path.join(_HERE, '..', 'xT_v3', 'checkpoints', 'split_match_ids.json')
with open(split_path) as f:
    split = json.load(f)
train_ids = set(split["train"])
mask = np.array([m in train_ids for m in match_ids])

loader = DataLoader(
    XTDataset(spatial[mask], scalar[mask], labels[mask]),
    batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True,
)
print(f"Batch size: {args.batch_size}  |  Batches/epoch: {len(loader):,}")

model = XTModel().to(device)
pos_rate = labels[mask].mean()
pos_weight = torch.tensor([min((1 - pos_rate) / (pos_rate + 1e-8), 10.0)]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Warmup: one mini-batch
spatial_b, scalar_b, labels_b = next(iter(loader))
model(spatial_b.to(device), scalar_b.to(device))
torch.cuda.synchronize()

start = time.perf_counter()
model.train()
for spatial_b, scalar_b, labels_b in loader:
    spatial_b = spatial_b.to(device, non_blocking=True)
    scalar_b  = scalar_b.to(device,  non_blocking=True)
    labels_b  = labels_b.to(device,  non_blocking=True)
    optimizer.zero_grad()
    loss = criterion(model(spatial_b, scalar_b), labels_b)
    loss.backward()
    optimizer.step()
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

print(f"1 epoch time: {elapsed:.1f}s  ({elapsed/60:.2f} min)")
print(f"Projected 50 epochs: {elapsed*50/60:.1f} min")
