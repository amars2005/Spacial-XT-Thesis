"""Generate EDA plots for the report."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "statsbomb_chained_dataset.csv")
print("Loading dataset...")
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df):,} events")

# ── 1. Class balance bar chart ────────────────────────────────────────────────
goal_count    = (df['chain_goal'] == 1).sum()
ngoal_count   = (df['chain_goal'] == 0).sum()
total         = len(df)
goal_pct      = 100 * goal_count / total
ngoal_pct     = 100 * ngoal_count / total

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(
    ['Goal Chain', 'No Goal Chain'],
    [goal_count, ngoal_count],
    color=['#e07b39', '#4c72b0'],
    edgecolor='white', linewidth=0.8
)
ax.set_ylabel('Number of Events', fontsize=11)
ax.set_title('Goal-Chain vs Background Event Distribution', fontsize=12, fontweight='bold')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))
for bar, pct in zip(bars, [goal_pct, ngoal_pct]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
            f'{pct:.2f}%', ha='center', va='bottom', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_class_balance.png"), dpi=150)
plt.close()
print("Saved eda_class_balance.png")

# ── 2. Event type distribution ────────────────────────────────────────────────
type_counts = df['type'].value_counts()
# keep top 8 types for readability
top_types = type_counts.head(8)

fig, ax = plt.subplots(figsize=(8, 4))
colors = plt.cm.tab10(np.linspace(0, 1, len(top_types)))
bars = ax.barh(top_types.index[::-1], top_types.values[::-1], color=colors[::-1],
               edgecolor='white', linewidth=0.6)
ax.set_xlabel('Number of Events', fontsize=11)
ax.set_title('Distribution of On-Ball Event Types', fontsize=12, fontweight='bold')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_event_types.png"), dpi=150)
plt.close()
print("Saved eda_event_types.png")

# ── 3. Spatial density of all events ─────────────────────────────────────────
valid = df.dropna(subset=['start_x', 'start_y'])
x = valid['start_x'].values
y = valid['start_y'].values

fig, ax = plt.subplots(figsize=(10, 6))
h = ax.hist2d(x, y, bins=[60, 40], range=[[0, 120], [0, 80]],
              cmap='YlOrRd', norm=LogNorm())
plt.colorbar(h[3], ax=ax, label='Event count (log scale)')

# Pitch markings
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xlim(0, 120); ax.set_ylim(0, 80)
ax.set_xlabel('Pitch x (m)', fontsize=11); ax.set_ylabel('Pitch y (m)', fontsize=11)
ax.set_title('Spatial Density of All Events Across the Pitch', fontsize=12, fontweight='bold')
# Add basic pitch lines
ax.axvline(60, color='gray', lw=0.8, ls='--', alpha=0.5)
ax.add_patch(mpatches.Rectangle((102, 18), 18, 44, fill=False, edgecolor='gray', lw=1))
ax.add_patch(mpatches.Rectangle((0, 18), 18, 44, fill=False, edgecolor='gray', lw=1))
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_spatial_density.png"), dpi=150)
plt.close()
print("Saved eda_spatial_density.png")

# ── 4. Goal-chain vs background event locations side by side ──────────────────
goal_df  = valid[valid['chain_goal'] == 1]
bkgd_df  = valid[valid['chain_goal'] == 0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
titles = ['Goal-Chain Events', 'Background Events']
dfs    = [goal_df, bkgd_df]

for ax, sub, title in zip(axes, dfs, titles):
    h = ax.hist2d(sub['start_x'].values, sub['start_y'].values,
                  bins=[60, 40], range=[[0, 120], [0, 80]],
                  cmap='YlOrRd', norm=LogNorm())
    plt.colorbar(h[3], ax=ax, label='Count (log scale)')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(0, 120); ax.set_ylim(0, 80)
    ax.set_xlabel('Pitch x (m)', fontsize=10)
    ax.set_ylabel('Pitch y (m)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axvline(60, color='gray', lw=0.8, ls='--', alpha=0.5)
    ax.add_patch(mpatches.Rectangle((102, 18), 18, 44, fill=False, edgecolor='gray', lw=1))

plt.suptitle('Spatial Distribution: Goal-Chain vs Background Events', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_goal_chain_locs.png"), dpi=150)
plt.close()
print("Saved eda_goal_chain_locs.png")

# ── 5. Goal rate by event type ────────────────────────────────────────────────
type_goal = df.groupby('type')['chain_goal'].mean().sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#e07b39' if v > type_goal.median() else '#4c72b0' for v in type_goal.values]
bars = ax.barh(type_goal.index, type_goal.values * 100, color=colors,
               edgecolor='white', linewidth=0.6)
ax.set_xlabel('Goal-Chain Rate (%)', fontsize=11)
ax.set_title('Goal-Chain Rate by Event Type', fontsize=12, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, val in zip(bars, type_goal.values):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
            f'{val*100:.2f}%', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_goal_rate_by_type.png"), dpi=150)
plt.close()
print("Saved eda_goal_rate_by_type.png")

# ── 6. Goal rate by pitch zone ────────────────────────────────────────────────
valid = df.dropna(subset=['start_x', 'start_y'])
x_bins = np.linspace(0, 120, 7)   # 6 columns
y_bins = np.linspace(0, 80,  5)   # 4 rows
zone_rate = np.zeros((4, 6))
zone_count = np.zeros((4, 6))

for _, row in valid.iterrows():
    xi = min(int((row['start_x'] / 120) * 6), 5)
    yi = min(int((row['start_y'] / 80)  * 4), 3)
    zone_rate[yi, xi]  += row['chain_goal']
    zone_count[yi, xi] += 1

with np.errstate(invalid='ignore'):
    zone_pct = np.where(zone_count > 0, zone_rate / zone_count * 100, np.nan)

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(zone_pct, cmap='RdYlGn', aspect='auto',
               extent=[0, 120, 0, 80], origin='lower', vmin=0)
plt.colorbar(im, ax=ax, label='Goal-Chain Rate (%)')
ax.set_xlabel('Pitch x (m)', fontsize=11)
ax.set_ylabel('Pitch y (m)', fontsize=11)
ax.set_title('Goal-Chain Rate by Pitch Zone', fontsize=12, fontweight='bold')
ax.axvline(40,  color='white', lw=1, ls='--', alpha=0.6)
ax.axvline(80,  color='white', lw=1, ls='--', alpha=0.6)
ax.add_patch(mpatches.Rectangle((102, 18), 18, 44, fill=False, edgecolor='white', lw=1.5))
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_goal_rate_by_zone.png"), dpi=150)
plt.close()
print("Saved eda_goal_rate_by_zone.png")

# ── 7. Possession chain length distribution ───────────────────────────────────
chain_lengths = df.groupby(['match_id', 'chain_id']).agg(
    length=('index', 'count'),
    is_goal=('chain_goal', 'max')
).reset_index()

goal_chains   = chain_lengths[chain_lengths['is_goal'] == 1]['length']
ngoal_chains  = chain_lengths[chain_lengths['is_goal'] == 0]['length']

fig, ax = plt.subplots(figsize=(9, 5))
bins = np.arange(1, min(goal_chains.max(), ngoal_chains.max(), 40) + 2)
ax.hist(ngoal_chains.clip(upper=40), bins=bins, density=True,
        alpha=0.7, color='#4c72b0', label='No-Goal Chains', edgecolor='white', lw=0.4)
ax.hist(goal_chains.clip(upper=40),  bins=bins, density=True,
        alpha=0.7, color='#e07b39', label='Goal Chains',    edgecolor='white', lw=0.4)
ax.set_xlabel('Chain Length (events)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Possession Chain Length Distribution', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_chain_lengths.png"), dpi=150)
plt.close()
print("Saved eda_chain_lengths.png")

# ── 8. Action type breakdown by pitch third ───────────────────────────────────
valid['pitch_third'] = pd.cut(
    valid['start_x'],
    bins=[0, 40, 80, 120],
    labels=['Defensive Third', 'Middle Third', 'Attacking Third']
)
top_types_list = df['type'].value_counts().head(6).index.tolist()
third_type = (valid[valid['type'].isin(top_types_list)]
              .groupby(['pitch_third', 'type'], observed=True)
              .size()
              .unstack(fill_value=0))
third_type_pct = third_type.div(third_type.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(9, 5))
colors_t = plt.cm.tab10(np.linspace(0, 1, len(third_type_pct.columns)))
bottom = np.zeros(len(third_type_pct))
for col, color in zip(third_type_pct.columns, colors_t):
    ax.bar(third_type_pct.index, third_type_pct[col], bottom=bottom,
           label=col, color=color, edgecolor='white', linewidth=0.5)
    bottom += third_type_pct[col].values
ax.set_ylabel('Proportion of Events (%)', fontsize=11)
ax.set_title('Event Type Distribution by Pitch Third', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=9, framealpha=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_action_by_third.png"), dpi=150)
plt.close()
print("Saved eda_action_by_third.png")

# ── 360 EDA: load v3 npz player tokens ───────────────────────────────────────
print("\nLoading v3 freeze-frame player tokens...")
V3_DIR = os.path.join(os.path.dirname(__file__), "..", "xT_v3", "data", "matches")
all_tokens = []
all_masks  = []
for fname in sorted(os.listdir(V3_DIR)):
    if not fname.endswith('.npz'):
        continue
    d = np.load(os.path.join(V3_DIR, fname))
    all_tokens.append(d['player_tokens'])   # (N, 22, 8)
    all_masks.append(d['player_mask'])       # (N, 22) bool — True = padding
tokens = np.concatenate(all_tokens, axis=0)  # (Total_N, 22, 8)
masks  = np.concatenate(all_masks,  axis=0)  # (Total_N, 22)
print(f"Loaded {len(tokens):,} freeze-frame events")

# token columns: [x_norm, y_norm, dx, dy, dist, is_teammate, is_keeper, is_actor]
valid_mask = ~masks  # True = real player

px = tokens[:, :, 0][valid_mask] * 120
py = tokens[:, :, 1][valid_mask] * 80
is_teammate = tokens[:, :, 5][valid_mask].astype(bool)

# ── 9. Player density heatmap (all players) ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
h = ax.hist2d(px, py, bins=[60, 40], range=[[0, 120], [0, 80]],
              cmap='Blues', norm=LogNorm())
plt.colorbar(h[3], ax=ax, label='Player count (log scale)')
ax.set_xlim(0, 120); ax.set_ylim(0, 80)
ax.set_xlabel('Pitch x (m)', fontsize=11)
ax.set_ylabel('Pitch y (m)', fontsize=11)
ax.set_title('Freeze-Frame Player Density (All Players)', fontsize=12, fontweight='bold')
ax.axvline(60, color='gray', lw=0.8, ls='--', alpha=0.5)
ax.add_patch(mpatches.Rectangle((102, 18), 18, 44, fill=False, edgecolor='gray', lw=1))
ax.add_patch(mpatches.Rectangle((0,   18), 18, 44, fill=False, edgecolor='gray', lw=1))
for sp in ax.spines.values():
    sp.set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_player_density.png"), dpi=150)
plt.close()
print("Saved eda_player_density.png")

# ── 10. Teammate vs opponent spatial distribution ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, flag, title, cmap in zip(
    axes,
    [True,  False],
    ['Attacking Team (Possession)', 'Defending Team'],
    ['Oranges', 'Blues']
):
    sel = (is_teammate == flag)
    h = ax.hist2d(px[sel], py[sel], bins=[60, 40], range=[[0, 120], [0, 80]],
                  cmap=cmap, norm=LogNorm())
    plt.colorbar(h[3], ax=ax, label='Count (log scale)')
    ax.set_xlim(0, 120); ax.set_ylim(0, 80)
    ax.set_xlabel('Pitch x (m)', fontsize=10)
    ax.set_ylabel('Pitch y (m)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axvline(60, color='gray', lw=0.8, ls='--', alpha=0.5)
    ax.add_patch(mpatches.Rectangle((102, 18), 18, 44, fill=False, edgecolor='gray', lw=1))
    for sp in ax.spines.values():
        sp.set_visible(False)
plt.suptitle('Player Positions During Freeze-Frame Events', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_team_positions.png"), dpi=150)
plt.close()
print("Saved eda_team_positions.png")

# ── 11. Visible player count distribution ────────────────────────────────────
player_counts = valid_mask.sum(axis=1)  # players visible per event
fig, ax = plt.subplots(figsize=(8, 5))
count_vals, count_bins = np.unique(player_counts, return_counts=True)
ax.bar(count_vals, count_bins, color='#4c72b0', edgecolor='white', linewidth=0.6)
ax.axvline(player_counts.mean(), color='#e07b39', lw=2, ls='--',
           label=f'Mean: {player_counts.mean():.1f} players')
ax.set_xlabel('Number of Visible Players per Event', fontsize=11)
ax.set_ylabel('Number of Events', fontsize=11)
ax.set_title('Visible Player Count in 360 Freeze Frames', fontsize=12, fontweight='bold')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e3:.0f}K' if x >= 1000 else str(int(x))))
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_player_count.png"), dpi=150)
plt.close()
print("Saved eda_player_count.png")

print("\nAll EDA plots generated.")
