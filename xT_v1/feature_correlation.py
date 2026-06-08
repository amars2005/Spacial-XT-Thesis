"""Generate feature correlation heatmap for Version 1 features."""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
from model import _build_team_form

CSV_PATH   = os.path.join(_ROOT, "statsbomb_chained_dataset.csv")
OUTPUT_PNG = os.path.join(_HERE, "xt_feature_correlation.png")

FEATURES = [
    'start_x', 'start_y', 'dist_to_goal', 'angle_to_goal',
    'type_Pass', 'type_Carry',
    'period', 'minute', 'score_diff', 'team_form',
]
LABELS = [
    'Start X', 'Start Y', 'Dist to Goal', 'Angle to Goal',
    'Pass', 'Carry',
    'Period', 'Minute', 'Score Diff', 'Team Form',
]

print("Loading dataset...")
df = pd.read_csv(CSV_PATH)
df['dx'] = 120 - df['start_x']
df['dy'] = 40  - df['start_y']
df['dist_to_goal']  = np.sqrt(df['dx']**2 + df['dy']**2)
df['angle_to_goal'] = np.arctan2(df['dy'], df['dx'])

# Score differential
df = df.sort_values(['match_id', 'period', 'index']).reset_index(drop=True)
df['_is_goal'] = ((df['type'] == 'Shot') & (df['success'] == 1)).astype(int)
df['_team_goals_before']  = df.groupby(['match_id', 'team_id'])['_is_goal'].cumsum() - df['_is_goal']
df['_total_goals_before'] = df.groupby('match_id')['_is_goal'].cumsum() - df['_is_goal']
df['_opp_goals_before']   = df['_total_goals_before'] - df['_team_goals_before']
df['score_diff']          = df['_team_goals_before'] - df['_opp_goals_before']
df.drop(columns=['_is_goal', '_team_goals_before', '_total_goals_before', '_opp_goals_before'], inplace=True)

# Team form
form_lookup = _build_team_form(df)
df['team_form'] = df.apply(lambda r: form_lookup.get((r['match_id'], r['team_id']), 0.5), axis=1)

df = pd.get_dummies(df, columns=['type'], prefix='type')
for col in ['type_Pass', 'type_Carry']:
    if col not in df.columns:
        df[col] = 0

sample = df[FEATURES].dropna().sample(n=min(200_000, len(df)), random_state=42)
corr   = sample.corr()

fig, ax = plt.subplots(figsize=(8, 7))
cmap = plt.cm.RdBu_r
im   = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Pearson Correlation')

ax.set_xticks(range(len(LABELS)))
ax.set_yticks(range(len(LABELS)))
ax.set_xticklabels(LABELS, rotation=40, ha='right', fontsize=8)
ax.set_yticklabels(LABELS, fontsize=8)

for i in range(len(LABELS)):
    for j in range(len(LABELS)):
        val   = corr.values[i, j]
        color = 'white' if abs(val) > 0.55 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)

ax.set_title('Version 1 Feature Correlation Matrix')
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight')
print(f"Saved to {OUTPUT_PNG}")
