"""
Generate feature importance bar chart for the Version 1 Random Forest model.
Trains the model (same split as model.py) then saves the chart to
xT_v1/xt_feature_importance.png.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
from model import _build_team_form

CSV_PATH   = os.path.join(_ROOT, "statsbomb_chained_dataset.csv")
OUTPUT_PNG = os.path.join(_HERE, "xt_feature_importance.png")

FEATURES = [
    'start_x', 'start_y', 'dist_to_goal', 'angle_to_goal',
    'type_Pass', 'type_Carry',
    'period', 'minute', 'score_diff', 'team_form',
]
FEATURE_LABELS = [
    'Start X', 'Start Y', 'Distance to Goal', 'Angle to Goal',
    'Action: Pass', 'Action: Carry',
    'Period', 'Minute', 'Score Difference', 'Team Form',
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
df[FEATURES] = df[FEATURES].fillna(0)

match_ids = df['match_id'].unique()
rng = np.random.default_rng(seed=42)
rng.shuffle(match_ids)
train_end     = int(len(match_ids) * 0.70)
train_matches = match_ids[:train_end]
train_data    = df[df['match_id'].isin(train_matches)]

X_train = train_data[FEATURES]
y_train = train_data['chain_goal']

print(f"Training Random Forest on {len(train_matches)} matches ({len(X_train):,} events)...")
rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

importances = rf.feature_importances_
order         = np.argsort(importances)[::-1]
sorted_labels = [FEATURE_LABELS[i] for i in order]
sorted_vals   = importances[order]

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.barh(sorted_labels[::-1], sorted_vals[::-1], color='steelblue', edgecolor='white')
ax.set_xlabel('Feature Importance (Mean Decrease in Impurity)')
ax.set_title('Version 1 Random Forest — Feature Importances')
ax.set_xlim(0, sorted_vals.max() * 1.15)

for bar, val in zip(bars, sorted_vals[::-1]):
    ax.text(val + sorted_vals.max() * 0.01, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight')
print(f"Chart saved to {OUTPUT_PNG}")
