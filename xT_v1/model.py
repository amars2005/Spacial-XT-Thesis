# model.py
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
from evaluate import compute_metrics, print_metrics, save_metrics

_DEFAULT_CSV   = os.path.join(_ROOT, "statsbomb_chained_dataset.csv")
_FORM_CACHE    = os.path.join(_HERE, "team_form_cache.csv")
_V3_SPLIT_PATH = os.path.join(_ROOT, "xT_v3", "checkpoints", "split_match_ids.json")


# ---------------------------------------------------------------------------
# Team-form helper
# ---------------------------------------------------------------------------

def _build_team_form(df):
    """
    Returns a dict {(match_id, team_id): team_form} where team_form is the
    fraction of points earned in the team's last 5 matches (0–1 scale,
    neutral prior 0.5 for teams with no previous data).

    Results are cached to team_form_cache.csv so StatsBombPy is only queried
    once.
    """
    if os.path.exists(_FORM_CACHE):
        cache = pd.read_csv(_FORM_CACHE)
        return dict(zip(zip(cache['match_id'], cache['team_id']), cache['team_form']))

    print("Building team form cache (fetching match dates from StatsBombPy)...")
    from statsbombpy import sb

    # --- 1. Goals per (match_id, team_id) from CSV data ---
    goals_df = df[(df['type'] == 'Shot') & (df['success'] == 1)]
    goals = goals_df.groupby(['match_id', 'team_id']).size().reset_index(name='goals_scored')
    match_teams = df[['match_id', 'team_id']].drop_duplicates()
    match_goals = match_teams.merge(goals, on=['match_id', 'team_id'], how='left')
    match_goals['goals_scored'] = match_goals['goals_scored'].fillna(0).astype(int)

    # Join opponent goals
    opp = match_goals.rename(columns={'team_id': 'opp_id', 'goals_scored': 'goals_conceded'})
    mr = match_goals.merge(opp, on='match_id')
    mr = mr[mr['team_id'] != mr['opp_id']].copy()
    mr['points'] = np.where(
        mr['goals_scored'] > mr['goals_conceded'], 3,
        np.where(mr['goals_scored'] == mr['goals_conceded'], 1, 0)
    )

    # --- 2. Match dates from StatsBombPy ---
    needed = set(df['match_id'].unique())
    date_rows = []
    comps = sb.competitions()
    for _, c in comps.iterrows():
        try:
            m = sb.matches(competition_id=c['competition_id'], season_id=c['season_id'])
            filt = m[m['match_id'].isin(needed)][['match_id', 'match_date']].copy()
            date_rows.append(filt)
        except Exception:
            pass
    dates = pd.concat(date_rows).drop_duplicates('match_id')
    dates['match_date'] = pd.to_datetime(dates['match_date'])

    mr = mr.merge(dates, on='match_id', how='left')
    mr = mr.sort_values(['team_id', 'match_date']).reset_index(drop=True)

    # --- 3. Rolling last-5 form (excluding current match) ---
    out_rows = []
    for team_id, grp in mr.groupby('team_id'):
        grp = grp.sort_values('match_date').reset_index(drop=True)
        pts = grp['points'].values
        form = np.empty(len(grp))
        _decay = 0.75  # exponential decay: most recent game weighted highest
        for i in range(len(grp)):
            window = pts[max(0, i - 5):i]
            if len(window) == 0:
                form[i] = 0.5
            else:
                # weights: [decay^(n-1), ..., decay^1, decay^0] — most recent = 1.0
                w = np.array([_decay ** (len(window) - 1 - j) for j in range(len(window))])
                form[i] = np.dot(w, window) / (3.0 * w.sum())
        grp['team_form'] = form
        out_rows.append(grp[['match_id', 'team_id', 'team_form']])

    result = pd.concat(out_rows, ignore_index=True)
    result.to_csv(_FORM_CACHE, index=False)
    return dict(zip(zip(result['match_id'], result['team_id']), result['team_form']))


class ExpectedThreatModel:
    def __init__(self, filepath=_DEFAULT_CSV):
        self.filepath = filepath
        self.model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)

    def load_and_prep(self):
        print("Loading dataset...")
        df = pd.read_csv(self.filepath)

        # --- Spatial features ---
        df['dx'] = 120 - df['start_x']
        df['dy'] = 40  - df['start_y']
        df['dist_to_goal']  = np.sqrt(df['dx']**2 + df['dy']**2)
        df['angle_to_goal'] = np.arctan2(df['dy'], df['dx'])

        # --- Score differential (running score at time of each event) ---
        df = df.sort_values(['match_id', 'period', 'index']).reset_index(drop=True)
        df['_is_goal'] = ((df['type'] == 'Shot') & (df['success'] == 1)).astype(int)
        df['_team_goals_before']  = (df.groupby(['match_id', 'team_id'])['_is_goal']
                                       .cumsum() - df['_is_goal'])
        df['_total_goals_before'] = (df.groupby('match_id')['_is_goal']
                                       .cumsum() - df['_is_goal'])
        df['_opp_goals_before']   = df['_total_goals_before'] - df['_team_goals_before']
        df['score_diff']          = df['_team_goals_before'] - df['_opp_goals_before']
        df.drop(columns=['_is_goal', '_team_goals_before',
                         '_total_goals_before', '_opp_goals_before'], inplace=True)

        # --- Team form (last-5 normalised, cached) ---
        form_lookup = _build_team_form(df)
        df['team_form'] = df.apply(
            lambda r: form_lookup.get((r['match_id'], r['team_id']), 0.5), axis=1
        )

        # --- One-hot encode action type ---
        df_encoded = pd.get_dummies(df, columns=['type'], prefix='type')
        for col in ['type_Pass', 'type_Carry']:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        features = [
            'start_x', 'start_y', 'dist_to_goal', 'angle_to_goal',
            'type_Pass', 'type_Carry',
            'period', 'minute', 'score_diff', 'team_form',
        ]
        df_encoded[features] = df_encoded[features].fillna(0)

        return df_encoded, features

    def train(self):
        df, features = self.load_and_prep()

        match_ids = df['match_id'].unique()
        rng = np.random.default_rng(seed=42)
        rng.shuffle(match_ids)
        train_end = int(len(match_ids) * 0.70)
        val_end   = train_end + int(len(match_ids) * 0.15)
        train_matches = match_ids[:train_end]
        test_matches  = match_ids[val_end:]

        print(f"Training on {len(train_matches)} matches, Testing on {len(test_matches)} matches.")

        train_data = df[df['match_id'].isin(train_matches)]
        test_data  = df[df['match_id'].isin(test_matches)]

        X_train, y_train = train_data[features], train_data['chain_goal']
        X_test,  y_test  = test_data[features],  test_data['chain_goal']

        print("Training Random Forest...")
        self.model.fit(X_train, y_train)

        probs   = self.model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test.values, probs)
        print_metrics(metrics, "v1 Test Results (Random Forest)")
        save_metrics(metrics, os.path.join(_ROOT, "metrics_v1.json"))

        all_probs        = self.model.predict_proba(df[features])[:, 1]
        df['pred_value'] = all_probs

        return df

    def train_360_subset(self, save_preds=False):
        """
        Train and evaluate the same Random Forest on only the 425-match 360-data
        subset, using the identical train/test split as V2/V3 (loaded from
        xT_v3/checkpoints/split_match_ids.json).

        This resolves the V1/V2 dataset confound: by holding architecture constant
        (RF with same hyperparameters) and restricting to the 360-data subset, the
        metric change from V1-full to V1-360 isolates the dataset effect, while the
        change from V1-360 to V2 isolates the CNN architecture benefit.
        """
        import json as _json

        if not os.path.exists(_V3_SPLIT_PATH):
            raise FileNotFoundError(
                f"V3 split file not found at {_V3_SPLIT_PATH}.\n"
                "Train xT_v3 first:  cd ../xT_v3 && python main.py --train"
            )

        with open(_V3_SPLIT_PATH) as f:
            split = _json.load(f)

        train_set = set(split["train"])
        test_set  = set(split["test"])
        all_360   = train_set | set(split["val"]) | test_set

        df, features = self.load_and_prep()

        subset = df[df['match_id'].isin(all_360)].copy()
        n_360  = subset['match_id'].nunique()
        print(f"360-data subset: {n_360} matches, {len(subset):,} events")

        train_data = subset[subset['match_id'].isin(train_set)]
        test_data  = subset[subset['match_id'].isin(test_set)]

        X_train, y_train = train_data[features], train_data['chain_goal']
        X_test,  y_test  = test_data[features],  test_data['chain_goal']

        print(
            f"360-subset split: {len(train_set)} train matches ({len(X_train):,} events)  |  "
            f"{len(test_set)} test matches ({len(X_test):,} events)"
        )
        print("Training Random Forest on 360-data subset...")
        rf_360 = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
        rf_360.fit(X_train, y_train)

        probs   = rf_360.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test.values, probs)
        print_metrics(metrics, "v1-RF (360-data subset) Test Results")
        save_metrics(metrics, os.path.join(_ROOT, "metrics_v1_360subset.json"))

        if save_preds:
            out_dir = os.path.join(_HERE, "predictions")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "preds.npz")
            np.savez(out_path,
                     probs=probs,
                     labels=y_test.values,
                     match_ids=test_data['match_id'].values)
            print(f"Predictions saved → {out_path}")

        return metrics

    # ------------------------------------------------------------------
    def _draw_pitch_markings(self, ax):
        lc, lw = 'white', 1.5
        ax.plot([0, 0],     [0, 80],   color=lc, linewidth=lw)
        ax.plot([120, 120], [0, 80],   color=lc, linewidth=lw)
        ax.plot([0, 120],   [0, 0],    color=lc, linewidth=lw)
        ax.plot([0, 120],   [80, 80],  color=lc, linewidth=lw)
        ax.plot([60, 60],   [0, 80],   color=lc, linewidth=lw)
        circle = mpatches.Circle((60, 40), 10, color=lc, fill=False, linewidth=lw)
        ax.add_patch(circle)
        ax.scatter(60, 40, color=lc, s=15)
        ax.plot([18, 18], [18, 62],   color=lc, linewidth=lw)
        ax.plot([0, 18],  [18, 18],   color=lc, linewidth=lw)
        ax.plot([0, 18],  [62, 62],   color=lc, linewidth=lw)
        ax.plot([102, 102], [18, 62], color=lc, linewidth=lw)
        ax.plot([120, 102], [18, 18], color=lc, linewidth=lw)
        ax.plot([120, 102], [62, 62], color=lc, linewidth=lw)
        ax.plot([6, 6],   [30, 50],   color=lc, linewidth=lw)
        ax.plot([0, 6],   [30, 30],   color=lc, linewidth=lw)
        ax.plot([0, 6],   [50, 50],   color=lc, linewidth=lw)
        ax.plot([114, 114], [30, 50], color=lc, linewidth=lw)
        ax.plot([120, 114], [30, 30], color=lc, linewidth=lw)
        ax.plot([120, 114], [50, 50], color=lc, linewidth=lw)

    def visualize_value_map(self, df):
        print("Generating Heatmap...")
        x_range = np.linspace(0, 120, 50)
        y_range = np.linspace(0, 80, 50)
        xx, yy  = np.meshgrid(x_range, y_range)

        grid = pd.DataFrame({'start_x': xx.ravel(), 'start_y': yy.ravel()})
        grid['dx']             = 120 - grid['start_x']
        grid['dy']             = 40  - grid['start_y']
        grid['dist_to_goal']   = np.sqrt(grid['dx']**2 + grid['dy']**2)
        grid['angle_to_goal']  = np.arctan2(grid['dy'], grid['dx'])
        grid['type_Pass']      = 1
        grid['type_Carry']     = 0
        grid['period']         = 2   # typical mid-game values
        grid['minute']         = 45
        grid['score_diff']     = 0
        grid['team_form']      = 0.5

        features = [
            'start_x', 'start_y', 'dist_to_goal', 'angle_to_goal',
            'type_Pass', 'type_Carry',
            'period', 'minute', 'score_diff', 'team_form',
        ]
        z = self.model.predict_proba(grid[features])[:, 1].reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(10, 7))
        contour = ax.contourf(xx, yy, z, levels=25, cmap='magma', vmin=0, vmax=0.15)
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label('Probability of Goal (xT)')
        self._draw_pitch_markings(ax)
        ax.set_title('Spatial Expected Threat Map (Random Forest)')
        ax.set_xlim(0, 120)
        ax.set_ylim(80, 0)
        ax.set_aspect('equal')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(_HERE, "xt_heatmap.png"), dpi=300, bbox_inches='tight')
        print("Heatmap saved to xt_heatmap.png")


# --- EXECUTION ---
if __name__ == "__main__":
    try:
        xt_model = ExpectedThreatModel()
        df_with_preds = xt_model.train()
        xt_model.visualize_value_map(df_with_preds)
        df_with_preds.to_csv(os.path.join(_HERE, "final_model_output.csv"), index=False)
    except FileNotFoundError:
        print("Error: 'statsbomb_chained_dataset.csv' not found.")
        print("Please run 'python main.py' first to generate the data.")
