import warnings
import numpy as np
import pandas as pd
import requests
from statsbombpy import sb

from config import ACTION_TYPES, SCALAR_COLS, PITCH_W, PITCH_H

warnings.filterwarnings('ignore')

_360_BASE_URL = (
    "https://raw.githubusercontent.com/statsbomb/open-data/master/data/three-sixty/{}.json"
)


class MatchParser360:
    """
    Parses a single match into:
      - A structured DataFrame of relevant events with scalar features attached.
      - A freeze-frame lookup dict:  event_uuid -> {players: [...], visible_area: [...]}

    The DataFrame columns will contain every key in SCALAR_COLS, plus the raw
    fields needed by the builder (match_id, id, index, period, timestamp,
    minute, second, type, team, start_x/y, end_x/y, success).
    """

    # Maps StatsBomb play-pattern names to a normalised 0-1 ordinal.
    # "Regular Play" scores lowest; set-pieces and restarts score higher.
    _PLAY_PATTERN_MAP = {
        'Regular Play':    0.0,
        'From Counter':    0.1,
        'From Keeper':     0.3,
        'From Goal Kick':  0.4,
        'From Throw In':   0.5,
        'From Free Kick':  0.7,
        'From Corner':     0.9,
        'From Kick Off':   1.0,
        'Other':           0.2,
    }

    def __init__(self, match_id: int):
        self.match_id = match_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self) -> tuple[pd.DataFrame, dict]:
        """
        Returns
        -------
        df : pd.DataFrame  — one row per relevant event, with all SCALAR_COLS present.
        frame_lookup : dict — event UUID -> {players, visible_area}
        """
        events = sb.events(match_id=self.match_id)
        frames = self._fetch_frames_direct()

        if events is None or events.empty:
            return pd.DataFrame(), {}

        # Compute rolling score differential before filtering
        events = self._compute_score_diff(events)

        # Keep only the action types we model
        # reset_index is critical — statsbombpy returns a non-contiguous index and
        # duplicate indices cause .loc assignment to fail in _extract_coordinates.
        df = events[events['type'].isin(ACTION_TYPES)].copy().reset_index(drop=True)
        if df.empty:
            return pd.DataFrame(), {}

        df = self._extract_coordinates(df)
        df = self._extract_scalars(df)

        df['match_id'] = self.match_id
        df['timestamp_dt'] = pd.to_timedelta(df['timestamp'], errors='coerce')
        df = df.sort_values(['period', 'timestamp_dt', 'index']).reset_index(drop=True)

        frame_lookup = self._build_frame_lookup(frames)

        return df, frame_lookup

    # ------------------------------------------------------------------
    # Score differential
    # ------------------------------------------------------------------

    def _compute_score_diff(self, events: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a 'score_diff' column to events representing the acting team's
        goal advantage *before* each event is executed.
        """
        events = events.copy()
        events['score_diff'] = 0.0

        if 'shot_outcome' not in events.columns or 'team' not in events.columns:
            return events

        teams = events['team'].dropna().unique()
        if len(teams) < 2:
            return events

        scores = {team: 0 for team in teams}
        score_diffs = []

        for _, row in events.iterrows():
            acting_team = row.get('team')

            if acting_team in scores:
                opponent_goals = sum(v for k, v in scores.items() if k != acting_team)
                score_diffs.append(float(scores[acting_team] - opponent_goals))
            else:
                score_diffs.append(0.0)

            # Update scores after this event
            if row.get('type') == 'Shot' and row.get('shot_outcome') == 'Goal':
                team = row.get('team')
                if team in scores:
                    scores[team] += 1

        events['score_diff'] = score_diffs
        return events

    # ------------------------------------------------------------------
    # Coordinate extraction
    # ------------------------------------------------------------------

    def _extract_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        df['start_x'] = np.nan
        df['start_y'] = np.nan
        df['end_x']   = np.nan
        df['end_y']   = np.nan

        # --- Start location (all event types) ---
        if 'location' in df.columns:
            loc = df['location'].dropna()
            if not loc.empty:
                coords = np.vstack(loc.values)
                df.loc[loc.index, 'start_x'] = coords[:, 0]
                df.loc[loc.index, 'start_y'] = coords[:, 1]

        # --- Pass end location ---
        if 'pass_end_location' in df.columns:
            mask = (df['type'] == 'Pass') & df['pass_end_location'].notna()
            if mask.any():
                coords = np.vstack(df.loc[mask, 'pass_end_location'].values)
                df.loc[mask, 'end_x'] = coords[:, 0]
                df.loc[mask, 'end_y'] = coords[:, 1]

        # --- Carry end location ---
        if 'carry_end_location' in df.columns:
            mask = (df['type'] == 'Carry') & df['carry_end_location'].notna()
            if mask.any():
                coords = np.vstack(df.loc[mask, 'carry_end_location'].values)
                df.loc[mask, 'end_x'] = coords[:, 0]
                df.loc[mask, 'end_y'] = coords[:, 1]

        # --- Shot end location (3D — take x, y only) ---
        if 'shot_end_location' in df.columns:
            mask = (df['type'] == 'Shot') & df['shot_end_location'].notna()
            if mask.any():
                raw = df.loc[mask, 'shot_end_location'].values
                coords = np.vstack([c[:2] for c in raw])
                df.loc[mask, 'end_x'] = coords[:, 0]
                df.loc[mask, 'end_y'] = coords[:, 1]

        # --- Success flag ---
        df['success'] = 0
        if 'pass_outcome' in df.columns:
            df.loc[(df['type'] == 'Pass') & df['pass_outcome'].isna(), 'success'] = 1
        df.loc[df['type'].isin(['Carry', 'Ball Receipt*']), 'success'] = 1
        if 'shot_outcome' in df.columns:
            df.loc[(df['type'] == 'Shot') & (df['shot_outcome'] == 'Goal'), 'success'] = 1
        if 'dribble_outcome' in df.columns:
            df.loc[(df['type'] == 'Dribble') & (df['dribble_outcome'] == 'Complete'), 'success'] = 1

        return df

    # ------------------------------------------------------------------
    # Scalar feature extraction
    # ------------------------------------------------------------------

    def _extract_scalars(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds every column in SCALAR_COLS to df.
        Non-applicable features (e.g. pass_length for a Carry) are zero-filled.
        """

        # 1. Action-type one-hot
        for atype in ACTION_TYPES:
            col = f"is_{atype.replace(' ', '_').replace('*', '').lower()}"
            df[col] = (df['type'] == atype).astype(float)

        # 2. Spatial features
        df['start_x_norm'] = df['start_x'].fillna(0) / PITCH_W
        df['start_y_norm'] = df['start_y'].fillna(0) / PITCH_H

        goal_x, goal_y = 120.0, 40.0
        dx = goal_x - df['start_x'].fillna(0)
        dy = goal_y - df['start_y'].fillna(0)
        max_dist = np.sqrt(goal_x ** 2 + goal_y ** 2)
        df['dist_to_goal_norm']  = np.sqrt(dx ** 2 + dy ** 2) / max_dist
        df['angle_to_goal_norm'] = np.arctan2(dy, dx) / np.pi  # normalised to [-1, 1]

        # 3. Match context
        df['under_pressure'] = (
            df['under_pressure'].fillna(False).astype(float)
            if 'under_pressure' in df.columns
            else 0.0
        )
        df['period_norm']     = df['period'].fillna(1) / 5.0
        df['minute_norm']     = df['minute'].fillna(0) / 120.0
        df['score_diff_norm'] = df['score_diff'].fillna(0).clip(-3, 3) / 3.0

        # 4. Pass-specific (zero for non-pass events)
        df['pass_length_norm'] = (
            df['pass_length'].fillna(0) / np.sqrt(PITCH_W ** 2 + PITCH_H ** 2)
            if 'pass_length' in df.columns else 0.0
        )
        df['pass_angle_norm'] = (
            df['pass_angle'].fillna(0) / np.pi
            if 'pass_angle' in df.columns else 0.0
        )

        height_map = {'Ground Pass': 0.0, 'Low Pass': 0.5, 'High Pass': 1.0}
        df['pass_height_norm'] = (
            df['pass_height'].map(height_map).fillna(0.0)
            if 'pass_height' in df.columns else 0.0
        )

        df['is_pass_switch']  = (
            df['pass_switch'].fillna(False).astype(float)
            if 'pass_switch' in df.columns else 0.0
        )
        df['is_through_ball'] = (
            df['pass_through_ball'].fillna(False).astype(float)
            if 'pass_through_ball' in df.columns else 0.0
        )

        # 5. End position (fall back to start if no end recorded)
        df['end_x_norm'] = df['end_x'].fillna(df['start_x'].fillna(0)) / PITCH_W
        df['end_y_norm'] = df['end_y'].fillna(df['start_y'].fillna(0)) / PITCH_H

        # Guarantee every SCALAR_COL column exists and is numeric
        for col in SCALAR_COLS:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        return df

    # ------------------------------------------------------------------
    # 360 freeze frame loading (direct JSON — bypasses statsbombpy bug)
    # ------------------------------------------------------------------

    def _fetch_frames_direct(self) -> list[dict]:
        """
        Fetch the StatsBomb 360 JSON directly from GitHub, bypassing
        statsbombpy's sb.frames() which is broken on newer pandas versions
        due to an internal pd.concat(axis=1) call with non-unique indices.

        Returns the raw list of 360 entries, or [] on 404 / network error.

        Each entry has the shape:
            {
              'event_uuid': str,
              'visible_area': [x1, y1, x2, y2, ...],   # flat coordinate list
              'freeze_frame': [{'teammate', 'actor', 'keeper', 'location'}, ...]
            }
        """
        url = _360_BASE_URL.format(self.match_id)
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception:
            return []

    def _build_frame_lookup(self, raw_frames: list[dict]) -> dict:
        """
        Convert the raw 360 JSON list into a fast lookup dict:
            event_uuid -> {'players': [...], 'visible_area': [[x,y], ...]}

        visible_area is stored as a flat [x1,y1,x2,y2,...] list in the JSON;
        we parse it into [[x1,y1], [x2,y2], ...] pairs here.
        """
        lookup: dict = {}

        for entry in raw_frames:
            event_uuid = entry.get('event_uuid')
            if not event_uuid:
                continue

            # Parse flat coordinate list into [x, y] pairs
            flat = entry.get('visible_area', [])
            visible_area = [
                [flat[i], flat[i + 1]]
                for i in range(0, len(flat) - 1, 2)
            ]

            players = []
            for player in entry.get('freeze_frame', []):
                loc = player.get('location')
                if isinstance(loc, list) and len(loc) >= 2:
                    players.append({
                        'location': loc,
                        'teammate': bool(player.get('teammate', False)),
                        'actor':    bool(player.get('actor',    False)),
                        'keeper':   bool(player.get('keeper',   False)),
                    })

            lookup[event_uuid] = {'players': players, 'visible_area': visible_area}

        return lookup
