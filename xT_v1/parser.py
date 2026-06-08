
import pandas as pd
import numpy as np
from statsbombpy import sb

class MatchParser:
    def __init__(self, match_id):
        self.match_id = match_id
        
    def parse(self):
        events = sb.events(match_id=self.match_id)
        cols_to_keep = [
            'id', 'index', 'period', 'timestamp', 'minute', 'second', 
            'type', 'location', 'pass_end_location', 'carry_end_location',
            'pass_outcome', 'shot_outcome', 'team_id'
        ]
        
        available_cols = [c for c in cols_to_keep if c in events.columns]
        df = events[available_cols].copy()
        
        df = df[df['type'].isin(['Pass', 'Carry', 'Shot'])].copy()
        
        df['start_x'] = np.nan; df['start_y'] = np.nan
        df['end_x'] = np.nan; df['end_y'] = np.nan

        if 'location' in df.columns:
            loc_data = df['location'].dropna()
            if not loc_data.empty:
                coords = np.vstack(loc_data.values)
                df.loc[loc_data.index, 'start_x'] = coords[:, 0]
                df.loc[loc_data.index, 'start_y'] = coords[:, 1]

        if 'pass_end_location' in df.columns:
            pass_mask = (df['type'] == 'Pass') & (df['pass_end_location'].notna())
            if pass_mask.any():
                pass_coords = np.vstack(df.loc[pass_mask, 'pass_end_location'].values)
                df.loc[pass_mask, 'end_x'] = pass_coords[:, 0]
                df.loc[pass_mask, 'end_y'] = pass_coords[:, 1]

        if 'carry_end_location' in df.columns:
            carry_mask = (df['type'] == 'Carry') & (df['carry_end_location'].notna())
            if carry_mask.any():
                carry_coords = np.vstack(df.loc[carry_mask, 'carry_end_location'].values)
                df.loc[carry_mask, 'end_x'] = carry_coords[:, 0]
                df.loc[carry_mask, 'end_y'] = carry_coords[:, 1]

        df['success'] = 0 
        if 'pass_outcome' in df.columns:
            df.loc[(df['type'] == 'Pass') & (df['pass_outcome'].isna()), 'success'] = 1
        df.loc[df['type'] == 'Carry', 'success'] = 1
        if 'shot_outcome' in df.columns:
            df.loc[(df['type'] == 'Shot') & (df['shot_outcome'] == 'Goal'), 'success'] = 1
            
        df['match_id'] = self.match_id
        
        df['timestamp_dt'] = pd.to_timedelta(df['timestamp'], errors='coerce')
        df = df.sort_values(by=['period', 'timestamp_dt', 'index'])
        
        return df[['match_id', 'index', 'period', 'timestamp', 'minute', 'second', 'type', 
                   'team_id', 'start_x', 'start_y', 'end_x', 'end_y', 'success']]