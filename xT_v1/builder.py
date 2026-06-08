import pandas as pd
from loader import StatsBombLoader
from parser import MatchParser

class DatasetBuilder:
    def __init__(self):
        self.loader = StatsBombLoader()
        
    def build_dataset(self, limit=None):
        match_ids = self.loader.get_all_free_matches()
        
        if limit:
            match_ids = match_ids[:limit]
            print(f"Limiting to first {limit} matches for testing.")
            
        master_frames = []
        
        for i, m_id in enumerate(match_ids):
            if i % 10 == 0:
                print(f"Processing match {i}/{len(match_ids)}...")
                
            try:
                parser = MatchParser(m_id)
                df_match = parser.parse()
                master_frames.append(df_match)
            except Exception as e:
                print(f"Error parsing match {m_id}: {e}")
                continue
                
        if not master_frames:
            return pd.DataFrame()
            
        full_df = pd.concat(master_frames, ignore_index=True)
        
        # --- FINAL SORT ---
        full_df['timestamp_dt'] = pd.to_timedelta(full_df['timestamp'], errors='coerce')
        full_df = full_df.sort_values(by=['match_id', 'period', 'timestamp_dt', 'index'])
        full_df = full_df.drop(columns=['timestamp_dt'])
        
        return full_df

    def process_chains(self, df):
        """
        Takes the master dataframe and adds Chain IDs and Goal outcomes.
        """
        print("Processing possession chains...")
        
        # Ensure strict sorting
        df = df.sort_values(by=['match_id', 'period', 'index']).reset_index(drop=True)
        
        # 1. Detect Breaks
        match_change = df['match_id'] != df['match_id'].shift(1)
        period_change = df['period'] != df['period'].shift(1)
        team_change = df['team_id'] != df['team_id'].shift(1)
        prev_was_shot = df['type'].shift(1) == 'Shot'
        
        is_new_chain = match_change | period_change | team_change | prev_was_shot
        
        # 2. Assign Chain IDs
        df['chain_id'] = is_new_chain.cumsum()
        
        # 3. Determine Goal Outcomes
        is_goal = (df['type'] == 'Shot') & (df['success'] == 1)
        goal_chains = df.loc[is_goal, 'chain_id'].unique()
        
        df['chain_goal'] = 0
        df.loc[df['chain_id'].isin(goal_chains), 'chain_goal'] = 1
        
        print(f"Processed {df['chain_id'].max()} unique chains.")
        return df