from statsbombpy import sb
import warnings

warnings.filterwarnings('ignore')

class StatsBombLoader:
    """
    Responsible for fetching metadata about competitions and matches.
    """
    def get_all_free_matches(self):
        print("Fetching competition data...")
        comps = sb.competitions()
        
        all_matches = []
        print(f"Found {len(comps)} competitions. Fetching matches...")
        
        for index, row in comps.iterrows():
            try:
                matches = sb.matches(competition_id=row['competition_id'], season_id=row['season_id'])
                match_ids = matches['match_id'].tolist()
                all_matches.extend(match_ids)
            except Exception as e:
                continue
                
        print(f"Total matches found: {len(all_matches)}")
        return list(set(all_matches))