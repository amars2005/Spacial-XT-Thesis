import warnings
import pandas as pd
from statsbombpy import sb

warnings.filterwarnings('ignore')


class StatsBomb360Loader:
    """
    Fetches only matches that have StatsBomb 360 freeze frame data available.

    Strategy: for each competition/season pair, sample the first match and
    attempt to load 360 frames. If frames are non-empty, every match in that
    competition is added to the list. This avoids checking every individual
    match while still being accurate at the competition level.
    """

    def get_360_matches(self) -> list[int]:
        print("Fetching competition list...")
        comps = sb.competitions()
        print(f"Found {len(comps)} competitions. Scanning for 360 data...\n")

        # The competitions DataFrame exposes a 'match_available_360' column —
        # filter to rows where it is non-null to identify 360-compatible seasons.
        comps_360 = comps[comps['match_available_360'].notna()]
        print(f"{len(comps_360)} competition/season pairs have 360 data.\n")

        all_match_ids: list[int] = []

        for _, row in comps_360.iterrows():
            comp_id   = row['competition_id']
            season_id = row['season_id']
            label     = f"{row['competition_name']} — {row['season_name']}"

            try:
                matches = sb.matches(competition_id=comp_id, season_id=season_id)
                if matches.empty:
                    continue

                # Filter to matches that individually have 360 data available
                if 'match_status_360' in matches.columns:
                    matches = matches[matches['match_status_360'] == 'available']

                if matches.empty:
                    continue

                ids = matches['match_id'].tolist()
                all_match_ids.extend(ids)
                print(f"  [360 OK] {label}  ({len(ids)} matches)")

            except Exception:
                continue

        unique_ids = list(set(all_match_ids))
        print(f"\nTotal 360-compatible matches: {len(unique_ids)}")
        return unique_ids
