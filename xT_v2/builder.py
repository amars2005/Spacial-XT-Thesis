import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import DATA_DIR, SCALAR_COLS, NUM_CHANNELS, GRID_H, GRID_W, BUILD_WORKERS
from loader import StatsBomb360Loader
from parser import MatchParser360
from encoder import encode_event


class DatasetBuilder:
    """
    Orchestrates the full data pipeline:
      1. Discover all 360-compatible match IDs via the loader.
      2. Parse each match (events + freeze frames) via MatchParser360.
      3. Assign possession chains and goal labels.
      4. Encode each event into a spatial tensor + scalar vector.
      5. Save each match as a compressed .npz file to DATA_DIR.

    On subsequent runs, already-built .npz files are skipped unless
    force_rebuild=True is passed to build().
    """

    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.loader = StatsBomb360Loader()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, limit: int | None = None, force_rebuild: bool = False) -> None:
        """
        Build (or incrementally update) the encoded dataset on disk.

        Parameters
        ----------
        limit : int, optional
            Cap the number of matches processed — useful for quick testing.
        force_rebuild : bool
            Re-encode matches even if their .npz already exists.
        """
        match_ids = self.loader.get_360_matches()

        if limit:
            match_ids = match_ids[:limit]
            print(f"Limited to first {limit} matches.\n")

        skipped  = 0
        failures = 0

        to_build = []
        for match_id in match_ids:
            out_path = os.path.join(DATA_DIR, f"{match_id}.npz")
            if os.path.exists(out_path) and not force_rebuild:
                skipped += 1
            else:
                to_build.append((match_id, out_path))

        print(f"Matches to build: {len(to_build)}  |  Already built (skipping): {skipped}\n")

        with ThreadPoolExecutor(max_workers=BUILD_WORKERS) as executor:
            futures = {
                executor.submit(self._process_and_save, mid, path): mid
                for mid, path in to_build
            }
            with tqdm(total=len(futures), desc="Building dataset") as pbar:
                for future in as_completed(futures):
                    mid = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        tqdm.write(f"  [SKIP] Match {mid}: {e}")
                        failures += 1
                    pbar.update(1)

        print(f"\nBuild complete.  Skipped (already built): {skipped}  |  Failures: {failures}")

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_split(
        self,
        match_ids: list[int],
        *,
        workers: int = 32,
        desc: str = "Loading",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load .npz files for a specific list of match IDs using a two-phase
        memory-efficient strategy:

        Phase 1 (fast, parallel): read only the tiny 'label' key per file to
          count events — does NOT load spatial data.
        Phase 2 (parallel, in-place): pre-allocate the full output arrays once,
          then each worker loads one file and writes directly into the correct
          slice.  Peak RAM ≈ output size + (workers × one-file size), vs 2×
          output size with np.concatenate.

        Returns
        -------
        spatial, scalar, labels, match_ids_arr
        """
        paths = []
        for mid in sorted(set(match_ids)):
            path = os.path.join(DATA_DIR, f"{mid}.npz")
            if os.path.exists(path):
                paths.append((mid, path))

        if not paths:
            raise FileNotFoundError(
                f"No .npz files found for any of the {len(match_ids)} requested match IDs."
            )

        n_workers = min(workers, len(paths))

        def _count(item):
            _, path = item
            with np.load(path) as d:
                return int(d['label'].shape[0])

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            counts = list(tqdm(
                ex.map(_count, paths),
                total=len(paths),
                desc=f"{desc} (counting)",
                leave=False,
            ))

        total = sum(counts)
        offsets = [0] * len(counts)
        for i in range(1, len(counts)):
            offsets[i] = offsets[i - 1] + counts[i - 1]

        from config import SCALAR_COLS as _SCALAR_COLS
        out_spatial  = np.empty((total, NUM_CHANNELS, GRID_H, GRID_W), dtype=np.float32)
        out_scalar   = np.empty((total, len(_SCALAR_COLS)),             dtype=np.float32)
        out_labels   = np.empty(total,                                  dtype=np.float32)
        out_mids_arr = np.empty(total,                                  dtype=np.int64)

        def _fill(i: int) -> None:
            mid, path = paths[i]
            off = offsets[i]
            n   = counts[i]
            with np.load(path) as d:
                out_spatial[off:off + n] = d['spatial']
                out_scalar[off:off + n]  = d['scalar']
                out_labels[off:off + n]  = d['label']
            out_mids_arr[off:off + n] = mid

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(_fill, i) for i in range(len(paths))]
            with tqdm(total=len(futs), desc=desc) as pbar:
                for f in as_completed(futs):
                    f.result()
                    pbar.update(1)

        return out_spatial, out_scalar, out_labels, out_mids_arr

    def load_all(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load all .npz files. Delegates to load_split with all available match IDs."""
        files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith('.npz'))

        if not files:
            raise FileNotFoundError(
                f"No .npz files found in {DATA_DIR}.\n"
                "Run  python main.py --build  first."
            )

        match_ids = [int(f[:-4]) for f in files]
        return self.load_split(match_ids, desc="Loading dataset")

    # ------------------------------------------------------------------
    # Internal: process one match
    # ------------------------------------------------------------------

    def _process_and_save(self, match_id: int, out_path: str) -> None:
        parser = MatchParser360(match_id)
        df, frame_lookup = parser.parse()

        if df.empty:
            return

        # Reject matches where the 360 JSON was unavailable (404 or network error).
        # These build without player channel data and would pollute the training set.
        if not frame_lookup:
            raise ValueError("No 360 freeze frame data returned — skipping match.")

        df = self._assign_chains(df)

        n = len(df)
        spatial_arr = np.zeros((n, NUM_CHANNELS, GRID_H, GRID_W), dtype=np.float32)
        scalar_arr  = np.zeros((n, len(SCALAR_COLS)),              dtype=np.float32)

        for i, (_, row) in enumerate(df.iterrows()):
            event_id   = row.get('id', None)
            frame_data = frame_lookup.get(event_id, None)

            spatial_arr[i] = encode_event(row.to_dict(), frame_data)

            for j, col in enumerate(SCALAR_COLS):
                val = row.get(col, 0.0)
                scalar_arr[i, j] = float(val) if not pd.isna(val) else 0.0

        labels = df['chain_goal'].values.astype(np.float32)

        np.savez_compressed(
            out_path,
            spatial  = spatial_arr,
            scalar   = scalar_arr,
            label    = labels,
            match_id = np.array([match_id]),
        )

    # ------------------------------------------------------------------
    # Internal: possession chains
    # ------------------------------------------------------------------

    def _assign_chains(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Segments events into possession chains and assigns a temporally
        binary label to each event.

        Every event in a goal-scoring chain receives label = 1.0.
        Events in non-goal chains receive label = 0.

        A new chain begins when:
          - The match changes
          - The period changes
          - The team in possession changes
          - The previous event was a Shot (possession always ends on a shot)
        """
        df = df.sort_values(['match_id', 'period', 'index']).reset_index(drop=True)

        match_change  = df['match_id'] != df['match_id'].shift(1)
        period_change = df['period']   != df['period'].shift(1)
        prev_was_shot = df['type'].shift(1) == 'Shot'

        # Determine team column
        if 'team' in df.columns:
            team_change = df['team'] != df['team'].shift(1)
        elif 'team_id' in df.columns:
            team_change = df['team_id'] != df['team_id'].shift(1)
        else:
            team_change = pd.Series(False, index=df.index)

        is_new_chain = match_change | period_change | team_change | prev_was_shot
        df['chain_id'] = is_new_chain.cumsum()

        # Identify goal chains
        is_goal     = (df['type'] == 'Shot') & (df['success'] == 1)
        goal_chains = set(df.loc[is_goal, 'chain_id'].unique())

        labels = np.zeros(len(df), dtype=np.float32)
        for chain_id, group in df.groupby('chain_id'):
            if chain_id not in goal_chains:
                continue
            labels[group.index] = 1.0

        df['chain_goal'] = labels
        return df
