"""
RIM_FG_PCT (Rim FG Percentage) processor.

Processes data for Rim Field Goal Percentage calculation.
"""

import time
import numpy as np
import pandas as pd

from .common import (
    _base_processing,
    _finalize_df,
    series_contains,
    case_when,
    RIM_ACTION_TYPES,
)


def process_rim_fg_pct_py(file_path, year, season_str):
    """
    Processes data for Rim FG Percentage factor calculation.
    Outputs Is_Rim_Make (0 or 1) for each rim attempt only.
    Rim FG% = made rim attempts / all rim attempts
    """
    print(f"  Starting RIM_FG_PCT Processing for {season_str}...")
    nba_df = _base_processing(file_path)
    if nba_df is None:
        return None
    start_time = time.time()

    # Filter to FGA events only (MAKE or MISS)
    initial_rows = len(nba_df)
    nba_df = nba_df[nba_df['event_type'].isin(['MAKE', 'MISS'])].copy()
    print(f"      Filtered to FGA events: {len(nba_df)} rows (from {initial_rows})")

    if nba_df.empty:
        print("      Warning: No FGA events found. Returning None.")
        return None

    # Identify rim attempts using vectorized approach
    if 'event_action_type' in nba_df.columns:
        nba_df['event_action_type_num'] = pd.to_numeric(nba_df['event_action_type'], errors='coerce')
        action_type_rim = nba_df['event_action_type_num'].isin(RIM_ACTION_TYPES)
    else:
        print("      Warning: 'event_action_type' column not found. Using description patterns only.")
        action_type_rim = pd.Series([False] * len(nba_df), index=nba_df.index)

    # Also check description patterns as fallback
    desc_combined = nba_df['home_description'].str.lower() + nba_df['visitor_description'].str.lower()
    desc_pattern_rim = desc_combined.str.contains('layup|dunk| tip ', case=False, regex=True, na=False)

    # Combine both checks
    is_rim = action_type_rim | desc_pattern_rim

    # Filter to rim attempts only
    fga_count = len(nba_df)
    nba_df = nba_df[is_rim].copy()
    print(f"      Filtered to rim attempts: {len(nba_df)} rows (from {fga_count} FGAs)")

    if nba_df.empty:
        print("      Warning: No rim attempts found. Returning None.")
        return None

    # Calculate Is_Rim_Make (1 if made, 0 if missed)
    nba_df['Is_Rim_Make'] = (nba_df['event_type'] == 'MAKE').astype(int)
    makes = nba_df['Is_Rim_Make'].sum()
    print(f"      Calculated Is_Rim_Make flag (found {makes} made rims out of {len(nba_df)} rim attempts)")

    # Each rim attempt is an "End of Possession" observation for this metric
    nba_df['End_of_Possession'] = True

    # Determine TeamOnOffense based on who took the shot
    nba_df['TeamOnOffense'] = case_when(
        series_contains(nba_df['home_description'], "PTS|MISS", case=False, regex=True), "Home",
        series_contains(nba_df['visitor_description'], "PTS|MISS", case=False, regex=True), "Away",
        ""
    )
    print("      Calculated TeamOnOffense for RIM_FG_PCT")

    # Map O/D Players
    o_mapping = {}
    d_mapping = {}
    for i in range(1, 6):
        o_mapping[f'O{i}'] = np.where(
            nba_df['TeamOnOffense'] == "Away", nba_df[f'a{i}'],
            np.where(nba_df['TeamOnOffense'] == "Home", nba_df[f'h{i}'], np.nan)
        )
        d_mapping[f'D{i}'] = np.where(
            nba_df['TeamOnOffense'] == "Away", nba_df[f'h{i}'],
            np.where(nba_df['TeamOnOffense'] == "Home", nba_df[f'a{i}'], np.nan)
        )
    nba_df = nba_df.assign(**o_mapping, **d_mapping)
    print("      Mapped O/D Players")

    # Filter for EOP (all rows since every rim attempt is an observation)
    nba_filt = nba_df[nba_df['End_of_Possession']].copy()
    print(f"      Final RIM_FG_PCT rows: {len(nba_filt)}")

    # Finalize
    nba_rim_fg_pct_output = _finalize_df(nba_filt, 'Is_Rim_Make', year)

    end_time = time.time()
    print(f"  Finished RIM_FG_PCT Processing. Time: {end_time - start_time:.2f} seconds.")
    return nba_rim_fg_pct_output
