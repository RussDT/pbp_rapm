"""
MIDRANGE_FREQ (Midrange Frequency) processor.

Processes data for Midrange Attempt Frequency calculation.
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


def process_midrange_freq_py(file_path, year, season_str):
    """
    Processes data for Midrange Frequency factor calculation.
    Outputs Is_Midrange_Attempt (0 or 1) for each FGA.
    Midrange = FGAs that are NOT 3-pointers AND NOT rim attempts
    """
    print(f"  Starting MIDRANGE_FREQ Processing for {season_str}...")
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

    # Identify 3-pointers via description pattern
    desc_combined = (nba_df['home_description'].fillna('').str.lower() +
                     nba_df['visitor_description'].fillna('').str.lower())
    is_3pt = desc_combined.str.contains('3pt', case=False, regex=False, na=False)

    # Identify rim attempts (reuse existing logic from rim_freq)
    if 'event_action_type' in nba_df.columns:
        nba_df['event_action_type_num'] = pd.to_numeric(nba_df['event_action_type'], errors='coerce')
        action_type_rim = nba_df['event_action_type_num'].isin(RIM_ACTION_TYPES)
    else:
        action_type_rim = pd.Series([False] * len(nba_df), index=nba_df.index)
    desc_pattern_rim = desc_combined.str.contains('layup|dunk| tip ', case=False, regex=True, na=False)
    is_rim = action_type_rim | desc_pattern_rim

    # Midrange = NOT 3PT AND NOT Rim
    nba_df['Is_Midrange_Attempt'] = (~is_3pt & ~is_rim).astype(int)
    midrange_count = nba_df['Is_Midrange_Attempt'].sum()
    print(f"      Found {midrange_count} midrange attempts out of {len(nba_df)} FGAs")

    # Each FGA is an observation
    nba_df['End_of_Possession'] = True

    # Determine TeamOnOffense
    nba_df['TeamOnOffense'] = case_when(
        series_contains(nba_df['home_description'], "PTS|MISS", case=False, regex=True), "Home",
        series_contains(nba_df['visitor_description'], "PTS|MISS", case=False, regex=True), "Away",
        ""
    )
    print("      Calculated TeamOnOffense for MIDRANGE_FREQ")

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

    # Filter for EOP (all rows since every FGA is an observation)
    nba_filt = nba_df[nba_df['End_of_Possession']].copy()
    print(f"      Final MIDRANGE_FREQ rows: {len(nba_filt)}")

    # Finalize
    nba_midrange_freq_output = _finalize_df(nba_filt, 'Is_Midrange_Attempt', year)

    end_time = time.time()
    print(f"  Finished MIDRANGE_FREQ Processing. Time: {end_time - start_time:.2f} seconds.")
    return nba_midrange_freq_output
