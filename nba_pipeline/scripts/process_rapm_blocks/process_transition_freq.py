"""
TRANSITION_FREQ (Transition Frequency) processor.

Processes data for Transition Shot Frequency calculation.
Only available for years 24-26.
"""

import time
import numpy as np
import pandas as pd

from .common import (
    _base_processing,
    _finalize_df,
    series_contains,
    case_when,
    normalize_season_end_year,
)


def process_transition_freq_py(file_path, year, season_str):
    """
    Processes data for Transition Frequency factor calculation.
    Outputs Is_Transition (0 or 1) for each FGA.

    Only available for years 24-26 (seasons 2023-24 through 2025-26).
    """
    season_end_year = normalize_season_end_year(year)
    if season_end_year < 2024:
        print(
            f"  Skipping TRANSITION_FREQ - only available for years 24-26 "
            f"(season end year {season_end_year} < 2024)"
        )
        return None

    print(f"  Starting TRANSITION_FREQ Processing for {season_str}...")
    nba_df = _base_processing(file_path)
    if nba_df is None:
        return None
    start_time = time.time()

    # Check if is_transition column exists
    if 'is_transition' not in nba_df.columns:
        print("      Warning: 'is_transition' column not found. Returning None.")
        return None

    # Filter to FGA events only (MAKE or MISS)
    initial_rows = len(nba_df)
    nba_df = nba_df[nba_df['event_type'].isin(['MAKE', 'MISS'])].copy()
    print(f"      Filtered to FGA events: {len(nba_df)} rows (from {initial_rows})")

    if nba_df.empty:
        print("      Warning: No FGA events found. Returning None.")
        return None

    # Calculate Is_Transition (handle None/NaN as False)
    nba_df['Is_Transition'] = (nba_df['is_transition'] == True).astype(int)
    transition_count = nba_df['Is_Transition'].sum()
    print(f"      Calculated Is_Transition flag (found {transition_count} transition shots out of {len(nba_df)} FGAs)")

    # Each FGA is an "End of Possession" observation for this metric
    nba_df['End_of_Possession'] = True

    # Determine TeamOnOffense based on who took the shot
    nba_df['TeamOnOffense'] = case_when(
        series_contains(nba_df['home_description'], "PTS|MISS", case=False, regex=True), "Home",
        series_contains(nba_df['visitor_description'], "PTS|MISS", case=False, regex=True), "Away",
        ""
    )
    print("      Calculated TeamOnOffense for TRANSITION_FREQ")

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
    print(f"      Final TRANSITION_FREQ rows: {len(nba_filt)}")

    # Finalize
    nba_transition_freq_output = _finalize_df(nba_filt, 'Is_Transition', year)

    end_time = time.time()
    print(f"  Finished TRANSITION_FREQ Processing. Time: {end_time - start_time:.2f} seconds.")
    return nba_transition_freq_output
