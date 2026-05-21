"""
REB (Rebounding) processor.

Processes data for Offensive Rebounding rate calculation.
"""

import time
import numpy as np
import pandas as pd

from .common import (
    _base_processing,
    _finalize_df,
    series_contains,
    case_when,
)


def process_reb_py(file_path, year, season_str):
    """Processes data for Rebounding factor calculation."""
    print(f"  Starting REB Processing for {season_str}...")
    nba_df = _base_processing(file_path)
    if nba_df is None:
        return None
    start_time = time.time()

    # Define Possession Characteristics (REB Version)
    time_col_exists = 'time_quarter' in nba_df.columns
    if not time_col_exists:
        print("      Warning: 'time_quarter' column not found. Offensive Rebound logic might be inaccurate.")

    nba_df['Offensive_Rebound'] = case_when(
        series_contains(nba_df['home_description'], "REBOUND", case=False) & \
            series_contains(nba_df['Prev_home_desc'], "MISS", case=False) & \
            (True if not time_col_exists else ~(series_contains(nba_df['time_quarter'], "0:00", case=False))) & \
            ~(nba_df['Next_event'] == "EndOfPeriod") & \
            ~series_contains(nba_df['Prev_home_desc'], r'(1 of [23]|2 of 3|Technical|Flagrant)', case=False, regex=True), 1,
        series_contains(nba_df['home_description'], "REBOUND", case=False) & \
            series_contains(nba_df['Prev_home_desc'], "Putback", case=False) & \
            (True if not time_col_exists else ~(series_contains(nba_df['time_quarter'], "0:00", case=False))) & \
            ~(nba_df['Next_event'] == "EndOfPeriod") & \
            ~series_contains(nba_df['Prev_home_desc'], r'(1 of [23]|2 of 3|Technical|Flagrant)', case=False, regex=True), 1,
        series_contains(nba_df['visitor_description'], "REBOUND", case=False) & \
            series_contains(nba_df['Prev_visitor_desc'], "MISS", case=False) & \
            (True if not time_col_exists else ~(series_contains(nba_df['time_quarter'], "0:00", case=False))) & \
            ~(nba_df['Next_event'] == "EndOfPeriod") & \
            ~series_contains(nba_df['Prev_visitor_desc'], r'(1 of [23]|2 of 3|Technical|Flagrant)', case=False, regex=True), 1,
        series_contains(nba_df['visitor_description'], "REBOUND", case=False) & \
            series_contains(nba_df['Prev_visitor_desc'], "Putback", case=False) & \
            (True if not time_col_exists else ~(series_contains(nba_df['time_quarter'], "0:00", case=False))) & \
            ~(nba_df['Next_event'] == "EndOfPeriod") & \
            ~series_contains(nba_df['Prev_visitor_desc'], r'(1 of [23]|2 of 3|Technical|Flagrant)', case=False, regex=True), 1,
        0
    ).astype(int)
    print(f"      Calculated Offensive_Rebound flag (found {nba_df['Offensive_Rebound'].sum()})")

    ft_exclude_pattern = r'\b(1 of [23]|2 of 3|Technical|Flagrant)\b'
    nba_df['End_of_Possession'] = case_when(
        series_contains(nba_df['home_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_visitor_desc'], "MISS", case=False) & ~series_contains(nba_df['Prev_visitor_desc'], ft_exclude_pattern, case=False, regex=True), True,
        series_contains(nba_df['visitor_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_home_desc'], "MISS", case=False) & ~series_contains(nba_df['Prev_home_desc'], ft_exclude_pattern, case=False, regex=True), True,
        series_contains(nba_df['home_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_home_desc'], "MISS", case=False) & ~series_contains(nba_df['Prev_home_desc'], ft_exclude_pattern, case=False, regex=True), True,
        series_contains(nba_df['visitor_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_visitor_desc'], "MISS", case=False) & ~series_contains(nba_df['Prev_visitor_desc'], ft_exclude_pattern, case=False, regex=True), True,
        False
    ).astype(bool)

    nba_df['TeamOnOffense'] = case_when(
        series_contains(nba_df['home_description'], r'Flagrant (2 of 2|3 of 3)', case=False, regex=True), "Home",
        series_contains(nba_df['visitor_description'], r'Flagrant (2 of 2|3 of 3)', case=False, regex=True), "Away",
        series_contains(nba_df['home_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_visitor_desc'], "MISS", case=False), "Away",
        series_contains(nba_df['visitor_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_home_desc'], "MISS", case=False), "Home",
        series_contains(nba_df['home_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_home_desc'], "MISS", case=False), "Home",
        series_contains(nba_df['visitor_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_visitor_desc'], "MISS", case=False), "Away",
        series_contains(nba_df['home_description'], "Turnover", case=False), "Home",
        series_contains(nba_df['visitor_description'], "Turnover", case=False), "Away",
        series_contains(nba_df['home_description'], "PTS", case=False) & (nba_df['event_type'] == "MAKE") & ~series_contains(nba_df['Next_visitor_desc'], "S.FOUL", case=True), "Home",
        series_contains(nba_df['visitor_description'], "PTS", case=False) & (nba_df['event_type'] == "MAKE") & ~series_contains(nba_df['Next_home_desc'], "S.FOUL", case=True), "Away",
        series_contains(nba_df['home_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False, regex=True), "Home",
        series_contains(nba_df['visitor_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False, regex=True), "Away",
        ""
    )
    print("      Calculated EOP, TeamOnOffense for REB")

    # Skip Propagation for REB
    print("      Skipping Player Propagation/FT Check for REB (as per R script structure)")
    nba_df_checked = nba_df

    # Map Players
    o_mapping = {}
    d_mapping = {}
    for i in range(1, 6):
        o_mapping[f'O{i}'] = np.where(
            nba_df_checked['TeamOnOffense'] == "Away", nba_df_checked[f'a{i}'],
            np.where(nba_df_checked['TeamOnOffense'] == "Home", nba_df_checked[f'h{i}'], np.nan)
        )
        d_mapping[f'D{i}'] = np.where(
            nba_df_checked['TeamOnOffense'] == "Away", nba_df_checked[f'h{i}'],
            np.where(nba_df_checked['TeamOnOffense'] == "Home", nba_df_checked[f'a{i}'], np.nan)
        )
    nba_df_checked = nba_df_checked.assign(**o_mapping, **d_mapping)
    print("      Mapped O/D Players")

    # Filter for EOP
    if 'End_of_Possession' in nba_df_checked.columns and pd.api.types.is_bool_dtype(nba_df_checked['End_of_Possession']):
        nba_filt = nba_df_checked[nba_df_checked['End_of_Possession']].copy()
    else:
        print("      Warning: 'End_of_Possession' column missing or not boolean for filtering.")
        nba_filt = nba_df_checked.copy()
    print(f"      Filtered down to {len(nba_filt)} REB End of Possession rows")

    # Finalize
    nba_reb_output = _finalize_df(nba_filt, 'Offensive_Rebound', year)

    end_time = time.time()
    print(f"  Finished REB Processing. Time: {end_time - start_time:.2f} seconds.")
    return nba_reb_output
