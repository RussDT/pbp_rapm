"""
THREE_FREQ processor.

Processes data for three-point attempt frequency.
"""

import time

from .common import (
    _base_processing,
    _finalize_df,
    build_shot_flags_df,
    map_players_from_team_on_offense,
    series_contains,
    case_when,
)


def process_three_freq_py(file_path, year, season_str):
    """
    Processes 3-point attempt frequency.
    Outputs Is_Three_Attempt (0 or 1) for each FGA.
    """
    print(f"  Starting THREE_FREQ Processing for {season_str}...")
    nba_df = _base_processing(file_path)
    if nba_df is None:
        return None
    start_time = time.time()

    shot_flags = build_shot_flags_df(nba_df)
    initial_rows = len(nba_df)
    nba_df = nba_df[shot_flags['is_fga']].copy()
    shot_flags = shot_flags.loc[nba_df.index]
    print(f"      Filtered to FGA events: {len(nba_df)} rows (from {initial_rows})")

    if nba_df.empty:
        print("      Warning: No FGA events found. Returning None.")
        return None

    nba_df['Is_Three_Attempt'] = shot_flags['is_3pt'].astype(int)
    print(f"      Calculated Is_Three_Attempt flag (found {nba_df['Is_Three_Attempt'].sum()} 3PA out of {len(nba_df)} FGAs)")

    nba_df['End_of_Possession'] = True
    nba_df['TeamOnOffense'] = case_when(
        series_contains(nba_df['home_description'], "PTS|MISS", case=False, regex=True), "Home",
        series_contains(nba_df['visitor_description'], "PTS|MISS", case=False, regex=True), "Away",
        ""
    )
    print("      Calculated TeamOnOffense for THREE_FREQ")

    nba_df = map_players_from_team_on_offense(nba_df)
    print("      Mapped O/D Players")

    nba_filt = nba_df[nba_df['End_of_Possession']].copy()
    print(f"      Final THREE_FREQ rows: {len(nba_filt)}")

    nba_three_freq_output = _finalize_df(nba_filt, 'Is_Three_Attempt', year)

    end_time = time.time()
    print(f"  Finished THREE_FREQ Processing. Time: {end_time - start_time:.2f} seconds.")
    return nba_three_freq_output
