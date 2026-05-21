"""
MIDRANGE_FG_PCT processor.

Processes data for midrange field-goal percentage.
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


def process_midrange_fg_pct_py(file_path, year, season_str):
    """
    Processes midrange field-goal percentage.
    Outputs Is_Midrange_Make (0 or 1) for each midrange FGA.
    Midrange = FGA that is neither a 3PA nor a rim attempt.
    """
    print(f"  Starting MIDRANGE_FG_PCT Processing for {season_str}...")
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

    fga_count = len(nba_df)
    midrange_mask = ~shot_flags.loc[nba_df.index, 'is_3pt'] & ~shot_flags.loc[nba_df.index, 'is_rim']
    nba_df = nba_df[midrange_mask].copy()
    print(f"      Filtered to midrange attempts: {len(nba_df)} rows (from {fga_count} FGAs)")

    if nba_df.empty:
        print("      Warning: No midrange attempts found. Returning None.")
        return None

    nba_df['Is_Midrange_Make'] = (nba_df['event_type'] == 'MAKE').astype(int)
    print(
        f"      Calculated Is_Midrange_Make flag "
        f"(found {nba_df['Is_Midrange_Make'].sum()} makes out of {len(nba_df)} midrange FGAs)"
    )

    nba_df['End_of_Possession'] = True
    nba_df['TeamOnOffense'] = case_when(
        series_contains(nba_df['home_description'], "PTS|MISS", case=False, regex=True), "Home",
        series_contains(nba_df['visitor_description'], "PTS|MISS", case=False, regex=True), "Away",
        ""
    )
    print("      Calculated TeamOnOffense for MIDRANGE_FG_PCT")

    nba_df = map_players_from_team_on_offense(nba_df)
    print("      Mapped O/D Players")

    nba_filt = nba_df[nba_df['End_of_Possession']].copy()
    print(f"      Final MIDRANGE_FG_PCT rows: {len(nba_filt)}")

    nba_midrange_fg_pct_output = _finalize_df(nba_filt, 'Is_Midrange_Make', year)

    end_time = time.time()
    print(f"  Finished MIDRANGE_FG_PCT Processing. Time: {end_time - start_time:.2f} seconds.")
    return nba_midrange_fg_pct_output
