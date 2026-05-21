"""
ASSIST_POINTS processor.

Processes data for assisted field-goal points per standard possession.
"""

import time

from .common import (
    _base_processing,
    _finalize_df,
    build_shot_flags_df,
    prepare_standard_possession_df,
)


def process_assist_points_py(file_path, year, season_str):
    """
    Processes assisted field-goal points using the standard possession definition.
    Outputs Assist_Points for each possession.
    """
    print(f"  Starting ASSIST_POINTS Processing for {season_str}...")
    nba_df = _base_processing(file_path)
    if nba_df is None:
        return None
    start_time = time.time()

    nba_df_checked, group_cols = prepare_standard_possession_df(nba_df, "ASSIST_POINTS")
    shot_flags = build_shot_flags_df(nba_df_checked)
    nba_df_checked['Assist_Points_Event'] = shot_flags['shot_points'].where(
        shot_flags['is_assisted_make'], 0
    ).astype(int)

    if group_cols:
        nba_df_checked['Assist_Points_Total'] = nba_df_checked.groupby(group_cols)['Assist_Points_Event'].cumsum()
    else:
        nba_df_checked['Assist_Points_Total'] = nba_df_checked['Assist_Points_Event'].cumsum()
    print(f"      Calculated Assist_Points_Event (found {nba_df_checked['Assist_Points_Event'].sum()} total assisted points)")

    if 'End_of_Possession' in nba_df_checked.columns and nba_df_checked['End_of_Possession'].dtype == bool:
        nba_filt = nba_df_checked[nba_df_checked['End_of_Possession']].copy()
    else:
        print("      Warning: 'End_of_Possession' column missing or not boolean for filtering. Skipping.")
        nba_filt = nba_df_checked.copy()
    print(f"      Filtered down to {len(nba_filt)} ASSIST_POINTS End of Possession rows")

    if not nba_filt.empty:
        if group_cols:
            nba_filt['Assist_Points'] = nba_filt['Assist_Points_Total'] - nba_filt.groupby(group_cols)['Assist_Points_Total'].shift(fill_value=0)
        else:
            nba_filt['Assist_Points'] = nba_filt['Assist_Points_Total'] - nba_filt['Assist_Points_Total'].shift(fill_value=0)
        nba_filt['Assist_Points'] = nba_filt['Assist_Points'].astype(int)
        print("      Calculated Assist_Points possession diffs")

    nba_assist_points_output = _finalize_df(nba_filt, 'Assist_Points', year)

    end_time = time.time()
    print(f"  Finished ASSIST_POINTS Processing. Time: {end_time - start_time:.2f} seconds.")
    return nba_assist_points_output
