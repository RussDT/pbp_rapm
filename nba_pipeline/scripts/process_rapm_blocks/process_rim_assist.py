"""
RIM_ASSIST processor.

Processes data for assisted rim makes per standard possession.
"""

import time

from .common import (
    _base_processing,
    _finalize_df,
    build_shot_flags_df,
    prepare_standard_possession_df,
)


def process_rim_assist_py(file_path, year, season_str):
    """
    Processes assisted rim makes using the standard possession definition.
    Outputs Is_Rim_Assist for each possession.
    """
    print(f"  Starting RIM_ASSIST Processing for {season_str}...")
    nba_df = _base_processing(file_path)
    if nba_df is None:
        return None
    start_time = time.time()

    nba_df_checked, group_cols = prepare_standard_possession_df(nba_df, "RIM_ASSIST")
    shot_flags = build_shot_flags_df(nba_df_checked)
    nba_df_checked['Rim_Assist_Event'] = (
        shot_flags['is_assisted_make'] & shot_flags['is_rim']
    ).astype(int)

    if group_cols:
        nba_df_checked['Rim_Assist_Total'] = nba_df_checked.groupby(group_cols)['Rim_Assist_Event'].cumsum()
    else:
        nba_df_checked['Rim_Assist_Total'] = nba_df_checked['Rim_Assist_Event'].cumsum()
    print(f"      Calculated Rim_Assist_Event (found {nba_df_checked['Rim_Assist_Event'].sum()} assisted rim makes)")

    if 'End_of_Possession' in nba_df_checked.columns and nba_df_checked['End_of_Possession'].dtype == bool:
        nba_filt = nba_df_checked[nba_df_checked['End_of_Possession']].copy()
    else:
        print("      Warning: 'End_of_Possession' column missing or not boolean for filtering. Skipping.")
        nba_filt = nba_df_checked.copy()
    print(f"      Filtered down to {len(nba_filt)} RIM_ASSIST End of Possession rows")

    if not nba_filt.empty:
        if group_cols:
            nba_filt['Is_Rim_Assist'] = nba_filt['Rim_Assist_Total'] - nba_filt.groupby(group_cols)['Rim_Assist_Total'].shift(fill_value=0)
        else:
            nba_filt['Is_Rim_Assist'] = nba_filt['Rim_Assist_Total'] - nba_filt['Rim_Assist_Total'].shift(fill_value=0)
        nba_filt['Is_Rim_Assist'] = nba_filt['Is_Rim_Assist'].astype(int)
        print("      Calculated Is_Rim_Assist possession diffs")

    nba_rim_assist_output = _finalize_df(nba_filt, 'Is_Rim_Assist', year)

    end_time = time.time()
    print(f"  Finished RIM_ASSIST Processing. Time: {end_time - start_time:.2f} seconds.")
    return nba_rim_assist_output
