"""
DUNK_ASSIST processor.

Processes assisted made dunks per standard possession.
"""

import time

from .common import (
    _base_processing,
    _finalize_df,
    build_shot_flags_df,
    prepare_standard_possession_df,
)


def process_dunk_assist_py(file_path, year, season_str):
    """
    Processes assisted made dunks using the standard possession definition.
    Outputs Is_Dunk_Assist for each possession.
    """
    print(f"  Starting DUNK_ASSIST Processing for {season_str}...")
    nba_df = _base_processing(file_path)
    if nba_df is None:
        return None
    start_time = time.time()

    nba_df_checked, group_cols = prepare_standard_possession_df(nba_df, "DUNK_ASSIST")
    shot_flags = build_shot_flags_df(nba_df_checked)
    nba_df_checked['Dunk_Assist_Event'] = (
        shot_flags['is_assisted_make'] & shot_flags['is_dunk']
    ).astype(int)

    if group_cols:
        nba_df_checked['Dunk_Assist_Total'] = nba_df_checked.groupby(group_cols)['Dunk_Assist_Event'].cumsum()
    else:
        nba_df_checked['Dunk_Assist_Total'] = nba_df_checked['Dunk_Assist_Event'].cumsum()
    print(f"      Calculated Dunk_Assist_Event (found {nba_df_checked['Dunk_Assist_Event'].sum()} assisted made dunks)")

    if 'End_of_Possession' in nba_df_checked.columns and nba_df_checked['End_of_Possession'].dtype == bool:
        nba_filt = nba_df_checked[nba_df_checked['End_of_Possession']].copy()
    else:
        print("      Warning: 'End_of_Possession' column missing or not boolean for filtering. Skipping.")
        nba_filt = nba_df_checked.copy()
    print(f"      Filtered down to {len(nba_filt)} DUNK_ASSIST End of Possession rows")

    if not nba_filt.empty:
        if group_cols:
            nba_filt['Is_Dunk_Assist'] = nba_filt['Dunk_Assist_Total'] - nba_filt.groupby(group_cols)['Dunk_Assist_Total'].shift(fill_value=0)
        else:
            nba_filt['Is_Dunk_Assist'] = nba_filt['Dunk_Assist_Total'] - nba_filt['Dunk_Assist_Total'].shift(fill_value=0)
        nba_filt['Is_Dunk_Assist'] = nba_filt['Is_Dunk_Assist'].astype(int)
        print("      Calculated Is_Dunk_Assist possession diffs")

    nba_dunk_assist_output = _finalize_df(nba_filt, 'Is_Dunk_Assist', year)

    end_time = time.time()
    print(f"  Finished DUNK_ASSIST Processing. Time: {end_time - start_time:.2f} seconds.")
    return nba_dunk_assist_output
