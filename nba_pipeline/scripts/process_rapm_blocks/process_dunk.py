"""
DUNK processor.

Processes made dunks per standard possession.
"""

import time

from .common import (
    _base_processing,
    _finalize_df,
    build_shot_flags_df,
    prepare_standard_possession_df,
)


def process_dunk_py(file_path, year, season_str):
    """
    Processes made dunks using the standard possession definition.
    Outputs Is_Dunk for each possession.
    """
    print(f"  Starting DUNK Processing for {season_str}...")
    nba_df = _base_processing(file_path)
    if nba_df is None:
        return None
    start_time = time.time()

    nba_df_checked, group_cols = prepare_standard_possession_df(nba_df, "DUNK")
    shot_flags = build_shot_flags_df(nba_df_checked)
    nba_df_checked['Dunk_Event'] = (
        (nba_df_checked['event_type'] == 'MAKE') & shot_flags['is_dunk']
    ).astype(int)

    if group_cols:
        nba_df_checked['Dunk_Total'] = nba_df_checked.groupby(group_cols)['Dunk_Event'].cumsum()
    else:
        nba_df_checked['Dunk_Total'] = nba_df_checked['Dunk_Event'].cumsum()
    print(f"      Calculated Dunk_Event (found {nba_df_checked['Dunk_Event'].sum()} made dunks)")

    if 'End_of_Possession' in nba_df_checked.columns and nba_df_checked['End_of_Possession'].dtype == bool:
        nba_filt = nba_df_checked[nba_df_checked['End_of_Possession']].copy()
    else:
        print("      Warning: 'End_of_Possession' column missing or not boolean for filtering. Skipping.")
        nba_filt = nba_df_checked.copy()
    print(f"      Filtered down to {len(nba_filt)} DUNK End of Possession rows")

    if not nba_filt.empty:
        if group_cols:
            nba_filt['Is_Dunk'] = nba_filt['Dunk_Total'] - nba_filt.groupby(group_cols)['Dunk_Total'].shift(fill_value=0)
        else:
            nba_filt['Is_Dunk'] = nba_filt['Dunk_Total'] - nba_filt['Dunk_Total'].shift(fill_value=0)
        nba_filt['Is_Dunk'] = nba_filt['Is_Dunk'].astype(int)
        print("      Calculated Is_Dunk possession diffs")

    nba_dunk_output = _finalize_df(nba_filt, 'Is_Dunk', year)

    end_time = time.time()
    print(f"  Finished DUNK Processing. Time: {end_time - start_time:.2f} seconds.")
    return nba_dunk_output
