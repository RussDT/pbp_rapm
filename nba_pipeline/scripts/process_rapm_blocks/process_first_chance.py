"""
FIRST_CHANCE processor.

Clean first-chance scoring on the standard RAPM possession denominator.
"""

import time

from .clean_chance_utils import build_clean_chance_base_py
from .common import _finalize_df


def process_first_chance_py(file_path, year, season_str):
    """Process clean first-chance scoring with turnover flags."""
    print(f"  Starting FIRST_CHANCE Processing for {season_str}...")
    start_time = time.time()
    nba_filt = build_clean_chance_base_py(file_path, year, season_str, "FIRST_CHANCE")
    if nba_filt is None or nba_filt.empty:
        return None

    nba_filt = nba_filt.copy()
    nba_filt['Net_Diff'] = nba_filt['First_Chance_Diff']

    numerator_cols = ['Net_Diff', 'Is_Turnover', 'Is_BadPass_TOV']
    for col in [
        'FC_SQ_Diff',
        'FC_MAKE_Diff',
        'FC_FT_Diff',
        'FC_FT_FREQ_Diff',
        'FC_FT_SEVERITY_Diff',
        'FC_EFG_Diff',
        'FC_EFG_BASELINE_Diff',
        'FC_EFG_VALUE_Diff',
        'FC_RIM_EFG_AA_Diff',
        'FC_MID_EFG_AA_Diff',
        'FC_THREE_EFG_AA_Diff',
        'FC_RIM_EFG_FREQ_Diff',
        'FC_RIM_EFG_FG_Diff',
        'FC_MID_EFG_FREQ_Diff',
        'FC_MID_EFG_FG_Diff',
        'FC_THREE_EFG_FREQ_Diff',
        'FC_THREE_EFG_FG_Diff',
    ]:
        if col in nba_filt.columns:
            numerator_cols.append(col)

    output = _finalize_df(nba_filt, numerator_cols, year)
    end_time = time.time()
    print(f"  Finished FIRST_CHANCE Processing. Time: {end_time - start_time:.2f} seconds.")
    return output
