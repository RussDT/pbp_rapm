"""
FIRST_CHANCE_CLEAN processor.

Direct first-chance scoring surface on the standard possession denominator:
non-turnover first-chance points use the existing ExpFT-adjusted
First_Chance_Diff, while first-chance turnover rows are set to zero.
"""

import time

import numpy as np
import pandas as pd

from .clean_chance_utils import build_clean_chance_base_py
from .common import _finalize_df


def process_first_chance_clean_py(file_path, year, season_str):
    """Process direct clean first-chance scoring with turnover rows zeroed."""
    print(f"  Starting FIRST_CHANCE_CLEAN Processing for {season_str}...")
    start_time = time.time()
    nba_filt = build_clean_chance_base_py(file_path, year, season_str, "FIRST_CHANCE_CLEAN")
    if nba_filt is None or nba_filt.empty:
        return None

    nba_filt = nba_filt.copy()
    first_chance = pd.to_numeric(nba_filt['First_Chance_Diff'], errors='coerce').fillna(0.0)
    is_turnover = pd.to_numeric(nba_filt['Is_Turnover'], errors='coerce').fillna(0).astype(int)

    nba_filt['FC_Raw_Diff'] = first_chance
    nba_filt['Net_Diff'] = np.where(is_turnover == 1, 0.0, first_chance)
    nba_filt['Off_Diff'] = nba_filt['Net_Diff']
    nba_filt['Def_Diff'] = -nba_filt['Off_Diff']

    numerator_cols = [
        'Net_Diff',
        'Off_Diff',
        'Def_Diff',
        'FC_Raw_Diff',
        'Is_Turnover',
        'Is_BadPass_TOV',
    ]
    output = _finalize_df(nba_filt, numerator_cols, year)
    end_time = time.time()
    print(f"  Finished FIRST_CHANCE_CLEAN Processing. Time: {end_time - start_time:.2f} seconds.")
    return output
