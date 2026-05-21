"""
SECOND_CHANCE_CLEAN processor.

RAPM-aligned second-chance scoring using the standard possession denominator
and expected FT handling.
"""

import time

from .clean_chance_utils import build_clean_chance_base_py
from .common import _finalize_df


def process_second_chance_clean_py(file_path, year, season_str):
    """Process clean second-chance scoring."""
    print(f"  Starting SECOND_CHANCE_CLEAN Processing for {season_str}...")
    start_time = time.time()
    nba_filt = build_clean_chance_base_py(file_path, year, season_str, "SECOND_CHANCE_CLEAN")
    if nba_filt is None or nba_filt.empty:
        return None

    nba_filt = nba_filt.copy()
    nba_filt['Net_Diff'] = nba_filt['Second_Chance_Diff']
    nba_filt['Off_Diff'] = nba_filt['Second_Chance_Diff']
    # Match the standard points/value convention used by TS and pure RAPM:
    # positive offense value implies negative defense value on the same row.
    nba_filt['Def_Diff'] = -nba_filt['Second_Chance_Diff']

    output = _finalize_df(nba_filt, ['Net_Diff', 'Off_Diff', 'Def_Diff'], year)
    end_time = time.time()
    print(f"  Finished SECOND_CHANCE_CLEAN Processing. Time: {end_time - start_time:.2f} seconds.")
    return output
