"""
RAPM (Regularized Adjusted Plus-Minus) processor.

Processes data for standard RAPM calculation with optional luck adjustment.
"""

import os
import time
import numpy as np
import pandas as pd

from .common import (
    _base_processing,
    _finalize_df,
    series_contains,
    case_when,
    fetch_player_stats_supabase,
    prepare_standard_possession_df,
)


def process_rapm_py(file_path, year, season_str, o_luck=1.0, d_luck=1.0, missing_ft_fallback="default"):
    """
    Processes data for standard RAPM calculation.

    Args:
        file_path: Path to input parquet file
        year: Season end year (e.g., 26 or 2026 for 2025-26)
        season_str: Season string (e.g., "2025-26")
        o_luck: Offensive luck adjustment weight (0.0 = no adjustment, 1.0 = full adjustment)
        d_luck: Defensive luck adjustment weight (0.0 = no adjustment, 1.0 = full adjustment)
        missing_ft_fallback: "default" uses 0.75 when FT% is missing; "actual"
            uses the observed FT result for only those missing-FT% rows.

    Returns:
        Tuple of (no_la_df, la_df) - both versions of RAPM output
    """
    print(f"  Starting RAPM Processing for {season_str} (o_luck={o_luck}, d_luck={d_luck})...")
    nba_df = _base_processing(file_path)
    if nba_df is None:
        return None
    start_time = time.time()

    nba_df_checked, group_cols = prepare_standard_possession_df(nba_df, "RAPM")

    # Join Player Season Stats from Supabase
    is_playoffs = "_PS" in os.path.basename(file_path).upper()
    player_stats_df = fetch_player_stats_supabase(year, is_playoffs)

    # Merge Player Stats
    shooter_id_col = 'player1_id'
    if player_stats_df is not None and shooter_id_col in nba_df_checked.columns:
        print("      Preparing data for player stats merge...")
        nba_df_checked['player1_id_numeric'] = pd.to_numeric(nba_df_checked[shooter_id_col], errors='coerce')
        player_stats_df = player_stats_df.copy()
        player_stats_df['PlayerID'] = pd.to_numeric(player_stats_df['PlayerID'], errors='coerce')
        player_stats_df = player_stats_df.dropna(subset=['PlayerID'])
        player_stats_df['PlayerID'] = player_stats_df['PlayerID'].astype('Int64')
        nba_df_checked['player1_id_numeric'] = nba_df_checked['player1_id_numeric'].astype('Int64')

        print(f"      Merging player stats onto {len(nba_df_checked)} rows...")
        nba_df_checked = pd.merge(
            nba_df_checked,
            player_stats_df[['PlayerID', 'FTPerc', 'ThreePerc']],
            left_on='player1_id_numeric',
            right_on='PlayerID',
            how='left'
        )
        nba_df_checked = nba_df_checked.drop(columns=['player1_id_numeric', 'PlayerID'], errors='ignore')
        print(f"      Successfully merged player stats for {nba_df_checked['FTPerc'].notna().sum()} rows.")
    else:
        print(f"      Skipping player stats merge. Using defaults.")
        if 'FTPerc' not in nba_df_checked.columns:
            nba_df_checked['FTPerc'] = np.nan
        if 'ThreePerc' not in nba_df_checked.columns:
            nba_df_checked['ThreePerc'] = np.nan

    # Calculate Expected Points
    default_ft_perc = 0.75
    default_3p_perc = 0.35

    ft_perc = pd.to_numeric(nba_df_checked['FTPerc'], errors='coerce')
    missing_ft_perc = ft_perc.isna()
    nba_df_checked['ExpFT'] = ft_perc.fillna(default_ft_perc).astype(float)
    nba_df_checked['Exp3PT'] = nba_df_checked['ThreePerc'].fillna(default_3p_perc).astype(float)

    print(f"      Assigned ExpFT (default: {default_ft_perc}) and Exp3PT (default: {default_3p_perc}).")

    is_ft_event = series_contains(nba_df_checked['home_description'], "Free Throw", case=False) | \
                  series_contains(nba_df_checked['visitor_description'], "Free Throw", case=False) | \
                  (nba_df_checked['event_type'] == "FreeThrow")
    is_2pt_make_event = (series_contains(nba_df_checked['home_description'], "PTS", case=False) & ~series_contains(nba_df_checked['home_description'], "3PT", case=False) & (nba_df_checked['event_type'] == "MAKE")) | \
                        (series_contains(nba_df_checked['visitor_description'], "PTS", case=False) & ~series_contains(nba_df_checked['visitor_description'], "3PT", case=False) & (nba_df_checked['event_type'] == "MAKE"))
    is_3pt_event = series_contains(nba_df_checked['home_description'], "3PT", case=False) | \
                   series_contains(nba_df_checked['visitor_description'], "3PT", case=False)
    actual_score = (nba_df_checked['Home_Action_Score'] + nba_df_checked['Away_Action_Score']).astype(float)

    exp_ft_value = nba_df_checked['ExpFT']
    if missing_ft_fallback == "actual":
        exp_ft_value = np.where(missing_ft_perc, actual_score, nba_df_checked['ExpFT'])
        print("      Missing FT% fallback for RAPM: actual FT results")
    exp_3pt_value = nba_df_checked['Exp3PT'] * 3.0

    # BASE VERSION (FT luck adjusted, no 3PT luck adjustment)
    nba_df_checked['ActualNet'] = case_when(
        is_ft_event, exp_ft_value,
        is_2pt_make_event, 2.0,
        is_3pt_event, actual_score,
        0.0
    )

    # LUCK ADJUSTMENT VERSION
    nba_df_checked['LA_OffNet'] = case_when(
        is_ft_event, exp_ft_value,
        is_2pt_make_event, 2.0,
        is_3pt_event, o_luck * exp_3pt_value + (1 - o_luck) * actual_score,
        0.0
    )

    nba_df_checked['LA_DefNet'] = case_when(
        is_ft_event, exp_ft_value,
        is_2pt_make_event, 2.0,
        is_3pt_event, d_luck * exp_3pt_value + (1 - d_luck) * actual_score,
        0.0
    )
    print(f"      Calculated ActualNet and LA_OffNet/LA_DefNet")

    # Calculate cumulative totals
    if group_cols:
        nba_df_checked['ActualTotal'] = nba_df_checked.groupby(group_cols)['ActualNet'].cumsum()
        nba_df_checked['LA_OffTotal'] = nba_df_checked.groupby(group_cols)['LA_OffNet'].cumsum()
        nba_df_checked['LA_DefTotal'] = nba_df_checked.groupby(group_cols)['LA_DefNet'].cumsum()
    else:
        nba_df_checked['ActualTotal'] = nba_df_checked['ActualNet'].cumsum()
        nba_df_checked['LA_OffTotal'] = nba_df_checked['LA_OffNet'].cumsum()
        nba_df_checked['LA_DefTotal'] = nba_df_checked['LA_DefNet'].cumsum()
    print("      Calculated cumulative totals for both versions")

    # Filter for EOP & Calculate Diffs
    if 'End_of_Possession' in nba_df_checked.columns and pd.api.types.is_bool_dtype(nba_df_checked['End_of_Possession']):
        nba_filt = nba_df_checked[nba_df_checked['End_of_Possession']].copy()
    else:
        print("      Warning: 'End_of_Possession' column missing or not boolean for filtering.")
        nba_filt = nba_df_checked.copy()
    print(f"      Filtered down to {len(nba_filt)} RAPM End of Possession rows")

    if not nba_filt.empty:
        # BASE VERSION DIFFS
        if 'ActualTotal' in nba_filt.columns:
            if group_cols:
                nba_filt['Off_Diff'] = nba_filt['ActualTotal'] - nba_filt.groupby(group_cols)['ActualTotal'].shift(fill_value=0)
            else:
                nba_filt['Off_Diff'] = nba_filt['ActualTotal'] - nba_filt['ActualTotal'].shift(fill_value=0)
            nba_filt['Def_Diff'] = nba_filt['Off_Diff']
            nba_filt['Net_Diff'] = nba_filt['Off_Diff']

        # LUCK ADJUSTMENT DIFFS
        if 'LA_OffTotal' in nba_filt.columns and 'LA_DefTotal' in nba_filt.columns:
            if group_cols:
                nba_filt['LA_Off_Diff'] = nba_filt['LA_OffTotal'] - nba_filt.groupby(group_cols)['LA_OffTotal'].shift(fill_value=0)
                nba_filt['LA_Def_Diff'] = nba_filt['LA_DefTotal'] - nba_filt.groupby(group_cols)['LA_DefTotal'].shift(fill_value=0)
            else:
                nba_filt['LA_Off_Diff'] = nba_filt['LA_OffTotal'] - nba_filt['LA_OffTotal'].shift(fill_value=0)
                nba_filt['LA_Def_Diff'] = nba_filt['LA_DefTotal'] - nba_filt['LA_DefTotal'].shift(fill_value=0)

        print("      Calculated diffs for both No-LA and LA versions")

    # Finalize BOTH versions
    no_la_cols = ['Net_Diff', 'Off_Diff', 'Def_Diff']
    no_la_cols_present = [c for c in no_la_cols if c in nba_filt.columns]
    if not no_la_cols_present:
        print("      Error: Could not find diff columns for No-LA output.")
        nba_rapm_no_la = None
    else:
        nba_rapm_no_la = _finalize_df(nba_filt, no_la_cols_present, year)

    # Version 2: With Luck Adjustment
    nba_filt_la = nba_filt.copy()
    if 'LA_Off_Diff' in nba_filt_la.columns:
        nba_filt_la['Off_Diff'] = nba_filt_la['LA_Off_Diff']
    if 'LA_Def_Diff' in nba_filt_la.columns:
        nba_filt_la['Def_Diff'] = nba_filt_la['LA_Def_Diff']

    la_cols = ['Net_Diff', 'Off_Diff', 'Def_Diff']
    la_cols_present = [c for c in la_cols if c in nba_filt_la.columns]
    if not la_cols_present:
        print("      Error: Could not find diff columns for LA output.")
        nba_rapm_la = None
    else:
        nba_rapm_la = _finalize_df(nba_filt_la, la_cols_present, year)

    end_time = time.time()
    print(f"  Finished RAPM Processing. Time: {end_time - start_time:.2f} seconds.")

    return nba_rapm_no_la, nba_rapm_la
