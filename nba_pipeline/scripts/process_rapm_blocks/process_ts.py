"""
TS (True Shooting) processor.

Processes data for True Shooting efficiency calculation.
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
    propagate_player_values_py,
    fetch_player_stats_supabase,
)


def process_ts_py(file_path, year, season_str, missing_ft_fallback="default"):
    """Processes data for True Shooting calculation."""
    print(f"  Starting TS Processing for {season_str}...")
    nba_df = _base_processing(file_path)
    if nba_df is None:
        return None
    start_time = time.time()

    # Define Possession Characteristics (TS Version)
    nba_df['End_of_Possession'] = case_when(
        (nba_df['event_type'] == "EndOfPeriod") & (nba_df['prev_seconds'] > 0), True,
        series_contains(nba_df['home_description'], r'Flagrant (2 of 2|3 of 3)', case=False, regex=True), True,
        series_contains(nba_df['visitor_description'], r'Flagrant (2 of 2|3 of 3)', case=False, regex=True), True,
        (nba_df['event_type'] == "Turnover"), True,
        series_contains(nba_df['visitor_description'], "MISS", case=False) & (nba_df['event_type'] == "MISS"), True,
        series_contains(nba_df['home_description'], "MISS", case=False) & (nba_df['event_type'] == "MISS"), True,
        series_contains(nba_df['home_description'], "PTS", case=False) & (nba_df['event_type'] == "MAKE") & ~series_contains(nba_df['Next_visitor_desc'], "S.FOUL", case=True), True,
        series_contains(nba_df['visitor_description'], "PTS", case=False) & (nba_df['event_type'] == "MAKE") & ~series_contains(nba_df['Next_home_desc'], "S.FOUL", case=True), True,
        series_contains(nba_df['home_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False, regex=True) & \
            ~(series_contains(nba_df['Prev_home_desc2'], "Transition", case=False) | series_contains(nba_df['Prev_home_desc'], "Transition", case=False) | series_contains(nba_df['home_description'], "Flagrant", case=False)), True,
        series_contains(nba_df['visitor_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False, regex=True) & \
            ~(series_contains(nba_df['Prev_visitor_desc2'], "Transition", case=False) | series_contains(nba_df['Prev_visitor_desc'], "Transition", case=False) | series_contains(nba_df['visitor_description'], "Flagrant", case=False)), True,
        False
    ).astype(bool)

    nba_df['TeamOnOffense'] = case_when(
        series_contains(nba_df['home_description'], r'Flagrant (2 of 2|3 of 3)', case=False, regex=True), "Home",
        series_contains(nba_df['visitor_description'], r'Flagrant (2 of 2|3 of 3)', case=False, regex=True), "Away",
        series_contains(nba_df['home_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_visitor_desc'], "MISS", case=False), "Away",
        series_contains(nba_df['visitor_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_home_desc'], "MISS", case=False), "Home",
        series_contains(nba_df['visitor_description'], "MISS", case=False), "Away",
        series_contains(nba_df['home_description'], "MISS", case=False), "Home",
        series_contains(nba_df['visitor_description'], "MAKE", case=False), "Away",
        series_contains(nba_df['home_description'], "MAKE", case=False), "Home",
        series_contains(nba_df['home_description'], "Turnover", case=False), "Home",
        series_contains(nba_df['visitor_description'], "Turnover", case=False), "Away",
        series_contains(nba_df['home_description'], "PTS", case=False), "Home",
        series_contains(nba_df['visitor_description'], "PTS", case=False), "Away",
        series_contains(nba_df['home_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False, regex=True), "Home",
        series_contains(nba_df['visitor_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False, regex=True), "Away",
        ""
    )
    print("      Calculated EOP, TeamOnOffense for TS")

    # Identify Potential Fouls
    window_size = 5
    group_cols = ['game_id'] if 'game_id' in nba_df.columns else []
    if group_cols:
        ft_events = nba_df.groupby(group_cols)['event_type'].transform(
            lambda x: (x == "FreeThrow").rolling(window=window_size, min_periods=1).sum().shift(-(window_size - 1)).fillna(0)
        )
        sub_events = nba_df.groupby(group_cols)['event_type'].transform(
            lambda x: (x == "Substitution").rolling(window=window_size, min_periods=1).sum().shift(-(window_size - 1)).fillna(0)
        )
    else:
        ft_events = (nba_df['event_type'] == "FreeThrow").rolling(window=window_size, min_periods=1).sum().shift(-(window_size - 1)).fillna(0)
        sub_events = (nba_df['event_type'] == "Substitution").rolling(window=window_size, min_periods=1).sum().shift(-(window_size - 1)).fillna(0)

    exclude_foul_pattern = r'T\.FOUL|FLAGRANT|Offens|Transition'
    nba_df['PotentialFoul'] = case_when(
        (ft_events > 0) & (sub_events > 0) & (nba_df['event_type'] == "Foul") & \
        ~series_contains(nba_df['visitor_description'], exclude_foul_pattern, case=False, regex=True) & \
        ~series_contains(nba_df['home_description'], exclude_foul_pattern, case=False, regex=True), True,
        False
    ).astype(bool)
    print(f"      Identified {nba_df['PotentialFoul'].sum()} potential fouls")

    # Apply Propagation (skip ft_off_check for TS)
    nba_df_propagated = propagate_player_values_py(nba_df)
    nba_df_checked = nba_df_propagated

    # Map O/D players
    o_mapping = {}
    d_mapping = {}
    for i in range(1, 6):
        o_mapping[f'O{i}'] = np.where(
            nba_df_checked['TeamOnOffense'] == "Away", nba_df_checked[f'a{i}'],
            np.where(nba_df_checked['TeamOnOffense'] == "Home", nba_df_checked[f'h{i}'], np.nan)
        )
        d_mapping[f'D{i}'] = np.where(
            nba_df_checked['TeamOnOffense'] == "Away", nba_df_checked[f'h{i}'],
            np.where(nba_df_checked['TeamOnOffense'] == "Home", nba_df_checked[f'a{i}'], np.nan)
        )
    nba_df_checked = nba_df_checked.assign(**o_mapping, **d_mapping)
    print("      Mapped O/D Players")

    # Join Player Season Stats for FT Luck Adjustment
    is_playoffs = "_PS" in os.path.basename(file_path).upper()
    player_stats_df = fetch_player_stats_supabase(year, is_playoffs)

    shooter_id_col = 'player1_id'
    if player_stats_df is not None and shooter_id_col in nba_df_checked.columns:
        print("      Preparing data for player stats merge (TS)...")
        nba_df_checked['player1_id_numeric'] = pd.to_numeric(nba_df_checked[shooter_id_col], errors='coerce')
        player_stats_df['PlayerID'] = pd.to_numeric(player_stats_df['PlayerID'], errors='coerce').astype('Int64')
        nba_df_checked = nba_df_checked.dropna(subset=['player1_id_numeric'])
        player_stats_df = player_stats_df.dropna(subset=['PlayerID'])
        nba_df_checked['player1_id_numeric'] = nba_df_checked['player1_id_numeric'].astype('Int64')

        nba_df_checked = pd.merge(
            nba_df_checked,
            player_stats_df[['PlayerID', 'FTPerc']],
            left_on='player1_id_numeric',
            right_on='PlayerID',
            how='left'
        )
        nba_df_checked = nba_df_checked.drop(columns=['player1_id_numeric', 'PlayerID'], errors='ignore')
        print(f"      Successfully merged FT stats for {nba_df_checked['FTPerc'].notna().sum()} rows (TS).")
    else:
        print(f"      Skipping player stats merge for TS. Using defaults.")
        if 'FTPerc' not in nba_df_checked.columns:
            nba_df_checked['FTPerc'] = np.nan

    # Calculate FT Luck-Adjusted Net_Total
    default_ft_perc = 0.75
    ft_perc = pd.to_numeric(nba_df_checked['FTPerc'], errors='coerce')
    missing_ft_perc = ft_perc.isna()
    nba_df_checked['ExpFT'] = ft_perc.fillna(default_ft_perc).astype(float)

    is_ft_event = series_contains(nba_df_checked['home_description'], "Free Throw", case=False) | \
                  series_contains(nba_df_checked['visitor_description'], "Free Throw", case=False) | \
                  (nba_df_checked['event_type'] == "FreeThrow")

    actual_action_score = (nba_df_checked['Home_Action_Score'] + nba_df_checked['Away_Action_Score']).astype(float)
    ft_action_score = nba_df_checked['ExpFT']
    if missing_ft_fallback == "actual":
        ft_action_score = np.where(missing_ft_perc, actual_action_score, nba_df_checked['ExpFT'])
        print("      Missing FT% fallback for TS: actual FT results")
    nba_df_checked['LA_Action_Score'] = np.where(is_ft_event, ft_action_score, actual_action_score)

    if group_cols:
        nba_df_checked['LA_Net_Total'] = nba_df_checked.groupby(group_cols)['LA_Action_Score'].cumsum()
    else:
        nba_df_checked['LA_Net_Total'] = nba_df_checked['LA_Action_Score'].cumsum()
    print("      Calculated FT Luck-Adjusted Net_Total for TS")

    # Filter for EOP & Calculate Diffs
    if 'End_of_Possession' in nba_df_checked.columns and pd.api.types.is_bool_dtype(nba_df_checked['End_of_Possession']):
        nba_filt = nba_df_checked[nba_df_checked['End_of_Possession']].copy()
    else:
        print("      Warning: 'End_of_Possession' column missing or not boolean for filtering.")
        nba_filt = nba_df_checked.copy()
    print(f"      Filtered down to {len(nba_filt)} TS End of Possession rows (before TO removal)")

    initial_rows_ts = len(nba_filt)
    nba_filt = nba_filt[nba_filt['event_type'] != 'Turnover'].copy()
    print(f"      Filtered out {initial_rows_ts - len(nba_filt)} Turnover possessions for TS")

    if not nba_filt.empty:
        if 'LA_Net_Total' in nba_filt.columns:
            if group_cols:
                nba_filt['Net_Diff'] = nba_filt['LA_Net_Total'] - nba_filt.groupby(group_cols)['LA_Net_Total'].shift(fill_value=0)
            else:
                nba_filt['Net_Diff'] = nba_filt['LA_Net_Total'] - nba_filt['LA_Net_Total'].shift(fill_value=0)
            print("      Calculated Net_Diff for TS (FT luck-adjusted)")
        else:
            print("      Warning: 'LA_Net_Total' column missing for TS Net_Diff calculation.")
            nba_filt['Net_Diff'] = 0

    # Finalize
    nba_ts_output = _finalize_df(nba_filt, 'Net_Diff', year)

    end_time = time.time()
    print(f"  Finished TS Processing. Time: {end_time - start_time:.2f} seconds.")
    return nba_ts_output
