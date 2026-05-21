"""
INITIAL_EV_BETA processor.

Possession-based RAPM using:
- Initial EV for ALL FGAs (rim and non-rim alike)
- Expected FT value for free throws
- 0 for turnovers and everything else

Unlike SPECIAL_RAPM (which uses actual 2/0 for rim attempts),
this substitutes initial_ev uniformly across all shot attempts.

Only available for years 24-26.
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
    ft_off_check_py,
    fetch_player_stats_supabase,
    normalize_season_end_year,
)


def process_initial_ev_beta_py(file_path, year, season_str):
    """
    Processes data for Initial EV Beta RAPM calculation.

    Uses:
    - Initial EV for all FGAs (both rim and non-rim)
    - Expected FT value for free throws
    - 0 for everything else (turnovers, etc.)

    Only available for years 24-26 (seasons 2023-24 through 2025-26).
    """
    season_end_year = normalize_season_end_year(year)
    if season_end_year < 2024:
        print(
            f"  Skipping INITIAL_EV_BETA - only available for years 24-26 "
            f"(season end year {season_end_year} < 2024)"
        )
        return None

    print(f"  Starting INITIAL_EV_BETA Processing for {season_str}...")
    nba_df = _base_processing(file_path)
    if nba_df is None:
        return None
    start_time = time.time()

    if 'initial_ev' not in nba_df.columns:
        print("      Warning: 'initial_ev' column not found. Returning None.")
        return None

    # --- Possession Boundaries (same as RAPM / SPECIAL_RAPM) ---
    ft_exclude_pattern = r'\b(1 of [23]|2 of 3|Technical|Flagrant)\b'

    nba_df['offensive_FT_Rebound'] = case_when(
        series_contains(nba_df['visitor_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_visitor_desc'], "MISS", case=False) & series_contains(nba_df['Prev_visitor_desc'], "Free Throw", case=False) & ~series_contains(nba_df['Prev_visitor_desc'], ft_exclude_pattern, case=False, regex=True), True,
        series_contains(nba_df['home_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_home_desc'], "MISS", case=False) & series_contains(nba_df['Prev_home_desc'], "Free Throw", case=False) & ~series_contains(nba_df['Prev_home_desc'], ft_exclude_pattern, case=False, regex=True), True,
        False
    ).astype(bool)

    nba_df['End_of_Possession'] = case_when(
        (nba_df['event_type'] == "EndOfPeriod") & (nba_df['prev_seconds'] > 0), True,
        series_contains(nba_df['home_description'], r'Flagrant (2 of 2|3 of 3)', case=False, regex=True), True,
        series_contains(nba_df['visitor_description'], r'Flagrant (2 of 2|3 of 3)', case=False, regex=True), True,
        (nba_df['event_type'] == "Turnover"), True,
        series_contains(nba_df['home_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_visitor_desc'], "MISS", case=False) & ~series_contains(nba_df['Prev_visitor_desc'], ft_exclude_pattern, case=False, regex=True), True,
        series_contains(nba_df['visitor_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_home_desc'], "MISS", case=False) & ~series_contains(nba_df['Prev_home_desc'], ft_exclude_pattern, case=False, regex=True), True,
        series_contains(nba_df['home_description'], "PTS", case=False) & (nba_df['event_type'] == "MAKE") & ~series_contains(nba_df['Next_visitor_desc'], "S.FOUL", case=True), True,
        series_contains(nba_df['visitor_description'], "PTS", case=False) & (nba_df['event_type'] == "MAKE") & ~series_contains(nba_df['Next_home_desc'], "S.FOUL", case=True), True,
        series_contains(nba_df['home_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False, regex=True) & series_contains(nba_df['home_description'], "PTS", case=False) & \
            ~(series_contains(nba_df['Prev_visitor_desc2'], "Transition", case=False) | series_contains(nba_df['Prev_visitor_desc'], "Transition", case=False) | series_contains(nba_df['home_description'], "Flagrant", case=False)), True,
        series_contains(nba_df['visitor_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False, regex=True) & series_contains(nba_df['visitor_description'], "PTS", case=False) & \
            ~(series_contains(nba_df['Prev_home_desc2'], "Transition", case=False) | series_contains(nba_df['Prev_home_desc'], "Transition", case=False) | series_contains(nba_df['visitor_description'], "Flagrant", case=False)), True,
        False
    ).astype(bool)

    nba_df['TeamOnOffense'] = case_when(
        series_contains(nba_df['home_description'], r'Flagrant (2 of 2|3 of 3)', case=False, regex=True), "Home",
        series_contains(nba_df['visitor_description'], r'Flagrant (2 of 2|3 of 3)', case=False, regex=True), "Away",
        series_contains(nba_df['home_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_visitor_desc'], "MISS", case=False), "Away",
        series_contains(nba_df['visitor_description'], "REBOUND", case=False) & series_contains(nba_df['Prev_home_desc'], "MISS", case=False), "Home",
        series_contains(nba_df['home_description'], "Turnover", case=False), "Home",
        series_contains(nba_df['visitor_description'], "Turnover", case=False), "Away",
        series_contains(nba_df['home_description'], "PTS", case=False) & (nba_df['event_type'] == "MAKE") & ~series_contains(nba_df['Next_visitor_desc'], "S.FOUL", case=True), "Home",
        series_contains(nba_df['visitor_description'], "PTS", case=False) & (nba_df['event_type'] == "MAKE") & ~series_contains(nba_df['Next_home_desc'], "S.FOUL", case=True), "Away",
        series_contains(nba_df['home_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False, regex=True) & series_contains(nba_df['home_description'], "PTS", case=False) & \
            ~(series_contains(nba_df['Prev_visitor_desc2'], "Transition", case=False) | series_contains(nba_df['Prev_visitor_desc'], "Transition", case=False) | series_contains(nba_df['home_description'], "Flagrant", case=False)), "Home",
        series_contains(nba_df['visitor_description'], r'\b(1 of 1|2 of 2|3 of 3)\b', case=False, regex=True) & series_contains(nba_df['visitor_description'], "PTS", case=False) & \
            ~(series_contains(nba_df['Prev_home_desc2'], "Transition", case=False) | series_contains(nba_df['Prev_home_desc'], "Transition", case=False) | series_contains(nba_df['visitor_description'], "Flagrant", case=False)), "Away",
        ""
    )
    print("      Calculated EOP, TeamOnOffense for INITIAL_EV_BETA")

    # --- Potential Fouls ---
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

    # --- Propagate & Map Players ---
    nba_df_propagated = propagate_player_values_py(nba_df)
    nba_df_checked = ft_off_check_py(nba_df_propagated)

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

    # --- FT Expected Value (player-specific FT%) ---
    is_playoffs = "_PS" in os.path.basename(file_path).upper()
    player_stats_df = fetch_player_stats_supabase(year, is_playoffs)

    shooter_id_col = 'player1_id'
    if player_stats_df is not None and shooter_id_col in nba_df_checked.columns:
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
        print(f"      Merged FT stats for {nba_df_checked['FTPerc'].notna().sum()} rows.")
    else:
        print("      Skipping player stats merge. Using defaults.")
        if 'FTPerc' not in nba_df_checked.columns:
            nba_df_checked['FTPerc'] = np.nan

    default_ft_perc = 0.75
    nba_df_checked['ExpFT'] = nba_df_checked['FTPerc'].fillna(default_ft_perc).astype(float)

    # --- Per-Event Value ---
    is_ft_event = (
        series_contains(nba_df_checked['home_description'], "Free Throw", case=False) |
        series_contains(nba_df_checked['visitor_description'], "Free Throw", case=False) |
        (nba_df_checked['event_type'] == "FreeThrow")
    )
    is_fga = nba_df_checked['event_type'].isin(['MAKE', 'MISS'])

    # Fill missing initial_ev with league average
    nba_df_checked['initial_ev_filled'] = nba_df_checked['initial_ev'].fillna(1.0)

    # FTs → ExpFT, all FGAs → initial_ev, everything else → 0
    nba_df_checked['BetaNet'] = case_when(
        is_ft_event, nba_df_checked['ExpFT'],
        is_fga,      nba_df_checked['initial_ev_filled'],
        0.0
    )
    print("      Calculated BetaNet (FTs=ExpFT, all FGAs=initial_ev)")

    # --- Cumulate within game ---
    if group_cols:
        nba_df_checked['BetaTotal'] = nba_df_checked.groupby(group_cols)['BetaNet'].cumsum()
    else:
        nba_df_checked['BetaTotal'] = nba_df_checked['BetaNet'].cumsum()

    # --- Filter to EOP & Compute Diffs ---
    if 'End_of_Possession' in nba_df_checked.columns and pd.api.types.is_bool_dtype(nba_df_checked['End_of_Possession']):
        nba_filt = nba_df_checked[nba_df_checked['End_of_Possession']].copy()
    else:
        print("      Warning: 'End_of_Possession' column missing or not boolean.")
        nba_filt = nba_df_checked.copy()
    print(f"      Filtered to {len(nba_filt)} INITIAL_EV_BETA End of Possession rows")

    if not nba_filt.empty and 'BetaTotal' in nba_filt.columns:
        if group_cols:
            nba_filt['Off_Diff'] = nba_filt['BetaTotal'] - nba_filt.groupby(group_cols)['BetaTotal'].shift(fill_value=0)
        else:
            nba_filt['Off_Diff'] = nba_filt['BetaTotal'] - nba_filt['BetaTotal'].shift(fill_value=0)
        nba_filt['Def_Diff'] = nba_filt['Off_Diff']
        nba_filt['Net_Diff'] = nba_filt['Off_Diff']
        print("      Calculated diffs for INITIAL_EV_BETA")

    diff_cols = ['Net_Diff', 'Off_Diff', 'Def_Diff']
    diff_cols_present = [c for c in diff_cols if c in nba_filt.columns]
    if not diff_cols_present:
        print("      Error: Could not find diff columns for INITIAL_EV_BETA output.")
        return None

    output = _finalize_df(nba_filt, diff_cols_present, year)

    end_time = time.time()
    print(f"  Finished INITIAL_EV_BETA Processing. Time: {end_time - start_time:.2f} seconds.")
    return output
