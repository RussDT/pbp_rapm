"""
SECOND_CHANCE processor.

Possession-based RAPM measuring second chance points per 100 possessions.
Every possession gets a row, but only points scored AFTER a missed FGA by the
offensive team within that possession count. Most possessions will have 0.

Definition: A "second chance" occurs when the offensive team misses a field goal
attempt but retains possession and subsequently scores. This captures:
- Putbacks after offensive rebounds
- Kick-outs to shooters after OREBs
- And-one FTs after miss → OREB → foul sequences
- Any scoring after the offense gets a second (or third, etc.) chance
"""

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
)


def process_second_chance_py(file_path, year, season_str):
    """
    Processes data for Second Chance Points RAPM calculation.

    Detects possessions where the offensive team misses a FGA but still scores.
    Only counts scoring events that occur after the first offensive miss in each
    possession. Uses unsigned scoring (always positive) since rapm.py handles
    offensive/defensive attribution via O/D player mapping.

    Returns:
        DataFrame with Off_Diff/Def_Diff/Net_Diff at EOP rows, or None on failure.
    """
    print(f"  Starting SECOND_CHANCE Processing for {season_str}...")
    nba_df = _base_processing(file_path)
    if nba_df is None:
        return None
    start_time = time.time()

    # --- Possession Boundaries (same as RAPM) ---
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
    print("      Calculated EOP, TeamOnOffense for SECOND_CHANCE")

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

    # --- Create Possession Groups ---
    # Possession group = cumsum of shifted EOP (new possession starts after each EOP)
    if group_cols:
        nba_df_checked['poss_group'] = nba_df_checked.groupby(group_cols)['End_of_Possession'].transform(
            lambda x: x.shift(1, fill_value=False).cumsum()
        )
    else:
        nba_df_checked['poss_group'] = nba_df_checked['End_of_Possession'].shift(1, fill_value=False).cumsum()

    poss_group_key = group_cols + ['poss_group'] if group_cols else ['poss_group']

    # --- Back-fill TeamOnOffense from EOP row to all rows in possession ---
    # TeamOnOffense is only set at EOP rows; we need it for all rows to know
    # which team's misses count as "offensive misses"
    nba_df_checked['poss_offense'] = nba_df_checked.groupby(poss_group_key)['TeamOnOffense'].transform(
        lambda x: x.replace('', np.nan).bfill().ffill()
    )

    # --- Detect Missed FGAs by the Offensive Team ---
    # event_type == "MISS" is a missed field goal (not FTs — those are event_type "FreeThrow")
    is_miss = nba_df_checked['event_type'] == "MISS"

    # Determine which team missed: home miss has MISS in home_description, away in visitor_description
    home_miss = is_miss & series_contains(nba_df_checked['home_description'], "MISS", case=False)
    away_miss = is_miss & series_contains(nba_df_checked['visitor_description'], "MISS", case=False)

    # Offensive miss = the miss belongs to whichever team is on offense for this possession
    offensive_miss = (
        (home_miss & (nba_df_checked['poss_offense'] == "Home")) |
        (away_miss & (nba_df_checked['poss_offense'] == "Away"))
    ).astype(bool)

    total_off_misses = offensive_miss.sum()
    print(f"      Detected {total_off_misses} offensive FGA misses")

    # --- Second Chance Flag: True from first offensive miss onward ---
    nba_df_checked['off_miss'] = offensive_miss
    nba_df_checked['after_miss'] = nba_df_checked.groupby(poss_group_key)['off_miss'].transform('cummax').astype(bool)

    # But the miss itself doesn't count — only events AFTER the miss.
    # Shift the flag forward: after_miss should be False on the miss row, True on next row onward.
    # Use cumsum > 0 on shifted values: if we've seen a miss in a prior row, we're in second chance.
    nba_df_checked['after_miss'] = nba_df_checked.groupby(poss_group_key)['off_miss'].transform(
        lambda x: x.shift(1, fill_value=False).cummax()
    ).astype(bool)

    sc_possessions = nba_df_checked.groupby(poss_group_key)['after_miss'].max().sum()
    print(f"      {sc_possessions} possessions with events after an offensive miss")

    # --- Per-Event Unsigned Scoring ---
    # Use Home_Action_Score + Away_Action_Score (always positive, like standard RAPM)
    # rapm.py handles offensive/defensive attribution via O/D player mapping
    event_pts = (nba_df_checked['Home_Action_Score'] + nba_df_checked['Away_Action_Score']).astype(float)

    # Second chance scoring: zero out events not after an offensive miss
    nba_df_checked['SC_Net'] = np.where(nba_df_checked['after_miss'], event_pts, 0.0)

    sc_total_pts = nba_df_checked['SC_Net'].sum()
    print(f"      Total second chance points in dataset: {sc_total_pts:.0f}")

    # --- Cumulate Within Game ---
    if group_cols:
        nba_df_checked['SC_Total'] = nba_df_checked.groupby(group_cols)['SC_Net'].cumsum()
    else:
        nba_df_checked['SC_Total'] = nba_df_checked['SC_Net'].cumsum()

    # --- Filter to EOP & Compute Diffs ---
    if 'End_of_Possession' in nba_df_checked.columns and pd.api.types.is_bool_dtype(nba_df_checked['End_of_Possession']):
        nba_filt = nba_df_checked[nba_df_checked['End_of_Possession']].copy()
    else:
        print("      Warning: 'End_of_Possession' column missing or not boolean.")
        nba_filt = nba_df_checked.copy()
    print(f"      Filtered to {len(nba_filt)} SECOND_CHANCE End of Possession rows")

    if not nba_filt.empty and 'SC_Total' in nba_filt.columns:
        if group_cols:
            nba_filt['Off_Diff'] = nba_filt['SC_Total'] - nba_filt.groupby(group_cols)['SC_Total'].shift(fill_value=0)
        else:
            nba_filt['Off_Diff'] = nba_filt['SC_Total'] - nba_filt['SC_Total'].shift(fill_value=0)
        # Use the same sign convention as other offense-valued metrics so
        # defensive coefficients remain comparable to parent RAPM/TS surfaces.
        nba_filt['Def_Diff'] = -nba_filt['Off_Diff']
        nba_filt['Net_Diff'] = nba_filt['Off_Diff']

        nonzero = (nba_filt['Off_Diff'] != 0).sum()
        print(f"      {nonzero} possessions with non-zero second chance points ({nonzero/len(nba_filt)*100:.1f}%)")
        print("      Calculated diffs for SECOND_CHANCE")

    diff_cols = ['Net_Diff', 'Off_Diff', 'Def_Diff']
    diff_cols_present = [c for c in diff_cols if c in nba_filt.columns]
    if not diff_cols_present:
        print("      Error: Could not find diff columns for SECOND_CHANCE output.")
        return None

    output = _finalize_df(nba_filt, diff_cols_present, year)

    end_time = time.time()
    print(f"  Finished SECOND_CHANCE Processing. Time: {end_time - start_time:.2f} seconds.")
    return output
