"""
CONTEST (Shot execution / contest quality, possession-based) processor.

Decomposition component of dTS. Uses the same TS denominator
(FGA + FT possessions, turnovers excluded) so ridge shrinkage is
consistent across the three components.

Per-event scoring:
  - FGA (MAKE or MISS) → actual_pts - initial_ev  (execution above expectation)
  - FT event           → 0.0                       (no FT signal in contest)
  - Everything else    → 0.0

Positive offense = shooting above shot quality expectation.
Negative defense = allowing opponents to shoot above their initial_ev.

Only available for years 24-26 (requires initial_ev column).
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
    normalize_season_end_year,
)


def process_contest_py(file_path, year, season_str):
    """
    Processes data for possession-based CONTEST RAPM.

    Uses TS denominator (FGA + FT possessions, no turnovers).
    FGA value = actual_pts - initial_ev, FT value = 0.0 (no FT signal).
    Captures shot-making execution and defensive contest quality.
    Only available for years 24-26.
    """
    season_end_year = normalize_season_end_year(year)
    if season_end_year < 2024:
        print(
            f"  Skipping CONTEST - only available for years 24-26 "
            f"(season end year {season_end_year} < 2024)"
        )
        return None

    print(f"  Starting CONTEST Processing for {season_str}...")
    nba_df = _base_processing(file_path)
    if nba_df is None:
        return None
    start_time = time.time()

    if 'initial_ev' not in nba_df.columns:
        print("      Warning: 'initial_ev' column not found. Returning None.")
        return None

    # --- TS Possession Boundaries (same denominator as TS RAPM) ---
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
    print("      Calculated EOP, TeamOnOffense for CONTEST")

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
    nba_df_checked = nba_df_propagated

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

    # --- Per-Event Value: FGA=actual_pts-initial_ev, FT=0.0, other=0.0 ---
    is_ft_event = (
        series_contains(nba_df_checked['home_description'], "Free Throw", case=False) |
        series_contains(nba_df_checked['visitor_description'], "Free Throw", case=False) |
        (nba_df_checked['event_type'] == "FreeThrow")
    )
    is_fga = nba_df_checked['event_type'].isin(['MAKE', 'MISS'])

    actual_action_score = (
        nba_df_checked['Home_Action_Score'] + nba_df_checked['Away_Action_Score']
    ).astype(float)

    # Compute avg actual pts per FGA for calibration and fill
    desc_all = (nba_df_checked['home_description'].fillna('') + ' ' +
                nba_df_checked['visitor_description'].fillna(''))
    fga_made = is_fga & (nba_df_checked['event_type'] == 'MAKE')
    fga_3pt = desc_all.str.contains('3PT', case=False, na=False)
    fga_pts = np.where(fga_made & fga_3pt, 3.0, np.where(fga_made, 2.0, 0.0))
    avg_actual_pts = fga_pts[is_fga].mean() if is_fga.any() else 1.09
    print(f"      Avg actual pts per FGA: {avg_actual_pts:.4f}")

    # Calibrate initial_ev: shift so mean matches avg actual pts
    ev_mask = is_fga & nba_df_checked['initial_ev'].notna()
    mean_ev = nba_df_checked.loc[ev_mask, 'initial_ev'].mean()
    ev_delta = avg_actual_pts - mean_ev
    nba_df_checked['initial_ev_calibrated'] = nba_df_checked['initial_ev'] + ev_delta
    print(f"      Calibrated initial_ev: mean {mean_ev:.4f} → {avg_actual_pts:.4f} (delta={ev_delta:+.4f})")

    # Fill missing with avg actual pts (actual - avg_actual = neutral for CONTEST)
    n_missing = nba_df_checked['initial_ev'].isna().sum()
    nba_df_checked['initial_ev_filled'] = nba_df_checked['initial_ev_calibrated'].fillna(avg_actual_pts)
    print(f"      Filled {n_missing:,} missing initial_ev with avg_actual_pts={avg_actual_pts:.4f}")

    nba_df_checked['Contest_Action_Score'] = case_when(
        is_ft_event, 0.0,
        is_fga,      actual_action_score - nba_df_checked['initial_ev_filled'],
        0.0
    )
    print("      Calculated Contest_Action_Score (FGA=actual_pts-calibrated_initial_ev, FT=0.0)")

    # --- Cumulate within game ---
    if group_cols:
        nba_df_checked['Contest_Total'] = nba_df_checked.groupby(group_cols)['Contest_Action_Score'].cumsum()
    else:
        nba_df_checked['Contest_Total'] = nba_df_checked['Contest_Action_Score'].cumsum()

    # --- Filter to EOP, remove turnovers ---
    if 'End_of_Possession' in nba_df_checked.columns and pd.api.types.is_bool_dtype(nba_df_checked['End_of_Possession']):
        nba_filt = nba_df_checked[nba_df_checked['End_of_Possession']].copy()
    else:
        print("      Warning: 'End_of_Possession' column missing or not boolean.")
        nba_filt = nba_df_checked.copy()
    print(f"      Filtered to {len(nba_filt)} EOP rows (before TO removal)")

    initial_rows = len(nba_filt)
    nba_filt = nba_filt[nba_filt['event_type'] != 'Turnover'].copy()
    print(f"      Removed {initial_rows - len(nba_filt)} Turnover possessions for CONTEST")

    # --- Compute diffs ---
    if not nba_filt.empty and 'Contest_Total' in nba_filt.columns:
        if group_cols:
            nba_filt['Net_Diff'] = nba_filt['Contest_Total'] - nba_filt.groupby(group_cols)['Contest_Total'].shift(fill_value=0)
        else:
            nba_filt['Net_Diff'] = nba_filt['Contest_Total'] - nba_filt['Contest_Total'].shift(fill_value=0)
        print("      Calculated Net_Diff for CONTEST")

    if 'Net_Diff' not in nba_filt.columns:
        print("      Error: Net_Diff missing for CONTEST.")
        return None

    output = _finalize_df(nba_filt, 'Net_Diff', year)

    end_time = time.time()
    print(f"  Finished CONTEST Processing. Time: {end_time - start_time:.2f} seconds.")
    return output
