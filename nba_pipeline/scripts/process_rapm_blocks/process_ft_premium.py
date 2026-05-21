"""
FT_PREMIUM (Free Throw premium, possession-based) processor.

Decomposition component of dTS. Uses the same TS denominator
(FGA + FT possessions, turnovers excluded) so ridge shrinkage is
consistent across the three components.

Per-event scoring varies by foul type. The baseline represents the
marginal value above what would have occurred on a normal possession
(avg_actual_pts ≈ 1.09 pts/FGA, computed dynamically from the data):

  - FGA (MAKE or MISS)    → 0.0  (neutral)
  - And-1 FT (1 of 1)     → ExpFT - 0            (pure bonus, basket already counted)
  - Technical FT           → ExpFT - 0            (pure bonus, possession continues)
  - 2-shot foul FT         → ExpFT - avg_pts/2    (replaces ~1.09 possession, per FT)
  - 3-shot foul FT         → ExpFT - avg_pts/3    (replaces ~1.09 possession, per FT)
  - Everything else        → 0.0

Example (90% FT shooter, avg_pts=1.09):
  And-1:   0.90 - 0     = +0.900 per FT  (total: +0.90)
  2-shot:  0.90 - 0.545 = +0.355 per FT  (total: +0.71)
  3-shot:  0.90 - 0.363 = +0.537 per FT  (total: +1.61)

Available for all seasons with play-by-play + FT% coverage. Unlike SQ_POSS
and CONTEST, this metric does not depend on ShotQuality inputs.
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


def process_ft_premium_py(file_path, year, season_str):
    """
    Processes data for possession-based FT Premium RAPM (FT_PREMIUM).

    Uses TS denominator (FGA + FT possessions, no turnovers).
    FGA value = 1.0 (neutral), FT value = ExpFT (player-specific).
    Captures FT drawing + FT shooting premium.
    Available historically because it only depends on FT events and FT%.
    """
    print(f"  Starting FT_PREMIUM Processing for {season_str}...")
    nba_df = _base_processing(file_path)
    if nba_df is None:
        return None
    start_time = time.time()

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
    print("      Calculated EOP, TeamOnOffense for FT_PREMIUM")

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

    # --- Compute avg actual pts per FGA (dynamic baseline) ---
    is_fga = nba_df_checked['event_type'].isin(['MAKE', 'MISS'])
    desc_all = (nba_df_checked['home_description'].fillna('') + ' ' +
                nba_df_checked['visitor_description'].fillna(''))
    fga_made = is_fga & (nba_df_checked['event_type'] == 'MAKE')
    fga_3pt = desc_all.str.contains('3PT', case=False, na=False)
    # Points per made FGA: 3 if 3PT, else 2; 0 for misses
    fga_pts = np.where(fga_made & fga_3pt, 3.0, np.where(fga_made, 2.0, 0.0))
    avg_actual_pts = fga_pts[is_fga].mean() if is_fga.any() else 1.09
    print(f"      Avg actual pts per FGA: {avg_actual_pts:.4f}")

    # --- Classify FT type from descriptions ---
    is_ft_event = (
        series_contains(nba_df_checked['home_description'], "Free Throw", case=False) |
        series_contains(nba_df_checked['visitor_description'], "Free Throw", case=False) |
        (nba_df_checked['event_type'] == "FreeThrow")
    )
    is_tech_ft = (
        series_contains(nba_df_checked['home_description'], "Technical", case=False) |
        series_contains(nba_df_checked['visitor_description'], "Technical", case=False)
    ) & is_ft_event
    is_1of1 = (
        series_contains(desc_all, r'\b1 of 1\b', regex=True) & is_ft_event
    )
    is_3shot = (
        series_contains(desc_all, r'\b[123] of 3\b', regex=True) & is_ft_event
    )
    is_2shot = (
        series_contains(desc_all, r'\b[12] of 2\b', regex=True) & is_ft_event
    )
    # And-1 = 1 of 1 that isn't a technical
    is_and1 = is_1of1 & ~is_tech_ft

    # Baseline per FT type:
    #   And-1 / Tech: 0 (pure bonus)
    #   2-shot: avg_actual_pts / 2
    #   3-shot: avg_actual_pts / 3
    ft_baseline = np.where(
        is_and1 | is_tech_ft, 0.0,
        np.where(is_3shot, avg_actual_pts / 3.0,
                 np.where(is_2shot, avg_actual_pts / 2.0,
                          0.0))  # fallback for unclassified FTs
    )

    nba_df_checked['FT_Action_Score'] = case_when(
        is_ft_event, nba_df_checked['ExpFT'] - ft_baseline,
        is_fga,      0.0,
        0.0
    )

    # Log FT type breakdown
    n_and1 = is_and1.sum()
    n_tech = is_tech_ft.sum()
    n_2shot = (is_2shot & ~is_tech_ft & ~is_1of1).sum()
    n_3shot = is_3shot.sum()
    print(f"      FT types: and-1={n_and1:,}, tech={n_tech:,}, 2-shot={n_2shot:,}, 3-shot={n_3shot:,}")
    print(f"      Baselines: and-1/tech=0, 2-shot={avg_actual_pts/2:.4f}, 3-shot={avg_actual_pts/3:.4f}")
    print(f"      Calculated FT_Action_Score (foul-type-aware baselines)")

    # --- Cumulate within game ---
    if group_cols:
        nba_df_checked['FT_Total'] = nba_df_checked.groupby(group_cols)['FT_Action_Score'].cumsum()
    else:
        nba_df_checked['FT_Total'] = nba_df_checked['FT_Action_Score'].cumsum()

    # --- Filter to EOP, remove turnovers ---
    if 'End_of_Possession' in nba_df_checked.columns and pd.api.types.is_bool_dtype(nba_df_checked['End_of_Possession']):
        nba_filt = nba_df_checked[nba_df_checked['End_of_Possession']].copy()
    else:
        print("      Warning: 'End_of_Possession' column missing or not boolean.")
        nba_filt = nba_df_checked.copy()
    print(f"      Filtered to {len(nba_filt)} EOP rows (before TO removal)")

    initial_rows = len(nba_filt)
    nba_filt = nba_filt[nba_filt['event_type'] != 'Turnover'].copy()
    print(f"      Removed {initial_rows - len(nba_filt)} Turnover possessions for FT_PREMIUM")

    # --- Compute diffs ---
    if not nba_filt.empty and 'FT_Total' in nba_filt.columns:
        if group_cols:
            nba_filt['Net_Diff'] = nba_filt['FT_Total'] - nba_filt.groupby(group_cols)['FT_Total'].shift(fill_value=0)
        else:
            nba_filt['Net_Diff'] = nba_filt['FT_Total'] - nba_filt['FT_Total'].shift(fill_value=0)
        print("      Calculated Net_Diff for FT_PREMIUM")

    if 'Net_Diff' not in nba_filt.columns:
        print("      Error: Net_Diff missing for FT_PREMIUM.")
        return None

    output = _finalize_df(nba_filt, 'Net_Diff', year)

    end_time = time.time()
    print(f"  Finished FT_PREMIUM Processing. Time: {end_time - start_time:.2f} seconds.")
    return output
