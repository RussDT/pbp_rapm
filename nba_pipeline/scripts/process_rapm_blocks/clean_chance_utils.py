"""
Shared helpers for the clean first-/second-chance decomposition.
"""

import os
import numpy as np
import pandas as pd

from .common import (
    _base_processing,
    build_shot_flags_df,
    case_when,
    fetch_player_stats_supabase,
    prepare_standard_possession_df,
    series_contains,
)


DECOMP_SHOT_ACTION_COLUMNS = [
    'FC_DECOMP_RIM_FREQ_Action_Score',
    'FC_DECOMP_RIM_FG_Action_Score',
    'FC_DECOMP_MID_FG_Action_Score',
    'FC_DECOMP_THREE_FREQ_Action_Score',
    'FC_DECOMP_THREE_FG_Action_Score',
]


def build_decomp_shot_action_components(
    actual_net: pd.Series,
    is_fga: pd.Series,
    is_rim_fga: pd.Series,
    is_mid_fga: pd.Series,
    is_three_fga: pd.Series,
    ft_sq_baseline: pd.Series,
    rim_avg_pts: float,
    mid_avg_pts: float,
    three_avg_pts: float,
) -> pd.DataFrame:
    """Return the exact five-child public DECOMP shot tree at action level."""
    components = pd.DataFrame(index=actual_net.index)
    components['FC_DECOMP_RIM_FREQ_Action_Score'] = np.where(
        is_rim_fga, rim_avg_pts - mid_avg_pts, 0.0
    )
    components['FC_DECOMP_RIM_FG_Action_Score'] = np.where(
        is_rim_fga, actual_net - rim_avg_pts, 0.0
    )
    components['FC_DECOMP_MID_FG_Action_Score'] = np.where(
        is_fga,
        np.where(is_mid_fga, actual_net, mid_avg_pts),
        ft_sq_baseline,
    )
    components['FC_DECOMP_THREE_FREQ_Action_Score'] = np.where(
        is_three_fga, three_avg_pts - mid_avg_pts, 0.0
    )
    components['FC_DECOMP_THREE_FG_Action_Score'] = np.where(
        is_three_fga, actual_net - three_avg_pts, 0.0
    )
    components['FC_DECOMP_EFG_Action_Score'] = components[DECOMP_SHOT_ACTION_COLUMNS].sum(axis=1)
    components['FC_DECOMP_MID_VALUE_Action_Score'] = 0.0
    return components


def _attach_rapm_scoring(nba_df_checked, file_path, year, label):
    """Attach RAPM-style event scoring and, when possible, TS subcomponents."""
    is_playoffs = "_PS" in os.path.basename(file_path).upper()
    player_stats_df = fetch_player_stats_supabase(year, is_playoffs)

    shooter_id_col = 'player1_id'
    if player_stats_df is not None and shooter_id_col in nba_df_checked.columns:
        print(f"      Preparing data for player stats merge ({label})...")
        nba_df_checked['player1_id_numeric'] = pd.to_numeric(
            nba_df_checked[shooter_id_col], errors='coerce'
        )
        player_stats_df['PlayerID'] = pd.to_numeric(
            player_stats_df['PlayerID'], errors='coerce'
        ).astype('Int64')

        nba_df_checked = nba_df_checked.dropna(subset=['player1_id_numeric'])
        player_stats_df = player_stats_df.dropna(subset=['PlayerID'])
        nba_df_checked['player1_id_numeric'] = nba_df_checked['player1_id_numeric'].astype('Int64')

        nba_df_checked = pd.merge(
            nba_df_checked,
            player_stats_df[['PlayerID', 'FTPerc', 'ThreePerc']],
            left_on='player1_id_numeric',
            right_on='PlayerID',
            how='left'
        )
        nba_df_checked = nba_df_checked.drop(
            columns=['player1_id_numeric', 'PlayerID'],
            errors='ignore'
        )
        print(
            f"      Successfully merged player stats for "
            f"{nba_df_checked['FTPerc'].notna().sum()} rows ({label})."
        )
    else:
        print(f"      Skipping player stats merge for {label}. Using defaults.")
        if 'FTPerc' not in nba_df_checked.columns:
            nba_df_checked['FTPerc'] = np.nan
        if 'ThreePerc' not in nba_df_checked.columns:
            nba_df_checked['ThreePerc'] = np.nan

    nba_df_checked['ExpFT'] = nba_df_checked['FTPerc'].fillna(0.75).astype(float)
    nba_df_checked['Exp3PT'] = nba_df_checked['ThreePerc'].fillna(0.35).astype(float)

    desc_all = (
        nba_df_checked['home_description'].fillna('').astype(str) + ' ' +
        nba_df_checked['visitor_description'].fillna('').astype(str)
    ).str.strip()

    is_ft_event = (
        series_contains(nba_df_checked['home_description'], "Free Throw", case=False) |
        series_contains(nba_df_checked['visitor_description'], "Free Throw", case=False) |
        (nba_df_checked['event_type'] == "FreeThrow")
    )
    is_fga = nba_df_checked['event_type'].isin(['MAKE', 'MISS'])
    is_2pt_make_event = (
        series_contains(nba_df_checked['home_description'], "PTS", case=False) &
        ~series_contains(nba_df_checked['home_description'], "3PT", case=False) &
        (nba_df_checked['event_type'] == "MAKE")
    ) | (
        series_contains(nba_df_checked['visitor_description'], "PTS", case=False) &
        ~series_contains(nba_df_checked['visitor_description'], "3PT", case=False) &
        (nba_df_checked['event_type'] == "MAKE")
    )
    is_3pt_event = (
        series_contains(nba_df_checked['home_description'], "3PT", case=False) |
        series_contains(nba_df_checked['visitor_description'], "3PT", case=False)
    )
    actual_score = (
        nba_df_checked['Home_Action_Score'] + nba_df_checked['Away_Action_Score']
    ).astype(float)

    nba_df_checked['ActualNet'] = case_when(
        is_ft_event, nba_df_checked['ExpFT'],
        is_2pt_make_event, 2.0,
        is_3pt_event, actual_score,
        0.0
    )
    print(f"      Calculated RAPM-style event scoring for {label}")

    fga_made = is_fga & (nba_df_checked['event_type'] == 'MAKE')
    fga_3pt = desc_all.str.contains('3PT', case=False, na=False)
    fga_pts = np.where(fga_made & fga_3pt, 3.0, np.where(fga_made, 2.0, 0.0))
    avg_actual_pts = float(fga_pts[is_fga].mean()) if is_fga.any() else 1.09

    is_tech_ft = (
        series_contains(nba_df_checked['home_description'], "Technical", case=False) |
        series_contains(nba_df_checked['visitor_description'], "Technical", case=False)
    ) & is_ft_event
    is_1of1 = series_contains(desc_all, r'\b1 of 1\b', regex=True) & is_ft_event
    is_3shot = series_contains(desc_all, r'\b[123] of 3\b', regex=True) & is_ft_event
    is_2shot = series_contains(desc_all, r'\b[12] of 2\b', regex=True) & is_ft_event
    is_and1 = is_1of1 & ~is_tech_ft

    ft_sq_baseline = np.where(
        is_and1 | is_tech_ft, 0.0,
        np.where(is_3shot, avg_actual_pts / 3.0,
                 np.where(is_2shot, avg_actual_pts / 2.0, 0.0))
    )

    nba_df_checked['FC_FT_Action_Score'] = case_when(
        is_ft_event, nba_df_checked['ExpFT'] - ft_sq_baseline,
        0.0
    )
    nba_df_checked['FC_FT_SQ_BASELINE_Action_Score'] = ft_sq_baseline
    print(f"      Calculated first-chance FT component (avg_actual_pts={avg_actual_pts:.4f}) for {label}")

    if 'initial_ev' not in nba_df_checked.columns:
        print(f"      Skipping first-chance SQ/MAKE subcomponents for {label}; no initial_ev column.")
        return nba_df_checked

    ev_mask = is_fga & nba_df_checked['initial_ev'].notna()
    mean_ev = float(nba_df_checked.loc[ev_mask, 'initial_ev'].mean()) if ev_mask.any() else avg_actual_pts
    ev_delta = avg_actual_pts - mean_ev
    initial_ev_filled = (nba_df_checked['initial_ev'] + ev_delta).fillna(avg_actual_pts).astype(float)

    nba_df_checked['FC_SQ_Action_Score'] = case_when(
        is_ft_event, ft_sq_baseline,
        is_fga, initial_ev_filled,
        0.0
    )
    nba_df_checked['FC_MAKE_Action_Score'] = case_when(
        is_ft_event, 0.0,
        is_fga, actual_score - initial_ev_filled,
        0.0
    )
    print(
        "      Calculated first-chance SQ/MAKE subcomponents "
        f"(avg_actual_pts={avg_actual_pts:.4f}, ev_delta={ev_delta:+.4f}) for {label}"
    )
    return nba_df_checked


def build_clean_chance_base_py(file_path, year, season_str, label):
    """
    Build a shared possession-level clean first-/second-chance split.

    Returns an EOP-filtered dataframe with:
    - First_Chance_Diff
    - Second_Chance_Diff
    - Is_Turnover
    """
    print(f"  Starting {label} base processing for {season_str}...")
    nba_df = _base_processing(file_path)
    if nba_df is None:
        return None

    nba_df_checked, group_cols = prepare_standard_possession_df(nba_df, label)
    nba_df_checked = _attach_rapm_scoring(nba_df_checked, file_path, year, label)

    if group_cols:
        game_groupers = [nba_df_checked[col] for col in group_cols]
        shifted_eop = nba_df_checked.groupby(group_cols)['End_of_Possession'].shift(
            1,
            fill_value=False,
        )
        nba_df_checked['poss_group'] = shifted_eop.groupby(game_groupers, sort=False).cumsum()
    else:
        nba_df_checked['poss_group'] = nba_df_checked['End_of_Possession'].shift(
            1, fill_value=False
        ).cumsum()

    poss_group_key = group_cols + ['poss_group'] if group_cols else ['poss_group']
    poss_offense_source = nba_df_checked['TeamOnOffense'].mask(
        nba_df_checked['TeamOnOffense'].eq(''),
        np.nan,
    )
    poss_groupers = [nba_df_checked[col] for col in poss_group_key]
    nba_df_checked['poss_offense'] = (
        poss_offense_source
        .groupby(poss_groupers, sort=False)
        .bfill()
        .groupby(poss_groupers, sort=False)
        .ffill()
    )

    is_miss = nba_df_checked['event_type'] == "MISS"
    home_miss = is_miss & series_contains(nba_df_checked['home_description'], "MISS", case=False)
    away_miss = is_miss & series_contains(nba_df_checked['visitor_description'], "MISS", case=False)
    nba_df_checked['off_miss'] = (
        (home_miss & (nba_df_checked['poss_offense'] == "Home")) |
        (away_miss & (nba_df_checked['poss_offense'] == "Away"))
    ).astype(bool)

    off_miss_seen = (
        nba_df_checked['off_miss']
        .astype(int)
        .groupby(poss_groupers, sort=False)
        .cumsum()
    )
    nba_df_checked['after_miss'] = (
        off_miss_seen
        .groupby(poss_groupers, sort=False)
        .shift(1, fill_value=0)
        .gt(0)
    )

    shot_flags = build_shot_flags_df(nba_df_checked)
    is_fga = shot_flags['is_fga'].astype(bool)
    is_rim_fga = is_fga & shot_flags['is_rim'].astype(bool) & ~shot_flags['is_3pt'].astype(bool)
    is_three_fga = is_fga & shot_flags['is_3pt'].astype(bool)
    is_mid_fga = is_fga & ~is_rim_fga & ~is_three_fga
    first_chance_fga = is_fga & ~nba_df_checked['after_miss']
    if 'is_transition' in nba_df_checked.columns:
        is_transition_fga = (
            nba_df_checked['is_transition']
            .fillna(False)
            .astype(bool)
            & first_chance_fga
        )
    else:
        is_transition_fga = pd.Series(False, index=nba_df_checked.index)
    nba_df_checked['Is_FC_Transition_Possession'] = (
        is_transition_fga
        .astype(int)
        .groupby(poss_groupers, sort=False)
        .transform('max')
        .astype(int)
    )
    nba_df_checked['Has_FC_FGA'] = (
        first_chance_fga
        .astype(int)
        .groupby(poss_groupers, sort=False)
        .transform('max')
        .astype(int)
    )
    if first_chance_fga.any():
        fc_efg_avg_pts = float(nba_df_checked.loc[first_chance_fga, 'ActualNet'].mean())
    elif is_fga.any():
        fc_efg_avg_pts = float(nba_df_checked.loc[is_fga, 'ActualNet'].mean())
    else:
        fc_efg_avg_pts = 1.09
    fc_rim_avg_pts = (
        float(nba_df_checked.loc[first_chance_fga & is_rim_fga, 'ActualNet'].mean())
        if (first_chance_fga & is_rim_fga).any()
        else fc_efg_avg_pts
    )
    fc_mid_avg_pts = (
        float(nba_df_checked.loc[first_chance_fga & is_mid_fga, 'ActualNet'].mean())
        if (first_chance_fga & is_mid_fga).any()
        else fc_efg_avg_pts
    )
    fc_three_avg_pts = (
        float(nba_df_checked.loc[first_chance_fga & is_three_fga, 'ActualNet'].mean())
        if (first_chance_fga & is_three_fga).any()
        else fc_efg_avg_pts
    )

    desc_all = (
        nba_df_checked['home_description'].fillna('').astype(str) + ' ' +
        nba_df_checked['visitor_description'].fillna('').astype(str)
    ).str.strip()
    is_ft_event = (
        series_contains(nba_df_checked['home_description'], "Free Throw", case=False) |
        series_contains(nba_df_checked['visitor_description'], "Free Throw", case=False) |
        (nba_df_checked['event_type'] == "FreeThrow")
    )
    is_tech_ft = (
        series_contains(nba_df_checked['home_description'], "Technical", case=False) |
        series_contains(nba_df_checked['visitor_description'], "Technical", case=False)
    ) & is_ft_event
    is_1of1 = series_contains(desc_all, r'\b1 of 1\b', regex=True) & is_ft_event
    is_3shot = series_contains(desc_all, r'\b[123] of 3\b', regex=True) & is_ft_event
    is_2shot = series_contains(desc_all, r'\b[12] of 2\b', regex=True) & is_ft_event
    is_and1 = is_1of1 & ~is_tech_ft
    fc_ft_sq_baseline = np.where(
        is_and1 | is_tech_ft,
        0.0,
        np.where(is_3shot, fc_efg_avg_pts / 3.0,
                 np.where(is_2shot, fc_efg_avg_pts / 2.0, 0.0))
    )
    nba_df_checked['FC_FT_Action_Score'] = np.where(
        is_ft_event,
        nba_df_checked['ExpFT'] - fc_ft_sq_baseline,
        0.0,
    )
    first_chance_ft = is_ft_event & ~nba_df_checked['after_miss'].fillna(False).astype(bool)
    prev_first_chance_ft = (
        first_chance_ft
        .groupby(poss_groupers, sort=False)
        .shift(1, fill_value=False)
    )
    ft_trip_start = first_chance_ft & ~prev_first_chance_ft
    ft_trip_id = (
        ft_trip_start
        .astype(int)
        .groupby(poss_groupers, sort=False)
        .cumsum()
    )
    ft_trip_value = (
        pd.to_numeric(nba_df_checked['FC_FT_Action_Score'], errors='coerce')
        .fillna(0.0)
        .where(first_chance_ft, 0.0)
        .groupby(poss_groupers + [ft_trip_id], sort=False)
        .transform('sum')
    )
    avg_ft_trip_value = float(ft_trip_value[ft_trip_start].mean()) if ft_trip_start.any() else 0.0
    nba_df_checked['FC_FT_FREQ_Action_Score'] = np.where(
        ft_trip_start,
        avg_ft_trip_value,
        0.0,
    )
    nba_df_checked['FC_FT_SEVERITY_Action_Score'] = np.where(
        ft_trip_start,
        ft_trip_value - avg_ft_trip_value,
        0.0,
    )
    ft_trip_action_gap = float(abs(
        (
            nba_df_checked['FC_FT_FREQ_Action_Score'] +
            nba_df_checked['FC_FT_SEVERITY_Action_Score'] -
            nba_df_checked['FC_FT_Action_Score'].where(first_chance_ft, 0.0)
        ).sum()
    ))
    print(
        "      Calculated first-chance FT frequency/severity components "
        f"(trips={int(ft_trip_start.sum())}, avg_trip_value={avg_ft_trip_value:.4f}, "
        f"sum gap={ft_trip_action_gap:.6f})"
    )
    nba_df_checked['FC_FT_SQ_BASELINE_Action_Score'] = fc_ft_sq_baseline
    if 'FC_SQ_Action_Score' in nba_df_checked.columns:
        nba_df_checked['FC_SQ_Action_Score'] = np.where(
            is_ft_event,
            fc_ft_sq_baseline,
            nba_df_checked['FC_SQ_Action_Score'],
        )

    nba_df_checked['FC_EFG_BASELINE_Action_Score'] = np.where(
        is_fga,
        fc_efg_avg_pts,
        pd.to_numeric(
            nba_df_checked.get('FC_FT_SQ_BASELINE_Action_Score', 0.0),
            errors='coerce',
        ).fillna(0.0),
    )
    nba_df_checked['FC_RIM_EFG_AA_Action_Score'] = np.where(
        is_rim_fga,
        nba_df_checked['ActualNet'] - fc_efg_avg_pts,
        0.0,
    )
    nba_df_checked['FC_RIM_EFG_FREQ_Action_Score'] = np.where(
        is_rim_fga,
        fc_rim_avg_pts - fc_efg_avg_pts,
        0.0,
    )
    nba_df_checked['FC_RIM_EFG_FG_Action_Score'] = np.where(
        is_rim_fga,
        nba_df_checked['ActualNet'] - fc_rim_avg_pts,
        0.0,
    )
    nba_df_checked['FC_MID_EFG_AA_Action_Score'] = np.where(
        is_mid_fga,
        nba_df_checked['ActualNet'] - fc_efg_avg_pts,
        0.0,
    )
    nba_df_checked['FC_MID_EFG_FREQ_Action_Score'] = np.where(
        is_mid_fga,
        fc_mid_avg_pts - fc_efg_avg_pts,
        0.0,
    )
    nba_df_checked['FC_MID_EFG_FG_Action_Score'] = np.where(
        is_mid_fga,
        nba_df_checked['ActualNet'] - fc_mid_avg_pts,
        0.0,
    )
    nba_df_checked['FC_THREE_EFG_AA_Action_Score'] = np.where(
        is_three_fga,
        nba_df_checked['ActualNet'] - fc_efg_avg_pts,
        0.0,
    )
    nba_df_checked['FC_THREE_EFG_FREQ_Action_Score'] = np.where(
        is_three_fga,
        fc_three_avg_pts - fc_efg_avg_pts,
        0.0,
    )
    nba_df_checked['FC_THREE_EFG_FG_Action_Score'] = np.where(
        is_three_fga,
        nba_df_checked['ActualNet'] - fc_three_avg_pts,
        0.0,
    )
    nba_df_checked['FC_EFG_VALUE_Action_Score'] = (
        nba_df_checked['FC_RIM_EFG_AA_Action_Score'] +
        nba_df_checked['FC_MID_EFG_AA_Action_Score'] +
        nba_df_checked['FC_THREE_EFG_AA_Action_Score']
    )
    efg_zone_action_sum = (
        nba_df_checked['FC_EFG_BASELINE_Action_Score'] +
        nba_df_checked['FC_RIM_EFG_AA_Action_Score'] +
        nba_df_checked['FC_MID_EFG_AA_Action_Score'] +
        nba_df_checked['FC_THREE_EFG_AA_Action_Score']
    )
    efg_action_target = np.where(
        is_fga,
        nba_df_checked['ActualNet'],
        pd.to_numeric(
            nba_df_checked.get('FC_FT_SQ_BASELINE_Action_Score', 0.0),
            errors='coerce',
        ).fillna(0.0),
    )
    efg_zone_action_gap = float(np.nanmax(np.abs(efg_zone_action_sum - efg_action_target)))
    print(
        "      Calculated first-chance EFG zone baseline/above-average components "
        f"(fc_efg_avg_pts={fc_efg_avg_pts:.4f}, action gap max={efg_zone_action_gap:.6f})"
    )
    efg_freq_fg_action_sum = (
        nba_df_checked['FC_RIM_EFG_FREQ_Action_Score'] +
        nba_df_checked['FC_RIM_EFG_FG_Action_Score'] +
        nba_df_checked['FC_MID_EFG_FREQ_Action_Score'] +
        nba_df_checked['FC_MID_EFG_FG_Action_Score'] +
        nba_df_checked['FC_THREE_EFG_FREQ_Action_Score'] +
        nba_df_checked['FC_THREE_EFG_FG_Action_Score']
    )
    efg_value_action_gap = float(np.nanmax(
        np.abs(
            efg_freq_fg_action_sum -
            nba_df_checked['FC_EFG_VALUE_Action_Score']
        )
    ))
    print(
        "      Calculated first-chance EFG frequency/FG components "
        f"(rim={fc_rim_avg_pts:.4f}, mid={fc_mid_avg_pts:.4f}, "
        f"three={fc_three_avg_pts:.4f}, value gap max={efg_value_action_gap:.6f})"
    )

    # Public DECOMP shot tree. Frequency is valued against the same midrange
    # counterfactual on every FGA; the counterfactual itself is carried by
    # DECOMP_MID_FG. This removes the old visible baseline while preserving an
    # exact five-child identity to first-chance EFG on every source row.
    ft_sq_baseline_values = pd.to_numeric(
        nba_df_checked.get('FC_FT_SQ_BASELINE_Action_Score', 0.0),
        errors='coerce',
    ).fillna(0.0)
    decomp_actions = build_decomp_shot_action_components(
        actual_net=nba_df_checked['ActualNet'],
        is_fga=is_fga,
        is_rim_fga=is_rim_fga,
        is_mid_fga=is_mid_fga,
        is_three_fga=is_three_fga,
        ft_sq_baseline=ft_sq_baseline_values,
        rim_avg_pts=fc_rim_avg_pts,
        mid_avg_pts=fc_mid_avg_pts,
        three_avg_pts=fc_three_avg_pts,
    )
    for col in decomp_actions.columns:
        nba_df_checked[col] = decomp_actions[col]
    decomp_action_gap = float(np.nanmax(np.abs(
        nba_df_checked['FC_DECOMP_EFG_Action_Score'] - efg_action_target
    )))
    if decomp_action_gap > 1e-9:
        raise ValueError(
            "First-chance DECOMP shot identity failed at the action level: "
            f"max gap={decomp_action_gap:.12f}"
        )
    print(
        "      Calculated public five-factor DECOMP shot tree "
        f"(mid counterfactual={fc_mid_avg_pts:.4f}, action gap max={decomp_action_gap:.6f})"
    )

    nba_df_checked['First_Chance_Net'] = np.where(
        nba_df_checked['after_miss'],
        0.0,
        nba_df_checked['ActualNet']
    )
    nba_df_checked['Second_Chance_Net'] = np.where(
        nba_df_checked['after_miss'],
        nba_df_checked['ActualNet'],
        0.0
    )

    component_pairs = [
        ('FC_SQ_Action_Score', 'First_Chance_SQ_Net'),
        ('FC_MAKE_Action_Score', 'First_Chance_MAKE_Net'),
        ('FC_FT_Action_Score', 'First_Chance_FT_Net'),
        ('FC_FT_FREQ_Action_Score', 'First_Chance_FT_FREQ_Net'),
        ('FC_FT_SEVERITY_Action_Score', 'First_Chance_FT_SEVERITY_Net'),
        ('FC_EFG_BASELINE_Action_Score', 'First_Chance_EFG_BASELINE_Net'),
        ('FC_EFG_VALUE_Action_Score', 'First_Chance_EFG_VALUE_Net'),
        ('FC_RIM_EFG_AA_Action_Score', 'First_Chance_RIM_EFG_AA_Net'),
        ('FC_MID_EFG_AA_Action_Score', 'First_Chance_MID_EFG_AA_Net'),
        ('FC_THREE_EFG_AA_Action_Score', 'First_Chance_THREE_EFG_AA_Net'),
        ('FC_RIM_EFG_FREQ_Action_Score', 'First_Chance_RIM_EFG_FREQ_Net'),
        ('FC_RIM_EFG_FG_Action_Score', 'First_Chance_RIM_EFG_FG_Net'),
        ('FC_MID_EFG_FREQ_Action_Score', 'First_Chance_MID_EFG_FREQ_Net'),
        ('FC_MID_EFG_FG_Action_Score', 'First_Chance_MID_EFG_FG_Net'),
        ('FC_THREE_EFG_FREQ_Action_Score', 'First_Chance_THREE_EFG_FREQ_Net'),
        ('FC_THREE_EFG_FG_Action_Score', 'First_Chance_THREE_EFG_FG_Net'),
        ('FC_DECOMP_EFG_Action_Score', 'First_Chance_DECOMP_EFG_Net'),
        ('FC_DECOMP_RIM_FREQ_Action_Score', 'First_Chance_DECOMP_RIM_FREQ_Net'),
        ('FC_DECOMP_RIM_FG_Action_Score', 'First_Chance_DECOMP_RIM_FG_Net'),
        ('FC_DECOMP_MID_FG_Action_Score', 'First_Chance_DECOMP_MID_FG_Net'),
        ('FC_DECOMP_THREE_FREQ_Action_Score', 'First_Chance_DECOMP_THREE_FREQ_Net'),
        ('FC_DECOMP_THREE_FG_Action_Score', 'First_Chance_DECOMP_THREE_FG_Net'),
        ('FC_DECOMP_MID_VALUE_Action_Score', 'First_Chance_DECOMP_MID_VALUE_Net'),
    ]
    for source_col, target_col in component_pairs:
        if source_col in nba_df_checked.columns:
            nba_df_checked[target_col] = np.where(
                nba_df_checked['after_miss'],
                0.0,
                nba_df_checked[source_col],
            )

    nba_df_checked['first_chance_completion_event'] = (
        ~nba_df_checked['after_miss'].fillna(False).astype(bool) &
        (is_fga | is_ft_event)
    )
    completion_seen = (
        nba_df_checked['first_chance_completion_event']
        .astype(int)
        .groupby(poss_groupers, sort=False)
        .cumsum()
    )
    nba_df_checked['had_first_chance_completion_before'] = (
        completion_seen
        .groupby(poss_groupers, sort=False)
        .shift(1, fill_value=0)
        .gt(0)
    )

    if group_cols:
        nba_df_checked['Actual_Total'] = nba_df_checked.groupby(group_cols)['ActualNet'].cumsum()
        nba_df_checked['First_Chance_Total'] = nba_df_checked.groupby(group_cols)['First_Chance_Net'].cumsum()
        nba_df_checked['Second_Chance_Total'] = nba_df_checked.groupby(group_cols)['Second_Chance_Net'].cumsum()
        for total_col, net_col in [
            ('First_Chance_SQ_Total', 'First_Chance_SQ_Net'),
            ('First_Chance_MAKE_Total', 'First_Chance_MAKE_Net'),
            ('First_Chance_FT_Total', 'First_Chance_FT_Net'),
            ('First_Chance_FT_FREQ_Total', 'First_Chance_FT_FREQ_Net'),
            ('First_Chance_FT_SEVERITY_Total', 'First_Chance_FT_SEVERITY_Net'),
            ('First_Chance_EFG_BASELINE_Total', 'First_Chance_EFG_BASELINE_Net'),
            ('First_Chance_EFG_VALUE_Total', 'First_Chance_EFG_VALUE_Net'),
            ('First_Chance_RIM_EFG_AA_Total', 'First_Chance_RIM_EFG_AA_Net'),
            ('First_Chance_MID_EFG_AA_Total', 'First_Chance_MID_EFG_AA_Net'),
            ('First_Chance_THREE_EFG_AA_Total', 'First_Chance_THREE_EFG_AA_Net'),
            ('First_Chance_RIM_EFG_FREQ_Total', 'First_Chance_RIM_EFG_FREQ_Net'),
            ('First_Chance_RIM_EFG_FG_Total', 'First_Chance_RIM_EFG_FG_Net'),
            ('First_Chance_MID_EFG_FREQ_Total', 'First_Chance_MID_EFG_FREQ_Net'),
            ('First_Chance_MID_EFG_FG_Total', 'First_Chance_MID_EFG_FG_Net'),
            ('First_Chance_THREE_EFG_FREQ_Total', 'First_Chance_THREE_EFG_FREQ_Net'),
            ('First_Chance_THREE_EFG_FG_Total', 'First_Chance_THREE_EFG_FG_Net'),
            ('First_Chance_DECOMP_EFG_Total', 'First_Chance_DECOMP_EFG_Net'),
            ('First_Chance_DECOMP_RIM_FREQ_Total', 'First_Chance_DECOMP_RIM_FREQ_Net'),
            ('First_Chance_DECOMP_RIM_FG_Total', 'First_Chance_DECOMP_RIM_FG_Net'),
            ('First_Chance_DECOMP_MID_FG_Total', 'First_Chance_DECOMP_MID_FG_Net'),
            ('First_Chance_DECOMP_THREE_FREQ_Total', 'First_Chance_DECOMP_THREE_FREQ_Net'),
            ('First_Chance_DECOMP_THREE_FG_Total', 'First_Chance_DECOMP_THREE_FG_Net'),
            ('First_Chance_DECOMP_MID_VALUE_Total', 'First_Chance_DECOMP_MID_VALUE_Net'),
        ]:
            if net_col in nba_df_checked.columns:
                nba_df_checked[total_col] = nba_df_checked.groupby(group_cols)[net_col].cumsum()
    else:
        nba_df_checked['Actual_Total'] = nba_df_checked['ActualNet'].cumsum()
        nba_df_checked['First_Chance_Total'] = nba_df_checked['First_Chance_Net'].cumsum()
        nba_df_checked['Second_Chance_Total'] = nba_df_checked['Second_Chance_Net'].cumsum()
        for total_col, net_col in [
            ('First_Chance_SQ_Total', 'First_Chance_SQ_Net'),
            ('First_Chance_MAKE_Total', 'First_Chance_MAKE_Net'),
            ('First_Chance_FT_Total', 'First_Chance_FT_Net'),
            ('First_Chance_FT_FREQ_Total', 'First_Chance_FT_FREQ_Net'),
            ('First_Chance_FT_SEVERITY_Total', 'First_Chance_FT_SEVERITY_Net'),
            ('First_Chance_EFG_BASELINE_Total', 'First_Chance_EFG_BASELINE_Net'),
            ('First_Chance_EFG_VALUE_Total', 'First_Chance_EFG_VALUE_Net'),
            ('First_Chance_RIM_EFG_AA_Total', 'First_Chance_RIM_EFG_AA_Net'),
            ('First_Chance_MID_EFG_AA_Total', 'First_Chance_MID_EFG_AA_Net'),
            ('First_Chance_THREE_EFG_AA_Total', 'First_Chance_THREE_EFG_AA_Net'),
            ('First_Chance_RIM_EFG_FREQ_Total', 'First_Chance_RIM_EFG_FREQ_Net'),
            ('First_Chance_RIM_EFG_FG_Total', 'First_Chance_RIM_EFG_FG_Net'),
            ('First_Chance_MID_EFG_FREQ_Total', 'First_Chance_MID_EFG_FREQ_Net'),
            ('First_Chance_MID_EFG_FG_Total', 'First_Chance_MID_EFG_FG_Net'),
            ('First_Chance_THREE_EFG_FREQ_Total', 'First_Chance_THREE_EFG_FREQ_Net'),
            ('First_Chance_THREE_EFG_FG_Total', 'First_Chance_THREE_EFG_FG_Net'),
            ('First_Chance_DECOMP_EFG_Total', 'First_Chance_DECOMP_EFG_Net'),
            ('First_Chance_DECOMP_RIM_FREQ_Total', 'First_Chance_DECOMP_RIM_FREQ_Net'),
            ('First_Chance_DECOMP_RIM_FG_Total', 'First_Chance_DECOMP_RIM_FG_Net'),
            ('First_Chance_DECOMP_MID_FG_Total', 'First_Chance_DECOMP_MID_FG_Net'),
            ('First_Chance_DECOMP_THREE_FREQ_Total', 'First_Chance_DECOMP_THREE_FREQ_Net'),
            ('First_Chance_DECOMP_THREE_FG_Total', 'First_Chance_DECOMP_THREE_FG_Net'),
            ('First_Chance_DECOMP_MID_VALUE_Total', 'First_Chance_DECOMP_MID_VALUE_Net'),
        ]:
            if net_col in nba_df_checked.columns:
                nba_df_checked[total_col] = nba_df_checked[net_col].cumsum()

    nba_filt = nba_df_checked[nba_df_checked['End_of_Possession']].copy()
    print(f"      Filtered to {len(nba_filt)} {label} End of Possession rows")

    if nba_filt.empty:
        return None

    if group_cols:
        nba_filt['Actual_Diff'] = (
            nba_filt['Actual_Total'] -
            nba_filt.groupby(group_cols)['Actual_Total'].shift(fill_value=0)
        )
        nba_filt['First_Chance_Diff'] = (
            nba_filt['First_Chance_Total'] -
            nba_filt.groupby(group_cols)['First_Chance_Total'].shift(fill_value=0)
        )
        nba_filt['Second_Chance_Diff'] = (
            nba_filt['Second_Chance_Total'] -
            nba_filt.groupby(group_cols)['Second_Chance_Total'].shift(fill_value=0)
        )
        for diff_col, total_col in [
            ('FC_SQ_Diff', 'First_Chance_SQ_Total'),
            ('FC_MAKE_Diff', 'First_Chance_MAKE_Total'),
            ('FC_FT_Diff', 'First_Chance_FT_Total'),
            ('FC_FT_FREQ_Diff', 'First_Chance_FT_FREQ_Total'),
            ('FC_FT_SEVERITY_Diff', 'First_Chance_FT_SEVERITY_Total'),
            ('FC_EFG_BASELINE_Diff', 'First_Chance_EFG_BASELINE_Total'),
            ('FC_EFG_VALUE_Diff', 'First_Chance_EFG_VALUE_Total'),
            ('FC_RIM_EFG_AA_Diff', 'First_Chance_RIM_EFG_AA_Total'),
            ('FC_MID_EFG_AA_Diff', 'First_Chance_MID_EFG_AA_Total'),
            ('FC_THREE_EFG_AA_Diff', 'First_Chance_THREE_EFG_AA_Total'),
            ('FC_RIM_EFG_FREQ_Diff', 'First_Chance_RIM_EFG_FREQ_Total'),
            ('FC_RIM_EFG_FG_Diff', 'First_Chance_RIM_EFG_FG_Total'),
            ('FC_MID_EFG_FREQ_Diff', 'First_Chance_MID_EFG_FREQ_Total'),
            ('FC_MID_EFG_FG_Diff', 'First_Chance_MID_EFG_FG_Total'),
            ('FC_THREE_EFG_FREQ_Diff', 'First_Chance_THREE_EFG_FREQ_Total'),
            ('FC_THREE_EFG_FG_Diff', 'First_Chance_THREE_EFG_FG_Total'),
            ('FC_DECOMP_EFG_Diff', 'First_Chance_DECOMP_EFG_Total'),
            ('FC_DECOMP_RIM_FREQ_Diff', 'First_Chance_DECOMP_RIM_FREQ_Total'),
            ('FC_DECOMP_RIM_FG_Diff', 'First_Chance_DECOMP_RIM_FG_Total'),
            ('FC_DECOMP_MID_FG_Diff', 'First_Chance_DECOMP_MID_FG_Total'),
            ('FC_DECOMP_THREE_FREQ_Diff', 'First_Chance_DECOMP_THREE_FREQ_Total'),
            ('FC_DECOMP_THREE_FG_Diff', 'First_Chance_DECOMP_THREE_FG_Total'),
            ('FC_DECOMP_MID_VALUE_Diff', 'First_Chance_DECOMP_MID_VALUE_Total'),
        ]:
            if total_col in nba_filt.columns:
                nba_filt[diff_col] = (
                    nba_filt[total_col] -
                    nba_filt.groupby(group_cols)[total_col].shift(fill_value=0)
                )
    else:
        nba_filt['Actual_Diff'] = (
            nba_filt['Actual_Total'] -
            nba_filt['Actual_Total'].shift(fill_value=0)
        )
        nba_filt['First_Chance_Diff'] = (
            nba_filt['First_Chance_Total'] -
            nba_filt['First_Chance_Total'].shift(fill_value=0)
        )
        nba_filt['Second_Chance_Diff'] = (
            nba_filt['Second_Chance_Total'] -
            nba_filt['Second_Chance_Total'].shift(fill_value=0)
        )
        for diff_col, total_col in [
            ('FC_SQ_Diff', 'First_Chance_SQ_Total'),
            ('FC_MAKE_Diff', 'First_Chance_MAKE_Total'),
            ('FC_FT_Diff', 'First_Chance_FT_Total'),
            ('FC_FT_FREQ_Diff', 'First_Chance_FT_FREQ_Total'),
            ('FC_FT_SEVERITY_Diff', 'First_Chance_FT_SEVERITY_Total'),
            ('FC_EFG_BASELINE_Diff', 'First_Chance_EFG_BASELINE_Total'),
            ('FC_EFG_VALUE_Diff', 'First_Chance_EFG_VALUE_Total'),
            ('FC_RIM_EFG_AA_Diff', 'First_Chance_RIM_EFG_AA_Total'),
            ('FC_MID_EFG_AA_Diff', 'First_Chance_MID_EFG_AA_Total'),
            ('FC_THREE_EFG_AA_Diff', 'First_Chance_THREE_EFG_AA_Total'),
            ('FC_RIM_EFG_FREQ_Diff', 'First_Chance_RIM_EFG_FREQ_Total'),
            ('FC_RIM_EFG_FG_Diff', 'First_Chance_RIM_EFG_FG_Total'),
            ('FC_MID_EFG_FREQ_Diff', 'First_Chance_MID_EFG_FREQ_Total'),
            ('FC_MID_EFG_FG_Diff', 'First_Chance_MID_EFG_FG_Total'),
            ('FC_THREE_EFG_FREQ_Diff', 'First_Chance_THREE_EFG_FREQ_Total'),
            ('FC_THREE_EFG_FG_Diff', 'First_Chance_THREE_EFG_FG_Total'),
            ('FC_DECOMP_EFG_Diff', 'First_Chance_DECOMP_EFG_Total'),
            ('FC_DECOMP_RIM_FREQ_Diff', 'First_Chance_DECOMP_RIM_FREQ_Total'),
            ('FC_DECOMP_RIM_FG_Diff', 'First_Chance_DECOMP_RIM_FG_Total'),
            ('FC_DECOMP_MID_FG_Diff', 'First_Chance_DECOMP_MID_FG_Total'),
            ('FC_DECOMP_THREE_FREQ_Diff', 'First_Chance_DECOMP_THREE_FREQ_Total'),
            ('FC_DECOMP_THREE_FG_Diff', 'First_Chance_DECOMP_THREE_FG_Total'),
            ('FC_DECOMP_MID_VALUE_Diff', 'First_Chance_DECOMP_MID_VALUE_Total'),
        ]:
            if total_col in nba_filt.columns:
                nba_filt[diff_col] = (
                    nba_filt[total_col] -
                    nba_filt[total_col].shift(fill_value=0)
                )

    desc_all = (
        nba_filt['home_description'].fillna('').astype(str) + ' ' +
        nba_filt['visitor_description'].fillna('').astype(str)
    ).str.strip()

    # First-chance turnover flags exclude possessions whose turnover happened
    # after a first-chance FGA/FT completion event; that prior scoring stays
    # in first chance, but the later terminal turnover is not the FC TOV bucket.
    nba_filt['Is_Turnover'] = (
        (nba_filt['event_type'] == 'Turnover') &
        ~nba_filt['after_miss'].fillna(False).astype(bool) &
        ~nba_filt['had_first_chance_completion_before'].fillna(False).astype(bool)
    ).astype(int)
    nba_filt['Is_BadPass_TOV'] = (
        nba_filt['Is_Turnover'].astype(bool) &
        desc_all.str.contains('Bad Pass', case=False, na=False)
    ).astype(int)

    identity_gap = (
        nba_filt['First_Chance_Diff'] +
        nba_filt['Second_Chance_Diff'] -
        nba_filt['Actual_Diff']
    ).abs().max()
    print(f"      Clean split built for {label}; identity gap max={identity_gap:.6f}")

    if {'FC_SQ_Diff', 'FC_MAKE_Diff', 'FC_FT_Diff'}.issubset(nba_filt.columns):
        non_turnover_mask = nba_filt['Is_Turnover'] == 0
        fc_ts_gap = (
            nba_filt.loc[non_turnover_mask, 'FC_SQ_Diff'] +
            nba_filt.loc[non_turnover_mask, 'FC_MAKE_Diff'] +
            nba_filt.loc[non_turnover_mask, 'FC_FT_Diff'] -
            nba_filt.loc[non_turnover_mask, 'First_Chance_Diff']
        ).abs().max()
        print(f"      First-chance TS component gap max={fc_ts_gap:.6f}")
    if 'FC_FT_Diff' in nba_filt.columns:
        if {'FC_FT_FREQ_Diff', 'FC_FT_SEVERITY_Diff'}.issubset(nba_filt.columns):
            fc_ft_freq_severity_gap = (
                nba_filt['FC_FT_Diff'] -
                (
                    pd.to_numeric(nba_filt['FC_FT_FREQ_Diff'], errors='coerce').fillna(0.0) +
                    pd.to_numeric(nba_filt['FC_FT_SEVERITY_Diff'], errors='coerce').fillna(0.0)
                )
            ).abs().max()
            print(f"      First-chance FT frequency/severity gap max={fc_ft_freq_severity_gap:.6f}")
        nba_filt['FC_EFG_Diff'] = (
            pd.to_numeric(nba_filt['First_Chance_Diff'], errors='coerce').fillna(0.0) -
            pd.to_numeric(nba_filt['FC_FT_Diff'], errors='coerce').fillna(0.0)
        )
        if {'FC_SQ_Diff', 'FC_MAKE_Diff'}.issubset(nba_filt.columns):
            fc_efg_gap = (
                nba_filt['FC_EFG_Diff'] -
                (
                    pd.to_numeric(nba_filt['FC_SQ_Diff'], errors='coerce').fillna(0.0) +
                    pd.to_numeric(nba_filt['FC_MAKE_Diff'], errors='coerce').fillna(0.0)
                )
            ).abs().max()
            print(f"      First-chance EFG component gap max={fc_efg_gap:.6f}")
        efg_zone_cols = [
            'FC_EFG_BASELINE_Diff',
            'FC_RIM_EFG_AA_Diff',
            'FC_MID_EFG_AA_Diff',
            'FC_THREE_EFG_AA_Diff',
        ]
        if set(efg_zone_cols).issubset(nba_filt.columns):
            fc_efg_zone_gap = (
                nba_filt['FC_EFG_Diff'] -
                sum(
                    pd.to_numeric(nba_filt[col], errors='coerce').fillna(0.0)
                    for col in efg_zone_cols
                )
            ).abs().max()
            print(f"      First-chance EFG zone decomposition gap max={fc_efg_zone_gap:.6f}")
        efg_value_cols = [
            'FC_RIM_EFG_AA_Diff',
            'FC_MID_EFG_AA_Diff',
            'FC_THREE_EFG_AA_Diff',
        ]
        if {'FC_EFG_VALUE_Diff', *efg_value_cols}.issubset(nba_filt.columns):
            fc_efg_value_gap = (
                nba_filt['FC_EFG_VALUE_Diff'] -
                sum(
                    pd.to_numeric(nba_filt[col], errors='coerce').fillna(0.0)
                    for col in efg_value_cols
                )
            ).abs().max()
            print(f"      First-chance EFG value-only zone gap max={fc_efg_value_gap:.6f}")
        efg_freq_fg_cols = [
            'FC_RIM_EFG_FREQ_Diff',
            'FC_RIM_EFG_FG_Diff',
            'FC_MID_EFG_FREQ_Diff',
            'FC_MID_EFG_FG_Diff',
            'FC_THREE_EFG_FREQ_Diff',
            'FC_THREE_EFG_FG_Diff',
        ]
        if {'FC_EFG_VALUE_Diff', *efg_freq_fg_cols}.issubset(nba_filt.columns):
            fc_efg_freq_fg_gap = (
                nba_filt['FC_EFG_VALUE_Diff'] -
                sum(
                    pd.to_numeric(nba_filt[col], errors='coerce').fillna(0.0)
                    for col in efg_freq_fg_cols
                )
            ).abs().max()
            freq_balance_gap = (
                nba_filt['FC_RIM_EFG_FREQ_Diff'] +
                nba_filt['FC_MID_EFG_FREQ_Diff'] +
                nba_filt['FC_THREE_EFG_FREQ_Diff']
            ).sum()
            print(
                "      First-chance EFG frequency/FG decomposition gap "
                f"max={fc_efg_freq_fg_gap:.6f}; frequency total={freq_balance_gap:.6f}"
            )
        decomp_cols = [
            'FC_DECOMP_RIM_FREQ_Diff',
            'FC_DECOMP_RIM_FG_Diff',
            'FC_DECOMP_MID_FG_Diff',
            'FC_DECOMP_THREE_FREQ_Diff',
            'FC_DECOMP_THREE_FG_Diff',
        ]
        if {'FC_EFG_Diff', 'FC_DECOMP_EFG_Diff', *decomp_cols}.issubset(nba_filt.columns):
            decomp_children = sum(
                pd.to_numeric(nba_filt[col], errors='coerce').fillna(0.0)
                for col in decomp_cols
            )
            parent_gap = float((nba_filt['FC_DECOMP_EFG_Diff'] - decomp_children).abs().max())
            efg_gap = float((nba_filt['FC_EFG_Diff'] - nba_filt['FC_DECOMP_EFG_Diff']).abs().max())
            if max(parent_gap, efg_gap) > 1e-9:
                raise ValueError(
                    "First-chance DECOMP possession identity failed: "
                    f"child gap={parent_gap:.12f}, EFG gap={efg_gap:.12f}"
                )
            print(
                "      Public DECOMP possession identity "
                f"child gap max={parent_gap:.6f}; EFG gap max={efg_gap:.6f}"
            )
    return nba_filt
