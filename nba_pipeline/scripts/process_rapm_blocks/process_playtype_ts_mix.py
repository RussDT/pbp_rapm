"""
PLAYTYPE_TS_MIX processor.

Processes a shot-based playtype-mix expected-points metric from ShotQuality
descriptor bundles. The output captures how lineups shift shot mix toward
higher- or lower-value descriptor families, independent of player IDs.
"""

import time
import numpy as np

from .common import (
    _base_processing,
    _finalize_df,
    build_shot_flags_df,
    map_players_from_team_on_offense,
    series_contains,
    case_when,
    normalize_season_end_year,
)


PLAYTYPE_SHRINK_K = 100.0


def process_playtype_ts_mix_py(file_path, year, season_str):
    """
    Build a descriptor-bundle expected-points surface and output Playtype_Exp_PTS
    for each field-goal attempt.

    Only available for seasons with ShotQuality descriptors in the raw parquet
    layer (currently 2023-24 onward / end years 24-26).
    """
    season_end_year = normalize_season_end_year(year)
    if season_end_year < 2024:
        print(f"  Skipping PLAYTYPE_TS_MIX - only available for years 24-26 (season end year {season_end_year} < 2024)")
        return None

    print(f"  Starting PLAYTYPE_TS_MIX Processing for {season_str}...")
    nba_df = _base_processing(file_path)
    if nba_df is None:
        return None
    start_time = time.time()

    if 'sq_descriptor_bundle' not in nba_df.columns:
        print("      Warning: 'sq_descriptor_bundle' column not found. Re-run 01b_enrich_pbp_shotquality.py first.")
        return None

    shot_flags = build_shot_flags_df(nba_df)
    initial_rows = len(nba_df)
    nba_df = nba_df[shot_flags['is_fga']].copy()
    shot_flags = shot_flags.loc[nba_df.index]
    print(f"      Filtered to FGA events: {len(nba_df)} rows (from {initial_rows})")

    if nba_df.empty:
        print("      Warning: No FGA events found. Returning None.")
        return None

    nba_df['Actual_Points'] = shot_flags['shot_points'].astype(float)
    bundle = nba_df['sq_descriptor_bundle'].fillna('').astype(str).str.strip()
    bundle = bundle.where(bundle != '', 'NO_TAGS')
    shot_value_tag = np.where(shot_flags['is_3pt'], 'SHOT_VALUE_3', 'SHOT_VALUE_2')
    nba_df['Playtype_Model_Key'] = bundle + '|' + shot_value_tag

    global_mean = float(nba_df['Actual_Points'].mean())
    bundle_stats = nba_df.groupby('Playtype_Model_Key', dropna=False)['Actual_Points'].agg(['mean', 'count'])
    bundle_stats['Playtype_Exp_PTS'] = (
        bundle_stats['count'] * bundle_stats['mean'] + PLAYTYPE_SHRINK_K * global_mean
    ) / (bundle_stats['count'] + PLAYTYPE_SHRINK_K)

    nba_df = nba_df.join(bundle_stats['Playtype_Exp_PTS'], on='Playtype_Model_Key')
    nba_df['Playtype_Exp_PTS'] = nba_df['Playtype_Exp_PTS'].fillna(global_mean)
    print(
        "      Built descriptor bundle EV surface "
        f"({len(bundle_stats)} bundles, global mean={global_mean:.3f}, "
        f"season mean fitted={nba_df['Playtype_Exp_PTS'].mean():.3f})"
    )

    nba_df['End_of_Possession'] = True
    nba_df['TeamOnOffense'] = case_when(
        series_contains(nba_df['home_description'], "PTS|MISS", case=False, regex=True), "Home",
        series_contains(nba_df['visitor_description'], "PTS|MISS", case=False, regex=True), "Away",
        ""
    )
    print("      Calculated TeamOnOffense for PLAYTYPE_TS_MIX")

    nba_df = map_players_from_team_on_offense(nba_df)
    print("      Mapped O/D Players")

    nba_filt = nba_df[nba_df['End_of_Possession']].copy()
    print(f"      Final PLAYTYPE_TS_MIX rows: {len(nba_filt)}")

    nba_playtype_output = _finalize_df(nba_filt, 'Playtype_Exp_PTS', year)

    end_time = time.time()
    print(f"  Finished PLAYTYPE_TS_MIX Processing. Time: {end_time - start_time:.2f} seconds.")
    return nba_playtype_output
