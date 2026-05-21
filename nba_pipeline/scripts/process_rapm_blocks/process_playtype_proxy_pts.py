"""
PLAYTYPE_PROXY_PTS processor.

Processes a shot-based proxy playtype value metric using a small Synergy-like
taxonomy and actual points per FGA by season.
"""

import time
import numpy as np
import pandas as pd

from .common import (
    _base_processing,
    _finalize_df,
    build_shot_flags_df,
    map_players_from_team_on_offense,
    series_contains,
    case_when,
    normalize_season_end_year,
)


PLAYTYPE_PROXY_SHRINK_K = 100.0
PLAYTYPE_PROXY_CATEGORIES = [
    "TRANSITION",
    "CUT",
    "SPOT_UP",
    "UNASSISTED_3",
    "UNASSISTED_2",
    "OTHER_ASSISTED_2",
]
SELF_CREATED_2_TAGS = [
    "sq_desc_off_drive",
    "sq_desc_pull_up",
    "sq_desc_post_up",
    "sq_desc_turnaround",
    "sq_desc_hook",
    "sq_desc_floater",
    "sq_desc_step_back",
]


def _bool_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    series = df[col]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(bool)


def process_playtype_proxy_pts_py(file_path, year, season_str):
    """
    Build a Synergy-proxy playtype mix value surface from actual points per FGA.

    For each season, every FGA is assigned to one mutually exclusive proxy
    bucket. Each shot then receives the shrunk bucket average actual points
    minus the seasonwide FGA average actual points.
    """
    season_end_year = normalize_season_end_year(year)
    if season_end_year < 2024:
        print(
            "  Skipping PLAYTYPE_PROXY_PTS - only available for years 24-26 "
            f"(season end year {season_end_year} < 2024)"
        )
        return None

    print(f"  Starting PLAYTYPE_PROXY_PTS Processing for {season_str}...")
    nba_df = _base_processing(file_path)
    if nba_df is None:
        return None
    start_time = time.time()

    if "sq_descriptor_bundle" not in nba_df.columns:
        print(
            "      Warning: 'sq_descriptor_bundle' column not found. "
            "Re-run 01b_enrich_pbp_shotquality.py first."
        )
        return None

    shot_flags = build_shot_flags_df(nba_df)
    initial_rows = len(nba_df)
    nba_df = nba_df[shot_flags["is_fga"]].copy()
    shot_flags = shot_flags.loc[nba_df.index]
    print(f"      Filtered to FGA events: {len(nba_df)} rows (from {initial_rows})")

    if nba_df.empty:
        print("      Warning: No FGA events found. Returning None.")
        return None

    nba_df["Actual_Points"] = shot_flags["shot_points"].astype(float)

    is_transition = _bool_col(nba_df, "sq_desc_transition")
    is_cut = _bool_col(nba_df, "sq_desc_off_cut") | _bool_col(nba_df, "sq_desc_alley_oop")
    is_spot_up = _bool_col(nba_df, "sq_desc_catch_and_shoot")

    self_created_2 = pd.Series(False, index=nba_df.index)
    for tag_col in SELF_CREATED_2_TAGS:
        self_created_2 = self_created_2 | _bool_col(nba_df, tag_col)

    is_three = shot_flags["is_3pt"]
    not_classified = ~(is_transition | is_cut | is_spot_up)

    conditions = [
        is_transition,
        (~is_transition) & is_cut,
        (~is_transition) & (~is_cut) & is_spot_up,
        not_classified & is_three,
        not_classified & (~is_three) & self_created_2,
    ]
    choices = [
        "TRANSITION",
        "CUT",
        "SPOT_UP",
        "UNASSISTED_3",
        "UNASSISTED_2",
    ]
    nba_df["Playtype_Proxy_Category"] = np.select(
        conditions,
        choices,
        default="OTHER_ASSISTED_2",
    )

    global_mean = float(nba_df["Actual_Points"].mean())
    category_stats = nba_df.groupby("Playtype_Proxy_Category", dropna=False)["Actual_Points"].agg(["mean", "count"])
    category_stats = category_stats.reindex(PLAYTYPE_PROXY_CATEGORIES).fillna({"mean": global_mean, "count": 0})
    category_stats["Playtype_Proxy_Mean_PTS"] = (
        category_stats["count"] * category_stats["mean"] + PLAYTYPE_PROXY_SHRINK_K * global_mean
    ) / (category_stats["count"] + PLAYTYPE_PROXY_SHRINK_K)
    category_stats["Playtype_Proxy_PTS"] = category_stats["Playtype_Proxy_Mean_PTS"] - global_mean

    nba_df = nba_df.join(category_stats[["Playtype_Proxy_PTS"]], on="Playtype_Proxy_Category")
    print(
        "      Built proxy playtype actual-points surface "
        f"(global mean={global_mean:.3f}, shrink_k={PLAYTYPE_PROXY_SHRINK_K:.0f})"
    )
    for category, row in category_stats.iterrows():
        print(
            "        "
            f"{category:<17} count={int(row['count']):>7,} "
            f"mean={row['mean']:.3f} "
            f"delta={row['Playtype_Proxy_PTS']:.3f}"
        )

    nba_df["End_of_Possession"] = True
    nba_df["TeamOnOffense"] = case_when(
        series_contains(nba_df["home_description"], "PTS|MISS", case=False, regex=True), "Home",
        series_contains(nba_df["visitor_description"], "PTS|MISS", case=False, regex=True), "Away",
        ""
    )
    print("      Calculated TeamOnOffense for PLAYTYPE_PROXY_PTS")

    nba_df = map_players_from_team_on_offense(nba_df)
    print("      Mapped O/D Players")

    nba_filt = nba_df[nba_df["End_of_Possession"]].copy()
    print(f"      Final PLAYTYPE_PROXY_PTS rows: {len(nba_filt)}")

    nba_proxy_output = _finalize_df(nba_filt, "Playtype_Proxy_PTS", year)

    end_time = time.time()
    print(f"  Finished PLAYTYPE_PROXY_PTS Processing. Time: {end_time - start_time:.2f} seconds.")
    return nba_proxy_output
