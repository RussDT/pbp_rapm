"""STOP possession processor.

One row is emitted per standard offensive possession.  The binary target is
one only when the possession is guaranteed to finish with zero points because
the offense turned the ball over or the defense recovered a blocked field-goal
attempt.
"""

import time

import numpy as np
import pandas as pd

from .common import _base_processing, _finalize_df, prepare_standard_possession_df, series_contains
from .process_block_recovery import _resolve_recovery_rows


def _blocked_fga_masks(nba_df):
    home_desc = nba_df["home_description"].fillna("").astype(str)
    visitor_desc = nba_df["visitor_description"].fillna("").astype(str)
    all_desc = home_desc + " " + visitor_desc

    home_blocked_fga = (
        nba_df["event_type"].eq("MISS")
        & series_contains(home_desc, r"\bMISS\b", case=False, regex=True)
        & series_contains(visitor_desc, r"\b(?:BLOCK|BLK)\b", case=False, regex=True)
    )
    away_blocked_fga = (
        nba_df["event_type"].eq("MISS")
        & series_contains(visitor_desc, r"\bMISS\b", case=False, regex=True)
        & series_contains(home_desc, r"\b(?:BLOCK|BLK)\b", case=False, regex=True)
    )
    any_blocked_fga = (
        nba_df["event_type"].eq("MISS")
        & series_contains(all_desc, r"\b(?:BLOCK|BLK)\b", case=False, regex=True)
        & (home_blocked_fga | away_blocked_fga)
    )
    return home_blocked_fga, away_blocked_fga, any_blocked_fga


def build_stop_flags(nba_df_checked):
    """Attach possession-level turnover, recovered-block, and STOP flags."""
    nba_df_checked = nba_df_checked.copy()
    home_blocked, away_blocked, any_blocked = _blocked_fga_masks(nba_df_checked)
    nba_df_checked["Is_Blocked_FGA"] = any_blocked.astype(int)

    # Standard possession preparation normally fills TeamOnOffense on shot
    # rows.  Set it explicitly here as a guard for synthetic/legacy inputs.
    nba_df_checked.loc[home_blocked, "TeamOnOffense"] = "Home"
    nba_df_checked.loc[away_blocked, "TeamOnOffense"] = "Away"

    recovery_side, resolution, _, _ = _resolve_recovery_rows(nba_df_checked)
    # On a turnover resolution the event side is the team losing the ball, so
    # possession is recovered by the opposite side.
    turnover_resolution = resolution.eq("turnover")
    turnover_recovery_side = recovery_side.map({"Home": "Away", "Away": "Home"})
    recovery_side = recovery_side.where(~turnover_resolution, turnover_recovery_side)
    defense_side = nba_df_checked["TeamOnOffense"].map(
        {"Home": "Away", "Away": "Home"}
    ).fillna("")
    nba_df_checked["_recovered_block_event"] = (
        any_blocked & recovery_side.eq(defense_side)
    ).astype(int)

    descriptions = (
        nba_df_checked["home_description"].fillna("").astype(str)
        + " "
        + nba_df_checked["visitor_description"].fillna("").astype(str)
    )
    nba_df_checked["_turnover_event"] = (
        nba_df_checked["event_type"].eq("Turnover")
        | (
            nba_df_checked["event_type"].eq("Foul")
            & series_contains(descriptions, r"\bOffens(?:ive)?\b", case=False, regex=True)
        )
    ).astype(int)

    group_cols = ["poss_group"]
    if "game_id" in nba_df_checked.columns:
        group_cols.insert(0, "game_id")
    grouped = nba_df_checked.groupby(group_cols, sort=False)
    nba_df_checked["Is_Turnover_Stop"] = grouped["_turnover_event"].transform("max").astype(int)
    nba_df_checked["Is_Recovered_Block_Stop"] = grouped[
        "_recovered_block_event"
    ].transform("max").astype(int)
    nba_df_checked["Is_Stop"] = np.maximum(
        nba_df_checked["Is_Turnover_Stop"],
        nba_df_checked["Is_Recovered_Block_Stop"],
    ).astype(int)
    nba_df_checked["Off_Diff"] = -nba_df_checked["Is_Stop"]
    nba_df_checked["Def_Diff"] = nba_df_checked["Is_Stop"]
    return nba_df_checked


def process_stop_py(file_path, year, season_str):
    """Process one season into possession-level STOP targets."""
    print(f"  Starting STOP Processing for {season_str}...")
    start_time = time.time()
    nba_df = _base_processing(file_path)
    if nba_df is None:
        return None

    nba_df_checked, _ = prepare_standard_possession_df(nba_df, "STOP")
    nba_df_checked = build_stop_flags(nba_df_checked)
    nba_filt = nba_df_checked[nba_df_checked["End_of_Possession"]].copy()

    stop_count = int(nba_filt["Is_Stop"].sum())
    turnover_count = int(nba_filt["Is_Turnover_Stop"].sum())
    block_count = int(nba_filt["Is_Recovered_Block_Stop"].sum())
    overlap_count = int(
        (
            nba_filt["Is_Turnover_Stop"].eq(1)
            & nba_filt["Is_Recovered_Block_Stop"].eq(1)
        ).sum()
    )
    print(
        "      STOP possessions: "
        f"{stop_count}/{len(nba_filt)} "
        f"(turnover={turnover_count}, recovered_block={block_count}, overlap={overlap_count})"
    )

    output = _finalize_df(
        nba_filt,
        [
            "Is_Stop",
            "Is_Turnover_Stop",
            "Is_Recovered_Block_Stop",
            "Off_Diff",
            "Def_Diff",
        ],
        year,
    )
    print(f"  Finished STOP Processing. Time: {time.time() - start_time:.2f} seconds.")
    return output
