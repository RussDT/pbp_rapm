"""Shooter/non-shooter scoring-role inputs for additive offensive RAPM.

Every standard possession is expanded to five focal-player rows:

* ``OnBall_Scoring_Points`` assigns scoring value to the shooter.
* ``OffBall_Scoring_Points`` assigns one quarter of that scoring value to each
  of the four offensive players who did not take the shot/free throw.

On fully resolved possessions, each role target sums across the five focal rows
to the parent possession scoring value. The hierarchy solver fits the
on-ball-minus-off-ball contrast and anchors it to offensive RAPM.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from .common import (
    _base_processing,
    _finalize_df,
    case_when,
    fetch_player_stats_supabase,
    prepare_standard_possession_df,
    series_contains,
)


OFFENSE_COLS = [f"O{i}" for i in range(1, 6)]
CONTEXT_COLS = [f"O{i}" for i in range(1, 5)]
DEFENSE_COLS = [f"D{i}" for i in range(1, 6)]


def _standard_scoring_event_value(
    nba_df: pd.DataFrame,
    year: int,
    file_path: str,
) -> pd.Series:
    """Match the standard RAPM offense target: actual FGs and expected FTs."""
    is_playoffs = "_PS" in os.path.basename(file_path).upper()
    player_stats = fetch_player_stats_supabase(year, is_playoffs)
    nba_df["player1_id_numeric"] = pd.to_numeric(
        nba_df.get("player1_id"), errors="coerce"
    ).astype("Int64")
    if player_stats is not None:
        player_stats = player_stats.copy()
        player_stats["PlayerID"] = pd.to_numeric(
            player_stats["PlayerID"], errors="coerce"
        ).astype("Int64")
        player_stats = player_stats.dropna(subset=["PlayerID"]).drop_duplicates(
            "PlayerID"
        )
        nba_df = nba_df.merge(
            player_stats[["PlayerID", "FTPerc"]],
            left_on="player1_id_numeric",
            right_on="PlayerID",
            how="left",
        )
    elif "FTPerc" not in nba_df.columns:
        nba_df["FTPerc"] = np.nan

    ft_pct = pd.to_numeric(nba_df["FTPerc"], errors="coerce").fillna(0.75)
    actual_score = (
        pd.to_numeric(nba_df["Home_Action_Score"], errors="coerce").fillna(0)
        + pd.to_numeric(nba_df["Away_Action_Score"], errors="coerce").fillna(0)
    ).astype(float)
    is_ft = (
        series_contains(nba_df["home_description"], "Free Throw", case=False)
        | series_contains(nba_df["visitor_description"], "Free Throw", case=False)
        | nba_df["event_type"].eq("FreeThrow")
    )
    is_two_make = (
        (
            series_contains(nba_df["home_description"], "PTS", case=False)
            & ~series_contains(nba_df["home_description"], "3PT", case=False)
            & nba_df["event_type"].eq("MAKE")
        )
        | (
            series_contains(nba_df["visitor_description"], "PTS", case=False)
            & ~series_contains(nba_df["visitor_description"], "3PT", case=False)
            & nba_df["event_type"].eq("MAKE")
        )
    )
    is_three = (
        series_contains(nba_df["home_description"], "3PT", case=False)
        | series_contains(nba_df["visitor_description"], "3PT", case=False)
    )
    values = case_when(
        is_ft,
        ft_pct,
        is_two_make,
        2.0,
        is_three,
        actual_score,
        0.0,
    )
    return pd.Series(values, index=nba_df.index, dtype=float)


def build_scoring_role_rows(
    terminal_rows: pd.DataFrame,
    scorer_points: pd.DataFrame,
) -> pd.DataFrame:
    """Expand terminal possessions and construct symmetric role targets."""
    base = terminal_rows.reset_index(drop=True).copy()
    repeated = base.loc[base.index.repeat(5)].reset_index(drop=True)
    offense = base[OFFENSE_COLS].apply(pd.to_numeric, errors="coerce").to_numpy()
    repeated["Focal_Player"] = offense.reshape(-1)

    context_blocks = [
        np.delete(offense, focal_idx, axis=1) for focal_idx in range(5)
    ]
    repeated[CONTEXT_COLS] = np.stack(context_blocks, axis=1).reshape(-1, 4)

    repeated = repeated.merge(
        scorer_points,
        on=["game_id", "poss_group", "Focal_Player"],
        how="left",
        validate="many_to_one",
    )
    repeated["OnBall_Scoring_Points"] = pd.to_numeric(
        repeated["OnBall_Scoring_Points"], errors="coerce"
    ).fillna(0.0)
    resolved = repeated.groupby(
        ["game_id", "poss_group"], sort=False
    )["OnBall_Scoring_Points"].transform("sum")
    repeated["Unresolved_Scoring_Points"] = (
        pd.to_numeric(repeated["Scoring_Points"], errors="coerce").fillna(0.0)
        - resolved
    )
    repeated["Role_Resolution_Complete"] = repeated[
        "Unresolved_Scoring_Points"
    ].abs().lt(1e-9)
    repeated["OffBall_Scoring_Points"] = (
        pd.to_numeric(repeated["Scoring_Points"], errors="coerce").fillna(0.0)
        - repeated["OnBall_Scoring_Points"]
    ) / 4.0
    return repeated


def expand_scoring_role_possessions(
    terminal_rows: pd.DataFrame,
    scorer_points: pd.DataFrame,
    year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Finalize separate on-ball and off-ball role parquet surfaces."""
    repeated = build_scoring_role_rows(terminal_rows, scorer_points)

    shared = [
        "Scoring_Points",
        "Unresolved_Scoring_Points",
        "Role_Resolution_Complete",
    ]
    onball = _finalize_df(
        repeated,
        ["OnBall_Scoring_Points", *shared],
        year,
        id_cols=["game_id", "poss_group", "Focal_Player"],
        o_cols=CONTEXT_COLS,
        d_cols=DEFENSE_COLS,
    )
    offball = _finalize_df(
        repeated,
        ["OffBall_Scoring_Points", *shared],
        year,
        id_cols=["game_id", "poss_group", "Focal_Player"],
        o_cols=CONTEXT_COLS,
        d_cols=DEFENSE_COLS,
    )
    return onball, offball


def process_scoring_roles_py(file_path, year, season_str):
    """Build on-ball, off-ball, and matching parent scoring parquets."""
    print(f"  Starting SCORING role processing for {season_str}...")
    nba_df = _base_processing(file_path)
    if nba_df is None:
        return None
    start_time = time.time()
    nba_df, group_cols = prepare_standard_possession_df(nba_df, "SCORING_ROLES")
    nba_df["Scoring_Points_Event"] = _standard_scoring_event_value(
        nba_df, year, file_path
    )
    nba_df["Scoring_Player"] = pd.to_numeric(
        nba_df.get("player1_id"), errors="coerce"
    ).where(nba_df["Scoring_Points_Event"].gt(0))
    scorer_in_lineup = nba_df[OFFENSE_COLS].eq(
        nba_df["Scoring_Player"], axis=0
    ).any(axis=1)
    nba_df.loc[
        nba_df["Scoring_Points_Event"].gt(0) & ~scorer_in_lineup,
        "Scoring_Player",
    ] = np.nan

    if group_cols:
        nba_df["Scoring_Points_Total"] = nba_df.groupby(group_cols)[
            "Scoring_Points_Event"
        ].cumsum()
    else:
        nba_df["Scoring_Points_Total"] = nba_df["Scoring_Points_Event"].cumsum()

    terminal = nba_df[nba_df["End_of_Possession"]].copy()
    if group_cols:
        terminal["Scoring_Points"] = terminal["Scoring_Points_Total"] - terminal.groupby(
            group_cols
        )["Scoring_Points_Total"].shift(fill_value=0)
    else:
        terminal["Scoring_Points"] = (
            terminal["Scoring_Points_Total"]
            - terminal["Scoring_Points_Total"].shift(fill_value=0)
        )
    scoring_events = nba_df[
        nba_df["Scoring_Points_Event"].gt(0)
        & nba_df["Scoring_Player"].notna()
    ].copy()
    scorer_points = (
        scoring_events.groupby(
            ["game_id", "poss_group", "Scoring_Player"],
            as_index=False,
            sort=False,
        )["Scoring_Points_Event"]
        .sum()
        .rename(
            columns={
                "Scoring_Player": "Focal_Player",
                "Scoring_Points_Event": "OnBall_Scoring_Points",
            }
        )
    )
    onball, offball = expand_scoring_role_possessions(terminal, scorer_points, year)
    parent_rows = terminal.copy()
    parent_rows["Off_Diff"] = parent_rows["Scoring_Points"]
    parent_rows["Def_Diff"] = parent_rows["Scoring_Points"]
    parent_rows["Net_Diff"] = parent_rows["Scoring_Points"]
    parent = _finalize_df(
        parent_rows,
        ["Net_Diff", "Off_Diff", "Def_Diff", "Scoring_Points"],
        year,
    )
    total = float(nba_df["Scoring_Points_Event"].sum())
    resolved = float(scorer_points["OnBall_Scoring_Points"].sum())
    print(
        f"      Scoring-role resolution: {resolved:,.1f}/{total:,.1f} "
        f"({100.0 * resolved / total:.3f}%)"
    )
    print(
        f"  Finished SCORING role processing in {time.time() - start_time:.2f}s."
    )
    return onball, offball, parent
