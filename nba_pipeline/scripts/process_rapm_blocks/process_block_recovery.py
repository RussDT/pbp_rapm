"""
BLOCK_RECOVERY processor.

Builds one row per blocked field-goal attempt. The target is whether the
defending team recovered the blocked shot.
"""

import time

import numpy as np
import pandas as pd

from .common import (
    _base_processing,
    _finalize_df,
    case_when,
    series_contains,
)


ADMIN_EVENT_TYPES = {
    "Substitution",
    "Timeout",
    "Ejection",
    "Empty",
    "StartOfPeriod",
    "Violation",
}


def _event_side(row, event_type):
    home_desc = str(row.get("home_description", ""))
    visitor_desc = str(row.get("visitor_description", ""))

    if event_type == "Rebound":
        if "REBOUND" in home_desc.upper():
            return "Home", "rebound"
        if "REBOUND" in visitor_desc.upper():
            return "Away", "rebound"
    elif event_type == "JumpBall" or "Jump Ball" in home_desc or "Jump Ball" in visitor_desc:
        if home_desc.strip():
            return "Home", "jump_ball"
        if visitor_desc.strip():
            return "Away", "jump_ball"
    elif event_type == "Turnover":
        if "Turnover" in home_desc:
            return "Home", "turnover"
        if "Turnover" in visitor_desc:
            return "Away", "turnover"
    elif event_type in {"MAKE", "MISS"}:
        if "PTS" in home_desc or "MISS" in home_desc:
            return "Home", "shot"
        if "PTS" in visitor_desc or "MISS" in visitor_desc:
            return "Away", "shot"

    return "", ""


def _resolve_recovery_rows(nba_df, max_lookahead=8):
    recovery_side = pd.Series("", index=nba_df.index, dtype=object)
    resolution = pd.Series("", index=nba_df.index, dtype=object)
    recovery_player = pd.Series(np.nan, index=nba_df.index, dtype="float64")
    resolution_event_num = pd.Series(np.nan, index=nba_df.index, dtype="float64")

    group_cols = ["game_id"] if "game_id" in nba_df.columns else []
    groups = nba_df.groupby(group_cols, sort=False) if group_cols else [(None, nba_df)]

    for _, game in groups:
        indices = list(game.index)
        positions = {idx: pos for pos, idx in enumerate(indices)}
        block_indices = game.index[game["Is_Blocked_FGA"].eq(1)].tolist()

        for block_idx in block_indices:
            start_pos = positions[block_idx] + 1
            block_period = game.at[block_idx, "period"] if "period" in game.columns else None
            for pos in range(start_pos, min(start_pos + max_lookahead, len(indices))):
                idx = indices[pos]
                if block_period is not None and game.at[idx, "period"] != block_period:
                    break

                event_type = game.at[idx, "event_type"]
                if event_type in ADMIN_EVENT_TYPES:
                    continue
                if event_type == "EndOfPeriod":
                    break

                side, kind = _event_side(game.loc[idx], event_type)
                if not side:
                    continue

                recovery_side.at[block_idx] = side
                resolution.at[block_idx] = kind
                resolution_event_num.at[block_idx] = pd.to_numeric(game.at[idx, "event_num"], errors="coerce")
                recovery_player.at[block_idx] = pd.to_numeric(game.at[idx, "player1_id"], errors="coerce")
                break

    return recovery_side, resolution, recovery_player, resolution_event_num


def process_block_recovery_py(file_path, year, season_str):
    """Processes blocked-FGA recovery opportunities."""
    print(f"  Starting BLOCK_RECOVERY Processing for {season_str}...")
    nba_df = _base_processing(file_path)
    if nba_df is None:
        return None
    start_time = time.time()

    home_desc = nba_df["home_description"].fillna("").astype(str)
    visitor_desc = nba_df["visitor_description"].fillna("").astype(str)
    all_desc = home_desc + " " + visitor_desc

    home_blocked_fga = (
        nba_df["event_type"].eq("MISS")
        & series_contains(home_desc, r"\bMISS\b", case=False, regex=True)
        & series_contains(visitor_desc, r"\b(BLOCK|BLK)\b", case=False, regex=True)
    )
    away_blocked_fga = (
        nba_df["event_type"].eq("MISS")
        & series_contains(visitor_desc, r"\bMISS\b", case=False, regex=True)
        & series_contains(home_desc, r"\b(BLOCK|BLK)\b", case=False, regex=True)
    )
    any_blocked_fga = (
        nba_df["event_type"].eq("MISS")
        & series_contains(all_desc, r"\b(BLOCK|BLK)\b", case=False, regex=True)
        & (home_blocked_fga | away_blocked_fga)
    )

    nba_df["Is_Blocked_FGA"] = any_blocked_fga.astype(int)
    nba_df["TeamOnOffense"] = case_when(
        home_blocked_fga, "Home",
        away_blocked_fga, "Away",
        "",
    )

    recovery_side, resolution, recovery_player, resolution_event_num = _resolve_recovery_rows(nba_df)
    nba_df["Recovery_Team"] = recovery_side
    nba_df["Recovery_Resolution"] = resolution
    nba_df["Recovery_Player"] = recovery_player
    nba_df["Recovery_Event_Num"] = resolution_event_num

    for i in range(1, 6):
        nba_df[f"O{i}"] = np.where(
            nba_df["TeamOnOffense"].eq("Away"),
            nba_df[f"a{i}"],
            np.where(nba_df["TeamOnOffense"].eq("Home"), nba_df[f"h{i}"], np.nan),
        )
        nba_df[f"D{i}"] = np.where(
            nba_df["TeamOnOffense"].eq("Away"),
            nba_df[f"h{i}"],
            np.where(nba_df["TeamOnOffense"].eq("Home"), nba_df[f"a{i}"], np.nan),
        )

    initial_rows = len(nba_df)
    nba_df = nba_df[nba_df["Is_Blocked_FGA"].eq(1)].copy()
    print(f"      Filtered to blocked FGA events: {len(nba_df)} rows (from {initial_rows})")
    if nba_df.empty:
        print("      Warning: No blocked FGA events found. Returning None.")
        return None

    nba_df["Shooter"] = pd.to_numeric(nba_df["player1_id"], errors="coerce")

    defense_side = nba_df["TeamOnOffense"].map({"Home": "Away", "Away": "Home"}).fillna("")
    nba_df["Block_Recovered_By_Defense"] = nba_df["Recovery_Team"].eq(defense_side).astype(int)
    print(
        "      Defensive block recoveries: "
        f"{int(nba_df['Block_Recovered_By_Defense'].sum())}/{len(nba_df)}"
    )

    nba_output = _finalize_df(
        nba_df,
        "Block_Recovered_By_Defense",
        year,
        id_cols=["game_id", "Shooter", "Recovery_Player", "Recovery_Event_Num"],
    )

    end_time = time.time()
    print(f"  Finished BLOCK_RECOVERY Processing. Time: {end_time - start_time:.2f} seconds.")
    return nba_output
