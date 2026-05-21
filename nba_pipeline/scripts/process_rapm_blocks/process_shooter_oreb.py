"""
SHOOTER_OREB processor.

Builds missed-FGA rebound opportunities with separate columns for the shooter,
the four non-shooter offensive players, and the five defensive players.
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


def _non_shooter_offense(row):
    shooter = row["Shooter"]
    if pd.isna(shooter):
        return pd.Series([np.nan, np.nan, np.nan, np.nan], index=[f"O{i}" for i in range(1, 5)])

    try:
        shooter_id = int(shooter)
    except (TypeError, ValueError):
        return pd.Series([np.nan, np.nan, np.nan, np.nan], index=[f"O{i}" for i in range(1, 5)])

    others = []
    for col in [f"RawO{i}" for i in range(1, 6)]:
        val = row[col]
        if pd.isna(val):
            continue
        try:
            player_id = int(val)
        except (TypeError, ValueError):
            continue
        if player_id != 0 and player_id != shooter_id:
            others.append(player_id)

    if len(others) != 4:
        return pd.Series([np.nan, np.nan, np.nan, np.nan], index=[f"O{i}" for i in range(1, 5)])

    return pd.Series(others, index=[f"O{i}" for i in range(1, 5)])


def process_shooter_oreb_py(file_path, year, season_str):
    """
    Processes missed field-goal rebound opportunities.

    Output rows are missed FGAs whose next event is a rebound. `O1-O4` are the
    four offensive players excluding the shooter, `Shooter` is a separate player
    column, and `D1-D5` are the defensive players.
    """
    print(f"  Starting SHOOTER_OREB Processing for {season_str}...")
    nba_df = _base_processing(file_path)
    if nba_df is None:
        return None
    start_time = time.time()

    group_cols = ["game_id"] if "game_id" in nba_df.columns else None
    if group_cols:
        nba_df["Next_player1_id"] = nba_df.groupby(group_cols)["player1_id"].shift(-1)
    else:
        nba_df["Next_player1_id"] = nba_df["player1_id"].shift(-1)

    initial_rows = len(nba_df)
    nba_df = nba_df[nba_df["event_type"] == "MISS"].copy()
    print(f"      Filtered to missed FGA events: {len(nba_df)} rows (from {initial_rows})")

    if nba_df.empty:
        print("      Warning: No missed FGA events found. Returning None.")
        return None

    home_miss = series_contains(nba_df["home_description"], "MISS", case=False)
    away_miss = series_contains(nba_df["visitor_description"], "MISS", case=False)
    next_home_reb = series_contains(nba_df["Next_home_desc"], "REBOUND", case=False)
    next_away_reb = series_contains(nba_df["Next_visitor_desc"], "REBOUND", case=False)

    nba_df["TeamOnOffense"] = case_when(
        home_miss, "Home",
        away_miss, "Away",
        "",
    )

    nba_df["Offensive_Rebound"] = case_when(
        (nba_df["TeamOnOffense"] == "Home") & next_home_reb, 1,
        (nba_df["TeamOnOffense"] == "Away") & next_away_reb, 1,
        0,
    ).astype(int)

    rebound_opportunity = (
        ((nba_df["TeamOnOffense"] == "Home") & (next_home_reb | next_away_reb))
        | ((nba_df["TeamOnOffense"] == "Away") & (next_home_reb | next_away_reb))
    )
    if "time_quarter" in nba_df.columns:
        rebound_opportunity &= ~series_contains(nba_df["time_quarter"], "0:00", case=False)

    nba_df = nba_df[rebound_opportunity].copy()
    print(
        "      Kept missed-FGA rebound opportunities: "
        f"{len(nba_df)} rows ({nba_df['Offensive_Rebound'].sum()} offensive rebounds)"
    )

    if nba_df.empty:
        print("      Warning: No rebound opportunities found. Returning None.")
        return None

    nba_df["Shooter"] = pd.to_numeric(nba_df["player1_id"], errors="coerce")
    nba_df["Rebounder"] = pd.to_numeric(nba_df["Next_player1_id"], errors="coerce")
    nba_df["Self_Offensive_Rebound"] = (
        (nba_df["Offensive_Rebound"] == 1)
        & nba_df["Shooter"].notna()
        & nba_df["Rebounder"].notna()
        & (nba_df["Shooter"].astype("Int64") == nba_df["Rebounder"].astype("Int64"))
    ).astype(int)

    for i in range(1, 6):
        nba_df[f"RawO{i}"] = np.where(
            nba_df["TeamOnOffense"] == "Away",
            nba_df[f"a{i}"],
            np.where(nba_df["TeamOnOffense"] == "Home", nba_df[f"h{i}"], np.nan),
        )
        nba_df[f"D{i}"] = np.where(
            nba_df["TeamOnOffense"] == "Away",
            nba_df[f"h{i}"],
            np.where(nba_df["TeamOnOffense"] == "Home", nba_df[f"a{i}"], np.nan),
        )

    nba_df[[f"O{i}" for i in range(1, 5)]] = nba_df.apply(_non_shooter_offense, axis=1)
    before_lineup_filter = len(nba_df)
    nba_df = nba_df.dropna(subset=["Shooter", *[f"O{i}" for i in range(1, 5)], *[f"D{i}" for i in range(1, 6)]])
    print(
        "      Rows after shooter/non-shooter lineup filter: "
        f"{len(nba_df)} (removed {before_lineup_filter - len(nba_df)})"
    )

    numerator_cols = ["Offensive_Rebound", "Self_Offensive_Rebound"]
    nba_output = _finalize_df(
        nba_df,
        numerator_cols,
        year,
        id_cols=["game_id", "Shooter", "Rebounder"],
        o_cols=[f"O{i}" for i in range(1, 5)],
        d_cols=[f"D{i}" for i in range(1, 6)],
    )

    end_time = time.time()
    print(f"  Finished SHOOTER_OREB Processing. Time: {end_time - start_time:.2f} seconds.")
    return nba_output
