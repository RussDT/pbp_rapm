#!/usr/bin/env python3
"""
Build and query 2026 lineup WOWY stats from raw play-by-play event counts.

This parser does not use RAPM possession outputs. It builds possession groups
from raw PBP, counts team events within each possession, then aggregates to
lineups and player on/off splits.

Examples:
    python lineup_stats.py build --year 26
    python lineup_stats.py query --year 26 --exact "203999,1627750,1629008,203932,1641774"
    python lineup_stats.py player 203999 --year 26
"""

from __future__ import annotations

import argparse
import sys
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

from process_rapm_blocks.common import (
    RIM_ACTION_TYPES,
    _base_processing,
    case_when,
    ft_off_check_py,
    propagate_player_values_py,
    series_contains,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_DIR.parent
PROCESSED_DIR = PIPELINE_DIR / "processed"
RAW_DATA_DIR = PIPELINE_DIR / "raw_data"
AUTOCOMPLETE_MAP = PROJECT_ROOT / "autocomplete_map.csv"

CANON_COLS = [f"P{i}" for i in range(1, 6)]
HOME_COLS = [f"h{i}" for i in range(1, 6)]
AWAY_COLS = [f"a{i}" for i in range(1, 6)]
POSS_KEY = ["game_id", "poss_group"]

OFF_RAW_COLS = [
    "off_poss",
    "off_points",
    "off_fga",
    "off_fgm",
    "off_fg_miss",
    "off_fta",
    "off_ftm",
    "off_orb",
    "off_tov",
    "off_rim_att",
    "off_rim_made",
    "off_mid_att",
    "off_mid_made",
    "off_three_att",
    "off_three_made",
]

DEF_RAW_COLS = [
    "def_poss",
    "def_opp_points",
    "def_opp_fga",
    "def_opp_fgm",
    "def_opp_fg_miss",
    "def_opp_fta",
    "def_opp_ftm",
    "def_opp_orb",
    "def_tov_forced",
    "def_opp_rim_att",
    "def_opp_rim_made",
    "def_opp_mid_att",
    "def_opp_mid_made",
    "def_opp_three_att",
    "def_opp_three_made",
]

DERIVED_COLS = [
    "off_tsa",
    "def_opp_tsa",
    "ortg",
    "drtg",
    "net",
    "off_ts_pct",
    "def_ts_pct_allowed",
    "off_fgorb_pct",
    "def_fgorb_pct_allowed",
    "off_rim_freq",
    "def_rim_freq_allowed",
    "off_mid_freq",
    "def_mid_freq_allowed",
    "off_three_freq",
    "def_three_freq_allowed",
    "off_rim_acc",
    "def_rim_acc_allowed",
    "off_mid_acc",
    "def_mid_acc_allowed",
    "off_three_acc",
    "def_three_acc_allowed",
]

LINEUP_DISPLAY_COLS = [
    "team_abbr",
    "lineup_key",
    *CANON_COLS,
    "off_poss",
    "def_poss",
    "ortg",
    "drtg",
    "net",
    "off_ts_pct",
    "def_ts_pct_allowed",
    "off_fgorb_pct",
    "def_fgorb_pct_allowed",
    "off_rim_freq",
    "def_rim_freq_allowed",
    "off_mid_freq",
    "def_mid_freq_allowed",
    "off_three_freq",
    "def_three_freq_allowed",
    "off_rim_acc",
    "def_rim_acc_allowed",
    "off_mid_acc",
    "def_mid_acc_allowed",
    "off_three_acc",
    "def_three_acc_allowed",
]

PLAYER_DISPLAY_COLS = [
    "team_abbr",
    "split",
    "off_poss",
    "def_poss",
    "off_points",
    "def_opp_points",
    "off_fga",
    "off_fta",
    "off_orb",
    "off_tov",
    "ortg",
    "drtg",
    "net",
    "off_ts_pct",
    "def_ts_pct_allowed",
    "off_fgorb_pct",
    "def_fgorb_pct_allowed",
    "off_rim_freq",
    "def_rim_freq_allowed",
    "off_mid_freq",
    "def_mid_freq_allowed",
    "off_three_freq",
    "def_three_freq_allowed",
    "off_rim_acc",
    "def_rim_acc_allowed",
    "off_mid_acc",
    "def_mid_acc_allowed",
    "off_three_acc",
    "def_three_acc_allowed",
]


def build_default_output_path(year: int) -> Path:
    return PROCESSED_DIR / f"LINEUP_STATS{year}.parquet"


def pct(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = numerator.astype(float)
    denominator = denominator.astype(float)
    return pd.Series(
        np.where(denominator > 0, 100.0 * numerator / denominator, np.nan),
        index=numerator.index,
    )


def canonicalize_frame(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    values = frame[cols].apply(pd.to_numeric, errors="coerce")
    valid_mask = values.notna().all(axis=1) & ~values.eq(0).any(axis=1)
    values = values.loc[valid_mask].astype(np.int64)
    lineup_values = values.to_numpy(copy=True)
    lineup_values.sort(axis=1)
    canon = pd.DataFrame(lineup_values, columns=CANON_COLS, index=values.index)
    return canon


def load_name_map(year: int) -> dict[int, str]:
    name_map: dict[int, str] = {}
    if AUTOCOMPLETE_MAP.exists():
        ac = pd.read_csv(AUTOCOMPLETE_MAP)
        if {"nba_id", "player_name"}.issubset(ac.columns):
            for row in ac.dropna(subset=["nba_id", "player_name"]).itertuples(index=False):
                name_map[int(row.nba_id)] = str(row.player_name)

    raw_path = RAW_DATA_DIR / f"NBA{year}.parquet"
    if raw_path.exists():
        raw = pd.read_parquet(
            raw_path,
            columns=[
                "player1_id", "player1_name",
                "player2_id", "player2_name",
                "player3_id", "player3_name",
            ],
        )
        for id_col, name_col in [
            ("player1_id", "player1_name"),
            ("player2_id", "player2_name"),
            ("player3_id", "player3_name"),
        ]:
            subset = raw[[id_col, name_col]].dropna()
            if subset.empty:
                continue
            subset[id_col] = pd.to_numeric(subset[id_col], errors="coerce")
            subset = subset.dropna()
            subset[name_col] = subset[name_col].astype(str).str.strip()
            subset = subset[subset[name_col] != ""]
            for player_id, player_name in subset.drop_duplicates(subset=[id_col], keep="last").itertuples(index=False):
                name_map.setdefault(int(player_id), player_name)
    return name_map


def resolve_player_query(token: str, autocomplete: pd.DataFrame) -> tuple[int, str]:
    token = token.strip()
    if not token:
        raise ValueError("Empty player token")
    try:
        player_id = int(float(token))
        row = autocomplete.loc[autocomplete["nba_id"] == player_id]
        if row.empty:
            return player_id, f"ID_{player_id}"
        return player_id, str(row.iloc[0]["player_name"])
    except ValueError:
        pass

    matches = autocomplete[autocomplete["player_name"].str.contains(token, case=False, na=False)]
    if matches.empty:
        raise ValueError(f"No player found matching '{token}'")
    if len(matches) > 1:
        choices = ", ".join(
            f"{row.player_name} ({int(row.nba_id)})"
            for row in matches.head(10).itertuples(index=False)
        )
        raise ValueError(f"Multiple matches for '{token}': {choices}")
    row = matches.iloc[0]
    return int(row["nba_id"]), str(row["player_name"])


def resolve_player_list(raw_value: str | None, autocomplete: pd.DataFrame) -> list[int]:
    if not raw_value:
        return []
    return [resolve_player_query(token, autocomplete)[0] for token in raw_value.split(",")]


def infer_game_team_map(df: pd.DataFrame) -> pd.DataFrame:
    vote_frames: list[pd.DataFrame] = []
    for idx in [1, 2, 3]:
        player_id = pd.to_numeric(df[f"player{idx}_id"], errors="coerce")
        team_abbr = df[f"player{idx}_team_abbreviation"].astype(str).str.strip()
        valid = player_id.notna() & team_abbr.ne("")
        if not valid.any():
            continue

        home_match = reduce(
            lambda left, right: left | right,
            [pd.to_numeric(df[col], errors="coerce").eq(player_id) for col in HOME_COLS],
        )
        away_match = reduce(
            lambda left, right: left | right,
            [pd.to_numeric(df[col], errors="coerce").eq(player_id) for col in AWAY_COLS],
        )

        home_votes = df.loc[valid & home_match, ["game_id"]].copy()
        home_votes["side"] = "Home"
        home_votes["team_abbr"] = team_abbr.loc[valid & home_match]
        vote_frames.append(home_votes)

        away_votes = df.loc[valid & away_match, ["game_id"]].copy()
        away_votes["side"] = "Away"
        away_votes["team_abbr"] = team_abbr.loc[valid & away_match]
        vote_frames.append(away_votes)

    if not vote_frames:
        raise ValueError("Could not infer game team abbreviations from raw data")

    votes = pd.concat(vote_frames, ignore_index=True)
    team_counts = (
        votes.groupby(["game_id", "side", "team_abbr"])
        .size()
        .reset_index(name="votes")
        .sort_values(["game_id", "side", "votes"], ascending=[True, True, False])
    )
    winners = team_counts.drop_duplicates(subset=["game_id", "side"], keep="first")
    pivot = winners.pivot(index="game_id", columns="side", values="team_abbr").reset_index()
    pivot = pivot.rename(columns={"Home": "home_team_abbr", "Away": "away_team_abbr"})
    return pivot


def prepare_raw_possessions(year: int) -> pd.DataFrame:
    raw_path = RAW_DATA_DIR / f"NBA{year}.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw file: {raw_path}")

    nba_df = _base_processing(raw_path)
    if nba_df is None or nba_df.empty:
        raise ValueError(f"Could not preprocess raw file: {raw_path}")

    team_map = infer_game_team_map(nba_df)
    nba_df = nba_df.merge(team_map, on="game_id", how="left")

    ft_exclude_pattern = r"\b(1 of [23]|2 of 3|Technical|Flagrant)\b"
    nba_df["offensive_FT_Rebound"] = case_when(
        series_contains(nba_df["visitor_description"], "REBOUND", case=False)
        & series_contains(nba_df["Prev_visitor_desc"], "MISS", case=False)
        & series_contains(nba_df["Prev_visitor_desc"], "Free Throw", case=False)
        & ~series_contains(nba_df["Prev_visitor_desc"], ft_exclude_pattern, case=False, regex=True),
        True,
        series_contains(nba_df["home_description"], "REBOUND", case=False)
        & series_contains(nba_df["Prev_home_desc"], "MISS", case=False)
        & series_contains(nba_df["Prev_home_desc"], "Free Throw", case=False)
        & ~series_contains(nba_df["Prev_home_desc"], ft_exclude_pattern, case=False, regex=True),
        True,
        False,
    ).astype(bool)

    home_offensive_foul_no_turnover = (
        (nba_df["event_type"] == "Foul")
        & series_contains(nba_df["home_description"], r"offensive|charge", case=False, regex=True)
        & ~nba_df["Next_event"].eq("Turnover")
    )
    away_offensive_foul_no_turnover = (
        (nba_df["event_type"] == "Foul")
        & series_contains(nba_df["visitor_description"], r"offensive|charge", case=False, regex=True)
        & ~nba_df["Next_event"].eq("Turnover")
    )
    next_home_rebound = nba_df["Next_event"].eq("Rebound") & series_contains(
        nba_df["Next_home_desc"], "REBOUND|Rebound", case=False, regex=True
    )
    next_away_rebound = nba_df["Next_event"].eq("Rebound") & series_contains(
        nba_df["Next_visitor_desc"], "REBOUND|Rebound", case=False, regex=True
    )
    next_home_deadball_rebound = next_home_rebound & series_contains(
        nba_df["Next_home_desc"], "deadball", case=False
    )
    next_away_deadball_rebound = next_away_rebound & series_contains(
        nba_df["Next_visitor_desc"], "deadball", case=False
    )
    next_home_team_off_rebound = next_home_rebound & series_contains(
        nba_df["Next_home_desc"], "team offensive rebound", case=False
    )
    next_home_team_def_rebound = next_home_rebound & series_contains(
        nba_df["Next_home_desc"], "team defensive rebound", case=False
    )
    next_away_team_off_rebound = next_away_rebound & series_contains(
        nba_df["Next_visitor_desc"], "team offensive rebound", case=False
    )
    next_away_team_def_rebound = next_away_rebound & series_contains(
        nba_df["Next_visitor_desc"], "team defensive rebound", case=False
    )
    home_missed_last_ft = (
        (nba_df["event_type"] == "FreeThrow")
        & series_contains(nba_df["home_description"], r"\b(1 of 1|2 of 2|3 of 3)\b", case=False, regex=True)
        & series_contains(nba_df["home_description"], r"\bMISS\b", case=False, regex=True)
        & ~series_contains(nba_df["home_description"], "technical", case=False)
    )
    away_missed_last_ft = (
        (nba_df["event_type"] == "FreeThrow")
        & series_contains(nba_df["visitor_description"], r"\b(1 of 1|2 of 2|3 of 3)\b", case=False, regex=True)
        & series_contains(nba_df["visitor_description"], r"\bMISS\b", case=False, regex=True)
        & ~series_contains(nba_df["visitor_description"], "technical", case=False)
    )
    home_live_same_team_off_reb = next_home_rebound & ~next_home_deadball_rebound & ~next_home_team_def_rebound
    away_live_same_team_off_reb = next_away_rebound & ~next_away_deadball_rebound & ~next_away_team_def_rebound
    home_live_other_team_def_reb = next_away_rebound & ~next_away_deadball_rebound & ~next_away_team_off_rebound
    away_live_other_team_def_reb = next_home_rebound & ~next_home_deadball_rebound & ~next_home_team_off_rebound
    home_missed_last_ft_eop = home_missed_last_ft & ~(home_live_same_team_off_reb | home_live_other_team_def_reb)
    away_missed_last_ft_eop = away_missed_last_ft & ~(away_live_same_team_off_reb | away_live_other_team_def_reb)

    nba_df["End_of_Possession"] = case_when(
        (nba_df["event_type"] == "EndOfPeriod") & (nba_df["prev_seconds"] > 0),
        True,
        series_contains(nba_df["home_description"], r"Flagrant (2 of 2|3 of 3)", case=False, regex=True),
        True,
        series_contains(nba_df["visitor_description"], r"Flagrant (2 of 2|3 of 3)", case=False, regex=True),
        True,
        nba_df["event_type"] == "Turnover",
        True,
        series_contains(nba_df["home_description"], "REBOUND", case=False)
        & series_contains(nba_df["Prev_visitor_desc"], "MISS", case=False)
        & ~series_contains(nba_df["Prev_visitor_desc"], ft_exclude_pattern, case=False, regex=True),
        True,
        series_contains(nba_df["visitor_description"], "REBOUND", case=False)
        & series_contains(nba_df["Prev_home_desc"], "MISS", case=False)
        & ~series_contains(nba_df["Prev_home_desc"], ft_exclude_pattern, case=False, regex=True),
        True,
        series_contains(nba_df["home_description"], "PTS", case=False)
        & (nba_df["event_type"] == "MAKE")
        & ~series_contains(nba_df["Next_visitor_desc"], "S.FOUL", case=True),
        True,
        series_contains(nba_df["visitor_description"], "PTS", case=False)
        & (nba_df["event_type"] == "MAKE")
        & ~series_contains(nba_df["Next_home_desc"], "S.FOUL", case=True),
        True,
        series_contains(nba_df["home_description"], r"\b(1 of 1|2 of 2|3 of 3)\b", case=False, regex=True)
        & series_contains(nba_df["home_description"], "PTS", case=False)
        & ~(
            series_contains(nba_df["Prev_visitor_desc2"], "Transition", case=False)
            | series_contains(nba_df["Prev_visitor_desc"], "Transition", case=False)
            | series_contains(nba_df["home_description"], "Flagrant", case=False)
        ),
        True,
        series_contains(nba_df["visitor_description"], r"\b(1 of 1|2 of 2|3 of 3)\b", case=False, regex=True)
        & series_contains(nba_df["visitor_description"], "PTS", case=False)
        & ~(
            series_contains(nba_df["Prev_home_desc2"], "Transition", case=False)
            | series_contains(nba_df["Prev_home_desc"], "Transition", case=False)
            | series_contains(nba_df["visitor_description"], "Flagrant", case=False)
        ),
        True,
        home_offensive_foul_no_turnover,
        True,
        away_offensive_foul_no_turnover,
        True,
        home_missed_last_ft_eop,
        True,
        away_missed_last_ft_eop,
        True,
        False,
    ).astype(bool)

    nba_df["TeamOnOffense"] = case_when(
        series_contains(nba_df["home_description"], r"Flagrant (2 of 2|3 of 3)", case=False, regex=True),
        "Home",
        series_contains(nba_df["visitor_description"], r"Flagrant (2 of 2|3 of 3)", case=False, regex=True),
        "Away",
        series_contains(nba_df["home_description"], "REBOUND", case=False)
        & series_contains(nba_df["Prev_visitor_desc"], "MISS", case=False),
        "Away",
        series_contains(nba_df["visitor_description"], "REBOUND", case=False)
        & series_contains(nba_df["Prev_home_desc"], "MISS", case=False),
        "Home",
        series_contains(nba_df["home_description"], "Turnover", case=False),
        "Home",
        series_contains(nba_df["visitor_description"], "Turnover", case=False),
        "Away",
        series_contains(nba_df["home_description"], "PTS", case=False)
        & (nba_df["event_type"] == "MAKE"),
        "Home",
        series_contains(nba_df["visitor_description"], "PTS", case=False)
        & (nba_df["event_type"] == "MAKE"),
        "Away",
        series_contains(nba_df["home_description"], r"\b(1 of 1|2 of 2|3 of 3)\b", case=False, regex=True)
        & series_contains(nba_df["home_description"], "PTS", case=False)
        & ~(
            series_contains(nba_df["Prev_visitor_desc2"], "Transition", case=False)
            | series_contains(nba_df["Prev_visitor_desc"], "Transition", case=False)
            | series_contains(nba_df["home_description"], "Flagrant", case=False)
        ),
        "Home",
        series_contains(nba_df["visitor_description"], r"\b(1 of 1|2 of 2|3 of 3)\b", case=False, regex=True)
        & series_contains(nba_df["visitor_description"], "PTS", case=False)
        & ~(
            series_contains(nba_df["Prev_home_desc2"], "Transition", case=False)
            | series_contains(nba_df["Prev_home_desc"], "Transition", case=False)
            | series_contains(nba_df["visitor_description"], "Flagrant", case=False)
        ),
        "Away",
        home_offensive_foul_no_turnover,
        "Home",
        away_offensive_foul_no_turnover,
        "Away",
        home_missed_last_ft_eop,
        "Home",
        away_missed_last_ft_eop,
        "Away",
        "",
    )

    home_ft_row = (nba_df["event_type"] == "FreeThrow") & series_contains(
        nba_df["home_description"], "Free Throw", case=False
    )
    away_ft_row = (nba_df["event_type"] == "FreeThrow") & series_contains(
        nba_df["visitor_description"], "Free Throw", case=False
    )
    nba_df.loc[nba_df["TeamOnOffense"].eq("") & home_ft_row, "TeamOnOffense"] = "Home"
    nba_df.loc[nba_df["TeamOnOffense"].eq("") & away_ft_row, "TeamOnOffense"] = "Away"

    window_size = 5
    ft_events = nba_df.groupby("game_id")["event_type"].transform(
        lambda x: (x == "FreeThrow").rolling(window=window_size, min_periods=1).sum().shift(-(window_size - 1)).fillna(0)
    )
    sub_events = nba_df.groupby("game_id")["event_type"].transform(
        lambda x: (x == "Substitution").rolling(window=window_size, min_periods=1).sum().shift(-(window_size - 1)).fillna(0)
    )
    exclude_foul_pattern = r"T\.FOUL|FLAGRANT|Offens|Transition"
    nba_df["PotentialFoul"] = case_when(
        (ft_events > 0)
        & (sub_events > 0)
        & (nba_df["event_type"] == "Foul")
        & ~series_contains(nba_df["visitor_description"], exclude_foul_pattern, case=False, regex=True)
        & ~series_contains(nba_df["home_description"], exclude_foul_pattern, case=False, regex=True),
        True,
        False,
    ).astype(bool)

    nba_df = propagate_player_values_py(nba_df)
    nba_df = ft_off_check_py(nba_df)

    nba_df["poss_group"] = nba_df.groupby("game_id")["End_of_Possession"].transform(
        lambda x: x.shift(1, fill_value=False).cumsum()
    )
    nba_df["poss_offense"] = nba_df.groupby(POSS_KEY)["TeamOnOffense"].transform(
        lambda x: x.mask(x.eq("")).bfill().ffill()
    )
    nba_df.loc[nba_df["poss_offense"].eq("") & home_ft_row, "poss_offense"] = "Home"
    nba_df.loc[nba_df["poss_offense"].eq("") & away_ft_row, "poss_offense"] = "Away"
    nba_df = nba_df[nba_df["poss_offense"].isin(["Home", "Away"])].copy()
    return nba_df


def build_possession_summary(year: int) -> pd.DataFrame:
    df = prepare_raw_possessions(year)

    shot_text = (df["home_description"].fillna("") + " " + df["visitor_description"].fillna("")).str.strip()
    home_desc = df["home_description"].fillna("")
    visitor_desc = df["visitor_description"].fillna("")
    event_action_type = pd.to_numeric(df["event_action_type"], errors="coerce")
    terminal_poss_offense = df.groupby(POSS_KEY)["poss_offense"].transform("last")
    offense_segment = df.groupby(POSS_KEY)["poss_offense"].transform(
        lambda x: x.ne(x.shift(fill_value=x.iloc[0])).cumsum()
    )

    off_home = df["poss_offense"].eq("Home")
    off_away = df["poss_offense"].eq("Away")

    home_make = (df["event_type"] == "MAKE") & series_contains(df["home_description"], "PTS", case=False)
    away_make = (df["event_type"] == "MAKE") & series_contains(df["visitor_description"], "PTS", case=False)
    home_miss = (df["event_type"] == "MISS") & series_contains(df["home_description"], "MISS", case=False)
    away_miss = (df["event_type"] == "MISS") & series_contains(df["visitor_description"], "MISS", case=False)
    home_ft = (df["event_type"] == "FreeThrow") & series_contains(df["home_description"], "Free Throw", case=False)
    away_ft = (df["event_type"] == "FreeThrow") & series_contains(df["visitor_description"], "Free Throw", case=False)

    off_fga = (off_home & (home_make | home_miss)) | (off_away & (away_make | away_miss))
    off_fgm = (off_home & home_make) | (off_away & away_make)
    off_fg_miss = (off_home & home_miss) | (off_away & away_miss)

    is_three = shot_text.str.contains("3PT", case=False, regex=False, na=False)
    is_rim = event_action_type.isin(RIM_ACTION_TYPES) | shot_text.str.contains(
        r"layup|dunk| tip ",
        case=False,
        regex=True,
        na=False,
    )
    is_mid = off_fga & ~is_three & ~is_rim
    off_rim_att = off_fga & is_rim
    off_rim_made = off_fgm & is_rim
    off_mid_att = is_mid
    off_mid_made = off_fgm & is_mid
    off_three_att = off_fga & is_three
    off_three_made = off_fgm & is_three

    off_fta = (off_home & home_ft) | (off_away & away_ft)
    off_ftm = off_fta & ~shot_text.str.contains(r"\bMISS\b", case=False, regex=True, na=False)
    tech_ft = (home_ft & series_contains(home_desc, "Technical", case=False)) | (
        away_ft & series_contains(visitor_desc, "Technical", case=False)
    )
    foreign_tech_ft = tech_ft & df["poss_offense"].ne(terminal_poss_offense)
    split_foreign = df["poss_offense"].ne(terminal_poss_offense) & ~foreign_tech_ft

    home_tov = (df["event_type"] == "Turnover") & series_contains(df["home_description"], "Turnover", case=False)
    away_tov = (df["event_type"] == "Turnover") & series_contains(df["visitor_description"], "Turnover", case=False)
    off_tov = (off_home & home_tov) | (off_away & away_tov)

    home_reb = (df["event_type"] == "Rebound") & series_contains(df["home_description"], "REBOUND|Rebound", case=False, regex=True)
    away_reb = (df["event_type"] == "Rebound") & series_contains(df["visitor_description"], "REBOUND|Rebound", case=False, regex=True)
    prev_home_fg_miss = (df["Prev_Event"] == "MISS") & series_contains(df["Prev_home_desc"], "MISS", case=False)
    prev_away_fg_miss = (df["Prev_Event"] == "MISS") & series_contains(df["Prev_visitor_desc"], "MISS", case=False)
    off_orb = (off_home & home_reb & prev_home_fg_miss) | (off_away & away_reb & prev_away_fg_miss)

    points_event = np.where(off_home, df["Home_Action_Score"], df["Away_Action_Score"]).astype(float)
    carry_points_event = np.where(split_foreign, points_event, 0.0)
    carry_fga_event = np.where(split_foreign, off_fga.astype(int), 0)
    carry_fgm_event = np.where(split_foreign, off_fgm.astype(int), 0)
    carry_fg_miss_event = np.where(split_foreign, off_fg_miss.astype(int), 0)
    carry_fta_event = np.where(split_foreign, off_fta.astype(int), 0)
    carry_ftm_event = np.where(split_foreign, off_ftm.astype(int), 0)
    carry_orb_event = np.where(split_foreign, off_orb.astype(int), 0)
    carry_tov_event = np.where(split_foreign, off_tov.astype(int), 0)
    carry_rim_att_event = np.where(split_foreign, off_rim_att.astype(int), 0)
    carry_rim_made_event = np.where(split_foreign, off_rim_made.astype(int), 0)
    carry_mid_att_event = np.where(split_foreign, off_mid_att.astype(int), 0)
    carry_mid_made_event = np.where(split_foreign, off_mid_made.astype(int), 0)
    carry_three_att_event = np.where(split_foreign, off_three_att.astype(int), 0)
    carry_three_made_event = np.where(split_foreign, off_three_made.astype(int), 0)

    masked_rows = foreign_tech_ft | split_foreign
    points_event = np.where(masked_rows, 0.0, points_event)
    off_fga_event = np.where(masked_rows, 0, off_fga.astype(int))
    off_fgm_event = np.where(masked_rows, 0, off_fgm.astype(int))
    off_fg_miss_event = np.where(masked_rows, 0, off_fg_miss.astype(int))
    off_fta_event = np.where(masked_rows, 0, off_fta.astype(int))
    off_ftm_event = np.where(masked_rows, 0, off_ftm.astype(int))
    off_orb_event = np.where(masked_rows, 0, off_orb.astype(int))
    off_tov_event = np.where(masked_rows, 0, off_tov.astype(int))
    off_rim_att_event = np.where(masked_rows, 0, off_rim_att.astype(int))
    off_rim_made_event = np.where(masked_rows, 0, off_rim_made.astype(int))
    off_mid_att_event = np.where(masked_rows, 0, off_mid_att.astype(int))
    off_mid_made_event = np.where(masked_rows, 0, off_mid_made.astype(int))
    off_three_att_event = np.where(masked_rows, 0, off_three_att.astype(int))
    off_three_made_event = np.where(masked_rows, 0, off_three_made.astype(int))

    df = df.assign(
        points_event=points_event,
        off_fga_event=off_fga_event,
        off_fgm_event=off_fgm_event,
        off_fg_miss_event=off_fg_miss_event,
        off_fta_event=off_fta_event,
        off_ftm_event=off_ftm_event,
        off_orb_event=off_orb_event,
        off_tov_event=off_tov_event,
        off_rim_att_event=off_rim_att_event,
        off_rim_made_event=off_rim_made_event,
        off_mid_att_event=off_mid_att_event,
        off_mid_made_event=off_mid_made_event,
        off_three_att_event=off_three_att_event,
        off_three_made_event=off_three_made_event,
        carry_points_event=carry_points_event,
        carry_fga_event=carry_fga_event,
        carry_fgm_event=carry_fgm_event,
        carry_fg_miss_event=carry_fg_miss_event,
        carry_fta_event=carry_fta_event,
        carry_ftm_event=carry_ftm_event,
        carry_orb_event=carry_orb_event,
        carry_tov_event=carry_tov_event,
        carry_rim_att_event=carry_rim_att_event,
        carry_rim_made_event=carry_rim_made_event,
        carry_mid_att_event=carry_mid_att_event,
        carry_mid_made_event=carry_mid_made_event,
        carry_three_att_event=carry_three_att_event,
        carry_three_made_event=carry_three_made_event,
        offense_segment=offense_segment,
    )

    counts = (
        df.groupby(POSS_KEY, as_index=False)
        .agg(
            Points=("points_event", "sum"),
            FGA=("off_fga_event", "sum"),
            FGM=("off_fgm_event", "sum"),
            FGMiss=("off_fg_miss_event", "sum"),
            FTA=("off_fta_event", "sum"),
            FTM=("off_ftm_event", "sum"),
            ORB=("off_orb_event", "sum"),
            TOV=("off_tov_event", "sum"),
            RimAtt=("off_rim_att_event", "sum"),
            RimMade=("off_rim_made_event", "sum"),
            MidAtt=("off_mid_att_event", "sum"),
            MidMade=("off_mid_made_event", "sum"),
            ThreeAtt=("off_three_att_event", "sum"),
            ThreeMade=("off_three_made_event", "sum"),
        )
    )

    eop_rows = df[df["End_of_Possession"]].copy()
    eop_rows = eop_rows.groupby(POSS_KEY, as_index=False).tail(1).copy()
    for idx in range(1, 6):
        eop_rows[f"O{idx}"] = np.where(eop_rows["poss_offense"] == "Home", eop_rows[f"h{idx}"], eop_rows[f"a{idx}"])
        eop_rows[f"D{idx}"] = np.where(eop_rows["poss_offense"] == "Home", eop_rows[f"a{idx}"], eop_rows[f"h{idx}"])

    eop_rows["off_team_abbr"] = np.where(
        eop_rows["poss_offense"] == "Home",
        eop_rows["home_team_abbr"],
        eop_rows["away_team_abbr"],
    )
    eop_rows["def_team_abbr"] = np.where(
        eop_rows["poss_offense"] == "Home",
        eop_rows["away_team_abbr"],
        eop_rows["home_team_abbr"],
    )

    poss = counts.merge(
        eop_rows[
            POSS_KEY
            + ["poss_offense", "off_team_abbr", "def_team_abbr"]
            + [f"O{i}" for i in range(1, 6)]
            + [f"D{i}" for i in range(1, 6)]
        ],
        on=POSS_KEY,
        how="inner",
    )
    poss["OffPoss"] = 1
    poss["DefPoss"] = 1

    carry_rows = df.loc[split_foreign].copy()
    if not carry_rows.empty:
        carry_key = POSS_KEY + ["offense_segment", "poss_offense"]
        carry_counts = (
            carry_rows.groupby(carry_key, as_index=False)
            .agg(
                Points=("carry_points_event", "sum"),
                FGA=("carry_fga_event", "sum"),
                FGM=("carry_fgm_event", "sum"),
                FGMiss=("carry_fg_miss_event", "sum"),
                FTA=("carry_fta_event", "sum"),
                FTM=("carry_ftm_event", "sum"),
                ORB=("carry_orb_event", "sum"),
                TOV=("carry_tov_event", "sum"),
                RimAtt=("carry_rim_att_event", "sum"),
                RimMade=("carry_rim_made_event", "sum"),
                MidAtt=("carry_mid_att_event", "sum"),
                MidMade=("carry_mid_made_event", "sum"),
                ThreeAtt=("carry_three_att_event", "sum"),
                ThreeMade=("carry_three_made_event", "sum"),
            )
        )
        carry_rows = carry_rows.groupby(carry_key, as_index=False).tail(1).copy()
        for idx in range(1, 6):
            carry_rows[f"O{idx}"] = np.where(
                carry_rows["poss_offense"] == "Home",
                carry_rows[f"h{idx}"],
                carry_rows[f"a{idx}"],
            )
            carry_rows[f"D{idx}"] = np.where(
                carry_rows["poss_offense"] == "Home",
                carry_rows[f"a{idx}"],
                carry_rows[f"h{idx}"],
            )

        carry_rows["off_team_abbr"] = np.where(
            carry_rows["poss_offense"] == "Home",
            carry_rows["home_team_abbr"],
            carry_rows["away_team_abbr"],
        )
        carry_rows["def_team_abbr"] = np.where(
            carry_rows["poss_offense"] == "Home",
            carry_rows["away_team_abbr"],
            carry_rows["home_team_abbr"],
        )

        carry_poss = carry_counts.merge(
            carry_rows[
                carry_key
                + ["off_team_abbr", "def_team_abbr"]
                + [f"O{i}" for i in range(1, 6)]
                + [f"D{i}" for i in range(1, 6)]
            ],
            on=carry_key,
            how="inner",
        )
        carry_poss["OffPoss"] = 1
        carry_poss["DefPoss"] = 1
        poss = pd.concat([poss, carry_poss], ignore_index=True, sort=False)

    # Preserve technical/admin FT scoring by the non-terminal side without
    # attributing it to the live-ball possession owner. These rows should
    # affect points/FTA but not possession counts.
    foreign_tech_rows = df.loc[foreign_tech_ft].copy()
    if not foreign_tech_rows.empty:
        for idx in range(1, 6):
            foreign_tech_rows[f"O{idx}"] = np.where(
                foreign_tech_rows["poss_offense"] == "Home",
                foreign_tech_rows[f"h{idx}"],
                foreign_tech_rows[f"a{idx}"],
            )
            foreign_tech_rows[f"D{idx}"] = np.where(
                foreign_tech_rows["poss_offense"] == "Home",
                foreign_tech_rows[f"a{idx}"],
                foreign_tech_rows[f"h{idx}"],
            )

        foreign_tech_rows["off_team_abbr"] = np.where(
            foreign_tech_rows["poss_offense"] == "Home",
            foreign_tech_rows["home_team_abbr"],
            foreign_tech_rows["away_team_abbr"],
        )
        foreign_tech_rows["def_team_abbr"] = np.where(
            foreign_tech_rows["poss_offense"] == "Home",
            foreign_tech_rows["away_team_abbr"],
            foreign_tech_rows["home_team_abbr"],
        )

        admin_poss = foreign_tech_rows[
            POSS_KEY
            + ["poss_offense", "off_team_abbr", "def_team_abbr"]
            + [f"O{i}" for i in range(1, 6)]
            + [f"D{i}" for i in range(1, 6)]
        ].copy()
        admin_poss["Points"] = np.where(
            foreign_tech_rows["poss_offense"].eq("Home"),
            foreign_tech_rows["Home_Action_Score"],
            foreign_tech_rows["Away_Action_Score"],
        ).astype(float)
        admin_poss["FGA"] = 0
        admin_poss["FGM"] = 0
        admin_poss["FGMiss"] = 0
        admin_poss["FTA"] = 1
        admin_poss["FTM"] = (~shot_text.loc[foreign_tech_rows.index].str.contains(r"\bMISS\b", case=False, regex=True, na=False)).astype(int).to_numpy()
        admin_poss["ORB"] = 0
        admin_poss["TOV"] = 0
        admin_poss["RimAtt"] = 0
        admin_poss["RimMade"] = 0
        admin_poss["MidAtt"] = 0
        admin_poss["MidMade"] = 0
        admin_poss["ThreeAtt"] = 0
        admin_poss["ThreeMade"] = 0
        admin_poss["OffPoss"] = 0
        admin_poss["DefPoss"] = 0
        poss = pd.concat([poss, admin_poss], ignore_index=True, sort=False)

    return poss


def derive_lineup_rates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["off_tsa"] = df["off_fga"] + 0.44 * df["off_fta"]
    df["def_opp_tsa"] = df["def_opp_fga"] + 0.44 * df["def_opp_fta"]
    df["ortg"] = pct(df["off_points"], df["off_poss"])
    df["drtg"] = pct(df["def_opp_points"], df["def_poss"])
    df["net"] = df["ortg"] - df["drtg"]
    df["off_ts_pct"] = pct(df["off_points"], 2 * df["off_tsa"])
    df["def_ts_pct_allowed"] = pct(df["def_opp_points"], 2 * df["def_opp_tsa"])
    df["off_fgorb_pct"] = pct(df["off_orb"], df["off_fg_miss"])
    df["def_fgorb_pct_allowed"] = pct(df["def_opp_orb"], df["def_opp_fg_miss"])
    df["off_rim_freq"] = pct(df["off_rim_att"], df["off_fga"])
    df["def_rim_freq_allowed"] = pct(df["def_opp_rim_att"], df["def_opp_fga"])
    df["off_mid_freq"] = pct(df["off_mid_att"], df["off_fga"])
    df["def_mid_freq_allowed"] = pct(df["def_opp_mid_att"], df["def_opp_fga"])
    df["off_three_freq"] = pct(df["off_three_att"], df["off_fga"])
    df["def_three_freq_allowed"] = pct(df["def_opp_three_att"], df["def_opp_fga"])
    df["off_rim_acc"] = pct(df["off_rim_made"], df["off_rim_att"])
    df["def_rim_acc_allowed"] = pct(df["def_opp_rim_made"], df["def_opp_rim_att"])
    df["off_mid_acc"] = pct(df["off_mid_made"], df["off_mid_att"])
    df["def_mid_acc_allowed"] = pct(df["def_opp_mid_made"], df["def_opp_mid_att"])
    df["off_three_acc"] = pct(df["off_three_made"], df["off_three_att"])
    df["def_three_acc_allowed"] = pct(df["def_opp_three_made"], df["def_opp_three_att"])
    return df


def build_lineup_stats(year: int, output_path: Path) -> pd.DataFrame:
    poss = build_possession_summary(year)

    off_canon = canonicalize_frame(poss, [f"O{i}" for i in range(1, 6)])
    off_bundle = pd.concat(
        [
            poss.loc[off_canon.index, ["game_id", "off_team_abbr"]].rename(columns={"off_team_abbr": "team_abbr"}),
            off_canon,
            poss.loc[
                off_canon.index,
                ["OffPoss", "Points", "FGA", "FGM", "FGMiss", "FTA", "FTM", "ORB", "TOV", "RimAtt", "RimMade", "MidAtt", "MidMade", "ThreeAtt", "ThreeMade"],
            ],
        ],
        axis=1,
    )
    off_agg = off_bundle.groupby(["team_abbr", *CANON_COLS], as_index=False).agg(
        off_games=("game_id", "nunique"),
        off_poss=("OffPoss", "sum"),
        off_points=("Points", "sum"),
        off_fga=("FGA", "sum"),
        off_fgm=("FGM", "sum"),
        off_fg_miss=("FGMiss", "sum"),
        off_fta=("FTA", "sum"),
        off_ftm=("FTM", "sum"),
        off_orb=("ORB", "sum"),
        off_tov=("TOV", "sum"),
        off_rim_att=("RimAtt", "sum"),
        off_rim_made=("RimMade", "sum"),
        off_mid_att=("MidAtt", "sum"),
        off_mid_made=("MidMade", "sum"),
        off_three_att=("ThreeAtt", "sum"),
        off_three_made=("ThreeMade", "sum"),
    )

    def_canon = canonicalize_frame(poss, [f"D{i}" for i in range(1, 6)])
    def_bundle = pd.concat(
        [
            poss.loc[def_canon.index, ["game_id", "def_team_abbr"]].rename(columns={"def_team_abbr": "team_abbr"}),
            def_canon,
            poss.loc[
                def_canon.index,
                ["DefPoss", "Points", "FGA", "FGM", "FGMiss", "FTA", "FTM", "ORB", "TOV", "RimAtt", "RimMade", "MidAtt", "MidMade", "ThreeAtt", "ThreeMade"],
            ],
        ],
        axis=1,
    )
    def_agg = def_bundle.groupby(["team_abbr", *CANON_COLS], as_index=False).agg(
        def_games=("game_id", "nunique"),
        def_poss=("DefPoss", "sum"),
        def_opp_points=("Points", "sum"),
        def_opp_fga=("FGA", "sum"),
        def_opp_fgm=("FGM", "sum"),
        def_opp_fg_miss=("FGMiss", "sum"),
        def_opp_fta=("FTA", "sum"),
        def_opp_ftm=("FTM", "sum"),
        def_opp_orb=("ORB", "sum"),
        def_tov_forced=("TOV", "sum"),
        def_opp_rim_att=("RimAtt", "sum"),
        def_opp_rim_made=("RimMade", "sum"),
        def_opp_mid_att=("MidAtt", "sum"),
        def_opp_mid_made=("MidMade", "sum"),
        def_opp_three_att=("ThreeAtt", "sum"),
        def_opp_three_made=("ThreeMade", "sum"),
    )

    merged = off_agg.merge(def_agg, on=["team_abbr", *CANON_COLS], how="outer")
    numeric_cols = [col for col in merged.columns if col not in {"team_abbr", *CANON_COLS}]
    merged[numeric_cols] = merged[numeric_cols].fillna(0)
    merged[CANON_COLS] = merged[CANON_COLS].astype(np.int64)

    name_map = load_name_map(year)
    for idx, player_col in enumerate(CANON_COLS, start=1):
        merged[f"P{idx}_name"] = merged[player_col].map(name_map).fillna(
            merged[player_col].map(lambda player_id: f"ID_{player_id}")
        )

    merged["lineup_key"] = merged[CANON_COLS].astype(str).agg("-".join, axis=1)
    merged["lineup_name"] = merged[[f"P{i}_name" for i in range(1, 6)]].agg(" | ".join, axis=1)
    merged = derive_lineup_rates(merged)

    ordered_cols = [
        "team_abbr",
        "lineup_key",
        *CANON_COLS,
        *[f"P{i}_name" for i in range(1, 6)],
        "lineup_name",
        "off_games",
        "def_games",
        *OFF_RAW_COLS,
        *DEF_RAW_COLS,
        *DERIVED_COLS,
    ]
    merged = merged[ordered_cols].sort_values(["team_abbr", "off_poss", "def_poss"], ascending=[True, False, False])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    return merged


def validate_lineup_stats(df: pd.DataFrame) -> dict[str, int]:
    checks = {
        "off_fg_miss_mismatch": int((df["off_fg_miss"] != (df["off_fga"] - df["off_fgm"])).sum()),
        "def_fg_miss_mismatch": int((df["def_opp_fg_miss"] != (df["def_opp_fga"] - df["def_opp_fgm"])).sum()),
        "off_zone_att_mismatch": int((df["off_rim_att"] + df["off_mid_att"] + df["off_three_att"] != df["off_fga"]).sum()),
        "def_zone_att_mismatch": int((df["def_opp_rim_att"] + df["def_opp_mid_att"] + df["def_opp_three_att"] != df["def_opp_fga"]).sum()),
        "off_zone_make_mismatch": int((df["off_rim_made"] + df["off_mid_made"] + df["off_three_made"] != df["off_fgm"]).sum()),
        "def_zone_make_mismatch": int((df["def_opp_rim_made"] + df["def_opp_mid_made"] + df["def_opp_three_made"] != df["def_opp_fgm"]).sum()),
    }
    return checks


def query_lineups(
    year: int,
    data_path: Path,
    include: str | None,
    exclude: str | None,
    exact: str | None,
    team: str | None,
    min_off_poss: int,
    min_def_poss: int,
    limit: int,
    sort_col: str,
    csv_path: Path | None,
) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Run `python lineup_stats.py build --year {year}` first")

    df = pd.read_parquet(data_path)
    autocomplete = pd.read_csv(AUTOCOMPLETE_MAP) if AUTOCOMPLETE_MAP.exists() else pd.DataFrame(columns=["player_name", "nba_id"])
    include_ids = resolve_player_list(include, autocomplete)
    exclude_ids = resolve_player_list(exclude, autocomplete)
    exact_ids = resolve_player_list(exact, autocomplete)

    mask = (df["off_poss"] >= min_off_poss) & (df["def_poss"] >= min_def_poss)
    if team:
        mask &= df["team_abbr"].eq(team.upper())
    if include_ids:
        include_set = set(include_ids)
        mask &= df[CANON_COLS].isin(include_set).sum(axis=1) == len(include_set)
    if exclude_ids:
        exclude_set = set(exclude_ids)
        mask &= ~df[CANON_COLS].isin(exclude_set).any(axis=1)
    if exact_ids:
        if len(exact_ids) != 5:
            raise ValueError("--exact requires exactly five players")
        exact_key = "-".join(str(pid) for pid in sorted(exact_ids))
        mask &= df["lineup_key"] == exact_key

    results = df.loc[mask].copy()
    if sort_col not in results.columns:
        raise ValueError(f"Unknown sort column '{sort_col}'")
    results = results.sort_values(sort_col, ascending=False).head(limit)
    if csv_path is not None:
        results.to_csv(csv_path, index=False)
    return results


def summarize_split(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [col for col in df.columns if col in OFF_RAW_COLS + DEF_RAW_COLS]
    summary = df[numeric_cols].sum().to_frame().T
    summary = derive_lineup_rates(summary)
    return summary


def player_on_off(year: int, player: str, data_path: Path, team: str | None) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Run `python lineup_stats.py build --year {year}` first")

    autocomplete = pd.read_csv(AUTOCOMPLETE_MAP) if AUTOCOMPLETE_MAP.exists() else pd.DataFrame(columns=["player_name", "nba_id"])
    player_id, player_name = resolve_player_query(player, autocomplete)

    df = pd.read_parquet(data_path)
    player_mask = df[CANON_COLS].eq(player_id).any(axis=1)
    teams = sorted(df.loc[player_mask, "team_abbr"].dropna().unique().tolist())
    if team:
        teams = [team.upper()]
    if not teams:
        raise ValueError(f"No lineup rows found for {player_name} ({player_id}) in {data_path.name}")

    outputs: list[pd.DataFrame] = []
    for team_abbr in teams:
        team_df = df[df["team_abbr"] == team_abbr].copy()
        if team_df.empty:
            continue
        on_df = team_df[team_df[CANON_COLS].eq(player_id).any(axis=1)]
        off_df = team_df[~team_df[CANON_COLS].eq(player_id).any(axis=1)]
        for split_name, split_df in [("on", on_df), ("off", off_df)]:
            summary = summarize_split(split_df)
            summary["team_abbr"] = team_abbr
            summary["split"] = split_name
            summary["player_id"] = player_id
            summary["player_name"] = player_name
            outputs.append(summary)

    if not outputs:
        raise ValueError(f"No on/off rows produced for {player_name} ({player_id})")

    result = pd.concat(outputs, ignore_index=True)
    ordered = ["player_id", "player_name", *PLAYER_DISPLAY_COLS]
    return result[ordered]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and query raw-count lineup WOWY stats.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build lineup stats from raw 2026 PBP.")
    build_parser.add_argument("--year", type=int, default=26, help="Two-digit season year")
    build_parser.add_argument("--output", type=Path, help="Optional output parquet path")
    build_parser.add_argument("--validate", action="store_true", help="Print invariant checks after build")

    query_parser = subparsers.add_parser("query", help="Query built lineup rows.")
    query_parser.add_argument("--year", type=int, default=26)
    query_parser.add_argument("--data", type=Path)
    query_parser.add_argument("--include", help="Comma-separated player IDs or names that must be in lineup")
    query_parser.add_argument("--exclude", help="Comma-separated player IDs or names that must not be in lineup")
    query_parser.add_argument("--exact", help="Comma-separated exact 5-man lineup")
    query_parser.add_argument("--team", help="Optional team abbreviation filter")
    query_parser.add_argument("--min-off-poss", type=int, default=0)
    query_parser.add_argument("--min-def-poss", type=int, default=0)
    query_parser.add_argument("--limit", type=int, default=25)
    query_parser.add_argument("--sort", default="off_poss")
    query_parser.add_argument("--csv", type=Path)
    query_parser.add_argument("--show-names", action="store_true")

    player_parser = subparsers.add_parser("player", help="Compute player on/off from lineup table.")
    player_parser.add_argument("player", help="NBA ID or player name")
    player_parser.add_argument("--year", type=int, default=26)
    player_parser.add_argument("--data", type=Path)
    player_parser.add_argument("--team", help="Optional team abbreviation if player was on multiple teams")
    player_parser.add_argument("--csv", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "build":
            output_path = args.output or build_default_output_path(args.year)
            df = build_lineup_stats(args.year, output_path)
            print(f"Built {len(df):,} lineup rows")
            print(f"Wrote {output_path}")
            if args.validate:
                checks = validate_lineup_stats(df)
                print(checks)
            return 0

        if args.command == "query":
            data_path = args.data or build_default_output_path(args.year)
            results = query_lineups(
                year=args.year,
                data_path=data_path,
                include=args.include,
                exclude=args.exclude,
                exact=args.exact,
                team=args.team,
                min_off_poss=args.min_off_poss,
                min_def_poss=args.min_def_poss,
                limit=args.limit,
                sort_col=args.sort,
                csv_path=args.csv,
            )
            if results.empty:
                print("No matching lineups.")
                return 0
            pd.set_option("display.width", 240)
            pd.set_option("display.max_columns", None)
            display_cols = LINEUP_DISPLAY_COLS + (["lineup_name"] if args.show_names else [])
            print(results[display_cols].to_string(index=False, float_format=lambda value: f"{value:0.2f}"))
            return 0

        data_path = args.data or build_default_output_path(args.year)
        result = player_on_off(args.year, args.player, data_path, args.team)
        if args.csv is not None:
            result.to_csv(args.csv, index=False)
        pd.set_option("display.width", 240)
        pd.set_option("display.max_columns", None)
        print(result.to_string(index=False, float_format=lambda value: f"{value:0.2f}"))
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
