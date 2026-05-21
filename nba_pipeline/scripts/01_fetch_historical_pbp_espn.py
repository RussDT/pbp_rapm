#!/usr/bin/env python3
"""
Fetch historical NBA play-by-play from SportsDataverse / hoopR ESPN season files
and normalize it into this repo's V2-style raw parquet contract.

Initial target is pre-2014 seasons where the NBA Stats API path is unreliable.

Usage:
    python nba_pipeline/scripts/01_fetch_historical_pbp_espn.py 2003
    python nba_pipeline/scripts/01_fetch_historical_pbp_espn.py 2003 PS
    python nba_pipeline/scripts/01_fetch_historical_pbp_espn.py 2003 --limit-games 5
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).parent
PIPELINE_ROOT = SCRIPT_DIR.parent
RAW_DATA_DIR = PIPELINE_ROOT / "raw_data"
CACHE_DIR = PIPELINE_ROOT / "cache" / "historical_espn"
AUTOCOMPLETE_MAP = PIPELINE_ROOT.parent / "autocomplete_map.csv"
UNRESOLVED_ID_OFFSET = 900_000_000

SHOT_TYPE_TO_LABEL = {
    "Jump Shot": "Jump Shot",
    "Layup Shot": "Layup Shot",
    "Dunk Shot": "Dunk Shot",
    "Hook Shot": "Hook Shot",
    "Tip Shot": "Tip Shot",
    "Driving Layup Shot": "Driving Layup Shot",
    "Running Jump Shot": "Running Jump Shot",
    "Turnaround Jump Shot": "Turnaround Jump Shot",
    "Floating Jump Shot": "Floating Jump Shot",
    "Fadeaway Jump Shot": "Fadeaway Jump Shot",
    "Pullup Jump Shot": "Pullup Jump Shot",
    "Step Back Jump Shot": "Step Back Jump Shot",
}


def run_rscript_export(season_end_year: int, pbp_path: Path, box_path: Path) -> None:
    """Materialize hoopR season releases as gzip CSVs so Python can read them."""
    if pbp_path.exists() and box_path.exists():
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    r_script = f"""
    suppressPackageStartupMessages(library(hoopR))
    season <- as.integer({season_end_year})
    pbp_path <- "{pbp_path}"
    box_path <- "{box_path}"

    if (!file.exists(pbp_path)) {{
      pbp <- load_nba_pbp(season)
      con <- gzfile(pbp_path, "w")
      write.csv(pbp, con, row.names = FALSE, na = "")
      close(con)
    }}

    if (!file.exists(box_path)) {{
      box <- load_nba_player_box(season)
      con <- gzfile(box_path, "w")
      write.csv(box, con, row.names = FALSE, na = "")
      close(con)
    }}
    """

    result = subprocess.run(
        ["Rscript", "-e", r_script],
        capture_output=True,
        text=True,
        cwd=PIPELINE_ROOT.parent,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to export historical hoopR data.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def compute_event_seconds_from_start(period: int, secs_remaining: float) -> float:
    total = 0.0
    for p in range(1, period):
        total += 12 * 60.0 if p <= 4 else 5 * 60.0
    period_len = 12 * 60.0 if period <= 4 else 5 * 60.0
    return total + (period_len - secs_remaining)


def mmss_to_seconds(clock_text: str, period: int) -> float:
    if not isinstance(clock_text, str) or ":" not in clock_text:
        return 12 * 60.0 if period <= 4 else 5 * 60.0
    minute_str, second_str = clock_text.split(":", 1)
    try:
        return int(minute_str) * 60.0 + float(second_str)
    except ValueError:
        return 12 * 60.0 if period <= 4 else 5 * 60.0


def season_type_code(short: str) -> int:
    mapping = {"RS": 2, "PS": 3}
    return mapping[short]


def season_suffix(season_end_year: int) -> str:
    return f"{season_end_year % 100:02d}"


def to_int(value) -> Optional[int]:
    if pd.isna(value) or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def normalize_name(name: Optional[str]) -> str:
    if not isinstance(name, str):
        return ""
    return re.sub(r"\s+", " ", name).strip()


def build_name_maps(box_df: pd.DataFrame) -> Tuple[Dict[int, str], Dict[int, Dict[str, object]], Dict[int, Dict[str, object]]]:
    player_name_map: Dict[int, str] = {}
    player_team_map: Dict[int, Dict[str, object]] = {}
    team_meta_map: Dict[int, Dict[str, object]] = {}

    for row in box_df.itertuples(index=False):
        athlete_id = to_int(getattr(row, "athlete_id", None))
        team_id = to_int(getattr(row, "team_id", None))
        if athlete_id is not None:
            player_name_map[athlete_id] = normalize_name(getattr(row, "athlete_display_name", None))
            player_team_map[athlete_id] = {
                "team_id": team_id,
                "team_city": normalize_name(getattr(row, "team_location", None)),
                "team_nickname": normalize_name(getattr(row, "team_name", None)),
                "team_abbreviation": normalize_name(getattr(row, "team_abbreviation", None)),
            }
        if team_id is not None and team_id not in team_meta_map:
            team_meta_map[team_id] = {
                "team_city": normalize_name(getattr(row, "team_location", None)),
                "team_nickname": normalize_name(getattr(row, "team_name", None)),
                "team_abbreviation": normalize_name(getattr(row, "team_abbreviation", None)),
            }
    return player_name_map, player_team_map, team_meta_map


def build_repo_player_id_map(box_df: pd.DataFrame) -> Dict[int, int]:
    """
    Map ESPN athlete IDs onto repo-stable player IDs.

    Exact-name matches use the repo's `autocomplete_map.csv` NBA IDs.
    Unresolved names fall back to a large synthetic namespace to avoid collisions
    with real NBA IDs.
    """
    if not AUTOCOMPLETE_MAP.exists():
        return {
            athlete_id: UNRESOLVED_ID_OFFSET + athlete_id
            for athlete_id in box_df["athlete_id"].dropna().astype(int).unique().tolist()
        }

    ac = pd.read_csv(AUTOCOMPLETE_MAP)
    ac = ac.dropna(subset=["player_name", "nba_id"]).copy()
    ac["normalized_name"] = ac["player_name"].astype(str).map(normalize_name).str.casefold()
    name_counts = ac["normalized_name"].value_counts()
    unique_name_map = {
        row.normalized_name: int(row.nba_id)
        for row in ac.itertuples(index=False)
        if name_counts.get(row.normalized_name, 0) == 1
    }

    athlete_to_repo: Dict[int, int] = {}
    unique_players = (
        box_df[["athlete_id", "athlete_display_name"]]
        .dropna(subset=["athlete_id"])
        .drop_duplicates("athlete_id")
        .copy()
    )
    for row in unique_players.itertuples(index=False):
        athlete_id = int(row.athlete_id)
        normalized_name = normalize_name(getattr(row, "athlete_display_name", None)).casefold()
        nba_id = unique_name_map.get(normalized_name)
        athlete_to_repo[athlete_id] = nba_id if nba_id is not None else UNRESOLVED_ID_OFFSET + athlete_id
    return athlete_to_repo


def apply_repo_player_ids(
    pbp_df: pd.DataFrame,
    box_df: pd.DataFrame,
    athlete_to_repo: Dict[int, int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pbp_df = pbp_df.copy()
    box_df = box_df.copy()

    for col in ["athlete_id", "athlete_id_1", "athlete_id_2", "athlete_id_3"]:
        if col in pbp_df.columns:
            pbp_df[col] = pd.to_numeric(pbp_df[col], errors="coerce").map(athlete_to_repo)
    if "athlete_id" in box_df.columns:
        box_df["athlete_id"] = pd.to_numeric(box_df["athlete_id"], errors="coerce").map(athlete_to_repo)
    return pbp_df, box_df


def build_starter_map(box_df: pd.DataFrame) -> Dict[Tuple[int, str], List[int]]:
    starters: Dict[Tuple[int, str], List[int]] = {}

    for (game_id, home_away), side_df in box_df.groupby(["game_id", "home_away"], sort=False):
        working = side_df.copy()
        for col in ["starter", "did_not_play", "active"]:
            if col in working.columns:
                working[col] = working[col].fillna(False).astype(bool)

        start_df = working[(working["starter"]) & (~working["did_not_play"])]
        if len(start_df) < 5:
            start_df = working[(~working["did_not_play"]) & (working["active"])]
            if "minutes" in start_df.columns:
                start_df = start_df.sort_values("minutes", ascending=False)

        ids = [to_int(v) for v in start_df["athlete_id"].tolist()]
        ids = [v for v in ids if v is not None]
        unique_ids: List[int] = []
        for pid in ids:
            if pid not in unique_ids:
                unique_ids.append(pid)
            if len(unique_ids) == 5:
                break
        starters[(int(game_id), str(home_away).lower())] = unique_ids
    return starters


def sort_lineup(ids: List[int]) -> List[Optional[int]]:
    unique_ids = []
    for pid in ids:
        if pid is not None and pid not in unique_ids:
            unique_ids.append(pid)
    unique_ids = sorted(unique_ids)
    return unique_ids[:5] + [None] * max(0, 5 - len(unique_ids[:5]))


def apply_substitution(lineup: List[Optional[int]], player_in: Optional[int], player_out: Optional[int]) -> List[Optional[int]]:
    current = [pid for pid in lineup if pid is not None]
    if player_out is not None and player_out in current:
        idx = current.index(player_out)
        if player_in is not None:
            current[idx] = player_in
        else:
            current.pop(idx)
    elif player_in is not None and player_in not in current:
        if len(current) < 5:
            current.append(player_in)
        else:
            # Fall back to replacing the last slot if the outgoing player was not found.
            current[-1] = player_in
    return sort_lineup(current)


def game_team_info(game_df: pd.DataFrame) -> Dict[str, Optional[int]]:
    first = game_df.iloc[0]
    return {
        "home_team_id": to_int(first.get("home_team_id")),
        "away_team_id": to_int(first.get("away_team_id")),
    }


def shot_points_from_text(text: str) -> int:
    lower = text.lower()
    if "three point" in lower or "3-pt" in lower or "3pt" in lower:
        return 3
    return 2


def format_assist_tag(text: str, secondary_id: Optional[int], player_name_map: Dict[int, str]) -> str:
    lower = text.lower()
    if "assist" not in lower or secondary_id is None:
        return ""
    assist_name = player_name_map.get(secondary_id)
    if not assist_name:
        return ""
    return f" ({assist_name} 1 AST)"


def format_shot_description(
    row: pd.Series,
    player_name_map: Dict[int, str],
) -> Tuple[Optional[str], Optional[str]]:
    text = normalize_name(row.get("text"))
    lower = text.lower()
    player1_id = to_int(row.get("athlete_id_1"))
    player2_id = to_int(row.get("athlete_id_2"))
    shooter = player_name_map.get(player1_id) or normalize_name(text.split(" made ")[0].split(" missed ")[0])
    shot_label = SHOT_TYPE_TO_LABEL.get(str(row.get("type_text")), "Jump Shot")
    points = shot_points_from_text(text)
    prefix = "3PT " if points == 3 else ""

    defense_desc = None
    if "blocked by" in lower and player2_id is not None:
        blocker = player_name_map.get(player2_id)
        if blocker:
            defense_desc = f"{blocker} BLOCK"

    if " made " in lower:
        offense_desc = f"{shooter} {prefix}{shot_label} ({points} PTS){format_assist_tag(text, player2_id, player_name_map)}"
    else:
        offense_desc = f"MISS {shooter} {prefix}{shot_label}"

    return offense_desc, defense_desc


def format_free_throw_description(row: pd.Series, player_name_map: Dict[int, str]) -> str:
    text = normalize_name(row.get("text"))
    lower = text.lower()
    player1_id = to_int(row.get("athlete_id_1"))
    shooter = player_name_map.get(player1_id) or normalize_name(text.split(" made ")[0].split(" missed ")[0])
    suffix = str(row.get("type_text")).replace("Free Throw - ", "").strip()
    if " made " in lower:
        return f"{shooter} Free Throw {suffix} (1 PTS)"
    return f"MISS {shooter} Free Throw {suffix}"


def format_rebound_description(row: pd.Series, player_name_map: Dict[int, str], team_meta_map: Dict[int, Dict[str, object]]) -> str:
    player1_id = to_int(row.get("athlete_id_1"))
    team_id = to_int(row.get("team_id"))
    if player1_id is not None and player1_id in player_name_map:
        return f"{player_name_map[player1_id]} REBOUND"
    team_abbrev = ""
    if team_id is not None:
        team_abbrev = str(team_meta_map.get(team_id, {}).get("team_abbreviation") or "").strip()
    return f"{team_abbrev} Rebound".strip()


def format_turnover_description(row: pd.Series, player_name_map: Dict[int, str]) -> Tuple[str, Optional[str]]:
    text = normalize_name(row.get("text"))
    type_text = str(row.get("type_text") or "").strip()
    player1_id = to_int(row.get("athlete_id_1"))
    player2_id = to_int(row.get("athlete_id_2"))
    actor = player_name_map.get(player1_id) if player1_id is not None else ""
    stealer_desc = None
    if player2_id is not None and player2_id in player_name_map and "turnover" in type_text.lower():
        if "steal" in text.lower():
            stealer_desc = f"{player_name_map[player2_id]} STEAL"

    if "Offensive Foul Turnover" == type_text:
        return f"{actor or text} Turnover (Offensive Foul)", stealer_desc
    if actor:
        label = type_text.replace("\n", " ").strip()
        return f"{actor} {label}", stealer_desc
    return text, stealer_desc


def format_foul_description(row: pd.Series, player_name_map: Dict[int, str]) -> str:
    type_text = str(row.get("type_text") or "").strip()
    player1_id = to_int(row.get("athlete_id_1"))
    actor = player_name_map.get(player1_id) if player1_id is not None else ""
    short_map = {
        "Shooting Foul": "S.FOUL",
        "Personal Foul": "P.FOUL",
        "Offensive Foul": "Offensive Foul",
        "Loose Ball Foul": "Loose Ball Foul",
        "Technical Foul": "Technical",
        "Away from Play Foul": "Away From Play Foul",
        "Flagrant Foul Type 1": "Flagrant 1",
        "Flagrant Foul Type 2": "Flagrant 2",
        "Clear Path Foul": "Clear Path Foul",
        "Inbound Foul": "Inbound Foul",
    }
    label = short_map.get(type_text, type_text)
    if actor:
        return f"{actor} {label}".strip()
    return label or normalize_name(row.get("text"))


def format_timeout_description(row: pd.Series, event_side: Optional[str], game_info: Dict[str, Optional[int]], team_meta_map: Dict[int, Dict[str, object]]) -> str:
    type_text = str(row.get("type_text") or "").strip()
    team_id = game_info["home_team_id"] if event_side == "home" else game_info["away_team_id"]
    abbrev = ""
    if team_id is not None:
        abbrev = str(team_meta_map.get(team_id, {}).get("team_abbreviation") or "").strip()
    label = type_text or "Timeout"
    return f"{abbrev} Timeout: {label}".strip()


def split_descriptions(
    row: pd.Series,
    player_name_map: Dict[int, str],
    team_meta_map: Dict[int, Dict[str, object]],
    game_info: Dict[str, Optional[int]],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    type_text = str(row.get("type_text") or "").strip()
    team_id = to_int(row.get("team_id"))
    event_side = None
    if team_id is not None:
        if team_id == game_info["home_team_id"]:
            event_side = "home"
        elif team_id == game_info["away_team_id"]:
            event_side = "away"

    home_desc = None
    neutral_desc = None
    visitor_desc = None

    if type_text in {"Start Period", "End Period", "End Game", "Jump Ball"}:
        neutral_desc = normalize_name(row.get("text"))
    elif type_text == "Substitution":
        player_in = player_name_map.get(to_int(row.get("athlete_id_1"))) or "Unknown"
        player_out = player_name_map.get(to_int(row.get("athlete_id_2"))) or "Unknown"
        desc = f"SUB: {player_in} FOR {player_out}"
        if event_side == "home":
            home_desc = desc
        elif event_side == "away":
            visitor_desc = desc
        else:
            neutral_desc = desc
    elif "Free Throw" in type_text:
        desc = format_free_throw_description(row, player_name_map)
        if event_side == "home":
            home_desc = desc
        elif event_side == "away":
            visitor_desc = desc
        else:
            neutral_desc = desc
    elif "Rebound" in type_text:
        desc = format_rebound_description(row, player_name_map, team_meta_map)
        if event_side == "home":
            home_desc = desc
        elif event_side == "away":
            visitor_desc = desc
        else:
            neutral_desc = desc
    elif "Turnover" in type_text:
        offense_desc, defense_desc = format_turnover_description(row, player_name_map)
        if event_side == "home":
            home_desc = offense_desc
            visitor_desc = defense_desc
        elif event_side == "away":
            visitor_desc = offense_desc
            home_desc = defense_desc
        else:
            neutral_desc = offense_desc
    elif "Foul" in type_text:
        desc = format_foul_description(row, player_name_map)
        if event_side == "home":
            home_desc = desc
        elif event_side == "away":
            visitor_desc = desc
        else:
            neutral_desc = desc
    elif "Timeout" in type_text:
        desc = format_timeout_description(row, event_side, game_info, team_meta_map)
        if event_side == "home":
            home_desc = desc
        elif event_side == "away":
            visitor_desc = desc
        else:
            neutral_desc = desc
    elif type_text.endswith("Shot"):
        offense_desc, defense_desc = format_shot_description(row, player_name_map)
        if event_side == "home":
            home_desc = offense_desc
            visitor_desc = defense_desc
        elif event_side == "away":
            visitor_desc = offense_desc
            home_desc = defense_desc
        else:
            neutral_desc = offense_desc
    else:
        neutral_desc = normalize_name(row.get("text"))

    return home_desc, neutral_desc, visitor_desc


def map_event_type_num(type_text: str, text: str) -> int:
    lower_type = type_text.lower()
    lower_text = text.lower()
    if type_text == "Start Period":
        return 12
    if type_text in {"End Period", "End Game"}:
        return 13
    if type_text == "Jump Ball":
        return 10
    if type_text == "Substitution":
        return 8
    if "timeout" in lower_type:
        return 9
    if "turnover" in lower_type:
        return 5
    if "rebound" in lower_type:
        return 4
    if "foul" in lower_type:
        return 6
    if "violation" in lower_type:
        return 7
    if "free throw" in lower_type:
        return 3
    if type_text.endswith("Shot"):
        if " made " in lower_text:
            return 1
        if " missed " in lower_text:
            return 2
    if "ejection" in lower_type:
        return 11
    return 14


def build_player_fields(player_id: Optional[int], player_name_map: Dict[int, str], player_team_map: Dict[int, Dict[str, object]]) -> Dict[str, object]:
    if player_id is None:
        return {
            "id": pd.NA,
            "name": None,
            "team_id": pd.NA,
            "team_city": None,
            "team_nickname": None,
            "team_abbreviation": None,
        }

    team_meta = player_team_map.get(player_id, {})
    return {
        "id": player_id,
        "name": player_name_map.get(player_id),
        "team_id": team_meta.get("team_id", pd.NA),
        "team_city": team_meta.get("team_city"),
        "team_nickname": team_meta.get("team_nickname"),
        "team_abbreviation": team_meta.get("team_abbreviation"),
    }


def build_score_string(home_score: Optional[int], away_score: Optional[int]) -> Optional[str]:
    if home_score is None or away_score is None:
        return None
    return f"{home_score} - {away_score}"


def build_output_rows(
    pbp_df: pd.DataFrame,
    starter_map: Dict[Tuple[int, str], List[int]],
    player_name_map: Dict[int, str],
    player_team_map: Dict[int, Dict[str, object]],
    team_meta_map: Dict[int, Dict[str, object]],
) -> pd.DataFrame:
    output_rows: List[Dict[str, object]] = []

    numeric_sort_cols = {
        "game_play_number": pd.to_numeric(pbp_df["game_play_number"], errors="coerce").fillna(0).astype(int),
        "sequence_number_num": pd.to_numeric(pbp_df["sequence_number"], errors="coerce").fillna(0).astype(int),
    }
    pbp_df = pbp_df.assign(**numeric_sort_cols)
    pbp_df = pbp_df.sort_values(["game_id", "period", "game_play_number", "sequence_number_num"], kind="stable")

    for game_id, game_df in pbp_df.groupby("game_id", sort=False):
        game_info = game_team_info(game_df)
        home_active = sort_lineup(starter_map.get((int(game_id), "home"), []))
        away_active = sort_lineup(starter_map.get((int(game_id), "away"), []))

        if all(pid is None for pid in home_active) or all(pid is None for pid in away_active):
            print(f"Skipping game {game_id}: missing starter seeds")
            continue

        for row in game_df.itertuples(index=False):
            row_dict = row._asdict()
            type_text = str(row_dict.get("type_text") or "")
            if type_text == "Substitution":
                player_in = to_int(row_dict.get("athlete_id_1"))
                player_out = to_int(row_dict.get("athlete_id_2"))
                team_id = to_int(row_dict.get("team_id"))
                if team_id == game_info["home_team_id"]:
                    home_active = apply_substitution(home_active, player_in, player_out)
                elif team_id == game_info["away_team_id"]:
                    away_active = apply_substitution(away_active, player_in, player_out)

            period = int(row_dict.get("period") or row_dict.get("period_number") or 0)
            time_quarter = str(row_dict.get("clock_display_value") or "")
            secs_remaining_q = mmss_to_seconds(time_quarter, period)
            event_secs = compute_event_seconds_from_start(period, secs_remaining_q)
            home_score = to_int(row_dict.get("home_score"))
            away_score = to_int(row_dict.get("away_score"))

            home_desc, neutral_desc, visitor_desc = split_descriptions(
                pd.Series(row_dict), player_name_map, team_meta_map, game_info
            )
            player1 = build_player_fields(to_int(row_dict.get("athlete_id_1")), player_name_map, player_team_map)
            player2 = build_player_fields(to_int(row_dict.get("athlete_id_2")), player_name_map, player_team_map)
            player3 = build_player_fields(to_int(row_dict.get("athlete_id_3")), player_name_map, player_team_map)

            output_row: Dict[str, object] = {
                "game_id": int(game_id),
                "event_num": int(row_dict.get("game_play_number") or row_dict.get("sequence_number_num") or 0),
                "event_type": map_event_type_num(type_text, str(row_dict.get("text") or "")),
                "event_action_type": 0,
                "period": period,
                "minute_game": event_secs / 60.0,
                "time_remaining": np.nan,
                "wc_time_string": None,
                "time_quarter": time_quarter,
                "minute_remaining_quarter": int(secs_remaining_q // 60),
                "seconds_remaining_quarter": int(secs_remaining_q % 60),
                "home_description": home_desc,
                "neutral_description": neutral_desc,
                "visitor_description": visitor_desc,
                "score": build_score_string(home_score, away_score),
                "away_score": away_score,
                "home_score": home_score,
                "score_margin": home_score - away_score if home_score is not None and away_score is not None else None,
                "person1type": 0,
                "player1_id": player1["id"],
                "player1_name": player1["name"],
                "player1_team_id": player1["team_id"],
                "player1_team_city": player1["team_city"],
                "player1_team_nickname": player1["team_nickname"],
                "player1_team_abbreviation": player1["team_abbreviation"],
                "person2type": 0,
                "player2_id": player2["id"],
                "player2_name": player2["name"],
                "player2_team_id": player2["team_id"],
                "player2_team_city": player2["team_city"],
                "player2_team_nickname": player2["team_nickname"],
                "player2_team_abbreviation": player2["team_abbreviation"],
                "person3type": 0,
                "player3_id": player3["id"],
                "player3_name": player3["name"],
                "player3_team_id": player3["team_id"],
                "player3_team_city": player3["team_city"],
                "player3_team_nickname": player3["team_nickname"],
                "player3_team_abbreviation": player3["team_abbreviation"],
                "video_available_flag": 0,
                "team_leading": None,
            }

            if home_score is not None and away_score is not None:
                if home_score > away_score:
                    output_row["team_leading"] = "Home"
                elif away_score > home_score:
                    output_row["team_leading"] = "Away"
                else:
                    output_row["team_leading"] = "TIE"

            for idx, pid in enumerate(away_active, start=1):
                output_row[f"away_player{idx}"] = pid
            for idx, pid in enumerate(home_active, start=1):
                output_row[f"home_player{idx}"] = pid

            output_rows.append(output_row)

    out = pd.DataFrame(output_rows)
    if out.empty:
        return out

    out["time_remaining"] = (
        out.groupby("game_id")["minute_game"].transform("max") - out["minute_game"]
    )

    int_cols = [
        "event_num", "event_type", "event_action_type", "period",
        "minute_remaining_quarter", "seconds_remaining_quarter",
        "away_score", "home_score",
        "player1_id", "player1_team_id", "player2_id", "player2_team_id",
        "player3_id", "player3_team_id",
        "away_player1", "away_player2", "away_player3", "away_player4", "away_player5",
        "home_player1", "home_player2", "home_player3", "home_player4", "home_player5",
    ]
    for col in int_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    out = out.sort_values(["game_id", "event_num"], kind="stable").reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch historical ESPN NBA play-by-play into repo raw parquet format.")
    parser.add_argument("year", type=int, help="4-digit season end year, e.g. 2003 for 2002-03.")
    parser.add_argument("season_type_short", nargs="?", choices=["RS", "PS"], default="RS")
    parser.add_argument("--limit-games", type=int, default=0, help="Optional game limit for trial runs.")
    args = parser.parse_args()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    season_end_year = int(args.year)
    if season_end_year < 2002:
        raise ValueError("This ESPN historical loader only supports season end years >= 2002.")

    pbp_cache = CACHE_DIR / f"espn_pbp_{season_end_year}.csv.gz"
    box_cache = CACHE_DIR / f"espn_player_box_{season_end_year}.csv.gz"
    run_rscript_export(season_end_year, pbp_cache, box_cache)

    print(f"Loading cached ESPN season data for {season_end_year}...")
    pbp_df = pd.read_csv(pbp_cache, low_memory=False)
    box_df = pd.read_csv(box_cache, low_memory=False)

    target_code = season_type_code(args.season_type_short)
    pbp_df = pbp_df[pd.to_numeric(pbp_df["season_type"], errors="coerce") == target_code].copy()
    box_df = box_df[pd.to_numeric(box_df["season_type"], errors="coerce") == target_code].copy()

    if args.limit_games > 0:
        keep_games = pbp_df["game_id"].drop_duplicates().head(args.limit_games).tolist()
        pbp_df = pbp_df[pbp_df["game_id"].isin(keep_games)].copy()
        box_df = box_df[box_df["game_id"].isin(keep_games)].copy()

    athlete_to_repo = build_repo_player_id_map(box_df)
    pbp_df, box_df = apply_repo_player_ids(pbp_df, box_df, athlete_to_repo)

    print(
        f"Historical ESPN season {season_end_year} {args.season_type_short}: "
        f"{pbp_df['game_id'].nunique()} games, {len(pbp_df)} PBP rows"
    )

    player_name_map, player_team_map, team_meta_map = build_name_maps(box_df)
    starter_map = build_starter_map(box_df)
    out = build_output_rows(pbp_df, starter_map, player_name_map, player_team_map, team_meta_map)

    if out.empty:
        raise RuntimeError("No output rows were generated.")

    suffix = season_suffix(season_end_year)
    ps_suffix = "_PS" if args.season_type_short == "PS" else ""
    out_path = RAW_DATA_DIR / f"NBA{suffix}{ps_suffix}.parquet"
    out.to_parquet(out_path, index=False)
    print(f"Saved historical raw parquet to {out_path} ({len(out)} rows)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
