#!/usr/bin/env python3
"""
Build pre-2014 season raw and processed parquets from Gabriel's merged PBP files.

Source:
  https://github.com/gabriel1200/merged_playbyplay/tree/master/old_data

The source is team-split parquet files named {TEAM}_{YEAR}_{rs|ps}.parquet.
This script downloads/checks out those files, merges all team files into one
season file, converts the friend/Gabriel schema into this repo's raw PBP
contract, and optionally processes RAPM/TOV/REB/TS parquets.

By default this targets 1997-2013 inclusive and overwrites the corresponding
NBA{YY}.parquet / NBA{YY}_PS.parquet outputs.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_ROOT.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from process_rapm_blocks import (  # noqa: E402
    process_assist_points_py,
    process_block_recovery_py,
    process_first_chance_clean_py,
    process_first_chance_py,
    process_ft_premium_py,
    process_midrange_fg_pct_py,
    process_midrange_freq_py,
    process_rapm_py,
    process_reb_py,
    process_dunk_assist_py,
    process_dunk_py,
    process_rim_assist_py,
    process_rim_fg_pct_py,
    process_rim_freq_py,
    process_second_chance_clean_py,
    process_second_chance_py,
    process_three_fg_pct_py,
    process_three_freq_py,
    process_tov_py,
    process_ts_py,
)


DEFAULT_REPO_URL = "https://github.com/gabriel1200/merged_playbyplay.git"
DEFAULT_SOURCE_DIR = PIPELINE_ROOT / "external" / "merged_playbyplay"
DEFAULT_RAW_DIR = PIPELINE_ROOT / "raw_data"
DEFAULT_PROCESSED_DIR = PIPELINE_ROOT / "processed"
DEFAULT_REPORT = PIPELINE_ROOT / "validation" / "gabriel_old_pbp" / "build_report.json"

DEFAULT_YEARS = list(range(1997, 2014))
SOURCE_TYPES = ("rs", "ps")
PROCESSORS = {
    "RAPM": lambda path, year, season: process_rapm_py(
        path,
        year,
        season,
        o_luck=0.0,
        d_luck=0.0,
        missing_ft_fallback="actual",
    )[0],
    "LA_RAPM": lambda path, year, season: process_rapm_py(
        path,
        year,
        season,
        o_luck=1.0,
        d_luck=1.0,
        missing_ft_fallback="actual",
    )[1],
    "TOV": lambda path, year, season: process_tov_py(path, year, season),
    "REB": lambda path, year, season: process_reb_py(path, year, season),
    "BLOCK_RECOVERY": lambda path, year, season: process_block_recovery_py(path, year, season),
    "TS": lambda path, year, season: process_ts_py(path, year, season, missing_ft_fallback="actual"),
    "RIM_FREQ": lambda path, year, season: process_rim_freq_py(path, year, season),
    "RIM_FG_PCT": lambda path, year, season: process_rim_fg_pct_py(path, year, season),
    "ASSIST_POINTS": lambda path, year, season: process_assist_points_py(path, year, season),
    "RIM_ASSIST": lambda path, year, season: process_rim_assist_py(path, year, season),
    "DUNK": lambda path, year, season: process_dunk_py(path, year, season),
    "DUNK_ASSIST": lambda path, year, season: process_dunk_assist_py(path, year, season),
    "THREE_FREQ": lambda path, year, season: process_three_freq_py(path, year, season),
    "THREE_FG_PCT": lambda path, year, season: process_three_fg_pct_py(path, year, season),
    "MIDRANGE_FREQ": lambda path, year, season: process_midrange_freq_py(path, year, season),
    "MIDRANGE_FG_PCT": lambda path, year, season: process_midrange_fg_pct_py(path, year, season),
    "FT_PREMIUM": lambda path, year, season: process_ft_premium_py(path, year, season),
    "SECOND_CHANCE": lambda path, year, season: process_second_chance_py(path, year, season),
    "FIRST_CHANCE": lambda path, year, season: process_first_chance_py(path, year, season),
    "FIRST_CHANCE_CLEAN": lambda path, year, season: process_first_chance_clean_py(path, year, season),
    "SECOND_CHANCE_CLEAN": lambda path, year, season: process_second_chance_clean_py(path, year, season),
}


def run_cmd(cmd: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=True)
    return result.stdout


def season_suffix(year: int) -> str:
    return f"{int(year) % 100:02d}"


def season_label(year: int) -> str:
    return f"{int(year) - 1}-{int(year) % 100:02d}"


def raw_output_path(raw_dir: Path, year: int, season_type: str) -> Path:
    suffix = season_suffix(year)
    return raw_dir / (f"NBA{suffix}_PS.parquet" if season_type == "ps" else f"NBA{suffix}.parquet")


def processed_output_path(processed_dir: Path, metric: str, year: int, season_type: str) -> Path:
    suffix = season_suffix(year)
    return processed_dir / (f"{metric}{suffix}_PS.parquet" if season_type == "ps" else f"{metric}{suffix}.parquet")


def parse_years(value: str | None) -> list[int]:
    if not value:
        return DEFAULT_YEARS
    years: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            years.update(range(int(start), int(end) + 1))
        else:
            years.add(int(part))
    return sorted(years)


def parse_season_types(value: str) -> list[str]:
    lowered = value.lower()
    if lowered == "all":
        return list(SOURCE_TYPES)
    out = [part.strip().lower() for part in lowered.split(",") if part.strip()]
    bad = sorted(set(out) - set(SOURCE_TYPES))
    if bad:
        raise ValueError(f"Unsupported season types: {bad}. Use rs, ps, or all.")
    return out


def normalize_player_id(value) -> float:
    if pd.isna(value):
        return np.nan
    try:
        player_id = int(float(value))
    except (TypeError, ValueError):
        return np.nan
    return float(player_id) if player_id > 0 else np.nan


def parse_player_pipe(value) -> list[float]:
    if pd.isna(value):
        return []
    players: list[float] = []
    for part in str(value).split("|"):
        player_id = normalize_player_id(part.strip())
        if pd.notna(player_id):
            players.append(float(player_id))
    return players


def parse_clock_seconds(clock_display) -> int:
    if pd.isna(clock_display):
        return 0
    match = re.search(r"(\d+):(\d+)", str(clock_display))
    if not match:
        return 0
    return int(match.group(1)) * 60 + int(match.group(2))


def normalize_game_id(value) -> int:
    return int(float(value))


def ensure_source_repo(source_dir: Path, repo_url: str) -> None:
    source_dir.parent.mkdir(parents=True, exist_ok=True)
    if (source_dir / ".git").exists():
        run_cmd(["git", "fetch", "--depth=1", "origin", "master"], cwd=source_dir)
        run_cmd(["git", "update-ref", "refs/heads/master", "FETCH_HEAD"], cwd=source_dir)
        return
    run_cmd(["git", "clone", "--filter=blob:none", "--no-checkout", repo_url, str(source_dir)])


def list_old_data_paths(source_dir: Path) -> list[str]:
    output = run_cmd(["git", "ls-tree", "-r", "--name-only", "HEAD", "old_data"], cwd=source_dir)
    return [line.strip() for line in output.splitlines() if line.strip().endswith(".parquet")]


def wanted_source_paths(all_paths: list[str], years: Iterable[int], season_types: Iterable[str]) -> list[str]:
    years_set = {int(year) for year in years}
    types_set = set(season_types)
    wanted: list[str] = []
    pattern = re.compile(r"old_data/[A-Z]{3}_(\d{4})_(rs|ps)\.parquet$")
    for path in all_paths:
        match = pattern.match(path)
        if not match:
            continue
        year = int(match.group(1))
        season_type = match.group(2)
        if year in years_set and season_type in types_set:
            wanted.append(path)
    return sorted(wanted)


def checkout_paths(source_dir: Path, paths: list[str]) -> None:
    missing = [path for path in paths if not (source_dir / path).exists()]
    if not missing:
        return
    chunk_size = 100
    for start in range(0, len(missing), chunk_size):
        chunk = missing[start : start + chunk_size]
        print(
            f"Checking out source parquet files {start + 1}-{start + len(chunk)} "
            f"of {len(missing)}",
            flush=True,
        )
        run_cmd(["git", "checkout", "HEAD", "--", *chunk], cwd=source_dir)


def file_paths_for_year(source_dir: Path, year: int, season_type: str) -> list[Path]:
    return sorted((source_dir / "old_data").glob(f"*_{int(year)}_{season_type}.parquet"))


def dedupe_team_split_rows(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = [
        col
        for col in [
            "game_id",
            "period",
            "clock_display",
            "actionNumber",
            "actionType",
            "description",
            "person_id",
            "assister_id",
            "team",
            "scoreHome",
            "scoreAway",
        ]
        if col in df.columns
    ]
    if not key_cols:
        return df.drop_duplicates().copy()
    return df.drop_duplicates(subset=key_cols, keep="first").copy()


def event_type_from_source(row: pd.Series) -> int:
    action = str(row.get("actionType", "")).lower()
    if action in {"2pt", "3pt"}:
        return 1 if str(row.get("shotResult", "")).lower() == "made" else 2
    if action == "freethrow":
        return 3
    if action == "rebound":
        return 4
    if action == "turnover":
        return 5
    if action == "foul":
        return 6
    if action == "substitution":
        return 8
    if action == "timeout":
        return 9
    if action == "jumpball":
        return 10
    if action == "ejection":
        return 11
    if action == "violation":
        return 7
    if action == "period":
        desc = str(row.get("description", "")).lower()
        return 13 if "end of" in desc else 12
    return 14


def event_action_type_from_source(row: pd.Series) -> int:
    desc = str(row.get("description", "")).lower()
    action = str(row.get("actionType", "")).lower()
    if action == "3pt":
        return 1
    if "dunk" in desc:
        return 7
    if "layup" in desc:
        if "driving" in desc:
            return 6
        if "running" in desc:
            return 41
        return 5
    if "tip" in desc:
        return 97
    return 0


def normalize_description(row: pd.Series) -> str:
    desc = "" if pd.isna(row.get("description")) else str(row.get("description"))
    action = str(row.get("actionType", "")).lower()
    if action in {"2pt", "3pt"} and str(row.get("shotResult", "")).lower() == "made" and "PTS" not in desc:
        points = "3" if action == "3pt" else "2"
        desc = f"{desc} ({points} PTS)"
    return desc


def score_margin_text(home_score, away_score):
    if pd.isna(home_score) or pd.isna(away_score):
        return np.nan
    margin = int(home_score) - int(away_score)
    return "TIE" if margin == 0 else str(margin)


def remove_auxiliary_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df["actionType"].astype(str).str.lower().isin({"block", "steal"})].copy()
    action_sets = df.groupby(["game_id", "actionNumber"])["actionType"].transform(
        lambda values: "|".join(sorted(set(str(value).lower() for value in values)))
    )
    real_action_pattern = "|".join(["2pt", "3pt", "freethrow", "rebound", "turnover", "foul"])
    duplicate_action_sub = (
        df["actionType"].astype(str).str.lower().eq("substitution")
        & action_sets.str.contains(real_action_pattern, regex=True)
    )
    return df[~duplicate_action_sub].copy()


def fill_missing_team_from_team_id(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing source team abbreviations from stable teamId mappings."""
    if "teamId" not in df.columns or "team" not in df.columns:
        return df

    out = df.copy()
    team_id = pd.to_numeric(out["teamId"], errors="coerce")
    team_text = out["team"].replace("", np.nan)
    known = team_id.notna() & team_id.ne(0) & team_text.notna()
    if not known.any():
        return out

    team_map = (
        pd.DataFrame({"teamId": team_id[known].astype("int64"), "team": team_text[known].astype(str)})
        .drop_duplicates()
        .groupby("teamId")["team"]
        .agg(lambda values: values.mode().iat[0])
        .to_dict()
    )
    missing = team_text.isna() & team_id.notna() & team_id.ne(0)
    out.loc[missing, "team"] = team_id[missing].astype("int64").map(team_map)
    return out


def known_lineup_tuple(row: pd.Series, cols: list[str]) -> tuple[int, ...]:
    players: list[int] = []
    for col in cols:
        player_id = normalize_player_id(row.get(col))
        if pd.notna(player_id) and int(player_id) > 0 and int(player_id) not in players:
            players.append(int(player_id))
    return tuple(players)


def is_complete_lineup(lineup: tuple[int, ...]) -> bool:
    return len(lineup) == 5 and len(set(lineup)) == 5 and all(player_id > 0 for player_id in lineup)


def lineup_contains_known(candidate: tuple[int, ...] | None, known: tuple[int, ...]) -> bool:
    if candidate is None or not known:
        return False
    known_set = set(known)
    candidate_set = set(candidate)
    return known_set.issubset(candidate_set) and len(candidate_set - known_set) <= 3


def repair_partial_lineups_from_stable_neighbors(game_rows: list[dict]) -> tuple[list[dict], dict]:
    """
    Repair transient partial lineup snapshots in Gabriel old-data conversion.

    The source applies same-clock substitution clusters row by row, so intermediate
    SUB out / SUB in rows can briefly show 3-4 players. Normal events after an
    incomplete substitution cluster can also carry a missing slot until the next
    complete row. Treat those partial snapshots as bookkeeping states and fill
    them from the nearest compatible complete lineup in the same period.
    """
    if not game_rows:
        return game_rows, {"home_repaired_rows": 0, "away_repaired_rows": 0}

    repaired_rows = [dict(row) for row in game_rows]
    report: dict[str, int] = {
        "home_partial_rows_before": 0,
        "home_repaired_rows": 0,
        "home_partial_rows_after": 0,
        "away_partial_rows_before": 0,
        "away_repaired_rows": 0,
        "away_partial_rows_after": 0,
    }

    for side in ("home", "away"):
        player_cols = [f"{side}_player{i}" for i in range(1, 6)]
        desc_col = f"{side}_description"
        report_key = f"{side}_repaired_rows"

        period_to_indices: dict[int, list[int]] = {}
        for idx, row in enumerate(repaired_rows):
            period_to_indices.setdefault(int(row["period"]), []).append(idx)

        for indices in period_to_indices.values():
            lineups = [known_lineup_tuple(repaired_rows[idx], player_cols) for idx in indices]
            complete = [is_complete_lineup(lineup) for lineup in lineups]
            partial_positions = [
                pos for pos, lineup in enumerate(lineups) if 0 < len(lineup) < 5 and not complete[pos]
            ]
            report[f"{side}_partial_rows_before"] += len(partial_positions)
            if not partial_positions:
                continue

            prev_complete: list[tuple[int, ...] | None] = []
            prev_complete_idx: list[int | None] = []
            current_complete: tuple[int, ...] | None = None
            current_complete_idx: int | None = None
            for pos, idx in enumerate(indices):
                if complete[pos]:
                    current_complete = lineups[pos]
                    current_complete_idx = idx
                prev_complete.append(current_complete)
                prev_complete_idx.append(current_complete_idx)

            next_complete: list[tuple[int, ...] | None] = [None] * len(indices)
            next_complete_idx: list[int | None] = [None] * len(indices)
            current_complete = None
            current_complete_idx = None
            for pos in range(len(indices) - 1, -1, -1):
                idx = indices[pos]
                if complete[pos]:
                    current_complete = lineups[pos]
                    current_complete_idx = idx
                next_complete[pos] = current_complete
                next_complete_idx[pos] = current_complete_idx

            last_side_sub_idx: list[int | None] = []
            current_sub_idx: int | None = None
            for idx in indices:
                row = repaired_rows[idx]
                has_side_sub = (
                    int(row.get("event_type", 0)) == 8
                    and pd.notna(row.get(desc_col))
                    and str(row.get(desc_col) or "").strip() != ""
                )
                if has_side_sub:
                    current_sub_idx = idx
                last_side_sub_idx.append(current_sub_idx)

            for pos in partial_positions:
                idx = indices[pos]
                known = lineups[pos]
                prev_candidate = prev_complete[pos]
                next_candidate = next_complete[pos]
                prev_ok = lineup_contains_known(prev_candidate, known)
                next_ok = lineup_contains_known(next_candidate, known)
                chosen = None

                if prev_ok and next_ok:
                    if prev_candidate == next_candidate:
                        chosen = prev_candidate
                    elif (
                        last_side_sub_idx[pos] is not None
                        and prev_complete_idx[pos] is not None
                        and last_side_sub_idx[pos] >= prev_complete_idx[pos]
                    ):
                        chosen = next_candidate
                    else:
                        prev_distance = abs(idx - prev_complete_idx[pos]) if prev_complete_idx[pos] is not None else 10**9
                        next_distance = abs(next_complete_idx[pos] - idx) if next_complete_idx[pos] is not None else 10**9
                        chosen = prev_candidate if prev_distance <= next_distance else next_candidate
                elif next_ok:
                    chosen = next_candidate
                elif prev_ok:
                    chosen = prev_candidate

                if chosen is None:
                    continue

                for offset, player_id in enumerate(chosen, start=1):
                    repaired_rows[idx][f"{side}_player{offset}"] = float(player_id)
                report[report_key] += 1

        partial_after = 0
        for row in repaired_rows:
            known_count = len(known_lineup_tuple(row, player_cols))
            if 0 < known_count < 5:
                partial_after += 1
        report[f"{side}_partial_rows_after"] = partial_after

    return repaired_rows, report


def sort_source_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_clock_seconds"] = out["clock_display"].map(parse_clock_seconds)
    out["_action_number"] = pd.to_numeric(out["actionNumber"], errors="coerce").fillna(0)
    out = out.sort_values(
        ["game_id", "period", "_clock_seconds", "_action_number"],
        ascending=[True, True, False, True],
        kind="stable",
    )
    return out


def infer_game_sides(game: pd.DataFrame) -> tuple[str | None, str | None]:
    game = sort_source_rows(game)
    home_votes: list[str] = []
    away_votes: list[str] = []
    prev_home = None
    prev_away = None
    for _, row in game.iterrows():
        team = row.get("team")
        if pd.isna(team):
            continue
        home_score = pd.to_numeric(row.get("scoreHome"), errors="coerce")
        away_score = pd.to_numeric(row.get("scoreAway"), errors="coerce")
        if pd.isna(home_score) or pd.isna(away_score):
            continue
        if prev_home is None:
            prev_home = home_score
            prev_away = away_score
            continue
        home_delta = home_score - prev_home
        away_delta = away_score - prev_away
        if home_delta > 0 and home_delta >= away_delta:
            home_votes.append(str(team))
        elif away_delta > 0:
            away_votes.append(str(team))
        prev_home = home_score
        prev_away = away_score

    home_team = pd.Series(home_votes).mode().iat[0] if home_votes else None
    away_team = pd.Series(away_votes).mode().iat[0] if away_votes else None
    if home_team is None or away_team is None or home_team == away_team:
        teams = [str(team) for team in game["team"].dropna().unique().tolist()]
        if home_team is None and teams:
            home_team = teams[0]
        if away_team is None:
            away_candidates = [team for team in teams if team != home_team]
            away_team = away_candidates[0] if away_candidates else None
    return home_team, away_team


def build_side_player_sets(game: pd.DataFrame, home_team: str | None, away_team: str | None) -> dict[str, set[float]]:
    side_sets = {"home": set(), "away": set()}
    offensive_actions = {"2pt", "3pt", "freethrow", "turnover"}
    if home_team is None or away_team is None:
        return side_sets
    for _, row in game.iterrows():
        action = str(row.get("actionType", "")).lower()
        team = None if pd.isna(row.get("team")) else str(row.get("team"))
        if action not in offensive_actions or team not in {home_team, away_team}:
            continue
        off = set(parse_player_pipe(row.get("off_players_on")))
        deff = set(parse_player_pipe(row.get("def_players_on")))
        if team == home_team:
            side_sets["home"].update(off)
            side_sets["away"].update(deff)
        else:
            side_sets["away"].update(off)
            side_sets["home"].update(deff)
    return side_sets


def side_for_team(team: str | None, home_team: str | None, away_team: str | None) -> str | None:
    if team == home_team:
        return "home"
    if team == away_team:
        return "away"
    return None


def opposite_side(side: str) -> str:
    return "away" if side == "home" else "home"


def add_side_vote(votes: dict[int, dict[str, int]], player_id, side: str | None, weight: int = 1) -> None:
    normalized = normalize_player_id(player_id)
    if pd.isna(normalized) or normalized <= 0 or side not in {"home", "away"}:
        return
    player_votes = votes.setdefault(int(normalized), {"home": 0, "away": 0})
    player_votes[side] += int(weight)


def build_game_player_side_map(
    game: pd.DataFrame,
    home_team: str | None,
    away_team: str | None,
) -> tuple[dict[int, str], dict[str, int]]:
    """
    Infer player side within a single game from event actor/team metadata.

    This intentionally avoids season-level team membership because players can
    be traded. Lineup snapshots can be corrupted, so actor rows carry the first
    vote. Scorer/rebounder/fouler/substitution `person_id` belongs to row team;
    assister belongs to the scoring team; blocker/stealer/foul-drawn player
    belongs to the opposite side for the row action.
    """
    votes: dict[int, dict[str, int]] = {}
    for _, row in game.iterrows():
        team = None if pd.isna(row.get("team")) else str(row.get("team"))
        action_side = side_for_team(team, home_team, away_team)
        if action_side is None:
            continue
        defense_side = opposite_side(action_side)
        add_side_vote(votes, row.get("person_id"), action_side, weight=3)
        add_side_vote(votes, row.get("assister_id"), action_side, weight=3)
        add_side_vote(votes, row.get("blockPersonId"), defense_side, weight=3)
        add_side_vote(votes, row.get("stealPersonId"), defense_side, weight=3)
        add_side_vote(votes, row.get("foulDrawnPersonId"), defense_side, weight=2)

    side_map: dict[int, str] = {}
    ambiguous = 0
    for player_id, player_votes in votes.items():
        home_votes = player_votes.get("home", 0)
        away_votes = player_votes.get("away", 0)
        if home_votes == away_votes:
            ambiguous += 1
            continue
        side_map[player_id] = "home" if home_votes > away_votes else "away"
    return side_map, {
        "side_map_players": len(side_map),
        "side_map_ambiguous_players": ambiguous,
    }


def filter_lineup_to_side(lineup: list[float], side: str, player_side_map: dict[int, str]) -> tuple[list[float], int]:
    kept: list[float] = []
    removed = 0
    for player_id in lineup:
        normalized = normalize_player_id(player_id)
        if pd.isna(normalized) or normalized <= 0:
            continue
        known_side = player_side_map.get(int(normalized))
        if known_side is not None and known_side != side:
            removed += 1
            continue
        if int(normalized) in {int(player_id) for player_id in kept}:
            removed += 1
            continue
        kept.append(float(normalized))
    return kept[:5], removed


def assign_home_away_lineup(
    row: pd.Series,
    side_sets: dict[str, set[float]],
    player_side_map: dict[int, str],
) -> tuple[list[float], list[float], dict[str, int]]:
    off = parse_player_pipe(row.get("off_players_on"))
    deff = parse_player_pipe(row.get("def_players_on"))
    home_set = side_sets.get("home", set())
    away_set = side_sets.get("away", set())
    off_home_overlap = len(set(off) & home_set)
    off_away_overlap = len(set(off) & away_set)
    def_home_overlap = len(set(deff) & home_set)
    if off_home_overlap > off_away_overlap or off_home_overlap >= def_home_overlap:
        home_players, away_players = off[:5], deff[:5]
    else:
        home_players, away_players = deff[:5], off[:5]
    home_players, home_removed = filter_lineup_to_side(home_players, "home", player_side_map)
    away_players, away_removed = filter_lineup_to_side(away_players, "away", player_side_map)
    return home_players, away_players, {
        "home_opposite_side_players_removed": home_removed,
        "away_opposite_side_players_removed": away_removed,
    }


def row_side_overlap(row: dict) -> set[int]:
    home = set(known_lineup_tuple(pd.Series(row), [f"home_player{i}" for i in range(1, 6)]))
    away = set(known_lineup_tuple(pd.Series(row), [f"away_player{i}" for i in range(1, 6)]))
    return home & away


def build_lineup_player_side_map(game_rows: list[dict]) -> tuple[dict[int, str], dict[str, int]]:
    votes: dict[int, dict[str, int]] = {}
    for row in game_rows:
        home = set(known_lineup_tuple(pd.Series(row), [f"home_player{i}" for i in range(1, 6)]))
        away = set(known_lineup_tuple(pd.Series(row), [f"away_player{i}" for i in range(1, 6)]))
        for player_id in home - away:
            add_side_vote(votes, player_id, "home")
        for player_id in away - home:
            add_side_vote(votes, player_id, "away")

    side_map: dict[int, str] = {}
    ambiguous = 0
    for player_id, player_votes in votes.items():
        home_votes = player_votes.get("home", 0)
        away_votes = player_votes.get("away", 0)
        if home_votes == away_votes:
            ambiguous += 1
            continue
        side_map[player_id] = "home" if home_votes > away_votes else "away"
    return side_map, {
        "lineup_side_map_players": len(side_map),
        "lineup_side_map_ambiguous_players": ambiguous,
    }


def merge_missing_side_map(base: dict[int, str], fallback: dict[int, str]) -> int:
    added = 0
    for player_id, side in fallback.items():
        if player_id not in base:
            base[player_id] = side
            added += 1
    return added


def sanitize_lineup_side_conflicts(
    game_rows: list[dict],
    player_side_map: dict[int, str],
) -> tuple[list[dict], dict[str, int]]:
    sanitized = [dict(row) for row in game_rows]
    report = {
        "same_player_side_overlap_rows_before": 0,
        "same_player_side_overlap_rows_after": 0,
        "same_player_side_overlap_players_removed": 0,
        "same_player_side_overlap_ambiguous_players_removed": 0,
        "same_player_side_overlap_unresolved_rows": 0,
    }
    for row in sanitized:
        overlap = row_side_overlap(row)
        if not overlap:
            continue
        report["same_player_side_overlap_rows_before"] += 1
        for player_id in overlap:
            known_side = player_side_map.get(int(player_id))
            if known_side == "home":
                remove_side = "away"
            elif known_side == "away":
                remove_side = "home"
            else:
                remove_side = None
            if remove_side is None:
                for side in ("home", "away"):
                    for slot in range(1, 6):
                        col = f"{side}_player{slot}"
                        normalized = normalize_player_id(row.get(col))
                        if pd.notna(normalized) and int(normalized) == int(player_id):
                            row[col] = np.nan
                            report["same_player_side_overlap_ambiguous_players_removed"] += 1
                continue
            for slot in range(1, 6):
                col = f"{remove_side}_player{slot}"
                normalized = normalize_player_id(row.get(col))
                if pd.notna(normalized) and int(normalized) == int(player_id):
                    row[col] = np.nan
                    report["same_player_side_overlap_players_removed"] += 1
        if row_side_overlap(row):
            report["same_player_side_overlap_unresolved_rows"] += 1
    for row in sanitized:
        if row_side_overlap(row):
            report["same_player_side_overlap_rows_after"] += 1
    return sanitized, report


def convert_source_season(df: pd.DataFrame, year: int, season_type: str) -> tuple[pd.DataFrame, dict]:
    if df.empty:
        return pd.DataFrame(), {"input_rows": 0, "converted_rows": 0, "games": 0}

    df = dedupe_team_split_rows(df)
    df = remove_auxiliary_rows(df)
    df = fill_missing_team_from_team_id(df)
    rows: list[dict] = []
    game_summaries: list[dict] = []
    for game_id, game in df.groupby("game_id", sort=True):
        game = sort_source_rows(game).reset_index(drop=True)
        home_team, away_team = infer_game_sides(game)
        player_side_map, player_side_summary = build_game_player_side_map(game, home_team, away_team)
        side_sets = build_side_player_sets(game, home_team, away_team)
        for player_id, side in player_side_map.items():
            side_sets[side].add(float(player_id))
        game_rows: list[dict] = []
        home_opposite_removed = 0
        away_opposite_removed = 0
        for seq, (_, row) in enumerate(game.iterrows(), start=1):
            team = None if pd.isna(row.get("team")) else str(row.get("team"))
            desc = normalize_description(row)
            home_desc = desc if team == home_team else None
            visitor_desc = desc if team == away_team else None
            neutral_desc = desc if team not in {home_team, away_team} else None
            home_players, away_players, side_filter_summary = assign_home_away_lineup(
                row,
                side_sets,
                player_side_map,
            )
            home_opposite_removed += side_filter_summary["home_opposite_side_players_removed"]
            away_opposite_removed += side_filter_summary["away_opposite_side_players_removed"]
            seconds = parse_clock_seconds(row.get("clock_display"))
            home_score = pd.to_numeric(row.get("scoreHome"), errors="coerce")
            away_score = pd.to_numeric(row.get("scoreAway"), errors="coerce")
            date_value = pd.to_numeric(row.get("date"), errors="coerce")
            game_date = (
                pd.to_datetime(str(int(date_value)), format="%Y%m%d", errors="coerce")
                if pd.notna(date_value)
                else pd.NaT
            )
            out = {
                "game_id": int(normalize_game_id(game_id)),
                "event_num": int(seq),
                "source_actionNumber": row.get("actionNumber"),
                "event_type": event_type_from_source(row),
                "event_action_type": event_action_type_from_source(row),
                "period": int(row["period"]),
                "minute_game": float(48.0 - float(row.get("minutes_left_in_game", np.nan)))
                if pd.notna(row.get("minutes_left_in_game"))
                else np.nan,
                "time_remaining": float(row.get("minutes_left_in_game", np.nan))
                if pd.notna(row.get("minutes_left_in_game"))
                else np.nan,
                "wc_time_string": np.nan,
                "time_quarter": row.get("clock_display"),
                "minute_remaining_quarter": seconds // 60,
                "seconds_remaining_quarter": seconds,
                "home_description": home_desc,
                "neutral_description": neutral_desc,
                "visitor_description": visitor_desc,
                "score": f"{int(away_score)} - {int(home_score)}"
                if pd.notna(away_score) and pd.notna(home_score)
                else None,
                "away_score": away_score,
                "home_score": home_score,
                "score_margin": score_margin_text(home_score, away_score),
                "person1type": 0,
                "player1_id": int(row["person_id"]) if pd.notna(row.get("person_id")) else 0,
                "player1_name": row.get("playerName"),
                "player1_team_id": int(row["teamId"]) if pd.notna(row.get("teamId")) else 0,
                "player1_team_city": None,
                "player1_team_nickname": None,
                "player1_team_abbreviation": team,
                "person2type": 0,
                "player2_id": row.get("assister_id"),
                "player2_name": None,
                "player2_team_id": np.nan,
                "player2_team_city": np.nan,
                "player2_team_nickname": np.nan,
                "player2_team_abbreviation": np.nan,
                "person3type": 0,
                "player3_id": row.get("blockPersonId")
                if pd.notna(row.get("blockPersonId"))
                else row.get("stealPersonId"),
                "player3_name": np.nan,
                "player3_team_id": np.nan,
                "player3_team_city": np.nan,
                "player3_team_nickname": np.nan,
                "player3_team_abbreviation": np.nan,
                "video_available_flag": 0,
                "team_leading": None,
                "game_date": game_date,
                "Season": int(year),
                "season_phase": "PS" if season_type == "ps" else "RS",
                "allow_partial_lineups": True,
            }
            for idx in range(5):
                out[f"home_player{idx + 1}"] = home_players[idx] if idx < len(home_players) else np.nan
                out[f"away_player{idx + 1}"] = away_players[idx] if idx < len(away_players) else np.nan
            game_rows.append(out)
        lineup_side_map, lineup_side_summary = build_lineup_player_side_map(game_rows)
        lineup_side_added = merge_missing_side_map(player_side_map, lineup_side_map)
        game_rows, pre_repair_overlap_summary = sanitize_lineup_side_conflicts(game_rows, player_side_map)
        game_rows, lineup_repair_summary = repair_partial_lineups_from_stable_neighbors(game_rows)
        post_lineup_side_map, post_lineup_side_summary = build_lineup_player_side_map(game_rows)
        post_lineup_side_added = merge_missing_side_map(player_side_map, post_lineup_side_map)
        game_rows, post_repair_overlap_summary = sanitize_lineup_side_conflicts(game_rows, player_side_map)
        rows.extend(game_rows)
        game_summaries.append(
            {
                "game_id": int(normalize_game_id(game_id)),
                "rows": len(game_rows),
                "home_team": home_team,
                "away_team": away_team,
                **player_side_summary,
                **lineup_side_summary,
                "lineup_side_map_added_players": lineup_side_added,
                "post_repair_lineup_side_map_players": post_lineup_side_summary["lineup_side_map_players"],
                "post_repair_lineup_side_map_ambiguous_players": post_lineup_side_summary[
                    "lineup_side_map_ambiguous_players"
                ],
                "post_repair_lineup_side_map_added_players": post_lineup_side_added,
                "home_opposite_side_players_removed": home_opposite_removed,
                "away_opposite_side_players_removed": away_opposite_removed,
                **{f"pre_repair_{key}": value for key, value in pre_repair_overlap_summary.items()},
                **lineup_repair_summary,
                **{f"post_repair_{key}": value for key, value in post_repair_overlap_summary.items()},
            }
        )

    converted = pd.DataFrame(rows)
    if not converted.empty:
        converted = converted.sort_values(["game_id", "event_num"], kind="stable").reset_index(drop=True)
    summary = {
        "input_rows_after_team_dedupe_and_aux_drop": int(len(df)),
        "converted_rows": int(len(converted)),
        "games": int(converted["game_id"].nunique()) if not converted.empty else 0,
        "game_side_failures": int(
            sum(1 for item in game_summaries if item["home_team"] is None or item["away_team"] is None)
        ),
        "home_partial_lineup_rows_before": int(
            sum(item.get("home_partial_rows_before", 0) for item in game_summaries)
        ),
        "home_partial_lineup_rows_repaired": int(
            sum(item.get("home_repaired_rows", 0) for item in game_summaries)
        ),
        "home_partial_lineup_rows_after": int(
            sum(item.get("home_partial_rows_after", 0) for item in game_summaries)
        ),
        "away_partial_lineup_rows_before": int(
            sum(item.get("away_partial_rows_before", 0) for item in game_summaries)
        ),
        "away_partial_lineup_rows_repaired": int(
            sum(item.get("away_repaired_rows", 0) for item in game_summaries)
        ),
        "away_partial_lineup_rows_after": int(
            sum(item.get("away_partial_rows_after", 0) for item in game_summaries)
        ),
        "game_player_side_map_players": int(
            sum(item.get("side_map_players", 0) for item in game_summaries)
        ),
        "game_player_side_map_ambiguous_players": int(
            sum(item.get("side_map_ambiguous_players", 0) for item in game_summaries)
        ),
        "lineup_side_map_players": int(
            sum(item.get("lineup_side_map_players", 0) for item in game_summaries)
        ),
        "lineup_side_map_ambiguous_players": int(
            sum(item.get("lineup_side_map_ambiguous_players", 0) for item in game_summaries)
        ),
        "lineup_side_map_added_players": int(
            sum(item.get("lineup_side_map_added_players", 0) for item in game_summaries)
        ),
        "post_repair_lineup_side_map_players": int(
            sum(item.get("post_repair_lineup_side_map_players", 0) for item in game_summaries)
        ),
        "post_repair_lineup_side_map_ambiguous_players": int(
            sum(item.get("post_repair_lineup_side_map_ambiguous_players", 0) for item in game_summaries)
        ),
        "post_repair_lineup_side_map_added_players": int(
            sum(item.get("post_repair_lineup_side_map_added_players", 0) for item in game_summaries)
        ),
        "home_opposite_side_players_removed": int(
            sum(item.get("home_opposite_side_players_removed", 0) for item in game_summaries)
        ),
        "away_opposite_side_players_removed": int(
            sum(item.get("away_opposite_side_players_removed", 0) for item in game_summaries)
        ),
        "pre_repair_same_player_side_overlap_rows_before": int(
            sum(item.get("pre_repair_same_player_side_overlap_rows_before", 0) for item in game_summaries)
        ),
        "pre_repair_same_player_side_overlap_rows_after": int(
            sum(item.get("pre_repair_same_player_side_overlap_rows_after", 0) for item in game_summaries)
        ),
        "pre_repair_same_player_side_overlap_players_removed": int(
            sum(item.get("pre_repair_same_player_side_overlap_players_removed", 0) for item in game_summaries)
        ),
        "pre_repair_same_player_side_overlap_ambiguous_players_removed": int(
            sum(
                item.get("pre_repair_same_player_side_overlap_ambiguous_players_removed", 0)
                for item in game_summaries
            )
        ),
        "pre_repair_same_player_side_overlap_unresolved_rows": int(
            sum(item.get("pre_repair_same_player_side_overlap_unresolved_rows", 0) for item in game_summaries)
        ),
        "post_repair_same_player_side_overlap_rows_before": int(
            sum(item.get("post_repair_same_player_side_overlap_rows_before", 0) for item in game_summaries)
        ),
        "post_repair_same_player_side_overlap_rows_after": int(
            sum(item.get("post_repair_same_player_side_overlap_rows_after", 0) for item in game_summaries)
        ),
        "post_repair_same_player_side_overlap_players_removed": int(
            sum(item.get("post_repair_same_player_side_overlap_players_removed", 0) for item in game_summaries)
        ),
        "post_repair_same_player_side_overlap_ambiguous_players_removed": int(
            sum(
                item.get("post_repair_same_player_side_overlap_ambiguous_players_removed", 0)
                for item in game_summaries
            )
        ),
        "post_repair_same_player_side_overlap_unresolved_rows": int(
            sum(item.get("post_repair_same_player_side_overlap_unresolved_rows", 0) for item in game_summaries)
        ),
    }
    return converted, summary


def build_raw_season(source_dir: Path, raw_dir: Path, year: int, season_type: str, overwrite: bool) -> dict:
    output_path = raw_output_path(raw_dir, year, season_type)
    if output_path.exists() and not overwrite:
        return {"raw_path": str(output_path), "skipped": True, "reason": "exists"}

    files = file_paths_for_year(source_dir, year, season_type)
    if not files:
        return {"raw_path": str(output_path), "skipped": True, "reason": "no_source_files"}

    frames = []
    for path in files:
        frame = pd.read_parquet(path)
        frame["_source_file"] = path.name
        frames.append(frame)
    source_df = pd.concat(frames, ignore_index=True, sort=False)
    converted, summary = convert_source_season(source_df, year, season_type)
    raw_dir.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    converted.to_parquet(tmp_output_path, index=False)
    tmp_output_path.replace(output_path)
    return {
        "raw_path": str(output_path),
        "skipped": False,
        "source_files": len(files),
        "source_rows": int(len(source_df)),
        **summary,
    }


def process_metrics(raw_path: Path, processed_dir: Path, year: int, season_type: str, metrics: list[str], overwrite: bool) -> dict:
    out: dict[str, dict] = {}
    label = season_label(year)
    processed_dir.mkdir(parents=True, exist_ok=True)
    for metric in metrics:
        metric = metric.upper()
        output_path = processed_output_path(processed_dir, metric, year, season_type)
        if output_path.exists() and not overwrite:
            out[metric] = {"path": str(output_path), "skipped": True, "reason": "exists"}
            continue
        try:
            processed = PROCESSORS[metric](raw_path, int(year), label)
            if processed is None or processed.empty:
                out[metric] = {"path": str(output_path), "skipped": True, "reason": "empty"}
                continue
            processed.to_parquet(output_path, index=False)
            out[metric] = {"path": str(output_path), "rows": int(len(processed)), "skipped": False}
        except Exception as exc:
            out[metric] = {"path": str(output_path), "skipped": True, "error": str(exc)}
    return out


def build_one_season_task(
    source_dir: str,
    raw_dir: str,
    processed_dir: str,
    year: int,
    season_type: str,
    metrics: list[str],
    overwrite: bool,
    raw_only: bool,
) -> tuple[str, dict, dict | None]:
    key = f"{year}_{season_type}"
    print(f"\n=== Building Gabriel raw {key} ===", flush=True)
    raw_summary = build_raw_season(
        Path(source_dir),
        Path(raw_dir),
        year,
        season_type,
        overwrite=overwrite,
    )
    raw_path = Path(raw_summary["raw_path"])
    processed_summary = None
    if not raw_only and not (raw_summary.get("skipped") and raw_summary.get("reason") == "no_source_files"):
        if raw_path.exists():
            print(f"=== Processing Gabriel metrics {key}: {','.join(metrics)} ===", flush=True)
            processed_summary = process_metrics(
                raw_path,
                Path(processed_dir),
                year,
                season_type,
                metrics,
                overwrite=overwrite,
            )
    return key, raw_summary, processed_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 1997-2013 Gabriel old PBP raw and metric parquets.")
    parser.add_argument("--years", default="1997-2013", help="Comma/range list, e.g. 1997-2013 or 2010,2011.")
    parser.add_argument("--season-types", default="all", help="rs, ps, or all.")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--metrics", default="RAPM,TOV,REB,TS")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--raw-only", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Year/season build workers. Default: 1.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    years = parse_years(args.years)
    season_types = parse_season_types(args.season_types)
    metrics = [part.strip().upper() for part in args.metrics.split(",") if part.strip()]
    bad_metrics = sorted(set(metrics) - set(PROCESSORS))
    if bad_metrics:
        raise ValueError(f"Unsupported metrics: {bad_metrics}. Supported: {sorted(PROCESSORS)}")

    if not args.skip_download:
        ensure_source_repo(args.source_dir, args.repo_url)
        all_paths = list_old_data_paths(args.source_dir)
        wanted = wanted_source_paths(all_paths, years, season_types)
        checkout_paths(args.source_dir, wanted)
    else:
        wanted = []

    report: dict = {
        "source_dir": str(args.source_dir),
        "years": years,
        "season_types": season_types,
        "metrics": metrics,
        "downloaded_or_checked_paths": len(wanted),
        "raw": {},
        "processed": {},
    }
    if args.download_only:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {args.report}")
        return 0

    overwrite = not args.no_overwrite
    tasks = [(year, season_type) for year in years for season_type in season_types]
    if args.workers <= 1:
        for year, season_type in tasks:
            key, raw_summary, processed_summary = build_one_season_task(
                str(args.source_dir),
                str(args.raw_dir),
                str(args.processed_dir),
                year,
                season_type,
                metrics,
                overwrite,
                args.raw_only,
            )
            report["raw"][key] = raw_summary
            if processed_summary is not None:
                report["processed"][key] = processed_summary
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    build_one_season_task,
                    str(args.source_dir),
                    str(args.raw_dir),
                    str(args.processed_dir),
                    year,
                    season_type,
                    metrics,
                    overwrite,
                    args.raw_only,
                ): (year, season_type)
                for year, season_type in tasks
            }
            for future in as_completed(futures):
                key, raw_summary, processed_summary = future.result()
                report["raw"][key] = raw_summary
                if processed_summary is not None:
                    report["processed"][key] = processed_summary

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
