#!/usr/bin/env python3
"""
Build career teammate shared-minute and net-rating summaries.

The builder combines two repo-native surfaces:
- raw NBA*.parquet files for elapsed seconds with same-team lineups
- processed RAPM*.parquet files for possession counts and scoring while together

Default output covers regular season plus playoffs from 1997 through 2026.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_DIR.parent
RAW_DATA_DIR = PIPELINE_DIR / "raw_data"
PROCESSED_DIR = PIPELINE_DIR / "processed"
DEFAULT_OUTPUT_DIR = PIPELINE_DIR / "results" / "career_teammates"
AUTOCOMPLETE_MAP = PROJECT_ROOT / "autocomplete_map.csv"

KEY_BASE = 100_000_000
PAIR_INDEXES = tuple((i, j) for i in range(5) for j in range(i + 1, 5))
RAW_HOME_COLS = [f"home_player{i}" for i in range(1, 6)]
RAW_AWAY_COLS = [f"away_player{i}" for i in range(1, 6)]
RAPM_OFF_COLS = [f"O{i}" for i in range(1, 6)]
RAPM_DEF_COLS = [f"D{i}" for i in range(1, 6)]


def season_to_year(season: int) -> int:
    return season if season >= 100 else 2000 + season if season < 70 else 1900 + season


def season_to_yy(season: int) -> str:
    return f"{season_to_year(season) % 100:02d}"


def parse_season_from_name(path: Path, prefix: str) -> tuple[int, str] | None:
    match = re.fullmatch(rf"{prefix}(\d{{2}})(_PS)?\.parquet", path.name)
    if not match:
        return None
    return season_to_year(int(match.group(1))), "ps" if match.group(2) else "rs"


def selected_files(
    directory: Path,
    prefix: str,
    start_year: int,
    end_year: int,
    season_types: set[str],
) -> list[Path]:
    files: list[Path] = []
    for path in directory.glob(f"{prefix}*.parquet"):
        parsed = parse_season_from_name(path, prefix)
        if not parsed:
            continue
        season, season_type = parsed
        if start_year <= season <= end_year and season_type in season_types:
            files.append(path)
    return sorted(files, key=lambda p: (parse_season_from_name(p, prefix) or (0, "")))


def encode_pair_ids(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lo = np.minimum(a, b).astype(np.int64, copy=False)
    hi = np.maximum(a, b).astype(np.int64, copy=False)
    return lo * KEY_BASE + hi


def decode_pair_key(key: int) -> tuple[int, int]:
    return int(key // KEY_BASE), int(key % KEY_BASE)


def add_weighted_unique(target: defaultdict[int, float], keys: np.ndarray, weights: np.ndarray) -> None:
    if keys.size == 0:
        return
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    sums = np.bincount(inverse, weights=weights.astype(float, copy=False))
    for key, value in zip(unique_keys.tolist(), sums.tolist()):
        target[int(key)] += float(value)


def add_count_unique(target: defaultdict[int, float], keys: np.ndarray) -> None:
    if keys.size == 0:
        return
    unique_keys, counts = np.unique(keys, return_counts=True)
    for key, value in zip(unique_keys.tolist(), counts.tolist()):
        target[int(key)] += float(value)


def add_player_seconds(
    player_seconds: defaultdict[int, float],
    lineup: np.ndarray,
    seconds: np.ndarray,
) -> None:
    repeated_seconds = np.repeat(seconds, lineup.shape[1])
    players = lineup.reshape(-1)
    valid = (repeated_seconds > 0) & np.isfinite(players) & (players > 0)
    add_weighted_unique(
        player_seconds,
        players[valid].astype(np.int64, copy=False),
        repeated_seconds[valid],
    )


def add_pair_seconds(
    pair_seconds: defaultdict[int, float],
    lineup: np.ndarray,
    seconds: np.ndarray,
) -> None:
    active_seconds = seconds.astype(float, copy=False)
    for i, j in PAIR_INDEXES:
        a = lineup[:, i]
        b = lineup[:, j]
        valid = (
            (active_seconds > 0)
            & np.isfinite(a)
            & np.isfinite(b)
            & (a > 0)
            & (b > 0)
            & (a != b)
        )
        add_weighted_unique(
            pair_seconds,
            encode_pair_ids(a[valid], b[valid]),
            active_seconds[valid],
        )


def add_pair_possessions(
    pair_possessions: defaultdict[int, float],
    pair_points: defaultdict[int, float],
    lineup: np.ndarray,
    points: np.ndarray,
) -> None:
    points = points.astype(float, copy=False)
    for i, j in PAIR_INDEXES:
        a = lineup[:, i]
        b = lineup[:, j]
        valid = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0) & (a != b)
        keys = encode_pair_ids(a[valid], b[valid])
        add_count_unique(pair_possessions, keys)
        add_weighted_unique(pair_points, keys, points[valid])


def add_player_possessions(
    player_possessions: defaultdict[int, float],
    player_points: defaultdict[int, float],
    lineup: np.ndarray,
    points: np.ndarray,
) -> None:
    repeated_points = np.repeat(points.astype(float, copy=False), lineup.shape[1])
    players = lineup.reshape(-1)
    valid = np.isfinite(players) & (players > 0)
    add_count_unique(player_possessions, players[valid].astype(np.int64, copy=False))
    add_weighted_unique(
        player_points,
        players[valid].astype(np.int64, copy=False),
        repeated_points[valid],
    )


def seconds_remaining_total(df: pd.DataFrame) -> np.ndarray:
    raw_seconds = pd.to_numeric(df["seconds_remaining_quarter"], errors="coerce").fillna(0).to_numpy(float)
    minutes = pd.to_numeric(df["minute_remaining_quarter"], errors="coerce").fillna(0).to_numpy(float)
    return np.where(raw_seconds <= 59, minutes * 60.0 + raw_seconds, raw_seconds)


def collect_names_from_raw(df: pd.DataFrame, name_map: dict[int, str]) -> None:
    for idx in (1, 2, 3):
        id_col = f"player{idx}_id"
        name_col = f"player{idx}_name"
        if id_col not in df or name_col not in df:
            continue
        ids = pd.to_numeric(df[id_col], errors="coerce")
        names = df[name_col].astype("string")
        valid = ids.notna() & names.notna() & (names.str.strip() != "")
        if not valid.any():
            continue
        slim = pd.DataFrame({"player_id": ids[valid].astype(np.int64), "player_name": names[valid].str.strip()})
        for row in slim.drop_duplicates("player_id", keep="last").itertuples(index=False):
            name_map.setdefault(int(row.player_id), str(row.player_name))


def aggregate_minutes(
    raw_files: Iterable[Path],
    player_seconds: defaultdict[int, float],
    pair_seconds: defaultdict[int, float],
    name_map: dict[int, str],
) -> None:
    columns = [
        "game_id",
        "event_num",
        "period",
        "minute_remaining_quarter",
        "seconds_remaining_quarter",
        *RAW_HOME_COLS,
        *RAW_AWAY_COLS,
        "player1_id",
        "player1_name",
        "player2_id",
        "player2_name",
        "player3_id",
        "player3_name",
    ]
    for path in raw_files:
        print(f"minutes: {path}")
        df = pd.read_parquet(path, columns=columns)
        collect_names_from_raw(df, name_map)
        df = df.sort_values(["game_id", "period", "event_num"], kind="mergesort")
        seconds = seconds_remaining_total(df)
        next_seconds = np.roll(seconds, -1)
        same_segment = (
            (df["game_id"].to_numpy() == np.roll(df["game_id"].to_numpy(), -1))
            & (df["period"].to_numpy() == np.roll(df["period"].to_numpy(), -1))
        )
        elapsed = np.where(same_segment, seconds - next_seconds, 0.0)
        elapsed = np.where((elapsed > 0) & (elapsed <= 720), elapsed, 0.0)

        for side_cols in (RAW_HOME_COLS, RAW_AWAY_COLS):
            lineup = df[side_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            add_player_seconds(player_seconds, lineup, elapsed)
            add_pair_seconds(pair_seconds, lineup, elapsed)


def aggregate_ratings(
    rapm_files: Iterable[Path],
    player_off_possessions: defaultdict[int, float],
    player_def_possessions: defaultdict[int, float],
    player_points_for: defaultdict[int, float],
    player_points_against: defaultdict[int, float],
    off_possessions: defaultdict[int, float],
    def_possessions: defaultdict[int, float],
    points_for: defaultdict[int, float],
    points_against: defaultdict[int, float],
) -> None:
    columns = [*RAPM_OFF_COLS, *RAPM_DEF_COLS, "Net_Diff"]
    for path in rapm_files:
        print(f"ratings: {path}")
        df = pd.read_parquet(path, columns=columns)
        points = pd.to_numeric(df["Net_Diff"], errors="coerce").fillna(0).to_numpy(float)
        off_lineup = df[RAPM_OFF_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        def_lineup = df[RAPM_DEF_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        add_player_possessions(player_off_possessions, player_points_for, off_lineup, points)
        add_player_possessions(player_def_possessions, player_points_against, def_lineup, points)
        add_pair_possessions(off_possessions, points_for, off_lineup, points)
        add_pair_possessions(def_possessions, points_against, def_lineup, points)


def load_name_map() -> dict[int, str]:
    name_map: dict[int, str] = {}
    if AUTOCOMPLETE_MAP.exists():
        ac = pd.read_csv(AUTOCOMPLETE_MAP)
        if {"nba_id", "player_name"}.issubset(ac.columns):
            for row in ac.dropna(subset=["nba_id", "player_name"]).itertuples(index=False):
                name_map[int(row.nba_id)] = str(row.player_name)
    return name_map


def player_name(player_id: int, name_map: dict[int, str]) -> str:
    return name_map.get(player_id, f"ID_{player_id}")


def rate(points: float, possessions: float) -> float:
    return float(100.0 * points / possessions) if possessions > 0 else np.nan


def build_pair_rows(
    pair_seconds: defaultdict[int, float],
    player_seconds: defaultdict[int, float],
    player_off_possessions: defaultdict[int, float],
    player_def_possessions: defaultdict[int, float],
    player_points_for: defaultdict[int, float],
    player_points_against: defaultdict[int, float],
    off_possessions: defaultdict[int, float],
    def_possessions: defaultdict[int, float],
    points_for: defaultdict[int, float],
    points_against: defaultdict[int, float],
    name_map: dict[int, str],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    all_pair_keys = set(pair_seconds) | set(off_possessions) | set(def_possessions)
    for key in sorted(all_pair_keys):
        player1_id, player2_id = decode_pair_key(key)
        shared_seconds = pair_seconds.get(key, 0.0)
        p1_seconds = player_seconds.get(player1_id, 0.0)
        p2_seconds = player_seconds.get(player2_id, 0.0)
        p1_off_poss = player_off_possessions.get(player1_id, 0.0)
        p1_def_poss = player_def_possessions.get(player1_id, 0.0)
        p1_pf = player_points_for.get(player1_id, 0.0)
        p1_pa = player_points_against.get(player1_id, 0.0)
        p2_off_poss = player_off_possessions.get(player2_id, 0.0)
        p2_def_poss = player_def_possessions.get(player2_id, 0.0)
        p2_pf = player_points_for.get(player2_id, 0.0)
        p2_pa = player_points_against.get(player2_id, 0.0)
        off_poss = off_possessions.get(key, 0.0)
        def_poss = def_possessions.get(key, 0.0)
        pf = points_for.get(key, 0.0)
        pa = points_against.get(key, 0.0)
        ortg = rate(pf, off_poss)
        drtg = rate(pa, def_poss)
        p1_ortg = rate(p1_pf, p1_off_poss)
        p1_drtg = rate(p1_pa, p1_def_poss)
        p2_ortg = rate(p2_pf, p2_off_poss)
        p2_drtg = rate(p2_pa, p2_def_poss)
        p1_without_ortg = rate(p1_pf - pf, p1_off_poss - off_poss)
        p1_without_drtg = rate(p1_pa - pa, p1_def_poss - def_poss)
        p2_without_ortg = rate(p2_pf - pf, p2_off_poss - off_poss)
        p2_without_drtg = rate(p2_pa - pa, p2_def_poss - def_poss)
        shared_poss = off_poss + def_poss
        p1_total_poss = p1_off_poss + p1_def_poss
        p2_total_poss = p2_off_poss + p2_def_poss
        rows.append(
            {
                "player1_id": player1_id,
                "player1_name": player_name(player1_id, name_map),
                "player2_id": player2_id,
                "player2_name": player_name(player2_id, name_map),
                "shared_minutes": shared_seconds / 60.0,
                "player1_total_minutes": p1_seconds / 60.0,
                "player2_total_minutes": p2_seconds / 60.0,
                "player1_pct_minutes_with_player2": 100.0 * shared_seconds / p1_seconds if p1_seconds > 0 else np.nan,
                "player2_pct_minutes_with_player1": 100.0 * shared_seconds / p2_seconds if p2_seconds > 0 else np.nan,
                "off_poss": off_poss,
                "def_poss": def_poss,
                "shared_poss": shared_poss,
                "points_for": pf,
                "points_against": pa,
                "ortg": ortg,
                "drtg": drtg,
                "net_rating": ortg - drtg if np.isfinite(ortg) and np.isfinite(drtg) else np.nan,
                "player1_off_poss": p1_off_poss,
                "player1_def_poss": p1_def_poss,
                "player1_total_poss": p1_total_poss,
                "player1_pct_poss_with_player2": 100.0 * shared_poss / p1_total_poss if p1_total_poss > 0 else np.nan,
                "player1_ortg": p1_ortg,
                "player1_drtg": p1_drtg,
                "player1_net_rating": p1_ortg - p1_drtg if np.isfinite(p1_ortg) and np.isfinite(p1_drtg) else np.nan,
                "player1_without_player2_off_poss": p1_off_poss - off_poss,
                "player1_without_player2_def_poss": p1_def_poss - def_poss,
                "player1_without_player2_ortg": p1_without_ortg,
                "player1_without_player2_drtg": p1_without_drtg,
                "player1_without_player2_net_rating": (
                    p1_without_ortg - p1_without_drtg
                    if np.isfinite(p1_without_ortg) and np.isfinite(p1_without_drtg)
                    else np.nan
                ),
                "player2_off_poss": p2_off_poss,
                "player2_def_poss": p2_def_poss,
                "player2_total_poss": p2_total_poss,
                "player2_pct_poss_with_player1": 100.0 * shared_poss / p2_total_poss if p2_total_poss > 0 else np.nan,
                "player2_ortg": p2_ortg,
                "player2_drtg": p2_drtg,
                "player2_net_rating": p2_ortg - p2_drtg if np.isfinite(p2_ortg) and np.isfinite(p2_drtg) else np.nan,
                "player2_without_player1_off_poss": p2_off_poss - off_poss,
                "player2_without_player1_def_poss": p2_def_poss - def_poss,
                "player2_without_player1_ortg": p2_without_ortg,
                "player2_without_player1_drtg": p2_without_drtg,
                "player2_without_player1_net_rating": (
                    p2_without_ortg - p2_without_drtg
                    if np.isfinite(p2_without_ortg) and np.isfinite(p2_without_drtg)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def build_oriented_top_pairs(pair_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    left = pair_df.rename(
        columns={
            "player1_id": "player_id",
            "player1_name": "player_name",
            "player2_id": "teammate_id",
            "player2_name": "teammate_name",
            "player1_total_minutes": "player_total_minutes",
            "player2_total_minutes": "teammate_total_minutes",
            "player1_pct_minutes_with_player2": "pct_player_minutes_with_teammate",
            "player2_pct_minutes_with_player1": "pct_teammate_minutes_with_player",
            "player1_off_poss": "player_off_poss",
            "player1_def_poss": "player_def_poss",
            "player1_total_poss": "player_total_poss",
            "player1_pct_poss_with_player2": "pct_player_poss_with_teammate",
            "player1_ortg": "player_ortg",
            "player1_drtg": "player_drtg",
            "player1_net_rating": "player_net_rating",
            "player1_without_player2_off_poss": "player_without_teammate_off_poss",
            "player1_without_player2_def_poss": "player_without_teammate_def_poss",
            "player1_without_player2_ortg": "player_without_teammate_ortg",
            "player1_without_player2_drtg": "player_without_teammate_drtg",
            "player1_without_player2_net_rating": "player_without_teammate_net_rating",
            "player2_off_poss": "teammate_off_poss",
            "player2_def_poss": "teammate_def_poss",
            "player2_total_poss": "teammate_total_poss",
            "player2_pct_poss_with_player1": "pct_teammate_poss_with_player",
            "player2_ortg": "teammate_ortg",
            "player2_drtg": "teammate_drtg",
            "player2_net_rating": "teammate_net_rating",
            "player2_without_player1_off_poss": "teammate_without_player_off_poss",
            "player2_without_player1_def_poss": "teammate_without_player_def_poss",
            "player2_without_player1_ortg": "teammate_without_player_ortg",
            "player2_without_player1_drtg": "teammate_without_player_drtg",
            "player2_without_player1_net_rating": "teammate_without_player_net_rating",
        }
    )
    right = pair_df.rename(
        columns={
            "player2_id": "player_id",
            "player2_name": "player_name",
            "player1_id": "teammate_id",
            "player1_name": "teammate_name",
            "player2_total_minutes": "player_total_minutes",
            "player1_total_minutes": "teammate_total_minutes",
            "player2_pct_minutes_with_player1": "pct_player_minutes_with_teammate",
            "player1_pct_minutes_with_player2": "pct_teammate_minutes_with_player",
            "player2_off_poss": "player_off_poss",
            "player2_def_poss": "player_def_poss",
            "player2_total_poss": "player_total_poss",
            "player2_pct_poss_with_player1": "pct_player_poss_with_teammate",
            "player2_ortg": "player_ortg",
            "player2_drtg": "player_drtg",
            "player2_net_rating": "player_net_rating",
            "player2_without_player1_off_poss": "player_without_teammate_off_poss",
            "player2_without_player1_def_poss": "player_without_teammate_def_poss",
            "player2_without_player1_ortg": "player_without_teammate_ortg",
            "player2_without_player1_drtg": "player_without_teammate_drtg",
            "player2_without_player1_net_rating": "player_without_teammate_net_rating",
            "player1_off_poss": "teammate_off_poss",
            "player1_def_poss": "teammate_def_poss",
            "player1_total_poss": "teammate_total_poss",
            "player1_pct_poss_with_player2": "pct_teammate_poss_with_player",
            "player1_ortg": "teammate_ortg",
            "player1_drtg": "teammate_drtg",
            "player1_net_rating": "teammate_net_rating",
            "player1_without_player2_off_poss": "teammate_without_player_off_poss",
            "player1_without_player2_def_poss": "teammate_without_player_def_poss",
            "player1_without_player2_ortg": "teammate_without_player_ortg",
            "player1_without_player2_drtg": "teammate_without_player_drtg",
            "player1_without_player2_net_rating": "teammate_without_player_net_rating",
        }
    )
    columns = [
        "player_id",
        "player_name",
        "teammate_id",
        "teammate_name",
        "shared_minutes",
        "player_total_minutes",
        "pct_player_minutes_with_teammate",
        "teammate_total_minutes",
        "pct_teammate_minutes_with_player",
        "off_poss",
        "def_poss",
        "shared_poss",
        "points_for",
        "points_against",
        "ortg",
        "drtg",
        "net_rating",
        "player_off_poss",
        "player_def_poss",
        "player_total_poss",
        "pct_player_poss_with_teammate",
        "player_ortg",
        "player_drtg",
        "player_net_rating",
        "player_without_teammate_off_poss",
        "player_without_teammate_def_poss",
        "player_without_teammate_ortg",
        "player_without_teammate_drtg",
        "player_without_teammate_net_rating",
        "teammate_off_poss",
        "teammate_def_poss",
        "teammate_total_poss",
        "pct_teammate_poss_with_player",
        "teammate_ortg",
        "teammate_drtg",
        "teammate_net_rating",
        "teammate_without_player_off_poss",
        "teammate_without_player_def_poss",
        "teammate_without_player_ortg",
        "teammate_without_player_drtg",
        "teammate_without_player_net_rating",
    ]
    oriented = pd.concat([left[columns], right[columns]], ignore_index=True)
    oriented = oriented.sort_values(
        ["player_id", "shared_poss", "shared_minutes", "teammate_id"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )
    oriented["rank"] = oriented.groupby("player_id").cumcount() + 1
    oriented = oriented[oriented["rank"] <= top_n].copy()
    return oriented[["rank", *columns]].sort_values(["player_name", "rank", "teammate_name"], kind="mergesort")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=1997)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument(
        "--season-types",
        choices=["all", "rs", "ps"],
        default="all",
        help="Use regular season, playoffs, or both.",
    )
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_year = season_to_year(args.start_year)
    end_year = season_to_year(args.end_year)
    season_types = {"rs", "ps"} if args.season_types == "all" else {args.season_types}

    raw_files = selected_files(args.raw_dir, "NBA", start_year, end_year, season_types)
    rapm_files = selected_files(args.processed_dir, "RAPM", start_year, end_year, season_types)
    if not raw_files:
        raise FileNotFoundError(f"No raw NBA parquet files found in {args.raw_dir}")
    if not rapm_files:
        raise FileNotFoundError(f"No processed RAPM parquet files found in {args.processed_dir}")

    print(f"Raw files: {len(raw_files)}")
    print(f"RAPM files: {len(rapm_files)}")

    player_seconds: defaultdict[int, float] = defaultdict(float)
    pair_seconds: defaultdict[int, float] = defaultdict(float)
    off_possessions: defaultdict[int, float] = defaultdict(float)
    def_possessions: defaultdict[int, float] = defaultdict(float)
    points_for: defaultdict[int, float] = defaultdict(float)
    points_against: defaultdict[int, float] = defaultdict(float)
    player_off_possessions: defaultdict[int, float] = defaultdict(float)
    player_def_possessions: defaultdict[int, float] = defaultdict(float)
    player_points_for: defaultdict[int, float] = defaultdict(float)
    player_points_against: defaultdict[int, float] = defaultdict(float)
    name_map = load_name_map()

    aggregate_minutes(raw_files, player_seconds, pair_seconds, name_map)
    aggregate_ratings(
        rapm_files,
        player_off_possessions,
        player_def_possessions,
        player_points_for,
        player_points_against,
        off_possessions,
        def_possessions,
        points_for,
        points_against,
    )

    pair_df = build_pair_rows(
        pair_seconds,
        player_seconds,
        player_off_possessions,
        player_def_possessions,
        player_points_for,
        player_points_against,
        off_possessions,
        def_possessions,
        points_for,
        points_against,
        name_map,
    )
    top_df = build_oriented_top_pairs(pair_df, args.top_n)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    window = f"{season_to_yy(start_year)}_{season_to_yy(end_year)}_{args.season_types}"
    pair_path = args.output_dir / f"career_teammate_pairs_{window}.csv"
    top_path = args.output_dir / f"career_teammate_top{args.top_n}_{window}.csv"
    pair_df.to_csv(pair_path, index=False)
    top_df.to_csv(top_path, index=False)
    print(f"Wrote {len(pair_df):,} pair rows to {pair_path}")
    print(f"Wrote {len(top_df):,} top teammate rows to {top_path}")


if __name__ == "__main__":
    main()
