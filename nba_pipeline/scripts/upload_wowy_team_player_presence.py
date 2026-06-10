#!/usr/bin/env python3
"""
Build and upload PBP-derived team-player presence for Databallr WOWY RAPM mode.

The table answers one narrow question efficiently: which players appeared in a
team's raw play-by-play lineups during a season window? RAPM values stay in the
existing player_alt3_efg_factors table.

Examples:
    python nba_pipeline/scripts/upload_wowy_team_player_presence.py --start-year 2022 --end-year 2026
    python nba_pipeline/scripts/upload_wowy_team_player_presence.py --start-year 2022 --end-year 2026 --upload
"""

from __future__ import annotations

import argparse
import os
import sys
from functools import reduce
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_DIR.parent
RAW_DATA_DIR = PIPELINE_DIR / "raw_data"
AUTOCOMPLETE_MAP = PROJECT_ROOT / "autocomplete_map.csv"

sys.path.append(str(SCRIPT_DIR))
from lineup_stats import load_name_map  # noqa: E402

HOME_COLS = [f"home_player{i}" for i in range(1, 6)]
AWAY_COLS = [f"away_player{i}" for i in range(1, 6)]
PLAYER_ID_NAME_COLS = [
    "player1_id",
    "player1_name",
    "player2_id",
    "player2_name",
    "player3_id",
    "player3_name",
]
RAW_REQUIRED_COLS = [
    "game_id",
    *HOME_COLS,
    *AWAY_COLS,
    "player1_id",
    "player1_team_abbreviation",
    "player2_id",
    "player2_team_abbreviation",
    "player3_id",
    "player3_team_abbreviation",
]

TEAM_ABBR_TO_ID = {
    "ATL": 1610612737,
    "BOS": 1610612738,
    "BKN": 1610612751,
    "BRK": 1610612751,
    "NJN": 1610612751,
    "CHA": 1610612766,
    "CHO": 1610612766,
    "CHH": 1610612766,
    "CHI": 1610612741,
    "CLE": 1610612739,
    "DAL": 1610612742,
    "DEN": 1610612743,
    "DET": 1610612765,
    "GSW": 1610612744,
    "HOU": 1610612745,
    "IND": 1610612754,
    "LAC": 1610612746,
    "LAL": 1610612747,
    "MEM": 1610612763,
    "VAN": 1610612763,
    "MIA": 1610612748,
    "MIL": 1610612749,
    "MIN": 1610612750,
    "NOP": 1610612740,
    "NOH": 1610612740,
    "NOK": 1610612740,
    "NYK": 1610612752,
    "OKC": 1610612760,
    "SEA": 1610612760,
    "ORL": 1610612753,
    "PHI": 1610612755,
    "PHX": 1610612756,
    "POR": 1610612757,
    "SAC": 1610612758,
    "SAS": 1610612759,
    "TOR": 1610612761,
    "UTA": 1610612762,
    "WAS": 1610612764,
    "WSB": 1610612764,
}

TEAM_ID_TO_CURRENT_ABBR = {
    1610612737: "ATL",
    1610612738: "BOS",
    1610612751: "BKN",
    1610612766: "CHA",
    1610612741: "CHI",
    1610612739: "CLE",
    1610612742: "DAL",
    1610612743: "DEN",
    1610612765: "DET",
    1610612744: "GSW",
    1610612745: "HOU",
    1610612754: "IND",
    1610612746: "LAC",
    1610612747: "LAL",
    1610612763: "MEM",
    1610612748: "MIA",
    1610612749: "MIL",
    1610612750: "MIN",
    1610612740: "NOP",
    1610612752: "NYK",
    1610612760: "OKC",
    1610612753: "ORL",
    1610612755: "PHI",
    1610612756: "PHX",
    1610612757: "POR",
    1610612758: "SAC",
    1610612759: "SAS",
    1610612761: "TOR",
    1610612762: "UTA",
    1610612764: "WAS",
}


def normalize_season(year: int) -> int:
    if year < 100:
        return 2000 + year
    return year


def season_to_suffix(season: int) -> int:
    return season % 100


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def get_raw_path(season: int, season_type: str) -> Path:
    suffix = season_to_suffix(season)
    if season_type == "PS":
        return RAW_DATA_DIR / f"NBA{suffix}_PS.parquet"
    return RAW_DATA_DIR / f"NBA{suffix}.parquet"


def extract_raw_name_map(raw_path: Path) -> dict[int, str]:
    name_map: dict[int, str] = {}
    if not raw_path.exists():
        return name_map

    raw_columns = set(pq.read_schema(raw_path).names)
    available_cols = [col for col in PLAYER_ID_NAME_COLS if col in raw_columns]
    if not available_cols:
        return name_map

    raw = pd.read_parquet(raw_path, columns=available_cols)
    for id_col, name_col in [
        ("player1_id", "player1_name"),
        ("player2_id", "player2_name"),
        ("player3_id", "player3_name"),
    ]:
        if id_col not in raw.columns or name_col not in raw.columns:
            continue
        subset = raw[[id_col, name_col]].dropna()
        if subset.empty:
            continue
        subset[id_col] = pd.to_numeric(subset[id_col], errors="coerce")
        subset = subset.dropna(subset=[id_col])
        subset[name_col] = subset[name_col].astype(str).str.strip()
        subset = subset[subset[name_col] != ""]
        for player_id, player_name in subset.drop_duplicates(subset=[id_col], keep="last").itertuples(index=False):
            name_map[int(player_id)] = player_name
    return name_map


def infer_game_team_map_from_raw(df: pd.DataFrame) -> pd.DataFrame:
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
    return pivot.rename(columns={"Home": "home_team_abbr", "Away": "away_team_abbr"})


def build_player_name_map(season: int, raw_path: Path) -> dict[int, str]:
    name_map = load_name_map(season_to_suffix(season))
    name_map.update(extract_raw_name_map(raw_path))
    if AUTOCOMPLETE_MAP.exists():
        ac = pd.read_csv(AUTOCOMPLETE_MAP)
        if {"nba_id", "player_name"}.issubset(ac.columns):
            for row in ac.dropna(subset=["nba_id", "player_name"]).itertuples(index=False):
                name_map.setdefault(int(row.nba_id), str(row.player_name))
    return name_map


def melt_side_presence(df: pd.DataFrame, side: str, season: int, season_type: str) -> pd.DataFrame:
    player_cols = HOME_COLS if side == "home" else AWAY_COLS
    team_col = "home_team_abbr" if side == "home" else "away_team_abbr"
    side_frame = df[["game_id", team_col, *player_cols]].copy()
    side_frame = side_frame.rename(columns={team_col: "raw_team_abbreviation"})
    melted = side_frame.melt(
        id_vars=["game_id", "raw_team_abbreviation"],
        value_vars=player_cols,
        value_name="nba_id",
    )
    melted["nba_id"] = pd.to_numeric(melted["nba_id"], errors="coerce")
    melted = melted.dropna(subset=["nba_id", "raw_team_abbreviation"])
    melted["nba_id"] = melted["nba_id"].astype(int)
    melted = melted[melted["nba_id"] > 0]
    melted["team_id"] = melted["raw_team_abbreviation"].map(TEAM_ABBR_TO_ID)
    melted = melted.dropna(subset=["team_id"])
    melted["team_id"] = melted["team_id"].astype(int)
    melted["team_abbreviation"] = melted["team_id"].map(TEAM_ID_TO_CURRENT_ABBR)
    melted["season"] = season
    melted["season_type"] = season_type
    melted["league"] = "nba"
    return melted


def build_presence_for_file(season: int, season_type: str, raw_path: Path) -> pd.DataFrame:
    if not raw_path.exists():
        print(f"[skip] Missing raw file: {raw_path}")
        return pd.DataFrame()

    raw_columns = pq.read_schema(raw_path).names
    missing_cols = [col for col in RAW_REQUIRED_COLS if col not in raw_columns]
    if missing_cols:
        raise ValueError(f"{raw_path} is missing required columns: {missing_cols}")

    df = pd.read_parquet(raw_path, columns=RAW_REQUIRED_COLS)
    team_map = infer_game_team_map_from_raw(df)
    df = df.merge(team_map, on="game_id", how="left")
    df = df.dropna(subset=["home_team_abbr", "away_team_abbr"])

    name_map = build_player_name_map(season, raw_path)
    presence = pd.concat(
        [
            melt_side_presence(df, "home", season, season_type),
            melt_side_presence(df, "away", season, season_type),
        ],
        ignore_index=True,
    )
    if presence.empty:
        return presence

    grouped = (
        presence.groupby(
            ["league", "season", "season_type", "team_id", "team_abbreviation", "nba_id"],
            as_index=False,
        )
        .agg(games=("game_id", "nunique"), lineup_rows=("game_id", "size"))
        .sort_values(["season", "season_type", "team_abbreviation", "nba_id"])
    )
    grouped["player_name"] = grouped["nba_id"].map(name_map)
    return grouped[
        [
            "league",
            "season",
            "season_type",
            "team_id",
            "team_abbreviation",
            "nba_id",
            "player_name",
            "games",
            "lineup_rows",
        ]
    ]


def build_presence(start_year: int, end_year: int, season_types: Iterable[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for season in range(start_year, end_year + 1):
        for season_type in season_types:
            raw_path = get_raw_path(season, season_type)
            season_frame = build_presence_for_file(season, season_type, raw_path)
            if not season_frame.empty:
                print(
                    f"[ok] {season} {season_type}: "
                    f"{len(season_frame):,} team-player rows from {raw_path.name}"
                )
                frames.append(season_frame)
    if not frames:
        return pd.DataFrame(
            columns=[
                "league",
                "season",
                "season_type",
                "team_id",
                "team_abbreviation",
                "nba_id",
                "player_name",
                "games",
                "lineup_rows",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def get_supabase_client():
    load_env_file(PROJECT_ROOT / ".env")
    url = os.getenv("SUPABASE_URL")
    key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_SERVICE_KEY")
        or os.getenv("SUPABASE_KEY")
    )
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required for upload")

    from supabase import create_client

    return create_client(url, key)


def upload_presence(df: pd.DataFrame, batch_size: int = 1000, clear_existing: bool = True) -> None:
    if df.empty:
        print("[upload] No presence rows to upload")
        return

    client = get_supabase_client()
    seasons = sorted(df["season"].unique().tolist())
    season_types = sorted(df["season_type"].unique().tolist())

    if clear_existing:
        for season in seasons:
            for season_type in season_types:
                client.table("wowy_team_player_presence").delete().eq("league", "nba").eq(
                    "season", int(season)
                ).eq("season_type", str(season_type)).execute()
        print(f"[upload] Cleared existing rows for seasons={seasons} season_types={season_types}")

    records = df.where(pd.notna(df), None).to_dict(orient="records")
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        client.table("wowy_team_player_presence").upsert(
            batch,
            on_conflict="league,season,season_type,team_id,nba_id",
        ).execute()
        print(f"[upload] Upserted {min(start + batch_size, len(records)):,}/{len(records):,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument(
        "--season-types",
        choices=["RS", "PS", "ALL"],
        default="ALL",
        help="Presence season type source to build.",
    )
    parser.add_argument("--output", type=Path, help="Optional CSV output path for validation.")
    parser.add_argument("--upload", action="store_true", help="Upload to Supabase.")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear existing rows before upload.")
    parser.add_argument("--batch-size", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_year = normalize_season(args.start_year)
    end_year = normalize_season(args.end_year)
    if start_year > end_year:
        raise ValueError("--start-year must be <= --end-year")

    season_types = ["RS", "PS"] if args.season_types == "ALL" else [args.season_types]
    presence = build_presence(start_year, end_year, season_types)
    print(
        f"[summary] Built {len(presence):,} rows, "
        f"{presence['team_id'].nunique() if not presence.empty else 0} teams, "
        f"{presence['nba_id'].nunique() if not presence.empty else 0} players"
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        presence.to_csv(args.output, index=False)
        print(f"[write] {args.output}")

    if args.upload:
        upload_presence(presence, batch_size=args.batch_size, clear_existing=not args.no_clear)


if __name__ == "__main__":
    main()
