#!/usr/bin/env python3
"""
Build a Basketball Reference FT% lookup keyed by NBA player id and season.

Input stat CSVs should be Basketball Reference per-game exports with:
  - Player
  - Team
  - FT%
  - Player-additional

The Gabriel site_Data index maps BRef slugs (`bref_id`) to NBA ids. When a
player has multiple BRef rows in a season, prefer the aggregate row (`TOT`,
`2TM`, `3TM`, etc.) and otherwise keep the largest-games row.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_ROOT.parent
DEFAULT_STATS_DIR = PROJECT_ROOT
DEFAULT_INDEX = PIPELINE_ROOT / "external" / "site_Data" / "index_master.csv"
DEFAULT_OUTPUT = PIPELINE_ROOT / "external" / "bref_ft_pct_1997_2000.csv"


AGGREGATE_TEAMS = {"TOT", "2TM", "3TM", "4TM", "5TM"}


def parse_years(value: str) -> list[int]:
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


def read_bref_stats(path: Path, year: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(col).strip() for col in df.columns]
    required = {"Player", "Team", "FT%", "Player-additional"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    df = df[df["Player-additional"].notna()].copy()
    df = df[df["Player-additional"].astype(str).ne("Player-additional")]
    df["year"] = int(year)
    df["bref_id"] = df["Player-additional"].astype(str).str.strip()
    df["team"] = df["Team"].astype(str).str.strip()
    df["ft_pct"] = pd.to_numeric(df["FT%"], errors="coerce")
    df["games"] = pd.to_numeric(df.get("G"), errors="coerce")
    df["is_aggregate_team"] = df["team"].isin(AGGREGATE_TEAMS)
    return df


def choose_player_season_rows(stats: pd.DataFrame) -> pd.DataFrame:
    ordered = stats.sort_values(
        ["year", "bref_id", "is_aggregate_team", "games"],
        ascending=[True, True, False, False],
        kind="stable",
    )
    return ordered.drop_duplicates(["year", "bref_id"], keep="first").copy()


def build_lookup(stats_dir: Path, index_path: Path, years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not index_path.exists():
        raise FileNotFoundError(
            f"Gabriel index not found at {index_path}. "
            "Download index_master.csv from gabriel1200/site_Data first."
        )

    frames = []
    for year in years:
        path = stats_dir / f"{year}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing BRef stats CSV: {path}")
        frames.append(read_bref_stats(path, year))

    stats = choose_player_season_rows(pd.concat(frames, ignore_index=True, sort=False))

    index = pd.read_csv(index_path, usecols=["player", "year", "team", "bref_id", "nba_id"])
    index["year"] = pd.to_numeric(index["year"], errors="coerce").astype("Int64")
    index["bref_id"] = index["bref_id"].astype(str).str.strip()
    index["nba_id"] = pd.to_numeric(index["nba_id"], errors="coerce")
    index = index.dropna(subset=["year", "bref_id", "nba_id"]).copy()
    index["year"] = index["year"].astype(int)
    index["nba_id"] = index["nba_id"].astype(int)
    id_map = (
        index[index["year"].isin(years)]
        .sort_values(["year", "bref_id", "team"], kind="stable")
        .drop_duplicates(["year", "bref_id", "nba_id"], keep="first")
    )

    conflicts = (
        id_map.groupby(["year", "bref_id"])["nba_id"]
        .nunique()
        .reset_index(name="nba_id_count")
        .query("nba_id_count > 1")
    )
    if not conflicts.empty:
        raise ValueError(f"Conflicting NBA ids for year+bref_id:\n{conflicts.to_string(index=False)}")

    id_map = id_map.drop_duplicates(["year", "bref_id"], keep="first")
    merged = stats.merge(
        id_map[["year", "bref_id", "nba_id"]],
        on=["year", "bref_id"],
        how="left",
        validate="one_to_one",
    )

    missing = merged[merged["nba_id"].isna()].copy()
    output = merged[merged["nba_id"].notna()].copy()
    output["nba_id"] = output["nba_id"].astype(int)
    output = output.rename(
        columns={
            "Player": "player_name",
            "FT%": "FT_PERC",
        }
    )
    output["FTPerc"] = output["ft_pct"]
    output = output[
        [
            "nba_id",
            "year",
            "player_name",
            "bref_id",
            "team",
            "FT_PERC",
            "FTPerc",
            "games",
            "is_aggregate_team",
        ]
    ].sort_values(["year", "player_name", "nba_id"], kind="stable")
    return output, missing[["year", "Player", "bref_id", "team", "FT%"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build historical BRef FT% lookup.")
    parser.add_argument("--years", default="1997-2000")
    parser.add_argument("--stats-dir", type=Path, default=DEFAULT_STATS_DIR)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    years = parse_years(args.years)
    output, missing = build_lookup(args.stats_dir, args.index, years)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output)} rows to {args.output}")
    print(f"Missing NBA ids: {len(missing)}")
    if not missing.empty:
        print(missing.to_string(index=False))
    print(f"Aggregate/team-total rows selected: {int(output['is_aggregate_team'].sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
