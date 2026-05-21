#!/usr/bin/env python3
"""Audit converted PBP files for player IDs appearing on both lineup sides.

This is intentionally linear in the number of raw event rows: each row expands
to at most ten lineup-slot records, then pandas groupbys identify conflicts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


HOME_COLS = [f"home_player{i}" for i in range(1, 6)]
AWAY_COLS = [f"away_player{i}" for i in range(1, 6)]
CONTEXT_COLS = [
    "game_id",
    "period",
    "event_num",
    "time_quarter",
    "event_type",
    "home_description",
    "visitor_description",
    "score",
]


def lineup_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    missing = [col for col in HOME_COLS + AWAY_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing lineup columns: {missing}")
    return HOME_COLS, AWAY_COLS


def build_long_lineup_table(df: pd.DataFrame) -> pd.DataFrame:
    home_cols, away_cols = lineup_columns(df)
    parts = []
    for side, cols in (("home", home_cols), ("away", away_cols)):
        side_df = df[["row_id", "game_id"] + cols].melt(
            id_vars=["row_id", "game_id"],
            value_vars=cols,
            var_name="slot",
            value_name="player_id",
        )
        side_df["side"] = side
        parts.append(side_df)

    long = pd.concat(parts, ignore_index=True)
    long["player_id"] = pd.to_numeric(long["player_id"], errors="coerce")
    long = long[long["player_id"].notna() & (long["player_id"] != 0)].copy()
    long["player_id"] = long["player_id"].astype(np.int64)
    return long[["row_id", "game_id", "side", "slot", "player_id"]]


def add_lineup_strings(df: pd.DataFrame, row_ids: pd.Series | np.ndarray | None = None) -> pd.DataFrame:
    def lineup_string(row: pd.Series, cols: list[str]) -> str:
        vals = pd.to_numeric(row[cols], errors="coerce").dropna()
        vals = vals[vals != 0].astype(int)
        return "-".join(str(value) for value in sorted(vals.tolist()))

    if row_ids is not None:
        wanted = set(pd.Series(row_ids).astype(np.int64).tolist())
        df = df[df["row_id"].isin(wanted)].copy()
    else:
        df = df.copy()
    df["home_lineup"] = df.apply(lambda row: lineup_string(row, HOME_COLS), axis=1)
    df["away_lineup"] = df.apply(lambda row: lineup_string(row, AWAY_COLS), axis=1)
    return df


def find_row_collisions(df: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    side_counts = (
        long.groupby(["row_id", "player_id"], observed=True)["side"]
        .nunique()
        .reset_index(name="side_count")
    )
    collisions = side_counts[side_counts["side_count"] > 1][["row_id", "player_id"]]
    if collisions.empty:
        return pd.DataFrame()

    context = add_lineup_strings(df, collisions["row_id"])
    keep_cols = ["row_id"] + [col for col in CONTEXT_COLS if col in context.columns] + [
        "home_lineup",
        "away_lineup",
    ]
    return collisions.merge(context[keep_cols], on="row_id", how="left")


def find_side_stability_flips(long: pd.DataFrame, threshold: float) -> pd.DataFrame:
    row_side = long.drop_duplicates(["row_id", "game_id", "player_id", "side"])
    counts = (
        row_side.groupby(["game_id", "player_id", "side"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for side in ("home", "away"):
        if side not in counts.columns:
            counts[side] = 0
    counts = counts.rename(columns={"home": "home_rows", "away": "away_rows"})
    counts["total_rows"] = counts["home_rows"] + counts["away_rows"]
    counts["dominant_side"] = np.where(counts["home_rows"] >= counts["away_rows"], "home", "away")
    counts["dominant_rows"] = counts[["home_rows", "away_rows"]].max(axis=1)
    counts["minority_rows"] = counts[["home_rows", "away_rows"]].min(axis=1)
    counts["dominant_share"] = counts["dominant_rows"] / counts["total_rows"]
    suspicious = counts[(counts["minority_rows"] > 0) & (counts["dominant_share"] >= threshold)]
    return suspicious.sort_values(["minority_rows", "total_rows"], ascending=False)


def find_side_stability_examples(
    df: pd.DataFrame,
    long: pd.DataFrame,
    suspicious: pd.DataFrame,
    max_examples_per_player: int,
) -> pd.DataFrame:
    if suspicious.empty:
        return pd.DataFrame()

    examples = suspicious[["game_id", "player_id", "dominant_side", "dominant_share", "minority_rows"]].copy()
    examples["minority_side"] = np.where(examples["dominant_side"] == "home", "away", "home")
    minority_hits = long.merge(
        examples,
        left_on=["game_id", "player_id", "side"],
        right_on=["game_id", "player_id", "minority_side"],
        how="inner",
    )
    minority_hits = minority_hits.drop_duplicates(["row_id", "player_id"])
    minority_hits["example_rank"] = minority_hits.groupby(["game_id", "player_id"]).cumcount() + 1
    minority_hits = minority_hits[minority_hits["example_rank"] <= max_examples_per_player]

    context = add_lineup_strings(df, minority_hits["row_id"])
    keep_cols = ["row_id"] + [col for col in CONTEXT_COLS if col in context.columns] + [
        "home_lineup",
        "away_lineup",
    ]
    return minority_hits.merge(context[keep_cols], on="row_id", how="left")


def audit_file(path: Path, output_dir: Path, threshold: float, max_examples: int) -> dict[str, int | str]:
    df = pd.read_parquet(path).reset_index(drop=True)
    df.insert(0, "row_id", np.arange(len(df), dtype=np.int64))
    long = build_long_lineup_table(df)

    collisions = find_row_collisions(df, long)
    suspicious = find_side_stability_flips(long, threshold)
    examples = find_side_stability_examples(df, long, suspicious, max_examples)

    stem = path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    collisions.to_csv(output_dir / f"{stem}_same_id_home_away_collisions.csv", index=False)
    suspicious.to_csv(output_dir / f"{stem}_player_side_stability_suspicious.csv", index=False)
    examples.to_csv(output_dir / f"{stem}_player_side_stability_examples.csv", index=False)

    return {
        "file": str(path),
        "rows": int(len(df)),
        "long_lineup_rows": int(len(long)),
        "row_collisions": int(len(collisions)),
        "rows_with_collision": int(collisions["row_id"].nunique()) if not collisions.empty else 0,
        "side_stability_flips": int(len(suspicious)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Converted raw PBP parquet files to audit.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nba_pipeline/validation/lineup_side_conflicts"),
        help="Directory for collision and side-stability CSV outputs.",
    )
    parser.add_argument(
        "--dominant-threshold",
        type=float,
        default=0.98,
        help="Flag game/player IDs with at least this share on one side and any rows on the other side.",
    )
    parser.add_argument(
        "--max-examples-per-player",
        type=int,
        default=10,
        help="Maximum minority-side example rows to write per game/player.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = [
        audit_file(path, args.output_dir, args.dominant_threshold, args.max_examples_per_player)
        for path in args.paths
    ]
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
