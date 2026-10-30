#!/usr/bin/env python3
"""
Build possession-weighted DARKO teammate-help artifacts for 1997-present NBA play.

For every valid five-player team exposure in processed RAPM possessions, each
player's teammate help is the sum of the other four players' DARKO values. The
canonical descriptive build applies each player's same-season DARKO value to
every exposure in that season, including games before the row's source date.
This deliberately uses hindsight to represent season-level teammate talent.
Strict game-date as-of lookup remains available as an audit mode.

The pair contribution contract is exact on fully rated exposures:

    season teammate help
      = sum(teammate weighted DARKO / anchor rated exposures)

This holds independently for DPM, offensive DPM, and defensive DPM.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_DIR.parent
PROCESSED_DIR = PIPELINE_DIR / "processed"
DEFAULT_CACHE_DIR = PIPELINE_DIR / ".cache" / "teammate_help"
DEFAULT_DARKO_PATH = DEFAULT_CACHE_DIR / "dpm_history.csv"
DEFAULT_OUTPUT_DIR = PIPELINE_DIR / "results" / "darko_teammate_help"
AUTOCOMPLETE_MAP = PROJECT_ROOT / "autocomplete_map.csv"
DARKO_HISTORY_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1mhwOLqPu2F9026EQiVxFPIN1t9RGafGpl-dokaIsm9c/"
    "export?format=csv&gid=142925152"
)

LINEUP_COLS = [f"O{i}" for i in range(1, 6)]
OPPONENT_COLS = [f"D{i}" for i in range(1, 6)]
METRICS = ("dpm", "o_dpm", "d_dpm")
METRIC_LABELS = {
    "dpm": "help_dpm",
    "o_dpm": "help_o_dpm",
    "d_dpm": "help_d_dpm",
}
CAREER_SCHEDULES = {
    "rs": {"rs"},
    "ps": {"ps"},
    "combined": {"rs", "ps"},
}
CAREER_PAIR_SHARDS = 64


def season_to_year(season: int) -> int:
    return season if season >= 100 else 2000 + season if season < 70 else 1900 + season


def parse_rapm_path(path: Path) -> tuple[int, str] | None:
    match = re.fullmatch(r"RAPM(\d{2})(_PS)?\.parquet", path.name)
    if not match:
        return None
    return season_to_year(int(match.group(1))), "ps" if match.group(2) else "rs"


def selected_rapm_files(
    processed_dir: Path,
    start_year: int,
    end_year: int,
    season_types: set[str],
) -> list[Path]:
    selected: list[Path] = []
    for path in processed_dir.glob("RAPM*.parquet"):
        parsed = parse_rapm_path(path)
        if parsed is None:
            continue
        season, season_type = parsed
        if start_year <= season <= end_year and season_type in season_types:
            selected.append(path)
    return sorted(selected, key=lambda path: parse_rapm_path(path) or (0, ""))


@dataclass(frozen=True)
class PlayerRatingHistory:
    dates: np.ndarray
    seasons: np.ndarray
    values: np.ndarray


def load_darko_history(path_or_url: str | Path) -> tuple[pd.DataFrame, dict[int, PlayerRatingHistory]]:
    usecols = ["nba_id", "player_name", "season", "date", *METRICS]
    darko = pd.read_csv(path_or_url, usecols=usecols, low_memory=False)
    darko["nba_id"] = pd.to_numeric(darko["nba_id"], errors="coerce")
    darko["season"] = pd.to_numeric(darko["season"], errors="coerce")
    darko["date"] = pd.to_datetime(darko["date"], errors="coerce").dt.normalize()
    for metric in METRICS:
        darko[metric] = pd.to_numeric(darko[metric], errors="coerce")
    darko = darko.dropna(subset=["nba_id", "season", "date"]).copy()
    darko["nba_id"] = darko["nba_id"].astype(np.int64)
    darko["season"] = darko["season"].astype(int)
    darko = darko.sort_values(["nba_id", "date", "season"], kind="mergesort")
    darko = darko.drop_duplicates(["nba_id", "date"], keep="last")

    histories: dict[int, PlayerRatingHistory] = {}
    for player_id, group in darko.groupby("nba_id", sort=False):
        histories[int(player_id)] = PlayerRatingHistory(
            dates=group["date"].to_numpy(dtype="datetime64[D]"),
            seasons=group["season"].to_numpy(dtype=np.int16),
            values=group[list(METRICS)].to_numpy(dtype=float),
        )
    return darko, histories


def lookup_asof_ratings(
    player_ids: np.ndarray,
    game_dates: np.ndarray,
    histories: dict[int, PlayerRatingHistory],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return backward-looking metric values, source dates, and source seasons."""
    player_ids = np.asarray(player_ids, dtype=np.int64)
    game_dates = np.asarray(game_dates, dtype="datetime64[D]")
    values = np.full((len(player_ids), len(METRICS)), np.nan, dtype=float)
    source_dates = np.full(len(player_ids), np.datetime64("NaT"), dtype="datetime64[D]")
    source_seasons = np.full(len(player_ids), -1, dtype=np.int16)

    for player_id in np.unique(player_ids):
        history = histories.get(int(player_id))
        if history is None or player_id <= 0:
            continue
        positions = np.flatnonzero(player_ids == player_id)
        indexes = np.searchsorted(history.dates, game_dates[positions], side="right") - 1
        found = indexes >= 0
        if not np.any(found):
            continue
        target = positions[found]
        selected = indexes[found]
        values[target] = history.values[selected]
        source_dates[target] = history.dates[selected]
        source_seasons[target] = history.seasons[selected]
    return values, source_dates, source_seasons


def lookup_same_season_ratings(
    player_ids: np.ndarray,
    season_end_year: int,
    histories: dict[int, PlayerRatingHistory],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return each player's DARKO row for the requested season, regardless of date."""
    player_ids = np.asarray(player_ids, dtype=np.int64)
    values = np.full((len(player_ids), len(METRICS)), np.nan, dtype=float)
    source_dates = np.full(len(player_ids), np.datetime64("NaT"), dtype="datetime64[D]")
    source_seasons = np.full(len(player_ids), -1, dtype=np.int16)

    for player_id in np.unique(player_ids):
        history = histories.get(int(player_id))
        if history is None or player_id <= 0:
            continue
        matches = np.flatnonzero(history.seasons == season_end_year)
        if not len(matches):
            continue
        selected = int(matches[-1])
        positions = np.flatnonzero(player_ids == player_id)
        values[positions] = history.values[selected]
        source_dates[positions] = history.dates[selected]
        source_seasons[positions] = history.seasons[selected]
    return values, source_dates, source_seasons


def collapse_team_exposures(possessions: pd.DataFrame) -> pd.DataFrame:
    """Collapse offense and defense lineup appearances to dated team exposures."""
    required = {"game_date", *LINEUP_COLS, *OPPONENT_COLS}
    missing = required - set(possessions.columns)
    if missing:
        raise ValueError(f"Processed possessions missing columns: {sorted(missing)}")

    dates = pd.to_datetime(possessions["game_date"], errors="coerce").dt.normalize()
    frames: list[pd.DataFrame] = []
    for columns in (LINEUP_COLS, OPPONENT_COLS):
        numeric = possessions[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        finite = np.isfinite(numeric).all(axis=1)
        ids = np.where(np.isfinite(numeric), numeric, -1).astype(np.int64)
        ids.sort(axis=1)
        valid_ids = (ids > 0).all(axis=1)
        unique_ids = np.apply_along_axis(lambda row: len(set(row.tolist())) == 5, 1, ids)
        valid = dates.notna().to_numpy() & finite & valid_ids & unique_ids
        if not np.any(valid):
            continue
        side = pd.DataFrame(ids[valid], columns=[f"P{i}" for i in range(1, 6)])
        side.insert(0, "game_date", dates.loc[valid].to_numpy())
        side["exposures"] = 1.0
        frames.append(side)

    if not frames:
        return pd.DataFrame(columns=["game_date", "P1", "P2", "P3", "P4", "P5", "exposures"])

    return (
        pd.concat(frames, ignore_index=True)
        .groupby(["game_date", "P1", "P2", "P3", "P4", "P5"], as_index=False, sort=False)[
            "exposures"
        ]
        .sum()
    )


def summarize_lineup_exposures(
    exposures: pd.DataFrame,
    histories: dict[int, PlayerRatingHistory],
    season_end_year: int | None = None,
    rating_timing: str = "same-season",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summarize player and oriented-pair help for one season/season type.

    A player's rated exposure requires all four teammates to have all three
    DARKO components available. The anchor player's own rating may be missing.
    """
    if exposures.empty:
        return pd.DataFrame(), pd.DataFrame()

    player_cols = [f"P{i}" for i in range(1, 6)]
    lineups = exposures[player_cols].to_numpy(dtype=np.int64)
    game_dates = exposures["game_date"].to_numpy(dtype="datetime64[D]")
    weights = exposures["exposures"].to_numpy(dtype=float)
    if season_end_year is None:
        calendar_years = pd.DatetimeIndex(game_dates).year.to_numpy()
        calendar_months = pd.DatetimeIndex(game_dates).month.to_numpy()
        inferred = calendar_years + (calendar_months >= 7).astype(int)
        season_end_year = int(pd.Series(inferred).mode().iloc[0])
    if rating_timing not in {"same-season", "asof"}:
        raise ValueError(f"Unknown rating timing: {rating_timing}")

    rating_values = np.empty((len(exposures), 5, len(METRICS)), dtype=float)
    source_dates = np.empty((len(exposures), 5), dtype="datetime64[D]")
    source_seasons = np.empty((len(exposures), 5), dtype=np.int16)
    for slot in range(5):
        if rating_timing == "same-season":
            values, dates, seasons = lookup_same_season_ratings(
                lineups[:, slot], season_end_year, histories
            )
        else:
            values, dates, seasons = lookup_asof_ratings(
                lineups[:, slot], game_dates, histories
            )
        rating_values[:, slot, :] = values
        source_dates[:, slot] = dates
        source_seasons[:, slot] = seasons

    complete_player_rating = np.isfinite(rating_values).all(axis=2)
    missing_count = (~complete_player_rating).sum(axis=1)
    finite_values = np.where(np.isfinite(rating_values), rating_values, 0.0)
    lineup_sums = finite_values.sum(axis=1)

    player_parts: list[pd.DataFrame] = []
    pair_parts: list[pd.DataFrame] = []
    for anchor_slot in range(5):
        anchors = lineups[:, anchor_slot]
        anchor_is_missing = ~complete_player_rating[:, anchor_slot]
        anchor_valid = missing_count - anchor_is_missing.astype(int) == 0
        help_values = lineup_sums - finite_values[:, anchor_slot, :]

        if rating_timing == "asof":
            teammate_source_dates = np.delete(source_dates, anchor_slot, axis=1)
            source_lag_days = (
                game_dates[:, None] - teammate_source_dates
            ).astype("timedelta64[D]").astype(float)
            mean_lag = np.nanmean(source_lag_days, axis=1)
            prior_snapshot = np.any(
                np.delete(source_seasons, anchor_slot, axis=1)
                < season_end_year,
                axis=1,
            )
            lag_is_defined = anchor_valid
        else:
            mean_lag = np.zeros(len(exposures), dtype=float)
            prior_snapshot = np.zeros(len(exposures), dtype=bool)
            lag_is_defined = np.zeros(len(exposures), dtype=bool)

        player_part = pd.DataFrame(
            {
                "player_id": anchors,
                "total_exposures": weights,
                "rated_exposures": np.where(anchor_valid, weights, 0.0),
                "weighted_help_dpm": np.where(anchor_valid, weights * help_values[:, 0], 0.0),
                "weighted_help_o_dpm": np.where(anchor_valid, weights * help_values[:, 1], 0.0),
                "weighted_help_d_dpm": np.where(anchor_valid, weights * help_values[:, 2], 0.0),
                "weighted_rating_lag_days": np.where(lag_is_defined, weights * mean_lag, 0.0),
                "rating_lag_exposures": np.where(lag_is_defined, weights, 0.0),
                "prior_snapshot_exposures": np.where(anchor_valid & prior_snapshot, weights, 0.0),
            }
        )
        player_parts.append(player_part)

        for teammate_slot in range(5):
            if teammate_slot == anchor_slot:
                continue
            pair_parts.append(
                pd.DataFrame(
                    {
                        "player_id": anchors,
                        "teammate_id": lineups[:, teammate_slot],
                        "shared_exposures": weights,
                        "rated_exposures": np.where(anchor_valid, weights, 0.0),
                        "weighted_dpm": np.where(
                            anchor_valid, weights * rating_values[:, teammate_slot, 0], 0.0
                        ),
                        "weighted_o_dpm": np.where(
                            anchor_valid, weights * rating_values[:, teammate_slot, 1], 0.0
                        ),
                        "weighted_d_dpm": np.where(
                            anchor_valid, weights * rating_values[:, teammate_slot, 2], 0.0
                        ),
                    }
                )
            )

    player = (
        pd.concat(player_parts, ignore_index=True)
        .groupby("player_id", as_index=False, sort=False)
        .sum(numeric_only=True)
    )
    pair = (
        pd.concat(pair_parts, ignore_index=True)
        .groupby(["player_id", "teammate_id"], as_index=False, sort=False)
        .sum(numeric_only=True)
    )

    for metric in METRICS:
        weighted = f"weighted_{metric}"
        help_col = METRIC_LABELS[metric]
        if metric == "dpm":
            weighted = "weighted_help_dpm"
        elif metric == "o_dpm":
            weighted = "weighted_help_o_dpm"
        else:
            weighted = "weighted_help_d_dpm"
        player[help_col] = player[weighted] / player["rated_exposures"].replace(0, np.nan)

        pair_weighted = f"weighted_{metric}"
        pair[f"avg_teammate_{metric}"] = pair[pair_weighted] / pair["rated_exposures"].replace(
            0, np.nan
        )
        pair[f"contribution_{metric}"] = pair[pair_weighted]

    player["coverage_pct"] = 100.0 * player["rated_exposures"] / player["total_exposures"]
    player["avg_rating_lag_days"] = (
        player["weighted_rating_lag_days"] / player["rating_lag_exposures"].replace(0, np.nan)
    )
    player["prior_snapshot_pct"] = (
        100.0 * player["prior_snapshot_exposures"] / player["rated_exposures"].replace(0, np.nan)
    )

    rated_denominator = player.set_index("player_id")["rated_exposures"]
    for metric in METRICS:
        pair[f"contribution_{metric}"] = pair[f"contribution_{metric}"] / pair[
            "player_id"
        ].map(rated_denominator).replace(0, np.nan)
    pair["share_of_player_exposures_pct"] = (
        100.0
        * pair["shared_exposures"]
        / pair["player_id"].map(player.set_index("player_id")["total_exposures"]).replace(0, np.nan)
    )
    return player, pair


def load_name_map(darko: pd.DataFrame) -> dict[int, str]:
    names: dict[int, str] = {}
    for row in darko.dropna(subset=["player_name"]).itertuples(index=False):
        names[int(row.nba_id)] = str(row.player_name)
    if AUTOCOMPLETE_MAP.exists():
        autocomplete = pd.read_csv(AUTOCOMPLETE_MAP)
        if {"nba_id", "player_name"}.issubset(autocomplete.columns):
            for row in autocomplete.dropna(subset=["nba_id", "player_name"]).itertuples(index=False):
                names.setdefault(int(row.nba_id), str(row.player_name))
    return names


def attach_names(frame: pd.DataFrame, names: dict[int, str]) -> pd.DataFrame:
    frame = frame.copy()
    frame.insert(
        1,
        "player_name",
        frame["player_id"].map(names).fillna(frame["player_id"].map(lambda value: f"ID {value}")),
    )
    if "teammate_id" in frame:
        teammate_position = frame.columns.get_loc("teammate_id") + 1
        frame.insert(
            teammate_position,
            "teammate_name",
            frame["teammate_id"]
            .map(names)
            .fillna(frame["teammate_id"].map(lambda value: f"ID {value}")),
        )
    return frame


def add_ranks(player: pd.DataFrame) -> pd.DataFrame:
    player = player.copy()
    group_cols = ["season", "season_type"]
    for metric in METRICS:
        help_col = METRIC_LABELS[metric]
        player[f"rank_{metric}"] = player.groupby(group_cols)[help_col].rank(
            method="min", ascending=False, na_option="bottom"
        )
        player[f"percentile_{metric}"] = player.groupby(group_cols)[help_col].rank(
            method="average", pct=True, ascending=True, na_option="bottom"
        )
    return player


def validate_reconciliation(player: pd.DataFrame, pair: pd.DataFrame, tolerance: float = 1e-10) -> dict:
    merged = player[["season", "season_type", "player_id", *METRIC_LABELS.values()]].copy()
    sums = (
        pair.groupby(["season", "season_type", "player_id"], as_index=False)[
            [f"contribution_{metric}" for metric in METRICS]
        ]
        .sum(min_count=1)
    )
    merged = merged.merge(sums, on=["season", "season_type", "player_id"], how="left")
    report: dict[str, float | int] = {"rows": int(len(merged))}
    for metric in METRICS:
        gap = (
            merged[METRIC_LABELS[metric]] - merged[f"contribution_{metric}"]
        ).abs()
        max_gap = float(gap.max(skipna=True)) if gap.notna().any() else 0.0
        report[f"max_{metric}_gap"] = max_gap
        if max_gap > tolerance:
            raise AssertionError(f"{metric} pair contribution gap {max_gap} exceeds {tolerance}")
    return report


def build_career_pair_summary(
    player: pd.DataFrame,
    pair: pd.DataFrame,
    schedule: str,
) -> pd.DataFrame:
    """Aggregate teammate exposure and DARKO contribution over a career schedule."""
    if schedule not in CAREER_SCHEDULES:
        raise ValueError(f"Unknown career schedule: {schedule}")
    season_types = CAREER_SCHEDULES[schedule]
    scoped_player = player[player["season_type"].isin(season_types)]
    scoped_pair = pair[pair["season_type"].isin(season_types)]
    if scoped_player.empty or scoped_pair.empty:
        return pd.DataFrame()

    player_totals = scoped_player.groupby("player_id", sort=False).agg(
        player_total_exposures=("total_exposures", "sum"),
        player_rated_exposures=("rated_exposures", "sum"),
    )
    career_pair = (
        scoped_pair.groupby(["player_id", "teammate_id"], as_index=False, sort=False)
        .agg(
            teammate_name=("teammate_name", "last"),
            shared_exposures=("shared_exposures", "sum"),
            rated_exposures=("rated_exposures", "sum"),
            weighted_dpm=("weighted_dpm", "sum"),
            weighted_o_dpm=("weighted_o_dpm", "sum"),
            weighted_d_dpm=("weighted_d_dpm", "sum"),
        )
    )
    total_exposures = career_pair["player_id"].map(
        player_totals["player_total_exposures"]
    )
    rated_exposures = career_pair["player_id"].map(
        player_totals["player_rated_exposures"]
    )
    career_pair["share_of_player_exposures_pct"] = (
        100.0 * career_pair["shared_exposures"] / total_exposures.replace(0, np.nan)
    )
    for metric in METRICS:
        career_pair[f"avg_teammate_{metric}"] = career_pair[
            f"weighted_{metric}"
        ] / career_pair["rated_exposures"].replace(0, np.nan)
        career_pair[f"contribution_{metric}"] = career_pair[
            f"weighted_{metric}"
        ] / rated_exposures.replace(0, np.nan)

    career_pair = career_pair.sort_values(
        ["player_id", "shared_exposures", "teammate_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    career_pair["frequency_rank"] = (
        career_pair.groupby("player_id", sort=False).cumcount() + 1
    )
    return career_pair


def write_app_json(
    player: pd.DataFrame,
    pair: pd.DataFrame,
    metadata: dict,
    output_path: Path,
) -> None:
    player_columns = [
        "season",
        "season_type",
        "player_id",
        "player_name",
        "total_exposures",
        "rated_exposures",
        "coverage_pct",
        "help_dpm",
        "help_o_dpm",
        "help_d_dpm",
        "rank_dpm",
        "rank_o_dpm",
        "rank_d_dpm",
        "percentile_dpm",
        "avg_rating_lag_days",
        "prior_snapshot_pct",
    ]
    pair_columns = [
        "season",
        "season_type",
        "player_id",
        "teammate_id",
        "teammate_name",
        "shared_exposures",
        "rated_exposures",
        "share_of_player_exposures_pct",
        "avg_teammate_dpm",
        "avg_teammate_o_dpm",
        "avg_teammate_d_dpm",
        "contribution_dpm",
        "contribution_o_dpm",
        "contribution_d_dpm",
    ]
    clean_player = player[player_columns].replace({np.nan: None})
    clean_pair = pair[pair_columns].replace({np.nan: None})
    payload = {
        "meta": metadata,
        "players": clean_player.to_dict(orient="records"),
        "teammates": clean_pair.to_dict(orient="records"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def _rounded_records(frame: pd.DataFrame, columns: list[str], decimals: int = 4) -> list[dict]:
    clean = frame[columns].copy()
    float_columns = clean.select_dtypes(include=["float", "float64", "float32"]).columns
    clean[float_columns] = clean[float_columns].round(decimals)
    clean = clean.replace({np.nan: None})
    return clean.to_dict(orient="records")


def write_split_app_data(
    player: pd.DataFrame,
    pair: pd.DataFrame,
    metadata: dict,
    output_dir: Path,
) -> None:
    """Write a small initial player index plus one lazily loaded pair file per season."""
    player_columns = [
        "season",
        "season_type",
        "player_id",
        "player_name",
        "total_exposures",
        "rated_exposures",
        "coverage_pct",
        "help_dpm",
        "help_o_dpm",
        "help_d_dpm",
        "rank_dpm",
        "rank_o_dpm",
        "rank_d_dpm",
        "percentile_dpm",
        "avg_rating_lag_days",
        "prior_snapshot_pct",
    ]
    pair_columns = [
        "player_id",
        "teammate_id",
        "teammate_name",
        "shared_exposures",
        "rated_exposures",
        "share_of_player_exposures_pct",
        "avg_teammate_dpm",
        "avg_teammate_o_dpm",
        "avg_teammate_d_dpm",
        "contribution_dpm",
        "contribution_o_dpm",
        "contribution_d_dpm",
        "frequency_rank",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir = output_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    career_pairs_dir = output_dir / "career-pairs"
    career_pairs_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, separators=(",", ":")), encoding="utf-8"
    )
    player_payload = json.dumps(
        _rounded_records(player, player_columns), separators=(",", ":")
    ).encode("utf-8")
    (output_dir / "players.json.gz").write_bytes(
        gzip.compress(player_payload, compresslevel=9, mtime=0)
    )
    pair_files: list[dict[str, object]] = []
    for (season, season_type), group in pair.groupby(["season", "season_type"], sort=True):
        filename = f"{int(season)}_{season_type}.json.gz"
        pair_payload = json.dumps(
            _rounded_records(group, pair_columns), separators=(",", ":")
        ).encode("utf-8")
        (pairs_dir / filename).write_bytes(
            gzip.compress(pair_payload, compresslevel=9, mtime=0)
        )
        pair_files.append(
            {
                "season": int(season),
                "season_type": str(season_type),
                "file": f"pairs/{filename}",
                "rows": int(len(group)),
            }
        )
    (output_dir / "pair_manifest.json").write_text(
        json.dumps(pair_files, separators=(",", ":")), encoding="utf-8"
    )
    career_pair_files: list[dict[str, object]] = []
    for schedule in CAREER_SCHEDULES:
        career_pair = build_career_pair_summary(player, pair, schedule)
        schedule_dir = career_pairs_dir / schedule
        schedule_dir.mkdir(parents=True, exist_ok=True)
        shard_rows: list[int] = []
        for shard in range(CAREER_PAIR_SHARDS):
            shard_frame = career_pair[
                career_pair["player_id"].mod(CAREER_PAIR_SHARDS) == shard
            ]
            pair_payload = json.dumps(
                _rounded_records(shard_frame, pair_columns), separators=(",", ":")
            ).encode("utf-8")
            (schedule_dir / f"{shard:02d}.json.gz").write_bytes(
                gzip.compress(pair_payload, compresslevel=9, mtime=0)
            )
            shard_rows.append(int(len(shard_frame)))
        career_pair_files.append(
            {
                "schedule": schedule,
                "directory": f"career-pairs/{schedule}",
                "rows": int(len(career_pair)),
                "shard_rows": shard_rows,
            }
        )
    (output_dir / "career_pair_manifest.json").write_text(
        json.dumps(
            {
                "shard_count": CAREER_PAIR_SHARDS,
                "schedules": career_pair_files,
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


def process_files(
    paths: Iterable[Path],
    histories: dict[int, PlayerRatingHistory],
    rating_timing: str = "same-season",
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    player_frames: list[pd.DataFrame] = []
    pair_frames: list[pd.DataFrame] = []
    file_reports: list[dict] = []
    columns = ["game_date", *LINEUP_COLS, *OPPONENT_COLS]
    for path in paths:
        parsed = parse_rapm_path(path)
        if parsed is None:
            continue
        season, season_type = parsed
        possessions = pd.read_parquet(path, columns=columns)
        date_coverage = float(pd.to_datetime(possessions["game_date"], errors="coerce").notna().mean())
        if date_coverage < 0.999:
            raise ValueError(
                f"{path} game_date coverage is {date_coverage:.2%}; refusing a leaky season-end fallback"
            )
        exposures = collapse_team_exposures(possessions)
        player, pair = summarize_lineup_exposures(
            exposures,
            histories,
            season_end_year=season,
            rating_timing=rating_timing,
        )
        player.insert(0, "season_type", season_type)
        player.insert(0, "season", season)
        pair.insert(0, "season_type", season_type)
        pair.insert(0, "season", season)
        player_frames.append(player)
        pair_frames.append(pair)
        file_reports.append(
            {
                "file": str(path),
                "season": season,
                "season_type": season_type,
                "possessions": int(len(possessions)),
                "team_exposures": int(exposures["exposures"].sum()),
                "lineup_date_groups": int(len(exposures)),
                "date_coverage_pct": 100.0 * date_coverage,
                "rated_anchor_coverage_pct": float(
                    100.0 * player["rated_exposures"].sum() / player["total_exposures"].sum()
                ),
            }
        )
        print(
            f"{season} {season_type}: {len(possessions):,} possessions, "
            f"{file_reports[-1]['rated_anchor_coverage_pct']:.1f}% rated anchor coverage"
        )
    if not player_frames:
        return pd.DataFrame(), pd.DataFrame(), file_reports
    return (
        pd.concat(player_frames, ignore_index=True),
        pd.concat(pair_frames, ignore_index=True),
        file_reports,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=1997)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--season-types", choices=["all", "rs", "ps"], default="all")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--darko", default=str(DEFAULT_DARKO_PATH))
    parser.add_argument(
        "--rating-timing",
        choices=["same-season", "asof"],
        default="same-season",
        help=(
            "DARKO timing contract. same-season applies the season's final value to all "
            "games (canonical descriptive build); asof preserves strict dated lookup."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--app-json",
        type=Path,
        default=None,
        help="Optional destination for a monolithic browser artifact (large).",
    )
    parser.add_argument(
        "--app-data-dir",
        type=Path,
        default=None,
        help="Optional destination for lazy-loadable browser JSON files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_year = season_to_year(args.start_year)
    end_year = season_to_year(args.end_year)
    season_types = {"rs", "ps"} if args.season_types == "all" else {args.season_types}
    paths = selected_rapm_files(args.processed_dir, start_year, end_year, season_types)
    if not paths:
        raise FileNotFoundError("No matching processed RAPM parquet files")

    darko_source: str | Path = args.darko
    if args.darko == str(DEFAULT_DARKO_PATH) and not DEFAULT_DARKO_PATH.exists():
        darko_source = DARKO_HISTORY_URL
    darko, histories = load_darko_history(darko_source)
    names = load_name_map(darko)
    player, pair, file_reports = process_files(
        paths,
        histories,
        rating_timing=args.rating_timing,
    )
    if player.empty:
        raise RuntimeError("No player help rows were generated")

    player = attach_names(add_ranks(player), names)
    pair = attach_names(pair, names)
    pair = pair.sort_values(
        ["season", "season_type", "player_id", "shared_exposures", "teammate_id"],
        ascending=[True, True, True, False, True],
        kind="mergesort",
    )
    pair["frequency_rank"] = pair.groupby(["season", "season_type", "player_id"]).cumcount() + 1
    reconciliation = validate_reconciliation(player, pair)

    if args.rating_timing == "same-season":
        method = (
            "Possession-weighted sum of the other four same-team players' same-season "
            "DARKO values, applied retrospectively to every game in that season."
        )
        historical_snapshot_note = (
            "The public DARKO history contains one row per player-season. The canonical "
            "descriptive build applies that row to the full season regardless of source "
            "date; it measures season-level teammate talent and is not an as-of forecast."
        )
    else:
        method = (
            "Possession-weighted sum of the other four same-team players' latest DARKO "
            "snapshot dated on or before each game."
        )
        historical_snapshot_note = (
            "The public DARKO history contains one dated snapshot per player-season, "
            "not a daily archive. Strict as-of lookup carries the latest prior snapshot "
            "forward and never uses a future value."
        )

    metadata = {
        "title": "DARKO Teammate Help",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "start_season": int(player["season"].min()),
        "end_season": int(player["season"].max()),
        "season_types": sorted(player["season_type"].unique().tolist()),
        "player_seasons": int(len(player)),
        "oriented_player_teammate_seasons": int(len(pair)),
        "darko_source": DARKO_HISTORY_URL,
        "darko_rows": int(len(darko)),
        "rating_timing": args.rating_timing,
        "method": method,
        "historical_snapshot_note": historical_snapshot_note,
        "coverage_pct": float(
            100.0 * player["rated_exposures"].sum() / player["total_exposures"].sum()
        ),
        "reconciliation": reconciliation,
        "files": file_reports,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    player_path = args.output_dir / "season_player_help.csv"
    pair_path = args.output_dir / "season_teammate_help.csv"
    metadata_path = args.output_dir / "metadata.json"
    player.to_csv(player_path, index=False)
    pair.to_csv(pair_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if args.app_json is not None:
        write_app_json(player, pair, metadata, args.app_json)
    if args.app_data_dir is not None:
        write_split_app_data(player, pair, metadata, args.app_data_dir)

    print(f"Wrote {len(player):,} player-season rows to {player_path}")
    print(f"Wrote {len(pair):,} oriented teammate rows to {pair_path}")
    print(f"Overall rated exposure coverage: {metadata['coverage_pct']:.2f}%")
    print(f"Reconciliation: {reconciliation}")


if __name__ == "__main__":
    main()
