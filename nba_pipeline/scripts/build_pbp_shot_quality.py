#!/usr/bin/env python3
"""
Build a repo-owned shot-quality table from raw NBA play-by-play.

This is deliberately independent of the external ShotQuality initial_ev field.
It parses each field-goal attempt from raw PBP descriptions, estimates a
leave-one-shot context EV from shot location/action descriptors, then adds a
separate empirical-Bayes shooter talent residual.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
RAW_DIR = PIPELINE_ROOT / "raw_data"
OUTPUT_DIR = PIPELINE_ROOT / "results" / "pbp_shot_quality"
PLAYER_MAP = PIPELINE_ROOT.parent / "autocomplete_map.csv"

RAW_COLUMNS = [
    "game_id",
    "event_num",
    "event_type",
    "event_action_type",
    "period",
    "time_quarter",
    "seconds_remaining_quarter",
    "home_description",
    "visitor_description",
    "player1_id",
    "player1_name",
    "player1_team_id",
    "player1_team_abbreviation",
    "away_player1",
    "away_player2",
    "away_player3",
    "away_player4",
    "away_player5",
    "home_player1",
    "home_player2",
    "home_player3",
    "home_player4",
    "home_player5",
]

SHOT_KEYWORDS = (
    "3pt",
    "jump shot",
    "jumper",
    "layup",
    "dunk",
    "hook",
    "floating",
    "floater",
    "tip",
    "fadeaway",
    "bank",
)

MODIFIER_PATTERNS = {
    "driving": r"\bdriving\b",
    "running": r"\brunning\b",
    "cutting": r"\bcutting\b",
    "pullup": r"\bpull[\s-]?up\b",
    "step_back": r"\bstep back\b",
    "turnaround": r"\bturnaround\b",
    "fadeaway": r"\bfadeaway\b",
    "bank": r"\bbank\b",
    "alley_oop": r"\balley oop\b",
    "putback": r"\bputback\b",
    "reverse": r"\breverse\b",
    "finger_roll": r"\bfinger roll\b",
    "floating": r"\bfloating|\bfloater\b",
}

DISTANCE_BINS = [-0.1, 2, 4, 7, 10, 14, 18, 22, 25, 28, 32, 100]
DISTANCE_LABELS = [
    "00_02",
    "03_04",
    "05_07",
    "08_10",
    "11_14",
    "15_18",
    "19_22",
    "23_25",
    "26_28",
    "29_32",
    "33_plus",
]


@dataclass(frozen=True)
class ModelConfig:
    season_shrink: float = 800.0
    zone_shrink: float = 500.0
    context_shrink: float = 250.0
    shooter_shrink: float = 250.0
    shooter_season_shrink: float = 150.0
    defender_shrink: float = 750.0


def parse_year_token(token: str) -> list[int]:
    token = token.strip()
    if "-" in token:
        start_s, end_s = token.split("-", 1)
        start = normalize_season_end_year(int(start_s))
        end = normalize_season_end_year(int(end_s))
        return list(range(start, end + 1))
    return [normalize_season_end_year(int(token))]


def normalize_season_end_year(year: int) -> int:
    if year < 100:
        return 1900 + year if year >= 70 else 2000 + year
    return year


def season_label(end_year: int) -> str:
    return f"{end_year - 1}-{end_year % 100:02d}"


def discover_raw_files(raw_dir: Path, years: set[int] | None, season_types: set[str]) -> list[Path]:
    files: list[tuple[int, str, Path]] = []
    pattern = re.compile(r"^NBA(\d{2})(_PS)?\.parquet$")
    for path in raw_dir.glob("NBA*.parquet"):
        match = pattern.match(path.name)
        if not match:
            continue
        end_year = normalize_season_end_year(int(match.group(1)))
        season_type = "ps" if match.group(2) else "rs"
        if years is not None and end_year not in years:
            continue
        if season_type not in season_types:
            continue
        files.append((end_year, season_type, path))
    return [path for _, _, path in sorted(files)]


def file_metadata(path: Path) -> tuple[int, str]:
    match = re.match(r"^NBA(\d{2})(_PS)?\.parquet$", path.name)
    if not match:
        raise ValueError(f"Unexpected raw file name: {path}")
    return normalize_season_end_year(int(match.group(1))), ("PS" if match.group(2) else "RS")


def normalize_game_id(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(10)
    )


def event_make_miss_flags(event_type: pd.Series) -> tuple[pd.Series, pd.Series]:
    event_num = pd.to_numeric(event_type, errors="coerce")
    event_str = event_type.astype(str).str.upper()
    is_make = event_num.eq(1) | event_str.isin({"MAKE", "MADE_SHOT", "FIELD_GOAL_MADE"})
    is_miss = event_num.eq(2) | event_str.isin({"MISS", "MISSED_SHOT", "FIELD_GOAL_MISSED"})
    return is_make, is_miss


def choose_shot_side(df: pd.DataFrame, is_make: pd.Series, is_miss: pd.Series) -> tuple[pd.Series, pd.Series]:
    home_desc = df["home_description"].fillna("").astype(str)
    away_desc = df["visitor_description"].fillna("").astype(str)
    shot_keyword_pattern = "|".join(re.escape(keyword) for keyword in SHOT_KEYWORDS)

    home_shot = (
        (is_make & home_desc.str.contains(r"\bPTS\b", case=False, regex=True, na=False))
        | (is_miss & home_desc.str.contains(r"\bMISS\b", case=False, regex=True, na=False))
        | home_desc.str.contains(shot_keyword_pattern, case=False, regex=True, na=False)
    ) & ~home_desc.str.contains(r"\bBLOCK\b", case=False, regex=True, na=False)

    away_shot = (
        (is_make & away_desc.str.contains(r"\bPTS\b", case=False, regex=True, na=False))
        | (is_miss & away_desc.str.contains(r"\bMISS\b", case=False, regex=True, na=False))
        | away_desc.str.contains(shot_keyword_pattern, case=False, regex=True, na=False)
    ) & ~away_desc.str.contains(r"\bBLOCK\b", case=False, regex=True, na=False)

    side = pd.Series("", index=df.index, dtype=object)
    side.loc[home_shot & ~away_shot] = "Home"
    side.loc[away_shot & ~home_shot] = "Away"
    side.loc[(side == "") & home_shot] = "Home"
    side.loc[(side == "") & away_shot] = "Away"

    desc = pd.Series(np.where(side.eq("Home"), home_desc, away_desc), index=df.index, dtype=object)
    unresolved = side.eq("")
    if unresolved.any():
        desc.loc[unresolved] = home_desc.where(home_desc.str.len().gt(0), away_desc).loc[unresolved]
        side.loc[unresolved & home_desc.str.len().gt(0)] = "Home"
        side.loc[unresolved & side.eq("") & away_desc.str.len().gt(0)] = "Away"
    return side, desc.str.replace(r"\s+", " ", regex=True).str.strip()


def load_shots_from_file(path: Path) -> pd.DataFrame:
    end_year, season_type = file_metadata(path)
    header = pq.read_schema(path).names
    use_columns = [col for col in RAW_COLUMNS if col in header]
    df = pd.read_parquet(path, columns=use_columns)
    before_dedupe = len(df)
    df = df.drop_duplicates().copy()
    duplicate_rows_removed = before_dedupe - len(df)

    is_make, is_miss = event_make_miss_flags(df["event_type"])
    df = df[is_make | is_miss].copy()
    is_make = is_make.loc[df.index]
    is_miss = is_miss.loc[df.index]
    if df.empty:
        return df

    shot_side, shot_description = choose_shot_side(df, is_make, is_miss)
    df = df[shot_side.isin(["Home", "Away"])].copy()
    is_make = is_make.loc[df.index]
    shot_side = shot_side.loc[df.index]
    shot_description = shot_description.loc[df.index]

    out = pd.DataFrame(index=df.index)
    out["source_file"] = path.name
    out["season_end_year"] = end_year
    out["season"] = season_label(end_year)
    out["season_type"] = season_type
    out["game_id"] = normalize_game_id(df["game_id"])
    out["event_num"] = pd.to_numeric(df["event_num"], errors="coerce").astype("Int64")
    out["shot_id"] = out["game_id"] + "_" + out["event_num"].astype(str)
    out["period"] = pd.to_numeric(df.get("period"), errors="coerce").astype("Int64")
    out["time_quarter"] = df.get("time_quarter", "").fillna("").astype(str)
    if "seconds_remaining_quarter" in df:
        out["seconds_remaining_quarter"] = pd.to_numeric(df["seconds_remaining_quarter"], errors="coerce")
    else:
        out["seconds_remaining_quarter"] = np.nan
    out["event_action_type"] = pd.to_numeric(df.get("event_action_type"), errors="coerce").astype("Int64")
    out["offense_side"] = shot_side.values
    out["shot_description"] = shot_description.values
    out["shooter_id"] = pd.to_numeric(df.get("player1_id"), errors="coerce").astype("Int64")
    out["shooter_name"] = df.get("player1_name", "").fillna("").astype(str)
    out["shooter_team_id"] = pd.to_numeric(df.get("player1_team_id"), errors="coerce").astype("Int64")
    out["shooter_team_abbreviation"] = df.get("player1_team_abbreviation", "").fillna("").astype(str)
    for idx in range(1, 6):
        away_col = f"away_player{idx}"
        home_col = f"home_player{idx}"
        away_ids = pd.to_numeric(df.get(away_col, 0), errors="coerce")
        home_ids = pd.to_numeric(df.get(home_col, 0), errors="coerce")
        out[f"O{idx}"] = pd.Series(
            np.where(shot_side.eq("Away"), away_ids, home_ids),
            index=df.index,
        ).astype("Int64")
        out[f"D{idx}"] = pd.Series(
            np.where(shot_side.eq("Away"), home_ids, away_ids),
            index=df.index,
        ).astype("Int64")
    out["is_make"] = is_make.astype(bool).values
    out["raw_rows_removed_as_duplicates"] = duplicate_rows_removed
    return out.reset_index(drop=True)


def add_shot_features(shots: pd.DataFrame) -> pd.DataFrame:
    shots = shots.copy()
    desc = shots["shot_description"].fillna("").astype(str)
    desc_lower = desc.str.lower()

    shots["is_3pt"] = desc_lower.str.contains(r"\b3pt\b", regex=True, na=False)
    shots["shot_value"] = np.where(shots["is_3pt"], 3.0, 2.0)
    shots["actual_points"] = np.where(shots["is_make"], shots["shot_value"], 0.0).astype(float)

    shots["distance_raw_ft"] = pd.to_numeric(desc.str.extract(r"(\d+)'\s*", expand=False), errors="coerce")
    is_dunk = desc_lower.str.contains(r"\bdunk\b|\bslam dunk\b", regex=True, na=False)
    is_layup = desc_lower.str.contains(r"\blayup\b|\bfinger roll\b", regex=True, na=False)
    is_tip = desc_lower.str.contains(r"\btip\b", regex=True, na=False)
    is_hook = desc_lower.str.contains(r"\bhook\b", regex=True, na=False)
    is_floater = desc_lower.str.contains(r"\bfloating\b|\bfloater\b", regex=True, na=False)
    is_jumper = desc_lower.str.contains(r"\bjump shot\b|\bjumper\b|\bpull[\s-]?up\b|\bstep back\b", regex=True, na=False)

    default_distance = np.select(
        [
            is_dunk | is_layup | is_tip,
            shots["is_3pt"],
            is_hook | is_floater,
            is_jumper,
        ],
        [1.0, 24.5, 8.0, 16.0],
        default=12.0,
    )
    shots["shot_distance_ft"] = shots["distance_raw_ft"].fillna(pd.Series(default_distance, index=shots.index)).astype(float)
    shots["distance_source"] = np.where(shots["distance_raw_ft"].notna(), "parsed_description", "imputed_from_type")
    shots["distance_bin"] = pd.cut(
        shots["shot_distance_ft"].clip(lower=0, upper=99),
        bins=DISTANCE_BINS,
        labels=DISTANCE_LABELS,
    ).astype(str)

    shots["shot_zone"] = np.select(
        [
            shots["is_3pt"],
            is_dunk | is_layup | is_tip | shots["shot_distance_ft"].le(4),
            shots["shot_distance_ft"].le(10),
            shots["shot_distance_ft"].le(18),
        ],
        ["three", "rim", "short_mid", "long_mid"],
        default="deep_two",
    )

    shots["shot_family"] = np.select(
        [
            is_dunk,
            is_layup,
            is_tip,
            is_hook,
            is_floater,
            shots["is_3pt"],
            is_jumper,
        ],
        ["dunk", "layup", "tip", "hook", "floater", "three", "jumper"],
        default="other_two",
    )

    shots["talent_bucket"] = np.select(
        [
            shots["is_3pt"],
            shots["shot_family"].isin(["dunk", "layup", "tip"]),
            shots["shot_family"].isin(["hook", "floater"]) | shots["shot_distance_ft"].le(10),
        ],
        ["three", "rim_finish", "short_touch"],
        default="midrange_jumper",
    )

    modifier_bundle = pd.Series("", index=shots.index, dtype=object)
    for label, pattern in MODIFIER_PATTERNS.items():
        col = f"desc_{label}"
        shots[col] = desc_lower.str.contains(pattern, regex=True, na=False)
        modifier_bundle = modifier_bundle + np.where(shots[col], f"|{label}", "")
    shots["modifier_bundle"] = modifier_bundle.str.strip("|").replace("", "none")
    shots["made_assisted_tag"] = shots["is_make"] & desc.str.contains(r"\([^()]+?\s+\d+\s+AST\)", regex=True, na=False)

    return shots


def apply_player_name_map(shots: pd.DataFrame, player_map_path: Path | None) -> pd.DataFrame:
    if player_map_path is None or not player_map_path.exists():
        return shots
    name_map_df = pd.read_csv(player_map_path)
    if not {"nba_id", "player_name"}.issubset(name_map_df.columns):
        return shots
    name_map = (
        name_map_df.dropna(subset=["nba_id", "player_name"])
        .assign(nba_id=lambda x: pd.to_numeric(x["nba_id"], errors="coerce").astype("Int64"))
        .dropna(subset=["nba_id"])
        .drop_duplicates("nba_id", keep="last")
        .set_index("nba_id")["player_name"]
    )
    mapped = shots["shooter_id"].map(name_map)
    shots = shots.copy()
    shots["raw_shooter_name"] = shots["shooter_name"]
    shots["shooter_name"] = mapped.fillna(shots["shooter_name"]).astype(str)
    shots["shooter_name_source"] = np.where(mapped.notna(), "autocomplete_map", "raw_pbp")
    return shots


def loo_eb_mean(
    df: pd.DataFrame,
    key_cols: list[str],
    target_col: str,
    prior: pd.Series,
    shrink: float,
    prefix: str,
) -> tuple[pd.Series, pd.Series]:
    stats = df.groupby(key_cols, dropna=False, observed=True)[target_col].agg(["sum", "count"])
    joined = df[key_cols].merge(stats, left_on=key_cols, right_index=True, how="left")
    loo_count = (joined["count"].astype(float) - 1.0).clip(lower=0.0)
    loo_sum = joined["sum"].astype(float) - df[target_col].astype(float).to_numpy()
    estimate = (loo_sum + shrink * prior.astype(float).to_numpy()) / (loo_count + shrink)
    df[f"{prefix}_loo_count"] = loo_count.to_numpy()
    return pd.Series(estimate, index=df.index), pd.Series(loo_count, index=df.index)


def add_context_ev(shots: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    shots = shots.copy()
    global_by_value = shots.groupby("shot_value")["actual_points"].mean()
    shots["global_shot_value_ev"] = shots["shot_value"].map(global_by_value).astype(float)

    season_key = ["season_end_year", "season_type", "shot_value"]
    shots["season_shot_value_ev"], _ = loo_eb_mean(
        shots,
        season_key,
        "actual_points",
        shots["global_shot_value_ev"],
        config.season_shrink,
        "season_value",
    )

    zone_key = ["shot_value", "shot_zone", "distance_bin", "shot_family"]
    shots["zone_context_ev"], _ = loo_eb_mean(
        shots,
        zone_key,
        "actual_points",
        shots["season_shot_value_ev"],
        config.zone_shrink,
        "zone_context",
    )

    context_key = [
        "shot_value",
        "shot_zone",
        "distance_bin",
        "shot_family",
        "event_action_type",
        "modifier_bundle",
    ]
    shots["shot_context_ev"], _ = loo_eb_mean(
        shots,
        context_key,
        "actual_points",
        shots["zone_context_ev"],
        config.context_shrink,
        "granular_context",
    )
    shots["shot_context_ev"] = shots["shot_context_ev"].clip(lower=0.01, upper=shots["shot_value"])
    shots["context_residual_points"] = shots["actual_points"] - shots["shot_context_ev"]
    return shots


def add_shooter_talent(shots: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    shots = shots.copy()
    talent_key = ["shooter_id", "talent_bucket"]
    stats = shots.groupby(talent_key, dropna=False, observed=True)["context_residual_points"].agg(["sum", "count"])
    joined = shots[talent_key].merge(stats, left_on=talent_key, right_index=True, how="left")
    loo_count = (joined["count"].astype(float) - 1.0).clip(lower=0.0)
    loo_sum = joined["sum"].astype(float) - shots["context_residual_points"].astype(float).to_numpy()
    shots["shooter_talent_attempts_loo"] = loo_count.to_numpy()
    shots["shooter_talent_ev_added"] = (loo_sum / (loo_count + config.shooter_shrink)).fillna(0.0).to_numpy()

    sort_cols = [
        "season_end_year",
        "season_type",
        "game_id",
        "period",
        "seconds_remaining_quarter",
        "event_num",
    ]
    ordered = shots.sort_values(
        sort_cols,
        ascending=[True, True, True, True, False, True],
        na_position="last",
    ).copy()
    group = ordered.groupby(talent_key, dropna=False, observed=True)["context_residual_points"]
    ordered["shooter_talent_prior_attempts"] = group.cumcount().astype(float)
    ordered["shooter_talent_prior_sum"] = group.cumsum() - ordered["context_residual_points"]
    ordered["shooter_talent_prior_ev_added"] = (
        ordered["shooter_talent_prior_sum"]
        / (ordered["shooter_talent_prior_attempts"] + config.shooter_shrink)
    ).fillna(0.0)

    prior_cols = ["shooter_talent_prior_attempts", "shooter_talent_prior_ev_added"]
    shots = shots.join(ordered[prior_cols], how="left")

    shots["shot_quality_ev"] = (shots["shot_context_ev"] + shots["shooter_talent_ev_added"]).clip(
        lower=0.01,
        upper=shots["shot_value"],
    )
    shots["shot_quality_prior_ev"] = (shots["shot_context_ev"] + shots["shooter_talent_prior_ev_added"]).clip(
        lower=0.01,
        upper=shots["shot_value"],
    )
    shots["shot_context_make_prob"] = (shots["shot_context_ev"] / shots["shot_value"]).clip(0.0, 1.0)
    shots["shot_quality_make_prob"] = (shots["shot_quality_ev"] / shots["shot_value"]).clip(0.0, 1.0)
    shots["shot_quality_prior_make_prob"] = (shots["shot_quality_prior_ev"] / shots["shot_value"]).clip(0.0, 1.0)
    return shots


def add_shooter_season_talent(shots: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    shots = shots.copy()
    season_talent_key = ["shooter_id", "season_end_year", "talent_bucket"]
    shots["shooter_season_talent_ev_added"], season_attempts = loo_eb_mean(
        shots,
        season_talent_key,
        "context_residual_points",
        shots["shooter_talent_ev_added"],
        config.shooter_season_shrink,
        "shooter_season_talent",
    )
    shots["shooter_season_talent_attempts_loo"] = season_attempts.to_numpy()
    shots["shot_quality_season_ev_unclipped"] = (
        shots["shot_context_ev"] + shots["shooter_season_talent_ev_added"]
    )
    shots["shot_quality_season_ev"] = shots["shot_quality_season_ev_unclipped"].clip(
        lower=0.01,
        upper=shots["shot_value"],
    )
    shots["shot_quality_season_make_prob"] = (
        shots["shot_quality_season_ev"] / shots["shot_value"]
    ).clip(0.0, 1.0)
    shots["season_shooter_residual_points"] = (
        shots["actual_points"] - shots["shot_quality_season_ev_unclipped"]
    )
    return shots


def add_defender_adjustment(shots: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    shots = shots.copy()
    defender_cols = [f"D{i}" for i in range(1, 6)]
    missing_cols = [col for col in defender_cols if col not in shots.columns]
    if missing_cols:
        raise ValueError(f"Missing defender columns: {missing_cols}")

    base = shots[["season_end_year", "talent_bucket", "season_shooter_residual_points"]].copy()
    base["row_id"] = np.arange(len(shots), dtype=np.int64)
    flattened = []
    for col in defender_cols:
        part = base.copy()
        part["defender_id"] = pd.to_numeric(shots[col], errors="coerce").fillna(0).astype("int64")
        flattened.append(part)
    defender_rows = pd.concat(flattened, ignore_index=True)
    defender_rows = defender_rows[defender_rows["defender_id"] > 0].copy()
    defender_rows = defender_rows.drop_duplicates(
        subset=["row_id", "defender_id", "season_end_year", "talent_bucket"],
        keep="first",
    )

    key = ["defender_id", "season_end_year", "talent_bucket"]
    stats = defender_rows.groupby(key, dropna=False, observed=True)["season_shooter_residual_points"].agg(["sum", "count"])
    defender_rows = defender_rows.merge(stats, left_on=key, right_index=True, how="left")
    defender_rows["defender_shot_attempts_loo"] = (defender_rows["count"].astype(float) - 1.0).clip(lower=0.0)
    defender_rows["defender_ev_allowed_added"] = (
        defender_rows["sum"].astype(float) - defender_rows["season_shooter_residual_points"].astype(float)
    ) / (defender_rows["defender_shot_attempts_loo"] + config.defender_shrink)

    row_effect = (
        defender_rows.groupby("row_id", observed=True)["defender_ev_allowed_added"]
        .mean()
        .reindex(np.arange(len(shots)), fill_value=0.0)
    )
    row_def_count = (
        defender_rows.groupby("row_id", observed=True)["defender_id"]
        .nunique()
        .reindex(np.arange(len(shots)), fill_value=0)
    )

    shots["defender_lineup_ev_allowed_added"] = row_effect.to_numpy()
    shots["defender_count"] = row_def_count.to_numpy().astype(int)
    shots["shot_quality_with_defense_ev"] = (
        shots["shot_quality_season_ev_unclipped"] + shots["defender_lineup_ev_allowed_added"]
    ).clip(lower=0.01, upper=shots["shot_value"])
    shots["shot_quality_with_defense_make_prob"] = (
        shots["shot_quality_with_defense_ev"] / shots["shot_value"]
    ).clip(0.0, 1.0)
    return shots


def build_shooter_summary(shots: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    grouped = (
        shots.groupby(["shooter_id", "shooter_name", "talent_bucket"], dropna=False, observed=True)
        .agg(
            attempts=("actual_points", "size"),
            actual_points_per_shot=("actual_points", "mean"),
            context_ev_per_shot=("shot_context_ev", "mean"),
            residual_sum=("context_residual_points", "sum"),
            residual_per_shot=("context_residual_points", "mean"),
            shot_quality_ev_per_shot=("shot_quality_ev", "mean"),
        )
        .reset_index()
    )
    grouped["shooter_talent_ev_added_eb"] = grouped["residual_sum"] / (
        grouped["attempts"] + config.shooter_shrink
    )
    return grouped.sort_values(["talent_bucket", "attempts"], ascending=[True, False])


def build_shooter_season_summary(shots: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    grouped = (
        shots.groupby(["shooter_id", "shooter_name", "season_end_year", "season", "talent_bucket"], dropna=False, observed=True)
        .agg(
            attempts=("actual_points", "size"),
            actual_points_per_shot=("actual_points", "mean"),
            context_ev_per_shot=("shot_context_ev", "mean"),
            career_talent_ev_added_per_shot=("shooter_talent_ev_added", "mean"),
            season_talent_ev_added_per_shot=("shooter_season_talent_ev_added", "mean"),
            shot_quality_season_ev_per_shot=("shot_quality_season_ev", "mean"),
            residual_sum=("context_residual_points", "sum"),
        )
        .reset_index()
    )
    grouped["season_talent_ev_added_eb"] = grouped["residual_sum"] / (
        grouped["attempts"] + config.shooter_season_shrink
    )
    grouped["season_talent_ev_added_eb"] = grouped["season_talent_ev_added_per_shot"]
    return grouped.sort_values(["season_end_year", "talent_bucket", "attempts"], ascending=[True, True, False])


def build_defender_season_summary(shots: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    defender_cols = [f"D{i}" for i in range(1, 6)]
    pieces = []
    for col in defender_cols:
        piece = shots[
            [
                "season_end_year",
                "season",
                "talent_bucket",
                "season_shooter_residual_points",
            ]
        ].copy()
        piece["defender_id"] = pd.to_numeric(shots[col], errors="coerce").fillna(0).astype("int64")
        pieces.append(piece)
    defender_rows = pd.concat(pieces, ignore_index=True)
    defender_rows = defender_rows[defender_rows["defender_id"] > 0].copy()
    summary = (
        defender_rows.groupby(["defender_id", "season_end_year", "season", "talent_bucket"], dropna=False, observed=True)
        .agg(
            shots_defended=("season_shooter_residual_points", "size"),
            opponent_residual_allowed_per_shot=("season_shooter_residual_points", "mean"),
            opponent_residual_allowed_sum=("season_shooter_residual_points", "sum"),
        )
        .reset_index()
    )
    summary["defender_ev_allowed_added_eb"] = summary["opponent_residual_allowed_sum"] / (
        summary["shots_defended"] + config.defender_shrink
    )

    name_map = (
        shots[["shooter_id", "shooter_name"]]
        .dropna(subset=["shooter_id"])
        .drop_duplicates("shooter_id", keep="last")
        .rename(columns={"shooter_id": "defender_id", "shooter_name": "defender_name"})
    )
    summary = summary.merge(name_map, on="defender_id", how="left")
    summary["defender_name"] = summary["defender_name"].fillna("ID_" + summary["defender_id"].astype(str))
    cols = ["defender_id", "defender_name", "season_end_year", "season", "talent_bucket"]
    return summary[cols + [c for c in summary.columns if c not in cols]].sort_values(
        ["season_end_year", "talent_bucket", "shots_defended"],
        ascending=[True, True, False],
    )


def build_calibration_summary(shots: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["season_end_year", "season_type", "shot_zone", "talent_bucket"]
    return (
        shots.groupby(group_cols, dropna=False, observed=True)
        .agg(
            attempts=("actual_points", "size"),
            actual_points_per_shot=("actual_points", "mean"),
            context_ev_per_shot=("shot_context_ev", "mean"),
            shot_quality_ev_per_shot=("shot_quality_ev", "mean"),
            shot_quality_season_ev_per_shot=("shot_quality_season_ev", "mean"),
            shot_quality_with_defense_ev_per_shot=("shot_quality_with_defense_ev", "mean"),
            shot_quality_prior_ev_per_shot=("shot_quality_prior_ev", "mean"),
            mean_abs_context_error=("context_residual_points", lambda x: float(np.mean(np.abs(x)))),
        )
        .reset_index()
        .sort_values(group_cols)
    )


def build_model_summary(shots: pd.DataFrame, files: list[Path], config: ModelConfig) -> pd.DataFrame:
    rows = [
        ("raw_files", len(files)),
        ("shots", len(shots)),
        ("seasons", shots["season_end_year"].nunique()),
        ("actual_points_per_shot", shots["actual_points"].mean()),
        ("shot_context_ev_mean", shots["shot_context_ev"].mean()),
        ("shot_quality_ev_mean", shots["shot_quality_ev"].mean()),
        ("shot_quality_season_ev_mean", shots["shot_quality_season_ev"].mean()),
        ("shot_quality_with_defense_ev_mean", shots["shot_quality_with_defense_ev"].mean()),
        ("shot_quality_prior_ev_mean", shots["shot_quality_prior_ev"].mean()),
        ("context_mae", np.mean(np.abs(shots["actual_points"] - shots["shot_context_ev"]))),
        ("shot_quality_mae", np.mean(np.abs(shots["actual_points"] - shots["shot_quality_ev"]))),
        ("shot_quality_season_mae", np.mean(np.abs(shots["actual_points"] - shots["shot_quality_season_ev"]))),
        ("shot_quality_with_defense_mae", np.mean(np.abs(shots["actual_points"] - shots["shot_quality_with_defense_ev"]))),
        ("season_shrink", config.season_shrink),
        ("zone_shrink", config.zone_shrink),
        ("context_shrink", config.context_shrink),
        ("shooter_shrink", config.shooter_shrink),
        ("shooter_season_shrink", config.shooter_season_shrink),
        ("defender_shrink", config.defender_shrink),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--years",
        nargs="*",
        help="Season end years to include, e.g. 2024 2025 or 1997-2026. Defaults to all NBA?? files.",
    )
    parser.add_argument(
        "--season-types",
        nargs="*",
        default=["rs", "ps"],
        choices=["rs", "ps", "all"],
        help="Season types to include. Defaults to rs ps.",
    )
    parser.add_argument("--season-shrink", type=float, default=800.0)
    parser.add_argument("--zone-shrink", type=float, default=500.0)
    parser.add_argument("--context-shrink", type=float, default=250.0)
    parser.add_argument("--shooter-shrink", type=float, default=250.0)
    parser.add_argument("--shooter-season-shrink", type=float, default=150.0)
    parser.add_argument("--defender-shrink", type=float, default=750.0)
    parser.add_argument("--output-name", default="pbp_shot_quality_all")
    parser.add_argument(
        "--player-map",
        type=Path,
        default=PLAYER_MAP,
        help="Optional nba_id/player_name CSV for canonical shooter names.",
    )
    parser.add_argument("--write-csv", action="store_true", help="Also write the full shot table as CSV.")
    parser.add_argument("--limit-files", type=int, help="Debug helper: process only the first N discovered files.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    config = ModelConfig(
        season_shrink=args.season_shrink,
        zone_shrink=args.zone_shrink,
        context_shrink=args.context_shrink,
        shooter_shrink=args.shooter_shrink,
        shooter_season_shrink=args.shooter_season_shrink,
        defender_shrink=args.defender_shrink,
    )

    years = None
    if args.years:
        years = {year for token in args.years for year in parse_year_token(token)}

    season_types = {item.lower() for item in args.season_types}
    if "all" in season_types:
        season_types = {"rs", "ps"}

    files = discover_raw_files(args.raw_dir, years, season_types)
    if args.limit_files:
        files = files[: args.limit_files]
    if not files:
        raise FileNotFoundError("No matching NBA raw parquet files found.")

    print(f"Discovered {len(files)} raw files.")
    shot_frames = []
    for path in files:
        print(f"  Loading {path.name}...")
        frame = load_shots_from_file(path)
        print(f"    shots={len(frame):,}")
        if not frame.empty:
            shot_frames.append(frame)
    if not shot_frames:
        raise ValueError("No FGA rows found in selected raw files.")

    shots = pd.concat(shot_frames, ignore_index=True)
    shots = shots.drop_duplicates(subset=["shot_id", "source_file"], keep="first").reset_index(drop=True)
    print(f"Building features for {len(shots):,} shots...")
    shots = apply_player_name_map(shots, args.player_map)
    shots = add_shot_features(shots)

    print("Estimating leave-one-shot context EV...")
    shots = add_context_ev(shots, config)

    print("Estimating shooter talent residuals...")
    shots = add_shooter_talent(shots, config)

    print("Estimating shooter-season talent residuals...")
    shots = add_shooter_season_talent(shots, config)

    print("Estimating defender-on-court residuals...")
    shots = add_defender_adjustment(shots, config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_parquet = args.output_dir / f"{args.output_name}.parquet"
    shots.to_parquet(full_parquet, index=False)
    print(f"Wrote {full_parquet}")

    if args.write_csv:
        full_csv = args.output_dir / f"{args.output_name}.csv"
        shots.to_csv(full_csv, index=False)
        print(f"Wrote {full_csv}")

    model_summary = build_model_summary(shots, files, config)
    model_summary_path = args.output_dir / f"{args.output_name}_model_summary.csv"
    model_summary.to_csv(model_summary_path, index=False)
    print(f"Wrote {model_summary_path}")

    shooter_summary = build_shooter_summary(shots, config)
    shooter_summary_path = args.output_dir / f"{args.output_name}_shooter_talent.csv"
    shooter_summary.to_csv(shooter_summary_path, index=False)
    print(f"Wrote {shooter_summary_path}")

    shooter_season_summary = build_shooter_season_summary(shots, config)
    shooter_season_summary_path = args.output_dir / f"{args.output_name}_shooter_season_talent.csv"
    shooter_season_summary.to_csv(shooter_season_summary_path, index=False)
    print(f"Wrote {shooter_season_summary_path}")

    defender_season_summary = build_defender_season_summary(shots, config)
    defender_season_summary_path = args.output_dir / f"{args.output_name}_defender_season_impact.csv"
    defender_season_summary.to_csv(defender_season_summary_path, index=False)
    print(f"Wrote {defender_season_summary_path}")

    calibration = build_calibration_summary(shots)
    calibration_path = args.output_dir / f"{args.output_name}_calibration.csv"
    calibration.to_csv(calibration_path, index=False)
    print(f"Wrote {calibration_path}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
