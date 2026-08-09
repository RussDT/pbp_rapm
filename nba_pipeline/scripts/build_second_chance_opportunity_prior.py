#!/usr/bin/env python3
"""
Build a game-cross-fitted second-chance opportunity prior for missed field goals.

This is an isolated research surface. It does not alter FIRST_CHANCE,
SECOND_CHANCE_CLEAN, or the active public DECOMP. The main counterfactual asks:

    What would the offensive-rebound probability have been if one of the four
    teammates took the same shot and the actual shooter moved into the
    non-shooter rebound group?

The model is intentionally interpretable:

    context + four non-shooter effects + five defender effects
            + shooter-by-zone effect

Every validation game is scored by a model trained on other games. The
second-chance points-per-OREB conversion is also estimated from training games
only.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.linear_model import Ridge
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
DEFAULT_PROCESSED_DIR = PIPELINE_ROOT / "processed"
DEFAULT_SHOT_QUALITY_PATH = (
    PIPELINE_ROOT / "results" / "pbp_shot_quality" / "pbp_shot_quality_1997_2026.parquet"
)
DEFAULT_OUTPUT_DIR = PIPELINE_ROOT / "results" / "second_chance_opportunity_prior"

NON_SHOOTER_COLS = ("O1", "O2", "O3", "O4")
DEFENDER_COLS = ("D1", "D2", "D3", "D4", "D5")
CONTEXT_COLS = (
    "season_sample",
    "shot_zone",
    "distance_bin",
    "shot_family",
    "event_action_type",
    "modifier_bundle",
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass(frozen=True)
class RoleSwapResult:
    actual_probability: float
    equal_teammate_swap_probability: float
    actual_raw_prediction: float
    equal_teammate_swap_raw: float
    actual_minus_swap_raw: float
    crash_group_delta_raw: float
    shooter_zone_delta_raw: float
    swap_predictions: Tuple[float, ...]
    swap_raw_predictions: Tuple[float, ...]


@dataclass(frozen=True)
class FeatureSpace:
    context_columns: Mapping[Tuple[str, str], int]
    non_shooter_columns: Mapping[int, int]
    defender_columns: Mapping[int, int]
    shooter_zone_columns: Mapping[Tuple[int, str], int]
    shooter_zone_start: int
    n_features: int


def normalize_game_id(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .str.zfill(10)
    )


def normalize_player_id(value: object) -> int:
    return int(float(value))


def normalize_end_year(value: int) -> int:
    value = int(value)
    if value < 100:
        return 1900 + value if value >= 97 else 2000 + value
    return value


def expand_year_window(start_year: int, end_year: int) -> List[int]:
    start = normalize_end_year(start_year)
    end = normalize_end_year(end_year)
    if end < start:
        end += 100
    return list(range(start, end + 1))


def season_type_label(value: str) -> str:
    label = str(value).upper()
    if label not in {"RS", "PS", "ALL"}:
        raise ValueError("season_type must be RS, PS, or ALL")
    return label


def clip_probability(value: float, bounds: Tuple[float, float]) -> float:
    return float(np.clip(float(value), bounds[0], bounds[1]))


def role_swap_counterfactual(
    *,
    actual_raw_prediction: float,
    shooter_id: int,
    non_shooter_ids: Sequence[int],
    shot_zone: str,
    non_shooter_coefficients: Mapping[int, float],
    shooter_zone_coefficients: Mapping[Tuple[int, str], float],
    clip_bounds: Tuple[float, float] = (0.01, 0.99),
) -> RoleSwapResult:
    """Swap each teammate into the shot and move the shooter into the crash group."""
    if len(non_shooter_ids) != 4:
        raise ValueError("role swap requires exactly four non-shooter teammates")

    shooter_id = normalize_player_id(shooter_id)
    teammates = [normalize_player_id(player_id) for player_id in non_shooter_ids]
    zone = str(shot_zone)
    shooter_crash = float(non_shooter_coefficients.get(shooter_id, 0.0))
    shooter_shot = float(shooter_zone_coefficients.get((shooter_id, zone), 0.0))

    swap_raw: List[float] = []
    crash_actual_minus_swap: List[float] = []
    shooter_actual_minus_swap: List[float] = []
    for teammate_id in teammates:
        teammate_crash = float(non_shooter_coefficients.get(teammate_id, 0.0))
        teammate_shot = float(shooter_zone_coefficients.get((teammate_id, zone), 0.0))
        raw = (
            float(actual_raw_prediction)
            + (shooter_crash - teammate_crash)
            + (teammate_shot - shooter_shot)
        )
        swap_raw.append(raw)
        crash_actual_minus_swap.append(teammate_crash - shooter_crash)
        shooter_actual_minus_swap.append(shooter_shot - teammate_shot)

    swap_probabilities = tuple(clip_probability(value, clip_bounds) for value in swap_raw)
    mean_swap_raw = float(np.mean(swap_raw))
    mean_swap_probability = float(np.mean(swap_probabilities))
    crash_delta = float(np.mean(crash_actual_minus_swap))
    shooter_delta = float(np.mean(shooter_actual_minus_swap))

    return RoleSwapResult(
        actual_probability=clip_probability(actual_raw_prediction, clip_bounds),
        equal_teammate_swap_probability=mean_swap_probability,
        actual_raw_prediction=float(actual_raw_prediction),
        equal_teammate_swap_raw=mean_swap_raw,
        actual_minus_swap_raw=float(actual_raw_prediction) - mean_swap_raw,
        crash_group_delta_raw=crash_delta,
        shooter_zone_delta_raw=shooter_delta,
        swap_predictions=swap_probabilities,
        swap_raw_predictions=tuple(float(value) for value in swap_raw),
    )


def second_chance_ev_per_attempt(
    *,
    make_probability: float,
    offensive_rebound_probability: float,
    points_per_offensive_rebound: float,
) -> float:
    """Price only the miss -> offensive rebound -> second-chance branch."""
    return float(
        (1.0 - float(make_probability))
        * float(offensive_rebound_probability)
        * float(points_per_offensive_rebound)
    )


def reporting_shot_zone(values: pd.Series) -> pd.Series:
    """Combine the two repo SQ zones beyond 10 feet into reader-facing long twos."""
    return values.replace({"long_mid": "long_two", "deep_two": "long_two"})


def game_cluster_bootstrap_mean(
    game_ids: Sequence[object],
    row_values: Sequence[float],
    *,
    repetitions: int,
    seed: int,
) -> Dict[str, float]:
    """Bootstrap games, retaining each sampled game's full row contribution."""
    frame = pd.DataFrame(
        {
            "game_id": pd.Series(game_ids, dtype=str),
            "value": np.asarray(row_values, dtype=float),
        }
    )
    grouped = frame.groupby("game_id", observed=True)["value"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    point = float(sums.sum() / counts.sum())
    if repetitions <= 0 or len(grouped) < 2:
        return {"point": point, "ci95_low": np.nan, "ci95_high": np.nan}

    rng = np.random.default_rng(seed)
    boot = np.empty(int(repetitions), dtype=float)
    for idx in range(int(repetitions)):
        sampled = rng.integers(0, len(grouped), size=len(grouped))
        boot[idx] = float(sums[sampled].sum() / counts[sampled].sum())
    low, high = np.quantile(boot, [0.025, 0.975])
    return {"point": point, "ci95_low": float(low), "ci95_high": float(high)}


def aggregate_player_zone_accounting(
    opportunities: pd.DataFrame,
    attempts: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate OOF miss predictions and price them over all shot attempts."""
    required_opportunities = {
        "shooter_id",
        "shooter_name",
        "shot_zone",
        "actual_oreb",
        "actual_config_probability_oof",
        "role_swap_probability_oof",
        "context_lineup_probability_oof",
        "crash_group_delta_raw",
        "shooter_zone_delta_raw",
        "points_per_oreb_oof",
    }
    required_attempts = {
        "shooter_id",
        "shooter_name",
        "shot_zone",
        "attempts",
        "shot_quality_prior_make_prob",
        "shot_quality_season_make_prob",
    }
    missing_opportunities = sorted(required_opportunities - set(opportunities.columns))
    missing_attempts = sorted(required_attempts - set(attempts.columns))
    if missing_opportunities:
        raise ValueError(f"Missing opportunity columns: {missing_opportunities}")
    if missing_attempts:
        raise ValueError(f"Missing attempt columns: {missing_attempts}")

    group_cols = ["shooter_id", "shot_zone"]
    miss_summary = (
        opportunities.groupby(group_cols, observed=True)
        .agg(
            rebound_opportunity_misses=("actual_oreb", "size"),
            actual_oreb_rate=("actual_oreb", "mean"),
            context_lineup_oreb_probability_oof=("context_lineup_probability_oof", "mean"),
            actual_config_oreb_probability_oof=("actual_config_probability_oof", "mean"),
            role_swap_oreb_probability_oof=("role_swap_probability_oof", "mean"),
            crash_group_delta_raw=("crash_group_delta_raw", "mean"),
            shooter_zone_delta_raw=("shooter_zone_delta_raw", "mean"),
            points_per_oreb_oof=("points_per_oreb_oof", "mean"),
        )
        .reset_index()
    )

    attempt_work = attempts.copy()
    if attempt_work.duplicated(group_cols).any():
        weighted = attempt_work.copy()
        for column in [
            "shot_quality_prior_make_prob",
            "shot_quality_season_make_prob",
        ]:
            weighted[f"{column}_weighted"] = weighted[column] * weighted["attempts"]
        attempt_work = (
            weighted.groupby(group_cols, observed=True)
            .agg(
                shooter_name=("shooter_name", "first"),
                attempts=("attempts", "sum"),
                prior_weighted=("shot_quality_prior_make_prob_weighted", "sum"),
                season_weighted=("shot_quality_season_make_prob_weighted", "sum"),
            )
            .reset_index()
        )
        attempt_work["shot_quality_prior_make_prob"] = (
            attempt_work["prior_weighted"] / attempt_work["attempts"]
        )
        attempt_work["shot_quality_season_make_prob"] = (
            attempt_work["season_weighted"] / attempt_work["attempts"]
        )
        attempt_work = attempt_work.drop(columns=["prior_weighted", "season_weighted"])

    out = attempt_work.merge(miss_summary, on=group_cols, how="left", validate="one_to_one")
    numeric_fill = [
        "rebound_opportunity_misses",
        "actual_oreb_rate",
        "context_lineup_oreb_probability_oof",
        "actual_config_oreb_probability_oof",
        "role_swap_oreb_probability_oof",
        "crash_group_delta_raw",
        "shooter_zone_delta_raw",
        "points_per_oreb_oof",
    ]
    out[numeric_fill] = out[numeric_fill].fillna(0.0)

    prior_make = out["shot_quality_prior_make_prob"].astype(float)
    conversion = out["points_per_oreb_oof"].astype(float)
    out["actual_config_sc_ev_per_attempt"] = (
        (1.0 - prior_make)
        * out["actual_config_oreb_probability_oof"]
        * conversion
    )
    out["role_swap_sc_ev_per_attempt"] = (
        (1.0 - prior_make)
        * out["role_swap_oreb_probability_oof"]
        * conversion
    )
    out["role_vacancy_sc_ev_per_attempt"] = (
        out["actual_config_sc_ev_per_attempt"] - out["role_swap_sc_ev_per_attempt"]
    )
    out["role_vacancy_sc_points_total"] = (
        out["role_vacancy_sc_ev_per_attempt"] * out["attempts"]
    )
    season_make = out["shot_quality_season_make_prob"].astype(float)
    out["actual_config_sc_ev_per_attempt_season_sq"] = (
        (1.0 - season_make)
        * out["actual_config_oreb_probability_oof"]
        * conversion
    )
    out["role_swap_sc_ev_per_attempt_season_sq"] = (
        (1.0 - season_make)
        * out["role_swap_oreb_probability_oof"]
        * conversion
    )
    out["role_vacancy_sc_ev_per_attempt_season_sq"] = (
        out["actual_config_sc_ev_per_attempt_season_sq"]
        - out["role_swap_sc_ev_per_attempt_season_sq"]
    )
    out["role_vacancy_sc_points_total_season_sq"] = (
        out["role_vacancy_sc_ev_per_attempt_season_sq"] * out["attempts"]
    )
    out["role_vacancy_oreb_probability_delta"] = (
        out["actual_config_oreb_probability_oof"]
        - out["role_swap_oreb_probability_oof"]
    )
    return out.sort_values(
        ["role_vacancy_sc_points_total", "attempts"],
        ascending=[True, False],
    ).reset_index(drop=True)


def _input_paths(
    processed_dir: Path,
    metric: str,
    years: Iterable[int],
    season_type: str,
) -> List[Tuple[Path, int, str]]:
    paths: List[Tuple[Path, int, str]] = []
    for year in years:
        suffix = f"{year % 100:02d}"
        if season_type in {"RS", "ALL"}:
            paths.append((processed_dir / f"{metric}{suffix}.parquet", year, "RS"))
        if season_type in {"PS", "ALL"}:
            paths.append((processed_dir / f"{metric}{suffix}_PS.parquet", year, "PS"))
    return paths


def _read_processed_window(
    processed_dir: Path,
    metric: str,
    years: Sequence[int],
    season_type: str,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path, year, sample in _input_paths(processed_dir, metric, years, season_type):
        if not path.exists():
            logging.warning("Skipping missing %s input %s", metric, path)
            continue
        frame = pd.read_parquet(path)
        frame["season_end_year"] = int(year)
        frame["season_type"] = sample
        frame["source_file"] = path.name
        frames.append(frame)
        logging.info("Loaded %s (%d rows)", path.name, len(frame))
    if not frames:
        raise FileNotFoundError(f"No {metric} inputs found for requested window")
    return pd.concat(frames, ignore_index=True)


def load_opportunities(
    processed_dir: Path,
    shot_quality_path: Path,
    years: Sequence[int],
    season_type: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    opportunities = _read_processed_window(
        processed_dir, "SHOOTER_OREB", years, season_type
    )
    required = [
        "game_id",
        "event_num",
        "Shooter",
        "Offensive_Rebound",
        *NON_SHOOTER_COLS,
        *DEFENDER_COLS,
    ]
    missing = sorted(set(required) - set(opportunities.columns))
    if missing:
        raise ValueError(f"SHOOTER_OREB inputs missing columns: {missing}")

    shot_columns = [
        "season_end_year",
        "season_type",
        "game_id",
        "event_num",
        "shooter_id",
        "shooter_name",
        "is_make",
        "shot_zone",
        "distance_bin",
        "shot_family",
        "event_action_type",
        "modifier_bundle",
        "shot_quality_prior_make_prob",
        "shot_quality_season_make_prob",
    ]
    if not shot_quality_path.exists():
        raise FileNotFoundError(f"Shot-quality table not found: {shot_quality_path}")
    normalized_years = [int(year) for year in years]
    shots = pd.read_parquet(
        shot_quality_path,
        columns=shot_columns,
        filters=[("season_end_year", "in", normalized_years)],
    )
    shots = shots[shots["season_end_year"].isin(normalized_years)].copy()
    if season_type != "ALL":
        shots = shots[shots["season_type"].str.upper().eq(season_type)].copy()
    shots["season_type"] = shots["season_type"].str.upper()
    shots["game_id"] = normalize_game_id(shots["game_id"])
    shots["event_num"] = pd.to_numeric(shots["event_num"], errors="coerce").astype("Int64")
    shots["shooter_id"] = pd.to_numeric(shots["shooter_id"], errors="coerce").astype("Int64")
    shots = shots.dropna(subset=["event_num", "shooter_id", "shot_zone"]).copy()
    shots["event_num"] = shots["event_num"].astype(int)
    shots["shooter_id"] = shots["shooter_id"].astype(int)

    attempts = (
        shots.groupby(["shooter_id", "shooter_name", "shot_zone"], observed=True)
        .agg(
            attempts=("event_num", "size"),
            shot_quality_prior_make_prob=("shot_quality_prior_make_prob", "mean"),
            shot_quality_season_make_prob=("shot_quality_season_make_prob", "mean"),
        )
        .reset_index()
    )

    opportunities["game_id"] = normalize_game_id(opportunities["game_id"])
    opportunities["event_num"] = pd.to_numeric(
        opportunities["event_num"], errors="coerce"
    ).astype("Int64")
    opportunities["Shooter"] = pd.to_numeric(
        opportunities["Shooter"], errors="coerce"
    ).astype("Int64")
    for column in (*NON_SHOOTER_COLS, *DEFENDER_COLS):
        opportunities[column] = pd.to_numeric(
            opportunities[column], errors="coerce"
        ).astype("Int64")
    before = len(opportunities)
    opportunities = opportunities.dropna(
        subset=["event_num", "Shooter", *NON_SHOOTER_COLS, *DEFENDER_COLS]
    ).copy()
    opportunities["event_num"] = opportunities["event_num"].astype(int)
    opportunities["Shooter"] = opportunities["Shooter"].astype(int)
    for column in (*NON_SHOOTER_COLS, *DEFENDER_COLS):
        opportunities[column] = opportunities[column].astype(int)

    join_cols = ["game_id", "event_num"]
    shot_misses = shots[~shots["is_make"].astype(bool)].drop_duplicates(join_cols)
    joined = opportunities.merge(
        shot_misses[
            join_cols
            + [
                "shooter_id",
                "shooter_name",
                "shot_zone",
                "distance_bin",
                "shot_family",
                "event_action_type",
                "modifier_bundle",
                "shot_quality_prior_make_prob",
                "shot_quality_season_make_prob",
            ]
        ],
        on=join_cols,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    shooter_match = joined["Shooter"].eq(joined["shooter_id"])
    matched = joined["_merge"].eq("both") & shooter_match
    audit = {
        "input_opportunity_rows": int(before),
        "complete_lineup_opportunity_rows": int(len(opportunities)),
        "shot_quality_joined_rows": int(matched.sum()),
        "shot_quality_join_coverage": float(matched.mean()) if len(joined) else 0.0,
        "shooter_id_mismatch_rows": int(
            (joined["_merge"].eq("both") & ~shooter_match).sum()
        ),
    }
    joined = joined[matched].drop(columns=["_merge"]).copy()
    joined = joined.rename(
        columns={
            "Shooter": "shooter_id_processed",
            "Offensive_Rebound": "actual_oreb",
        }
    )
    joined["shooter_id"] = joined["shooter_id"].astype(int)
    joined["actual_oreb"] = joined["actual_oreb"].astype(float)
    joined["season_sample"] = (
        joined["season_end_year"].astype(str) + "_" + joined["season_type"].astype(str)
    )
    for column in CONTEXT_COLS:
        joined[column] = joined[column].fillna("unknown").astype(str)
    return joined, attempts, audit


def build_feature_space(df: pd.DataFrame) -> FeatureSpace:
    offset = 0
    context_columns: Dict[Tuple[str, str], int] = {}
    for column in CONTEXT_COLS:
        for value in sorted(df[column].astype(str).unique()):
            context_columns[(column, value)] = offset
            offset += 1

    non_shooter_ids = sorted(
        {
            normalize_player_id(value)
            for column in NON_SHOOTER_COLS
            for value in df[column].dropna().unique()
        }
    )
    non_shooter_columns = {
        player_id: offset + idx for idx, player_id in enumerate(non_shooter_ids)
    }
    offset += len(non_shooter_columns)

    defender_ids = sorted(
        {
            normalize_player_id(value)
            for column in DEFENDER_COLS
            for value in df[column].dropna().unique()
        }
    )
    defender_columns = {
        player_id: offset + idx for idx, player_id in enumerate(defender_ids)
    }
    offset += len(defender_columns)
    shooter_zone_start = offset

    shooter_zone_keys = sorted(
        {
            (normalize_player_id(shooter), str(zone))
            for shooter, zone in df[["shooter_id", "shot_zone"]].itertuples(
                index=False, name=None
            )
        }
    )
    shooter_zone_columns = {
        key: offset + idx for idx, key in enumerate(shooter_zone_keys)
    }
    offset += len(shooter_zone_columns)
    return FeatureSpace(
        context_columns=context_columns,
        non_shooter_columns=non_shooter_columns,
        defender_columns=defender_columns,
        shooter_zone_columns=shooter_zone_columns,
        shooter_zone_start=shooter_zone_start,
        n_features=offset,
    )


def build_design_matrix(df: pd.DataFrame, space: FeatureSpace) -> csr_matrix:
    row_indices: List[int] = []
    column_indices: List[int] = []
    data: List[float] = []

    def add(row_idx: int, column_idx: int) -> None:
        row_indices.append(row_idx)
        column_indices.append(column_idx)
        data.append(1.0)

    for row_idx, row in enumerate(df.itertuples(index=False)):
        values = row._asdict()
        for column in CONTEXT_COLS:
            feature = space.context_columns.get((column, str(values[column])))
            if feature is not None:
                add(row_idx, feature)
        for column in NON_SHOOTER_COLS:
            feature = space.non_shooter_columns.get(
                normalize_player_id(values[column])
            )
            if feature is not None:
                add(row_idx, feature)
        for column in DEFENDER_COLS:
            feature = space.defender_columns.get(normalize_player_id(values[column]))
            if feature is not None:
                add(row_idx, feature)
        key = (
            normalize_player_id(values["shooter_id"]),
            str(values["shot_zone"]),
        )
        feature = space.shooter_zone_columns.get(key)
        if feature is not None:
            add(row_idx, feature)

    return coo_matrix(
        (data, (row_indices, column_indices)),
        shape=(len(df), space.n_features),
        dtype=np.float64,
    ).tocsr()


def _coefficient_maps(
    coefficients: np.ndarray,
    space: FeatureSpace,
) -> Tuple[Dict[int, float], Dict[Tuple[int, str], float]]:
    non_shooter = {
        player_id: float(coefficients[column])
        for player_id, column in space.non_shooter_columns.items()
    }
    shooter_zone = {
        key: float(coefficients[column])
        for key, column in space.shooter_zone_columns.items()
    }
    return non_shooter, shooter_zone


def _read_second_chance_conversion(
    processed_dir: Path,
    years: Sequence[int],
    season_type: str,
) -> pd.DataFrame:
    sc = _read_processed_window(
        processed_dir, "SECOND_CHANCE_CLEAN", years, season_type
    )
    if "Off_Diff" not in sc.columns:
        raise ValueError("SECOND_CHANCE_CLEAN input is missing Off_Diff")
    sc["game_id"] = normalize_game_id(sc["game_id"])
    return (
        sc.groupby("game_id", observed=True)["Off_Diff"]
        .sum()
        .rename("second_chance_points")
        .reset_index()
    )


def crossfit_opportunity_prior(
    opportunities: pd.DataFrame,
    second_chance_by_game: pd.DataFrame,
    *,
    folds: int,
    alpha: float,
    seed: int,
    bootstrap_repetitions: int = 1000,
    clip_bounds: Tuple[float, float] = (0.01, 0.99),
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if opportunities["game_id"].nunique() < folds:
        raise ValueError("Not enough games for requested number of folds")

    space = build_feature_space(opportunities)
    logging.info(
        "Building sparse design: %d rows, %d base features, %d shooter-zone features",
        len(opportunities),
        space.shooter_zone_start,
        space.n_features - space.shooter_zone_start,
    )
    design = build_design_matrix(opportunities, space)
    base_design = design[:, : space.shooter_zone_start]
    y = opportunities["actual_oreb"].to_numpy(dtype=float)
    groups = opportunities["game_id"].astype(str).to_numpy()

    context_lineup_probability = np.full(len(opportunities), np.nan)
    actual_config_probability = np.full(len(opportunities), np.nan)
    role_swap_probability = np.full(len(opportunities), np.nan)
    actual_minus_swap_raw = np.full(len(opportunities), np.nan)
    crash_group_delta_raw = np.full(len(opportunities), np.nan)
    shooter_zone_delta_raw = np.full(len(opportunities), np.nan)
    points_per_oreb = np.full(len(opportunities), np.nan)
    fold_id = np.full(len(opportunities), -1, dtype=int)
    fold_metrics: List[Dict[str, object]] = []

    splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    sc_game = second_chance_by_game.set_index("game_id")["second_chance_points"]
    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(design, y, groups=groups), start=1
    ):
        base_model = Ridge(
            alpha=float(alpha),
            fit_intercept=True,
            solver="lsqr",
            tol=1e-5,
            max_iter=2000,
        )
        enhanced_model = Ridge(
            alpha=float(alpha),
            fit_intercept=True,
            solver="lsqr",
            tol=1e-5,
            max_iter=2000,
        )
        base_model.fit(base_design[train_idx], y[train_idx])
        enhanced_model.fit(design[train_idx], y[train_idx])

        base_raw = base_model.predict(base_design[valid_idx])
        enhanced_raw = enhanced_model.predict(design[valid_idx])
        base_prob = np.clip(base_raw, clip_bounds[0], clip_bounds[1])
        enhanced_prob = np.clip(enhanced_raw, clip_bounds[0], clip_bounds[1])
        context_lineup_probability[valid_idx] = base_prob
        actual_config_probability[valid_idx] = enhanced_prob
        fold_id[valid_idx] = fold

        non_shooter_coefficients, shooter_zone_coefficients = _coefficient_maps(
            enhanced_model.coef_, space
        )
        for local_idx, global_idx in enumerate(valid_idx):
            row = opportunities.iloc[global_idx]
            swap = role_swap_counterfactual(
                actual_raw_prediction=float(enhanced_raw[local_idx]),
                shooter_id=int(row["shooter_id"]),
                non_shooter_ids=[int(row[column]) for column in NON_SHOOTER_COLS],
                shot_zone=str(row["shot_zone"]),
                non_shooter_coefficients=non_shooter_coefficients,
                shooter_zone_coefficients=shooter_zone_coefficients,
                clip_bounds=clip_bounds,
            )
            role_swap_probability[global_idx] = swap.equal_teammate_swap_probability
            actual_minus_swap_raw[global_idx] = swap.actual_minus_swap_raw
            crash_group_delta_raw[global_idx] = swap.crash_group_delta_raw
            shooter_zone_delta_raw[global_idx] = swap.shooter_zone_delta_raw

        train_games = set(groups[train_idx])
        train_orebs = float(y[train_idx].sum())
        train_sc_points = float(sc_game.reindex(list(train_games)).fillna(0.0).sum())
        conversion = train_sc_points / train_orebs if train_orebs > 0 else 0.0
        points_per_oreb[valid_idx] = conversion
        fold_metrics.append(
            {
                "fold": fold,
                "train_rows": int(len(train_idx)),
                "validation_rows": int(len(valid_idx)),
                "train_games": int(len(train_games)),
                "validation_games": int(len(set(groups[valid_idx]))),
                "game_overlap": int(
                    len(train_games.intersection(set(groups[valid_idx])))
                ),
                "points_per_offensive_rebound_train_only": conversion,
                "base_brier": float(brier_score_loss(y[valid_idx], base_prob)),
                "enhanced_brier": float(
                    brier_score_loss(y[valid_idx], enhanced_prob)
                ),
                "base_log_loss": float(
                    log_loss(y[valid_idx], base_prob, labels=[0.0, 1.0])
                ),
                "enhanced_log_loss": float(
                    log_loss(y[valid_idx], enhanced_prob, labels=[0.0, 1.0])
                ),
            }
        )
        logging.info(
            "Fold %d/%d complete: base Brier %.6f, enhanced %.6f, conversion %.4f",
            fold,
            folds,
            fold_metrics[-1]["base_brier"],
            fold_metrics[-1]["enhanced_brier"],
            conversion,
        )

    out = opportunities.copy()
    out["fold"] = fold_id
    out["context_lineup_probability_oof"] = context_lineup_probability
    out["actual_config_probability_oof"] = actual_config_probability
    out["role_swap_probability_oof"] = role_swap_probability
    out["role_vacancy_oreb_probability_delta"] = (
        actual_config_probability - role_swap_probability
    )
    out["actual_minus_swap_raw"] = actual_minus_swap_raw
    out["crash_group_delta_raw"] = crash_group_delta_raw
    out["shooter_zone_delta_raw"] = shooter_zone_delta_raw
    out["role_swap_raw_reconciliation_gap"] = (
        actual_minus_swap_raw - crash_group_delta_raw - shooter_zone_delta_raw
    )
    out["points_per_oreb_oof"] = points_per_oreb
    out["actual_config_sc_ev_on_observed_attempt"] = [
        second_chance_ev_per_attempt(
            make_probability=make_prob,
            offensive_rebound_probability=oreb_probability,
            points_per_offensive_rebound=conversion,
        )
        for make_prob, oreb_probability, conversion in zip(
            out["shot_quality_prior_make_prob"],
            out["actual_config_probability_oof"],
            out["points_per_oreb_oof"],
        )
    ]
    out["role_swap_sc_ev_on_observed_attempt"] = [
        second_chance_ev_per_attempt(
            make_probability=make_prob,
            offensive_rebound_probability=oreb_probability,
            points_per_offensive_rebound=conversion,
        )
        for make_prob, oreb_probability, conversion in zip(
            out["shot_quality_prior_make_prob"],
            out["role_swap_probability_oof"],
            out["points_per_oreb_oof"],
        )
    ]

    base_all = np.clip(context_lineup_probability, clip_bounds[0], clip_bounds[1])
    enhanced_all = np.clip(actual_config_probability, clip_bounds[0], clip_bounds[1])
    brier_row_gain = (y - base_all) ** 2 - (y - enhanced_all) ** 2
    log_loss_row_gain = (
        -(y * np.log(base_all) + (1.0 - y) * np.log(1.0 - base_all))
        + (y * np.log(enhanced_all) + (1.0 - y) * np.log(1.0 - enhanced_all))
    )
    brier_bootstrap = game_cluster_bootstrap_mean(
        groups,
        brier_row_gain,
        repetitions=bootstrap_repetitions,
        seed=seed + 101,
    )
    log_loss_bootstrap = game_cluster_bootstrap_mean(
        groups,
        log_loss_row_gain,
        repetitions=bootstrap_repetitions,
        seed=seed + 202,
    )
    metrics: Dict[str, object] = {
        "rows": int(len(out)),
        "games": int(out["game_id"].nunique()),
        "folds": int(folds),
        "alpha": float(alpha),
        "seed": int(seed),
        "base_brier_oof": float(brier_score_loss(y, base_all)),
        "enhanced_brier_oof": float(brier_score_loss(y, enhanced_all)),
        "brier_gain": float(
            brier_score_loss(y, base_all) - brier_score_loss(y, enhanced_all)
        ),
        "base_log_loss_oof": float(log_loss(y, base_all, labels=[0.0, 1.0])),
        "enhanced_log_loss_oof": float(
            log_loss(y, enhanced_all, labels=[0.0, 1.0])
        ),
        "log_loss_gain": float(
            log_loss(y, base_all, labels=[0.0, 1.0])
            - log_loss(y, enhanced_all, labels=[0.0, 1.0])
        ),
        "brier_gain_game_bootstrap": brier_bootstrap,
        "log_loss_gain_game_bootstrap": log_loss_bootstrap,
        "bootstrap_repetitions": int(bootstrap_repetitions),
        "max_abs_role_swap_raw_reconciliation_gap": float(
            np.nanmax(np.abs(out["role_swap_raw_reconciliation_gap"]))
        ),
        "fold_metrics": fold_metrics,
    }
    return out, metrics


def build_player_name_map(attempts: pd.DataFrame) -> Dict[int, str]:
    names: Dict[int, str] = {}
    for player_id, name in attempts[["shooter_id", "shooter_name"]].dropna().itertuples(
        index=False, name=None
    ):
        key = normalize_player_id(player_id)
        candidate = str(name).strip()
        current = names.get(key, "")
        if len(candidate) > len(current):
            names[key] = candidate
    return names


def write_outputs(
    output_dir: Path,
    opportunities: pd.DataFrame,
    player_zone: pd.DataFrame,
    player_shot_mix: pd.DataFrame,
    metrics: Mapping[str, object],
    manifest: Mapping[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    opportunities.to_parquet(
        output_dir / "miss_opportunity_predictions.parquet", index=False
    )
    player_zone.to_csv(output_dir / "player_zone_accounting.csv", index=False)
    player_shot_mix.to_csv(
        output_dir / "player_shot_mix_accounting.csv", index=False
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the cross-fitted second-chance opportunity prior."
    )
    parser.add_argument("start_year", type=int)
    parser.add_argument("end_year", type=int)
    parser.add_argument(
        "--season-type",
        default="ALL",
        choices=["RS", "PS", "ALL", "rs", "ps", "all"],
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument(
        "--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR
    )
    parser.add_argument(
        "--shot-quality-path", type=Path, default=DEFAULT_SHOT_QUALITY_PATH
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    years = expand_year_window(args.start_year, args.end_year)
    season_type = season_type_label(args.season_type)
    opportunities, attempts, join_audit = load_opportunities(
        args.processed_dir,
        args.shot_quality_path,
        years,
        season_type,
    )
    second_chance_by_game = _read_second_chance_conversion(
        args.processed_dir,
        years,
        season_type,
    )
    predictions, metrics = crossfit_opportunity_prior(
        opportunities,
        second_chance_by_game,
        folds=args.folds,
        alpha=args.alpha,
        seed=args.seed,
        bootstrap_repetitions=args.bootstrap_repetitions,
    )
    metrics["join_audit"] = join_audit
    player_zone = aggregate_player_zone_accounting(predictions, attempts)
    reporting_opportunities = predictions.copy()
    reporting_attempts = attempts.copy()
    reporting_opportunities["shot_zone"] = reporting_shot_zone(
        reporting_opportunities["shot_zone"]
    )
    reporting_attempts["shot_zone"] = reporting_shot_zone(
        reporting_attempts["shot_zone"]
    )
    player_shot_mix = aggregate_player_zone_accounting(
        reporting_opportunities,
        reporting_attempts,
    )
    manifest = {
        "artifact": "second_chance_opportunity_prior",
        "status": "retrospective_research",
        "years": years,
        "season_type": season_type,
        "inputs": {
            "shooter_oreb": str(args.processed_dir / "SHOOTER_OREB{YY}[_PS].parquet"),
            "shot_quality": str(args.shot_quality_path),
            "second_chance_clean": str(
                args.processed_dir / "SECOND_CHANCE_CLEAN{YY}[_PS].parquet"
            ),
        },
        "model": {
            "target": "P(offensive rebound | eligible missed FGA)",
            "validation": f"{args.folds}-fold GroupKFold by game_id",
            "alpha": float(args.alpha),
            "base_features": [
                *CONTEXT_COLS,
                "four exact non-shooter identities",
                "five exact defender identities",
            ],
            "enhanced_feature": "shooter identity x shot zone",
            "counterfactual": (
                "Each of the four teammates takes the same zone shot in turn; "
                "the actual shooter replaces that teammate in the non-shooter group."
            ),
            "ev_formula": (
                "(1 - repo_owned_SQ_prior_make_probability) "
                "* P(OREB | miss) * training_fold_points_per_OREB"
            ),
        },
        "interpretation_guardrails": [
            "This is empirical role-vacancy accounting, not optical rebound tracking.",
            "Player effects can contain scheme, lineup, transition-defense, and health context.",
            "The role swap is not a claim that the teammate would literally take the attempt.",
            "No active public weighted_factors_decomp_* input or output is modified.",
            "Legacy Alt3 remains an audit surface and is not used as this model's target.",
        ],
        "outputs": {
            "miss_predictions": "miss_opportunity_predictions.parquet",
            "player_zone_accounting": (
                "player_zone_accounting.csv (canonical SQ zones)"
            ),
            "player_shot_mix_accounting": (
                "player_shot_mix_accounting.csv "
                "(long_mid + deep_two reported as long_two)"
            ),
            "metrics": "metrics.json",
        },
    }
    write_outputs(
        args.output_dir,
        predictions,
        player_zone,
        player_shot_mix,
        metrics,
        manifest,
    )
    logging.info("Wrote second-chance opportunity prior to %s", args.output_dir)


if __name__ == "__main__":
    main()
