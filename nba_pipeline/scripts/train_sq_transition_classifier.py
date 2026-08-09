#!/usr/bin/env python3
"""
Train historical-safe transition classifiers from ShotQuality labels.

Labels come from the recovered ShotQuality descriptor lists. Features come
only from ordinary NBA play-by-play fields available outside the licensed SQ
window: current shot action/location text, period clock, and preceding event
sequence. Player/team identity and current-shot make/miss are excluded.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

try:
    from shotquality_modeling import (
        normalize_game_id,
        parse_descriptor_tags,
        probability_metrics,
        write_json,
    )
except ImportError:  # pragma: no cover - package import in tests
    from nba_pipeline.scripts.shotquality_modeling import (
        normalize_game_id,
        parse_descriptor_tags,
        probability_metrics,
        write_json,
    )


LOGGER = logging.getLogger("sq_transition_classifier")
DEFAULT_SQ_ROOT = Path(
    "/Users/russ/Documents/Codex/basketball/nba/research/shotquality/nba_games/2025"
)
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_PATHS = [
    PIPELINE_ROOT / "raw_data" / "NBA25.parquet",
    PIPELINE_ROOT / "raw_data" / "NBA25_PS.parquet",
]
DEFAULT_FORWARD_RAW_PATHS = [
    PIPELINE_ROOT / "raw_data" / "NBA26.parquet",
    PIPELINE_ROOT / "raw_data" / "NBA26_PS.parquet",
]
DEFAULT_OUTPUT_DIR = PIPELINE_ROOT / "results" / "sq_transition_classifier"

LABEL_SOURCE_COLUMNS = [
    "game_id",
    "GAME_EVENT_ID",
    "action_type",
    "play_descriptors",
]
RAW_COLUMNS = [
    "game_id",
    "event_num",
    "event_type",
    "event_action_type",
    "period",
    "time_quarter",
    "minute_remaining_quarter",
    "seconds_remaining_quarter",
    "home_description",
    "neutral_description",
    "visitor_description",
]
SHOT_ACTION_FLAGS = [
    "3PT",
    "DUNK",
    "LAYUP",
    "HOOK",
    "FLOAT",
    "JUMP",
    "PULLUP",
    "STEP BACK",
    "FADEAWAY",
    "CUTTING",
    "DRIVING",
    "PUTBACK",
    "TURNAROUND",
    "ALLEY OOP",
    "BANK",
]
PRIOR_DESCRIPTION_FLAGS = [
    "STEAL",
    "REBOUND",
    "TURNOVER",
    "MISS",
    "PTS",
    "BLOCK",
    "FREE THROW",
    "JUMP BALL",
]
EVENT_TYPE_CODES = list(range(1, 14)) + [18]
CHANGE_EVENT_TYPES = {1, 4, 5, 10}


def load_sq_transition_labels(root: Path, *, limit_games: int = 0) -> pd.DataFrame:
    files = sorted(root.glob("*.csv"))
    if limit_games:
        files = files[:limit_games]
    if not files:
        raise FileNotFoundError(f"No ShotQuality game CSVs found under {root}")

    frames: list[pd.DataFrame] = []
    for file_path in files:
        header = set(pd.read_csv(file_path, nrows=0).columns)
        usecols = [column for column in LABEL_SOURCE_COLUMNS if column in header]
        frame = pd.read_csv(file_path, usecols=usecols, low_memory=False)
        for column in LABEL_SOURCE_COLUMNS:
            if column not in frame.columns:
                frame[column] = np.nan
        frames.append(frame)
    labels = pd.concat(frames, ignore_index=True)
    labels = labels[
        labels["action_type"].astype("string").str.lower().isin(["2pt", "3pt"])
        & labels["GAME_EVENT_ID"].notna()
    ].copy()
    labels["game_id"] = normalize_game_id(labels["game_id"])
    labels["event_num"] = pd.to_numeric(
        labels["GAME_EVENT_ID"], errors="coerce"
    ).astype("Int64")
    tags = labels["play_descriptors"].map(parse_descriptor_tags)
    labels["is_transition"] = tags.map(lambda values: int("transition" in values))
    labels["is_transition_dunk_or_layup"] = tags.map(
        lambda values: int(
            "transition" in values and bool({"dunk", "lay_up"} & set(values))
        )
    )
    labels = labels.dropna(subset=["game_id", "event_num"])
    return labels.drop_duplicates(["game_id", "event_num"], keep="first")


def load_raw_pbp(
    paths: Iterable[Path],
    *,
    extra_columns: Iterable[str] = (),
) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        columns = list(dict.fromkeys([*RAW_COLUMNS, *extra_columns]))
        available_columns = set(pq.ParquetFile(path).schema.names)
        required_columns = set(RAW_COLUMNS) - {"time_quarter"}
        missing_required = sorted(required_columns - available_columns)
        if missing_required:
            raise ValueError(
                f"{path} is missing required transition fields: "
                f"{missing_required}"
            )
        columns = [
            column for column in columns if column in available_columns
        ]
        frame = pd.read_parquet(path, columns=columns)
        frame["_raw_source"] = path.name
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    raw["game_id"] = normalize_game_id(raw["game_id"])
    raw["event_num"] = pd.to_numeric(raw["event_num"], errors="coerce").astype("Int64")
    return raw.drop_duplicates().reset_index(drop=True)


def _description_side(frame: pd.DataFrame) -> pd.Series:
    home = frame["home_description"].fillna("").astype(str)
    visitor = frame["visitor_description"].fillna("").astype(str)
    home_present = home.str.strip().ne("")
    visitor_present = visitor.str.strip().ne("")
    return pd.Series(
        np.select(
            [home_present & ~visitor_present, visitor_present & ~home_present],
            ["home", "away"],
            default="neutral",
        ),
        index=frame.index,
        dtype="string",
    )


def _shot_description(frame: pd.DataFrame) -> pd.Series:
    home = frame["home_description"].fillna("").astype(str)
    visitor = frame["visitor_description"].fillna("").astype(str)
    shot_pattern = r"\bMISS\b|\bPTS\b|\b3PT\b|\bDUNK\b|\bLAYUP\b|\bJUMP SHOT\b"
    visitor_is_shot = visitor.str.contains(shot_pattern, case=False, regex=True)
    return home.where(~visitor_is_shot, visitor).str.upper()


def prepare_sequence_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Sort clock-first and attach only information known before each event."""
    frame = raw.copy()
    minute = pd.to_numeric(
        frame["minute_remaining_quarter"], errors="coerce"
    )
    second = pd.to_numeric(
        frame["seconds_remaining_quarter"], errors="coerce"
    )
    component_clock = minute * 60 + second
    source = (
        frame["_raw_source"].astype(str)
        if "_raw_source" in frame.columns
        else pd.Series("__single_frame__", index=frame.index)
    )
    seconds_are_total = second.groupby(source).transform("max").gt(59)
    fallback_clock = component_clock.where(~seconds_are_total, second)

    if "time_quarter" in frame.columns:
        parsed = frame["time_quarter"].astype("string").str.extract(
            r"^\s*(\d+):(\d+(?:\.\d+)?)\s*$"
        )
        parsed_clock = (
            pd.to_numeric(parsed[0], errors="coerce") * 60
            + pd.to_numeric(parsed[1], errors="coerce")
        )
        frame["clock_sec"] = parsed_clock.fillna(fallback_clock)
    else:
        frame["clock_sec"] = fallback_clock
    frame = frame.sort_values(
        ["game_id", "period", "clock_sec", "event_num"],
        ascending=[True, True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    group_keys = [frame["game_id"], frame["period"]]
    frame["event_side"] = _description_side(frame)
    combined_description = (
        frame["home_description"].fillna("").astype(str)
        + " "
        + frame["neutral_description"].fillna("").astype(str)
        + " "
        + frame["visitor_description"].fillna("").astype(str)
    ).str.upper()
    frame["combined_description"] = combined_description

    for lag in (1, 2, 3):
        frame[f"prev{lag}_event_type"] = frame.groupby(
            ["game_id", "period"], sort=False
        )["event_type"].shift(lag)
        frame[f"prev{lag}_clock_sec"] = frame.groupby(
            ["game_id", "period"], sort=False
        )["clock_sec"].shift(lag)
        frame[f"prev{lag}_event_side"] = frame.groupby(
            ["game_id", "period"], sort=False
        )["event_side"].shift(lag)
        frame[f"prev{lag}_description"] = frame.groupby(
            ["game_id", "period"], sort=False
        )["combined_description"].shift(lag)
        frame[f"elapsed_since_prev{lag}"] = (
            frame[f"prev{lag}_clock_sec"] - frame["clock_sec"]
        ).clip(lower=0)

    change_clock = frame["clock_sec"].where(frame["event_type"].isin(CHANGE_EVENT_TYPES))
    change_type = frame["event_type"].where(frame["event_type"].isin(CHANGE_EVENT_TYPES))
    frame["last_change_clock_sec"] = change_clock.groupby(group_keys, sort=False).ffill()
    frame["last_change_event_type"] = change_type.groupby(group_keys, sort=False).ffill()
    frame["last_change_clock_sec"] = frame.groupby(
        ["game_id", "period"], sort=False
    )["last_change_clock_sec"].shift(1)
    frame["last_change_event_type"] = frame.groupby(
        ["game_id", "period"], sort=False
    )["last_change_event_type"].shift(1)
    frame["seconds_since_last_change"] = (
        frame["last_change_clock_sec"] - frame["clock_sec"]
    ).clip(lower=0)
    return frame


def build_historical_feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Build fields available in ordinary historical play-by-play."""
    feature_values: dict[str, pd.Series] = {}
    numeric = [
        "period",
        "clock_sec",
        "event_action_type",
        "elapsed_since_prev1",
        "elapsed_since_prev2",
        "elapsed_since_prev3",
        "seconds_since_last_change",
    ]
    for column in numeric:
        feature_values[column] = pd.to_numeric(frame[column], errors="coerce")

    shot_description = _shot_description(frame)
    distance = shot_description.str.extract(r"(\d{1,2})'", expand=False)
    feature_values["shot_distance_ft"] = pd.to_numeric(distance, errors="coerce")
    for label in SHOT_ACTION_FLAGS:
        safe_name = re.sub(r"[^A-Z0-9]+", "_", label).strip("_").lower()
        feature_values[f"shot_action::{safe_name}"] = shot_description.str.contains(
            re.escape(label), regex=True
        ).astype(int)

    action_type = pd.to_numeric(frame["event_action_type"], errors="coerce")
    for code in sorted(action_type.dropna().astype(int).unique()):
        feature_values[f"shot_action_type::{code}"] = action_type.eq(code).astype(int)

    current_side = _description_side(frame)
    for lag in (1, 2, 3):
        previous_type = pd.to_numeric(
            frame[f"prev{lag}_event_type"], errors="coerce"
        )
        for code in EVENT_TYPE_CODES:
            feature_values[f"prev{lag}_event_type::{code}"] = previous_type.eq(code).astype(int)
        previous_side = frame[f"prev{lag}_event_side"].astype("string")
        feature_values[f"prev{lag}_was_opponent_side"] = (
            previous_side.notna()
            & current_side.isin(["home", "away"])
            & previous_side.isin(["home", "away"])
            & previous_side.ne(current_side)
        ).astype(int)
        previous_description = (
            frame[f"prev{lag}_description"].fillna("").astype(str).str.upper()
        )
        for label in PRIOR_DESCRIPTION_FLAGS:
            safe_name = re.sub(r"[^A-Z0-9]+", "_", label).strip("_").lower()
            feature_values[f"prev{lag}_desc::{safe_name}"] = (
                previous_description.str.contains(re.escape(label), regex=True).astype(int)
            )

    last_change_type = pd.to_numeric(
        frame["last_change_event_type"], errors="coerce"
    )
    for code in sorted(CHANGE_EVENT_TYPES):
        feature_values[f"last_change_type::{code}"] = last_change_type.eq(code).astype(int)

    features = pd.DataFrame(feature_values, index=frame.index)
    validate_historical_feature_names(features.columns)
    return features.astype(float)


def validate_historical_feature_names(feature_names: Iterable[str]) -> None:
    forbidden_tokens = (
        "transition",
        "made",
        "shot_made",
        "initial_ev",
        "dynamic_ev",
        "player_id",
        "team_id",
        "score_margin",
    )
    invalid = [
        name
        for name in feature_names
        if any(token in name.lower() for token in forbidden_tokens)
    ]
    if invalid:
        raise ValueError(f"Leaking/nonportable transition features: {invalid}")


def new_model(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.06,
        max_iter=180,
        max_leaf_nodes=31,
        min_samples_leaf=80,
        l2_regularization=3.0,
        early_stopping=False,
        random_state=seed,
    )


def grouped_oof(
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    *,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions = np.full(len(features), np.nan)
    prevalence_baseline = np.full(len(features), np.nan)
    fold_ids = np.full(len(features), -1)
    splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold_id, (train_idx, test_idx) in enumerate(
        splitter.split(features, target, groups)
    ):
        model = new_model(seed + fold_id)
        model.fit(features.iloc[train_idx], target.iloc[train_idx])
        predictions[test_idx] = model.predict_proba(features.iloc[test_idx])[:, 1]
        prevalence_baseline[test_idx] = float(target.iloc[train_idx].mean())
        fold_ids[test_idx] = fold_id
    return predictions, prevalence_baseline, fold_ids


def chronological_holdout(
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    *,
    seed: int,
    fraction: float = 0.2,
) -> dict[str, object]:
    games = np.array(sorted(groups.astype(str).unique()))
    cut = max(1, int(np.floor(len(games) * (1.0 - fraction))))
    train_games = set(games[:cut])
    train_mask = groups.astype(str).isin(train_games).to_numpy()
    test_mask = ~train_mask
    model = new_model(seed)
    model.fit(features.loc[train_mask], target.loc[train_mask])
    probability = model.predict_proba(features.loc[test_mask])[:, 1]
    prevalence = np.repeat(float(target.loc[train_mask].mean()), test_mask.sum())
    return {
        "model": probability_metrics(target.loc[test_mask], probability),
        "prevalence_baseline": probability_metrics(
            target.loc[test_mask], prevalence
        ),
        "test_game_count": int(len(games[cut:])),
        "first_test_game": str(games[cut]),
        "last_test_game": str(games[-1]),
    }


def run_target(
    joined: pd.DataFrame,
    features: pd.DataFrame,
    target_name: str,
    *,
    folds: int,
    seed: int,
    output_dir: Path,
) -> dict[str, object]:
    target = joined[target_name].astype(int).reset_index(drop=True)
    groups = joined["game_id"].astype(str).reset_index(drop=True)
    probability, baseline, fold_ids = grouped_oof(
        features, target, groups, folds=folds, seed=seed
    )
    metrics = {
        "model": probability_metrics(target, probability),
        "prevalence_baseline": probability_metrics(target, baseline),
    }


def evaluate_forward_descriptor_season(
    *,
    raw_paths: Iterable[Path],
    output_dir: Path,
    training_feature_columns: list[str],
    training_games: Iterable[str] = (),
) -> dict[str, object]:
    """Score untouched SQ labels from the next season without refitting."""
    label_columns = ["is_transition", "is_transition_dunk_or_layup"]
    forward = prepare_sequence_frame(
        load_raw_pbp(raw_paths, extra_columns=label_columns)
    )
    forward = forward[
        forward["event_type"].isin([1, 2])
        & forward["is_transition"].notna()
        & forward["is_transition_dunk_or_layup"].notna()
    ].reset_index(drop=True)
    if forward.empty:
        raise ValueError("Forward raw files contain no non-null SQ transition labels")
    overlap = set(forward["game_id"].astype(str)) & set(map(str, training_games))
    if overlap:
        raise ValueError(
            f"Forward evaluation overlaps {len(overlap)} training game ids"
        )

    forward_features = build_historical_feature_matrix(forward)
    forward_features = forward_features.reindex(
        columns=training_feature_columns, fill_value=0.0
    )
    output: dict[str, object] = {
        "raw_paths": [str(path) for path in raw_paths],
        "rows": int(len(forward)),
        "games": int(forward["game_id"].nunique()),
        "targets": {},
    }
    for target_name in label_columns:
        bundle = joblib.load(
            output_dir / f"{target_name}_historical_safe_model.joblib"
        )
        if list(bundle["feature_columns"]) != training_feature_columns:
            raise ValueError(
                f"{target_name} model feature contract differs from metrics JSON"
            )
        target = forward[target_name].astype(bool).astype(int)
        probability = bundle["model"].predict_proba(forward_features)[:, 1]
        output["targets"][target_name] = {
            "positive_rows": int(target.sum()),
            "positive_rate": float(target.mean()),
            "metrics": probability_metrics(target, probability),
        }
        pd.DataFrame(
            {
                "game_id": forward["game_id"].astype(str),
                "event_num": forward["event_num"],
                "target": target,
                "forward_probability": probability,
            }
        ).to_parquet(
            output_dir / f"{target_name}_forward_2025_26_predictions.parquet",
            index=False,
        )
    return output
    temporal = chronological_holdout(
        features, target, groups, seed=seed
    )
    final_model = new_model(seed)
    final_model.fit(features, target)
    model_path = output_dir / f"{target_name}_historical_safe_model.joblib"
    joblib.dump(
        {
            "model": final_model,
            "feature_columns": list(features.columns),
            "target": target_name,
            "training_rows": int(len(joined)),
            "training_games": int(groups.nunique()),
            "leakage_contract": (
                "Ordinary PBP only; current-shot make/miss, score, player/team "
                "identity, and all ShotQuality features/EV outputs excluded."
            ),
        },
        model_path,
    )
    pd.DataFrame(
        {
            "game_id": groups,
            "event_num": joined["event_num"],
            "target": target,
            "oof_probability": probability,
            "prevalence_baseline": baseline,
            "outer_fold": fold_ids.astype(int),
        }
    ).to_parquet(
        output_dir / f"{target_name}_oof_predictions.parquet", index=False
    )
    return {
        "rows": int(len(joined)),
        "positive_rows": int(target.sum()),
        "positive_rate": float(target.mean()),
        "games": int(groups.nunique()),
        "metrics": metrics,
        "chronological_last_20pct_holdout": temporal,
        "model_path": str(model_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sq-root", type=Path, default=DEFAULT_SQ_ROOT)
    parser.add_argument(
        "--raw-paths", nargs="+", type=Path, default=DEFAULT_RAW_PATHS
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--limit-games", type=int, default=0)
    parser.add_argument(
        "--forward-raw-paths",
        nargs="+",
        type=Path,
        default=DEFAULT_FORWARD_RAW_PATHS,
    )
    parser.add_argument(
        "--reuse-models-forward-only",
        action="store_true",
        help=(
            "Load the existing metrics/model artifacts and only refresh the "
            "untouched 2025-26 descriptor-label evaluation."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = args.output_dir / "sq_transition_classifier_metrics.json"
    if args.reuse_models_forward_only:
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        import json

        results = json.loads(metrics_path.read_text())
        results["forward_2025_26_descriptor_holdout"] = (
            evaluate_forward_descriptor_season(
                raw_paths=args.forward_raw_paths,
                output_dir=args.output_dir,
                training_feature_columns=results["model_spec"]["feature_columns"],
            )
        )
        write_json(metrics_path, results)
        LOGGER.info("Wrote %s", metrics_path)
        return

    labels = load_sq_transition_labels(args.sq_root, limit_games=args.limit_games)
    raw = prepare_sequence_frame(load_raw_pbp(args.raw_paths))
    joined = raw.merge(
        labels[
            [
                "game_id",
                "event_num",
                "is_transition",
                "is_transition_dunk_or_layup",
            ]
        ],
        on=["game_id", "event_num"],
        how="inner",
        validate="one_to_one",
    ).reset_index(drop=True)
    features = build_historical_feature_matrix(joined).reset_index(drop=True)
    results = {
        "inventory": {
            "sq_root": str(args.sq_root),
            "sq_labeled_rows": int(len(labels)),
            "sq_games": int(labels["game_id"].nunique()),
            "raw_paths": [str(path) for path in args.raw_paths],
            "raw_rows": int(len(raw)),
            "joined_shot_rows": int(len(joined)),
            "join_coverage": float(len(joined) / len(labels)),
            "joined_games": int(joined["game_id"].nunique()),
        },
        "model_spec": {
            "estimator": "HistGradientBoostingClassifier",
            "feature_count": int(features.shape[1]),
            "feature_columns": list(features.columns),
            "outer_validation": (
                f"{args.folds}-fold shuffled GroupKFold with game_id groups"
            ),
            "secondary_validation": "last 20% of ordered game ids held out",
        },
        "targets": {},
    }
    for target_name in [
        "is_transition",
        "is_transition_dunk_or_layup",
    ]:
        LOGGER.info("Training %s", target_name)
        results["targets"][target_name] = run_target(
            joined,
            features,
            target_name,
            folds=args.folds,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    results["forward_2025_26_descriptor_holdout"] = (
        evaluate_forward_descriptor_season(
            raw_paths=args.forward_raw_paths,
            output_dir=args.output_dir,
            training_feature_columns=list(features.columns),
            training_games=joined["game_id"].astype(str).unique(),
        )
    )
    write_json(metrics_path, results)
    LOGGER.info("Wrote %s", metrics_path)


if __name__ == "__main__":
    main()
