#!/usr/bin/env python3
"""
Train same-row ShotQuality challengers for midrange shots and three-pointers.

The feature-only model excludes every ShotQuality EV/shot-taking output. The
anchored model adds only `ev.initial_ev.value / shot_value`, allowing the
tracking features to learn an out-of-game residual correction to initial_ev.
Every reported prediction is out of fold with game-level group isolation.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from shotquality_modeling import (
        normalize_game_id,
        paired_cluster_bootstrap,
        parse_descriptor_tags,
        probability_metrics,
        write_json,
    )
except ImportError:  # pragma: no cover - package import in tests
    from nba_pipeline.scripts.shotquality_modeling import (
        normalize_game_id,
        paired_cluster_bootstrap,
        parse_descriptor_tags,
        probability_metrics,
        write_json,
    )


LOGGER = logging.getLogger("sq_shot_value_challenger")

DEFAULT_SQ_ROOT = Path(
    "/Users/russ/Documents/Codex/basketball/nba/research/shotquality/nba_games/2025"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "sq_challenger"

IDENTIFIER_COLUMNS = [
    "game_id",
    "GAME_EVENT_ID",
    "player_id",
    "personId",
]
TARGET_COLUMNS = [
    "made",
    "action_type",
    "SHOT_ZONE_BASIC",
    "play_descriptors",
    "ev.initial_ev.value",
]
BASE_NUMERIC_FEATURES = [
    "period",
    "minutes",
    "seconds",
    "shot_x",
    "shot_y",
    "contest_score",
    "spacing",
    "shooting_bench_direction",
    "feature_store.Shot Distance (ft.)",
    "feature_store.Player Three Prior",
    "feature_store.Player Jumpshot Prior",
    "feature_store.Player Midrange Prior",
    "feature_store.Three Prior (Men's)",
    "feature_store.Midrange Prior (Men's)",
    "feature_store.Closest Defender Behind",
    "feature_store.Closest Defender Inside",
    "feature_store.Closest Defender Blocking",
    "feature_store.Second-Closest Defender Behind",
    "feature_store.Second-Closest Defender Inside",
    "feature_store.Second-Closest Defender Blocking",
    "feature_store.Closest Defender Three Prior",
    "feature_store.Closest Defender Jumpshot Prior",
    "feature_store.Closest Defender Midrange Prior",
    "feature_store.Harmonic Mean Defender Distance",
    "feature_store.Average Defender Distance (Feet)",
    "feature_store.Angle with the Closest Defender",
    "feature_store.Angle with the Second-Closest Defender",
    "feature_store.Angle with the Third-Closest Defender",
    "feature_store.Distance to the Closest Defender (Feet)",
    "feature_store.Height of the Closest Defender (Inches)",
    "feature_store.Weight of the Closest Defender (Pounds)",
    "feature_store.Number of Defenders Closer to the Basket",
    "feature_store.Cosine of the Angle with the Closest Defender",
    "feature_store.Distance to the Second-Closest Defender (Feet)",
    "feature_store.Height of the Second-Closest Defender (Inches)",
    "feature_store.Weight of the Second-Closest Defender (Pounds)",
    "feature_store.Cosine of the Angle with the Second-Closest Defender",
    "feature_store.Height of the Closest Defender's Bounding Box (Pixels)",
    "feature_store.Height of the Second-Closest Defender's Bounding Box (Pixels)",
    "feature_store.Height of the Third-Closest Defender's Bounding Box (Pixels)",
    "feature_store.Difference in Bounding Box Height with the Closest Defender (Pixels)",
    "feature_store.Difference in Bounding Box Height with the Second-Closest Defender (Pixels)",
    "feature_store.Shooter in Restricted Area",
    "feature_store.Shooter in the Paint",
    "feature_store.Defenders in the Restricted Area",
    "feature_store.Defenders in Paint",
    "feature_store.Number of Teammates in the Restricted Area",
    "feature_store.Number of Teammates in the Paint",
    "feature_store.Shooter Velocity Angle",
    "feature_store.Shooter Velocity (ft/s)",
    "feature_store.Shooter Velocity to the Basket (ft/s)",
    "feature_store.Shooter Velocity along the X-Axis (ft/s)",
    "feature_store.Shooter Velocity along the Y-Axis (ft/s)",
    "feature_store.Shooter Acceleration Angle",
    "feature_store.Shooter Acceleration (ft/s^2)",
    "feature_store.Shooter Acceleration to the Basket (ft/s^2)",
    "feature_store.Shooter Acceleration along the X-Axis (ft/s)",
    "feature_store.Shooter Acceleration along the Y-Axis (ft/s)",
    "feature_store.Closest Defender Velocity Angle",
    "feature_store.Closest Defender Velocity (ft/s)",
    "feature_store.Closest Defender Velocity to the Basket (ft/s)",
    "feature_store.Closest Defender Velocity to the Shooter (ft/s)",
    "feature_store.Closest Defender Velocity along the X-Axis (ft/s)",
    "feature_store.Closest Defender Velocity along the Y-Axis (ft/s)",
    "feature_store.Closest Defender Acceleration Angle",
    "feature_store.Closest Defender Acceleration (ft/s^2)",
    "feature_store.Closest Defender Acceleration to the Basket (ft/s^2)",
    "feature_store.Closest Defender Acceleration to the Shooter (ft/s^2)",
    "feature_store.Closest Defender Acceleration along the X-Axis (ft/s^2)",
    "feature_store.Closest Defender Acceleration along the Y-Axis (ft/s^2)",
    "feature_store.Second-Closest Defender Velocity Angle",
    "feature_store.Second-Closest Defender Velocity (ft/s)",
    "feature_store.Second-Closest Defender Velocity to the Basket (ft/s)",
    "feature_store.Second-Closest Defender Velocity to the Shooter (ft/s)",
    "feature_store.Second-Closest Defender Velocity along the X-Axis (ft/s)",
    "feature_store.Second-Closest Defender Velocity along the Y-Axis (ft/s)",
    "feature_store.Second-Closest Defender Acceleration Angle",
    "feature_store.Second-Closest Defender Acceleration (ft/s^2)",
    "feature_store.Second-Closest Defender Acceleration to the Basket (ft/s^2)",
    "feature_store.Second-Closest Defender Acceleration to the Shooter (ft/s^2)",
    "feature_store.Second-Closest Defender Acceleration along the X-Axis (ft/s^2)",
    "feature_store.Second-Closest Defender Acceleration along the Y-Axis (ft/s^2)",
]
DESCRIPTOR_TAGS = [
    "alley_oop",
    "catch_and_shoot",
    "dunk",
    "floater",
    "hook",
    "in_paint",
    "jumpshot",
    "lay_up",
    "off_cut",
    "off_drive",
    "pick_and_roll",
    "post_up",
    "pull_up",
    "reverse",
    "step_back",
    "tip_in",
    "transition",
    "turnaround",
]
ZONE_CATEGORIES = [
    "Above the Break 3",
    "Left Corner 3",
    "Right Corner 3",
    "Mid-Range",
]
FORBIDDEN_TOKENS = (
    "ev.",
    "shot_taking",
    "made",
    "shot_made",
    "points",
    "assist",
    "blockplayer",
    "rebound",
)
LOGIT_C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1]


def required_source_columns() -> list[str]:
    return list(
        dict.fromkeys(
            IDENTIFIER_COLUMNS
            + TARGET_COLUMNS
            + BASE_NUMERIC_FEATURES
        )
    )


def load_sq_games(root: Path, *, limit_games: int = 0) -> pd.DataFrame:
    files = sorted(root.glob("*.csv"))
    if limit_games:
        files = files[:limit_games]
    if not files:
        raise FileNotFoundError(f"No ShotQuality game CSVs found under {root}")

    wanted = required_source_columns()
    frames: list[pd.DataFrame] = []
    for file_path in files:
        header = set(pd.read_csv(file_path, nrows=0).columns)
        usecols = [column for column in wanted if column in header]
        frame = pd.read_csv(file_path, usecols=usecols, low_memory=False)
        for column in wanted:
            if column not in frame.columns:
                frame[column] = np.nan
        frame["_source_file"] = file_path.name
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined["game_id"] = normalize_game_id(combined["game_id"])
    return combined


def segment_rows(frame: pd.DataFrame, segment: str) -> pd.Series:
    action = frame["action_type"].astype("string").str.lower()
    if segment == "midrange":
        return (action == "2pt") & frame["SHOT_ZONE_BASIC"].eq("Mid-Range")
    if segment == "three":
        return action == "3pt"
    raise ValueError(f"Unknown segment: {segment}")


def build_feature_matrix(
    frame: pd.DataFrame,
    *,
    include_initial_ev: bool,
    shot_value: int,
) -> pd.DataFrame:
    feature_values: dict[str, pd.Series] = {}
    for column in BASE_NUMERIC_FEATURES:
        feature_values[column] = pd.to_numeric(frame.get(column), errors="coerce")

    parsed_tags = frame["play_descriptors"].map(parse_descriptor_tags)
    for tag in DESCRIPTOR_TAGS:
        feature_values[f"descriptor::{tag}"] = parsed_tags.map(
            lambda tags, tag=tag: int(tag in tags)
        )

    zone = frame["SHOT_ZONE_BASIC"].fillna("UNKNOWN").astype(str)
    for category in ZONE_CATEGORIES:
        feature_values[f"zone::{category}"] = zone.eq(category).astype(int)
    feature_values["zone::UNKNOWN"] = zone.eq("UNKNOWN").astype(int)

    if include_initial_ev:
        feature_values["baseline_initial_make_prob"] = (
            pd.to_numeric(frame["ev.initial_ev.value"], errors="coerce") / shot_value
        ).clip(0.001, 0.999)

    features = pd.DataFrame(feature_values, index=frame.index)
    all_missing = features.columns[features.notna().sum(axis=0) == 0]
    features = features.drop(columns=list(all_missing))
    validate_feature_names(features.columns, include_initial_ev=include_initial_ev)
    return features.astype(float)


def validate_feature_names(
    feature_names: Iterable[str],
    *,
    include_initial_ev: bool,
) -> None:
    allowed_anchor = {"baseline_initial_make_prob"} if include_initial_ev else set()
    forbidden = [
        name
        for name in feature_names
        if name not in allowed_anchor
        and any(token in name.lower() for token in FORBIDDEN_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"Outcome/EV leakage columns entered the model: {forbidden}")


def new_model(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.06,
        max_iter=180,
        max_leaf_nodes=31,
        min_samples_leaf=60,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=seed,
    )


def logit_challenger_columns(
    feature_only: pd.DataFrame,
    *,
    segment: str,
) -> list[str]:
    """Small, auditable feature set for the regularized residual challenger."""
    common = [
        "baseline_initial_logit",
        "contest_score",
        "spacing",
        "shot_x",
        "shot_y",
        "feature_store.Shot Distance (ft.)",
        "feature_store.Distance to the Closest Defender (Feet)",
        "feature_store.Distance to the Second-Closest Defender (Feet)",
        "feature_store.Height of the Closest Defender (Inches)",
        "feature_store.Weight of the Closest Defender (Pounds)",
        "feature_store.Number of Defenders Closer to the Basket",
        "feature_store.Cosine of the Angle with the Closest Defender",
        "feature_store.Cosine of the Angle with the Second-Closest Defender",
        "feature_store.Closest Defender Behind",
        "feature_store.Closest Defender Inside",
        "feature_store.Closest Defender Blocking",
    ]
    if segment == "midrange":
        common.extend(
            [
                "feature_store.Player Midrange Prior",
                "feature_store.Player Jumpshot Prior",
                "feature_store.Midrange Prior (Men's)",
                "feature_store.Closest Defender Midrange Prior",
                "feature_store.Closest Defender Jumpshot Prior",
            ]
        )
    else:
        common.extend(
            [
                "feature_store.Player Three Prior",
                "feature_store.Player Jumpshot Prior",
                "feature_store.Three Prior (Men's)",
                "feature_store.Closest Defender Three Prior",
                "feature_store.Closest Defender Jumpshot Prior",
            ]
        )
    common.extend(
        column
        for column in feature_only.columns
        if column.startswith("descriptor::") or column.startswith("zone::")
    )
    return [column for column in common if column in feature_only.columns or column == "baseline_initial_logit"]


def build_regularized_logit_features(
    feature_only: pd.DataFrame,
    baseline_probability: pd.Series,
    *,
    segment: str,
) -> pd.DataFrame:
    values = feature_only.copy()
    baseline = np.clip(np.asarray(baseline_probability, dtype=float), 0.001, 0.999)
    values.insert(0, "baseline_initial_logit", np.log(baseline / (1.0 - baseline)))
    return values[logit_challenger_columns(values, segment=segment)].copy()


def new_logit_model(c_value: float):
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(C=c_value, max_iter=1000, solver="lbfgs"),
    )


def select_logit_c(
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    *,
    seed: int,
    inner_folds: int = 3,
) -> tuple[float, dict[str, float]]:
    """Choose regularization entirely within the supplied training games."""
    splitter = GroupKFold(
        n_splits=inner_folds, shuffle=True, random_state=seed
    )
    scores: dict[str, float] = {}
    for c_value in LOGIT_C_GRID:
        predictions = np.full(len(features), np.nan)
        for train_idx, test_idx in splitter.split(features, target, groups):
            model = new_logit_model(c_value)
            model.fit(features.iloc[train_idx], target.iloc[train_idx])
            predictions[test_idx] = model.predict_proba(features.iloc[test_idx])[:, 1]
        scores[str(c_value)] = float(log_loss(target, predictions, labels=[0, 1]))
    selected = min(LOGIT_C_GRID, key=lambda value: (scores[str(value)], value))
    return float(selected), scores


def nested_grouped_logit_oof(
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    fold_ids: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Nested game-group CV: C selection never sees the outer test games."""
    predictions = np.full(len(features), np.nan)
    selections: list[dict[str, object]] = []
    for fold_id in sorted(np.unique(fold_ids)):
        train_mask = fold_ids != fold_id
        test_mask = ~train_mask
        selected_c, inner_scores = select_logit_c(
            features.loc[train_mask].reset_index(drop=True),
            target.loc[train_mask].reset_index(drop=True),
            groups.loc[train_mask].reset_index(drop=True),
            seed=seed + 100 + int(fold_id),
        )
        model = new_logit_model(selected_c)
        model.fit(features.loc[train_mask], target.loc[train_mask])
        predictions[test_mask] = model.predict_proba(features.loc[test_mask])[:, 1]
        selections.append(
            {
                "outer_fold": int(fold_id),
                "selected_c": selected_c,
                "inner_log_loss_by_c": inner_scores,
            }
        )
    if np.isnan(predictions).any():
        raise RuntimeError("Nested regularized-logit OOF coverage is incomplete")
    return predictions, selections


def grouped_oof_predictions(
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    *,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.full(len(features), np.nan, dtype=float)
    fold_ids = np.full(len(features), -1, dtype=int)
    splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold_id, (train_idx, test_idx) in enumerate(
        splitter.split(features, target, groups)
    ):
        model = new_model(seed + fold_id)
        model.fit(features.iloc[train_idx], target.iloc[train_idx])
        predictions[test_idx] = model.predict_proba(features.iloc[test_idx])[:, 1]
        fold_ids[test_idx] = fold_id
    if np.isnan(predictions).any() or (fold_ids < 0).any():
        raise RuntimeError("OOF prediction coverage is incomplete")
    return predictions, fold_ids


def chronological_holdout(
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    *,
    seed: int,
    fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered_games = np.array(sorted(groups.astype(str).unique()))
    cut = max(1, int(np.floor(len(ordered_games) * (1.0 - fraction))))
    train_games = set(ordered_games[:cut])
    train_mask = groups.astype(str).isin(train_games).to_numpy()
    test_mask = ~train_mask
    model = new_model(seed)
    model.fit(features.loc[train_mask], target.loc[train_mask])
    predictions = model.predict_proba(features.loc[test_mask])[:, 1]
    return test_mask, predictions, ordered_games[cut:]


def run_segment(
    source: pd.DataFrame,
    segment: str,
    *,
    folds: int,
    seed: int,
    output_dir: Path,
    bootstrap_iterations: int,
    reuse_hgb_oof: bool = False,
    previous_result: dict[str, object] | None = None,
    refresh_hgb_importance: bool = False,
) -> dict[str, object]:
    shot_value = 2 if segment == "midrange" else 3
    mask = segment_rows(source, segment)
    data = source.loc[mask].copy()
    data["made"] = data["made"].astype("boolean")
    data["ev.initial_ev.value"] = pd.to_numeric(
        data["ev.initial_ev.value"], errors="coerce"
    )
    data = data[
        data["made"].notna()
        & data["ev.initial_ev.value"].notna()
        & data["game_id"].notna()
    ].reset_index(drop=True)
    if data.empty:
        raise ValueError(f"No fair-comparison rows available for {segment}")

    target = data["made"].astype(int)
    groups = data["game_id"].astype(str)
    baseline = (data["ev.initial_ev.value"] / shot_value).clip(0.001, 0.999)
    feature_only = build_feature_matrix(
        data, include_initial_ev=False, shot_value=shot_value
    )
    anchored = build_feature_matrix(
        data, include_initial_ev=True, shot_value=shot_value
    )

    predictions_path = output_dir / f"sq_{segment}_oof_predictions.parquet"
    if reuse_hgb_oof and predictions_path.exists():
        existing = pd.read_parquet(predictions_path)
        if len(existing) != len(data):
            raise ValueError(
                f"Cannot reuse {predictions_path}: {len(existing)} rows != {len(data)}"
            )
        if not existing["game_id"].astype(str).reset_index(drop=True).equals(
            groups.reset_index(drop=True)
        ):
            raise ValueError(f"Cannot reuse {predictions_path}: game order changed")
        existing_event = pd.to_numeric(
            existing["game_event_id"], errors="coerce"
        ).reset_index(drop=True)
        current_event = pd.to_numeric(
            data["GAME_EVENT_ID"], errors="coerce"
        ).reset_index(drop=True)
        if not existing_event.equals(current_event):
            raise ValueError(f"Cannot reuse {predictions_path}: event order changed")
        if not existing["made"].astype(int).reset_index(drop=True).equals(
            target.reset_index(drop=True)
        ):
            raise ValueError(f"Cannot reuse {predictions_path}: targets changed")
        if not np.allclose(
            existing["initial_ev"].to_numpy(dtype=float),
            data["ev.initial_ev.value"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"Cannot reuse {predictions_path}: initial_ev changed")
        feature_probability = existing[
            "feature_only_oof_probability"
        ].to_numpy()
        anchored_probability = existing["anchored_oof_probability"].to_numpy()
        fold_ids = existing["outer_fold"].to_numpy(dtype=int)
    else:
        feature_probability, fold_ids = grouped_oof_predictions(
            feature_only, target, groups, folds=folds, seed=seed
        )
        anchored_probability, anchored_fold_ids = grouped_oof_predictions(
            anchored, target, groups, folds=folds, seed=seed
        )
        if not np.array_equal(fold_ids, anchored_fold_ids):
            raise RuntimeError("Model families used different outer folds")

    logit_features = build_regularized_logit_features(
        feature_only, baseline, segment=segment
    )
    logit_probability, logit_selections = nested_grouped_logit_oof(
        logit_features,
        target,
        groups,
        fold_ids,
        seed=seed,
    )

    models = {
        "initial_ev": np.asarray(baseline),
        "feature_only": feature_probability,
        "anchored_residual": anchored_probability,
        "regularized_logit_challenger": logit_probability,
    }
    metrics = {
        name: probability_metrics(target, probability, shot_value=shot_value)
        for name, probability in models.items()
    }
    comparisons = {
        name: paired_cluster_bootstrap(
            target,
            probability,
            baseline,
            groups,
            iterations=bootstrap_iterations,
            seed=seed,
        )
        for name, probability in models.items()
        if name != "initial_ev"
    }

    temporal: dict[str, object] = {}
    if reuse_hgb_oof and previous_result is not None:
        temporal.update(previous_result["chronological_last_20pct_holdout"])
    else:
        for name, feature_frame in [
            ("feature_only", feature_only),
            ("anchored_residual", anchored),
        ]:
            test_mask, temporal_probability, test_games = chronological_holdout(
                feature_frame, target, groups, seed=seed
            )
            temporal[name] = {
                "metrics": probability_metrics(
                    target.loc[test_mask],
                    temporal_probability,
                    shot_value=shot_value,
                ),
                "test_game_count": int(len(test_games)),
                "first_test_game": str(test_games[0]),
                "last_test_game": str(test_games[-1]),
            }
        temporal_test_mask, _, temporal_games = chronological_holdout(
            anchored, target, groups, seed=seed
        )
        temporal["initial_ev"] = {
            "metrics": probability_metrics(
                target.loc[temporal_test_mask],
                baseline.loc[temporal_test_mask],
                shot_value=shot_value,
            ),
            "test_game_count": int(len(temporal_games)),
            "first_test_game": str(temporal_games[0]),
            "last_test_game": str(temporal_games[-1]),
        }

    ordered_games = np.array(sorted(groups.unique()))
    temporal_cut = max(1, int(np.floor(len(ordered_games) * 0.8)))
    temporal_train_games = set(ordered_games[:temporal_cut])
    temporal_train_mask = groups.isin(temporal_train_games).to_numpy()
    temporal_test_mask = ~temporal_train_mask
    temporal_c, temporal_c_scores = select_logit_c(
        logit_features.loc[temporal_train_mask].reset_index(drop=True),
        target.loc[temporal_train_mask].reset_index(drop=True),
        groups.loc[temporal_train_mask].reset_index(drop=True),
        seed=seed + 500,
    )
    temporal_logit_model = new_logit_model(temporal_c)
    temporal_logit_model.fit(
        logit_features.loc[temporal_train_mask],
        target.loc[temporal_train_mask],
    )
    temporal_logit_probability = temporal_logit_model.predict_proba(
        logit_features.loc[temporal_test_mask]
    )[:, 1]
    temporal["regularized_logit_challenger"] = {
        "metrics": probability_metrics(
            target.loc[temporal_test_mask],
            temporal_logit_probability,
            shot_value=shot_value,
        ),
        "selected_c": temporal_c,
        "inner_log_loss_by_c": temporal_c_scores,
        "test_game_count": int(len(ordered_games[temporal_cut:])),
        "first_test_game": str(ordered_games[temporal_cut]),
        "last_test_game": str(ordered_games[-1]),
    }

    model_path = output_dir / f"sq_{segment}_anchored_model.joblib"
    if not reuse_hgb_oof or not model_path.exists():
        final_model = new_model(seed)
        final_model.fit(anchored, target)
        joblib.dump(
            {
                "model": final_model,
                "feature_columns": list(anchored.columns),
                "segment": segment,
                "shot_value": shot_value,
                "training_rows": int(len(data)),
                "training_games": int(groups.nunique()),
                "leakage_contract": (
                    "Only initial_ev make probability plus explicitly allowlisted "
                    "pre-shot tracking/context features; no dynamic/live EV, "
                    "shot-taking output, current-shot outcome, assist, rebound, or block fields."
                ),
            },
            model_path,
        )

    if not reuse_hgb_oof or refresh_hgb_importance:
        importance_model = new_model(seed + 1000)
        importance_model.fit(
            anchored.loc[temporal_train_mask],
            target.loc[temporal_train_mask],
        )
        importance_indices = np.flatnonzero(temporal_test_mask)
        if len(importance_indices) > 20000:
            rng = np.random.default_rng(seed)
            importance_indices = np.sort(
                rng.choice(importance_indices, size=20000, replace=False)
            )
        importance = permutation_importance(
            importance_model,
            anchored.iloc[importance_indices],
            target.iloc[importance_indices],
            scoring="neg_log_loss",
            n_repeats=3,
            random_state=seed,
            n_jobs=1,
        )
        importance_frame = pd.DataFrame(
            {
                "feature": anchored.columns,
                "permutation_log_loss_increase_mean": importance.importances_mean,
                "permutation_log_loss_increase_std": importance.importances_std,
                "evaluation_split": "ordered_last_20pct_games",
            }
        ).sort_values("permutation_log_loss_increase_mean", ascending=False)
        importance_frame.to_csv(
            output_dir / f"sq_{segment}_feature_importance.csv", index=False
        )

    final_logit_c, final_logit_scores = select_logit_c(
        logit_features,
        target,
        groups,
        seed=seed + 900,
    )
    final_logit_model = new_logit_model(final_logit_c)
    final_logit_model.fit(logit_features, target)
    logit_model_path = (
        output_dir / f"sq_{segment}_regularized_logit_model.joblib"
    )
    joblib.dump(
        {
            "model": final_logit_model,
            "feature_columns": list(logit_features.columns),
            "segment": segment,
            "shot_value": shot_value,
            "selected_c": final_logit_c,
            "inner_log_loss_by_c": final_logit_scores,
            "training_rows": int(len(data)),
            "training_games": int(groups.nunique()),
            "leakage_contract": (
                "Initial_ev enters only as a logit anchor. All remaining fields "
                "are explicitly allowlisted pre-shot tracking/context features. "
                "Regularization is selected with game-group inner CV."
            ),
        },
        logit_model_path,
    )

    predictions = pd.DataFrame(
        {
            "game_id": groups,
            "game_event_id": data["GAME_EVENT_ID"],
            "segment": segment,
            "made": target,
            "initial_ev": data["ev.initial_ev.value"],
            "initial_ev_make_probability": baseline,
            "feature_only_oof_probability": feature_probability,
            "anchored_oof_probability": anchored_probability,
            "regularized_logit_oof_probability": logit_probability,
            "outer_fold": fold_ids,
        }
    )
    predictions.to_parquet(
        output_dir / f"sq_{segment}_oof_predictions.parquet", index=False
    )

    return {
        "segment": segment,
        "shot_value": shot_value,
        "rows": int(len(data)),
        "games": int(groups.nunique()),
        "outer_folds": int(folds),
        "metrics": metrics,
        "paired_game_cluster_bootstrap": comparisons,
        "chronological_last_20pct_holdout": temporal,
        "feature_only_columns": list(feature_only.columns),
        "anchored_columns": list(anchored.columns),
        "regularized_logit_columns": list(logit_features.columns),
        "regularized_logit_outer_selections": logit_selections,
        "regularized_logit_final_c": final_logit_c,
        "regularized_logit_final_inner_scores": final_logit_scores,
        "model_path": str(model_path),
        "regularized_logit_model_path": str(logit_model_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sq-root", type=Path, default=DEFAULT_SQ_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--segments",
        nargs="+",
        choices=["midrange", "three"],
        default=["midrange", "three"],
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--limit-games", type=int, default=0)
    parser.add_argument(
        "--reuse-hgb-oof",
        action="store_true",
        help=(
            "Reuse existing HGB OOF predictions after exact row/order checks; "
            "useful when adding the nested regularized-logit benchmark."
        ),
    )
    parser.add_argument(
        "--refresh-hgb-importance",
        action="store_true",
        help=(
            "Refit HGB on the first 80% of ordered games and rewrite permutation "
            "importance from the untouched last 20%."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = load_sq_games(args.sq_root, limit_games=args.limit_games)
    inventory = {
        "sq_root": str(args.sq_root),
        "source_files": int(source["_source_file"].nunique()),
        "source_rows": int(len(source)),
        "source_games": int(source["game_id"].nunique()),
        "source_columns_requested": required_source_columns(),
    }
    results = {
        "inventory": inventory,
        "model_spec": {
            "estimator": "HistGradientBoostingClassifier",
            "loss": "log_loss",
            "learning_rate": 0.06,
            "max_iter": 180,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 60,
            "l2_regularization": 2.0,
            "outer_validation": (
                f"{args.folds}-fold shuffled GroupKFold with game_id groups"
            ),
            "secondary_validation": "last 20% of ordered game ids held out",
        },
        "segments": {},
    }
    previous_results = None
    previous_metrics_path = (
        args.output_dir / "sq_shot_value_challenger_metrics.json"
    )
    if args.reuse_hgb_oof and previous_metrics_path.exists():
        previous_results = json.loads(previous_metrics_path.read_text())
    for segment in args.segments:
        LOGGER.info("Training %s challenger", segment)
        results["segments"][segment] = run_segment(
            source,
            segment,
            folds=args.folds,
            seed=args.seed,
            output_dir=args.output_dir,
            bootstrap_iterations=args.bootstrap_iterations,
            reuse_hgb_oof=args.reuse_hgb_oof,
            previous_result=(
                previous_results["segments"].get(segment)
                if previous_results is not None
                else None
            ),
            refresh_hgb_importance=args.refresh_hgb_importance,
        )
    write_json(args.output_dir / "sq_shot_value_challenger_metrics.json", results)
    LOGGER.info(
        "Wrote %s",
        args.output_dir / "sq_shot_value_challenger_metrics.json",
    )


if __name__ == "__main__":
    main()
