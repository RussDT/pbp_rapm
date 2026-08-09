#!/usr/bin/env python3
"""Train a modern ShotQuality teacher and a historical-PBP student.

The teacher is evaluated against made/missed outcomes with ShotQuality
``initial_ev`` as the held-out baseline.  It may use modern SQ descriptor
features.  The student is trained to reproduce the teacher using only fields
available across the historical raw-PBP archive, allowing the model to be
scored before the licensed SQ era.

Models and calibration are separate for midrange attempts and threes.  The
test set is the chronologically latest group of games; no shot from a held-out
game is used to fit either model.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from nba_pipeline.scripts.build_pbp_shot_quality import (
    add_shot_features,
    discover_raw_files,
    file_metadata,
    load_shots_from_file,
    normalize_game_id,
    parse_year_token,
)
from nba_pipeline.scripts.build_shot_clock_estimates import estimate_shot_clock


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PIPELINE_ROOT / "raw_data"
OUTPUT_DIR = PIPELINE_ROOT / "results" / "historical_shot_quality"
DEFAULT_RICH_OOF = [
    PIPELINE_ROOT / "results" / "sq_challenger" / "sq_midrange_oof_predictions.parquet",
    PIPELINE_ROOT / "results" / "sq_challenger" / "sq_three_oof_predictions.parquet",
]

HISTORICAL_CATEGORICAL = [
    "shot_zone",
    "distance_bin",
    "shot_family",
    "modifier_bundle",
    "event_action_type_str",
    "season_type",
    "distance_source",
    "offense_side",
    "shot_clock_reset_reason",
    "shot_clock_confidence",
]
HISTORICAL_NUMERIC = [
    "shot_distance_ft",
    "shot_clock_est",
    "shot_clock_fraction",
    "elapsed_since_reset",
    "effective_reset_len",
    "period",
    "seconds_remaining_quarter",
    "shooter_pregame_make_rate",
    "shooter_pregame_log_attempts",
]
TEACHER_BASE_NUMERIC = ["initial_make_prob"]
MODEL_BUCKETS = ("midrange", "three")


def _safe_probability(values: Iterable[float]) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 1e-5, 1 - 1e-5)


def _logit(values: Iterable[float]) -> np.ndarray:
    p = _safe_probability(values)
    return np.log(p / (1.0 - p)).reshape(-1, 1)


def fit_probability_calibrator(raw_probability: Iterable[float], y: Iterable[int]) -> dict[str, float]:
    """Fit a one-dimensional Platt calibration map."""
    y_array = np.asarray(y, dtype=int)
    if len(np.unique(y_array)) < 2:
        return {"slope": 1.0, "intercept": 0.0}
    model = LogisticRegression(C=1e6, solver="lbfgs")
    model.fit(_logit(raw_probability), y_array)
    return {
        "slope": float(model.coef_[0, 0]),
        "intercept": float(model.intercept_[0]),
    }


def apply_probability_calibrator(
    raw_probability: Iterable[float],
    calibrator: dict[str, float],
) -> np.ndarray:
    logits = _logit(raw_probability).ravel()
    calibrated_logit = calibrator["slope"] * logits + calibrator["intercept"]
    return _safe_probability(1.0 / (1.0 + np.exp(-calibrated_logit)))


def _build_preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                    max_categories=254,
                ),
            ),
        ]
    )
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    return ColumnTransformer(
        [
            ("categorical", categorical_pipe, categorical),
            ("numeric", numeric_pipe, numeric),
        ],
        verbose_feature_names_out=False,
    )


def build_classifier(categorical: list[str], numeric: list[str]) -> Pipeline:
    preprocessor = _build_preprocessor(categorical, numeric)
    categorical_indices = list(range(len(categorical)))
    return Pipeline(
        [
            ("features", preprocessor),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=160,
                    max_leaf_nodes=31,
                    min_samples_leaf=80,
                    l2_regularization=2.0,
                    categorical_features=categorical_indices,
                    random_state=17,
                ),
            ),
        ]
    )


def build_regressor(categorical: list[str], numeric: list[str]) -> Pipeline:
    preprocessor = _build_preprocessor(categorical, numeric)
    categorical_indices = list(range(len(categorical)))
    return Pipeline(
        [
            ("features", preprocessor),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=0.05,
                    max_iter=180,
                    max_leaf_nodes=31,
                    min_samples_leaf=80,
                    l2_regularization=3.0,
                    categorical_features=categorical_indices,
                    random_state=23,
                ),
            ),
        ]
    )


def _normalize_event_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _add_model_bucket(shots: pd.DataFrame) -> pd.DataFrame:
    out = shots.copy()
    is_three = out["is_3pt"].fillna(False)
    is_midrange = ~is_three & out["shot_distance_ft"].ge(5.0)
    out["model_bucket"] = np.select(
        [is_three, is_midrange],
        ["three", "midrange"],
        default="excluded",
    )
    return out


def load_feature_rows(path: Path, *, require_sq: bool) -> pd.DataFrame:
    """Load one raw file into leakage-safe shot features and optional SQ labels."""
    end_year, _ = file_metadata(path)
    raw = pd.read_parquet(path)
    shots = add_shot_features(load_shots_from_file(path))
    shots = _add_model_bucket(shots)
    shots["event_action_type_str"] = shots["event_action_type"].astype(str).replace("<NA>", "missing")

    clock = estimate_shot_clock(raw, season_end_year=end_year)[
        [
            "game_id",
            "event_num",
            "shot_clock_est",
            "elapsed_since_reset",
            "reset_len",
            "reset_reason",
            "confidence",
            "needs_review",
        ]
    ].copy()
    clock["game_id"] = normalize_game_id(clock["game_id"])
    clock["event_num"] = _normalize_event_num(clock["event_num"])
    clock = clock.drop_duplicates(["game_id", "event_num"], keep="first")
    shots = shots.merge(
        clock,
        on=["game_id", "event_num"],
        how="left",
        validate="one_to_one",
    )
    shots["shot_clock_est"] = pd.to_numeric(shots["shot_clock_est"], errors="coerce")
    shots["elapsed_since_reset"] = pd.to_numeric(shots["elapsed_since_reset"], errors="coerce")
    shots["effective_reset_len"] = pd.to_numeric(shots["reset_len"], errors="coerce")
    shots["shot_clock_fraction"] = (
        shots["shot_clock_est"] / shots["effective_reset_len"].replace(0, np.nan)
    ).clip(0, 1)
    shots["shot_clock_reset_reason"] = shots["reset_reason"].fillna("missing").astype(str)
    shots["shot_clock_confidence"] = shots["confidence"].fillna("missing").astype(str)

    sq_columns = [
        name
        for name in raw.columns
        if name.startswith("sq_desc_")
        or name in {"initial_ev", "dynamic_ev", "is_transition", "sq_descriptor_bundle"}
    ]
    if require_sq and "initial_ev" not in sq_columns:
        raise ValueError(f"{path.name} has no initial_ev labels.")
    if sq_columns:
        sq = raw[["game_id", "event_num", *sq_columns]].copy()
        sq["game_id"] = normalize_game_id(sq["game_id"])
        sq["event_num"] = _normalize_event_num(sq["event_num"])
        sq = sq.drop_duplicates(["game_id", "event_num"], keep="first")
        shots = shots.merge(sq, on=["game_id", "event_num"], how="left", validate="one_to_one")

    if "initial_ev" not in shots:
        shots["initial_ev"] = np.nan
    if "dynamic_ev" not in shots:
        shots["dynamic_ev"] = np.nan
    shots["initial_ev"] = pd.to_numeric(shots["initial_ev"], errors="coerce")
    shots["dynamic_ev"] = pd.to_numeric(shots["dynamic_ev"], errors="coerce")
    shots["initial_make_prob"] = (shots["initial_ev"] / shots["shot_value"]).clip(0.001, 0.999)
    shots["dynamic_make_prob"] = (
        shots["dynamic_ev"].fillna(shots["initial_ev"]) / shots["shot_value"]
    ).clip(0.001, 0.999)
    return shots


def add_pregame_shooter_priors(rows: pd.DataFrame, *, shrink: float = 200.0) -> pd.DataFrame:
    """Add current-game-excluded shooter/bucket make priors.

    The prior for a game is built only from earlier ordered games.  It is
    player-informed and must not be labeled player-agnostic xFG.
    """
    out = rows.copy()
    if "is_make" not in out.columns:
        out["shooter_pregame_make_rate"] = out["model_bucket"].map(
            {"midrange": 0.41, "three": 0.36}
        ).fillna(0.40)
        out["shooter_pregame_log_attempts"] = 0.0
        return out
    game_stats = (
        out.groupby(
            ["shooter_id", "model_bucket", "season_end_year", "season_type", "game_id"],
            dropna=False,
            observed=True,
        )
        .agg(game_attempts=("is_make", "size"), game_makes=("is_make", "sum"))
        .reset_index()
        .sort_values(
            ["shooter_id", "model_bucket", "season_end_year", "season_type", "game_id"],
            kind="stable",
        )
    )
    group = game_stats.groupby(["shooter_id", "model_bucket"], dropna=False, observed=True)
    game_stats["prior_attempts"] = group["game_attempts"].cumsum() - game_stats["game_attempts"]
    game_stats["prior_makes"] = group["game_makes"].cumsum() - game_stats["game_makes"]
    game_stats["fixed_prior"] = game_stats["model_bucket"].map(
        {"midrange": 0.41, "three": 0.36}
    ).fillna(0.40)
    game_stats["shooter_pregame_make_rate"] = (
        game_stats["prior_makes"] + shrink * game_stats["fixed_prior"]
    ) / (game_stats["prior_attempts"] + shrink)
    game_stats["shooter_pregame_log_attempts"] = np.log1p(game_stats["prior_attempts"])
    keys = ["shooter_id", "model_bucket", "season_end_year", "season_type", "game_id"]
    return out.merge(
        game_stats[
            keys
            + [
                "shooter_pregame_make_rate",
                "shooter_pregame_log_attempts",
            ]
        ],
        on=keys,
        how="left",
        validate="many_to_one",
    )


def discover_modern_files(raw_dir: Path) -> list[Path]:
    files = []
    for path in discover_raw_files(raw_dir, None, {"rs", "ps"}):
        names = set(pq.read_schema(path).names)
        if "initial_ev" in names and any(name.startswith("sq_desc_") for name in names):
            files.append(path)
    return files


def load_rich_oof_teacher_rows(
    prediction_files: list[Path],
    *,
    raw_dir: Path,
) -> pd.DataFrame:
    """Join 2024-25 rich-feature OOF teacher predictions to historical features."""
    predictions = []
    for path in prediction_files:
        frame = pd.read_parquet(path)
        required = {
            "game_id",
            "game_event_id",
            "segment",
            "regularized_logit_oof_probability",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing rich OOF columns: {sorted(missing)}")
        frame = frame[list(required)].copy()
        frame["game_id"] = normalize_game_id(frame["game_id"])
        frame["event_num"] = _normalize_event_num(frame["game_event_id"])
        frame["external_teacher_probability"] = _safe_probability(
            frame["regularized_logit_oof_probability"]
        )
        predictions.append(
            frame[["game_id", "event_num", "segment", "external_teacher_probability"]]
        )
    teacher = pd.concat(predictions, ignore_index=True).drop_duplicates(
        ["game_id", "event_num"],
        keep="first",
    )
    raw_paths = [
        path
        for path in [raw_dir / "NBA25.parquet", raw_dir / "NBA25_PS.parquet"]
        if path.exists()
    ]
    if not raw_paths:
        raise FileNotFoundError("NBA25 RS/PS raw files are required for rich OOF transfer.")
    historical = pd.concat(
        [load_feature_rows(path, require_sq=False) for path in raw_paths],
        ignore_index=True,
    )
    joined = historical.merge(
        teacher,
        on=["game_id", "event_num"],
        how="inner",
        validate="one_to_one",
    )
    joined["model_bucket"] = joined["segment"].astype(str)
    return joined.drop(columns=["segment"])


def chronological_game_split(
    rows: pd.DataFrame,
    *,
    test_fraction: float = 0.20,
) -> tuple[pd.Series, pd.Series]:
    """Split on complete games, holding out the chronologically latest games."""
    game_order = (
        rows[["season_end_year", "season_type", "game_id"]]
        .drop_duplicates()
        .sort_values(["season_end_year", "season_type", "game_id"], kind="stable")
        .reset_index(drop=True)
    )
    if len(game_order) < 5:
        raise ValueError("At least five games are required for a temporal holdout.")
    cut = max(1, min(len(game_order) - 1, int(round(len(game_order) * (1.0 - test_fraction)))))
    train_games = set(game_order.iloc[:cut]["game_id"])
    test_games = set(game_order.iloc[cut:]["game_id"])
    return rows["game_id"].isin(train_games), rows["game_id"].isin(test_games)


def probability_metrics(y: pd.Series, probability: Iterable[float]) -> dict[str, float]:
    p = _safe_probability(probability)
    target = y.astype(int).to_numpy()
    return {
        "brier": float(brier_score_loss(target, p)),
        "log_loss": float(log_loss(target, p, labels=[0, 1])),
        "calibration_bias": float(p.mean() - target.mean()),
    }


def paired_game_loss_intervals(
    rows: pd.DataFrame,
    probability_column: str,
    *,
    bootstrap_repetitions: int = 2000,
) -> dict[str, float]:
    """Cluster-bootstrap paired loss improvements by game."""
    y = rows["is_make"].astype(int).to_numpy()
    baseline = _safe_probability(rows["initial_probability"])
    candidate = _safe_probability(rows[probability_column])
    brier_delta = (y - baseline) ** 2 - (y - candidate) ** 2
    baseline_log = -(y * np.log(baseline) + (1 - y) * np.log(1 - baseline))
    candidate_log = -(y * np.log(candidate) + (1 - y) * np.log(1 - candidate))
    work = pd.DataFrame(
        {
            "game_id": rows["game_id"].astype(str).to_numpy(),
            "brier_delta": brier_delta,
            "log_delta": baseline_log - candidate_log,
        }
    )
    grouped = work.groupby("game_id", observed=True).agg(
        brier_sum=("brier_delta", "sum"),
        log_sum=("log_delta", "sum"),
        shots=("brier_delta", "size"),
    )
    rng = np.random.default_rng(20260723)
    sample_idx = rng.integers(
        0,
        len(grouped),
        size=(bootstrap_repetitions, len(grouped)),
    )
    counts = grouped["shots"].to_numpy()
    denominators = counts[sample_idx].sum(axis=1)
    brier_samples = grouped["brier_sum"].to_numpy()[sample_idx].sum(axis=1) / denominators
    log_samples = grouped["log_sum"].to_numpy()[sample_idx].sum(axis=1) / denominators
    return {
        "brier_improvement_ci_low": float(np.quantile(brier_samples, 0.025)),
        "brier_improvement_ci_high": float(np.quantile(brier_samples, 0.975)),
        "log_loss_improvement_ci_low": float(np.quantile(log_samples, 0.025)),
        "log_loss_improvement_ci_high": float(np.quantile(log_samples, 0.975)),
    }


def calibration_table(rows: pd.DataFrame, probability_columns: list[str]) -> pd.DataFrame:
    pieces = []
    for bucket, bucket_df in rows.groupby("model_bucket", observed=True):
        for column in probability_columns:
            work = bucket_df[["is_make", column]].dropna().copy()
            if work.empty:
                continue
            work["decile"] = pd.qcut(
                work[column].rank(method="first"),
                q=min(10, len(work)),
                labels=False,
                duplicates="drop",
            )
            grouped = (
                work.groupby("decile", observed=True)
                .agg(
                    shots=("is_make", "size"),
                    predicted_make_rate=(column, "mean"),
                    actual_make_rate=("is_make", "mean"),
                )
                .reset_index()
            )
            grouped.insert(0, "prediction", column)
            grouped.insert(0, "model_bucket", bucket)
            pieces.append(grouped)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def fit_bucket_models(
    bucket_rows: pd.DataFrame,
    *,
    sq_descriptor_columns: list[str],
) -> tuple[dict[str, object], pd.DataFrame, list[dict[str, object]]]:
    train_mask, test_mask = chronological_game_split(bucket_rows)
    train = bucket_rows[train_mask].copy()
    test = bucket_rows[test_mask & bucket_rows["initial_make_prob"].notna()].copy()
    if test.empty:
        raise ValueError("Temporal test block has no initial_ev rows for a paired benchmark.")

    fit_mask, calibration_mask = chronological_game_split(train, test_fraction=0.20)
    teacher_numeric = [*HISTORICAL_NUMERIC, *TEACHER_BASE_NUMERIC, *sq_descriptor_columns]
    teacher_train = train[train["initial_make_prob"].notna()].copy()
    teacher_fit_mask, teacher_calibration_mask = chronological_game_split(
        teacher_train,
        test_fraction=0.20,
    )
    teacher_fit = build_classifier(HISTORICAL_CATEGORICAL, teacher_numeric)
    teacher_fit.fit(
        teacher_train.loc[teacher_fit_mask],
        teacher_train.loc[teacher_fit_mask, "is_make"].astype(int),
    )
    calibration_raw = teacher_fit.predict_proba(
        teacher_train.loc[teacher_calibration_mask]
    )[:, 1]
    teacher_calibrator = fit_probability_calibrator(
        calibration_raw,
        teacher_train.loc[teacher_calibration_mask, "is_make"].astype(int),
    )

    def transfer_target(
        model: Pipeline,
        calibrator: dict[str, float],
        frame: pd.DataFrame,
    ) -> np.ndarray:
        target = pd.to_numeric(
            frame.get("external_teacher_probability", np.nan),
            errors="coerce",
        )
        if not isinstance(target, pd.Series):
            target = pd.Series(np.nan, index=frame.index, dtype=float)
        target = target.reindex(frame.index)
        needs_prediction = target.isna() & frame["initial_make_prob"].notna()
        if needs_prediction.any():
            target.loc[needs_prediction] = apply_probability_calibrator(
                model.predict_proba(frame.loc[needs_prediction])[:, 1],
                calibrator,
            )
        if target.isna().any():
            raise ValueError("Historical training rows lack both rich OOF and modern teacher targets.")
        return _safe_probability(target)

    fit_teacher_probability = transfer_target(
        teacher_fit,
        teacher_calibrator,
        train.loc[fit_mask],
    )

    # Two leakage-safe historical candidates are fit on the early training
    # block: direct outcome prediction and teacher distillation.  Their blend
    # weight and calibration are selected only on the later training block.
    student_fit = build_regressor(HISTORICAL_CATEGORICAL, HISTORICAL_NUMERIC)
    student_fit.fit(train.loc[fit_mask], fit_teacher_probability)
    calibration_student_raw = _safe_probability(student_fit.predict(train.loc[calibration_mask]))
    student_calibrator = fit_probability_calibrator(
        calibration_student_raw,
        train.loc[calibration_mask, "is_make"].astype(int),
    )
    calibration_student = apply_probability_calibrator(
        calibration_student_raw,
        student_calibrator,
    )

    direct_fit = build_classifier(HISTORICAL_CATEGORICAL, HISTORICAL_NUMERIC)
    direct_fit.fit(train.loc[fit_mask], train.loc[fit_mask, "is_make"].astype(int))
    calibration_direct_raw = direct_fit.predict_proba(train.loc[calibration_mask])[:, 1]
    direct_calibrator = fit_probability_calibrator(
        calibration_direct_raw,
        train.loc[calibration_mask, "is_make"].astype(int),
    )
    calibration_direct = apply_probability_calibrator(
        calibration_direct_raw,
        direct_calibrator,
    )
    calibration_y = train.loc[calibration_mask, "is_make"].astype(int)
    blend_candidates = np.linspace(0.0, 1.0, 11)
    blend_losses = [
        brier_score_loss(
            calibration_y,
            weight * calibration_direct + (1.0 - weight) * calibration_student,
        )
        for weight in blend_candidates
    ]
    direct_blend_weight = float(blend_candidates[int(np.argmin(blend_losses))])

    teacher = build_classifier(HISTORICAL_CATEGORICAL, teacher_numeric)
    teacher.fit(teacher_train, teacher_train["is_make"].astype(int))
    train_teacher = transfer_target(teacher, teacher_calibrator, train)
    test_teacher = apply_probability_calibrator(
        teacher.predict_proba(test)[:, 1],
        teacher_calibrator,
    )

    student = build_regressor(HISTORICAL_CATEGORICAL, HISTORICAL_NUMERIC)
    student.fit(train, train_teacher)
    test_distilled = apply_probability_calibrator(
        _safe_probability(student.predict(test)),
        student_calibrator,
    )
    direct = build_classifier(HISTORICAL_CATEGORICAL, HISTORICAL_NUMERIC)
    direct.fit(train, train["is_make"].astype(int))
    test_direct = apply_probability_calibrator(
        direct.predict_proba(test)[:, 1],
        direct_calibrator,
    )
    test_student = _safe_probability(
        direct_blend_weight * test_direct
        + (1.0 - direct_blend_weight) * test_distilled
    )

    test["initial_probability"] = _safe_probability(test["initial_make_prob"])
    test["teacher_probability"] = test_teacher
    test["direct_probability"] = test_direct
    test["distilled_probability"] = test_distilled
    test["student_probability"] = test_student
    test["teacher_ev"] = test["teacher_probability"] * test["shot_value"]
    test["student_ev"] = test["student_probability"] * test["shot_value"]

    metric_rows: list[dict[str, object]] = []
    baseline_metrics = probability_metrics(test["is_make"], test["initial_probability"])
    for model_name, column in [
        ("initial_ev_baseline", "initial_probability"),
        ("sq_feature_teacher", "teacher_probability"),
        ("historical_direct_outcome", "direct_probability"),
        ("historical_teacher_distilled", "distilled_probability"),
        ("historical_pbp_student", "student_probability"),
    ]:
        values = probability_metrics(test["is_make"], test[column])
        intervals = paired_game_loss_intervals(test, column)
        metric_rows.append(
            {
                "model_bucket": str(test["model_bucket"].iloc[0]),
                "model": model_name,
                "shots": int(len(test)),
                "train_games": int(train["game_id"].nunique()),
                "test_games": int(test["game_id"].nunique()),
                **values,
                "brier_improvement_vs_initial_ev": baseline_metrics["brier"] - values["brier"],
                "log_loss_improvement_vs_initial_ev": baseline_metrics["log_loss"] - values["log_loss"],
                **intervals,
                "direct_blend_weight": direct_blend_weight if model_name == "historical_pbp_student" else np.nan,
            }
        )
    metric_rows.append(
        {
            "model_bucket": str(test["model_bucket"].iloc[0]),
            "model": "student_distillation_error",
            "shots": int(len(test)),
            "train_games": int(train["game_id"].nunique()),
            "test_games": int(test["game_id"].nunique()),
            "brier": np.nan,
            "log_loss": np.nan,
            "calibration_bias": float(test_distilled.mean() - test_teacher.mean()),
            "brier_improvement_vs_initial_ev": np.nan,
            "log_loss_improvement_vs_initial_ev": np.nan,
            "teacher_student_mae": float(mean_absolute_error(test_teacher, test_distilled)),
            "teacher_student_rmse": float(mean_squared_error(test_teacher, test_distilled) ** 0.5),
        }
    )
    package = {
        "teacher": teacher,
        "teacher_calibrator": teacher_calibrator,
        "student": student,
        "student_calibrator": student_calibrator,
        "direct": direct,
        "direct_calibrator": direct_calibrator,
        "direct_blend_weight": direct_blend_weight,
        "teacher_categorical": HISTORICAL_CATEGORICAL,
        "teacher_numeric": teacher_numeric,
        "student_categorical": HISTORICAL_CATEGORICAL,
        "student_numeric": HISTORICAL_NUMERIC,
    }
    return package, test, metric_rows


def train_transfer_models(modern_rows: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    modern_rows = add_pregame_shooter_priors(modern_rows)
    eligible = modern_rows[
        modern_rows["model_bucket"].isin(MODEL_BUCKETS)
        & (
            modern_rows["initial_make_prob"].notna()
            | modern_rows.get(
                "external_teacher_probability",
                pd.Series(np.nan, index=modern_rows.index),
            ).notna()
        )
    ].copy()
    sq_descriptor_columns = sorted(
        column for column in eligible.columns if column.startswith("sq_desc_")
    )
    for column in sq_descriptor_columns:
        eligible[column] = pd.to_numeric(eligible[column], errors="coerce").fillna(0.0)

    packages: dict[str, object] = {}
    tests = []
    metrics: list[dict[str, object]] = []
    for bucket in MODEL_BUCKETS:
        bucket_rows = eligible[eligible["model_bucket"].eq(bucket)].copy()
        if bucket_rows["game_id"].nunique() < 5:
            raise ValueError(f"Not enough games to train {bucket}.")
        package, test, bucket_metrics = fit_bucket_models(
            bucket_rows,
            sq_descriptor_columns=sq_descriptor_columns,
        )
        packages[bucket] = package
        tests.append(test)
        metrics.extend(bucket_metrics)

    artifact = {
        "schema_version": 1,
        "model_buckets": list(MODEL_BUCKETS),
        "sq_descriptor_columns": sq_descriptor_columns,
        "historical_categorical": HISTORICAL_CATEGORICAL,
        "historical_numeric": HISTORICAL_NUMERIC,
        "models": packages,
        "training_seasons": sorted(int(v) for v in eligible["season_end_year"].unique()),
        "training_files": sorted(str(v) for v in eligible["source_file"].unique()),
        "notes": (
            "Modern teacher uses ShotQuality initial EV and pre-shot SQ descriptors; dynamic EV is excluded. "
            "Historical student blends direct outcome and rich/modern-teacher distillation models using only "
            "historical PBP plus current-game-excluded shooter priors. Models/calibration are separate by bucket."
        ),
    }
    return artifact, pd.concat(tests, ignore_index=True), pd.DataFrame(metrics)


def score_student_rows(rows: pd.DataFrame, artifact: dict[str, object]) -> pd.DataFrame:
    scored = rows.copy()
    if "shooter_pregame_make_rate" not in scored:
        scored["shooter_pregame_make_rate"] = scored["model_bucket"].map(
            {"midrange": 0.41, "three": 0.36}
        ).fillna(0.40)
    if "shooter_pregame_log_attempts" not in scored:
        scored["shooter_pregame_log_attempts"] = 0.0
    scored["historical_sq_make_prob"] = np.nan
    scored["historical_sq_direct_make_prob"] = np.nan
    scored["historical_sq_distilled_make_prob"] = np.nan
    for bucket in artifact["model_buckets"]:
        mask = scored["model_bucket"].eq(bucket)
        if not mask.any():
            continue
        package = artifact["models"][bucket]
        distilled = apply_probability_calibrator(
            _safe_probability(package["student"].predict(scored.loc[mask])),
            package["student_calibrator"],
        )
        direct = apply_probability_calibrator(
            package["direct"].predict_proba(scored.loc[mask])[:, 1],
            package["direct_calibrator"],
        )
        weight = package["direct_blend_weight"]
        scored.loc[mask, "historical_sq_direct_make_prob"] = direct
        scored.loc[mask, "historical_sq_distilled_make_prob"] = distilled
        scored.loc[mask, "historical_sq_make_prob"] = _safe_probability(
            weight * direct + (1.0 - weight) * distilled
        )
    scored["historical_sq_ev"] = scored["historical_sq_make_prob"] * scored["shot_value"]
    scored["historical_sq_component_gap"] = (
        scored["historical_sq_direct_make_prob"]
        - scored["historical_sq_distilled_make_prob"]
    ).abs()
    return scored


def _parse_score_files(raw_dir: Path, tokens: list[str] | None) -> list[Path]:
    if not tokens:
        return []
    years = {year for token in tokens for year in parse_year_token(token)}
    return discover_raw_files(raw_dir, years, {"rs", "ps"})


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--modern-files", nargs="*", type=Path)
    parser.add_argument(
        "--rich-oof-predictions",
        nargs="*",
        type=Path,
        help="Optional 2024-25 rich-feature OOF prediction parquets.",
    )
    parser.add_argument(
        "--no-rich-oof",
        action="store_true",
        help="Do not auto-use the local 2024-25 SQ challenger OOF artifacts.",
    )
    parser.add_argument("--score-years", nargs="*", help="Optional historical season-end years/ranges.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-name", default="historical_sq_transfer")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    modern_files = args.modern_files or discover_modern_files(args.raw_dir)
    if not modern_files:
        raise FileNotFoundError("No raw files with initial_ev and SQ descriptors were found.")

    print("Modern labeled files:", ", ".join(path.name for path in modern_files))
    modern_frames = []
    for path in modern_files:
        frame = load_feature_rows(path, require_sq=True)
        modern_frames.append(frame)
        print(f"  {path.name}: {len(frame):,} FGA rows, {frame['initial_ev'].notna().sum():,} SQ labels")
    rich_paths = args.rich_oof_predictions
    if rich_paths is None and not args.no_rich_oof:
        rich_paths = [path for path in DEFAULT_RICH_OOF if path.exists()]
    if rich_paths:
        rich_rows = load_rich_oof_teacher_rows(rich_paths, raw_dir=args.raw_dir)
        modern_frames.append(rich_rows)
        print(
            f"  Rich OOF transfer: {len(rich_rows):,} joined 2024-25 rows "
            f"from {len(rich_paths)} prediction files"
        )
    modern = pd.concat(modern_frames, ignore_index=True, sort=False)
    artifact, heldout, metrics = train_transfer_models(modern)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / f"{args.output_name}_models.joblib"
    benchmark_path = args.output_dir / f"{args.output_name}_benchmark.csv"
    heldout_path = args.output_dir / f"{args.output_name}_heldout_predictions.parquet"
    calibration_path = args.output_dir / f"{args.output_name}_calibration.csv"
    manifest_path = args.output_dir / f"{args.output_name}_manifest.json"
    joblib.dump(artifact, model_path)
    metrics.to_csv(benchmark_path, index=False)
    heldout.to_parquet(heldout_path, index=False)
    calibration_table(
        heldout,
        [
            "initial_probability",
            "teacher_probability",
            "direct_probability",
            "distilled_probability",
            "student_probability",
        ],
    ).to_csv(calibration_path, index=False)

    score_files = _parse_score_files(args.raw_dir, args.score_years)
    scored_paths = []
    if score_files:
        score_frames = [load_feature_rows(path, require_sq=False) for path in score_files]
        all_score_rows = add_pregame_shooter_priors(pd.concat(score_frames, ignore_index=True))
        all_scored = score_student_rows(all_score_rows, artifact)
        for path in score_files:
            scored = all_scored[all_scored["source_file"].eq(path.name)].copy()
            output_path = args.output_dir / f"{args.output_name}_{path.stem}.parquet"
            scored.to_parquet(output_path, index=False)
            scored_paths.append(str(output_path))
            print(f"Scored {len(scored):,} FGA rows -> {output_path}")

    manifest = {
        "model_path": str(model_path),
        "benchmark_path": str(benchmark_path),
        "heldout_predictions_path": str(heldout_path),
        "calibration_path": str(calibration_path),
        "modern_files": [str(path) for path in modern_files],
        "rich_oof_prediction_files": [str(path) for path in (rich_paths or [])],
        "scored_files": scored_paths,
        "training_seasons": artifact["training_seasons"],
        "temporal_holdout": "latest 20% of games by season/type/game_id",
        "season_holdout_available": len(artifact["training_seasons"]) > 1,
        "independent_shot_clock_labels_available": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(metrics.to_string(index=False))
    print(f"Wrote model package to {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
