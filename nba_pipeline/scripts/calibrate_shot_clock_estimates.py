#!/usr/bin/env python3
"""Calibrate rule-based shot-clock estimates against independent shot logs.

The supported label source is the preserved 2014-15 NBA player-shot-log
export.  It contains player-game shot ordinal, game clock, and the arena
``SHOT_CLOCK`` value.  Labels are joined without using shot clock itself:
game + shooter + player-game FGA ordinal, followed by strict period,
make/miss, point-value, and game-clock consistency gates.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from nba_pipeline.scripts.build_pbp_shot_quality import (
    add_shot_features,
    load_shots_from_file,
)
from nba_pipeline.scripts.build_shot_clock_estimates import estimate_shot_clock


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = PIPELINE_ROOT / "raw_data" / "NBA15.parquet"
DEFAULT_LABELS = PIPELINE_ROOT / ".cache" / "shot_clock_labels" / "shot_logs.csv"
DEFAULT_OUTPUT = PIPELINE_ROOT / "validation" / "shot_clock_calibration"

CATEGORICAL_FEATURES = ["reset_reason", "confidence"]
NUMERIC_FEATURES = [
    "shot_clock_est",
    "shot_clock_raw",
    "elapsed_since_reset",
    "reset_len",
    "rule_reset_len",
    "clock_sec",
    "reset_clock_sec",
    "period",
    "needs_review_numeric",
    "game_clock_below_24",
    "overtime",
]


def normalize_game_id(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(10)
    )


def parse_clock_seconds(series: pd.Series) -> pd.Series:
    parts = series.fillna("").astype(str).str.extract(
        r"(?P<minute>\d+):(?P<second>\d+(?:\.\d+)?)"
    )
    return (
        pd.to_numeric(parts["minute"], errors="coerce") * 60
        + pd.to_numeric(parts["second"], errors="coerce")
    )


def join_player_shot_log_labels(
    raw_path: Path,
    label_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Map independent shot-clock labels to raw event numbers."""
    shots = add_shot_features(load_shots_from_file(raw_path))
    shots["clock_sec_join"] = parse_clock_seconds(shots["time_quarter"])
    shots = shots.sort_values(
        ["game_id", "period", "clock_sec_join", "event_num"],
        ascending=[True, True, False, True],
        kind="stable",
    )
    join_key = ["game_id", "shooter_id", "player_game_shot_number"]
    shots["player_game_shot_number"] = (
        shots.groupby(["game_id", "shooter_id"], dropna=False, observed=True).cumcount() + 1
    )
    raw_duplicate_keys = shots.duplicated(join_key, keep=False)

    labels = pd.read_csv(label_path)
    required = {
        "GAME_ID",
        "PERIOD",
        "GAME_CLOCK",
        "SHOT_CLOCK",
        "SHOT_NUMBER",
        "PTS_TYPE",
        "SHOT_RESULT",
        "player_id",
    }
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"Shot-log label file is missing: {missing}")
    labels["game_id"] = normalize_game_id(labels["GAME_ID"])
    labels["shooter_id"] = pd.to_numeric(labels["player_id"], errors="coerce").astype("Int64")
    labels["player_game_shot_number"] = pd.to_numeric(
        labels["SHOT_NUMBER"], errors="coerce"
    ).astype("Int64")
    labels["label_period"] = pd.to_numeric(labels["PERIOD"], errors="coerce")
    labels["label_game_clock_sec"] = parse_clock_seconds(labels["GAME_CLOCK"])
    labels["label_is_make"] = labels["SHOT_RESULT"].astype(str).str.lower().eq("made")
    labels["label_shot_value"] = pd.to_numeric(labels["PTS_TYPE"], errors="coerce")
    labels["shot_clock_actual"] = pd.to_numeric(labels["SHOT_CLOCK"], errors="coerce")
    label_duplicate_keys = labels.duplicated(join_key, keep=False)

    raw_unique = shots[~raw_duplicate_keys].copy()
    label_unique = labels[~label_duplicate_keys].copy()
    candidates = raw_unique.merge(
        label_unique[
            join_key
            + [
                "label_period",
                "label_game_clock_sec",
                "label_is_make",
                "label_shot_value",
                "shot_clock_actual",
            ]
        ],
        on=join_key,
        how="inner",
        validate="one_to_one",
    )
    candidates["game_clock_gap"] = (
        candidates["clock_sec_join"] - candidates["label_game_clock_sec"]
    ).abs()
    candidates["period_agrees"] = candidates["period"].eq(candidates["label_period"])
    candidates["outcome_agrees"] = candidates["is_make"].eq(candidates["label_is_make"])
    candidates["value_agrees"] = candidates["shot_value"].eq(candidates["label_shot_value"])
    candidates["clock_agrees"] = candidates["game_clock_gap"].le(10)
    candidates["join_valid"] = (
        candidates[
            ["period_agrees", "outcome_agrees", "value_agrees", "clock_agrees"]
        ].all(axis=1)
        & candidates["shot_clock_actual"].between(0, 24, inclusive="both")
    )
    valid = candidates[candidates["join_valid"]].copy()
    mapped = valid[
        [
            "game_id",
            "period",
            "event_num",
            "shot_clock_actual",
            "game_clock_gap",
            "shooter_id",
            "player_game_shot_number",
        ]
    ].copy()

    nonnull_labels = int(labels["shot_clock_actual"].notna().sum())
    audit = {
        "raw_fga_rows": int(len(shots)),
        "raw_games": int(shots["game_id"].nunique()),
        "label_rows": int(len(labels)),
        "label_games": int(labels["game_id"].nunique()),
        "nonnull_label_rows": nonnull_labels,
        "raw_duplicate_join_keys": int(raw_duplicate_keys.sum()),
        "label_duplicate_join_keys": int(label_duplicate_keys.sum()),
        "ordinal_join_candidates": int(len(candidates)),
        "period_agreement_rate": float(candidates["period_agrees"].mean()),
        "outcome_agreement_rate": float(candidates["outcome_agrees"].mean()),
        "point_value_agreement_rate": float(candidates["value_agrees"].mean()),
        "within_10s_game_clock_rate": float(candidates["clock_agrees"].mean()),
        "valid_joined_labels": int(len(mapped)),
        "valid_coverage_of_nonnull_labels": float(len(mapped) / nonnull_labels),
        "unmatched_or_rejected_nonnull_labels": int(nonnull_labels - len(mapped)),
    }
    return mapped, audit


def build_calibration_features(estimates: pd.DataFrame) -> pd.DataFrame:
    features = estimates.copy()
    features["reset_clock_sec"] = parse_clock_seconds(features["reset_time_quarter"])
    features["needs_review_numeric"] = features["needs_review"].fillna(False).astype(int)
    features["game_clock_below_24"] = features["clock_sec"].lt(24).astype(int)
    features["overtime"] = features["period"].gt(4).astype(int)
    return features


def build_residual_model() -> Pipeline:
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )
    numeric = Pipeline([("impute", SimpleImputer(strategy="median"))])
    transform = ColumnTransformer(
        [
            ("categorical", categorical, CATEGORICAL_FEATURES),
            ("numeric", numeric, NUMERIC_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("features", transform),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="absolute_error",
                    learning_rate=0.05,
                    max_iter=300,
                    max_leaf_nodes=31,
                    min_samples_leaf=100,
                    l2_regularization=5.0,
                    categorical_features=list(range(len(CATEGORICAL_FEATURES))),
                    random_state=29,
                ),
            ),
        ]
    )


def apply_calibrator(
    estimates: pd.DataFrame,
    package: dict[str, object],
) -> pd.DataFrame:
    """Apply a serialized residual model while preserving the raw estimate."""
    scored = build_calibration_features(estimates)
    scored["shot_clock_calibrated"] = (
        scored["shot_clock_est"] + package["model"].predict(scored)
    ).clip(lower=0, upper=scored["reset_len"])
    group_q90 = package["group_q90"]
    scored = scored.merge(
        group_q90[["reset_reason", "confidence", "q90"]],
        on=["reset_reason", "confidence"],
        how="left",
    )
    scored["q90"] = scored["q90"].fillna(float(package["global_q90"]))
    scored["shot_clock_calibrated_lower"] = (
        scored["shot_clock_calibrated"] - scored["q90"]
    ).clip(lower=0)
    scored["shot_clock_calibrated_upper"] = np.minimum(
        scored["reset_len"],
        scored["shot_clock_calibrated"] + scored["q90"],
    )
    return scored


def _prediction_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    error = predicted.to_numpy(dtype=float) - actual.to_numpy(dtype=float)
    absolute = np.abs(error)
    return {
        "rows": int(len(actual)),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(error.mean()),
        "within_1s_rate": float((absolute <= 1).mean()),
        "within_2s_rate": float((absolute <= 2).mean()),
        "within_3s_rate": float((absolute <= 3).mean()),
    }


def fit_and_benchmark_calibrator(
    labeled_estimates: pd.DataFrame,
    *,
    test_fraction: float = 0.20,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = build_calibration_features(labeled_estimates).sort_values(
        ["game_id", "period", "clock_sec", "event_num"],
        ascending=[True, True, False, True],
        kind="stable",
    )
    games = sorted(rows["game_id"].astype(str).unique())
    cut = max(1, min(len(games) - 1, int(round(len(games) * (1 - test_fraction)))))
    train_games = set(games[:cut])
    test_games = set(games[cut:])
    train = rows[rows["game_id"].isin(train_games)].copy()
    test = rows[rows["game_id"].isin(test_games)].copy()

    # Game-grouped OOF residuals on training games provide an honest conformal
    # width without looking at the final ordered-game holdout.
    n_splits = min(3, train["game_id"].nunique())
    grouped = GroupKFold(n_splits=n_splits)
    train_oof = pd.Series(np.nan, index=train.index, dtype=float)
    for fit_idx, validation_idx in grouped.split(train, groups=train["game_id"]):
        fold_model = build_residual_model()
        fold_model.fit(
            train.iloc[fit_idx],
            (
                train.iloc[fit_idx]["shot_clock_actual"]
                - train.iloc[fit_idx]["shot_clock_est"]
            ),
        )
        train_oof.iloc[validation_idx] = fold_model.predict(train.iloc[validation_idx])
    oof_calibrated = (
        train["shot_clock_est"] + train_oof
    ).clip(lower=0, upper=train["reset_len"])
    oof_abs_error = (train["shot_clock_actual"] - oof_calibrated).abs()
    global_q90 = float(oof_abs_error.quantile(0.90))
    grouped_q90 = (
        train.assign(oof_abs_error=oof_abs_error)
        .groupby(["reset_reason", "confidence"], observed=True)["oof_abs_error"]
        .agg(["count", lambda x: float(x.quantile(0.90))])
        .reset_index()
        .rename(columns={"<lambda_0>": "q90"})
    )
    grouped_q90.loc[grouped_q90["count"].lt(200), "q90"] = global_q90

    model = build_residual_model()
    model.fit(train, train["shot_clock_actual"] - train["shot_clock_est"])
    package = {
        "schema_version": 1,
        "model": model,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "group_q90": grouped_q90,
        "global_q90": global_q90,
        "training_games": len(train_games),
        "heldout_games": len(test_games),
        "label_season_end_year": 2015,
    }
    test = apply_calibrator(test, package)
    test["calibrated_interval_covered"] = test["shot_clock_actual"].between(
        test["shot_clock_calibrated_lower"],
        test["shot_clock_calibrated_upper"],
        inclusive="both",
    )

    raw_metrics = _prediction_metrics(test["shot_clock_actual"], test["shot_clock_est"])
    calibrated_metrics = _prediction_metrics(
        test["shot_clock_actual"], test["shot_clock_calibrated"]
    )
    benchmark = pd.DataFrame(
        [
            {"model": "raw_rule_estimate", **raw_metrics},
            {
                "model": "game_heldout_residual_calibrator",
                **calibrated_metrics,
                "interval_90_coverage": float(test["calibrated_interval_covered"].mean()),
                "interval_half_width_global_q90": global_q90,
            },
        ]
    )
    by_reason = (
        test.assign(
            raw_error=test["shot_clock_est"] - test["shot_clock_actual"],
            calibrated_error=test["shot_clock_calibrated"] - test["shot_clock_actual"],
        )
        .groupby(["reset_reason", "confidence"], observed=True)
        .agg(
            rows=("shot_clock_actual", "size"),
            raw_mae=("raw_error", lambda x: float(x.abs().mean())),
            raw_bias=("raw_error", "mean"),
            calibrated_mae=("calibrated_error", lambda x: float(x.abs().mean())),
            calibrated_bias=("calibrated_error", "mean"),
            interval_90_coverage=("calibrated_interval_covered", "mean"),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    benchmark["training_games"] = len(train_games)
    benchmark["heldout_games"] = len(test_games)
    return package, test, benchmark, by_reason


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-name", default="shot_clock_2015")
    parser.add_argument(
        "--apply-estimates",
        type=Path,
        help="Apply an existing model package to an estimates parquet/CSV.",
    )
    parser.add_argument("--model", type=Path, help="Model package for --apply-estimates.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.apply_estimates:
        if not args.model:
            raise ValueError("--model is required with --apply-estimates.")
        package = joblib.load(args.model)
        if args.apply_estimates.suffix.lower() == ".parquet":
            estimates = pd.read_parquet(args.apply_estimates)
        else:
            estimates = pd.read_csv(args.apply_estimates)
        scored = apply_calibrator(estimates, package)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / f"{args.output_name}_applied.parquet"
        scored.to_parquet(output_path, index=False)
        print(f"Wrote {len(scored):,} calibrated estimates to {output_path}")
        return 0
    mapped, join_audit = join_player_shot_log_labels(args.raw, args.labels)
    raw = pd.read_parquet(args.raw)
    estimates = estimate_shot_clock(raw, season_end_year=2015)
    labeled = estimates.merge(
        mapped,
        on=["game_id", "period", "event_num"],
        how="inner",
        validate="one_to_one",
    )
    package, heldout, benchmark, by_reason = fit_and_benchmark_calibrator(labeled)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / args.output_name
    joblib.dump(package, Path(f"{prefix}_model.joblib"))
    heldout.to_parquet(Path(f"{prefix}_heldout.parquet"), index=False)
    benchmark.to_csv(Path(f"{prefix}_benchmark.csv"), index=False)
    by_reason.to_csv(Path(f"{prefix}_by_reason.csv"), index=False)
    Path(f"{prefix}_join_audit.json").write_text(
        json.dumps(join_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(join_audit, indent=2, sort_keys=True))
    print(benchmark.to_string(index=False))
    print(by_reason.head(12).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
