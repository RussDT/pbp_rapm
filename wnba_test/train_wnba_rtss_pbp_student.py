#!/usr/bin/env python3
"""Distill WNBA RTSS qSQ into an outcome-safe PBP ShotQuality student.

RTSS is the richer teacher on exactly matched 2023-2025 shots.  The student
uses only fields available in the Gabriel play-by-play shot table, so it can
score games and seasons without RTSS.  Validation is chronological:

* fit 2023, score 2024;
* fit 2023-2024, score 2025.

Season 2026 is fenced out because it is reserved for the existing RTSS
confirmation protocol.  This script is a shot-level research experiment; it
does not write possession targets or alter the published WNBA VPM board.
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RTSS_ROOT = ROOT / "downloads" / "rtss-data" / "wnba" / "downloads"
DEFAULT_INVENTORY = (
    ROOT / "wnba_test" / "Results" / "rtss_feature_audit" / "game_inventory.csv"
)
DEFAULT_PBP_PREDICTIONS = (
    ROOT
    / "wnba_test"
    / "Results"
    / "pbp_shot_quality"
    / "wnba_pbp_shot_quality_forward.parquet"
)
DEFAULT_OUTPUT = ROOT / "wnba_test" / "Results" / "rtss_pbp_student"

DEVELOPMENT_SEASONS = (2023, 2024, 2025)
CONFIRMATION_SEASON = 2026
IMPACT_TWO_POINT_TYPES = {"jumper", "post", "floater"}

NUMERIC_FEATURES = [
    "shot_value",
    "shot_distance",
    "xlegacy",
    "ylegacy",
    "period",
    "seconds_remaining_quarter",
    "corner_three",
    "desc_driving",
    "desc_running",
    "desc_cutting",
    "desc_pullup",
    "desc_step_back",
    "desc_turnaround",
    "desc_fadeaway",
    "desc_bank",
    "desc_alley_oop",
    "desc_putback",
    "desc_reverse",
    "desc_finger_roll",
    "desc_floating",
    "desc_tip",
]
CATEGORICAL_FEATURES = [
    "shot_zone",
    "distance_bin",
    "shot_family",
    "clock_bucket",
    "location_side",
]
FORBIDDEN_STUDENT_FEATURES = {
    "actual_points",
    "is_make",
    "shotresult",
    "actiontype",
    "description",
    "made_assisted_tag",
    "qSQ",
    "qSP",
    "teacher_qsq_ev",
    "provider_qsp_ev",
}


@dataclass
class StudentModel:
    encoder: OrdinalEncoder
    model: HistGradientBoostingRegressor


def normalize_game_id(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(10)
    )


def q_score(value: Any) -> float | None:
    """Normalize an RTSS q value stored as a fraction or percent."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    if result > 1.0:
        result /= 100.0
    if not 0.0 <= result <= 1.0:
        return None
    return float(result)


def q_expected_points(value: Any) -> float | None:
    """Convert provider expected-eFG q to expected points per FGA."""
    score = q_score(value)
    return None if score is None else 2.0 * score


def extract_rtss_teacher_game(args: tuple[str, str, int]) -> list[dict[str, Any]]:
    """Extract the MID/3 teacher contract from one 2D markings file."""
    path_string, official_game_id, inventory_season = args
    path = Path(path_string)
    payload = json.loads(path.read_text())
    rows: list[dict[str, Any]] = []
    for shot in payload.get("shots", []):
        try:
            season = int(shot.get("season"))
        except (TypeError, ValueError):
            continue
        if season != int(inventory_season) or season not in DEVELOPMENT_SEASONS:
            continue
        is_three = bool(shot.get("three"))
        shot_type = str(shot.get("shotType") or "").strip().lower()
        if not (is_three or shot_type in IMPACT_TWO_POINT_TYPES):
            continue
        qsq_ev = q_expected_points(shot.get("qSQ"))
        qsp_ev = q_expected_points(shot.get("qSP"))
        try:
            event_num = int(float(shot.get("pbpId")))
            period = int(float(shot.get("period")))
        except (TypeError, ValueError):
            continue
        if qsq_ev is None:
            continue
        rows.append(
            {
                "season": season,
                "game_id": str(official_game_id).zfill(10),
                "event_num": event_num,
                "provider_game_id": path.parent.name,
                "provider_shot_id": str(shot.get("id") or ""),
                "period_rtss": period,
                "shot_value_rtss": 3.0 if is_three else 2.0,
                "outcome_rtss": bool(shot.get("outcome")),
                "teacher_qsq_ev": qsq_ev,
                "provider_qsp_ev": qsp_ev,
            }
        )
    return rows


def load_rtss_teacher_shots(
    rtss_root: Path,
    inventory_path: Path,
    workers: int,
) -> pd.DataFrame:
    inventory = pd.read_csv(inventory_path, dtype={"game_id": str})
    required = {"game_id", "provider_game_id", "season"}
    missing = required.difference(inventory.columns)
    if missing:
        raise ValueError(f"RTSS game inventory missing columns: {sorted(missing)}")
    inventory["season"] = pd.to_numeric(inventory["season"], errors="coerce")
    inventory = inventory.loc[inventory["season"].isin(DEVELOPMENT_SEASONS)].copy()
    if inventory["provider_game_id"].duplicated().any():
        raise ValueError("RTSS game inventory has duplicate provider game ids")
    jobs = []
    for row in inventory.itertuples(index=False):
        path = rtss_root / str(row.provider_game_id) / "basketball-markings.json"
        if path.exists():
            jobs.append((str(path), str(row.game_id), int(row.season)))
    if not jobs:
        raise FileNotFoundError(f"No development RTSS markings found under {rtss_root}")
    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for rows in pool.map(extract_rtss_teacher_game, jobs, chunksize=2):
            records.extend(rows)
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise RuntimeError("No eligible RTSS teacher shots were extracted")
    if int(frame["season"].max()) >= CONFIRMATION_SEASON:
        raise RuntimeError("RTSS confirmation season crossed the development fence")
    return frame


def load_pbp_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {
        "season_end_year",
        "game_id",
        "event_num",
        "actual_points",
        "is_make",
        "shot_value",
        "shot_value_baseline_ev",
        "shot_context_ev",
        "shot_quality_prior_ev",
        *NUMERIC_FEATURES,
        *CATEGORICAL_FEATURES,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"PBP prediction artifact missing columns: {sorted(missing)}")
    frame = frame.loc[
        pd.to_numeric(frame["season_end_year"], errors="coerce").isin(
            DEVELOPMENT_SEASONS
        )
    ].copy()
    frame["season_end_year"] = frame["season_end_year"].astype(int)
    frame["game_id"] = normalize_game_id(frame["game_id"])
    frame["event_num"] = pd.to_numeric(frame["event_num"], errors="coerce").astype(
        "Int64"
    )
    if frame.duplicated(["game_id", "event_num"], keep=False).any():
        raise ValueError("PBP prediction artifact has ambiguous game/event shot keys")
    return frame.reset_index(drop=True)


def exact_teacher_join(
    teacher: pd.DataFrame,
    pbp: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join on audited game/event keys and exclude every conflicting payload."""
    duplicate_teacher = teacher.duplicated(["game_id", "event_num"], keep=False)
    unique_teacher = teacher.loc[~duplicate_teacher].copy()
    joined = unique_teacher.merge(
        pbp,
        on=["game_id", "event_num"],
        how="inner",
        validate="one_to_one",
        suffixes=("_rtss", ""),
    )
    joined["period_agree"] = joined["period_rtss"].eq(joined["period"])
    joined["shot_value_agree"] = joined["shot_value_rtss"].eq(joined["shot_value"])
    joined["outcome_agree"] = joined["outcome_rtss"].eq(joined["is_make"])
    valid = (
        joined["period_agree"]
        & joined["shot_value_agree"]
        & joined["outcome_agree"]
    )
    audit = {
        "rtss_eligible_shots": int(len(teacher)),
        "rtss_ambiguous_game_event_rows": int(duplicate_teacher.sum()),
        "exact_key_matches": int(len(joined)),
        "period_mismatches": int((~joined["period_agree"]).sum()),
        "shot_value_mismatches": int((~joined["shot_value_agree"]).sum()),
        "outcome_mismatches": int((~joined["outcome_agree"]).sum()),
        "validated_teacher_matches": int(valid.sum()),
        "validated_by_season": {
            str(int(year)): int(count)
            for year, count in joined.loc[valid].groupby("season_end_year").size().items()
        },
    }
    return joined.loc[valid].reset_index(drop=True), audit


def _feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    forbidden = FORBIDDEN_STUDENT_FEATURES.intersection(features)
    if forbidden:
        raise RuntimeError(f"Outcome/provider fields entered student features: {sorted(forbidden)}")
    return frame[features].copy()


def _numeric_matrix(frame: pd.DataFrame) -> np.ndarray:
    columns = []
    for column in NUMERIC_FEATURES:
        values = pd.to_numeric(frame[column], errors="coerce").astype(float)
        columns.append(values.to_numpy())
    return np.column_stack(columns)


def fit_student(train: pd.DataFrame, target_column: str) -> StudentModel:
    if train.empty:
        raise ValueError("Cannot fit the RTSS/PBP student on an empty frame")
    _feature_frame(train)
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
        encoded_missing_value=np.nan,
        dtype=float,
    )
    categorical = encoder.fit_transform(
        train[CATEGORICAL_FEATURES].fillna("missing").astype(str)
    )
    numeric = _numeric_matrix(train)
    matrix = np.column_stack([numeric, categorical])
    categorical_mask = np.array(
        [False] * len(NUMERIC_FEATURES) + [True] * len(CATEGORICAL_FEATURES)
    )
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=100,
        l2_regularization=5.0,
        categorical_features=categorical_mask,
        early_stopping=False,
        random_state=20260825,
    )
    target = pd.to_numeric(train[target_column], errors="coerce").to_numpy(float)
    model.fit(matrix, target)
    return StudentModel(encoder=encoder, model=model)


def predict_student(bundle: StudentModel, validation: pd.DataFrame) -> np.ndarray:
    _feature_frame(validation)
    categorical = bundle.encoder.transform(
        validation[CATEGORICAL_FEATURES].fillna("missing").astype(str)
    )
    numeric = _numeric_matrix(validation)
    prediction = bundle.model.predict(np.column_stack([numeric, categorical]))
    shot_value = pd.to_numeric(validation["shot_value"], errors="coerce").to_numpy(float)
    return np.minimum(np.maximum(prediction, 0.01), shot_value)


def prediction_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    actual = pd.to_numeric(frame["actual_points"], errors="coerce").to_numpy(float)
    shot_value = pd.to_numeric(frame["shot_value"], errors="coerce").to_numpy(float)
    make = (actual > 0).astype(float)
    probability = np.clip(prediction / shot_value, 1e-6, 1 - 1e-6)
    mse = float(np.mean(np.square(actual - prediction)))
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "brier": float(np.mean(np.square(make - probability))),
        "mean_prediction": float(np.mean(prediction)),
        "mean_actual": float(np.mean(actual)),
    }


def paired_game_bootstrap(
    frame: pd.DataFrame,
    candidate: np.ndarray,
    baseline: np.ndarray,
    draws: int,
) -> dict[str, float]:
    """Cluster by game; positive baseline-minus-candidate MSE favors candidate."""
    actual = pd.to_numeric(frame["actual_points"], errors="coerce").to_numpy(float)
    delta = np.square(actual - baseline) - np.square(actual - candidate)
    clusters = pd.DataFrame(
        {"game_id": frame["game_id"].astype(str).to_numpy(), "delta": delta}
    ).groupby("game_id")["delta"].agg(["sum", "size"])
    rng = np.random.default_rng(20260825)
    sums = clusters["sum"].to_numpy(float)
    sizes = clusters["size"].to_numpy(float)
    sampled = rng.integers(0, len(clusters), size=(int(draws), len(clusters)))
    boot = sums[sampled].sum(axis=1) / sizes[sampled].sum(axis=1)
    return {
        "mse_gain": float(delta.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "games": int(len(clusters)),
    }


def evaluate(
    pbp: pd.DataFrame,
    teacher_matches: pd.DataFrame,
    bootstrap_draws: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for validation_year in (2024, 2025):
        train_teacher = teacher_matches.loc[
            teacher_matches["season_end_year"].lt(validation_year)
        ].copy()
        validation_all = pbp.loc[pbp["season_end_year"].eq(validation_year)].copy()
        qsq_student = fit_student(train_teacher, "teacher_qsq_ev")
        outcome_control = fit_student(train_teacher, "actual_points")
        scored = validation_all.copy()
        scored["student_qsq_ev"] = predict_student(qsq_student, validation_all)
        scored["outcome_control_ev"] = predict_student(outcome_control, validation_all)
        scored["incumbent_context_ev"] = pd.to_numeric(
            validation_all["shot_context_ev"], errors="coerce"
        )
        scored["incumbent_shooter_ev"] = pd.to_numeric(
            validation_all["shot_quality_prior_ev"], errors="coerce"
        )
        scored["value_baseline_ev"] = pd.to_numeric(
            validation_all["shot_value_baseline_ev"], errors="coerce"
        )
        provider = teacher_matches.loc[
            teacher_matches["season_end_year"].eq(validation_year),
            ["game_id", "event_num", "teacher_qsq_ev", "provider_qsp_ev"],
        ]
        scored = scored.merge(
            provider,
            on=["game_id", "event_num"],
            how="left",
            validate="one_to_one",
        )
        scored["rtss_matched"] = scored["teacher_qsq_ev"].notna()

        models = {
            "value_baseline": "value_baseline_ev",
            "incumbent_pbp_context": "incumbent_context_ev",
            "incumbent_pbp_context_plus_shooter": "incumbent_shooter_ev",
            "pbp_outcome_control_same_rows": "outcome_control_ev",
            "rtss_qsq_pbp_student": "student_qsq_ev",
        }
        for scope, subset in (
            ("all_pbp", scored),
            ("rtss_matched", scored.loc[scored["rtss_matched"]].copy()),
        ):
            scope_models = dict(models)
            if scope == "rtss_matched":
                scope_models["provider_qSQ"] = "teacher_qsq_ev"
                scope_models["provider_qSP"] = "provider_qsp_ev"
            for label, column in scope_models.items():
                values = pd.to_numeric(subset[column], errors="coerce").to_numpy(float)
                metrics = prediction_metrics(subset, values)
                metric_rows.append(
                    {
                        "validation_year": validation_year,
                        "train_years": ",".join(
                            map(
                                str,
                                sorted(train_teacher["season_end_year"].unique()),
                            )
                        ),
                        "scope": scope,
                        "model": label,
                        "shots": int(len(subset)),
                        **metrics,
                    }
                )
            candidate = pd.to_numeric(
                subset["student_qsq_ev"], errors="coerce"
            ).to_numpy(float)
            for baseline_label, baseline_column in (
                ("incumbent_pbp_context", "incumbent_context_ev"),
                (
                    "incumbent_pbp_context_plus_shooter",
                    "incumbent_shooter_ev",
                ),
                ("pbp_outcome_control_same_rows", "outcome_control_ev"),
            ):
                baseline = pd.to_numeric(
                    subset[baseline_column], errors="coerce"
                ).to_numpy(float)
                comparison_rows.append(
                    {
                        "validation_year": validation_year,
                        "scope": scope,
                        "candidate": "rtss_qsq_pbp_student",
                        "baseline": baseline_label,
                        **paired_game_bootstrap(
                            subset,
                            candidate,
                            baseline,
                            bootstrap_draws,
                        ),
                    }
                )

        common = scored.loc[scored["rtss_matched"]].copy()
        fidelity_error = (
            pd.to_numeric(common["student_qsq_ev"], errors="coerce")
            - pd.to_numeric(common["teacher_qsq_ev"], errors="coerce")
        )
        common["student_teacher_error"] = fidelity_error
        common["student_teacher_rmse"] = float(
            np.sqrt(np.mean(np.square(fidelity_error)))
        )
        common["student_teacher_correlation"] = float(
            common["student_qsq_ev"].corr(common["teacher_qsq_ev"])
        )
        prediction_frames.append(scored)
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(comparison_rows),
        pd.concat(prediction_frames, ignore_index=True),
    )


def add_2024_selected_blends(
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    predictions: pd.DataFrame,
    bootstrap_draws: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Select one blend weight on 2024 and apply it unchanged to 2025.

    Selection uses the deployable all-PBP scope.  The matched-shot result is a
    diagnostic scored with that same weight, not a separately tuned blend.
    """
    development = predictions.loc[predictions["season_end_year"].eq(2024)].copy()
    scored = predictions.copy()
    selections: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for baseline_label, baseline_column, blend_column in (
        (
            "incumbent_pbp_context",
            "incumbent_context_ev",
            "blend_context_student_ev",
        ),
        (
            "incumbent_pbp_context_plus_shooter",
            "incumbent_shooter_ev",
            "blend_shooter_student_ev",
        ),
    ):
        actual = pd.to_numeric(
            development["actual_points"], errors="coerce"
        ).to_numpy(float)
        baseline = pd.to_numeric(
            development[baseline_column], errors="coerce"
        ).to_numpy(float)
        student = pd.to_numeric(
            development["student_qsq_ev"], errors="coerce"
        ).to_numpy(float)
        direction = student - baseline
        denominator = float(np.sum(np.square(direction)))
        raw_weight = (
            float(np.sum((actual - baseline) * direction) / denominator)
            if denominator
            else 0.0
        )
        weight = float(np.clip(raw_weight, 0.0, 1.0))
        selections.append(
            {
                "selection_year": 2024,
                "selection_scope": "all_pbp",
                "baseline": baseline_label,
                "student_weight": weight,
                "raw_unbounded_weight": raw_weight,
            }
        )
        scored[blend_column] = (
            pd.to_numeric(scored[baseline_column], errors="coerce")
            + weight
            * (
                pd.to_numeric(scored["student_qsq_ev"], errors="coerce")
                - pd.to_numeric(scored[baseline_column], errors="coerce")
            )
        )
        validation = scored.loc[scored["season_end_year"].eq(2025)].copy()
        for scope, subset in (
            ("all_pbp", validation),
            (
                "rtss_matched",
                validation.loc[validation["rtss_matched"]].copy(),
            ),
        ):
            candidate = pd.to_numeric(
                subset[blend_column], errors="coerce"
            ).to_numpy(float)
            baseline_values = pd.to_numeric(
                subset[baseline_column], errors="coerce"
            ).to_numpy(float)
            metric_rows.append(
                {
                    "validation_year": 2025,
                    "train_years": "2023,2024; blend weight selected on 2024",
                    "scope": scope,
                    "model": f"rtss_student_2024_selected_blend_vs_{baseline_label}",
                    "shots": int(len(subset)),
                    "selected_student_weight": weight,
                    **prediction_metrics(subset, candidate),
                }
            )
            comparison_rows.append(
                {
                    "validation_year": 2025,
                    "scope": scope,
                    "candidate": (
                        f"rtss_student_2024_selected_blend_vs_{baseline_label}"
                    ),
                    "baseline": baseline_label,
                    "selected_student_weight": weight,
                    **paired_game_bootstrap(
                        subset,
                        candidate,
                        baseline_values,
                        bootstrap_draws,
                    ),
                }
            )
    return (
        pd.concat([metrics, pd.DataFrame(metric_rows)], ignore_index=True),
        pd.concat(
            [comparisons, pd.DataFrame(comparison_rows)], ignore_index=True
        ),
        scored,
        selections,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtss-root", type=Path, default=DEFAULT_RTSS_ROOT)
    parser.add_argument("--game-inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument(
        "--pbp-predictions", type=Path, default=DEFAULT_PBP_PREDICTIONS
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    teacher = load_rtss_teacher_shots(
        args.rtss_root,
        args.game_inventory,
        args.workers,
    )
    pbp = load_pbp_predictions(args.pbp_predictions)
    matches, join_audit = exact_teacher_join(teacher, pbp)
    metrics, comparisons, predictions = evaluate(
        pbp,
        matches,
        args.bootstrap_draws,
    )
    metrics, comparisons, predictions, blend_selections = add_2024_selected_blends(
        metrics,
        comparisons,
        predictions,
        args.bootstrap_draws,
    )
    fidelity_rows = []
    for year, frame in predictions.loc[predictions["rtss_matched"]].groupby(
        "season_end_year"
    ):
        student = pd.to_numeric(frame["student_qsq_ev"], errors="coerce")
        teacher_values = pd.to_numeric(frame["teacher_qsq_ev"], errors="coerce")
        fidelity_rows.append(
            {
                "validation_year": int(year),
                "shots": int(len(frame)),
                "teacher_rmse": float(
                    np.sqrt(np.mean(np.square(student - teacher_values)))
                ),
                "teacher_correlation": float(student.corr(teacher_values)),
                "student_mean_ev": float(student.mean()),
                "teacher_mean_ev": float(teacher_values.mean()),
            }
        )
    fidelity = pd.DataFrame(fidelity_rows)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "forward_metrics.csv", index=False)
    comparisons.to_csv(output / "paired_game_clustered_comparisons.csv", index=False)
    fidelity.to_csv(output / "teacher_fidelity.csv", index=False)
    predictions.to_parquet(output / "forward_predictions.parquet", index=False)
    summary = {
        "status": "complete",
        "model": "WNBA RTSS qSQ to PBP ShotQuality student v1",
        "development_seasons": list(DEVELOPMENT_SEASONS),
        "confirmation_season": CONFIRMATION_SEASON,
        "confirmation_status": "sealed; not read or scored",
        "teacher": "RTSS qSQ calibrated as expected points = 2 * qSQ",
        "student_features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "forbidden_student_features": sorted(FORBIDDEN_STUDENT_FEATURES),
        "join_audit": join_audit,
        "validation_folds": [
            {"train": [2023], "score": 2024},
            {"train": [2023, 2024], "score": 2025},
        ],
        "blend_selection": blend_selections,
        "published_vpm_changed": False,
        "outputs": {
            "metrics": str(output / "forward_metrics.csv"),
            "comparisons": str(output / "paired_game_clustered_comparisons.csv"),
            "teacher_fidelity": str(output / "teacher_fidelity.csv"),
            "predictions": str(output / "forward_predictions.parquet"),
        },
    }
    (output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(join_audit, indent=2))
    print(metrics.to_string(index=False))
    print(comparisons.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
