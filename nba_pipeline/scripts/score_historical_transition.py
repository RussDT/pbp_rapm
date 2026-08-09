#!/usr/bin/env python3
"""
Score historical NBA field-goal attempts with the SQ transition classifiers.

The scorer reconstructs the exact historical-safe feature contract stored in
the model bundles, applies calibration learned only from seasons with surviving
SQ labels, writes one parquet per raw RS/PS file, and emits file/era/feature
drift audits.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Iterable, Mapping

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold

try:
    from shotquality_modeling import calibration_error, clip_probability, write_json
    from train_sq_transition_classifier import (
        RAW_COLUMNS,
        build_historical_feature_matrix,
        load_raw_pbp,
        prepare_sequence_frame,
    )
except ImportError:  # pragma: no cover - package import in tests
    from nba_pipeline.scripts.shotquality_modeling import (
        calibration_error,
        clip_probability,
        write_json,
    )
    from nba_pipeline.scripts.train_sq_transition_classifier import (
        RAW_COLUMNS,
        build_historical_feature_matrix,
        load_raw_pbp,
        prepare_sequence_frame,
    )


LOGGER = logging.getLogger("score_historical_transition")
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PIPELINE_ROOT / "raw_data"
DEFAULT_MODELS_DIR = PIPELINE_ROOT / "results" / "sq_transition_classifier"
DEFAULT_OUTPUT_DIR = PIPELINE_ROOT / "results" / "sq_transition_backfill"
TARGETS = ["is_transition", "is_transition_dunk_or_layup"]
SHOT_ID_COLUMNS = ["player1_id", "player1_name", "player1_team_id"]
FEATURE_CONTRACT_VERSION = "sq_transition_historical_safe_v1_97"
CALIBRATION_VERSION = "game_group_crossfit_logit_v2"


def parse_raw_filename(path: Path) -> tuple[int, str]:
    match = re.fullmatch(r"NBA(\d{2})(_PS)?\.parquet", path.name)
    if not match:
        raise ValueError(f"Unrecognized raw NBA filename: {path.name}")
    suffix = int(match.group(1))
    year = 2000 + suffix if suffix <= 26 else 1900 + suffix
    season_type = "ps" if match.group(2) else "rs"
    return year, season_type


def era_label(year: int) -> str:
    if year <= 2003:
        return "1997-2003"
    if year <= 2010:
        return "2004-2010"
    if year <= 2016:
        return "2011-2016"
    if year <= 2020:
        return "2017-2020"
    if year <= 2024:
        return "2021-2024"
    return "2025-2026"


def discover_raw_paths(
    raw_dir: Path,
    *,
    start_year: int,
    end_year: int,
    season_types: str,
) -> list[Path]:
    selected: list[Path] = []
    allowed = {"rs", "ps"} if season_types == "all" else {season_types}
    for path in sorted(raw_dir.glob("NBA??*.parquet")):
        try:
            year, season_type = parse_raw_filename(path)
        except ValueError:
            continue
        if start_year <= year <= end_year and season_type in allowed:
            selected.append(path)
    if not selected:
        raise FileNotFoundError(
            f"No raw parquets for {start_year}-{end_year} {season_types} under {raw_dir}"
        )
    return selected


def _logit(values: Iterable[float]) -> np.ndarray:
    p = clip_probability(values)
    return np.log(p / (1.0 - p))


def fit_logit_calibrator(
    probability: Iterable[float],
    target: Iterable[int],
) -> dict[str, float]:
    p = clip_probability(probability)
    y = np.asarray(target, dtype=int)
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    model.fit(_logit(p).reshape(-1, 1), y)
    return {
        "intercept": float(model.intercept_[0]),
        "slope": float(model.coef_[0, 0]),
    }


def apply_logit_calibrator(
    probability: Iterable[float],
    calibrator: Mapping[str, float],
) -> np.ndarray:
    linear = (
        float(calibrator["intercept"])
        + float(calibrator["slope"]) * _logit(probability)
    )
    return 1.0 / (1.0 + np.exp(-np.clip(linear, -30.0, 30.0)))


def operating_threshold(
    target: Iterable[int],
    probability: Iterable[float],
) -> dict[str, float]:
    y = np.asarray(target, dtype=int)
    p = np.asarray(probability, dtype=float)
    candidates = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 199)))
    best_threshold: float | None = None
    best_score = -np.inf
    for threshold in candidates:
        score = float(f1_score(y, p >= threshold, zero_division=0))
        if score > best_score or (
            np.isclose(score, best_score)
            and (
                best_threshold is None
                or float(threshold) < best_threshold
            )
        ):
            best_threshold = float(threshold)
            best_score = score
    assert best_threshold is not None
    threshold = best_threshold
    label = p >= threshold
    return {
        "threshold": threshold,
        "f1": best_score,
        "precision": float(precision_score(y, label, zero_division=0)),
        "recall": float(recall_score(y, label, zero_division=0)),
    }


def calibration_diagnostics(
    target: np.ndarray,
    raw_probability: np.ndarray,
    calibrated_probability: np.ndarray,
) -> dict[str, float]:
    return {
        "rows": int(len(target)),
        "positive_rate": float(target.mean()),
        "raw_mean_probability": float(raw_probability.mean()),
        "calibrated_mean_probability": float(calibrated_probability.mean()),
        "raw_log_loss": float(log_loss(target, raw_probability, labels=[0, 1])),
        "calibrated_log_loss": float(
            log_loss(target, calibrated_probability, labels=[0, 1])
        ),
        "raw_brier": float(brier_score_loss(target, raw_probability)),
        "calibrated_brier": float(
            brier_score_loss(target, calibrated_probability)
        ),
        "raw_ece_10": calibration_error(target, raw_probability),
        "calibrated_ece_10": calibration_error(
            target, calibrated_probability
        ),
    }


def build_group_crossfit_calibration(
    frame: pd.DataFrame,
    *,
    folds: int = 5,
    seed: int = 20260723,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    predictions = np.full(len(frame), np.nan)
    fold_assignment = np.full(len(frame), -1, dtype=np.int16)
    splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    groups = (
        frame["calibration_season"].astype(str)
        + "::"
        + frame["game_id"].astype(str)
    )
    fold_payloads: list[dict[str, object]] = []
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(frame, frame["target"], groups)
    ):
        fit_game_ids = sorted(groups.iloc[train_idx].unique().tolist())
        validation_game_ids = sorted(groups.iloc[test_idx].unique().tolist())
        if set(fit_game_ids) & set(validation_game_ids):
            raise AssertionError("Calibration fit and validation games overlap")
        params = fit_logit_calibrator(
            frame.iloc[train_idx]["raw_probability"],
            frame.iloc[train_idx]["target"],
        )
        predictions[test_idx] = apply_logit_calibrator(
            frame.iloc[test_idx]["raw_probability"], params
        )
        fold_assignment[test_idx] = fold
        fold_payloads.append(
            {
                "fold": fold,
                "parameters": params,
                "fit_game_ids": fit_game_ids,
                "validation_game_ids": validation_game_ids,
            }
        )
    if np.isnan(predictions).any() or (fold_assignment < 0).any():
        raise AssertionError("Cross-fit calibration left rows unassigned")
    return predictions, fold_assignment, fold_payloads


def crossfit_calibration_diagnostics(
    frame: pd.DataFrame,
    *,
    folds: int = 5,
    seed: int = 20260723,
) -> dict[str, float]:
    predictions, _, _ = build_group_crossfit_calibration(
        frame, folds=folds, seed=seed
    )
    return calibration_diagnostics(
        frame["target"].to_numpy(dtype=int),
        frame["raw_probability"].to_numpy(dtype=float),
        predictions,
    )


def load_labeled_calibration_rows(models_dir: Path, target_name: str) -> pd.DataFrame:
    oof_path = models_dir / f"{target_name}_oof_predictions.parquet"
    forward_path = (
        models_dir / f"{target_name}_forward_2025_26_predictions.parquet"
    )
    if not oof_path.exists() or not forward_path.exists():
        raise FileNotFoundError(
            "Calibration requires both out-of-fold 2024-25 and untouched "
            f"2025-26 prediction artifacts; missing {oof_path} or {forward_path}"
        )
    oof = pd.read_parquet(oof_path)[
        ["game_id", "event_num", "target", "oof_probability"]
    ].rename(columns={"oof_probability": "raw_probability"})
    oof["calibration_season"] = 2025
    oof["base_probability_scope"] = "2025_game_group_oof"
    forward = pd.read_parquet(forward_path)[
        ["game_id", "event_num", "target", "forward_probability"]
    ].rename(columns={"forward_probability": "raw_probability"})
    forward["calibration_season"] = 2026
    forward["base_probability_scope"] = "2026_untouched_forward"
    return pd.concat([oof, forward], ignore_index=True)


def build_calibration_artifact(
    models_dir: Path,
    output_path: Path,
    *,
    seed: int = 20260723,
) -> dict[str, object]:
    artifact: dict[str, object] = {
        "version": CALIBRATION_VERSION,
        "policy": {
            "labeled_seasons": [2025, 2026],
            "2025": (
                "Base predictions are game-group OOF. Each row receives a "
                "calibrator fit only on other games in its calibration fold."
            ),
            "2026": (
                "Base predictions are untouched 2025-trained forward scores. "
                "Each row receives a calibrator fit only on other games in "
                "its calibration fold."
            ),
            "unlabeled_seasons": (
                "Pooled 2025+2026 labeled-season calibrator. No historical "
                "make/miss, transition heuristic, or inferred label is used "
                "to fit a historical season-specific calibrator. This is a "
                "retrospective labeled-era calibration, not prospective proof "
                "of calibration in the historical era."
            ),
            "binary_labels": (
                "The retrospective operating threshold maximizes F1 on pooled "
                "game-cross-fitted labeled-season probabilities. A separate "
                ">=0.5 label is also written."
            ),
        },
        "targets": {},
    }
    for target_name in TARGETS:
        labeled = load_labeled_calibration_rows(models_dir, target_name)
        target_payload: dict[str, object] = {"seasons": {}}
        pooled_crossfit_probability = np.full(len(labeled), np.nan)
        for season, season_frame in labeled.groupby(
            "calibration_season", sort=True
        ):
            season_frame = season_frame.copy()
            calibrated, fold_assignment, fold_payloads = (
                build_group_crossfit_calibration(
                    season_frame, seed=seed + int(season)
                )
            )
            pooled_crossfit_probability[season_frame.index.to_numpy()] = calibrated
            season_frame["calibration_fold"] = fold_assignment
            target_payload["seasons"][str(int(season))] = {
                "base_probability_scope": str(
                    season_frame["base_probability_scope"].iloc[0]
                ),
                "crossfit_diagnostics": calibration_diagnostics(
                    season_frame["target"].to_numpy(dtype=int),
                    season_frame["raw_probability"].to_numpy(dtype=float),
                    calibrated,
                ),
                "folds": fold_payloads,
            }
        if np.isnan(pooled_crossfit_probability).any():
            raise AssertionError("Pooled calibration rows were not all assigned")
        pooled_params = fit_logit_calibrator(
            labeled["raw_probability"], labeled["target"]
        )
        pooled_calibrated = apply_logit_calibrator(
            labeled["raw_probability"], pooled_params
        )
        target_payload["pooled_retrospective"] = {
            "parameters": pooled_params,
            "in_sample_diagnostics": calibration_diagnostics(
                labeled["target"].to_numpy(dtype=int),
                labeled["raw_probability"].to_numpy(dtype=float),
                pooled_calibrated,
            ),
            "game_crossfit_labeled_season_diagnostics": (
                calibration_diagnostics(
                    labeled["target"].to_numpy(dtype=int),
                    labeled["raw_probability"].to_numpy(dtype=float),
                    pooled_crossfit_probability,
                )
            ),
            "operating_point": operating_threshold(
                labeled["target"], pooled_crossfit_probability
            ),
        }
        artifact["targets"][target_name] = target_payload
    write_json(output_path, artifact)
    return artifact


def calibration_folds_for_season(
    artifact: Mapping[str, object],
    target_name: str,
    year: int,
) -> tuple[dict[int, Mapping[str, float]], dict[str, int], str, float]:
    target_payload = artifact["targets"][target_name]
    season_payload = target_payload["seasons"].get(str(year))
    if season_payload is None:
        raise ValueError(f"No labeled-season crossfit calibration for {year}")
    parameters_by_fold: dict[int, Mapping[str, float]] = {}
    game_to_fold: dict[str, int] = {}
    for fold_payload in season_payload["folds"]:
        fold = int(fold_payload["fold"])
        parameters_by_fold[fold] = fold_payload["parameters"]
        for prefixed_game_id in fold_payload["validation_game_ids"]:
            _, game_id = str(prefixed_game_id).split("::", 1)
            if game_id in game_to_fold:
                raise AssertionError(f"Game assigned to two calibration folds: {game_id}")
            game_to_fold[game_id] = fold
    threshold = float(
        target_payload["pooled_retrospective"]["operating_point"]["threshold"]
    )
    return (
        parameters_by_fold,
        game_to_fold,
        str(season_payload["base_probability_scope"]),
        threshold,
    )


def retrospective_calibration(
    artifact: Mapping[str, object],
    target_name: str,
) -> tuple[Mapping[str, float], str, float]:
    payload = artifact["targets"][target_name]["pooled_retrospective"]
    return (
        payload["parameters"],
        "retrospective_pooled_2025_2026_for_unlabeled_season",
        float(payload["operating_point"]["threshold"]),
    )


def load_model_bundles(models_dir: Path) -> dict[str, dict[str, object]]:
    bundles = {}
    feature_contract: list[str] | None = None
    for target_name in TARGETS:
        path = models_dir / f"{target_name}_historical_safe_model.joblib"
        if not path.exists():
            raise FileNotFoundError(path)
        bundle = joblib.load(path)
        columns = list(bundle["feature_columns"])
        if feature_contract is None:
            feature_contract = columns
        elif columns != feature_contract:
            raise ValueError("Transition model bundles have different feature contracts")
        bundles[target_name] = bundle
    assert feature_contract is not None
    if len(feature_contract) != 97:
        raise ValueError(
            f"Expected the frozen 97-feature contract, got {len(feature_contract)}"
        )
    return bundles


def reference_feature_statistics(
    raw_dir: Path,
    feature_columns: list[str],
) -> pd.DataFrame:
    paths = [
        raw_dir / "NBA25.parquet",
        raw_dir / "NBA25_PS.parquet",
    ]
    raw = prepare_sequence_frame(load_raw_pbp(paths, extra_columns=SHOT_ID_COLUMNS))
    shots = raw[raw["event_type"].isin([1, 2])].reset_index(drop=True)
    features = build_historical_feature_matrix(shots).reindex(
        columns=feature_columns, fill_value=0.0
    )
    rows = []
    for column in feature_columns:
        values = pd.to_numeric(features[column], errors="coerce")
        rows.append(
            {
                "feature": column,
                "reference_rows": int(len(values)),
                "reference_missing_rate": float(values.isna().mean()),
                "reference_mean": float(values.mean()) if values.notna().any() else None,
                "reference_std": float(values.std(ddof=0))
                if values.notna().any()
                else None,
                "reference_min": float(values.min()) if values.notna().any() else None,
                "reference_max": float(values.max()) if values.notna().any() else None,
            }
        )
    return pd.DataFrame(rows)


def feature_drift_rows(
    features: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    source_file: str,
    year: int,
    season_type: str,
) -> list[dict[str, object]]:
    ref = reference.set_index("feature")
    rows: list[dict[str, object]] = []
    for column in features.columns:
        values = pd.to_numeric(features[column], errors="coerce")
        reference_row = ref.loc[column]
        missing_rate = float(values.isna().mean())
        mean = float(values.mean()) if values.notna().any() else None
        std = float(values.std(ddof=0)) if values.notna().any() else None
        reference_mean = reference_row["reference_mean"]
        reference_std = reference_row["reference_std"]
        reference_constant_change = False
        if pd.notna(reference_mean) and mean is not None:
            safe_reference_std = (
                float(reference_std) if pd.notna(reference_std) else 0.0
            )
            if safe_reference_std >= 1e-6:
                standardized_shift = (
                    mean - float(reference_mean)
                ) / safe_reference_std
            else:
                standardized_shift = None
                reference_constant_change = not np.isclose(
                    mean, float(reference_mean), atol=1e-9, rtol=0.0
                )
        else:
            standardized_shift = None
        missing_delta = missing_rate - float(reference_row["reference_missing_rate"])
        severe = bool(
            abs(missing_delta) >= 0.25
            or reference_constant_change
            or (
                standardized_shift is not None
                and abs(float(standardized_shift)) >= 1.0
            )
        )
        rows.append(
            {
                "source_file": source_file,
                "season_end_year": year,
                "season_type": season_type,
                "era": era_label(year),
                "feature": column,
                "rows": int(len(values)),
                "missing_rate": missing_rate,
                "reference_missing_rate": float(
                    reference_row["reference_missing_rate"]
                ),
                "missing_rate_delta": missing_delta,
                "mean": mean,
                "reference_mean": (
                    float(reference_mean) if pd.notna(reference_mean) else None
                ),
                "std": std,
                "reference_std": (
                    float(reference_std) if pd.notna(reference_std) else None
                ),
                "standardized_mean_shift": standardized_shift,
                "reference_constant_change": reference_constant_change,
                "severe_drift_flag": severe,
            }
        )
    return rows


def calibrate_file_probabilities(
    *,
    artifact: Mapping[str, object],
    target_name: str,
    year: int,
    game_id: pd.Series,
    event_num: pd.Series,
    full_model_probability: np.ndarray,
    labeled_rows: pd.DataFrame,
) -> dict[str, object]:
    if year not in {2025, 2026}:
        params, calibration_scope, threshold = retrospective_calibration(
            artifact, target_name
        )
        calibrated = apply_logit_calibrator(full_model_probability, params)
        return {
            "raw_probability": full_model_probability,
            "calibrated_probability": calibrated,
            "calibration_scope": np.repeat(calibration_scope, len(game_id)),
            "base_probability_scope": np.repeat(
                "full_2025_trained_model_retrospective", len(game_id)
            ),
            "threshold": threshold,
            "target": pd.array([pd.NA] * len(game_id), dtype="boolean"),
        }

    parameters_by_fold, game_to_fold, expected_scope, threshold = (
        calibration_folds_for_season(artifact, target_name, year)
    )
    season_rows = labeled_rows[
        labeled_rows["calibration_season"].eq(year)
    ].copy()
    season_rows["game_id"] = season_rows["game_id"].astype(str)
    season_rows["event_num"] = pd.to_numeric(
        season_rows["event_num"], errors="coerce"
    ).astype("Int64")
    if season_rows.duplicated(["game_id", "event_num"]).any():
        raise ValueError(
            f"Duplicate labeled calibration keys for {target_name} {year}"
        )
    annotation = pd.DataFrame(
        {
            "game_id": game_id.astype(str).to_numpy(),
            "event_num": pd.to_numeric(event_num, errors="coerce").astype(
                "Int64"
            ),
        }
    ).merge(
        season_rows[
            [
                "game_id",
                "event_num",
                "target",
                "raw_probability",
                "base_probability_scope",
            ]
        ],
        on=["game_id", "event_num"],
        how="left",
        sort=False,
        validate="many_to_one",
    )
    raw_probability = np.asarray(full_model_probability, dtype=float).copy()
    matched = annotation["raw_probability"].notna().to_numpy()
    raw_probability[matched] = annotation.loc[
        matched, "raw_probability"
    ].to_numpy(dtype=float)
    base_scope = np.where(
        matched,
        annotation["base_probability_scope"].fillna(expected_scope),
        f"{year}_full_model_unlabeled_row_fallback",
    )

    available_folds = sorted(parameters_by_fold)
    if not available_folds:
        raise AssertionError(f"No calibration folds for {target_name} {year}")
    default_fold = available_folds[0]
    row_folds = (
        annotation["game_id"].map(game_to_fold).fillna(default_fold).astype(int)
    )
    calibrated = np.full(len(annotation), np.nan)
    for fold, indices in row_folds.groupby(row_folds).groups.items():
        row_index = np.asarray(list(indices), dtype=int)
        calibrated[row_index] = apply_logit_calibrator(
            raw_probability[row_index], parameters_by_fold[int(fold)]
        )
    if np.isnan(calibrated).any():
        raise AssertionError("Labeled-season calibration left rows unassigned")
    return {
        "raw_probability": raw_probability,
        "calibrated_probability": calibrated,
        "calibration_scope": np.asarray(
            [
                f"season_{year}_game_group_crossfit_fold_{fold}"
                for fold in row_folds
            ],
            dtype=object,
        ),
        "base_probability_scope": np.asarray(base_scope, dtype=object),
        "threshold": threshold,
        "target": pd.array(annotation["target"], dtype="boolean"),
    }


def score_raw_file(
    path: Path,
    *,
    bundles: Mapping[str, Mapping[str, object]],
    calibration: Mapping[str, object],
    labeled_rows_by_target: Mapping[str, pd.DataFrame],
    output_dir: Path,
    reference_stats: pd.DataFrame,
    overwrite: bool,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    year, season_type = parse_raw_filename(path)
    output_path = output_dir / f"transition_backfill_{year}_{season_type}.parquet"
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists; pass --overwrite to replace it"
        )

    extra_columns = SHOT_ID_COLUMNS + TARGETS
    available_columns = set(pq.ParquetFile(path).schema.names)
    present_extra = [column for column in extra_columns if column in available_columns]
    raw = prepare_sequence_frame(
        load_raw_pbp([path], extra_columns=present_extra)
    )
    shots = raw[raw["event_type"].isin([1, 2])].reset_index(drop=True)
    if shots.empty:
        raise ValueError(f"No FGA event_type 1/2 rows found in {path}")

    feature_columns = list(next(iter(bundles.values()))["feature_columns"])
    features = build_historical_feature_matrix(shots).reindex(
        columns=feature_columns, fill_value=0.0
    )
    drift = feature_drift_rows(
        features,
        reference_stats,
        source_file=path.name,
        year=year,
        season_type=season_type,
    )
    missing_count = features.isna().sum(axis=1).astype("int16")

    result = pd.DataFrame(
        {
            "season_end_year": year,
            "season_type": season_type,
            "source_file": path.name,
            "feature_contract_version": FEATURE_CONTRACT_VERSION,
            "game_id": shots["game_id"].astype(str),
            "event_num": shots["event_num"],
            "period": shots["period"],
            "clock_sec": shots["clock_sec"],
            "event_type": shots["event_type"],
            "event_action_type": shots["event_action_type"],
            "shot_distance_ft": features["shot_distance_ft"],
            "feature_missing_count": missing_count,
            "feature_missing_rate": missing_count / len(feature_columns),
        }
    )
    for column in SHOT_ID_COLUMNS:
        result[column] = shots[column] if column in shots.columns else pd.NA

    calibration_scopes: set[str] = set()
    base_probability_scopes: set[str] = set()
    for target_name, bundle in bundles.items():
        full_model_probability = bundle["model"].predict_proba(features)[:, 1]
        calibrated_payload = calibrate_file_probabilities(
            artifact=calibration,
            target_name=target_name,
            year=year,
            game_id=shots["game_id"],
            event_num=shots["event_num"],
            full_model_probability=full_model_probability,
            labeled_rows=labeled_rows_by_target[target_name],
        )
        raw_probability = np.asarray(
            calibrated_payload["raw_probability"], dtype=float
        )
        calibrated = np.asarray(
            calibrated_payload["calibrated_probability"], dtype=float
        )
        scopes = np.asarray(calibrated_payload["calibration_scope"], dtype=object)
        base_scopes = np.asarray(
            calibrated_payload["base_probability_scope"], dtype=object
        )
        threshold = float(calibrated_payload["threshold"])
        prefix = (
            "transition"
            if target_name == "is_transition"
            else "transition_rim"
        )
        result[f"{prefix}_probability_full_model"] = (
            full_model_probability.astype("float32")
        )
        result[f"{prefix}_probability_raw"] = raw_probability.astype("float32")
        result[f"{prefix}_probability"] = calibrated.astype("float32")
        result[f"{prefix}_label_050"] = calibrated >= 0.5
        result[f"{prefix}_label_operating"] = calibrated >= threshold
        result[f"{prefix}_operating_threshold"] = np.float32(threshold)
        result[f"{prefix}_calibration_scope"] = scopes
        result[f"{prefix}_base_probability_scope"] = base_scopes
        calibration_scopes.update(map(str, np.unique(scopes)))
        base_probability_scopes.update(map(str, np.unique(base_scopes)))
        artifact_target = pd.Series(
            calibrated_payload["target"], index=result.index, dtype="boolean"
        )
        if target_name in shots.columns:
            raw_target = shots[target_name].astype("boolean").reset_index(drop=True)
            artifact_target = artifact_target.fillna(raw_target)
        result[f"sq_{target_name}"] = artifact_target

    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    file_audit = {
        "source_file": path.name,
        "output_file": output_path.name,
        "season_end_year": year,
        "season_type": season_type,
        "era": era_label(year),
        "raw_rows": int(len(raw)),
        "shot_rows": int(len(result)),
        "games": int(result["game_id"].nunique()),
        "feature_count": int(len(feature_columns)),
        "mean_feature_missing_rate": float(result["feature_missing_rate"].mean()),
        "max_feature_missing_rate": float(result["feature_missing_rate"].max()),
        "severe_drift_features": int(
            sum(bool(row["severe_drift_flag"]) for row in drift)
        ),
        "transition_probability_mean": float(
            result["transition_probability"].mean()
        ),
        "transition_probability_min": float(
            result["transition_probability"].min()
        ),
        "transition_probability_max": float(
            result["transition_probability"].max()
        ),
        "transition_label_050_rate": float(
            result["transition_label_050"].mean()
        ),
        "transition_label_operating_rate": float(
            result["transition_label_operating"].mean()
        ),
        "transition_rim_probability_mean": float(
            result["transition_rim_probability"].mean()
        ),
        "transition_rim_label_050_rate": float(
            result["transition_rim_label_050"].mean()
        ),
        "transition_rim_label_operating_rate": float(
            result["transition_rim_label_operating"].mean()
        ),
        "calibration_scopes": "|".join(sorted(calibration_scopes)),
        "base_probability_scopes": "|".join(
            sorted(base_probability_scopes)
        ),
    }
    return file_audit, drift


def build_era_audit(file_audit: pd.DataFrame) -> pd.DataFrame:
    weighted_columns = [
        "mean_feature_missing_rate",
        "transition_probability_mean",
        "transition_label_050_rate",
        "transition_label_operating_rate",
        "transition_rim_probability_mean",
        "transition_rim_label_050_rate",
        "transition_rim_label_operating_rate",
    ]
    rows = []
    for (era, season_type), group in file_audit.groupby(
        ["era", "season_type"], sort=True
    ):
        weights = group["shot_rows"].to_numpy(dtype=float)
        row = {
            "era": era,
            "season_type": season_type,
            "files": int(len(group)),
            "seasons": int(group["season_end_year"].nunique()),
            "shot_rows": int(group["shot_rows"].sum()),
            "games": int(group["games"].sum()),
            "severe_drift_feature_file_pairs": int(
                group["severe_drift_features"].sum()
            ),
        }
        for column in weighted_columns:
            row[column] = float(
                np.average(group[column].to_numpy(dtype=float), weights=weights)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=1997)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument(
        "--season-types", choices=["rs", "ps", "all"], default="all"
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--rebuild-calibration",
        action="store_true",
        help="Refit labeled-season calibration even if the JSON already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.start_year > args.end_year:
        raise ValueError("--start-year must be <= --end-year")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bundles = load_model_bundles(args.models_dir)
    labeled_rows_by_target = {
        target_name: load_labeled_calibration_rows(
            args.models_dir, target_name
        )
        for target_name in TARGETS
    }
    feature_columns = list(next(iter(bundles.values()))["feature_columns"])
    calibration_path = args.output_dir / "transition_calibration.json"
    if args.rebuild_calibration or not calibration_path.exists():
        calibration = build_calibration_artifact(
            args.models_dir, calibration_path
        )
    else:
        calibration = json.loads(calibration_path.read_text())
    if calibration.get("version") != CALIBRATION_VERSION:
        raise ValueError(
            f"Unsupported calibration version: {calibration.get('version')}"
        )

    reference_path = args.output_dir / "transition_feature_reference_2025.csv"
    reference = reference_feature_statistics(args.raw_dir, feature_columns)
    reference.to_csv(reference_path, index=False)

    raw_paths = discover_raw_paths(
        args.raw_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        season_types=args.season_types,
    )
    file_rows: list[dict[str, object]] = []
    drift_rows: list[dict[str, object]] = []
    for index, path in enumerate(raw_paths, start=1):
        LOGGER.info("Scoring %s (%s/%s)", path.name, index, len(raw_paths))
        file_audit, drift = score_raw_file(
            path,
            bundles=bundles,
            calibration=calibration,
            labeled_rows_by_target=labeled_rows_by_target,
            output_dir=args.output_dir,
            reference_stats=reference,
            overwrite=args.overwrite,
        )
        file_rows.append(file_audit)
        drift_rows.extend(drift)

    file_audit = pd.DataFrame(file_rows).sort_values(
        ["season_end_year", "season_type"]
    )
    drift_audit = pd.DataFrame(drift_rows).sort_values(
        ["season_end_year", "season_type", "feature"]
    )
    era_audit = build_era_audit(file_audit)
    file_audit.to_csv(
        args.output_dir / "transition_backfill_file_audit.csv", index=False
    )
    drift_audit.to_csv(
        args.output_dir / "transition_backfill_feature_drift.csv", index=False
    )
    era_audit.to_csv(
        args.output_dir / "transition_backfill_era_audit.csv", index=False
    )
    summary = {
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "calibration_version": CALIBRATION_VERSION,
        "start_year": args.start_year,
        "end_year": args.end_year,
        "season_types": args.season_types,
        "files": int(len(file_audit)),
        "shot_rows": int(file_audit["shot_rows"].sum()),
        "games_file_sum": int(file_audit["games"].sum()),
        "output_dir": str(args.output_dir),
        "calibration_path": str(calibration_path),
        "reference_path": str(reference_path),
        "file_audit_path": str(
            args.output_dir / "transition_backfill_file_audit.csv"
        ),
        "feature_drift_path": str(
            args.output_dir / "transition_backfill_feature_drift.csv"
        ),
        "era_audit_path": str(
            args.output_dir / "transition_backfill_era_audit.csv"
        ),
    }
    write_json(args.output_dir / "transition_backfill_summary.json", summary)
    LOGGER.info(
        "Completed %s files and %s shot rows",
        len(file_audit),
        f"{file_audit['shot_rows'].sum():,}",
    )


if __name__ == "__main__":
    main()
