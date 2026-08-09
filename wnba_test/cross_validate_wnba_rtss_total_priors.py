#!/usr/bin/env python3
"""Test WNBA RTSS features against low-shrinkage annual ORAPM/DRAPM targets.

This is deliberately a research lane.  It builds pure annual RAPM targets from
the clean FIRST_CHANCE + SECOND_CHANCE_CLEAN possession surface, varies the
target solve penalties, and then uses symmetric leave-one-season-out CV over
2023-2025 to compare RTSS-only, box-only, and combined player-season priors.

Feature importance is measured on held-out seasons.  The script reports
individual permutation importance, grouped mechanism-family importance, and
the sign stability of standardized ridge coefficients.  It does not select
features from an in-sample coefficient ranking.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wnba_test.build_wnba_box_priors import (
    TARGET_PREFIXES,
    feature_columns,
    load_box_features,
)
from wnba_test.clean_decomp_wnba import (
    _load_clean_total_df,
    _load_name_map,
    _prepare_direct_ridge_problem,
    _solve_direct_ridge_problem,
)
from wnba_test.cross_validate_wnba_rtss_priors import (
    _season_standardize_rtss,
    _weighted_metrics,
)


YEARS = (2023, 2024, 2025)
TARGET_ALPHA_SPECS: tuple[tuple[str, float, float], ...] = (
    # Orthogonal sweeps: do not attribute a change to one side while moving
    # both penalties in the coupled offense/defense solve.
    ("off25_def1000", 25.0, 1000.0),
    ("off50_def1000", 50.0, 1000.0),
    ("off100_def1000", 100.0, 1000.0),
    ("off250_def1000", 250.0, 1000.0),
    ("off500_def1000", 500.0, 1000.0),
    ("off750_def1000", 750.0, 1000.0),
    ("off750_def50", 750.0, 50.0),
    ("off750_def100", 750.0, 100.0),
    ("off750_def250", 750.0, 250.0),
    ("off750_def500", 750.0, 500.0),
)
MODEL_ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
SELECTION_ALPHA = 100.0
MAX_COMPACT_RTSS_FEATURES = 8
DEFAULT_FEATURES = (
    ROOT / "wnba_test" / "Results" / "rtss_prior_research" / "player_season_features.csv"
)
DEFAULT_OUTPUT = ROOT / "wnba_test" / "Results" / "rtss_total_prior_cv"


def target_path(output_dir: Path, label: str, year: int) -> Path:
    return output_dir / "annual_targets" / label / f"{year}.csv"


def build_targets(output_dir: Path, rebuild: bool) -> None:
    """Build each annual clean-total target once, then reuse its design grid."""
    name_map = _load_name_map()
    for year in YEARS:
        needed = [
            (label, off_alpha, def_alpha)
            for label, off_alpha, def_alpha in TARGET_ALPHA_SPECS
            if rebuild or not target_path(output_dir, label, year).exists()
        ]
        if not needed:
            continue
        logging.info("Preparing %s WNBA clean-total RAPM design", year)
        frame = _load_clean_total_df([year % 100], "ALL")
        problem = _prepare_direct_ridge_problem(frame, time_decay_days=None)
        for label, off_alpha, def_alpha in needed:
            logging.info(
                "Solving %s target with offense alpha %.0f / defense alpha %.0f",
                year,
                off_alpha,
                def_alpha,
            )
            result = _solve_direct_ridge_problem(
                problem,
                name_map,
                off_alpha=off_alpha,
                def_alpha=def_alpha,
            )
            path = target_path(output_dir, label, year)
            path.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(path, index=False)


def _normalize_ids(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["player_id"] = (
        pd.to_numeric(frame["player_id"], errors="coerce")
        .astype("Int64")
        .astype(str)
    )
    return frame.loc[frame["player_id"].ne("<NA>")]


def load_target_rows(output_dir: Path, label: str, year: int, side: str) -> pd.DataFrame:
    target = _normalize_ids(pd.read_csv(target_path(output_dir, label, year)))
    value_column = side
    poss_column = f"{side}_poss"
    target_value = pd.to_numeric(target[value_column], errors="coerce")
    if side == "def":
        # The solver stores lower-is-better defense.  Positive-good targets make
        # coefficient and importance interpretation consistent with offense.
        target_value = -target_value
    return pd.DataFrame(
        {
            "player_id": target["player_id"],
            "target": target_value,
            "target_poss": pd.to_numeric(target[poss_column], errors="coerce"),
        }
    ).dropna()


def box_features_for_side(side: str) -> list[str]:
    return list(
        dict.fromkeys(
            feature
            for prefix in TARGET_PREFIXES
            for feature in feature_columns(prefix, side)
        )
    )


def rtss_features_for_side(columns: Iterable[str], side: str) -> list[str]:
    fields = [column for column in columns if column.startswith("z_rtss_")]
    defense_prefixes = (
        "z_rtss_def_",
        "z_rtss_drive_help",
        "z_rtss_pass_deflection",
    )
    if side == "def":
        return [column for column in fields if column.startswith(defense_prefixes)]
    return [column for column in fields if not column.startswith(defense_prefixes)]


def rtss_feature_family(feature: str, side: str) -> str:
    """Assign each RTSS field to one interpretable, non-overlapping family."""
    name = feature.removeprefix("z_")
    if side == "def":
        if "qsq" in name or "qsp" in name:
            return "shot_quality_allowed"
        if any(token in name for token in ("contest", "closeout", "rearview", "closest")):
            return "contest_geometry"
        if any(token in name for token in ("matchup", "rotation", "drive_help", "deflection")):
            return "help_rotation_activity"
        if "rb_opp" in name:
            return "rebounding_position"
        return "other_defense"

    if any(token in name for token in ("qsq", "qsp", "shooter_adjustment")):
        return "shot_quality_and_talent"
    if "fg_pct" in name:
        return "realized_finishing"
    if any(token in name for token in ("pass_", "potential_assist")):
        return "playmaking"
    if any(token in name for token in ("pick", "screen", "handoff")):
        return "screening_offball"
    if any(token in name for token in ("crash", "rb_opp")):
        return "rebounding_position"
    if any(
        token in name
        for token in (
            "self_created",
            "assist_opp",
            "catch_shoot",
            "drive",
            "iso_post",
            "on_ball",
            "touch_",
        )
    ):
        return "creation_role"
    if name.endswith("three_share") or name.endswith("mid_share"):
        return "shot_profile"
    return "other_offense"


def _fit(rows: pd.DataFrame, features: list[str], alpha: float) -> Pipeline:
    model = Pipeline(
        [("scale", StandardScaler()), ("ridge", Ridge(alpha=float(alpha)))]
    )
    weight = np.sqrt(rows["target_poss"].clip(lower=1.0, upper=2500.0))
    model.fit(
        rows[features],
        rows["target"],
        ridge__sample_weight=(weight / weight.mean()).to_numpy(),
    )
    return model


def select_alpha(
    rows_by_year: dict[int, pd.DataFrame],
    years: list[int],
    features: list[str],
    tolerance: float = 0.005,
) -> tuple[float, pd.DataFrame]:
    """Inner season CV; prefer stronger shrinkage within 0.5% of min RMSE."""
    results: list[dict[str, float]] = []
    for alpha in MODEL_ALPHAS:
        actual: list[float] = []
        prediction: list[float] = []
        weight: list[float] = []
        for heldout in years:
            train_years = [year for year in years if year != heldout]
            train = pd.concat([rows_by_year[year] for year in train_years], ignore_index=True)
            validation = rows_by_year[heldout]
            model = _fit(train, features, alpha)
            actual.extend(validation["target"].tolist())
            prediction.extend(model.predict(validation[features]).tolist())
            weight.extend(np.sqrt(validation["target_poss"]).tolist())
        metrics = _weighted_metrics(
            np.asarray(actual), np.asarray(prediction), np.asarray(weight)
        )
        results.append({"alpha": float(alpha), "rmse": metrics["rmse"]})
    table = pd.DataFrame(results)
    threshold = float(table["rmse"].min()) * (1.0 + tolerance)
    selected = float(table.loc[table["rmse"] <= threshold, "alpha"].max())
    return selected, table


def _available_features(rows_by_year: dict[int, pd.DataFrame], candidates: list[str]) -> list[str]:
    common = set.intersection(*(set(frame.columns) for frame in rows_by_year.values()))
    available = [feature for feature in candidates if feature in common]
    # Drop fields that are constant across every development season.
    return [
        feature
        for feature in available
        if any(rows_by_year[year][feature].std(ddof=0) > 1e-10 for year in YEARS)
    ]


def _cross_validated_rmse(
    rows_by_year: dict[int, pd.DataFrame],
    years: list[int],
    features: list[str],
    alpha: float,
) -> tuple[float, dict[int, float]]:
    """Score a fixed feature set across seasons without reusing validation rows."""
    actual: list[float] = []
    prediction: list[float] = []
    weight: list[float] = []
    fold_rmse: dict[int, float] = {}
    for heldout in years:
        train_years = [year for year in years if year != heldout]
        train = pd.concat([rows_by_year[year] for year in train_years], ignore_index=True)
        validation = rows_by_year[heldout]
        validation_weight = np.sqrt(validation["target_poss"].to_numpy())
        if features:
            model = _fit(train, features, alpha)
            fold_prediction = model.predict(validation[features])
        else:
            train_weight = np.sqrt(train["target_poss"].clip(lower=1.0, upper=2500.0))
            fold_prediction = np.repeat(
                np.average(train["target"], weights=train_weight), len(validation)
            )
        fold_rmse[heldout] = _weighted_metrics(
            validation["target"].to_numpy(), fold_prediction, validation_weight
        )["rmse"]
        actual.extend(validation["target"].tolist())
        prediction.extend(fold_prediction.tolist())
        weight.extend(validation_weight.tolist())
    pooled = _weighted_metrics(
        np.asarray(actual), np.asarray(prediction), np.asarray(weight)
    )["rmse"]
    return pooled, fold_rmse


def select_compact_rtss_features(
    rows_by_year: dict[int, pd.DataFrame],
    years: list[int],
    base_features: list[str],
    rtss_candidates: list[str],
    max_features: int = MAX_COMPACT_RTSS_FEATURES,
    minimum_relative_gain: float = 0.001,
) -> tuple[list[str], pd.DataFrame]:
    """Nested forward selection requiring improvement in every inner season.

    A candidate must improve the current feature set in both inner held-out
    seasons and lower pooled RMSE by at least 0.1%.  This deliberately rejects
    features whose apparent value comes from only one season.
    """
    selected: list[str] = []
    remaining = list(rtss_candidates)
    history: list[dict[str, Any]] = []
    current_rmse, current_folds = _cross_validated_rmse(
        rows_by_year, years, base_features, SELECTION_ALPHA
    )
    for step in range(1, max_features + 1):
        candidates: list[dict[str, Any]] = []
        for feature in remaining:
            trial_features = base_features + selected + [feature]
            trial_rmse, trial_folds = _cross_validated_rmse(
                rows_by_year, years, trial_features, SELECTION_ALPHA
            )
            fold_wins = all(
                trial_folds[year] < current_folds[year] for year in years
            )
            relative_gain = (current_rmse - trial_rmse) / current_rmse
            candidates.append(
                {
                    "step": step,
                    "feature": feature,
                    "trial_rmse": trial_rmse,
                    "relative_gain": relative_gain,
                    "fold_wins": int(fold_wins),
                    "accepted": 0,
                    "fold_rmse": trial_folds,
                }
            )
        eligible = [
            row
            for row in candidates
            if row["fold_wins"] and row["relative_gain"] >= minimum_relative_gain
        ]
        if not eligible:
            history.extend(candidates)
            break
        winner = min(eligible, key=lambda row: (row["trial_rmse"], row["feature"]))
        winner["accepted"] = 1
        history.extend(candidates)
        selected.append(winner["feature"])
        remaining.remove(winner["feature"])
        current_rmse = float(winner["trial_rmse"])
        current_folds = dict(winner["fold_rmse"])
    history_frame = pd.DataFrame(history)
    if not history_frame.empty:
        history_frame["fold_rmse"] = history_frame["fold_rmse"].map(
            lambda value: json.dumps(value, sort_keys=True)
        )
    return selected, history_frame


def _permutation_delta(
    model: Pipeline,
    validation: pd.DataFrame,
    features: list[str],
    shuffled_features: list[str],
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    weight = np.sqrt(validation["target_poss"].to_numpy())
    base = _weighted_metrics(
        validation["target"].to_numpy(),
        model.predict(validation[features]),
        weight,
    )["rmse"]
    deltas: list[float] = []
    rng = np.random.default_rng(seed)
    for _ in range(repeats):
        shuffled = validation[features].copy()
        order = rng.permutation(len(shuffled))
        shuffled.loc[:, shuffled_features] = shuffled[shuffled_features].to_numpy()[order]
        rmse = _weighted_metrics(
            validation["target"].to_numpy(), model.predict(shuffled), weight
        )["rmse"]
        deltas.append(float(rmse - base))
    return float(np.mean(deltas)), float(np.std(deltas, ddof=1) if repeats > 1 else 0.0)


def _target_reliability(
    rows_by_year: dict[int, pd.DataFrame], label: str, side: str
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    pooled = pd.concat(rows_by_year.values(), ignore_index=True)
    weights = np.sqrt(pooled["target_poss"].to_numpy())
    mean = np.average(pooled["target"], weights=weights)
    sd = np.sqrt(np.average(np.square(pooled["target"] - mean), weights=weights))
    output.append(
        {
            "target_spec": label,
            "side": side,
            "comparison": "pooled",
            "rows": len(pooled),
            "target_sd": float(sd),
            "returning_player_correlation": np.nan,
        }
    )
    for first, second in zip(YEARS[:-1], YEARS[1:]):
        paired = rows_by_year[first][["player_id", "target"]].merge(
            rows_by_year[second][["player_id", "target"]],
            on="player_id",
            suffixes=("_first", "_second"),
        )
        corr = paired["target_first"].corr(paired["target_second"])
        output.append(
            {
                "target_spec": label,
                "side": side,
                "comparison": f"{first}_to_{second}",
                "rows": len(paired),
                "target_sd": np.nan,
                "returning_player_correlation": float(corr),
            }
        )
    return output


def cross_validate(
    feature_path: Path,
    output_dir: Path,
    min_target_poss: float,
    min_rtss_chances: float,
    permutation_repeats: int,
) -> dict[str, pd.DataFrame]:
    rtss = _normalize_ids(pd.read_csv(feature_path, dtype={"player_id": str}))
    rtss = _season_standardize_rtss(rtss)
    box_by_year = {year: _normalize_ids(load_box_features(year)) for year in YEARS}

    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    feature_importance_rows: list[dict[str, Any]] = []
    family_importance_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    full_fit_rows: list[dict[str, Any]] = []

    for label, target_off_alpha, target_def_alpha in TARGET_ALPHA_SPECS:
        for side in ("off", "def"):
            rows_by_year: dict[int, pd.DataFrame] = {}
            chance_column = f"{side}_chances"
            for year in YEARS:
                target = load_target_rows(output_dir, label, year, side)
                merged = target.merge(box_by_year[year], on="player_id", how="inner")
                merged = merged.merge(
                    rtss.loc[rtss["season"].eq(year)],
                    on="player_id",
                    how="inner",
                    suffixes=("", "_rtss"),
                )
                rows_by_year[year] = merged.loc[
                    merged["target_poss"].ge(min_target_poss)
                    & merged[chance_column].ge(min_rtss_chances)
                ].copy()

            reliability_rows.extend(_target_reliability(rows_by_year, label, side))
            box_features = _available_features(rows_by_year, box_features_for_side(side))
            rtss_features = _available_features(
                rows_by_year,
                rtss_features_for_side(rows_by_year[YEARS[0]].columns, side),
            )
            for heldout_year in YEARS:
                train_years = [year for year in YEARS if year != heldout_year]
                train = pd.concat([rows_by_year[year] for year in train_years], ignore_index=True)
                validation = rows_by_year[heldout_year].copy()
                rtss_compact, rtss_selection = select_compact_rtss_features(
                    rows_by_year,
                    train_years,
                    base_features=[],
                    rtss_candidates=rtss_features,
                )
                combined_compact, combined_selection = select_compact_rtss_features(
                    rows_by_year,
                    train_years,
                    base_features=box_features,
                    rtss_candidates=rtss_features,
                )
                for selection_lane, selected, history in (
                    ("rtss_compact", rtss_compact, rtss_selection),
                    ("box_plus_rtss_compact", combined_compact, combined_selection),
                ):
                    if not history.empty:
                        for record in history.to_dict("records"):
                            selection_rows.append(
                                {
                                    "target_spec": label,
                                    "side": side,
                                    "outer_heldout_year": heldout_year,
                                    "lane": selection_lane,
                                    **record,
                                }
                            )
                lanes = {
                    "rtss_only": rtss_features,
                    "rtss_compact": rtss_compact or ["intercept_only"],
                    "box_only": box_features,
                    "box_plus_rtss": box_features + rtss_features,
                    "box_plus_rtss_compact": box_features + combined_compact,
                }
                validation["intercept_only"] = 0.0
                train["intercept_only"] = 0.0
                for year in YEARS:
                    rows_by_year[year]["intercept_only"] = 0.0
                for lane, features in lanes.items():
                    selected_alpha, _ = select_alpha(rows_by_year, train_years, features)
                    model = _fit(train, features, selected_alpha)
                    prediction = model.predict(validation[features])
                    weight = np.sqrt(validation["target_poss"].to_numpy())
                    metrics = _weighted_metrics(
                        validation["target"].to_numpy(), prediction, weight
                    )
                    fold_rows.append(
                        {
                            "target_spec": label,
                            "target_off_alpha": target_off_alpha,
                            "target_def_alpha": target_def_alpha,
                            "side": side,
                            "lane": lane,
                            "heldout_year": heldout_year,
                            "training_years": "|".join(map(str, train_years)),
                            "rows": len(validation),
                            "feature_count": len(features),
                            "model_alpha": selected_alpha,
                            **metrics,
                        }
                    )
                    for position, (_, row) in enumerate(validation.iterrows()):
                        prediction_rows.append(
                            {
                                "target_spec": label,
                                "side": side,
                                "lane": lane,
                                "heldout_year": heldout_year,
                                "player_id": row["player_id"],
                                "player_name": row.get("player_name", row.get("player_name_rtss", "")),
                                "target": float(row["target"]),
                                "target_poss": float(row["target_poss"]),
                                "prediction": float(prediction[position]),
                            }
                        )

                    if lane not in {
                        "rtss_only",
                        "rtss_compact",
                        "box_plus_rtss",
                        "box_plus_rtss_compact",
                    }:
                        continue
                    lane_rtss_features = [
                        feature for feature in features if feature in rtss_features
                    ]
                    coefficients = model.named_steps["ridge"].coef_
                    for feature in lane_rtss_features:
                        coefficient_rows.append(
                            {
                                "target_spec": label,
                                "side": side,
                                "lane": lane,
                                "heldout_year": heldout_year,
                                "feature": feature.removeprefix("z_"),
                                "family": rtss_feature_family(feature, side),
                                "standardized_coefficient": float(
                                    coefficients[features.index(feature)]
                                ),
                            }
                        )
                        delta, delta_sd = _permutation_delta(
                            model,
                            validation,
                            features,
                            [feature],
                            permutation_repeats,
                            seed=heldout_year * 1009 + features.index(feature),
                        )
                        feature_importance_rows.append(
                            {
                                "target_spec": label,
                                "side": side,
                                "lane": lane,
                                "heldout_year": heldout_year,
                                "feature": feature.removeprefix("z_"),
                                "family": rtss_feature_family(feature, side),
                                "permutation_delta_rmse": delta,
                                "permutation_delta_rmse_sd": delta_sd,
                            }
                        )
                    lane_families: dict[str, list[str]] = {}
                    for feature in lane_rtss_features:
                        lane_families.setdefault(
                            rtss_feature_family(feature, side), []
                        ).append(feature)
                    for family, family_features in lane_families.items():
                        delta, delta_sd = _permutation_delta(
                            model,
                            validation,
                            features,
                            family_features,
                            permutation_repeats,
                            seed=heldout_year * 9173 + sum(map(ord, family)),
                        )
                        family_importance_rows.append(
                            {
                                "target_spec": label,
                                "side": side,
                                "lane": lane,
                                "heldout_year": heldout_year,
                                "family": family,
                                "feature_count": len(family_features),
                                "permutation_delta_rmse": delta,
                                "permutation_delta_rmse_sd": delta_sd,
                            }
                        )

            full_alpha, _ = select_alpha(rows_by_year, list(YEARS), rtss_features)
            full_rows = pd.concat(rows_by_year.values(), ignore_index=True)
            full_model = _fit(full_rows, rtss_features, full_alpha)
            scaler = full_model.named_steps["scale"]
            ridge = full_model.named_steps["ridge"]
            raw_intercept = float(
                ridge.intercept_
                - np.sum(ridge.coef_ * scaler.mean_ / scaler.scale_)
            )
            for position, feature in enumerate(rtss_features):
                full_fit_rows.append(
                    {
                        "target_spec": label,
                        "side": side,
                        "status": (
                            "exploratory_offense_candidate"
                            if label == "off250_def1000" and side == "off"
                            else "rejected_defense_diagnostic"
                            if label == "off750_def1000" and side == "def"
                            else "alpha_sensitivity_model"
                        ),
                        "model_alpha": full_alpha,
                        "feature": feature.removeprefix("z_"),
                        "family": rtss_feature_family(feature, side),
                        "pipeline_input_mean": float(scaler.mean_[position]),
                        "pipeline_input_scale": float(scaler.scale_[position]),
                        "standardized_coefficient": float(ridge.coef_[position]),
                        "raw_z_input_coefficient": float(
                            ridge.coef_[position] / scaler.scale_[position]
                        ),
                        "raw_z_input_intercept": raw_intercept,
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    summary_rows: list[dict[str, Any]] = []
    for (label, side, lane), group in predictions.groupby(["target_spec", "side", "lane"]):
        metrics = _weighted_metrics(
            group["target"].to_numpy(),
            group["prediction"].to_numpy(),
            np.sqrt(group["target_poss"].to_numpy()),
        )
        summary_rows.append(
            {
                "target_spec": label,
                "side": side,
                "lane": lane,
                "rows": len(group),
                **metrics,
            }
        )

    coefficients = pd.DataFrame(coefficient_rows)
    candidate_oof = pd.concat(
        [
            predictions.loc[
                predictions["target_spec"].eq("off250_def1000")
                & predictions["side"].eq("off")
                & predictions["lane"].eq("rtss_only")
            ].assign(status="exploratory_offense_candidate"),
            predictions.loc[
                predictions["target_spec"].eq("off750_def1000")
                & predictions["side"].eq("def")
                & predictions["lane"].eq("rtss_only")
            ].assign(status="rejected_defense_diagnostic"),
        ],
        ignore_index=True,
    )
    candidate_oof["solver_prior_value"] = np.where(
        candidate_oof["side"].eq("def"),
        -candidate_oof["prediction"],
        candidate_oof["prediction"],
    )
    candidate_oof = candidate_oof.rename(
        columns={"heldout_year": "season", "prediction": "positive_good_prediction"}
    )
    coefficient_summary: list[dict[str, Any]] = []
    for (label, side, lane, feature, family), group in coefficients.groupby(
        ["target_spec", "side", "lane", "feature", "family"]
    ):
        values = group["standardized_coefficient"].to_numpy()
        nonzero = values[np.abs(values) > 1e-12]
        sign_stability = (
            max(np.mean(nonzero > 0), np.mean(nonzero < 0)) if len(nonzero) else 0.0
        )
        coefficient_summary.append(
            {
                "target_spec": label,
                "side": side,
                "lane": lane,
                "feature": feature,
                "family": family,
                "mean_standardized_coefficient": float(np.mean(values)),
                "mean_abs_standardized_coefficient": float(np.mean(np.abs(values))),
                "sign_stability": float(sign_stability),
                "folds": len(values),
            }
        )

    return {
        "fold_metrics": pd.DataFrame(fold_rows),
        "predictions": predictions,
        "model_summary": pd.DataFrame(summary_rows),
        "target_reliability": pd.DataFrame(reliability_rows),
        "coefficients": coefficients,
        "coefficient_summary": pd.DataFrame(coefficient_summary),
        "feature_importance": pd.DataFrame(feature_importance_rows),
        "family_importance": pd.DataFrame(family_importance_rows),
        "selection_history": pd.DataFrame(selection_rows),
        "full_fit_coefficients": pd.DataFrame(full_fit_rows),
        "candidate_oof_priors": candidate_oof,
    }


def build_report(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    summary = tables["model_summary"]
    model_pivot = summary.pivot_table(
        index=["target_spec", "side"], columns="lane", values=["rmse", "r2", "correlation"]
    )
    comparisons: list[dict[str, Any]] = []
    for target_spec, side in model_pivot.index:
        row = model_pivot.loc[(target_spec, side)]
        comparisons.append(
            {
                "target_spec": target_spec,
                "side": side,
                "rtss_only_rmse": float(row[("rmse", "rtss_only")]),
                "rtss_only_r2": float(row[("r2", "rtss_only")]),
                "rtss_only_correlation": float(row[("correlation", "rtss_only")]),
                "rtss_compact_rmse": float(row[("rmse", "rtss_compact")]),
                "rtss_compact_r2": float(row[("r2", "rtss_compact")]),
                "rtss_compact_correlation": float(row[("correlation", "rtss_compact")]),
                "box_rmse": float(row[("rmse", "box_only")]),
                "combined_rmse": float(row[("rmse", "box_plus_rtss")]),
                "combined_r2": float(row[("r2", "box_plus_rtss")]),
                "combined_correlation": float(row[("correlation", "box_plus_rtss")]),
                "combined_vs_box_rmse_pct": float(
                    100.0
                    * (row[("rmse", "box_plus_rtss")] - row[("rmse", "box_only")])
                    / row[("rmse", "box_only")]
                ),
                "compact_combined_rmse": float(
                    row[("rmse", "box_plus_rtss_compact")]
                ),
                "compact_combined_r2": float(
                    row[("r2", "box_plus_rtss_compact")]
                ),
                "compact_combined_correlation": float(
                    row[("correlation", "box_plus_rtss_compact")]
                ),
                "compact_combined_vs_box_rmse_pct": float(
                    100.0
                    * (
                        row[("rmse", "box_plus_rtss_compact")]
                        - row[("rmse", "box_only")]
                    )
                    / row[("rmse", "box_only")]
                ),
            }
        )

    importance = tables["feature_importance"].loc[
        tables["feature_importance"]["lane"].eq("rtss_only")
    ]
    feature_rank = (
        importance.groupby(["target_spec", "side", "feature", "family"], as_index=False)
        .agg(
            mean_delta_rmse=("permutation_delta_rmse", "mean"),
            folds_positive=("permutation_delta_rmse", lambda x: int((x > 0).sum())),
        )
        .sort_values(["target_spec", "side", "mean_delta_rmse"], ascending=[True, True, False])
    )
    top_features: dict[str, list[dict[str, Any]]] = {}
    for (target_spec, side), group in feature_rank.groupby(["target_spec", "side"]):
        top_features[f"{target_spec}:{side}"] = group.head(12).to_dict("records")

    return {
        "status": "research_only",
        "years": list(YEARS),
        "cv_design": "symmetric leave-one-season-out; all three seasons are development data",
        "target_definition": "annual clean FIRST_CHANCE + SECOND_CHANCE_CLEAN pure ORAPM/DRAPM",
        "defense_convention_in_models": "positive is good; saved annual target CSV retains raw solver defense",
        "target_alpha_specs": [
            {"label": label, "off_alpha": off, "def_alpha": defense}
            for label, off, defense in TARGET_ALPHA_SPECS
        ],
        "model_comparisons": comparisons,
        "top_rtss_only_features_by_permutation": top_features,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rebuild-targets", action="store_true")
    parser.add_argument("--min-target-poss", type=float, default=250.0)
    parser.add_argument("--min-rtss-chances", type=float, default=100.0)
    parser.add_argument("--permutation-repeats", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    build_targets(args.output_dir, args.rebuild_targets)
    tables = cross_validate(
        args.feature_path,
        args.output_dir,
        args.min_target_poss,
        args.min_rtss_chances,
        args.permutation_repeats,
    )
    for name, table in tables.items():
        table.to_csv(args.output_dir / f"{name}.csv", index=False)
    report = build_report(tables)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    logging.info("Wrote WNBA total-prior CV outputs to %s", args.output_dir)


if __name__ == "__main__":
    main()
