#!/usr/bin/env python3
"""Three-season cross-validation for WNBA RTSS component-prior additions.

The box backbone uses every non-held-out box season, including 2022. RTSS is
fit only as a residual correction to cross-fitted box predictions. The outer
fold leaves out one of 2023, 2024, or 2025; the two remaining RTSS seasons tune
and fit the residual layer. This is symmetric model-development CV, not a
historical as-of simulation or fresh confirmation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wnba_test.build_wnba_box_priors import (
    DEFAULT_OUTPUT as BOX_PRIOR_OUTPUT,
    RIDGE_ALPHAS as BOX_ALPHAS,
    TARGET_PREFIXES,
    feature_columns,
    load_box_features,
    load_target,
)


DEFAULT_FEATURES = (
    ROOT / "wnba_test" / "Results" / "rtss_prior_research" / "player_season_features.csv"
)
DEFAULT_OUTPUT = (
    ROOT / "wnba_test" / "Results" / "rtss_prior_cross_validation"
)
RTSS_YEARS = (2023, 2024, 2025)
BOX_YEARS = (2022, 2023, 2024, 2025)
RESIDUAL_ALPHAS = (1.0, 10.0, 100.0, 1000.0)


CREATION_FEATURES: dict[tuple[str, str], list[str]] = {
    ("ALT_EFG_RIM_FREQ", "off"): [
        "rtss_rim_self_created_rate100",
        "rtss_rim_assist_opp_share",
        "rtss_drive_rate100",
        "rtss_potential_assist_received_rate100",
    ],
    ("ALT_EFG_RIM_FG", "off"): [
        "rtss_rim_assist_opp_share",
        "rtss_after_pass_self_created_share",
        "rtss_drive_blowby_share",
        "rtss_potential_assist_received_rate100",
    ],
    ("ALT_EFG_MID_FREQ", "off"): [
        "rtss_mid_self_created_rate100",
        "rtss_mid_assist_opp_share",
        "rtss_touch_dribbles",
        "rtss_iso_post_rate100",
    ],
    ("ALT_EFG_MID_FG", "off"): [
        "rtss_mid_assist_opp_share",
        "rtss_after_pass_self_created_share",
        "rtss_catch_shoot_share",
        "rtss_touch_dribbles",
    ],
    ("ALT_EFG_THREE_FREQ", "off"): [
        "rtss_three_self_created_rate100",
        "rtss_three_assist_opp_share",
        "rtss_potential_assist_received_rate100",
        "rtss_offball_screen_rate100",
    ],
    ("ALT_EFG_THREE_FG", "off"): [
        "rtss_three_assist_opp_share",
        "rtss_after_pass_self_created_share",
        "rtss_catch_shoot_share",
        "rtss_potential_assist_received_rate100",
    ],
    ("ALT_FT", "off"): [
        "rtss_self_created_shot_rate100",
        "rtss_drive_rate100",
        "rtss_touch_double_team_share",
    ],
    ("ALT_FT_FREQ", "off"): [
        "rtss_self_created_shot_rate100",
        "rtss_drive_rate100",
        "rtss_touch_double_team_share",
    ],
    ("ALT_FC_COMPLETION", "off"): [
        "rtss_potential_assist_rate100",
        "rtss_pass_assist_opp_share",
        "rtss_pass_turnover_share",
        "rtss_self_created_shot_rate100",
    ],
    ("ALT_TOV_VALUE", "off"): [
        "rtss_potential_assist_rate100",
        "rtss_pass_assist_opp_share",
        "rtss_pass_turnover_share",
        "rtss_self_created_shot_rate100",
    ],
    ("ALT_BADPASS_TOV_VALUE", "off"): [
        "rtss_potential_assist_rate100",
        "rtss_pass_assist_opp_share",
        "rtss_pass_turnover_share",
    ],
    ("SECOND_CHANCE_CLEAN", "off"): [
        "rtss_off_rb_opp_rate100",
        "rtss_off_crash_rate100",
        "rtss_picks_set_rate100",
    ],
}

ROLE_ONLY_FEATURES: dict[tuple[str, str], list[str]] = {
    ("ALT_EFG_RIM_FREQ", "off"): [
        "rtss_rim_self_created_rate100",
        "rtss_rim_assist_opp_share",
    ],
    ("ALT_EFG_RIM_FG", "off"): [
        "rtss_rim_assist_opp_share",
        "rtss_after_pass_self_created_share",
        "rtss_potential_assist_received_rate100",
    ],
    ("ALT_EFG_MID_FREQ", "off"): [
        "rtss_mid_self_created_rate100",
        "rtss_mid_assist_opp_share",
    ],
    ("ALT_EFG_MID_FG", "off"): [
        "rtss_mid_assist_opp_share",
        "rtss_after_pass_self_created_share",
    ],
    ("ALT_EFG_THREE_FREQ", "off"): [
        "rtss_three_self_created_rate100",
        "rtss_three_assist_opp_share",
        "rtss_potential_assist_received_rate100",
    ],
    ("ALT_EFG_THREE_FG", "off"): [
        "rtss_three_assist_opp_share",
        "rtss_after_pass_self_created_share",
        "rtss_potential_assist_received_rate100",
    ],
    ("ALT_FT", "off"): ["rtss_self_created_shot_rate100"],
    ("ALT_FT_FREQ", "off"): ["rtss_self_created_shot_rate100"],
    ("ALT_FC_COMPLETION", "off"): [
        "rtss_potential_assist_rate100",
        "rtss_pass_assist_opp_share",
        "rtss_self_created_shot_rate100",
    ],
    ("ALT_TOV_VALUE", "off"): [
        "rtss_potential_assist_rate100",
        "rtss_pass_assist_opp_share",
        "rtss_self_created_shot_rate100",
    ],
    ("ALT_BADPASS_TOV_VALUE", "off"): [
        "rtss_potential_assist_rate100",
        "rtss_pass_assist_opp_share",
    ],
}

ROLE_FINISHING_FEATURES: dict[tuple[str, str], list[str]] = {
    ("ALT_EFG_RIM_FG", "off"): [
        "rtss_rim_assist_opp_share",
        "rtss_after_pass_self_created_share",
        "rtss_rim_assist_opp_fg_pct",
        "rtss_rim_self_created_fg_pct",
    ],
    ("ALT_EFG_MID_FG", "off"): [
        "rtss_mid_assist_opp_share",
        "rtss_after_pass_self_created_share",
        "rtss_mid_assist_opp_fg_pct",
        "rtss_mid_self_created_fg_pct",
    ],
    ("ALT_EFG_THREE_FG", "off"): [
        "rtss_three_assist_opp_share",
        "rtss_after_pass_self_created_share",
        "rtss_three_assist_opp_fg_pct",
        "rtss_three_self_created_fg_pct",
    ],
}

ROLE_QUALITY_FEATURES: dict[tuple[str, str], list[str]] = {
    ("ALT_EFG_MID_FG", "off"): [
        "rtss_mid_assist_opp_qsq_epv",
        "rtss_mid_assist_opp_shooter_adjustment_epv",
        "rtss_mid_self_created_qsq_epv",
        "rtss_mid_self_created_shooter_adjustment_epv",
    ],
    ("ALT_EFG_THREE_FG", "off"): [
        "rtss_three_assist_opp_qsq_epv",
        "rtss_three_assist_opp_shooter_adjustment_epv",
        "rtss_three_self_created_qsq_epv",
        "rtss_three_self_created_shooter_adjustment_epv",
    ],
}


MECHANISM_FEATURES: dict[tuple[str, str], list[str]] = {
    ("ALT_EFG_RIM_FREQ", "def"): [
        "rtss_drive_help_rate100",
        "rtss_def_rotation_rate100",
        "rtss_def_matchup_rate100",
        "rtss_def_rim_dfga_rate100",
    ],
    ("ALT_EFG_RIM_FG", "def"): [
        "rtss_drive_help_rate100",
        "rtss_def_rotation_rate100",
        "rtss_def_contest_rate100",
        "rtss_def_rim_dfga_rate100",
        "rtss_def_rim_fg_pct_allowed",
        "rtss_def_rim_heavy_contest_share",
    ],
    ("ALT_EFG_MID_FREQ", "def"): [
        "rtss_def_contest_rate100",
        "rtss_def_closeout_rate100",
        "rtss_def_rotation_rate100",
        "rtss_def_mid3_catch_shoot_share",
        "rtss_def_mid_dfga_rate100",
    ],
    ("ALT_EFG_MID_FG", "def"): [
        "rtss_def_contest_rate100",
        "rtss_def_heavy_contest_share",
        "rtss_def_contest_distance",
        "rtss_def_closeout_rate100",
        "rtss_def_rotation_rate100",
        "rtss_def_mid_fg_pct_allowed",
        "rtss_def_mid_heavy_contest_share",
        "rtss_def_mid_closest_distance",
    ],
    ("ALT_EFG_THREE_FREQ", "def"): [
        "rtss_def_contest_rate100",
        "rtss_def_closeout_rate100",
        "rtss_def_rotation_rate100",
        "rtss_def_mid3_catch_shoot_share",
        "rtss_def_three_dfga_rate100",
    ],
    ("ALT_EFG_THREE_FG", "def"): [
        "rtss_def_contest_rate100",
        "rtss_def_heavy_contest_share",
        "rtss_def_contest_distance",
        "rtss_def_closeout_rate100",
        "rtss_def_rearview_share",
        "rtss_def_three_fg_pct_allowed",
        "rtss_def_three_heavy_contest_share",
        "rtss_def_three_closest_distance",
    ],
    ("ALT_FT", "def"): [
        "rtss_drive_help_rate100",
        "rtss_def_rotation_rate100",
        "rtss_def_contest_rate100",
        "rtss_def_shooting_fouls_rate100",
        "rtss_def_fouls_committed_rate100",
    ],
    ("ALT_FT_FREQ", "def"): [
        "rtss_drive_help_rate100",
        "rtss_def_rotation_rate100",
        "rtss_def_contest_rate100",
        "rtss_def_shooting_fouls_rate100",
        "rtss_def_non_shooting_fouls_rate100",
    ],
    ("ALT_FC_COMPLETION", "def"): [
        "rtss_def_matchup_rate100",
        "rtss_def_rotation_rate100",
        "rtss_drive_help_rate100",
        "rtss_def_off_fouls_drawn_rate100",
    ],
    ("ALT_TOV_VALUE", "def"): [
        "rtss_def_matchup_rate100",
        "rtss_def_rotation_rate100",
        "rtss_drive_help_rate100",
        "rtss_def_off_fouls_drawn_rate100",
        "rtss_def_charges_drawn_rate100",
    ],
    ("ALT_BADPASS_TOV_VALUE", "def"): [
        "rtss_def_matchup_rate100",
        "rtss_def_rotation_rate100",
        "rtss_def_off_fouls_drawn_rate100",
    ],
    ("SECOND_CHANCE_CLEAN", "def"): [
        "rtss_def_rb_opp_rate100",
        "rtss_def_rotation_rate100",
    ],
}


Q_FEATURES: dict[tuple[str, str], list[str]] = {
    ("ALT_EFG_MID_FREQ", "off"): ["rtss_mid_qsq_epv"],
    ("ALT_EFG_MID_FG", "off"): [
        "rtss_mid_qsq_epv",
        "rtss_mid_shooter_adjustment_epv",
    ],
    ("ALT_EFG_THREE_FREQ", "off"): ["rtss_three_qsq_epv"],
    ("ALT_EFG_THREE_FG", "off"): [
        "rtss_three_qsq_epv",
        "rtss_three_shooter_adjustment_epv",
    ],
    # Defensive allowed shot quality, split by zone. The pooled mid3 average
    # mixes two different defensive skills, so each zone gets its own arm.
    ("ALT_EFG_MID_FREQ", "def"): [
        "rtss_def_mid_qsq_epv",
        "rtss_def_mid_qsp_epv",
    ],
    ("ALT_EFG_MID_FG", "def"): [
        "rtss_def_mid_qsq_epv",
        "rtss_def_mid_qsp_epv",
        "rtss_def_mid_closest_distance",
    ],
    ("ALT_EFG_THREE_FREQ", "def"): [
        "rtss_def_three_qsq_epv",
        "rtss_def_three_qsp_epv",
    ],
    ("ALT_EFG_THREE_FG", "def"): [
        "rtss_def_three_qsq_epv",
        "rtss_def_three_qsp_epv",
        "rtss_def_three_closest_distance",
    ],
}


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _weighted_metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    weight = np.asarray(weight, dtype=float)
    mean = np.average(actual, weights=weight)
    squared_error = np.average(np.square(prediction - actual), weights=weight)
    total_error = np.average(np.square(actual - mean), weights=weight)
    actual_centered = actual - mean
    prediction_centered = prediction - np.average(prediction, weights=weight)
    covariance = np.average(actual_centered * prediction_centered, weights=weight)
    variance = np.average(np.square(actual_centered), weights=weight) * np.average(
        np.square(prediction_centered), weights=weight
    )
    return {
        "rmse": float(np.sqrt(squared_error)),
        "r2": float(1.0 - squared_error / total_error) if total_error > 0 else np.nan,
        "correlation": float(covariance / np.sqrt(variance)) if variance > 0 else np.nan,
    }


def _fit(rows: pd.DataFrame, features: list[str], alpha: float) -> Pipeline:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )
    weight = np.sqrt(rows["target_poss"].clip(lower=1.0, upper=2500.0))
    model.fit(
        rows[features],
        rows["target"],
        ridge__sample_weight=weight / weight.mean(),
    )
    return model


def _target_rows(
    year: int,
    prefix: str,
    side: str,
    box_by_year: dict[int, pd.DataFrame],
    target_dir: Path,
    min_target_poss: float,
) -> pd.DataFrame:
    target = load_target(target_dir, year, prefix)[
        ["player_id", side, f"{side}_poss"]
    ].rename(columns={side: "target", f"{side}_poss": "target_poss"})
    rows = box_by_year[year].merge(target, on="player_id", how="inner")
    return rows[rows["target_poss"] >= min_target_poss].copy()


def _select_alpha_by_year(
    rows_by_year: dict[int, pd.DataFrame],
    years: list[int],
    features: list[str],
    alphas: tuple[float, ...] | list[float],
) -> float:
    best: tuple[float, float] | None = None
    for alpha in alphas:
        actual: list[float] = []
        prediction: list[float] = []
        weight: list[float] = []
        for heldout in years:
            train_years = [year for year in years if year != heldout]
            if not train_years:
                continue
            train = pd.concat([rows_by_year[year] for year in train_years], ignore_index=True)
            validation = rows_by_year[heldout]
            model = _fit(train, features, alpha)
            actual.extend(validation["target"].tolist())
            prediction.extend(model.predict(validation[features]).tolist())
            weight.extend(np.sqrt(validation["target_poss"]).tolist())
        if not actual:
            continue
        rmse = _weighted_metrics(
            np.asarray(actual), np.asarray(prediction), np.asarray(weight)
        )["rmse"]
        candidate = (rmse, float(alpha))
        if best is None or candidate < best:
            best = candidate
    return 100.0 if best is None else best[1]


def _season_standardize_rtss(rtss: pd.DataFrame) -> pd.DataFrame:
    """Convert tracking fields to within-season exposure-weighted z scores."""
    rtss = rtss.copy()
    feature_columns = [column for column in rtss if column.startswith("rtss_")]
    for column in feature_columns:
        rtss[f"z_{column}"] = 0.0
    for _season, index in rtss.groupby("season").groups.items():
        group = rtss.loc[index]
        for column in feature_columns:
            side = (
                "def_chances"
                if column.startswith("rtss_def_")
                or column.startswith("rtss_drive_help")
                else "off_chances"
            )
            weight = group[side].clip(lower=1.0).to_numpy(dtype=float)
            values = group[column].to_numpy(dtype=float)
            mean = np.average(values, weights=weight)
            variance = np.average(np.square(values - mean), weights=weight)
            scale = np.sqrt(variance)
            rtss.loc[index, f"z_{column}"] = (
                (values - mean) / scale if scale > 1e-12 else 0.0
            )
    return rtss


def _rtss_rows(
    base_rows: pd.DataFrame,
    rtss_year: pd.DataFrame,
    side: str,
    min_rtss_chances: float,
) -> pd.DataFrame:
    rows = base_rows.merge(
        rtss_year,
        on="player_id",
        how="inner",
        suffixes=("", "_rtss"),
    )
    return rows[rows[f"{side}_chances"] >= min_rtss_chances].copy()


def _lane_features(prefix: str, side: str) -> dict[str, list[str]]:
    base = (
        CREATION_FEATURES[(prefix, side)]
        if side == "off"
        else MECHANISM_FEATURES[(prefix, side)]
    )
    q = Q_FEATURES.get((prefix, side), [])
    lanes = {"tracking_core": base}
    role_only = ROLE_ONLY_FEATURES.get((prefix, side), [])
    if role_only:
        lanes["creation_roles_only"] = role_only
    role_finishing = ROLE_FINISHING_FEATURES.get((prefix, side), [])
    if role_finishing:
        lanes["role_split_finishing"] = role_finishing
        role_quality = ROLE_QUALITY_FEATURES.get((prefix, side), [])
        if role_quality:
            lanes["role_split_finishing_plus_q"] = _dedupe(
                role_finishing + role_quality
            )
    if q:
        lanes["tracking_core_plus_q"] = _dedupe(base + q)
    return {
        lane: [f"z_{feature}" for feature in features]
        for lane, features in lanes.items()
    }


def cross_validate(
    feature_path: Path,
    target_dir: Path,
    output_dir: Path,
    min_rtss_chances: float,
    min_target_poss: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rtss = pd.read_csv(feature_path, dtype={"player_id": str})
    rtss["player_id"] = (
        pd.to_numeric(rtss["player_id"], errors="coerce").astype("Int64").astype(str)
    )
    rtss = _season_standardize_rtss(rtss)
    box_by_year = {year: load_box_features(year) for year in BOX_YEARS}

    fold_results: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    coefficients: list[dict[str, Any]] = []

    for prefix in TARGET_PREFIXES:
        for side in ("off", "def"):
            box_features = feature_columns(prefix, side)
            box_rows = {
                year: _target_rows(
                    year,
                    prefix,
                    side,
                    box_by_year,
                    target_dir,
                    min_target_poss,
                )
                for year in BOX_YEARS
            }
            tracking_rows = {
                year: _rtss_rows(
                    box_rows[year],
                    rtss[rtss["season"] == year],
                    side,
                    min_rtss_chances,
                )
                for year in RTSS_YEARS
            }
            lanes = _lane_features(prefix, side)
            available = set.intersection(
                *(set(rows.columns) for rows in tracking_rows.values())
            )
            missing = sorted(
                feature
                for features in lanes.values()
                for feature in features
                if feature not in available
            )
            if missing:
                raise KeyError(f"{prefix} {side} missing RTSS fields: {missing}")

            for heldout_year in RTSS_YEARS:
                box_train_years = [year for year in BOX_YEARS if year != heldout_year]
                selected_box_alpha = _select_alpha_by_year(
                    box_rows,
                    box_train_years,
                    box_features,
                    BOX_ALPHAS,
                )
                box_train = pd.concat(
                    [box_rows[year] for year in box_train_years],
                    ignore_index=True,
                )
                box_model = _fit(box_train, box_features, selected_box_alpha)
                validation = tracking_rows[heldout_year].copy()
                box_prediction = box_model.predict(validation[box_features])
                validation["box_prediction"] = box_prediction

                residual_train_years = [
                    year for year in RTSS_YEARS if year != heldout_year
                ]
                residual_by_year: dict[int, pd.DataFrame] = {}
                for residual_year in residual_train_years:
                    inner_box_years = [
                        year
                        for year in box_train_years
                        if year != residual_year
                    ]
                    inner_box_train = pd.concat(
                        [box_rows[year] for year in inner_box_years],
                        ignore_index=True,
                    )
                    inner_box_model = _fit(
                        inner_box_train,
                        box_features,
                        selected_box_alpha,
                    )
                    residual_rows = tracking_rows[residual_year].copy()
                    residual_rows["box_prediction"] = inner_box_model.predict(
                        residual_rows[box_features]
                    )
                    residual_rows["target"] = (
                        residual_rows["target"] - residual_rows["box_prediction"]
                    )
                    residual_by_year[residual_year] = residual_rows

                baseline_metrics = _weighted_metrics(
                    validation["target"].to_numpy(),
                    box_prediction,
                    np.sqrt(validation["target_poss"].to_numpy()),
                )
                for lane, tracking_features in lanes.items():
                    selected_residual_alpha = _select_alpha_by_year(
                        residual_by_year,
                        residual_train_years,
                        tracking_features,
                        RESIDUAL_ALPHAS,
                    )
                    residual_train = pd.concat(
                        [residual_by_year[year] for year in residual_train_years],
                        ignore_index=True,
                    )
                    residual_model = _fit(
                        residual_train,
                        tracking_features,
                        selected_residual_alpha,
                    )
                    residual_prediction = residual_model.predict(
                        validation[tracking_features]
                    )
                    augmented_prediction = box_prediction + residual_prediction
                    augmented_metrics = _weighted_metrics(
                        validation["target"].to_numpy(),
                        augmented_prediction,
                        np.sqrt(validation["target_poss"].to_numpy()),
                    )
                    fold_results.append(
                        {
                            "prefix": prefix,
                            "side": side,
                            "lane": lane,
                            "heldout_year": heldout_year,
                            "box_training_years": "|".join(map(str, box_train_years)),
                            "rtss_training_years": "|".join(
                                map(str, residual_train_years)
                            ),
                            "validation_rows": len(validation),
                            "box_alpha": selected_box_alpha,
                            "residual_alpha": selected_residual_alpha,
                            "tracking_feature_count": len(tracking_features),
                            "tracking_features": "|".join(tracking_features),
                            "box_rmse": baseline_metrics["rmse"],
                            "augmented_rmse": augmented_metrics["rmse"],
                            "delta_rmse": augmented_metrics["rmse"]
                            - baseline_metrics["rmse"],
                            "box_r2": baseline_metrics["r2"],
                            "augmented_r2": augmented_metrics["r2"],
                            "delta_r2": augmented_metrics["r2"]
                            - baseline_metrics["r2"],
                            "box_correlation": baseline_metrics["correlation"],
                            "augmented_correlation": augmented_metrics["correlation"],
                            "delta_correlation": augmented_metrics["correlation"]
                            - baseline_metrics["correlation"],
                        }
                    )
                    for feature, coefficient in zip(
                        tracking_features,
                        residual_model.named_steps["ridge"].coef_,
                    ):
                        coefficients.append(
                            {
                                "prefix": prefix,
                                "side": side,
                                "lane": lane,
                                "heldout_year": heldout_year,
                                "feature": feature.removeprefix("z_"),
                                "standardized_residual_coefficient": float(coefficient),
                            }
                        )
                    for row_number, (_, row) in enumerate(validation.iterrows()):
                        predictions.append(
                            {
                                "prefix": prefix,
                                "side": side,
                                "lane": lane,
                                "heldout_year": heldout_year,
                                "player_id": row["player_id"],
                                "player_name": row.get("player_name", ""),
                                "target": float(row["target"]),
                                "target_poss": float(row["target_poss"]),
                                "box_prediction": float(box_prediction[row_number]),
                                "residual_prediction": float(
                                    residual_prediction[row_number]
                                ),
                                "augmented_prediction": float(
                                    augmented_prediction[row_number]
                                ),
                            }
                        )

    folds = pd.DataFrame(fold_results)
    prediction_frame = pd.DataFrame(predictions)
    coefficient_frame = pd.DataFrame(coefficients)

    pooled_rows: list[dict[str, Any]] = []
    for (prefix, side, lane), group in prediction_frame.groupby(
        ["prefix", "side", "lane"]
    ):
        weights = np.sqrt(group["target_poss"].to_numpy())
        box_metrics = _weighted_metrics(
            group["target"].to_numpy(),
            group["box_prediction"].to_numpy(),
            weights,
        )
        augmented_metrics = _weighted_metrics(
            group["target"].to_numpy(),
            group["augmented_prediction"].to_numpy(),
            weights,
        )
        pooled_rows.append(
            {
                "prefix": prefix,
                "side": side,
                "lane": lane,
                "rows": len(group),
                "seasons": "|".join(map(str, sorted(group["heldout_year"].unique()))),
                "box_rmse": box_metrics["rmse"],
                "augmented_rmse": augmented_metrics["rmse"],
                "delta_rmse": augmented_metrics["rmse"] - box_metrics["rmse"],
                "relative_rmse_change_pct": 100.0
                * (augmented_metrics["rmse"] - box_metrics["rmse"])
                / box_metrics["rmse"],
                "box_r2": box_metrics["r2"],
                "augmented_r2": augmented_metrics["r2"],
                "delta_r2": augmented_metrics["r2"] - box_metrics["r2"],
                "box_correlation": box_metrics["correlation"],
                "augmented_correlation": augmented_metrics["correlation"],
                "delta_correlation": augmented_metrics["correlation"]
                - box_metrics["correlation"],
                "fold_rmse_wins": int(
                    (
                        folds[
                            (folds["prefix"] == prefix)
                            & (folds["side"] == side)
                            & (folds["lane"] == lane)
                        ]["delta_rmse"]
                        < 0
                    ).sum()
                ),
            }
        )
    pooled = pd.DataFrame(pooled_rows)
    best = (
        pooled.sort_values(
            ["prefix", "side", "augmented_rmse", "augmented_correlation"],
            ascending=[True, True, True, False],
        )
        .groupby(["prefix", "side"], as_index=False)
        .first()
    )
    best["selected_prior_lane"] = np.where(
        best["delta_rmse"] < 0,
        best["lane"],
        "box_only",
    )
    best["two_metric_candidate"] = (
        (best["delta_rmse"] < 0) & (best["delta_correlation"] > 0)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output_dir / "season_folds.csv", index=False)
    pooled.to_csv(output_dir / "pooled_results.csv", index=False)
    best.to_csv(output_dir / "best_by_component.csv", index=False)
    prediction_frame.to_csv(output_dir / "out_of_fold_predictions.csv", index=False)
    coefficient_frame.to_csv(output_dir / "fold_coefficients.csv", index=False)

    shot_prefixes = {
        "ALT_EFG_RIM_FREQ",
        "ALT_EFG_RIM_FG",
        "ALT_EFG_MID_FREQ",
        "ALT_EFG_MID_FG",
        "ALT_EFG_THREE_FREQ",
        "ALT_EFG_THREE_FG",
    }
    offensive_shots = best[
        (best["side"] == "off") & best["prefix"].isin(shot_prefixes)
    ]
    manifest = {
        "status": "research_cross_validation_not_live",
        "design": (
            "symmetric leave-one-RTSS-season-out CV; box backbone uses every "
            "non-held-out 2022-2025 season; RTSS predicts cross-fitted box residuals"
        ),
        "outer_seasons": list(RTSS_YEARS),
        "warning": (
            "2023/2024 folds use future seasons and are not historical as-of tests; "
            "all three seasons are now development data"
        ),
        "assist_semantics": {
            "shots.assistOpp": "potential-assisted attempt on made and missed shots",
            "shots.assisted": "made-shot result; not used for self-created labeling",
            "self_created": "not shots.assistOpp",
            "passes.assistOpp": "potential assist created",
        },
        "aggregate": {
            "component_sides": int(len(best)),
            "tracking_rmse_candidates": int((best["delta_rmse"] < 0).sum()),
            "tracking_two_metric_candidates": int(
                ((best["delta_rmse"] < 0) & (best["delta_correlation"] > 0)).sum()
            ),
            "offensive_shot_components": int(len(offensive_shots)),
            "offensive_shot_rmse_wins": int(
                (offensive_shots["delta_rmse"] < 0).sum()
            ),
            "offensive_shot_both_wins": int(
                (
                    (offensive_shots["delta_rmse"] < 0)
                    & (offensive_shots["delta_correlation"] > 0)
                ).sum()
            ),
        },
    }
    (output_dir / "cross_validation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return folds, best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--target-dir", type=Path, default=BOX_PRIOR_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-rtss-chances", type=float, default=100.0)
    parser.add_argument("--min-target-poss", type=float, default=250.0)
    args = parser.parse_args()
    _, best = cross_validate(
        args.features.resolve(),
        args.target_dir.resolve(),
        args.output_dir.resolve(),
        args.min_rtss_chances,
        args.min_target_poss,
    )
    print(
        best[
            [
                "prefix",
                "side",
                "selected_prior_lane",
                "rows",
                "relative_rmse_change_pct",
                "delta_r2",
                "delta_correlation",
                "fold_rmse_wins",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
