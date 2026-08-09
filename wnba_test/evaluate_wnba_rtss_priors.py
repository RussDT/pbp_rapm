#!/usr/bin/env python3
"""Season-forward test of RTSS features as WNBA component-prior inputs.

Model and alpha selection use 2023 -> 2024 only.  The selected challenger is
then refit on 2023-24 and opened once on 2025.  Box-only and challenger models
use the exact same mapped-player rows so the comparison is paired.
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
    RIDGE_ALPHAS,
    TARGET_PREFIXES,
    feature_columns,
    load_box_features,
    load_target,
)


DEFAULT_FEATURES = (
    ROOT / "wnba_test" / "Results" / "rtss_prior_research" / "player_season_features.csv"
)
DEFAULT_OUTPUT = ROOT / "wnba_test" / "Results" / "rtss_prior_research"


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
    ("ALT_EFG_MID_FREQ", "def"): [
        "rtss_def_mid3_qsq_epv",
        "rtss_def_mid3_qsp_epv",
    ],
    ("ALT_EFG_MID_FG", "def"): [
        "rtss_def_mid3_qsq_epv",
        "rtss_def_mid3_qsp_epv",
        "rtss_def_closest_distance",
    ],
    ("ALT_EFG_THREE_FREQ", "def"): [
        "rtss_def_mid3_qsq_epv",
        "rtss_def_mid3_qsp_epv",
    ],
    ("ALT_EFG_THREE_FG", "def"): [
        "rtss_def_mid3_qsq_epv",
        "rtss_def_mid3_qsp_epv",
        "rtss_def_closest_distance",
    ],
}


OFF_CREATION = [
    "rtss_touch_rate100",
    "rtss_touch_time",
    "rtss_touch_dribbles",
    "rtss_touch_double_team_share",
]
OFF_PASSING = [
    "rtss_pass_rate100",
    "rtss_pass_led_shot_share",
    "rtss_pass_assist_opp_share",
    "rtss_pass_turnover_share",
]
OFF_SCREENING = [
    "rtss_picks_set_rate100",
    "rtss_offball_screen_rate100",
    "rtss_pick_contact_share",
]
DEF_CONTEST = [
    "rtss_def_contest_rate100",
    "rtss_def_heavy_contest_share",
    "rtss_def_contest_distance",
    "rtss_def_closeout_rate100",
    "rtss_def_rotation_rate100",
]


MECHANISM_FEATURES: dict[tuple[str, str], list[str]] = {
    ("ALT_EFG_RIM_FREQ", "off"): [
        "rtss_drive_rate100",
        "rtss_drive_blowby_share",
        *OFF_CREATION,
    ],
    ("ALT_EFG_RIM_FG", "off"): [
        "rtss_drive_rate100",
        "rtss_pass_assist_opp_share",
        "rtss_picks_used_rate100",
    ],
    ("ALT_EFG_MID_FREQ", "off"): [
        "rtss_mid_share",
        "rtss_touch_dribbles",
        "rtss_iso_post_rate100",
    ],
    ("ALT_EFG_MID_FG", "off"): [
        "rtss_catch_shoot_share",
        "rtss_touch_double_team_share",
    ],
    ("ALT_EFG_THREE_FREQ", "off"): [
        "rtss_three_share",
        "rtss_catch_shoot_share",
        "rtss_pass_led_shot_share",
        "rtss_offball_screen_rate100",
    ],
    ("ALT_EFG_THREE_FG", "off"): [
        "rtss_catch_shoot_share",
        "rtss_pass_assist_opp_share",
    ],
    ("ALT_FT", "off"): [
        "rtss_drive_rate100",
        "rtss_drive_blowby_share",
        "rtss_touch_double_team_share",
    ],
    ("ALT_FT_FREQ", "off"): [
        "rtss_drive_rate100",
        "rtss_drive_blowby_share",
        "rtss_touch_double_team_share",
    ],
    ("ALT_FC_COMPLETION", "off"): OFF_CREATION + OFF_PASSING,
    ("ALT_TOV_VALUE", "off"): OFF_CREATION + OFF_PASSING,
    ("ALT_BADPASS_TOV_VALUE", "off"): OFF_PASSING + ["rtss_pass_reversal_share"],
    ("SECOND_CHANCE_CLEAN", "off"): [
        "rtss_off_crash_rate100",
        "rtss_off_rb_opp_rate100",
        "rtss_picks_set_rate100",
    ],
    ("ALT_EFG_RIM_FREQ", "def"): [
        "rtss_drive_help_rate100",
        "rtss_def_rotation_rate100",
        "rtss_def_matchup_rate100",
    ],
    ("ALT_EFG_RIM_FG", "def"): [
        "rtss_drive_help_rate100",
        "rtss_def_rotation_rate100",
        "rtss_def_contest_rate100",
    ],
    ("ALT_EFG_MID_FREQ", "def"): DEF_CONTEST,
    ("ALT_EFG_MID_FG", "def"): DEF_CONTEST,
    ("ALT_EFG_THREE_FREQ", "def"): DEF_CONTEST + ["rtss_def_mid3_catch_shoot_share"],
    ("ALT_EFG_THREE_FG", "def"): DEF_CONTEST + ["rtss_def_rearview_share"],
    ("ALT_FT", "def"): [
        "rtss_drive_help_rate100",
        "rtss_def_rotation_rate100",
        "rtss_def_contest_rate100",
    ],
    ("ALT_FT_FREQ", "def"): [
        "rtss_drive_help_rate100",
        "rtss_def_rotation_rate100",
        "rtss_def_contest_rate100",
    ],
    ("ALT_FC_COMPLETION", "def"): [
        "rtss_def_matchup_rate100",
        "rtss_def_rotation_rate100",
        "rtss_drive_help_rate100",
    ],
    ("ALT_TOV_VALUE", "def"): [
        "rtss_def_matchup_rate100",
        "rtss_def_rotation_rate100",
        "rtss_drive_help_rate100",
    ],
    ("ALT_BADPASS_TOV_VALUE", "def"): [
        "rtss_def_matchup_rate100",
        "rtss_def_rotation_rate100",
    ],
    ("SECOND_CHANCE_CLEAN", "def"): [
        "rtss_def_rb_opp_rate100",
        "rtss_def_rotation_rate100",
    ],
}


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _weighted_metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
) -> dict[str, float]:
    weight = np.asarray(weight, dtype=float)
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    rmse = float(np.sqrt(np.average(np.square(prediction - actual), weights=weight)))
    actual_centered = actual - np.average(actual, weights=weight)
    prediction_centered = prediction - np.average(prediction, weights=weight)
    covariance = np.average(actual_centered * prediction_centered, weights=weight)
    variance = np.average(np.square(actual_centered), weights=weight) * np.average(
        np.square(prediction_centered), weights=weight
    )
    correlation = float(covariance / np.sqrt(variance)) if variance > 0 else np.nan
    return {"rmse": rmse, "correlation": correlation}


def _fit(rows: pd.DataFrame, features: list[str], alpha: float) -> Pipeline:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )
    weights = np.sqrt(rows["target_poss"].clip(lower=1.0, upper=2500.0))
    weights = weights / weights.mean()
    model.fit(rows[features], rows["target"], ridge__sample_weight=weights)
    return model


def _rows_for_year(
    year: int,
    prefix: str,
    side: str,
    box_by_year: dict[int, pd.DataFrame],
    rtss: pd.DataFrame,
    target_dir: Path,
    min_rtss_chances: float,
    min_target_poss: float,
) -> pd.DataFrame:
    target = load_target(target_dir, year, prefix)[
        ["player_id", side, f"{side}_poss"]
    ].rename(columns={side: "target", f"{side}_poss": "target_poss"})
    tracking = rtss[rtss["season"] == year].copy()
    exposure = f"{side}_chances"
    frame = box_by_year[year].merge(tracking, on="player_id", how="inner", suffixes=("", "_rtss"))
    frame = frame.merge(target, on="player_id", how="inner")
    frame = frame[
        (frame[exposure] >= min_rtss_chances)
        & (frame["target_poss"] >= min_target_poss)
    ].copy()
    frame["year"] = int(year)
    return frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["target", "target_poss"]
    )


def _lane_features(prefix: str, side: str) -> dict[str, list[str]]:
    box = feature_columns(prefix, side)
    q = Q_FEATURES.get((prefix, side), [])
    mechanism = MECHANISM_FEATURES[(prefix, side)]
    lanes = {
        "box_only": box,
        "box_plus_q": _dedupe(box + q),
        "box_plus_mechanism": _dedupe(box + mechanism),
        "box_plus_all": _dedupe(box + q + mechanism),
    }
    return {
        name: features
        for name, features in lanes.items()
        if name == "box_only" or len(features) > len(box)
    }


def evaluate(
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
    box_by_year = {year: load_box_features(year) for year in (2023, 2024, 2025)}

    development_rows: list[dict[str, Any]] = []
    confirmation_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []

    for prefix in TARGET_PREFIXES:
        for side in ("off", "def"):
            by_year = {
                year: _rows_for_year(
                    year,
                    prefix,
                    side,
                    box_by_year,
                    rtss,
                    target_dir,
                    min_rtss_chances,
                    min_target_poss,
                )
                for year in (2023, 2024, 2025)
            }
            lanes = _lane_features(prefix, side)
            available = set.intersection(*(set(frame.columns) for frame in by_year.values()))
            missing = sorted(
                {
                    feature
                    for features in lanes.values()
                    for feature in features
                    if feature not in available
                }
            )
            if missing:
                raise KeyError(f"{prefix} {side} missing features: {missing}")

            dev_scores: list[dict[str, Any]] = []
            for lane, features in lanes.items():
                for alpha in RIDGE_ALPHAS:
                    model = _fit(by_year[2023], features, alpha)
                    prediction = model.predict(by_year[2024][features])
                    metrics = _weighted_metrics(
                        by_year[2024]["target"].to_numpy(),
                        prediction,
                        np.sqrt(by_year[2024]["target_poss"].to_numpy()),
                    )
                    dev_scores.append(
                        {
                            "prefix": prefix,
                            "side": side,
                            "lane": lane,
                            "alpha": float(alpha),
                            "feature_count": len(features),
                            "features": "|".join(features),
                            "train_rows": len(by_year[2023]),
                            "validation_rows": len(by_year[2024]),
                            **metrics,
                        }
                    )
            development_rows.extend(dev_scores)
            scored = pd.DataFrame(dev_scores)
            best_by_lane = (
                scored.sort_values(
                    ["lane", "rmse", "correlation"],
                    ascending=[True, True, False],
                )
                .groupby("lane", as_index=False)
                .first()
            )
            baseline = best_by_lane[best_by_lane["lane"] == "box_only"].iloc[0]
            challenger = (
                best_by_lane[best_by_lane["lane"] != "box_only"]
                .sort_values(["rmse", "correlation"], ascending=[True, False])
                .iloc[0]
            )

            train = pd.concat([by_year[2023], by_year[2024]], ignore_index=True)
            validation = by_year[2025]
            baseline_features = lanes["box_only"]
            challenger_features = lanes[str(challenger["lane"])]
            baseline_model = _fit(train, baseline_features, float(baseline["alpha"]))
            challenger_model = _fit(
                train, challenger_features, float(challenger["alpha"])
            )
            for feature, coefficient in zip(
                challenger_features,
                challenger_model.named_steps["ridge"].coef_,
            ):
                coefficient_rows.append(
                    {
                        "prefix": prefix,
                        "side": side,
                        "selected_challenger": str(challenger["lane"]),
                        "feature": feature,
                        "is_rtss": feature.startswith("rtss_"),
                        "standardized_coefficient": float(coefficient),
                    }
                )
            baseline_prediction = baseline_model.predict(validation[baseline_features])
            challenger_prediction = challenger_model.predict(
                validation[challenger_features]
            )
            weights = np.sqrt(validation["target_poss"].to_numpy())
            baseline_metrics = _weighted_metrics(
                validation["target"].to_numpy(), baseline_prediction, weights
            )
            challenger_metrics = _weighted_metrics(
                validation["target"].to_numpy(), challenger_prediction, weights
            )
            confirmation_rows.append(
                {
                    "prefix": prefix,
                    "side": side,
                    "selected_challenger": str(challenger["lane"]),
                    "development_box_rmse": float(baseline["rmse"]),
                    "development_challenger_rmse": float(challenger["rmse"]),
                    "development_delta_rmse": float(challenger["rmse"] - baseline["rmse"]),
                    "development_eligible": bool(challenger["rmse"] < baseline["rmse"]),
                    "box_alpha": float(baseline["alpha"]),
                    "challenger_alpha": float(challenger["alpha"]),
                    "challenger_features": "|".join(challenger_features),
                    "train_rows": len(train),
                    "confirmation_rows": len(validation),
                    "box_rmse": baseline_metrics["rmse"],
                    "challenger_rmse": challenger_metrics["rmse"],
                    "delta_rmse": challenger_metrics["rmse"] - baseline_metrics["rmse"],
                    "box_correlation": baseline_metrics["correlation"],
                    "challenger_correlation": challenger_metrics["correlation"],
                    "delta_correlation": (
                        challenger_metrics["correlation"]
                        - baseline_metrics["correlation"]
                    ),
                }
            )
            for row_number, (_, row) in enumerate(validation.iterrows()):
                prediction_rows.append(
                    {
                        "prefix": prefix,
                        "side": side,
                        "player_id": row["player_id"],
                        "player_name": row.get("player_name", row.get("player_name_rtss", "")),
                        "target": float(row["target"]),
                        "target_poss": float(row["target_poss"]),
                        "box_prediction": float(baseline_prediction[row_number]),
                        "challenger_prediction": float(challenger_prediction[row_number]),
                        "selected_challenger": str(challenger["lane"]),
                    }
                )

    development = pd.DataFrame(development_rows)
    confirmation = pd.DataFrame(confirmation_rows)
    confirmation["decision"] = np.where(
        confirmation["development_eligible"],
        np.where(
            (confirmation["delta_rmse"] < 0)
            & (confirmation["delta_correlation"] > 0),
            "confirmed_both",
            "failed_confirmation",
        ),
        "retain_box_failed_development",
    )
    confirmation["gated_delta_rmse"] = np.where(
        confirmation["development_eligible"], confirmation["delta_rmse"], 0.0
    )
    confirmation["gated_delta_correlation"] = np.where(
        confirmation["development_eligible"], confirmation["delta_correlation"], 0.0
    )
    predictions = pd.DataFrame(prediction_rows)
    coefficients = pd.DataFrame(coefficient_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    development.to_csv(output_dir / "development_2024_grid.csv", index=False)
    confirmation.to_csv(output_dir / "confirmation_2025.csv", index=False)
    predictions.to_csv(output_dir / "confirmation_2025_predictions.csv", index=False)
    coefficients.to_csv(output_dir / "selected_model_coefficients.csv", index=False)

    weight = confirmation["confirmation_rows"].clip(lower=1.0)
    eligible = confirmation[confirmation["development_eligible"]].copy()
    eligible_weight = eligible["confirmation_rows"].clip(lower=1.0)
    aggregate = {
        "component_side_tests": len(confirmation),
        "confirmation_rows_total": int(confirmation["confirmation_rows"].sum()),
        "raw_rmse_wins": int((confirmation["delta_rmse"] < 0).sum()),
        "raw_correlation_wins": int((confirmation["delta_correlation"] > 0).sum()),
        "development_eligible": int(confirmation["development_eligible"].sum()),
        "eligible_rmse_wins": int((eligible["delta_rmse"] < 0).sum()),
        "eligible_correlation_wins": int((eligible["delta_correlation"] > 0).sum()),
        "eligible_both_wins": int(
            ((eligible["delta_rmse"] < 0) & (eligible["delta_correlation"] > 0)).sum()
        ),
        "eligible_weighted_mean_delta_rmse": float(
            np.average(eligible["delta_rmse"], weights=eligible_weight)
        ),
        "eligible_weighted_mean_delta_correlation": float(
            np.average(eligible["delta_correlation"], weights=eligible_weight)
        ),
        "gated_weighted_mean_delta_rmse": float(
            np.average(confirmation["gated_delta_rmse"], weights=weight)
        ),
        "gated_weighted_mean_delta_correlation": float(
            np.average(confirmation["gated_delta_correlation"], weights=weight)
        ),
    }
    manifest = {
        "status": "research_only_not_live",
        "selection": "fit 2023, select lane and alpha on 2024",
        "confirmation": "refit 2023-24, evaluate selected challenger once on 2025",
        "paired_sample": True,
        "min_rtss_chances": min_rtss_chances,
        "min_target_poss": min_target_poss,
        "q_contract": {
            "qSQ": "league-average-shooter shot context, per project definition",
            "qSP": "includes shooter, per project definition",
            "shooter_adjustment": "qSP - qSQ",
            "zones": "MID and 3P only",
        },
        "q_season_weighted_means": {
            str(year): {
                column: float(
                    np.average(
                        group[column],
                        weights=group["off_chances"].clip(lower=1.0),
                    )
                )
                for column in (
                    "rtss_mid_qsq_epv",
                    "rtss_mid_shooter_adjustment_epv",
                    "rtss_three_qsq_epv",
                    "rtss_three_shooter_adjustment_epv",
                )
            }
            for year, group in rtss.groupby("season")
        },
        "aggregate": aggregate,
    }
    (output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return development, confirmation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--target-dir", type=Path, default=BOX_PRIOR_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-rtss-chances", type=float, default=100.0)
    parser.add_argument("--min-target-poss", type=float, default=250.0)
    args = parser.parse_args()
    _, confirmation = evaluate(
        args.features.resolve(),
        args.target_dir.resolve(),
        args.output_dir.resolve(),
        args.min_rtss_chances,
        args.min_target_poss,
    )
    print(
        confirmation[
            [
                "prefix",
                "side",
                "selected_challenger",
                "confirmation_rows",
                "delta_rmse",
                "delta_correlation",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
