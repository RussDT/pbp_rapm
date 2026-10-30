#!/usr/bin/env python3
"""Build a PBP-only shot-role difficulty proxy and test future offensive TS.

The proxy intentionally ignores make/miss, points, and made-only assist tags
when assigning a shot family.  A feature-season player's shot-family mix is
priced with the immediately preceding season's league results, then compared
with the existing Synergy-derived ``playtype_diff`` on the same player-season
and future-oTS samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

try:
    from .analyze_shot_clock_difficulty_ots import (
        ANNUAL_TARGET_TEMPLATE,
        DEFAULT_BOX,
        DEFAULT_ZTS,
        PIPELINE_ROOT,
        RIDGE_ALPHAS,
        _weighted_metrics,
        clustered_forward_bootstrap_rmse_change,
        load_zts_features,
        season_forward_ridge,
    )
    from .process_rapm_blocks.common import _base_processing, build_shot_flags_df
except ImportError:  # Direct script execution.
    from analyze_shot_clock_difficulty_ots import (
        ANNUAL_TARGET_TEMPLATE,
        DEFAULT_BOX,
        DEFAULT_ZTS,
        PIPELINE_ROOT,
        RIDGE_ALPHAS,
        _weighted_metrics,
        clustered_forward_bootstrap_rmse_change,
        load_zts_features,
        season_forward_ridge,
    )
    from process_rapm_blocks.common import _base_processing, build_shot_flags_df


DEFAULT_RAW_DIR = PIPELINE_ROOT / "raw_data"
DEFAULT_OUTPUT = PIPELINE_ROOT / "results" / "pbp_playtype_difficulty" / "2023_2026"
DEFAULT_ROLE_CACHE = (
    PIPELINE_ROOT / "cache" / "five_man_fit" / "player_role_features_2014_2026.parquet"
)
DEFAULT_PASSING_RS = Path(
    "/Users/russ/Documents/Codex/basketball/data/csvs/passing_rs_updated.csv"
)
DEFAULT_MERGED_RS_2025 = Path(
    "/Users/russ/Documents/Codex/basketball/nba/pipelines/core/"
    "nba_rapm/data/outputs/merged_rs_2025.csv"
)
PRICE_SHRINK_FGA = 100.0
PBP_CATEGORIES = [
    "CUT_2",
    "SELF_CREATED_3",
    "OTHER_3",
    "SELF_CREATED_2",
    "OTHER_RIM_2",
    "OTHER_2",
]
SELF_CREATED_PATTERN = (
    r"pull.?up|step.?back|turnaround|fadeaway|driving|floating|floater|hook"
)
CUT_PATTERN = r"cutting|alley oop"
RIM_PATTERN = r"dunk|layup|tip"
CREATION_CONTROL_FEATURES = [
    "on_ball_time_pct",
    "potential_assists_per100",
    "rim_assists_per100",
]
FLOATING_SCORING_FEATURES = ["pts100", "TSA100"]


def classify_pbp_shots(frame: pd.DataFrame, is_three: pd.Series) -> pd.Series:
    """Assign outcome-invariant broad shot families from NBA descriptions."""
    description = (
        frame.get("home_description", pd.Series("", index=frame.index))
        .fillna("")
        .astype(str)
        .str.cat(
            frame.get("visitor_description", pd.Series("", index=frame.index))
            .fillna("")
            .astype(str),
            sep=" ",
        )
        .str.lower()
    )
    cut = description.str.contains(CUT_PATTERN, regex=True)
    self_created = description.str.contains(SELF_CREATED_PATTERN, regex=True)
    rim = description.str.contains(RIM_PATTERN, regex=True)
    conditions = [
        cut & ~is_three,
        self_created & is_three,
        is_three,
        self_created & ~is_three,
        rim & ~is_three,
    ]
    choices = [
        "CUT_2",
        "SELF_CREATED_3",
        "OTHER_3",
        "SELF_CREATED_2",
        "OTHER_RIM_2",
    ]
    return pd.Series(
        np.select(conditions, choices, default="OTHER_2"),
        index=frame.index,
        name="pbp_playtype_category",
    )


def load_pbp_shots(raw_path: Path, season: int) -> pd.DataFrame:
    """Load regular-season FGA with only shooter, category, and realized points."""
    raw = _base_processing(raw_path)
    if raw is None:
        raise RuntimeError(f"Could not process {raw_path}")
    flags = build_shot_flags_df(raw)
    shots = raw.loc[flags["is_fga"]].copy()
    shot_flags = flags.loc[shots.index]
    shots["pbp_playtype_category"] = classify_pbp_shots(
        shots, shot_flags["is_3pt"].astype(bool)
    )
    shots["actual_points"] = shot_flags["shot_points"].astype(float)
    shots["player_id"] = pd.to_numeric(shots["player1_id"], errors="coerce")
    shots["player_name"] = shots["player1_name"].fillna("").astype(str)
    shots["season"] = int(season)
    return shots.dropna(subset=["player_id"])[
        [
            "season",
            "game_id",
            "event_num",
            "player_id",
            "player_name",
            "pbp_playtype_category",
            "actual_points",
        ]
    ]


def estimate_category_prices(
    pricing_shots: pd.DataFrame,
    *,
    pricing_season: int,
    feature_season: int,
    shrink_fga: float = PRICE_SHRINK_FGA,
) -> pd.DataFrame:
    """Estimate category points/FGA from one completed prior season."""
    global_mean = float(pricing_shots["actual_points"].mean())
    prices = (
        pricing_shots.groupby("pbp_playtype_category", sort=False)["actual_points"]
        .agg(["mean", "count"])
        .reindex(PBP_CATEGORIES)
    )
    prices["mean"] = prices["mean"].fillna(global_mean)
    prices["count"] = prices["count"].fillna(0)
    prices["expected_points_per_fga"] = (
        prices["count"] * prices["mean"] + shrink_fga * global_mean
    ) / (prices["count"] + shrink_fga)
    prices["league_points_per_fga"] = global_mean
    prices["category_value_vs_league"] = (
        prices["expected_points_per_fga"] - global_mean
    )
    prices["pricing_season"] = int(pricing_season)
    prices["feature_season"] = int(feature_season)
    return prices.reset_index()


def aggregate_player_difficulty(
    feature_shots: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    feature_season: int,
) -> pd.DataFrame:
    """Price current shot mix with prior-season category values."""
    price_columns = [
        "pbp_playtype_category",
        "expected_points_per_fga",
        "league_points_per_fga",
    ]
    scored = feature_shots.merge(
        prices[price_columns],
        on="pbp_playtype_category",
        how="left",
        validate="many_to_one",
    )
    if scored["expected_points_per_fga"].isna().any():
        raise ValueError("Every feature-season PBP category must have a prior price")
    rows = []
    for player_id, group in scored.groupby("player_id", sort=False):
        expected = float(group["expected_points_per_fga"].mean())
        league = float(group["league_points_per_fga"].iloc[0])
        counts = group["pbp_playtype_category"].value_counts()
        row = {
            "feature_season": int(feature_season),
            "player_id": int(player_id),
            "pbp_player_name": group["player_name"].iloc[-1],
            "pbp_fga": int(len(group)),
            "pbp_expected_points_per_fga": expected,
            "pbp_league_points_per_fga": league,
            "pbp_difficulty_points_per_fga": league - expected,
            # TS% = points / (2 * TSA); this is the FGA-only equivalent in
            # percentage points, used on the same scale as playtype_diff.
            "pbp_difficulty_pp": 50.0 * (league - expected),
        }
        for category in PBP_CATEGORIES:
            row[f"share_{category.lower()}"] = float(counts.get(category, 0) / len(group))
        rows.append(row)
    return pd.DataFrame(rows)


def build_pbp_feature_panel(
    raw_dir: Path,
    *,
    first_feature_season: int,
    last_feature_season: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build lag-priced PBP difficulty for consecutive feature seasons."""
    required_seasons = range(first_feature_season - 1, last_feature_season + 1)
    shots_by_season = {}
    for season in required_seasons:
        raw_path = raw_dir / f"NBA{str(season)[-2:]}.parquet"
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)
        shots_by_season[season] = load_pbp_shots(raw_path, season)
    feature_parts = []
    price_parts = []
    for feature_season in range(first_feature_season, last_feature_season + 1):
        pricing_season = feature_season - 1
        prices = estimate_category_prices(
            shots_by_season[pricing_season],
            pricing_season=pricing_season,
            feature_season=feature_season,
        )
        feature = aggregate_player_difficulty(
            shots_by_season[feature_season],
            prices,
            feature_season=feature_season,
        )
        feature_parts.append(feature)
        price_parts.append(prices)
    return (
        pd.concat(feature_parts, ignore_index=True),
        pd.concat(price_parts, ignore_index=True),
    )


def _annual_target_path(target_season: int) -> Path:
    return PIPELINE_ROOT / "results" / ANNUAL_TARGET_TEMPLATE.format(
        year=str(target_season)[-2:]
    )


def build_forward_rows(
    pbp_features: pd.DataFrame,
    *,
    box_path: Path,
    zts_path: Path,
    first_feature_season: int,
    last_feature_season: int,
    min_fga: int,
    min_minutes: float,
    creation_controls: pd.DataFrame | None = None,
) -> pd.DataFrame:
    box_columns = [
        "year",
        "nba_id",
        "ShortName",
        "Minutes",
        "Pts75",
        "TSA100",
    ]
    box = pd.read_csv(box_path, usecols=box_columns)
    box = box[box["year"].between(first_feature_season, last_feature_season)].copy()
    box = box.rename(columns={"year": "feature_season", "nba_id": "player_id"})
    box["player_id"] = pd.to_numeric(box["player_id"], errors="coerce")
    zts = load_zts_features(
        zts_path, first_feature_season, last_feature_season
    ).rename(columns={"year": "feature_season"})
    box = box.merge(
        zts,
        on=["feature_season", "player_id"],
        how="left",
        validate="one_to_one",
    )
    rows = box.merge(
        pbp_features,
        on=["feature_season", "player_id"],
        how="inner",
        validate="one_to_one",
    )
    target_parts = []
    for target_season in range(first_feature_season + 1, last_feature_season + 2):
        target = pd.read_csv(_annual_target_path(target_season))[
            ["player_id", "player_name", "off", "off_poss"]
        ].rename(
            columns={
                "player_name": "target_player_name",
                "off": "future_oTS",
                "off_poss": "future_off_poss",
            }
        )
        target["player_id"] = pd.to_numeric(target["player_id"], errors="coerce")
        target["target_season"] = int(target_season)
        target["feature_season"] = int(target_season - 1)
        target_parts.append(target)
    rows = rows.merge(
        pd.concat(target_parts, ignore_index=True),
        on=["feature_season", "player_id"],
        how="inner",
        validate="one_to_one",
    )
    rows = rows[
        rows["pbp_fga"].ge(min_fga)
        & pd.to_numeric(rows["Minutes"], errors="coerce").ge(min_minutes)
        & rows["future_off_poss"].gt(0)
    ].copy()
    if creation_controls is not None:
        rows = rows.merge(
            creation_controls,
            on=["feature_season", "player_id"],
            how="left",
            validate="one_to_one",
        )
    # The decompressed/floating scoring model uses raw points and TSA as
    # separate regressors, allowing the future-oTS fit to learn its own
    # scoring break-even. Pts75 is raw personal points per 75 possessions.
    rows["pts100"] = pd.to_numeric(rows["Pts75"], errors="coerce") * (100.0 / 75.0)
    rows["pbp_role_added_100"] = (
        2 * rows["TSA100"] * rows["pbp_difficulty_pp"] / 100
    )
    rows["synergy_role_added_100"] = (
        2 * rows["TSA100"] * rows["playtype_diff"] / 100
    )
    return rows


def load_creation_controls(
    role_cache: Path,
    passing_rs: Path,
    merged_rs_2025: Path,
    *,
    first_feature_season: int,
    last_feature_season: int,
) -> pd.DataFrame:
    """Load feature-season creator-role controls without target-season data."""
    role = pd.read_parquet(role_cache)[
        [
            "year",
            "nba_id",
            "PASSING_on-ball-time%",
            "PASSING_Potential Assists/100",
        ]
    ].rename(
        columns={
            "year": "feature_season",
            "nba_id": "player_id",
            "PASSING_on-ball-time%": "on_ball_time_pct",
            "PASSING_Potential Assists/100": "potential_assists_per100",
        }
    )
    historical_rim = pd.read_csv(
        passing_rs,
        usecols=["year", "nba_id", "AtRimAssists/100"],
    ).rename(
        columns={
            "year": "feature_season",
            "nba_id": "player_id",
            "AtRimAssists/100": "rim_assists_per100",
        }
    )
    current_rim = pd.read_csv(
        merged_rs_2025,
        usecols=["year", "nba_id", "PASSING_AtRimAssists/100"],
    ).rename(
        columns={
            "year": "feature_season",
            "nba_id": "player_id",
            "PASSING_AtRimAssists/100": "rim_assists_per100",
        }
    )
    rim = pd.concat([historical_rim, current_rim], ignore_index=True)

    for frame in (role, rim):
        for column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    role = role.drop_duplicates(["feature_season", "player_id"], keep="last")
    rim = rim.drop_duplicates(["feature_season", "player_id"], keep="last")
    controls = role.merge(
        rim,
        on=["feature_season", "player_id"],
        how="inner",
        validate="one_to_one",
    )
    controls = controls[
        controls["feature_season"].between(
            first_feature_season, last_feature_season
        )
    ].copy()
    return controls.dropna(subset=CREATION_CONTROL_FEATURES).reset_index(drop=True)


def build_model_features(*, include_creation_controls: bool) -> dict[str, list[str]]:
    controls = CREATION_CONTROL_FEATURES if include_creation_controls else []
    return {
        "box_scoring_baseline": FLOATING_SCORING_FEATURES + controls,
        "pbp_role_added": FLOATING_SCORING_FEATURES
        + ["pbp_role_added_100"]
        + controls,
        "synergy_role_added": FLOATING_SCORING_FEATURES
        + ["synergy_role_added_100"]
        + controls,
        "both_roles_added": FLOATING_SCORING_FEATURES
        + ["pbp_role_added_100", "synergy_role_added_100"]
        + controls,
    }


def _select_alpha(
    X: np.ndarray, y: np.ndarray, weights: np.ndarray
) -> float:
    folds = KFold(n_splits=5, shuffle=True, random_state=43)
    losses = []
    for alpha in RIDGE_ALPHAS:
        fold_losses = []
        for fit_index, validation_index in folds.split(X):
            model = Ridge(alpha=float(alpha)).fit(
                X[fit_index], y[fit_index], sample_weight=weights[fit_index]
            )
            prediction = model.predict(X[validation_index])
            fold_losses.append(
                mean_squared_error(
                    y[validation_index],
                    prediction,
                    sample_weight=weights[validation_index],
                )
            )
        losses.append(float(np.mean(fold_losses)))
    return float(RIDGE_ALPHAS[int(np.argmin(losses))])


def expanding_season_forward_ridge(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    first_test_target_season: int = 2025,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Fit only earlier target transitions and score each later transition."""
    complete = frame.dropna(
        subset=feature_columns
        + ["future_oTS", "future_off_poss", "feature_season", "target_season"]
    ).copy()
    prediction_parts = []
    selected_alphas = []
    test_seasons = [
        int(year)
        for year in sorted(complete["target_season"].unique())
        if int(year) >= first_test_target_season
    ]
    for test_target_season in test_seasons:
        train = complete[complete["target_season"].lt(test_target_season)]
        test = complete[complete["target_season"].eq(test_target_season)]
        if train.empty or test.empty:
            continue
        X_train_raw = train[feature_columns].to_numpy(dtype=float)
        X_test_raw = test[feature_columns].to_numpy(dtype=float)
        y_train = train["future_oTS"].to_numpy(dtype=float)
        weights_train = train["future_off_poss"].to_numpy(dtype=float)
        scaler = StandardScaler().fit(X_train_raw)
        X_train = scaler.transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        selected_alpha = _select_alpha(X_train, y_train, weights_train)
        selected_alphas.append(selected_alpha)
        model = Ridge(alpha=selected_alpha).fit(
            X_train, y_train, sample_weight=weights_train
        )
        prediction = test[
            [
                "player_id",
                "target_player_name",
                "feature_season",
                "target_season",
                "future_oTS",
                "future_off_poss",
            ]
        ].copy()
        prediction["prediction"] = model.predict(X_test)
        prediction["residual"] = (
            prediction["future_oTS"] - prediction["prediction"]
        )
        prediction_parts.append(prediction)
    if not prediction_parts:
        raise ValueError("No chronological season-forward folds were available")
    predictions = pd.concat(prediction_parts, ignore_index=True)
    metrics = _weighted_metrics(
        predictions["future_oTS"].to_numpy(dtype=float),
        predictions["prediction"].to_numpy(dtype=float),
        predictions["future_off_poss"].to_numpy(dtype=float),
    )
    metrics.update(
        {
            "player_seasons": int(len(predictions)),
            "players": int(predictions["player_id"].nunique()),
            "features": int(len(feature_columns)),
            "median_selected_alpha": float(np.median(selected_alphas)),
        }
    )
    return metrics, predictions


def joint_season_forward_ridge(
    frame: pd.DataFrame,
    model_features: dict[str, list[str]],
    *,
    chronological: bool,
    first_test_target_season: int = 2025,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every model with the alpha selected by the box baseline per fold."""
    required = sorted(
        {
            column
            for feature_columns in model_features.values()
            for column in feature_columns
        }
        | {"future_oTS", "future_off_poss", "feature_season", "target_season"}
    )
    complete = frame.dropna(subset=required).copy()
    prediction_parts = []
    selected_alphas = []
    test_seasons = [
        int(year)
        for year in sorted(complete["target_season"].unique())
        if int(year) >= first_test_target_season
    ]
    for test_target_season in test_seasons:
        if chronological:
            train = complete[complete["target_season"].lt(test_target_season)]
        else:
            train = complete[~complete["target_season"].eq(test_target_season)]
        test = complete[complete["target_season"].eq(test_target_season)]
        if train.empty or test.empty:
            continue
        baseline_features = model_features["box_scoring_baseline"]
        baseline_scaler = StandardScaler().fit(
            train[baseline_features].to_numpy(dtype=float)
        )
        baseline_X = baseline_scaler.transform(
            train[baseline_features].to_numpy(dtype=float)
        )
        y_train = train["future_oTS"].to_numpy(dtype=float)
        weights_train = train["future_off_poss"].to_numpy(dtype=float)
        selected_alpha = _select_alpha(baseline_X, y_train, weights_train)
        selected_alphas.append(
            {"target_season": test_target_season, "selected_alpha": selected_alpha}
        )
        for model_name, feature_columns in model_features.items():
            scaler = StandardScaler().fit(
                train[feature_columns].to_numpy(dtype=float)
            )
            X_train = scaler.transform(train[feature_columns].to_numpy(dtype=float))
            X_test = scaler.transform(test[feature_columns].to_numpy(dtype=float))
            model = Ridge(alpha=selected_alpha).fit(
                X_train, y_train, sample_weight=weights_train
            )
            prediction = test[
                [
                    "player_id",
                    "target_player_name",
                    "feature_season",
                    "target_season",
                    "future_oTS",
                    "future_off_poss",
                ]
            ].copy()
            prediction.insert(0, "model", model_name)
            prediction["selected_alpha"] = selected_alpha
            prediction["prediction"] = model.predict(X_test)
            prediction["residual"] = (
                prediction["future_oTS"] - prediction["prediction"]
            )
            prediction_parts.append(prediction)
    if not prediction_parts:
        raise ValueError("No joint season-forward folds were available")
    predictions = pd.concat(prediction_parts, ignore_index=True)
    metric_rows = []
    for model_name, group in predictions.groupby("model", sort=False):
        metrics = _weighted_metrics(
            group["future_oTS"].to_numpy(dtype=float),
            group["prediction"].to_numpy(dtype=float),
            group["future_off_poss"].to_numpy(dtype=float),
        )
        metrics.update(
            {
                "model": model_name,
                "player_seasons": int(len(group)),
                "players": int(group["player_id"].nunique()),
                "features": int(len(model_features[model_name])),
                "median_selected_alpha": float(
                    np.median([row["selected_alpha"] for row in selected_alphas])
                ),
                "alpha_selection_model": "box_scoring_baseline",
            }
        )
        metric_rows.append(metrics)
    return pd.DataFrame(metric_rows), predictions


def role_coefficient_audit(
    frame: pd.DataFrame,
    model_features: dict[str, list[str]],
    *,
    first_test_target_season: int = 2025,
) -> pd.DataFrame:
    """Export fold coefficients for the decompressed floating-three model."""
    required = sorted(
        set().union(*(set(features) for features in model_features.values()))
        | {"future_oTS", "future_off_poss", "target_season"}
    )
    complete = frame.dropna(subset=required).copy()
    rows = []
    for test_target_season in sorted(complete["target_season"].unique()):
        test_target_season = int(test_target_season)
        if test_target_season < first_test_target_season:
            continue
        train = complete[complete["target_season"].lt(test_target_season)]
        test = complete[complete["target_season"].eq(test_target_season)]
        if train.empty or test.empty:
            continue
        baseline_features = model_features["box_scoring_baseline"]
        baseline_scaler = StandardScaler().fit(
            train[baseline_features].to_numpy(dtype=float)
        )
        baseline_X = baseline_scaler.transform(
            train[baseline_features].to_numpy(dtype=float)
        )
        y_train = train["future_oTS"].to_numpy(dtype=float)
        weights_train = train["future_off_poss"].to_numpy(dtype=float)
        selected_alpha = _select_alpha(baseline_X, y_train, weights_train)
        for model_name, feature_columns in model_features.items():
            role_features = [
                column for column in feature_columns if column.endswith("_role_added_100")
            ]
            if not role_features:
                continue
            scaler = StandardScaler().fit(
                train[feature_columns].to_numpy(dtype=float)
            )
            X_train = scaler.transform(train[feature_columns].to_numpy(dtype=float))
            model = Ridge(alpha=selected_alpha).fit(
                X_train, y_train, sample_weight=weights_train
            )
            standardized = dict(zip(feature_columns, model.coef_))
            physical = {
                feature: float(coef / scale)
                for feature, coef, scale in zip(
                    feature_columns, model.coef_, scaler.scale_
                )
            }
            points_beta = physical["pts100"]
            for role_feature in role_features:
                role_beta = physical[role_feature]
                rows.append(
                    {
                        "target_season": test_target_season,
                        "model": model_name,
                        "role_feature": role_feature,
                        "train_rows": int(len(train)),
                        "test_rows": int(len(test)),
                        "selected_alpha": float(selected_alpha),
                        "standardized_role_coefficient": float(
                            standardized[role_feature]
                        ),
                        "physical_role_coefficient": role_beta,
                        "physical_points_coefficient": points_beta,
                        "role_to_points_ratio": (
                            role_beta / points_beta if points_beta != 0 else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def compare_models(
    predictions: pd.DataFrame,
    specs: list[tuple[str, str, str]],
    *,
    evaluation: str,
) -> pd.DataFrame:
    rows = []
    for comparison, baseline_model, challenger_model in specs:
        baseline = predictions[predictions["model"].eq(baseline_model)]
        challenger = predictions[predictions["model"].eq(challenger_model)]
        paired = baseline[
            [
                "player_id",
                "target_season",
                "future_oTS",
                "future_off_poss",
                "prediction",
            ]
        ].merge(
            challenger[["player_id", "target_season", "prediction"]],
            on=["player_id", "target_season"],
            suffixes=("_baseline", "_challenger"),
            validate="one_to_one",
        )
        y = paired["future_oTS"].to_numpy(dtype=float)
        weights = paired["future_off_poss"].to_numpy(dtype=float)
        baseline_rmse = float(
            np.sqrt(
                np.average(
                    (y - paired["prediction_baseline"].to_numpy(dtype=float)) ** 2,
                    weights=weights,
                )
            )
        )
        challenger_rmse = float(
            np.sqrt(
                np.average(
                    (y - paired["prediction_challenger"].to_numpy(dtype=float)) ** 2,
                    weights=weights,
                )
            )
        )
        lower, upper, probability = clustered_forward_bootstrap_rmse_change(
            baseline, challenger
        )
        rows.append(
            {
                "evaluation": evaluation,
                "comparison": comparison,
                "baseline_model": baseline_model,
                "challenger_model": challenger_model,
                "rows": int(len(paired)),
                "players": int(paired["player_id"].nunique()),
                "baseline_rmse": baseline_rmse,
                "challenger_rmse": challenger_rmse,
                "rmse_change": challenger_rmse - baseline_rmse,
                "rmse_change_ci95_lower": lower,
                "rmse_change_ci95_upper": upper,
                "bootstrap_probability_improves": probability,
            }
        )
    return pd.DataFrame(rows)


def overlap_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    common = frame.dropna(subset=["pbp_difficulty_pp", "playtype_diff"])
    groups = [("all", common)] + [
        (str(int(season)), group)
        for season, group in common.groupby("feature_season", sort=True)
    ]
    for label, group in groups:
        rows.append(
            {
                "feature_season": label,
                "player_seasons": int(len(group)),
                "players": int(group["player_id"].nunique()),
                "pearson_pbp_vs_synergy": group["pbp_difficulty_pp"].corr(
                    group["playtype_diff"]
                ),
                "spearman_pbp_vs_synergy": group["pbp_difficulty_pp"].corr(
                    group["playtype_diff"], method="spearman"
                ),
            }
        )
    return pd.DataFrame(rows)


def per_target_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, target_season), group in predictions.groupby(
        ["model", "target_season"], sort=True
    ):
        metrics = _weighted_metrics(
            group["future_oTS"].to_numpy(dtype=float),
            group["prediction"].to_numpy(dtype=float),
            group["future_off_poss"].to_numpy(dtype=float),
        )
        metrics.update(
            {
                "model": model,
                "target_season": int(target_season),
                "rows": int(len(group)),
                "players": int(group["player_id"].nunique()),
            }
        )
        rows.append(metrics)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--box", type=Path, default=DEFAULT_BOX)
    parser.add_argument("--zts", type=Path, default=DEFAULT_ZTS)
    parser.add_argument("--first-feature-season", type=int, default=2023)
    parser.add_argument("--last-feature-season", type=int, default=2025)
    parser.add_argument("--min-fga", type=int, default=100)
    parser.add_argument("--min-minutes", type=float, default=250)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--role-cache", type=Path, default=DEFAULT_ROLE_CACHE)
    parser.add_argument("--passing-rs", type=Path, default=DEFAULT_PASSING_RS)
    parser.add_argument(
        "--merged-rs-2025", type=Path, default=DEFAULT_MERGED_RS_2025
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pbp_features, category_prices = build_pbp_feature_panel(
        args.raw_dir,
        first_feature_season=args.first_feature_season,
        last_feature_season=args.last_feature_season,
    )
    creation_controls = load_creation_controls(
        args.role_cache,
        args.passing_rs,
        args.merged_rs_2025,
        first_feature_season=args.first_feature_season,
        last_feature_season=args.last_feature_season,
    )
    forward = build_forward_rows(
        pbp_features,
        box_path=args.box,
        zts_path=args.zts,
        first_feature_season=args.first_feature_season,
        last_feature_season=args.last_feature_season,
        min_fga=args.min_fga,
        min_minutes=args.min_minutes,
        creation_controls=creation_controls,
    )

    model_features = build_model_features(include_creation_controls=True)
    required = sorted(
        {
            column
            for feature_columns in model_features.values()
            for column in feature_columns
        }
        | {"future_oTS", "future_off_poss"}
    )
    common = forward.dropna(subset=required).copy()

    model_specific_chronological_metric_rows = []
    model_specific_chronological_prediction_parts = []
    for model, features in model_features.items():
        metrics, predictions = expanding_season_forward_ridge(common, features)
        metrics["model"] = model
        model_specific_chronological_metric_rows.append(metrics)
        predictions.insert(0, "model", model)
        model_specific_chronological_prediction_parts.append(predictions)
    model_specific_chronological_metrics = pd.DataFrame(
        model_specific_chronological_metric_rows
    )
    model_specific_chronological_predictions = pd.concat(
        model_specific_chronological_prediction_parts, ignore_index=True
    )
    chronological_metrics, chronological_predictions = joint_season_forward_ridge(
        common,
        model_features,
        chronological=True,
    )

    control_ladder_rows = []
    control_ladder_coefficient_parts = []
    control_groups = {
        "shooting_only": [],
        "plus_on_ball": ["on_ball_time_pct"],
        "plus_potential_assists": [
            "on_ball_time_pct",
            "potential_assists_per100",
        ],
        "plus_rim_assists": CREATION_CONTROL_FEATURES,
    }
    for control_stage, controls in control_groups.items():
        stage_features = build_model_features(include_creation_controls=False)
        stage_features = {
            model: features + controls for model, features in stage_features.items()
        }
        stage_required = sorted(
            set().union(*(set(features) for features in stage_features.values()))
            | {"future_oTS", "future_off_poss"}
        )
        stage_common = forward.dropna(subset=stage_required).copy()
        stage_metrics, _ = joint_season_forward_ridge(
            stage_common,
            stage_features,
            chronological=True,
        )
        stage_coefficients = role_coefficient_audit(stage_common, stage_features)
        stage_coefficients.insert(0, "control_stage", control_stage)
        control_ladder_coefficient_parts.append(stage_coefficients)
        rmse = stage_metrics.set_index("model")["rmse"]
        control_ladder_rows.append(
            {
                "control_stage": control_stage,
                "rows": int(len(stage_common)),
                "players": int(stage_common["player_id"].nunique()),
                "box_rmse": float(rmse["box_scoring_baseline"]),
                "pbp_rmse": float(rmse["pbp_role_added"]),
                "synergy_rmse": float(rmse["synergy_role_added"]),
                "pbp_minus_box": float(
                    rmse["pbp_role_added"] - rmse["box_scoring_baseline"]
                ),
                "synergy_minus_box": float(
                    rmse["synergy_role_added"] - rmse["box_scoring_baseline"]
                ),
                "pbp_minus_synergy": float(
                    rmse["pbp_role_added"] - rmse["synergy_role_added"]
                ),
            }
        )
    control_ladder = pd.DataFrame(control_ladder_rows)
    control_ladder_coefficients = pd.concat(
        control_ladder_coefficient_parts, ignore_index=True
    )

    legacy = common[common["target_season"].ge(2025)].copy()
    model_specific_legacy_metric_rows = []
    model_specific_legacy_prediction_parts = []
    for model, features in model_features.items():
        metrics, predictions = season_forward_ridge(legacy, features)
        metrics["model"] = model
        model_specific_legacy_metric_rows.append(metrics)
        predictions.insert(0, "model", model)
        model_specific_legacy_prediction_parts.append(predictions)
    model_specific_legacy_metrics = pd.DataFrame(model_specific_legacy_metric_rows)
    model_specific_legacy_predictions = pd.concat(
        model_specific_legacy_prediction_parts, ignore_index=True
    )
    legacy_metrics, legacy_predictions = joint_season_forward_ridge(
        legacy,
        model_features,
        chronological=False,
    )

    comparisons = [
        ("pbp_vs_box", "box_scoring_baseline", "pbp_role_added"),
        ("synergy_vs_box", "box_scoring_baseline", "synergy_role_added"),
        ("synergy_after_pbp", "pbp_role_added", "both_roles_added"),
        ("pbp_after_synergy", "synergy_role_added", "both_roles_added"),
        ("synergy_vs_pbp", "pbp_role_added", "synergy_role_added"),
    ]
    comparison_table = pd.concat(
        [
            compare_models(
                chronological_predictions,
                comparisons,
                evaluation="chronological_expanding",
            ),
            compare_models(
                legacy_predictions,
                comparisons,
                evaluation="legacy_leave_one_transition_out",
            ),
        ],
        ignore_index=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pbp_features.to_csv(args.output_dir / "pbp_playtype_player_season.csv", index=False)
    category_prices.to_csv(args.output_dir / "pbp_playtype_category_prices.csv", index=False)
    common.to_csv(args.output_dir / "pbp_playtype_future_ots_analysis.csv", index=False)
    overlap_summary(common).to_csv(
        args.output_dir / "pbp_vs_synergy_overlap.csv", index=False
    )
    chronological_metrics.to_csv(
        args.output_dir / "chronological_metrics.csv", index=False
    )
    chronological_predictions.to_csv(
        args.output_dir / "chronological_predictions.csv", index=False
    )
    control_ladder.to_csv(
        args.output_dir / "creation_control_ladder_metrics.csv", index=False
    )
    control_ladder_coefficients.to_csv(
        args.output_dir / "creation_control_ladder_role_coefficients.csv",
        index=False,
    )
    model_specific_chronological_metrics.to_csv(
        args.output_dir / "chronological_model_specific_alpha_metrics.csv",
        index=False,
    )
    model_specific_chronological_predictions.to_csv(
        args.output_dir / "chronological_model_specific_alpha_predictions.csv",
        index=False,
    )
    legacy_metrics.to_csv(args.output_dir / "legacy_loto_metrics.csv", index=False)
    legacy_predictions.to_csv(
        args.output_dir / "legacy_loto_predictions.csv", index=False
    )
    model_specific_legacy_metrics.to_csv(
        args.output_dir / "legacy_loto_model_specific_alpha_metrics.csv",
        index=False,
    )
    model_specific_legacy_predictions.to_csv(
        args.output_dir / "legacy_loto_model_specific_alpha_predictions.csv",
        index=False,
    )
    comparison_table.to_csv(
        args.output_dir / "incremental_comparisons.csv", index=False
    )
    per_target_metrics(chronological_predictions).to_csv(
        args.output_dir / "chronological_per_target_metrics.csv", index=False
    )
    role_coefficient_audit(common, model_features).to_csv(
        args.output_dir / "chronological_role_coefficients.csv", index=False
    )
    manifest = {
        "feature_seasons": [
            int(args.first_feature_season),
            int(args.last_feature_season),
        ],
        "pricing_rule": "immediately preceding regular season only",
        "category_price_shrink_fga": PRICE_SHRINK_FGA,
        "categories": PBP_CATEGORIES,
        "category_inputs": "shot value plus outcome-invariant NBA PBP description tokens",
        "forbidden_category_inputs": [
            "make_or_miss",
            "actual_points",
            "score",
            "made_only_assist_tag",
        ],
        "difficulty_definition": "50 * (prior-season league points/FGA - prior-priced player shot-mix points/FGA)",
        "forward_targets": list(
            range(args.first_feature_season + 1, args.last_feature_season + 2)
        ),
        "chronological_scored_targets": [2025, 2026],
        "chronological_training": "2025 fit on 2023->2024; 2026 fit on 2023->2024 plus 2024->2025",
        "legacy_surface": "2024->2025 and 2025->2026, each fit on the other transition",
        "primary_alpha_rule": "select alpha on box baseline training rows, then apply it to every candidate in that fold",
        "sensitivity_alpha_rule": "select alpha independently for each model",
        "sample_rule": {
            "minimum_feature_fga": int(args.min_fga),
            "minimum_feature_minutes": float(args.min_minutes),
            "exact_common_rows_across_models": True,
        },
        "creation_controls": CREATION_CONTROL_FEATURES,
        "creation_control_rule": "feature-season values included identically in every primary model; standardized and ridge-shrunk with the other regressors",
        "floating_scoring_features": FLOATING_SCORING_FEATURES,
        "pts100_definition": "Pts75 * 100 / 75",
        "role_coefficient_rule": "floating-three decomposition: raw pts100, TSA100, and 2*TSA100*difficulty/100 enter separately; divide the physical role coefficient by the physical pts100 coefficient for the scoring-relative role multiplier",
        "role_cache": str(args.role_cache),
        "rim_assist_sources": [str(args.passing_rs), str(args.merged_rs_2025)],
        "box_source": str(args.box),
        "synergy_source": str(args.zts),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print("\nChronological expanding evaluation:\n")
    print(chronological_metrics.sort_values("rmse").to_string(index=False))
    print("\nLegacy leave-one-transition-out evaluation:\n")
    print(legacy_metrics.sort_values("rmse").to_string(index=False))
    print("\nPaired incremental comparisons:\n")
    print(comparison_table.to_string(index=False))
    print("\nPBP-versus-Synergy overlap:\n")
    print(overlap_summary(common).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
