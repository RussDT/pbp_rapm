#!/usr/bin/env python3
"""Build player shot-clock difficulty and test it against offensive TS RAPM.

The primary stat uses the raw rule-based clock because that surface is
available for every endpoint and remains stable through zero. A secondary
surface uses the shot-calibrated clock with a four-second floor, reflecting
the calibration's weak extreme tail.

Clock expectations are cross-fit by game. Every endpoint is priced by an
isotonic clock curve trained on other games from the same season and reset
family. Positive difficulty means the player's attempt mix was expected to be
harder than the league-average attempt mix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, RepeatedKFold
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "nba_pipeline"
DEFAULT_OUTCOMES = (
    PIPELINE_ROOT
    / "results"
    / "turnover_shot_clock"
    / "2024_2026"
    / "turnover_shot_clock_shooting_outcomes.parquet"
)
DEFAULT_TARGET = (
    PIPELINE_ROOT
    / "validation"
    / "rapms_reference"
    / "master_results"
    / "weighted_factors_24_26_all_rb_se_a2000_4000.csv"
)
DEFAULT_BOX = Path(
    "/Users/russ/Documents/Codex/basketball/data/csvs/player_stats_export.csv"
)
DEFAULT_ZTS = Path(
    "/Users/russ/Documents/Codex/basketball/nba/pipelines/core/"
    "nba_rapm/data/outputs/playtype_weighted_rts.csv"
)
DEFAULT_PLAYTYPE_ROWS = Path(
    "/Users/russ/Documents/Codex/basketball/nba/pipelines/core/"
    "nba_rapm/data/outputs/playtype_rows_with_estimates.csv"
)
DEFAULT_OUTPUT = PIPELINE_ROOT / "results" / "shot_clock_difficulty" / "2024_2026"
ANNUAL_TARGET_TEMPLATE = "ts_{year}_all_rb_se_a2000_4000_results.csv"
BOX_FEATURES = ["TSA100", "Pts75", "TS_added_100", "2PA_100", "3PA_100", "FTA_100"]
ZTS_COLUMNS = ["playtype_weighted_rTS", "playtype_diff", "pt_adj_rTS"]
ROLE_GROUPS = ["creation", "shooting", "ee", "transition"]
RIDGE_ALPHAS = np.logspace(-3, 4, 15)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return np.nan
    return float(np.average(values[valid], weights=weights[valid]))


def _prepare_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    required = {
        "game_id",
        "season",
        "shooter_id",
        "shooter_name",
        "points",
        "tsa",
        "shot_clock_est",
        "clock",
        "rule_reset_len",
    }
    missing = sorted(required - set(outcomes.columns))
    if missing:
        raise ValueError(f"Shooting outcomes are missing columns: {missing}")
    work = outcomes.copy()
    for column in [
        "shooter_id",
        "points",
        "tsa",
        "shot_clock_est",
        "clock",
        "rule_reset_len",
    ]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work[
        work["shooter_id"].gt(0)
        & work["tsa"].gt(0)
        & work["points"].notna()
        & work["shot_clock_est"].notna()
    ].copy()
    work["shooter_id"] = work["shooter_id"].astype(np.int64)
    work["reset_family"] = np.where(work["rule_reset_len"].eq(14), 14, 24)
    work["raw_rule_clock"] = work["shot_clock_est"].clip(0, work["reset_family"])
    work["calibrated_floor4_clock"] = (
        work["clock"].clip(lower=4).clip(upper=work["reset_family"])
    )
    work["endpoint_tsx2"] = work["points"] / work["tsa"]
    return work.reset_index(drop=True)


def crossfit_clock_expectation(
    outcomes: pd.DataFrame,
    *,
    clock_column: str,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Return game-held-out clock expectation and league reference per row."""
    work = outcomes.copy()
    work["expected_tsx2"] = np.nan
    work["league_tsx2"] = np.nan
    unique_games = work["game_id"].nunique()
    folds = GroupKFold(n_splits=min(n_splits, unique_games))
    groups = work["game_id"].astype(str)
    for train_index, validation_index in folds.split(work, groups=groups):
        train = work.iloc[train_index]
        validation = work.iloc[validation_index]
        league_tsx2 = _weighted_mean(train["endpoint_tsx2"], train["tsa"])
        work.loc[work.index[validation_index], "league_tsx2"] = league_tsx2
        for group_key, validation_group in validation.groupby(
            ["season", "reset_family"], sort=False
        ):
            fit = train[
                train["season"].eq(group_key[0])
                & train["reset_family"].eq(group_key[1])
            ]
            if len(fit) < 100:
                fit = train[train["reset_family"].eq(group_key[1])]
            model = IsotonicRegression(increasing=True, out_of_bounds="clip")
            model.fit(
                fit[clock_column].to_numpy(dtype=float),
                fit["endpoint_tsx2"].to_numpy(dtype=float),
                sample_weight=fit["tsa"].to_numpy(dtype=float),
            )
            work.loc[validation_group.index, "expected_tsx2"] = model.predict(
                validation_group[clock_column].to_numpy(dtype=float)
            )
    if work[["expected_tsx2", "league_tsx2"]].isna().any().any():
        raise ValueError("Cross-fit clock pricing left null predictions")
    work["expected_points"] = work["expected_tsx2"] * work["tsa"]
    work["league_reference_points"] = work["league_tsx2"] * work["tsa"]
    work["clock_adjustment_points"] = (
        work["league_reference_points"] - work["expected_points"]
    )
    work["clock_adjusted_points"] = work["points"] + work["clock_adjustment_points"]
    return work


def aggregate_player_clock_stats(scored: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Aggregate endpoint prices to player-season with an exact TS identity."""
    work = scored.copy()
    work["clock_weight"] = work["tsa"] * work["clock_used"]
    work["late4_weight"] = work["tsa"] * work["clock_used"].le(4)
    work["late8_weight"] = work["tsa"] * work["clock_used"].le(8)
    grouped = (
        work.groupby(["season", "shooter_id"], sort=False)
        .agg(
            shooter_name=("shooter_name", "last"),
            endpoints=("game_id", "size"),
            games=("game_id", "nunique"),
            tsa=("tsa", "sum"),
            points=("points", "sum"),
            expected_points=("expected_points", "sum"),
            league_reference_points=("league_reference_points", "sum"),
            clock_adjustment_points=("clock_adjustment_points", "sum"),
            clock_adjusted_points=("clock_adjusted_points", "sum"),
            clock_weight=("clock_weight", "sum"),
            late4_weight=("late4_weight", "sum"),
            late8_weight=("late8_weight", "sum"),
        )
        .reset_index()
    )
    grouped[f"{prefix}_raw_ts_pct"] = 100 * grouped["points"] / (2 * grouped["tsa"])
    grouped[f"{prefix}_clock_expected_ts_pct"] = (
        100 * grouped["expected_points"] / (2 * grouped["tsa"])
    )
    grouped[f"{prefix}_league_reference_ts_pct"] = (
        100 * grouped["league_reference_points"] / (2 * grouped["tsa"])
    )
    grouped[f"{prefix}_shot_clock_difficulty_pp"] = (
        100 * grouped["clock_adjustment_points"] / (2 * grouped["tsa"])
    )
    grouped[f"{prefix}_clock_adjusted_ts_pct"] = (
        100 * grouped["clock_adjusted_points"] / (2 * grouped["tsa"])
    )
    grouped[f"{prefix}_average_clock"] = grouped["clock_weight"] / grouped["tsa"]
    grouped[f"{prefix}_late4_tsa_share"] = grouped["late4_weight"] / grouped["tsa"]
    grouped[f"{prefix}_late8_tsa_share"] = grouped["late8_weight"] / grouped["tsa"]
    gap = (
        grouped[f"{prefix}_raw_ts_pct"]
        + grouped[f"{prefix}_shot_clock_difficulty_pp"]
        - grouped[f"{prefix}_clock_adjusted_ts_pct"]
    ).abs().max()
    if float(gap) > 1e-10:
        raise ValueError(f"Clock-adjusted TS identity failed for {prefix}: {gap}")
    keep = [
        "season",
        "shooter_id",
        "shooter_name",
        "endpoints",
        "games",
        "tsa",
        "points",
    ] + [column for column in grouped if column.startswith(f"{prefix}_")]
    return grouped[keep]


def aggregate_across_seasons(player_season: pd.DataFrame, prefix: str) -> pd.DataFrame:
    value_columns = [
        f"{prefix}_raw_ts_pct",
        f"{prefix}_clock_expected_ts_pct",
        f"{prefix}_league_reference_ts_pct",
        f"{prefix}_shot_clock_difficulty_pp",
        f"{prefix}_clock_adjusted_ts_pct",
        f"{prefix}_average_clock",
        f"{prefix}_late4_tsa_share",
        f"{prefix}_late8_tsa_share",
    ]
    rows = []
    for shooter_id, group in player_season.groupby("shooter_id", sort=False):
        weights = group["tsa"]
        row = {
            "player_id": int(shooter_id),
            "player_name": group.loc[group["season"].idxmax(), "shooter_name"],
            "seasons": int(group["season"].nunique()),
            "endpoints": int(group["endpoints"].sum()),
            "games": int(group["games"].sum()),
            "tsa": float(weights.sum()),
            "points": float(group["points"].sum()),
        }
        for column in value_columns:
            row[column] = _weighted_mean(group[column], weights)
        rows.append(row)
    return pd.DataFrame(rows)


def load_zts_features(path: Path, start_year: int, end_year: int) -> pd.DataFrame:
    zts = pd.read_csv(path, usecols=["year", "nba_id"] + ZTS_COLUMNS)
    zts = zts[zts["year"].between(start_year, end_year)].copy()
    zts = zts.rename(columns={"nba_id": "player_id"})
    zts["player_id"] = pd.to_numeric(zts["player_id"], errors="coerce")
    for column in ZTS_COLUMNS:
        zts[column] = pd.to_numeric(zts[column], errors="coerce")
    return zts.dropna(subset=["player_id"]).drop_duplicates()


def aggregate_box_features(
    path: Path,
    start_year: int,
    end_year: int,
    *,
    zts_path: Path = DEFAULT_ZTS,
) -> pd.DataFrame:
    columns = ["year", "nba_id", "ShortName", "Minutes"] + BOX_FEATURES
    box = pd.read_csv(path, usecols=columns)
    box = box[box["year"].between(start_year, end_year)].copy()
    box = box.rename(columns={"nba_id": "player_id"})
    box["player_id"] = pd.to_numeric(box["player_id"], errors="coerce")
    box["Minutes"] = pd.to_numeric(box["Minutes"], errors="coerce").fillna(0)
    zts = load_zts_features(zts_path, start_year, end_year)
    box = box.merge(
        zts,
        on=["year", "player_id"],
        how="left",
        validate="one_to_one",
    )
    rows = []
    for player_id, group in box.dropna(subset=["player_id"]).groupby("player_id", sort=False):
        weights = group["Minutes"]
        row = {
            "player_id": int(player_id),
            "box_player_name": group.loc[group["year"].idxmax(), "ShortName"],
            "box_minutes": float(weights.sum()),
        }
        for column in BOX_FEATURES + ZTS_COLUMNS:
            row[column] = _weighted_mean(
                pd.to_numeric(group[column], errors="coerce"), weights
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_zts_role_numbers(
    path: Path,
    zts_path: Path,
    *,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Return the four player-season role buckets underlying the zTS surface."""
    columns = [
        "year",
        "nba_id",
        "player_name",
        "playtype_group",
        "TSA",
        "TSA100",
        "ts_percent",
        "playtype_avg_ts",
        "playtype_rTS",
    ]
    rows = pd.read_csv(path, usecols=columns)
    rows = rows[
        rows["year"].between(start_year, end_year)
        & rows["playtype_group"].isin(ROLE_GROUPS)
    ].copy()
    for column in [
        "nba_id",
        "TSA",
        "TSA100",
        "ts_percent",
        "playtype_avg_ts",
        "playtype_rTS",
    ]:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows = rows.dropna(subset=["nba_id", "TSA", "playtype_group"])
    rows["actual_ts_weight"] = rows["TSA"] * rows["ts_percent"]
    rows["expected_ts_weight"] = rows["TSA"] * rows["playtype_avg_ts"]
    rows["rts_weight"] = rows["TSA"] * rows["playtype_rTS"]
    role = (
        rows.groupby(["year", "nba_id", "playtype_group"], sort=False)
        .agg(
            player_name=("player_name", "last"),
            role_tsa=("TSA", "sum"),
            role_tsa100=("TSA100", "sum"),
            actual_ts_weight=("actual_ts_weight", "sum"),
            expected_ts_weight=("expected_ts_weight", "sum"),
            rts_weight=("rts_weight", "sum"),
        )
        .reset_index()
        .rename(columns={"nba_id": "player_id", "playtype_group": "role"})
    )
    role["role_actual_ts_pct"] = 100 * role["actual_ts_weight"] / role["role_tsa"]
    role["role_expected_ts_pct"] = (
        100 * role["expected_ts_weight"] / role["role_tsa"]
    )
    role["role_rts_pct_pts"] = 100 * role["rts_weight"] / role["role_tsa"]
    role["role_attempt_share"] = role["role_tsa"] / role.groupby(
        ["year", "player_id"]
    )["role_tsa"].transform("sum")
    zts = load_zts_features(zts_path, start_year, end_year).rename(
        columns={"year": "year"}
    )
    role = role.merge(zts, on=["year", "player_id"], how="left", validate="many_to_one")
    return role[
        [
            "year",
            "player_id",
            "player_name",
            "role",
            "role_tsa",
            "role_tsa100",
            "role_attempt_share",
            "role_actual_ts_pct",
            "role_expected_ts_pct",
            "role_rts_pct_pts",
            *ZTS_COLUMNS,
        ]
    ]


def _weighted_metrics(y: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, prediction, sample_weight=weights))),
        "mae": float(mean_absolute_error(y, prediction, sample_weight=weights)),
        "r2": float(r2_score(y, prediction, sample_weight=weights)),
        "pearson": float(np.corrcoef(y, prediction)[0, 1]),
        "spearman": float(spearmanr(y, prediction).statistic),
    }


def paired_bootstrap_rmse_change(
    baseline: pd.DataFrame,
    challenger: pd.DataFrame,
    *,
    draws: int = 4000,
    seed: int = 71,
) -> tuple[float, float, float]:
    paired = baseline[["player_id", "oTS", "off_poss", "prediction"]].merge(
        challenger[["player_id", "prediction"]],
        on="player_id",
        how="inner",
        suffixes=("_baseline", "_challenger"),
        validate="one_to_one",
    )
    y = paired["oTS"].to_numpy(dtype=float)
    weights = paired["off_poss"].to_numpy(dtype=float)
    baseline_prediction = paired["prediction_baseline"].to_numpy(dtype=float)
    challenger_prediction = paired["prediction_challenger"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    changes = np.empty(draws, dtype=float)
    for draw in range(draws):
        sample = rng.integers(0, len(paired), size=len(paired))
        baseline_rmse = np.sqrt(
            np.average((y[sample] - baseline_prediction[sample]) ** 2, weights=weights[sample])
        )
        challenger_rmse = np.sqrt(
            np.average((y[sample] - challenger_prediction[sample]) ** 2, weights=weights[sample])
        )
        changes[draw] = challenger_rmse - baseline_rmse
    lower, upper = np.quantile(changes, [0.025, 0.975])
    return float(lower), float(upper), float((changes < 0).mean())


def clustered_forward_bootstrap_rmse_change(
    baseline: pd.DataFrame,
    challenger: pd.DataFrame,
    *,
    draws: int = 4000,
    seed: int = 83,
) -> tuple[float, float, float]:
    paired = baseline[
        ["player_id", "target_season", "future_oTS", "future_off_poss", "prediction"]
    ].merge(
        challenger[["player_id", "target_season", "prediction"]],
        on=["player_id", "target_season"],
        how="inner",
        suffixes=("_baseline", "_challenger"),
        validate="one_to_one",
    )
    player_ids = paired["player_id"].unique()
    row_indices = {
        player_id: group.index.to_numpy(dtype=int)
        for player_id, group in paired.groupby("player_id", sort=False)
    }
    y = paired["future_oTS"].to_numpy(dtype=float)
    weights = paired["future_off_poss"].to_numpy(dtype=float)
    baseline_prediction = paired["prediction_baseline"].to_numpy(dtype=float)
    challenger_prediction = paired["prediction_challenger"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    changes = np.empty(draws, dtype=float)
    for draw in range(draws):
        sampled_players = rng.choice(player_ids, size=len(player_ids), replace=True)
        sample = np.concatenate([row_indices[player_id] for player_id in sampled_players])
        baseline_rmse = np.sqrt(
            np.average(
                (y[sample] - baseline_prediction[sample]) ** 2,
                weights=weights[sample],
            )
        )
        challenger_rmse = np.sqrt(
            np.average(
                (y[sample] - challenger_prediction[sample]) ** 2,
                weights=weights[sample],
            )
        )
        changes[draw] = challenger_rmse - baseline_rmse
    lower, upper = np.quantile(changes, [0.025, 0.975])
    return float(lower), float(upper), float((changes < 0).mean())


def cross_validated_ridge(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    target_column: str = "oTS",
    weight_column: str = "off_poss",
    n_repeats: int = 5,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    complete = frame.dropna(subset=feature_columns + [target_column, weight_column]).copy()
    complete = complete.reset_index(drop=True)
    X = complete[feature_columns].to_numpy(dtype=float)
    y = complete[target_column].to_numpy(dtype=float)
    weights = complete[weight_column].to_numpy(dtype=float)
    prediction_sum = np.zeros(len(complete), dtype=float)
    prediction_count = np.zeros(len(complete), dtype=int)
    selected_alphas = []
    outer = RepeatedKFold(n_splits=5, n_repeats=n_repeats, random_state=29)
    for train_index, validation_index in outer.split(X):
        scaler = StandardScaler().fit(X[train_index])
        X_train = scaler.transform(X[train_index])
        X_validation = scaler.transform(X[validation_index])
        inner = KFold(n_splits=4, shuffle=True, random_state=37)
        alpha_losses = []
        for alpha in RIDGE_ALPHAS:
            fold_losses = []
            for fit_index, test_index in inner.split(X_train):
                model = Ridge(alpha=float(alpha))
                model.fit(
                    X_train[fit_index],
                    y[train_index][fit_index],
                    sample_weight=weights[train_index][fit_index],
                )
                fold_prediction = model.predict(X_train[test_index])
                fold_losses.append(
                    mean_squared_error(
                        y[train_index][test_index],
                        fold_prediction,
                        sample_weight=weights[train_index][test_index],
                    )
                )
            alpha_losses.append(float(np.mean(fold_losses)))
        selected_alpha = float(RIDGE_ALPHAS[int(np.argmin(alpha_losses))])
        selected_alphas.append(selected_alpha)
        model = Ridge(alpha=selected_alpha)
        model.fit(X_train, y[train_index], sample_weight=weights[train_index])
        prediction_sum[validation_index] += model.predict(X_validation)
        prediction_count[validation_index] += 1
    predictions = prediction_sum / prediction_count
    metrics = _weighted_metrics(y, predictions, weights)
    metrics.update(
        {
            "players": int(len(complete)),
            "features": int(len(feature_columns)),
            "median_selected_alpha": float(np.median(selected_alphas)),
        }
    )
    full_scaler = StandardScaler().fit(X)
    final_alpha = metrics["median_selected_alpha"]
    final_model = Ridge(alpha=final_alpha).fit(
        full_scaler.transform(X), y, sample_weight=weights
    )
    coefficients = pd.DataFrame(
        {
            "feature": feature_columns,
            "standardized_coefficient": final_model.coef_,
        }
    )
    predictions_frame = complete[["player_id", "player_name", target_column, weight_column]].copy()
    predictions_frame["prediction"] = predictions
    predictions_frame["residual"] = predictions_frame[target_column] - predictions
    return metrics, coefficients, predictions_frame


def build_season_forward_rows(
    player_season: pd.DataFrame,
    box_path: Path,
    *,
    zts_path: Path = DEFAULT_ZTS,
    start_year: int,
    end_year: int,
    min_tsa: float = 100,
    min_minutes: float = 250,
) -> pd.DataFrame:
    box_columns = ["year", "nba_id", "ShortName", "Minutes"] + BOX_FEATURES
    box = pd.read_csv(box_path, usecols=box_columns)
    box = box[box["year"].between(start_year, end_year - 1)].copy()
    box = box.rename(columns={"year": "feature_season", "nba_id": "player_id"})
    zts = load_zts_features(zts_path, start_year, end_year - 1).rename(
        columns={"year": "feature_season"}
    )
    box = box.merge(
        zts,
        on=["feature_season", "player_id"],
        how="left",
        validate="one_to_one",
    )
    rows = player_season.rename(
        columns={"season": "feature_season", "shooter_id": "player_id"}
    ).merge(box, on=["feature_season", "player_id"], how="inner", validate="one_to_one")
    target_parts = []
    for target_year in range(start_year + 1, end_year + 1):
        target_path = PIPELINE_ROOT / "results" / ANNUAL_TARGET_TEMPLATE.format(
            year=str(target_year)[-2:]
        )
        target = pd.read_csv(target_path)[
            ["player_id", "player_name", "off", "off_poss"]
        ].rename(
            columns={
                "player_name": "target_player_name",
                "off": "future_oTS",
                "off_poss": "future_off_poss",
            }
        )
        target["target_season"] = target_year
        target["feature_season"] = target_year - 1
        target_parts.append(target)
    future = pd.concat(target_parts, ignore_index=True)
    rows = rows.merge(
        future,
        on=["feature_season", "player_id"],
        how="inner",
        validate="one_to_one",
    )
    rows = rows[
        rows["tsa"].ge(min_tsa)
        & pd.to_numeric(rows["Minutes"], errors="coerce").ge(min_minutes)
        & rows["future_off_poss"].gt(0)
    ].copy()
    for prefix in ["raw", "cal4"]:
        rows[f"{prefix}_clock_adjusted_TS_added_100"] = (
            rows["TS_added_100"]
            + (
                rows[f"{prefix}_clock_adjusted_ts_pct"]
                - rows[f"{prefix}_raw_ts_pct"]
            )
            / 100
            * 2
            * rows["TSA100"]
        )
    rows["zTS_added_100"] = (
        rows["TS_added_100"]
        + 2 * rows["TSA100"] * rows["playtype_diff"] / 100
    )
    rows["zTS_raw_clock_adjusted_TS_added_100"] = (
        rows["zTS_added_100"]
        + 2
        * rows["TSA100"]
        * rows["raw_shot_clock_difficulty_pp"]
        / 100
    )
    rows["zts_sample_marker"] = np.where(rows["playtype_diff"].notna(), 0.0, np.nan)
    return rows


def season_forward_ridge(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[dict[str, float], pd.DataFrame]:
    complete = frame.dropna(
        subset=feature_columns
        + ["future_oTS", "future_off_poss", "feature_season", "target_season"]
    ).copy()
    prediction_parts = []
    selected_alphas = []
    for test_target_season in sorted(complete["target_season"].unique()):
        train = complete[~complete["target_season"].eq(test_target_season)]
        test = complete[complete["target_season"].eq(test_target_season)]
        X_train_raw = train[feature_columns].to_numpy(dtype=float)
        X_test_raw = test[feature_columns].to_numpy(dtype=float)
        y_train = train["future_oTS"].to_numpy(dtype=float)
        weights_train = train["future_off_poss"].to_numpy(dtype=float)
        scaler = StandardScaler().fit(X_train_raw)
        X_train = scaler.transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        folds = KFold(n_splits=5, shuffle=True, random_state=43)
        losses = []
        for alpha in RIDGE_ALPHAS:
            fold_losses = []
            for fit_index, validation_index in folds.split(X_train):
                model = Ridge(alpha=float(alpha)).fit(
                    X_train[fit_index],
                    y_train[fit_index],
                    sample_weight=weights_train[fit_index],
                )
                prediction = model.predict(X_train[validation_index])
                fold_losses.append(
                    mean_squared_error(
                        y_train[validation_index],
                        prediction,
                        sample_weight=weights_train[validation_index],
                    )
                )
            losses.append(float(np.mean(fold_losses)))
        selected_alpha = float(RIDGE_ALPHAS[int(np.argmin(losses))])
        selected_alphas.append(selected_alpha)
        model = Ridge(alpha=selected_alpha).fit(
            X_train, y_train, sample_weight=weights_train
        )
        prediction = test[
            ["player_id", "target_player_name", "feature_season", "target_season", "future_oTS", "future_off_poss"]
        ].copy()
        prediction["prediction"] = model.predict(X_test)
        prediction["residual"] = prediction["future_oTS"] - prediction["prediction"]
        prediction_parts.append(prediction)
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


def build_incremental_comparisons(
    predictions: pd.DataFrame,
    comparisons: list[tuple[str, str, str]],
    *,
    forward: bool,
) -> pd.DataFrame:
    """Compare specified models on their exact common held-out rows."""
    rows = []
    for comparison, baseline_model, challenger_model in comparisons:
        baseline = predictions[predictions["model"].eq(baseline_model)]
        challenger = predictions[predictions["model"].eq(challenger_model)]
        if forward:
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
                how="inner",
                suffixes=("_baseline", "_challenger"),
                validate="one_to_one",
            )
            target_column = "future_oTS"
            weight_column = "future_off_poss"
            lower, upper, probability = clustered_forward_bootstrap_rmse_change(
                baseline, challenger
            )
        else:
            paired = baseline[
                ["player_id", "oTS", "off_poss", "prediction"]
            ].merge(
                challenger[["player_id", "prediction"]],
                on="player_id",
                how="inner",
                suffixes=("_baseline", "_challenger"),
                validate="one_to_one",
            )
            target_column = "oTS"
            weight_column = "off_poss"
            lower, upper, probability = paired_bootstrap_rmse_change(
                baseline, challenger
            )
        target = paired[target_column].to_numpy(dtype=float)
        weights = paired[weight_column].to_numpy(dtype=float)
        baseline_rmse = float(
            np.sqrt(
                np.average(
                    (target - paired["prediction_baseline"].to_numpy(dtype=float))
                    ** 2,
                    weights=weights,
                )
            )
        )
        challenger_rmse = float(
            np.sqrt(
                np.average(
                    (target - paired["prediction_challenger"].to_numpy(dtype=float))
                    ** 2,
                    weights=weights,
                )
            )
        )
        rows.append(
            {
                "evaluation": "season_forward" if forward else "same_window",
                "comparison": comparison,
                "baseline_model": baseline_model,
                "challenger_model": challenger_model,
                "rows": len(paired),
                "players": paired["player_id"].nunique(),
                "baseline_rmse": baseline_rmse,
                "challenger_rmse": challenger_rmse,
                "rmse_change": challenger_rmse - baseline_rmse,
                "rmse_change_ci95_lower": lower,
                "rmse_change_ci95_upper": upper,
                "bootstrap_probability_improves": probability,
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--box", type=Path, default=DEFAULT_BOX)
    parser.add_argument("--zts", type=Path, default=DEFAULT_ZTS)
    parser.add_argument("--playtype-rows", type=Path, default=DEFAULT_PLAYTYPE_ROWS)
    parser.add_argument("--start-year", type=int, default=2024)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--min-tsa", type=float, default=200)
    parser.add_argument("--min-box-minutes", type=float, default=500)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outcomes = _prepare_outcomes(pd.read_parquet(args.outcomes))
    surfaces = {
        "raw": "raw_rule_clock",
        "cal4": "calibrated_floor4_clock",
    }
    player_season_parts = []
    aggregate_parts = []
    curve_parts = []
    for prefix, clock_column in surfaces.items():
        scored = crossfit_clock_expectation(outcomes, clock_column=clock_column)
        scored["clock_used"] = scored[clock_column]
        player_season = aggregate_player_clock_stats(scored, prefix)
        aggregate = aggregate_across_seasons(player_season, prefix)
        player_season_parts.append(player_season)
        aggregate_parts.append(aggregate)
        curve = (
            scored.assign(clock_second=scored[clock_column].round().astype(int))
            .groupby(["season", "reset_family", "clock_second"], sort=False)
            .apply(
                lambda group: pd.Series(
                    {
                        "endpoints": len(group),
                        "tsa": group["tsa"].sum(),
                        "actual_tsx2": _weighted_mean(group["endpoint_tsx2"], group["tsa"]),
                        "crossfit_expected_tsx2": _weighted_mean(group["expected_tsx2"], group["tsa"]),
                    }
                ),
                include_groups=False,
            )
            .reset_index()
        )
        curve.insert(0, "clock_surface", prefix)
        curve_parts.append(curve)

    player_season = player_season_parts[0].merge(
        player_season_parts[1].drop(
            columns=["shooter_name", "endpoints", "games", "tsa", "points"]
        ),
        on=["season", "shooter_id"],
        how="outer",
        validate="one_to_one",
    )
    aggregate = aggregate_parts[0].merge(
        aggregate_parts[1].drop(
            columns=["player_name", "seasons", "endpoints", "games", "tsa", "points"]
        ),
        on="player_id",
        how="outer",
        validate="one_to_one",
    )
    box = aggregate_box_features(
        args.box,
        args.start_year,
        args.end_year,
        zts_path=args.zts,
    )
    target = pd.read_csv(args.target)[
        ["player_id", "player_name", "oTS", "off_poss"]
    ].copy()
    target["player_id"] = pd.to_numeric(target["player_id"], errors="coerce").astype("Int64")
    analysis = target.merge(aggregate, on="player_id", how="inner", suffixes=("", "_clock"))
    analysis = analysis.merge(box, on="player_id", how="inner", validate="one_to_one")
    analysis = analysis[
        analysis["tsa"].ge(args.min_tsa)
        & analysis["box_minutes"].ge(args.min_box_minutes)
    ].copy()
    for prefix in surfaces:
        analysis[f"{prefix}_clock_adjusted_TS_added_100"] = (
            analysis["TS_added_100"]
            + (
                analysis[f"{prefix}_clock_adjusted_ts_pct"]
                - analysis[f"{prefix}_raw_ts_pct"]
            )
            / 100
            * 2
            * analysis["TSA100"]
        )
    analysis["zTS_added_100"] = (
        analysis["TS_added_100"]
        + 2 * analysis["TSA100"] * analysis["playtype_diff"] / 100
    )
    analysis["zTS_raw_clock_adjusted_TS_added_100"] = (
        analysis["zTS_added_100"]
        + 2
        * analysis["TSA100"]
        * analysis["raw_shot_clock_difficulty_pp"]
        / 100
    )
    analysis["zts_sample_marker"] = np.where(
        analysis["playtype_diff"].notna(), 0.0, np.nan
    )

    model_features = {
        "raw_ts_only": ["raw_raw_ts_pct"],
        "raw_clock_adjusted_ts_only": ["raw_clock_adjusted_ts_pct"],
        "cal4_clock_adjusted_ts_only": ["cal4_clock_adjusted_ts_pct"],
        "box_scoring_baseline": BOX_FEATURES,
        "box_scoring_raw_clock_adjusted": [
            "TSA100",
            "Pts75",
            "raw_clock_adjusted_TS_added_100",
            "2PA_100",
            "3PA_100",
            "FTA_100",
        ],
        "box_scoring_cal4_clock_adjusted": [
            "TSA100",
            "Pts75",
            "cal4_clock_adjusted_TS_added_100",
            "2PA_100",
            "3PA_100",
            "FTA_100",
        ],
        "box_scoring_plus_raw_difficulty": BOX_FEATURES
        + ["raw_shot_clock_difficulty_pp"],
        "box_scoring_plus_raw_average_clock": BOX_FEATURES + ["raw_average_clock"],
        "box_scoring_plus_raw_late4_share": BOX_FEATURES + ["raw_late4_tsa_share"],
        "box_scoring_plus_raw_clock_context": BOX_FEATURES
        + [
            "raw_shot_clock_difficulty_pp",
            "raw_average_clock",
            "raw_late4_tsa_share",
        ],
        "box_scoring_baseline_zts_sample": BOX_FEATURES + ["zts_sample_marker"],
        "box_scoring_zts_adjusted": [
            "TSA100",
            "Pts75",
            "zTS_added_100",
            "2PA_100",
            "3PA_100",
            "FTA_100",
        ],
        "box_scoring_zts_plus_raw_clock_adjusted": [
            "TSA100",
            "Pts75",
            "zTS_raw_clock_adjusted_TS_added_100",
            "2PA_100",
            "3PA_100",
            "FTA_100",
        ],
        "box_scoring_zts_plus_raw_difficulty": [
            "TSA100",
            "Pts75",
            "zTS_added_100",
            "2PA_100",
            "3PA_100",
            "FTA_100",
            "raw_shot_clock_difficulty_pp",
        ],
    }
    metric_rows = []
    coefficient_parts = []
    prediction_parts = []
    for model_name, features in model_features.items():
        metrics, coefficients, predictions = cross_validated_ridge(analysis, features)
        metrics["model"] = model_name
        metric_rows.append(metrics)
        coefficients.insert(0, "model", model_name)
        coefficient_parts.append(coefficients)
        predictions.insert(0, "model", model_name)
        prediction_parts.append(predictions)
    metrics = pd.DataFrame(metric_rows)
    all_predictions = pd.concat(prediction_parts, ignore_index=True)
    baseline_rmse = float(
        metrics.loc[metrics["model"].eq("box_scoring_baseline"), "rmse"].iloc[0]
    )
    metrics["rmse_change_vs_box_baseline"] = metrics["rmse"] - baseline_rmse
    baseline_predictions = all_predictions[
        all_predictions["model"].eq("box_scoring_baseline")
    ]
    intervals = []
    for model_name in metrics["model"]:
        if model_name == "box_scoring_baseline":
            intervals.append((0.0, 0.0, np.nan))
            continue
        challenger = all_predictions[all_predictions["model"].eq(model_name)]
        intervals.append(
            paired_bootstrap_rmse_change(baseline_predictions, challenger)
        )
    metrics[[
        "rmse_change_ci95_lower",
        "rmse_change_ci95_upper",
        "bootstrap_probability_improves",
    ]] = intervals

    forward_rows = build_season_forward_rows(
        player_season,
        args.box,
        zts_path=args.zts,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    forward_model_features = {
        "box_scoring_baseline": BOX_FEATURES,
        "box_scoring_raw_clock_adjusted": [
            "TSA100",
            "Pts75",
            "raw_clock_adjusted_TS_added_100",
            "2PA_100",
            "3PA_100",
            "FTA_100",
        ],
        "box_scoring_cal4_clock_adjusted": [
            "TSA100",
            "Pts75",
            "cal4_clock_adjusted_TS_added_100",
            "2PA_100",
            "3PA_100",
            "FTA_100",
        ],
        "box_scoring_plus_raw_difficulty": BOX_FEATURES
        + ["raw_shot_clock_difficulty_pp"],
        "box_scoring_plus_raw_clock_context": BOX_FEATURES
        + [
            "raw_shot_clock_difficulty_pp",
            "raw_average_clock",
            "raw_late4_tsa_share",
        ],
        "box_scoring_baseline_zts_sample": BOX_FEATURES + ["zts_sample_marker"],
        "box_scoring_zts_adjusted": [
            "TSA100",
            "Pts75",
            "zTS_added_100",
            "2PA_100",
            "3PA_100",
            "FTA_100",
        ],
        "box_scoring_zts_plus_raw_clock_adjusted": [
            "TSA100",
            "Pts75",
            "zTS_raw_clock_adjusted_TS_added_100",
            "2PA_100",
            "3PA_100",
            "FTA_100",
        ],
        "box_scoring_zts_plus_raw_difficulty": [
            "TSA100",
            "Pts75",
            "zTS_added_100",
            "2PA_100",
            "3PA_100",
            "FTA_100",
            "raw_shot_clock_difficulty_pp",
        ],
    }
    forward_metric_rows = []
    forward_prediction_parts = []
    for model_name, features in forward_model_features.items():
        forward_metrics, forward_predictions = season_forward_ridge(
            forward_rows, features
        )
        forward_metrics["model"] = model_name
        forward_metric_rows.append(forward_metrics)
        forward_predictions.insert(0, "model", model_name)
        forward_prediction_parts.append(forward_predictions)
    forward_metrics = pd.DataFrame(forward_metric_rows)
    all_forward_predictions = pd.concat(forward_prediction_parts, ignore_index=True)
    forward_baseline_rmse = float(
        forward_metrics.loc[
            forward_metrics["model"].eq("box_scoring_baseline"), "rmse"
        ].iloc[0]
    )
    forward_metrics["rmse_change_vs_box_baseline"] = (
        forward_metrics["rmse"] - forward_baseline_rmse
    )
    forward_baseline_predictions = all_forward_predictions[
        all_forward_predictions["model"].eq("box_scoring_baseline")
    ]
    forward_intervals = []
    for model_name in forward_metrics["model"]:
        if model_name == "box_scoring_baseline":
            forward_intervals.append((0.0, 0.0, np.nan))
            continue
        challenger = all_forward_predictions[
            all_forward_predictions["model"].eq(model_name)
        ]
        forward_intervals.append(
            clustered_forward_bootstrap_rmse_change(
                forward_baseline_predictions, challenger
            )
        )
    forward_metrics[[
        "rmse_change_ci95_lower",
        "rmse_change_ci95_upper",
        "bootstrap_probability_improves",
    ]] = forward_intervals

    comparison_specs = [
        (
            "zts_vs_box_on_zts_sample",
            "box_scoring_baseline_zts_sample",
            "box_scoring_zts_adjusted",
        ),
        (
            "clock_replacement_on_top_of_zts",
            "box_scoring_zts_adjusted",
            "box_scoring_zts_plus_raw_clock_adjusted",
        ),
        (
            "clock_feature_on_top_of_zts",
            "box_scoring_zts_adjusted",
            "box_scoring_zts_plus_raw_difficulty",
        ),
    ]
    incremental = pd.concat(
        [
            build_incremental_comparisons(
                all_predictions, comparison_specs, forward=False
            ),
            build_incremental_comparisons(
                all_forward_predictions, comparison_specs, forward=True
            ),
        ],
        ignore_index=True,
    )

    role_numbers = build_zts_role_numbers(
        args.playtype_rows,
        args.zts,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    overlap_rows = []
    overlap = forward_rows.dropna(
        subset=["playtype_diff", "raw_shot_clock_difficulty_pp"]
    )
    overlap_groups = [("all", overlap)] + [
        (str(int(year)), group)
        for year, group in overlap.groupby("feature_season", sort=True)
    ]
    for season, group in overlap_groups:
        overlap_rows.append(
            {
                "feature_season": season,
                "player_seasons": len(group),
                "players": group["player_id"].nunique(),
                "pearson_playtype_vs_clock": group["playtype_diff"].corr(
                    group["raw_shot_clock_difficulty_pp"]
                ),
                "spearman_playtype_vs_clock": group["playtype_diff"].corr(
                    group["raw_shot_clock_difficulty_pp"], method="spearman"
                ),
            }
        )
    overlap_summary = pd.DataFrame(overlap_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    player_season.to_csv(args.output_dir / "shot_clock_difficulty_player_season.csv", index=False)
    aggregate.to_csv(args.output_dir / "shot_clock_difficulty_24_26.csv", index=False)
    analysis.to_csv(args.output_dir / "shot_clock_difficulty_ots_analysis.csv", index=False)
    forward_rows.to_csv(
        args.output_dir / "shot_clock_difficulty_future_ots_analysis.csv", index=False
    )
    pd.concat(curve_parts, ignore_index=True).to_csv(
        args.output_dir / "shot_clock_difficulty_crossfit_curves.csv", index=False
    )
    metrics.to_csv(args.output_dir / "shot_clock_difficulty_ots_regression_metrics.csv", index=False)
    pd.concat(coefficient_parts, ignore_index=True).to_csv(
        args.output_dir / "shot_clock_difficulty_ots_regression_coefficients.csv", index=False
    )
    all_predictions.to_csv(
        args.output_dir / "shot_clock_difficulty_ots_oof_predictions.csv", index=False
    )
    forward_metrics.to_csv(
        args.output_dir / "shot_clock_difficulty_future_ots_regression_metrics.csv",
        index=False,
    )
    all_forward_predictions.to_csv(
        args.output_dir / "shot_clock_difficulty_future_ots_predictions.csv",
        index=False,
    )
    incremental.to_csv(
        args.output_dir / "shot_clock_difficulty_zts_incremental_metrics.csv",
        index=False,
    )
    overlap_summary.to_csv(
        args.output_dir / "shot_clock_difficulty_zts_overlap.csv", index=False
    )
    role_numbers.to_csv(
        args.output_dir / "zts_role_numbers_player_season.csv", index=False
    )
    manifest = {
        "seasons": [args.start_year, args.end_year],
        "clock_surfaces": surfaces,
        "clock_curve": "game-held-out isotonic TSx2 by season and 24/14-second reset family",
        "difficulty_definition": "league reference TS% minus clock-expected TS%",
        "adjusted_ts_identity": "raw TS% + shot-clock difficulty = clock-adjusted TS%",
        "regression_target": str(args.target),
        "regression_sample_players": int(len(analysis)),
        "season_forward_player_seasons": int(len(forward_rows)),
        "season_forward_folds": "2024 features to 2025 oTS and 2025 features to 2026 oTS; each transition trained on the other",
        "primary_surface": "raw",
        "calibrated_sensitivity": "calibrated clock floored at four seconds",
        "zts_definition": "pt_adj_rTS = rTSPct + playtype_diff",
        "zts_source": str(args.zts),
        "zts_role_source": str(args.playtype_rows),
        "zts_roles": ROLE_GROUPS,
        "zts_role_rows": int(len(role_numbers)),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(metrics.sort_values("rmse").to_string(index=False))
    print("\nSeason-forward oTS:\n")
    print(forward_metrics.sort_values("rmse").to_string(index=False))
    print("\nzTS / clock incremental comparisons:\n")
    print(incremental.to_string(index=False))
    print("\nPlaytype-versus-clock overlap:\n")
    print(overlap_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
