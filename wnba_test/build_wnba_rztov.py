#!/usr/bin/env python3
"""Build a box-estimated WNBA rZTOV research board.

The WNBA RTSS mirror observes the two role inputs that ordinary box scores do
not: whether a shot was potentially assisted and how many potential assists a
passer created.  This builder uses 2023-2025 RTSS only as a training label,
then estimates both quantities from the repo's PBP Stats-style WNBA box panel.

The shot ledger is constrained rather than independently predicted:

    estimated self-created TSA + estimated assisted TSA = box TSA

where TSA counts FGA plus non-and-one shooting foul trips and nonshooting foul trips.  Potential assists are estimated with a
separate compact ridge model.  WNBA-specific turnover break-evens are then fit
jointly against the existing total, bad-pass, and implied scoring-turnover
offensive RAPM targets.  The resulting rate follows the NBA rZTOV convention:

    rZTOV = 100 * (actual turnovers - allowance) / plays

Negative is good.  This is a research-only estimated surface; it does not
replace the observed NBA rZTOV build or the live WNBA VPM board.
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

from wnba_test.build_wnba_box_priors import (  # noqa: E402
    BOX_DATA_DIR,
    DEFAULT_OUTPUT as BOX_PRIOR_OUTPUT,
    load_box_features,
)

DEFAULT_RTSS_FEATURES = (
    ROOT
    / "wnba_test"
    / "Results"
    / "rtss_prior_research"
    / "player_season_features.csv"
)
DEFAULT_OUTPUT = ROOT / "wnba_test" / "Results" / "wnba_rztov"
TRAIN_SEASONS = (2023, 2024, 2025)
HISTORY_START_SEASON = 2022
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
MIN_RTSS_CHANCES = 250.0
QUALIFY_OFF_POSS = 1000.0
QUALIFY_INVOLVEMENT = 10.0

ASSIST_SHARE_FEATURES = [
    "tsa100",
    "rim_freq",
    "mid_freq",
    "three_freq",
    "fta100",
    "assists100",
    "assist_points100",
    "rim_assists100",
    "turnovers100",
    "badpass_tov100",
    "usage_proxy",
]
POTENTIAL_ASSIST_FEATURES = [
    "assists100",
    "assist_points100",
    "rim_assists100",
    "badpass_tov100",
    "turnovers100",
    "tsa100",
    "usage_proxy",
]


def _number(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def load_raw_box(year: int) -> pd.DataFrame:
    """Load exact current counts/rates without the prior smoothing used by priors."""
    path = BOX_DATA_DIR / f"{year}_pbp.csv"
    raw = pd.read_csv(path)
    out = pd.DataFrame(index=raw.index)
    out["player_id"] = (
        pd.to_numeric(raw["EntityId"], errors="coerce")
        .astype("Int64")
        .astype(str)
    )
    out["player_name"] = raw["Name"].fillna(raw.get("ShortName", "")).astype(str)
    out["season"] = int(year)
    out["off_poss"] = _number(raw, "OffPoss")
    fga = _number(raw, "FG2A") + _number(raw, "FG3A")
    required = ["TwoPtShootingFoulsDrawn", "ThreePtShootingFoulsDrawn",
                "2pt And 1 Free Throw Trips", "3pt And 1 Free Throw Trips",
                "NonShootingFoulsDrawn"]
    missing = set(required) - set(raw)
    if missing:
        raise ValueError(f"Missing canonical TSA fields: {sorted(missing)}")
    tsa = (fga + _number(raw, required[0]) + _number(raw, required[1])
           - _number(raw, required[2]) - _number(raw, required[3])
           + _number(raw, required[4]))
    tov = _number(raw, "Turnovers")
    badpass = np.minimum(_number(raw, "BadPassTurnovers"), tov)
    denominator = out["off_poss"].replace(0.0, np.nan)
    out["fga100_raw"] = 100.0 * fga / denominator
    out["tsa100_raw"] = 100.0 * tsa / denominator
    out["tov100_raw"] = 100.0 * tov / denominator
    out["bp_tov100_raw"] = 100.0 * badpass / denominator
    out["sc_tov100_raw"] = out["tov100_raw"] - out["bp_tov100_raw"]
    out["turnovers"] = tov
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def load_model_box(year: int) -> pd.DataFrame:
    frame = load_box_features(year).rename(columns={"year": "season"})
    frame["player_id"] = frame["player_id"].astype(str)
    return frame


def build_training_panel(rtss_features: Path) -> pd.DataFrame:
    rtss = pd.read_csv(rtss_features)
    rtss["player_id"] = rtss["player_id"].astype(str)
    rtss = rtss[rtss["season"].isin(TRAIN_SEASONS)].copy()
    rtss["assist_opp_share"] = pd.to_numeric(
        rtss["rtss_shot_assist_opp_share"], errors="coerce"
    )
    rtss["potential_assists100"] = pd.to_numeric(
        rtss["rtss_potential_assist_rate100"], errors="coerce"
    )
    pieces = []
    for season in TRAIN_SEASONS:
        labels = rtss[rtss["season"].eq(season)]
        model_box = load_model_box(season)
        raw_box = load_raw_box(season)
        merged = labels.merge(
            model_box,
            on=["season", "player_id"],
            how="inner",
            suffixes=("_rtss", ""),
            validate="one_to_one",
        ).merge(
            raw_box.drop(columns=["player_name", "off_poss"]),
            on=["season", "player_id"],
            how="inner",
            validate="one_to_one",
        )
        pieces.append(merged)
    panel = pd.concat(pieces, ignore_index=True)
    required = [
        "assist_opp_share",
        "potential_assists100",
        "off_chances",
        "off_poss",
        *ASSIST_SHARE_FEATURES,
        *POTENTIAL_ASSIST_FEATURES,
    ]
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    return panel[panel["off_chances"].ge(MIN_RTSS_CHANCES)].copy()


def _model(features: list[str], alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )


def _weighted_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    weight: np.ndarray,
) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    weight = np.maximum(np.asarray(weight, dtype=float), 0.0)
    total_weight = max(float(weight.sum()), 1.0)
    mean = float(np.sum(weight * actual) / total_weight)
    mse = float(np.sum(weight * np.square(actual - predicted)) / total_weight)
    variance = float(np.sum(weight * np.square(actual - mean)) / total_weight)
    centered_prediction = predicted - float(np.sum(weight * predicted) / total_weight)
    centered_actual = actual - mean
    denominator = float(
        np.sqrt(
            np.sum(weight * np.square(centered_actual))
            * np.sum(weight * np.square(centered_prediction))
        )
    )
    return {
        "rmse": float(np.sqrt(mse)),
        "r2": 1.0 - mse / variance if variance > 0.0 else np.nan,
        "correlation": (
            float(np.sum(weight * centered_actual * centered_prediction) / denominator)
            if denominator > 0.0
            else np.nan
        ),
    }


def select_and_fit_label_model(
    panel: pd.DataFrame,
    target: str,
    features: list[str],
    lower: float,
    upper: float | None,
) -> tuple[Pipeline, float, pd.DataFrame, pd.DataFrame]:
    """Select ridge alpha by symmetric leave-one-season-out development CV."""
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        for heldout in TRAIN_SEASONS:
            train = panel[panel["season"].ne(heldout)]
            test = panel[panel["season"].eq(heldout)]
            fitted = _model(features, alpha)
            fitted.fit(
                train[features],
                train[target],
                ridge__sample_weight=train["off_chances"],
            )
            prediction = fitted.predict(test[features])
            prediction = np.maximum(prediction, lower)
            if upper is not None:
                prediction = np.minimum(prediction, upper)
            metrics = _weighted_metrics(
                test[target].to_numpy(float),
                prediction,
                test["off_chances"].to_numpy(float),
            )
            fold_rows.append(
                {
                    "target": target,
                    "alpha": float(alpha),
                    "heldout_season": int(heldout),
                    "rows": int(len(test)),
                    **metrics,
                }
            )
            for row, estimate in zip(test.itertuples(index=False), prediction):
                prediction_rows.append(
                    {
                        "target": target,
                        "alpha": float(alpha),
                        "heldout_season": int(heldout),
                        "player_id": str(row.player_id),
                        "off_chances": float(row.off_chances),
                        "actual": float(getattr(row, target)),
                        "prediction": float(estimate),
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    pooled = []
    for alpha, group in predictions.groupby("alpha"):
        pooled.append(
            {
                "alpha": float(alpha),
                **_weighted_metrics(
                    group["actual"].to_numpy(float),
                    group["prediction"].to_numpy(float),
                    group["off_chances"].to_numpy(float),
                ),
            }
        )
    pooled_frame = pd.DataFrame(pooled).sort_values(["rmse", "alpha"])
    selected_alpha = float(pooled_frame.iloc[0]["alpha"])
    folds = pd.DataFrame(fold_rows).merge(
        pooled_frame.add_prefix("pooled_"),
        left_on="alpha",
        right_on="pooled_alpha",
        how="left",
    )
    folds["selected"] = folds["alpha"].eq(selected_alpha)
    selected_predictions = predictions[predictions["alpha"].eq(selected_alpha)].copy()

    final_model = _model(features, selected_alpha)
    final_model.fit(
        panel[features],
        panel[target],
        ridge__sample_weight=panel["off_chances"],
    )
    return final_model, selected_alpha, folds, selected_predictions


def _load_turnover_targets(season: int) -> pd.DataFrame:
    directory = BOX_PRIOR_OUTPUT / "annual_targets" / str(season)
    total = pd.read_csv(directory / "ALT_TOV_VALUE.csv")
    passing = pd.read_csv(directory / "ALT_BADPASS_TOV_VALUE.csv")
    for frame in (total, passing):
        frame["player_id"] = (
            pd.to_numeric(frame["player_id"], errors="coerce")
            .astype("Int64")
            .astype(str)
        )
    targets = total[["player_id", "off", "off_poss"]].rename(
        columns={"off": "y_total", "off_poss": "target_off_poss"}
    )
    targets = targets.merge(
        passing[["player_id", "off"]].rename(columns={"off": "y_badpass"}),
        on="player_id",
        how="inner",
        validate="one_to_one",
    )
    targets["y_scoring"] = targets["y_total"] - targets["y_badpass"]
    targets["season"] = int(season)
    return targets


def fit_break_evens(panel: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    """Fit the NBA-style joint two-lane allowance on observed RTSS roles."""
    pieces = []
    for season in TRAIN_SEASONS:
        rows = panel[panel["season"].eq(season)].merge(
            _load_turnover_targets(season),
            on=["season", "player_id"],
            how="inner",
            validate="one_to_one",
        )
        pieces.append(rows)
    fit = pd.concat(pieces, ignore_index=True)
    fit = fit[fit["target_off_poss"].ge(QUALIFY_OFF_POSS)].copy()
    fit["v_assisted"] = fit["tsa100_raw"] * fit["assist_opp_share"].clip(0.0, 1.0)
    fit["v_self"] = fit["tsa100_raw"] - fit["v_assisted"]
    fit["v_potast"] = fit["potential_assists100"].clip(lower=0.0)
    fit["actual"] = fit["sc_tov100_raw"] + fit["bp_tov100_raw"]

    weight = fit["off_chances"].to_numpy(float)
    sc = fit["sc_tov100_raw"].to_numpy(float)
    bp = fit["bp_tov100_raw"].to_numpy(float)
    actual = fit["actual"].to_numpy(float)
    v_self = fit["v_self"].to_numpy(float)
    v_assisted = fit["v_assisted"].to_numpy(float)
    v_potast = fit["v_potast"].to_numpy(float)
    y_sc = fit["y_scoring"].to_numpy(float)
    y_bp = fit["y_badpass"].to_numpy(float)
    y_total = fit["y_total"].to_numpy(float)
    zero = np.zeros_like(sc)
    sqrt_weight = np.sqrt(weight)

    def through_origin(target: np.ndarray, design: np.ndarray) -> np.ndarray:
        return np.linalg.lstsq(
            design * sqrt_weight[:, None],
            target * sqrt_weight,
            rcond=None,
        )[0]

    scoring_anchor = through_origin(
        y_sc, np.column_stack([sc, v_self, v_assisted])
    )
    passing_anchor = through_origin(y_bp, np.column_stack([bp, v_potast]))
    be_self = float(-scoring_anchor[1] / scoring_anchor[0])
    be_assisted = float(-scoring_anchor[2] / scoring_anchor[0])
    be_potast = float(-passing_anchor[1] / passing_anchor[0])

    for _ in range(300):
        scoring_allowance = be_self * v_self + be_assisted * v_assisted
        passing_allowance = be_potast * v_potast
        scoring_excess = sc - scoring_allowance
        passing_excess = bp - passing_allowance
        total_excess = actual - scoring_allowance - passing_allowance

        def slope(target: np.ndarray, excess: np.ndarray) -> float:
            denominator = float(np.sum(weight * np.square(excess)))
            return float(np.sum(weight * target * excess) / max(denominator, 1e-12))

        k_sc = slope(y_sc, scoring_excess)
        k_bp = slope(y_bp, passing_excess)
        k_total = slope(y_total, total_excess)
        variance_sc = max(
            float(np.average(np.square(y_sc - k_sc * scoring_excess), weights=weight)),
            1e-12,
        )
        variance_bp = max(
            float(np.average(np.square(y_bp - k_bp * passing_excess), weights=weight)),
            1e-12,
        )
        variance_total = max(
            float(np.average(np.square(y_total - k_total * total_excess), weights=weight)),
            1e-12,
        )
        design = np.vstack(
            [
                np.column_stack([-k_sc * v_self, -k_sc * v_assisted, zero]),
                np.column_stack([zero, zero, -k_bp * v_potast]),
                np.column_stack(
                    [-k_total * v_self, -k_total * v_assisted, -k_total * v_potast]
                ),
            ]
        )
        target = np.concatenate(
            [y_sc - k_sc * sc, y_bp - k_bp * bp, y_total - k_total * actual]
        )
        stacked_weight = np.sqrt(
            np.concatenate(
                [
                    weight / variance_sc,
                    weight / variance_bp,
                    weight / variance_total,
                ]
            )
        )
        previous = (be_self, be_assisted, be_potast)
        be_self, be_assisted, be_potast = np.linalg.lstsq(
            design * stacked_weight[:, None],
            target * stacked_weight,
            rcond=None,
        )[0]
        if max(
            abs(be_self - previous[0]),
            abs(be_assisted - previous[1]),
            abs(be_potast - previous[2]),
        ) < 1e-10:
            break

    scoring_allowance = be_self * v_self + be_assisted * v_assisted
    passing_allowance = be_potast * v_potast
    scoring_excess = sc - scoring_allowance
    passing_excess = bp - passing_allowance
    total_excess = actual - scoring_allowance - passing_allowance

    def fit_diagnostics(target: np.ndarray, excess: np.ndarray) -> tuple[float, float]:
        fitted_slope = float(
            np.sum(weight * target * excess)
            / max(float(np.sum(weight * np.square(excess))), 1e-12)
        )
        prediction = fitted_slope * excess
        target_mean = float(np.average(target, weights=weight))
        denominator = float(np.sum(weight * np.square(target - target_mean)))
        r2 = 1.0 - float(np.sum(weight * np.square(target - prediction))) / max(
            denominator, 1e-12
        )
        return fitted_slope, r2

    k_sc, r2_sc = fit_diagnostics(y_sc, scoring_excess)
    k_bp, r2_bp = fit_diagnostics(y_bp, passing_excess)
    k_total, r2_total = fit_diagnostics(y_total, total_excess)
    result = {
        "self": float(be_self),
        "assisted": float(be_assisted),
        "potential_assist": float(be_potast),
        "lane_self": float(-scoring_anchor[1] / scoring_anchor[0]),
        "lane_assisted": float(-scoring_anchor[2] / scoring_anchor[0]),
        "lane_potential_assist": float(-passing_anchor[1] / passing_anchor[0]),
        "k_scoring": k_sc,
        "k_badpass": k_bp,
        "k_total": k_total,
        "r2_scoring": r2_sc,
        "r2_badpass": r2_bp,
        "r2_total": r2_total,
        "rows": int(len(fit)),
        "off_chances": float(weight.sum()),
    }
    return result, fit


def score_box_season(
    season: int,
    share_model: Pipeline,
    potast_model: Pipeline,
    prices: dict[str, float],
) -> pd.DataFrame:
    model_box = load_model_box(season)
    raw_box = load_raw_box(season)
    rows = model_box.merge(
        raw_box.drop(columns="off_poss"),
        on=["season", "player_id"],
        how="inner",
        suffixes=("", "_raw"),
        validate="one_to_one",
    )
    rows["estimated_assisted_share"] = np.clip(
        share_model.predict(rows[ASSIST_SHARE_FEATURES]), 0.0, 1.0
    )
    rows["estimated_self_created_tsa100"] = (
        rows["tsa100_raw"] * (1.0 - rows["estimated_assisted_share"])
    )
    rows["estimated_assisted_tsa100"] = (
        rows["tsa100_raw"] * rows["estimated_assisted_share"]
    )
    rows["estimated_potential_assists100"] = np.maximum(
        potast_model.predict(rows[POTENTIAL_ASSIST_FEATURES]), 0.0
    )
    rows["shot_volume_identity_gap"] = (
        rows["estimated_self_created_tsa100"]
        + rows["estimated_assisted_tsa100"]
        - rows["tsa100_raw"]
    )
    rows["scoring_allowance100_uncentered"] = (
        prices["self"] * rows["estimated_self_created_tsa100"]
        + prices["assisted"] * rows["estimated_assisted_tsa100"]
    )
    rows["badpass_allowance100_uncentered"] = (
        prices["potential_assist"] * rows["estimated_potential_assists100"]
    )
    eligible = rows["off_poss"].gt(0.0)
    weights = rows.loc[eligible, "off_poss"]
    scoring_center = float(
        np.average(
            rows.loc[eligible, "sc_tov100_raw"]
            - rows.loc[eligible, "scoring_allowance100_uncentered"],
            weights=weights,
        )
    )
    badpass_center = float(
        np.average(
            rows.loc[eligible, "bp_tov100_raw"]
            - rows.loc[eligible, "badpass_allowance100_uncentered"],
            weights=weights,
        )
    )
    rows["scoring_center100"] = scoring_center
    rows["badpass_center100"] = badpass_center
    rows["scoring_allowance100"] = (
        rows["scoring_allowance100_uncentered"] + scoring_center
    )
    rows["badpass_allowance100"] = (
        rows["badpass_allowance100_uncentered"] + badpass_center
    )
    rows["turnover_allowance100"] = (
        rows["scoring_allowance100"] + rows["badpass_allowance100"]
    )
    rows["scoring_excess100"] = (
        rows["sc_tov100_raw"] - rows["scoring_allowance100"]
    )
    rows["badpass_excess100"] = (
        rows["bp_tov100_raw"] - rows["badpass_allowance100"]
    )
    rows["turnover_excess100"] = (
        rows["scoring_excess100"] + rows["badpass_excess100"]
    )
    rows["involvement100"] = (
        rows["estimated_self_created_tsa100"]
        + rows["estimated_assisted_tsa100"]
        + rows["estimated_potential_assists100"]
    )
    rows["plays100"] = rows["involvement100"] + rows["tov100_raw"]
    rows["rztov_pct"] = np.where(
        rows["plays100"].gt(0.0),
        100.0 * rows["turnover_excess100"] / rows["plays100"],
        np.nan,
    )
    rows["ztov_pct"] = rows["rztov_pct"]
    rows["turnovers_saved100"] = -rows["turnover_excess100"]
    rows["qualified"] = (
        rows["off_poss"].ge(QUALIFY_OFF_POSS)
        & rows["involvement100"].ge(QUALIFY_INVOLVEMENT)
    ).astype(int)
    if float(rows["shot_volume_identity_gap"].abs().max()) > 1e-10:
        raise AssertionError("estimated self-created + assisted volume does not equal TSA")
    return rows


def _historical_score_status(season: int) -> str:
    if season < min(TRAIN_SEASONS):
        return "pre_training_retrospective_score"
    if season in TRAIN_SEASONS:
        return "training_period_retrospective_score"
    return "current_unlabeled_score"


def score_box_history(
    seasons: list[int],
    share_model: Pipeline,
    potast_model: Pipeline,
    prices: dict[str, float],
) -> pd.DataFrame:
    """Score one fixed retrospective model/price contract over many seasons."""
    pieces = []
    training_label = "-".join(str(season) for season in TRAIN_SEASONS)
    for season in seasons:
        board = score_box_season(season, share_model, potast_model, prices).copy()
        board["series"] = "wnba_box_estimated_rztov_retrospective_fixed_fit"
        board["historical_score_status"] = _historical_score_status(season)
        board["role_model_training_seasons"] = training_label
        board["turnover_price_training_seasons"] = training_label
        pieces.append(board)
    history = pd.concat(pieces, ignore_index=True)
    if history.duplicated(["season", "player_id"]).any():
        raise AssertionError("historical WNBA rZTOV has duplicate player-season keys")
    if float(history["shot_volume_identity_gap"].abs().max()) > 1e-10:
        raise AssertionError("historical WNBA shot-volume identity does not close")
    return history


def build(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = build_training_panel(args.rtss_features)
    share_model, share_alpha, share_folds, share_predictions = (
        select_and_fit_label_model(
            panel,
            "assist_opp_share",
            ASSIST_SHARE_FEATURES,
            lower=0.0,
            upper=1.0,
        )
    )
    potast_model, potast_alpha, potast_folds, potast_predictions = (
        select_and_fit_label_model(
            panel,
            "potential_assists100",
            POTENTIAL_ASSIST_FEATURES,
            lower=0.0,
            upper=None,
        )
    )
    folds = pd.concat([share_folds, potast_folds], ignore_index=True)
    folds.to_csv(args.output_dir / "label_model_loso_folds.csv", index=False)
    predictions = pd.concat(
        [share_predictions, potast_predictions], ignore_index=True
    )
    predictions.to_csv(
        args.output_dir / "label_model_loso_predictions.csv", index=False
    )

    prices, price_rows = fit_break_evens(panel)
    price_rows[
        [
            "season",
            "player_id",
            "player_name_rtss",
            "off_chances",
            "target_off_poss",
            "assist_opp_share",
            "potential_assists100",
            "tsa100_raw",
            "sc_tov100_raw",
            "bp_tov100_raw",
            "y_scoring",
            "y_badpass",
            "y_total",
        ]
    ].to_csv(args.output_dir / "break_even_training_rows.csv", index=False)

    history_seasons = list(range(args.history_start_season, args.season + 1))
    if not history_seasons:
        raise ValueError("history start season must not exceed the scored season")
    history = score_box_history(
        history_seasons, share_model, potast_model, prices
    )
    board = history[history["season"].eq(args.season)].copy()
    output_columns = [
        "season",
        "player_id",
        "player_name",
        "off_poss",
        "tsa100_raw",
        "estimated_self_created_tsa100",
        "estimated_assisted_tsa100",
        "estimated_assisted_share",
        "estimated_potential_assists100",
        "sc_tov100_raw",
        "bp_tov100_raw",
        "tov100_raw",
        "scoring_allowance100",
        "badpass_allowance100",
        "turnover_allowance100",
        "scoring_excess100",
        "badpass_excess100",
        "turnover_excess100",
        "involvement100",
        "plays100",
        "rztov_pct",
        "ztov_pct",
        "turnovers_saved100",
        "qualified",
        "shot_volume_identity_gap",
    ]
    board_path = args.output_dir / f"wnba_rztov_{args.season}.csv"
    board[output_columns].sort_values(
        ["qualified", "rztov_pct"], ascending=[False, True]
    ).to_csv(board_path, index=False)
    history_columns = [
        "series",
        "historical_score_status",
        "role_model_training_seasons",
        "turnover_price_training_seasons",
        *output_columns,
    ]
    history_path = (
        args.output_dir
        / f"wnba_box_estimated_rztov_{args.history_start_season}_{args.season}.csv"
    )
    history[history_columns].sort_values(
        ["season", "qualified", "rztov_pct"],
        ascending=[True, False, True],
    ).to_csv(history_path, index=False)

    selected_predictions = predictions.copy()
    validation: dict[str, dict[str, float]] = {}
    for target, group in selected_predictions.groupby("target"):
        validation[target] = _weighted_metrics(
            group["actual"].to_numpy(float),
            group["prediction"].to_numpy(float),
            group["off_chances"].to_numpy(float),
        )
    qualified = board[board["qualified"].eq(1)]
    manifest = {
        "status": "research_only_box_estimated_not_live",
        "season": int(args.season),
        "historical_coverage": [
            int(args.history_start_season),
            int(args.season),
        ],
        "historical_contract": {
            "role_model": "one fixed retrospective fit on 2023-2025 RTSS labels",
            "turnover_prices": "one fixed retrospective fit on 2023-2025 WNBA targets",
            "season_statuses": {
                "2022": "pre_training_retrospective_score",
                "2023-2025": "training_period_retrospective_score",
                "2026": "current_unlabeled_score",
            },
            "date_safe": False,
            "interpretation": "descriptive workload/rZTOV history, not a walk-forward predictive series",
        },
        "training_seasons": list(TRAIN_SEASONS),
        "rtss_label_source": str(args.rtss_features),
        "box_source": str(BOX_DATA_DIR),
        "turnover_target_source": str(BOX_PRIOR_OUTPUT / "annual_targets"),
        "training_rows": int(len(panel)),
        "minimum_rtss_chances": MIN_RTSS_CHANCES,
        "label_models": {
            "assist_opp_share": {
                "features": ASSIST_SHARE_FEATURES,
                "selected_alpha": share_alpha,
                **validation["assist_opp_share"],
            },
            "potential_assists100": {
                "features": POTENTIAL_ASSIST_FEATURES,
                "selected_alpha": potast_alpha,
                **validation["potential_assists100"],
            },
        },
        "break_evens": prices,
        "identities": {
            "shot_volume": "estimated_self_created_tsa100 + estimated_assisted_tsa100 = box TSA100",
            "max_shot_volume_gap": float(board["shot_volume_identity_gap"].abs().max()),
            "turnover_lanes": "scoring_excess100 + badpass_excess100 = turnover_excess100",
            "rate": "rztov_pct = 100 * turnover_excess100 / plays100; negative is good",
        },
        "output": str(board_path),
        "historical_output": str(history_path),
        "players": int(len(board)),
        "qualified_players": int(len(qualified)),
        "historical_player_seasons": int(len(history)),
        "historical_qualified_player_seasons": int(history["qualified"].sum()),
        "qualification": {
            "off_poss": QUALIFY_OFF_POSS,
            "involvement100": QUALIFY_INVOLVEMENT,
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=float) + "\n"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument(
        "--history-start-season",
        type=int,
        default=HISTORY_START_SEASON,
        help="First season to include in the consolidated retrospective history.",
    )
    parser.add_argument("--rtss-features", type=Path, default=DEFAULT_RTSS_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    manifest = build(parse_args())
    print(json.dumps(manifest, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
