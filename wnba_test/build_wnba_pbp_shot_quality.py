#!/usr/bin/env python3
"""Build a leakage-safe WNBA midrange/three play-by-play shot-quality model.

The context model uses only information observable before the shot result:
location, distance, shot family, description-derived shot-form modifiers, and
period-clock context. Outcome labels, made-only assist tags, and the raw
Made Shot / Missed Shot action label are retained only for auditing.

Every published validation row is a midrange or three-point attempt and is
season-forward. Rim is audit-only and excluded from tuning, validation, and
impact use. The optional shooter adjustment for validation season Y uses
shooter residual history from seasons strictly before Y.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = ROOT / "wnba_test" / "GabrielNormalizedClean"
DEFAULT_CLASSIC_DIR = (
    ROOT / "nba_pipeline" / "external" / "gabriel_wnba" / "pbp_data"
)
DEFAULT_OUTPUT_DIR = ROOT / "wnba_test" / "Results" / "pbp_shot_quality"
IMPACT_TARGET_ZONES = {"short_mid", "long_mid", "deep_two", "three"}
IMPACT_TARGET_COMPONENTS = {
    "ALT_EFG_MID_FREQ",
    "ALT_EFG_MID_FG",
    "ALT_EFG_THREE_FREQ",
    "ALT_EFG_THREE_FG",
}

DISTANCE_BINS = [-0.1, 2, 4, 7, 10, 14, 18, 22, 25, 28, 32, 100]
DISTANCE_LABELS = [
    "00_02",
    "03_04",
    "05_07",
    "08_10",
    "11_14",
    "15_18",
    "19_22",
    "23_25",
    "26_28",
    "29_32",
    "33_plus",
]
MODIFIER_PATTERNS = {
    "driving": r"\bdriving\b",
    "running": r"\brunning\b",
    "cutting": r"\bcutting\b",
    "pullup": r"\bpull[\s-]?up\b|\bpullup\b",
    "step_back": r"\bstep[\s-]?back\b",
    "turnaround": r"\bturnaround\b",
    "fadeaway": r"\bfadeaway\b",
    "bank": r"\bbank\b",
    "alley_oop": r"\balley[\s-]?oop\b",
    "putback": r"\bputback\b",
    "reverse": r"\breverse\b",
    "finger_roll": r"\bfinger[\s-]?roll\b",
    "floating": r"\bfloating\b|\bfloater\b",
    "tip": r"\btip\b",
}

ZONE_KEYS = [
    "shot_value",
    "shot_zone",
    "distance_bin",
    "shot_family",
]
CONTEXT_KEYS = [
    *ZONE_KEYS,
    "modifier_bundle",
    "clock_bucket",
]
MODEL_FEATURE_COLUMNS = [
    "shot_value",
    "shot_distance",
    "shot_zone",
    "distance_bin",
    "shot_family",
    "modifier_bundle",
    "clock_bucket",
    *[f"desc_{label}" for label in MODIFIER_PATTERNS],
]


@dataclass(frozen=True)
class ShotQualityConfig:
    zone_shrink: float = 350.0
    context_shrink: float = 175.0
    shooter_shrink: float = 250.0


@dataclass
class ContextModel:
    config: ShotQualityConfig
    global_table: pd.DataFrame
    zone_table: pd.DataFrame
    context_table: pd.DataFrame


def normalize_game_id(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(10)
    )


def season_forward_splits(years: Iterable[int]) -> list[tuple[list[int], int]]:
    ordered = sorted({int(year) for year in years})
    return [(ordered[:idx], year) for idx, year in enumerate(ordered[1:], start=1)]


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _shot_family(subtype: pd.Series, description: pd.Series) -> pd.Series:
    subtype_lower = subtype.fillna("").astype(str).str.lower()
    desc_lower = description.fillna("").astype(str).str.lower()
    return pd.Series(
        np.select(
            [
                subtype_lower.str.contains("dunk") | desc_lower.str.contains(r"\bdunk\b", regex=True),
                subtype_lower.str.contains("layup") | desc_lower.str.contains(r"\blayup\b|\bfinger[\s-]?roll\b", regex=True),
                subtype_lower.str.contains("hook") | desc_lower.str.contains(r"\bhook\b", regex=True),
                desc_lower.str.contains(r"\bfloating\b|\bfloater\b", regex=True),
                desc_lower.str.contains(r"\btip\b", regex=True),
            ],
            ["dunk", "layup", "hook", "floater", "tip"],
            default="jumper",
        ),
        index=subtype.index,
        dtype=object,
    )


def build_shot_rows(
    raw: pd.DataFrame,
    season_end_year: int,
    season_type: str,
) -> pd.DataFrame:
    """Normalize official FGA rows and attach outcome-safe context features."""
    is_fga = _numeric(raw, "isfieldgoal").eq(1)
    result = raw.get("shotresult", pd.Series("", index=raw.index)).fillna("").astype(str)
    is_make = result.str.lower().eq("made")
    is_miss = result.str.lower().eq("missed")
    frame = raw.loc[is_fga & (is_make | is_miss)].copy()
    if frame.empty:
        return frame

    result = frame["shotresult"].fillna("").astype(str)
    description = frame.get("description", pd.Series("", index=frame.index)).fillna("").astype(str)
    description_lower = description.str.lower()
    shot_value = _numeric(frame, "shotvalue", 2.0).where(lambda x: x.isin([2, 3]), 2.0)
    distance = _numeric(frame, "shotdistance").clip(lower=0.0, upper=99.0)
    xlegacy = _numeric(frame, "xlegacy")
    ylegacy = _numeric(frame, "ylegacy")
    seconds = _numeric(frame, "seconds_remaining_quarter", np.nan)
    subtype = frame.get("subtype", pd.Series("", index=frame.index)).fillna("").astype(str)

    shots = pd.DataFrame(index=frame.index)
    shots["season_end_year"] = int(season_end_year)
    shots["season_type"] = str(season_type).upper()
    shots["game_id"] = normalize_game_id(frame["game_id"])
    shots["event_num"] = _numeric(frame, "event_num").astype(int)
    shots["shot_id"] = shots["game_id"] + "_" + shots["event_num"].astype(str)
    shots["period"] = _numeric(frame, "period").astype(int)
    shots["seconds_remaining_quarter"] = seconds
    shots["shooter_id"] = _numeric(frame, "personid").astype("Int64")
    shots["shooter_name"] = frame.get(
        "playername", pd.Series("", index=frame.index)
    ).fillna("").astype(str)
    shots["shot_value"] = shot_value.astype(float)
    shots["shot_distance"] = distance.astype(float)
    shots["xlegacy"] = xlegacy.astype(float)
    shots["ylegacy"] = ylegacy.astype(float)
    geometry_flag = frame.get(
        "geometry_available",
        xlegacy.ne(0) | ylegacy.ne(0),
    )
    shots["geometry_available"] = geometry_flag.fillna(False).astype(bool)
    shots["is_make"] = result.str.lower().eq("made")
    shots["actual_points"] = np.where(
        shots["is_make"], shots["shot_value"], 0.0
    ).astype(float)

    family = _shot_family(subtype, description)
    shots["shot_family"] = family
    shots["shot_zone"] = np.select(
        [
            shots["shot_value"].eq(3),
            family.isin(["dunk", "layup", "tip"]),
            distance.le(10),
            distance.le(18),
        ],
        ["three", "rim", "short_mid", "long_mid"],
        default="deep_two",
    )
    shots["impact_target_eligible"] = shots["shot_zone"].isin(
        IMPACT_TARGET_ZONES
    )
    shots["distance_bin"] = pd.cut(
        distance,
        bins=DISTANCE_BINS,
        labels=DISTANCE_LABELS,
        include_lowest=True,
    ).astype(str)
    shots["location_side"] = np.select(
        [xlegacy.lt(-55), xlegacy.gt(55)],
        ["left", "right"],
        default="center",
    )
    shots["corner_three"] = (
        shots["shot_value"].eq(3) & xlegacy.abs().ge(210) & ylegacy.le(140)
    )
    shots["clock_bucket"] = np.select(
        [seconds.le(4), seconds.le(10), shots["period"].gt(4)],
        ["end_period_4", "end_period_10", "overtime"],
        default="normal",
    )

    modifier_bundle = pd.Series("", index=frame.index, dtype=object)
    for label, pattern in MODIFIER_PATTERNS.items():
        values = description_lower.str.contains(pattern, regex=True, na=False)
        shots[f"desc_{label}"] = values
        modifier_bundle = modifier_bundle + np.where(values, f"|{label}", "")
    shots["modifier_bundle"] = modifier_bundle.str.strip("|").replace("", "none")
    shots["talent_bucket"] = np.select(
        [
            shots["shot_value"].eq(3),
            family.isin(["dunk", "layup", "tip"]),
            family.isin(["hook", "floater"]) | distance.le(10),
        ],
        ["three", "rim_finish", "short_touch"],
        default="midrange_jumper",
    )

    # Audit-only raw fields. None are members of MODEL_FEATURE_COLUMNS.
    shots["shotresult"] = result
    shots["actiontype"] = frame.get(
        "actiontype", pd.Series("", index=frame.index)
    ).fillna("").astype(str)
    shots["description"] = description
    shots["made_assisted_tag"] = shots["is_make"] & description.str.contains(
        r"\([^()]+?\s+\d+\s+AST\)", regex=True, na=False
    )
    return shots.reset_index(drop=True)


def _clock_seconds(value: pd.Series) -> pd.Series:
    extracted = value.fillna("").astype(str).str.extract(
        r"^\s*(\d+):(\d+(?:\.\d+)?)\s*$"
    )
    minutes = pd.to_numeric(extracted[0], errors="coerce")
    seconds = pd.to_numeric(extracted[1], errors="coerce")
    return (60.0 * minutes + seconds).fillna(np.nan)


def build_classic_shot_rows(
    raw: pd.DataFrame,
    season_end_year: int,
    season_type: str,
) -> pd.DataFrame:
    """Adapt Gabriel classic stats-style PBP into the common shot schema."""
    msg_type = pd.to_numeric(raw["EVENTMSGTYPE"], errors="coerce")
    classic = raw.loc[msg_type.isin([1, 2])].copy()
    if classic.empty:
        return pd.DataFrame()

    msg_type = pd.to_numeric(classic["EVENTMSGTYPE"], errors="coerce")
    description = (
        classic.get(
            "HOMEDESCRIPTION", pd.Series("", index=classic.index)
        ).fillna("").astype(str)
        + " "
        + classic.get(
            "VISITORDESCRIPTION", pd.Series("", index=classic.index)
        ).fillna("").astype(str)
    ).str.strip()
    is_three = description.str.contains("3PT", case=False, regex=False)
    extracted_distance = pd.to_numeric(
        description.str.extract(r"\b(\d{1,2})'", expand=False),
        errors="coerce",
    )
    rim_description = description.str.contains(
        r"\blayup\b|\bdunk\b|\btip\b",
        case=False,
        regex=True,
    )
    fallback_distance = np.select(
        [is_three, rim_description],
        [25.0, 2.0],
        default=15.0,
    )

    normalized = pd.DataFrame(index=classic.index)
    normalized["game_id"] = classic["GAME_ID"]
    normalized["event_num"] = classic["EVENTNUM"]
    normalized["period"] = classic["PERIOD"]
    normalized["seconds_remaining_quarter"] = _clock_seconds(
        classic["PCTIMESTRING"]
    )
    normalized["personid"] = classic["PLAYER1_ID"]
    normalized["playername"] = classic["PLAYER1_NAME"]
    normalized["isfieldgoal"] = 1
    normalized["shotresult"] = np.where(msg_type.eq(1), "Made", "Missed")
    normalized["shotvalue"] = np.where(is_three, 3, 2)
    normalized["shotdistance"] = extracted_distance.fillna(
        pd.Series(fallback_distance, index=classic.index)
    )
    normalized["xlegacy"] = 0.0
    normalized["ylegacy"] = 0.0
    normalized["geometry_available"] = False
    normalized["actiontype"] = classic["EVENTMSGACTIONTYPE"].astype(str)
    normalized["subtype"] = description
    normalized["description"] = description
    return build_shot_rows(normalized, season_end_year, season_type)


def select_impact_target_shots(shots: pd.DataFrame) -> pd.DataFrame:
    """Keep only midrange and three-point attempts for impact-model use."""
    if "impact_target_eligible" not in shots:
        raise KeyError("impact_target_eligible")
    return shots.loc[shots["impact_target_eligible"].astype(bool)].copy()


def _fit_smoothed_table(
    rows: pd.DataFrame,
    keys: list[str],
    prior_table: pd.DataFrame,
    prior_keys: list[str],
    prior_column: str,
    shrink: float,
    output_column: str,
) -> pd.DataFrame:
    stats = (
        rows.groupby(keys, dropna=False, observed=True)["actual_points"]
        .agg(["sum", "count"])
        .reset_index()
    )
    stats = stats.merge(
        prior_table[prior_keys + [prior_column]],
        on=prior_keys,
        how="left",
        validate="many_to_one",
    )
    stats[output_column] = (
        stats["sum"] + float(shrink) * stats[prior_column]
    ) / (stats["count"] + float(shrink))
    return stats[keys + ["count", output_column]]


def fit_context_model(
    train: pd.DataFrame,
    config: ShotQualityConfig = ShotQualityConfig(),
) -> ContextModel:
    if train.empty:
        raise ValueError("Cannot fit WNBA shot quality on an empty training set")
    global_table = (
        train.groupby(["shot_value"], observed=True)["actual_points"]
        .agg(["sum", "count"])
        .reset_index()
    )
    global_table["global_ev"] = global_table["sum"] / global_table["count"]
    global_table = global_table[["shot_value", "count", "global_ev"]]
    zone_table = _fit_smoothed_table(
        train,
        ZONE_KEYS,
        global_table,
        ["shot_value"],
        "global_ev",
        config.zone_shrink,
        "zone_ev",
    )
    context_table = _fit_smoothed_table(
        train,
        CONTEXT_KEYS,
        zone_table,
        ZONE_KEYS,
        "zone_ev",
        config.context_shrink,
        "context_ev",
    )
    return ContextModel(config, global_table, zone_table, context_table)


def predict_context_ev(model: ContextModel, shots: pd.DataFrame) -> pd.Series:
    if shots.empty:
        return pd.Series(dtype=float)
    scored = shots.reset_index(drop=False).rename(columns={"index": "_source_index"})
    scored["_row_order"] = np.arange(len(scored))
    scored = scored.merge(
        model.global_table[["shot_value", "global_ev"]],
        on=["shot_value"],
        how="left",
        validate="many_to_one",
    )
    scored = scored.merge(
        model.zone_table[ZONE_KEYS + ["zone_ev"]],
        on=ZONE_KEYS,
        how="left",
        validate="many_to_one",
    )
    scored = scored.merge(
        model.context_table[CONTEXT_KEYS + ["context_ev"]],
        on=CONTEXT_KEYS,
        how="left",
        validate="many_to_one",
    )
    prediction = (
        scored["context_ev"]
        .fillna(scored["zone_ev"])
        .fillna(scored["global_ev"])
        .fillna(scored["shot_value"] * 0.45)
        .clip(lower=0.01)
    )
    prediction = np.minimum(prediction, scored["shot_value"])
    ordered = pd.DataFrame(
        {
            "_row_order": scored["_row_order"],
            "prediction": prediction,
        }
    ).sort_values("_row_order")
    return pd.Series(ordered["prediction"].to_numpy(), index=shots.index, dtype=float)


def predict_zone_ev(model: ContextModel, shots: pd.DataFrame) -> pd.Series:
    """Predict from shot value plus coarse location/family geometry only."""
    if shots.empty:
        return pd.Series(dtype=float)
    scored = shots.reset_index(drop=False).rename(columns={"index": "_source_index"})
    scored["_row_order"] = np.arange(len(scored))
    scored = scored.merge(
        model.global_table[["shot_value", "global_ev"]],
        on=["shot_value"],
        how="left",
        validate="many_to_one",
    )
    scored = scored.merge(
        model.zone_table[ZONE_KEYS + ["zone_ev"]],
        on=ZONE_KEYS,
        how="left",
        validate="many_to_one",
    )
    prediction = (
        scored["zone_ev"]
        .fillna(scored["global_ev"])
        .fillna(scored["shot_value"] * 0.45)
        .clip(lower=0.01)
    )
    prediction = np.minimum(prediction, scored["shot_value"])
    ordered = pd.DataFrame(
        {
            "_row_order": scored["_row_order"],
            "prediction": prediction,
        }
    ).sort_values("_row_order")
    return pd.Series(ordered["prediction"].to_numpy(), index=shots.index, dtype=float)


def predict_shooter_prior_ev(
    model: ContextModel,
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    train_context = predict_context_ev(model, train)
    residuals = train[["shooter_id", "talent_bucket"]].copy()
    residuals["residual"] = train["actual_points"].to_numpy() - train_context.to_numpy()
    talent = (
        residuals.groupby(["shooter_id", "talent_bucket"], dropna=False, observed=True)[
            "residual"
        ]
        .agg(["sum", "count"])
        .reset_index()
    )
    talent["shooter_ev_added"] = talent["sum"] / (
        talent["count"] + model.config.shooter_shrink
    )
    lookup = validation[["shooter_id", "talent_bucket"]].merge(
        talent[["shooter_id", "talent_bucket", "count", "shooter_ev_added"]],
        on=["shooter_id", "talent_bucket"],
        how="left",
        validate="many_to_one",
    )
    context = predict_context_ev(model, validation)
    added = lookup["shooter_ev_added"].fillna(0.0)
    prediction = np.minimum(
        np.maximum(context.to_numpy() + added.to_numpy(), 0.01),
        validation["shot_value"].to_numpy(),
    )
    return (
        pd.Series(prediction, index=validation.index, dtype=float),
        pd.Series(lookup["count"].fillna(0.0).to_numpy(), index=validation.index),
    )


def _prediction_metrics(
    actual_points: pd.Series,
    shot_value: pd.Series,
    prediction: pd.Series,
) -> dict[str, float]:
    actual = actual_points.to_numpy(dtype=float)
    value = shot_value.to_numpy(dtype=float)
    pred = prediction.to_numpy(dtype=float)
    make = (actual > 0).astype(float)
    probability = np.clip(pred / value, 1e-6, 1 - 1e-6)
    mse = float(np.mean(np.square(actual - pred)))
    baseline_mean = float(np.mean(actual))
    denominator = float(np.sum(np.square(actual - baseline_mean)))
    r2 = 1.0 - float(np.sum(np.square(actual - pred))) / denominator if denominator else 0.0
    return {
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(actual - pred))),
        "r2": r2,
        "brier": float(np.mean(np.square(make - probability))),
        "log_loss": float(
            -np.mean(make * np.log(probability) + (1 - make) * np.log(1 - probability))
        ),
        "mean_prediction": float(np.mean(pred)),
        "mean_actual": float(np.mean(actual)),
    }


def evaluate_season_forward(
    shots: pd.DataFrame,
    config: ShotQualityConfig = ShotQualityConfig(),
    keep_predictions: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = sorted(shots["season_end_year"].astype(int).unique())
    reports = []
    prediction_frames = []
    for train_years, validation_year in season_forward_splits(years):
        train = shots.loc[shots["season_end_year"].isin(train_years)].copy()
        validation = shots.loc[shots["season_end_year"].eq(validation_year)].copy()
        model = fit_context_model(train, config)
        global_map = model.global_table.set_index("shot_value")["global_ev"]
        baseline = validation["shot_value"].map(global_map).fillna(
            train["actual_points"].mean()
        )
        zone = predict_zone_ev(model, validation)
        context = predict_context_ev(model, validation)
        shooter, shooter_attempts = predict_shooter_prior_ev(model, train, validation)
        predictions = {
            "shot_value_baseline": baseline,
            "zone_geometry": zone,
            "context": context,
            "context_plus_prior_shooter": shooter,
        }
        fold_metrics = {}
        for label, values in predictions.items():
            metrics = _prediction_metrics(
                validation["actual_points"],
                validation["shot_value"],
                values,
            )
            fold_metrics[label] = metrics
            reports.append(
                {
                    "validation_year": int(validation_year),
                    "train_years": ",".join(map(str, train_years)),
                    "model": label,
                    "shots": int(len(validation)),
                    **metrics,
                }
            )
        baseline_mse = fold_metrics["shot_value_baseline"]["rmse"] ** 2
        context_mse = fold_metrics["context"]["rmse"] ** 2
        shooter_mse = fold_metrics["context_plus_prior_shooter"]["rmse"] ** 2
        zone_mse = fold_metrics["zone_geometry"]["rmse"] ** 2
        reports[-3]["mse_gain_vs_value_baseline_pct"] = 100.0 * (
            1.0 - zone_mse / baseline_mse
        )
        reports[-2]["mse_gain_vs_value_baseline_pct"] = 100.0 * (
            1.0 - context_mse / baseline_mse
        )
        reports[-2]["mse_gain_vs_zone_pct"] = 100.0 * (
            1.0 - context_mse / zone_mse
        )
        reports[-1]["mse_gain_vs_value_baseline_pct"] = 100.0 * (
            1.0 - shooter_mse / baseline_mse
        )
        reports[-1]["mse_gain_vs_context_pct"] = 100.0 * (
            1.0 - shooter_mse / context_mse
        )

        if keep_predictions:
            output = validation.copy()
            output["train_years"] = ",".join(map(str, train_years))
            output["shot_value_baseline_ev"] = baseline.to_numpy()
            output["shot_zone_ev"] = zone.to_numpy()
            output["shot_context_ev"] = context.to_numpy()
            output["prior_shooter_attempts"] = shooter_attempts.to_numpy()
            output["shot_quality_prior_ev"] = shooter.to_numpy()
            output["context_residual_points"] = (
                output["actual_points"] - output["shot_context_ev"]
            )
            output["prior_model_residual_points"] = (
                output["actual_points"] - output["shot_quality_prior_ev"]
            )
            prediction_frames.append(output)
    prediction_frame = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    return pd.DataFrame(reports), prediction_frame


def select_config(
    shots: pd.DataFrame,
    holdout_year: int,
    zone_shrinks: Iterable[float] = (150.0, 350.0, 700.0),
    context_shrinks: Iterable[float] = (75.0, 175.0, 350.0),
    shooter_shrinks: Iterable[float] = (100.0, 250.0, 500.0),
) -> tuple[ShotQualityConfig, pd.DataFrame]:
    """Select smoothing only on folds strictly before the named holdout."""
    development = shots.loc[shots["season_end_year"].lt(int(holdout_year))].copy()
    rows = []
    for zone_shrink in zone_shrinks:
        for context_shrink in context_shrinks:
            for shooter_shrink in shooter_shrinks:
                config = ShotQualityConfig(
                    zone_shrink=float(zone_shrink),
                    context_shrink=float(context_shrink),
                    shooter_shrink=float(shooter_shrink),
                )
                report, _ = evaluate_season_forward(
                    development,
                    config,
                    keep_predictions=False,
                )
                scored = report.loc[
                    report["model"].eq("context_plus_prior_shooter")
                ].copy()
                pooled_mse = float(
                    np.average(
                        np.square(scored["rmse"]),
                        weights=scored["shots"],
                    )
                )
                rows.append(
                    {
                        **asdict(config),
                        "selection_folds": ",".join(
                            map(str, sorted(scored["validation_year"].astype(int).unique()))
                        ),
                        "pooled_rmse": math.sqrt(pooled_mse),
                        "mean_r2": float(
                            np.average(scored["r2"], weights=scored["shots"])
                        ),
                        "mean_brier": float(
                            np.average(scored["brier"], weights=scored["shots"])
                        ),
                    }
                )
    grid = pd.DataFrame(rows).sort_values(
        ["pooled_rmse", "mean_brier", "zone_shrink", "context_shrink", "shooter_shrink"]
    ).reset_index(drop=True)
    winner = grid.iloc[0]
    selected = ShotQualityConfig(
        zone_shrink=float(winner["zone_shrink"]),
        context_shrink=float(winner["context_shrink"]),
        shooter_shrink=float(winner["shooter_shrink"]),
    )
    return selected, grid


def discover_files(
    raw_dir: Path,
    start_year: int,
    end_year: int,
) -> list[tuple[int, str, Path]]:
    files = []
    for year in range(start_year, end_year + 1):
        for season_type, suffix in (("RS", ""), ("PS", "_PS")):
            if year == end_year and season_type == "PS":
                continue
            path = raw_dir / f"WNBA{year % 100:02d}{suffix}.csv"
            if path.exists():
                files.append((year, season_type, path))
    return files


def discover_classic_files(
    classic_dir: Path,
    start_year: int,
    end_year: int,
) -> list[tuple[int, str, Path]]:
    files = []
    for path in classic_dir.glob("*.csv"):
        game_id = path.stem
        if len(game_id) < 5 or not game_id.isdigit():
            continue
        year = 2000 + int(game_id[3:5])
        if not int(start_year) <= year <= int(end_year):
            continue
        season_type = "PS" if game_id.startswith("104") else "RS"
        files.append((year, season_type, path))
    return sorted(files, key=lambda item: (item[0], item[1], item[2].name))


def load_classic_shots(
    classic_dir: Path,
    start_year: int,
    end_year: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    pieces = []
    summary: dict[tuple[int, str], dict[str, object]] = {}
    columns = [
        "GAME_ID",
        "EVENTNUM",
        "EVENTMSGTYPE",
        "EVENTMSGACTIONTYPE",
        "PERIOD",
        "PCTIMESTRING",
        "HOMEDESCRIPTION",
        "VISITORDESCRIPTION",
        "PLAYER1_ID",
        "PLAYER1_NAME",
    ]
    for year, season_type, path in discover_classic_files(
        classic_dir,
        start_year,
        end_year,
    ):
        raw = pd.read_csv(path, usecols=columns, low_memory=False)
        shots = build_classic_shot_rows(raw, year, season_type)
        if not shots.empty:
            pieces.append(shots)
        key = (year, season_type)
        if key not in summary:
            summary[key] = {
                "path": str(classic_dir),
                "season_end_year": year,
                "season_type": season_type,
                "games": 0,
                "rows": 0,
                "shots": 0,
            }
        summary[key]["games"] += int(not raw.empty)
        summary[key]["rows"] += int(len(raw))
        summary[key]["shots"] += int(len(shots))
    if not pieces:
        return pd.DataFrame(), list(summary.values())
    combined = pd.concat(pieces, ignore_index=True).drop_duplicates(
        ["season_end_year", "season_type", "shot_id"],
        keep="first",
    )
    return combined.reset_index(drop=True), list(summary.values())


def load_shots(
    raw_dir: Path,
    start_year: int,
    end_year: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    pieces = []
    sources = []
    for year, season_type, path in discover_files(raw_dir, start_year, end_year):
        print(f"Loading {path.name}...")
        raw = pd.read_csv(path, low_memory=False)
        shots = build_shot_rows(raw, year, season_type)
        print(f"  shots={len(shots):,}")
        pieces.append(shots)
        sources.append(
            {
                "path": str(path),
                "season_end_year": year,
                "season_type": season_type,
                "rows": int(len(raw)),
                "shots": int(len(shots)),
            }
        )
    if not pieces:
        raise FileNotFoundError(f"No WNBA normalized CSVs found under {raw_dir}")
    combined = pd.concat(pieces, ignore_index=True)
    combined = combined.drop_duplicates(
        ["season_end_year", "season_type", "shot_id"], keep="first"
    ).reset_index(drop=True)
    return combined, sources


def build_shooter_season_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby(
            [
                "season_end_year",
                "shooter_id",
                "shooter_name",
                "talent_bucket",
            ],
            dropna=False,
            observed=True,
        )
        .agg(
            attempts=("actual_points", "size"),
            actual_points_per_shot=("actual_points", "mean"),
            context_ev_per_shot=("shot_context_ev", "mean"),
            prior_model_ev_per_shot=("shot_quality_prior_ev", "mean"),
            context_residual_per_shot=("context_residual_points", "mean"),
            prior_model_residual_per_shot=("prior_model_residual_points", "mean"),
        )
        .reset_index()
        .sort_values(
            ["season_end_year", "talent_bucket", "attempts"],
            ascending=[True, True, False],
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--classic-dir", type=Path, default=DEFAULT_CLASSIC_DIR)
    parser.add_argument("--classic-start-year", type=int, default=2018)
    parser.add_argument(
        "--include-classic",
        action="store_true",
        help=(
            "Add Gabriel classic PBP before the modern start year. Historical "
            "windows are comparison-only unless they clear the modern holdout."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-year", type=int, default=2022)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--zone-shrink", type=float, default=350.0)
    parser.add_argument("--context-shrink", type=float, default=175.0)
    parser.add_argument("--shooter-shrink", type=float, default=250.0)
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Use the supplied smoothing constants instead of pre-holdout selection.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requested_config = ShotQualityConfig(
        zone_shrink=args.zone_shrink,
        context_shrink=args.context_shrink,
        shooter_shrink=args.shooter_shrink,
    )
    modern_shots, modern_sources = load_shots(
        args.raw_dir,
        args.start_year,
        args.end_year,
    )
    if args.include_classic:
        classic_shots, classic_sources = load_classic_shots(
            args.classic_dir,
            args.classic_start_year,
            args.start_year - 1,
        )
    else:
        classic_shots = pd.DataFrame()
        classic_sources: list[dict[str, object]] = []
    all_shots = pd.concat(
        [frame for frame in (classic_shots, modern_shots) if not frame.empty],
        ignore_index=True,
    ).drop_duplicates(
        ["season_end_year", "season_type", "shot_id"],
        keep="last",
    )
    sources = classic_sources + modern_sources
    shots = select_impact_target_shots(all_shots)
    print(
        "Impact-model scope: "
        f"{len(shots):,}/{len(all_shots):,} midrange/three attempts; "
        "rim excluded"
    )
    if args.skip_tuning:
        config = requested_config
        grid = pd.DataFrame()
        selection_folds: list[int] = []
    else:
        print(f"Selecting smoothing on folds strictly before {args.end_year}...")
        config, grid = select_config(shots, holdout_year=args.end_year)
        selection_folds = sorted(
            int(year)
            for year in shots["season_end_year"].astype(int).unique()
            if year < args.end_year
            and year > shots["season_end_year"].astype(int).min()
        )
        print(f"Selected config: {config}")
    report, predictions = evaluate_season_forward(shots, config)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "forward_validation.csv"
    prediction_path = output_dir / "wnba_pbp_shot_quality_forward.parquet"
    current_path = output_dir / f"wnba_pbp_shot_quality_{args.end_year}.parquet"
    shooter_path = output_dir / "shooter_season_summary.csv"
    grid_path = output_dir / "smoothing_grid.csv"
    manifest_path = output_dir / "manifest.json"
    report.to_csv(report_path, index=False)
    if not grid.empty:
        grid.to_csv(grid_path, index=False)
    predictions.to_parquet(prediction_path, index=False)
    predictions.loc[
        predictions["season_end_year"].eq(args.end_year)
    ].to_parquet(current_path, index=False)
    build_shooter_season_summary(predictions).to_csv(shooter_path, index=False)
    manifest = {
        "status": "complete",
        "model": "WNBA repo-owned PBP shot quality v1",
        "impact_integration_scope": {
            "zones": sorted(IMPACT_TARGET_ZONES),
            "components": sorted(IMPACT_TARGET_COMPONENTS),
            "rim_policy": "audit only; excluded from model tuning, validation, and impact target",
        },
        "config": asdict(config),
        "selection_policy": (
            "smoothing selected on season-forward folds strictly before the final "
            "holdout; every validation season excluded from fit"
        ),
        "selection_folds": selection_folds,
        "holdout_year": int(args.end_year),
        "model_feature_columns": MODEL_FEATURE_COLUMNS,
        "forbidden_model_fields": [
            "shotresult",
            "actiontype",
            "description",
            "is_make",
            "actual_points",
            "made_assisted_tag",
        ],
        "sources": sources,
        "source_fga": int(len(all_shots)),
        "modeled_midrange_three_shots": int(len(shots)),
        "seasons": sorted(shots["season_end_year"].astype(int).unique().tolist()),
        "outputs": {
            "validation": str(report_path),
            "predictions": str(prediction_path),
            "current": str(current_path),
            "shooter_summary": str(shooter_path),
            "smoothing_grid": str(grid_path) if not grid.empty else None,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(report.to_string(index=False))
    print(f"Wrote {prediction_path} with {len(predictions):,} forward-scored shots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
