#!/usr/bin/env python3
"""Season-forward challengers to provider WNBA RTSS qSQ and qSP.

qSQ is treated as the provider's league-average-shooter expected eFG rate.
qSP is treated as the provider expected eFG after adding shooter talent.  The
challengers are anchored on those probabilities and may use only release-time
2D tracking fields.  The qSP challenger additionally receives a strongly
shrunk shooter residual calculated from completed *prior games*.

The 2026 RTSS season is deliberately fenced out: it is reserved by the WNBA
RTSS confirmation preregistration.  This script develops on 2023-2025 only and
scores chronological folds 2023 -> 2024 and 2023-2024 -> 2025.
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import OrdinalEncoder


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RTSS_ROOT = ROOT / "downloads" / "rtss-data" / "wnba"
DEFAULT_OUTPUT = ROOT / "wnba_test" / "Results" / "rtss_q_models"
DEVELOPMENT_SEASONS = (2023, 2024, 2025)
CONFIRMATION_SEASON = 2026
IMPACT_TWO_POINT_TYPES = {"jumper", "post", "floater"}
SHOOTER_PRIOR_ATTEMPTS = 250.0
EPSILON = 1e-5

# Explicit allowlist. Same-row realized fields such as outcome, assisted,
# blocked, fouled, rimDepth, arcAngle, and complexOutcome are never features.
CONTEXT_NUMERIC = [
    "qSQ_prob",
    "shot_value",
    "distance",
    "shot_clock",
    "shot_clock_missing",
    "location_x_abs",
    "location_y_abs",
    "dribbles_before",
    "catch_and_shoot",
    "created_from_paint",
    "putback",
    "contested",
    "rearview_contest",
    "closest_def_distance",
    "shot_angle_cos",
    "shot_angle_sin",
    "shooter_speed",
    "shooter_velocity_cos",
    "shooter_velocity_sin",
]
CONTEXT_CATEGORICAL = ["region", "shot_type", "complex_shot_type", "contest_level"]
SHOOTER_NUMERIC = [
    "qSP_prob",
    "provider_shooter_adjustment",
    "shooter_prior_residual",
    "shooter_prior_attempts_log",
]


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def q_score(value: Any) -> float | None:
    """Normalize a provider q score stored as either a fraction or percent."""
    result = _number(value)
    if result is None:
        return None
    if result > 1.0:
        result /= 100.0
    if result < 0.0 or result > 1.0:
        return None
    return float(result)


def q_make_probability(value: Any, shot_value: float) -> float | None:
    """Convert the provider's expected-eFG q score to make probability.

    A q score is on an effective-field-goal scale: two-point make probability
    equals q, while three-point make probability is ``q * 2 / 3``. This scale
    is evident in every season: three-point qSQ is about .49-.50 while WNBA
    three-point make rate is about .33-.34.
    """
    score = q_score(value)
    if score is None or shot_value not in {2.0, 3.0}:
        return None
    return float(np.clip(score * 2.0 / shot_value, 0.0, 1.0))


def _angle_parts(value: Any) -> tuple[float | None, float | None]:
    angle = _number(value)
    if angle is None:
        return None, None
    radians = math.radians(angle)
    return math.cos(radians), math.sin(radians)


def _zone(shot: dict[str, Any]) -> str:
    if bool(shot.get("three")):
        return "three"
    region = str(shot.get("region") or "").strip().lower()
    shot_type = str(shot.get("shotType") or "").strip().lower()
    distance = _number(shot.get("distance"))
    if region in {"ra", "restricted area"} or shot_type in {"dunk", "layup", "tip"}:
        return "rim"
    if distance is not None and distance <= 7.0:
        return "rim"
    return "mid"


def extract_game(path_string: str) -> list[dict[str, Any]]:
    """Extract outcome labels and the pre-release feature allowlist for one game."""
    path = Path(path_string)
    payload = json.loads(path.read_text())
    shots = payload.get("shots", [])
    if not shots:
        return []
    season = next(
        (
            int(row["season"])
            for row in shots
            if _number(row.get("season")) is not None
        ),
        None,
    )
    if season not in DEVELOPMENT_SEASONS:
        return []

    result: list[dict[str, Any]] = []
    for shot in shots:
        three = bool(shot.get("three"))
        shot_value = 3.0 if three else 2.0
        qsq_score = q_score(shot.get("qSQ"))
        qsp_score = q_score(shot.get("qSP"))
        qsq = q_make_probability(shot.get("qSQ"), shot_value)
        qsp = q_make_probability(shot.get("qSP"), shot_value)
        outcome = shot.get("outcome")
        if qsq is None or qsp is None or outcome is None:
            continue
        location = shot.get("location") or [None, None]
        location_x = _number(location[0]) if len(location) >= 1 else None
        location_y = _number(location[1]) if len(location) >= 2 else None
        shot_clock = _number(shot.get("shotClock"))
        shot_angle_cos, shot_angle_sin = _angle_parts(shot.get("shotAngle"))
        velocity_cos, velocity_sin = _angle_parts(shot.get("shooterVelAngle"))
        wall_clock = _number(shot.get("startWallClock"))
        result.append(
            {
                "season": season,
                "game_id": str(shot.get("gameId") or path.parent.name),
                "shot_id": str(shot.get("id") or ""),
                "game_epoch": wall_clock,
                "period": _number(shot.get("period")),
                "game_clock": _number(shot.get("startGameClock")),
                "shooter_id": str(shot.get("shooterId") or ""),
                "is_make": int(bool(outcome)),
                "shot_value": shot_value,
                "actual_points": shot_value * int(bool(outcome)),
                "qSQ_score": qsq_score,
                "qSP_score": qsp_score,
                "qSQ_prob": qsq,
                "qSP_prob": qsp,
                "provider_shooter_adjustment": qsp - qsq,
                "zone": _zone(shot),
                "distance": _number(shot.get("distance")),
                "shot_clock": shot_clock,
                "shot_clock_missing": float(shot_clock is None),
                "location_x_abs": abs(location_x) if location_x is not None else None,
                "location_y_abs": abs(location_y) if location_y is not None else None,
                "dribbles_before": _number(shot.get("dribblesBefore")),
                "catch_and_shoot": float(bool(shot.get("catchAndShoot"))),
                "created_from_paint": float(bool(shot.get("createdFromPaint"))),
                "putback": float(bool(shot.get("putback"))),
                "contested": float(bool(shot.get("contested"))),
                "rearview_contest": float(bool(shot.get("hasRearviewContest"))),
                "closest_def_distance": _number(shot.get("closestDefDist")),
                "shot_angle_cos": shot_angle_cos,
                "shot_angle_sin": shot_angle_sin,
                "shooter_speed": _number(shot.get("shooterSpeed")),
                "shooter_velocity_cos": velocity_cos,
                "shooter_velocity_sin": velocity_sin,
                "region": str(shot.get("region") or "missing"),
                "shot_type": str(shot.get("shotType") or "missing"),
                "complex_shot_type": str(shot.get("complexShotType") or "missing"),
                "contest_level": str(shot.get("contestLevel") or "missing"),
            }
        )
    return result


def extract_shots(rtss_root: Path, workers: int) -> pd.DataFrame:
    paths = sorted(rtss_root.rglob("basketball-markings.json"))
    if not paths:
        raise FileNotFoundError(f"no RTSS markings files under {rtss_root}")
    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for rows in pool.map(extract_game, map(str, paths), chunksize=2):
            records.extend(rows)
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise RuntimeError("no development-season RTSS shots were extracted")
    if int(frame["season"].max()) >= CONFIRMATION_SEASON:
        raise RuntimeError("confirmation-season outcomes crossed the development fence")
    frame["game_epoch"] = pd.to_numeric(frame["game_epoch"], errors="coerce")
    # All shots in a game share the same chronological game key. This is what
    # makes every empirical shooter history current-game-excluded.
    game_epoch = frame.groupby("game_id")["game_epoch"].transform("min")
    frame["game_epoch"] = game_epoch.fillna(
        frame["season"].astype(float) * 1e15
    )
    return add_prior_game_shooter_history(frame)


def add_prior_game_shooter_history(
    shots: pd.DataFrame,
    shrink_attempts: float = SHOOTER_PRIOR_ATTEMPTS,
) -> pd.DataFrame:
    """Attach qSQ-relative shooter talent using completed earlier games only."""
    frame = shots.copy()
    game = (
        frame.groupby(["shooter_id", "game_id"], as_index=False)
        .agg(
            game_epoch=("game_epoch", "min"),
            game_attempts=("is_make", "size"),
            game_residual=("is_make", "sum"),
            game_qsq_sum=("qSQ_prob", "sum"),
        )
        .sort_values(["shooter_id", "game_epoch", "game_id"])
    )
    game["game_residual"] -= game["game_qsq_sum"]
    game["prior_attempts"] = game.groupby("shooter_id")["game_attempts"].cumsum() - game["game_attempts"]
    game["prior_residual_sum"] = game.groupby("shooter_id")["game_residual"].cumsum() - game["game_residual"]
    game["shooter_prior_residual"] = game["prior_residual_sum"] / (
        game["prior_attempts"] + float(shrink_attempts)
    )
    game["shooter_prior_attempts_log"] = np.log1p(game["prior_attempts"])
    attached = frame.merge(
        game[
            [
                "shooter_id",
                "game_id",
                "prior_attempts",
                "shooter_prior_residual",
                "shooter_prior_attempts_log",
            ]
        ],
        on=["shooter_id", "game_id"],
        how="left",
        validate="many_to_one",
    )
    return attached


@dataclass(frozen=True)
class FeatureSpec:
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]


class AnchoredTree:
    """Small, regularized tree model with train-fitted categorical encoding."""

    def __init__(self, spec: FeatureSpec):
        self.spec = spec
        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
            dtype=np.float64,
        )
        self.numeric_medians: pd.Series | None = None
        self.model = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.05,
            max_iter=180,
            max_leaf_nodes=15,
            min_samples_leaf=100,
            l2_regularization=4.0,
            early_stopping=False,
            random_state=20260812,
            categorical_features=[True] * len(spec.categorical)
            + [False] * len(spec.numeric),
        )

    def _transform(self, frame: pd.DataFrame, fit: bool) -> np.ndarray:
        categorical = frame.loc[:, self.spec.categorical].fillna("missing").astype(str)
        if fit:
            encoded = self.encoder.fit_transform(categorical)
            numeric = frame.loc[:, self.spec.numeric].apply(pd.to_numeric, errors="coerce")
            self.numeric_medians = numeric.median().fillna(0.0)
        else:
            encoded = self.encoder.transform(categorical)
            numeric = frame.loc[:, self.spec.numeric].apply(pd.to_numeric, errors="coerce")
        assert self.numeric_medians is not None
        numeric_array = numeric.fillna(self.numeric_medians).to_numpy(dtype=float)
        return np.column_stack([encoded, numeric_array])

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> "AnchoredTree":
        self.model.fit(self._transform(frame, fit=True), target)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.clip(
            self.model.predict_proba(self._transform(frame, fit=False))[:, 1],
            EPSILON,
            1.0 - EPSILON,
        )


def _metric_row(
    season: int,
    scope: str,
    model: str,
    frame: pd.DataFrame,
    probability: np.ndarray,
) -> dict[str, Any]:
    actual = frame["is_make"].to_numpy(dtype=float)
    value = frame["shot_value"].to_numpy(dtype=float)
    prediction = np.clip(np.asarray(probability, dtype=float), EPSILON, 1.0 - EPSILON)
    return {
        "season": season,
        "scope": scope,
        "model": model,
        "shots": len(frame),
        "games": int(frame["game_id"].nunique()),
        "actual_fg": float(actual.mean()),
        "predicted_fg": float(prediction.mean()),
        "calibration_bias": float(prediction.mean() - actual.mean()),
        "brier": float(np.mean((prediction - actual) ** 2)),
        "log_loss": float(log_loss(actual, prediction, labels=[0, 1])),
        "points_mse": float(np.mean((prediction * value - actual * value) ** 2)),
    }


def clustered_ci(
    frame: pd.DataFrame,
    challenger: np.ndarray,
    baseline: np.ndarray,
    metric: str,
    draws: int = 2000,
) -> tuple[float, float, float]:
    actual = frame["is_make"].to_numpy(dtype=float)
    value = frame["shot_value"].to_numpy(dtype=float)
    challenger = np.clip(challenger, EPSILON, 1.0 - EPSILON)
    baseline = np.clip(baseline, EPSILON, 1.0 - EPSILON)
    if metric == "brier":
        delta = (challenger - actual) ** 2 - (baseline - actual) ** 2
    elif metric == "log_loss":
        delta = -actual * np.log(challenger) - (1 - actual) * np.log(1 - challenger)
        delta -= -actual * np.log(baseline) - (1 - actual) * np.log(1 - baseline)
    elif metric == "points_mse":
        delta = (challenger * value - actual * value) ** 2
        delta -= (baseline * value - actual * value) ** 2
    else:
        raise ValueError(metric)
    grouped = pd.DataFrame(
        {"game_id": frame["game_id"].to_numpy(), "delta": delta}
    ).groupby("game_id")["delta"].agg(["sum", "size"])
    point = float(delta.mean())
    rng = np.random.default_rng(20260812)
    indices = rng.integers(0, len(grouped), size=(draws, len(grouped)))
    sums = grouped["sum"].to_numpy()[indices].sum(axis=1)
    counts = grouped["size"].to_numpy()[indices].sum(axis=1)
    samples = sums / counts
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return point, float(lower), float(upper)


def _scope_frame(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "all":
        return frame
    if scope == "mid_three":
        shot_type = frame["shot_type"].fillna("").astype(str).str.lower()
        return frame[
            frame["shot_value"].eq(3.0) | shot_type.isin(IMPACT_TWO_POINT_TYPES)
        ]
    raise ValueError(scope)


def run_forward_models(shots: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    context_spec = FeatureSpec(tuple(CONTEXT_NUMERIC), tuple(CONTEXT_CATEGORICAL))
    shooter_spec = FeatureSpec(
        tuple(CONTEXT_NUMERIC + SHOOTER_NUMERIC), tuple(CONTEXT_CATEGORICAL)
    )

    for scope in ("all", "mid_three"):
        scoped = _scope_frame(shots, scope).copy()
        for test_season in (2024, 2025):
            train = scoped[scoped["season"] < test_season].copy()
            test = scoped[scoped["season"] == test_season].copy()
            if train.empty or test.empty:
                continue
            context_model = AnchoredTree(context_spec).fit(
                train, train["is_make"].to_numpy(dtype=int)
            )
            shooter_model = AnchoredTree(shooter_spec).fit(
                train, train["is_make"].to_numpy(dtype=int)
            )
            context_prediction = context_model.predict(test)
            shooter_prediction = shooter_model.predict(test)
            qsq = test["qSQ_prob"].to_numpy(dtype=float)
            qsp = test["qSP_prob"].to_numpy(dtype=float)
            rim = test["zone"].eq("rim").to_numpy()
            # Conservative interpretable challengers: the recurring provider
            # miss is at the rim, while MID/3 q values are already strong.
            # Keep provider predictions untouched off-rim and apply the learned
            # correction only where both chronological folds support it.
            rim_gated_context = qsq.copy()
            rim_gated_context[rim] = context_prediction[rim]
            rim_gated_shooter = qsp.copy()
            rim_gated_shooter[rim] = shooter_prediction[rim]

            candidates = {
                "provider_qSQ": qsq,
                "provider_qSP": qsp,
                "anchored_context_qSQ": context_prediction,
                "anchored_context_plus_shooter": shooter_prediction,
                "rim_gated_context_qSQ": rim_gated_context,
                "rim_gated_context_plus_shooter": rim_gated_shooter,
            }
            for name, prediction in candidates.items():
                metrics.append(_metric_row(test_season, scope, name, test, prediction))
            for challenger_name, challenger, baseline_name, baseline in (
                ("anchored_context_qSQ", context_prediction, "provider_qSQ", qsq),
                ("anchored_context_plus_shooter", shooter_prediction, "provider_qSP", qsp),
                ("anchored_context_plus_shooter", shooter_prediction, "provider_qSQ", qsq),
                ("rim_gated_context_qSQ", rim_gated_context, "provider_qSQ", qsq),
                (
                    "rim_gated_context_plus_shooter",
                    rim_gated_shooter,
                    "provider_qSP",
                    qsp,
                ),
            ):
                for metric in ("brier", "log_loss", "points_mse"):
                    point, lower, upper = clustered_ci(
                        test, challenger, baseline, metric
                    )
                    comparisons.append(
                        {
                            "season": test_season,
                            "scope": scope,
                            "challenger": challenger_name,
                            "baseline": baseline_name,
                            "metric": metric,
                            "delta": point,
                            "ci_low": lower,
                            "ci_high": upper,
                            "games": int(test["game_id"].nunique()),
                            "shots": len(test),
                        }
                    )
            prediction_frame = test[
                [
                    "season",
                    "game_id",
                    "shot_id",
                    "shooter_id",
                    "zone",
                    "is_make",
                    "shot_value",
                    "qSQ_prob",
                    "qSP_prob",
                ]
            ].copy()
            prediction_frame["scope"] = scope
            prediction_frame["anchored_context_qSQ"] = context_prediction
            prediction_frame["anchored_context_plus_shooter"] = shooter_prediction
            prediction_frame["rim_gated_context_qSQ"] = rim_gated_context
            prediction_frame["rim_gated_context_plus_shooter"] = rim_gated_shooter
            predictions.append(prediction_frame)
    return (
        pd.DataFrame(metrics),
        pd.DataFrame(comparisons),
        pd.concat(predictions, ignore_index=True),
    )


def coverage_summary(shots: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for season, group in shots.groupby("season"):
        row: dict[str, Any] = {
            "season": int(season),
            "games": int(group["game_id"].nunique()),
            "shots": len(group),
            "mid_three_shots": int(
                (
                    group["shot_value"].eq(3.0)
                    | group["shot_type"]
                    .fillna("")
                    .astype(str)
                    .str.lower()
                    .isin(IMPACT_TWO_POINT_TYPES)
                ).sum()
            ),
            "actual_fg": float(group["is_make"].mean()),
            "qSQ_mean": float(group["qSQ_prob"].mean()),
            "qSP_mean": float(group["qSP_prob"].mean()),
        }
        for feature in CONTEXT_NUMERIC + CONTEXT_CATEGORICAL:
            row[f"{feature}_coverage"] = float(group[feature].notna().mean())
        rows.append(row)
    return pd.DataFrame(rows)


def write_manifest(
    output: Path,
    shots: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    manifest = {
        "status": "research_only_development",
        "development_seasons": list(DEVELOPMENT_SEASONS),
        "confirmation_season_fenced": CONFIRMATION_SEASON,
        "q_contract": {
            "qSQ": "provider league-average-shooter expected eFG score",
            "qSP": "provider expected eFG score including shooter",
            "make_probability_conversion": "q_score * 2 / shot_value",
        },
        "rows": int(len(shots)),
        "games": int(shots["game_id"].nunique()),
        "feature_contract": {
            "context_numeric": CONTEXT_NUMERIC,
            "context_categorical": CONTEXT_CATEGORICAL,
            "shooter_only_numeric": SHOOTER_NUMERIC,
            "forbidden_same_row_labels": [
                "outcome",
                "actual_points",
                "assisted",
                "assistOpp",
                "blocked",
                "fouled",
                "rimDepth",
                "rimLeftRight",
                "arcAngle",
                "complexOutcome",
            ],
        },
        "shooter_history": {
            "current_game_excluded": True,
            "shrink_attempts": SHOOTER_PRIOR_ATTEMPTS,
            "target": "actual make minus provider qSQ",
        },
        "models": {
            "type": "regularized HistGradientBoostingClassifier",
            "parameters": asdict(
                _ModelParameters()
            ),
        },
        "metric_rows": int(len(metrics)),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


@dataclass(frozen=True)
class _ModelParameters:
    learning_rate: float = 0.05
    max_iter: int = 180
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 100
    l2_regularization: float = 4.0


def run(rtss_root: Path, output: Path, workers: int, rebuild: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    shot_path = output / "development_shots.parquet"
    if shot_path.exists() and not rebuild:
        shots = pd.read_parquet(shot_path)
    else:
        shots = extract_shots(rtss_root, workers)
        shots.to_parquet(shot_path, index=False)
    if int(shots["season"].max()) >= CONFIRMATION_SEASON:
        raise RuntimeError("confirmation-season rows found in cached development data")
    metrics, comparisons, predictions = run_forward_models(shots)
    coverage_summary(shots).to_csv(output / "coverage.csv", index=False)
    metrics.to_csv(output / "forward_metrics.csv", index=False)
    comparisons.to_csv(output / "paired_game_clustered_comparisons.csv", index=False)
    predictions.to_parquet(output / "forward_predictions.parquet", index=False)
    write_manifest(output, shots, metrics)
    print(metrics.to_string(index=False))
    print("\nPaired challenger-minus-provider loss (negative is better):")
    print(comparisons.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtss-root", type=Path, default=DEFAULT_RTSS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    run(
        args.rtss_root.resolve(),
        args.output.resolve(),
        max(1, args.workers),
        args.rebuild,
    )


if __name__ == "__main__":
    main()
