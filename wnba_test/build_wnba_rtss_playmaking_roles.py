#!/usr/bin/env python3
"""Estimate WNBA shooter, passer, and presence shot-value roles from RTSS.

This is a research-only lane.  Each valid RTSS shot supplies one exact
five-on-five row from ``chance_players``.  The joint ridge model contains:

* an exposure-averaged offensive parent for every player;
* a centered shooter interaction;
* a centered potential-assist passer interaction; and
* a defensive player block.

The omitted offensive role is presence.  Centering the two explicit roles by
each player's observed shot exposure makes the offensive parent equal the
exposure-weighted sum of shooter, passer, and presence values by construction.

The script also prices RTSS pass turnovers and runs leave-one-season-out
development tests of a compact playmaking prior:

``on-ball time % + potential assists/100 + rim assists/100``.

Nothing here changes the frozen live WNBA rating.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wnba_test.build_wnba_box_priors import load_box_features
from wnba_test.build_wnba_rtss_prior_features import (
    DEFAULT_CROSSWALK,
    DEFAULT_OUTPUT as DEFAULT_FEATURES,
    DEFAULT_RTSS_ROOT,
    VALID_CROSSWALK_STATUSES,
    _number,
    _player,
    _season,
    build_features,
)

DEFAULT_OUTPUT = (
    ROOT / "wnba_test" / "Results" / "rtss_playmaking_roles"
)
ROLE_ALPHA_GRID = (100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0)
PRIOR_ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
RIM_REGIONS = {"ra", "restricted area"}


def _flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "made"}
    return bool(value)


def _q_probability(value: Any) -> float | None:
    """Normalize provider q values that can appear as either 0-1 or 0-100."""
    numeric = _number(value)
    if numeric is None:
        return None
    probability = numeric / 100.0 if numeric > 1.0 else numeric
    if probability < 0.0 or probability > 1.0:
        return None
    return probability


def _zone(row: dict[str, Any]) -> str:
    if _flag(row.get("three")):
        return "three"
    region = str(row.get("region") or "").strip().lower()
    return "rim" if region in RIM_REGIONS else "mid"


def _valid_chance_lineups(
    data: dict[str, Any],
) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in data.get("chance_players", []):
        chance_id = _player(row.get("chanceId"))
        if chance_id is not None:
            grouped[chance_id].append(row)

    result: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for chance_id, rows in grouped.items():
        offense = tuple(
            sorted(
                {
                    str(row["playerId"])
                    for row in rows
                    if _flag(row.get("offense")) and _player(row.get("playerId"))
                }
            )
        )
        defense = tuple(
            sorted(
                {
                    str(row["playerId"])
                    for row in rows
                    if not _flag(row.get("offense")) and _player(row.get("playerId"))
                }
            )
        )
        if (
            len(rows) == 10
            and len(offense) == 5
            and len(defense) == 5
            and not set(offense).intersection(defense)
        ):
            result[chance_id] = (offense, defense)
    return result


def _extract_game(path_string: str) -> dict[str, Any]:
    path = Path(path_string)
    data = json.loads(path.read_text())
    season = _season(data)
    if season not in {2023, 2024, 2025}:
        return {"season": season, "shots": [], "pass_stats": [], "diagnostics": {}}

    lineups = _valid_chance_lineups(data)
    diagnostics = Counter(
        games=1,
        chance_groups=len(
            {
                str(row.get("chanceId"))
                for row in data.get("chance_players", [])
                if row.get("chanceId") is not None
            }
        ),
        valid_chance_groups=len(lineups),
    )
    shots: list[dict[str, Any]] = []
    for row in data.get("shots", []):
        diagnostics["shot_rows"] += 1
        chance_id = _player(row.get("chanceId"))
        if chance_id is None or chance_id not in lineups:
            diagnostics["shot_missing_valid_lineup"] += 1
            continue
        offense, defense = lineups[chance_id]
        shooter = _player(row.get("shooterId"))
        if shooter is None or shooter not in offense:
            diagnostics["shot_invalid_shooter"] += 1
            continue
        assist_opp = _flag(row.get("assistOpp"))
        raw_passer = _player(row.get("passerId"))
        passer = (
            raw_passer
            if assist_opp
            and raw_passer in offense
            and raw_passer != shooter
            else None
        )
        if assist_opp and passer is None:
            diagnostics["assist_opp_missing_valid_passer"] += 1

        shot_value = 3.0 if _flag(row.get("three")) else 2.0
        q_probability = _q_probability(row.get("qSQ"))
        shots.append(
            {
                "season": int(season),
                "game_id": str(row.get("gameId") or path.parent.name),
                "shot_id": str(row.get("id") or ""),
                "chance_id": chance_id,
                "offense": offense,
                "defense": defense,
                "shooter_id": shooter,
                "passer_id": passer,
                "assist_opp": assist_opp,
                "zone": _zone(row),
                "actual_points": shot_value if _flag(row.get("outcome")) else 0.0,
                "provider_qsq_points": (
                    shot_value * q_probability
                    if q_probability is not None
                    else np.nan
                ),
            }
        )
        diagnostics["valid_shots"] += 1
        diagnostics[f"valid_{_zone(row)}_shots"] += 1
        diagnostics["potential_assisted_shots"] += int(passer is not None)

    pass_stats: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in data.get("passes", []):
        passer = _player(row.get("passerId"))
        if passer is None:
            continue
        pass_stats[passer]["passes"] += 1
        pass_stats[passer]["pass_turnovers"] += int(_flag(row.get("turnover")))
        pass_stats[passer]["potential_assists"] += int(_flag(row.get("assistOpp")))

    return {
        "season": int(season),
        "shots": shots,
        "pass_stats": [
            {
                "season": int(season),
                "provider_player_id": player_id,
                **dict(values),
            }
            for player_id, values in pass_stats.items()
        ],
        "diagnostics": dict(diagnostics),
    }


def _load_rtss(
    rtss_root: Path,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = sorted(rtss_root.rglob("basketball-markings.json"))
    if not paths:
        raise FileNotFoundError(f"no basketball-markings.json files under {rtss_root}")

    shot_rows: list[dict[str, Any]] = []
    pass_totals: defaultdict[tuple[int, str], Counter[str]] = defaultdict(Counter)
    diagnostic_totals: defaultdict[int, Counter[str]] = defaultdict(Counter)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for game_number, result in enumerate(
            pool.map(_extract_game, map(str, paths), chunksize=2),
            start=1,
        ):
            season = result.get("season")
            if season in {2023, 2024, 2025}:
                shot_rows.extend(result["shots"])
                diagnostic_totals[int(season)].update(result["diagnostics"])
                for row in result["pass_stats"]:
                    key = (int(row["season"]), str(row["provider_player_id"]))
                    pass_totals[key].update(
                        {
                            "passes": row.get("passes", 0),
                            "pass_turnovers": row.get("pass_turnovers", 0),
                            "potential_assists": row.get("potential_assists", 0),
                        }
                    )
            if game_number % 50 == 0 or game_number == len(paths):
                print(f"loaded {game_number}/{len(paths)} RTSS games", flush=True)

    shots = pd.DataFrame(shot_rows)
    passes = pd.DataFrame(
        [
            {
                "season": season,
                "provider_player_id": player_id,
                **dict(values),
            }
            for (season, player_id), values in pass_totals.items()
        ]
    )
    diagnostics = pd.DataFrame(
        [
            {"season": season, **dict(values)}
            for season, values in sorted(diagnostic_totals.items())
        ]
    )
    return shots, passes, diagnostics


def _role_exposures(
    shots: pd.DataFrame,
) -> tuple[list[str], dict[str, int], np.ndarray, np.ndarray, np.ndarray]:
    on_floor = Counter()
    shooter = Counter()
    passer = Counter()
    players: set[str] = set()
    for row in shots.itertuples(index=False):
        offense = tuple(row.offense)
        defense = tuple(row.defense)
        players.update(offense)
        players.update(defense)
        for player_id in offense:
            on_floor[player_id] += 1
        shooter[str(row.shooter_id)] += 1
        if row.passer_id is not None and not pd.isna(row.passer_id):
            passer[str(row.passer_id)] += 1
    player_list = sorted(players)
    index = {player_id: position for position, player_id in enumerate(player_list)}
    exposure = np.array([on_floor[player_id] for player_id in player_list], dtype=float)
    shooter_share = np.array(
        [
            shooter[player_id] / max(on_floor[player_id], 1)
            for player_id in player_list
        ],
        dtype=float,
    )
    passer_share = np.array(
        [
            passer[player_id] / max(on_floor[player_id], 1)
            for player_id in player_list
        ],
        dtype=float,
    )
    return player_list, index, exposure, shooter_share, passer_share


def _role_design(
    shots: pd.DataFrame,
) -> tuple[sparse.csr_matrix, list[str], np.ndarray, np.ndarray, np.ndarray]:
    (
        players,
        player_index,
        exposure,
        shooter_share,
        passer_share,
    ) = _role_exposures(shots)
    n_players = len(players)
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    for row_number, row in enumerate(shots.itertuples(index=False)):
        passer = (
            str(row.passer_id)
            if row.passer_id is not None and not pd.isna(row.passer_id)
            else None
        )
        for player_id in row.offense:
            position = player_index[str(player_id)]
            row_indices.append(row_number)
            column_indices.append(position)
            values.append(1.0)

            shooter_value = float(str(player_id) == str(row.shooter_id)) - shooter_share[position]
            if shooter_value:
                row_indices.append(row_number)
                column_indices.append(2 * n_players + position)
                values.append(shooter_value)

            passer_value = float(str(player_id) == passer) - passer_share[position]
            if passer_value:
                row_indices.append(row_number)
                column_indices.append(3 * n_players + position)
                values.append(passer_value)

        for player_id in row.defense:
            position = player_index[str(player_id)]
            row_indices.append(row_number)
            column_indices.append(n_players + position)
            values.append(1.0)

    design = sparse.coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(shots), 4 * n_players),
        dtype=float,
    ).tocsr()
    return design, players, exposure, shooter_share, passer_share


def _fit_scaled_ridge(
    design: sparse.csr_matrix,
    target: np.ndarray,
    base_alpha: float,
    role_alpha: float,
) -> tuple[Ridge, np.ndarray, np.ndarray]:
    n_players = design.shape[1] // 4
    penalties = np.concatenate(
        [
            np.full(2 * n_players, base_alpha, dtype=float),
            np.full(2 * n_players, role_alpha, dtype=float),
        ]
    )
    scales = np.sqrt(penalties)
    scaled_design = design @ sparse.diags(1.0 / scales)
    model = Ridge(
        alpha=1.0,
        fit_intercept=True,
        solver="lsqr",
        tol=1e-6,
    )
    model.fit(scaled_design, target)
    coefficients = np.asarray(model.coef_, dtype=float) / scales
    return model, scaled_design, coefficients


def _choose_role_alpha(
    design: sparse.csr_matrix,
    target: np.ndarray,
    groups: np.ndarray,
    base_alpha: float,
    alpha_grid: tuple[float, ...],
) -> tuple[float, pd.DataFrame]:
    unique_groups = np.unique(groups)
    splits = min(5, len(unique_groups))
    if splits < 2:
        return float(alpha_grid[-1]), pd.DataFrame()
    fold_iterator = list(GroupKFold(n_splits=splits).split(design, target, groups))
    records: list[dict[str, float]] = []
    for role_alpha in alpha_grid:
        fold_mse = []
        for fold, (train_index, test_index) in enumerate(fold_iterator, start=1):
            model, train_design, _ = _fit_scaled_ridge(
                design[train_index],
                target[train_index],
                base_alpha,
                float(role_alpha),
            )
            n_players = design.shape[1] // 4
            penalties = np.concatenate(
                [
                    np.full(2 * n_players, base_alpha, dtype=float),
                    np.full(2 * n_players, role_alpha, dtype=float),
                ]
            )
            test_design = design[test_index] @ sparse.diags(
                1.0 / np.sqrt(penalties)
            )
            prediction = model.predict(test_design)
            mse = float(np.mean(np.square(target[test_index] - prediction)))
            fold_mse.append(mse)
            records.append(
                {
                    "role_alpha": float(role_alpha),
                    "fold": fold,
                    "mse": mse,
                    "train_rows": len(train_index),
                    "test_rows": len(test_index),
                }
            )
    cv = pd.DataFrame(records)
    summary = (
        cv.groupby("role_alpha", as_index=False)
        .agg(mean_mse=("mse", "mean"), se_mse=("mse", "sem"))
        .sort_values("role_alpha")
    )
    best = summary.loc[summary["mean_mse"].idxmin()]
    standard_error = (
        float(best["se_mse"]) if np.isfinite(best["se_mse"]) else 0.0
    )
    threshold = float(best["mean_mse"] + standard_error)
    eligible = summary[summary["mean_mse"] <= threshold]
    selected = float(eligible["role_alpha"].max())
    cv = cv.merge(summary, on="role_alpha", how="left")
    cv["selected"] = cv["role_alpha"].eq(selected)
    return selected, cv


def _conditional_role_frame(
    season: int,
    target_name: str,
    players: list[str],
    exposure: np.ndarray,
    shooter_share: np.ndarray,
    passer_share: np.ndarray,
    coefficients: np.ndarray,
    intercept: float,
    base_alpha: float,
    role_alpha: float,
) -> pd.DataFrame:
    n_players = len(players)
    offense_parent = coefficients[:n_players]
    defense_effect = coefficients[n_players : 2 * n_players]
    shooter_delta = coefficients[2 * n_players : 3 * n_players]
    passer_delta = coefficients[3 * n_players : 4 * n_players]
    presence_share = 1.0 - shooter_share - passer_share

    conditional_presence = (
        offense_parent
        - shooter_share * shooter_delta
        - passer_share * passer_delta
    )
    conditional_shooter = conditional_presence + shooter_delta
    conditional_passer = conditional_presence + passer_delta
    shooter_contribution = shooter_share * conditional_shooter
    passer_contribution = passer_share * conditional_passer
    presence_contribution = presence_share * conditional_presence
    identity_gap = (
        shooter_contribution
        + passer_contribution
        + presence_contribution
        - offense_parent
    )
    if float(np.max(np.abs(identity_gap))) > 1e-10:
        raise AssertionError("shooter + passer + presence does not close to parent")

    return pd.DataFrame(
        {
            "season": int(season),
            "target": target_name,
            "provider_player_id": players,
            "shots_on_floor": exposure,
            "shooter_share": shooter_share,
            "passer_share": passer_share,
            "presence_share": presence_share,
            "offensive_parent_per_shot": offense_parent,
            "defensive_effect_per_shot": defense_effect,
            "shooter_delta_vs_presence": shooter_delta,
            "passer_delta_vs_presence": passer_delta,
            "conditional_shooter_value": conditional_shooter,
            "conditional_passer_value": conditional_passer,
            "conditional_presence_value": conditional_presence,
            "shooter_contribution_per_shot": shooter_contribution,
            "passer_contribution_per_shot": passer_contribution,
            "presence_contribution_per_shot": presence_contribution,
            "role_identity_gap": identity_gap,
            "intercept": float(intercept),
            "base_alpha": float(base_alpha),
            "role_alpha": float(role_alpha),
        }
    )


def _fit_season_roles(
    shots: pd.DataFrame,
    season: int,
    target_name: str,
    base_alpha: float,
    alpha_grid: tuple[float, ...],
    selected_alpha: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_rows = shots[shots["season"].eq(season)].copy()
    model_rows = model_rows[np.isfinite(pd.to_numeric(model_rows[target_name], errors="coerce"))]
    model_rows = model_rows.reset_index(drop=True)
    design, players, exposure, shooter_share, passer_share = _role_design(model_rows)
    target = pd.to_numeric(model_rows[target_name], errors="raise").to_numpy(float)
    if selected_alpha is None:
        selected_alpha, cv = _choose_role_alpha(
            design,
            target,
            model_rows["game_id"].astype(str).to_numpy(),
            base_alpha,
            alpha_grid,
        )
    else:
        cv = pd.DataFrame()
    model, _scaled_design, coefficients = _fit_scaled_ridge(
        design,
        target,
        base_alpha,
        selected_alpha,
    )
    roles = _conditional_role_frame(
        season,
        target_name,
        players,
        exposure,
        shooter_share,
        passer_share,
        coefficients,
        float(model.intercept_),
        base_alpha,
        selected_alpha,
    )
    if not cv.empty:
        cv.insert(0, "target", target_name)
        cv.insert(0, "season", int(season))
    return roles, cv


def _common_role_alpha(cv: pd.DataFrame) -> float:
    """Choose one comparable role shrinkage level across all target seasons."""
    summary = (
        cv.groupby("role_alpha", as_index=False)
        .agg(mean_mse=("mse", "mean"), se_mse=("mse", "sem"))
        .sort_values("role_alpha")
    )
    best = summary.loc[summary["mean_mse"].idxmin()]
    standard_error = (
        float(best["se_mse"]) if np.isfinite(best["se_mse"]) else 0.0
    )
    threshold = float(best["mean_mse"] + standard_error)
    return float(summary.loc[summary["mean_mse"] <= threshold, "role_alpha"].max())


def _weighted_metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    weight = np.asarray(weight, dtype=float)
    weight = np.maximum(weight, 0.0)
    weight_sum = max(float(weight.sum()), 1.0)
    mean_actual = float(np.sum(weight * actual) / weight_sum)
    residual = actual - prediction
    mse = float(np.sum(weight * np.square(residual)) / weight_sum)
    total = float(np.sum(weight * np.square(actual - mean_actual)) / weight_sum)
    centered_actual = actual - mean_actual
    mean_prediction = float(np.sum(weight * prediction) / weight_sum)
    centered_prediction = prediction - mean_prediction
    covariance = float(np.sum(weight * centered_actual * centered_prediction))
    denominator = float(
        np.sqrt(
            np.sum(weight * np.square(centered_actual))
            * np.sum(weight * np.square(centered_prediction))
        )
    )
    return {
        "rmse": float(np.sqrt(mse)),
        "r2": 1.0 - mse / total if total > 0 else np.nan,
        "correlation": covariance / denominator if denominator > 0 else np.nan,
    }


def _fit_prior(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    alpha: float,
) -> Pipeline:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )
    model.fit(
        frame[features],
        frame[target],
        ridge__sample_weight=frame["off_chances"],
    )
    return model


def _select_prior_alpha(
    train: pd.DataFrame,
    features: list[str],
    target: str,
) -> float:
    years = sorted(train["season"].unique())
    if len(years) < 2:
        return 10.0
    scores = []
    for alpha in PRIOR_ALPHA_GRID:
        fold_mse = []
        for heldout in years:
            fit_rows = train[train["season"].ne(heldout)]
            test_rows = train[train["season"].eq(heldout)]
            if fit_rows.empty or test_rows.empty:
                continue
            model = _fit_prior(fit_rows, features, target, alpha)
            prediction = model.predict(test_rows[features])
            metrics = _weighted_metrics(
                test_rows[target].to_numpy(),
                prediction,
                test_rows["off_chances"].to_numpy(),
            )
            fold_mse.append(metrics["rmse"] ** 2)
        if fold_mse:
            scores.append((float(np.mean(fold_mse)), float(alpha)))
    return min(scores)[1] if scores else 10.0


def _evaluate_compact_priors(
    frame: pd.DataFrame,
    min_chances: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_sets = {
        "onball_only": ["rtss_on_ball_time_pct"],
        "onball_rim": ["rtss_on_ball_time_pct", "rim_assists100"],
        "onball_potast": [
            "rtss_on_ball_time_pct",
            "rtss_potential_assist_rate100",
        ],
        "potast_rim": ["rtss_potential_assist_rate100", "rim_assists100"],
        "compact_three": [
            "rtss_on_ball_time_pct",
            "rtss_potential_assist_rate100",
            "rim_assists100",
        ],
    }
    targets = [
        "passer_shot_value_per100_chances",
        "net_playmaking_value_per100_chances",
        "presence_value_per100_chances",
    ]
    required = sorted(
        {
            "season",
            "player_id",
            "player_name",
            "off_chances",
            *targets,
            *(feature for features in feature_sets.values() for feature in features),
        }
    )
    rows = frame[required].replace([np.inf, -np.inf], np.nan).dropna().copy()
    rows = rows[rows["off_chances"] >= min_chances].copy()

    fold_records: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    for target in targets:
        for heldout in sorted(rows["season"].unique()):
            train = rows[rows["season"].ne(heldout)]
            test = rows[rows["season"].eq(heldout)]
            baseline = float(
                np.average(train[target], weights=train["off_chances"])
            )
            baseline_prediction = np.full(len(test), baseline)
            baseline_metrics = _weighted_metrics(
                test[target].to_numpy(),
                baseline_prediction,
                test["off_chances"].to_numpy(),
            )
            fold_records.append(
                {
                    "target": target,
                    "heldout_season": int(heldout),
                    "model": "train_weighted_mean",
                    "alpha": np.nan,
                    **baseline_metrics,
                    "rows": len(test),
                }
            )
            for row, prediction in zip(
                test.itertuples(index=False),
                baseline_prediction,
            ):
                prediction_records.append(
                    {
                        "target": target,
                        "heldout_season": int(heldout),
                        "model": "train_weighted_mean",
                        "player_id": row.player_id,
                        "player_name": row.player_name,
                        "off_chances": row.off_chances,
                        "actual": getattr(row, target),
                        "prediction": float(prediction),
                    }
                )

            for model_name, features in feature_sets.items():
                alpha = _select_prior_alpha(train, features, target)
                model = _fit_prior(train, features, target, alpha)
                prediction = model.predict(test[features])
                metrics = _weighted_metrics(
                    test[target].to_numpy(),
                    prediction,
                    test["off_chances"].to_numpy(),
                )
                fold_records.append(
                    {
                        "target": target,
                        "heldout_season": int(heldout),
                        "model": model_name,
                        "alpha": alpha,
                        **metrics,
                        "rows": len(test),
                    }
                )
                for row, estimate in zip(
                    test.itertuples(index=False),
                    prediction,
                ):
                    prediction_records.append(
                        {
                            "target": target,
                            "heldout_season": int(heldout),
                            "model": model_name,
                            "player_id": row.player_id,
                            "player_name": row.player_name,
                            "off_chances": row.off_chances,
                            "actual": getattr(row, target),
                            "prediction": float(estimate),
                        }
                    )

    folds = pd.DataFrame(fold_records)
    predictions = pd.DataFrame(prediction_records)
    summaries = []
    for (target, model_name), group in predictions.groupby(["target", "model"]):
        metrics = _weighted_metrics(
            group["actual"].to_numpy(),
            group["prediction"].to_numpy(),
            group["off_chances"].to_numpy(),
        )
        summaries.append(
            {
                "target": target,
                "model": model_name,
                **metrics,
                "rows": len(group),
                "winning_folds_vs_mean": int(
                    (
                        folds[
                            (folds["target"].eq(target))
                            & (folds["model"].eq(model_name))
                        ]
                        .set_index("heldout_season")["rmse"]
                        < folds[
                            (folds["target"].eq(target))
                            & (folds["model"].eq("train_weighted_mean"))
                        ]
                        .set_index("heldout_season")["rmse"]
                    ).sum()
                )
                if model_name != "train_weighted_mean"
                else 0,
            }
        )
    return folds, pd.DataFrame(summaries), predictions


def _mapped_pass_stats(
    passes: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    mapped = passes.merge(
        crosswalk[["provider_player_id", "player_id"]],
        on="provider_player_id",
        how="inner",
        validate="many_to_one",
    )
    return (
        mapped.groupby(["season", "player_id"], as_index=False)[
            ["passes", "pass_turnovers", "potential_assists"]
        ]
        .sum()
    )


def _add_value_identities(
    roles: pd.DataFrame,
    diagnostics: pd.DataFrame,
    features: pd.DataFrame,
    passes: pd.DataFrame,
) -> pd.DataFrame:
    season_context = diagnostics.copy()
    season_context["shot_rate_per_chance"] = (
        season_context["valid_shots"]
        / season_context["valid_chance_groups"].clip(lower=1.0)
    )
    actual_means = (
        roles.attrs.get("actual_shot_means")
        if "actual_shot_means" in roles.attrs
        else {}
    )
    season_context["average_points_per_shot"] = season_context["season"].map(
        actual_means
    )
    result = roles.merge(
        season_context[
            ["season", "shot_rate_per_chance", "average_points_per_shot"]
        ],
        on="season",
        how="left",
        validate="many_to_one",
    )
    result = result.merge(
        features[
            [
                "season",
                "player_id",
                "player_name",
                "off_chances",
                "rtss_on_ball_time_pct",
                "rtss_potential_assist_rate100",
            ]
        ],
        on=["season", "player_id"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_feature"),
    )
    result = result.merge(
        passes,
        on=["season", "player_id"],
        how="left",
        validate="many_to_one",
    )
    for column in ("passes", "pass_turnovers", "potential_assists"):
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)

    multiplier = 100.0 * result["shot_rate_per_chance"]
    result["scoring_value_per100_chances"] = (
        multiplier * result["shooter_contribution_per_shot"]
    )
    result["passer_shot_value_per100_chances"] = (
        multiplier * result["passer_contribution_per_shot"]
    )
    result["presence_value_per100_chances"] = (
        multiplier * result["presence_contribution_per_shot"]
    )
    result["shot_parent_value_per100_chances"] = (
        multiplier * result["offensive_parent_per_shot"]
    )
    result["pass_turnover_cost_per100_chances"] = np.where(
        result["off_chances"] > 0,
        -100.0
        * result["pass_turnovers"]
        / result["off_chances"]
        * result["average_points_per_shot"],
        0.0,
    )
    result["net_playmaking_value_per100_chances"] = (
        result["passer_shot_value_per100_chances"]
        + result["pass_turnover_cost_per100_chances"]
    )
    result["creation_parent_value_per100_chances"] = (
        result["shot_parent_value_per100_chances"]
        + result["pass_turnover_cost_per100_chances"]
    )
    result["shot_role_identity_gap"] = (
        result["scoring_value_per100_chances"]
        + result["passer_shot_value_per100_chances"]
        + result["presence_value_per100_chances"]
        - result["shot_parent_value_per100_chances"]
    )
    result["creation_identity_gap"] = (
        result["scoring_value_per100_chances"]
        + result["net_playmaking_value_per100_chances"]
        + result["presence_value_per100_chances"]
        - result["creation_parent_value_per100_chances"]
    )
    if result[["shot_role_identity_gap", "creation_identity_gap"]].abs().max().max() > 1e-8:
        raise AssertionError("published role value identities do not close")
    return result


def run_research(
    rtss_root: Path,
    crosswalk_path: Path,
    features_path: Path,
    output_dir: Path,
    workers: int,
    base_alpha: float,
    min_chances: float,
    rebuild_features: bool,
) -> dict[str, Any]:
    if rebuild_features or not features_path.exists():
        build_features(rtss_root, crosswalk_path, features_path, workers)
    features = pd.read_csv(features_path)
    if "rtss_on_ball_time_pct" not in features:
        features = build_features(rtss_root, crosswalk_path, features_path, workers)
    features["player_id"] = features["player_id"].astype(str)

    shots, provider_passes, diagnostics = _load_rtss(rtss_root, workers)
    crosswalk = pd.read_csv(crosswalk_path)
    crosswalk = crosswalk[crosswalk["status"].isin(VALID_CROSSWALK_STATUSES)].copy()
    crosswalk["player_id"] = (
        pd.to_numeric(crosswalk["nba_id"], errors="coerce")
        .astype("Int64")
        .astype(str)
    )
    crosswalk = crosswalk[crosswalk["player_id"].ne("<NA>")]
    crosswalk = crosswalk.drop_duplicates("provider_player_id")

    role_frames = []
    cv_frames = []
    actual_shot_means: dict[int, float] = {}
    for season in (2023, 2024, 2025):
        season_shots = shots[shots["season"].eq(season)]
        actual_shot_means[season] = float(season_shots["actual_points"].mean())
        _screen_roles, actual_cv = _fit_season_roles(
            shots,
            season,
            "actual_points",
            base_alpha,
            ROLE_ALPHA_GRID,
        )
        cv_frames.append(actual_cv)

    combined_cv = pd.concat(cv_frames, ignore_index=True)
    selected_alpha = _common_role_alpha(combined_cv)
    combined_cv["common_selected"] = combined_cv["role_alpha"].eq(selected_alpha)
    role_frames = []
    for season in (2023, 2024, 2025):
        actual_roles, _ = _fit_season_roles(
            shots,
            season,
            "actual_points",
            base_alpha,
            ROLE_ALPHA_GRID,
            selected_alpha=selected_alpha,
        )
        role_frames.append(actual_roles)
        q_roles, _ = _fit_season_roles(
            shots[shots["zone"].isin(["mid", "three"])],
            season,
            "provider_qsq_points",
            base_alpha,
            ROLE_ALPHA_GRID,
            selected_alpha=selected_alpha,
        )
        role_frames.append(q_roles)

    all_roles = pd.concat(role_frames, ignore_index=True)
    all_roles = all_roles.merge(
        crosswalk[
            [
                "provider_player_id",
                "player_id",
                "player_name",
                "status",
            ]
        ],
        on="provider_player_id",
        how="left",
        validate="many_to_one",
    )
    actual_roles = all_roles[
        all_roles["target"].eq("actual_points") & all_roles["player_id"].notna()
    ].copy()
    actual_roles.attrs["actual_shot_means"] = actual_shot_means

    mapped_passes = _mapped_pass_stats(provider_passes, crosswalk)
    player_values = _add_value_identities(
        actual_roles,
        diagnostics,
        features,
        mapped_passes,
    )
    box = pd.concat(
        [load_box_features(year) for year in (2023, 2024, 2025)],
        ignore_index=True,
    ).rename(columns={"year": "season"})
    box["player_id"] = box["player_id"].astype(str)
    player_values = player_values.merge(
        box[
            [
                "season",
                "player_id",
                "points100",
                "tsa100",
                "rim_assists100",
            ]
        ],
        on=["season", "player_id"],
        how="left",
        validate="many_to_one",
    )
    folds, summary, predictions = _evaluate_compact_priors(
        player_values,
        min_chances=min_chances,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    all_roles.to_csv(output_dir / "role_coefficients.csv", index=False)
    player_values.to_csv(output_dir / "player_season_role_values.csv", index=False)
    combined_cv.to_csv(
        output_dir / "role_alpha_cv.csv",
        index=False,
    )
    folds.to_csv(output_dir / "playmaking_prior_loso_folds.csv", index=False)
    summary.to_csv(output_dir / "playmaking_prior_loso_summary.csv", index=False)
    predictions.to_csv(
        output_dir / "playmaking_prior_loso_predictions.csv",
        index=False,
    )
    diagnostics.to_csv(output_dir / "source_diagnostics.csv", index=False)

    manifest = {
        "status": "research_only_not_live",
        "rtss_root": str(rtss_root),
        "crosswalk": str(crosswalk_path),
        "feature_surface": str(features_path),
        "seasons": [2023, 2024, 2025],
        "valid_shots": int(len(shots)),
        "mapped_player_seasons": int(len(player_values)),
        "base_alpha": float(base_alpha),
        "selected_common_role_alpha": selected_alpha,
        "prior_min_off_chances": float(min_chances),
        "role_definition": {
            "shooter": "RTSS shooterId",
            "passer": "shots.assistOpp and valid distinct shots.passerId",
            "presence": "offensive chance_players other than shooter and passer",
            "parent": "exposure-averaged joint offensive shot-value coefficient",
        },
        "identities": {
            "shot_parent": "scoring + passer_shot + presence",
            "net_playmaking": "passer_shot + pass_turnover_cost",
            "creation_parent": "scoring + net_playmaking + presence",
            "max_shot_gap": float(player_values["shot_role_identity_gap"].abs().max()),
            "max_creation_gap": float(player_values["creation_identity_gap"].abs().max()),
        },
        "provider_qsq": (
            "MID/THREE benchmark only; q values normalized from 0-1 or 0-100; "
            "not used as the primary playmaking target"
        ),
        "validation": (
            "symmetric leave-one-2023/2024/2025-season-out development; "
            "not historical as-of confirmation"
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtss-root", type=Path, default=DEFAULT_RTSS_ROOT)
    parser.add_argument("--crosswalk", type=Path, default=DEFAULT_CROSSWALK)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base-alpha", type=float, default=3000.0)
    parser.add_argument("--min-chances", type=float, default=250.0)
    parser.add_argument("--rebuild-features", action="store_true")
    args = parser.parse_args()
    manifest = run_research(
        args.rtss_root.resolve(),
        args.crosswalk.resolve(),
        args.features.resolve(),
        args.output_dir.resolve(),
        max(1, args.workers),
        float(args.base_alpha),
        float(args.min_chances),
        bool(args.rebuild_features),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
