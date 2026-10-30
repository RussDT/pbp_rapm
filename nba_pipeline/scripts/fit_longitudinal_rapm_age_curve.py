#!/usr/bin/env python3
"""Fit offense and defense aging curves from consecutive player seasons.

The public input is a player-season panel with positive-good ``orapm`` and
``drapm`` columns.  The model works on within-player consecutive-season
changes and includes a calendar-pair nuisance effect.  The otherwise
unidentified constant aging drift is assigned to the age curve by constraining
the possession-weighted mean calendar-pair effect to zero.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer


REQUIRED_COLUMNS = {
    "player_id",
    "season",
    "age",
    "orapm",
    "drapm",
    "off_poss",
    "def_poss",
}


@dataclass(frozen=True)
class LongitudinalCurveFit:
    curve: pd.DataFrame
    pairs: pd.DataFrame
    year_effects: pd.DataFrame
    cv_results: pd.DataFrame
    receipt: dict


def _consecutive_pairs(panel: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(panel.columns)
    if missing:
        raise ValueError(f"Player-season panel missing columns: {sorted(missing)}")

    current = panel.copy()
    current["player_id"] = current["player_id"].astype(str)
    current["season"] = pd.to_numeric(current["season"], errors="coerce")
    numeric = ["age", "orapm", "drapm", "off_poss", "def_poss"]
    for column in numeric:
        current[column] = pd.to_numeric(current[column], errors="coerce")
    current = current.dropna(subset=["player_id", "season", *numeric]).copy()
    current["season"] = current["season"].astype(int)
    current = current.sort_values(["player_id", "season"]).drop_duplicates(
        ["player_id", "season"], keep="last"
    )

    following = current.rename(
        columns={
            "season": "season_to",
            "age": "age_to",
            "orapm": "orapm_to",
            "drapm": "drapm_to",
            "off_poss": "off_poss_to",
            "def_poss": "def_poss_to",
        }
    )
    pairs = current.rename(
        columns={
            "season": "season_from",
            "age": "age_from",
            "orapm": "orapm_from",
            "drapm": "drapm_from",
            "off_poss": "off_poss_from",
            "def_poss": "def_poss_from",
        }
    ).merge(following, on="player_id", how="inner")
    pairs = pairs[pairs["season_to"] == pairs["season_from"] + 1].copy()
    if pairs.empty:
        raise ValueError("No consecutive player-season pairs were found.")

    pairs["transition_age"] = (pairs["age_from"] + pairs["age_to"]) / 2.0
    pairs["orapm_change"] = pairs["orapm_to"] - pairs["orapm_from"]
    pairs["drapm_change"] = pairs["drapm_to"] - pairs["drapm_from"]
    return pairs.reset_index(drop=True)


def _precision_proxy(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.maximum(np.asarray(first, dtype=float), 1.0)
    second = np.maximum(np.asarray(second, dtype=float), 1.0)
    precision = 1.0 / ((1.0 / first) + (1.0 / second))
    cap = float(np.quantile(precision, 0.95))
    if cap > 0:
        precision = np.minimum(precision, cap)
    mean = float(precision.mean())
    return precision / mean if mean > 0 else np.ones_like(precision)


def _weighted_group_means(values: np.ndarray, groups: np.ndarray, weights: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values, dtype=float)
    for group in np.unique(groups):
        mask = groups == group
        result[mask] = np.average(values[mask], axis=0, weights=weights[mask])
    return result


def _fit_side(
    pairs: pd.DataFrame,
    side: str,
    *,
    age_min: int,
    age_max: int,
    reference_age: int,
    smoothing_alpha: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    change_column = f"{side}rapm_change"
    poss_column = "off_poss" if side == "o" else "def_poss"
    weights = _precision_proxy(
        pairs[f"{poss_column}_from"].to_numpy(),
        pairs[f"{poss_column}_to"].to_numpy(),
    )
    ages = pairs["transition_age"].to_numpy(dtype=float)
    groups = pairs["season_to"].to_numpy(dtype=int)
    target = pairs[change_column].to_numpy(dtype=float)

    n_unique_ages = max(4, int(pd.Series(ages).round(1).nunique()))
    n_knots = min(7, n_unique_ages)
    transformer = SplineTransformer(
        n_knots=n_knots,
        degree=3,
        include_bias=True,
        extrapolation="linear",
    )
    basis = transformer.fit_transform(ages.reshape(-1, 1))
    target_within = target - _weighted_group_means(target, groups, weights)
    basis_within = basis - _weighted_group_means(basis, groups, weights)

    sqrt_weights = np.sqrt(weights)
    weighted_basis = basis_within * sqrt_weights[:, None]
    weighted_target = target_within * sqrt_weights
    penalty = float(smoothing_alpha) * np.eye(weighted_basis.shape[1])
    beta = np.linalg.solve(
        weighted_basis.T @ weighted_basis + penalty,
        weighted_basis.T @ weighted_target,
    )

    shape_increment = basis @ beta
    residual = target - shape_increment
    raw_year_effect = _weighted_group_means(residual, groups, weights)
    constant_increment = float(np.average(raw_year_effect, weights=weights))
    year_effect = raw_year_effect - constant_increment

    year_rows = []
    for season_to in sorted(np.unique(groups)):
        mask = groups == season_to
        year_rows.append(
            {
                "side": side,
                "season_from": int(season_to - 1),
                "season_to": int(season_to),
                "effect": float(np.average(year_effect[mask], weights=weights[mask])),
                "pairs": int(mask.sum()),
                "weight": float(weights[mask].sum()),
            }
        )

    transition_grid = np.arange(age_min, age_max, dtype=float) + 0.5
    grid_basis = transformer.transform(transition_grid.reshape(-1, 1))
    increments = grid_basis @ beta + constant_increment
    effects = {int(reference_age): 0.0}
    for age in range(reference_age + 1, age_max + 1):
        effects[age] = effects[age - 1] + float(increments[(age - 1) - age_min])
    for age in range(reference_age - 1, age_min - 1, -1):
        effects[age] = effects[age + 1] - float(increments[age - age_min])
    return np.array([effects[age] for age in range(age_min, age_max + 1)]), pd.DataFrame(year_rows)


def _predict_pair_changes(
    effects: np.ndarray,
    year_effects: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    side: str,
    age_min: int,
    age_max: int,
) -> np.ndarray:
    transition_grid = np.arange(age_min, age_max, dtype=float) + 0.5
    increments = np.diff(effects)
    age_prediction = np.interp(
        pairs["transition_age"].to_numpy(dtype=float),
        transition_grid,
        increments,
        left=float(increments[0]),
        right=float(increments[-1]),
    )
    side_years = year_effects[year_effects["side"] == side]
    year_lookup = dict(zip(side_years["season_to"], side_years["effect"]))
    calendar_prediction = pairs["season_to"].map(year_lookup).fillna(0.0).to_numpy(dtype=float)
    return age_prediction + calendar_prediction


def _select_smoothing_alpha(
    pairs: pd.DataFrame,
    side: str,
    grid: list[float],
    *,
    cv_folds: int,
    random_seed: int,
    age_min: int,
    age_max: int,
    reference_age: int,
) -> tuple[float, list[dict]]:
    players = pairs["player_id"].drop_duplicates().to_numpy()
    if len(players) < 2:
        raise ValueError("Grouped-player CV requires at least two players.")
    folds = min(int(cv_folds), len(players))
    if folds < 2:
        raise ValueError("cv_folds must be at least 2.")
    rng = np.random.default_rng(random_seed)
    shuffled = players.copy()
    rng.shuffle(shuffled)
    heldout_groups = np.array_split(shuffled, folds)
    poss_column = "off_poss" if side == "o" else "def_poss"
    target_column = f"{side}rapm_change"

    alpha_losses: dict[float, list[float]] = {float(alpha): [] for alpha in grid}
    for heldout_players in heldout_groups:
        heldout = set(heldout_players.tolist())
        train = pairs[~pairs["player_id"].isin(heldout)].copy()
        test = pairs[pairs["player_id"].isin(heldout)].copy()
        test_weights = _precision_proxy(
            test[f"{poss_column}_from"].to_numpy(),
            test[f"{poss_column}_to"].to_numpy(),
        )
        actual = test[target_column].to_numpy(dtype=float)
        for alpha in alpha_losses:
            effects, year_effects = _fit_side(
                train,
                side,
                age_min=age_min,
                age_max=age_max,
                reference_age=reference_age,
                smoothing_alpha=alpha,
            )
            predicted = _predict_pair_changes(
                effects,
                year_effects,
                test,
                side=side,
                age_min=age_min,
                age_max=age_max,
            )
            alpha_losses[alpha].append(float(np.average((actual - predicted) ** 2, weights=test_weights)))

    rows = []
    for alpha in sorted(alpha_losses):
        losses = np.asarray(alpha_losses[alpha], dtype=float)
        rows.append(
            {
                "side": side,
                "alpha": float(alpha),
                "mean_mse": float(losses.mean()),
                "se_mse": float(losses.std(ddof=1) / np.sqrt(len(losses))) if len(losses) > 1 else 0.0,
                "folds": int(len(losses)),
            }
        )
    best = min(rows, key=lambda row: row["mean_mse"])
    threshold = best["mean_mse"] + best["se_mse"]
    selected = float(best["alpha"])
    for row in rows:
        row["one_se_threshold"] = float(threshold)
        row["selected"] = bool(row["alpha"] == selected)
    return float(selected), rows


def fit_longitudinal_curves(
    panel: pd.DataFrame,
    *,
    age_min: int = 18,
    age_max: int = 40,
    reference_age: int = 27,
    smoothing_alpha: float | None = 10.0,
    smoothing_grid: list[float] | None = None,
    cv_folds: int = 5,
    bootstrap_reps: int = 0,
    random_seed: int = 20260826,
) -> LongitudinalCurveFit:
    """Fit positive-good offensive and defensive age curves."""
    if not age_min < reference_age < age_max:
        raise ValueError("reference_age must lie strictly inside the age range")
    if smoothing_alpha is not None and smoothing_alpha <= 0:
        raise ValueError("smoothing_alpha must be positive")
    if bootstrap_reps < 0:
        raise ValueError("bootstrap_reps cannot be negative")

    pairs = _consecutive_pairs(panel)
    cv_rows: list[dict] = []
    if smoothing_alpha is None:
        grid = smoothing_grid or [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        if not grid or any(float(alpha) <= 0 for alpha in grid):
            raise ValueError("smoothing_grid must contain positive values")
        off_alpha, off_rows = _select_smoothing_alpha(
            pairs,
            "o",
            [float(alpha) for alpha in grid],
            cv_folds=cv_folds,
            random_seed=random_seed,
            age_min=age_min,
            age_max=age_max,
            reference_age=reference_age,
        )
        def_alpha, def_rows = _select_smoothing_alpha(
            pairs,
            "d",
            [float(alpha) for alpha in grid],
            cv_folds=cv_folds,
            random_seed=random_seed + 1,
            age_min=age_min,
            age_max=age_max,
            reference_age=reference_age,
        )
        cv_rows = off_rows + def_rows
        smoothing_selection = "grouped_player_minimum_mse"
    else:
        off_alpha = def_alpha = float(smoothing_alpha)
        smoothing_selection = "fixed_shared"
    off_effect, off_year = _fit_side(
        pairs,
        "o",
        age_min=age_min,
        age_max=age_max,
        reference_age=reference_age,
        smoothing_alpha=off_alpha,
    )
    def_effect, def_year = _fit_side(
        pairs,
        "d",
        age_min=age_min,
        age_max=age_max,
        reference_age=reference_age,
        smoothing_alpha=def_alpha,
    )
    curve = pd.DataFrame(
        {
            "age": np.arange(age_min, age_max + 1, dtype=int),
            "orapm_age_effect": off_effect,
            "drapm_age_effect": def_effect,
        }
    )
    support_age = np.floor(pairs["transition_age"].to_numpy(dtype=float) + 0.5).astype(int)
    support = (
        pairs.assign(_support_age=support_age)
        .groupby("_support_age")
        .agg(
            transition_pairs=("player_id", "size"),
            transition_players=("player_id", "nunique"),
        )
    )
    curve["transition_pairs"] = curve["age"].map(support["transition_pairs"]).fillna(0).astype(int)
    curve["transition_players"] = curve["age"].map(support["transition_players"]).fillna(0).astype(int)
    curve["supported_age"] = (
        (curve["transition_pairs"] >= 30) & (curve["transition_players"] >= 20)
    )
    curve.loc[curve["age"] == reference_age, ["orapm_age_effect", "drapm_age_effect"]] = 0.0
    if bootstrap_reps:
        rng = np.random.default_rng(random_seed)
        player_ids = pairs["player_id"].drop_duplicates().to_numpy()
        player_row_indices = {
            player_id: np.flatnonzero(pairs["player_id"].to_numpy() == player_id)
            for player_id in player_ids
        }
        off_draws = []
        def_draws = []
        for _ in range(bootstrap_reps):
            sampled_ids = rng.choice(player_ids, size=len(player_ids), replace=True)
            sampled_indices = np.concatenate([player_row_indices[player_id] for player_id in sampled_ids])
            sampled_pairs = pairs.iloc[sampled_indices].reset_index(drop=True)
            sampled_off, _ = _fit_side(
                sampled_pairs,
                "o",
                age_min=age_min,
                age_max=age_max,
                reference_age=reference_age,
                smoothing_alpha=off_alpha,
            )
            sampled_def, _ = _fit_side(
                sampled_pairs,
                "d",
                age_min=age_min,
                age_max=age_max,
                reference_age=reference_age,
                smoothing_alpha=def_alpha,
            )
            off_draws.append(sampled_off)
            def_draws.append(sampled_def)

        for side, draws, point in [
            ("orapm", np.asarray(off_draws), off_effect),
            ("drapm", np.asarray(def_draws), def_effect),
        ]:
            low = np.quantile(draws, 0.025, axis=0)
            high = np.quantile(draws, 0.975, axis=0)
            curve[f"{side}_ci_low"] = np.minimum(low, point)
            curve[f"{side}_ci_high"] = np.maximum(high, point)
            curve.loc[
                curve["age"] == reference_age,
                [f"{side}_ci_low", f"{side}_ci_high"],
            ] = 0.0
    receipt = {
        "method": "within_player_consecutive_season_changes",
        "identification": "calendar_pair_effects_weighted_mean_zero",
        "reference_age": int(reference_age),
        "age_min": int(age_min),
        "age_max": int(age_max),
        "player_seasons": int(len(panel)),
        "consecutive_pairs": int(len(pairs)),
        "players": int(pairs["player_id"].nunique()),
        "smoothing_selection": smoothing_selection,
        "off_smoothing_alpha": float(off_alpha),
        "def_smoothing_alpha": float(def_alpha),
        "cv_folds": int(cv_folds) if smoothing_alpha is None else 0,
        "bootstrap_reps": int(bootstrap_reps),
        "bootstrap_cluster": "player_id",
        "random_seed": int(random_seed),
        "supported_age_min_pairs": 30,
        "supported_age_min_players": 20,
    }
    return LongitudinalCurveFit(
        curve=curve,
        pairs=pairs,
        year_effects=pd.concat([off_year, def_year], ignore_index=True),
        cv_results=pd.DataFrame(cv_rows),
        receipt=receipt,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit longitudinal O-RAPM/D-RAPM age curves from a player-season panel."
    )
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--age-min", type=int, default=18)
    parser.add_argument("--age-max", type=int, default=40)
    parser.add_argument("--reference-age", type=int, default=27)
    parser.add_argument("--smoothing-alpha", type=float, default=None)
    parser.add_argument(
        "--smoothing-grid",
        default="0.01,0.1,1,10,100,1000",
        help="Comma-separated alpha grid used when --smoothing-alpha is omitted.",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--bootstrap-reps", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=20260826)
    return parser.parse_args(argv)


def write_fit_artifacts(fitted: LongitudinalCurveFit, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fitted.curve.to_csv(output_dir / "age_curve.csv", index=False)
    fitted.pairs.to_csv(output_dir / "consecutive_player_season_pairs.csv", index=False)
    fitted.year_effects.to_csv(output_dir / "calendar_pair_effects.csv", index=False)
    if not fitted.cv_results.empty:
        fitted.cv_results.to_csv(output_dir / "smoothing_cv.csv", index=False)
    with (output_dir / "fit_receipt.json").open("w", encoding="utf-8") as handle:
        json.dump(fitted.receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    panel = pd.read_csv(args.panel)
    fitted = fit_longitudinal_curves(
        panel,
        age_min=args.age_min,
        age_max=args.age_max,
        reference_age=args.reference_age,
        smoothing_alpha=args.smoothing_alpha,
        smoothing_grid=[float(value) for value in args.smoothing_grid.split(",") if value.strip()],
        cv_folds=args.cv_folds,
        bootstrap_reps=args.bootstrap_reps,
        random_seed=args.random_seed,
    )
    write_fit_artifacts(fitted, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
