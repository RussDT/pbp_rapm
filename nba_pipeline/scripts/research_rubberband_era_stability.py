#!/usr/bin/env python3
"""Test whether the smooth RAPM rubberband surface is constant across eras.

The first 1997-2001 surface is frozen. For each 2002-2026 test season, player
coefficients are fit on the preceding five seasons under three contexts:
no rubberband, the frozen early surface, and a jointly re-estimated rolling
surface. All predictions use identical future possessions.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import rapm  # noqa: E402
from research_rubberband_adjustment import (  # noqa: E402
    append_rubberband_design,
    build_base_design_pair,
    calibration_rmse,
    paired_cluster_bootstrap,
    parse_int_list,
    prepare_season,
)
from rubberband_model import build_rubberband_basis  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-years", default=",".join(str(year) for year in range(2002, 2027)))
    parser.add_argument("--train-window", type=int, default=5)
    parser.add_argument("--season-type", choices=["RS", "ALL"], default="ALL")
    parser.add_argument("--off-alpha", type=float, default=2000.0)
    parser.add_argument("--def-alpha", type=float, default=4000.0)
    parser.add_argument("--rubberband-alpha", type=float, default=100.0)
    parser.add_argument("--bootstrap-draws", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PIPELINE_ROOT / "results" / "rubberband_era_stability",
    )
    return parser.parse_args()


def surface_states() -> pd.DataFrame:
    rows = []
    for margin in range(-30, 31, 2):
        for seconds in (2520, 1800, 1080, 600, 360, 120):
            rows.append(
                {
                    "score_diff": float(margin),
                    "period": min(4, max(1, 4 - int(max(seconds - 1, 0) // 720))),
                    "seconds_remaining_game": float(seconds),
                    "regulation_progress": 1.0 - seconds / 2880.0,
                    "is_overtime": 0,
                    "overtime_progress": 0.0,
                }
            )
    return pd.DataFrame(rows)


def representative_states() -> pd.DataFrame:
    rows = []
    for label, seconds in (("q1_mid", 2520), ("q2_mid", 1800), ("q3_mid", 1080),
                           ("q4_6m", 360), ("q4_2m", 120)):
        for margin in (-20, -10, -5, 5, 10, 20):
            rows.append(
                {
                    "state": label,
                    "score_diff": float(margin),
                    "period": min(4, max(1, 4 - int(max(seconds - 1, 0) // 720))),
                    "seconds_remaining_game": float(seconds),
                    "regulation_progress": 1.0 - seconds / 2880.0,
                    "is_overtime": 0,
                    "overtime_progress": 0.0,
                }
            )
    return pd.DataFrame(rows)


def solve_players(
    matrix,
    target,
    mapping,
    players,
    off_alpha,
    def_alpha,
):
    return rapm.joint_generalized_rapm(
        matrix,
        target,
        mapping,
        players,
        alpha_offense=off_alpha,
        alpha_defense=def_alpha,
        rubberband=False,
    )


def solve_rolling(
    matrix,
    target,
    mapping,
    players,
    off_alpha,
    def_alpha,
    rubberband_alpha,
):
    return rapm.joint_generalized_rapm(
        matrix,
        target,
        mapping,
        players,
        alpha_offense=off_alpha,
        alpha_defense=def_alpha,
        rubberband=True,
        alpha_rb=rubberband_alpha,
    )


def score_prediction(test, target, prediction, label):
    residual = target - prediction
    squared_error = np.square(residual)
    absolute_error = np.abs(residual)
    metric = {
        "label": label,
        "rows": int(len(test)),
        "games": int(test["gameid"].nunique()),
        "mse": float(squared_error.mean()),
        "mae": float(absolute_error.mean()),
        "calibration_rmse": calibration_rmse(test, residual),
    }
    game_loss = pd.DataFrame(
        {
            "gameid": test["gameid"].astype(str).to_numpy(),
            "squared_error": squared_error,
            "absolute_error": absolute_error,
        }
    ).groupby("gameid", as_index=False).agg(
        squared_error_sum=("squared_error", "sum"),
        absolute_error_sum=("absolute_error", "sum"),
        rows=("squared_error", "size"),
    )
    game_loss["label"] = label
    return metric, game_loss


def era_for_year(year: int) -> str:
    if year <= 2009:
        return "2002-2009"
    if year <= 2017:
        return "2010-2017"
    return "2018-2026"


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.8g}"
        )
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    test_years = parse_int_list(args.test_years)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[int, pd.DataFrame] = {}
    metrics = []
    game_losses = []
    surface_rows = []
    representative_rows = []
    early_coefficients: dict[str, float] | None = None
    early_surface: np.ndarray | None = None
    previous_surface: np.ndarray | None = None
    dense_states = surface_states()
    dense_basis = build_rubberband_basis(dense_states, "smooth")
    readable_states = representative_states()
    readable_basis = build_rubberband_basis(readable_states, "smooth")

    def get(year: int) -> pd.DataFrame:
        if year not in cache:
            logging.info("Preparing season %s", year)
            cache[year] = prepare_season(year, args.season_type)
        return cache[year]

    for test_year in test_years:
        train_years = list(range(test_year - args.train_window, test_year))
        train = pd.concat([get(year) for year in train_years], ignore_index=True)
        test = get(test_year)
        logging.info(
            "Fold %s: train=%s-%s (%s rows), test=%s (%s rows)",
            test_year,
            train_years[0],
            train_years[-1],
            len(train),
            test_year,
            len(test),
        )
        base = build_base_design_pair(train, test)
        smooth = append_rubberband_design(base, train, test, "smooth")
        x_train, x_test, y_train, y_test, players, smooth_mapping = smooth
        base_x_train, base_x_test, _, _, _, base_mapping = base
        train_mean = float(y_train.mean())

        beta_none = solve_players(
            base_x_train,
            y_train - train_mean,
            base_mapping,
            players,
            args.off_alpha,
            args.def_alpha,
        )
        pred_none = np.asarray(base_x_test @ beta_none).ravel() + train_mean

        beta_rolling = solve_rolling(
            x_train,
            y_train - train_mean,
            smooth_mapping,
            players,
            args.off_alpha,
            args.def_alpha,
            args.rubberband_alpha,
        )
        pred_rolling = np.asarray(x_test @ beta_rolling).ravel() + train_mean
        rolling_coefficients = {
            name: float(beta_rolling[index])
            for name, index in smooth_mapping.items()
            if name.startswith("rb_smooth_")
        }
        if early_coefficients is None:
            early_coefficients = dict(rolling_coefficients)

        rolling_vector = np.array(
            [rolling_coefficients[name] for name in dense_basis.feature_names],
            dtype=np.float64,
        )
        early_vector = np.array(
            [early_coefficients[name] for name in dense_basis.feature_names],
            dtype=np.float64,
        )
        rolling_surface = np.asarray(dense_basis.matrix @ rolling_vector).ravel() * 100.0
        if early_surface is None:
            early_surface = np.asarray(dense_basis.matrix @ early_vector).ravel() * 100.0

        rb_indices = np.array(
            [smooth_mapping[name] for name in dense_basis.feature_names],
            dtype=int,
        )
        fixed_train = np.asarray(x_train[:, rb_indices] @ early_vector).ravel()
        fixed_test = np.asarray(x_test[:, rb_indices] @ early_vector).ravel()
        fixed_center = float(fixed_train.mean())
        beta_fixed_players = solve_players(
            base_x_train,
            y_train - train_mean - (fixed_train - fixed_center),
            base_mapping,
            players,
            args.off_alpha,
            args.def_alpha,
        )
        pred_fixed = (
            np.asarray(base_x_test @ beta_fixed_players).ravel()
            + train_mean
            + fixed_test
            - fixed_center
        )

        for label, prediction in (
            ("none", pred_none),
            ("fixed_1997_2001", pred_fixed),
            ("rolling_5y", pred_rolling),
        ):
            metric, losses = score_prediction(test, y_test, prediction, label)
            metric.update(
                {
                    "test_year": test_year,
                    "train_start": train_years[0],
                    "train_end": train_years[-1],
                    "era": era_for_year(test_year),
                }
            )
            metrics.append(metric)
            losses["test_year"] = test_year
            losses["era"] = era_for_year(test_year)
            game_losses.append(losses)

        correlation = float(np.corrcoef(early_surface, rolling_surface)[0, 1])
        surface_rows.append(
            {
                "test_year": test_year,
                "train_start": train_years[0],
                "train_end": train_years[-1],
                "era": era_for_year(test_year),
                "surface_correlation_vs_1997_2001": correlation,
                "surface_rmse_vs_1997_2001_per_100": float(
                    np.sqrt(np.mean(np.square(rolling_surface - early_surface)))
                ),
                "surface_rmse_vs_previous_window_per_100": (
                    np.nan
                    if previous_surface is None
                    else float(np.sqrt(np.mean(np.square(rolling_surface - previous_surface))))
                ),
            }
        )
        previous_surface = rolling_surface

        readable_vector = np.array(
            [rolling_coefficients[name] for name in readable_basis.feature_names],
            dtype=np.float64,
        )
        readable_effects = np.asarray(readable_basis.matrix @ readable_vector).ravel() * 100.0
        for state, effect in zip(readable_states.itertuples(index=False), readable_effects):
            representative_rows.append(
                {
                    "test_year": test_year,
                    "train_start": train_years[0],
                    "train_end": train_years[-1],
                    "era": era_for_year(test_year),
                    "state": state.state,
                    "score_margin": state.score_diff,
                    "effect_per_100": float(effect),
                }
            )

        oldest_needed = test_year - args.train_window + 1
        cache = {year: frame for year, frame in cache.items() if year >= oldest_needed}

    metrics_df = pd.DataFrame(metrics)
    losses_df = pd.concat(game_losses, ignore_index=True)
    surface_df = pd.DataFrame(surface_rows)
    representative_df = pd.DataFrame(representative_rows)

    summary = metrics_df.groupby(["era", "label"], as_index=False).agg(
        folds=("test_year", "nunique"),
        mean_mse=("mse", "mean"),
        mean_mae=("mae", "mean"),
        mean_calibration_rmse=("calibration_rmse", "mean"),
    ).sort_values(["era", "mean_mse"], kind="stable")

    comparisons = []
    for era_index, era in enumerate(["2002-2009", "2010-2017", "2018-2026", "all"]):
        era_losses = losses_df if era == "all" else losses_df[losses_df["era"] == era]
        for pair_index, (challenger, reference) in enumerate(
            (("rolling_5y", "none"), ("fixed_1997_2001", "none"),
             ("rolling_5y", "fixed_1997_2001"))
        ):
            result = paired_cluster_bootstrap(
                era_losses,
                challenger,
                reference,
                args.bootstrap_draws,
                args.seed + era_index * 10 + pair_index,
            )
            result["era"] = era
            comparisons.append(result)
    comparisons_df = pd.DataFrame(comparisons)

    era_effects = representative_df.groupby(
        ["era", "state", "score_margin"], as_index=False
    )["effect_per_100"].mean()
    selected_effects = era_effects[
        era_effects["state"].isin(["q1_mid", "q4_6m", "q4_2m"])
        & era_effects["score_margin"].isin([-20.0, -10.0, 10.0, 20.0])
    ]

    metrics_df.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    losses_df.to_csv(args.output_dir / "game_losses.csv", index=False)
    surface_df.to_csv(args.output_dir / "surface_drift.csv", index=False)
    representative_df.to_csv(args.output_dir / "representative_effects.csv", index=False)
    comparisons_df.to_csv(args.output_dir / "paired_comparisons.csv", index=False)

    final_drift = surface_df.iloc[-1].to_dict()
    report = [
        "# Rubberband Era Stability",
        "",
        "## Design",
        "",
        "The 1997-2001 smooth alpha-100 surface is frozen and carried forward unchanged. Each 2002-2026 fold refits player coefficients on the trailing five seasons under no rubberband, the fixed early surface, and a jointly re-estimated rolling five-season surface.",
        "",
        "Historical score state is reconstructed from sparse scoreboard snapshots and venue orientation is propagated through lineup continuity before this test.",
        "",
        "## Equal-fold results by test era",
        "",
        markdown_table(summary),
        "",
        "## Paired game-clustered comparisons",
        "",
        markdown_table(comparisons_df),
        "",
        "## Mean representative rolling effects by era",
        "",
        markdown_table(selected_effects),
        "",
        "## Surface drift",
        "",
        markdown_table(surface_df),
        "",
        "Latest-window drift versus 1997-2001:",
        "",
        "```json",
        json.dumps(final_drift, indent=2, sort_keys=True),
        "```",
        "",
        "## Interpretation limit",
        "",
        "This is a historical next-season possession-fit diagnostic. It tests whether an early nuisance surface transports unchanged, not whether a complete RAPM/VPM game forecast beats an external metric. All test seasons are now spent evidence.",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(report) + "\n")
    logging.info("Wrote era-stability artifacts to %s", args.output_dir)


if __name__ == "__main__":
    main()
