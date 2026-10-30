#!/usr/bin/env python3
"""Forward-test a global home-court nuisance term inside pure RAPM.

The experiment uses rolling five-season training windows and next-season tests.
It compares plain RAPM, a centered home-offense term, the selected smooth
rubberband surface, and both adjustments together on identical possessions.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack

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


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass(frozen=True)
class ContextSpec:
    label: str
    rubberband: bool
    home: bool


SPECS = (
    ContextSpec("none", rubberband=False, home=False),
    ContextSpec("home", rubberband=False, home=True),
    ContextSpec("smooth", rubberband=True, home=False),
    ContextSpec("smooth_home", rubberband=True, home=True),
)
MIN_VALID_HOME_POSSESSION_SHARE = 0.40
MAX_VALID_HOME_POSSESSION_SHARE = 0.60
MIN_RESOLVED_HOME_ORIENTATION_SHARE = 0.98


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-years", default="2023")
    parser.add_argument("--stress-years", default="2024,2025,2026")
    parser.add_argument("--train-window", type=int, default=5)
    parser.add_argument("--season-type", choices=["RS", "ALL"], default="ALL")
    parser.add_argument("--off-alpha", type=float, default=2000.0)
    parser.add_argument("--def-alpha", type=float, default=4000.0)
    parser.add_argument("--rubberband-alpha", type=float, default=100.0)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PIPELINE_ROOT / "results" / "homecourt_research",
    )
    return parser.parse_args()


def centered_home_columns(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Code home offense around its training-sample share.

    The fitted coefficient is the home-minus-away possession scoring contrast.
    Centering prevents the unpenalized context column from acting as an
    intercept against the penalized player columns.
    """

    train_home = pd.to_numeric(train["home_poss"], errors="raise").to_numpy(dtype=float)
    test_home = pd.to_numeric(test["home_poss"], errors="raise").to_numpy(dtype=float)
    train_resolved = pd.to_numeric(
        train["home_orientation_resolved"], errors="raise"
    ).to_numpy(dtype=bool)
    test_resolved = pd.to_numeric(
        test["home_orientation_resolved"], errors="raise"
    ).to_numpy(dtype=bool)
    if not train_resolved.any():
        raise ValueError("Training sample has no resolved venue orientations.")
    share = float(train_home[train_resolved].mean())
    train_column = np.where(train_resolved, train_home - share, 0.0)
    test_column = np.where(test_resolved, test_home - share, 0.0)
    return train_column, test_column, share


def validate_home_label_balance(frame: pd.DataFrame, year: int) -> float:
    """Reject seasons whose recovered venue orientation is incomplete or imbalanced."""

    resolved = pd.to_numeric(
        frame["home_orientation_resolved"], errors="raise"
    ).to_numpy(dtype=bool)
    resolved_share = float(resolved.mean())
    if resolved_share < MIN_RESOLVED_HOME_ORIENTATION_SHARE:
        raise ValueError(
            f"Season {year} resolves venue orientation on only {resolved_share:.4f} "
            "of possessions; home-court research requires at least "
            f"{MIN_RESOLVED_HOME_ORIENTATION_SHARE:.2f}."
        )
    share = float(
        pd.to_numeric(frame.loc[resolved, "home_poss"], errors="raise").mean()
    )
    if not MIN_VALID_HOME_POSSESSION_SHARE <= share <= MAX_VALID_HOME_POSSESSION_SHARE:
        raise ValueError(
            f"Season {year} has home_poss share {share:.4f}; venue orientation is not "
            "trustworthy for home-court research."
        )
    return share


def add_home_design(
    design,
    train: pd.DataFrame,
    test: pd.DataFrame,
):
    x_train, x_test, y_train, y_test, players, mapping = design
    train_home, test_home, share = centered_home_columns(train, test)
    context_mapping = dict(mapping)
    context_mapping["context_home_offense"] = x_train.shape[1]
    return (
        hstack([x_train, train_home[:, None]], format="csr"),
        hstack([x_test, test_home[:, None]], format="csr"),
        y_train,
        y_test,
        players,
        context_mapping,
        share,
    )


def solve_and_score(
    train: pd.DataFrame,
    test: pd.DataFrame,
    design,
    spec: ContextSpec,
    off_alpha: float,
    def_alpha: float,
    rubberband_alpha: float,
) -> tuple[dict[str, float | int | str], pd.DataFrame, np.ndarray, dict[str, int]]:
    if spec.home:
        x_train, x_test, y_train, y_test, players, mapping, home_share = add_home_design(
            design,
            train,
            test,
        )
    else:
        x_train, x_test, y_train, y_test, players, mapping = design
        home_share = float(train["home_poss"].mean())

    train_mean = float(y_train.mean())
    beta = rapm.joint_generalized_rapm(
        x_train,
        y_train - train_mean,
        mapping,
        players,
        alpha_offense=off_alpha,
        alpha_defense=def_alpha,
        rubberband=spec.rubberband,
        alpha_rb=rubberband_alpha if spec.rubberband else None,
    )
    prediction = np.asarray(x_test @ beta).ravel() + train_mean
    residual = y_test - prediction
    squared_error = np.square(residual)
    absolute_error = np.abs(residual)
    resolved_mask = test["home_orientation_resolved"].to_numpy(dtype=bool)
    home_mask = test["home_poss"].to_numpy(dtype=bool) & resolved_mask
    away_mask = ~test["home_poss"].to_numpy(dtype=bool) & resolved_mask
    home_residual = float(residual[home_mask].mean())
    away_residual = float(residual[away_mask].mean())
    row = {
        "label": spec.label,
        "rubberband": spec.rubberband,
        "home": spec.home,
        "rows": int(len(test)),
        "games": int(test["gameid"].nunique()),
        "mse": float(squared_error.mean()),
        "mae": float(absolute_error.mean()),
        "state_calibration_rmse": calibration_rmse(test, residual),
        "train_home_share": home_share,
        "home_effect_per_100": (
            float(beta[mapping["context_home_offense"]] * 100.0) if spec.home else np.nan
        ),
        "test_home_residual_per_100": home_residual * 100.0,
        "test_away_residual_per_100": away_residual * 100.0,
        "test_home_away_residual_gap_per_100": (home_residual - away_residual) * 100.0,
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
    game_loss["label"] = spec.label
    return row, game_loss, beta, mapping


def evaluate_folds(
    test_years: list[int],
    train_window: int,
    season_type: str,
    off_alpha: float,
    def_alpha: float,
    rubberband_alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, float | int | str]] = []
    losses: list[pd.DataFrame] = []
    cache: dict[int, pd.DataFrame] = {}

    def get(year: int) -> pd.DataFrame:
        if year not in cache:
            logging.info("Preparing season %s", year)
            cache[year] = prepare_season(year, season_type)
            validate_home_label_balance(cache[year], year)
        return cache[year]

    for test_year in test_years:
        train_years = list(range(test_year - train_window, test_year))
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
        for spec in SPECS:
            design = smooth if spec.rubberband else base
            logging.info("Scoring %s on fold %s", spec.label, test_year)
            row, game_loss, _, _ = solve_and_score(
                train,
                test,
                design,
                spec,
                off_alpha,
                def_alpha,
                rubberband_alpha,
            )
            row["test_year"] = test_year
            row["train_start"] = train_years[0]
            row["train_end"] = train_years[-1]
            metrics.append(row)
            game_loss["test_year"] = test_year
            losses.append(game_loss)
        oldest_needed = test_year - train_window + 1
        cache = {year: frame for year, frame in cache.items() if year >= oldest_needed}
    return pd.DataFrame(metrics), pd.concat(losses, ignore_index=True)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.groupby("label", as_index=False).agg(
        folds=("test_year", "nunique"),
        mean_mse=("mse", "mean"),
        mean_mae=("mae", "mean"),
        mean_state_calibration_rmse=("state_calibration_rmse", "mean"),
        mean_abs_home_away_residual_gap_per_100=(
            "test_home_away_residual_gap_per_100",
            lambda values: float(np.mean(np.abs(values))),
        ),
    ).sort_values("mean_mse", kind="stable")


def player_stability(
    final_train: pd.DataFrame,
    final_test: pd.DataFrame,
    off_alpha: float,
    def_alpha: float,
    rubberband_alpha: float,
) -> dict[str, float | int]:
    base = build_base_design_pair(final_train, final_test)
    smooth = append_rubberband_design(base, final_train, final_test, "smooth")
    fits: dict[str, dict[str, float]] = {}
    for spec in (SPECS[2], SPECS[3]):
        _, _, beta, mapping = solve_and_score(
            final_train,
            final_test,
            smooth,
            spec,
            off_alpha,
            def_alpha,
            rubberband_alpha,
        )
        players = sorted({name[:-4] for name in mapping if name.endswith("_off")})
        fits[spec.label] = {
            player: 100.0 * (
                beta[mapping[f"{player}_off"]] - beta[mapping[f"{player}_def"]]
            )
            for player in players
        }
    common = sorted(set(fits["smooth"]) & set(fits["smooth_home"]))
    baseline = np.array([fits["smooth"][player] for player in common])
    home = np.array([fits["smooth_home"][player] for player in common])
    delta = home - baseline
    return {
        "players": len(common),
        "correlation": float(np.corrcoef(baseline, home)[0, 1]),
        "median_abs_shift_per_100": float(np.median(np.abs(delta))),
        "p90_abs_shift_per_100": float(np.quantile(np.abs(delta), 0.90)),
        "max_abs_shift_per_100": float(np.max(np.abs(delta))),
    }


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
    development_years = parse_int_list(args.development_years)
    stress_years = parse_int_list(args.stress_years)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dev_metrics, dev_losses = evaluate_folds(
        development_years,
        args.train_window,
        args.season_type,
        args.off_alpha,
        args.def_alpha,
        args.rubberband_alpha,
    )
    stress_metrics, stress_losses = evaluate_folds(
        stress_years,
        args.train_window,
        args.season_type,
        args.off_alpha,
        args.def_alpha,
        args.rubberband_alpha,
    )
    dev_summary = summarize(dev_metrics)
    stress_summary = summarize(stress_metrics)

    comparisons = []
    comparison_pairs = (
        ("home", "none"),
        ("smooth", "none"),
        ("smooth_home", "smooth"),
        ("smooth_home", "home"),
    )
    for offset, (challenger, reference) in enumerate(comparison_pairs):
        comparisons.append(
            paired_cluster_bootstrap(
                stress_losses,
                challenger,
                reference,
                args.bootstrap_draws,
                args.seed + offset,
            )
        )

    final_year = max(stress_years)
    final_train = pd.concat(
        [prepare_season(year, args.season_type) for year in range(final_year - args.train_window, final_year)],
        ignore_index=True,
    )
    final_test = prepare_season(final_year, args.season_type)
    stability = player_stability(
        final_train,
        final_test,
        args.off_alpha,
        args.def_alpha,
        args.rubberband_alpha,
    )

    dev_metrics.to_csv(args.output_dir / "development_fold_metrics.csv", index=False)
    dev_losses.to_csv(args.output_dir / "development_game_losses.csv", index=False)
    stress_metrics.to_csv(args.output_dir / "stress_fold_metrics.csv", index=False)
    stress_losses.to_csv(args.output_dir / "stress_game_losses.csv", index=False)
    pd.DataFrame(comparisons).to_csv(args.output_dir / "stress_paired_comparisons.csv", index=False)
    (args.output_dir / "player_stability.json").write_text(
        json.dumps(stability, indent=2, sort_keys=True) + "\n"
    )

    coefficient_columns = [
        "test_year",
        "label",
        "home_effect_per_100",
        "test_home_away_residual_gap_per_100",
        "mse",
    ]
    coefficient_table = pd.concat([dev_metrics, stress_metrics], ignore_index=True)
    coefficient_table = coefficient_table.loc[
        coefficient_table["home"], coefficient_columns
    ].sort_values(["test_year", "label"])
    coefficient_table.to_csv(args.output_dir / "home_effect_by_fold.csv", index=False)

    report = [
        "# Home-Court RAPM Adjustment Research",
        "",
        "## Design",
        "",
        "Rolling five-season training windows predict the next season on identical possessions. The global home-offense indicator is centered on the training home-possession share and estimated as an unpenalized nuisance coefficient in the same joint generalized-ridge system as the player and smooth rubberband blocks.",
        "",
        f"Development seasons: {development_years}. Historical stress seasons: {stress_years}.",
        "",
        "## Development",
        "",
        markdown_table(dev_summary),
        "",
        "## Historical stress",
        "",
        markdown_table(stress_summary),
        "",
        "## Paired game-clustered stress comparisons",
        "",
        markdown_table(pd.DataFrame(comparisons)),
        "",
        "## Home coefficient and residual calibration by fold",
        "",
        markdown_table(coefficient_table),
        "",
        "## Player stability: smooth + home versus smooth",
        "",
        "```json",
        json.dumps(stability, indent=2, sort_keys=True),
        "```",
        "",
        "## Interpretation limit",
        "",
        "This tests a global home-offense context adjustment inside pure RAPM. It does not estimate team-specific arena effects, travel, altitude, rest, crowd mechanisms, or a complete pregame forecast. Historical venue orientation is reconstructed from sparse scoreboard snapshots and lineup continuity; unresolved possessions receive zero home context and are excluded from the home/away residual diagnostic. The test seasons were previously opened for other RAPM research, so this is historical stress evidence rather than untouched confirmation.",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(report) + "\n")
    logging.info("Wrote home-court research artifacts to %s", args.output_dir)


if __name__ == "__main__":
    main()
