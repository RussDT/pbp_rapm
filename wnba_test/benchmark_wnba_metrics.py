"""Benchmark frozen WNBA metrics on held-out possession and game outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "wnba_test" / "Processed"
DEFAULT_RAW = ROOT / "wnba_test" / "GabrielNormalizedClean" / "WNBA26.csv"
DEFAULT_NATIVE = (
    ROOT / "wnba_test" / "Results" / "rapm_22_23_24_25_all" / "weighted_factors.csv"
)
DEFAULT_CLEAN = (
    ROOT
    / "wnba_test"
    / "Results"
    / "clean_decomp_22_25_all_a1750_4000"
    / "weighted_factors_alt3_efg_value_22_25_all_a1750_4000.csv"
)
DEFAULT_BOX_PRIOR = (
    ROOT
    / "wnba_test"
    / "Results"
    / "box_prior_true_prior"
    / "frozen_2025_box_true_prior_total.csv"
)
DEFAULT_RATINGS = ROOT / "nba_pipeline" / "external" / "wnba_darko_daily"
DEFAULT_OUTPUT = ROOT / "wnba_test" / "Results" / "benchmarks" / "holdout_2026"


def normalize_id(series: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
    )


def load_holdout(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"game_id": str}, low_memory=False)
    frame["game_id"] = frame["game_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)
    frame["event_num"] = pd.to_numeric(frame["event_num"], errors="coerce").astype("Int64")
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    frame["actual_points"] = pd.to_numeric(frame["Net_Diff"], errors="coerce")
    required = [
        "game_id",
        "event_num",
        "game_date",
        "actual_points",
        *[f"O{i}" for i in range(1, 6)],
        *[f"D{i}" for i in range(1, 6)],
    ]
    if frame[required].isna().any().any():
        raise ValueError("held-out possession rows contain missing keys, dates, targets, or lineups")
    return frame


def training_mean(start_year: int, end_year: int) -> float:
    values = []
    for year in range(start_year, end_year + 1):
        for suffix in ("", "_PS"):
            path = PROCESSED_DIR / f"RAPM{year:02d}{suffix}.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path, usecols=["Net_Diff"])
            values.append(pd.to_numeric(frame["Net_Diff"], errors="coerce"))
    return float(pd.concat(values, ignore_index=True).mean())


def static_rating_map(
    path: Path,
    defense_multiplier: float = 1.0,
) -> tuple[dict[str, float], dict[str, float]]:
    frame = pd.read_csv(path, dtype={"player_id": str})
    frame["player_id"] = normalize_id(frame["player_id"])
    offense = dict(zip(frame["player_id"], pd.to_numeric(frame["off"], errors="coerce")))
    defense = dict(
        zip(
            frame["player_id"],
            defense_multiplier * pd.to_numeric(frame["def"], errors="coerce"),
        )
    )
    return offense, defense


def static_predictions(
    holdout: pd.DataFrame,
    offense: dict[str, float],
    defense: dict[str, float],
    intercept: float,
) -> tuple[np.ndarray, int, int]:
    prediction = np.full(len(holdout), intercept, dtype=float)
    rated_slots = np.zeros(len(holdout), dtype=int)
    for column in [f"O{i}" for i in range(1, 6)]:
        player_ids = normalize_id(holdout[column])
        values = player_ids.map(offense)
        prediction += values.fillna(0.0).to_numpy(dtype=float) / 100.0
        rated_slots += values.notna().to_numpy(dtype=int)
    for column in [f"D{i}" for i in range(1, 6)]:
        player_ids = normalize_id(holdout[column])
        values = player_ids.map(defense)
        prediction -= values.fillna(0.0).to_numpy(dtype=float) / 100.0
        rated_slots += values.notna().to_numpy(dtype=int)
    return prediction, int(rated_slots.sum()), int((rated_slots == 10).sum())


def latest_snapshot(
    ratings: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    eligible = ratings.loc[ratings["date"] < cutoff]
    return eligible.sort_values(["nba_id", "date"]).drop_duplicates(
        subset=["nba_id"], keep="last"
    )


def dynamic_predictions(
    holdout: pd.DataFrame,
    ratings: pd.DataFrame,
    offense_column: str,
    defense_column: str,
    intercept: float,
    frozen_cutoff: pd.Timestamp | None,
) -> tuple[np.ndarray, int, int]:
    prediction = np.full(len(holdout), intercept, dtype=float)
    rated_slots = np.zeros(len(holdout), dtype=int)
    ratings = ratings.copy()
    ratings["date"] = pd.to_datetime(ratings["date"], errors="coerce")
    ratings["nba_id"] = normalize_id(ratings["nba_id"])

    frozen_snapshot = (
        latest_snapshot(ratings, frozen_cutoff)
        if frozen_cutoff is not None
        else None
    )
    snapshots = (
        {}
        if frozen_snapshot is not None
        else {
            pd.Timestamp(game_date): latest_snapshot(
                ratings, pd.Timestamp(game_date)
            )
            for game_date in holdout["game_date"].unique()
        }
    )

    for game_date, row_index in holdout.groupby("game_date", sort=False).groups.items():
        snapshot = (
            frozen_snapshot
            if frozen_snapshot is not None
            else snapshots[pd.Timestamp(game_date)]
        )
        offense = dict(
            zip(snapshot["nba_id"], pd.to_numeric(snapshot[offense_column], errors="coerce"))
        )
        defense = dict(
            zip(snapshot["nba_id"], pd.to_numeric(snapshot[defense_column], errors="coerce"))
        )
        positions = holdout.index.get_indexer(row_index)
        subset = holdout.loc[row_index]
        for column in [f"O{i}" for i in range(1, 6)]:
            values = normalize_id(subset[column]).map(offense)
            prediction[positions] += values.fillna(0.0).to_numpy(dtype=float) / 100.0
            rated_slots[positions] += values.notna().to_numpy(dtype=int)
        for column in [f"D{i}" for i in range(1, 6)]:
            values = normalize_id(subset[column]).map(defense)
            prediction[positions] -= values.fillna(0.0).to_numpy(dtype=float) / 100.0
            rated_slots[positions] += values.notna().to_numpy(dtype=int)
    return prediction, int(rated_slots.sum()), int((rated_slots == 10).sum())


def resolve_offense_side(holdout: pd.DataFrame, raw_path: Path) -> pd.Series:
    columns = [
        "game_id",
        "event_num",
        *[f"away_player{i}" for i in range(1, 6)],
        *[f"home_player{i}" for i in range(1, 6)],
    ]
    raw = pd.read_csv(raw_path, usecols=columns, dtype={"game_id": str}, low_memory=False)
    raw["game_id"] = raw["game_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)
    raw["event_num"] = pd.to_numeric(raw["event_num"], errors="coerce").astype("Int64")
    if raw.duplicated(["game_id", "event_num"]).any():
        raise ValueError("raw holdout contains duplicate game/event keys")
    away_rosters = {
        game_id: {
            int(value)
            for column in [f"away_player{i}" for i in range(1, 6)]
            for value in group[column]
            if pd.notna(value) and int(value) > 0
        }
        for game_id, group in raw.groupby("game_id")
    }
    home_rosters = {
        game_id: {
            int(value)
            for column in [f"home_player{i}" for i in range(1, 6)]
            for value in group[column]
            if pd.notna(value) and int(value) > 0
        }
        for game_id, group in raw.groupby("game_id")
    }
    merged = holdout[["game_id", "event_num", *[f"O{i}" for i in range(1, 6)]]].merge(
        raw, on=["game_id", "event_num"], how="left", validate="one_to_one"
    )
    sides = []
    for _, row in merged.iterrows():
        offense = {int(row[f"O{i}"]) for i in range(1, 6)}
        away = {int(row[f"away_player{i}"]) for i in range(1, 6)}
        home = {int(row[f"home_player{i}"]) for i in range(1, 6)}
        if offense == away:
            sides.append("away")
        elif offense == home:
            sides.append("home")
        else:
            game_id = row["game_id"]
            away_overlap = len(offense & away_rosters.get(game_id, set()))
            home_overlap = len(offense & home_rosters.get(game_id, set()))
            if away_overlap > home_overlap:
                sides.append("away")
            elif home_overlap > away_overlap:
                sides.append("home")
            else:
                sides.append("")
    return pd.Series(sides, index=holdout.index)


def metric_summary(
    frame: pd.DataFrame,
    prediction_column: str,
    rated_slots: int,
    complete_rows: int,
) -> dict[str, Any]:
    actual = frame["actual_points"].to_numpy(dtype=float)
    prediction = frame[prediction_column].to_numpy(dtype=float)
    error = prediction - actual
    team_games = (
        frame.loc[frame["offense_side"].ne("")]
        .groupby(["game_id", "offense_side"], as_index=False)
        .agg(actual_points=("actual_points", "sum"), predicted_points=(prediction_column, "sum"))
    )
    team_game_error = team_games["predicted_points"] - team_games["actual_points"]
    game_wide = team_games.pivot(
        index="game_id",
        columns="offense_side",
        values=["actual_points", "predicted_points"],
    ).dropna()
    actual_margin = game_wide[("actual_points", "home")] - game_wide[("actual_points", "away")]
    predicted_margin = (
        game_wide[("predicted_points", "home")]
        - game_wide[("predicted_points", "away")]
    )
    margin_error = predicted_margin - actual_margin
    total_slots = len(frame) * 10
    return {
        "possessions": len(frame),
        "possession_rmse": float(np.sqrt(np.mean(error**2))),
        "possession_mae": float(np.mean(np.abs(error))),
        "team_games": len(team_games),
        "team_game_points_rmse": float(np.sqrt(np.mean(team_game_error**2))),
        "team_game_points_mae": float(np.mean(np.abs(team_game_error))),
        "games": len(game_wide),
        "game_margin_rmse": float(np.sqrt(np.mean(margin_error**2))),
        "game_margin_mae": float(np.mean(np.abs(margin_error))),
        "game_margin_correlation": float(actual_margin.corr(predicted_margin)),
        "rated_player_slot_pct": 100.0 * rated_slots / total_slots,
        "complete_rated_possession_pct": 100.0 * complete_rows / len(frame),
    }


def game_margin_errors(frame: pd.DataFrame, prediction_column: str) -> pd.Series:
    team_games = (
        frame.loc[frame["offense_side"].ne("")]
        .groupby(["game_id", "offense_side"], as_index=False)
        .agg(
            actual_points=("actual_points", "sum"),
            predicted_points=(prediction_column, "sum"),
        )
    )
    game_wide = team_games.pivot(
        index="game_id",
        columns="offense_side",
        values=["actual_points", "predicted_points"],
    ).dropna()
    actual_margin = (
        game_wide[("actual_points", "home")]
        - game_wide[("actual_points", "away")]
    )
    predicted_margin = (
        game_wide[("predicted_points", "home")]
        - game_wide[("predicted_points", "away")]
    )
    return (predicted_margin - actual_margin).sort_index()


def paired_bootstrap_rmse_difference(
    metric_errors: pd.Series,
    baseline_errors: pd.Series,
    samples: int = 4000,
    seed: int = 20260723,
) -> dict[str, float]:
    paired = pd.concat(
        [metric_errors.rename("metric"), baseline_errors.rename("baseline")],
        axis=1,
        join="inner",
    ).dropna()
    values = paired.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    differences = np.empty(samples, dtype=float)
    for sample_idx in range(samples):
        draw = rng.integers(0, len(values), size=len(values))
        sampled = values[draw]
        metric_rmse = np.sqrt(np.mean(sampled[:, 0] ** 2))
        baseline_rmse = np.sqrt(np.mean(sampled[:, 1] ** 2))
        differences[sample_idx] = metric_rmse - baseline_rmse
    observed = float(
        np.sqrt(np.mean(values[:, 0] ** 2))
        - np.sqrt(np.mean(values[:, 1] ** 2))
    )
    return {
        "observed_rmse_difference": observed,
        "ci95_low": float(np.quantile(differences, 0.025)),
        "ci95_high": float(np.quantile(differences, 0.975)),
        "bootstrap_probability_metric_better": float(
            np.mean(differences < 0)
        ),
        "games": len(values),
        "samples": samples,
    }


def run_benchmark(
    holdout_path: Path,
    raw_path: Path,
    native_path: Path,
    clean_path: Path,
    box_prior_path: Path,
    ratings_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    holdout = load_holdout(holdout_path)
    holdout["offense_side"] = resolve_offense_side(holdout, raw_path)
    intercept = training_mean(22, 25)
    preseason_cutoff = holdout["game_date"].min()
    metric_coverage: dict[str, tuple[int, int]] = {}
    baseline_slots = len(holdout) * 10
    holdout["league_mean"] = intercept
    metric_coverage["league_mean"] = (baseline_slots, len(holdout))

    static_specs = [
        ("ours_native_frozen", native_path, 1.0),
        ("ours_clean_frozen", clean_path, 1.0),
        # The direct prior solver stores defense as points allowed, while the
        # display files and benchmark expect positive-good defense.
        ("ours_box_trueprior_frozen", box_prior_path, -1.0),
    ]
    for name, path, defense_multiplier in static_specs:
        offense, defense = static_rating_map(path, defense_multiplier)
        prediction, slots, complete = static_predictions(holdout, offense, defense, intercept)
        holdout[name] = prediction
        metric_coverage[name] = (slots, complete)

    rating_specs = [
        ("darko_dpm", "dpm_cols.parq", "o_dpm", "d_dpm"),
        ("pi_rapm", "pi_rapm_ratings.parq", "pi_orapm", "pi_drapm"),
        ("npi_rapm", "npi_rapm_ratings.parq", "npi_orapm", "npi_drapm"),
    ]
    for name, filename, offense_column, defense_column in rating_specs:
        ratings = pd.read_parquet(ratings_dir / filename)
        for policy, cutoff in [("preseason", preseason_cutoff), ("live_prior_day", None)]:
            column = f"{name}_{policy}"
            prediction, slots, complete = dynamic_predictions(
                holdout,
                ratings,
                offense_column,
                defense_column,
                intercept,
                frozen_cutoff=cutoff,
            )
            holdout[column] = prediction
            metric_coverage[column] = (slots, complete)

    summaries = {
        column: metric_summary(holdout, column, *metric_coverage[column])
        for column in metric_coverage
    }
    baseline_errors = game_margin_errors(holdout, "league_mean")
    for column in summaries:
        if column == "league_mean":
            continue
        bootstrap = paired_bootstrap_rmse_difference(
            game_margin_errors(holdout, column),
            baseline_errors,
        )
        for key, value in bootstrap.items():
            summaries[column][f"vs_league_{key}"] = value
    report = {
        "train_window": "2022-2025 RS+PS clean lineup surface",
        "holdout_window": "2026 RS through the source checkout date",
        "evaluation_unit": (
            "Conditional-on-observed-lineup possession predictions, summed to "
            "team-game points and game margins using observed possession counts; "
            "this is not a pregame rotation or pace forecast."
        ),
        "training_points_per_possession_intercept": intercept,
        "preseason_cutoff": preseason_cutoff.date().isoformat(),
        "rating_join_rule": (
            "preseason uses latest rating strictly before the first holdout game; "
            "live_prior_day uses latest rating strictly before each game date"
        ),
        "resolved_offense_side_pct": 100.0
        * holdout["offense_side"].ne("").mean(),
        "metrics": summaries,
        "paired_game_margin_bootstrap": {
            f"{left}_vs_{right}": {
                "interpretation": (
                    "Negative RMSE difference and probability above 0.5 favor "
                    f"{left} over {right}."
                ),
                **paired_bootstrap_rmse_difference(
                    game_margin_errors(holdout, left),
                    game_margin_errors(holdout, right),
                ),
            }
            for left, right in [
                ("ours_box_trueprior_frozen", "darko_dpm_live_prior_day"),
                ("ours_box_trueprior_frozen", "pi_rapm_live_prior_day"),
                ("ours_box_trueprior_frozen", "npi_rapm_live_prior_day"),
                ("ours_native_frozen", "darko_dpm_live_prior_day"),
                ("ours_native_frozen", "npi_rapm_live_prior_day"),
                ("ours_native_frozen", "npi_rapm_preseason"),
                ("darko_dpm_live_prior_day", "npi_rapm_live_prior_day"),
            ]
        },
    }
    summary_frame = (
        pd.DataFrame.from_dict(summaries, orient="index")
        .rename_axis("metric")
        .reset_index()
        .sort_values("game_margin_rmse")
    )
    return summary_frame, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=Path, default=PROCESSED_DIR / "RAPM26.csv")
    parser.add_argument("--raw-holdout", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--native", type=Path, default=DEFAULT_NATIVE)
    parser.add_argument("--clean", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--box-prior", type=Path, default=DEFAULT_BOX_PRIOR)
    parser.add_argument("--ratings-dir", type=Path, default=DEFAULT_RATINGS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    summary, report = run_benchmark(
        args.holdout,
        args.raw_holdout,
        args.native,
        args.clean,
        args.box_prior,
        args.ratings_dir,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.csv"
    report_path = args.output_dir / "report.json"
    summary.to_csv(summary_path, index=False)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(summary.to_string(index=False))
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
