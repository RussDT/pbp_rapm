#!/usr/bin/env python3
"""Select and stress-test a clock-aware RAPM rubberband adjustment.

The harness uses rolling five-season training windows and next-season tests.
Candidate/penalty selection is restricted to the development test years;
later seasons are opened only after one candidate is frozen.
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
from scipy.sparse import diags, hstack, vstack
from scipy.sparse.linalg import lsqr

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_ROOT.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import rapm  # noqa: E402
from rubberband_model import (  # noqa: E402
    build_rubberband_basis,
    compute_pre_possession_clock_state,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass(frozen=True)
class ModelSpec:
    model: str
    alpha: float | None

    @property
    def label(self) -> str:
        return self.model if self.alpha is None else f"{self.model}_a{self.alpha:g}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-years", default="2018,2019,2020,2021")
    parser.add_argument("--stress-years", default="2022,2023,2024,2025,2026")
    parser.add_argument("--train-window", type=int, default=5)
    parser.add_argument("--alphas", default="1,10,100")
    parser.add_argument("--season-type", choices=["RS", "ALL"], default="ALL")
    parser.add_argument("--off-alpha", type=float, default=2000.0)
    parser.add_argument("--def-alpha", type=float, default=4000.0)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PIPELINE_ROOT / "results" / "rubberband_research",
    )
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def season_paths(year: int, season_type: str) -> list[Path]:
    paths = [PIPELINE_ROOT / "processed" / f"RAPM{year % 100:02d}.parquet"]
    if season_type == "ALL":
        paths.append(PIPELINE_ROOT / "processed" / f"RAPM{year % 100:02d}_PS.parquet")
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing processed RAPM files: {missing}")
    return paths


def prepare_season(year: int, season_type: str) -> pd.DataFrame:
    frames = []
    for path in season_paths(year, season_type):
        frame = pd.read_parquet(path)
        frame["season_phase"] = "PS" if path.stem.endswith("_PS") else "RS"
        frame["source_season_end_year"] = year
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    prepared = rapm.detect_file_type_and_prepare(raw, pure=True, prefix="RAPM")
    transformed = rapm.transform_to_home_away_format(prepared).reset_index(drop=True)
    clock = compute_pre_possession_clock_state(prepared.reset_index(drop=True))
    for column in clock.columns:
        transformed[column] = clock[column].to_numpy()
    return transformed


def build_player_mapping(df: pd.DataFrame) -> tuple[list[str], dict[str, int]]:
    all_players: set[str] = set()
    for column in [f"a{i}" for i in range(1, 6)] + [f"h{i}" for i in range(1, 6)]:
        values = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)
        all_players.update(str(value) for value in values.unique() if value > 0)
    players = sorted(all_players)
    mapping: dict[str, int] = {}
    for player in players:
        mapping[f"{player}_off"] = len(mapping)
        mapping[f"{player}_def"] = len(mapping)
    return players, mapping


def build_base_design_pair(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[object, object, np.ndarray, np.ndarray, list[str], dict[str, int]]:
    combined = pd.concat([train, test], ignore_index=True)
    players, base_mapping = build_player_mapping(combined)
    train_mapping = dict(base_mapping)
    test_mapping = dict(base_mapping)
    x_train, y_train, _ = rapm.build_simple_design_matrix(train, train_mapping)
    x_test, y_test, _ = rapm.build_simple_design_matrix(test, test_mapping)

    return x_train, x_test, y_train, y_test, players, train_mapping


def append_rubberband_design(
    base_design: tuple[object, object, np.ndarray, np.ndarray, list[str], dict[str, int]],
    train: pd.DataFrame,
    test: pd.DataFrame,
    model: str,
) -> tuple[object, object, np.ndarray, np.ndarray, list[str], dict[str, int]]:
    x_train, x_test, y_train, y_test, players, base_mapping = base_design
    train_mapping = dict(base_mapping)

    train_basis = build_rubberband_basis(train, model)
    test_basis = build_rubberband_basis(test, model)
    if train_basis.feature_names != test_basis.feature_names:
        raise AssertionError("Train/test rubberband basis schema differs.")
    if train_basis.feature_names:
        start = x_train.shape[1]
        for offset, name in enumerate(train_basis.feature_names):
            train_mapping[name] = start + offset
        x_train = hstack([x_train, train_basis.matrix], format="csr")
        x_test = hstack([x_test, test_basis.matrix], format="csr")
    return x_train, x_test, y_train, y_test, players, train_mapping


def build_design_pair(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model: str,
) -> tuple[object, object, np.ndarray, np.ndarray, list[str], dict[str, int]]:
    base_design = build_base_design_pair(train, test)
    return append_rubberband_design(base_design, train, test, model)


def calibration_rmse(df: pd.DataFrame, residual: np.ndarray) -> float:
    margin_edges = [-np.inf, -15, -10, -5, 5, 10, 15, np.inf]
    time_edges = [-0.1, 120, 720, 1440, 2160, np.inf]
    audit = pd.DataFrame(
        {
            "margin_bin": pd.cut(df["score_diff"], margin_edges, right=True),
            "time_bin": pd.cut(df["seconds_remaining_game"], time_edges, right=True),
            "residual": residual,
        }
    )
    grouped = audit.groupby(["margin_bin", "time_bin"], observed=True)["residual"].agg(["mean", "size"])
    return float(np.sqrt(np.average(np.square(grouped["mean"]), weights=grouped["size"])))


def solve_generalized_ridge(
    matrix,
    target: np.ndarray,
    mapping: dict[str, int],
    off_alpha: float,
    def_alpha: float,
    rb_alpha: float | None,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Solve the different-penalty RAPM objective in one joint LSQR system."""

    penalties = np.zeros(matrix.shape[1], dtype=np.float64)
    for name, index in mapping.items():
        if name.endswith("_off"):
            penalties[index] = off_alpha
        elif name.endswith("_def"):
            penalties[index] = def_alpha
        elif name.startswith("rb_"):
            if rb_alpha is None:
                raise ValueError("Rubberband feature found without rb_alpha.")
            penalties[index] = rb_alpha
        else:
            raise ValueError(f"Unrecognized generalized-ridge feature: {name}")
    if np.any(penalties <= 0):
        raise ValueError("Every research-harness feature must have a positive penalty.")

    penalty_rows = diags(np.sqrt(penalties), offsets=0, format="csr")
    augmented_matrix = vstack([matrix, penalty_rows], format="csr")
    augmented_target = np.concatenate([target, np.zeros(matrix.shape[1], dtype=np.float64)])
    solution = lsqr(
        augmented_matrix,
        augmented_target,
        atol=1e-7,
        btol=1e-7,
        iter_lim=1000,
        show=False,
    )
    meta = {
        "lsqr_stop": int(solution[1]),
        "lsqr_iterations": int(solution[2]),
        "lsqr_residual_norm": float(solution[3]),
        "lsqr_normal_residual_norm": float(solution[4]),
    }
    if meta["lsqr_stop"] == 7:
        raise RuntimeError("Generalized ridge LSQR reached its iteration limit.")
    return np.asarray(solution[0], dtype=np.float64), meta


def fit_and_score(
    train: pd.DataFrame,
    test: pd.DataFrame,
    spec: ModelSpec,
    off_alpha: float,
    def_alpha: float,
    design: tuple[object, object, np.ndarray, np.ndarray, list[str], dict[str, int]] | None = None,
) -> tuple[dict[str, float | int | str], pd.DataFrame, np.ndarray, dict[str, int]]:
    if design is None:
        design = build_design_pair(train, test, spec.model)
    x_train, x_test, y_train, y_test, players, mapping = design
    train_mean = float(y_train.mean())
    has_rb = spec.model != "none"
    beta, solver_meta = solve_generalized_ridge(
        x_train,
        y_train - train_mean,
        mapping,
        off_alpha,
        def_alpha,
        spec.alpha if has_rb else None,
    )
    prediction = np.asarray(x_test @ beta).ravel() + train_mean
    residual = y_test - prediction
    squared_error = np.square(residual)
    absolute_error = np.abs(residual)
    row = {
        "model": spec.model,
        "alpha": np.nan if spec.alpha is None else spec.alpha,
        "label": spec.label,
        "rows": int(len(test)),
        "games": int(test["gameid"].nunique()),
        "mse": float(squared_error.mean()),
        "mae": float(absolute_error.mean()),
        "calibration_rmse": calibration_rmse(test, residual),
        **solver_meta,
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
    specs: list[ModelSpec],
    off_alpha: float,
    def_alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = []
    game_losses = []
    cache: dict[int, pd.DataFrame] = {}

    def get(year: int) -> pd.DataFrame:
        if year not in cache:
            logging.info("Preparing season %s", year)
            cache[year] = prepare_season(year, season_type)
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
        base_design = build_base_design_pair(train, test)
        design_cache: dict[
            str,
            tuple[object, object, np.ndarray, np.ndarray, list[str], dict[str, int]],
        ] = {"none": base_design}
        for spec in specs:
            logging.info("Scoring %s on fold %s", spec.label, test_year)
            if spec.model not in design_cache:
                design_cache[spec.model] = append_rubberband_design(
                    base_design,
                    train,
                    test,
                    spec.model,
                )
            row, losses, _, _ = fit_and_score(
                train,
                test,
                spec,
                off_alpha,
                def_alpha,
                design=design_cache[spec.model],
            )
            row["test_year"] = test_year
            row["train_start"] = train_years[0]
            row["train_end"] = train_years[-1]
            metrics.append(row)
            losses["test_year"] = test_year
            game_losses.append(losses)
        oldest_needed = test_year - train_window + 1
        cache = {year: frame for year, frame in cache.items() if year >= oldest_needed}

    return pd.DataFrame(metrics), pd.concat(game_losses, ignore_index=True)


def equal_fold_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.groupby(["label", "model", "alpha"], dropna=False, as_index=False).agg(
        folds=("test_year", "nunique"),
        mean_mse=("mse", "mean"),
        mean_mae=("mae", "mean"),
        mean_calibration_rmse=("calibration_rmse", "mean"),
    ).sort_values("mean_mse", kind="stable")


def paired_cluster_bootstrap(
    game_losses: pd.DataFrame,
    challenger: str,
    reference: str,
    draws: int,
    seed: int,
) -> dict[str, float | int | str]:
    left = game_losses[game_losses["label"] == challenger].copy()
    right = game_losses[game_losses["label"] == reference].copy()
    merged = left.merge(right, on=["test_year", "gameid"], suffixes=("_challenger", "_reference"))
    if merged.empty:
        raise ValueError(f"No common games for {challenger} versus {reference}")
    delta_sum = merged["squared_error_sum_reference"].to_numpy() - merged["squared_error_sum_challenger"].to_numpy()
    counts = merged["rows_reference"].to_numpy(dtype=float)
    point = float(delta_sum.sum() / counts.sum())
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=float)
    n_games = len(merged)
    for draw in range(draws):
        index = rng.integers(0, n_games, size=n_games)
        samples[draw] = delta_sum[index].sum() / counts[index].sum()
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "challenger": challenger,
        "reference": reference,
        "games": n_games,
        "mse_improvement": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "win_probability": float((samples > 0).mean()),
    }


def player_stability(
    final_train: pd.DataFrame,
    final_test: pd.DataFrame,
    winner: ModelSpec,
    off_alpha: float,
    def_alpha: float,
) -> dict[str, float | int | str]:
    fitted = {}
    for spec in (ModelSpec("legacy", 10.0), winner):
        _, _, beta, mapping = fit_and_score(final_train, final_test, spec, off_alpha, def_alpha)
        player_ids = sorted({key[:-4] for key in mapping if key.endswith("_off")})
        net = {
            player: 100.0 * (
                beta[mapping[f"{player}_off"]] - beta[mapping[f"{player}_def"]]
            )
            for player in player_ids
        }
        fitted[spec.label] = net
    common = sorted(set(fitted["legacy_a10"]) & set(fitted[winner.label]))
    legacy = np.array([fitted["legacy_a10"][player] for player in common])
    challenger = np.array([fitted[winner.label][player] for player in common])
    delta = challenger - legacy
    return {
        "winner": winner.label,
        "reference": "legacy_a10",
        "players": len(common),
        "correlation": float(np.corrcoef(legacy, challenger)[0, 1]),
        "median_abs_shift_per_100": float(np.median(np.abs(delta))),
        "p90_abs_shift_per_100": float(np.quantile(np.abs(delta), 0.90)),
        "max_abs_shift_per_100": float(np.max(np.abs(delta))),
    }


def write_report(
    output_path: Path,
    development_summary: pd.DataFrame,
    stress_summary: pd.DataFrame,
    comparisons: list[dict[str, float | int | str]],
    stability: dict[str, float | int | str],
    winner: ModelSpec,
    development_years: list[int],
    stress_years: list[int],
) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        display = frame.copy()
        for column in display.select_dtypes(include=[np.number]).columns:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.8g}"
            )
        headers = [str(column) for column in display.columns]
        rows = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
        table = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        table.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(table)

    lines = [
        "# Continuous Rubberband Adjustment Research",
        "",
        "## Decision",
        "",
        f"Selected on development seasons {development_years}: `{winner.label}`.",
        f"Chronological stress seasons: {stress_years}.",
        "",
        "The candidate is promoted only if stress MSE improves versus both no rubberband and the legacy buckets, with the game-clustered interval clearing zero, while state calibration and player stability remain acceptable.",
        "",
        "## Development selection",
        "",
        markdown_table(development_summary),
        "",
        "## Frozen stress results",
        "",
        markdown_table(stress_summary),
        "",
        "## Paired game-clustered comparisons",
        "",
        markdown_table(pd.DataFrame(comparisons)),
        "",
        "## Player-coefficient stability versus legacy buckets",
        "",
        "```json",
        json.dumps(stability, indent=2, sort_keys=True),
        "```",
        "",
        "## Interpretation limits",
        "",
        "These are historical forward tests and therefore development/stress evidence, not untouched 2026-27 confirmation. The rubberband block remains a nuisance adjustment jointly fit with player coefficients; it is not a standalone player metric.",
    ]
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    development_years = parse_int_list(args.development_years)
    stress_years = parse_int_list(args.stress_years)
    alphas = parse_float_list(args.alphas)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    development_specs = [ModelSpec("none", None), ModelSpec("legacy", 10.0)]
    for model in ("pressure", "smooth"):
        development_specs.extend(ModelSpec(model, alpha) for alpha in alphas)

    dev_metrics, dev_losses = evaluate_folds(
        development_years,
        args.train_window,
        args.season_type,
        development_specs,
        args.off_alpha,
        args.def_alpha,
    )
    dev_summary = equal_fold_summary(dev_metrics)
    candidate_summary = dev_summary[dev_summary["model"].isin(["pressure", "smooth"])]
    winner_row = candidate_summary.iloc[0]
    winner = ModelSpec(str(winner_row["model"]), float(winner_row["alpha"]))
    logging.info("Frozen winner from development: %s", winner.label)

    stress_specs = [ModelSpec("none", None), ModelSpec("legacy", 10.0), winner]
    stress_metrics, stress_losses = evaluate_folds(
        stress_years,
        args.train_window,
        args.season_type,
        stress_specs,
        args.off_alpha,
        args.def_alpha,
    )
    stress_summary = equal_fold_summary(stress_metrics)
    comparisons = [
        paired_cluster_bootstrap(
            stress_losses,
            winner.label,
            reference,
            args.bootstrap_draws,
            args.seed + offset,
        )
        for offset, reference in enumerate(("none", "legacy_a10"))
    ]

    final_test_year = max(stress_years)
    final_train = pd.concat(
        [prepare_season(year, args.season_type) for year in range(final_test_year - args.train_window, final_test_year)],
        ignore_index=True,
    )
    final_test = prepare_season(final_test_year, args.season_type)
    stability = player_stability(final_train, final_test, winner, args.off_alpha, args.def_alpha)

    dev_metrics.to_csv(args.output_dir / "development_fold_metrics.csv", index=False)
    dev_losses.to_csv(args.output_dir / "development_game_losses.csv", index=False)
    stress_metrics.to_csv(args.output_dir / "stress_fold_metrics.csv", index=False)
    stress_losses.to_csv(args.output_dir / "stress_game_losses.csv", index=False)
    pd.DataFrame(comparisons).to_csv(args.output_dir / "stress_paired_comparisons.csv", index=False)
    (args.output_dir / "player_stability.json").write_text(json.dumps(stability, indent=2, sort_keys=True) + "\n")
    write_report(
        args.output_dir / "REPORT.md",
        dev_summary,
        stress_summary,
        comparisons,
        stability,
        winner,
        development_years,
        stress_years,
    )
    logging.info("Wrote rubberband research artifacts to %s", args.output_dir)


if __name__ == "__main__":
    main()
