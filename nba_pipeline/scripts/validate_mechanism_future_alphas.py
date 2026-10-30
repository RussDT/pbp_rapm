#!/usr/bin/env python3
"""Validate mechanism RAPM penalties on later two-season possession outcomes.

The production grouped-game CV answers same-window fit.  This harness asks the
more important transport question: fit a trailing five-season TD550 model,
then score the next non-overlapping two-season block.  Future season/phase
intercepts are evaluation-only nuisance terms; no future outcome enters the
player fit.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PIPELINE_ROOT.parent
RAPM_PATH = Path(__file__).resolve().with_name("rapm.py")
REGISTRY_PATH = PIPELINE_ROOT / "config" / "rapm_table_registry.json"
DEFAULT_OUTPUT = PIPELINE_ROOT / "results" / "mechanism_future_alpha_validation"
STANDARD_PAIRS = ((100, 100), (300, 300), (1000, 1000), (3000, 3000), (10000, 10000))
STANDARD_ALPHAS = tuple(pair[0] for pair in STANDARD_PAIRS)


def _load_rapm_module():
    spec = importlib.util.spec_from_file_location("rapm_future_alpha_core", RAPM_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {RAPM_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class Mechanism:
    mechanism_id: str
    metric: str
    prefix: str
    current_off_alpha: int
    current_def_alpha: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanism", action="append", default=[])
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--processed-dir", type=Path, default=PIPELINE_ROOT / "processed")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-seasons", type=int, default=5)
    parser.add_argument("--future-seasons", type=int, default=2)
    parser.add_argument("--half-life", type=float, default=550.0)
    parser.add_argument("--min-blocks", type=int, default=3)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument(
        "--full-alpha-grid",
        action="store_true",
        help="Test the full Cartesian offense-by-defense grid over the standard alpha values.",
    )
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--alpha-rb", type=float, default=100.0)
    parser.add_argument("--future-rapm-target-alpha", type=float, default=3000.0)
    parser.add_argument("--min-player-poss", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-existing", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def load_mechanisms(path: Path, requested: Iterable[str]) -> list[Mechanism]:
    registry = json.loads(path.read_text())
    current = registry["current_mechanism_view"]
    current_ids = [row["mechanism_id"] for row in current["mechanisms"]]
    wanted = list(requested) or current_ids
    by_id = {row["mechanism_id"]: row for row in registry["mechanism_rapms"]}
    manifest_path = PIPELINE_ROOT / "results" / "mechanism_td550_current" / "run_manifest.json"
    selected = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for row in manifest.get("records", manifest.get("completed", [])):
            if row.get("status") not in {"built", "complete", "completed"}:
                continue
            selected[row.get("mechanism_id")] = (
                int(float(row.get("selected_off_alpha", row.get("off_alpha", 3000)))),
                int(float(row.get("selected_def_alpha", row.get("def_alpha", 3000)))),
            )
    result = []
    for mechanism_id in wanted:
        row = by_id.get(mechanism_id)
        if row is None or not row.get("metric"):
            logging.warning("Skipping unknown or specialized mechanism %s", mechanism_id)
            continue
        default_pair = tuple(int(value) for value in row.get("alphas", [3000, 3000]))
        off_alpha, def_alpha = selected.get(mechanism_id, default_pair)
        result.append(
            Mechanism(
                mechanism_id=mechanism_id,
                metric=str(row["metric"]),
                prefix=str(row["processed_prefix"]),
                current_off_alpha=off_alpha,
                current_def_alpha=def_alpha,
            )
        )
    return result


def season_path(processed_dir: Path, prefix: str, year: int, phase: str) -> Path:
    suffix = "_PS" if phase == "PS" else ""
    return processed_dir / f"{prefix}{year % 100:02d}{suffix}.parquet"


def complete_years(processed_dir: Path, prefix: str, end_year: int) -> list[int]:
    return [
        year
        for year in range(1997, end_year + 1)
        if season_path(processed_dir, prefix, year, "RS").exists()
        and season_path(processed_dir, prefix, year, "PS").exists()
    ]


def nonoverlapping_blocks(
    years: list[int], train_seasons: int, future_seasons: int, max_blocks: int | None
) -> list[tuple[list[int], list[int]]]:
    available = set(years)
    if not years:
        return []
    blocks = []
    latest_cutoff = max(years) - future_seasons
    for cutoff in range(latest_cutoff, min(years) + train_seasons - 2, -future_seasons):
        train = list(range(cutoff - train_seasons + 1, cutoff + 1))
        future = list(range(cutoff + 1, cutoff + future_seasons + 1))
        if set(train + future).issubset(available):
            blocks.append((train, future))
    blocks.reverse()
    if max_blocks is not None:
        blocks = blocks[-max_blocks:]
    return blocks


def candidate_pairs(mechanism: Mechanism, full_alpha_grid: bool = False) -> list[tuple[int, int, str]]:
    current = (mechanism.current_off_alpha, mechanism.current_def_alpha)
    if full_alpha_grid:
        pairs = {
            (off_alpha, def_alpha): f"grid_{off_alpha}_{def_alpha}"
            for off_alpha in STANDARD_ALPHAS
            for def_alpha in STANDARD_ALPHAS
        }
    else:
        pairs = {(a, d): f"diag_{a}" for a, d in STANDARD_PAIRS}
    pairs[(current[0], 3000)] = "current_off_def3000"
    pairs[(3000, current[1])] = "off3000_current_def"
    # Assign current last so a default 3000/3000 pair is still identifiable
    # in receipts even when it coincides with both mixed diagnostic pairs.
    pairs[current] = "current_game_cv"
    return [
        (off, defense, label)
        for (off, defense), label in sorted(
            pairs.items(), key=lambda item: (item[0][0] + item[0][1], item[0])
        )
    ]


def load_transformed(module, mechanism: Mechanism, processed_dir: Path, years: list[int]) -> pd.DataFrame:
    paths = [
        season_path(processed_dir, mechanism.prefix, year, phase)
        for year in years
        for phase in ("RS", "PS")
    ]
    frames = module._load_files_sequential([str(path) for path in paths], False, prefix=mechanism.metric)
    if len(frames) != len(paths):
        raise RuntimeError(f"{mechanism.mechanism_id}: loaded {len(frames)} of {len(paths)} inputs")
    raw = pd.concat(frames, ignore_index=True)
    transformed = module.transform_to_home_away_format(raw)
    transformed["season"] = pd.to_numeric(transformed["season"], errors="raise").astype(int)
    transformed["season_phase"] = transformed["season_phase"].astype(str).str.upper()
    return transformed


def build_design(module, frame: pd.DataFrame):
    players = set()
    for column in [f"a{i}" for i in range(1, 6)] + [f"h{i}" for i in range(1, 6)]:
        values = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(int)
        players.update(str(value) for value in values.unique() if value > 0)
    all_players = sorted(players)
    mapping = {}
    for player in all_players:
        mapping[f"{player}_off"] = len(mapping)
        mapping[f"{player}_def"] = len(mapping)
    X, y, metadata = module.build_design_matrix_with_smooth_rubberband(
        frame, mapping, season_effects=True
    )
    return X, np.asarray(y, dtype=float), mapping, all_players, metadata


def oracle_baseline_mse(frame: pd.DataFrame, actual: np.ndarray, predicted: np.ndarray) -> tuple[float, list[dict]]:
    scored = frame[["season", "season_phase"]].copy()
    scored["actual"] = actual
    scored["predicted"] = predicted
    scored["residual"] = scored["actual"] - scored["predicted"]
    scored["centered_residual"] = scored["residual"] - scored.groupby(
        ["season", "season_phase"], sort=False
    )["residual"].transform("mean")
    season_rows = []
    for season, group in scored.groupby("season", sort=True):
        season_rows.append(
            {
                "season": int(season),
                "mse": float(np.mean(np.square(group["centered_residual"]))),
                "rows": int(len(group)),
            }
        )
    return float(np.mean([row["mse"] for row in season_rows])), season_rows


def weighted_coefficient_agreement(
    train_beta: np.ndarray,
    future_beta: np.ndarray,
    X_train,
    X_future,
    mapping: dict[str, int],
    side: str,
    min_player_poss: int,
) -> dict[str, float | int]:
    suffix = f"_{side}"
    indices = np.array(
        [index for name, index in mapping.items() if name.endswith(suffix) and name[:-4].isdigit()],
        dtype=int,
    )
    train_poss = np.asarray(X_train[:, indices].sum(axis=0)).ravel()
    future_poss = np.asarray(X_future[:, indices].sum(axis=0)).ravel()
    keep = (train_poss >= min_player_poss) & (future_poss >= min_player_poss)
    if keep.sum() < 3:
        return {"players": int(keep.sum()), "rmse_x100": np.nan, "correlation": np.nan}
    weights = np.minimum(train_poss[keep], future_poss[keep])
    train_values = train_beta[indices][keep]
    future_values = future_beta[indices][keep]
    train_values = train_values - np.average(train_values, weights=weights)
    future_values = future_values - np.average(future_values, weights=weights)
    rmse = math.sqrt(float(np.average(np.square(train_values - future_values), weights=weights)))
    covariance = float(np.average(train_values * future_values, weights=weights))
    train_sd = math.sqrt(float(np.average(np.square(train_values), weights=weights)))
    future_sd = math.sqrt(float(np.average(np.square(future_values), weights=weights)))
    correlation = covariance / (train_sd * future_sd) if train_sd > 0 and future_sd > 0 else np.nan
    return {"players": int(keep.sum()), "rmse_x100": rmse * 100.0, "correlation": correlation}


def paired_one_se(summary: pd.DataFrame) -> tuple[tuple[int, int], pd.DataFrame]:
    pivot = summary.pivot_table(
        index=["off_alpha", "def_alpha", "candidate"], columns="block", values="future_mse"
    ).reset_index()
    block_columns = [column for column in pivot.columns if isinstance(column, int)]
    if len(block_columns) < 2:
        raise ValueError("Paired one-SE selection requires at least two future blocks")
    pivot["mean_future_mse"] = pivot[block_columns].mean(axis=1)
    best = pivot.sort_values(["mean_future_mse", "off_alpha", "def_alpha"]).iloc[0]
    best_losses = best[block_columns].to_numpy(dtype=float)
    excess_means, excess_ses = [], []
    for _, row in pivot.iterrows():
        differences = row[block_columns].to_numpy(dtype=float) - best_losses
        excess_means.append(float(differences.mean()))
        excess_ses.append(float(differences.std(ddof=1) / math.sqrt(len(differences))))
    pivot["paired_mean_excess"] = excess_means
    pivot["paired_excess_se"] = excess_ses
    pivot["within_paired_one_se"] = pivot["paired_mean_excess"] <= pivot["paired_excess_se"] + 1e-15
    # Lower inverse-alpha mass means a more strongly shrunk two-sided model.
    # This behaves sensibly for asymmetric pairs, unlike simply summing alphas.
    pivot["shrinkage_complexity"] = 1.0 / pivot["off_alpha"] + 1.0 / pivot["def_alpha"]
    selected = pivot[pivot["within_paired_one_se"]].sort_values(
        ["shrinkage_complexity", "off_alpha", "def_alpha"], ascending=[True, False, False]
    ).iloc[0]
    pair = (int(selected["off_alpha"]), int(selected["def_alpha"]))
    pivot["selected"] = (pivot["off_alpha"] == pair[0]) & (pivot["def_alpha"] == pair[1])
    return pair, pivot


def summarize_existing(output_dir: Path) -> pd.DataFrame:
    rows = []
    for detail_path in sorted(output_dir.glob("*_future_blocks.csv")):
        detail = pd.read_csv(detail_path)
        if detail.empty:
            continue
        mechanism_id = str(detail["mechanism_id"].iloc[0])
        selected_pair, selection = paired_one_se(detail)
        selection.to_csv(output_dir / f"{mechanism_id}_selection.csv", index=False)
        exact = selection.sort_values("mean_future_mse").iloc[0]
        selected = selection[selection["selected"]].iloc[0]
        current_rows = selection[selection["candidate"] == "current_game_cv"]
        current = current_rows.iloc[0] if not current_rows.empty else None
        rows.append(
            {
                "mechanism_id": mechanism_id,
                "blocks": int(detail["block"].nunique()),
                "first_future_year": int(detail["future_start"].min()),
                "last_future_year": int(detail["future_end"].max()),
                "current_off_alpha": int(current["off_alpha"]) if current is not None else np.nan,
                "current_def_alpha": int(current["def_alpha"]) if current is not None else np.nan,
                "future_selected_off_alpha": selected_pair[0],
                "future_selected_def_alpha": selected_pair[1],
                "exact_min_off_alpha": int(exact["off_alpha"]),
                "exact_min_def_alpha": int(exact["def_alpha"]),
                "selected_future_mse": float(selected["mean_future_mse"]),
                "current_future_mse": float(current["mean_future_mse"]) if current is not None else np.nan,
                "selected_minus_current_mse": (
                    float(selected["mean_future_mse"] - current["mean_future_mse"])
                    if current is not None
                    else np.nan
                ),
                "selected_relative_mse_change_pct": (
                    100.0 * float(selected["mean_future_mse"] / current["mean_future_mse"] - 1.0)
                    if current is not None
                    else np.nan
                ),
            }
        )
    summary = pd.DataFrame(rows).sort_values("mechanism_id").reset_index(drop=True)
    summary.to_csv(output_dir / "summary.csv", index=False)
    return summary


def run_mechanism(module, mechanism: Mechanism, args: argparse.Namespace) -> dict:
    years = complete_years(args.processed_dir, mechanism.prefix, args.end_year)
    blocks = nonoverlapping_blocks(years, args.train_seasons, args.future_seasons, args.max_blocks)
    payload = {
        "mechanism_id": mechanism.mechanism_id,
        "metric": mechanism.metric,
        "coverage": [min(years), max(years)] if years else None,
        "blocks": [{"train": train, "future": future} for train, future in blocks],
        "current_game_cv_pair": [mechanism.current_off_alpha, mechanism.current_def_alpha],
        "candidates": [list(pair) for pair in candidate_pairs(mechanism, args.full_alpha_grid)],
    }
    if len(blocks) < args.min_blocks:
        payload["status"] = "insufficient_chronological_blocks"
        return payload
    if args.dry_run:
        payload["status"] = "dry_run"
        return payload

    used_years = sorted({year for train, future in blocks for year in train + future})
    frame = load_transformed(module, mechanism, args.processed_dir, used_years)
    X, y, mapping, players, _ = build_design(module, frame)
    decay_base = module.half_life_to_decay_base(args.half_life)
    rows = []
    for block_index, (train_years, future_years) in enumerate(blocks, start=1):
        train_mask = frame["season"].isin(train_years).to_numpy()
        future_mask = frame["season"].isin(future_years).to_numpy()
        train_idx = np.flatnonzero(train_mask)
        future_idx = np.flatnonzero(future_mask)
        weights = module.compute_time_decay_weights(frame.iloc[train_idx], decay_base=decay_base)
        train_mean = float(np.average(y[train_idx], weights=weights))
        y_train = y[train_idx] - train_mean
        y_future_centered = y[future_idx] - float(np.mean(y[future_idx]))
        future_beta = module.joint_generalized_rapm(
            X[future_idx],
            y_future_centered,
            mapping,
            players,
            alpha_offense=args.future_rapm_target_alpha,
            alpha_defense=args.future_rapm_target_alpha,
            rubberband=True,
            alpha_rb=args.alpha_rb,
        )
        for off_alpha, def_alpha, label in candidate_pairs(mechanism, args.full_alpha_grid):
            beta = module.joint_generalized_rapm(
                X[train_idx],
                y_train,
                mapping,
                players,
                alpha_offense=off_alpha,
                alpha_defense=def_alpha,
                weights=weights,
                rubberband=True,
                alpha_rb=args.alpha_rb,
            )
            future_prediction = np.asarray(X[future_idx] @ beta).ravel()
            block_mse, season_scores = oracle_baseline_mse(
                frame.iloc[future_idx], y[future_idx], future_prediction
            )
            off_agreement = weighted_coefficient_agreement(
                beta, future_beta, X[train_idx], X[future_idx], mapping, "off", args.min_player_poss
            )
            def_agreement = weighted_coefficient_agreement(
                beta, future_beta, X[train_idx], X[future_idx], mapping, "def", args.min_player_poss
            )
            rows.append(
                {
                    "mechanism_id": mechanism.mechanism_id,
                    "block": block_index,
                    "train_start": min(train_years),
                    "train_end": max(train_years),
                    "future_start": min(future_years),
                    "future_end": max(future_years),
                    "off_alpha": off_alpha,
                    "def_alpha": def_alpha,
                    "candidate": label,
                    "future_mse": block_mse,
                    "future_rmse_x100": math.sqrt(block_mse) * 100.0,
                    "train_rows": len(train_idx),
                    "future_rows": len(future_idx),
                    "future_rapm_target_alpha": args.future_rapm_target_alpha,
                    "future_rapm_off_rmse_x100": off_agreement["rmse_x100"],
                    "future_rapm_def_rmse_x100": def_agreement["rmse_x100"],
                    "future_rapm_off_correlation": off_agreement["correlation"],
                    "future_rapm_def_correlation": def_agreement["correlation"],
                    "future_rapm_off_players": off_agreement["players"],
                    "future_rapm_def_players": def_agreement["players"],
                    "season_scores": json.dumps(season_scores, separators=(",", ":")),
                }
            )
        logging.info(
            "%s block %d/%d complete: train %d-%d -> future %d-%d",
            mechanism.mechanism_id,
            block_index,
            len(blocks),
            min(train_years),
            max(train_years),
            min(future_years),
            max(future_years),
        )

    detailed = pd.DataFrame(rows)
    selected_pair, selection = paired_one_se(detailed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(args.output_dir / f"{mechanism.mechanism_id}_future_blocks.csv", index=False)
    selection.to_csv(args.output_dir / f"{mechanism.mechanism_id}_selection.csv", index=False)
    payload.update(
        {
            "status": "complete",
            "selected_future_pair": list(selected_pair),
            "exact_minimum_pair": [
                int(selection.sort_values("mean_future_mse").iloc[0]["off_alpha"]),
                int(selection.sort_values("mean_future_mse").iloc[0]["def_alpha"]),
            ],
            "detail_csv": str(args.output_dir / f"{mechanism.mechanism_id}_future_blocks.csv"),
            "selection_csv": str(args.output_dir / f"{mechanism.mechanism_id}_selection.csv"),
        }
    )
    return payload


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    module = _load_rapm_module()
    if args.summarize_existing:
        print(summarize_existing(args.output_dir).to_string(index=False))
        return
    mechanisms = load_mechanisms(args.registry, args.mechanism)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    prior_results = []
    if manifest_path.exists():
        try:
            prior_results = json.loads(manifest_path.read_text()).get("results", [])
        except (json.JSONDecodeError, OSError):
            prior_results = []
    results_by_id = {row["mechanism_id"]: row for row in prior_results}
    for mechanism in mechanisms:
        logging.info("Validating %s", mechanism.mechanism_id)
        result = run_mechanism(module, mechanism, args)
        results_by_id[mechanism.mechanism_id] = result
        results = [results_by_id[key] for key in sorted(results_by_id)]
        manifest_path.write_text(
            json.dumps(
                {
                    "method": "trailing_5y_td550_to_nonoverlapping_future_2y",
                    "primary_metric": "equal-season-weight future possession MSE after future season-phase intercepts",
                    "selection": "most-shrunk tested pair within paired one standard error of exact minimum",
                    "future_data_use": "evaluation only; no future outcome enters player fit",
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
