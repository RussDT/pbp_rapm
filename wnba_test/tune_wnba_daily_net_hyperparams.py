#!/usr/bin/env python3
"""Tune multi-year WNBA net-DECOMP ridge penalties and time decay.

This is a pure, zero-centered tuning lane.  Each rolling-origin fold trains on
completed seasons only and predicts the next full season, preventing the
whole-season continuation constants in the current processed files from
leaking into an earlier same-season cutoff.  The primary score is the summed
component prediction's MSE against future actual possession points.

The structural target family keeps direct turnover value separate from the
neutral net baseline.  Optional SQ families split MID/THREE net-FG into EV and
actual-minus-EV residual children while retaining exact row closure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wnba_test.build_wnba_sq_net_targets import (
    MID_RESIDUAL_COLUMN,
    THREE_RESIDUAL_COLUMN,
)
from wnba_test.master_wnba_rapm import build_design_matrix, transform_to_home_away_format


WNBA_ROOT = ROOT / "wnba_test"
DEFAULT_PROCESSED_DIR = WNBA_ROOT / "Processed"
DEFAULT_SQ_DIR = WNBA_ROOT / "Results" / "wnba_net_sq_targets"
DEFAULT_OUTPUT_DIR = WNBA_ROOT / "Results" / "wnba_daily_net_tuning"

KEY_COLUMNS = [
    "game_id",
    "event_num",
    *[f"O{i}" for i in range(1, 6)],
    *[f"D{i}" for i in range(1, 6)],
]
SHOT_COMPONENTS = [
    ("RIM_FREQ", "FC_NET_RIM_EFG_FREQ_Diff"),
    ("RIM_FG", "FC_NET_RIM_EFG_FG_Diff"),
    ("MID_FREQ", "FC_NET_MID_EFG_FREQ_Diff"),
    ("MID_FG", "FC_NET_MID_EFG_FG_Diff"),
    ("THREE_FREQ", "FC_NET_THREE_EFG_FREQ_Diff"),
    ("THREE_FG", "FC_NET_THREE_EFG_FG_Diff"),
]


def parse_float_grid(value: str, *, allow_none: bool = False) -> list[float | None]:
    parsed: list[float | None] = []
    for raw in value.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if allow_none and item in {"none", "no", "inf"}:
            parsed.append(None)
        else:
            parsed.append(float(item))
    if not parsed:
        raise argparse.ArgumentTypeError("grid must contain at least one value")
    return parsed


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pair(fc_path: Path, sc_path: Path, year: int) -> pd.DataFrame:
    fc = pd.read_csv(fc_path)
    sc = pd.read_csv(sc_path)
    if len(fc) != len(sc):
        raise ValueError(f"row mismatch for {fc_path.name}: {len(fc)} vs {len(sc)}")
    for column in KEY_COLUMNS:
        if column not in fc.columns or column not in sc.columns:
            raise ValueError(f"missing alignment key {column} in {fc_path} / {sc_path}")
        left = fc[column].astype(str).reset_index(drop=True)
        right = sc[column].astype(str).reset_index(drop=True)
        if not left.equals(right):
            raise ValueError(f"row alignment mismatch on {column} for {fc_path.name}")
    out = fc.copy()
    out["season_end_year"] = int(year)
    out["_sc_net"] = pd.to_numeric(sc["SC_NET_CONT_Diff"], errors="coerce").fillna(0.0)
    out["_actual_points"] = (
        pd.to_numeric(fc["Net_Diff"], errors="coerce").fillna(0.0)
        + pd.to_numeric(sc["Net_Diff"], errors="coerce").fillna(0.0)
    )
    return out


def load_target_rows(
    years: Iterable[int],
    *,
    target_family: str,
    processed_dir: Path,
    sq_dir: Path,
    include_playoffs: bool,
) -> tuple[pd.DataFrame, list[Path]]:
    if target_family == "net":
        fc_dir = processed_dir
    elif target_family == "net_sq_context":
        fc_dir = sq_dir / "shot_context_ev"
    elif target_family == "net_sq_prior_shooter":
        fc_dir = sq_dir / "shot_quality_prior_ev"
    else:
        raise ValueError(f"unsupported target family: {target_family}")

    pieces: list[pd.DataFrame] = []
    sources: list[Path] = []
    for year in years:
        for suffix in ("", "_PS"):
            if suffix and not include_playoffs:
                continue
            fc_path = fc_dir / f"FIRST_CHANCE{year % 100:02d}{suffix}.csv"
            sc_path = processed_dir / f"SECOND_CHANCE_CLEAN{year % 100:02d}{suffix}.csv"
            if not fc_path.exists() or not sc_path.exists():
                continue
            pieces.append(_load_pair(fc_path, sc_path, year))
            sources.extend([fc_path, sc_path])
    if not pieces:
        raise FileNotFoundError(f"no rows found for {target_family}")
    rows = pd.concat(pieces, ignore_index=True)
    dates = pd.to_datetime(rows["game_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"{int(dates.isna().sum())} rows have missing exact game_date")
    rows["game_date"] = dates
    if int(rows["season_end_year"].max()) >= 2027:
        raise ValueError("2027 is embargoed and may not enter WNBA hyperparameter selection")
    return rows, sources


def build_component_targets(rows: pd.DataFrame, target_family: str) -> tuple[pd.DataFrame, list[str]]:
    """Construct a structural component matrix that closes to actual points."""
    out = rows.copy()
    non_turnover = pd.to_numeric(out["Is_Turnover"], errors="coerce").fillna(0).eq(0)
    if not non_turnover.any():
        raise ValueError("target rows have no non-turnover possessions")

    components: list[tuple[str, str]] = list(SHOT_COMPONENTS)
    if target_family != "net":
        components = [
            (name, column)
            for name, column in components
            if name not in {"MID_FG", "THREE_FG"}
        ]
        components.extend(
            [
                ("MID_SQ_EV", "FC_NET_MID_EFG_FG_Diff"),
                ("MID_SQ_RESID", MID_RESIDUAL_COLUMN),
                ("THREE_SQ_EV", "FC_NET_THREE_EFG_FG_Diff"),
                ("THREE_SQ_RESID", THREE_RESIDUAL_COLUMN),
            ]
        )

    target_names: list[str] = []
    first_chance_children: list[np.ndarray] = []
    is_turnover = ~non_turnover
    for name, source in components:
        if source not in out.columns:
            raise ValueError(f"{target_family} is missing target column {source}")
        values = pd.to_numeric(out[source], errors="coerce").fillna(0.0)
        baseline = float(values.loc[non_turnover].mean())
        target = np.where(is_turnover, baseline, values)
        column = f"target_{name}"
        out[column] = target
        target_names.append(column)
        first_chance_children.append(np.asarray(target, dtype=float))

    ft_values = pd.to_numeric(out["FC_FT_Diff"], errors="coerce").fillna(0.0)
    ft_baseline = float(ft_values.loc[non_turnover].mean())
    out["target_FT"] = np.where(is_turnover, ft_baseline, ft_values)
    target_names.append("target_FT")
    first_chance_children.append(out["target_FT"].to_numpy(float))

    fc_parent = pd.to_numeric(out["FC_TS_NET_Diff"], errors="coerce").fillna(0.0)
    fc_non_tov_mean = float(fc_parent.loc[non_turnover].mean())
    out["target_TOV"] = np.where(is_turnover, fc_parent - fc_non_tov_mean, 0.0)
    target_names.append("target_TOV")

    child_sum_without_base = np.sum(np.vstack(first_chance_children), axis=0)
    out["target_NET_BASE"] = (
        fc_parent.to_numpy(float)
        - child_sum_without_base
        - out["target_TOV"].to_numpy(float)
    )
    target_names.append("target_NET_BASE")

    out["target_SECOND_CHANCE"] = pd.to_numeric(
        out["_sc_net"], errors="coerce"
    ).fillna(0.0)
    target_names.append("target_SECOND_CHANCE")

    target_sum = out[target_names].sum(axis=1)
    closure = target_sum - pd.to_numeric(out["_actual_points"], errors="coerce")
    max_gap = float(closure.abs().max())
    if max_gap > 1e-9:
        raise ValueError(
            f"{target_family} structural targets do not close to actual points: {max_gap:.3g}"
        )
    return out, target_names


def _player_mapping(transformed: pd.DataFrame) -> dict[str, int]:
    players: set[str] = set()
    for column in [f"a{i}" for i in range(1, 6)] + [f"h{i}" for i in range(1, 6)]:
        players.update(transformed[column].astype(int).astype(str).unique())
    players.discard("0")
    mapping: dict[str, int] = {}
    for player in sorted(players):
        mapping[f"{player}_off"] = len(mapping)
        mapping[f"{player}_def"] = len(mapping)
    return mapping


@dataclass
class Fold:
    validation_year: int
    cutoff_date: pd.Timestamp
    train_dates: np.ndarray
    train_x: object
    train_targets: np.ndarray
    validation_x: object
    validation_actual: np.ndarray
    validation_games: np.ndarray
    off_indices: np.ndarray
    def_indices: np.ndarray
    residual_target_indices: np.ndarray
    cache: dict[float | None, tuple[np.ndarray, np.ndarray, np.ndarray]]

    def weighted_system(self, half_life: float | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if half_life not in self.cache:
            cutoff = self.cutoff_date
            if half_life is None:
                weights = np.ones(len(self.train_dates), dtype=float)
            else:
                age_days = (
                    cutoff - pd.to_datetime(self.train_dates)
                ).days.astype(float)
                weights = np.asarray(
                    np.power(0.5, age_days / float(half_life)), dtype=float
                )
            target_means = np.average(self.train_targets, axis=0, weights=weights)
            centered = self.train_targets - target_means
            weighted_x = self.train_x.multiply(np.sqrt(weights)[:, None])
            gram = (weighted_x.T @ weighted_x).toarray()
            rhs = np.asarray(self.train_x.T @ (weights[:, None] * centered))
            self.cache[half_life] = (gram, rhs, target_means)
        return self.cache[half_life]


def prepare_folds(
    rows: pd.DataFrame,
    target_family: str,
    validation_years: Iterable[int],
) -> tuple[list[Fold], pd.DataFrame, list[str]]:
    ordered = rows.sort_values(["game_id", "event_num"], kind="stable").reset_index(drop=True)
    solver_frame = ordered.copy()
    solver_frame["Off_Diff"] = solver_frame["_actual_points"]
    solver_frame["Def_Diff"] = -solver_frame["_actual_points"]
    transformed = transform_to_home_away_format(solver_frame)
    source_games = ordered["game_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    transformed_games = transformed["gameid"].astype(str).str.replace(r"\.0$", "", regex=True)
    if not source_games.reset_index(drop=True).equals(transformed_games.reset_index(drop=True)):
        raise ValueError("home/away transform changed row order; target alignment is unsafe")

    audit_rows: list[dict[str, object]] = []
    folds: list[Fold] = []
    target_names: list[str] | None = None
    for validation_year in validation_years:
        train_mask = ordered["season_end_year"].lt(validation_year).to_numpy()
        validation_mask = ordered["season_end_year"].eq(validation_year).to_numpy()
        if not train_mask.any() or not validation_mask.any():
            continue
        train_transformed = transformed.loc[train_mask].reset_index(drop=True)
        validation_transformed = transformed.loc[validation_mask].reset_index(drop=True)
        train_prepared, fold_target_names = build_component_targets(
            ordered.loc[train_mask].reset_index(drop=True),
            target_family,
        )
        if target_names is None:
            target_names = fold_target_names
        elif target_names != fold_target_names:
            raise ValueError("component target names changed across folds")
        mapping = _player_mapping(train_transformed)
        train_x, _ = build_design_matrix(train_transformed, mapping)
        validation_x, _ = build_design_matrix(validation_transformed, mapping)
        train_dates = ordered.loc[train_mask, "game_date"].to_numpy()
        validation_dates = ordered.loc[validation_mask, "game_date"]
        if pd.Timestamp(pd.to_datetime(train_dates).max()) >= validation_dates.min():
            raise ValueError(f"fold {validation_year} training overlaps validation")
        off_indices = np.asarray(
            [index for key, index in mapping.items() if key.endswith("_off")], dtype=int
        )
        def_indices = np.asarray(
            [index for key, index in mapping.items() if key.endswith("_def")], dtype=int
        )
        folds.append(
            Fold(
                validation_year=int(validation_year),
                cutoff_date=pd.Timestamp(validation_dates.min()),
                train_dates=train_dates,
                train_x=train_x,
                train_targets=train_prepared[target_names].to_numpy(float),
                validation_x=validation_x,
                validation_actual=ordered.loc[validation_mask, "_actual_points"].to_numpy(float),
                validation_games=(
                    ordered.loc[validation_mask, "game_id"]
                    .astype(str)
                    .str.replace(r"\.0$", "", regex=True)
                    .to_numpy()
                ),
                off_indices=off_indices,
                def_indices=def_indices,
                residual_target_indices=np.asarray(
                    [
                        index
                        for index, name in enumerate(target_names)
                        if name in {
                            "target_MID_SQ_RESID",
                            "target_THREE_SQ_RESID",
                        }
                    ],
                    dtype=int,
                ),
                cache={},
            )
        )
        audit_rows.append(
            {
                "validation_year": validation_year,
                "train_min_date": str(pd.to_datetime(train_dates).min().date()),
                "train_max_date": str(pd.to_datetime(train_dates).max().date()),
                "validation_min_date": str(validation_dates.min().date()),
                "validation_max_date": str(validation_dates.max().date()),
                "train_rows": int(train_mask.sum()),
                "validation_rows": int(validation_mask.sum()),
                "validation_games": int(ordered.loc[validation_mask, "game_id"].nunique()),
                "train_players": len(mapping) // 2,
            }
        )
    if not folds:
        raise ValueError("no complete rolling-origin folds were prepared")
    assert target_names is not None
    return folds, pd.DataFrame(audit_rows), target_names


def evaluate_candidate(
    folds: list[Fold],
    off_alpha: float,
    def_alpha: float,
    half_life: float | None,
    residual_off_alpha: float | None = None,
    residual_def_alpha: float | None = None,
) -> tuple[float, pd.DataFrame]:
    game_rows: list[pd.DataFrame] = []
    for fold in folds:
        gram, rhs, means = fold.weighted_system(half_life)
        penalties = np.zeros(gram.shape[0], dtype=float)
        penalties[fold.off_indices] = float(off_alpha)
        penalties[fold.def_indices] = float(def_alpha)
        coefficients = np.linalg.solve(gram + np.diag(penalties), rhs)
        if fold.residual_target_indices.size and (
            residual_off_alpha is not None or residual_def_alpha is not None
        ):
            residual_penalties = np.zeros(gram.shape[0], dtype=float)
            residual_penalties[fold.off_indices] = float(
                residual_off_alpha if residual_off_alpha is not None else off_alpha
            )
            residual_penalties[fold.def_indices] = float(
                residual_def_alpha if residual_def_alpha is not None else def_alpha
            )
            residual_columns = fold.residual_target_indices
            coefficients[:, residual_columns] = np.linalg.solve(
                gram + np.diag(residual_penalties),
                rhs[:, residual_columns],
            )
        summed_coefficients = coefficients.sum(axis=1)
        prediction = means.sum() + fold.validation_x @ summed_coefficients
        errors = np.square(fold.validation_actual - np.asarray(prediction).reshape(-1))
        game_frame = pd.DataFrame(
            {
                "validation_year": fold.validation_year,
                "game_id": fold.validation_games,
                "squared_error": errors,
            }
        )
        game_rows.append(
            game_frame.groupby(["validation_year", "game_id"], as_index=False)[
                "squared_error"
            ].mean()
        )
    by_game = pd.concat(game_rows, ignore_index=True)
    score = float(
        by_game.groupby("validation_year")["squared_error"].mean().mean()
    )
    return score, by_game


def paired_one_se(
    candidate: pd.DataFrame,
    best: pd.DataFrame,
) -> float:
    merged = candidate.merge(
        best,
        on=["validation_year", "game_id"],
        suffixes=("_candidate", "_best"),
        validate="one_to_one",
    )
    merged["difference"] = (
        merged["squared_error_candidate"] - merged["squared_error_best"]
    )
    year_variances = []
    for _, group in merged.groupby("validation_year"):
        if len(group) <= 1:
            year_variances.append(0.0)
        else:
            year_variances.append(float(group["difference"].var(ddof=1) / len(group)))
    return float(np.sqrt(np.sum(year_variances)) / len(year_variances))


def tune(
    folds: list[Fold],
    off_grid: Iterable[float],
    def_grid: Iterable[float],
    half_life_grid: Iterable[float | None],
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    scores: list[dict[str, object]] = []
    game_scores: dict[tuple[float, float, float | None], pd.DataFrame] = {}
    for half_life in half_life_grid:
        for off_alpha in off_grid:
            for def_alpha in def_grid:
                key = (float(off_alpha), float(def_alpha), half_life)
                score, by_game = evaluate_candidate(folds, *key)
                game_scores[key] = by_game
                scores.append(
                    {
                        "off_alpha": key[0],
                        "def_alpha": key[1],
                        "half_life_days": half_life,
                        "mean_game_weighted_possession_mse": score,
                    }
                )
    grid = pd.DataFrame(scores).sort_values("mean_game_weighted_possession_mse").reset_index(drop=True)
    best_row = grid.iloc[0]
    best_key = (
        float(best_row["off_alpha"]),
        float(best_row["def_alpha"]),
        None if pd.isna(best_row["half_life_days"]) else float(best_row["half_life_days"]),
    )
    best_games = game_scores[best_key]

    one_se_values = []
    for row in grid.itertuples(index=False):
        key = (
            float(row.off_alpha),
            float(row.def_alpha),
            None if pd.isna(row.half_life_days) else float(row.half_life_days),
        )
        one_se_values.append(paired_one_se(game_scores[key], best_games))
    grid["paired_one_se"] = one_se_values
    grid["within_one_se"] = (
        grid["mean_game_weighted_possession_mse"]
        <= float(best_row["mean_game_weighted_possession_mse"]) + grid["paired_one_se"]
    )
    eligible = grid.loc[grid["within_one_se"]].copy()
    eligible["shrinkage"] = np.sqrt(eligible["off_alpha"] * eligible["def_alpha"])
    eligible["half_life_preference"] = eligible["half_life_days"].fillna(np.inf)
    selected_row = eligible.sort_values(
        ["shrinkage", "half_life_preference"], ascending=[False, False]
    ).iloc[0]
    selected = {
        "off_alpha": float(selected_row["off_alpha"]),
        "def_alpha": float(selected_row["def_alpha"]),
        "half_life_days": (
            None
            if pd.isna(selected_row["half_life_days"])
            else float(selected_row["half_life_days"])
        ),
        "mean_game_weighted_possession_mse": float(
            selected_row["mean_game_weighted_possession_mse"]
        ),
        "best_unconstrained_mse": float(best_row["mean_game_weighted_possession_mse"]),
        "selection_rule": "most-shrunk then longest-history candidate within paired one-SE of best",
    }
    selected_key = (
        selected["off_alpha"], selected["def_alpha"], selected["half_life_days"]
    )
    return grid, selected, game_scores[selected_key]


def tune_sq_residual_alphas(
    folds: list[Fold],
    base_selected: dict[str, object],
    residual_off_grid: Iterable[float],
    residual_def_grid: Iterable[float],
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    """Tune only the two actual-minus-SQ residual children around a fixed base."""
    base_off = float(base_selected["off_alpha"])
    base_def = float(base_selected["def_alpha"])
    half_life = base_selected["half_life_days"]
    half_life = None if half_life is None else float(half_life)
    scores: list[dict[str, object]] = []
    game_scores: dict[tuple[float, float], pd.DataFrame] = {}
    for residual_off in residual_off_grid:
        for residual_def in residual_def_grid:
            key = (float(residual_off), float(residual_def))
            score, by_game = evaluate_candidate(
                folds,
                base_off,
                base_def,
                half_life,
                residual_off_alpha=key[0],
                residual_def_alpha=key[1],
            )
            game_scores[key] = by_game
            scores.append(
                {
                    "base_off_alpha": base_off,
                    "base_def_alpha": base_def,
                    "half_life_days": half_life,
                    "residual_off_alpha": key[0],
                    "residual_def_alpha": key[1],
                    "mean_game_weighted_possession_mse": score,
                }
            )
    grid = pd.DataFrame(scores).sort_values(
        "mean_game_weighted_possession_mse"
    ).reset_index(drop=True)
    best_row = grid.iloc[0]
    best_key = (
        float(best_row["residual_off_alpha"]),
        float(best_row["residual_def_alpha"]),
    )
    best_games = game_scores[best_key]
    grid["paired_one_se"] = [
        paired_one_se(
            game_scores[(float(row.residual_off_alpha), float(row.residual_def_alpha))],
            best_games,
        )
        for row in grid.itertuples(index=False)
    ]
    grid["within_one_se"] = (
        grid["mean_game_weighted_possession_mse"]
        <= float(best_row["mean_game_weighted_possession_mse"]) + grid["paired_one_se"]
    )
    eligible = grid.loc[grid["within_one_se"]].copy()
    eligible["residual_shrinkage"] = np.sqrt(
        eligible["residual_off_alpha"] * eligible["residual_def_alpha"]
    )
    selected_row = eligible.sort_values(
        "residual_shrinkage", ascending=False
    ).iloc[0]
    selected = {
        **base_selected,
        "residual_off_alpha": float(selected_row["residual_off_alpha"]),
        "residual_def_alpha": float(selected_row["residual_def_alpha"]),
        "mean_game_weighted_possession_mse": float(
            selected_row["mean_game_weighted_possession_mse"]
        ),
        "best_unconstrained_grouped_mse": float(
            best_row["mean_game_weighted_possession_mse"]
        ),
        "gain_vs_common_alpha_x1e6": (
            float(base_selected["mean_game_weighted_possession_mse"])
            - float(selected_row["mean_game_weighted_possession_mse"])
        )
        * 1e6,
        "selection_rule": (
            "base alpha/decay selected first; then most-shrunk SQ residual "
            "penalties within paired one-SE of the grouped optimum"
        ),
    }
    selected_key = (
        selected["residual_off_alpha"], selected["residual_def_alpha"]
    )
    return grid, selected, game_scores[selected_key]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-family",
        choices=["net", "net_sq_context", "net_sq_prior_shooter"],
        default="net",
    )
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--sq-dir", type=Path, default=DEFAULT_SQ_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--exclude-playoffs", action="store_true")
    parser.add_argument("--start-year", type=int, default=2022)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--first-validation-year", type=int, default=2024)
    parser.add_argument("--off-grid", default="1000,2000,5000,10000,20000")
    parser.add_argument("--def-grid", default="2000,5000,10000,20000,40000")
    parser.add_argument("--half-life-grid", default="365,550,730,1100,none")
    parser.add_argument(
        "--residual-off-grid", default="2000,5000,10000,20000,40000,80000"
    )
    parser.add_argument(
        "--residual-def-grid", default="5000,10000,20000,40000,80000,160000"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    years = list(range(args.start_year, args.end_year + 1))
    rows, sources = load_target_rows(
        years,
        target_family=args.target_family,
        processed_dir=args.processed_dir,
        sq_dir=args.sq_dir,
        include_playoffs=not args.exclude_playoffs,
    )
    validation_years = list(range(args.first_validation_year, args.end_year + 1))
    folds, fold_audit, target_names = prepare_folds(
        rows, args.target_family, validation_years
    )
    off_grid = [float(value) for value in parse_float_grid(args.off_grid)]
    def_grid = [float(value) for value in parse_float_grid(args.def_grid)]
    half_life_grid = parse_float_grid(args.half_life_grid, allow_none=True)
    grid, selected, selected_game_scores = tune(
        folds, off_grid, def_grid, half_life_grid
    )

    residual_grid = None
    if args.target_family != "net":
        residual_off_grid = [
            float(value) for value in parse_float_grid(args.residual_off_grid)
        ]
        residual_def_grid = [
            float(value) for value in parse_float_grid(args.residual_def_grid)
        ]
        residual_grid, selected, selected_game_scores = tune_sq_residual_alphas(
            folds,
            selected,
            residual_off_grid,
            residual_def_grid,
        )

    output_dir = args.output_dir / args.target_family
    output_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(output_dir / "grid_scores.csv", index=False)
    if residual_grid is not None:
        residual_grid.to_csv(output_dir / "residual_grid_scores.csv", index=False)
    fold_audit.to_csv(output_dir / "fold_audit.csv", index=False)
    selected_game_scores.to_csv(output_dir / "selected_game_scores.csv", index=False)
    source_hashes = {str(path): _file_hash(path) for path in sorted(set(sources))}
    report = {
        "status": "development_only",
        "target_family": args.target_family,
        "pure_zero_centered": True,
        "prior_centered": False,
        "selection_uses_2026": 2026 in validation_years,
        "next_untouched_confirmation": 2027,
        "warning": (
            "2026 is spent evidence. Whole-season kappa inputs make these "
            "completed-season rolling folds safe, but they must not be reused "
            "for within-season historical daily cuts without causal kappa rebuilds."
        ),
        "years": years,
        "validation_years": validation_years,
        "include_playoffs": not args.exclude_playoffs,
        "target_components": target_names,
        "selected": selected,
        "grid": {
            "off_alpha": off_grid,
            "def_alpha": def_grid,
            "half_life_days": half_life_grid,
            "residual_off_alpha": (
                [float(value) for value in parse_float_grid(args.residual_off_grid)]
                if args.target_family != "net"
                else None
            ),
            "residual_def_alpha": (
                [float(value) for value in parse_float_grid(args.residual_def_grid)]
                if args.target_family != "net"
                else None
            ),
        },
        "source_hashes": source_hashes,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output_dir / "frozen_candidate.json").write_text(
        json.dumps(
            {
                "status": "development_candidate_not_promoted",
                "target_family": args.target_family,
                **selected,
                "source_report": str(output_dir / "report.json"),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(report, indent=2))
    print(grid.head(15).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
