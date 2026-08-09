#!/usr/bin/env python3
"""Fit joint oTS RAPM with a player-specific centered shooter interaction.

The model uses one row per resolved TS attempt:

    y = season baseline
        + five offensive player effects
        + five defensive player effects
        + sum_i (is_shooter_i - player_season_shot_share_i) * role_i

The centered interaction makes each offensive coefficient the player's
usage-averaged oTS effect.  The role coefficient is the self-shot versus
teammate-shot contrast, so the two conditional rates reconcile exactly:

    self = parent + (1 - usage) * role
    teammate = parent - usage * role
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

try:
    from .run_assist_points_hierarchy import (
        PROCESSED_DIR,
        RESULTS_DIR,
        build_player_name_map,
        exposure_counts,
        role_paths,
        season_suffix,
        weighted_center,
    )
    from .run_ts_shot_role_hierarchy import find_ts_parent
except ImportError:
    from run_assist_points_hierarchy import (
        PROCESSED_DIR,
        RESULTS_DIR,
        build_player_name_map,
        exposure_counts,
        role_paths,
        season_suffix,
        weighted_center,
    )
    from run_ts_shot_role_hierarchy import find_ts_parent


OFFENSE_COLS = ["Focal_Player", *[f"O{i}" for i in range(1, 5)]]
DEFENSE_COLS = [f"D{i}" for i in range(1, 6)]
DEFAULT_ROLE_ALPHA_GRID = [
    10.0,
    30.0,
    50.0,
    100.0,
    200.0,
    300.0,
    500.0,
    700.0,
    1000.0,
    3000.0,
    10000.0,
    1.0e12,
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def read_joint_inputs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            logging.warning("Missing input %s", path)
            continue
        frame = pd.read_parquet(path)
        frame["source_file"] = path.name
        frames.append(frame)
        logging.info("Loaded %s (%d rows)", path.name, len(frame))
    if not frames:
        raise FileNotFoundError("No TS_SELF_SHOT parquets found.")

    df = pd.concat(frames, ignore_index=True)
    required = [
        "game_id",
        "Focal_Player",
        "TS_Points",
        "Season",
        *OFFENSE_COLS[1:],
        *DEFENSE_COLS,
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required joint-oTS columns: {missing}")
    for column in [*OFFENSE_COLS, *DEFENSE_COLS, "TS_Points", "Season"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    before = len(df)
    df = df.dropna(subset=required).copy()
    for column in [*OFFENSE_COLS, *DEFENSE_COLS, "Season"]:
        df[column] = df[column].astype(np.int64)
    logging.info("Kept %d complete attempt rows (removed %d)", len(df), before - len(df))
    return df


def compute_player_season_shot_shares(df: pd.DataFrame) -> pd.DataFrame:
    """Return each player's observed share of team TS attempts by season."""
    numerator = df.groupby(["Season", "Focal_Player"], sort=False).size()
    numerator.index = numerator.index.set_names(["Season", "player_id"])
    denominator = None
    for column in OFFENSE_COLS:
        counts = df.groupby(["Season", column], sort=False).size()
        counts.index = counts.index.set_names(["Season", "player_id"])
        denominator = counts if denominator is None else denominator.add(
            counts, fill_value=0
        )
    shares = (
        pd.DataFrame(
            {
                "self_attempts": numerator,
                "offensive_attempt_exposures": denominator,
            }
        )
        .fillna(0)
        .reset_index()
    )
    shares["shot_share"] = (
        shares["self_attempts"] / shares["offensive_attempt_exposures"]
    )
    if not shares["shot_share"].between(0, 1).all():
        raise ValueError("Computed an invalid player-season shot share.")
    return shares


def collect_player_ids(df: pd.DataFrame) -> np.ndarray:
    values = pd.to_numeric(
        pd.Series(df[[*OFFENSE_COLS, *DEFENSE_COLS]].to_numpy().ravel()),
        errors="coerce",
    ).dropna().astype(np.int64)
    return np.sort(values[values > 0].unique())


def _append_feature_block(
    row_indices: list[np.ndarray],
    col_indices: list[np.ndarray],
    data: list[np.ndarray],
    players: np.ndarray,
    feature_values: np.ndarray,
    player_ids: np.ndarray,
    block_offset: int,
) -> None:
    flat_players = players.reshape(-1).astype(np.int64)
    flat_values = feature_values.reshape(-1).astype(np.float64)
    rows = np.repeat(
        np.arange(players.shape[0], dtype=np.int64), players.shape[1]
    )
    valid = (flat_players > 0) & np.isfinite(flat_values)
    positions = np.searchsorted(player_ids, flat_players[valid])
    matched = (
        (positions < len(player_ids))
        & (player_ids[positions] == flat_players[valid])
    )
    row_indices.append(rows[valid][matched])
    col_indices.append(block_offset + positions[matched])
    data.append(flat_values[valid][matched])


def build_joint_design_matrix(
    df: pd.DataFrame,
    player_ids: np.ndarray,
    shot_shares: pd.DataFrame,
    off_alpha: float,
    def_alpha: float,
    role_alpha: float,
):
    """Build offense, defense, and centered shooter-interaction blocks."""
    share_lookup = shot_shares.set_index(["Season", "player_id"])[
        "shot_share"
    ]
    offense = df[OFFENSE_COLS].to_numpy(dtype=np.int64)
    defense = df[DEFENSE_COLS].to_numpy(dtype=np.int64)
    seasons = df["Season"].to_numpy(dtype=np.int64)

    interaction = np.empty_like(offense, dtype=np.float64)
    for idx, column in enumerate(OFFENSE_COLS):
        keys = pd.MultiIndex.from_arrays(
            [seasons, offense[:, idx]], names=["Season", "player_id"]
        )
        usage = share_lookup.reindex(keys).to_numpy(dtype=np.float64)
        if np.isnan(usage).any():
            raise ValueError(f"Missing shot-share values while mapping {column}.")
        interaction[:, idx] = -usage
    interaction[:, 0] += 1.0

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    values: list[np.ndarray] = []
    n_players = len(player_ids)
    _append_feature_block(
        rows,
        cols,
        values,
        offense,
        np.full_like(offense, 1.0 / np.sqrt(off_alpha), dtype=np.float64),
        player_ids,
        0,
    )
    _append_feature_block(
        rows,
        cols,
        values,
        defense,
        np.full_like(defense, 1.0 / np.sqrt(def_alpha), dtype=np.float64),
        player_ids,
        n_players,
    )
    _append_feature_block(
        rows,
        cols,
        values,
        offense,
        interaction / np.sqrt(role_alpha),
        player_ids,
        2 * n_players,
    )
    matrix = coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(df), 3 * n_players),
        dtype=np.float64,
    ).tocsr()
    return matrix, interaction


def season_centered_target(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    baselines = (
        df.groupby("Season", as_index=False)["TS_Points"]
        .mean()
        .rename(columns={"TS_Points": "baseline_per_attempt"})
        .sort_values("Season")
    )
    lookup = baselines.set_index("Season")["baseline_per_attempt"]
    centered = (
        df["TS_Points"].to_numpy(dtype=np.float64)
        - df["Season"].map(lookup).to_numpy(dtype=np.float64)
    )
    return centered, baselines


def fit_coefficients(X, y: np.ndarray) -> np.ndarray:
    model = Ridge(alpha=1.0, fit_intercept=False, solver="lsqr")
    model.fit(X, y)
    return model.coef_.astype(np.float64)


def parse_alpha_grid(value: str | None) -> list[float]:
    if value is None:
        return list(DEFAULT_ROLE_ALPHA_GRID)
    parsed = sorted(
        {
            float(part.strip())
            for part in value.split(",")
            if part.strip()
        }
    )
    if not parsed or any(alpha <= 0 for alpha in parsed):
        raise ValueError("Role alpha grid must contain positive values.")
    return parsed


def cross_validate_role_alpha(
    df: pd.DataFrame,
    y: np.ndarray,
    player_ids: np.ndarray,
    shot_shares: pd.DataFrame,
    off_alpha: float,
    def_alpha: float,
    candidates: list[float],
    folds: int,
) -> pd.DataFrame:
    """Choose the role penalty with held-out games."""
    groups = df["game_id"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=int(folds))
    splits = list(splitter.split(np.zeros(len(df)), groups=groups))
    rows = []
    for role_alpha in candidates:
        logging.info("Cross-validating shooter-role alpha %.0f", role_alpha)
        X, _ = build_joint_design_matrix(
            df,
            player_ids,
            shot_shares,
            off_alpha,
            def_alpha,
            role_alpha,
        )
        fold_mse = []
        for fold_idx, (train_idx, valid_idx) in enumerate(splits, start=1):
            beta = fit_coefficients(X[train_idx], y[train_idx])
            prediction = X[valid_idx] @ beta
            mse = float(np.mean((y[valid_idx] - prediction) ** 2))
            fold_mse.append(mse)
            logging.info(
                "  alpha %.0f fold %d/%d MSE %.9f",
                role_alpha,
                fold_idx,
                len(splits),
                mse,
            )
        rows.append(
            {
                "role_alpha": role_alpha,
                "mean_mse": float(np.mean(fold_mse)),
                "std_mse": float(np.std(fold_mse)),
                **{
                    f"fold_{idx + 1}_mse": value
                    for idx, value in enumerate(fold_mse)
                },
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mean_mse", "role_alpha"], ascending=[True, False]
    )


def select_role_alpha_one_se(
    cv_results: pd.DataFrame,
    folds: int,
) -> tuple[float, pd.DataFrame]:
    """Prefer the strongest role shrinkage within one SE of minimum CV error."""
    if cv_results.empty:
        raise ValueError("Cannot select a role alpha from empty CV results.")
    result = cv_results.copy()
    best = result.sort_values(
        ["mean_mse", "role_alpha"], ascending=[True, False]
    ).iloc[0]
    best_se = float(best["std_mse"]) / np.sqrt(int(folds))
    threshold = float(best["mean_mse"]) + best_se
    result["minimum_mean_mse"] = float(best["mean_mse"])
    result["minimum_standard_error"] = best_se
    result["one_se_threshold"] = threshold
    result["within_one_se"] = result["mean_mse"].le(threshold)
    selected = float(result.loc[result["within_one_se"], "role_alpha"].max())
    result["selected_role_alpha"] = selected
    return selected, result


def aggregate_player_usage(
    df: pd.DataFrame,
    player_ids: np.ndarray,
) -> tuple[dict[int, int], dict[int, int], dict[int, float]]:
    self_counts = exposure_counts(df, ["Focal_Player"])
    total_counts = exposure_counts(df, OFFENSE_COLS)
    shares = {
        int(player_id): (
            self_counts.get(int(player_id), 0)
            / total_counts.get(int(player_id), 1)
        )
        for player_id in player_ids
    }
    return self_counts, total_counts, shares


def build_results(
    df: pd.DataFrame,
    player_ids: np.ndarray,
    beta: np.ndarray,
    off_alpha: float,
    def_alpha: float,
    role_alpha: float,
) -> tuple[pd.DataFrame, dict]:
    n_players = len(player_ids)
    offense = beta[:n_players] / np.sqrt(off_alpha)
    defense = beta[n_players : 2 * n_players] / np.sqrt(def_alpha)
    role = beta[2 * n_players :] / np.sqrt(role_alpha)

    off_counts = exposure_counts(df, OFFENSE_COLS)
    def_counts = exposure_counts(df, DEFENSE_COLS)
    off_center = weighted_center(offense, player_ids, off_counts)
    def_center = weighted_center(defense, player_ids, def_counts)
    self_counts, total_counts, usage = aggregate_player_usage(df, player_ids)
    names = build_player_name_map(set(int(value) for value in player_ids))

    rows = []
    for idx, player_id_value in enumerate(player_ids):
        player_id = int(player_id_value)
        parent = float(offense[idx] * 100.0)
        contrast = float(role[idx] * 100.0)
        shot_share = float(usage[player_id])
        conditional_self = parent + (1.0 - shot_share) * contrast
        conditional_teammate = parent - shot_share * contrast
        self_contribution = shot_share * conditional_self
        teammate_contribution = (1.0 - shot_share) * conditional_teammate
        rows.append(
            {
                "player_id": player_id,
                "player_name": names.get(player_id, f"ID_{player_id}"),
                "joint_ots_rapm": parent,
                "shooter_role_contrast_rapm": contrast,
                "conditional_self_shot_ots": conditional_self,
                "conditional_teammate_shot_ots": conditional_teammate,
                "self_shot_ots_contribution": self_contribution,
                "teammate_shot_ots_contribution": teammate_contribution,
                "shot_share": shot_share,
                "self_shot_attempts": int(self_counts.get(player_id, 0)),
                "teammate_shot_attempts": int(
                    total_counts.get(player_id, 0)
                    - self_counts.get(player_id, 0)
                ),
                "def_allowed_ots_rapm": float(defense[idx] * 100.0),
                "joint_ots_decomp_gap": (
                    self_contribution + teammate_contribution - parent
                ),
            }
        )
    result = pd.DataFrame(rows).sort_values(
        "joint_ots_rapm", ascending=False
    )
    metadata = {
        "offense_centering_offset": float(off_center),
        "defense_centering_offset": float(def_center),
        "players": len(player_ids),
        "rows": len(df),
    }
    return result, metadata


def merge_existing_parent(
    result: pd.DataFrame,
    parent_path: Path | None,
) -> pd.DataFrame:
    if parent_path is None:
        return result
    parent = pd.read_csv(parent_path)
    if not {"player_id", "off"}.issubset(parent.columns):
        raise ValueError(f"Existing TS result {parent_path} lacks player_id/off.")
    parent = parent[["player_id", "off"]].copy()
    parent["player_id"] = pd.to_numeric(
        parent["player_id"], errors="coerce"
    ).astype("Int64")
    parent = parent.rename(columns={"off": "existing_ots_rapm"})
    merged = result.merge(parent, on="player_id", how="left", validate="one_to_one")
    merged["joint_minus_existing_ots"] = (
        merged["joint_ots_rapm"] - merged["existing_ots_rapm"]
    )
    return merged


def run(args: argparse.Namespace) -> Path:
    season_type = args.season_type.upper()
    paths = role_paths(
        "TS_SELF_SHOT", args.start_year, args.end_year, season_type
    )
    df = read_joint_inputs(paths)
    shot_shares = compute_player_season_shot_shares(df)
    player_ids = collect_player_ids(df)
    y, baselines = season_centered_target(df)

    if args.role_alpha is not None:
        selected_role_alpha = float(args.role_alpha)
        cv_results = None
    else:
        candidates = parse_alpha_grid(args.role_alpha_grid)
        cv_results = cross_validate_role_alpha(
            df,
            y,
            player_ids,
            shot_shares,
            args.off_alpha,
            args.def_alpha,
            candidates,
            args.cv_folds,
        )
        selected_role_alpha, cv_results = select_role_alpha_one_se(
            cv_results, args.cv_folds
        )
        logging.info(
            "Selected shooter-role alpha %.0f with the one-SE rule",
            selected_role_alpha,
        )

    X, interaction = build_joint_design_matrix(
        df,
        player_ids,
        shot_shares,
        args.off_alpha,
        args.def_alpha,
        selected_role_alpha,
    )
    # Every player-season interaction must sum to zero over that player's
    # offensive attempt exposures.
    interaction_audits = []
    for idx, column in enumerate(OFFENSE_COLS):
        interaction_audits.append(
            pd.DataFrame(
                {
                    "Season": df["Season"].to_numpy(),
                    "player_id": df[column].to_numpy(),
                    "interaction": interaction[:, idx],
                }
            )
        )
    interaction_audit = pd.concat(interaction_audits, ignore_index=True)
    grouped = interaction_audit.groupby(
        ["Season", "player_id"]
    )["interaction"].sum()
    max_interaction_gap = float(grouped.abs().max())
    if max_interaction_gap > 1e-8:
        raise ValueError(
            f"Shooter interactions do not center exactly: {max_interaction_gap}"
        )

    beta = fit_coefficients(X, y)
    result, metadata = build_results(
        df,
        player_ids,
        beta,
        args.off_alpha,
        args.def_alpha,
        selected_role_alpha,
    )
    parent_path = find_ts_parent(
        args.start_year,
        args.end_year,
        season_type,
        args.parent_results,
    )
    result = merge_existing_parent(result, parent_path)

    suffix = (
        f"{season_suffix(args.start_year)}_{season_suffix(args.end_year)}_"
        f"{season_type.lower()}_a{int(args.off_alpha)}_{int(args.def_alpha)}_"
        f"role{int(selected_role_alpha)}"
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"joint_ts_shooter_{suffix}_results.csv"
    result.to_csv(output_path, index=False)
    baseline_path = RESULTS_DIR / f"joint_ts_shooter_{suffix}_season_baselines.csv"
    baselines.to_csv(baseline_path, index=False)
    if cv_results is not None:
        cv_path = RESULTS_DIR / f"joint_ts_shooter_{suffix}_role_alpha_cv.csv"
        cv_results.to_csv(cv_path, index=False)
        logging.info("Wrote %s", cv_path)

    logging.info("Wrote %s", output_path)
    logging.info("Wrote %s", baseline_path)
    logging.info("Model metadata: %s", metadata)
    logging.info(
        "Maximum centered-interaction sum gap: %.12g", max_interaction_gap
    )
    logging.info(
        "Maximum player decomposition gap: %.12g",
        result["joint_ots_decomp_gap"].abs().max(),
    )
    display_cols = [
        "player_name",
        "joint_ots_rapm",
        "conditional_self_shot_ots",
        "conditional_teammate_shot_ots",
        "self_shot_ots_contribution",
        "teammate_shot_ots_contribution",
        "shot_share",
    ]
    print("\nTop joint oTS RAPM")
    print(result.head(args.display)[display_cols].to_string(index=False))
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit joint oTS RAPM with centered shooter interactions."
    )
    parser.add_argument("start_year", type=int)
    parser.add_argument("end_year", type=int)
    parser.add_argument(
        "season_type", choices=["RS", "PS", "ALL", "rs", "ps", "all"]
    )
    parser.add_argument("--off-alpha", type=float, default=3000.0)
    parser.add_argument("--def-alpha", type=float, default=5000.0)
    parser.add_argument(
        "--role-alpha",
        type=float,
        help="Fixed shooter-role alpha. Omit to select it by grouped CV.",
    )
    parser.add_argument(
        "--role-alpha-grid",
        help="Comma-separated role-alpha candidates for grouped CV.",
    )
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--parent-results")
    parser.add_argument("--display", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
