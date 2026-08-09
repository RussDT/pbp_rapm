#!/usr/bin/env python3
"""Decompose offensive ASSIST_POINTS RAPM into passer and receiver components.

The role parquets contain five focal-player opportunity rows per standard team
possession. A three-block ridge model estimates the passer-minus-receiver
contrast:

    focal role contrast + four offensive teammates + five defenders

The standard ASSIST_POINTS offensive RAPM is the parent total. The two children
are recovered from the sum and contrast equations:

    passer + receiver = parent ASSIST_POINTS RAPM
    passer - receiver = fitted role contrast

This makes the decomposition close exactly for every player.
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.linear_model import Ridge


SCRIPT_DIR = Path(__file__).parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_ROOT.parent
PROCESSED_DIR = PIPELINE_ROOT / "processed"
RESULTS_DIR = PIPELINE_ROOT / "results"
RAW_DATA_DIR = PIPELINE_ROOT / "raw_data"
CONTEXT_COLS = [f"O{i}" for i in range(1, 5)]
DEFENSE_COLS = [f"D{i}" for i in range(1, 6)]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def normalize_end_year(year: int) -> int:
    year = int(year)
    if year < 100:
        return 1900 + year if year >= 97 else 2000 + year
    return year


def season_suffix(year: int) -> str:
    return f"{normalize_end_year(year) % 100:02d}"


def expand_year_window(start_year: int, end_year: int) -> list[int]:
    start_full = normalize_end_year(start_year)
    end_full = normalize_end_year(end_year)
    if end_full < start_full:
        end_full += 100
    return list(range(start_full, end_full + 1))


def role_paths(prefix: str, start_year: int, end_year: int, season_type: str) -> list[Path]:
    paths = []
    for year in expand_year_window(start_year, end_year):
        suffix = season_suffix(year)
        if season_type in {"RS", "ALL"}:
            paths.append(PROCESSED_DIR / f"{prefix}{suffix}.parquet")
        if season_type in {"PS", "ALL"}:
            paths.append(PROCESSED_DIR / f"{prefix}{suffix}_PS.parquet")
    return paths


def read_role_inputs(paths: list[Path], target_col: str) -> pd.DataFrame:
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
        raise FileNotFoundError(f"No role parquets found for {target_col}.")
    df = pd.concat(frames, ignore_index=True)
    required = [
        "game_id",
        "poss_group",
        "Focal_Player",
        target_col,
        "Role_Resolution_Complete",
        *CONTEXT_COLS,
        *DEFENSE_COLS,
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {target_col}: {missing}")
    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    before = len(df)
    df = df.dropna(subset=required).copy()
    logging.info("Kept %d complete rows (removed %d)", len(df), before - len(df))
    return df


def build_role_contrast_input(
    receiver_df: pd.DataFrame,
    passer_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a common resolved sample with passer-minus-receiver as the target."""
    keys = ["game_id", "poss_group", "Focal_Player"]
    context = [*CONTEXT_COLS, *DEFENSE_COLS]
    receiver_keep = [
        *keys,
        *context,
        "Receiver_Assist_Points",
        "Role_Resolution_Complete",
    ]
    passer_keep = [
        *keys,
        "Passer_Assist_Points",
        "Role_Resolution_Complete",
    ]
    merged = receiver_df[receiver_keep].merge(
        passer_df[passer_keep],
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_receiver", "_passer"),
    )
    complete = (
        merged["Role_Resolution_Complete_receiver"].astype(bool)
        & merged["Role_Resolution_Complete_passer"].astype(bool)
    )
    merged = merged.loc[complete].copy()
    merged["Passer_Receiver_Contrast"] = (
        pd.to_numeric(merged["Passer_Assist_Points"], errors="coerce")
        - pd.to_numeric(merged["Receiver_Assist_Points"], errors="coerce")
    )
    return merged


def collect_player_ids(df: pd.DataFrame) -> np.ndarray:
    values = pd.to_numeric(
        pd.Series(df[["Focal_Player", *CONTEXT_COLS, *DEFENSE_COLS]].to_numpy().ravel()),
        errors="coerce",
    ).dropna().astype(np.int64)
    return np.sort(values[values > 0].unique())


def add_feature_block(
    row_indices: list[np.ndarray],
    col_indices: list[np.ndarray],
    data: list[np.ndarray],
    values: np.ndarray,
    player_ids: np.ndarray,
    block_offset: int,
    feature_scale: float,
) -> None:
    flat = values.reshape(-1).astype(np.int64)
    rows = np.repeat(np.arange(values.shape[0], dtype=np.int64), values.shape[1])
    valid = flat > 0
    positions = np.searchsorted(player_ids, flat[valid])
    matched = (positions < len(player_ids)) & (player_ids[positions] == flat[valid])
    row_indices.append(rows[valid][matched])
    col_indices.append(block_offset + positions[matched])
    data.append(np.full(int(matched.sum()), feature_scale, dtype=np.float64))


def build_design_matrix(
    df: pd.DataFrame,
    player_ids: np.ndarray,
    role_alpha: float,
    teammate_alpha: float,
    defense_alpha: float,
):
    n_players = len(player_ids)
    row_indices: list[np.ndarray] = []
    col_indices: list[np.ndarray] = []
    data: list[np.ndarray] = []
    add_feature_block(
        row_indices,
        col_indices,
        data,
        df[["Focal_Player"]].to_numpy(),
        player_ids,
        0,
        1.0 / np.sqrt(role_alpha),
    )
    add_feature_block(
        row_indices,
        col_indices,
        data,
        df[CONTEXT_COLS].to_numpy(),
        player_ids,
        n_players,
        1.0 / np.sqrt(teammate_alpha),
    )
    add_feature_block(
        row_indices,
        col_indices,
        data,
        df[DEFENSE_COLS].to_numpy(),
        player_ids,
        2 * n_players,
        1.0 / np.sqrt(defense_alpha),
    )
    return coo_matrix(
        (
            np.concatenate(data),
            (np.concatenate(row_indices), np.concatenate(col_indices)),
        ),
        shape=(len(df), 3 * n_players),
        dtype=np.float64,
    ).tocsr()


def exposure_counts(df: pd.DataFrame, columns: list[str]) -> dict[int, int]:
    values = pd.to_numeric(
        pd.Series(df[columns].to_numpy().ravel()), errors="coerce"
    ).dropna().astype(np.int64)
    return values[values > 0].value_counts().astype(int).to_dict()


def weighted_center(block: np.ndarray, player_ids: np.ndarray, counts: dict[int, int]) -> float:
    weights = np.array([counts.get(int(player_id), 0) for player_id in player_ids], dtype=float)
    if not weights.sum():
        return 0.0
    offset = float(np.average(block, weights=weights))
    block -= offset
    return offset


def build_player_name_map(player_ids: set[int]) -> dict[int, str]:
    names: dict[int, str] = {}

    def store(player_id, player_name):
        try:
            player_id = int(float(player_id))
        except (TypeError, ValueError):
            return
        if player_id not in player_ids or pd.isna(player_name):
            return
        player_name = str(player_name).strip()
        if player_name and (
            player_id not in names
            or (len(player_name.split()), len(player_name))
            > (len(names[player_id].split()), len(names[player_id]))
        ):
            names[player_id] = player_name

    autocomplete = PROJECT_ROOT / "autocomplete_map.csv"
    if autocomplete.exists():
        frame = pd.read_csv(autocomplete)
        if {"nba_id", "player_name"}.issubset(frame.columns):
            for row in frame[["nba_id", "player_name"]].itertuples(index=False):
                store(*row)

    unresolved = player_ids - set(names)
    if unresolved:
        for raw_path in sorted(RAW_DATA_DIR.glob("NBA*.parquet"), reverse=True):
            try:
                frame = pd.read_parquet(raw_path, columns=["player1_id", "player1_name"])
            except Exception:
                continue
            for row in frame[["player1_id", "player1_name"]].dropna().itertuples(index=False):
                store(*row)
            unresolved = player_ids - set(names)
            if not unresolved:
                break
    return names


def fit_role_contrast_model(
    df: pd.DataFrame,
    role_alpha: float,
    teammate_alpha: float,
    defense_alpha: float,
    target_col: str = "Passer_Receiver_Contrast",
    output_col: str = "passer_minus_receiver_rapm",
) -> tuple[pd.DataFrame, dict]:
    player_ids = collect_player_ids(df)
    X = build_design_matrix(
        df, player_ids, role_alpha, teammate_alpha, defense_alpha
    )
    y = df[target_col].to_numpy(dtype=np.float64)
    baseline = float(y.mean())
    model = Ridge(alpha=1.0, fit_intercept=False, solver="lsqr")
    model.fit(X, y - baseline)

    n_players = len(player_ids)
    scaled_beta = model.coef_.astype(np.float64)
    role_beta = scaled_beta[:n_players] / np.sqrt(role_alpha)
    teammate_beta = scaled_beta[n_players : 2 * n_players] / np.sqrt(teammate_alpha)
    defense_beta = scaled_beta[2 * n_players :] / np.sqrt(defense_alpha)

    role_counts = exposure_counts(df, ["Focal_Player"])
    teammate_counts = exposure_counts(df, CONTEXT_COLS)
    defense_counts = exposure_counts(df, DEFENSE_COLS)
    offsets = {
        "role": weighted_center(role_beta, player_ids, role_counts),
        "teammate": weighted_center(teammate_beta, player_ids, teammate_counts),
        "defense": weighted_center(defense_beta, player_ids, defense_counts),
    }

    points_by_player = (
        df.groupby(df["Focal_Player"].astype(np.int64))[target_col].sum().to_dict()
    )
    names = build_player_name_map(set(int(value) for value in player_ids))
    rows = []
    for idx, player_id in enumerate(player_ids):
        player_id = int(player_id)
        focal_poss = int(role_counts.get(player_id, 0))
        raw_points = float(points_by_player.get(player_id, 0.0))
        rows.append(
            {
                "player_id": player_id,
                "player_name": names.get(player_id, f"ID_{player_id}"),
                output_col: role_beta[idx] * 100.0,
                "teammate_context_diagnostic": teammate_beta[idx] * 100.0,
                "def_allowed_rapm": defense_beta[idx] * 500.0,
                "def_suppression_rapm": -defense_beta[idx] * 500.0,
                "focal_poss": focal_poss,
                "raw_passer_minus_receiver_points": raw_points,
                "raw_passer_minus_receiver_per_100": (
                    100.0 * raw_points / focal_poss if focal_poss else np.nan
                ),
                "teammate_candidate_rows": int(teammate_counts.get(player_id, 0)),
                "def_candidate_rows": int(defense_counts.get(player_id, 0)),
            }
        )
    result = pd.DataFrame(rows).sort_values(
        [output_col, "focal_poss"], ascending=False
    )
    metadata = {
        "target": target_col,
        "rows": len(df),
        "players": len(player_ids),
        "candidate_baseline_per_100": baseline * 100.0,
        "team_baseline_per_100": baseline * 500.0,
        "role_alpha": role_alpha,
        "teammate_alpha": teammate_alpha,
        "defense_alpha": defense_alpha,
        "centering_offsets": offsets,
    }
    return result, metadata


def reconcile_role_components(hierarchy: pd.DataFrame) -> pd.DataFrame:
    """Anchor the role contrast to the parent and enforce exact child closure."""
    required = {"assist_points_off_rapm", "passer_minus_receiver_rapm"}
    missing = required - set(hierarchy.columns)
    if missing:
        raise ValueError(f"Cannot reconcile assist-point roles; missing {sorted(missing)}")
    result = hierarchy.copy()
    total = pd.to_numeric(result["assist_points_off_rapm"], errors="coerce")
    contrast = pd.to_numeric(
        result["passer_minus_receiver_rapm"], errors="coerce"
    )
    result["passer_assist_points_rapm"] = (total + contrast) / 2.0
    result["receiver_assist_points_rapm"] = (total - contrast) / 2.0
    result["assist_points_decomp_gap"] = (
        result["passer_assist_points_rapm"]
        + result["receiver_assist_points_rapm"]
        - total
    )
    return result


def find_parent_results(
    start_year: int,
    end_year: int,
    season_type: str,
    parent_results: str | None,
) -> Path | None:
    if parent_results:
        path = Path(parent_results)
        return path if path.exists() else None
    pattern = (
        f"assist_points_{season_suffix(start_year)}_{season_suffix(end_year)}_"
        f"{season_type.lower()}*_results.csv"
    )
    matches = sorted(RESULTS_DIR.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    if not matches and normalize_end_year(start_year) == normalize_end_year(end_year):
        pattern = (
            f"assist_points_{season_suffix(start_year)}_"
            f"{season_type.lower()}*_results.csv"
        )
        matches = sorted(
            RESULTS_DIR.glob(pattern),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    return matches[0] if matches else None


def run(args: argparse.Namespace) -> Path:
    season_type = args.season_type.upper()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = (
        f"{season_suffix(args.start_year)}_{season_suffix(args.end_year)}_"
        f"{season_type.lower()}_a{int(args.role_alpha)}_"
        f"{int(args.teammate_alpha)}_{int(args.defense_alpha)}"
    )

    receiver_df = read_role_inputs(
        role_paths("ASSIST_POINTS_RECEIVER", args.start_year, args.end_year, season_type),
        "Receiver_Assist_Points",
    )
    passer_df = read_role_inputs(
        role_paths("ASSIST_POINTS_PASSER", args.start_year, args.end_year, season_type),
        "Passer_Assist_Points",
    )
    contrast_df = build_role_contrast_input(receiver_df, passer_df)
    logging.info(
        "Using %d common fully resolved focal rows (%.2f%% of receiver rows)",
        len(contrast_df),
        100.0 * len(contrast_df) / len(receiver_df),
    )
    hierarchy, contrast_meta = fit_role_contrast_model(
        contrast_df,
        args.role_alpha,
        args.teammate_alpha,
        args.defense_alpha,
    )

    parent_path = find_parent_results(
        args.start_year, args.end_year, season_type, args.parent_results
    )
    if parent_path is None:
        raise FileNotFoundError(
            "No parent ASSIST_POINTS result found. Run rapm.py ASSIST_POINTS "
            "for the same window or pass --parent-results."
        )
    parent = pd.read_csv(parent_path)
    parent["player_id"] = pd.to_numeric(
        parent["player_id"], errors="coerce"
    ).astype("Int64")
    keep = [
        column
        for column in [
            "player_id",
            "off",
            "def",
            "net_rapm",
            "possessions",
            "off_poss",
            "def_poss",
        ]
        if column in parent.columns
    ]
    if "off" not in keep:
        raise ValueError(f"Parent result {parent_path} has no offensive RAPM column.")
    parent = parent[keep].rename(
        columns={
            "off": "assist_points_off_rapm",
            "def": "assist_points_def_allowed_rapm",
            "net_rapm": "assist_points_net_rapm",
            "possessions": "assist_points_possessions",
            "off_poss": "assist_points_off_poss",
            "def_poss": "assist_points_def_poss",
        }
    )
    hierarchy = hierarchy.merge(
        parent, on="player_id", how="inner", validate="one_to_one"
    )
    logging.info("Joined parent ASSIST_POINTS result %s", parent_path)
    hierarchy = reconcile_role_components(hierarchy)
    hierarchy = hierarchy.sort_values(
        "assist_points_off_rapm",
        ascending=False,
        na_position="last",
    )
    output_path = RESULTS_DIR / f"assist_points_hierarchy_{suffix}_results.csv"
    hierarchy.to_csv(output_path, index=False)
    logging.info("Wrote %s", output_path)

    print("\nTop offensive ASSIST_POINTS RAPM with additive role decomposition")
    print(
        hierarchy.head(args.display)[
            [
                "player_name",
                "assist_points_off_rapm",
                "passer_assist_points_rapm",
                "receiver_assist_points_rapm",
                "passer_minus_receiver_rapm",
                "assist_points_decomp_gap",
            ]
        ]
        .to_string(index=False)
    )
    print("\nContrast model")
    print(pd.DataFrame([contrast_meta]).to_string(index=False))
    max_gap = hierarchy["assist_points_decomp_gap"].abs().max()
    logging.info("Maximum player-level decomposition gap: %.12g", max_gap)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompose ASSIST_POINTS RAPM into additive passer and receiver components."
    )
    parser.add_argument("start_year", type=int)
    parser.add_argument("end_year", type=int)
    parser.add_argument("season_type", choices=["RS", "PS", "ALL", "rs", "ps", "all"])
    parser.add_argument("--role-alpha", type=float, default=3000.0)
    parser.add_argument("--teammate-alpha", type=float, default=3000.0)
    parser.add_argument("--defense-alpha", type=float, default=5000.0)
    parser.add_argument("--parent-results")
    parser.add_argument("--display", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
