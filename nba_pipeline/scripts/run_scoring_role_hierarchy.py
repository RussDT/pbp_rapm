#!/usr/bin/env python3
"""Decompose offensive RAPM into additive on-ball and off-ball scoring roles."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

try:
    from .run_assist_points_hierarchy import (
        RESULTS_DIR,
        fit_role_contrast_model,
        read_role_inputs,
        role_paths,
        season_suffix,
    )
except ImportError:
    from run_assist_points_hierarchy import (
        RESULTS_DIR,
        fit_role_contrast_model,
        read_role_inputs,
        role_paths,
        season_suffix,
    )


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def build_scoring_contrast_input(
    onball_df: pd.DataFrame,
    offball_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join role surfaces on their common fully resolved focal-player sample."""
    keys = ["game_id", "poss_group", "Focal_Player"]
    context = [*[f"O{i}" for i in range(1, 5)], *[f"D{i}" for i in range(1, 6)]]
    onball = onball_df[
        [
            *keys,
            *context,
            "OnBall_Scoring_Points",
            "Role_Resolution_Complete",
        ]
    ]
    offball = offball_df[
        [*keys, "OffBall_Scoring_Points", "Role_Resolution_Complete"]
    ]
    merged = onball.merge(
        offball,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_onball", "_offball"),
    )
    complete = (
        merged["Role_Resolution_Complete_onball"].astype(bool)
        & merged["Role_Resolution_Complete_offball"].astype(bool)
    )
    merged = merged.loc[complete].copy()
    merged["OnBall_OffBall_Contrast"] = (
        pd.to_numeric(merged["OnBall_Scoring_Points"], errors="coerce")
        - pd.to_numeric(merged["OffBall_Scoring_Points"], errors="coerce")
    )
    return merged


def find_scoring_parent(
    start_year: int,
    end_year: int,
    season_type: str,
    parent_results: str | None,
) -> Path | None:
    if parent_results:
        path = Path(parent_results)
        return path if path.exists() else None
    pattern = (
        f"scoring_parent_{season_suffix(start_year)}_{season_suffix(end_year)}_"
        f"{season_type.lower()}*_results.csv"
    )
    matches = sorted(
        RESULTS_DIR.glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def reconcile_scoring_components(hierarchy: pd.DataFrame) -> pd.DataFrame:
    """Apply the parent sum and fitted role-contrast equations."""
    result = hierarchy.copy()
    total = pd.to_numeric(result["offensive_rapm"], errors="coerce")
    contrast = pd.to_numeric(
        result["onball_minus_offball_rapm"], errors="coerce"
    )
    result["onball_scoring_rapm"] = (total + contrast) / 2.0
    result["offball_scoring_rapm"] = (total - contrast) / 2.0
    result["scoring_role_decomp_gap"] = (
        result["onball_scoring_rapm"]
        + result["offball_scoring_rapm"]
        - total
    )
    return result


def run(args: argparse.Namespace) -> Path:
    season_type = args.season_type.upper()
    suffix = (
        f"{season_suffix(args.start_year)}_{season_suffix(args.end_year)}_"
        f"{season_type.lower()}_a{int(args.role_alpha)}_"
        f"{int(args.teammate_alpha)}_{int(args.defense_alpha)}"
    )
    onball_df = read_role_inputs(
        role_paths("SCORING_ONBALL", args.start_year, args.end_year, season_type),
        "OnBall_Scoring_Points",
    )
    offball_df = read_role_inputs(
        role_paths("SCORING_OFFBALL", args.start_year, args.end_year, season_type),
        "OffBall_Scoring_Points",
    )
    contrast_df = build_scoring_contrast_input(onball_df, offball_df)
    logging.info(
        "Using %d common fully resolved focal rows (%.2f%% of on-ball rows)",
        len(contrast_df),
        100.0 * len(contrast_df) / len(onball_df),
    )
    hierarchy, metadata = fit_role_contrast_model(
        contrast_df,
        args.role_alpha,
        args.teammate_alpha,
        args.defense_alpha,
        target_col="OnBall_OffBall_Contrast",
        output_col="onball_minus_offball_rapm",
    )
    hierarchy = hierarchy.rename(
        columns={
            "raw_passer_minus_receiver_points": "raw_onball_minus_offball_points",
            "raw_passer_minus_receiver_per_100": "raw_onball_minus_offball_per_100",
        }
    )

    parent_path = find_scoring_parent(
        args.start_year, args.end_year, season_type, args.parent_results
    )
    if parent_path is None:
        raise FileNotFoundError(
            "No matching scoring-parent result found. Run rapm.py "
            "SCORING_PARENT for the "
            "same window or pass --parent-results."
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
    parent = parent[keep].rename(
        columns={
            "off": "offensive_rapm",
            "def": "defensive_rapm",
            "net_rapm": "total_rapm",
        }
    )
    hierarchy = hierarchy.merge(
        parent, on="player_id", how="inner", validate="one_to_one"
    )
    hierarchy = reconcile_scoring_components(hierarchy).sort_values(
        "offensive_rapm", ascending=False
    )
    output_path = RESULTS_DIR / f"scoring_role_hierarchy_{suffix}_results.csv"
    hierarchy.to_csv(output_path, index=False)
    logging.info("Wrote %s", output_path)
    logging.info(
        "Maximum player-level decomposition gap: %.12g",
        hierarchy["scoring_role_decomp_gap"].abs().max(),
    )

    print("\nTop offensive RAPM with additive scoring-role decomposition")
    print(
        hierarchy.head(args.display)[
            [
                "player_name",
                "offensive_rapm",
                "onball_scoring_rapm",
                "offball_scoring_rapm",
                "onball_minus_offball_rapm",
                "scoring_role_decomp_gap",
            ]
        ].to_string(index=False)
    )
    print("\nContrast model")
    print(pd.DataFrame([metadata]).to_string(index=False))
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompose offensive RAPM into on-ball and off-ball scoring roles."
    )
    parser.add_argument("start_year", type=int)
    parser.add_argument("end_year", type=int)
    parser.add_argument(
        "season_type", choices=["RS", "PS", "ALL", "rs", "ps", "all"]
    )
    parser.add_argument("--role-alpha", type=float, default=3000.0)
    parser.add_argument("--teammate-alpha", type=float, default=3000.0)
    parser.add_argument("--defense-alpha", type=float, default=5000.0)
    parser.add_argument("--parent-results")
    parser.add_argument("--display", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
