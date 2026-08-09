#!/usr/bin/env python3
"""Decompose offensive TS RAPM by self-shot, teammate-shot, and shot mix.

Two role-conditioned RAPM models estimate:

* ``a``: the player's centered TS effect when taking the attempt;
* ``b``: the player's centered TS effect when a teammate takes the attempt.

Their contrast is anchored to the existing offensive TS RAPM parent ``P`` at
the player's observed shot share ``u``:

    a - b = fitted role contrast
    u*a + (1-u)*b = P

An exact Shapley decomposition relative to the neutral five-player shot share
``u0 = 0.2`` then gives self-shot, teammate-shot, and shot-mix contributions.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .run_assist_points_hierarchy import (
        CONTEXT_COLS,
        DEFENSE_COLS,
        PROCESSED_DIR,
        RESULTS_DIR,
        fit_role_contrast_model,
        role_paths,
        season_suffix,
    )
except ImportError:
    from run_assist_points_hierarchy import (
        CONTEXT_COLS,
        DEFENSE_COLS,
        PROCESSED_DIR,
        RESULTS_DIR,
        fit_role_contrast_model,
        role_paths,
        season_suffix,
    )


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def read_ts_role_inputs(paths: list[Path]) -> pd.DataFrame:
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
        raise FileNotFoundError("No TS shot-role parquets found.")
    df = pd.concat(frames, ignore_index=True)
    required = [
        "game_id",
        "event_num",
        "Focal_Player",
        "TS_Points",
        "Role_Resolution_Complete",
        *CONTEXT_COLS,
        *DEFENSE_COLS,
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required TS role columns: {missing}")
    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    before = len(df)
    df = df.dropna(subset=required).copy()
    df = df[df["Role_Resolution_Complete"].astype(bool)].copy()
    logging.info("Kept %d resolved role rows (removed %d)", len(df), before - len(df))
    return df


def anchor_conditional_ts_effects(
    parent: pd.Series,
    contrast: pd.Series,
    shot_share: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Recover conditional self/teammate rates from parent, contrast, and usage."""
    self_rate = parent + (1.0 - shot_share) * contrast
    teammate_rate = parent - shot_share * contrast
    return self_rate, teammate_rate


def shapley_ts_components(
    self_rate: pd.Series,
    teammate_rate: pd.Series,
    shot_share: pd.Series,
    reference_share: float = 0.2,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Allocate u*a + (1-u)*b relative to (u0, a=0, b=0)."""
    average_share = 0.5 * (shot_share + reference_share)
    self_component = average_share * self_rate
    teammate_component = (1.0 - average_share) * teammate_rate
    shot_mix_component = (
        0.5 * (shot_share - reference_share) * (self_rate - teammate_rate)
    )
    return self_component, teammate_component, shot_mix_component


def reconcile_ts_hierarchy(
    hierarchy: pd.DataFrame,
    reference_share: float = 0.2,
) -> pd.DataFrame:
    result = hierarchy.copy()
    parent = pd.to_numeric(result["ots_rapm"], errors="coerce")
    contrast = pd.to_numeric(
        result["self_minus_teammate_ts_rapm"], errors="coerce"
    )
    shot_share = pd.to_numeric(result["shot_share"], errors="coerce")
    self_rate, teammate_rate = anchor_conditional_ts_effects(
        parent, contrast, shot_share
    )
    self_component, teammate_component, shot_mix_component = shapley_ts_components(
        self_rate, teammate_rate, shot_share, reference_share
    )
    result["conditional_self_shot_ts_rapm"] = self_rate
    result["conditional_teammate_shot_ts_rapm"] = teammate_rate
    result["self_shot_ts_component"] = self_component
    result["teammate_shot_ts_component"] = teammate_component
    result["shot_mix_ts_component"] = shot_mix_component
    result["shot_share_reference"] = float(reference_share)
    result["conditional_anchor_gap"] = (
        shot_share * self_rate + (1.0 - shot_share) * teammate_rate - parent
    )
    result["ts_role_decomp_gap"] = (
        self_component + teammate_component + shot_mix_component - parent
    )
    return result


def find_ts_parent(
    start_year: int,
    end_year: int,
    season_type: str,
    parent_results: str | None,
) -> Path | None:
    if parent_results:
        path = Path(parent_results)
        return path if path.exists() else None
    patterns = [
        (
            f"ts_{season_suffix(start_year)}_{season_suffix(end_year)}_"
            f"{season_type.lower()}*_results.csv"
        ),
    ]
    if int(start_year) == int(end_year):
        patterns.append(
            f"ts_{season_suffix(start_year)}_{season_type.lower()}*_results.csv"
        )
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(RESULTS_DIR.glob(pattern))
    matches = sorted(
        set(matches), key=lambda path: path.stat().st_mtime, reverse=True
    )
    return matches[0] if matches else None


def _fit_conditional_model(
    frame: pd.DataFrame,
    output_col: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict]:
    result, metadata = fit_role_contrast_model(
        frame,
        args.role_alpha,
        args.teammate_alpha,
        args.defense_alpha,
        target_col="TS_Points",
        output_col=output_col,
    )
    return result, metadata


def run(args: argparse.Namespace) -> Path:
    season_type = args.season_type.upper()
    self_df = read_ts_role_inputs(
        role_paths("TS_SELF_SHOT", args.start_year, args.end_year, season_type)
    )
    teammate_df = read_ts_role_inputs(
        role_paths("TS_TEAMMATE_SHOT", args.start_year, args.end_year, season_type)
    )
    self_fit, self_meta = _fit_conditional_model(
        self_df, "raw_conditional_self_shot_ts_rapm", args
    )
    teammate_fit, teammate_meta = _fit_conditional_model(
        teammate_df, "raw_conditional_teammate_shot_ts_rapm", args
    )

    self_keep = self_fit[
        [
            "player_id",
            "player_name",
            "raw_conditional_self_shot_ts_rapm",
            "focal_poss",
        ]
    ].rename(columns={"focal_poss": "self_shot_attempts"})
    teammate_keep = teammate_fit[
        [
            "player_id",
            "raw_conditional_teammate_shot_ts_rapm",
            "focal_poss",
        ]
    ].rename(columns={"focal_poss": "teammate_shot_attempts"})
    hierarchy = self_keep.merge(
        teammate_keep, on="player_id", how="inner", validate="one_to_one"
    )
    hierarchy["self_minus_teammate_ts_rapm"] = (
        hierarchy["raw_conditional_self_shot_ts_rapm"]
        - hierarchy["raw_conditional_teammate_shot_ts_rapm"]
    )
    total_role_attempts = (
        hierarchy["self_shot_attempts"] + hierarchy["teammate_shot_attempts"]
    )
    hierarchy["shot_share"] = (
        hierarchy["self_shot_attempts"] / total_role_attempts.replace(0, np.nan)
    )

    parent_path = find_ts_parent(
        args.start_year, args.end_year, season_type, args.parent_results
    )
    if parent_path is None:
        raise FileNotFoundError(
            "No matching TS parent result found. Run rapm.py TS for the same "
            "window or pass --parent-results."
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
        raise ValueError(f"Parent result {parent_path} has no offensive TS column.")
    parent = parent[keep].rename(
        columns={
            "off": "ots_rapm",
            "def": "dts_allowed_rapm",
            "net_rapm": "ts_net_rapm",
            "possessions": "ts_possessions",
            "off_poss": "ts_off_poss",
            "def_poss": "ts_def_poss",
        }
    )
    hierarchy = hierarchy.merge(
        parent, on="player_id", how="inner", validate="one_to_one"
    )
    hierarchy = reconcile_ts_hierarchy(
        hierarchy, reference_share=args.reference_share
    ).sort_values("ots_rapm", ascending=False)

    suffix = (
        f"{season_suffix(args.start_year)}_{season_suffix(args.end_year)}_"
        f"{season_type.lower()}_a{int(args.role_alpha)}_"
        f"{int(args.teammate_alpha)}_{int(args.defense_alpha)}"
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"ts_shot_role_hierarchy_{suffix}_results.csv"
    hierarchy.to_csv(output_path, index=False)
    logging.info("Joined parent TS result %s", parent_path)
    logging.info("Wrote %s", output_path)
    logging.info(
        "Maximum TS decomposition gap: %.12g",
        hierarchy["ts_role_decomp_gap"].abs().max(),
    )

    display_cols = [
        "player_name",
        "ots_rapm",
        "self_shot_ts_component",
        "teammate_shot_ts_component",
        "shot_mix_ts_component",
        "conditional_self_shot_ts_rapm",
        "conditional_teammate_shot_ts_rapm",
        "shot_share",
    ]
    print("\nTop offensive TS RAPM with shot-role decomposition")
    print(hierarchy.head(args.display)[display_cols].to_string(index=False))
    print("\nConditional role models")
    print(pd.DataFrame([{"role": "self", **self_meta}, {"role": "teammate", **teammate_meta}]).to_string(index=False))
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose offensive TS RAPM into self-shot, teammate-shot, "
            "and shot-mix contributions."
        )
    )
    parser.add_argument("start_year", type=int)
    parser.add_argument("end_year", type=int)
    parser.add_argument(
        "season_type", choices=["RS", "PS", "ALL", "rs", "ps", "all"]
    )
    parser.add_argument("--role-alpha", type=float, default=3000.0)
    parser.add_argument("--teammate-alpha", type=float, default=3000.0)
    parser.add_argument("--defense-alpha", type=float, default=5000.0)
    parser.add_argument("--reference-share", type=float, default=0.2)
    parser.add_argument("--parent-results")
    parser.add_argument("--display", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
