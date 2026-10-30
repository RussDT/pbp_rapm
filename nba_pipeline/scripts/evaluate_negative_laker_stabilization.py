#!/usr/bin/env python3
"""Select LAKER prior stabilization on consecutive-season forward ratings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .build_negative_laker_priors import _normalize_nba_id, load_age_curve
except ImportError:
    from build_negative_laker_priors import _normalize_nba_id, load_age_curve


def build_player_seasons(rows: pd.DataFrame, curve: pd.DataFrame) -> pd.DataFrame:
    frame = rows.copy()
    frame["player_id"] = frame["nba_id"].map(_normalize_nba_id)
    for column in ["year", "age", "playoffs", "minutes", "orapm", "drapm", "rapm"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[
        frame["playoffs"].eq(0)
        & frame["player_id"].notna()
        & frame["minutes"].gt(0)
        & frame[["age", "orapm", "drapm", "rapm"]].notna().all(axis=1)
        & ~frame["team_abbreviation"].astype(str).str.upper().eq("TOT")
    ].copy()
    ages = curve["age"].to_numpy(dtype=float)
    clipped = frame["age"].clip(float(ages.min()), float(ages.max()))
    frame["off"] = frame["orapm"] - np.interp(
        clipped, ages, curve["orapm_age_effect"].to_numpy(dtype=float)
    )
    frame["def"] = frame["drapm"] - np.interp(
        clipped, ages, curve["drapm_age_effect"].to_numpy(dtype=float)
    )
    frame["net"] = frame["off"] + frame["def"]
    for side in ["off", "def", "net"]:
        frame[f"weighted_{side}"] = frame["minutes"] * frame[side]
    seasons = frame.groupby(["player_id", "year"], as_index=False).agg(
        minutes=("minutes", "sum"),
        weighted_off=("weighted_off", "sum"),
        weighted_def=("weighted_def", "sum"),
        weighted_net=("weighted_net", "sum"),
    )
    for side in ["off", "def", "net"]:
        seasons[side] = seasons[f"weighted_{side}"] / seasons["minutes"]
    return seasons


def build_forward_transitions(
    seasons: pd.DataFrame,
    *,
    minute_threshold: float,
) -> pd.DataFrame:
    rows = []
    for player_id, history in seasons.sort_values("year").groupby("player_id"):
        hist_minutes = 0.0
        hist_sums = {side: 0.0 for side in ["off", "def", "net"]}
        last_year = None
        for row in history.to_dict("records"):
            consecutive = last_year is not None and int(row["year"]) == int(last_year) + 1
            hist_net = hist_sums["net"] / hist_minutes if hist_minutes else np.nan
            if consecutive and hist_minutes <= minute_threshold and hist_net < 0:
                rows.append({
                    "player_id": player_id,
                    "target_year": int(row["year"]),
                    "history_minutes": hist_minutes,
                    "history_off": hist_sums["off"] / hist_minutes,
                    "history_def": hist_sums["def"] / hist_minutes,
                    "history_net": hist_net,
                    "target_off": float(row["off"]),
                    "target_def": float(row["def"]),
                    "target_net": float(row["net"]),
                    "target_minutes": float(row["minutes"]),
                })
            hist_minutes += float(row["minutes"])
            for side in ["off", "def", "net"]:
                hist_sums[side] += float(row[f"weighted_{side}"])
            last_year = int(row["year"])
    return pd.DataFrame(rows)


def evaluate_grid(transitions: pd.DataFrame, grid: list[float]) -> pd.DataFrame:
    if transitions.empty:
        raise ValueError("No eligible consecutive-season transitions.")
    weights = transitions["target_minutes"].to_numpy(dtype=float)
    baseline = {
        side: float(np.average(transitions[f"target_{side}"] ** 2, weights=weights))
        for side in ["off", "def", "net"]
    }
    results = []
    for stabilization in grid:
        reliability = transitions["history_minutes"] / (
            transitions["history_minutes"] + float(stabilization)
        )
        row = {"stabilization_minutes": float(stabilization)}
        for side in ["off", "def", "net"]:
            prediction = reliability * transitions[f"history_{side}"]
            mse = float(np.average(
                (prediction - transitions[f"target_{side}"]) ** 2,
                weights=weights,
            ))
            row[f"{side}_mse"] = mse
            row[f"{side}_mse_gain_vs_zero"] = baseline[side] - mse
        row["off_def_mse_sum"] = row["off_mse"] + row["def_mse"]
        row["off_def_gain_vs_zero"] = (
            baseline["off"] + baseline["def"] - row["off_def_mse_sum"]
        )
        results.append(row)
    return pd.DataFrame(results).sort_values("stabilization_minutes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--age-curve", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minute-threshold", type=float, default=1000.0)
    parser.add_argument(
        "--grid",
        default="0,25,50,100,250,500,1000,2000,2344,4000",
    )
    args = parser.parse_args()

    grid = [float(value) for value in args.grid.split(",")]
    seasons = build_player_seasons(pd.read_csv(args.source_csv), load_age_curve(args.age_curve))
    transitions = build_forward_transitions(
        seasons,
        minute_threshold=args.minute_threshold,
    )
    results = evaluate_grid(transitions, grid)
    selected = results.sort_values(
        ["off_def_mse_sum", "stabilization_minutes"], ascending=[True, False]
    ).iloc[0]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "negative_laker_stabilization_validation.csv"
    receipt_path = args.output_dir / "negative_laker_stabilization_validation_receipt.json"
    results.to_csv(csv_path, index=False)
    receipt = {
        "design": "consecutive-season age-27 LAKER prediction",
        "eligibility": "prior history age-27 LAKER net < 0 and cumulative history minutes <= threshold",
        "weight": "target-season regular-season minutes",
        "selection_metric": "offense MSE + defense MSE",
        "minute_threshold": float(args.minute_threshold),
        "transitions": int(len(transitions)),
        "target_minutes": float(transitions["target_minutes"].sum()),
        "selected_stabilization_minutes": float(selected["stabilization_minutes"]),
        "selected_off_def_mse_sum": float(selected["off_def_mse_sum"]),
        "selected_off_def_gain_vs_zero": float(selected["off_def_gain_vs_zero"]),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
