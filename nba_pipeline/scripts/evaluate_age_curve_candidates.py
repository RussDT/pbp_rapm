#!/usr/bin/env python3
"""Score frozen age curves against consecutive annual O/D RAPM changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


PANEL_COLUMNS = {
    "player_id",
    "season",
    "age",
    "orapm",
    "drapm",
    "off_poss",
    "def_poss",
}
CURVE_COLUMNS = {"age", "orapm_age_effect", "drapm_age_effect"}


def _harmonic(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    denominator = left + right
    return np.divide(2.0 * left * right, denominator, out=np.zeros_like(left), where=denominator > 0)


def _build_pairs(panel: pd.DataFrame) -> pd.DataFrame:
    working = panel.copy()
    for column in PANEL_COLUMNS - {"player_id"}:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.dropna(subset=list(PANEL_COLUMNS)).sort_values(["player_id", "season"])
    previous = working.groupby("player_id", sort=False).shift(1)
    pairs = working.loc[(working["season"] - previous["season"]) == 1].copy()
    for column in ["age", "orapm", "drapm", "off_poss", "def_poss"]:
        pairs[f"previous_{column}"] = previous.loc[pairs.index, column]
    return pairs


def evaluate_candidates(
    annual_panel: pd.DataFrame,
    candidates: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict]:
    missing_panel = PANEL_COLUMNS - set(annual_panel.columns)
    if missing_panel:
        raise ValueError(f"annual panel missing columns: {sorted(missing_panel)}")
    pairs = _build_pairs(annual_panel)
    if pairs.empty:
        raise ValueError("annual panel has no consecutive player-season pairs")

    prepared: dict[str, pd.DataFrame | None] = {"zero": None}
    for label, curve in candidates.items():
        if label == "zero":
            raise ValueError("zero is a reserved candidate label")
        missing_curve = CURVE_COLUMNS - set(curve.columns)
        if missing_curve:
            raise ValueError(f"curve {label} missing columns: {sorted(missing_curve)}")
        candidate = curve[list(CURVE_COLUMNS)].copy()
        for column in CURVE_COLUMNS:
            candidate[column] = pd.to_numeric(candidate[column], errors="coerce")
        candidate = candidate.dropna().sort_values("age").drop_duplicates("age", keep="last")
        prepared[label] = candidate

    rows: list[dict] = []
    for label, curve in prepared.items():
        predictions: dict[str, np.ndarray] = {}
        for target, rating, possession in (
            ("off", "orapm", "off_poss"),
            ("def", "drapm", "def_poss"),
        ):
            actual = (pairs[rating] - pairs[f"previous_{rating}"]).to_numpy(dtype=float)
            if curve is None:
                prediction = np.zeros(len(pairs), dtype=float)
            else:
                curve_column = f"{rating}_age_effect"
                prediction = np.interp(pairs["age"], curve["age"], curve[curve_column]) - np.interp(
                    pairs["previous_age"], curve["age"], curve[curve_column]
                )
            predictions[target] = prediction
            weights = _harmonic(
                pairs[possession].to_numpy(dtype=float),
                pairs[f"previous_{possession}"].to_numpy(dtype=float),
            )
            error = actual - prediction
            rows.append(
                {
                    "candidate": label,
                    "target": target,
                    "pairs": int(len(pairs)),
                    "weighted_mse": float(np.average(np.square(error), weights=weights)),
                    "weighted_mae": float(np.average(np.abs(error), weights=weights)),
                    "weighted_bias": float(np.average(error, weights=weights)),
                }
            )

        actual_net = (
            pairs["orapm"]
            + pairs["drapm"]
            - pairs["previous_orapm"]
            - pairs["previous_drapm"]
        ).to_numpy(dtype=float)
        prediction_net = predictions["off"] + predictions["def"]
        current_poss = (pairs["off_poss"] + pairs["def_poss"]).to_numpy(dtype=float)
        previous_poss = (
            pairs["previous_off_poss"] + pairs["previous_def_poss"]
        ).to_numpy(dtype=float)
        net_weights = _harmonic(current_poss, previous_poss)
        net_error = actual_net - prediction_net
        rows.append(
            {
                "candidate": label,
                "target": "net",
                "pairs": int(len(pairs)),
                "weighted_mse": float(np.average(np.square(net_error), weights=net_weights)),
                "weighted_mae": float(np.average(np.abs(net_error), weights=net_weights)),
                "weighted_bias": float(np.average(net_error, weights=net_weights)),
            }
        )

    scores = pd.DataFrame(rows)
    zero = scores.loc[scores["candidate"] == "zero", ["target", "weighted_mse"]].rename(
        columns={"weighted_mse": "zero_mse"}
    )
    scores = scores.merge(zero, on="target", validate="many_to_one")
    scores["mse_improvement_vs_zero"] = scores["zero_mse"] - scores["weighted_mse"]
    receipt = {
        "method": "consecutive_annual_rapm_change",
        "consecutive_pairs": int(len(pairs)),
        "players": int(pairs["player_id"].nunique()),
        "first_target_season": int(pairs["season"].min()),
        "last_target_season": int(pairs["season"].max()),
        "weighting": "harmonic_consecutive_season_possessions",
        "interpretation": "diagnostic_not_independent_possession_holdout",
    }
    return scores, receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annual-panel", type=Path, required=True)
    parser.add_argument("--curve", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    curves: dict[str, pd.DataFrame] = {}
    for value in args.curve:
        if "=" not in value:
            raise ValueError(f"curve must be LABEL=PATH: {value}")
        label, path = value.split("=", 1)
        curves[label] = pd.read_csv(path)
    scores, receipt = evaluate_candidates(pd.read_csv(args.annual_panel), curves)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output, index=False)
    receipt_path = args.receipt or args.output.with_name(f"{args.output.stem}_receipt.json")
    with receipt_path.open("w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
