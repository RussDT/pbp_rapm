#!/usr/bin/env python3
"""Build a reader-safe age-27 O/D RAPM board with extrapolation diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


RESULT_COLUMNS = {
    "player_id",
    "player_name",
    "off",
    "def",
    "possessions",
    "off_poss",
    "def_poss",
}
PANEL_COLUMNS = {"player_id", "season", "age", "off_poss", "def_poss"}
CURVE_COLUMNS = {"age", "orapm_age_effect", "drapm_age_effect", "supported_age"}


def _string_ids(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.astype("Int64").astype(str)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return float("nan")
    return float(np.average(values.loc[valid], weights=weights.loc[valid]))


def _support_label(distance: float) -> str:
    if not np.isfinite(distance):
        return "missing_age_history"
    if distance <= 1.0:
        return "near_reference"
    if distance <= 4.0:
        return "moderate_extrapolation"
    return "long_extrapolation"


def build_parent_audit(
    results: pd.DataFrame,
    annual_panel: pd.DataFrame,
    age_curve: pd.DataFrame,
    *,
    reference_age: float = 27.0,
) -> tuple[pd.DataFrame, dict]:
    """Attach exposure-weighted age adjustments to a raw ``rapm.py`` result.

    ``rapm.py`` exports defense as points allowed (lower is better). This public
    audit flips it exactly once and reports every rating as positive-good.
    """
    for label, frame, required in (
        ("results", results, RESULT_COLUMNS),
        ("annual panel", annual_panel, PANEL_COLUMNS),
        ("age curve", age_curve, CURVE_COLUMNS),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} missing columns: {sorted(missing)}")

    curve = age_curve.copy()
    for column in ["age", "orapm_age_effect", "drapm_age_effect"]:
        curve[column] = pd.to_numeric(curve[column], errors="coerce")
    curve = curve.dropna(subset=["age", "orapm_age_effect", "drapm_age_effect"])
    curve = curve.sort_values("age").drop_duplicates("age", keep="last")
    if curve.empty:
        raise ValueError("age curve has no usable rows")
    supported = curve.loc[curve["supported_age"].astype(bool), "age"]
    if supported.empty:
        raise ValueError("age curve has no supported ages")
    supported_min = float(supported.min())
    supported_max = float(supported.max())

    panel = annual_panel.copy()
    panel["player_id"] = _string_ids(panel["player_id"])
    for column in ["age", "off_poss", "def_poss"]:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    panel = panel.dropna(subset=["age"])
    panel["off_age_effect"] = np.interp(
        panel["age"], curve["age"], curve["orapm_age_effect"]
    )
    panel["def_age_effect"] = np.interp(
        panel["age"], curve["age"], curve["drapm_age_effect"]
    )
    panel["within_supported_age_range"] = panel["age"].between(
        supported_min, supported_max, inclusive="both"
    )

    rows: list[dict] = []
    for player_id, group in panel.groupby("player_id", sort=False):
        ages = group["age"].dropna()
        nearest = float((ages - reference_age).abs().min()) if not ages.empty else float("nan")
        off_total = float(group.loc[group["off_poss"] > 0, "off_poss"].sum())
        def_total = float(group.loc[group["def_poss"] > 0, "def_poss"].sum())
        rows.append(
            {
                "player_id": player_id,
                "first_age": float(ages.min()) if not ages.empty else float("nan"),
                "last_age": float(ages.max()) if not ages.empty else float("nan"),
                "nearest_age_to_reference": nearest,
                "reference_age_support": _support_label(nearest),
                "off_mean_age": _weighted_mean(group["age"], group["off_poss"]),
                "def_mean_age": _weighted_mean(group["age"], group["def_poss"]),
                "off_mean_age_effect": _weighted_mean(
                    group["off_age_effect"], group["off_poss"]
                ),
                "def_mean_age_effect": _weighted_mean(
                    group["def_age_effect"], group["def_poss"]
                ),
                "off_supported_exposure_share": (
                    float(
                        group.loc[
                            group["within_supported_age_range"] & (group["off_poss"] > 0),
                            "off_poss",
                        ].sum()
                        / off_total
                    )
                    if off_total > 0
                    else float("nan")
                ),
                "def_supported_exposure_share": (
                    float(
                        group.loc[
                            group["within_supported_age_range"] & (group["def_poss"] > 0),
                            "def_poss",
                        ].sum()
                        / def_total
                    )
                    if def_total > 0
                    else float("nan")
                ),
            }
        )
    age_audit = pd.DataFrame(rows)

    output = results.copy()
    output["player_id"] = _string_ids(output["player_id"])
    output["orapm_age27"] = pd.to_numeric(output["off"], errors="coerce")
    output["drapm_age27"] = -pd.to_numeric(output["def"], errors="coerce")
    output["net_age27"] = output["orapm_age27"] + output["drapm_age27"]
    output = output.merge(age_audit, on="player_id", how="left", validate="one_to_one")
    output["reference_age_support"] = output["reference_age_support"].fillna(
        "missing_age_history"
    )
    output["off_age_adjustment_to_27"] = -output["off_mean_age_effect"]
    output["def_age_adjustment_to_27"] = -output["def_mean_age_effect"]
    output["net_age_adjustment_to_27"] = (
        output["off_age_adjustment_to_27"] + output["def_age_adjustment_to_27"]
    )
    output["orapm_at_observed_age"] = (
        output["orapm_age27"] + output["off_mean_age_effect"]
    )
    output["drapm_at_observed_age"] = (
        output["drapm_age27"] + output["def_mean_age_effect"]
    )
    output["net_at_observed_age"] = (
        output["orapm_at_observed_age"] + output["drapm_at_observed_age"]
    )
    output = output.drop(columns=["off", "def", "net_rapm"], errors="ignore")
    output = output.sort_values(["net_age27", "player_id"], ascending=[False, True]).reset_index(
        drop=True
    )

    audited = int(output["off_mean_age"].notna().sum())
    receipt = {
        "result_players": int(len(output)),
        "players_with_age_audit": audited,
        "players_missing_age_audit": int(len(output) - audited),
        "reference_age": float(reference_age),
        "supported_age_min": supported_min,
        "supported_age_max": supported_max,
        "result_raw_defense_sign": "points_allowed_lower_good",
        "defense_output_sign": "positive_good",
        "age_exposure_sample": "annual_regular_season_panel",
        "age_curve_interpolation": "piecewise_linear_endpoint_clipped",
    }
    return output, receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--annual-panel", type=Path, required=True)
    parser.add_argument("--age-curve", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, default=None)
    parser.add_argument("--reference-age", type=float, default=27.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output, receipt = build_parent_audit(
        pd.read_csv(args.results),
        pd.read_csv(args.annual_panel),
        pd.read_csv(args.age_curve),
        reference_age=args.reference_age,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    receipt_path = args.receipt or args.output.with_name(f"{args.output.stem}_receipt.json")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
