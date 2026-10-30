#!/usr/bin/env python3
"""Smooth joint possession-level RAPM age bins into a frozen O/D curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "age",
    "off_raw_effect_per_100",
    "def_raw_effect_per_100",
    "off_slots",
    "def_slots",
}


def _weighted_r2(y: np.ndarray, fitted: np.ndarray, weights: np.ndarray) -> float:
    center = float(np.average(y, weights=weights))
    total = float(np.sum(weights * np.square(y - center)))
    residual = float(np.sum(weights * np.square(y - fitted)))
    return float(1.0 - residual / total) if total > 0 else float("nan")


def fit_joint_age_curve(
    age_effects: pd.DataFrame,
    *,
    degree: int = 3,
    age_min: int = 20,
    age_max: int = 38,
    reference_age: float = 27.0,
) -> tuple[pd.DataFrame, dict]:
    """Fit slot-weighted polynomials and return positive-good O/D values."""
    missing = REQUIRED_COLUMNS - set(age_effects.columns)
    if missing:
        raise ValueError(f"age effects missing columns: {sorted(missing)}")
    if degree < 1:
        raise ValueError("degree must be at least one")
    if age_max <= age_min:
        raise ValueError("age_max must exceed age_min")
    if not age_min <= reference_age <= age_max:
        raise ValueError("reference_age must lie inside the fitted range")

    working = age_effects[list(REQUIRED_COLUMNS)].copy()
    for column in REQUIRED_COLUMNS:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.dropna().sort_values("age")
    working = working[working["age"].between(age_min, age_max, inclusive="both")]
    if len(working) <= degree:
        raise ValueError("not enough supported age bins for requested polynomial degree")

    output_ages = np.arange(age_min, age_max + 1, dtype=float)
    rows: dict[str, np.ndarray] = {"age": output_ages.astype(int)}
    receipt: dict = {
        "method": "joint_possession_age_bins_slot_weighted_polynomial",
        "degree": int(degree),
        "fit_age_min": int(age_min),
        "fit_age_max": int(age_max),
        "reference_age": float(reference_age),
        "age_18_19_policy": "excluded_from_fit_and_endpoint_clipped",
        "age_above_max_policy": "endpoint_clipped",
        "source_defense_sign": "points_allowed_lower_good",
        "defense_output_sign": "positive_good",
        "sides": {},
    }

    x = working["age"].to_numpy(dtype=float)
    for side, output_column, source_column, weight_column, sign in (
        ("off", "orapm_age_effect", "off_raw_effect_per_100", "off_slots", 1.0),
        ("def", "drapm_age_effect", "def_raw_effect_per_100", "def_slots", -1.0),
    ):
        y = sign * working[source_column].to_numpy(dtype=float)
        weights = working[weight_column].to_numpy(dtype=float)
        if np.any(weights <= 0):
            raise ValueError(f"{side} slot weights must be positive")
        coefficients = np.polyfit(x, y, degree, w=np.sqrt(weights))
        fitted_source = np.polyval(coefficients, x)
        reference_value = float(np.polyval(coefficients, reference_age))
        fitted_output = np.polyval(coefficients, output_ages) - reference_value
        rows[output_column] = fitted_output
        receipt["sides"][side] = {
            "slot_weight_sum": float(weights.sum()),
            "weighted_r2": _weighted_r2(y, fitted_source, weights),
            "coefficients_high_to_low": [float(value) for value in coefficients],
        }

    curve = pd.DataFrame(rows)
    curve["supported_age"] = True
    return curve, receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--age-effects", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, default=None)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--age-min", type=int, default=20)
    parser.add_argument("--age-max", type=int, default=38)
    parser.add_argument("--reference-age", type=float, default=27.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    curve, receipt = fit_joint_age_curve(
        pd.read_csv(args.age_effects),
        degree=args.degree,
        age_min=args.age_min,
        age_max=args.age_max,
        reference_age=args.reference_age,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(args.output, index=False)
    receipt_path = args.receipt or args.output.with_name(f"{args.output.stem}_receipt.json")
    with receipt_path.open("w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
