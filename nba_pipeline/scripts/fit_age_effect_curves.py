#!/usr/bin/env python3
"""
Fit polynomial curves to age-effect CSVs emitted by rapm.py --age-dummies.

Usage:
    python nba_pipeline/scripts/fit_age_effect_curves.py \
        nba_pipeline/results/rapm_14_26_all_rb_agefe \
        --degree 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SIDE_CONFIG = {
    "off": ("off_centered_effect_per_100", "off_slots"),
    "def": ("def_centered_effect_per_100", "def_slots"),
    "net": ("net_centered_effect_per_100", "total_slots"),
}


def metric_key(path: Path) -> str:
    name = path.name
    name = name.replace("_age_effects.csv", "")
    name = name.replace("_14_26_all_pure_rb_agefe", "")
    name = name.replace("_14_26_all_rb_agefe", "")
    name = name.replace("_pure", "")
    return name


def weighted_r2(y: np.ndarray, y_hat: np.ndarray, weights: np.ndarray) -> float:
    if weights.sum() <= 0:
        return float("nan")
    y_bar = np.average(y, weights=weights)
    ss_res = np.sum(weights * np.square(y - y_hat))
    ss_tot = np.sum(weights * np.square(y - y_bar))
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - (ss_res / ss_tot))


def fit_curve(
    metric: str,
    side: str,
    df: pd.DataFrame,
    degree: int,
) -> tuple[dict, list[dict]]:
    value_col, weight_col = SIDE_CONFIG[side]
    working = df[["age", value_col, weight_col]].copy()
    working["age"] = pd.to_numeric(working["age"], errors="coerce")
    working[value_col] = pd.to_numeric(working[value_col], errors="coerce")
    working[weight_col] = pd.to_numeric(working[weight_col], errors="coerce").fillna(0.0)
    working = working.dropna(subset=["age", value_col])
    working = working[working[weight_col] > 0].sort_values("age")

    if len(working) <= degree:
        raise ValueError(f"Not enough points to fit degree {degree}: {metric} {side}")

    x = working["age"].to_numpy(dtype=float)
    y = working[value_col].to_numpy(dtype=float)
    weights = working[weight_col].to_numpy(dtype=float)

    coeffs = np.polyfit(x, y, degree, w=np.sqrt(weights))
    y_hat = np.polyval(coeffs, x)

    coef_row = {
        "metric": metric,
        "side": side,
        "degree": degree,
        "weighted_r2": weighted_r2(y, y_hat, weights),
        "age_min": int(x.min()),
        "age_max": int(x.max()),
        "weight_sum": float(weights.sum()),
    }
    for i, coef in enumerate(coeffs):
        power = degree - i
        coef_row[f"coef_age_pow_{power}"] = float(coef)

    fit_rows = []
    for age, observed, fitted, weight in zip(x, y, y_hat, weights):
        fit_rows.append(
            {
                "metric": metric,
                "side": side,
                "age": int(age),
                "observed_centered_effect_per_100": float(observed),
                "fitted_centered_effect_per_100": float(fitted),
                "residual": float(observed - fitted),
                "weight": float(weight),
            }
        )

    return coef_row, fit_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit polynomial age-effect curves.")
    parser.add_argument("input_dir", type=Path, help="Directory containing *_age_effects.csv files.")
    parser.add_argument("--degree", type=int, default=3, help="Polynomial degree to fit. Default: 3.")
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output filename prefix. Default: age_curve_poly{degree}",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)
    if args.degree < 1:
        raise ValueError("--degree must be >= 1")

    output_prefix = args.output_prefix or f"age_curve_poly{args.degree}"
    files = sorted(input_dir.glob("*_age_effects.csv"))
    if not files:
        raise FileNotFoundError(f"No *_age_effects.csv files found in {input_dir}")

    coefficient_rows = []
    fitted_rows = []

    for path in files:
        metric = metric_key(path)
        df = pd.read_csv(path)
        for side in SIDE_CONFIG:
            coef_row, side_fit_rows = fit_curve(metric, side, df, args.degree)
            coefficient_rows.append(coef_row)
            fitted_rows.extend(side_fit_rows)

    coefficients = pd.DataFrame(coefficient_rows).sort_values(["metric", "side"]).reset_index(drop=True)
    fitted_values = pd.DataFrame(fitted_rows).sort_values(["metric", "side", "age"]).reset_index(drop=True)

    coef_path = input_dir / f"{output_prefix}_coefficients.csv"
    fitted_path = input_dir / f"{output_prefix}_fitted_values.csv"
    coefficients.to_csv(coef_path, index=False)
    fitted_values.to_csv(fitted_path, index=False)

    print(f"Wrote {coef_path}")
    print(f"Wrote {fitted_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
