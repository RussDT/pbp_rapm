#!/usr/bin/env python3
"""Validate inferred shot clocks, with optional arena-clock labels.

The estimator can always be checked for state-machine integrity.  Absolute
error metrics are emitted only when an independent label file is supplied;
made/missed outcomes are not treated as shot-clock labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLUMNS = ["game_id", "period", "event_num"]


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def validate_estimates(
    estimates: pd.DataFrame,
    labels: pd.DataFrame | None = None,
    *,
    label_column: str = "shot_clock_actual",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Return row diagnostics, grouped metrics, and a JSON-safe summary."""
    required = {
        *KEY_COLUMNS,
        "shot_clock_est",
        "shot_clock_raw",
        "shot_clock_lower",
        "shot_clock_upper",
        "reset_len",
        "confidence",
        "reset_reason",
    }
    missing = sorted(required - set(estimates.columns))
    if missing:
        raise ValueError(f"Estimate table is missing required columns: {missing}")

    rows = estimates.copy()
    rows["bound_violation"] = (
        pd.to_numeric(rows["shot_clock_est"], errors="coerce").lt(0)
        | pd.to_numeric(rows["shot_clock_est"], errors="coerce").gt(
            pd.to_numeric(rows["reset_len"], errors="coerce")
        )
    )
    rows["raw_out_of_range"] = (
        pd.to_numeric(rows["shot_clock_raw"], errors="coerce").lt(-2)
        | pd.to_numeric(rows["shot_clock_raw"], errors="coerce").gt(
            pd.to_numeric(rows["reset_len"], errors="coerce") + 2
        )
    )
    rows["side_mismatch"] = rows.get(
        "confidence_reason", pd.Series("", index=rows.index)
    ).fillna("").astype(str).str.contains("side_mismatch", regex=False)

    has_labels = labels is not None
    if has_labels:
        label_required = {*KEY_COLUMNS, label_column}
        label_missing = sorted(label_required - set(labels.columns))
        if label_missing:
            raise ValueError(f"Label table is missing required columns: {label_missing}")
        label_rows = labels[list(KEY_COLUMNS) + [label_column]].copy()
        if label_rows.duplicated(KEY_COLUMNS).any():
            raise ValueError("Shot-clock labels must be unique by game_id/period/event_num.")
        label_rows[label_column] = pd.to_numeric(label_rows[label_column], errors="coerce")
        label_rows = label_rows[label_rows[label_column].between(0, 24, inclusive="both")]
        rows = rows.merge(label_rows, on=KEY_COLUMNS, how="left", validate="one_to_one")
        rows["signed_error"] = rows["shot_clock_est"] - rows[label_column]
        rows["abs_error"] = rows["signed_error"].abs()
        rows["squared_error"] = rows["signed_error"].pow(2)
        rows["within_1s"] = rows["abs_error"].le(1)
        rows["within_2s"] = rows["abs_error"].le(2)
        rows["interval_covered"] = rows[label_column].between(
            rows["shot_clock_lower"], rows["shot_clock_upper"], inclusive="both"
        )

    agg_spec: dict[str, tuple[str, object]] = {
        "rows": ("shot_clock_est", "size"),
        "mean_estimate": ("shot_clock_est", "mean"),
        "bound_violations": ("bound_violation", "sum"),
        "raw_out_of_range": ("raw_out_of_range", "sum"),
        "side_mismatches": ("side_mismatch", "sum"),
    }
    if has_labels:
        agg_spec.update(
            {
                "labeled_rows": (label_column, "count"),
                "mae": ("abs_error", "mean"),
                "rmse": ("squared_error", lambda x: float(np.sqrt(x.mean()))),
                "bias": ("signed_error", "mean"),
                "within_1s_rate": ("within_1s", "mean"),
                "within_2s_rate": ("within_2s", "mean"),
                "interval_coverage": ("interval_covered", "mean"),
            }
        )
    grouped = (
        rows.groupby(["confidence", "reset_reason"], dropna=False, observed=True)
        .agg(**agg_spec)
        .reset_index()
        .sort_values(["confidence", "rows"], ascending=[True, False])
    )

    summary: dict[str, object] = {
        "rows": int(len(rows)),
        "high_confidence_rows": int(rows["confidence"].eq("high").sum()),
        "high_confidence_rate": float(rows["confidence"].eq("high").mean()) if len(rows) else 0.0,
        "bound_violations": int(rows["bound_violation"].sum()),
        "raw_out_of_range": int(rows["raw_out_of_range"].sum()),
        "side_mismatches": int(rows["side_mismatch"].sum()),
        "has_independent_labels": bool(has_labels),
    }
    if has_labels:
        labeled = rows[rows[label_column].notna()]
        summary.update(
            {
                "labeled_rows": int(len(labeled)),
                "label_coverage": float(len(labeled) / len(rows)) if len(rows) else 0.0,
                "mae": float(labeled["abs_error"].mean()) if len(labeled) else None,
                "rmse": float(np.sqrt(labeled["squared_error"].mean())) if len(labeled) else None,
                "bias": float(labeled["signed_error"].mean()) if len(labeled) else None,
                "within_1s_rate": float(labeled["within_1s"].mean()) if len(labeled) else None,
                "within_2s_rate": float(labeled["within_2s"].mean()) if len(labeled) else None,
                "interval_coverage": float(labeled["interval_covered"].mean()) if len(labeled) else None,
            }
        )
    return rows, grouped, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimates", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--label-column", default="shot_clock_actual")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="shot_clock_validation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    estimates = _read_table(args.estimates)
    labels = _read_table(args.labels) if args.labels else None
    rows, grouped, summary = validate_estimates(
        estimates,
        labels,
        label_column=args.label_column,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(args.output_dir / f"{args.output_name}_rows.parquet", index=False)
    grouped.to_csv(args.output_dir / f"{args.output_name}_groups.csv", index=False)
    (args.output_dir / f"{args.output_name}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
