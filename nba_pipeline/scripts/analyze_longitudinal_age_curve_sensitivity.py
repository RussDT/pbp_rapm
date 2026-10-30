#!/usr/bin/env python3
"""Compare longitudinal RAPM age curves across annual ridge penalties."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


CURVE_COLUMNS = {"age", "orapm_age_effect", "drapm_age_effect"}


def compare_age_curves(
    curves: Mapping[str, pd.DataFrame],
    *,
    reference_label: str,
    convergence_label: str,
    net_tolerance: float = 0.20,
) -> tuple[pd.DataFrame, dict]:
    if reference_label not in curves or convergence_label not in curves:
        raise ValueError("reference_label and convergence_label must name supplied curves")
    frames = []
    for label, raw in curves.items():
        missing = CURVE_COLUMNS - set(raw.columns)
        if missing:
            raise ValueError(f"Curve {label} missing columns: {sorted(missing)}")
        frame = raw.copy()
        frame["model"] = label
        frame["net_age_effect"] = frame["orapm_age_effect"] + frame["drapm_age_effect"]
        if "supported_age" not in frame:
            frame["supported_age"] = frame["age"].between(20, 39)
        frames.append(frame)
    comparison = pd.concat(frames, ignore_index=True)

    reference = comparison[comparison["model"] == reference_label].set_index("age")
    convergence = comparison[comparison["model"] == convergence_label].set_index("age")
    common = reference.index.intersection(convergence.index)
    supported = (
        reference.loc[common, "supported_age"].astype(bool)
        & convergence.loc[common, "supported_age"].astype(bool)
    )
    ages = common[supported.to_numpy()]
    if len(ages) == 0:
        raise ValueError("No commonly supported ages for convergence comparison")

    differences = {}
    for side, column in [
        ("off", "orapm_age_effect"),
        ("def", "drapm_age_effect"),
        ("net", "net_age_effect"),
    ]:
        differences[side] = (
            reference.loc[ages, column].to_numpy(dtype=float)
            - convergence.loc[ages, column].to_numpy(dtype=float)
        )
    max_abs_net = float(np.max(np.abs(differences["net"])))
    receipt = {
        "reference_label": reference_label,
        "convergence_label": convergence_label,
        "supported_age_min": int(np.min(ages)),
        "supported_age_max": int(np.max(ages)),
        "supported_ages": int(len(ages)),
        "max_abs_off_difference": round(float(np.max(np.abs(differences["off"]))), 12),
        "max_abs_def_difference": round(float(np.max(np.abs(differences["def"]))), 12),
        "max_abs_net_difference": round(max_abs_net, 12),
        "net_tolerance": float(net_tolerance),
        "converged": bool(max_abs_net <= net_tolerance),
    }
    return comparison.sort_values(["model", "age"]).reset_index(drop=True), receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--curve",
        action="append",
        required=True,
        help="Curve specification LABEL=PATH; repeat for each annual-alpha run.",
    )
    parser.add_argument("--reference-label", required=True)
    parser.add_argument("--convergence-label", required=True)
    parser.add_argument("--net-tolerance", type=float, default=0.20)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    curves = {}
    for spec in args.curve:
        label, separator, path = spec.partition("=")
        if not separator or not label or not path:
            raise ValueError(f"Invalid --curve specification: {spec}")
        curves[label] = pd.read_csv(path)
    comparison, receipt = compare_age_curves(
        curves,
        reference_label=args.reference_label,
        convergence_label=args.convergence_label,
        net_tolerance=args.net_tolerance,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_dir / "alpha_sensitivity.csv", index=False)
    with (args.output_dir / "alpha_sensitivity_receipt.json").open("w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
