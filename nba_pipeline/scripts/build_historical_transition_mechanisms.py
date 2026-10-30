#!/usr/bin/env python3
"""Materialize labeled historical transition mechanism inputs.

The source is the frozen 1997-2026 SQ-descriptor transition classifier archive.
Pre-2025 rows are retrospective backcasts, so these outputs deliberately use
BACKCAST prefixes and must not silently replace exact provider-era targets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED = PIPELINE_ROOT / "processed"
DEFAULT_TRANSITION = PIPELINE_ROOT / "results" / "sq_transition_backfill"


def normalize_game_id(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype("Int64")
    return numeric.astype(str).str.replace("<NA>", "", regex=False).str.zfill(10)


def attach_backcast(base: pd.DataFrame, labels: pd.DataFrame, *, year: int, phase: str) -> pd.DataFrame:
    base = base.copy()
    labels = labels.copy()
    for frame in (base, labels):
        frame["game_id"] = normalize_game_id(frame["game_id"])
        frame["event_num"] = pd.to_numeric(frame["event_num"], errors="coerce").astype("Int64")
    if base[["game_id", "event_num"]].duplicated().any():
        raise ValueError(f"{year} {phase}: duplicate base FGA keys")
    if labels[["game_id", "event_num"]].duplicated().any():
        raise ValueError(f"{year} {phase}: duplicate transition archive keys")
    keep = [
        "game_id",
        "event_num",
        "transition_label_operating",
        "transition_rim_label_operating",
        "transition_probability",
        "transition_rim_probability",
        "transition_calibration_scope",
        "transition_rim_calibration_scope",
        "feature_contract_version",
    ]
    merged = base.merge(labels[keep], on=["game_id", "event_num"], how="left", validate="one_to_one")
    required = ["transition_label_operating", "transition_rim_label_operating"]
    if merged[required].isna().any().any():
        missing = int(merged[required].isna().any(axis=1).sum())
        raise ValueError(f"{year} {phase}: {missing} FGA rows lack transition backcasts")
    if len(merged) != len(base):
        raise AssertionError("Transition merge changed the base FGA row count")
    merged["Is_Transition_Backcast"] = merged["transition_label_operating"].astype(int)
    merged["Is_Transition_Rim_Backcast"] = merged["transition_rim_label_operating"].astype(int)
    merged["target_provenance"] = "frozen_sq_descriptor_classifier_backcast"
    return merged


def output_frame(merged: pd.DataFrame, target: str) -> pd.DataFrame:
    base_columns = [
        "game_id", "O1", "O2", "O3", "O4", "O5", "D1", "D2", "D3", "D4", "D5",
        "score", "period", "time_quarter", "away_score", "home_score", "score_margin",
        "event_num", "game_date", "Season",
    ]
    audit_columns = [
        "target_provenance", "feature_contract_version", "transition_calibration_scope",
        "transition_rim_calibration_scope", "transition_probability", "transition_rim_probability",
    ]
    return merged[base_columns + [target] + audit_columns].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=1997)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--transition-dir", type=Path, default=DEFAULT_TRANSITION)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    for year in range(args.start_year, args.end_year + 1):
        suffix = f"{year % 100:02d}"
        for phase, phase_suffix in (("rs", ""), ("ps", "_PS")):
            base_path = args.processed_dir / f"RIM_FREQ{suffix}{phase_suffix}.parquet"
            label_path = args.transition_dir / f"transition_backfill_{year}_{phase}.parquet"
            if not base_path.exists() or not label_path.exists():
                raise FileNotFoundError(f"Missing input: {base_path if not base_path.exists() else label_path}")
            destinations = {
                "Is_Transition_Backcast": args.processed_dir / f"TRANSITION_FREQ_BACKCAST{suffix}{phase_suffix}.parquet",
                "Is_Transition_Rim_Backcast": args.processed_dir / f"TRANSITION_RIM_BACKCAST{suffix}{phase_suffix}.parquet",
            }
            if not args.overwrite and all(path.exists() for path in destinations.values()):
                records.append({"year": year, "phase": phase, "status": "existing"})
                continue
            base = pd.read_parquet(base_path)
            labels = pd.read_parquet(label_path)
            merged = attach_backcast(base, labels, year=year, phase=phase)
            for target, destination in destinations.items():
                if args.overwrite or not destination.exists():
                    output_frame(merged, target).to_parquet(destination, index=False)
            records.append(
                {
                    "year": year,
                    "phase": phase,
                    "status": "built",
                    "rows": len(merged),
                    "transition_rate": float(merged["Is_Transition_Backcast"].mean()),
                    "transition_rim_rate": float(merged["Is_Transition_Rim_Backcast"].mean()),
                }
            )
    manifest = {
        "series": "retrospective_transition_backcast_mechanisms",
        "warning": "Pre-2025 targets use a classifier frozen from 2025-2026 provider labels; do not label as exact historical provider data.",
        "records": records,
    }
    manifest_path = args.processed_dir / "transition_backcast_mechanism_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
