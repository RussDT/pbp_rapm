#!/usr/bin/env python3
"""
Build a combined master file for the three shot-family RAPM surfaces.

Usage:
    python build_master_frequency_rapm.py 23 26 ALL
    python build_master_frequency_rapm.py 21 26 ALL --timedecay --half-life 700
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).parent
PIPELINE_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PIPELINE_ROOT / "results"
MASTER_RESULTS_DIR = PIPELINE_ROOT / "master_results"

METRIC_CONFIG = {
    "rim_freq": {
        "label": "rim frequency",
        "sample_prefix": "rim_freq",
        "stems": ["rim_freq", "rimfreq"],
        "rename": {
            "off": "off_rim_freq_rapm",
            "def": "def_rim_freq_suppression_rapm",
            "net_rapm": "net_rim_freq_rapm",
        },
    },
    "midrange_freq": {
        "label": "midrange frequency",
        "sample_prefix": "midrange_freq",
        "stems": ["midrange_freq", "midrangefreq"],
        "rename": {
            "off": "off_midrange_freq_rapm",
            "def": "def_midrange_freq_suppression_rapm",
            "net_rapm": "net_midrange_freq_rapm",
        },
    },
    "three_freq": {
        "label": "three frequency",
        "sample_prefix": "three_freq",
        "stems": ["three_freq", "threefreq"],
        "rename": {
            "off": "off_three_freq_rapm",
            "def": "def_three_freq_suppression_rapm",
            "net_rapm": "net_three_freq_rapm",
        },
    },
    "rim_fg_pct": {
        "label": "rim fg%",
        "sample_prefix": "rim_fg_pct",
        "stems": ["rim_fg_pct", "rimfgpct"],
        "rename": {
            "off": "off_rim_fg_pct_rapm",
            "def": "def_rim_fg_pct_suppression_rapm",
            "net_rapm": "net_rim_fg_pct_rapm",
        },
    },
    "midrange_fg_pct": {
        "label": "midrange fg%",
        "sample_prefix": "midrange_fg_pct",
        "stems": ["midrange_fg_pct", "midrangefgpct"],
        "rename": {
            "off": "off_midrange_fg_pct_rapm",
            "def": "def_midrange_fg_pct_suppression_rapm",
            "net_rapm": "net_midrange_fg_pct_rapm",
        },
    },
    "three_fg_pct": {
        "label": "three fg%",
        "sample_prefix": "three_fg_pct",
        "stems": ["three_fg_pct", "threefgpct"],
        "rename": {
            "off": "off_three_fg_pct_rapm",
            "def": "def_three_fg_pct_suppression_rapm",
            "net_rapm": "net_three_fg_pct_rapm",
        },
    },
}


def build_suffix(start_year: int, end_year: int, season_type: str, *, timedecay: bool, half_life: int) -> str:
    suffix = f"{start_year:02d}_{end_year:02d}_{season_type.lower()}"
    if timedecay:
        suffix += f"_td{int(half_life)}"
    return suffix


def pick_latest(paths: list[Path]) -> Path:
    return max(paths, key=lambda path: path.stat().st_mtime)


def resolve_metric_file(metric_key: str, suffix: str) -> Path:
    matches: list[Path] = []
    for stem in METRIC_CONFIG[metric_key]["stems"]:
        target_name = f"{stem}_{suffix}_results.csv"
        matches.extend(RESULTS_DIR.glob(f"**/{target_name}"))
        matches.extend(MASTER_RESULTS_DIR.glob(f"**/{target_name}"))

    unique_matches = sorted(set(matches))
    if not unique_matches:
        searched = ", ".join(
            f"{stem}_{suffix}_results.csv" for stem in METRIC_CONFIG[metric_key]["stems"]
        )
        raise FileNotFoundError(
            f"Could not find {METRIC_CONFIG[metric_key]['label']} RAPM result for suffix '{suffix}'. "
            f"Searched for: {searched}"
        )

    chosen = pick_latest(unique_matches)
    logging.info("Using %s file: %s", METRIC_CONFIG[metric_key]["label"], chosen)
    return chosen


def load_metric(metric_key: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["player_id", "player_name", "off", "def", "net_rapm", "possessions", "off_poss", "def_poss"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    rename_map = dict(METRIC_CONFIG[metric_key]["rename"])
    sample_prefix = METRIC_CONFIG[metric_key]["sample_prefix"]
    rename_map.update({
        "possessions": f"{sample_prefix}_possessions",
        "off_poss": f"{sample_prefix}_off_poss",
        "def_poss": f"{sample_prefix}_def_poss",
    })

    keep_cols = list(rename_map.keys()) + ["player_id", "player_name"]
    return df[keep_cols].rename(columns=rename_map)


def collapse_sample_columns(df: pd.DataFrame) -> pd.DataFrame:
    sample_groups = {
        "possessions": [f"{metric}_possessions" for metric in ("rim_freq", "midrange_freq", "three_freq")],
        "off_poss": [f"{metric}_off_poss" for metric in ("rim_freq", "midrange_freq", "three_freq")],
        "def_poss": [f"{metric}_def_poss" for metric in ("rim_freq", "midrange_freq", "three_freq")],
    }

    for final_name, cols in sample_groups.items():
        present = [col for col in cols if col in df.columns]
        if not present:
            continue

        anchor = present[0]
        equal_mask = pd.Series(True, index=df.index)
        for col in present[1:]:
            equal_mask &= df[col].fillna(-1).eq(df[anchor].fillna(-1))

        if bool(equal_mask.all()):
            df[final_name] = df[anchor]
            df = df.drop(columns=present)

    return df


def build_master_dataframe(start_year: int, end_year: int, season_type: str, *, timedecay: bool, half_life: int) -> pd.DataFrame:
    suffix = build_suffix(start_year, end_year, season_type, timedecay=timedecay, half_life=half_life)
    merged: pd.DataFrame | None = None

    for metric_key in (
        "rim_freq",
        "midrange_freq",
        "three_freq",
        "rim_fg_pct",
        "midrange_fg_pct",
        "three_fg_pct",
    ):
        metric_path = resolve_metric_file(metric_key, suffix)
        metric_df = load_metric(metric_key, metric_path)
        if merged is None:
            merged = metric_df
        else:
            merged = merged.merge(metric_df, on=["player_id", "player_name"], how="outer")

    assert merged is not None
    merged = collapse_sample_columns(merged)

    ordered_cols = [
        "player_id",
        "player_name",
        "off_rim_freq_rapm",
        "def_rim_freq_suppression_rapm",
        "net_rim_freq_rapm",
        "off_midrange_freq_rapm",
        "def_midrange_freq_suppression_rapm",
        "net_midrange_freq_rapm",
        "off_three_freq_rapm",
        "def_three_freq_suppression_rapm",
        "net_three_freq_rapm",
        "off_rim_fg_pct_rapm",
        "def_rim_fg_pct_suppression_rapm",
        "net_rim_fg_pct_rapm",
        "off_midrange_fg_pct_rapm",
        "def_midrange_fg_pct_suppression_rapm",
        "net_midrange_fg_pct_rapm",
        "off_three_fg_pct_rapm",
        "def_three_fg_pct_suppression_rapm",
        "net_three_fg_pct_rapm",
    ]

    trailing_cols = [
        col for col in [
            "possessions",
            "off_poss",
            "def_poss",
            "rim_fg_pct_possessions",
            "rim_fg_pct_off_poss",
            "rim_fg_pct_def_poss",
            "midrange_fg_pct_possessions",
            "midrange_fg_pct_off_poss",
            "midrange_fg_pct_def_poss",
            "three_fg_pct_possessions",
            "three_fg_pct_off_poss",
            "three_fg_pct_def_poss",
        ] if col in merged.columns
    ]

    merged = merged[ordered_cols + trailing_cols]
    merged = merged.sort_values(["player_name", "player_id"], kind="stable").reset_index(drop=True)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a master shot-family RAPM CSV from rim/mid/three frequency and FG% outputs."
    )
    parser.add_argument("start_year", type=int, help="Two-digit start year, e.g. 23")
    parser.add_argument("end_year", type=int, help="Two-digit end year, e.g. 26")
    parser.add_argument("season_type", choices=["RS", "PS", "ALL"], help="Season type")
    parser.add_argument("--timedecay", "-td", action="store_true", help="Use td suffix when resolving input files")
    parser.add_argument("--half-life", type=int, default=700, help="Half-life for td suffix resolution")
    parser.add_argument("--output-file", type=Path, default=None, help="Optional explicit output file path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    suffix = build_suffix(
        args.start_year,
        args.end_year,
        args.season_type,
        timedecay=args.timedecay,
        half_life=args.half_life,
    )
    output_file = args.output_file or (MASTER_RESULTS_DIR / f"frequency_rapm_master_{suffix}.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    master_df = build_master_dataframe(
        args.start_year,
        args.end_year,
        args.season_type,
        timedecay=args.timedecay,
        half_life=args.half_life,
    )
    master_df.to_csv(output_file, index=False)

    logging.info("Wrote %s rows to %s", len(master_df), output_file)


if __name__ == "__main__":
    main()
