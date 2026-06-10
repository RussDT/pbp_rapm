#!/usr/bin/env python3
"""Build a RAPM parquet that uses actual FT outcomes instead of expected FT%.

This is an audit surface for raw on-court ORtg/DRtg comparisons. It keeps the
standard RAPM possession parsing and three-point handling, but free throw rows
use their observed made/missed point value.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from process_rapm_blocks import PROCESSED_DIR, RAW_DATA_DIR, process_rapm_py


def infer_year_and_season(raw_path: Path) -> tuple[int, str, str]:
    match = re.search(r"NBA(\d{2,4})(_PS)?\.parquet$", raw_path.name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not infer season from {raw_path.name}")

    year_text = match.group(1)
    ending_year = 2000 + int(year_text) if len(year_text) == 2 else int(year_text)
    season = f"{ending_year - 1}-{ending_year % 100:02d}"
    suffix = f"{ending_year % 100:02d}"
    ps_suffix = "_PS" if match.group(2) else ""
    return ending_year, season, f"{suffix}{ps_suffix}"


def default_raw_path(year: int, season_type: str) -> Path:
    suffix = "_PS" if season_type.upper() == "PS" else ""
    return RAW_DATA_DIR / f"NBA{year % 100:02d}{suffix}.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=26, help="Season end year, e.g. 26 or 2026")
    parser.add_argument("--season-type", choices=["RS", "PS"], default="PS")
    parser.add_argument("--input", type=Path, help="Optional raw NBA parquet path")
    parser.add_argument("--output", type=Path, help="Optional output parquet path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_path = args.input or default_raw_path(args.year, args.season_type)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw parquet not found: {raw_path}")

    ending_year, season, output_suffix = infer_year_and_season(raw_path)
    output_path = args.output or PROCESSED_DIR / f"RAPMnoFT{output_suffix}.parquet"

    result = process_rapm_py(
        raw_path,
        ending_year,
        season,
        o_luck=1.0,
        d_luck=1.0,
        ft_value_mode="actual",
    )
    if result is None or result[0] is None or result[0].empty:
        raise RuntimeError("RAPMnoFT build produced no rows")

    noft_df = result[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    noft_df.to_parquet(output_path, index=False)
    print(f"Wrote {len(noft_df):,} rows to {output_path}")


if __name__ == "__main__":
    main()
