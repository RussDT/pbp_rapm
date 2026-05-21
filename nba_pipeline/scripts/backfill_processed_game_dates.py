#!/usr/bin/env python3
"""
Backfill missing game_date values in processed RAPM metric parquets.

Time-decay RAPM relies on processed `game_date`. Some processed surfaces can
have null dates even when game IDs are valid. This script fills missing dates
from a schedule file and, secondarily, from already dated processed surfaces.
Legacy pre-2014 raw game IDs can also encode the calendar date directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "processed"
RESULTS_DIR = ROOT / "results"
DEFAULT_SCHEDULE_URL = "https://raw.githubusercontent.com/gabriel1200/shot_data/refs/heads/master/game_dates.csv"
DEFAULT_METRICS = ["RAPM", "TOV", "TS", "REB"]
DEFAULT_YEARS = [24, 25, 26]


def normalize_game_id(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    return str(int(digits))


def infer_legacy_game_date(value) -> str | None:
    """
    Infer dates from older 9-digit legacy IDs like 221101004.

    Those IDs use a calendar code where the first two digits are
    `calendar_year - 1980`, followed by MMDD and a game sequence.
    Example: 221101004 -> 2002-11-01.
    """
    text = normalize_game_id(value)
    if text is None or len(text) != 9:
        return None
    try:
        year = 1980 + int(text[:2])
        month = int(text[2:4])
        day = int(text[4:6])
        date = pd.Timestamp(year=year, month=month, day=day)
    except Exception:
        return None
    return date.strftime("%Y-%m-%d")


def load_schedule_dates(schedule: str) -> pd.DataFrame:
    schedule_df = pd.read_csv(schedule)
    required = {"GAME_ID", "date"}
    missing = required - set(schedule_df.columns)
    if missing:
        raise ValueError(f"Schedule missing required columns: {sorted(missing)}")

    out = schedule_df[["GAME_ID", "date"]].copy()
    out["game_id_norm"] = out["GAME_ID"].map(normalize_game_id)
    out["game_date"] = pd.to_datetime(out["date"].astype(str), format="%Y%m%d", errors="coerce")
    out = out.dropna(subset=["game_id_norm", "game_date"])
    out["game_date"] = out["game_date"].dt.strftime("%Y-%m-%d")
    return out[["game_id_norm", "game_date"]].drop_duplicates("game_id_norm", keep="first")


def load_existing_processed_dates(metrics: list[str], years: list[int], processed_dir: Path) -> pd.DataFrame:
    frames = []
    for metric in metrics:
        for year in years:
            path = processed_dir / f"{metric}{year}.parquet"
            if not path.exists():
                continue
            cols = pd.read_parquet(path, columns=["game_id", "game_date"])
            dates = pd.to_datetime(cols["game_date"], errors="coerce")
            cols = cols.loc[dates.notna(), ["game_id", "game_date"]].copy()
            if cols.empty:
                continue
            cols["game_id_norm"] = cols["game_id"].map(normalize_game_id)
            cols["game_date"] = pd.to_datetime(cols["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
            frames.append(cols[["game_id_norm", "game_date"]])

    if not frames:
        return pd.DataFrame(columns=["game_id_norm", "game_date"])
    return pd.concat(frames, ignore_index=True).dropna().drop_duplicates("game_id_norm", keep="first")


def build_date_map(schedule: str, metrics: list[str], years: list[int], processed_dir: Path) -> pd.DataFrame:
    schedule_dates = load_schedule_dates(schedule)
    existing_dates = load_existing_processed_dates(metrics, years, processed_dir)
    combined = pd.concat([existing_dates, schedule_dates], ignore_index=True)
    return combined.dropna().drop_duplicates("game_id_norm", keep="first")


def backfill_file(
    path: Path,
    date_map: pd.DataFrame,
    write: bool,
    infer_legacy_id_dates: bool = False,
) -> dict[str, object]:
    df = pd.read_parquet(path)
    if "game_id" not in df.columns:
        raise ValueError(f"{path} is missing game_id")
    if "game_date" not in df.columns:
        df["game_date"] = pd.NA

    before = int(pd.to_datetime(df["game_date"], errors="coerce").notna().sum())
    missing_mask = pd.to_datetime(df["game_date"], errors="coerce").isna()

    lookup = date_map.set_index("game_id_norm")["game_date"]
    normalized_ids = df.loc[missing_mask, "game_id"].map(normalize_game_id)
    fill_values = normalized_ids.map(lookup)

    schedule_fill_count = int(fill_values.notna().sum())
    legacy_fill_count = 0
    if infer_legacy_id_dates:
        still_missing = fill_values.isna()
        legacy_values = df.loc[missing_mask, "game_id"].map(infer_legacy_game_date)
        legacy_fill_count = int((still_missing & legacy_values.notna()).sum())
        fill_values = fill_values.mask(still_missing, legacy_values)

    fill_count = int(fill_values.notna().sum())

    if write and fill_count:
        df.loc[missing_mask, "game_date"] = fill_values.values
        df.to_parquet(path, index=False)

    after = before + fill_count if write else before
    return {
        "file": str(path),
        "rows": len(df),
        "dates_before": before,
        "filled": fill_count,
        "schedule_filled": schedule_fill_count,
        "legacy_inferred_filled": legacy_fill_count,
        "dates_after": after,
        "write": write,
    }


def iter_target_paths(processed_dir: Path, metrics: list[str], years: list[int], all_files: bool) -> list[Path]:
    if all_files:
        return sorted(processed_dir.glob("*.parquet"))

    paths = []
    for metric in metrics:
        for year in years:
            path = processed_dir / f"{metric}{year}.parquet"
            paths.append(path)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill processed RAPM metric game_date columns.")
    parser.add_argument("--schedule", default=DEFAULT_SCHEDULE_URL)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--all-files", action="store_true", help="Scan every parquet in the processed directory.")
    parser.add_argument(
        "--infer-legacy-id-dates",
        action="store_true",
        help="Infer dates for legacy 9-digit pre-2014 game IDs when schedule lookup misses.",
    )
    parser.add_argument("--write", action="store_true", help="Rewrite parquet files in place.")
    parser.add_argument(
        "--report",
        type=Path,
        default=RESULTS_DIR / "processed_game_date_backfill_report.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    date_map = build_date_map(args.schedule, args.metrics, args.years, args.processed_dir)
    if date_map.empty:
        raise RuntimeError("No schedule dates available for backfill")

    rows = []
    for path in iter_target_paths(args.processed_dir, args.metrics, args.years, args.all_files):
        if not path.exists():
            rows.append({"file": str(path), "error": "missing"})
            continue
        try:
            rows.append(backfill_file(path, date_map, args.write, args.infer_legacy_id_dates))
        except Exception as exc:
            rows.append({"file": str(path), "error": str(exc), "write": args.write})

    report = pd.DataFrame(rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.report, index=False)
    print(report.to_string(index=False))
    print(f"Saved report: {args.report}")


if __name__ == "__main__":
    main()
