#!/usr/bin/env python3
"""Build isolated WNBA net-DECOMP targets with a ShotQuality EV split.

The canonical processed files remain unchanged.  This adapter maps the
chronological first FGA in each possession to the terminal FIRST_CHANCE row,
then splits only the MID and THREE net-FG children into:

    net opportunity EV + (net actual - net opportunity EV)

The two children exactly reproduce the original net-FG child.  Rim, shot
frequency, FT, turnover, and second-chance targets are untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nba_pipeline.scripts.build_vpm_target_swap import _classify_fgas
from nba_pipeline.scripts.process_rapm_blocks.common import normalize_game_id_series


WNBA_ROOT = ROOT / "wnba_test"
DEFAULT_PROCESSED_DIR = WNBA_ROOT / "Processed"
DEFAULT_RAW_DIR = WNBA_ROOT / "GabrielNormalizedClean"
DEFAULT_SQ_PATH = (
    WNBA_ROOT
    / "Results"
    / "pbp_shot_quality"
    / "wnba_pbp_shot_quality_forward.parquet"
)
DEFAULT_OUTPUT_DIR = WNBA_ROOT / "Results" / "wnba_net_sq_targets"

NET_SHOT_COLUMNS = [
    "FC_NET_RIM_EFG_FREQ_Diff",
    "FC_NET_RIM_EFG_FG_Diff",
    "FC_NET_MID_EFG_FREQ_Diff",
    "FC_NET_MID_EFG_FG_Diff",
    "FC_NET_THREE_EFG_FREQ_Diff",
    "FC_NET_THREE_EFG_FG_Diff",
]
MID_TARGET_COLUMN = "FC_NET_MID_EFG_FG_Diff"
THREE_TARGET_COLUMN = "FC_NET_THREE_EFG_FG_Diff"
MID_RESIDUAL_COLUMN = "FC_NET_MID_SQ_RESID_Diff"
THREE_RESIDUAL_COLUMN = "FC_NET_THREE_SQ_RESID_Diff"
ATTACHED_COLUMN = "FC_NET_SQ_ATTACHED"
REQUIRED_FIRST_CHANCE_COLUMNS = {
    "game_id",
    "event_num",
    "FC_CONT_KAPPA_Diff",
    "FC_TS_NET_Diff",
    *NET_SHOT_COLUMNS,
}
REQUIRED_RAW_COLUMNS = {
    "game_id",
    "event_num",
    "event_type",
    "home_description",
    "visitor_description",
}


def parse_years(value: str) -> list[int]:
    text = value.strip()
    if "-" in text and "," not in text:
        start, end = (int(part) for part in text.split("-", maxsplit=1))
        if end < start:
            raise argparse.ArgumentTypeError("year range end must be >= start")
        return list(range(start, end + 1))
    years = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not years:
        raise argparse.ArgumentTypeError("at least one year is required")
    return years


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def prepare_sq_table(sq: pd.DataFrame, ev_column: str) -> pd.DataFrame:
    """Normalize one leakage-safe per-shot EV column to mid/three keys."""
    required = {"game_id", "event_num", "shot_zone", "shot_value", ev_column}
    _require_columns(sq, required, "WNBA SQ parquet")
    out = sq[list(required)].copy()
    out["game_id"] = normalize_game_id_series(out["game_id"])
    out["event_num"] = pd.to_numeric(out["event_num"], errors="coerce")
    out["shot_value"] = pd.to_numeric(out["shot_value"], errors="coerce")
    out["ev_pts"] = pd.to_numeric(out[ev_column], errors="coerce")
    out["zone"] = np.where(
        out["shot_zone"].astype(str).str.lower().eq("three"),
        "three",
        "mid",
    )
    out = out.dropna(subset=["event_num", "shot_value", "ev_pts"])
    out["event_num"] = out["event_num"].astype(np.int64)
    out["p_make"] = (out["ev_pts"] / out["shot_value"]).clip(0.0, 1.0)
    out = out[["game_id", "event_num", "zone", "ev_pts", "p_make"]]

    duplicates = out.duplicated(["game_id", "event_num"], keep=False)
    if duplicates.any():
        conflict = (
            out.loc[duplicates]
            .groupby(["game_id", "event_num"], sort=False)
            .agg(zones=("zone", "nunique"), evs=("ev_pts", "nunique"))
        )
        conflict = conflict[(conflict["zones"] > 1) | (conflict["evs"] > 1)]
        if not conflict.empty:
            examples = [f"{game}/{event}" for game, event in conflict.index[:5]]
            raise ValueError(
                "WNBA SQ parquet has conflicting duplicate game/event keys: "
                + ", ".join(examples)
            )
        out = out.drop_duplicates(["game_id", "event_num"], keep="first")
    return out.reset_index(drop=True)


def assign_first_fga_to_possessions(
    first_chance: pd.DataFrame,
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Map raw first-FGA keys to terminal possession rows in clock-first order."""
    _require_columns(first_chance, REQUIRED_FIRST_CHANCE_COLUMNS, "FIRST_CHANCE")
    _require_columns(raw, REQUIRED_RAW_COLUMNS, "raw WNBA PBP")

    fc = first_chance[["game_id", "event_num"]].copy()
    fc["game_id"] = normalize_game_id_series(fc["game_id"])
    fc["event_num"] = pd.to_numeric(fc["event_num"], errors="coerce")
    if fc["event_num"].isna().any():
        raise ValueError("FIRST_CHANCE has missing/non-numeric event_num values")
    fc["event_num"] = fc["event_num"].astype(np.int64)
    fc["_row"] = np.arange(len(fc), dtype=np.int64)
    if fc.duplicated(["game_id", "event_num"]).any():
        raise ValueError("FIRST_CHANCE game/event terminal keys must be unique")

    raw_work = raw.copy()
    raw_work["game_id"] = normalize_game_id_series(raw_work["game_id"])
    raw_work["event_num"] = pd.to_numeric(raw_work["event_num"], errors="coerce")
    raw_work = raw_work.dropna(subset=["event_num"])
    raw_work["event_num"] = raw_work["event_num"].astype(np.int64)
    # Gabriel's WNBA feed stores the whole period clock in
    # seconds_remaining_quarter.  The shared clock-first sorter expects both
    # minute and second fields, so make the absent minute contribution explicit.
    if "minute_remaining_quarter" not in raw_work.columns:
        raw_work["minute_remaining_quarter"] = 0
    fga, ordered = _classify_fgas(raw_work)

    raw_keys = ordered[["game_id", "event_num", "_raw_position"]].drop_duplicates(
        ["game_id", "event_num"], keep="first"
    )
    terminals = fc.merge(
        raw_keys,
        on=["game_id", "event_num"],
        how="left",
        validate="one_to_one",
    )

    assignments: list[pd.DataFrame] = []
    for game_id, game_fga in fga.groupby("game_id", sort=False):
        game_term = terminals[
            terminals["game_id"].eq(game_id) & terminals["_raw_position"].notna()
        ].sort_values("_raw_position")
        if game_term.empty:
            continue
        terminal_positions = game_term["_raw_position"].to_numpy(dtype=np.int64)
        fga_positions = game_fga["_raw_position"].to_numpy(dtype=np.int64)
        next_terminal = np.searchsorted(terminal_positions, fga_positions, side="left")
        valid = next_terminal < len(terminal_positions)
        if not valid.any():
            continue
        assigned = game_fga.loc[valid].copy()
        assigned["_row"] = game_term["_row"].to_numpy(dtype=np.int64)[
            next_terminal[valid]
        ]
        assignments.append(assigned)

    columns = ["_row", "game_id", "event_num", "_raw_position", "_zone", "_actual_pts"]
    if assignments:
        attached = pd.concat(assignments, ignore_index=True)
        attached = attached.sort_values("_raw_position", kind="stable")
        attached = attached.drop_duplicates("_row", keep="first")
        attached = attached[columns].reset_index(drop=True)
    else:
        attached = pd.DataFrame(columns=columns)
    return attached, {
        "terminal_rows": int(len(fc)),
        "terminal_keys_found_in_raw": int(terminals["_raw_position"].notna().sum()),
        "first_fgas_assigned": int(len(attached)),
    }


def _recover_zone_kappa(
    assignments: pd.DataFrame,
    kappa_by_row: np.ndarray,
    zone: str,
) -> tuple[float, float]:
    miss_rows = assignments.loc[
        assignments["_zone"].eq(zone) & assignments["_actual_pts"].eq(0.0),
        "_row",
    ].to_numpy(dtype=np.int64)
    values = kappa_by_row[miss_rows]
    values = values[values > 0]
    if not values.size:
        raise ValueError(f"no positive continuation kappa found for {zone}")
    rounded = np.round(values, 9)
    unique, counts = np.unique(rounded, return_counts=True)
    selected = float(unique[np.argmax(counts)])
    return selected, float(np.mean(np.isclose(values, selected, atol=1e-8)))


def attach_sq_net_targets(
    first_chance: pd.DataFrame,
    raw: pd.DataFrame,
    sq: pd.DataFrame,
    *,
    ev_column: str = "shot_context_ev",
    identity_tolerance: float = 0.02,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Attach the WNBA net EV/residual split and prove row closure."""
    prepared_sq = prepare_sq_table(sq, ev_column)
    assignments, stats = assign_first_fga_to_possessions(first_chance, raw)
    out = first_chance.reset_index(drop=True).copy()
    out["game_id"] = normalize_game_id_series(out["game_id"])
    out[MID_RESIDUAL_COLUMN] = 0.0
    out[THREE_RESIDUAL_COLUMN] = 0.0
    out[ATTACHED_COLUMN] = False

    original_sum = (
        out[NET_SHOT_COLUMNS]
        .apply(pd.to_numeric, errors="coerce")
        .sum(axis=1)
        .to_numpy(dtype=np.float64)
    )
    kappa_by_row = pd.to_numeric(
        out["FC_CONT_KAPPA_Diff"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if np.isnan(kappa_by_row).any():
        raise ValueError("FC_CONT_KAPPA_Diff contains missing values")

    keyed_sq = prepared_sq.rename(columns={"zone": "_ev_zone"})
    assignments = assignments.merge(
        keyed_sq,
        on=["game_id", "event_num"],
        how="left",
        validate="one_to_one",
    )

    for zone, target_column, residual_column in [
        ("mid", MID_TARGET_COLUMN, MID_RESIDUAL_COLUMN),
        ("three", THREE_TARGET_COLUMN, THREE_RESIDUAL_COLUMN),
    ]:
        kappa, agreement = _recover_zone_kappa(assignments, kappa_by_row, zone)
        zone_rows = assignments[assignments["_zone"].eq(zone)].set_index("_row")
        aligned = zone_rows.reindex(out.index)
        actual = pd.to_numeric(aligned["_actual_pts"], errors="coerce").to_numpy(float)
        ev = pd.to_numeric(aligned["ev_pts"], errors="coerce").to_numpy(float)
        p_make = pd.to_numeric(aligned["p_make"], errors="coerce").to_numpy(float)
        target = pd.to_numeric(out[target_column], errors="coerce").to_numpy(float)
        candidate = ~np.isnan(actual)
        net_actual = np.where(actual == 0.0, kappa, actual)
        if candidate.any():
            replacement_constant = float(np.nanmedian(net_actual[candidate] - target[candidate]))
            identity_ok = candidate & (
                np.abs((net_actual - target) - replacement_constant) < identity_tolerance
            )
        else:
            replacement_constant = float("nan")
            identity_ok = np.zeros(len(out), dtype=bool)
        zone_match = (
            aligned["_ev_zone"].astype("string").eq(zone).fillna(False).to_numpy(bool)
        )
        swap = identity_ok & ~np.isnan(ev) & ~np.isnan(p_make) & zone_match
        net_ev = ev + kappa * (1.0 - p_make)
        residual = np.zeros(len(out), dtype=float)
        residual[swap] = net_actual[swap] - net_ev[swap]
        out[target_column] = target - residual
        out[residual_column] = residual
        out.loc[swap, ATTACHED_COLUMN] = True
        stats.update(
            {
                f"{zone}_kappa": kappa,
                f"{zone}_kappa_agreement": agreement,
                f"{zone}_first_chance_fgas": int(candidate.sum()),
                f"{zone}_identity_ok": int(identity_ok.sum()),
                f"{zone}_sq_matches": int(swap.sum()),
                f"{zone}_replacement_constant": replacement_constant,
            }
        )

    swapped_sum = (
        out[NET_SHOT_COLUMNS].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        + pd.to_numeric(out[MID_RESIDUAL_COLUMN], errors="coerce")
        + pd.to_numeric(out[THREE_RESIDUAL_COLUMN], errors="coerce")
    ).to_numpy(float)
    gap = swapped_sum - original_sum
    stats.update(
        {
            "ev_column": ev_column,
            "rows": int(len(out)),
            "sq_attached_rows": int(out[ATTACHED_COLUMN].sum()),
            "max_abs_net_sq_closure_gap": float(np.max(np.abs(gap))),
            "sum_net_sq_closure_gap": float(gap.sum()),
        }
    )
    if stats["max_abs_net_sq_closure_gap"] > 1e-9:
        raise ValueError(
            "WNBA net SQ target split broke row closure: "
            f"max gap={stats['max_abs_net_sq_closure_gap']:.3g}"
        )
    return out, stats


def build_target_files(
    years: Iterable[int],
    *,
    processed_dir: Path,
    raw_dir: Path,
    sq_path: Path,
    output_dir: Path,
    ev_column: str,
    include_playoffs: bool,
) -> pd.DataFrame:
    sq = pd.read_parquet(sq_path)
    target_dir = output_dir / ev_column
    target_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, object]] = []
    for year in years:
        for suffix, season_type in [("", "RS"), ("_PS", "PS")]:
            if suffix and not include_playoffs:
                continue
            fc_path = processed_dir / f"FIRST_CHANCE{year % 100:02d}{suffix}.csv"
            raw_path = raw_dir / f"WNBA{year % 100:02d}{suffix}.csv"
            if not fc_path.exists() or not raw_path.exists():
                continue
            first_chance = pd.read_csv(fc_path)
            raw = pd.read_csv(raw_path)
            swapped, stats = attach_sq_net_targets(
                first_chance,
                raw,
                sq,
                ev_column=ev_column,
            )
            destination = target_dir / fc_path.name
            swapped.to_csv(destination, index=False)
            reports.append(
                {
                    "year": year,
                    "season_type": season_type,
                    "source": str(fc_path),
                    "output": str(destination),
                    **stats,
                }
            )
    report = pd.DataFrame(reports)
    report.to_csv(target_dir / "attach_report.csv", index=False)
    manifest = {
        "status": "research_only",
        "ev_column": ev_column,
        "sq_path": str(sq_path),
        "years": sorted({int(year) for year in years}),
        "include_playoffs": bool(include_playoffs),
        "rows": int(report["rows"].sum()) if not report.empty else 0,
        "sq_attached_rows": int(report["sq_attached_rows"].sum()) if not report.empty else 0,
        "max_abs_net_sq_closure_gap": (
            float(report["max_abs_net_sq_closure_gap"].max()) if not report.empty else 0.0
        ),
    }
    (target_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=parse_years, default=parse_years("2022-2026"))
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--sq-path", type=Path, default=DEFAULT_SQ_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--ev-column",
        choices=["shot_context_ev", "shot_quality_prior_ev"],
        default="shot_context_ev",
    )
    parser.add_argument("--exclude-playoffs", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_target_files(
        args.years,
        processed_dir=args.processed_dir,
        raw_dir=args.raw_dir,
        sq_path=args.sq_path,
        output_dir=args.output_dir,
        ev_column=args.ev_column,
        include_playoffs=not args.exclude_playoffs,
    )
    print(report.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
