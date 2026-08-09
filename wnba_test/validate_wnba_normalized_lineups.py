"""Validate normalized WNBA event lineups and optionally write a clean solve surface."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT_DIR = Path("wnba_test/GabrielNormalized")
DEFAULT_OUTPUT_DIR = Path("wnba_test/GabrielNormalizedClean")


def positive_lineups(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    columns = [f"{prefix}_player{i}" for i in range(1, 6)]
    return (
        frame[columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(int)
    )


def duplicate_positive_player(row: Any) -> bool:
    values = [int(value) for value in row if int(value) > 0]
    return len(values) != len(set(values))


def mode_int(series: pd.Series) -> int:
    mode = pd.to_numeric(series, errors="coerce").dropna().astype(int).mode()
    return int(mode.iloc[0]) if not mode.empty else 0


def lineup_quality_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    away = positive_lineups(frame, "away")
    home = positive_lineups(frame, "home")
    team_id = pd.to_numeric(frame["teamid"], errors="coerce").fillna(0).astype(int)
    person_id = (
        pd.to_numeric(frame["personid"], errors="coerce").fillna(0).astype(int)
    )
    action_type = frame["actiontype"].fillna("").astype(str).str.lower()
    subtype = frame["subtype"].fillna("").astype(str).str.lower()

    missing_player = (away.le(0).any(axis=1) | home.le(0).any(axis=1))
    duplicate_side = pd.Series(
        [
            duplicate_positive_player(away_row)
            or duplicate_positive_player(home_row)
            for away_row, home_row in zip(away.to_numpy(), home.to_numpy())
        ],
        index=frame.index,
    )
    cross_side = pd.Series(
        [
            bool(set(away_row[away_row > 0]) & set(home_row[home_row > 0]))
            for away_row, home_row in zip(away.to_numpy(), home.to_numpy())
        ],
        index=frame.index,
    )

    away_team_ids = (
        frame.loc[frame["location"].eq("v") & team_id.gt(0)]
        .groupby("game_id")["teamid"]
        .agg(mode_int)
    )
    home_team_ids = (
        frame.loc[frame["location"].eq("h") & team_id.gt(0)]
        .groupby("game_id")["teamid"]
        .agg(mode_int)
    )
    away_team_event = frame["game_id"].map(away_team_ids).fillna(0).astype(int).eq(team_id)
    home_team_event = frame["game_id"].map(home_team_ids).fillna(0).astype(int).eq(team_id)
    actor_in_away = pd.Series(
        [
            int(actor) in set(lineup)
            for actor, lineup in zip(person_id, away.to_numpy())
        ],
        index=frame.index,
    )
    actor_in_home = pd.Series(
        [
            int(actor) in set(lineup)
            for actor, lineup in zip(person_id, home.to_numpy())
        ],
        index=frame.index,
    )
    actor_off_recorded_side = (
        person_id.gt(0)
        & team_id.gt(0)
        & action_type.ne("substitution")
        & (
            (away_team_event & ~actor_in_away)
            | (home_team_event & ~actor_in_home)
        )
    )
    admin_actor = action_type.eq("ejection") | (
        action_type.eq("foul") & subtype.eq("technical")
    )
    live_actor_off_lineup = actor_off_recorded_side & ~admin_actor
    invalid = missing_player | duplicate_side | cross_side | live_actor_off_lineup
    return {
        "missing_player": missing_player,
        "duplicate_side": duplicate_side,
        "cross_side": cross_side,
        "admin_actor_off_lineup": actor_off_recorded_side & admin_actor,
        "live_actor_off_lineup": live_actor_off_lineup,
        "invalid": invalid,
    }


def validate_file(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(path, dtype={"game_id": str}, low_memory=False)
    frame["game_id"] = (
        frame["game_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)
    )
    masks = lineup_quality_masks(frame)
    invalid_games = sorted(frame.loc[masks["invalid"], "game_id"].unique().tolist())
    clean = frame.loc[~frame["game_id"].isin(invalid_games)].copy()
    report = {
        "path": str(path),
        "rows": len(frame),
        "games": int(frame["game_id"].nunique()),
        "invalid_games": invalid_games,
        "invalid_game_count": len(invalid_games),
        "clean_rows": len(clean),
        "clean_games": int(clean["game_id"].nunique()),
        "row_issues": {
            key: int(mask.sum()) for key, mask in masks.items() if key != "invalid"
        },
    }
    return clean, report


def validate_directory(
    input_dir: Path,
    output_dir: Path,
    write_clean: bool,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    if write_clean:
        output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(input_dir.glob("WNBA*.csv")):
        clean, report = validate_file(path)
        reports[path.name] = report
        if write_clean:
            target = output_dir / path.name
            clean.to_csv(target, index=False)
            print(
                f"{path.name}: kept {report['clean_games']:,}/{report['games']:,} games "
                f"and {report['clean_rows']:,}/{report['rows']:,} rows -> {target}"
            )
    if not reports:
        raise RuntimeError(f"no normalized WNBA CSVs found in {input_dir}")
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "input_directory": str(input_dir),
        "clean_directory": str(output_dir) if write_clean else None,
        "files": reports,
        "totals": {
            "rows": sum(report["rows"] for report in reports.values()),
            "games": sum(report["games"] for report in reports.values()),
            "invalid_games": sum(
                report["invalid_game_count"] for report in reports.values()
            ),
            "clean_rows": sum(report["clean_rows"] for report in reports.values()),
            "clean_games": sum(
                report["clean_games"] for report in reports.values()
            ),
        },
        "policy": (
            "The full normalized archive is preserved. The clean solve surface excludes "
            "an entire game when any row has an incomplete five, a duplicate/cross-side "
            "player, or a non-admin event actor absent from the recorded team lineup."
        ),
    }
    report_dir = output_dir if write_clean else input_dir
    report_path = report_dir / "lineup_validation_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Validation report: {report_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--write-clean",
        action="store_true",
        help="Write season CSVs with every ambiguous game excluded",
    )
    args = parser.parse_args()
    validate_directory(args.input_dir, args.output_dir, args.write_clean)


if __name__ == "__main__":
    main()
