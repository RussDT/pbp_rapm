"""Per-player-game WNBA zone shot lines (rim / mid / three) from modern PBP.

Contract:
- Source: ``wnba_test/GabrielNormalizedClean/WNBA{YY}[_PS].csv`` (2022+),
  parsed with the exact shot-row normalization and zone classification from
  ``build_wnba_pbp_shot_quality.build_shot_rows`` (rim = dunk/layup/tip
  family; mid = short_mid + long_mid + deep_two; three = shot_value 3).
- Output: one row per (game_id, person_id) with attempts and makes per zone,
  written to ``Results/gameday/estimated_skills/player_game_zone_lines.csv``.
- Reconciliation: joined against ``Results/gameday/player_game_logs.csv``
  played rows; reports per-season shares of exact FGA / FG3A / FG3M / FGM
  agreement, coverage of game-log games by PBP, and residual diagnostics in
  ``zone_reconciliation.json``. PBP-derived counts remain the zone source of
  truth for the skills panel; the report only quantifies the box seam.
- Classic (pre-2022) per-game files carry no game dates, so they are out of
  scope here; extending warm-up before 2022 needs a date source first.

Run: ``.venv/bin/python wnba_test/build_wnba_zone_game_lines.py [--start 2022] [--end 2026]``
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import build_wnba_pbp_shot_quality as sq
import wnba_gameday_common as common

OUT_DIR = common.GAMEDAY_DIR / "estimated_skills"
ZONE_MAP = {
    "rim": "rim",
    "short_mid": "mid",
    "long_mid": "mid",
    "deep_two": "mid",
    "three": "three",
}


def build_zone_lines(start_year: int, end_year: int) -> tuple[pd.DataFrame, list[dict]]:
    shots, sources = sq.load_shots(sq.DEFAULT_RAW_DIR, start_year, end_year)
    shots = shots[shots["shooter_id"].notna()].copy()
    shots["zone3"] = shots["shot_zone"].map(ZONE_MAP)
    if shots["zone3"].isna().any():
        bad = shots.loc[shots["zone3"].isna(), "shot_zone"].unique()
        raise SystemExit(f"unmapped shot zones: {bad}")
    shots["person_id"] = shots["shooter_id"].astype(int)

    grouped = (
        shots.groupby(
            ["season_end_year", "season_type", "game_id", "person_id", "zone3"],
            observed=True,
        )
        .agg(fga=("is_make", "size"), fgm=("is_make", "sum"))
        .reset_index()
    )
    lines = grouped.pivot_table(
        index=["season_end_year", "season_type", "game_id", "person_id"],
        columns="zone3",
        values=["fga", "fgm"],
        fill_value=0,
        aggfunc="sum",
    )
    lines.columns = [f"{zone}_{kind}" for kind, zone in lines.columns]
    lines = lines.reset_index()
    for col in ("rim_fga", "rim_fgm", "mid_fga", "mid_fgm", "three_fga", "three_fgm"):
        if col not in lines:
            lines[col] = 0
        lines[col] = lines[col].astype(int)
    names = (
        shots.groupby("person_id")["shooter_name"]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else "")
        .rename("player_name")
    )
    lines = lines.merge(names, on="person_id", how="left")
    lines = lines.rename(columns={"season_end_year": "season"})
    lines["season_type"] = lines["season_type"].str.lower()
    lines = lines.sort_values(["season", "game_id", "person_id"]).reset_index(drop=True)
    return lines, sources


def reconcile(lines: pd.DataFrame) -> dict:
    log = pd.read_csv(common.GAME_LOG_PATH)
    log = log[(log["played"] == 1)].copy()
    log["game_id"] = sq.normalize_game_id(log["game_id"])
    merged = log.merge(
        lines.drop(columns=["season", "season_type", "player_name"]),
        on=["game_id", "person_id"],
        how="left",
        indicator=True,
    )
    covered_games = set(lines["game_id"])
    merged["pbp_covered"] = merged["game_id"].isin(covered_games)
    zone_cols = ["rim_fga", "rim_fgm", "mid_fga", "mid_fgm", "three_fga", "three_fgm"]
    covered = merged[merged["pbp_covered"]].copy()
    covered[zone_cols] = covered[zone_cols].fillna(0).astype(int)
    covered["zone_fga"] = covered[["rim_fga", "mid_fga", "three_fga"]].sum(axis=1)
    covered["zone_fgm"] = covered[["rim_fgm", "mid_fgm", "three_fgm"]].sum(axis=1)

    report: dict[str, object] = {
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "n_zone_rows": int(len(lines)),
        "n_zone_games": int(lines["game_id"].nunique()),
        "n_log_played_rows": int(len(merged)),
        "n_log_games": int(merged["game_id"].nunique()),
        "n_log_games_without_pbp": int(
            merged.loc[~merged["pbp_covered"], "game_id"].nunique()
        ),
    }
    per_season = []
    for season, grp in covered.groupby("season"):
        n = len(grp)
        per_season.append(
            {
                "season": int(season),
                "rows": int(n),
                "games": int(grp["game_id"].nunique()),
                "fga_exact_share": float((grp["zone_fga"] == grp["fga"]).mean()),
                "fgm_exact_share": float((grp["zone_fgm"] == grp["fgm"]).mean()),
                "fg3a_exact_share": float((grp["three_fga"] == grp["fg3a"]).mean()),
                "fg3m_exact_share": float((grp["three_fgm"] == grp["fg3m"]).mean()),
                "fga_mean_abs_diff": float((grp["zone_fga"] - grp["fga"]).abs().mean()),
                "fga_total_pbp": int(grp["zone_fga"].sum()),
                "fga_total_box": int(grp["fga"].sum()),
                "fg3a_total_pbp": int(grp["three_fga"].sum()),
                "fg3a_total_box": int(grp["fg3a"].sum()),
                "rim_share_of_pbp_fga": float(
                    grp["rim_fga"].sum() / max(grp["zone_fga"].sum(), 1)
                ),
                "mid_share_of_pbp_fga": float(
                    grp["mid_fga"].sum() / max(grp["zone_fga"].sum(), 1)
                ),
            }
        )
    report["per_season"] = per_season
    missing_games = sorted(
        merged.loc[~merged["pbp_covered"], "game_id"].unique().tolist()
    )
    report["log_games_without_pbp"] = missing_games[:50]
    # Zone rows with no played game-log row (identity or seam mismatches).
    anti = lines.merge(
        log[["game_id", "person_id"]].assign(_in_log=1),
        on=["game_id", "person_id"],
        how="left",
    )
    report["zone_rows_without_log_row"] = int(anti["_in_log"].isna().sum())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=int, default=2022)
    parser.add_argument("--end", type=int, default=2026)
    args = parser.parse_args()

    lines, sources = build_zone_lines(args.start, args.end)
    report = reconcile(lines)
    report["sources"] = sources

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines.to_csv(OUT_DIR / "player_game_zone_lines.csv", index=False)
    (OUT_DIR / "zone_reconciliation.json").write_text(json.dumps(report, indent=2))

    print(f"\nwrote {OUT_DIR}/player_game_zone_lines.csv ({len(lines)} rows)")
    print(f"log games without PBP coverage: {report['n_log_games_without_pbp']}")
    for row in report["per_season"]:
        print(
            f"  {row['season']}: rows={row['rows']:>5} games={row['games']:>3} "
            f"fga_exact={row['fga_exact_share']:.4f} fg3a_exact={row['fg3a_exact_share']:.4f} "
            f"rim_share={row['rim_share_of_pbp_fga']:.3f} mid_share={row['mid_share_of_pbp_fga']:.3f}"
        )


if __name__ == "__main__":
    main()
