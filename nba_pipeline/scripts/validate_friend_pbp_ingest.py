#!/usr/bin/env python3
"""
Validate a friend's team-season PBP parquet against the repo raw contract.

This harness is intentionally side-effect-light:
- writes the official BKN subset raw parquet for inspection
- writes converted friend raw + processed artifacts under validation/
- runs a small set of RAPM solves into validation/results/

It does not overwrite nba_pipeline/processed or nba_pipeline/results.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_ROOT.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from process_rapm_blocks import (  # noqa: E402
    process_assist_points_py,
    process_midrange_fg_pct_py,
    process_midrange_freq_py,
    process_rapm_py,
    process_reb_py,
    process_rim_fg_pct_py,
    process_rim_freq_py,
    process_three_fg_pct_py,
    process_three_freq_py,
    process_tov_py,
    process_ts_py,
)

import rapm as rapm_solver  # noqa: E402


DEFAULT_FRIEND_RAW = PIPELINE_ROOT / "raw_data" / "BKN_2020_rs.parquet"
DEFAULT_OFFICIAL_RAW = PIPELINE_ROOT / "raw_data" / "NBA20.parquet"
DEFAULT_OUTPUT_ROOT = PIPELINE_ROOT / "validation" / "friend_ingest"
SEASON_END_YEAR = 2020
SEASON_LABEL = "2019-20"


PROCESSORS = {
    "RAPM": lambda path: process_rapm_py(path, SEASON_END_YEAR, SEASON_LABEL, o_luck=0.0, d_luck=0.0)[0],
    "TS": lambda path: process_ts_py(path, SEASON_END_YEAR, SEASON_LABEL),
    "TOV": lambda path: process_tov_py(path, SEASON_END_YEAR, SEASON_LABEL),
    "REB": lambda path: process_reb_py(path, SEASON_END_YEAR, SEASON_LABEL),
    "RIM_FREQ": lambda path: process_rim_freq_py(path, SEASON_END_YEAR, SEASON_LABEL),
    "RIM_FG_PCT": lambda path: process_rim_fg_pct_py(path, SEASON_END_YEAR, SEASON_LABEL),
    "THREE_FREQ": lambda path: process_three_freq_py(path, SEASON_END_YEAR, SEASON_LABEL),
    "THREE_FG_PCT": lambda path: process_three_fg_pct_py(path, SEASON_END_YEAR, SEASON_LABEL),
    "MIDRANGE_FREQ": lambda path: process_midrange_freq_py(path, SEASON_END_YEAR, SEASON_LABEL),
    "MIDRANGE_FG_PCT": lambda path: process_midrange_fg_pct_py(path, SEASON_END_YEAR, SEASON_LABEL),
    "ASSIST_POINTS": lambda path: process_assist_points_py(path, SEASON_END_YEAR, SEASON_LABEL),
}

SOLVE_METRICS = ["RAPM", "TS", "TOV", "REB"]


def parse_player_pipe(value) -> list[float]:
    if pd.isna(value):
        return []
    out: list[float] = []
    for part in str(value).split("|"):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            continue
    return out


def parse_clock_seconds(clock_display) -> int:
    if pd.isna(clock_display):
        return 0
    text = str(clock_display)
    match = re.search(r"(\d+):(\d+)", text)
    if not match:
        return 0
    return int(match.group(1)) * 60 + int(match.group(2))


def infer_bkn_games(official_df: pd.DataFrame, friend_game_ids: set[int], team: str) -> pd.DataFrame:
    rows = []
    for game_id, game in official_df[official_df["game_id"].isin(friend_game_ids)].groupby("game_id"):
        home_events = game[
            game["home_description"].notna() & game["player1_team_abbreviation"].notna()
        ]["player1_team_abbreviation"]
        away_events = game[
            game["visitor_description"].notna() & game["player1_team_abbreviation"].notna()
        ]["player1_team_abbreviation"]
        home = home_events.mode().iat[0] if not home_events.mode().empty else None
        away = away_events.mode().iat[0] if not away_events.mode().empty else None
        if home == team or away == team:
            rows.append({"game_id": int(game_id), "home_team": home, "away_team": away})
    return pd.DataFrame(rows).sort_values("game_id").reset_index(drop=True)


def build_game_side_player_sets(official_subset: pd.DataFrame) -> dict[int, dict[str, set[float]]]:
    side_sets: dict[int, dict[str, set[float]]] = {}
    for game_id, game in official_subset.groupby("game_id"):
        home_cols = [f"home_player{i}" for i in range(1, 6)]
        away_cols = [f"away_player{i}" for i in range(1, 6)]
        side_sets[int(game_id)] = {
            "home": set(pd.to_numeric(game[home_cols].stack(), errors="coerce").dropna().astype(float)),
            "away": set(pd.to_numeric(game[away_cols].stack(), errors="coerce").dropna().astype(float)),
        }
    return side_sets


def event_type_from_friend(row: pd.Series) -> int:
    action = str(row.get("actionType", "")).lower()
    if action in {"2pt", "3pt"}:
        return 1 if str(row.get("shotResult", "")).lower() == "made" else 2
    if action == "freethrow":
        return 3
    if action == "rebound":
        return 4
    if action == "turnover":
        return 5
    if action == "foul":
        return 6
    if action == "substitution":
        return 8
    if action == "timeout":
        return 9
    if action == "jumpball":
        return 10
    if action == "ejection":
        return 11
    if action == "violation":
        return 7
    if action == "period":
        desc = str(row.get("description", "")).lower()
        return 13 if "end of" in desc else 12
    return 14


def event_action_type_from_friend(row: pd.Series) -> int:
    desc = str(row.get("description", "")).lower()
    action = str(row.get("actionType", "")).lower()
    if action == "3pt":
        return 1
    if "dunk" in desc:
        return 7
    if "layup" in desc:
        if "driving" in desc:
            return 6
        if "running" in desc:
            return 41
        return 5
    if "tip" in desc:
        return 97
    return 0


def normalize_friend_description(row: pd.Series) -> str:
    desc = "" if pd.isna(row.get("description")) else str(row.get("description"))
    action = str(row.get("actionType", "")).lower()
    if action in {"2pt", "3pt"} and str(row.get("shotResult", "")).lower() == "made" and "PTS" not in desc:
        points = "3" if action == "3pt" else "2"
        desc = f"{desc} ({points} PTS)"
    return desc


def assign_home_away_lineup(row: pd.Series, side_sets: dict[int, dict[str, set[float]]]) -> tuple[list[float], list[float]]:
    home_cols = [f"home_player{i}" for i in range(1, 6)]
    away_cols = [f"away_player{i}" for i in range(1, 6)]
    if all(col in row.index for col in home_cols + away_cols):
        home_players = [
            float(row[col])
            for col in home_cols
            if pd.notna(row.get(col))
        ]
        away_players = [
            float(row[col])
            for col in away_cols
            if pd.notna(row.get(col))
        ]
        if home_players or away_players:
            return home_players, away_players

    game_sets = side_sets.get(int(row["game_id"]), {"home": set(), "away": set()})
    home_set = game_sets["home"]
    off = parse_player_pipe(row.get("off_players_on"))
    deff = parse_player_pipe(row.get("def_players_on"))

    off_home_overlap = len(set(off) & home_set)
    def_home_overlap = len(set(deff) & home_set)
    if off_home_overlap >= def_home_overlap:
        home_players, away_players = off, deff
    else:
        home_players, away_players = deff, off

    return home_players[:5], away_players[:5]


def infer_offense_side(row: pd.Series, side_sets: dict[int, dict[str, set[float]]]) -> str | None:
    """Infer whether the friend row's offense lineup is home or away."""
    game_sets = side_sets.get(int(row["game_id"]), {"home": set(), "away": set()})
    home_set = game_sets["home"]
    away_set = game_sets["away"]
    off = set(parse_player_pipe(row.get("off_players_on")))
    if not off:
        return None
    home_overlap = len(off & home_set)
    away_overlap = len(off & away_set)
    if home_overlap > away_overlap:
        return "home"
    if away_overlap > home_overlap:
        return "away"
    return None


def add_synthetic_admin_rows(friend_df: pd.DataFrame) -> pd.DataFrame:
    """Insert omitted non-scoring admin rows that alter possession parsing."""
    friend_df = friend_df.sort_values(["game_id", "period", "actionNumber"], kind="stable").copy()
    synthetic_rows: list[dict] = []
    used_keys = set(zip(friend_df["game_id"], friend_df["period"], friend_df["actionNumber"]))

    for (_, _), game_period in friend_df.groupby(["game_id", "period"], sort=False):
        rows = list(game_period.to_dict("records"))
        for idx, row in enumerate(rows):
            action = str(row.get("actionType", "")).lower()
            desc = "" if pd.isna(row.get("description")) else str(row.get("description"))
            next_row = rows[idx + 1] if idx + 1 < len(rows) else None
            next_next_row = rows[idx + 2] if idx + 2 < len(rows) else None
            if next_row is None:
                continue

            current_action_num = float(row["actionNumber"])
            next_action_num = float(next_row["actionNumber"])
            candidate_action_num = current_action_num + 1
            if (row["game_id"], row["period"], candidate_action_num) in used_keys or candidate_action_num >= next_action_num:
                candidate_action_num = (current_action_num + next_action_num) / 2.0

            is_missed_final_ft = (
                action == "freethrow"
                and "MISS" in desc.upper()
                and re.search(r"\b(1 of 1|2 of 2|3 of 3)\b", desc, re.IGNORECASE)
            )
            next_desc = "" if pd.isna(next_row.get("description")) else str(next_row.get("description"))
            delayed_team_def_rebound = (
                str(next_row.get("actionType", "")).lower() == "rebound"
                and "TEAM defensive REBOUND" in next_desc
                and parse_clock_seconds(row.get("clock_display")) - parse_clock_seconds(next_row.get("clock_display")) >= 3
            )
            if is_missed_final_ft and delayed_team_def_rebound:
                synth = dict(next_row)
                synth["actionNumber"] = candidate_action_num
                synth["actionType"] = "jumpball"
                synth["team"] = np.nan
                synth["teamId"] = np.nan
                synth["person_id"] = 0
                synth["playerName"] = None
                synth["description"] = "Synthetic Jump Ball before delayed team rebound"
                synthetic_rows.append(synth)
                used_keys.add((row["game_id"], row["period"], candidate_action_num))

    if not synthetic_rows:
        return friend_df
    return pd.concat([friend_df, pd.DataFrame(synthetic_rows)], ignore_index=True, sort=False)


def convert_friend_to_repo_raw(
    friend_df: pd.DataFrame,
    game_sides: pd.DataFrame,
    side_sets: dict[int, dict[str, set[float]]],
    allow_partial_lineups: bool = False,
) -> pd.DataFrame:
    # Friend PBP emits blocks and steals as standalone rows sharing the same
    # actionNumber as the missed shot / turnover. Official NBA PBP embeds those
    # details on the shot/turnover row. Keeping them as standalone rows breaks
    # lag-based possession parsing, especially rebounds after blocked shots.
    friend_df = friend_df[
        ~friend_df["actionType"].astype(str).str.lower().isin({"block", "steal"})
    ].copy()
    action_sets = friend_df.groupby(["game_id", "actionNumber"])["actionType"].transform(
        lambda values: "|".join(sorted(set(str(value).lower() for value in values)))
    )
    friend_df = friend_df[
        ~(
            friend_df["actionType"].astype(str).str.lower().eq("substitution")
            & action_sets.str.contains("|".join(["2pt", "3pt", "freethrow", "rebound", "turnover", "foul"]), regex=True)
        )
    ].copy()
    friend_df["_clock_seconds"] = friend_df["clock_display"].map(parse_clock_seconds)
    friend_df = friend_df.sort_values(["game_id", "period", "actionNumber"], kind="stable").copy()
    prev_clock = friend_df.groupby(["game_id", "period"])["_clock_seconds"].shift(1)
    out_of_order_sub = (
        friend_df["actionType"].astype(str).str.lower().eq("substitution")
        & prev_clock.notna()
        & friend_df["_clock_seconds"].notna()
        & (friend_df["_clock_seconds"] > prev_clock)
    )
    friend_df = friend_df[~out_of_order_sub].drop(columns=["_clock_seconds"]).copy()
    friend_df = add_synthetic_admin_rows(friend_df)

    side_lookup = game_sides.set_index("game_id")[["home_team", "away_team"]].to_dict("index")
    rows = []
    for _, row in friend_df.sort_values(["game_id", "period", "actionNumber"], kind="stable").iterrows():
        game_id = int(row["game_id"])
        sides = side_lookup.get(game_id, {})
        home_team = sides.get("home_team")
        away_team = sides.get("away_team")
        event_team = None if pd.isna(row.get("team")) else str(row.get("team"))
        action = str(row.get("actionType", "")).lower()
        if event_team not in {home_team, away_team} and action in {"2pt", "3pt", "freethrow", "turnover"}:
            inferred_side = infer_offense_side(row, side_sets)
            if inferred_side == "home":
                event_team = home_team
            elif inferred_side == "away":
                event_team = away_team
        desc = normalize_friend_description(row)
        home_desc = desc if event_team == home_team else None
        visitor_desc = desc if event_team == away_team else None
        neutral_desc = desc if event_team not in {home_team, away_team} else None
        home_players, away_players = assign_home_away_lineup(row, side_sets)
        seconds = parse_clock_seconds(row.get("clock_display"))
        home_score = pd.to_numeric(row.get("scoreHome"), errors="coerce")
        away_score = pd.to_numeric(row.get("scoreAway"), errors="coerce")

        out = {
            "game_id": game_id,
            "event_num": int(row["actionNumber"]),
            "event_type": event_type_from_friend(row),
            "event_action_type": event_action_type_from_friend(row),
            "period": int(row["period"]),
            "minute_game": float(48.0 - float(row.get("minutes_left_in_game", np.nan)))
            if pd.notna(row.get("minutes_left_in_game"))
            else np.nan,
            "time_remaining": float(row.get("minutes_left_in_game", np.nan))
            if pd.notna(row.get("minutes_left_in_game"))
            else np.nan,
            "wc_time_string": np.nan,
            "time_quarter": row.get("clock_display"),
            "minute_remaining_quarter": seconds // 60,
            "seconds_remaining_quarter": seconds,
            "home_description": home_desc,
            "neutral_description": neutral_desc,
            "visitor_description": visitor_desc,
            "score": f"{int(away_score)} - {int(home_score)}"
            if pd.notna(away_score) and pd.notna(home_score)
            else None,
            "away_score": away_score,
            "home_score": home_score,
            "score_margin": (home_score - away_score) if pd.notna(home_score) and pd.notna(away_score) else np.nan,
            "person1type": 0,
            "player1_id": int(row["person_id"]) if pd.notna(row.get("person_id")) else 0,
            "player1_name": row.get("playerName"),
            "player1_team_id": int(row["teamId"]) if pd.notna(row.get("teamId")) else 0,
            "player1_team_city": None,
            "player1_team_nickname": None,
            "player1_team_abbreviation": event_team,
            "person2type": 0,
            "player2_id": row.get("assister_id"),
            "player2_name": None,
            "player2_team_id": np.nan,
            "player2_team_city": np.nan,
            "player2_team_nickname": np.nan,
            "player2_team_abbreviation": np.nan,
            "person3type": 0,
            "player3_id": row.get("blockPersonId")
            if pd.notna(row.get("blockPersonId"))
            else row.get("stealPersonId"),
            "player3_name": np.nan,
            "player3_team_id": np.nan,
            "player3_team_city": np.nan,
            "player3_team_nickname": np.nan,
            "player3_team_abbreviation": np.nan,
            "video_available_flag": 0,
            "team_leading": None,
        }
        for idx in range(5):
            out[f"home_player{idx + 1}"] = home_players[idx] if idx < len(home_players) else np.nan
            out[f"away_player{idx + 1}"] = away_players[idx] if idx < len(away_players) else np.nan
        out["allow_partial_lineups"] = bool(allow_partial_lineups)
        rows.append(out)

    converted = pd.DataFrame(rows)
    lineup_cols = [f"{side}_player{i}" for side in ("home", "away") for i in range(1, 6)]
    if not allow_partial_lineups:
        converted[lineup_cols] = converted.groupby("game_id")[lineup_cols].ffill().bfill()
    return converted


def write_processed(metric: str, df: pd.DataFrame | None, output_dir: Path) -> Path | None:
    if df is None or df.empty:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{metric}20.parquet"
    df.to_parquet(path, index=False)
    return path


def metric_summary(path: Path | None, metric_col: str | list[str] | None) -> dict:
    if path is None:
        return {"exists": False}
    df = pd.read_parquet(path)
    summary = {
        "exists": True,
        "rows": int(len(df)),
        "games": int(df["game_id"].nunique()) if "game_id" in df.columns else 0,
        "unique_off_lineups": int(df[[f"O{i}" for i in range(1, 6)]].drop_duplicates().shape[0])
        if all(f"O{i}" in df.columns for i in range(1, 6))
        else 0,
    }
    cols = metric_col if isinstance(metric_col, list) else [metric_col]
    for col in [c for c in cols if c and c in df.columns]:
        vals = pd.to_numeric(df[col], errors="coerce")
        summary[f"{col}_sum"] = float(vals.sum())
        summary[f"{col}_mean"] = float(vals.mean())
    return summary


def compare_processed(official_path: Path | None, friend_path: Path | None, metric_cols: list[str]) -> dict:
    if official_path is None or friend_path is None:
        return {"comparable": False}
    left = pd.read_parquet(official_path)
    right = pd.read_parquet(friend_path)
    keys = [col for col in ["game_id", "event_num"] if col in left.columns and col in right.columns]
    out = {"comparable": True, "official_rows": int(len(left)), "friend_rows": int(len(right))}
    if not keys:
        return out
    merged = left.merge(right, on=keys, suffixes=("_official", "_friend"))
    out["matched_rows"] = int(len(merged))
    out["official_unmatched_rows"] = int(len(left) - len(merged))
    out["friend_unmatched_rows"] = int(len(right) - len(merged))
    for col in metric_cols:
        lcol = f"{col}_official"
        rcol = f"{col}_friend"
        if lcol not in merged.columns or rcol not in merged.columns or merged.empty:
            continue
        lvals = pd.to_numeric(merged[lcol], errors="coerce")
        rvals = pd.to_numeric(merged[rcol], errors="coerce")
        diff = rvals - lvals
        out[f"{col}_mae"] = float(diff.abs().mean())
        out[f"{col}_exact_rate"] = float((diff == 0).mean())
        out[f"{col}_corr"] = float(lvals.corr(rvals)) if lvals.nunique() > 1 and rvals.nunique() > 1 else None
    return out


def compare_solver_results(official_csv: Path, friend_csv: Path) -> dict:
    left = pd.read_csv(official_csv)
    right = pd.read_csv(friend_csv)
    merged = left.merge(right, on="player_id", suffixes=("_official", "_friend"))
    out = {
        "matched_players": int(len(merged)),
        "official_players": int(len(left)),
        "friend_players": int(len(right)),
    }
    for col in ["off", "def", "net_rapm"]:
        lcol = f"{col}_official"
        rcol = f"{col}_friend"
        diff = merged[rcol] - merged[lcol]
        out[f"{col}_mae"] = float(diff.abs().mean())
        out[f"{col}_corr"] = float(merged[lcol].corr(merged[rcol]))
    return out


def run_metric_solves(processed_root: Path, results_root: Path) -> dict:
    original_results_dir = rapm_solver.RESULTS_DIR
    outputs = {}
    for side in ["official", "friend_converted"]:
        side_results = results_root / side
        rapm_solver.RESULTS_DIR = side_results
        for metric in SOLVE_METRICS:
            input_file = processed_root / side / f"{metric}20.parquet"
            if not input_file.exists():
                continue
            result = rapm_solver.run_simplified_rapm(
                [str(input_file)],
                name_map_file=PROJECT_ROOT / "autocomplete_map.csv",
                pure=(metric == "RAPM"),
                prefix=metric,
            )
            if result is not None:
                expected_name = (
                    f"{metric.lower()}_20_rs_pure_results.csv"
                    if metric == "RAPM"
                    else f"{metric.lower()}_20_rs_results.csv"
                )
                expected = side_results / expected_name
                outputs[f"{side}:{metric}"] = str(expected)
    rapm_solver.RESULTS_DIR = original_results_dir
    comparisons = {}
    for metric in SOLVE_METRICS:
        expected_name = (
            f"{metric.lower()}_20_rs_pure_results.csv"
            if metric == "RAPM"
            else f"{metric.lower()}_20_rs_results.csv"
        )
        official_csv = results_root / "official" / expected_name
        friend_csv = results_root / "friend_converted" / expected_name
        if official_csv.exists() and friend_csv.exists():
            comparisons[metric] = compare_solver_results(official_csv, friend_csv)
    return {"outputs": outputs, "comparisons": comparisons}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--friend-raw", type=Path, default=DEFAULT_FRIEND_RAW)
    parser.add_argument("--official-raw", type=Path, default=DEFAULT_OFFICIAL_RAW)
    parser.add_argument("--team", default="BKN")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--skip-solves", action="store_true")
    parser.add_argument(
        "--allow-partial-lineups",
        action="store_true",
        help="Preserve partial known lineups instead of filling missing slots before processing.",
    )
    args = parser.parse_args()

    output_root = args.output_root
    raw_out = output_root / "raw"
    processed_root = output_root / "processed"
    results_root = output_root / "results"
    for path in [raw_out, processed_root, results_root]:
        path.mkdir(parents=True, exist_ok=True)

    friend_df = pd.read_parquet(args.friend_raw)
    official_df = pd.read_parquet(args.official_raw)
    friend_game_ids = set(pd.to_numeric(friend_df["game_id"], errors="coerce").dropna().astype(int))
    game_sides = infer_bkn_games(official_df, friend_game_ids, args.team)
    official_subset = official_df[official_df["game_id"].isin(game_sides["game_id"])].copy()
    official_subset = official_subset.sort_values(["game_id", "period", "event_num"], kind="stable")
    official_subset_path = PIPELINE_ROOT / "raw_data" / f"{args.team}_2020_rs_ours.parquet"
    official_subset.to_parquet(official_subset_path, index=False)
    shutil.copy2(official_subset_path, raw_out / official_subset_path.name)

    side_sets = build_game_side_player_sets(official_subset)
    converted = convert_friend_to_repo_raw(
        friend_df,
        game_sides,
        side_sets,
        allow_partial_lineups=args.allow_partial_lineups,
    )
    converted_path = raw_out / f"{args.team}_2020_rs_friend_converted.parquet"
    converted.to_parquet(converted_path, index=False)

    metric_cols = {
        "RAPM": ["Net_Diff", "Off_Diff", "Def_Diff"],
        "TS": ["Net_Diff"],
        "TOV": ["Is_Turnover", "Is_BadPass_TOV"],
        "REB": ["Offensive_Rebound"],
        "RIM_FREQ": ["Is_Rim_Attempt"],
        "RIM_FG_PCT": ["Is_Rim_Make"],
        "THREE_FREQ": ["Is_Three_Attempt"],
        "THREE_FG_PCT": ["Is_Three_Make"],
        "MIDRANGE_FREQ": ["Is_Midrange_Attempt"],
        "MIDRANGE_FG_PCT": ["Is_Midrange_Make"],
        "ASSIST_POINTS": ["Assist_Points"],
    }

    processed_paths: dict[str, dict[str, str | None]] = {"official": {}, "friend_converted": {}}
    summaries: dict[str, dict] = {"official": {}, "friend_converted": {}}
    comparisons: dict[str, dict] = {}
    for label, raw_path in [("official", official_subset_path), ("friend_converted", converted_path)]:
        side_dir = processed_root / label
        for metric, processor in PROCESSORS.items():
            print(f"\n=== Processing {label} {metric} ===")
            try:
                processed = processor(raw_path)
            except Exception as exc:  # Keep the harness useful when one metric fails.
                summaries[label][metric] = {"exists": False, "error": str(exc)}
                processed_paths[label][metric] = None
                continue
            path = write_processed(metric, processed, side_dir)
            processed_paths[label][metric] = str(path) if path else None
            summaries[label][metric] = metric_summary(path, metric_cols[metric])

    for metric, cols in metric_cols.items():
        official_path = Path(processed_paths["official"][metric]) if processed_paths["official"].get(metric) else None
        friend_path = (
            Path(processed_paths["friend_converted"][metric])
            if processed_paths["friend_converted"].get(metric)
            else None
        )
        comparisons[metric] = compare_processed(official_path, friend_path, cols)

    solve_report = {} if args.skip_solves else run_metric_solves(processed_root, results_root)

    report = {
        "inputs": {
            "friend_raw": str(args.friend_raw),
            "official_raw": str(args.official_raw),
            "team": args.team,
        },
        "outputs": {
            "official_subset_raw": str(official_subset_path),
            "converted_friend_raw": str(converted_path),
            "processed_root": str(processed_root),
            "results_root": str(results_root),
        },
        "raw_summary": {
            "friend_rows": int(len(friend_df)),
            "official_subset_rows": int(len(official_subset)),
            "converted_rows": int(len(converted)),
            "friend_games": int(friend_df["game_id"].nunique()),
            "official_subset_games": int(official_subset["game_id"].nunique()),
            "converted_games": int(converted["game_id"].nunique()),
            "bkn_home_games": int((game_sides["home_team"] == args.team).sum()),
            "bkn_away_games": int((game_sides["away_team"] == args.team).sum()),
        },
        "processed_summaries": summaries,
        "processed_comparisons": comparisons,
        "solve_report": solve_report,
    }
    report_path = output_root / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote validation report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
