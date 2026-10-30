#!/usr/bin/env python3
"""Analyze fourth-quarter after-timeout possessions by clock and score state.

The analysis uses the repo's standard possession parser. An ATO possession is
one where the offensive team called a regular/short timeout before the first
competitive action in that possession. Coach challenges, official timeouts,
defensive timeouts, and timeouts after a competitive action are excluded.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from lineup_stats import infer_game_team_map
from process_rapm_blocks.common import _base_processing, prepare_standard_possession_df


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
RAW_DATA_DIR = PIPELINE_DIR / "raw_data"
RESULTS_DIR = PIPELINE_DIR / "results" / "endgame_ato"

GROUP_KEYS = ["game_id", "period", "poss_group"]
COMPETITIVE_EVENTS = {"MAKE", "MISS", "FreeThrow", "Turnover", "JumpBall", "Foul", "Violation"}
TEAM_TIMEOUT_RE = r"TIMEOUT\s*:\s*(?:REGULAR|SHORT)"
CLOCK_LABELS = ["1", "2", "3", "4", "5", "6", "7", "8-24"]


def season_raw_path(year: int, season_type: str) -> Path:
    suffix = "_PS" if season_type.upper() == "PS" else ""
    return RAW_DATA_DIR / f"NBA{int(year):02d}{suffix}.parquet"


def clock_seconds(series: pd.Series) -> pd.Series:
    """Parse MM:SS(.t) strings into seconds remaining."""
    pieces = series.astype(str).str.extract(r"^(?:(\d+):)?(\d+)(?:\.(\d+))?$")
    minutes = pd.to_numeric(pieces[0], errors="coerce").fillna(0.0)
    seconds = pd.to_numeric(pieces[1], errors="coerce")
    tenths = pd.to_numeric("0." + pieces[2].fillna("0"), errors="coerce").fillna(0.0)
    return minutes * 60.0 + seconds + tenths


def clock_bucket(seconds: pd.Series) -> pd.Series:
    """Bucket (0,1], (1,2], ..., (6,7], and (7,24] seconds."""
    ceiling = np.ceil(pd.to_numeric(seconds, errors="coerce")).clip(1, 24)
    return pd.Series(np.where(ceiling <= 7, ceiling.astype("Int64").astype(str), "8-24"), index=seconds.index)


def repair_sparse_score_state(frame: pd.DataFrame) -> pd.DataFrame:
    """Carry real score states across historical 0-0 placeholder rows.

    Modern PBP carries the score on every row. Some 2014-17 files instead put
    0-0 on non-scoring rows after scoring has begun. Action-score cumulative
    totals identify those impossible placeholders without changing a genuine
    opening 0-0 state.
    """
    repaired = frame.copy()
    home_actions = pd.to_numeric(repaired["Home_Action_Score"], errors="coerce").fillna(0.0)
    away_actions = pd.to_numeric(repaired["Away_Action_Score"], errors="coerce").fillna(0.0)
    cumulative_points = (home_actions + away_actions).groupby(repaired["game_id"], sort=False).cumsum()
    home_score = pd.to_numeric(repaired["home_score"], errors="coerce")
    away_score = pd.to_numeric(repaired["away_score"], errors="coerce")
    placeholder = home_score.eq(0) & away_score.eq(0) & cumulative_points.gt(0)
    repaired["home_score"] = home_score.mask(placeholder).groupby(repaired["game_id"], sort=False).ffill().fillna(0.0)
    repaired["away_score"] = away_score.mask(placeholder).groupby(repaired["game_id"], sort=False).ffill().fillna(0.0)
    return repaired


def _first_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.sort_values([*GROUP_KEYS, "row_order"], kind="stable")
        .drop_duplicates(GROUP_KEYS, keep="first")
        .copy()
    )


def build_season_endgame_possessions(year: int, season_type: str) -> pd.DataFrame:
    """Return one row per eligible late-Q4 standard possession."""
    raw_path = season_raw_path(year, season_type)
    base = _base_processing(raw_path)
    if base is None or base.empty:
        raise RuntimeError(f"Could not load {raw_path}")

    team_map = infer_game_team_map(base)
    parsed_input = base.merge(team_map, on="game_id", how="left")
    parsed, _ = prepare_standard_possession_df(parsed_input, "ENDGAME_ATO")
    parsed = parsed.reset_index(drop=True)
    parsed["row_order"] = np.arange(len(parsed), dtype=np.int64)
    parsed = repair_sparse_score_state(parsed)
    parsed["period"] = pd.to_numeric(parsed["period"], errors="coerce")
    parsed = parsed.loc[parsed["period"].eq(4)].copy()
    if parsed.empty:
        return pd.DataFrame()

    parsed["clock_sec"] = clock_seconds(parsed["time_quarter"])
    parsed["is_eop"] = parsed["End_of_Possession"].fillna(False).astype(bool)
    parsed["is_competitive"] = parsed["event_type"].isin(COMPETITIVE_EVENTS)
    parsed["competitive_order"] = np.where(parsed["is_competitive"], parsed["row_order"], np.nan)
    parsed["event_text"] = (
        parsed["home_description"].fillna("").astype(str)
        + " "
        + parsed["visitor_description"].fillna("").astype(str)
        + " "
        + parsed["neutral_description"].fillna("").astype(str)
    ).str.strip()
    parsed["is_team_timeout"] = (
        parsed["event_type"].eq("Timeout")
        & parsed["event_text"].str.contains(TEAM_TIMEOUT_RE, case=False, regex=True, na=False)
    )
    home_has_text = parsed["home_description"].fillna("").astype(str).str.strip().ne("")
    away_has_text = parsed["visitor_description"].fillna("").astype(str).str.strip().ne("")
    parsed["timeout_caller_side"] = np.select(
        [parsed["is_team_timeout"] & home_has_text & ~away_has_text,
         parsed["is_team_timeout"] & away_has_text & ~home_has_text],
        ["Home", "Away"],
        default="",
    )

    parsed["away_score_num"] = pd.to_numeric(parsed["away_score"], errors="coerce")
    parsed["home_score_num"] = pd.to_numeric(parsed["home_score"], errors="coerce")
    parsed["away_action_points"] = pd.to_numeric(parsed["Away_Action_Score"], errors="coerce").fillna(0.0)
    parsed["home_action_points"] = pd.to_numeric(parsed["Home_Action_Score"], errors="coerce").fillna(0.0)

    first = _first_rows(parsed)[
        [*GROUP_KEYS, "row_order", "clock_sec", "away_score_num", "home_score_num",
         "away_action_points", "home_action_points"]
    ].rename(columns={
        "row_order": "first_order",
        "clock_sec": "first_event_clock",
        "away_score_num": "first_away_score",
        "home_score_num": "first_home_score",
        "away_action_points": "first_away_action_points",
        "home_action_points": "first_home_action_points",
    })

    first_comp = (
        parsed.groupby(GROUP_KEYS, sort=False)["competitive_order"]
        .min()
        .rename("first_competitive_order")
        .reset_index()
    )
    points = (
        parsed.groupby(GROUP_KEYS, sort=False)
        .agg(home_points=("home_action_points", "sum"), away_points=("away_action_points", "sum"))
        .reset_index()
    )
    terminal = (
        parsed.loc[parsed["is_eop"]]
        .sort_values([*GROUP_KEYS, "row_order"], kind="stable")
        .drop_duplicates(GROUP_KEYS, keep="last")
        [[*GROUP_KEYS, "row_order", "clock_sec", "event_num", "event_type", "poss_offense",
          "home_team_abbr", "away_team_abbr"]]
        .rename(columns={
            "row_order": "terminal_order",
            "clock_sec": "end_clock",
            "event_num": "end_event_num",
            "event_type": "terminal_event_type",
        })
    )

    possessions = first.merge(first_comp, on=GROUP_KEYS, how="inner")
    possessions = possessions.merge(points, on=GROUP_KEYS, how="inner")
    possessions = possessions.merge(terminal, on=GROUP_KEYS, how="inner")
    possessions = possessions.loc[possessions["poss_offense"].isin(["Home", "Away"])].copy()
    possessions["pre_away_score"] = possessions["first_away_score"] - possessions["first_away_action_points"]
    possessions["pre_home_score"] = possessions["first_home_score"] - possessions["first_home_action_points"]
    possessions["off_team_abbr"] = np.where(
        possessions["poss_offense"].eq("Home"),
        possessions["home_team_abbr"],
        possessions["away_team_abbr"],
    )
    possessions["off_points"] = np.where(
        possessions["poss_offense"].eq("Home"), possessions["home_points"], possessions["away_points"]
    ).astype(float)
    possessions["def_points"] = np.where(
        possessions["poss_offense"].eq("Home"), possessions["away_points"], possessions["home_points"]
    ).astype(float)
    possessions["pre_margin"] = np.where(
        possessions["poss_offense"].eq("Home"),
        possessions["pre_home_score"] - possessions["pre_away_score"],
        possessions["pre_away_score"] - possessions["pre_home_score"],
    ).astype(float)

    possessions = possessions.sort_values(["game_id", "first_order"], kind="stable")
    possessions["previous_end_clock"] = possessions.groupby("game_id", sort=False)["end_clock"].shift(1)
    possessions["previous_end_clock"] = possessions["previous_end_clock"].fillna(720.0)

    timeouts = parsed.loc[
        parsed["is_team_timeout"] & parsed["timeout_caller_side"].ne(""),
        [*GROUP_KEYS, "row_order", "clock_sec", "timeout_caller_side"],
    ].copy()
    timeouts = timeouts.merge(
        possessions[[*GROUP_KEYS, "poss_offense", "first_competitive_order"]],
        on=GROUP_KEYS,
        how="inner",
    )
    timeouts["is_pre_action"] = (
        timeouts["first_competitive_order"].notna()
        & timeouts["row_order"].lt(timeouts["first_competitive_order"])
    )
    pre_action = timeouts.loc[timeouts["is_pre_action"]].copy()
    any_pre_action = (
        pre_action.groupby(GROUP_KEYS, sort=False).size().rename("pre_action_timeout_count").reset_index()
    )
    offense_timeouts = pre_action.loc[
        pre_action["timeout_caller_side"].eq(pre_action["poss_offense"])
    ].copy()
    offense_timeouts = (
        offense_timeouts.sort_values([*GROUP_KEYS, "row_order"], kind="stable")
        .drop_duplicates(GROUP_KEYS, keep="last")
        [[*GROUP_KEYS, "row_order", "clock_sec"]]
        .rename(columns={"row_order": "ato_timeout_order", "clock_sec": "ato_clock_sec"})
    )

    possessions = possessions.merge(any_pre_action, on=GROUP_KEYS, how="left")
    possessions = possessions.merge(offense_timeouts, on=GROUP_KEYS, how="left")
    possessions["pre_action_timeout_count"] = possessions["pre_action_timeout_count"].fillna(0).astype(int)
    possessions["is_ato"] = possessions["ato_timeout_order"].notna()
    possessions["analysis_clock"] = np.where(
        possessions["is_ato"], possessions["ato_clock_sec"], possessions["previous_end_clock"]
    )

    action_rows = parsed.merge(
        possessions[[*GROUP_KEYS, "is_ato", "ato_timeout_order", "first_order"]],
        on=GROUP_KEYS,
        how="inner",
    )
    action_rows["action_threshold"] = np.where(
        action_rows["is_ato"], action_rows["ato_timeout_order"], action_rows["first_order"] - 1
    )
    action_rows = action_rows.loc[action_rows["row_order"].gt(action_rows["action_threshold"])].copy()
    action_rows["is_fga"] = action_rows["event_type"].isin(["MAKE", "MISS"])
    action_rows["fga_order"] = np.where(action_rows["is_fga"], action_rows["row_order"], np.nan)
    action_rows["turnover_order"] = np.where(
        action_rows["event_type"].eq("Turnover"), action_rows["row_order"], np.nan
    )
    action_summary = (
        action_rows.groupby(GROUP_KEYS, sort=False)
        .agg(first_fga_order=("fga_order", "min"), first_turnover_order=("turnover_order", "min"))
        .reset_index()
    )
    first_fga = (
        action_rows.loc[action_rows["is_fga"]]
        .sort_values([*GROUP_KEYS, "row_order"], kind="stable")
        .drop_duplicates(GROUP_KEYS, keep="first")
        [[*GROUP_KEYS, "event_text"]]
    )
    first_fga["first_fga_type"] = np.where(
        first_fga["event_text"].str.contains("3PT", case=False, regex=False, na=False), "3PA", "2PA"
    )
    first_fga = first_fga.drop(columns="event_text")
    possessions = possessions.merge(action_summary, on=GROUP_KEYS, how="left")
    possessions = possessions.merge(first_fga, on=GROUP_KEYS, how="left")
    possessions["turnover_before_fga"] = possessions["first_turnover_order"].notna() & (
        possessions["first_fga_order"].isna()
        | possessions["first_turnover_order"].lt(possessions["first_fga_order"])
    )

    possessions["deficit"] = (-possessions["pre_margin"]).round().astype("Int64")
    possessions["clock_bucket"] = clock_bucket(possessions["analysis_clock"])
    possessions["post_margin"] = possessions["pre_margin"] + possessions["off_points"] - possessions["def_points"]
    possessions["tie_or_lead"] = np.where(
        possessions["deficit"].eq(0), possessions["post_margin"].gt(0), possessions["post_margin"].ge(0)
    )
    possessions["outcome_bucket"] = np.select(
        [possessions["off_points"].eq(0), possessions["off_points"].eq(1), possessions["off_points"].eq(2)],
        ["0", "1", "2"],
        default="3+",
    )
    possessions["year"] = int(year)
    possessions["season_type"] = season_type.upper()
    possessions["is_baseline"] = possessions["pre_action_timeout_count"].eq(0)

    eligible = (
        possessions["analysis_clock"].gt(0)
        & possessions["analysis_clock"].le(24)
        & possessions["deficit"].between(0, 3, inclusive="both")
    )
    output_cols = [
        "game_id", "year", "season_type", "poss_group", "off_team_abbr", "poss_offense",
        "analysis_clock", "clock_bucket", "deficit", "pre_margin", "post_margin",
        "off_points", "def_points", "is_ato", "is_baseline", "pre_action_timeout_count",
        "first_fga_type", "turnover_before_fga", "tie_or_lead", "outcome_bucket",
        "first_order", "terminal_order", "end_event_num", "terminal_event_type",
    ]
    return possessions.loc[eligible, output_cols].reset_index(drop=True)


def _cluster_bootstrap_ppp(frame: pd.DataFrame, reps: int, seed: int) -> np.ndarray:
    if frame.empty:
        return np.array([], dtype=float)
    clusters = frame.groupby("game_id", sort=False)["off_points"].agg(["sum", "count"])
    values = clusters.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(reps, len(values)))
    sampled = values[draws]
    return sampled[:, :, 0].sum(axis=1) / sampled[:, :, 1].sum(axis=1)


def _summarize_cell(frame: pd.DataFrame) -> dict[str, float | int]:
    n = len(frame)
    if not n:
        return {
            "possessions": 0, "games": 0, "points": 0.0, "ppp": np.nan,
            "zero_rate": np.nan, "one_rate": np.nan, "two_rate": np.nan,
            "three_plus_rate": np.nan, "turnover_rate": np.nan,
            "first_fga_n": 0, "three_share_first_fga": np.nan, "tie_or_lead_rate": np.nan,
        }
    return {
        "possessions": int(n),
        "games": int(frame["game_id"].nunique()),
        "points": float(frame["off_points"].sum()),
        "ppp": float(frame["off_points"].mean()),
        "zero_rate": float(frame["outcome_bucket"].eq("0").mean()),
        "one_rate": float(frame["outcome_bucket"].eq("1").mean()),
        "two_rate": float(frame["outcome_bucket"].eq("2").mean()),
        "three_plus_rate": float(frame["outcome_bucket"].eq("3+").mean()),
        "turnover_rate": float(frame["turnover_before_fga"].mean()),
        "first_fga_n": int(frame["first_fga_type"].notna().sum()),
        "three_share_first_fga": float(frame["first_fga_type"].eq("3PA").sum() / frame["first_fga_type"].notna().sum())
        if frame["first_fga_type"].notna().any() else np.nan,
        "tie_or_lead_rate": float(frame["tie_or_lead"].mean()),
    }


def summarize_cells(possessions: pd.DataFrame, bootstrap_reps: int = 2000) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for deficit in range(4):
        for bucket_index, bucket in enumerate(CLOCK_LABELS):
            cell = possessions.loc[
                possessions["deficit"].eq(deficit) & possessions["clock_bucket"].eq(bucket)
            ]
            ato = cell.loc[cell["is_ato"]]
            baseline = cell.loc[cell["is_baseline"]]
            row: dict[str, object] = {"deficit": deficit, "clock_bucket": bucket}
            row.update({f"ato_{key}": value for key, value in _summarize_cell(ato).items()})
            row.update({f"baseline_{key}": value for key, value in _summarize_cell(baseline).items()})

            seed = 20260812 + deficit * 100 + bucket_index
            ato_boot = _cluster_bootstrap_ppp(ato, bootstrap_reps, seed)
            baseline_boot = _cluster_bootstrap_ppp(baseline, bootstrap_reps, seed + 10000)
            if len(ato_boot):
                row["ato_ppp_ci_low"], row["ato_ppp_ci_high"] = np.quantile(ato_boot, [0.025, 0.975])
            else:
                row["ato_ppp_ci_low"] = row["ato_ppp_ci_high"] = np.nan
            if len(baseline_boot):
                row["baseline_ppp_ci_low"], row["baseline_ppp_ci_high"] = np.quantile(
                    baseline_boot, [0.025, 0.975]
                )
            else:
                row["baseline_ppp_ci_low"] = row["baseline_ppp_ci_high"] = np.nan
            if len(ato_boot) and len(baseline_boot):
                delta_boot = ato_boot - baseline_boot
                row["ppp_delta"] = row["ato_ppp"] - row["baseline_ppp"]
                row["ppp_delta_ci_low"], row["ppp_delta_ci_high"] = np.quantile(delta_boot, [0.025, 0.975])
            else:
                row["ppp_delta"] = row["ppp_delta_ci_low"] = row["ppp_delta_ci_high"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_deficits(possessions: pd.DataFrame, bootstrap_reps: int = 4000) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for deficit in range(4):
        cell = possessions.loc[possessions["deficit"].eq(deficit)]
        ato = cell.loc[cell["is_ato"]]
        baseline = cell.loc[cell["is_baseline"]]
        row: dict[str, object] = {"deficit": deficit}
        row.update({f"ato_{key}": value for key, value in _summarize_cell(ato).items()})
        row.update({f"baseline_{key}": value for key, value in _summarize_cell(baseline).items()})
        seed = 20261812 + deficit
        ato_boot = _cluster_bootstrap_ppp(ato, bootstrap_reps, seed)
        baseline_boot = _cluster_bootstrap_ppp(baseline, bootstrap_reps, seed + 10000)
        delta_boot = ato_boot - baseline_boot
        row["ato_ppp_ci_low"], row["ato_ppp_ci_high"] = np.quantile(ato_boot, [0.025, 0.975])
        row["baseline_ppp_ci_low"], row["baseline_ppp_ci_high"] = np.quantile(
            baseline_boot, [0.025, 0.975]
        )
        row["ppp_delta"] = row["ato_ppp"] - row["baseline_ppp"]
        row["ppp_delta_ci_low"], row["ppp_delta_ci_high"] = np.quantile(delta_boot, [0.025, 0.975])
        rows.append(row)
    return pd.DataFrame(rows)


def build_quality_report(possessions: pd.DataFrame) -> dict[str, object]:
    ato = possessions.loc[possessions["is_ato"]]
    return {
        "eligible_possessions": int(len(possessions)),
        "eligible_games": int(possessions["game_id"].nunique()),
        "ato_possessions": int(len(ato)),
        "ato_games": int(ato["game_id"].nunique()),
        "baseline_possessions": int(possessions["is_baseline"].sum()),
        "duplicate_possession_keys": int(
            possessions.duplicated(["game_id", "year", "season_type", "poss_group"]).sum()
        ),
        "ato_missing_team": int(ato["off_team_abbr"].fillna("").eq("").sum()),
        "ato_invalid_clock": int((~ato["analysis_clock"].gt(0) | ~ato["analysis_clock"].le(24)).sum()),
        "ato_noninteger_deficit": int((ato["pre_margin"] % 1 != 0).sum()),
        "ato_points_outside_0_6": int((~ato["off_points"].between(0, 6)).sum()),
        "ato_multiple_pre_action_timeouts": int(ato["pre_action_timeout_count"].gt(1).sum()),
        "seasons": sorted(int(value) for value in possessions["year"].unique()),
        "season_types": sorted(str(value) for value in possessions["season_type"].unique()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=14)
    parser.add_argument("--end-year", type=int, default=26)
    parser.add_argument("--season-types", choices=["rs", "ps", "all"], default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    season_types = ["RS", "PS"] if args.season_types == "all" else [args.season_types.upper()]
    jobs = [
        (year, season_type)
        for year in range(args.start_year, args.end_year + 1)
        for season_type in season_types
        if season_raw_path(year, season_type).exists()
    ]
    if not jobs:
        raise FileNotFoundError("No requested raw season files exist")

    frames: list[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(build_season_endgame_possessions, year, season_type): (year, season_type)
            for year, season_type in jobs
        }
        for future in as_completed(futures):
            year, season_type = futures[future]
            frame = future.result()
            frames.append(frame)
            print(f"Finished {year:02d} {season_type}: {len(frame):,} eligible possessions")

    possessions = pd.concat(frames, ignore_index=True)
    possessions = possessions.sort_values(
        ["year", "season_type", "game_id", "first_order"], kind="stable"
    ).reset_index(drop=True)
    summary = summarize_cells(possessions, bootstrap_reps=args.bootstrap_reps)
    deficits = summarize_deficits(possessions, bootstrap_reps=max(args.bootstrap_reps, 4000))
    quality = build_quality_report(possessions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "endgame_ato_possessions_2014_2026.parquet"
    summary_path = args.output_dir / "endgame_ato_by_clock_deficit_2014_2026.csv"
    deficit_path = args.output_dir / "endgame_ato_by_deficit_2014_2026.csv"
    quality_path = args.output_dir / "endgame_ato_quality_2014_2026.json"
    possessions.to_parquet(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    deficits.to_csv(deficit_path, index=False)
    quality_path.write_text(json.dumps(quality, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {deficit_path}")
    print(f"Wrote {quality_path}")


if __name__ == "__main__":
    main()
