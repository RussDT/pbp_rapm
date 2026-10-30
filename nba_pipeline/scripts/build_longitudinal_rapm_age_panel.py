#!/usr/bin/env python3
"""Assemble the annual positive-good O-RAPM/D-RAPM panel used for aging."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


RESULT_COLUMNS = {
    "player_id",
    "player_name",
    "off",
    "def",
    "off_poss",
    "def_poss",
}
AGE_COLUMNS = {"nba_id", "season", "date", "age"}


def _string_ids(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.astype("Int64").astype(str)


def assemble_age_panel(
    annual_results: Mapping[int, pd.DataFrame],
    age_history: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Join annual raw-solver RAPM results to the latest age observation."""
    missing_age = AGE_COLUMNS - set(age_history.columns)
    if missing_age:
        raise ValueError(f"Age history missing columns: {sorted(missing_age)}")

    age = age_history[list(AGE_COLUMNS)].copy()
    age["player_id"] = _string_ids(age["nba_id"])
    age["season"] = pd.to_numeric(age["season"], errors="coerce").astype("Int64")
    age["age"] = pd.to_numeric(age["age"], errors="coerce")
    age["age_date"] = pd.to_datetime(age["date"], errors="coerce")
    age = age.dropna(subset=["player_id", "season", "age"]).copy()
    age["season"] = age["season"].astype(int)
    age = age.sort_values(["player_id", "season", "age_date"]).drop_duplicates(
        ["player_id", "season"], keep="last"
    )[["player_id", "season", "age", "age_date"]]

    frames = []
    for season, result in sorted(annual_results.items()):
        missing_result = RESULT_COLUMNS - set(result.columns)
        if missing_result:
            raise ValueError(
                f"Annual RAPM result for {season} missing columns: {sorted(missing_result)}"
            )
        frame = result[list(RESULT_COLUMNS)].copy()
        frame["player_id"] = _string_ids(frame["player_id"])
        frame["season"] = int(season)
        frame["orapm"] = pd.to_numeric(frame["off"], errors="coerce")
        # rapm.py stores defense as points allowed (lower is better).
        frame["drapm"] = -pd.to_numeric(frame["def"], errors="coerce")
        frame["off_poss"] = pd.to_numeric(frame["off_poss"], errors="coerce")
        frame["def_poss"] = pd.to_numeric(frame["def_poss"], errors="coerce")
        frames.append(
            frame[
                [
                    "player_id",
                    "player_name",
                    "season",
                    "orapm",
                    "drapm",
                    "off_poss",
                    "def_poss",
                ]
            ]
        )
    if not frames:
        raise ValueError("No annual RAPM results were supplied.")

    ratings = pd.concat(frames, ignore_index=True)
    panel = ratings.merge(age, on=["player_id", "season"], how="left", validate="one_to_one")
    covered = int(panel["age"].notna().sum())
    receipt = {
        "rows": int(len(panel)),
        "players": int(panel["player_id"].nunique()),
        "seasons": int(panel["season"].nunique()),
        "age_rows_covered": covered,
        "age_rows_missing": int(len(panel) - covered),
        "age_coverage": float(covered / len(panel)) if len(panel) else 0.0,
        "defense_sign": "positive_good",
    }
    return panel.sort_values(["player_id", "season"]).reset_index(drop=True), receipt

