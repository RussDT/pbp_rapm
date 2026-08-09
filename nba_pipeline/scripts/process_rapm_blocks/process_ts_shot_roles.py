"""Build self-shot and teammate-shot role samples from the existing TS surface.

The standard TS parquet has one row per scoring attempt and already excludes
turnover-ended possessions.  This module resolves the terminal shooter from the
matching raw event and creates two conditional samples:

* ``TS_SELF_SHOT``: one focal-shooter row per resolved attempt.
* ``TS_TEAMMATE_SHOT``: four focal-nonshooter rows per resolved attempt.

Both samples keep the full attempt value.  The hierarchy runner estimates the
two conditional role effects and then decomposes parent offensive TS RAPM into
self-shot, teammate-shot, and shot-mix contributions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .process_ts import process_ts_py


OFFENSE_COLS = [f"O{i}" for i in range(1, 6)]
CONTEXT_COLS = [f"O{i}" for i in range(1, 5)]
DEFENSE_COLS = [f"D{i}" for i in range(1, 6)]
KEY_COLS = ["game_id", "event_num"]


def _numeric_key(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_game_key"] = pd.to_numeric(
        result["game_id"], errors="coerce"
    ).astype("Int64")
    result["_event_key"] = pd.to_numeric(
        result["event_num"], errors="coerce"
    ).astype("Int64")
    return result


def _resolved_shooters(raw_df: pd.DataFrame) -> pd.DataFrame:
    required = {*KEY_COLS, "player1_id"}
    missing = required - set(raw_df.columns)
    if missing:
        raise ValueError(f"Raw PBP is missing shooter merge columns: {sorted(missing)}")
    shooter = _numeric_key(raw_df[list(required)])
    shooter["Shooter"] = pd.to_numeric(
        shooter["player1_id"], errors="coerce"
    ).astype("Int64")
    shooter = shooter[["_game_key", "_event_key", "Shooter"]].drop_duplicates()

    # A corrected event key with conflicting player ids is not safely resolvable.
    conflict = (
        shooter.dropna(subset=["Shooter"])
        .groupby(["_game_key", "_event_key"])["Shooter"]
        .nunique()
        .gt(1)
    )
    conflict_keys = conflict[conflict].index
    if len(conflict_keys):
        shooter = shooter.set_index(["_game_key", "_event_key"])
        shooter.loc[conflict_keys, "Shooter"] = pd.NA
        shooter = shooter.reset_index()
    return shooter.drop_duplicates(["_game_key", "_event_key"], keep="last")


def _shared_output_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "game_id",
        "event_num",
        "Focal_Player",
        *CONTEXT_COLS,
        *DEFENSE_COLS,
        "score",
        "period",
        "time_quarter",
        "away_score",
        "home_score",
        "score_margin",
        "TS_Points",
        "Role_Resolution_Complete",
        "game_date",
        "Season",
    ]
    return [column for column in preferred if column in frame.columns]


def build_ts_shot_role_rows(
    ts_df: pd.DataFrame,
    raw_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expand resolved TS attempts into shooter and nonshooter focal samples."""
    required = {*KEY_COLS, "Net_Diff", *OFFENSE_COLS, *DEFENSE_COLS}
    missing = required - set(ts_df.columns)
    if missing:
        raise ValueError(f"TS parquet is missing required columns: {sorted(missing)}")

    attempts = _numeric_key(ts_df).merge(
        _resolved_shooters(raw_df),
        on=["_game_key", "_event_key"],
        how="left",
        validate="many_to_one",
    )
    offense = attempts[OFFENSE_COLS].apply(pd.to_numeric, errors="coerce")
    defense = attempts[DEFENSE_COLS].apply(pd.to_numeric, errors="coerce")
    shooter = pd.to_numeric(attempts["Shooter"], errors="coerce")
    shooter_match = offense.eq(shooter, axis=0)
    complete_lineups = (
        offense.notna().all(axis=1)
        & defense.notna().all(axis=1)
        & offense.gt(0).all(axis=1)
        & defense.gt(0).all(axis=1)
    )
    resolved = shooter_match.sum(axis=1).eq(1) & complete_lineups
    attempts = attempts.loc[resolved].reset_index(drop=True)
    offense_values = (
        attempts[OFFENSE_COLS].apply(pd.to_numeric, errors="coerce").to_numpy()
    )
    shooter_values = pd.to_numeric(
        attempts["Shooter"], errors="coerce"
    ).to_numpy()
    shooter_indices = (offense_values == shooter_values[:, None]).argmax(axis=1)
    attempts["TS_Points"] = pd.to_numeric(
        attempts["Net_Diff"], errors="coerce"
    ).astype(float)
    attempts["Role_Resolution_Complete"] = True

    # One shooter-focal row per attempt.
    self_rows = attempts.copy()
    self_rows["Focal_Player"] = shooter_values
    self_context = np.empty((len(attempts), 4), dtype=float)
    for shooter_idx in range(5):
        mask = shooter_indices == shooter_idx
        if mask.any():
            self_context[mask] = np.delete(
                offense_values[mask], shooter_idx, axis=1
            )
    self_rows[CONTEXT_COLS] = self_context

    # Four nonshooter-focal rows per attempt.  The shooter remains in context.
    teammate_records: list[pd.DataFrame] = []
    for focal_idx in range(5):
        mask = shooter_indices != focal_idx
        if not mask.any():
            continue
        focal_rows = attempts.loc[mask].copy()
        focal_rows["Focal_Player"] = offense_values[mask, focal_idx]
        focal_rows[CONTEXT_COLS] = np.delete(
            offense_values[mask], focal_idx, axis=1
        )
        teammate_records.append(focal_rows)
    teammate_rows = pd.concat(teammate_records, ignore_index=True)

    self_rows = self_rows[_shared_output_columns(self_rows)].reset_index(drop=True)
    teammate_rows = teammate_rows[
        _shared_output_columns(teammate_rows)
    ].reset_index(drop=True)
    return self_rows, teammate_rows


def build_ts_shot_roles_from_processed(
    ts_df: pd.DataFrame,
    raw_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load only raw shooter keys and build role rows from a processed TS frame."""
    raw_df = pd.read_parquet(raw_path, columns=[*KEY_COLS, "player1_id"])
    return build_ts_shot_role_rows(ts_df, raw_df)


def process_ts_shot_roles_py(
    file_path: str,
    year: int,
    season_str: str,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Rebuild standard TS and return its paired role surfaces."""
    ts_df = process_ts_py(file_path, year, season_str)
    if ts_df is None or ts_df.empty:
        return None
    self_rows, teammate_rows = build_ts_shot_roles_from_processed(ts_df, file_path)
    resolved = len(self_rows)
    print(
        f"      TS shooter resolution: {resolved:,}/{len(ts_df):,} "
        f"({100.0 * resolved / len(ts_df):.3f}%)"
    )
    return self_rows, teammate_rows
