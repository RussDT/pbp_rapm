#!/usr/bin/env python3
"""Audit and analyze Russell Westbrook's free-throw sequence around 2017-18.

The primary source is the repo's regular-season raw NBA play-by-play parquet
surface.  The script deliberately reconstructs trips from clock-first event
order and description text instead of trusting a possession-level FT flag.

Generated tables and charts are research artifacts and belong under
``nba_pipeline/results/westbrook_free_throw_routine`` (gitignored).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import scipy.optimize as opt
import scipy.special as special
import scipy.stats as st
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint


WESTBROOK_ID = 201566
RULE_SEASON_END = 2018
DEFAULT_GAME_DATES = Path(
    "/Users/russ/Documents/Codex/basketball/nba/research/shotquality/nba_game_dates.csv"
)
DEFAULT_2026_CROSSWALK = Path(
    "/Users/russ/Documents/Codex/basketball/nba/research/shotquality/"
    "court_analysis/coordinate_repair/nba_game_crosswalk_2025_26.csv"
)
FT_PATTERN = re.compile(r"Free Throw(?:\s+.*?)*?\s+(\d+) of (\d+)", re.IGNORECASE)
UNUSUAL_CONTEXT_PATTERN = re.compile(
    r"TECH(?:NICAL)?|FLAGRANT|CLEAR\s*PATH|TRANSITION\s+TAKE|"
    r"AWAY\s+FROM\s+PLAY|DEF(?:ENSIVE)?\.?\s*3(?:\s*SECOND)?",
    re.IGNORECASE,
)


def normalize_game_id(value: object) -> str:
    """Return the repo-standard stable 10-character NBA game id."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(10)


def parse_clock_seconds(value: object, minutes: object, seconds: object) -> float:
    """Parse the canonical quarter clock, with legacy total-second fallback."""
    if not pd.isna(value):
        text = str(value).strip()
        match = re.match(r"^(\d+):(\d+(?:\.\d+)?)$", text)
        if match:
            return 60.0 * float(match.group(1)) + float(match.group(2))
        match = re.match(r"^PT(?:(\d+)M)?([\d.]+)S$", text, re.IGNORECASE)
        if match:
            return 60.0 * float(match.group(1) or 0) + float(match.group(2))
    sec = pd.to_numeric(pd.Series([seconds]), errors="coerce").iloc[0]
    minute = pd.to_numeric(pd.Series([minutes]), errors="coerce").iloc[0]
    if pd.notna(sec) and float(sec) > 59:
        return float(sec)
    if pd.notna(minute) and pd.notna(sec):
        return 60.0 * float(minute) + float(sec)
    return np.nan


@dataclass(frozen=True)
class ParsedFreeThrow:
    made: int
    declared_num: int | None
    declared_total: int | None
    explicit_kind: str
    ambiguous_no_outcome_marker: bool


def parse_ft_description(description: str) -> ParsedFreeThrow:
    """Parse outcome, ordinal, and explicit administrative FT wording."""
    text = str(description or "")
    if "free throw" not in text.lower():
        raise ValueError(f"Not a free-throw description: {text!r}")
    has_miss = re.search(r"\bMISS\b", text, re.IGNORECASE) is not None
    has_made_points_marker = re.search(r"\(\d+ PTS\)", text, re.IGNORECASE) is not None
    ambiguous_no_outcome_marker = not has_miss and not has_made_points_marker
    # In this NBA feed a credited make carries the cumulative-player-points
    # marker. Three Westbrook rows omit both MISS and that marker; cumulative
    # points prove all three are misses, so "not MISS" is not sufficient.
    made = int(not has_miss and has_made_points_marker)
    if re.search(r"\bTechnical\b", text, re.IGNORECASE):
        kind = "technical"
    elif re.search(r"\bFlagrant\b", text, re.IGNORECASE):
        kind = "flagrant"
    elif re.search(r"\bClear\s*Path\b", text, re.IGNORECASE):
        kind = "clear_path"
    else:
        kind = "ordinary"
    match = FT_PATTERN.search(text)
    if match:
        num, total = int(match.group(1)), int(match.group(2))
    elif kind == "technical":
        num, total = None, None
    else:
        num, total = None, None
    return ParsedFreeThrow(made, num, total, kind, ambiguous_no_outcome_marker)


def season_label(season_end: int) -> str:
    return f"{season_end - 1}-{str(season_end)[-2:]}"


def _description(df: pd.DataFrame) -> pd.Series:
    cols = ["home_description", "neutral_description", "visitor_description"]
    return (
        df.reindex(columns=cols, fill_value="")
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def load_game_dates(path: Path, supplemental_path: Path | None = None) -> pd.DataFrame:
    dates = pd.read_csv(path)
    required = {"GAME_ID", "HTM", "VTM", "date"}
    missing = required.difference(dates.columns)
    if missing:
        raise ValueError(f"Game-date file is missing columns: {sorted(missing)}")
    dates = dates.copy()
    dates["game_id"] = dates["GAME_ID"].map(normalize_game_id)
    dates["game_date"] = pd.to_datetime(
        dates["date"].astype(str).str.replace(r"\.0$", "", regex=True),
        format="%Y%m%d",
        errors="coerce",
    )
    conflicts = (
        dates.groupby("game_id", observed=True)[["game_date", "HTM", "VTM"]]
        .nunique(dropna=False)
        .max(axis=1)
    )
    if (conflicts > 1).any():
        bad = conflicts[conflicts > 1].index[:5].tolist()
        raise ValueError(f"Conflicting schedule rows for game ids: {bad}")
    dates = dates.drop_duplicates("game_id")[["game_id", "game_date", "HTM", "VTM"]]
    if supplemental_path is not None and supplemental_path.exists():
        extra = pd.read_csv(supplemental_path)
        extra = extra[extra["nba_game_id"].notna() & extra["nba_game_date"].notna()].copy()
        extra["game_id"] = extra["nba_game_id"].map(normalize_game_id)
        extra["game_date"] = pd.to_datetime(
            extra["nba_game_date"].astype(str).str.replace(r"\.0$", "", regex=True),
            format="%Y%m%d",
            errors="coerce",
        )
        extra = extra.rename(columns={"home_abbr": "HTM", "away_abbr": "VTM"})
        conflicts = (
            extra.groupby("game_id", observed=True)[["game_date", "HTM", "VTM"]]
            .nunique(dropna=False)
            .max(axis=1)
        )
        if (conflicts > 1).any():
            bad = conflicts[conflicts > 1].index[:5].tolist()
            raise ValueError(f"Conflicting supplemental schedule rows: {bad}")
        extra = extra.drop_duplicates("game_id")[["game_id", "game_date", "HTM", "VTM"]]
        dates = pd.concat([dates, extra], ignore_index=True).drop_duplicates("game_id", keep="first")
    return dates


def _read_columns(path: Path, requested: Sequence[str]) -> pd.DataFrame:
    import pyarrow.parquet as pq

    available = set(pq.ParquetFile(path).schema.names)
    columns = [column for column in requested if column in available]
    return pd.read_parquet(path, columns=columns)


def _build_pre_event_score(df: pd.DataFrame) -> pd.DataFrame:
    """Attach pre-event score while ignoring literal post-scoring 0-0 placeholders."""
    out = df.copy()
    away = pd.to_numeric(out.get("away_score"), errors="coerce")
    home = pd.to_numeric(out.get("home_score"), errors="coerce")
    total = away.fillna(0) + home.fillna(0)
    positive_seen = total.gt(0).groupby(out["game_id"], sort=False).cummax()
    valid = total.gt(0) | ~positive_seen
    away_state = away.where(valid).groupby(out["game_id"], sort=False).ffill().fillna(0)
    home_state = home.where(valid).groupby(out["game_id"], sort=False).ffill().fillna(0)
    out["pre_away_score"] = away_state.groupby(out["game_id"], sort=False).shift().fillna(0)
    out["pre_home_score"] = home_state.groupby(out["game_id"], sort=False).shift().fillna(0)
    return out


def _assign_trips(ft: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Assign trip ids, repairing only sequence-proven ordinal contradictions."""
    out = ft.copy()
    trip_ids: list[str] = []
    reconstructed_nums: list[int] = []
    trip_meta: dict[str, dict[str, object]] = {}
    active: dict[tuple[str, int, float, str], str] = {}
    counter = 0
    orphan_rows = 0

    for row in out.itertuples():
        kind = row.explicit_kind
        key = (row.game_id, int(row.period), float(row.clock_sec), kind)
        declared_num = row.declared_num
        declared_total = row.declared_total

        if kind == "technical":
            counter += 1
            trip_id = f"{row.game_id}-{counter:04d}"
            num = 1
            trip_meta[trip_id] = {
                "declared_totals": {1},
                "positions": {1},
                "kind": kind,
                "orphan": False,
            }
        elif declared_num == 1:
            counter += 1
            trip_id = f"{row.game_id}-{counter:04d}"
            num = 1
            trip_meta[trip_id] = {
                "declared_totals": {int(declared_total)},
                "positions": {1},
                "kind": kind,
                "orphan": False,
            }
            active[key] = trip_id
        elif declared_num is not None and declared_num > 1:
            candidate = active.get(key)
            can_join = False
            if candidate is not None:
                positions = trip_meta[candidate]["positions"]
                expected = max(positions) + 1
                can_join = expected == int(declared_num)
            if can_join:
                trip_id = candidate
                num = int(declared_num)
                trip_meta[trip_id]["positions"].add(num)
                trip_meta[trip_id]["declared_totals"].add(int(declared_total))
            else:
                counter += 1
                trip_id = f"{row.game_id}-{counter:04d}"
                num = int(declared_num)
                orphan_rows += 1
                trip_meta[trip_id] = {
                    "declared_totals": {int(declared_total)},
                    "positions": {num},
                    "kind": kind,
                    "orphan": True,
                }
                active[key] = trip_id
        else:
            counter += 1
            trip_id = f"{row.game_id}-{counter:04d}"
            num = 1
            orphan_rows += 1
            trip_meta[trip_id] = {
                "declared_totals": {1},
                "positions": {1},
                "kind": kind,
                "orphan": True,
            }
        trip_ids.append(trip_id)
        reconstructed_nums.append(num)

    out["trip_id"] = trip_ids
    out["ft_num"] = reconstructed_nums
    repaired_first_rows = 0
    trip_total: dict[str, int] = {}
    trip_complete: dict[str, bool] = {}
    label_anomaly: dict[str, bool] = {}
    for trip_id, meta in trip_meta.items():
        totals = set(meta["declared_totals"])
        positions = set(meta["positions"])
        total = max(max(totals), max(positions))
        anomaly = len(totals) > 1
        # The known V3 contradiction is "1 of 1" followed by "2 of 2".
        if totals == {1, 2} and positions == {1, 2}:
            repaired_first_rows += 1
        trip_total[trip_id] = total
        trip_complete[trip_id] = not bool(meta["orphan"]) and positions == set(range(1, total + 1))
        label_anomaly[trip_id] = anomaly
    out["trip_total"] = out["trip_id"].map(trip_total).astype(int)
    out["trip_complete"] = out["trip_id"].map(trip_complete).astype(bool)
    out["label_anomaly"] = out["trip_id"].map(label_anomaly).astype(bool)
    return out, {
        "trip_count": len(trip_meta),
        "orphan_ft_rows": orphan_rows,
        "incomplete_trip_count": int(sum(not value for value in trip_complete.values())),
        "label_anomaly_trip_count": int(sum(label_anomaly.values())),
        "sequence_repaired_first_rows": repaired_first_rows,
    }


def build_westbrook_attempts(
    raw_dir: Path,
    game_dates_path: Path,
    supplemental_game_dates_path: Path | None = DEFAULT_2026_CROSSWALK,
    start_year: int = 2009,
    end_year: int = 2026,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build the audited Westbrook regular-season FT attempt table."""
    schedule = load_game_dates(game_dates_path, supplemental_game_dates_path)
    requested = [
        "game_id",
        "event_num",
        "source_actionNumber",
        "event_type",
        "event_action_type",
        "period",
        "minute_game",
        "time_quarter",
        "minute_remaining_quarter",
        "seconds_remaining_quarter",
        "home_description",
        "neutral_description",
        "visitor_description",
        "away_score",
        "home_score",
        "player1_id",
        "player1_name",
        "player1_team_id",
        "player1_team_abbreviation",
    ]
    season_attempts: list[pd.DataFrame] = []
    audit: dict[str, object] = {"seasons": {}}

    for season_end in range(start_year, end_year + 1):
        yy = season_end % 100
        path = raw_dir / f"NBA{yy:02d}.parquet"
        if not path.exists():
            continue
        raw = _read_columns(path, requested)
        raw["game_id"] = raw["game_id"].map(normalize_game_id)
        raw["description"] = _description(raw)
        raw["event_type_num"] = pd.to_numeric(raw["event_type"], errors="coerce")
        raw["clock_sec"] = [
            parse_clock_seconds(value, minute, second)
            for value, minute, second in zip(
                raw.get("time_quarter", pd.Series(index=raw.index, dtype=object)),
                raw.get("minute_remaining_quarter", pd.Series(index=raw.index, dtype=float)),
                raw.get("seconds_remaining_quarter", pd.Series(index=raw.index, dtype=float)),
            )
        ]
        wb_ft_mask = (
            pd.to_numeric(raw["player1_id"], errors="coerce").eq(WESTBROOK_ID)
            & raw["event_type_num"].eq(3)
            & raw["description"].str.contains("Free Throw", case=False, na=False)
        )
        game_ids = raw.loc[wb_ft_mask, "game_id"].unique()
        if not len(game_ids):
            continue
        raw = raw[raw["game_id"].isin(game_ids)].copy()
        before_dedupe = len(raw)
        raw = raw.drop_duplicates().copy()
        exact_duplicates_removed = before_dedupe - len(raw)
        raw = raw.sort_values(
            ["game_id", "period", "clock_sec", "event_num"],
            ascending=[True, True, False, True],
            kind="stable",
        ).reset_index(drop=True)
        raw["event_order"] = np.arange(len(raw))
        raw = _build_pre_event_score(raw)

        ft = raw[
            pd.to_numeric(raw["player1_id"], errors="coerce").eq(WESTBROOK_ID)
            & raw["event_type_num"].eq(3)
            & raw["description"].str.contains("Free Throw", case=False, na=False)
        ].copy()
        parsed = ft["description"].map(parse_ft_description)
        ft["made"] = parsed.map(lambda value: value.made).astype(int)
        ft["declared_num"] = parsed.map(lambda value: value.declared_num)
        ft["declared_total"] = parsed.map(lambda value: value.declared_total)
        ft["explicit_kind"] = parsed.map(lambda value: value.explicit_kind)
        ft["ambiguous_no_outcome_marker"] = parsed.map(
            lambda value: value.ambiguous_no_outcome_marker
        )
        ft["season_end"] = season_end
        ft["season"] = season_label(season_end)
        ft["post_rule"] = int(season_end >= RULE_SEASON_END)

        ft, trip_audit = _assign_trips(ft)
        first_rows = ft.sort_values("event_order").drop_duplicates("trip_id")
        trip_context: dict[str, dict[str, object]] = {}
        for row in first_rows.itertuples():
            same_clock = raw[
                raw["game_id"].eq(row.game_id)
                & raw["period"].eq(row.period)
                & np.isclose(raw["clock_sec"], row.clock_sec, equal_nan=False)
                & raw["event_order"].ne(row.event_order)
            ]
            made_fg = same_clock[same_clock["event_type_num"].eq(1)]
            follows_any = not made_fg.empty
            same_shooter_made = made_fg[
                pd.to_numeric(made_fg["player1_id"], errors="coerce").eq(WESTBROOK_ID)
            ]
            follows_same = not same_shooter_made.empty
            corrected_after = bool(
                follows_same and same_shooter_made["event_order"].gt(row.event_order).any()
            )
            context = " || ".join(same_clock["description"].tolist())
            context_match = UNUSUAL_CONTEXT_PATTERN.search(context)
            trip_context[row.trip_id] = {
                "trip_follows_made_fg": follows_any,
                "trip_follows_same_shooter_made_fg": follows_same,
                "same_clock_made_fg_after_ft_event": corrected_after,
                "pre_ft_same_clock_context": context,
                "context_unusual_label": context_match.group(0) if context_match else "",
            }
        for column in [
            "trip_follows_made_fg",
            "trip_follows_same_shooter_made_fg",
            "same_clock_made_fg_after_ft_event",
            "pre_ft_same_clock_context",
            "context_unusual_label",
        ]:
            ft[column] = ft["trip_id"].map(lambda key: trip_context[key][column])

        first_rows = ft.sort_values("event_order").drop_duplicates("trip_id")

        def classify_trip(row: pd.Series) -> str:
            if row["explicit_kind"] == "technical":
                return "technical"
            if row["explicit_kind"] == "flagrant":
                return "flagrant"
            if row["explicit_kind"] == "clear_path":
                return "clear_path"
            if row["trip_total"] == 3:
                return "ordinary_three"
            if row["trip_total"] == 2:
                return "ordinary_two"
            if row["trip_total"] == 1:
                if row["trip_follows_same_shooter_made_fg"]:
                    return "and1"
                return "other_one_shot"
            return "unresolved"

        trip_types = first_rows.set_index("trip_id").apply(classify_trip, axis=1)
        ft["trip_type"] = ft["trip_id"].map(trip_types)
        ft["genuine_one_shot"] = ft["trip_type"].isin(["and1", "other_one_shot"])
        ft["ordinary_multi_first"] = (
            ft["trip_type"].isin(["ordinary_two", "ordinary_three"]) & ft["ft_num"].eq(1)
        )

        ft["is_home"] = ft["home_description"].fillna("").str.contains(
            "Free Throw", case=False, na=False
        )
        ft["pre_score_margin_shooter"] = np.where(
            ft["is_home"],
            ft["pre_home_score"] - ft["pre_away_score"],
            ft["pre_away_score"] - ft["pre_home_score"],
        )
        ft["elapsed_game_minute"] = pd.to_numeric(ft.get("minute_game"), errors="coerce")

        schedule_cols = schedule[schedule["game_id"].isin(game_ids)]
        ft = ft.merge(schedule_cols, on="game_id", how="left", validate="many_to_one")
        ft["team"] = np.where(ft["is_home"], ft["HTM"], ft["VTM"])
        ft["opponent"] = np.where(ft["is_home"], ft["VTM"], ft["HTM"])

        conflict_keys = raw.groupby(["game_id", "event_num"], observed=True).size()
        season_audit = {
            "attempts": int(len(ft)),
            "makes": int(ft["made"].sum()),
            "games": int(ft["game_id"].nunique()),
            "exact_context_rows_removed": int(exact_duplicates_removed),
            "conflicting_or_repeated_event_keys_in_selected_games": int((conflict_keys > 1).sum()),
            "date_missing_attempts": int(ft["game_date"].isna().sum()),
            "ambiguous_no_outcome_marker_repaired_as_miss": int(
                ft["ambiguous_no_outcome_marker"].sum()
            ),
            **trip_audit,
        }
        audit["seasons"][str(season_end)] = season_audit
        season_attempts.append(ft)

    attempts = pd.concat(season_attempts, ignore_index=True)
    game_dates = attempts[["game_id", "game_date"]].drop_duplicates().sort_values("game_date")
    game_dates["days_since_previous_game"] = game_dates["game_date"].diff().dt.days
    attempts = attempts.merge(
        game_dates[["game_id", "days_since_previous_game"]],
        on="game_id",
        how="left",
        validate="many_to_one",
    )
    attempts["back_to_back"] = attempts["days_since_previous_game"].eq(1)
    attempts = attempts.sort_values(
        ["game_date", "game_id", "period", "clock_sec", "event_num"],
        ascending=[True, True, True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    attempts["career_attempt"] = np.arange(1, len(attempts) + 1)

    audit["attempts"] = int(len(attempts))
    audit["makes"] = int(attempts["made"].sum())
    audit["games"] = int(attempts["game_id"].nunique())
    audit["date_coverage"] = float(attempts["game_date"].notna().mean())
    audit["parsed_ordinal_coverage"] = float(
        (attempts["declared_num"].notna() | attempts["explicit_kind"].eq("technical")).mean()
    )
    audit["complete_trip_attempt_coverage"] = float(attempts["trip_complete"].mean())
    audit["same_clock_corrected_made_fg_after_ft_event_attempts"] = int(
        attempts["same_clock_made_fg_after_ft_event"].sum()
    )
    audit["trip_type_counts"] = {
        str(key): int(value)
        for key, value in attempts.drop_duplicates("trip_id")["trip_type"].value_counts().items()
    }
    return attempts, audit


def rate_ci(frame: pd.DataFrame) -> tuple[int, int, float, float, float]:
    makes = int(frame["made"].sum())
    attempts = int(len(frame))
    if attempts == 0:
        return makes, attempts, np.nan, np.nan, np.nan
    low, high = proportion_confint(makes, attempts, method="wilson")
    return makes, attempts, makes / attempts, float(low), float(high)


def rate_record(frame: pd.DataFrame, prefix: str = "") -> dict[str, object]:
    makes, attempts, rate, low, high = rate_ci(frame)
    return {
        f"{prefix}makes": makes,
        f"{prefix}attempts": attempts,
        f"{prefix}pct": 100 * rate if pd.notna(rate) else np.nan,
        f"{prefix}ci_low": 100 * low if pd.notna(low) else np.nan,
        f"{prefix}ci_high": 100 * high if pd.notna(high) else np.nan,
    }


def build_season_table(attempts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season_end, frame in attempts.groupby("season_end", sort=True):
        row: dict[str, object] = {
            "season": season_label(int(season_end)),
            "season_end": int(season_end),
        }
        row.update(rate_record(frame, "overall_"))
        categories = {
            "ft1_": frame["ft_num"].eq(1),
            "ft2_": frame["ft_num"].eq(2),
            "ft3_": frame["ft_num"].eq(3),
            "one_shot_": frame["trip_total"].eq(1),
            "two_first_": frame["trip_type"].eq("ordinary_two") & frame["ft_num"].eq(1),
            "two_second_": frame["trip_type"].eq("ordinary_two") & frame["ft_num"].eq(2),
            "three_first_": frame["trip_type"].eq("ordinary_three") & frame["ft_num"].eq(1),
            "final_": frame["ft_num"].eq(frame["trip_total"]),
        }
        for prefix, mask in categories.items():
            row.update(rate_record(frame[mask], prefix))
        row["ft2_minus_ft1_pp"] = row["ft2_pct"] - row["ft1_pct"]
        rows.append(row)
    return pd.DataFrame(rows)


def cluster_bootstrap(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    cluster: str = "game_id",
    n_boot: int = 5000,
    seed: int = 20260820,
) -> tuple[float, float, float, int]:
    work = frame.dropna(subset=[cluster]).copy()
    point = float(statistic(work))
    groups = {key: value for key, value in work.groupby(cluster, sort=False)}
    keys = np.asarray(list(groups), dtype=object)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        boot = pd.concat([groups[key] for key in sampled], ignore_index=True)
        draws[index] = statistic(boot)
    draws = draws[np.isfinite(draws)]
    low, high = np.quantile(draws, [0.025, 0.975])
    return point, float(low), float(high), len(keys)


def stratified_period_bootstrap(
    pre: pd.DataFrame,
    post: pd.DataFrame,
    statistic: Callable[[pd.DataFrame, pd.DataFrame], float],
    n_boot: int = 5000,
    seed: int = 20260820,
) -> tuple[float, float, float]:
    point = float(statistic(pre, post))
    pre_groups = {key: value for key, value in pre.groupby("game_id", sort=False)}
    post_groups = {key: value for key, value in post.groupby("game_id", sort=False)}
    pre_keys = np.asarray(list(pre_groups), dtype=object)
    post_keys = np.asarray(list(post_groups), dtype=object)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        pre_sample = rng.choice(pre_keys, len(pre_keys), replace=True)
        post_sample = rng.choice(post_keys, len(post_keys), replace=True)
        pre_boot = pd.concat([pre_groups[key] for key in pre_sample], ignore_index=True)
        post_boot = pd.concat([post_groups[key] for key in post_sample], ignore_index=True)
        draws[index] = statistic(pre_boot, post_boot)
    draws = draws[np.isfinite(draws)]
    low, high = np.quantile(draws, [0.025, 0.975])
    return point, float(low), float(high)


def _rate_contrast_distribution(
    frame: pd.DataFrame,
    masks: Sequence[pd.Series],
    coefficients: Sequence[float],
    n_boot: int,
    seed: int,
    outcome: pd.Series | None = None,
) -> tuple[float, np.ndarray]:
    """Vectorized whole-game bootstrap for a linear contrast of rates."""
    if outcome is None:
        outcome = frame["made"]
    temp = pd.DataFrame({"game_id": frame["game_id"], "outcome": outcome}, index=frame.index)
    count_columns = []
    make_columns = []
    for index, mask in enumerate(masks):
        count_col = f"n_{index}"
        make_col = f"y_{index}"
        selected = mask.reindex(frame.index).fillna(False).astype(float)
        temp[count_col] = selected
        temp[make_col] = selected * temp["outcome"].astype(float)
        count_columns.append(count_col)
        make_columns.append(make_col)
    grouped = temp.groupby("game_id", sort=False)[count_columns + make_columns].sum()
    counts = grouped[count_columns].to_numpy(float)
    makes = grouped[make_columns].to_numpy(float)
    coefficient_array = np.asarray(coefficients, dtype=float)
    point_rates = makes.sum(axis=0) / counts.sum(axis=0)
    point = float(point_rates @ coefficient_array)
    rng = np.random.default_rng(seed)
    game_count = len(grouped)
    weights = rng.multinomial(
        game_count,
        np.full(game_count, 1.0 / game_count),
        size=n_boot,
    )
    draw_counts = weights @ counts
    draw_makes = weights @ makes
    with np.errstate(divide="ignore", invalid="ignore"):
        draw_rates = draw_makes / draw_counts
    draws = draw_rates @ coefficient_array
    return point, draws[np.isfinite(draws)]


def _mean_distribution(
    frame: pd.DataFrame,
    values: pd.Series,
    n_boot: int,
    seed: int,
) -> tuple[float, np.ndarray]:
    temp = pd.DataFrame(
        {"game_id": frame["game_id"], "value": values.astype(float), "n": 1.0},
        index=frame.index,
    )
    grouped = temp.groupby("game_id", sort=False)[["value", "n"]].sum()
    point = float(grouped["value"].sum() / grouped["n"].sum())
    rng = np.random.default_rng(seed)
    game_count = len(grouped)
    weights = rng.multinomial(
        game_count,
        np.full(game_count, 1.0 / game_count),
        size=n_boot,
    )
    sums = weights @ grouped["value"].to_numpy(float)
    counts = weights @ grouped["n"].to_numpy(float)
    return point, sums / counts


def _interval(draws: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(draws[np.isfinite(draws)], [0.025, 0.975])
    return float(low), float(high)


def _two_shot_trip_table(frame: pd.DataFrame) -> pd.DataFrame:
    ordinary = frame[frame["trip_type"].eq("ordinary_two") & frame["trip_complete"]].copy()
    pivot = ordinary.pivot_table(
        index=["trip_id", "game_id", "season_end"],
        columns="ft_num",
        values="made",
        aggfunc="first",
    ).reset_index()
    return pivot.dropna(subset=[1, 2]).rename(columns={1: "ft1_made", 2: "ft2_made"})


def build_ft1_ft2_period_table(attempts: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    trips = _two_shot_trip_table(attempts)
    periods = [
        ("career pre-rule", lambda x: x["season_end"].le(2017)),
        ("last 3 pre-rule", lambda x: x["season_end"].between(2015, 2017)),
        ("2016-17", lambda x: x["season_end"].eq(2017)),
        ("2017-18", lambda x: x["season_end"].eq(2018)),
        ("2018-19", lambda x: x["season_end"].eq(2019)),
        ("first 2 post-rule", lambda x: x["season_end"].between(2018, 2019)),
        ("first 3 post-rule", lambda x: x["season_end"].between(2018, 2020)),
    ]
    rows = []
    for label, selector in periods:
        frame = trips[selector(trips)].copy()
        point_raw, draws = _mean_distribution(
            frame,
            frame["ft2_made"] - frame["ft1_made"],
            n_boot=n_boot,
            seed=20260820 + len(rows),
        )
        low_raw, high_raw = _interval(draws)
        point, low, high = 100 * point_raw, 100 * low_raw, 100 * high_raw
        games = frame["game_id"].nunique()
        ft1_low, ft1_high = proportion_confint(frame["ft1_made"].sum(), len(frame), method="wilson")
        ft2_low, ft2_high = proportion_confint(frame["ft2_made"].sum(), len(frame), method="wilson")
        rows.append(
            {
                "period": label,
                "trips_each_position": len(frame),
                "games": games,
                "ft1_pct": 100 * frame["ft1_made"].mean(),
                "ft1_ci_low": 100 * ft1_low,
                "ft1_ci_high": 100 * ft1_high,
                "ft2_pct": 100 * frame["ft2_made"].mean(),
                "ft2_ci_low": 100 * ft2_low,
                "ft2_ci_high": 100 * ft2_high,
                "ft2_minus_ft1_pp": point,
                "diff_ci_low": low,
                "diff_ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def _paired_diff(frame: pd.DataFrame) -> float:
    return 100 * float((frame["ft2_made"] - frame["ft1_made"]).mean())


def build_did_results(attempts: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    trips = _two_shot_trip_table(attempts)
    comparisons = [
        ("last 3 pre vs first 2 post", (2015, 2017), (2018, 2019)),
        ("career pre vs first 2 post", (2009, 2017), (2018, 2019)),
        ("last 3 pre vs first 3 post", (2015, 2017), (2018, 2020)),
    ]
    rows = []
    for index, (label, pre_years, post_years) in enumerate(comparisons):
        pre = trips[trips["season_end"].between(*pre_years)]
        post = trips[trips["season_end"].between(*post_years)]
        pre_point, pre_draws = _mean_distribution(
            pre,
            pre["ft2_made"] - pre["ft1_made"],
            n_boot=n_boot,
            seed=20260820 + 2 * index,
        )
        post_point, post_draws = _mean_distribution(
            post,
            post["ft2_made"] - post["ft1_made"],
            n_boot=n_boot,
            seed=20260821 + 2 * index,
        )
        point = 100 * (post_point - pre_point)
        low_raw, high_raw = _interval(100 * (post_draws - pre_draws))
        low, high = low_raw, high_raw
        rows.append(
            {
                "comparison": label,
                "pre_trips": len(pre),
                "post_trips": len(post),
                "pre_ft2_minus_ft1_pp": _paired_diff(pre),
                "post_ft2_minus_ft1_pp": _paired_diff(post),
                "did_pp": point,
                "did_ci_low": low,
                "did_ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def category_mask(frame: pd.DataFrame, category: str) -> pd.Series:
    masks = {
        "ordinary two-shot FT1": frame["trip_type"].eq("ordinary_two") & frame["ft_num"].eq(1),
        "ordinary two-shot FT2": frame["trip_type"].eq("ordinary_two") & frame["ft_num"].eq(2),
        "and-1": frame["trip_type"].eq("and1"),
        "technical": frame["trip_type"].eq("technical"),
        "other one-shot": frame["trip_type"].eq("other_one_shot"),
        "all one-shot": frame["trip_total"].eq(1),
        "genuine one-shot": frame["genuine_one_shot"],
        "multi-shot FT1": frame["ordinary_multi_first"],
        "flagrant / clear-path": frame["trip_type"].isin(["flagrant", "clear_path"]),
    }
    return masks[category]


def build_focused_pre_post(attempts: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    periods = {
        "career pre-rule": attempts["season_end"].le(2017),
        "last 3 pre-rule": attempts["season_end"].between(2015, 2017),
        "2016-17": attempts["season_end"].eq(2017),
        "2017-18": attempts["season_end"].eq(2018),
        "2018-19": attempts["season_end"].eq(2019),
        "first 2 post-rule": attempts["season_end"].between(2018, 2019),
        "first 3 post-rule": attempts["season_end"].between(2018, 2020),
        "all post-rule": attempts["season_end"].ge(2018),
    }
    categories = [
        "ordinary two-shot FT1",
        "ordinary two-shot FT2",
        "and-1",
        "technical",
        "other one-shot",
        "all one-shot",
        "genuine one-shot",
        "multi-shot FT1",
        "flagrant / clear-path",
    ]
    rows = []
    for period_index, (period, period_mask) in enumerate(periods.items()):
        frame = attempts[period_mask]
        for category_index, category in enumerate(categories):
            sub = frame[category_mask(frame, category)]
            record = {"period": period, "category": category}
            record.update(rate_record(sub))
            if not sub.empty:
                _, draws = _rate_contrast_distribution(
                    sub,
                    [pd.Series(True, index=sub.index)],
                    [1.0],
                    n_boot,
                    20260850 + 100 * period_index + category_index,
                )
                record["cluster_ci_low"], record["cluster_ci_high"] = _interval(
                    100 * draws
                )
            rows.append(record)
    table = pd.DataFrame(rows)

    # Add game-clustered decline intervals to the first-3-post rows.
    pre = attempts[attempts["season_end"].between(2015, 2017)]
    post = attempts[attempts["season_end"].between(2018, 2020)]
    for index, category in enumerate(categories):
        pre_sub = pre[category_mask(pre, category)]
        post_sub = post[category_mask(post, category)]
        if pre_sub.empty or post_sub.empty:
            continue
        pre_point, pre_draws = _rate_contrast_distribution(
            pre_sub,
            [pd.Series(True, index=pre_sub.index)],
            [1.0],
            n_boot,
            20260900 + 2 * index,
        )
        post_point, post_draws = _rate_contrast_distribution(
            post_sub,
            [pd.Series(True, index=post_sub.index)],
            [1.0],
            n_boot,
            20260901 + 2 * index,
        )
        point = 100 * (post_point - pre_point)
        low, high = _interval(100 * (post_draws - pre_draws))
        target = table["period"].eq("first 3 post-rule") & table["category"].eq(category)
        table.loc[target, "change_vs_last3_pre_pp"] = point
        table.loc[target, "change_ci_low"] = low
        table.loc[target, "change_ci_high"] = high
    return table


def build_one_shot_contrasts(attempts: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    periods = [
        ("career pre vs first 3 post", (2009, 2017), (2018, 2020)),
        ("last 3 pre vs first 3 post", (2015, 2017), (2018, 2020)),
    ]
    contrasts = [
        ("and-1 minus ordinary two-shot FT1", "and-1", "ordinary two-shot FT1"),
        ("genuine one-shot minus multi-shot FT1", "genuine one-shot", "multi-shot FT1"),
        ("technical minus ordinary two-shot FT1", "technical", "ordinary two-shot FT1"),
    ]
    rows = []
    for p_index, (period_label, pre_years, post_years) in enumerate(periods):
        pre_all = attempts[attempts["season_end"].between(*pre_years)]
        post_all = attempts[attempts["season_end"].between(*post_years)]
        for c_index, (label, category_a, category_b) in enumerate(contrasts):
            def period_diff(frame: pd.DataFrame) -> float:
                a = frame[category_mask(frame, category_a)]["made"].mean()
                b = frame[category_mask(frame, category_b)]["made"].mean()
                return 100 * float(a - b)

            pre_point, pre_draws = _rate_contrast_distribution(
                pre_all,
                [category_mask(pre_all, category_a), category_mask(pre_all, category_b)],
                [1.0, -1.0],
                n_boot,
                20261000 + 20 * p_index + 2 * c_index,
            )
            post_point, post_draws = _rate_contrast_distribution(
                post_all,
                [category_mask(post_all, category_a), category_mask(post_all, category_b)],
                [1.0, -1.0],
                n_boot,
                20261001 + 20 * p_index + 2 * c_index,
            )
            point = 100 * (post_point - pre_point)
            low, high = _interval(100 * (post_draws - pre_draws))
            pre_low, pre_high = _interval(100 * pre_draws)
            post_low, post_high = _interval(100 * post_draws)
            rows.append(
                {
                    "period_comparison": period_label,
                    "contrast": label,
                    "pre_difference_pp": period_diff(pre_all),
                    "pre_difference_ci_low": pre_low,
                    "pre_difference_ci_high": pre_high,
                    "post_difference_pp": period_diff(post_all),
                    "post_difference_ci_low": post_low,
                    "post_difference_ci_high": post_high,
                    "did_pp": point,
                    "did_ci_low": low,
                    "did_ci_high": high,
                    "pre_a_n": int(category_mask(pre_all, category_a).sum()),
                    "pre_b_n": int(category_mask(pre_all, category_b).sum()),
                    "post_a_n": int(category_mask(post_all, category_a).sum()),
                    "post_b_n": int(category_mask(post_all, category_b).sum()),
                }
            )
    return pd.DataFrame(rows)


def build_conditional_ft2(attempts: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    trips = _two_shot_trip_table(attempts)
    periods = {
        "career pre-rule": trips["season_end"].le(2017),
        "last 3 pre-rule": trips["season_end"].between(2015, 2017),
        "first 2 post-rule": trips["season_end"].between(2018, 2019),
        "first 3 post-rule": trips["season_end"].between(2018, 2020),
        "all post-rule": trips["season_end"].ge(2018),
    }
    rows = []
    for index, (label, mask) in enumerate(periods.items()):
        frame = trips[mask].copy()
        after_make = frame[frame["ft1_made"].eq(1)]
        after_miss = frame[frame["ft1_made"].eq(0)]

        point_raw, draws = _rate_contrast_distribution(
            frame,
            [frame["ft1_made"].eq(0), frame["ft1_made"].eq(1)],
            [1.0, -1.0],
            n_boot,
            20261100 + index,
            outcome=frame["ft2_made"],
        )
        point = 100 * point_raw
        low_raw, high_raw = _interval(100 * draws)
        low, high = low_raw, high_raw
        games = frame["game_id"].nunique()
        record = {
            "period": label,
            "games": games,
            "ft2_after_ft1_make_pct": 100 * after_make["ft2_made"].mean(),
            "n_after_make": len(after_make),
            "ft2_after_ft1_miss_pct": 100 * after_miss["ft2_made"].mean(),
            "n_after_miss": len(after_miss),
            "after_miss_minus_after_make_pp": point,
            "contrast_ci_low": low,
            "contrast_ci_high": high,
        }
        rows.append(record)
    table = pd.DataFrame(rows)

    pre = trips[trips["season_end"].between(2015, 2017)]
    post = trips[trips["season_end"].between(2018, 2020)]

    pre_point, pre_draws = _rate_contrast_distribution(
        pre,
        [pre["ft1_made"].eq(0), pre["ft1_made"].eq(1)],
        [1.0, -1.0],
        n_boot,
        20261200,
        outcome=pre["ft2_made"],
    )
    post_point, post_draws = _rate_contrast_distribution(
        post,
        [post["ft1_made"].eq(0), post["ft1_made"].eq(1)],
        [1.0, -1.0],
        n_boot,
        20261201,
        outcome=post["ft2_made"],
    )
    point = 100 * (post_point - pre_point)
    low, high = _interval(100 * (post_draws - pre_draws))
    table.attrs["correction_interaction"] = {
        "last3_pre_vs_first3_post_pp": point,
        "ci_low": low,
        "ci_high": high,
    }
    return table


def _beta_binomial_mle(season: pd.DataFrame) -> tuple[float, float]:
    y = season["makes"].to_numpy(dtype=int)
    n = season["attempts"].to_numpy(dtype=int)

    def objective(params: np.ndarray) -> float:
        mu = special.expit(params[0])
        phi = np.exp(params[1])
        alpha = mu * phi
        beta = (1 - mu) * phi
        logp = (
            special.gammaln(n + 1)
            - special.gammaln(y + 1)
            - special.gammaln(n - y + 1)
            + special.betaln(y + alpha, n - y + beta)
            - special.betaln(alpha, beta)
        )
        return -float(logp.sum())

    start_mu = y.sum() / n.sum()
    result = opt.minimize(
        objective,
        np.array([special.logit(start_mu), math.log(100.0)]),
        method="L-BFGS-B",
        bounds=[(-8, 8), (-4, 12)],
    )
    if not result.success:
        raise RuntimeError(f"Beta-binomial MLE failed: {result.message}")
    mu = special.expit(result.x[0])
    phi = np.exp(result.x[1])
    return float(mu * phi), float((1 - mu) * phi)


def build_extremeness_table(
    attempts: pd.DataFrame,
    season_table: pd.DataFrame,
    n_boot: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    pre = attempts[attempts["season_end"].le(2017)]
    baseline = pre["made"].mean()
    pre_by_season = (
        pre.groupby("season_end", observed=True)["made"]
        .agg(makes="sum", attempts="size")
        .reset_index()
    )
    alpha, beta = _beta_binomial_mle(pre_by_season)
    mu = alpha / (alpha + beta)
    season_sd = math.sqrt(mu * (1 - mu) / (alpha + beta + 1))

    periods = [
        ("pre-2017 baseline", attempts["season_end"].le(2017)),
        ("2017-18", attempts["season_end"].eq(2018)),
        ("2018-19", attempts["season_end"].eq(2019)),
        ("first two post-rule seasons", attempts["season_end"].between(2018, 2019)),
    ]
    rows = []
    for index, (label, mask) in enumerate(periods):
        frame = attempts[mask]
        record = {"period": label, **rate_record(frame)}
        _, draws = _rate_contrast_distribution(
            frame,
            [pd.Series(True, index=frame.index)],
            [1.0],
            n_boot,
            20261300 + index,
        )
        record["game_cluster_ci_low"], record["game_cluster_ci_high"] = _interval(
            100 * draws
        )
        record["decline_from_pre_pp"] = record["pct"] - 100 * baseline
        if label != "pre-2017 baseline":
            y, n = int(frame["made"].sum()), len(frame)
            record["binomial_lower_tail_p"] = float(st.binom.cdf(y, n, baseline))
            if label == "first two post-rule seasons":
                first = attempts[attempts["season_end"].eq(2018)]
                second = attempts[attempts["season_end"].eq(2019)]
                pmf1 = st.betabinom.pmf(
                    np.arange(len(first) + 1), len(first), alpha, beta
                )
                pmf2 = st.betabinom.pmf(
                    np.arange(len(second) + 1), len(second), alpha, beta
                )
                combined = np.convolve(pmf1, pmf2)
                record["beta_binomial_lower_tail_p"] = float(
                    combined[: y + 1].sum()
                )
            else:
                record["beta_binomial_lower_tail_p"] = float(
                    st.betabinom.cdf(y, n, alpha, beta)
                )
        rows.append(record)
    meta = {
        "pre_baseline": float(baseline),
        "beta_binomial_alpha": alpha,
        "beta_binomial_beta": beta,
        "beta_binomial_mean": mu,
        "pre_season_ability_sd": season_sd,
    }
    return pd.DataFrame(rows), meta


def _model_category(frame: pd.DataFrame) -> pd.Series:
    category = pd.Series(index=frame.index, dtype=object)
    category.loc[frame["trip_type"].eq("ordinary_two") & frame["ft_num"].eq(2)] = "two_second"
    category.loc[frame["trip_type"].eq("ordinary_two") & frame["ft_num"].eq(1)] = "two_first"
    category.loc[frame["genuine_one_shot"]] = "genuine_one"
    return category


def fit_clustered_logistic_model(attempts: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    work = attempts[attempts["season_end"].between(2013, 2020)].copy()
    work["category"] = _model_category(work)
    work = work.dropna(
        subset=[
            "category",
            "made",
            "game_id",
            "period",
            "pre_score_margin_shooter",
            "elapsed_game_minute",
        ]
    ).copy()
    work["post"] = work["season_end"].ge(2018).astype(int)
    work["season_center"] = work["season_end"] - 2017
    work["margin_clip"] = work["pre_score_margin_shooter"].clip(-20, 20)
    work["home"] = work["is_home"].astype(int)
    work["b2b"] = work["back_to_back"].astype(int)
    work["period_group"] = np.where(work["period"].le(4), work["period"].astype(str), "OT")
    formula = (
        "made ~ post * C(category, Treatment(reference='two_second')) + "
        "season_center + C(period_group) + home + b2b + margin_clip + elapsed_game_minute"
    )
    model = sm.GLM.from_formula(formula, data=work, family=sm.families.Binomial())
    result = model.fit(cov_type="cluster", cov_kwds={"groups": work["game_id"]})

    design_info = result.model.data.design_info
    base = work.copy()
    base["season_center"] = 0

    def prediction_gradient(post: int, category: str) -> tuple[float, np.ndarray]:
        counterfactual = base.copy()
        counterfactual["post"] = post
        counterfactual["category"] = category
        matrix = np.asarray(
            patsy.build_design_matrices([design_info], counterfactual, return_type="dataframe")[0]
        )
        linear = matrix @ result.params.to_numpy()
        probability = special.expit(linear)
        gradient = (probability * (1 - probability))[:, None] * matrix
        return float(probability.mean()), gradient.mean(axis=0)

    predictions: dict[tuple[int, str], tuple[float, np.ndarray]] = {}
    for post in (0, 1):
        for category in ("two_second", "two_first", "genuine_one"):
            predictions[(post, category)] = prediction_gradient(post, category)

    covariance = np.asarray(result.cov_params())

    def contrast(
        label: str,
        signed_cells: Sequence[tuple[float, tuple[int, str]]],
    ) -> dict[str, object]:
        value = sum(weight * predictions[cell][0] for weight, cell in signed_cells)
        gradient = sum(weight * predictions[cell][1] for weight, cell in signed_cells)
        se = float(np.sqrt(gradient @ covariance @ gradient))
        return {
            "effect": label,
            "estimate_pp": 100 * value,
            "ci_low": 100 * (value - 1.96 * se),
            "ci_high": 100 * (value + 1.96 * se),
        }

    rows = [
        contrast("modeled post change: ordinary two-shot FT2", [(1, (1, "two_second")), (-1, (0, "two_second"))]),
        contrast("modeled post change: ordinary two-shot FT1", [(1, (1, "two_first")), (-1, (0, "two_first"))]),
        contrast("modeled post change: genuine one-shot", [(1, (1, "genuine_one")), (-1, (0, "genuine_one"))]),
        contrast(
            "additional modeled FT1 decline versus FT2",
            [
                (-1, (1, "two_first")),
                (1, (0, "two_first")),
                (1, (1, "two_second")),
                (-1, (0, "two_second")),
            ],
        ),
        contrast(
            "additional modeled genuine-one-shot decline versus FT2",
            [
                (-1, (1, "genuine_one")),
                (1, (0, "genuine_one")),
                (1, (1, "two_second")),
                (-1, (0, "two_second")),
            ],
        ),
    ]
    meta = {
        "formula": formula,
        "seasons": "2012-13 through 2019-20",
        "attempts": int(len(work)),
        "games": int(work["game_id"].nunique()),
        "covariance": "game-clustered sandwich",
        "coefficient_table": [
            {
                "term": str(term),
                "coefficient_log_odds": float(value),
                "standard_error": float(result.bse[term]),
            }
            for term, value in result.params.items()
        ],
    }
    return pd.DataFrame(rows), meta


def build_composition_decomposition(attempts: pd.DataFrame) -> pd.DataFrame:
    work = attempts.copy()
    work["composition_category"] = np.select(
        [
            work["trip_type"].eq("ordinary_two") & work["ft_num"].eq(1),
            work["trip_type"].eq("ordinary_two") & work["ft_num"].eq(2),
            work["trip_type"].eq("ordinary_three") & work["ft_num"].eq(1),
            work["trip_type"].eq("ordinary_three") & work["ft_num"].eq(2),
            work["trip_type"].eq("ordinary_three") & work["ft_num"].eq(3),
            work["trip_type"].eq("and1"),
            work["trip_type"].eq("technical"),
            work["trip_type"].eq("other_one_shot"),
            work["trip_type"].isin(["flagrant", "clear_path"]),
        ],
        [
            "two_ft1",
            "two_ft2",
            "three_ft1",
            "three_ft2",
            "three_ft3",
            "and1",
            "technical",
            "other_one",
            "unusual_multi",
        ],
        default="unresolved",
    )
    pre = work[work["season_end"].between(2015, 2017)]
    post = work[work["season_end"].between(2018, 2020)]
    pre_rate = pre.groupby("composition_category")["made"].mean()
    pre_mix = pre["composition_category"].value_counts(normalize=True)
    post_mix = post["composition_category"].value_counts(normalize=True)
    cats = sorted(set(pre_rate.index) | set(pre_mix.index) | set(post_mix.index))
    pre_observed = pre["made"].mean()
    post_observed = post["made"].mean()
    counterfactual = sum(post_mix.get(cat, 0) * pre_rate.get(cat, pre_observed) for cat in cats)
    return pd.DataFrame(
        [
            {"quantity": "last-3-pre observed FT%", "value_pct": 100 * pre_observed},
            {"quantity": "first-3-post observed FT%", "value_pct": 100 * post_observed},
            {
                "quantity": "post mix at pre category rates",
                "value_pct": 100 * counterfactual,
            },
            {
                "quantity": "composition contribution to change",
                "value_pct": 100 * (counterfactual - pre_observed),
            },
            {
                "quantity": "within-category performance contribution",
                "value_pct": 100 * (post_observed - counterfactual),
            },
        ]
    )


def build_monthly_table(attempts: pd.DataFrame) -> pd.DataFrame:
    season = attempts[attempts["season_end"].eq(2018)].copy()
    season["month"] = season["game_date"].dt.to_period("M").astype(str)
    rows = []
    for month, frame in season.groupby("month", sort=True):
        for category in ["overall", "ordinary two-shot FT1", "ordinary two-shot FT2", "genuine one-shot"]:
            sub = frame if category == "overall" else frame[category_mask(frame, category)]
            rows.append({"month": month, "category": category, **rate_record(sub)})
    return pd.DataFrame(rows)


def build_rolling_diagnostics(attempts: pd.DataFrame) -> pd.DataFrame:
    work = attempts.sort_values("career_attempt").copy()
    baseline = 100 * work.loc[work["season_end"].le(2017), "made"].mean()
    boundary = int(work.loc[work["season_end"].eq(2018), "career_attempt"].min())
    rows = []
    for window in (50, 100, 200):
        rolling = 100 * work["made"].rolling(window, min_periods=window).mean()
        eligible = (
            work["career_attempt"].ge(boundary + window - 1)
            & work["season_end"].between(2018, 2020)
        )
        eligible_values = rolling[eligible]
        max_index = eligible_values.idxmax()
        returned = eligible & rolling.ge(baseline)
        first_return_index = rolling[returned].index.min() if returned.any() else None
        rows.append(
            {
                "window_attempts": window,
                "pre_rule_baseline_pct": baseline,
                "post_window_max_pct": float(rolling.loc[max_index]),
                "post_window_max_date": str(work.loc[max_index, "game_date"].date()),
                "first_return_to_baseline_date": (
                    str(work.loc[first_return_index, "game_date"].date())
                    if first_return_index is not None
                    else ""
                ),
                "share_of_first3_post_windows_at_or_above_baseline": float(
                    rolling[eligible].ge(baseline).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_change_point(attempts: pd.DataFrame) -> dict[str, object]:
    work = attempts[attempts["season_end"].between(2015, 2019)].sort_values("career_attempt").copy()
    outcomes = work["made"].to_numpy(dtype=int)
    boundary = int(np.searchsorted(work["season_end"].to_numpy(), 2018))
    candidates = range(max(150, boundary - 300), min(len(work) - 150, boundary + 301))

    def bernoulli_loglike(values: np.ndarray) -> float:
        p = np.clip(values.mean(), 1e-8, 1 - 1e-8)
        return float((values * np.log(p) + (1 - values) * np.log(1 - p)).sum())

    ll_null = bernoulli_loglike(outcomes)
    scored = []
    for split in candidates:
        ll = bernoulli_loglike(outcomes[:split]) + bernoulli_loglike(outcomes[split:])
        scored.append((ll, split))
    best_ll, best_split = max(scored)
    fixed_ll = bernoulli_loglike(outcomes[:boundary]) + bernoulli_loglike(outcomes[boundary:])
    best_row = work.iloc[best_split]
    return {
        "window": "2014-15 through 2018-19",
        "candidate_range": "within 300 attempts of the 2017-18 boundary, at least 150 per side",
        "best_split_career_attempt": int(best_row["career_attempt"]),
        "best_split_date": str(best_row["game_date"].date()),
        "attempts_from_rule_boundary": int(best_split - boundary),
        "best_pre_pct": 100 * float(outcomes[:best_split].mean()),
        "best_post_pct": 100 * float(outcomes[best_split:].mean()),
        "best_two_rate_bic": float(-2 * best_ll + 2 * math.log(len(outcomes))),
        "fixed_rule_boundary_bic": float(-2 * fixed_ll + 2 * math.log(len(outcomes))),
        "one_rate_bic": float(-2 * ll_null + math.log(len(outcomes))),
        "warning": "Descriptive Bernoulli change-point only; ignores game/trip dependence and was searched after seeing the data.",
    }


def build_peer_comparison(raw_dir: Path) -> pd.DataFrame:
    season_rows = []
    for season_end in (2017, 2018):
        path = raw_dir / f"NBA{season_end % 100:02d}.parquet"
        raw = _read_columns(
            path,
            [
                "game_id",
                "event_num",
                "event_type",
                "player1_id",
                "player1_name",
                "home_description",
                "neutral_description",
                "visitor_description",
            ],
        ).drop_duplicates()
        raw["event_type_num"] = pd.to_numeric(raw["event_type"], errors="coerce")
        raw["description"] = _description(raw)
        ft = raw[
            raw["event_type_num"].eq(3)
            & raw["description"].str.contains("Free Throw", case=False, na=False)
            & pd.to_numeric(raw["player1_id"], errors="coerce").gt(0)
        ].copy()
        ft["made"] = (
            ~ft["description"].str.contains(r"\bMISS\b", case=False, regex=True)
            & ft["description"].str.contains(r"\(\d+ PTS\)", case=False, regex=True)
        )
        grouped = (
            ft.groupby(["player1_id", "player1_name"], dropna=False)["made"]
            .agg(makes="sum", attempts="size")
            .reset_index()
        )
        grouped["season_end"] = season_end
        season_rows.append(grouped)
    panel = pd.concat(season_rows).pivot_table(
        index=["player1_id", "player1_name"],
        columns="season_end",
        values=["makes", "attempts"],
        aggfunc="first",
    )
    panel.columns = [f"{field}_{season}" for field, season in panel.columns]
    panel = panel.reset_index().dropna()
    panel = panel[(panel["attempts_2017"] >= 300) & (panel["attempts_2018"] >= 300)].copy()
    panel["ft_pct_2017"] = 100 * panel["makes_2017"] / panel["attempts_2017"]
    panel["ft_pct_2018"] = 100 * panel["makes_2018"] / panel["attempts_2018"]
    panel["change_pp"] = panel["ft_pct_2018"] - panel["ft_pct_2017"]
    panel["change_rank_worst"] = panel["change_pp"].rank(method="min", ascending=True).astype(int)
    return panel.sort_values("change_pp").reset_index(drop=True)


def _plot_season_chart(season: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.8))
    x = np.arange(len(season))
    y = season["overall_pct"]
    yerr = np.vstack([y - season["overall_ci_low"], season["overall_ci_high"] - y])
    ax.errorbar(x, y, yerr=yerr, marker="o", color="#1d3557", capsize=3, lw=1.8)
    ax.axvline(np.where(season["season_end"].eq(2018))[0][0] - 0.5, color="#d62828", ls="--", lw=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(season["season"], rotation=45, ha="right")
    ax.set_ylabel("FT% (Wilson 95% interval)")
    ax.set_title("Russell Westbrook regular-season free-throw percentage")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_ft1_ft2(season: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.8))
    x = np.arange(len(season))
    ax.plot(x, season["two_first_pct"], marker="o", label="FT1, ordinary 2-shot trip", color="#d95f02")
    ax.plot(x, season["two_second_pct"], marker="o", label="FT2, ordinary 2-shot trip", color="#1b9e77")
    ax.axvline(np.where(season["season_end"].eq(2018))[0][0] - 0.5, color="#d62828", ls="--", lw=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(season["season"], rotation=45, ha="right")
    ax.set_ylabel("FT%")
    ax.set_title("Where the decline occurred within ordinary two-shot trips")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_category_changes(focused: pd.DataFrame, output: Path) -> None:
    plot = focused[
        focused["period"].eq("first 3 post-rule")
        & focused["category"].isin(
            [
                "ordinary two-shot FT1",
                "ordinary two-shot FT2",
                "and-1",
                "genuine one-shot",
                "technical",
            ]
        )
    ].copy()
    plot = plot.sort_values("change_vs_last3_pre_pp")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    y = np.arange(len(plot))
    x = plot["change_vs_last3_pre_pp"]
    xerr = np.vstack([x - plot["change_ci_low"], plot["change_ci_high"] - x])
    ax.errorbar(x, y, xerr=xerr, fmt="o", color="#264653", capsize=4)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot["category"])
    ax.set_xlabel("Change from 2014-15–2016-17 to first 3 post seasons (pp)\nGame-clustered 95% bootstrap interval")
    ax.set_title("Context-specific free-throw changes")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_rolling(attempts: pd.DataFrame, output: Path) -> None:
    plot = attempts.sort_values("career_attempt").copy()
    boundary = plot.loc[plot["season_end"].eq(2018), "career_attempt"].min()
    plot = plot[plot["season_end"].between(2015, 2020)]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    colors = {50: "#e76f51", 100: "#2a9d8f", 200: "#264653"}
    for window in (50, 100, 200):
        rolling = plot["made"].rolling(window, min_periods=window).mean() * 100
        ax.plot(plot["career_attempt"], rolling, label=f"trailing {window}", color=colors[window], alpha=0.9)
    ax.axvline(boundary, color="#d62828", ls="--", lw=1.5, label="start of 2017-18")
    ax.set_xlabel("Career regular-season FT attempt")
    ax.set_ylabel("Trailing FT%")
    ax.set_title("Rolling free-throw percentage around the intervention")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_ft2_conditionals(conditional: pd.DataFrame, output: Path) -> None:
    plot = conditional[conditional["period"].isin(["last 3 pre-rule", "first 3 post-rule"])].copy()
    x = np.arange(len(plot))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(x - width / 2, plot["ft2_after_ft1_make_pct"], width, label="FT2 after FT1 make", color="#457b9d")
    ax.bar(x + width / 2, plot["ft2_after_ft1_miss_pct"], width, label="FT2 after FT1 miss", color="#e9c46a")
    ax.set_xticks(x)
    ax.set_xticklabels(plot["period"])
    ax.set_ylim(50, 100)
    ax.set_ylabel("FT2%")
    ax.set_title("Does FT1 outcome predict FT2 differently after the rule change?")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_outputs(
    output_dir: Path,
    attempts: pd.DataFrame,
    audit: dict[str, object],
    n_boot: int,
    raw_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    attempts.to_parquet(output_dir / "westbrook_ft_attempts.parquet", index=False)

    season = build_season_table(attempts)
    focused = build_focused_pre_post(attempts, n_boot=n_boot)
    ft1_ft2 = build_ft1_ft2_period_table(attempts, n_boot=n_boot)
    did = build_did_results(attempts, n_boot=n_boot)
    one_shot = build_one_shot_contrasts(attempts, n_boot=n_boot)
    conditional = build_conditional_ft2(attempts, n_boot=n_boot)
    extremeness, beta_meta = build_extremeness_table(attempts, season, n_boot=n_boot)
    model_effects, model_meta = fit_clustered_logistic_model(attempts)
    composition = build_composition_decomposition(attempts)
    monthly = build_monthly_table(attempts)
    rolling_diagnostics = build_rolling_diagnostics(attempts)
    change_point = build_change_point(attempts)
    peers = build_peer_comparison(raw_dir)

    tables = {
        "season_table.csv": season,
        "focused_pre_post.csv": focused,
        "ft1_ft2_periods.csv": ft1_ft2,
        "ft1_ft2_did.csv": did,
        "one_shot_contrasts.csv": one_shot,
        "conditional_ft2.csv": conditional,
        "extremeness.csv": extremeness,
        "clustered_logistic_effects.csv": model_effects,
        "composition_decomposition.csv": composition,
        "monthly_2017_18.csv": monthly,
        "rolling_diagnostics.csv": rolling_diagnostics,
        "peer_comparison_2017_to_2018.csv": peers,
    }
    for name, table in tables.items():
        table.to_csv(output_dir / name, index=False)

    _plot_season_chart(season, output_dir / "01_season_ft_pct.png")
    _plot_ft1_ft2(season, output_dir / "02_ft1_ft2_by_season.png")
    _plot_category_changes(focused, output_dir / "03_category_changes.png")
    _plot_rolling(attempts, output_dir / "04_rolling_ft_pct.png")
    _plot_ft2_conditionals(conditional, output_dir / "05_ft2_conditionals.png")

    audit["conditional_correction_interaction"] = conditional.attrs[
        "correction_interaction"
    ]
    audit["beta_binomial"] = beta_meta
    audit["clustered_logistic_model"] = model_meta
    audit["change_point"] = change_point
    with (output_dir / "audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, default=str)

    summary = {
        "pre_baseline_pct": float(
            extremeness.loc[extremeness["period"].eq("pre-2017 baseline"), "pct"].iloc[0]
        ),
        "season_2017_18_pct": float(
            extremeness.loc[extremeness["period"].eq("2017-18"), "pct"].iloc[0]
        ),
        "season_2018_19_pct": float(
            extremeness.loc[extremeness["period"].eq("2018-19"), "pct"].iloc[0]
        ),
        "first_two_post_pct": float(
            extremeness.loc[
                extremeness["period"].eq("first two post-rule seasons"), "pct"
            ].iloc[0]
        ),
        "did_primary": did.iloc[0].to_dict(),
        "one_shot_primary": one_shot[
            one_shot["period_comparison"].eq("last 3 pre vs first 3 post")
        ].to_dict(orient="records"),
        "conditional_interaction": conditional.attrs["correction_interaction"],
        "model_effects": model_effects.to_dict(orient="records"),
        "composition": composition.to_dict(orient="records"),
        "rolling_diagnostics": rolling_diagnostics.to_dict(orient="records"),
        "change_point": change_point,
        "peer_count": int(len(peers)),
        "westbrook_peer_row": peers[
            pd.to_numeric(peers["player1_id"], errors="coerce").eq(WESTBROOK_ID)
        ].to_dict(orient="records"),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    return summary


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=root / "raw_data")
    parser.add_argument("--game-dates", type=Path, default=DEFAULT_GAME_DATES)
    parser.add_argument("--supplemental-game-dates", type=Path, default=DEFAULT_2026_CROSSWALK)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results" / "westbrook_free_throw_routine",
    )
    parser.add_argument("--start-year", type=int, default=2009)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    attempts, audit = build_westbrook_attempts(
        args.raw_dir,
        args.game_dates,
        supplemental_game_dates_path=args.supplemental_game_dates,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    summary = write_outputs(
        args.output_dir,
        attempts,
        audit,
        n_boot=args.bootstrap_reps,
        raw_dir=args.raw_dir,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
