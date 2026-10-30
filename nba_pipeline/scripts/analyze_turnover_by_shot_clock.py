#!/usr/bin/env python3
"""Describe when turnovers occur within inferred NBA shot-clock cycles.

This is an exploratory event-level diagnostic, not a RAPM input.  It reuses
the repo's shot-clock state machine by inserting marker rows immediately
before turnovers and shooting-foul trips.  The marker receives the live clock
estimate, while the original event still applies the normal reset rule.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nba_pipeline.scripts.build_shot_clock_estimates import (
    estimate_shot_clock,
    normalize_clock_seconds,
)
from nba_pipeline.scripts.calibrate_shot_clock_estimates import apply_calibrator


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (
    PIPELINE_ROOT
    / "validation"
    / "shot_clock_calibration"
    / "shot_clock_2015_absolute_model.joblib"
)
DEFAULT_OUTPUT = PIPELINE_ROOT / "results" / "turnover_shot_clock"
MARKER_BASE = -9_000_000_000


def _all_description(frame: pd.DataFrame) -> pd.Series:
    return (
        frame.get("home_description", pd.Series("", index=frame.index))
        .fillna("")
        .astype(str)
        + " "
        + frame.get("visitor_description", pd.Series("", index=frame.index))
        .fillna("")
        .astype(str)
        + " "
        + frame.get("neutral_description", pd.Series("", index=frame.index))
        .fillna("")
        .astype(str)
    ).str.strip()


def _normalize_event_type(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    mapping = {
        1: "MAKE",
        2: "MISS",
        3: "FreeThrow",
        4: "Rebound",
        5: "Turnover",
        6: "Foul",
        7: "Violation",
        10: "JumpBall",
        12: "StartOfPeriod",
        13: "EndOfPeriod",
    }
    return numeric.map(mapping).fillna(series.astype(str))


def classify_marker_events(raw: pd.DataFrame) -> pd.DataFrame:
    """Return raw events that need a synthetic clock-observation marker."""
    work = raw.copy()
    work["_event_name"] = _normalize_event_type(work["event_type"])
    work["_desc"] = _all_description(work)
    group = work.groupby(["game_id", "period"], sort=False)

    next_event = group["_event_name"].shift(-1)

    is_turnover = work["_event_name"].eq("Turnover")
    is_offensive_foul = (
        work["_event_name"].eq("Foul")
        & work["_desc"].str.contains(r"offensive|charge", case=False, regex=True, na=False)
    )
    offensive_foul_without_companion = is_offensive_foul & ~next_event.eq("Turnover")

    # Map each ordinary FT backward to the nearest foul at the same game-clock
    # timestamp. This is safer than a fixed look-ahead window when substitutions
    # or replay rows sit between the foul and the first FT. And-one fouls remain
    # attached to the made FGA rather than becoming a separate attempt endpoint.
    excluded_ft = work["_desc"].str.contains(
        r"technical|flagrant|clear path|transition take|offensive|charge",
        case=False,
        regex=True,
        na=False,
    )
    is_foul = work["_event_name"].eq("Foul")
    is_fga = work["_event_name"].isin(["MAKE", "MISS"])
    row_number = pd.Series(np.arange(len(work), dtype=float), index=work.index)
    last_foul_row = row_number.where(is_foul).groupby(
        [work["game_id"], work["period"]], sort=False
    ).ffill()
    last_foul_clock = work["time_quarter"].where(is_foul).groupby(
        [work["game_id"], work["period"]], sort=False
    ).ffill()
    last_fga_clock = work["time_quarter"].where(is_fga).groupby(
        [work["game_id"], work["period"]], sort=False
    ).ffill()
    last_fga_make = work["_event_name"].eq("MAKE").astype("boolean").where(is_fga).groupby(
        [work["game_id"], work["period"]], sort=False
    ).ffill().fillna(False)
    and_one_foul = is_foul & last_fga_make & last_fga_clock.eq(work["time_quarter"])
    ordinary_ft = work["_event_name"].eq("FreeThrow") & ~excluded_ft
    ft_foul_rows = last_foul_row[
        ordinary_ft & last_foul_clock.eq(work["time_quarter"])
    ].dropna().astype(int)
    is_ft_trip = pd.Series(False, index=work.index)
    is_ft_trip.iloc[ft_foul_rows.unique()] = True
    is_ft_trip &= ~and_one_foul & ~excluded_ft

    selected = work[is_turnover | offensive_foul_without_companion | is_ft_trip].copy()
    selected["endpoint_type"] = np.select(
        [is_turnover.loc[selected.index], offensive_foul_without_companion.loc[selected.index]],
        ["turnover", "offensive_foul_no_tov"],
        default="ft_trip",
    )
    selected["marker_id"] = MARKER_BASE - np.arange(1, len(selected) + 1, dtype=np.int64)
    selected["source_event_num"] = pd.to_numeric(selected["event_num"], errors="coerce").fillna(0).astype(int)
    turnover_desc = selected["_desc"]
    selected["turnover_type"] = np.select(
        [
            selected["endpoint_type"].eq("offensive_foul_no_tov")
            | turnover_desc.str.contains(r"offensive foul|charge", case=False, regex=True, na=False),
            turnover_desc.str.contains("bad pass", case=False, na=False),
            turnover_desc.str.contains("lost ball", case=False, na=False),
            turnover_desc.str.contains(r"shot clock|24 second", case=False, regex=True, na=False),
            turnover_desc.str.contains(
                r"travel|double dribble|palming|3 second|8 second|backcourt|out of bounds|stepped out|lane violation",
                case=False,
                regex=True,
                na=False,
            ),
            selected["endpoint_type"].eq("turnover"),
        ],
        ["offensive_foul", "bad_pass", "lost_ball", "shot_clock", "other_violation", "team_or_other"],
        default="not_turnover",
    )
    return selected


def build_endpoint_estimates(
    raw: pd.DataFrame,
    *,
    season_end_year: int,
    calibrator: dict[str, object] | None,
) -> pd.DataFrame:
    """Estimate live clocks for FGA, turnover, offensive-foul, and FT endpoints."""
    raw = raw.drop_duplicates().copy()
    raw["game_id"] = raw["game_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)
    raw["_clock_sec_sort"] = normalize_clock_seconds(raw)
    raw["_event_num_sort"] = pd.to_numeric(raw["event_num"], errors="coerce").fillna(0)
    raw["_input_order"] = np.arange(len(raw), dtype=np.int64)
    raw = raw.sort_values(
        ["game_id", "period", "_clock_sec_sort", "_event_num_sort", "_input_order"],
        ascending=[True, True, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    raw["_source_order"] = np.arange(len(raw), dtype=np.int64)
    markers = classify_marker_events(raw)

    synthetic = markers[raw.columns].copy()
    synthetic["event_type"] = 2  # Observe the clock as a synthetic missed FGA.
    synthetic["player1_id"] = markers["marker_id"].to_numpy()
    synthetic["player1_name"] = markers["endpoint_type"].to_numpy()
    synthetic["_source_order"] = synthetic["_source_order"] - 0.5

    augmented = pd.concat([raw, synthetic], ignore_index=True).sort_values(
        "_source_order", kind="stable"
    )
    estimates = estimate_shot_clock(augmented, season_end_year=season_end_year)
    estimates["shooter_id"] = pd.to_numeric(estimates["shooter_id"], errors="coerce")

    marker_key = markers[
        ["marker_id", "endpoint_type", "source_event_num", "turnover_type"]
    ].rename(
        columns={"marker_id": "shooter_id"}
    )
    marker_estimate_rows = estimates[estimates["shooter_id"].lt(MARKER_BASE + 1)].copy()
    duplicate_marker_rows = marker_estimate_rows.duplicated("shooter_id", keep=False)
    if duplicate_marker_rows.any():
        duplicated = marker_estimate_rows.loc[
            duplicate_marker_rows,
            ["game_id", "period", "event_num", "shooter_id", "clock_sec"],
        ]
        if duplicated.drop_duplicates().duplicated("shooter_id", keep=False).any():
            raise ValueError("Synthetic clock markers mapped to conflicting event estimates")
        marker_estimate_rows = marker_estimate_rows.drop_duplicates("shooter_id", keep="first")
    synthetic_estimates = marker_estimate_rows.merge(
        marker_key, on="shooter_id", how="inner", validate="one_to_one"
    )
    synthetic_estimates["event_num"] = synthetic_estimates["source_event_num"]

    real_fga = estimates[estimates["shooter_id"].ge(0) | estimates["shooter_id"].isna()].copy()
    real_fga["endpoint_type"] = "fga"
    real_fga["turnover_type"] = "not_turnover"
    endpoints = pd.concat([real_fga, synthetic_estimates], ignore_index=True, sort=False)
    endpoints["is_turnover"] = endpoints["endpoint_type"].isin(
        ["turnover", "offensive_foul_no_tov"]
    )
    if calibrator is not None:
        endpoints = apply_calibrator(endpoints, calibrator)
        endpoints["clock"] = endpoints["shot_clock_calibrated"]
    else:
        endpoints["clock"] = endpoints["shot_clock_est"].astype(float)
    endpoints["clock"] = endpoints["clock"].clip(0, endpoints["reset_len"])
    return endpoints


def build_shooting_outcomes(
    raw: pd.DataFrame,
    endpoints: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Attach actual FGA/FT points to their live shot-clock endpoint.

    Ordinary FT attempts map to the nearest preceding foul at the same game
    clock. And-one FTs map to the preceding made FGA. Administrative FTs are
    excluded because they do not represent a normal shot-clock decision.
    """
    work = raw.drop_duplicates().copy()
    work["game_id"] = (
        work["game_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)
    )
    work["clock_sec"] = normalize_clock_seconds(work)
    work["event_num"] = pd.to_numeric(work["event_num"], errors="coerce").fillna(0).astype(int)
    work["period"] = pd.to_numeric(work["period"], errors="coerce").fillna(0).astype(int)
    work["_input_order"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(
        ["game_id", "period", "clock_sec", "event_num", "_input_order"],
        ascending=[True, True, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    work["_event_name"] = _normalize_event_type(work["event_type"])
    work["_desc"] = _all_description(work)

    anchor_columns = [
        "game_id",
        "period",
        "event_num",
        "endpoint_type",
        "rule_reset_len",
        "reset_len",
        "shot_clock_est",
        "clock",
    ]
    anchors = endpoints[endpoints["endpoint_type"].isin(["fga", "ft_trip"])][
        anchor_columns
    ].copy()
    work = work.merge(
        anchors,
        on=["game_id", "period", "event_num"],
        how="left",
        validate="many_to_one",
    )
    row_id = work["game_id"] + "|" + work["period"].astype(str) + "|" + work["event_num"].astype(str)
    groupers = [work["game_id"], work["period"]]

    is_fga = work["_event_name"].isin(["MAKE", "MISS"])
    last_fga_id = row_id.where(is_fga).groupby(groupers, sort=False).ffill()
    last_fga_clock = work["clock_sec"].where(is_fga).groupby(groupers, sort=False).ffill()
    last_fga_make = (
        work["_event_name"]
        .eq("MAKE")
        .astype("boolean")
        .where(is_fga)
        .groupby(groupers, sort=False)
        .ffill()
        .fillna(False)
    )

    is_foul = work["_event_name"].eq("Foul")
    and_one_foul = is_foul & last_fga_make & last_fga_clock.eq(work["clock_sec"])
    foul_anchor = pd.Series(np.where(and_one_foul, last_fga_id, row_id), index=work.index)
    foul_map = pd.DataFrame(
        {
            "foul_id": row_id[is_foul].to_numpy(),
            "foul_clock": work.loc[is_foul, "clock_sec"].to_numpy(),
            "anchor_id": foul_anchor[is_foul].to_numpy(),
        }
    )
    last_foul_id = row_id.where(is_foul).groupby(groupers, sort=False).ffill()
    last_foul_clock = work["clock_sec"].where(is_foul).groupby(groupers, sort=False).ffill()

    is_ft = work["_event_name"].eq("FreeThrow")
    is_admin_ft = work["_desc"].str.contains(
        r"Technical|Flagrant|Clear Path|Def\.?\s*3|Defensive 3|Transition Take",
        case=False,
        regex=True,
        na=False,
    )
    ordinary_ft = is_ft & ~is_admin_ft
    ft = work[ordinary_ft].copy()
    ft["foul_id"] = last_foul_id[ordinary_ft].to_numpy()
    ft["foul_clock"] = last_foul_clock[ordinary_ft].to_numpy()
    ft = ft.merge(foul_map, on=["foul_id", "foul_clock"], how="left", validate="many_to_one")
    anchor_ids = set(row_id[work["endpoint_type"].notna()])
    ft["matched"] = ft["anchor_id"].isin(anchor_ids)
    ft["fta"] = 1
    ft["ft_points"] = (~ft["_desc"].str.contains(r"\bMISS\b", case=False, regex=True, na=False)).astype(int)
    ft["ft_shooter_id"] = pd.to_numeric(ft["player1_id"], errors="coerce")
    ft["ft_shooter_name"] = ft["player1_name"]
    conflicting_ft_shooters = (
        ft[ft["matched"]]
        .groupby("anchor_id", sort=False)["ft_shooter_id"]
        .nunique(dropna=True)
        .gt(1)
    )
    conflicting_ft_anchor_ids = set(
        conflicting_ft_shooters[conflicting_ft_shooters].index
    )
    ft_agg = (
        ft[ft["matched"]]
        .groupby("anchor_id", sort=False)
        .agg(
            fta=("fta", "sum"),
            ft_points=("ft_points", "sum"),
            ft_shooter_id=("ft_shooter_id", "first"),
            ft_shooter_name=("ft_shooter_name", "first"),
        )
        .reset_index()
    )
    ambiguous_ft_anchor = ft_agg["anchor_id"].isin(conflicting_ft_anchor_ids)
    ft_agg.loc[ambiguous_ft_anchor, ["ft_shooter_id", "ft_shooter_name"]] = np.nan

    outcomes = work[work["endpoint_type"].notna()].copy()
    outcomes["anchor_id"] = row_id[work["endpoint_type"].notna()].to_numpy()
    outcomes = outcomes.merge(ft_agg, on="anchor_id", how="left")
    outcomes[["fta", "ft_points"]] = outcomes[["fta", "ft_points"]].fillna(0)
    outcomes["fga"] = outcomes["endpoint_type"].eq("fga").astype(int)
    outcomes["fg_points"] = np.where(
        outcomes["_event_name"].eq("MAKE"),
        np.where(outcomes["_desc"].str.contains("3PT", case=False, na=False), 3, 2),
        0,
    )
    outcomes["points"] = outcomes["fg_points"] + outcomes["ft_points"]
    outcomes["tsa"] = outcomes["fga"] + 0.44 * outcomes["fta"]
    outcomes["shooter_id"] = pd.to_numeric(
        outcomes["player1_id"].where(outcomes["fga"].eq(1), outcomes["ft_shooter_id"]),
        errors="coerce",
    )
    outcomes["shooter_name"] = outcomes["player1_name"].where(
        outcomes["fga"].eq(1), outcomes["ft_shooter_name"]
    )

    audit = {
        "raw_ft_attempts": int(is_ft.sum()),
        "administrative_ft_attempts_excluded": int((is_ft & is_admin_ft).sum()),
        "ordinary_ft_attempts": int(ordinary_ft.sum()),
        "ordinary_ft_attempts_matched": int(ft["matched"].sum()),
        "ordinary_ft_attempts_unmatched": int((~ft["matched"]).sum()),
        "ambiguous_ft_shooter_anchors": int(len(conflicting_ft_anchor_ids)),
        "shooting_endpoints": int(len(outcomes)),
        "fga": int(outcomes["fga"].sum()),
        "matched_fta": int(outcomes["fta"].sum()),
        "points": int(outcomes["points"].sum()),
        "shooting_endpoints_with_shooter": int(outcomes["shooter_id"].notna().sum()),
    }
    return outcomes, audit


def clock_bin(clock: pd.Series) -> pd.Categorical:
    labels = ["0-4", "4-8", "8-12", "12-16", "16-20", "20-24"]
    return pd.cut(clock, bins=[-0.001, 4, 8, 12, 16, 20, 24.001], labels=labels, include_lowest=True)


def summarize_endpoints(endpoints: pd.DataFrame) -> pd.DataFrame:
    work = endpoints.copy()
    work["clock_bin"] = clock_bin(work["clock"])
    work["reset_family"] = np.where(work["rule_reset_len"].eq(14), "14-second reset", "24-second start")
    grouped = (
        work.groupby(["reset_family", "clock_bin"], observed=False)
        .agg(
            endpoints=("endpoint_type", "size"),
            turnovers=("is_turnover", "sum"),
            fga=("endpoint_type", lambda x: int(x.eq("fga").sum())),
            ft_trips=("endpoint_type", lambda x: int(x.eq("ft_trip").sum())),
            offensive_fouls_without_companion=(
                "endpoint_type",
                lambda x: int(x.eq("offensive_foul_no_tov").sum()),
            ),
        )
        .reset_index()
    )
    grouped["turnover_endpoint_rate"] = grouped["turnovers"] / grouped["endpoints"]
    grouped["share_of_turnovers"] = grouped["turnovers"] / grouped.groupby("reset_family")["turnovers"].transform("sum")
    return grouped


def summarize_discrete_hazard(endpoints: pd.DataFrame) -> pd.DataFrame:
    """Discrete clock-second hazard among observed decision-ending cycles."""
    rows: list[dict[str, object]] = []
    work = endpoints.copy()
    work["clock_second"] = np.floor(work["clock"] + 0.5).clip(0, 24).astype(int)
    work["reset_family"] = np.where(work["rule_reset_len"].eq(14), "14-second reset", "24-second start")
    for reset_family, family in work.groupby("reset_family", sort=False):
        max_second = 14 if reset_family == "14-second reset" else 24
        for second in range(max_second, -1, -1):
            at_risk = family[
                family["clock_second"].le(second) & family["reset_len"].ge(second)
            ]
            turnovers = family[family["clock_second"].eq(second) & family["is_turnover"]]
            all_endpoints = family[family["clock_second"].eq(second)]
            rows.append(
                {
                    "reset_family": reset_family,
                    "clock_second": second,
                    "cycles_at_risk": len(at_risk),
                    "turnovers": len(turnovers),
                    "all_endpoints": len(all_endpoints),
                    "turnover_hazard": len(turnovers) / len(at_risk) if len(at_risk) else np.nan,
                    "any_endpoint_hazard": len(all_endpoints) / len(at_risk) if len(at_risk) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_shot_value(outcomes: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for clock_basis, clock_column in [
        ("raw_rule", "shot_clock_est"),
        ("shot_calibrated", "clock"),
    ]:
        work = outcomes.copy()
        work["clock_bin"] = clock_bin(pd.to_numeric(work[clock_column], errors="coerce"))
        work["reset_family"] = np.where(
            work["rule_reset_len"].eq(14), "14-second reset", "24-second start"
        )
        summary = (
            work.groupby(["reset_family", "clock_bin"], observed=False)
            .agg(
                shooting_endpoints=("endpoint_type", "size"),
                points=("points", "sum"),
                fga=("fga", "sum"),
                fta=("fta", "sum"),
                fg_points=("fg_points", "sum"),
                ft_points=("ft_points", "sum"),
            )
            .reset_index()
        )
        summary["true_shooting_attempts"] = summary["fga"] + 0.44 * summary["fta"]
        summary["tsx2_points_per_shot"] = (
            summary["points"] / summary["true_shooting_attempts"].replace(0, np.nan)
        )
        summary["points_per_fga"] = summary["fg_points"] / summary["fga"].replace(0, np.nan)
        summary.insert(0, "clock_basis", clock_basis)
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True)


def build_remaining_possession_values(
    raw: pd.DataFrame,
    endpoints: pd.DataFrame,
    first_chance_path: Path,
    second_chance_path: Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Map an initial 24-second cycle to its full canonical possession value."""
    key = ["game_id", "period", "event_num"]
    work = raw.drop_duplicates().copy()
    work["game_id"] = (
        work["game_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)
    )
    work["period"] = pd.to_numeric(work["period"], errors="coerce").fillna(0).astype(int)
    work["event_num"] = pd.to_numeric(work["event_num"], errors="coerce").fillna(0).astype(int)
    work["clock_sec"] = normalize_clock_seconds(work)
    work["_input_order"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(
        ["game_id", "period", "clock_sec", "event_num", "_input_order"],
        ascending=[True, True, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    work["_event_order"] = work.groupby(["game_id", "period"], sort=False).cumcount()
    ambiguous_raw_key = work.duplicated(key, keep=False)
    order_map = work.loc[~ambiguous_raw_key, key + ["_event_order"]]

    initial = endpoints[endpoints["rule_reset_len"].eq(24)].merge(
        order_map, on=key, how="left", validate="one_to_one"
    )
    first = pd.read_parquet(first_chance_path, columns=key + ["Net_Diff"])
    second = pd.read_parquet(second_chance_path, columns=key + ["Net_Diff"])
    possessions = first.merge(
        second,
        on=key,
        validate="one_to_one",
        suffixes=("_first_chance", "_second_chance"),
    )
    possessions["game_id"] = (
        possessions["game_id"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(10)
    )
    possessions["period"] = pd.to_numeric(possessions["period"], errors="coerce").astype(int)
    possessions["event_num"] = pd.to_numeric(
        possessions["event_num"], errors="coerce"
    ).astype(int)
    possessions = possessions.merge(
        order_map, on=key, how="left", validate="one_to_one"
    ).dropna(subset=["_event_order"])
    possession_groups = {
        group_key: group.sort_values("_event_order")
        for group_key, group in possessions.groupby(["game_id", "period"], sort=False)
    }

    mapped_parts = []
    for group_key, endpoint_group in initial.dropna(subset=["_event_order"]).groupby(
        ["game_id", "period"], sort=False
    ):
        possession_group = possession_groups.get(group_key)
        if possession_group is None:
            continue
        endpoint_order = endpoint_group["_event_order"].to_numpy()
        terminal_order = possession_group["_event_order"].to_numpy()
        indices = np.searchsorted(terminal_order, endpoint_order, side="left")
        valid = indices < len(terminal_order)
        matched = endpoint_group.loc[valid].copy()
        terminals = possession_group.iloc[indices[valid]].reset_index(drop=True)
        matched["terminal_event_num"] = terminals["event_num"].to_numpy()
        matched["first_chance_value"] = terminals["Net_Diff_first_chance"].to_numpy()
        matched["second_chance_value"] = terminals["Net_Diff_second_chance"].to_numpy()
        matched["possession_value"] = (
            matched["first_chance_value"] + matched["second_chance_value"]
        )
        mapped_parts.append(matched)

    mapped = pd.concat(mapped_parts, ignore_index=True)
    terminal_key = ["game_id", "period", "terminal_event_num"]
    duplicate_terminal_rows = mapped.duplicated(terminal_key, keep=False)
    duplicate_terminal_count = int(duplicate_terminal_rows.sum())
    mapped = (
        mapped.sort_values(["game_id", "period", "terminal_event_num", "_event_order"])
        .drop_duplicates(terminal_key, keep="first")
        .reset_index(drop=True)
    )
    closure_gap = (
        mapped["first_chance_value"]
        + mapped["second_chance_value"]
        - mapped["possession_value"]
    ).abs().max()
    if float(closure_gap) > 1e-12:
        raise ValueError(f"Possession-value closure failed: {closure_gap}")
    audit = {
        "initial_24_second_endpoints": int(len(initial)),
        "initial_endpoints_with_raw_order": int(initial["_event_order"].notna().sum()),
        "forward_terminal_mappings": int(sum(len(part) for part in mapped_parts)),
        "duplicate_terminal_mapping_rows": duplicate_terminal_count,
        "unique_possession_mappings": int(len(mapped)),
    }
    return mapped, audit


def summarize_remaining_possession_value(
    mapped: pd.DataFrame, *, include_calibrated: bool = True
) -> pd.DataFrame:
    rows = []
    clock_views = [("raw_rule", "shot_clock_est")]
    if include_calibrated:
        clock_views.append(("shot_calibrated", "clock"))
    for clock_basis, clock_column in clock_views:
        for seconds_remaining in np.arange(24.0, -0.001, -0.25):
            at_risk = mapped[
                mapped["reset_len"].ge(seconds_remaining)
                & pd.to_numeric(mapped[clock_column], errors="coerce").le(seconds_remaining)
            ]
            standard_error = float(
                at_risk["possession_value"].std(ddof=1) / np.sqrt(len(at_risk))
            ) if len(at_risk) > 1 else np.nan
            rows.append(
                {
                    "clock_basis": clock_basis,
                    "seconds_remaining": float(seconds_remaining),
                    "possessions_alive": int(len(at_risk)),
                    "first_chance_value": float(at_risk["first_chance_value"].mean()),
                    "second_chance_value": float(at_risk["second_chance_value"].mean()),
                    "remaining_possession_value": float(at_risk["possession_value"].mean()),
                    "standard_error": standard_error,
                    "ci95_lower": float(at_risk["possession_value"].mean() - 1.96 * standard_error),
                    "ci95_upper": float(at_risk["possession_value"].mean() + 1.96 * standard_error),
                    "turnover_outcome_rate": float(at_risk["is_turnover"].mean()),
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="+", type=int, default=[24, 25, 26])
    parser.add_argument("--season-types", nargs="+", choices=["RS", "PS"], default=["RS", "PS"])
    parser.add_argument("--pipeline-root", type=Path, default=PIPELINE_ROOT)
    parser.add_argument("--calibration-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    calibrator = None if args.no_calibration else joblib.load(args.calibration_model)
    all_endpoints = []
    all_shooting_outcomes = []
    ft_join_audits = []
    all_possession_values = []
    possession_value_audits = []
    for season in args.seasons:
        for season_type in args.season_types:
            suffix = "_PS" if season_type == "PS" else ""
            path = args.pipeline_root / "raw_data" / f"NBA{season:02d}{suffix}.parquet"
            if not path.exists():
                continue
            print(f"Reading {path}")
            raw = pd.read_parquet(path)
            endpoints = build_endpoint_estimates(
                raw,
                season_end_year=season,
                calibrator=calibrator,
            )
            shooting_outcomes, ft_join_audit = build_shooting_outcomes(raw, endpoints)
            suffix = "_PS" if season_type == "PS" else ""
            possession_values, possession_value_audit = build_remaining_possession_values(
                raw,
                endpoints,
                args.pipeline_root / "processed" / f"FIRST_CHANCE{season:02d}{suffix}.parquet",
                args.pipeline_root
                / "processed"
                / f"SECOND_CHANCE_CLEAN{season:02d}{suffix}.parquet",
            )
            endpoints["season"] = 2000 + season
            endpoints["season_type"] = season_type
            shooting_outcomes["season"] = 2000 + season
            shooting_outcomes["season_type"] = season_type
            possession_values["season"] = 2000 + season
            possession_values["season_type"] = season_type
            ft_join_audit.update({"season": 2000 + season, "season_type": season_type})
            possession_value_audit.update(
                {"season": 2000 + season, "season_type": season_type}
            )
            all_endpoints.append(endpoints)
            all_shooting_outcomes.append(shooting_outcomes)
            ft_join_audits.append(ft_join_audit)
            all_possession_values.append(possession_values)
            possession_value_audits.append(possession_value_audit)
            print(
                f"  endpoints={len(endpoints):,}, turnovers={int(endpoints['is_turnover'].sum()):,}, "
                f"high-confidence={endpoints['confidence'].eq('high').mean():.2%}"
            )
    if not all_endpoints:
        raise FileNotFoundError("No requested raw season files were found")

    endpoints = pd.concat(all_endpoints, ignore_index=True)
    shooting_outcomes = pd.concat(all_shooting_outcomes, ignore_index=True)
    possession_values = pd.concat(all_possession_values, ignore_index=True)
    clock_views = [("raw_rule", "shot_clock_est")]
    if calibrator is not None:
        clock_views.append(("shot_calibrated", "clock"))
    summaries = []
    hazards = []
    for clock_basis, clock_column in clock_views:
        view = endpoints.copy()
        view["clock"] = pd.to_numeric(view[clock_column], errors="coerce")
        summary_view = summarize_endpoints(view)
        summary_view.insert(0, "clock_basis", clock_basis)
        summaries.append(summary_view)
        hazard_view = summarize_discrete_hazard(view)
        hazard_view.insert(0, "clock_basis", clock_basis)
        hazards.append(hazard_view)
    summary = pd.concat(summaries, ignore_index=True)
    hazard = pd.concat(hazards, ignore_index=True)
    shot_value = summarize_shot_value(shooting_outcomes)
    remaining_value = summarize_remaining_possession_value(
        possession_values, include_calibrated=calibrator is not None
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    endpoints.to_parquet(args.output_dir / "turnover_shot_clock_endpoints.parquet", index=False)
    shooting_outcomes.to_parquet(
        args.output_dir / "turnover_shot_clock_shooting_outcomes.parquet", index=False
    )
    possession_values.to_parquet(
        args.output_dir / "turnover_shot_clock_possession_values.parquet", index=False
    )
    summary.to_csv(args.output_dir / "turnover_shot_clock_summary.csv", index=False)
    hazard.to_csv(args.output_dir / "turnover_shot_clock_hazard.csv", index=False)
    shot_value.to_csv(args.output_dir / "turnover_shot_clock_shot_value.csv", index=False)
    remaining_value.to_csv(
        args.output_dir / "turnover_shot_clock_remaining_possession_value.csv", index=False
    )
    pd.DataFrame(ft_join_audits).to_csv(
        args.output_dir / "turnover_shot_clock_ft_join_audit.csv", index=False
    )
    pd.DataFrame(possession_value_audits).to_csv(
        args.output_dir / "turnover_shot_clock_possession_value_audit.csv", index=False
    )
    print("\n" + summary.to_string(index=False))
    print("\n" + shot_value.to_string(index=False))
    print("\n" + remaining_value.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
