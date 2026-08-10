#!/usr/bin/env python3
"""Build isolated FIRST_CHANCE parquets for the EV-target VPM research lane.

The active public DECOMP target remains actual points and is never modified by
this script.  VPM instead copies FIRST_CHANCE into a separate output directory,
replaces the first-chance midrange and three-point make targets with per-shot
expected value, and carries ``actual - EV`` in two residual solver aliases:

* ``DECOMP_EFG`` / ``FC_DECOMP_EFG_Diff``: three-point residual
* ``DECOMP_MID_VALUE`` / ``FC_DECOMP_MID_VALUE_Diff``: midrange residual

For an eligible shot with original make target ``D``:

    residual = actual_points - expected_points
    EV_target = D - residual

Consequently, the five EV-adjusted public shot children plus the two residuals
still equal the original five-child actual-points shot tree on every row.

This is target construction only.  It does not train an EV model, tune ridge
alphas, apply priors, or publish a VPM table.  The supplied EV parquet must be
walk-forward/leakage-safe if the output will be used for validation.

``--net-lane`` additionally builds the net-continuation research lane in an
isolated output directory.  The source FIRST_CHANCE parquets must carry the
``FC_DECOMP_NET_*`` family (net-priced shot tree, ``net = actual + kappa(zone)``
on first-chance misses).  The net swap mirrors the gross swap exactly:

    net_EV      = ev_pts + kappa(zone) * (1 - p_make)
    net_residual = net_actual - net_EV
    net_target   = D_net - net_residual

with ``FC_DECOMP_NET_EFG_Diff`` / ``FC_DECOMP_NET_MID_VALUE_Diff`` overwritten
as the net residual aliases.  Per-zone kappa is recovered exactly from the
file's own ``FC_CONT_KAPPA_Diff`` column, so the swap composes with the same
constants the processed builder used.  SECOND_CHANCE_CLEAN is rewritten (not
symlinked) with ``SC_NET_CONT_Diff`` as its Net/Off/Def target so the lane's
component sum still closes to total actual points.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from .process_rapm_blocks.common import (
        RIM_ACTION_TYPES,
        normalize_game_id_series,
        sort_raw_pbp_chronologically,
    )
except ImportError:
    from process_rapm_blocks.common import (
        RIM_ACTION_TYPES,
        normalize_game_id_series,
        sort_raw_pbp_chronologically,
    )


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PIPELINE_ROOT / "raw_data"
DEFAULT_PROCESSED_DIR = PIPELINE_ROOT / "processed"
DEFAULT_OUTPUT_DIR = PIPELINE_ROOT / "results" / "vpm_target_swap" / "processed"
DEFAULT_REPORT = PIPELINE_ROOT / "results" / "vpm_target_swap" / "attach_report.csv"

PUBLIC_SHOT_COLUMNS = [
    "FC_DECOMP_RIM_FREQ_Diff",
    "FC_DECOMP_RIM_FG_Diff",
    "FC_DECOMP_MID_FG_Diff",
    "FC_DECOMP_THREE_FREQ_Diff",
    "FC_DECOMP_THREE_FG_Diff",
]
THREE_TARGET_COLUMN = "FC_DECOMP_THREE_FG_Diff"
THREE_RESIDUAL_COLUMN = "FC_DECOMP_EFG_Diff"
MID_TARGET_COLUMN = "FC_DECOMP_MID_FG_Diff"
MID_RESIDUAL_COLUMN = "FC_DECOMP_MID_VALUE_Diff"

NET_SHOT_COLUMNS = [
    "FC_DECOMP_NET_RIM_FREQ_Diff",
    "FC_DECOMP_NET_RIM_FG_Diff",
    "FC_DECOMP_NET_MID_FG_Diff",
    "FC_DECOMP_NET_THREE_FREQ_Diff",
    "FC_DECOMP_NET_THREE_FG_Diff",
]
NET_THREE_TARGET_COLUMN = "FC_DECOMP_NET_THREE_FG_Diff"
NET_THREE_RESIDUAL_COLUMN = "FC_DECOMP_NET_EFG_Diff"
NET_MID_TARGET_COLUMN = "FC_DECOMP_NET_MID_FG_Diff"
NET_MID_RESIDUAL_COLUMN = "FC_DECOMP_NET_MID_VALUE_Diff"
KAPPA_COLUMN = "FC_CONT_KAPPA_Diff"
SC_NET_CONT_COLUMN = "SC_NET_CONT_Diff"
REQUIRED_NET_FIRST_CHANCE_COLUMNS = {
    *NET_SHOT_COLUMNS,
    NET_THREE_RESIDUAL_COLUMN,
    NET_MID_RESIDUAL_COLUMN,
    KAPPA_COLUMN,
    "FC_TS_NET_Diff",
}
NET_DEFAULT_OUTPUT_DIR = (
    PIPELINE_ROOT / "results" / "vpm_target_swap_netcont" / "processed"
)
NET_DEFAULT_REPORT = (
    PIPELINE_ROOT / "results" / "vpm_target_swap_netcont" / "attach_report.csv"
)

REQUIRED_FIRST_CHANCE_COLUMNS = {
    "game_id",
    "event_num",
    *PUBLIC_SHOT_COLUMNS,
    THREE_RESIDUAL_COLUMN,
    MID_RESIDUAL_COLUMN,
}
REQUIRED_RAW_COLUMNS = {
    "game_id",
    "event_num",
    "event_type",
    "home_description",
    "visitor_description",
}
REQUIRED_EV_COLUMNS = {"game_id", "event_num", "zone", "ev_pts"}


def parse_years(value: str) -> list[int]:
    """Parse ``2021-2026`` or a comma-separated year list."""
    text = value.strip()
    if "-" in text and "," not in text:
        start, end = (int(part) for part in text.split("-", maxsplit=1))
        if end < start:
            raise argparse.ArgumentTypeError("year range end must be >= start")
        return list(range(start, end + 1))
    try:
        years = [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid year list: {value}") from exc
    if not years:
        raise argparse.ArgumentTypeError("at least one year is required")
    return years


def _require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def prepare_ev_table(ev: pd.DataFrame, *, net_lane: bool = False) -> pd.DataFrame:
    """Validate and normalize a per-shot EV table keyed by game/event."""
    _require_columns(ev, REQUIRED_EV_COLUMNS, "EV parquet")
    keep = sorted(REQUIRED_EV_COLUMNS)
    keep += [c for c in ("p_make", "p_v2", "shot_value") if c in ev.columns]
    out = ev.loc[:, keep].copy()
    out["game_id"] = normalize_game_id_series(out["game_id"])
    out["event_num"] = pd.to_numeric(out["event_num"], errors="coerce")
    out["ev_pts"] = pd.to_numeric(out["ev_pts"], errors="coerce")
    out["zone"] = out["zone"].astype(str).str.strip().str.lower()
    out = out.dropna(subset=["event_num", "ev_pts"])
    out["event_num"] = out["event_num"].astype(np.int64)

    if net_lane:
        if "p_make" in out.columns:
            p_make = pd.to_numeric(out["p_make"], errors="coerce")
        elif "p_v2" in out.columns:
            p_make = pd.to_numeric(out["p_v2"], errors="coerce")
        elif "shot_value" in out.columns:
            shot_value = pd.to_numeric(out["shot_value"], errors="coerce")
            p_make = out["ev_pts"] / shot_value
        else:
            raise ValueError(
                "net lane requires a make probability: EV parquet must carry "
                "p_make, p_v2, or shot_value"
            )
        if p_make.isna().any():
            raise ValueError("net lane: EV make probability has missing values")
        if (p_make < -1e-9).any() or (p_make > 1 + 1e-9).any():
            raise ValueError("net lane: EV make probability must lie in [0, 1]")
        out["p_make"] = p_make.clip(0.0, 1.0)
    out = out.drop(columns=[c for c in ("p_v2", "shot_value") if c in out.columns])

    invalid_zones = sorted(set(out["zone"]) - {"mid", "three"})
    if invalid_zones:
        raise ValueError(
            "EV parquet zone must contain only 'mid'/'three'; "
            f"found {invalid_zones}"
        )

    duplicate_mask = out.duplicated(["game_id", "event_num"], keep=False)
    if duplicate_mask.any():
        duplicate_groups = out.loc[duplicate_mask].groupby(
            ["game_id", "event_num"], sort=False
        )
        conflicts = duplicate_groups.agg(
            zones=("zone", "nunique"),
            ev_values=("ev_pts", "nunique"),
        )
        conflicts = conflicts[(conflicts["zones"] > 1) | (conflicts["ev_values"] > 1)]
        if not conflicts.empty:
            examples = [f"{gid}/{event}" for gid, event in conflicts.index[:5]]
            raise ValueError(
                "EV parquet has conflicting duplicate game/event keys: "
                + ", ".join(examples)
            )
        out = out.drop_duplicates(["game_id", "event_num"], keep="first")

    return out.reset_index(drop=True)


def _normalize_raw_event_types(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    event_num = pd.to_numeric(out["event_type"], errors="coerce")
    event_text = out["event_type"].astype(str).str.strip().str.upper()
    out["_is_make"] = event_num.eq(1) | event_text.eq("MAKE")
    out["_is_miss"] = event_num.eq(2) | event_text.eq("MISS")
    out["_is_fga"] = out["_is_make"] | out["_is_miss"]
    return out


def _classify_fgas(raw: pd.DataFrame) -> pd.DataFrame:
    """Return chronologically ordered FGA keys, zones, and actual points."""
    ordered = sort_raw_pbp_chronologically(raw).reset_index(drop=True)
    ordered["_raw_position"] = np.arange(len(ordered), dtype=np.int64)
    ordered = _normalize_raw_event_types(ordered)

    shot_text = (
        ordered["home_description"].fillna("").astype(str)
        + " "
        + ordered["visitor_description"].fillna("").astype(str)
    )
    is_three = shot_text.str.contains("3PT", case=False, regex=False, na=False)
    if "event_action_type" in ordered.columns:
        action_type = pd.to_numeric(ordered["event_action_type"], errors="coerce")
        is_rim = action_type.isin(RIM_ACTION_TYPES)
    else:
        is_rim = pd.Series(False, index=ordered.index)
    is_rim = is_rim | shot_text.str.contains(
        r"\blayup\b|\bdunk\b|\btip\b", case=False, regex=True, na=False
    )

    fga = ordered.loc[ordered["_is_fga"]].copy()
    fga["_zone"] = np.where(
        is_three.loc[fga.index],
        "three",
        np.where(is_rim.loc[fga.index], "rim", "mid"),
    )
    fga["_actual_pts"] = np.where(
        fga["_is_make"],
        np.where(fga["_zone"].eq("three"), 3.0, 2.0),
        0.0,
    )
    return fga[
        ["game_id", "event_num", "_raw_position", "_zone", "_actual_pts"]
    ].reset_index(drop=True), ordered


def assign_first_fga_to_possessions(
    first_chance: pd.DataFrame,
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Map the chronological first FGA to each FIRST_CHANCE terminal row.

    Terminal membership is determined in clock-first raw order.  This avoids
    the event-number-first failure mode caused by corrected actions appended
    with a late event number at an earlier clock.
    """
    _require_columns(first_chance, REQUIRED_FIRST_CHANCE_COLUMNS, "FIRST_CHANCE")
    _require_columns(raw, REQUIRED_RAW_COLUMNS, "raw PBP")

    fc = first_chance[["game_id", "event_num"]].copy()
    fc["game_id"] = normalize_game_id_series(fc["game_id"])
    fc["event_num"] = pd.to_numeric(fc["event_num"], errors="coerce")
    if fc["event_num"].isna().any():
        raise ValueError("FIRST_CHANCE contains null/non-numeric event_num values")
    fc["event_num"] = fc["event_num"].astype(np.int64)
    fc["_row"] = np.arange(len(fc), dtype=np.int64)
    if fc.duplicated(["game_id", "event_num"]).any():
        raise ValueError("FIRST_CHANCE game_id/event_num keys must be unique")

    raw_work = raw.copy()
    raw_work["game_id"] = normalize_game_id_series(raw_work["game_id"])
    raw_work["event_num"] = pd.to_numeric(raw_work["event_num"], errors="coerce")
    raw_work = raw_work.dropna(subset=["event_num"])
    raw_work["event_num"] = raw_work["event_num"].astype(np.int64)
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

    columns = [
        "_row",
        "game_id",
        "event_num",
        "_raw_position",
        "_zone",
        "_actual_pts",
    ]
    if assignments:
        attached = pd.concat(assignments, ignore_index=True)
        attached = attached.sort_values("_raw_position", kind="stable")
        attached = attached.drop_duplicates("_row", keep="first")
        attached = attached[columns].reset_index(drop=True)
    else:
        attached = pd.DataFrame(columns=columns)

    stats = {
        "terminal_rows": int(len(fc)),
        "terminal_keys_found_in_raw": int(terminals["_raw_position"].notna().sum()),
        "first_fgas_assigned": int(len(attached)),
    }
    return attached, stats


def attach_ev_targets(
    first_chance: pd.DataFrame,
    raw: pd.DataFrame,
    ev: pd.DataFrame,
    *,
    identity_tolerance: float = 0.02,
    net_lane: bool = False,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Return one isolated EV-swapped FIRST_CHANCE frame and QA statistics."""
    prepared_ev = prepare_ev_table(ev, net_lane=net_lane)
    assignments, stats = assign_first_fga_to_possessions(first_chance, raw)

    out = first_chance.reset_index(drop=True).copy()
    out["game_id"] = normalize_game_id_series(out["game_id"])
    out["VPM_ORIGINAL_DECOMP_EFG_Diff"] = pd.to_numeric(
        out[THREE_RESIDUAL_COLUMN], errors="coerce"
    )
    out["VPM_ORIGINAL_DECOMP_MID_VALUE_Diff"] = pd.to_numeric(
        out[MID_RESIDUAL_COLUMN], errors="coerce"
    )
    out["VPM_THREE_RESID_Diff"] = 0.0
    out["VPM_MID_RESID_Diff"] = 0.0
    out["VPM_EV_ATTACHED"] = False

    original_children = out[PUBLIC_SHOT_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    if original_children.isna().any().any():
        raise ValueError("FIRST_CHANCE public shot children contain missing values")
    original_shot_sum = original_children.sum(axis=1).to_numpy(dtype=np.float64)

    kappa_all: np.ndarray | None = None
    original_net_shot_sum: np.ndarray | None = None
    if net_lane:
        _require_columns(
            out, REQUIRED_NET_FIRST_CHANCE_COLUMNS, "FIRST_CHANCE (net lane)"
        )
        out["VPM_ORIGINAL_DECOMP_NET_EFG_Diff"] = pd.to_numeric(
            out[NET_THREE_RESIDUAL_COLUMN], errors="coerce"
        )
        out["VPM_ORIGINAL_DECOMP_NET_MID_VALUE_Diff"] = pd.to_numeric(
            out[NET_MID_RESIDUAL_COLUMN], errors="coerce"
        )
        out["VPM_THREE_RESID_NET_Diff"] = 0.0
        out["VPM_MID_RESID_NET_Diff"] = 0.0

        original_net_children = out[NET_SHOT_COLUMNS].apply(
            pd.to_numeric, errors="coerce"
        )
        if original_net_children.isna().any().any():
            raise ValueError("FIRST_CHANCE net shot children contain missing values")
        original_net_shot_sum = original_net_children.sum(axis=1).to_numpy(
            dtype=np.float64
        )
        kappa_all = pd.to_numeric(out[KAPPA_COLUMN], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if np.isnan(kappa_all).any():
            raise ValueError(f"{KAPPA_COLUMN} contains missing values")
        # The net tree must exceed the gross tree by exactly the kappa column
        # (net = actual + kappa on first-chance misses) on every row.
        net_tree_kappa_gap = (original_net_shot_sum - original_shot_sum) - kappa_all
        stats["max_abs_net_tree_kappa_gap"] = float(np.max(np.abs(net_tree_kappa_gap)))
        if stats["max_abs_net_tree_kappa_gap"] > 1e-9:
            raise ValueError(
                "net lane: net shot tree minus gross shot tree does not equal "
                f"{KAPPA_COLUMN}: max gap="
                f"{stats['max_abs_net_tree_kappa_gap']:.3g}"
            )

    keyed_ev = prepared_ev.rename(
        columns={"zone": "_ev_zone", "ev_pts": "_ev_pts", "p_make": "_ev_p"}
    )
    assignments = assignments.merge(
        keyed_ev,
        on=["game_id", "event_num"],
        how="left",
        validate="one_to_one",
    )

    zone_kappas: dict[str, float] = {}
    if net_lane:
        assert kappa_all is not None
        zone_kappas, kappa_agreement = _recover_zone_kappas(assignments, kappa_all)
        for zone_name, value in zone_kappas.items():
            stats[f"kappa_{zone_name}"] = value
        stats["kappa_agreement_min"] = min(kappa_agreement.values())

    for (
        zone,
        target_column,
        residual_column,
        audit_column,
        net_target_column,
        net_residual_column,
        net_audit_column,
    ) in [
        (
            "three",
            THREE_TARGET_COLUMN,
            THREE_RESIDUAL_COLUMN,
            "VPM_THREE_RESID_Diff",
            NET_THREE_TARGET_COLUMN,
            NET_THREE_RESIDUAL_COLUMN,
            "VPM_THREE_RESID_NET_Diff",
        ),
        (
            "mid",
            MID_TARGET_COLUMN,
            MID_RESIDUAL_COLUMN,
            "VPM_MID_RESID_Diff",
            NET_MID_TARGET_COLUMN,
            NET_MID_RESIDUAL_COLUMN,
            "VPM_MID_RESID_NET_Diff",
        ),
    ]:
        zone_rows = assignments[assignments["_zone"].eq(zone)].set_index("_row")
        aligned = zone_rows.reindex(out.index)
        actual_points = pd.to_numeric(
            aligned["_actual_pts"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        expected_points = pd.to_numeric(
            aligned["_ev_pts"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        ev_zone = aligned["_ev_zone"].astype("string").to_numpy()
        target = pd.to_numeric(out[target_column], errors="coerce").to_numpy(
            dtype=np.float64
        )

        candidate = ~np.isnan(actual_points)
        if candidate.any():
            replacement_constant = float(
                np.nanmedian(actual_points[candidate] - target[candidate])
            )
            identity_ok = candidate & (
                np.abs((actual_points - target) - replacement_constant)
                < identity_tolerance
            )
        else:
            replacement_constant = float("nan")
            identity_ok = np.zeros(len(out), dtype=bool)

        has_ev = ~np.isnan(expected_points)
        zone_match = np.asarray(
            pd.Series(ev_zone, dtype="string").eq(zone).fillna(False),
            dtype=bool,
        )
        swap = identity_ok & has_ev & zone_match

        if net_lane:
            # Gate the net tree with its own identity check, then swap both
            # trees on the common row set so the two lanes stay row-comparable.
            kappa_z = zone_kappas[zone]
            net_actual = np.where(actual_points == 0.0, kappa_z, actual_points)
            net_target = pd.to_numeric(
                out[net_target_column], errors="coerce"
            ).to_numpy(dtype=np.float64)
            if candidate.any():
                net_replacement_constant = float(
                    np.nanmedian(net_actual[candidate] - net_target[candidate])
                )
                net_identity_ok = candidate & (
                    np.abs((net_actual - net_target) - net_replacement_constant)
                    < identity_tolerance
                )
            else:
                net_replacement_constant = float("nan")
                net_identity_ok = np.zeros(len(out), dtype=bool)
            gate_mismatch = int((identity_ok != net_identity_ok).sum())
            swap = swap & net_identity_ok

            expected_p = pd.to_numeric(aligned["_ev_p"], errors="coerce").to_numpy(
                dtype=np.float64
            )
            net_expected = expected_points + kappa_z * (1.0 - expected_p)
            net_residual = np.zeros(len(out), dtype=np.float64)
            net_residual[swap] = net_actual[swap] - net_expected[swap]

        residual = np.zeros(len(out), dtype=np.float64)
        residual[swap] = actual_points[swap] - expected_points[swap]

        out[target_column] = target - residual
        out[residual_column] = residual
        out[audit_column] = residual
        out.loc[swap, "VPM_EV_ATTACHED"] = True

        if net_lane:
            out[net_target_column] = net_target - net_residual
            out[net_residual_column] = net_residual
            out[net_audit_column] = net_residual
            stats[f"{zone}_net_identity_ok"] = int(net_identity_ok.sum())
            stats[f"{zone}_net_gate_mismatch"] = gate_mismatch
            stats[f"{zone}_net_replacement_constant"] = net_replacement_constant
            stats[f"{zone}_net_residual_sum"] = float(net_residual.sum())

        stats[f"{zone}_first_chance_fgas"] = int(candidate.sum())
        stats[f"{zone}_identity_ok"] = int(identity_ok.sum())
        stats[f"{zone}_ev_key_matches"] = int((identity_ok & has_ev).sum())
        stats[f"{zone}_zone_matches"] = int(
            (identity_ok & has_ev & zone_match).sum()
        )
        stats[f"{zone}_swapped"] = int(swap.sum())
        stats[f"{zone}_replacement_constant"] = replacement_constant
        stats[f"{zone}_residual_sum"] = float(residual.sum())

    swapped_shot_sum = (
        out[PUBLIC_SHOT_COLUMNS].sum(axis=1)
        + out[THREE_RESIDUAL_COLUMN]
        + out[MID_RESIDUAL_COLUMN]
    ).to_numpy(dtype=np.float64)
    closure_gap = swapped_shot_sum - original_shot_sum
    stats["rows"] = int(len(out))
    stats["ev_attached_rows"] = int(out["VPM_EV_ATTACHED"].sum())
    stats["max_abs_vpm_shot_closure_gap"] = float(np.max(np.abs(closure_gap)))
    stats["sum_vpm_shot_closure_gap"] = float(closure_gap.sum())

    if stats["max_abs_vpm_shot_closure_gap"] > 1e-9:
        raise ValueError(
            "VPM target swap broke shot-tree closure: "
            f"max gap={stats['max_abs_vpm_shot_closure_gap']:.3g}"
        )

    if net_lane:
        assert original_net_shot_sum is not None
        swapped_net_shot_sum = (
            out[NET_SHOT_COLUMNS].sum(axis=1)
            + out[NET_THREE_RESIDUAL_COLUMN]
            + out[NET_MID_RESIDUAL_COLUMN]
        ).to_numpy(dtype=np.float64)
        net_closure_gap = swapped_net_shot_sum - original_net_shot_sum
        stats["max_abs_vpm_net_shot_closure_gap"] = float(
            np.max(np.abs(net_closure_gap))
        )
        stats["sum_vpm_net_shot_closure_gap"] = float(net_closure_gap.sum())
        if stats["max_abs_vpm_net_shot_closure_gap"] > 1e-9:
            raise ValueError(
                "VPM net target swap broke net shot-tree closure: "
                f"max gap={stats['max_abs_vpm_net_shot_closure_gap']:.3g}"
            )
    return out, stats


def _recover_zone_kappas(
    assignments: pd.DataFrame,
    kappa_values: np.ndarray,
) -> tuple[dict[str, float], dict[str, float]]:
    """Recover the per-zone continuation constants from ``FC_CONT_KAPPA_Diff``.

    The processed builder prices every first-chance miss with one constant per
    shot zone, so each file carries exactly three distinct positive kappas.
    Assigned first-FGA misses are grouped by raw-classified zone and the modal
    positive kappa is taken per zone; the agreement fraction guards against a
    zone-classifier mismatch between this script and the processed pipeline.
    """
    kappas: dict[str, float] = {}
    agreement: dict[str, float] = {}
    for zone in ("rim", "mid", "three"):
        rows = assignments.loc[
            assignments["_zone"].eq(zone) & assignments["_actual_pts"].eq(0.0),
            "_row",
        ].to_numpy(dtype=np.int64)
        values = kappa_values[rows]
        values = values[values > 0]
        if values.size == 0:
            raise ValueError(
                f"net lane: no positive {KAPPA_COLUMN} observed for zone {zone}"
            )
        rounded = np.round(values, 9)
        uniques, counts = np.unique(rounded, return_counts=True)
        kappas[zone] = float(uniques[np.argmax(counts)])
        agreement[zone] = float(counts.max() / counts.sum())
        if agreement[zone] < 0.98:
            raise ValueError(
                f"net lane: kappa recovery for zone {zone} is ambiguous "
                f"(agreement {agreement[zone]:.4f})"
            )
    if len({round(v, 9) for v in kappas.values()}) != 3:
        raise ValueError(f"net lane: recovered zone kappas are not distinct: {kappas}")
    return kappas, agreement


def rewrite_net_second_chance(
    source_path: Path,
    output_path: Path,
    fc_kappa: np.ndarray | None = None,
) -> dict[str, float | int | str]:
    """Write a SECOND_CHANCE_CLEAN parquet with ``SC_NET_CONT`` as the target.

    Verifies the source identities ``Off == Net``, ``Def == -Net``, and
    ``SC_NET_CONT == Net - kappa`` before overwriting the three solver target
    columns; the original net target is preserved in an audit column.  When
    ``fc_kappa`` is supplied (the matching FIRST_CHANCE kappa column), row
    order alignment between the two families is asserted as well.
    """
    sc = pd.read_parquet(source_path)
    _require_columns(
        sc,
        {"Off_Diff", "Def_Diff", "Net_Diff", SC_NET_CONT_COLUMN, KAPPA_COLUMN},
        "SECOND_CHANCE_CLEAN (net lane)",
    )
    net = pd.to_numeric(sc["Net_Diff"], errors="coerce").to_numpy(dtype=np.float64)
    off = pd.to_numeric(sc["Off_Diff"], errors="coerce").to_numpy(dtype=np.float64)
    deff = pd.to_numeric(sc["Def_Diff"], errors="coerce").to_numpy(dtype=np.float64)
    sc_net = pd.to_numeric(sc[SC_NET_CONT_COLUMN], errors="coerce").to_numpy(
        dtype=np.float64
    )
    kappa = pd.to_numeric(sc[KAPPA_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
    checks = {
        "sc_max_abs_off_net_gap": float(np.max(np.abs(off - net))),
        "sc_max_abs_def_net_gap": float(np.max(np.abs(deff + net))),
        "sc_max_abs_netcont_identity_gap": float(
            np.max(np.abs(sc_net - (net - kappa)))
        ),
    }
    if fc_kappa is not None:
        if len(fc_kappa) != len(kappa):
            raise ValueError(
                "net lane: FIRST_CHANCE and SECOND_CHANCE_CLEAN row counts differ "
                f"({len(fc_kappa)} vs {len(kappa)}) for {source_path.name}"
            )
        checks["sc_max_abs_fc_kappa_gap"] = float(np.max(np.abs(fc_kappa - kappa)))
    if max(checks.values()) > 1e-9:
        raise ValueError(
            f"net lane SECOND_CHANCE_CLEAN identity check failed for "
            f"{source_path.name}: {checks}"
        )
    out = sc.copy()
    out["VPM_ORIGINAL_SC_Net_Diff"] = sc["Net_Diff"]
    out["Net_Diff"] = sc_net
    out["Off_Diff"] = sc_net
    out["Def_Diff"] = -sc_net
    out.to_parquet(output_path, index=False)
    return {
        "file": source_path.name,
        "sc_rows": int(len(sc)),
        "sc_net_cont_sum": float(sc_net.sum()),
        **checks,
    }


def _season_suffixes(season_types: str) -> list[str]:
    if season_types == "rs":
        return [""]
    if season_types == "ps":
        return ["_PS"]
    return ["", "_PS"]


def _link_unchanged_families(
    source_dir: Path,
    output_dir: Path,
    years: Iterable[int],
    season_types: str,
    overwrite: bool,
    families: Iterable[str] = ("SECOND_CHANCE_CLEAN", "RAPM"),
) -> None:
    for year in years:
        yy = year % 100
        for suffix in _season_suffixes(season_types):
            for family in families:
                source = source_dir / f"{family}{yy:02d}{suffix}.parquet"
                destination = output_dir / source.name
                if not source.exists():
                    continue
                if destination.exists() or destination.is_symlink():
                    if not overwrite:
                        continue
                    destination.unlink()
                relative_source = os.path.relpath(source, start=output_dir)
                destination.symlink_to(relative_source)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ev-parquet",
        type=Path,
        required=True,
        help="Per-shot walk-forward EV parquet with game_id/event_num/zone/ev_pts.",
    )
    parser.add_argument("--years", type=parse_years, required=True)
    parser.add_argument(
        "--season-types",
        choices=["rs", "ps", "all"],
        default="all",
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--identity-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--link-unchanged-families",
        action="store_true",
        help="Add relative symlinks for RAPM and SECOND_CHANCE_CLEAN inputs. "
        "In net-lane mode SECOND_CHANCE_CLEAN is rewritten instead, so only "
        "RAPM is linked (from --rapm-source-dir).",
    )
    parser.add_argument(
        "--net-lane",
        action="store_true",
        help="Also swap the FC_DECOMP_NET_* tree with kappa-adjusted EV and "
        "rewrite SECOND_CHANCE_CLEAN to SC_NET_CONT targets (isolated "
        "net-continuation research lane).",
    )
    parser.add_argument(
        "--rapm-source-dir",
        type=Path,
        default=None,
        help="Directory holding RAPM parquets to symlink in net-lane mode "
        "(default: the canonical processed directory).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing files in the isolated output directory.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.net_lane:
        # Never let the net lane clobber the gross-lane default directory.
        if args.output_dir == DEFAULT_OUTPUT_DIR:
            args.output_dir = NET_DEFAULT_OUTPUT_DIR
        if args.report == DEFAULT_REPORT:
            args.report = NET_DEFAULT_REPORT
    ev = pd.read_parquet(args.ev_parquet)
    ev = prepare_ev_table(ev, net_lane=args.net_lane)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, float | int | str]] = []

    for year in args.years:
        yy = year % 100
        for suffix in _season_suffixes(args.season_types):
            fc_path = args.processed_dir / f"FIRST_CHANCE{yy:02d}{suffix}.parquet"
            raw_path = args.raw_dir / f"NBA{yy:02d}{suffix}.parquet"
            output_path = args.output_dir / fc_path.name
            if not fc_path.exists() or not raw_path.exists():
                missing = [str(path) for path in [fc_path, raw_path] if not path.exists()]
                raise FileNotFoundError("Missing VPM input(s): " + ", ".join(missing))
            if output_path.exists() and not args.overwrite:
                raise FileExistsError(
                    f"{output_path} already exists; pass --overwrite to replace it"
                )

            first_chance = pd.read_parquet(fc_path)
            raw_columns = sorted(
                REQUIRED_RAW_COLUMNS
                | {
                    "event_action_type",
                    "period",
                    "minute_remaining_quarter",
                    "seconds_remaining_quarter",
                }
            )
            raw = pd.read_parquet(raw_path, columns=raw_columns)
            swapped, stats = attach_ev_targets(
                first_chance,
                raw,
                ev,
                identity_tolerance=args.identity_tolerance,
                net_lane=args.net_lane,
            )
            swapped.to_parquet(output_path, index=False)
            stats = {"file": fc_path.name, **stats}

            if args.net_lane:
                sc_source = (
                    args.processed_dir
                    / f"SECOND_CHANCE_CLEAN{yy:02d}{suffix}.parquet"
                )
                if not sc_source.exists():
                    raise FileNotFoundError(f"Missing net-lane input: {sc_source}")
                sc_output = args.output_dir / sc_source.name
                if sc_output.exists() and not args.overwrite:
                    raise FileExistsError(
                        f"{sc_output} already exists; pass --overwrite to replace it"
                    )
                fc_kappa = pd.to_numeric(
                    swapped[KAPPA_COLUMN], errors="coerce"
                ).to_numpy(dtype=np.float64)
                sc_stats = rewrite_net_second_chance(
                    sc_source, sc_output, fc_kappa=fc_kappa
                )
                stats.update(
                    {key: value for key, value in sc_stats.items() if key != "file"}
                )

            reports.append(stats)
            message = (
                f"{fc_path.name}: attached={stats['ev_attached_rows']:,} "
                f"closure={stats['max_abs_vpm_shot_closure_gap']:.3g}"
            )
            if args.net_lane:
                message += (
                    f" net_closure={stats['max_abs_vpm_net_shot_closure_gap']:.3g}"
                )
            print(message, flush=True)

    if args.link_unchanged_families:
        if args.net_lane:
            _link_unchanged_families(
                args.rapm_source_dir or DEFAULT_PROCESSED_DIR,
                args.output_dir,
                args.years,
                args.season_types,
                args.overwrite,
                families=("RAPM",),
            )
        else:
            _link_unchanged_families(
                args.processed_dir,
                args.output_dir,
                args.years,
                args.season_types,
                args.overwrite,
            )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(reports).to_csv(args.report, index=False)
    print(f"Wrote {args.report}", flush=True)


if __name__ == "__main__":
    main()
