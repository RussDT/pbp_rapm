#!/usr/bin/env python3
"""Season-forward rim ShotQuality challenger with an explicit block channel.

For test season S, every fitted model uses only seasons before S.  Shooter and
assigned-defender outcome priors are shifted by whole game, while Dunks &
Threes predictive rates come from the latest requested snapshot on or before
the game date.  The current shot's make/miss and block outcomes are labels
only.

This is an isolated research challenger.  It does not change
RUSSELL_SHOTQUALITY, DECOMP, or the accepted VPM target surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from nba_pipeline.scripts.train_sq_walk_forward_challenger import (
    DEFENDER_GEOMETRY_FEATURES,
    DESCRIPTOR_TAGS,
    binary_log_loss,
    classify_segment,
    clustered_delta_interval,
    descriptor_features,
    load_game_dates,
    metric_row,
    normalize_game_id,
    resolve_shot_rows,
)


DEFAULT_SOURCE = Path(
    "/Users/russ/Documents/Codex/basketball/nba/research/vpm/"
    "vpm/data/sq_shots_keyed.parquet"
)
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = PIPELINE_ROOT / "raw_data"
DEFAULT_PROCESSED = PIPELINE_ROOT / "processed"
DEFAULT_EPM_CACHE = PIPELINE_ROOT / "cache" / "epm"
DEFAULT_OUTPUT = PIPELINE_ROOT / "results" / "sq_rim_block_challenger"

BASE_FEATURES = (
    "period",
    "minutes",
    "seconds",
    "shot_x",
    "shot_y",
    "spacing",
    "feature_store.Shot Distance (ft.)",
    "shooter_prior_make_probability",
    "shooter_prior_make_log_attempts",
)

SHOOTER_EPM_FEATURES = (
    "epm_shooter_rim_make_probability",
    "epm_shooter_rim_attempts_100",
    "epm_shooter_snapshot_age_days",
)

BLOCK_FEATURES = (
    "shooter_prior_block_probability",
    "shooter_prior_block_log_attempts",
    "assigned_defender_prior_block_probability",
    "assigned_defender_prior_block_log_attempts",
    "epm_assigned_defender_blk_100",
    "epm_assigned_defender_snapshot_age_days",
    "epm_defense_lineup_blk_sum",
    "epm_defense_lineup_blk_max",
    "epm_defense_lineup_blk_top2",
    "epm_defense_lineup_blk_coverage",
    "primary_defender_identity_available",
    "defense_lineup_available",
)

MODEL_FEATURES = {
    "context_internal": BASE_FEATURES,
    "context_epm": BASE_FEATURES + SHOOTER_EPM_FEATURES,
    "context_epm_geometry": (
        BASE_FEATURES + SHOOTER_EPM_FEATURES + DEFENDER_GEOMETRY_FEATURES
    ),
    "context_epm_block": BASE_FEATURES + SHOOTER_EPM_FEATURES + BLOCK_FEATURES,
    "context_epm_geometry_block": (
        BASE_FEATURES
        + SHOOTER_EPM_FEATURES
        + DEFENDER_GEOMETRY_FEATURES
        + BLOCK_FEATURES
    ),
    "anchored_initial_ev_geometry_block": (
        ("initial_make_probability",)
        + BASE_FEATURES
        + SHOOTER_EPM_FEATURES
        + DEFENDER_GEOMETRY_FEATURES
        + BLOCK_FEATURES
    ),
}


def raw_event_paths(raw_dir: Path, seasons: Iterable[int]) -> List[Path]:
    paths: List[Path] = []
    for season in sorted(set(int(value) for value in seasons)):
        suffix = str(season)[-2:]
        for postseason in ("", "_PS"):
            path = raw_dir / f"NBA{suffix}{postseason}.parquet"
            if path.exists():
                paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No modern raw PBP parquets found under {raw_dir}")
    return paths


def load_raw_shot_context(
    raw_dir: Path,
    seasons: Iterable[int],
) -> Tuple[pd.DataFrame, Mapping[str, int]]:
    """Load block labels and lineups, rejecting conflicting raw event keys."""
    context_columns = [
        "game_id",
        "event_num",
        "player1_id",
        "home_description",
        "visitor_description",
        *[
            f"{side}_player{slot}"
            for side in ("home", "away")
            for slot in range(1, 6)
        ],
    ]
    frames = [
        pd.read_parquet(path, columns=context_columns)
        for path in raw_event_paths(raw_dir, seasons)
    ]
    raw = pd.concat(frames, ignore_index=True)
    raw["game_id_normalized"] = normalize_game_id(raw["game_id"])
    raw["GAME_EVENT_ID"] = pd.to_numeric(
        raw["event_num"], errors="coerce"
    ).astype("Int64")
    keys = ["game_id_normalized", "GAME_EVENT_ID"]
    raw = raw.dropna(subset=keys).copy()
    source_rows = len(raw)
    raw = raw.drop_duplicates().copy()
    duplicate_rows_removed = source_rows - len(raw)
    counts = raw.groupby(keys, dropna=False).size()
    conflicting = counts[counts > 1].index
    if len(conflicting):
        key_index = pd.MultiIndex.from_frame(raw[keys])
        conflict_mask = key_index.isin(conflicting)
        conflict_rows_removed = int(conflict_mask.sum())
        raw = raw.loc[~conflict_mask].copy()
    else:
        conflict_rows_removed = 0
    if raw.duplicated(keys).any():
        raise AssertionError("Raw shot context did not resolve to unique event keys")

    descriptions = raw[["home_description", "visitor_description"]].fillna("")
    raw["blocked"] = descriptions.agg(" ".join, axis=1).str.contains(
        "BLOCK", case=False, regex=False
    )
    keep = keys + [
        "player1_id",
        "blocked",
        *[
            f"{side}_player{slot}"
            for side in ("home", "away")
            for slot in range(1, 6)
        ],
    ]
    audit = {
        "raw_context_source_rows": int(source_rows),
        "raw_context_exact_duplicate_rows_removed": int(duplicate_rows_removed),
        "raw_context_conflicting_keys_removed": int(len(conflicting)),
        "raw_context_conflicting_rows_removed": int(conflict_rows_removed),
        "raw_context_resolved_rows": int(len(raw)),
    }
    return raw[keep].reset_index(drop=True), audit


def attach_block_labels_and_lineups(
    shots: pd.DataFrame,
    raw_context: pd.DataFrame,
) -> Tuple[pd.DataFrame, Mapping[str, float]]:
    out = shots.merge(
        raw_context,
        on=["game_id_normalized", "GAME_EVENT_ID"],
        how="left",
        validate="one_to_one",
        indicator="_raw_context_join",
    )
    matched = out["_raw_context_join"].eq("both")
    unmatched_rows = int((~matched).sum())
    out = out.loc[matched].drop(columns="_raw_context_join").copy()
    out["blocked"] = out["blocked"].astype(bool)
    out["made"] = out["made"].astype(bool).astype(int)
    ambiguous_outcome = out["blocked"] & out["made"].eq(1)
    made_and_blocked = int(ambiguous_outcome.sum())
    out = out.loc[~ambiguous_outcome].copy()

    shooter = pd.to_numeric(out["personId"], errors="coerce")
    raw_shooter = pd.to_numeric(out["player1_id"], errors="coerce")
    shooter_match = shooter.eq(raw_shooter) | shooter.isna() | raw_shooter.isna()
    mismatch_rows = int((~shooter_match).sum())
    out = out.loc[shooter_match].copy()

    home_columns = [f"home_player{slot}" for slot in range(1, 6)]
    away_columns = [f"away_player{slot}" for slot in range(1, 6)]
    home = out[home_columns].apply(pd.to_numeric, errors="coerce")
    away = out[away_columns].apply(pd.to_numeric, errors="coerce")
    shooter = pd.to_numeric(out["personId"], errors="coerce")
    shooter_home = home.eq(shooter, axis=0).any(axis=1)
    shooter_away = away.eq(shooter, axis=0).any(axis=1)
    resolved_side = shooter_home ^ shooter_away
    for slot in range(1, 6):
        out[f"D{slot}"] = np.where(
            shooter_home,
            pd.to_numeric(out[f"away_player{slot}"], errors="coerce"),
            np.where(
                shooter_away,
                pd.to_numeric(out[f"home_player{slot}"], errors="coerce"),
                np.nan,
            ),
        )
    out["defense_lineup_available"] = resolved_side.astype(float)
    out = out.drop(columns=home_columns + away_columns + ["player1_id"])
    audit = {
        "rim_rows_without_raw_context_removed": unmatched_rows,
        "rim_rows_made_and_blocked_removed": made_and_blocked,
        "rim_rows_shooter_key_mismatch_removed": mismatch_rows,
        "rim_rows_with_defense_lineup": int(resolved_side.sum()),
        "rim_defense_lineup_coverage": float(resolved_side.mean()),
        "rim_block_rate": float(out["blocked"].mean()),
    }
    return out.reset_index(drop=True), audit


def add_game_shifted_binary_prior(
    frame: pd.DataFrame,
    *,
    player_column: str,
    target_column: str,
    prefix: str,
    shrink_attempts: float,
    baseline: float,
) -> pd.DataFrame:
    """Attach a player prior whose current-game outcomes are fully excluded."""
    out = frame.copy()
    id_column = f"_{prefix}_id"
    out[id_column] = pd.to_numeric(out[player_column], errors="coerce").astype(
        "Int64"
    )
    valid = out[id_column].notna() & out[id_column].gt(0)
    games = (
        out.loc[valid]
        .groupby(
            [id_column, "game_date", "game_id_normalized"],
            dropna=False,
            sort=True,
        )
        .agg(
            game_events=(target_column, "sum"),
            game_attempts=(target_column, "size"),
        )
        .reset_index()
        .sort_values(
            [id_column, "game_date", "game_id_normalized"], kind="stable"
        )
    )
    grouped = games.groupby(id_column, sort=False)
    games["prior_events"] = grouped["game_events"].cumsum() - games["game_events"]
    games["prior_attempts"] = (
        grouped["game_attempts"].cumsum() - games["game_attempts"]
    )
    games[f"{prefix}_probability"] = (
        games["prior_events"] + float(shrink_attempts) * float(baseline)
    ) / (games["prior_attempts"] + float(shrink_attempts))
    games[f"{prefix}_log_attempts"] = np.log1p(games["prior_attempts"])
    join_columns = [id_column, "game_date", "game_id_normalized"]
    values = [f"{prefix}_probability", f"{prefix}_log_attempts"]
    out = out.merge(
        games[join_columns + values],
        on=join_columns,
        how="left",
        validate="many_to_one",
    )
    out[f"{prefix}_probability"] = out[f"{prefix}_probability"].fillna(
        float(baseline)
    )
    out[f"{prefix}_log_attempts"] = out[f"{prefix}_log_attempts"].fillna(0.0)
    return out.drop(columns=id_column)


def load_epm_snapshots(cache_dir: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(cache_dir.glob("epm_all_*_go0.csv")):
        parts = path.stem.split("_")
        if len(parts) < 4:
            continue
        snapshot_date = pd.to_datetime(parts[2], errors="coerce")
        if pd.isna(snapshot_date):
            continue
        available = pd.read_csv(path, nrows=0).columns
        wanted = [
            "player_id",
            "p_fgpct_rim",
            "p_fga_rim_100",
            "p_blk_100",
        ]
        if not set(wanted).issubset(available):
            continue
        snap = pd.read_csv(path, usecols=wanted)
        snap["snapshot_date"] = snapshot_date
        frames.append(snap)
    if not frames:
        raise FileNotFoundError(f"No usable rim/block EPM snapshots under {cache_dir}")
    out = pd.concat(frames, ignore_index=True)
    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").astype(
        "Int64"
    )
    out = out.dropna(subset=["player_id"])
    duplicates = out.duplicated(["player_id", "snapshot_date"], keep=False)
    if duplicates.any():
        out = out.drop_duplicates()
        if out.duplicated(["player_id", "snapshot_date"]).any():
            raise ValueError("Conflicting EPM player/date rows found")
    return out.sort_values(["snapshot_date", "player_id"], kind="stable")


def attach_epm_player(
    frame: pd.DataFrame,
    snapshots: pd.DataFrame,
    *,
    player_column: str,
    prefix: str,
) -> pd.DataFrame:
    out = frame.copy()
    out["_epm_player_id"] = pd.to_numeric(
        out[player_column], errors="coerce"
    ).astype("Int64")
    out["_row_order"] = np.arange(len(out), dtype="int64")
    left = out.sort_values(["game_date", "_epm_player_id"], kind="stable")
    right = snapshots.rename(columns={"player_id": "_epm_player_id"}).sort_values(
        ["snapshot_date", "_epm_player_id"], kind="stable"
    )
    merged = pd.merge_asof(
        left,
        right,
        left_on="game_date",
        right_on="snapshot_date",
        by="_epm_player_id",
        direction="backward",
        allow_exact_matches=True,
    )
    rename = {
        "p_fgpct_rim": f"{prefix}_rim_make_probability",
        "p_fga_rim_100": f"{prefix}_rim_attempts_100",
        "p_blk_100": f"{prefix}_blk_100",
        "snapshot_date": f"{prefix}_snapshot_date",
    }
    merged = merged.rename(columns=rename)
    merged[f"{prefix}_snapshot_age_days"] = (
        merged["game_date"] - merged[f"{prefix}_snapshot_date"]
    ).dt.days
    return (
        merged.sort_values("_row_order", kind="stable")
        .drop(columns=["_epm_player_id", "_row_order"])
        .reset_index(drop=True)
    )


def attach_epm_features(
    frame: pd.DataFrame,
    snapshots: pd.DataFrame,
) -> pd.DataFrame:
    out = attach_epm_player(
        frame, snapshots, player_column="personId", prefix="epm_shooter"
    )
    unsupported = pd.to_numeric(
        out["epm_shooter_rim_attempts_100"], errors="coerce"
    ).fillna(0.0).lt(0.5)
    out.loc[unsupported, "epm_shooter_rim_make_probability"] = np.nan

    out = attach_epm_player(
        out,
        snapshots,
        player_column="opp_personId",
        prefix="epm_assigned_defender",
    )
    for slot in range(1, 6):
        out = attach_epm_player(
            out,
            snapshots,
            player_column=f"D{slot}",
            prefix=f"epm_d{slot}",
        )
    block_columns = [f"epm_d{slot}_blk_100" for slot in range(1, 6)]
    block_values = out[block_columns].apply(pd.to_numeric, errors="coerce")
    out["epm_defense_lineup_blk_sum"] = block_values.sum(axis=1, min_count=1)
    out["epm_defense_lineup_blk_max"] = block_values.max(axis=1)
    sorted_values = np.sort(block_values.fillna(-np.inf).to_numpy(), axis=1)
    top_two = sorted_values[:, -2:]
    top_two[~np.isfinite(top_two)] = np.nan
    out["epm_defense_lineup_blk_top2"] = np.nansum(top_two, axis=1)
    all_missing = block_values.notna().sum(axis=1).eq(0)
    out.loc[all_missing, "epm_defense_lineup_blk_top2"] = np.nan
    out["epm_defense_lineup_blk_coverage"] = block_values.notna().mean(axis=1)

    drop = []
    for slot in range(1, 6):
        drop.extend(
            [
                f"epm_d{slot}_rim_make_probability",
                f"epm_d{slot}_rim_attempts_100",
                f"epm_d{slot}_snapshot_date",
                f"epm_d{slot}_snapshot_age_days",
            ]
        )
    drop.extend(
        [
            "epm_assigned_defender_rim_make_probability",
            "epm_assigned_defender_rim_attempts_100",
            "epm_assigned_defender_snapshot_date",
            "epm_shooter_blk_100",
            "epm_shooter_snapshot_date",
        ]
    )
    return out.drop(columns=[column for column in drop if column in out])


def build_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = sorted(
        {
            "initial_make_probability",
            *BASE_FEATURES,
            *SHOOTER_EPM_FEATURES,
            *DEFENDER_GEOMETRY_FEATURES,
            *BLOCK_FEATURES,
        }
    )
    features = pd.DataFrame(index=frame.index)
    for column in numeric_columns:
        features[column] = pd.to_numeric(frame[column], errors="coerce")
    return pd.concat(
        [features, descriptor_features(frame["play_descriptors"])], axis=1
    ).astype(float)


def new_model(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=450,
        learning_rate=0.04,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=3.0,
        early_stopping=True,
        validation_fraction=0.08,
        n_iter_no_change=30,
        random_state=seed,
    )


def prepare_shots(
    source: Path,
    *,
    raw_dir: Path,
    processed_dir: Path,
    epm_cache: Path,
    shooter_make_shrink: float,
    shooter_block_shrink: float,
    defender_block_shrink: float,
) -> Tuple[pd.DataFrame, Mapping[str, object]]:
    frame = pd.read_parquet(source)
    frame, provider_audit = resolve_shot_rows(frame)
    frame["game_id_normalized"] = normalize_game_id(frame["game_id"])
    frame["GAME_EVENT_ID"] = pd.to_numeric(
        frame["GAME_EVENT_ID"], errors="coerce"
    ).astype("Int64")
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce").astype(
        "Int64"
    )
    frame = frame.dropna(subset=["season", "made"]).copy()
    frame["season"] = frame["season"].astype(int)
    frame["segment"] = classify_segment(frame)
    frame = frame.loc[frame["segment"].eq("rim")].copy()
    frame["shot_value"] = 2.0
    frame["initial_make_probability"] = (
        pd.to_numeric(frame["ev.initial_ev.value"], errors="coerce") / 2.0
    ).clip(0.001, 0.999)
    frame = frame.dropna(subset=["initial_make_probability"]).copy()

    game_dates = load_game_dates(processed_dir, frame["season"].unique())
    frame = frame.merge(
        game_dates,
        on="game_id_normalized",
        how="left",
        validate="many_to_one",
    )
    if frame["game_date"].isna().any():
        raise ValueError(
            f"Missing game dates for {int(frame['game_date'].isna().sum())} rim rows"
        )
    raw_context, raw_audit = load_raw_shot_context(raw_dir, frame["season"].unique())
    frame, join_audit = attach_block_labels_and_lineups(frame, raw_context)
    frame = frame.sort_values(
        ["game_date", "game_id_normalized", "period", "minutes", "seconds"],
        ascending=[True, True, True, False, False],
        kind="stable",
    ).reset_index(drop=True)

    frame = add_game_shifted_binary_prior(
        frame,
        player_column="personId",
        target_column="made",
        prefix="shooter_prior_make",
        shrink_attempts=shooter_make_shrink,
        baseline=0.60,
    )
    frame = add_game_shifted_binary_prior(
        frame,
        player_column="personId",
        target_column="blocked",
        prefix="shooter_prior_block",
        shrink_attempts=shooter_block_shrink,
        baseline=0.122,
    )
    frame = add_game_shifted_binary_prior(
        frame,
        player_column="opp_personId",
        target_column="blocked",
        prefix="assigned_defender_prior_block",
        shrink_attempts=defender_block_shrink,
        baseline=0.122,
    )
    frame["primary_defender_identity_available"] = (
        pd.to_numeric(frame["opp_personId"], errors="coerce").notna().astype(float)
    )
    closest = pd.to_numeric(
        frame["feature_store.Distance to the Closest Defender (Feet)"],
        errors="coerce",
    )
    frame["defender_geometry_available"] = closest.notna().astype(float)
    frame = attach_epm_features(frame, load_epm_snapshots(epm_cache))

    audit = {
        **provider_audit,
        **raw_audit,
        **join_audit,
        "model_rows": int(len(frame)),
        "games": int(frame["game_id_normalized"].nunique()),
        "seasons": sorted(frame["season"].unique().astype(int).tolist()),
        "initial_ev_coverage": float(frame["initial_make_probability"].notna().mean()),
        "geometry_coverage": float(frame["defender_geometry_available"].mean()),
        "shooter_epm_coverage": float(
            frame["epm_shooter_rim_make_probability"].notna().mean()
        ),
        "assigned_defender_epm_block_coverage": float(
            frame["epm_assigned_defender_blk_100"].notna().mean()
        ),
        "lineup_epm_block_full_coverage": float(
            frame["epm_defense_lineup_blk_coverage"].eq(1.0).mean()
        ),
    }
    return frame, audit


def block_metric_row(
    frame: pd.DataFrame,
    probability: np.ndarray,
    *,
    model: str,
    coverage_slice: str,
) -> Dict[str, object]:
    target = frame["blocked"].to_numpy(dtype=float)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    return {
        "test_season": int(frame["season"].iloc[0]),
        "coverage_slice": coverage_slice,
        "model": model,
        "shots": int(len(frame)),
        "games": int(frame["game_id_normalized"].nunique()),
        "actual_block_rate": float(target.mean()),
        "predicted_block_rate": float(probability.mean()),
        "brier": float(np.mean((target - probability) ** 2)),
        "log_loss": float(binary_log_loss(target, probability).mean()),
        "roc_auc": float(roc_auc_score(target, probability)),
    }


def run_walk_forward(
    frame: pd.DataFrame,
    *,
    test_seasons: Sequence[int],
    bootstrap_iterations: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = build_feature_frame(frame)
    descriptor_columns = [
        column for column in features.columns if column.startswith("descriptor::")
    ]
    metrics: List[Dict[str, object]] = []
    block_metrics: List[Dict[str, object]] = []
    intervals: List[Dict[str, object]] = []
    prediction_frames: List[pd.DataFrame] = []

    for test_season in test_seasons:
        train_mask = frame["season"].lt(test_season)
        test_mask = frame["season"].eq(test_season)
        train = frame.loc[train_mask]
        test = frame.loc[test_mask]
        if train.empty or test.empty:
            continue
        predictions: Dict[str, np.ndarray] = {
            "initial_ev": test["initial_make_probability"].to_numpy(dtype=float)
        }
        fitted = {}
        for arm_index, (arm, columns) in enumerate(MODEL_FEATURES.items()):
            print(
                f"fit test_season={test_season} arm={arm} "
                f"train={len(train):,} test={len(test):,}",
                flush=True,
            )
            arm_columns = list(columns) + descriptor_columns
            model = new_model(seed + 100 * test_season + 10 * arm_index)
            model.fit(
                features.loc[train_mask, arm_columns],
                train["made"].to_numpy(dtype=int),
            )
            predictions[arm] = model.predict_proba(
                features.loc[test_mask, arm_columns]
            )[:, 1]
            fitted[arm] = model

        full_columns = list(MODEL_FEATURES["context_epm_geometry_block"]) + descriptor_columns
        anchored_full_columns = (
            list(MODEL_FEATURES["anchored_initial_ev_geometry_block"])
            + descriptor_columns
        )
        block_model = new_model(seed + 100 * test_season + 901)
        block_model.fit(
            features.loc[train_mask, full_columns],
            train["blocked"].to_numpy(dtype=int),
        )
        p_block = block_model.predict_proba(features.loc[test_mask, full_columns])[:, 1]
        nonblock_train_mask = train_mask & ~frame["blocked"]
        nonblock_make_model = new_model(seed + 100 * test_season + 902)
        nonblock_make_model.fit(
            features.loc[nonblock_train_mask, full_columns],
            frame.loc[nonblock_train_mask, "made"].to_numpy(dtype=int),
        )
        p_make_if_unblocked = nonblock_make_model.predict_proba(
            features.loc[test_mask, full_columns]
        )[:, 1]
        predictions["staged_block_make"] = np.clip(
            (1.0 - p_block) * p_make_if_unblocked, 0.001, 0.999
        )
        anchored_nonblock_make_model = new_model(seed + 100 * test_season + 903)
        anchored_nonblock_make_model.fit(
            features.loc[nonblock_train_mask, anchored_full_columns],
            frame.loc[nonblock_train_mask, "made"].to_numpy(dtype=int),
        )
        p_make_if_unblocked_anchored = anchored_nonblock_make_model.predict_proba(
            features.loc[test_mask, anchored_full_columns]
        )[:, 1]
        predictions["staged_block_make_anchored"] = np.clip(
            (1.0 - p_block) * p_make_if_unblocked_anchored, 0.001, 0.999
        )

        output_columns = [
            "game_id_normalized",
            "GAME_EVENT_ID",
            "game_date",
            "season",
            "personId",
            "opp_personId",
            "made",
            "blocked",
            "shot_value",
            "defender_geometry_available",
            "epm_shooter_rim_make_probability",
            "epm_assigned_defender_blk_100",
            "epm_defense_lineup_blk_sum",
        ]
        output = test[output_columns].copy()
        output["p_block"] = p_block
        output["p_make_if_unblocked"] = p_make_if_unblocked
        output["p_make_if_unblocked_anchored"] = p_make_if_unblocked_anchored
        for model_name, probability in predictions.items():
            output[f"p_{model_name}"] = probability
            output[f"pps_{model_name}"] = probability * 2.0
        prediction_frames.append(output)

        slices = {
            "all_shots": np.ones(len(test), dtype=bool),
            "geometry_available": test[
                "defender_geometry_available"
            ].to_numpy(dtype=bool),
            "geometry_missing": ~test[
                "defender_geometry_available"
            ].to_numpy(dtype=bool),
        }
        test_reset = test.reset_index(drop=True)
        for slice_name, slice_mask in slices.items():
            sliced = test_reset.loc[slice_mask].reset_index(drop=True)
            if sliced.empty:
                continue
            for model_name, probability in predictions.items():
                metrics.append(
                    metric_row(
                        sliced,
                        probability[slice_mask],
                        model=model_name,
                        coverage_slice=slice_name,
                    )
                )
            block_metrics.append(
                block_metric_row(
                    sliced,
                    p_block[slice_mask],
                    model="staged_block_channel",
                    coverage_slice=slice_name,
                )
            )
            comparisons = (
                ("context_epm", "context_internal"),
                ("context_epm_geometry", "context_epm"),
                ("context_epm_block", "context_epm"),
                ("context_epm_geometry_block", "context_epm_geometry"),
                ("context_epm_geometry_block", "context_epm_block"),
                ("staged_block_make", "context_epm_geometry_block"),
                (
                    "anchored_initial_ev_geometry_block",
                    "initial_ev",
                ),
                (
                    "staged_block_make_anchored",
                    "anchored_initial_ev_geometry_block",
                ),
                ("context_epm_geometry_block", "initial_ev"),
                ("staged_block_make", "initial_ev"),
                ("staged_block_make_anchored", "initial_ev"),
            )
            for challenger_name, reference_name in comparisons:
                challenger = predictions[challenger_name][slice_mask]
                reference = predictions[reference_name][slice_mask]
                for metric in ("points_mse", "log_loss", "brier"):
                    result = clustered_delta_interval(
                        sliced,
                        challenger,
                        reference,
                        metric=metric,
                        iterations=bootstrap_iterations,
                        seed=(
                            seed
                            + test_season
                            + len(intervals)
                            + len(challenger_name)
                        ),
                    )
                    intervals.append(
                        {
                            "test_season": int(test_season),
                            "segment": "rim",
                            "coverage_slice": slice_name,
                            "challenger": challenger_name,
                            "reference": reference_name,
                            "metric": metric,
                            **result,
                        }
                    )
    if not prediction_frames:
        raise ValueError("No rim test-season predictions were produced")
    return (
        pd.DataFrame(metrics),
        pd.DataFrame(block_metrics),
        pd.DataFrame(intervals),
        pd.concat(prediction_frames, ignore_index=True),
    )


def write_outputs(
    output_dir: Path,
    *,
    metrics: pd.DataFrame,
    block_metrics: pd.DataFrame,
    intervals: pd.DataFrame,
    predictions: pd.DataFrame,
    audit: Mapping[str, object],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "walk_forward_metrics.csv", index=False)
    block_metrics.to_csv(output_dir / "block_channel_metrics.csv", index=False)
    intervals.to_csv(output_dir / "walk_forward_intervals.csv", index=False)
    predictions.to_parquet(output_dir / "walk_forward_predictions.parquet", index=False)
    manifest = {
        "schema_version": 1,
        "source": str(args.source),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "test_seasons": list(args.test_seasons),
        "shooter_make_shrink": args.shooter_make_shrink,
        "shooter_block_shrink": args.shooter_block_shrink,
        "defender_block_shrink": args.defender_block_shrink,
        "bootstrap_iterations": args.bootstrap_iterations,
        "seed": args.seed,
        "staged_identity": "P(make)=(1-P(block))*P(make|not blocked)",
        "validation_contract": (
            "Test season S is scored by models fit only on seasons < S; "
            "outcome priors exclude the whole current game and EPM fields use "
            "the latest requested snapshot on or before game date."
        ),
        "provider_contract": (
            "ShotQuality geometry/primary defender and raw NBA block/lineup "
            "context remain separately identified; current-shot block is a "
            "label only."
        ),
        "model_features": {
            key: list(value) + [f"descriptor::{tag}" for tag in DESCRIPTOR_TAGS]
            for key, value in MODEL_FEATURES.items()
        },
        "data_audit": dict(audit),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--epm-cache", type=Path, default=DEFAULT_EPM_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-seasons", type=int, nargs="+", default=[2025, 2026])
    parser.add_argument("--shooter-make-shrink", type=float, default=100.0)
    parser.add_argument("--shooter-block-shrink", type=float, default=150.0)
    parser.add_argument("--defender-block-shrink", type=float, default=500.0)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=131)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shots, audit = prepare_shots(
        args.source,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        epm_cache=args.epm_cache,
        shooter_make_shrink=args.shooter_make_shrink,
        shooter_block_shrink=args.shooter_block_shrink,
        defender_block_shrink=args.defender_block_shrink,
    )
    metrics, block_metrics, intervals, predictions = run_walk_forward(
        shots,
        test_seasons=args.test_seasons,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    write_outputs(
        args.output_dir,
        metrics=metrics,
        block_metrics=block_metrics,
        intervals=intervals,
        predictions=predictions,
        audit=audit,
        args=args,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(metrics.to_string(index=False))
    print(block_metrics.to_string(index=False))
    print(intervals.to_string(index=False))


if __name__ == "__main__":
    main()
