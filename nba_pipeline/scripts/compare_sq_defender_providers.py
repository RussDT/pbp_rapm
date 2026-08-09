#!/usr/bin/env python3
"""Compare NBA assignments, lineup D-EPM, and SQ geometry on forward 3P prediction.

The NBA `matchup_fga` provider is an assignment source, not a tracking-distance
source. Multi-defender `DEF_ID` values are preserved and summarized through
prior-game-only, strongly shrunk defender make-rate-allowed features.

The comparison fits the context, rich-geometry, and NBA-assignment arms on the
same currently resolved rows before fitting a season-forward Platt calibration
of provider `initial_ev`. Opposing five-man lineup D-EPM uses only provider
snapshots requested on or before the shot date.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from nba_pipeline.scripts.train_sq_walk_forward_challenger import (
    BASE_NUMERIC_FEATURES,
    DEFENDER_GEOMETRY_FEATURES,
    DEFENDER_IDENTITY_FEATURES,
    EPM_FEATURES,
    binary_log_loss,
    build_feature_frame,
    clustered_delta_interval,
    metric_row,
    new_model,
    normalize_game_id,
    prepare_shots,
)


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATCHUP_ROOT = Path(
    "/Users/russ/Documents/Codex/basketball/nba/data-sources/matchup-fga"
)
DEFAULT_RAW_DIR = PIPELINE_ROOT / "raw_data"
DEFAULT_OUTPUT = (
    PIPELINE_ROOT / "results" / "sq_defender_provider_comparison"
)

NBA_ASSIGNMENT_FEATURES = (
    "nba_assignment_prior_make_probability_allowed",
    "nba_assignment_prior_log_attempts",
    "nba_assignment_count",
    "nba_single_assignment",
    "nba_assignment_available",
)

LINEUP_DEPM_FEATURES = (
    "def_lineup_depm_sum",
    "def_lineup_depm_min",
    "def_lineup_depm_max",
    "def_lineup_depm_std",
    "def_lineup_depm_coverage",
    "def_lineup_available",
)

LINEUP_DEPM_TOTAL_FEATURES = (
    "def_lineup_depm_sum",
    "def_lineup_depm_coverage",
    "def_lineup_available",
)

HOME_LINEUP_COLUMNS = tuple(f"home_player{index}" for index in range(1, 6))
AWAY_LINEUP_COLUMNS = tuple(f"away_player{index}" for index in range(1, 6))
OFF_LINEUP_COLUMNS = tuple(f"off_lineup_player{index}" for index in range(1, 6))
DEF_LINEUP_COLUMNS = tuple(f"def_lineup_player{index}" for index in range(1, 6))


def parse_defender_ids(value: object) -> Tuple[int, ...]:
    """Normalize a provider DEF_ID payload without selecting a first defender."""
    if pd.isna(value):
        return ()
    defenders: set[int] = set()
    for token in str(value).split("|"):
        token = token.strip()
        if not token:
            continue
        try:
            defender_id = int(float(token))
        except ValueError:
            continue
        if defender_id > 0:
            defenders.add(defender_id)
    return tuple(sorted(defenders))


def load_event_lineups(
    raw_dir: Path,
    *,
    seasons: Iterable[int],
) -> Tuple[pd.DataFrame, Mapping[str, object]]:
    """Load one unambiguous home/away lineup row per raw PBP event."""
    frames: List[pd.DataFrame] = []
    files: List[str] = []
    columns = [
        "game_id",
        "event_num",
        *HOME_LINEUP_COLUMNS,
        *AWAY_LINEUP_COLUMNS,
    ]
    for season in sorted({int(value) for value in seasons}):
        suffix = str(season)[-2:]
        for postseason in ("", "_PS"):
            path = raw_dir / f"NBA{suffix}{postseason}.parquet"
            if not path.exists():
                continue
            frames.append(pd.read_parquet(path, columns=columns))
            files.append(str(path))
    if not frames:
        raise FileNotFoundError(
            f"No raw event-lineup parquets found under {raw_dir}"
        )

    raw = pd.concat(frames, ignore_index=True)
    source_rows = len(raw)
    raw["game_id_normalized"] = normalize_game_id(raw["game_id"])
    raw["GAME_EVENT_ID"] = pd.to_numeric(
        raw["event_num"], errors="coerce"
    ).astype("Int64")
    raw = raw[
        raw["game_id_normalized"].ne("") & raw["GAME_EVENT_ID"].notna()
    ].copy()
    payload_columns = [
        "game_id_normalized",
        "GAME_EVENT_ID",
        *HOME_LINEUP_COLUMNS,
        *AWAY_LINEUP_COLUMNS,
    ]
    exact = raw[payload_columns].drop_duplicates().copy()
    key_columns = ["game_id_normalized", "GAME_EVENT_ID"]
    counts = exact.groupby(key_columns, dropna=False).size()
    conflicting_keys = counts[counts > 1].index
    if len(conflicting_keys):
        keyed = pd.MultiIndex.from_frame(exact[key_columns])
        conflict_mask = keyed.isin(conflicting_keys)
        conflict_rows = int(conflict_mask.sum())
        exact = exact.loc[~conflict_mask].copy()
    else:
        conflict_rows = 0
    if exact.duplicated(key_columns).any():
        raise AssertionError("Resolved raw event lineups are not event-unique")
    audit = {
        "files": files,
        "source_rows": int(source_rows),
        "missing_key_rows_removed": int(source_rows - len(raw)),
        "exact_duplicate_rows_removed": int(
            len(raw) - len(exact) - conflict_rows
        ),
        "conflicting_event_keys_removed": int(len(conflicting_keys)),
        "conflicting_rows_removed": conflict_rows,
        "resolved_events": int(len(exact)),
    }
    return exact.reset_index(drop=True), audit


def attach_event_lineups(
    shots: pd.DataFrame,
    event_lineups: pd.DataFrame,
) -> Tuple[pd.DataFrame, Mapping[str, object]]:
    """Orient home/away event lineups using the side containing the shooter."""
    out = shots.copy()
    out["_lineup_row_order"] = np.arange(len(out), dtype="int64")
    out = out.merge(
        event_lineups,
        on=["game_id_normalized", "GAME_EVENT_ID"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    shooter = pd.to_numeric(out["personId"], errors="coerce")
    home_values = out[list(HOME_LINEUP_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    )
    away_values = out[list(AWAY_LINEUP_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    )
    home_positive = home_values.gt(0)
    away_positive = away_values.gt(0)
    valid_home = home_positive.sum(axis=1).eq(5) & home_values.nunique(
        axis=1, dropna=True
    ).eq(5)
    valid_away = away_positive.sum(axis=1).eq(5) & away_values.nunique(
        axis=1, dropna=True
    ).eq(5)
    shooter_home = home_values.eq(shooter, axis=0).any(axis=1)
    shooter_away = away_values.eq(shooter, axis=0).any(axis=1)
    home_offense = valid_home & valid_away & shooter_home & ~shooter_away
    away_offense = valid_home & valid_away & shooter_away & ~shooter_home
    valid_orientation = home_offense | away_offense

    for index, (off_column, def_column) in enumerate(
        zip(OFF_LINEUP_COLUMNS, DEF_LINEUP_COLUMNS),
        start=1,
    ):
        home_column = f"home_player{index}"
        away_column = f"away_player{index}"
        out[off_column] = np.where(
            home_offense,
            home_values[home_column],
            np.where(away_offense, away_values[away_column], np.nan),
        )
        out[def_column] = np.where(
            home_offense,
            away_values[away_column],
            np.where(away_offense, home_values[home_column], np.nan),
        )
        out[off_column] = pd.to_numeric(out[off_column], errors="coerce").astype(
            "Int64"
        )
        out[def_column] = pd.to_numeric(out[def_column], errors="coerce").astype(
            "Int64"
        )
    out["def_lineup_available"] = valid_orientation.astype(float)
    matched_event = out[list(HOME_LINEUP_COLUMNS)].notna().any(axis=1)
    audit = {
        "shot_rows": int(len(out)),
        "matched_event_rows": int(matched_event.sum()),
        "matched_event_coverage": float(matched_event.mean()),
        "valid_oriented_lineups": int(valid_orientation.sum()),
        "valid_oriented_coverage": float(valid_orientation.mean()),
        "invalid_or_ambiguous_orientation_rows": int(
            matched_event.sum() - valid_orientation.sum()
        ),
        "unmatched_event_rows": int((~matched_event).sum()),
    }
    drop_columns = [
        "_lineup_row_order",
        *HOME_LINEUP_COLUMNS,
        *AWAY_LINEUP_COLUMNS,
    ]
    return out.drop(columns=drop_columns), audit


def load_lineup_epm_snapshots(cache_dir: Path) -> pd.DataFrame:
    """Load date-stamped player D-EPM snapshots for lineup aggregation."""
    frames: List[pd.DataFrame] = []
    for path in sorted(cache_dir.glob("epm_all_*_go0.csv")):
        parts = path.stem.split("_")
        if len(parts) < 4:
            continue
        snapshot_date = pd.to_datetime(parts[2], errors="coerce")
        if pd.isna(snapshot_date):
            continue
        available = pd.read_csv(path, nrows=0).columns
        if not {"player_id", "depm"}.issubset(available):
            continue
        frame = pd.read_csv(path, usecols=["player_id", "depm"])
        frame["snapshot_date"] = snapshot_date
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"No EPM snapshots with player_id/depm found under {cache_dir}"
        )
    out = pd.concat(frames, ignore_index=True)
    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").astype(
        "Int64"
    )
    out["depm"] = pd.to_numeric(out["depm"], errors="coerce")
    out = out.dropna(subset=["player_id"])
    deduped = out.drop_duplicates()
    if deduped.duplicated(["player_id", "snapshot_date"]).any():
        raise ValueError("Conflicting D-EPM player/date rows found")
    return deduped.sort_values(
        ["snapshot_date", "player_id"], kind="stable"
    ).reset_index(drop=True)


def attach_lineup_depm_features(
    shots: pd.DataFrame,
    snapshots: pd.DataFrame,
) -> Tuple[pd.DataFrame, Mapping[str, object]]:
    """Attach five-defender as-of D-EPM aggregates to every shot."""
    out = shots.copy()
    out["_lineup_epm_row_id"] = np.arange(len(out), dtype="int64")
    links = out[
        ["_lineup_epm_row_id", "game_date", *DEF_LINEUP_COLUMNS]
    ].melt(
        id_vars=["_lineup_epm_row_id", "game_date"],
        value_vars=list(DEF_LINEUP_COLUMNS),
        var_name="defender_slot",
        value_name="player_id",
    )
    links["player_id"] = pd.to_numeric(
        links["player_id"], errors="coerce"
    ).astype("Int64")
    valid_links = links["player_id"].notna() & links["player_id"].gt(0)
    links = links.loc[valid_links].copy()
    links["_link_order"] = np.arange(len(links), dtype="int64")
    left = links.sort_values(["game_date", "player_id"], kind="stable")
    right = snapshots.sort_values(
        ["snapshot_date", "player_id"], kind="stable"
    )
    links = pd.merge_asof(
        left,
        right,
        left_on="game_date",
        right_on="snapshot_date",
        by="player_id",
        direction="backward",
        allow_exact_matches=True,
    ).sort_values("_link_order", kind="stable")
    if (
        links["snapshot_date"].notna()
        & links["snapshot_date"].gt(links["game_date"])
    ).any():
        raise AssertionError("Future D-EPM snapshot entered lineup features")
    links["snapshot_age_days"] = (
        links["game_date"] - links["snapshot_date"]
    ).dt.days

    complete_rows = pd.DataFrame(
        {
            "_lineup_epm_row_id": np.repeat(
                out["_lineup_epm_row_id"].to_numpy(), 5
            ),
            "slot_index": np.tile(np.arange(5), len(out)),
        }
    )
    link_values = links[
        ["_lineup_epm_row_id", "defender_slot", "depm", "snapshot_age_days"]
    ].copy()
    slot_map = {
        column: index for index, column in enumerate(DEF_LINEUP_COLUMNS)
    }
    link_values["slot_index"] = link_values["defender_slot"].map(slot_map)
    complete_rows = complete_rows.merge(
        link_values.drop(columns="defender_slot"),
        on=["_lineup_epm_row_id", "slot_index"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    complete_rows["depm_known"] = complete_rows["depm"].notna()
    complete_rows["depm_imputed"] = complete_rows["depm"].fillna(0.0)
    summaries = (
        complete_rows.groupby("_lineup_epm_row_id")
        .agg(
            def_lineup_depm_sum=("depm_imputed", "sum"),
            def_lineup_depm_min=("depm_imputed", "min"),
            def_lineup_depm_max=("depm_imputed", "max"),
            def_lineup_depm_std=("depm_imputed", "std"),
            def_lineup_depm_coverage=("depm_known", "sum"),
            def_lineup_epm_snapshot_age_mean=("snapshot_age_days", "mean"),
            def_lineup_epm_snapshot_age_max=("snapshot_age_days", "max"),
        )
        .reset_index()
    )
    out = out.merge(
        summaries,
        on="_lineup_epm_row_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    lineup_available = out["def_lineup_available"].eq(1.0)
    for column in LINEUP_DEPM_FEATURES:
        if column == "def_lineup_available":
            continue
        out.loc[~lineup_available, column] = np.nan
    coverage = pd.to_numeric(
        out["def_lineup_depm_coverage"], errors="coerce"
    )
    audit = {
        "shot_rows": int(len(out)),
        "lineup_available_rows": int(lineup_available.sum()),
        "lineup_available_coverage": float(lineup_available.mean()),
        "complete_five_player_epm_rows": int(
            (lineup_available & coverage.eq(5)).sum()
        ),
        "complete_five_player_epm_coverage": float(
            (lineup_available & coverage.eq(5)).mean()
        ),
        "lineup_rows_with_any_epm": int(
            (lineup_available & coverage.gt(0)).sum()
        ),
        "max_snapshot_age_days": (
            None
            if out["def_lineup_epm_snapshot_age_max"].dropna().empty
            else int(out["def_lineup_epm_snapshot_age_max"].max())
        ),
    }
    return out.drop(columns="_lineup_epm_row_id"), audit


def load_nba_assignments(
    matchup_root: Path,
    *,
    seasons: Iterable[int],
) -> Tuple[pd.DataFrame, Mapping[str, object]]:
    """Load one provider row per unambiguous shot event from player CSVs."""
    frames: List[pd.DataFrame] = []
    files_by_season: Dict[str, int] = {}
    usecols = [
        "GAME_ID",
        "GAME_EVENT_ID",
        "PLAYER_ID",
        "SHOT_MADE_FLAG",
        "GAME_DATE",
        "DEF_ID",
    ]
    for season in sorted({int(value) for value in seasons}):
        paths = sorted((matchup_root / "defender" / str(season)).glob("*.csv"))
        if not paths:
            raise FileNotFoundError(
                f"No NBA matchup defender CSVs found for season {season}"
            )
        files_by_season[str(season)] = len(paths)
        for path in paths:
            frame = pd.read_csv(
                path,
                usecols=lambda column: column in usecols,
                dtype=str,
                low_memory=False,
            )
            missing = sorted(set(usecols) - set(frame.columns))
            if missing:
                raise ValueError(f"{path} is missing columns: {missing}")
            frame["provider_season"] = season
            frames.append(frame)

    raw = pd.concat(frames, ignore_index=True)
    source_rows = len(raw)
    raw["game_id_normalized"] = normalize_game_id(raw["GAME_ID"])
    raw["GAME_EVENT_ID"] = pd.to_numeric(
        raw["GAME_EVENT_ID"], errors="coerce"
    ).astype("Int64")
    raw["nba_defender_ids"] = raw["DEF_ID"].map(parse_defender_ids)
    raw = raw[
        raw["game_id_normalized"].ne("")
        & raw["GAME_EVENT_ID"].notna()
        & raw["nba_defender_ids"].map(bool)
    ].copy()

    payload_columns = [
        "game_id_normalized",
        "GAME_EVENT_ID",
        "PLAYER_ID",
        "SHOT_MADE_FLAG",
        "GAME_DATE",
        "provider_season",
        "nba_defender_ids",
    ]
    exact = raw[payload_columns].drop_duplicates().copy()
    key_columns = ["game_id_normalized", "GAME_EVENT_ID"]
    counts = exact.groupby(key_columns, dropna=False).size()
    conflicting_keys = counts[counts > 1].index
    if len(conflicting_keys):
        keyed = pd.MultiIndex.from_frame(exact[key_columns])
        conflict_mask = keyed.isin(conflicting_keys)
        conflict_rows = int(conflict_mask.sum())
        exact = exact.loc[~conflict_mask].copy()
    else:
        conflict_rows = 0
    if exact.duplicated(key_columns).any():
        raise AssertionError("NBA matchup assignments are not event-unique")

    exact["nba_assignment_count"] = exact["nba_defender_ids"].map(len).astype(float)
    exact["nba_single_assignment"] = (
        exact["nba_assignment_count"].eq(1).astype(float)
    )
    exact["nba_assignment_available"] = 1.0
    audit = {
        "source_rows": int(source_rows),
        "files_by_season": files_by_season,
        "rows_without_usable_key_or_assignment": int(source_rows - len(raw)),
        "exact_duplicate_rows_removed": int(len(raw) - len(exact) - conflict_rows),
        "conflicting_event_keys_removed": int(len(conflicting_keys)),
        "conflicting_rows_removed": conflict_rows,
        "resolved_events": int(len(exact)),
        "multi_assignment_events": int(exact["nba_assignment_count"].gt(1).sum()),
    }
    return exact.reset_index(drop=True), audit


def attach_nba_assignment_priors(
    shots: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    shrink_attempts: float,
    baseline_by_segment: Mapping[str, float],
) -> Tuple[pd.DataFrame, Mapping[str, object]]:
    """Attach prior-game-only summaries across every assigned NBA defender."""
    out = shots.copy()
    out["_provider_row_id"] = np.arange(len(out), dtype="int64")
    merge_columns = [
        "game_id_normalized",
        "GAME_EVENT_ID",
        "nba_defender_ids",
        "nba_assignment_count",
        "nba_single_assignment",
        "nba_assignment_available",
    ]
    out = out.merge(
        assignments[merge_columns],
        on=["game_id_normalized", "GAME_EVENT_ID"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    matched = out["nba_assignment_available"].eq(1.0)

    links = out.loc[
        matched,
        [
            "_provider_row_id",
            "game_id_normalized",
            "game_date",
            "segment",
            "made",
            "nba_defender_ids",
        ],
    ].explode("nba_defender_ids")
    links["nba_defender_id"] = pd.to_numeric(
        links["nba_defender_ids"], errors="coerce"
    ).astype("Int64")
    links = links.dropna(subset=["nba_defender_id"]).copy()
    links = links.sort_values(
        ["game_date", "game_id_normalized", "_provider_row_id"],
        kind="stable",
    )
    games = (
        links.groupby(
            ["nba_defender_id", "segment", "game_date", "game_id_normalized"],
            sort=True,
        )
        .agg(game_makes=("made", "sum"), game_attempts=("made", "size"))
        .reset_index()
        .sort_values(
            ["nba_defender_id", "segment", "game_date", "game_id_normalized"],
            kind="stable",
        )
    )
    grouped = games.groupby(["nba_defender_id", "segment"], sort=False)
    games["prior_makes"] = grouped["game_makes"].cumsum() - games["game_makes"]
    games["prior_attempts"] = (
        grouped["game_attempts"].cumsum() - games["game_attempts"]
    )
    games["baseline"] = games["segment"].map(baseline_by_segment).astype(float)
    games["defender_prior"] = (
        games["prior_makes"] + shrink_attempts * games["baseline"]
    ) / (games["prior_attempts"] + shrink_attempts)

    links = links.merge(
        games[
            [
                "nba_defender_id",
                "segment",
                "game_date",
                "game_id_normalized",
                "defender_prior",
                "prior_attempts",
            ]
        ],
        on=[
            "nba_defender_id",
            "segment",
            "game_date",
            "game_id_normalized",
        ],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    summaries = (
        links.groupby("_provider_row_id")
        .agg(
            nba_assignment_prior_make_probability_allowed=(
                "defender_prior",
                "mean",
            ),
            nba_assignment_prior_attempts=("prior_attempts", "mean"),
        )
        .reset_index()
    )
    summaries["nba_assignment_prior_log_attempts"] = np.log1p(
        summaries.pop("nba_assignment_prior_attempts")
    )
    out = out.merge(
        summaries,
        on="_provider_row_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    baseline = out["segment"].map(baseline_by_segment).astype(float)
    out["nba_assignment_prior_make_probability_allowed"] = out[
        "nba_assignment_prior_make_probability_allowed"
    ].fillna(baseline)
    out["nba_assignment_prior_log_attempts"] = out[
        "nba_assignment_prior_log_attempts"
    ].fillna(0.0)
    for column in (
        "nba_assignment_count",
        "nba_single_assignment",
        "nba_assignment_available",
    ):
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)

    audit = {
        "shot_rows": int(len(out)),
        "matched_events": int(matched.sum()),
        "event_coverage": float(matched.mean()),
        "matched_single_assignment_events": int(
            out.loc[matched, "nba_single_assignment"].sum()
        ),
        "matched_multi_assignment_events": int(
            out.loc[matched, "nba_assignment_count"].gt(1).sum()
        ),
        "expanded_assignment_links": int(len(links)),
    }
    return (
        out.drop(columns=["_provider_row_id", "nba_defender_ids"]),
        audit,
    )


def platt_calibrate(
    train_probability: np.ndarray,
    train_target: np.ndarray,
    test_probability: np.ndarray,
) -> Tuple[np.ndarray, Mapping[str, float]]:
    """Fit a prior-season-only logistic calibration on provider probabilities."""
    train_probability = np.clip(train_probability.astype(float), 1e-6, 1 - 1e-6)
    test_probability = np.clip(test_probability.astype(float), 1e-6, 1 - 1e-6)
    train_logit = np.log(train_probability / (1.0 - train_probability)).reshape(
        -1, 1
    )
    test_logit = np.log(test_probability / (1.0 - test_probability)).reshape(
        -1, 1
    )
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(train_logit, train_target.astype(int))
    calibrated = model.predict_proba(test_logit)[:, 1]
    return calibrated, {
        "intercept": float(model.intercept_[0]),
        "slope": float(model.coef_[0, 0]),
    }


def comparison_metric_row(
    frame: pd.DataFrame,
    probability: np.ndarray,
    *,
    model: str,
    coverage_slice: str,
) -> Dict[str, object]:
    row = metric_row(
        frame,
        probability,
        model=model,
        coverage_slice=coverage_slice,
    )
    row["calibration_abs_error_pps"] = abs(row["pps_bias"])
    return row


def run_comparison(
    shots: pd.DataFrame,
    *,
    test_season: int,
    segment: str,
    bootstrap_iterations: int,
    model_seed: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Mapping[str, float]]:
    train_mask = shots["season"].lt(test_season) & shots["segment"].eq(segment)
    test_mask = shots["season"].eq(test_season) & shots["segment"].eq(segment)
    train = shots.loc[train_mask]
    test = shots.loc[test_mask].copy()
    if train.empty or test.empty:
        raise ValueError("The requested forward train/test split is empty")

    base_features = build_feature_frame(shots)
    for column in (*NBA_ASSIGNMENT_FEATURES, *LINEUP_DEPM_FEATURES):
        base_features[column] = pd.to_numeric(shots[column], errors="coerce")
    descriptor_columns = [
        column
        for column in base_features.columns
        if column.startswith("descriptor::")
    ]
    context_columns = (
        list(BASE_NUMERIC_FEATURES)
        + list(EPM_FEATURES)
        + descriptor_columns
    )
    arm_columns = {
        "context_epm": context_columns,
        "nba_assignment": context_columns + list(NBA_ASSIGNMENT_FEATURES),
        "lineup_depm_total": (
            context_columns + list(LINEUP_DEPM_TOTAL_FEATURES)
        ),
        "lineup_depm": context_columns + list(LINEUP_DEPM_FEATURES),
        "lineup_depm_nba_assignment": (
            context_columns
            + list(LINEUP_DEPM_FEATURES)
            + list(NBA_ASSIGNMENT_FEATURES)
        ),
        "sq_geometry": context_columns + list(DEFENDER_GEOMETRY_FEATURES),
        "sq_geometry_identity": (
            context_columns
            + list(DEFENDER_GEOMETRY_FEATURES)
            + list(DEFENDER_IDENTITY_FEATURES)
        ),
        "lineup_depm_sq_geometry": (
            context_columns
            + list(LINEUP_DEPM_FEATURES)
            + list(DEFENDER_GEOMETRY_FEATURES)
        ),
        "lineup_depm_sq_geometry_identity": (
            context_columns
            + list(LINEUP_DEPM_FEATURES)
            + list(DEFENDER_GEOMETRY_FEATURES)
            + list(DEFENDER_IDENTITY_FEATURES)
        ),
        "lineup_depm_total_sq_geometry_identity": (
            context_columns
            + list(LINEUP_DEPM_TOTAL_FEATURES)
            + list(DEFENDER_GEOMETRY_FEATURES)
            + list(DEFENDER_IDENTITY_FEATURES)
        ),
    }
    arm_seeds = {
        "context_epm": model_seed + 100 * test_season + 10,
        "sq_geometry": model_seed + 100 * test_season + 20,
        "sq_geometry_identity": model_seed + 100 * test_season + 30,
        "nba_assignment": model_seed + 100 * test_season + 40,
        # Nested lineup arms reuse the exact reference-arm seed so the
        # early-stopping validation split and random choices remain aligned.
        "lineup_depm": model_seed + 100 * test_season + 10,
        "lineup_depm_total": model_seed + 100 * test_season + 10,
        "lineup_depm_nba_assignment": model_seed + 100 * test_season + 40,
        "lineup_depm_sq_geometry": model_seed + 100 * test_season + 20,
        "lineup_depm_sq_geometry_identity": (
            model_seed + 100 * test_season + 30
        ),
        "lineup_depm_total_sq_geometry_identity": (
            model_seed + 100 * test_season + 30
        ),
    }
    predictions: Dict[str, np.ndarray] = {}
    for arm, columns in arm_columns.items():
        model = new_model(arm_seeds[arm])
        model.fit(
            base_features.loc[train_mask, columns],
            train["made"].to_numpy(dtype=int),
        )
        predictions[arm] = model.predict_proba(
            base_features.loc[test_mask, columns]
        )[:, 1]
    calibrated_initial, calibration = platt_calibrate(
        train["initial_make_probability"].to_numpy(dtype=float),
        train["made"].to_numpy(dtype=int),
        test["initial_make_probability"].to_numpy(dtype=float),
    )

    predictions = {
        "initial_ev_raw": test["initial_make_probability"].to_numpy(dtype=float),
        "initial_ev_calibrated": calibrated_initial,
        **predictions,
    }
    slices = {
        "all_shots": np.ones(len(test), dtype=bool),
        "lineup_epm_complete": test["def_lineup_depm_coverage"]
        .fillna(0)
        .eq(5)
        .to_numpy(dtype=bool),
        "nba_assignment_available": test[
            "nba_assignment_available"
        ].to_numpy(dtype=bool),
        "both_providers_available": (
            test["nba_assignment_available"].to_numpy(dtype=bool)
            & test["defender_geometry_available"].to_numpy(dtype=bool)
        ),
    }
    metrics: List[Dict[str, object]] = []
    intervals: List[Dict[str, object]] = []
    comparisons = (
        ("nba_assignment", "context_epm"),
        ("lineup_depm_total", "context_epm"),
        ("lineup_depm", "context_epm"),
        ("lineup_depm_nba_assignment", "lineup_depm"),
        ("lineup_depm_nba_assignment", "nba_assignment"),
        ("sq_geometry", "context_epm"),
        ("lineup_depm_sq_geometry", "sq_geometry"),
        ("lineup_depm_sq_geometry", "lineup_depm"),
        (
            "lineup_depm_sq_geometry_identity",
            "sq_geometry_identity",
        ),
        (
            "lineup_depm_sq_geometry_identity",
            "lineup_depm_sq_geometry",
        ),
        (
            "lineup_depm_sq_geometry_identity",
            "initial_ev_raw",
        ),
        (
            "lineup_depm_total_sq_geometry_identity",
            "sq_geometry_identity",
        ),
        (
            "lineup_depm_total_sq_geometry_identity",
            "lineup_depm_sq_geometry_identity",
        ),
        (
            "lineup_depm_total_sq_geometry_identity",
            "initial_ev_raw",
        ),
        ("nba_assignment", "sq_geometry"),
        ("nba_assignment", "sq_geometry_identity"),
        ("nba_assignment", "initial_ev_calibrated"),
        ("nba_assignment", "initial_ev_raw"),
        ("sq_geometry", "initial_ev_calibrated"),
        ("sq_geometry", "initial_ev_raw"),
        ("sq_geometry_identity", "sq_geometry"),
        ("sq_geometry_identity", "initial_ev_calibrated"),
        ("sq_geometry_identity", "initial_ev_raw"),
        ("initial_ev_calibrated", "initial_ev_raw"),
    )
    test = test.reset_index(drop=True)
    for slice_name, slice_mask in slices.items():
        sliced = test.loc[slice_mask].reset_index(drop=True)
        if sliced.empty:
            continue
        for model_name, probability in predictions.items():
            metrics.append(
                comparison_metric_row(
                    sliced,
                    probability[slice_mask],
                    model=model_name,
                    coverage_slice=slice_name,
                )
            )
        for challenger_name, reference_name in comparisons:
            for metric in ("points_mse", "log_loss", "brier"):
                result = clustered_delta_interval(
                    sliced,
                    predictions[challenger_name][slice_mask],
                    predictions[reference_name][slice_mask],
                    metric=metric,
                    iterations=bootstrap_iterations,
                    seed=seed + len(intervals),
                )
                intervals.append(
                    {
                        "test_season": test_season,
                        "segment": segment,
                        "coverage_slice": slice_name,
                        "challenger": challenger_name,
                        "reference": reference_name,
                        "metric": metric,
                        **result,
                    }
                )

    output_columns = [
        "game_id_normalized",
        "GAME_EVENT_ID",
        "game_date",
        "season",
        "segment",
        "personId",
        "made",
        "shot_value",
        "nba_assignment_available",
        "nba_assignment_count",
        "defender_geometry_available",
        "def_lineup_available",
        "def_lineup_depm_coverage",
        "def_lineup_depm_sum",
    ]
    output = test[output_columns].copy()
    for model_name, probability in predictions.items():
        output[f"p_{model_name}"] = probability
        output[f"pps_{model_name}"] = probability * output["shot_value"]
    return (
        pd.DataFrame(metrics),
        pd.DataFrame(intervals),
        output,
        calibration,
    )


def write_outputs(
    output_dir: Path,
    *,
    metrics: pd.DataFrame,
    intervals: pd.DataFrame,
    predictions: pd.DataFrame,
    manifest: Mapping[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    intervals.to_csv(output_dir / "intervals.csv", index=False)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    (output_dir / "manifest.json").write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "/Users/russ/Documents/Codex/basketball/nba/research/vpm/"
            "vpm/data/sq_shots_keyed.parquet"
        ),
    )
    parser.add_argument("--processed-dir", type=Path, default=PIPELINE_ROOT / "processed")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--epm-cache", type=Path, default=PIPELINE_ROOT / "cache" / "epm")
    parser.add_argument("--matchup-root", type=Path, default=DEFAULT_MATCHUP_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-season", type=int, default=2025)
    parser.add_argument("--segment", choices=["mid", "three"], default="three")
    parser.add_argument("--shooter-shrink", type=float, default=100.0)
    parser.add_argument("--sq-defender-shrink", type=float, default=250.0)
    parser.add_argument("--nba-defender-shrink", type=float, default=250.0)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument(
        "--model-seed",
        type=int,
        default=73,
        help="Seed used by the source SQ walk-forward run.",
    )
    parser.add_argument("--seed", type=int, default=113)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shots, shot_audit = prepare_shots(
        args.source,
        processed_dir=args.processed_dir,
        epm_cache=args.epm_cache,
        shooter_shrink=args.shooter_shrink,
        defender_shrink=args.sq_defender_shrink,
    )
    needed_seasons = sorted(
        shots.loc[shots["season"].le(args.test_season), "season"].unique()
    )
    event_lineups, event_lineup_source_audit = load_event_lineups(
        args.raw_dir,
        seasons=needed_seasons,
    )
    shots, event_lineup_join_audit = attach_event_lineups(
        shots,
        event_lineups,
    )
    shots, lineup_epm_audit = attach_lineup_depm_features(
        shots,
        load_lineup_epm_snapshots(args.epm_cache),
    )
    assignments, provider_audit = load_nba_assignments(
        args.matchup_root,
        seasons=needed_seasons,
    )
    shots, join_audit = attach_nba_assignment_priors(
        shots,
        assignments,
        shrink_attempts=args.nba_defender_shrink,
        baseline_by_segment={"mid": 0.44, "three": 0.36},
    )
    metrics, intervals, predictions, calibration = run_comparison(
        shots,
        test_season=args.test_season,
        segment=args.segment,
        bootstrap_iterations=args.bootstrap_iterations,
        model_seed=args.model_seed,
        seed=args.seed,
    )
    manifest = {
        "schema_version": 2,
        "source": str(args.source),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "matchup_root": str(args.matchup_root),
        "matchup_commit": (
            __import__("subprocess")
            .check_output(
                ["git", "-C", str(args.matchup_root), "rev-parse", "HEAD"],
                text=True,
            )
            .strip()
        ),
        "test_season": args.test_season,
        "segment": args.segment,
        "nba_defender_shrink_attempts": args.nba_defender_shrink,
        "lineup_depm_features": list(LINEUP_DEPM_FEATURES),
        "lineup_depm_total_features": list(LINEUP_DEPM_TOTAL_FEATURES),
        "bootstrap_iterations": args.bootstrap_iterations,
        "model_seed": args.model_seed,
        "seed": args.seed,
        "platt_calibration": calibration,
        "shot_audit": dict(shot_audit),
        "event_lineup_source_audit": dict(event_lineup_source_audit),
        "event_lineup_join_audit": dict(event_lineup_join_audit),
        "lineup_epm_audit": dict(lineup_epm_audit),
        "provider_audit": dict(provider_audit),
        "join_audit": dict(join_audit),
        "validation_contract": (
            "Model and calibration parameters train only on seasons before the "
            "test season. NBA defender outcome priors exclude the whole current "
            "game and preserve every pipe-separated provider assignment. "
            "Lineup D-EPM uses only snapshots requested on or before game date."
        ),
        "provider_contract": (
            "NBA matchup DEF_ID is an assignment, not physical closest-defender "
            "distance. SQ geometry remains a separate provider feature family."
        ),
        "target_contract": (
            "Immediate made-field-goal points per shot only. No OREB or "
            "second-chance continuation value enters this comparison."
        ),
    }
    write_outputs(
        args.output_dir,
        metrics=metrics,
        intervals=intervals,
        predictions=predictions,
        manifest=manifest,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(metrics.to_string(index=False))
    print(intervals.to_string(index=False))


if __name__ == "__main__":
    main()
