#!/usr/bin/env python3
"""Season-forward Shot Quality test with EPM and closest-defender features.

For test season S, model coefficients are fit only on seasons before S.
Shooter and assigned-defender empirical-Bayes priors are updated by completed
games and are shifted so the current game's outcomes never enter its features.

This is an isolated research challenger. It does not change RUSSELL_SHOTQUALITY
or the active public DECOMP outputs.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


DEFAULT_SOURCE = Path(
    "/Users/russ/Documents/Codex/basketball/nba/research/vpm/"
    "vpm/data/sq_shots_keyed.parquet"
)
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED = PIPELINE_ROOT / "processed"
DEFAULT_EPM_CACHE = PIPELINE_ROOT / "cache" / "epm"
DEFAULT_OUTPUT = PIPELINE_ROOT / "results" / "sq_walk_forward_challenger"

DESCRIPTOR_TAGS = (
    "alley_oop",
    "catch_and_shoot",
    "dunk",
    "floater",
    "hook",
    "in_paint",
    "jumpshot",
    "lay_up",
    "off_cut",
    "off_drive",
    "pick_and_roll",
    "post_up",
    "pull_up",
    "reverse",
    "step_back",
    "tip_in",
    "transition",
    "turnaround",
)

BASE_NUMERIC_FEATURES = (
    "period",
    "minutes",
    "seconds",
    "shot_x",
    "shot_y",
    "spacing",
    "feature_store.Shot Distance (ft.)",
    "shooter_prior_make_probability",
    "shooter_prior_log_attempts",
)

EPM_FEATURES = (
    "epm_shooter_make_probability",
    "epm_shooter_attempts_100",
    "epm_snapshot_age_days",
)

DEFENDER_GEOMETRY_FEATURES = (
    "contest_score",
    "feature_store.Defenders in Paint",
    "feature_store.Closest Defender Behind",
    "feature_store.Closest Defender Inside",
    "feature_store.Closest Defender Blocking",
    "feature_store.Second-Closest Defender Behind",
    "feature_store.Harmonic Mean Defender Distance",
    "feature_store.Average Defender Distance (Feet)",
    "feature_store.Second-Closest Defender Blocking",
    "feature_store.Distance to the Closest Defender (Feet)",
    "feature_store.Height of the Closest Defender (Inches)",
    "feature_store.Number of Defenders Closer to the Basket",
    "feature_store.Cosine of the Angle with the Closest Defender",
    "feature_store.Distance to the Second-Closest Defender (Feet)",
    "feature_store.Cosine of the Angle with the Second-Closest Defender",
    "feature_store.Difference in Bounding Box Height with the Closest Defender (Pixels)",
    "defender_geometry_available",
)

DEFENDER_IDENTITY_FEATURES = (
    "defender_prior_make_probability_allowed",
    "defender_prior_log_attempts",
    "primary_defender_identity_available",
)

MODEL_FEATURES = {
    "context_internal": BASE_NUMERIC_FEATURES,
    "context_epm": BASE_NUMERIC_FEATURES + EPM_FEATURES,
    "context_epm_def_geometry": (
        BASE_NUMERIC_FEATURES + EPM_FEATURES + DEFENDER_GEOMETRY_FEATURES
    ),
    "context_epm_def_geometry_identity": (
        BASE_NUMERIC_FEATURES
        + EPM_FEATURES
        + DEFENDER_GEOMETRY_FEATURES
        + DEFENDER_IDENTITY_FEATURES
    ),
}


def normalize_game_id(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.lstrip("0")


def resolve_shot_rows(
    frame: pd.DataFrame,
    *,
    keys: Sequence[str] = ("game_id", "GAME_EVENT_ID"),
) -> Tuple[pd.DataFrame, Mapping[str, int]]:
    """Collapse exact duplicates and exclude conflicting provider event keys."""
    missing = sorted(set(keys) - set(frame.columns))
    if missing:
        raise ValueError(f"Shot table is missing event keys: {missing}")
    original_rows = len(frame)
    missing_key_mask = frame[list(keys)].isna().any(axis=1)
    missing_key_rows = int(missing_key_mask.sum())
    keyed_frame = frame.loc[~missing_key_mask].copy()
    exact = keyed_frame.drop_duplicates().copy()
    distinct_counts = exact.groupby(list(keys), dropna=False).size()
    conflicting_keys = distinct_counts[distinct_counts > 1].index
    if len(conflicting_keys):
        keyed = pd.MultiIndex.from_frame(exact[list(keys)])
        conflict_mask = keyed.isin(conflicting_keys)
        conflict_rows = int(conflict_mask.sum())
        exact = exact.loc[~conflict_mask].copy()
    else:
        conflict_rows = 0
    if exact.duplicated(list(keys)).any():
        raise AssertionError("Resolved shot rows are not unique by provider event key")
    audit = {
        "source_rows": int(original_rows),
        "missing_event_key_rows_removed": missing_key_rows,
        "exact_duplicate_rows_removed": int(
            len(keyed_frame) - len(keyed_frame.drop_duplicates())
        ),
        "conflicting_event_keys_removed": int(len(conflicting_keys)),
        "conflicting_rows_removed": conflict_rows,
        "resolved_rows": int(len(exact)),
    }
    return exact.reset_index(drop=True), audit


def load_game_dates(processed_dir: Path, seasons: Iterable[int]) -> pd.DataFrame:
    mappings: List[pd.DataFrame] = []
    for season in sorted(set(int(value) for value in seasons)):
        suffix = str(season)[-2:]
        for postseason in ("", "_PS"):
            path = processed_dir / f"FIRST_CHANCE{suffix}{postseason}.parquet"
            if not path.exists():
                continue
            mapping = pd.read_parquet(path, columns=["game_id", "game_date"])
            mapping = mapping.drop_duplicates("game_id")
            mapping["game_id_normalized"] = normalize_game_id(mapping["game_id"])
            mappings.append(mapping[["game_id_normalized", "game_date"]])
    if not mappings:
        raise FileNotFoundError("No FIRST_CHANCE game-date parquets were found")
    out = pd.concat(mappings, ignore_index=True)
    conflicts = out.groupby("game_id_normalized")["game_date"].nunique(dropna=True)
    if (conflicts > 1).any():
        raise ValueError("Conflicting game dates found for normalized game ids")
    out = out.drop_duplicates("game_id_normalized")
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    return out


def classify_segment(frame: pd.DataFrame) -> pd.Series:
    action = frame["action_type"].fillna("").astype(str).str.lower()
    descriptors = frame["play_descriptors"].fillna("").astype(str).str.lower()
    distance = pd.to_numeric(
        frame["feature_store.Shot Distance (ft.)"], errors="coerce"
    )
    rim_text = descriptors.str.contains(
        r"dunk|lay_up|tip_in|alley_oop", regex=True
    )
    return pd.Series(
        np.where(
            action.eq("3pt"),
            "three",
            np.where(distance.le(4.5) | rim_text, "rim", "mid"),
        ),
        index=frame.index,
    )


def add_game_shifted_prior(
    frame: pd.DataFrame,
    *,
    player_column: str,
    prefix: str,
    shrink_attempts: float,
    baseline_by_segment: Mapping[str, float],
) -> pd.DataFrame:
    """Add prior-game-only player/segment make rates to every shot row."""
    out = frame.copy()
    player = pd.to_numeric(out[player_column], errors="coerce").astype("Int64")
    out[f"_{prefix}_id"] = player
    valid = player.notna() & player.gt(0)
    games = (
        out.loc[valid]
        .groupby(
            [f"_{prefix}_id", "segment", "game_date", "game_id_normalized"],
            dropna=False,
            sort=True,
        )
        .agg(game_makes=("made", "sum"), game_attempts=("made", "size"))
        .reset_index()
        .sort_values(
            [f"_{prefix}_id", "segment", "game_date", "game_id_normalized"],
            kind="stable",
        )
    )
    grouped = games.groupby([f"_{prefix}_id", "segment"], sort=False)
    games["prior_makes"] = grouped["game_makes"].cumsum() - games["game_makes"]
    games["prior_attempts"] = (
        grouped["game_attempts"].cumsum() - games["game_attempts"]
    )
    games["baseline"] = games["segment"].map(baseline_by_segment).astype(float)
    games[f"{prefix}_prior_make_probability"] = (
        games["prior_makes"] + float(shrink_attempts) * games["baseline"]
    ) / (games["prior_attempts"] + float(shrink_attempts))
    games[f"{prefix}_prior_log_attempts"] = np.log1p(games["prior_attempts"])

    join_columns = [
        f"_{prefix}_id",
        "segment",
        "game_date",
        "game_id_normalized",
    ]
    value_columns = [
        f"{prefix}_prior_make_probability",
        f"{prefix}_prior_log_attempts",
    ]
    out = out.merge(
        games[join_columns + value_columns],
        on=join_columns,
        how="left",
        validate="many_to_one",
        sort=False,
    )
    baseline = out["segment"].map(baseline_by_segment).astype(float)
    out[f"{prefix}_prior_make_probability"] = out[
        f"{prefix}_prior_make_probability"
    ].fillna(baseline)
    out[f"{prefix}_prior_log_attempts"] = out[
        f"{prefix}_prior_log_attempts"
    ].fillna(0.0)
    return out.drop(columns=[f"_{prefix}_id"])


def load_epm_snapshots(cache_dir: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(cache_dir.glob("epm_all_*_go0.csv")):
        parts = path.stem.split("_")
        if len(parts) < 4:
            continue
        requested_date = pd.to_datetime(parts[2], errors="coerce")
        if pd.isna(requested_date):
            continue
        available = pd.read_csv(path, nrows=0).columns
        columns = [
            column
            for column in (
                "player_id",
                "p_fg3pct",
                "p_fgpct_mid",
                "p_fg3a_100",
                "p_fga_mid_100",
            )
            if column in available
        ]
        if len(columns) < 5:
            continue
        snap = pd.read_csv(path, usecols=columns)
        snap["snapshot_date"] = requested_date
        frames.append(snap)
    if not frames:
        raise FileNotFoundError(f"No usable EPM snapshots found under {cache_dir}")
    out = pd.concat(frames, ignore_index=True)
    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").astype(
        "Int64"
    )
    out = out.dropna(subset=["player_id"])
    conflicts = out.duplicated(["player_id", "snapshot_date"], keep=False)
    if conflicts.any():
        deduped = out.drop_duplicates()
        if deduped.duplicated(["player_id", "snapshot_date"]).any():
            raise ValueError("Conflicting EPM player/date rows found")
        out = deduped
    return out.sort_values(["snapshot_date", "player_id"], kind="stable")


def attach_epm_features(frame: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["_epm_player_id"] = pd.to_numeric(
        out["personId"], errors="coerce"
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
    is_three = merged["segment"].eq("three")
    merged["epm_shooter_make_probability"] = np.where(
        is_three, merged["p_fg3pct"], merged["p_fgpct_mid"]
    )
    merged["epm_shooter_attempts_100"] = np.where(
        is_three, merged["p_fg3a_100"], merged["p_fga_mid_100"]
    )
    unsupported = pd.to_numeric(
        merged["epm_shooter_attempts_100"], errors="coerce"
    ).fillna(0.0).lt(0.5)
    merged.loc[unsupported, "epm_shooter_make_probability"] = np.nan
    merged["epm_snapshot_age_days"] = (
        merged["game_date"] - merged["snapshot_date"]
    ).dt.days
    return (
        merged.sort_values("_row_order", kind="stable")
        .drop(columns=["_epm_player_id", "_row_order"])
        .reset_index(drop=True)
    )


def descriptor_features(values: pd.Series) -> pd.DataFrame:
    def parse(value) -> set:
        if pd.isna(value):
            return set()
        if isinstance(value, (list, tuple, set)):
            return {str(item).strip().lower() for item in value}
        try:
            parsed = ast.literal_eval(str(value))
            if isinstance(parsed, (list, tuple, set)):
                return {str(item).strip().lower() for item in parsed}
        except (SyntaxError, ValueError):
            pass
        return {
            token.strip().lower()
            for token in str(value).replace("|", ",").split(",")
            if token.strip()
        }

    parsed = values.map(parse)
    return pd.DataFrame(
        {
            f"descriptor::{tag}": parsed.map(lambda tags, tag=tag: tag in tags).astype(
                float
            )
            for tag in DESCRIPTOR_TAGS
        },
        index=values.index,
    )


def build_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = sorted(
        {
            *BASE_NUMERIC_FEATURES,
            *EPM_FEATURES,
            *DEFENDER_GEOMETRY_FEATURES,
            *DEFENDER_IDENTITY_FEATURES,
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


def binary_log_loss(target: np.ndarray, probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return -(
        target * np.log(probability)
        + (1.0 - target) * np.log(1.0 - probability)
    )


def metric_row(
    frame: pd.DataFrame,
    probability: np.ndarray,
    *,
    model: str,
    coverage_slice: str,
) -> Dict[str, object]:
    target = frame["made"].to_numpy(dtype=float)
    shot_value = frame["shot_value"].to_numpy(dtype=float)
    actual_points = target * shot_value
    predicted_points = np.asarray(probability, dtype=float) * shot_value
    return {
        "test_season": int(frame["season"].iloc[0]),
        "segment": str(frame["segment"].iloc[0]),
        "coverage_slice": coverage_slice,
        "model": model,
        "shots": int(len(frame)),
        "games": int(frame["game_id_normalized"].nunique()),
        "actual_pps": float(actual_points.mean()),
        "predicted_pps": float(predicted_points.mean()),
        "pps_bias": float(predicted_points.mean() - actual_points.mean()),
        "points_mse": float(np.mean((actual_points - predicted_points) ** 2)),
        "log_loss": float(binary_log_loss(target, probability).mean()),
        "brier": float(np.mean((target - probability) ** 2)),
    }


def clustered_delta_interval(
    frame: pd.DataFrame,
    challenger: np.ndarray,
    reference: np.ndarray,
    *,
    metric: str,
    iterations: int,
    seed: int,
) -> Dict[str, float]:
    target = frame["made"].to_numpy(dtype=float)
    shot_value = frame["shot_value"].to_numpy(dtype=float)
    if metric == "points_mse":
        actual = target * shot_value
        delta = (actual - challenger * shot_value) ** 2 - (
            actual - reference * shot_value
        ) ** 2
    elif metric == "log_loss":
        delta = binary_log_loss(target, challenger) - binary_log_loss(
            target, reference
        )
    elif metric == "brier":
        delta = (target - challenger) ** 2 - (target - reference) ** 2
    else:
        raise ValueError(f"Unsupported interval metric: {metric}")
    grouped = (
        pd.DataFrame(
            {
                "game_id": frame["game_id_normalized"].to_numpy(),
                "delta": delta,
            }
        )
        .groupby("game_id")
        .agg(delta_sum=("delta", "sum"), rows=("delta", "size"))
    )
    rng = np.random.default_rng(seed)
    values = grouped[["delta_sum", "rows"]].to_numpy(dtype=float)
    samples = np.empty(iterations, dtype=float)
    for index in range(iterations):
        draw = rng.integers(0, len(values), size=len(values))
        selected = values[draw]
        samples[index] = selected[:, 0].sum() / selected[:, 1].sum()
    return {
        "delta": float(delta.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "probability_delta_below_zero": float(np.mean(samples < 0.0)),
    }


def prepare_shots(
    source: Path,
    *,
    processed_dir: Path,
    epm_cache: Path,
    shooter_shrink: float,
    defender_shrink: float,
) -> Tuple[pd.DataFrame, Mapping[str, object]]:
    frame = pd.read_parquet(source)
    frame, duplicate_audit = resolve_shot_rows(frame)
    frame["game_id_normalized"] = normalize_game_id(frame["game_id"])
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["season", "made"]).copy()
    frame["season"] = frame["season"].astype(int)
    frame["made"] = frame["made"].astype(bool).astype(int)
    frame["segment"] = classify_segment(frame)
    frame = frame[frame["segment"].isin(["mid", "three"])].copy()
    frame["shot_value"] = np.where(frame["segment"].eq("three"), 3.0, 2.0)
    frame["initial_make_probability"] = (
        pd.to_numeric(frame["ev.initial_ev.value"], errors="coerce")
        / frame["shot_value"]
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
        missing = int(frame["game_date"].isna().sum())
        raise ValueError(f"Missing game dates for {missing} shot rows")
    frame = frame.sort_values(
        ["game_date", "game_id_normalized", "period", "minutes", "seconds"],
        ascending=[True, True, True, False, False],
        kind="stable",
    ).reset_index(drop=True)

    baseline = {"mid": 0.44, "three": 0.36}
    frame = add_game_shifted_prior(
        frame,
        player_column="personId",
        prefix="shooter",
        shrink_attempts=shooter_shrink,
        baseline_by_segment=baseline,
    )
    frame = add_game_shifted_prior(
        frame,
        player_column="opp_personId",
        prefix="defender",
        shrink_attempts=defender_shrink,
        baseline_by_segment=baseline,
    )
    frame = frame.rename(
        columns={
            "defender_prior_make_probability": (
                "defender_prior_make_probability_allowed"
            )
        }
    )
    frame["primary_defender_identity_available"] = (
        pd.to_numeric(frame["opp_personId"], errors="coerce").notna().astype(float)
    )
    closest_distance = pd.to_numeric(
        frame["feature_store.Distance to the Closest Defender (Feet)"],
        errors="coerce",
    )
    frame["defender_geometry_available"] = closest_distance.notna().astype(float)
    frame = attach_epm_features(frame, load_epm_snapshots(epm_cache))

    audit = {
        **duplicate_audit,
        "model_rows": int(len(frame)),
        "games": int(frame["game_id_normalized"].nunique()),
        "seasons": sorted(frame["season"].unique().astype(int).tolist()),
        "epm_coverage": float(
            frame["epm_shooter_make_probability"].notna().mean()
        ),
        "defender_identity_coverage": float(
            frame["primary_defender_identity_available"].mean()
        ),
        "defender_geometry_coverage": float(
            frame["defender_geometry_available"].mean()
        ),
    }
    return frame, audit


def run_walk_forward(
    frame: pd.DataFrame,
    *,
    test_seasons: Sequence[int],
    bootstrap_iterations: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = build_feature_frame(frame)
    descriptor_columns = [
        column for column in features.columns if column.startswith("descriptor::")
    ]
    metrics: List[Dict[str, object]] = []
    intervals: List[Dict[str, object]] = []
    prediction_frames: List[pd.DataFrame] = []

    for test_season in test_seasons:
        for segment in ("mid", "three"):
            train_mask = frame["season"].lt(test_season) & frame["segment"].eq(segment)
            test_mask = frame["season"].eq(test_season) & frame["segment"].eq(segment)
            train = frame.loc[train_mask]
            test = frame.loc[test_mask]
            if train.empty or test.empty:
                continue
            predictions: Dict[str, np.ndarray] = {
                "initial_ev": test["initial_make_probability"].to_numpy(dtype=float)
            }
            for arm_index, (arm, columns) in enumerate(MODEL_FEATURES.items()):
                print(
                    f"fit test_season={test_season} segment={segment} arm={arm} "
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

            output = test[
                [
                    "game_id_normalized",
                    "GAME_EVENT_ID",
                    "game_date",
                    "season",
                    "segment",
                    "personId",
                    "opp_personId",
                    "made",
                    "shot_value",
                    "defender_geometry_available",
                    "primary_defender_identity_available",
                    "epm_shooter_make_probability",
                ]
            ].copy()
            for model_name, probability in predictions.items():
                output[f"p_{model_name}"] = probability
                output[f"pps_{model_name}"] = probability * output["shot_value"]
            prediction_frames.append(output)

            slices = {
                "all_shots": np.ones(len(test), dtype=bool),
                "geometry_available": test[
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
                comparisons = (
                    ("context_epm", "context_internal"),
                    ("context_epm_def_geometry", "context_epm"),
                    (
                        "context_epm_def_geometry_identity",
                        "context_epm_def_geometry",
                    ),
                    ("context_epm_def_geometry_identity", "context_epm"),
                    ("context_epm_def_geometry_identity", "initial_ev"),
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
                            seed=seed
                            + test_season
                            + len(intervals)
                            + len(challenger_name),
                        )
                        intervals.append(
                            {
                                "test_season": int(test_season),
                                "segment": segment,
                                "coverage_slice": slice_name,
                                "challenger": challenger_name,
                                "reference": reference_name,
                                "metric": metric,
                                **result,
                            }
                        )
    if not prediction_frames:
        raise ValueError("No test-season predictions were produced")
    return (
        pd.DataFrame(metrics),
        pd.DataFrame(intervals),
        pd.concat(prediction_frames, ignore_index=True),
    )


def write_outputs(
    output_dir: Path,
    *,
    metrics: pd.DataFrame,
    intervals: pd.DataFrame,
    predictions: pd.DataFrame,
    audit: Mapping[str, object],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "walk_forward_metrics.csv", index=False)
    intervals.to_csv(output_dir / "walk_forward_intervals.csv", index=False)
    predictions.to_parquet(
        output_dir / "walk_forward_predictions.parquet", index=False
    )
    manifest = {
        "schema_version": 1,
        "source": str(args.source),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "test_seasons": list(args.test_seasons),
        "shooter_shrink_attempts": args.shooter_shrink,
        "defender_shrink_attempts": args.defender_shrink,
        "bootstrap_iterations": args.bootstrap_iterations,
        "seed": args.seed,
        "validation_contract": (
            "Test season S is scored by models fit only on seasons < S; "
            "shooter/defender outcome priors exclude the whole current game."
        ),
        "provider_contract": (
            "opp_personId and geometry are ShotQuality-provider fields; "
            "NBA matchup defenders are not coalesced into them."
        ),
        "model_features": {
            key: list(value) + [f"descriptor::{tag}" for tag in DESCRIPTOR_TAGS]
            for key, value in MODEL_FEATURES.items()
        },
        "data_audit": dict(audit),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--epm-cache", type=Path, default=DEFAULT_EPM_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-seasons", type=int, nargs="+", default=[2025, 2026])
    parser.add_argument("--shooter-shrink", type=float, default=100.0)
    parser.add_argument("--defender-shrink", type=float, default=250.0)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=73)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shots, audit = prepare_shots(
        args.source,
        processed_dir=args.processed_dir,
        epm_cache=args.epm_cache,
        shooter_shrink=args.shooter_shrink,
        defender_shrink=args.defender_shrink,
    )
    metrics, intervals, predictions = run_walk_forward(
        shots,
        test_seasons=args.test_seasons,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    write_outputs(
        args.output_dir,
        metrics=metrics,
        intervals=intervals,
        predictions=predictions,
        audit=audit,
        args=args,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(
        metrics[
            [
                "test_season",
                "segment",
                "coverage_slice",
                "model",
                "shots",
                "actual_pps",
                "predicted_pps",
                "points_mse",
                "log_loss",
                "brier",
            ]
        ].to_string(index=False)
    )
    print(
        intervals[
            [
                "test_season",
                "segment",
                "coverage_slice",
                "challenger",
                "metric",
                "delta",
                "ci_low",
                "ci_high",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
