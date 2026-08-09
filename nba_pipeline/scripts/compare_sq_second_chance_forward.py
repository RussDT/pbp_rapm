#!/usr/bin/env python3
"""Forward-test shooter-specific second-chance value over shot-quality models.

The benchmark fits the existing interpretable OREB opportunity model on
seasons strictly before the test season, then scores the first field-goal
attempt of each test-season possession.  The prediction target is deliberately
narrow:

    points on the initial FGA + SECOND_CHANCE_CLEAN points after its miss

First-chance free throws are excluded so the nested comparison isolates the
miss/rebound continuation branch.  This is an RAPM research artifact; it does
not modify canonical RUSSELL_SHOTQUALITY or active public DECOMP.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from nba_pipeline.scripts.build_second_chance_opportunity_prior import (
    CONTEXT_COLS,
    DEFENDER_COLS,
    NON_SHOOTER_COLS,
    _coefficient_maps,
    build_design_matrix,
    build_feature_space,
    load_opportunities,
    normalize_game_id,
    role_swap_counterfactual,
)
from nba_pipeline.scripts.process_rapm_blocks.common import (
    sort_raw_pbp_chronologically,
)


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = (
    PIPELINE_ROOT
    / "results"
    / "sq_walk_forward_challenger"
    / "walk_forward_predictions.parquet"
)
DEFAULT_REPO_SQ = (
    PIPELINE_ROOT
    / "results"
    / "pbp_shot_quality"
    / "pbp_shot_quality_1997_2026.parquet"
)
DEFAULT_OUTPUT = (
    PIPELINE_ROOT / "results" / "sq_second_chance_forward" / "2025"
)

IMMEDIATE_MODELS = {
    "initial_ev": "p_initial_ev",
    "context_epm": "p_context_epm",
    "geometry_identity": "p_context_epm_def_geometry_identity",
}


def release_unused_memory() -> None:
    """Return temporary parquet buffers before loading the next large slice."""
    gc.collect()
    try:
        import pyarrow as pa

        pa.default_memory_pool().release_unused()
    except (ImportError, AttributeError):
        pass


def build_release_map_from_frames(
    raw: pd.DataFrame,
    terminals: pd.DataFrame,
    second_chance: pd.DataFrame,
) -> Tuple[pd.DataFrame, Mapping[str, int]]:
    """Map every FGA to its terminal standard-possession row."""
    required_raw = {
        "game_id",
        "event_num",
        "event_type",
        "period",
        "minute_remaining_quarter",
        "seconds_remaining_quarter",
    }
    missing = sorted(required_raw - set(raw.columns))
    if missing:
        raise ValueError(f"Raw release-map input missing columns: {missing}")
    if not {"game_id", "event_num"}.issubset(terminals.columns):
        raise ValueError("Terminal input requires game_id and event_num")
    if not {"game_id", "event_num", "Off_Diff"}.issubset(
        second_chance.columns
    ):
        raise ValueError(
            "Second-chance input requires game_id, event_num, and Off_Diff"
        )

    events = raw.drop_duplicates().copy()
    events["game_id"] = normalize_game_id(events["game_id"])
    events["event_num"] = pd.to_numeric(
        events["event_num"], errors="coerce"
    ).astype("Int64")
    events = events.dropna(subset=["event_num"]).copy()
    events["event_num"] = events["event_num"].astype(int)
    events = sort_raw_pbp_chronologically(events)

    terminal = terminals[["game_id", "event_num"]].copy()
    terminal["game_id"] = normalize_game_id(terminal["game_id"])
    terminal["event_num"] = pd.to_numeric(
        terminal["event_num"], errors="coerce"
    ).astype("Int64")
    terminal = terminal.dropna().drop_duplicates()
    terminal["event_num"] = terminal["event_num"].astype(int)

    event_keys = pd.MultiIndex.from_frame(events[["game_id", "event_num"]])
    terminal_keys = pd.MultiIndex.from_frame(
        terminal[["game_id", "event_num"]]
    )
    events["is_terminal"] = event_keys.isin(terminal_keys)
    events["terminal_event_num"] = (
        events["event_num"]
        .where(events["is_terminal"])
        .groupby(events["game_id"], sort=False)
        .bfill()
    )
    events["is_fga"] = pd.to_numeric(
        events["event_type"], errors="coerce"
    ).isin([1, 2])
    shots = events[
        events["is_fga"] & events["terminal_event_num"].notna()
    ].copy()
    shots["terminal_event_num"] = shots["terminal_event_num"].astype(int)
    shots["fga_rank"] = shots.groupby(
        ["game_id", "terminal_event_num"], sort=False
    ).cumcount()

    continuation = second_chance[
        ["game_id", "event_num", "Off_Diff"]
    ].copy()
    continuation["game_id"] = normalize_game_id(continuation["game_id"])
    continuation["terminal_event_num"] = pd.to_numeric(
        continuation.pop("event_num"), errors="coerce"
    ).astype("Int64")
    continuation["realized_second_chance_points"] = pd.to_numeric(
        continuation.pop("Off_Diff"), errors="coerce"
    )
    continuation = continuation.dropna(
        subset=["terminal_event_num", "realized_second_chance_points"]
    )
    continuation["terminal_event_num"] = continuation[
        "terminal_event_num"
    ].astype(int)
    if continuation.duplicated(
        ["game_id", "terminal_event_num"]
    ).any():
        raise ValueError("Second-chance terminal keys are not unique")

    shots = shots.merge(
        continuation,
        on=["game_id", "terminal_event_num"],
        how="left",
        validate="many_to_one",
    )
    shots["realized_second_chance_points"] = shots[
        "realized_second_chance_points"
    ].fillna(0.0)
    output = shots[
        [
            "game_id",
            "event_num",
            "terminal_event_num",
            "fga_rank",
            "realized_second_chance_points",
        ]
    ].drop_duplicates(["game_id", "event_num"])
    audit = {
        "raw_rows": int(len(events)),
        "terminal_rows": int(len(terminal)),
        "terminal_rows_found": int(events["is_terminal"].sum()),
        "mapped_fga_rows": int(len(output)),
        "first_fga_rows": int(output["fga_rank"].eq(0).sum()),
    }
    return output.reset_index(drop=True), audit


def load_release_maps(
    raw_dir: Path,
    processed_dir: Path,
    *,
    years: Iterable[int],
    season_type: str,
) -> Tuple[pd.DataFrame, Sequence[Mapping[str, object]]]:
    frames: List[pd.DataFrame] = []
    audits: List[Mapping[str, object]] = []
    samples = (("", "RS"), ("_PS", "PS"))
    requested_type = str(season_type).upper()
    for year in sorted({int(value) for value in years}):
        suffix = str(year)[-2:]
        for postseason_suffix, sample in samples:
            if requested_type != "ALL" and sample != requested_type:
                continue
            raw_path = raw_dir / f"NBA{suffix}{postseason_suffix}.parquet"
            terminal_path = (
                processed_dir / f"RAPM{suffix}{postseason_suffix}.parquet"
            )
            second_chance_path = (
                processed_dir
                / f"SECOND_CHANCE_CLEAN{suffix}{postseason_suffix}.parquet"
            )
            if not (
                raw_path.exists()
                and terminal_path.exists()
                and second_chance_path.exists()
            ):
                continue
            raw = pd.read_parquet(
                raw_path,
                columns=[
                    "game_id",
                    "event_num",
                    "event_type",
                    "period",
                    "minute_remaining_quarter",
                    "seconds_remaining_quarter",
                ],
            )
            terminals = pd.read_parquet(
                terminal_path, columns=["game_id", "event_num"]
            )
            second_chance = pd.read_parquet(
                second_chance_path,
                columns=["game_id", "event_num", "Off_Diff"],
            )
            release_map, audit = build_release_map_from_frames(
                raw, terminals, second_chance
            )
            release_map["season_end_year"] = int(year)
            release_map["season_type"] = sample
            frames.append(release_map)
            audits.append(
                {
                    "season_end_year": int(year),
                    "season_type": sample,
                    "raw_path": str(raw_path),
                    **audit,
                }
            )
    if not frames:
        raise FileNotFoundError("No complete raw/terminal/second-chance samples")
    combined = pd.concat(frames, ignore_index=True)
    keys = ["game_id", "event_num", "season_end_year", "season_type"]
    if combined.duplicated(keys).any():
        raise ValueError("Release-map keys are not unique across samples")
    return combined, audits


def derive_non_shooter_lineups(
    shots: pd.DataFrame,
) -> Tuple[pd.DataFrame, Mapping[str, int]]:
    """Convert O1-O5 into the four non-shooters required by the OREB model."""
    required = {"shooter_id", "O1", "O2", "O3", "O4", "O5", *DEFENDER_COLS}
    missing = sorted(required - set(shots.columns))
    if missing:
        raise ValueError(f"Shot lineup input missing columns: {missing}")
    out = shots.copy()
    non_shooters: List[Tuple[int, int, int, int]] = []
    valid: List[bool] = []
    for values in out[
        ["shooter_id", "O1", "O2", "O3", "O4", "O5"]
    ].itertuples(index=False, name=None):
        try:
            shooter = int(float(values[0]))
        except (TypeError, ValueError):
            shooter = 0
        lineup: List[int] = []
        for value in values[1:]:
            try:
                player_id = int(float(value))
            except (TypeError, ValueError):
                continue
            if player_id > 0 and player_id not in lineup:
                lineup.append(player_id)
        row_valid = (
            shooter > 0
            and shooter in lineup
            and len(lineup) == 5
        )
        teammates = (
            [player for player in lineup if player != shooter]
            if row_valid
            else []
        )
        row_valid = row_valid and len(teammates) == 4
        valid.append(bool(row_valid))
        non_shooters.append(
            tuple(teammates) if row_valid else (0, 0, 0, 0)
        )
    for index, column in enumerate(NON_SHOOTER_COLS):
        out[column] = [values[index] for values in non_shooters]
    defender_valid = np.ones(len(out), dtype=bool)
    for column in DEFENDER_COLS:
        numeric = pd.to_numeric(out[column], errors="coerce").fillna(0).astype(int)
        out[column] = numeric
        defender_valid &= numeric.gt(0).to_numpy()
    valid_array = np.asarray(valid, dtype=bool) & defender_valid
    audit = {
        "rows": int(len(out)),
        "valid_complete_lineup_rows": int(valid_array.sum()),
        "invalid_lineup_rows": int((~valid_array).sum()),
    }
    return out.loc[valid_array].copy(), audit


def segment_from_zone(zone: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(zone.astype(str).eq("three"), "three", "mid"),
        index=zone.index,
    )


def fit_and_score_oreb_models(
    training: pd.DataFrame,
    test: pd.DataFrame,
    *,
    alpha: float,
    clip_bounds: Tuple[float, float] = (0.01, 0.99),
) -> Tuple[pd.DataFrame, Mapping[str, object]]:
    """Fit pre-test OREB models and score actual/role-swap configurations."""
    train = training.copy()
    score = test.copy()
    # A test-season dummy cannot be transported.  A common historical token
    # retains the feature contract while leaving season level to the intercept.
    train["season_sample"] = "historical"
    score["season_sample"] = "historical"
    for column in CONTEXT_COLS:
        train[column] = train[column].fillna("unknown").astype(str)
        score[column] = score[column].fillna("unknown").astype(str)

    space = build_feature_space(train)
    train_design = build_design_matrix(train, space)
    test_design = build_design_matrix(score, space)
    target = pd.to_numeric(train["actual_oreb"], errors="coerce").to_numpy(
        dtype=float
    )
    base_model = Ridge(
        alpha=float(alpha),
        fit_intercept=True,
        solver="lsqr",
        tol=1e-5,
        max_iter=2000,
    )
    enhanced_model = Ridge(
        alpha=float(alpha),
        fit_intercept=True,
        solver="lsqr",
        tol=1e-5,
        max_iter=2000,
    )
    base_model.fit(train_design[:, : space.shooter_zone_start], target)
    enhanced_model.fit(train_design, target)
    base_raw = base_model.predict(
        test_design[:, : space.shooter_zone_start]
    )
    enhanced_raw = enhanced_model.predict(test_design)
    score["oreb_probability_context_lineup"] = np.clip(
        base_raw, *clip_bounds
    )
    score["oreb_probability_actual_config"] = np.clip(
        enhanced_raw, *clip_bounds
    )

    non_shooter_coefficients, shooter_zone_coefficients = _coefficient_maps(
        enhanced_model.coef_, space
    )
    swap_probabilities = np.full(len(score), np.nan)
    raw_reconciliation = np.full(len(score), np.nan)
    role_columns = [
        "shooter_id",
        *NON_SHOOTER_COLS,
        "shot_zone",
    ]
    for row_index, values in enumerate(
        score[role_columns].itertuples(index=False, name=None)
    ):
        shooter_id = int(values[0])
        non_shooter_ids = [int(value) for value in values[1:5]]
        shot_zone = str(values[5])
        swap = role_swap_counterfactual(
            actual_raw_prediction=float(enhanced_raw[row_index]),
            shooter_id=shooter_id,
            non_shooter_ids=non_shooter_ids,
            shot_zone=shot_zone,
            non_shooter_coefficients=non_shooter_coefficients,
            shooter_zone_coefficients=shooter_zone_coefficients,
            clip_bounds=clip_bounds,
        )
        swap_probabilities[row_index] = (
            swap.equal_teammate_swap_probability
        )
        raw_reconciliation[row_index] = (
            swap.actual_minus_swap_raw
            - swap.crash_group_delta_raw
            - swap.shooter_zone_delta_raw
        )
    score["oreb_probability_role_swap"] = swap_probabilities
    score["role_vacancy_oreb_probability_delta"] = (
        score["oreb_probability_actual_config"]
        - score["oreb_probability_role_swap"]
    )
    audit = {
        "training_rows": int(len(train)),
        "test_rows": int(len(score)),
        "features": int(space.n_features),
        "base_features": int(space.shooter_zone_start),
        "shooter_zone_features": int(
            space.n_features - space.shooter_zone_start
        ),
        "alpha": float(alpha),
        "training_oreb_rate": float(np.mean(target)),
        "max_abs_role_swap_raw_reconciliation_gap": float(
            np.nanmax(np.abs(raw_reconciliation))
        ),
    }
    return score, audit


def train_only_conversion_by_segment(
    training: pd.DataFrame,
) -> Mapping[str, float]:
    """Price an initial-shot OREB using only pre-test continuation points."""
    work = training.copy()
    work["segment"] = segment_from_zone(work["shot_zone"])
    conversions: Dict[str, float] = {}
    for segment, frame in work.groupby("segment", observed=True):
        orebs = float(pd.to_numeric(frame["actual_oreb"]).sum())
        points = float(
            pd.to_numeric(
                frame["realized_second_chance_points"], errors="coerce"
            )
            .fillna(0.0)
            .sum()
        )
        conversions[str(segment)] = points / orebs if orebs > 0 else 0.0
    return conversions


def add_release_value_predictions(
    shots: pd.DataFrame,
    *,
    conversions: Mapping[str, float],
) -> pd.DataFrame:
    out = shots.copy()
    out["segment"] = out["segment"].astype(str)
    out["points_per_oreb_train_only"] = out["segment"].map(conversions)
    if out["points_per_oreb_train_only"].isna().any():
        raise ValueError("Missing train-only points-per-OREB conversion")
    out["actual_release_points"] = (
        pd.to_numeric(out["made"], errors="coerce")
        * pd.to_numeric(out["shot_value"], errors="coerce")
        + pd.to_numeric(
            out["realized_second_chance_points"], errors="coerce"
        ).fillna(0.0)
    )
    continuation_lanes = {
        "none": np.zeros(len(out), dtype=float),
        "context_lineup": out[
            "oreb_probability_context_lineup"
        ].to_numpy(dtype=float),
        "actual_config": out[
            "oreb_probability_actual_config"
        ].to_numpy(dtype=float),
    }
    shot_value = pd.to_numeric(out["shot_value"], errors="coerce").to_numpy(
        dtype=float
    )
    conversion = out["points_per_oreb_train_only"].to_numpy(dtype=float)
    for model, probability_column in IMMEDIATE_MODELS.items():
        make_probability = pd.to_numeric(
            out[probability_column], errors="coerce"
        ).to_numpy(dtype=float)
        out[f"immediate_ev::{model}"] = make_probability * shot_value
        for lane, oreb_probability in continuation_lanes.items():
            out[f"second_chance_ev::{model}::{lane}"] = (
                (1.0 - make_probability) * oreb_probability * conversion
            )
            out[f"release_ev::{model}::{lane}"] = (
                out[f"immediate_ev::{model}"]
                + out[f"second_chance_ev::{model}::{lane}"]
            )
        out[f"role_vacancy_sc_ev::{model}"] = (
            (1.0 - make_probability)
            * (
                out["oreb_probability_actual_config"].to_numpy(dtype=float)
                - out["oreb_probability_role_swap"].to_numpy(dtype=float)
            )
            * conversion
        )
    return out


def metric_rows(shots: pd.DataFrame) -> pd.DataFrame:
    rows: List[Mapping[str, object]] = []
    slices = {"all": np.ones(len(shots), dtype=bool)}
    slices.update(
        {
            segment: shots["segment"].eq(segment).to_numpy()
            for segment in sorted(shots["segment"].unique())
        }
    )
    for segment, mask in slices.items():
        frame = shots.loc[mask]
        actual = frame["actual_release_points"].to_numpy(dtype=float)
        for model in IMMEDIATE_MODELS:
            for lane in ("none", "context_lineup", "actual_config"):
                predicted = frame[
                    f"release_ev::{model}::{lane}"
                ].to_numpy(dtype=float)
                rows.append(
                    {
                        "segment": segment,
                        "model": model,
                        "continuation": lane,
                        "shots": int(len(frame)),
                        "games": int(frame["game_id"].nunique()),
                        "actual_points_per_release": float(actual.mean()),
                        "predicted_points_per_release": float(
                            predicted.mean()
                        ),
                        "bias": float(predicted.mean() - actual.mean()),
                        "points_mse": float(
                            np.mean(np.square(actual - predicted))
                        ),
                        "points_mae": float(
                            np.mean(np.abs(actual - predicted))
                        ),
                    }
                )
    return pd.DataFrame(rows)


def cluster_mse_interval(
    shots: pd.DataFrame,
    challenger_column: str,
    reference_column: str,
    *,
    iterations: int,
    seed: int,
) -> Mapping[str, float]:
    actual = shots["actual_release_points"].to_numpy(dtype=float)
    challenger = shots[challenger_column].to_numpy(dtype=float)
    reference = shots[reference_column].to_numpy(dtype=float)
    delta = np.square(actual - challenger) - np.square(actual - reference)
    grouped = (
        pd.DataFrame({"game_id": shots["game_id"], "delta": delta})
        .groupby("game_id", observed=True)
        .agg(delta_sum=("delta", "sum"), rows=("delta", "size"))
    )
    values = grouped[["delta_sum", "rows"]].to_numpy(dtype=float)
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(iterations), dtype=float)
    for index in range(int(iterations)):
        draw = rng.integers(0, len(values), size=len(values))
        selected = values[draw]
        samples[index] = selected[:, 0].sum() / selected[:, 1].sum()
    return {
        "delta": float(delta.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "probability_delta_below_zero": float(np.mean(samples < 0.0)),
    }


def interval_rows(
    shots: pd.DataFrame,
    *,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    comparisons: List[Tuple[str, str, str]] = []
    for model in IMMEDIATE_MODELS:
        comparisons.append(
            (
                f"{model}+actual_config",
                f"{model}+context_lineup",
                model,
            )
        )
    comparisons.extend(
        [
            (
                "context_epm+actual_config",
                "initial_ev+actual_config",
                "cross_immediate",
            ),
            (
                "geometry_identity+actual_config",
                "initial_ev+actual_config",
                "cross_immediate",
            ),
            (
                "context_epm+actual_config",
                "initial_ev+context_lineup",
                "full_challenger",
            ),
            (
                "geometry_identity+actual_config",
                "initial_ev+context_lineup",
                "full_challenger",
            ),
            (
                "initial_ev+context_lineup",
                "initial_ev+none",
                "continuation",
            ),
        ]
    )
    rows: List[Mapping[str, object]] = []
    slices = {"all": np.ones(len(shots), dtype=bool)}
    slices.update(
        {
            segment: shots["segment"].eq(segment).to_numpy()
            for segment in sorted(shots["segment"].unique())
        }
    )
    for segment, mask in slices.items():
        frame = shots.loc[mask]
        for index, (challenger, reference, family) in enumerate(comparisons):
            challenger_model, challenger_lane = challenger.split("+")
            reference_model, reference_lane = reference.split("+")
            interval = cluster_mse_interval(
                frame,
                f"release_ev::{challenger_model}::{challenger_lane}",
                f"release_ev::{reference_model}::{reference_lane}",
                iterations=iterations,
                seed=seed + 100 * index + len(rows),
            )
            rows.append(
                {
                    "segment": segment,
                    "family": family,
                    "challenger": challenger,
                    "reference": reference,
                    "metric": "points_mse",
                    **interval,
                }
            )
    return pd.DataFrame(rows)


def load_test_shots(
    predictions_path: Path,
    repo_sq_path: Path,
    *,
    test_season: int,
) -> Tuple[pd.DataFrame, Mapping[str, object]]:
    predictions = pd.read_parquet(predictions_path)
    predictions = predictions[
        predictions["season"].eq(int(test_season))
        & predictions["segment"].isin(["mid", "three"])
    ].copy()
    predictions["game_id"] = normalize_game_id(
        predictions["game_id_normalized"]
    )
    predictions["event_num"] = pd.to_numeric(
        predictions["GAME_EVENT_ID"], errors="coerce"
    ).astype("Int64")
    predictions["shooter_id"] = pd.to_numeric(
        predictions["personId"], errors="coerce"
    ).astype("Int64")
    predictions = predictions.dropna(
        subset=["event_num", "shooter_id"]
    ).copy()
    predictions["event_num"] = predictions["event_num"].astype(int)
    predictions["shooter_id"] = predictions["shooter_id"].astype(int)

    sq_columns = [
        "season_end_year",
        "season_type",
        "game_id",
        "event_num",
        "shooter_id",
        "shooter_name",
        "shot_zone",
        "distance_bin",
        "shot_family",
        "event_action_type",
        "modifier_bundle",
        "O1",
        "O2",
        "O3",
        "O4",
        "O5",
        *DEFENDER_COLS,
    ]
    repo_sq = pd.read_parquet(
        repo_sq_path,
        columns=sq_columns,
        filters=[("season_end_year", "=", int(test_season))],
    )
    repo_sq = repo_sq[repo_sq["season_end_year"].eq(int(test_season))].copy()
    repo_sq["game_id"] = normalize_game_id(repo_sq["game_id"])
    repo_sq["event_num"] = pd.to_numeric(
        repo_sq["event_num"], errors="coerce"
    ).astype("Int64")
    repo_sq["shooter_id"] = pd.to_numeric(
        repo_sq["shooter_id"], errors="coerce"
    ).astype("Int64")
    repo_sq = repo_sq.dropna(subset=["event_num", "shooter_id"]).copy()
    repo_sq["event_num"] = repo_sq["event_num"].astype(int)
    repo_sq["shooter_id"] = repo_sq["shooter_id"].astype(int)
    keys = ["game_id", "event_num", "shooter_id"]
    repo_sq = repo_sq.drop_duplicates(keys)
    joined = predictions.merge(
        repo_sq,
        on=keys,
        how="left",
        validate="one_to_one",
        indicator="_repo_sq_merge",
    )
    matched = joined["_repo_sq_merge"].eq("both")
    audit = {
        "prediction_rows": int(len(predictions)),
        "repo_sq_joined_rows": int(matched.sum()),
        "repo_sq_missing_rows": int((~matched).sum()),
    }
    return joined.loc[matched].drop(columns="_repo_sq_merge"), audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-season", type=int, default=2025)
    parser.add_argument(
        "--train-start-season", type=int, default=2022
    )
    parser.add_argument(
        "--predictions", type=Path, default=DEFAULT_PREDICTIONS
    )
    parser.add_argument("--repo-sq", type=Path, default=DEFAULT_REPO_SQ)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PIPELINE_ROOT / "processed",
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=PIPELINE_ROOT / "raw_data"
    )
    parser.add_argument("--season-type", choices=["RS", "PS", "ALL"], default="ALL")
    parser.add_argument("--alpha", type=float, default=300.0)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_years = list(
        range(int(args.train_start_season), int(args.test_season))
    )
    training_release_maps, training_release_audits = load_release_maps(
        args.raw_dir,
        args.processed_dir,
        years=train_years,
        season_type=args.season_type,
    )
    opportunities, _, opportunity_audit = load_opportunities(
        args.processed_dir,
        args.repo_sq,
        train_years,
        args.season_type,
    )
    print(f"Loaded {len(opportunities):,} pre-test OREB opportunities")
    release_unused_memory()
    training = opportunities.merge(
        training_release_maps,
        on=["game_id", "event_num", "season_end_year", "season_type"],
        how="left",
        validate="one_to_one",
        indicator="_release_merge",
    )
    training_matched = training["_release_merge"].eq("both")
    training = training[
        training_matched & training["fga_rank"].eq(0)
    ].drop(columns="_release_merge")
    training["segment"] = segment_from_zone(training["shot_zone"])
    training = training[training["segment"].isin(["mid", "three"])].copy()
    print(f"Retained {len(training):,} first-FGA training misses")
    del opportunities, training_release_maps
    release_unused_memory()

    test, test_join_audit = load_test_shots(
        args.predictions, args.repo_sq, test_season=args.test_season
    )
    print(f"Joined {len(test):,} test SQ rows")
    release_unused_memory()
    test, lineup_audit = derive_non_shooter_lineups(test)
    print(f"Retained {len(test):,} complete test lineups")
    test_release_maps, test_release_audits = load_release_maps(
        args.raw_dir,
        args.processed_dir,
        years=[int(args.test_season)],
        season_type=args.season_type,
    )
    test_release = test_release_maps.drop(
        columns=["season_end_year", "season_type"]
    )
    test = test.merge(
        test_release,
        on=["game_id", "event_num"],
        how="left",
        validate="one_to_one",
        indicator="_release_merge",
    )
    test_release_matched = test["_release_merge"].eq("both")
    test = test[
        test_release_matched & test["fga_rank"].eq(0)
    ].drop(columns="_release_merge")
    print(f"Retained {len(test):,} first-FGA test releases")
    del test_release_maps, test_release
    release_unused_memory()

    scored, model_audit = fit_and_score_oreb_models(
        training, test, alpha=args.alpha
    )
    print("Scored forward OREB context, shooter, and role-swap lanes")
    conversions = train_only_conversion_by_segment(training)
    scored = add_release_value_predictions(scored, conversions=conversions)
    metrics = metric_rows(scored)
    intervals = interval_rows(
        scored,
        iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    print("Completed game-cluster release-value intervals")
    player_zone = (
        scored.groupby(
            ["shooter_id", "shooter_name", "shot_zone"], observed=True
        )
        .agg(
            attempts=("event_num", "size"),
            oreb_probability_context_lineup=(
                "oreb_probability_context_lineup",
                "mean",
            ),
            oreb_probability_actual_config=(
                "oreb_probability_actual_config",
                "mean",
            ),
            oreb_probability_role_swap=(
                "oreb_probability_role_swap",
                "mean",
            ),
            role_vacancy_oreb_probability_delta=(
                "role_vacancy_oreb_probability_delta",
                "mean",
            ),
            role_vacancy_sc_ev_per_attempt_initial_ev=(
                "role_vacancy_sc_ev::initial_ev",
                "mean",
            ),
            role_vacancy_sc_ev_total_initial_ev=(
                "role_vacancy_sc_ev::initial_ev",
                "sum",
            ),
        )
        .reset_index()
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    intervals.to_csv(args.output_dir / "intervals.csv", index=False)
    scored.to_parquet(
        args.output_dir / "predictions.parquet", index=False
    )
    player_zone.to_csv(
        args.output_dir / "player_zone_accounting.csv", index=False
    )
    manifest = {
        "test_season": int(args.test_season),
        "train_years": train_years,
        "season_type": args.season_type,
        "alpha": float(args.alpha),
        "bootstrap_iterations": int(args.bootstrap_iterations),
        "seed": int(args.seed),
        "prediction_source": str(args.predictions),
        "repo_sq_source": str(args.repo_sq),
        "release_target": (
            "Initial FGA points plus SECOND_CHANCE_CLEAN points after its "
            "miss; first-chance free throws excluded."
        ),
        "validation_contract": (
            "OREB coefficients, shooter-zone effects, and points-per-OREB "
            "conversion use seasons strictly before the test season."
        ),
        "public_boundary": (
            "RAPM research challenger only. No canonical RUSSELL_SHOTQUALITY "
            "or active weighted_factors_decomp_* output is modified."
        ),
        "conversions": dict(conversions),
        "opportunity_audit": dict(opportunity_audit),
        "training_release_joined_rows": int(training_matched.sum()),
        "training_first_fga_rows": int(len(training)),
        "test_join_audit": dict(test_join_audit),
        "lineup_audit": dict(lineup_audit),
        "test_release_joined_rows": int(test_release_matched.sum()),
        "test_first_fga_rows": int(len(scored)),
        "model_audit": dict(model_audit),
        "release_map_audits": [
            *training_release_audits,
            *test_release_audits,
        ],
    }
    with (args.output_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
