#!/usr/bin/env python3
"""
Train a game-held-out shot-clock challenger over repo-owned Shot Quality.

The predictive model estimates make probability using the existing
chronological-prior SQ probability plus pre-release shot-clock state. The same
fold model also scores a neutral-clock counterfactual (default: 14 seconds)
while holding shot context fixed.

Generated shot rows and grenade summaries are RAPM-owned research artifacts.
They are not canonical RUSSELL_SHOTQUALITY, public DECOMP, or raw VPM inputs.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from nba_pipeline.scripts.shotquality_modeling import (
        normalize_game_id,
        paired_cluster_bootstrap,
        probability_metrics,
        write_json,
    )
except ModuleNotFoundError:
    from shotquality_modeling import (  # type: ignore
        normalize_game_id,
        paired_cluster_bootstrap,
        probability_metrics,
        write_json,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
DEFAULT_SQ_PATH = (
    PIPELINE_ROOT
    / "results"
    / "pbp_shot_quality"
    / "pbp_shot_quality_1997_2026.parquet"
)
DEFAULT_CLOCK_PATH = (
    PIPELINE_ROOT
    / "validation"
    / "shot_clock_calibration"
    / "shot_clock_26_rs_applied.parquet"
)
DEFAULT_OUTPUT_DIR = (
    PIPELINE_ROOT / "results" / "sq_shot_clock_challenger" / "2026_rs"
)
DEFAULT_SECOND_CHANCE_PATH = (
    PIPELINE_ROOT
    / "results"
    / "second_chance_opportunity_prior"
    / "22_26_all_a300"
    / "player_zone_accounting.csv"
)

JOIN_KEYS = ("game_id", "period", "event_num")
CLOCK_REQUIRED_COLUMNS = (
    "shot_clock_calibrated",
    "shot_clock_calibrated_lower",
    "shot_clock_calibrated_upper",
    "confidence_score",
    "needs_review",
    "game_clock_below_24",
    "reset_reason",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def _as_bool_numeric(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(int)
    normalized = values.astype("string").str.lower().str.strip()
    return normalized.isin({"1", "true", "t", "yes"}).astype(int)


def build_clock_feature_matrix(
    shots: pd.DataFrame,
    *,
    clock_override: float | Sequence[float] | None = None,
    neutral_clock: float = 14.0,
) -> pd.DataFrame:
    """Build the explicit pre-release feature contract used by the challenger."""
    required = {
        "shot_quality_prior_make_prob",
        "shot_clock_calibrated",
        *CLOCK_REQUIRED_COLUMNS[1:],
        "shot_value",
        "shot_zone",
        "reset_reason",
    }
    missing = sorted(required - set(shots.columns))
    if missing:
        raise ValueError(f"Missing shot-clock challenger columns: {missing}")

    baseline = np.clip(
        pd.to_numeric(
            shots["shot_quality_prior_make_prob"], errors="coerce"
        ).to_numpy(dtype=float),
        0.001,
        0.999,
    )
    if clock_override is None:
        clock = pd.to_numeric(
            shots["shot_clock_calibrated"], errors="coerce"
        ).to_numpy(dtype=float)
    elif np.isscalar(clock_override):
        clock = np.full(len(shots), float(clock_override), dtype=float)
    else:
        clock = np.asarray(clock_override, dtype=float)
        if len(clock) != len(shots):
            raise ValueError("clock_override length must match shots")

    centered = clock - float(neutral_clock)
    late_seconds = np.maximum(0.0, 7.0 - clock)
    grenade_seconds = np.maximum(0.0, 4.0 - clock)
    output: Dict[str, object] = {
        "baseline_logit": np.log(baseline / (1.0 - baseline)),
        "clock_centered": centered,
        "clock_centered_sq": np.square(centered),
        "late_7_seconds": late_seconds,
        "grenade_4_seconds": grenade_seconds,
        "clock_interval_width": (
            pd.to_numeric(
                shots["shot_clock_calibrated_upper"], errors="coerce"
            )
            - pd.to_numeric(
                shots["shot_clock_calibrated_lower"], errors="coerce"
            )
        ),
        "clock_confidence_score": pd.to_numeric(
            shots["confidence_score"], errors="coerce"
        ),
        "clock_needs_review": _as_bool_numeric(shots["needs_review"]),
        "game_clock_below_24": _as_bool_numeric(
            shots["game_clock_below_24"]
        ),
        "is_three": pd.to_numeric(
            shots["shot_value"], errors="coerce"
        ).eq(3.0).astype(int),
    }

    zones = shots["shot_zone"].fillna("unknown").astype(str)
    for zone in sorted(zones.unique()):
        indicator = zones.eq(zone).astype(float).to_numpy()
        output[f"clock_centered::zone={zone}"] = centered * indicator
        output[f"late_7::zone={zone}"] = late_seconds * indicator
        output[f"grenade_4::zone={zone}"] = grenade_seconds * indicator

    reset_reasons = shots["reset_reason"].fillna("unknown").astype(str)
    for reason in sorted(reset_reasons.unique()):
        output[f"reset_reason::{reason}"] = (
            reset_reasons.eq(reason).astype(int)
        )

    return pd.DataFrame(output, index=shots.index).astype(float)


def attach_clock_value_columns(
    shots: pd.DataFrame,
    *,
    conditioned_probability: Sequence[float],
    neutral_probability: Sequence[float],
    grenade_threshold: float = 4.0,
) -> pd.DataFrame:
    """Attach the exact conditioned + neutralization = neutral EV identity."""
    out = shots.copy()
    conditioned = np.clip(
        np.asarray(conditioned_probability, dtype=float), 0.001, 0.999
    )
    neutral = np.clip(
        np.asarray(neutral_probability, dtype=float), 0.001, 0.999
    )
    if len(conditioned) != len(out) or len(neutral) != len(out):
        raise ValueError("Probability arrays must match shot rows")
    shot_value = pd.to_numeric(out["shot_value"], errors="coerce").to_numpy(
        dtype=float
    )
    out["clock_conditioned_make_probability_oof"] = conditioned
    out["neutral_clock_make_probability_oof"] = neutral
    out["clock_conditioned_ev"] = conditioned * shot_value
    out["neutral_clock_ev"] = neutral * shot_value
    out["clock_drag_ev"] = (
        out["clock_conditioned_ev"] - out["neutral_clock_ev"]
    )
    out["grenade_burden_ev"] = -out["clock_drag_ev"]
    out["is_grenade"] = (
        pd.to_numeric(out["shot_clock_calibrated"], errors="coerce")
        <= float(grenade_threshold)
    )
    out["clock_value_identity_gap"] = (
        out["clock_conditioned_ev"]
        + out["grenade_burden_ev"]
        - out["neutral_clock_ev"]
    )
    return out


def attach_possession_value_columns(shots: pd.DataFrame) -> pd.DataFrame:
    """Add expected miss/rebound continuation to conditioned and neutral SQ."""
    required = {
        "clock_conditioned_make_probability_oof",
        "neutral_clock_make_probability_oof",
        "shot_value",
        "actual_config_oreb_probability_oof",
        "points_per_oreb_oof",
    }
    missing = sorted(required - set(shots.columns))
    if missing:
        raise ValueError(f"Missing possession-value columns: {missing}")

    out = shots.copy()
    conditioned_make = pd.to_numeric(
        out["clock_conditioned_make_probability_oof"], errors="coerce"
    )
    neutral_make = pd.to_numeric(
        out["neutral_clock_make_probability_oof"], errors="coerce"
    )
    shot_value = pd.to_numeric(out["shot_value"], errors="coerce")
    oreb_probability = pd.to_numeric(
        out["actual_config_oreb_probability_oof"], errors="coerce"
    )
    continuation_value = pd.to_numeric(
        out["points_per_oreb_oof"], errors="coerce"
    )

    out["clock_conditioned_shot_ev"] = conditioned_make * shot_value
    out["neutral_clock_shot_ev"] = neutral_make * shot_value
    out["clock_conditioned_second_chance_ev"] = (
        (1.0 - conditioned_make)
        * oreb_probability
        * continuation_value
    )
    out["neutral_clock_second_chance_ev"] = (
        (1.0 - neutral_make)
        * oreb_probability
        * continuation_value
    )
    out["clock_conditioned_possession_ev"] = (
        out["clock_conditioned_shot_ev"]
        + out["clock_conditioned_second_chance_ev"]
    )
    out["neutral_clock_possession_ev"] = (
        out["neutral_clock_shot_ev"]
        + out["neutral_clock_second_chance_ev"]
    )
    out["grenade_shot_burden_ev"] = (
        out["neutral_clock_shot_ev"] - out["clock_conditioned_shot_ev"]
    )
    out["second_chance_recovery_offset_ev"] = (
        out["clock_conditioned_second_chance_ev"]
        - out["neutral_clock_second_chance_ev"]
    )
    out["grenade_possession_burden_ev"] = (
        out["neutral_clock_possession_ev"]
        - out["clock_conditioned_possession_ev"]
    )
    out["possession_clock_identity_gap"] = (
        out["clock_conditioned_possession_ev"]
        + out["grenade_possession_burden_ev"]
        - out["neutral_clock_possession_ev"]
    )
    out["grenade_burden_bridge_gap"] = (
        out["grenade_shot_burden_ev"]
        - out["second_chance_recovery_offset_ev"]
        - out["grenade_possession_burden_ev"]
    )
    return out


def attach_second_chance_prior(
    shots: pd.DataFrame,
    accounting: pd.DataFrame,
) -> Tuple[pd.DataFrame, Mapping[str, object]]:
    """Attach the player-zone OREB continuation prior with explicit coverage."""
    required = {
        "shooter_id",
        "shot_zone",
        "actual_config_oreb_probability_oof",
        "points_per_oreb_oof",
    }
    missing = sorted(required - set(accounting.columns))
    if missing:
        raise ValueError(f"Second-chance accounting missing columns: {missing}")
    lookup = accounting[
        [
            "shooter_id",
            "shot_zone",
            "actual_config_oreb_probability_oof",
            "points_per_oreb_oof",
        ]
    ].copy()
    lookup["shooter_id"] = pd.to_numeric(
        lookup["shooter_id"], errors="coerce"
    ).astype("Int64")
    if lookup.duplicated(["shooter_id", "shot_zone"]).any():
        raise ValueError("Second-chance accounting has duplicate player-zone keys")
    out = shots.merge(
        lookup,
        on=["shooter_id", "shot_zone"],
        how="left",
        validate="many_to_one",
        indicator="_second_chance_merge",
    )
    matched = out["_second_chance_merge"].eq("both")
    audit = {
        "rows": int(len(out)),
        "matched_rows": int(matched.sum()),
        "coverage": float(matched.mean()) if len(out) else 0.0,
        "missing_rows": int((~matched).sum()),
    }
    out = out.drop(columns=["_second_chance_merge"])
    if not matched.all():
        raise ValueError(
            "Second-chance player-zone coverage is incomplete: "
            f"{audit['matched_rows']}/{audit['rows']}"
        )
    return attach_possession_value_columns(out), audit


def summarize_player_clock_burden(shots: pd.DataFrame) -> pd.DataFrame:
    """Create shooter-level grenade load without treating it as causal credit."""
    required = {
        "shooter_id",
        "shooter_name",
        "is_make",
        "shot_value",
        "is_grenade",
        "clock_conditioned_ev",
        "neutral_clock_ev",
        "clock_drag_ev",
        "grenade_burden_ev",
    }
    missing = sorted(required - set(shots.columns))
    if missing:
        raise ValueError(f"Missing player-summary columns: {missing}")

    work = shots.copy()
    work["actual_points"] = (
        pd.to_numeric(work["is_make"], errors="coerce").fillna(0.0)
        * pd.to_numeric(work["shot_value"], errors="coerce").fillna(0.0)
    )
    work["grenade_actual_points"] = np.where(
        work["is_grenade"], work["actual_points"], 0.0
    )
    work["grenade_burden_only"] = np.where(
        work["is_grenade"], work["grenade_burden_ev"], 0.0
    )
    work["grenade_conditioned_ev"] = np.where(
        work["is_grenade"], work["clock_conditioned_ev"], 0.0
    )
    work["grenade_neutral_ev"] = np.where(
        work["is_grenade"], work["neutral_clock_ev"], 0.0
    )
    if "grenade_possession_burden_ev" in work.columns:
        work["grenade_possession_burden_only"] = np.where(
            work["is_grenade"],
            work["grenade_possession_burden_ev"],
            0.0,
        )
        work["grenade_second_chance_offset_only"] = np.where(
            work["is_grenade"],
            work["second_chance_recovery_offset_ev"],
            0.0,
        )

    summary = (
        work.groupby("shooter_id", observed=True)
        .agg(
            shooter_name=("shooter_name", "first"),
            attempts=("is_make", "size"),
            makes=("is_make", "sum"),
            actual_points=("actual_points", "sum"),
            grenade_attempts=("is_grenade", "sum"),
            grenade_makes=("is_make", lambda values: int(
                values[work.loc[values.index, "is_grenade"]].sum()
            )),
            grenade_actual_points=("grenade_actual_points", "sum"),
            clock_conditioned_ev_total=("clock_conditioned_ev", "sum"),
            neutral_clock_ev_total=("neutral_clock_ev", "sum"),
            clock_drag_ev_total=("clock_drag_ev", "sum"),
            grenade_burden_ev_total=("grenade_burden_only", "sum"),
            grenade_conditioned_ev_total=("grenade_conditioned_ev", "sum"),
            grenade_neutral_ev_total=("grenade_neutral_ev", "sum"),
        )
        .reset_index()
    )
    summary["grenade_rate"] = (
        summary["grenade_attempts"] / summary["attempts"]
    )
    summary["actual_points_per_shot"] = (
        summary["actual_points"] / summary["attempts"]
    )
    summary["clock_conditioned_ev_per_shot"] = (
        summary["clock_conditioned_ev_total"] / summary["attempts"]
    )
    summary["neutral_clock_ev_per_shot"] = (
        summary["neutral_clock_ev_total"] / summary["attempts"]
    )
    summary["clock_drag_ev_per_shot"] = (
        summary["clock_drag_ev_total"] / summary["attempts"]
    )
    grenade_denominator = summary["grenade_attempts"].replace(0, np.nan)
    summary["grenade_fg_pct"] = (
        summary["grenade_makes"] / grenade_denominator
    )
    summary["grenade_actual_points_per_shot"] = (
        summary["grenade_actual_points"] / grenade_denominator
    )
    summary["grenade_burden_ev_per_grenade"] = (
        summary["grenade_burden_ev_total"] / grenade_denominator
    )
    if "grenade_possession_burden_only" in work.columns:
        possession = (
            work.groupby("shooter_id", observed=True)
            .agg(
                grenade_possession_burden_ev_total=(
                    "grenade_possession_burden_only",
                    "sum",
                ),
                grenade_second_chance_recovery_offset_total=(
                    "grenade_second_chance_offset_only",
                    "sum",
                ),
            )
            .reset_index()
        )
        summary = summary.merge(
            possession,
            on="shooter_id",
            how="left",
            validate="one_to_one",
        )
        summary["grenade_possession_burden_ev_per_grenade"] = (
            summary["grenade_possession_burden_ev_total"]
            / grenade_denominator
        )
    return summary.sort_values(
        ["grenade_burden_ev_total", "grenade_attempts"],
        ascending=[False, False],
    ).reset_index(drop=True)


def _new_logit_model(c_value: float):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            C=float(c_value),
            max_iter=750,
            solver="lbfgs",
        ),
    )


def grouped_oof_with_counterfactual(
    actual_features: pd.DataFrame,
    neutral_features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    *,
    folds: int,
    seed: int,
    c_value: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if list(actual_features.columns) != list(neutral_features.columns):
        raise ValueError("Actual and neutral feature contracts differ")
    actual_probability = np.full(len(target), np.nan)
    neutral_probability = np.full(len(target), np.nan)
    fold_ids = np.full(len(target), -1, dtype=int)
    splitter = GroupKFold(
        n_splits=int(folds),
        shuffle=True,
        random_state=int(seed),
    )
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(actual_features, target, groups)
    ):
        model = _new_logit_model(c_value)
        model.fit(actual_features.iloc[train_idx], target.iloc[train_idx])
        actual_probability[test_idx] = model.predict_proba(
            actual_features.iloc[test_idx]
        )[:, 1]
        neutral_probability[test_idx] = model.predict_proba(
            neutral_features.iloc[test_idx]
        )[:, 1]
        fold_ids[test_idx] = fold
    if (
        np.isnan(actual_probability).any()
        or np.isnan(neutral_probability).any()
        or (fold_ids < 0).any()
    ):
        raise RuntimeError("Grouped OOF coverage is incomplete")
    return actual_probability, neutral_probability, fold_ids


def chronological_holdout(
    actual_features: pd.DataFrame,
    neutral_features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    *,
    c_value: float,
    fraction: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Sequence[str]]:
    ordered_games = np.asarray(sorted(groups.astype(str).unique()))
    cut = max(1, int(np.floor(len(ordered_games) * (1.0 - fraction))))
    train_games = set(ordered_games[:cut])
    train_mask = groups.astype(str).isin(train_games).to_numpy()
    test_mask = ~train_mask
    model = _new_logit_model(c_value)
    model.fit(actual_features.loc[train_mask], target.loc[train_mask])
    actual_probability = model.predict_proba(
        actual_features.loc[test_mask]
    )[:, 1]
    neutral_probability = model.predict_proba(
        neutral_features.loc[test_mask]
    )[:, 1]
    return test_mask, actual_probability, neutral_probability, ordered_games[cut:]


def load_joined_shots(
    sq_path: Path,
    clock_path: Path,
    *,
    season_end_year: int,
    season_type: str,
) -> Tuple[pd.DataFrame, Mapping[str, object]]:
    sq_columns = [
        "season_end_year",
        "season_type",
        "game_id",
        "period",
        "event_num",
        "shooter_id",
        "shooter_name",
        "shot_value",
        "is_make",
        "shot_quality_prior_make_prob",
        "shot_quality_season_make_prob",
        "shot_zone",
        "distance_bin",
        "shot_family",
        "event_action_type",
        "modifier_bundle",
    ]
    clock_columns = [
        *JOIN_KEYS,
        "shooter_id",
        "shot_clock_est",
        *CLOCK_REQUIRED_COLUMNS,
        "shot_clock_calibrated_lower",
        "shot_clock_calibrated_upper",
        "confidence",
        "confidence_reason",
    ]
    sq = pd.read_parquet(sq_path, columns=sq_columns)
    sq = sq[
        sq["season_end_year"].eq(int(season_end_year))
        & sq["season_type"].astype(str).str.upper().eq(season_type.upper())
    ].copy()
    clock = pd.read_parquet(clock_path, columns=list(dict.fromkeys(clock_columns)))
    if sq.empty:
        raise ValueError("No SQ rows for requested season/sample")
    if clock.empty:
        raise ValueError("No shot-clock rows")

    for frame in (sq, clock):
        frame["game_id"] = normalize_game_id(frame["game_id"])
        for column in ("period", "event_num"):
            frame[column] = pd.to_numeric(
                frame[column], errors="coerce"
            ).astype("Int64")
    sq = sq.dropna(subset=list(JOIN_KEYS)).copy()
    clock = clock.dropna(subset=list(JOIN_KEYS)).copy()
    if sq.duplicated(list(JOIN_KEYS)).any():
        raise ValueError("SQ input has duplicate shot keys")
    if clock.duplicated(list(JOIN_KEYS)).any():
        raise ValueError("Shot-clock input has duplicate shot keys")

    joined = sq.merge(
        clock,
        on=list(JOIN_KEYS),
        how="left",
        suffixes=("", "_clock"),
        validate="one_to_one",
        indicator=True,
    )
    shooter_match = pd.to_numeric(
        joined["shooter_id"], errors="coerce"
    ).eq(pd.to_numeric(joined["shooter_id_clock"], errors="coerce"))
    matched = joined["_merge"].eq("both") & shooter_match
    audit = {
        "sq_rows": int(len(sq)),
        "clock_rows": int(len(clock)),
        "joined_rows": int(matched.sum()),
        "join_coverage": float(matched.mean()),
        "shooter_id_mismatches": int(
            (joined["_merge"].eq("both") & ~shooter_match).sum()
        ),
        "sq_games": int(sq["game_id"].nunique()),
        "clock_games": int(clock["game_id"].nunique()),
    }
    joined = joined[matched].drop(
        columns=["_merge", "shooter_id_clock"]
    ).reset_index(drop=True)
    joined["shooter_id"] = pd.to_numeric(
        joined["shooter_id"], errors="coerce"
    ).astype(int)
    joined["is_make"] = joined["is_make"].astype(bool).astype(int)
    return joined, audit


def _clock_bin(values: pd.Series) -> pd.Series:
    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=[-0.001, 4.0, 7.0, 14.0, 24.001],
        labels=["grenade_0_4", "late_4_7", "middle_7_14", "early_14_24"],
        include_lowest=True,
    )


def build_clock_bin_metrics(shots: pd.DataFrame) -> Sequence[Mapping[str, object]]:
    work = shots.copy()
    work["clock_bin"] = _clock_bin(work["shot_clock_calibrated"])
    output = []
    for clock_bin, group in work.groupby("clock_bin", observed=True):
        output.append(
            {
                "clock_bin": str(clock_bin),
                "shots": int(len(group)),
                "actual_make_rate": float(group["is_make"].mean()),
                "canonical_sq_mean": float(
                    group["shot_quality_prior_make_prob"].mean()
                ),
                "clock_conditioned_mean": float(
                    group["clock_conditioned_make_probability_oof"].mean()
                ),
                "neutral_clock_mean": float(
                    group["neutral_clock_make_probability_oof"].mean()
                ),
                "clock_drag_ev_mean": float(group["clock_drag_ev"].mean()),
            }
        )
    return output


def run_experiment(
    shots: pd.DataFrame,
    *,
    second_chance_accounting: pd.DataFrame | None = None,
    folds: int,
    seed: int,
    c_value: float,
    neutral_clock: float,
    grenade_threshold: float,
    bootstrap_iterations: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Mapping[str, object], object]:
    target = shots["is_make"].astype(int)
    groups = shots["game_id"].astype(str)
    canonical = np.clip(
        shots["shot_quality_prior_make_prob"].to_numpy(dtype=float),
        0.001,
        0.999,
    )
    baseline_features = pd.DataFrame(
        {
            "baseline_logit": np.log(canonical / (1.0 - canonical)),
        },
        index=shots.index,
    )
    actual_features = build_clock_feature_matrix(
        shots, neutral_clock=neutral_clock
    )
    neutral_features = build_clock_feature_matrix(
        shots,
        clock_override=neutral_clock,
        neutral_clock=neutral_clock,
    )

    baseline_oof, _, baseline_fold = grouped_oof_with_counterfactual(
        baseline_features,
        baseline_features,
        target,
        groups,
        folds=folds,
        seed=seed,
        c_value=c_value,
    )
    conditioned_oof, neutral_oof, clock_fold = (
        grouped_oof_with_counterfactual(
            actual_features,
            neutral_features,
            target,
            groups,
            folds=folds,
            seed=seed,
            c_value=c_value,
        )
    )
    if not np.array_equal(baseline_fold, clock_fold):
        raise RuntimeError("Baseline and clock models used different folds")

    predictions = shots.copy()
    predictions["outer_fold"] = clock_fold
    predictions["canonical_sq_make_probability"] = canonical
    predictions["baseline_recalibrated_make_probability_oof"] = baseline_oof
    predictions = attach_clock_value_columns(
        predictions,
        conditioned_probability=conditioned_oof,
        neutral_probability=neutral_oof,
        grenade_threshold=grenade_threshold,
    )
    second_chance_audit = None
    if second_chance_accounting is not None:
        predictions, second_chance_audit = attach_second_chance_prior(
            predictions,
            second_chance_accounting,
        )

    models = {
        "canonical_sq_prior": canonical,
        "baseline_recalibrated_oof": baseline_oof,
        "clock_conditioned_oof": conditioned_oof,
    }
    metrics = {
        name: probability_metrics(target, probability)
        for name, probability in models.items()
    }
    comparisons = {
        "clock_vs_canonical": paired_cluster_bootstrap(
            target,
            conditioned_oof,
            canonical,
            groups,
            iterations=bootstrap_iterations,
            seed=seed,
        ),
        "clock_vs_baseline_recalibrated": paired_cluster_bootstrap(
            target,
            conditioned_oof,
            baseline_oof,
            groups,
            iterations=bootstrap_iterations,
            seed=seed + 1,
        ),
    }

    temporal_test, temporal_conditioned, temporal_neutral, temporal_games = (
        chronological_holdout(
            actual_features,
            neutral_features,
            target,
            groups,
            c_value=c_value,
        )
    )
    temporal_baseline_test, temporal_baseline, _, baseline_games = (
        chronological_holdout(
            baseline_features,
            baseline_features,
            target,
            groups,
            c_value=c_value,
        )
    )
    if (
        not np.array_equal(temporal_test, temporal_baseline_test)
        or list(temporal_games) != list(baseline_games)
    ):
        raise RuntimeError("Chronological model families used different games")
    chronological = {
        "test_games": int(len(temporal_games)),
        "first_test_game": str(temporal_games[0]),
        "last_test_game": str(temporal_games[-1]),
        "canonical_sq_prior": probability_metrics(
            target.loc[temporal_test],
            canonical[temporal_test],
        ),
        "baseline_recalibrated": probability_metrics(
            target.loc[temporal_test],
            temporal_baseline,
        ),
        "clock_conditioned": probability_metrics(
            target.loc[temporal_test],
            temporal_conditioned,
        ),
        "mean_neutral_clock_probability": float(
            np.mean(temporal_neutral)
        ),
    }

    final_model = _new_logit_model(c_value)
    final_model.fit(actual_features, target)
    player_summary = summarize_player_clock_burden(predictions)
    result = {
        "rows": int(len(shots)),
        "games": int(groups.nunique()),
        "folds": int(folds),
        "seed": int(seed),
        "logistic_c": float(c_value),
        "neutral_clock_seconds": float(neutral_clock),
        "grenade_threshold_seconds": float(grenade_threshold),
        "feature_columns": list(actual_features.columns),
        "metrics": metrics,
        "paired_game_cluster_bootstrap": comparisons,
        "chronological_last_20pct_games": chronological,
        "clock_bin_metrics": build_clock_bin_metrics(predictions),
        "max_abs_clock_value_identity_gap": float(
            predictions["clock_value_identity_gap"].abs().max()
        ),
        "interpretation": {
            "clock_drag_ev": (
                "clock-conditioned shot EV minus the same fold model scored "
                "at the neutral clock; negative values are clock burden"
            ),
            "grenade_burden_ev": (
                "neutral-clock EV minus clock-conditioned EV; player summaries "
                "sum this only on attempts at or below the grenade threshold"
            ),
        },
    }
    if second_chance_audit is not None:
        result["second_chance_join_audit"] = second_chance_audit
        result["max_abs_possession_clock_identity_gap"] = float(
            predictions["possession_clock_identity_gap"].abs().max()
        )
        result["max_abs_grenade_burden_bridge_gap"] = float(
            predictions["grenade_burden_bridge_gap"].abs().max()
        )
    return predictions, player_summary, result, final_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sq-path", type=Path, default=DEFAULT_SQ_PATH)
    parser.add_argument("--clock-path", type=Path, default=DEFAULT_CLOCK_PATH)
    parser.add_argument(
        "--second-chance-accounting-path",
        type=Path,
        default=DEFAULT_SECOND_CHANCE_PATH,
    )
    parser.add_argument("--season-end-year", type=int, default=2026)
    parser.add_argument(
        "--season-type",
        choices=["RS", "PS", "rs", "ps"],
        default="RS",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--logistic-c", type=float, default=0.1)
    parser.add_argument("--neutral-clock", type=float, default=14.0)
    parser.add_argument("--grenade-threshold", type=float, default=4.0)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shots, join_audit = load_joined_shots(
        args.sq_path,
        args.clock_path,
        season_end_year=args.season_end_year,
        season_type=args.season_type.upper(),
    )
    logging.info(
        "Joined %d shots across %d games (coverage %.3f%%)",
        len(shots),
        shots["game_id"].nunique(),
        100.0 * float(join_audit["join_coverage"]),
    )
    second_chance_accounting = None
    if args.second_chance_accounting_path:
        if not args.second_chance_accounting_path.exists():
            raise FileNotFoundError(
                "Second-chance accounting file not found: "
                f"{args.second_chance_accounting_path}"
            )
        second_chance_accounting = pd.read_csv(
            args.second_chance_accounting_path
        )
    predictions, player_summary, metrics, final_model = run_experiment(
        shots,
        second_chance_accounting=second_chance_accounting,
        folds=args.folds,
        seed=args.seed,
        c_value=args.logistic_c,
        neutral_clock=args.neutral_clock,
        grenade_threshold=args.grenade_threshold,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    metrics = dict(metrics)
    metrics["join_audit"] = dict(join_audit)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(
        args.output_dir / "sq_shot_clock_oof_predictions.parquet",
        index=False,
    )
    player_summary.to_csv(
        args.output_dir / "player_grenade_burden.csv",
        index=False,
    )
    write_json(args.output_dir / "metrics.json", metrics)
    joblib.dump(
        {
            "model": final_model,
            "feature_columns": metrics["feature_columns"],
            "neutral_clock_seconds": float(args.neutral_clock),
            "season_end_year": int(args.season_end_year),
            "season_type": args.season_type.upper(),
            "training_rows": int(len(shots)),
            "training_games": int(shots["game_id"].nunique()),
            "leakage_contract": (
                "Existing chronological-prior SQ anchor plus calibrated "
                "pre-release clock state only; no current shot outcome, "
                "assist, rebound, future event, or raw defender join"
            ),
        },
        args.output_dir / "sq_shot_clock_model.joblib",
    )
    manifest = {
        "artifact": "sq_shot_clock_challenger",
        "status": "RAPM-owned experimental overlay",
        "inputs": {
            "repo_owned_sq": str(args.sq_path),
            "calibrated_shot_clock": str(args.clock_path),
            "second_chance_player_zone_prior": str(
                args.second_chance_accounting_path
            ),
        },
        "outputs": {
            "shot_oof_predictions": "sq_shot_clock_oof_predictions.parquet",
            "player_grenade_summary": "player_grenade_burden.csv",
            "metrics": "metrics.json",
            "model": "sq_shot_clock_model.joblib",
        },
        "vpm_boundary": {
            "never_direct_vpm_inputs": [
                "raw shot rows",
                "raw event/provider joins",
                "shot-level reset reasons",
                "raw defender assignments",
                "fold-level predictions",
            ],
            "possible_future_vpm_priors_after_reliability_tests": [
                "stabilized player-season grenade attempt rate",
                "stabilized player-season grenade burden per attempt",
                "clock-adjusted shooter talent residual",
            ],
            "required_gates": [
                "multi-season backfill",
                "forward-season validation",
                "year-to-year stability",
                "minimum-attempt reliability",
                "incremental VPM target validation",
            ],
        },
        "public_guardrail": (
            "Does not modify RUSSELL_SHOTQUALITY, FIRST_CHANCE, "
            "SECOND_CHANCE_CLEAN, legacy Alt3, or active public DECOMP."
        ),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    logging.info("Wrote shot-clock SQ challenger to %s", args.output_dir)


if __name__ == "__main__":
    main()
