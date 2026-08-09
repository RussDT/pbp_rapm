#!/usr/bin/env python3
"""Leakage-controlled retrospective historical ShotQuality crossfit.

This is a deliberately conservative research harness.  Every target encoding
is rebuilt inside each outer game fold:

* outer-training rows receive maps with their entire game removed;
* outer-validation rows receive maps fitted on outer-training games only;
* validation outcomes are never read while validation features are built.

The frozen rich-transfer-v3 score is treated as an upstream model prediction,
not recomputed here.  Direct (non-residual) empirical-Bayes encodings add
season/bucket, shot context, shooter career, shooter season, and defender
allowed information.  A small HGB stacker is evaluated strictly out of game.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from nba_pipeline.scripts.audit_historical_shot_quality_competitors import (
    paired_game_cluster_interval,
)


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V3_DIR = (
    PIPELINE_ROOT / "results" / "historical_shot_quality" / "rich_transfer_v3_scored"
)
DEFAULT_REFERENCE = (
    PIPELINE_ROOT / "results" / "pbp_shot_quality" / "pbp_shot_quality_1997_2026.parquet"
)
DEFAULT_OUTPUT_DIR = (
    PIPELINE_ROOT / "results" / "historical_shot_quality" / "safe_crossfit"
)

KEYS = ["source_file", "game_id", "event_num"]
DEFENDER_COLUMNS = ["D1", "D2", "D3", "D4", "D5"]
MODEL_BUCKETS = ("midrange", "three")
FIXED_PRIORS = {"midrange": 0.41, "three": 0.36}

V3_COLUMNS = [
    *KEYS,
    "shot_id",
    "season_end_year",
    "season_type",
    "model_bucket",
    "shot_value",
    "is_make",
    "period",
    "shooter_id",
    "shot_distance_ft",
    "shot_zone",
    "shot_family",
    "modifier_bundle",
    "event_action_type_str",
    "shot_clock_est",
    "shot_clock_fraction",
    "elapsed_since_reset",
    "effective_reset_len",
    "shot_clock_reset_reason",
    "shot_clock_confidence",
    "historical_sq_direct_make_prob",
    "historical_sq_distilled_make_prob",
    "historical_sq_component_gap",
    "historical_sq_make_prob",
]
REFERENCE_COLUMNS = [
    *KEYS,
    "is_make",
    "shot_value",
    "shot_quality_with_defense_make_prob",
    *DEFENDER_COLUMNS,
]

ENCODING_SPECS = {
    "safe_season_bucket": (
        ["season_end_year", "season_type", "model_bucket"],
        800.0,
    ),
    "safe_context": (
        [
            "model_bucket",
            "shot_zone",
            "shot_family",
            "event_action_type_str",
            "modifier_bundle",
            "distance_bin",
        ],
        180.0,
    ),
    "safe_shooter_career": (["shooter_id", "model_bucket"], 220.0),
    "safe_shooter_season": (
        ["shooter_id", "season_end_year", "season_type", "model_bucket"],
        120.0,
    ),
}
ENCODING_FEATURES = [
    f"{name}_{suffix}"
    for name in [*ENCODING_SPECS, "safe_defender_allowed"]
    for suffix in ["prob", "log_support"]
]

NUMERIC_FEATURES = [
    "historical_sq_make_prob",
    "historical_sq_direct_make_prob",
    "historical_sq_distilled_make_prob",
    "historical_sq_component_gap",
    "shot_distance_ft",
    "shot_clock_est",
    "shot_clock_fraction",
    "elapsed_since_reset",
    "effective_reset_len",
    "period",
    "season_end_year",
    *ENCODING_FEATURES,
]
CATEGORICAL_FEATURES = [
    "season_type",
    "shot_zone",
    "shot_family",
    "modifier_bundle",
    "event_action_type_str",
    "shot_clock_reset_reason",
    "shot_clock_confidence",
]
FEATURES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]


def era_label(season: int) -> str:
    if season <= 2003:
        return "1997-2003"
    if season <= 2013:
        return "2004-2013"
    if season <= 2023:
        return "2014-2023"
    return "2024-2026"


def write_json_atomic(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def fixed_prior(rows: pd.DataFrame) -> pd.Series:
    return rows["model_bucket"].map(FIXED_PRIORS).fillna(0.40).astype(float)


def _lookup_aggregates(
    rows: pd.DataFrame,
    aggregates: pd.DataFrame,
    keys: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Lookup sum/count maps without using any columns outside ``keys``."""
    left = rows[keys].copy()
    left["_safe_row_order"] = np.arange(len(left), dtype=np.int64)
    found = (
        left.merge(aggregates, on=keys, how="left", sort=False, validate="many_to_one")
        .sort_values("_safe_row_order", kind="stable")
    )
    return (
        found["_target_sum"].fillna(0.0).to_numpy(dtype=float),
        found["_target_count"].fillna(0.0).to_numpy(dtype=float),
    )


def direct_encode_fold(
    fit_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    *,
    keys: list[str],
    shrink: float,
    target_column: str = "is_make",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Direct EB map with whole-game exclusion for fit and fit-only validation.

    No validation target is accessed.  The fit encoding removes all rows from
    the current game, rather than merely subtracting the current shot.
    """
    if "_game_key" not in fit_rows or "_game_key" not in validation_rows:
        raise KeyError("Both frames must include _game_key.")
    target = pd.to_numeric(fit_rows[target_column], errors="coerce")
    if target.isna().any():
        raise ValueError("Fit targets must be complete.")
    work = fit_rows[[*keys, "_game_key"]].copy()
    work["_target"] = target.astype(float).to_numpy()
    totals = (
        work.groupby(keys, dropna=False, observed=True)["_target"]
        .agg(_target_sum="sum", _target_count="size")
        .reset_index()
    )
    game_totals = (
        work.groupby([*keys, "_game_key"], dropna=False, observed=True)["_target"]
        .agg(_game_sum="sum", _game_count="size")
        .reset_index()
    )
    fit_sum, fit_count = _lookup_aggregates(fit_rows, totals, keys)
    fit_game = fit_rows[[*keys, "_game_key"]].copy()
    fit_game["_safe_row_order"] = np.arange(len(fit_game), dtype=np.int64)
    found_game = (
        fit_game.merge(
            game_totals,
            on=[*keys, "_game_key"],
            how="left",
            sort=False,
            validate="many_to_one",
        )
        .sort_values("_safe_row_order", kind="stable")
    )
    fit_sum -= found_game["_game_sum"].fillna(0.0).to_numpy(dtype=float)
    fit_count -= found_game["_game_count"].fillna(0.0).to_numpy(dtype=float)
    fit_count = np.maximum(fit_count, 0.0)
    fit_probability = (fit_sum + shrink * fixed_prior(fit_rows).to_numpy()) / (
        fit_count + shrink
    )

    validation_sum, validation_count = _lookup_aggregates(validation_rows, totals, keys)
    validation_probability = (
        validation_sum + shrink * fixed_prior(validation_rows).to_numpy()
    ) / (validation_count + shrink)
    return (
        np.clip(fit_probability, 0.01, 0.99),
        fit_count,
        np.clip(validation_probability, 0.01, 0.99),
        validation_count,
    )


def _explode_defenders(rows: pd.DataFrame) -> pd.DataFrame:
    base_columns = [
        "_safe_row_id",
        "_game_key",
        "season_end_year",
        "model_bucket",
        "is_make",
    ]
    parts: list[pd.DataFrame] = []
    for column in DEFENDER_COLUMNS:
        part = rows[base_columns].copy()
        part["defender_id"] = (
            pd.to_numeric(rows[column], errors="coerce").fillna(0).astype(np.int64)
        )
        parts.append(part[part["defender_id"].gt(0)])
    if not parts:
        return pd.DataFrame(columns=[*base_columns, "defender_id"])
    return pd.concat(parts, ignore_index=True).drop_duplicates(
        ["_safe_row_id", "defender_id"], keep="first"
    )


def add_fold_target_encodings(
    fit_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build all outcome-derived features under the outer-fold contract."""
    fit = fit_rows.copy()
    validation = validation_rows.copy()
    for name, (keys, shrink) in ENCODING_SPECS.items():
        fit_prob, fit_support, valid_prob, valid_support = direct_encode_fold(
            fit,
            validation,
            keys=keys,
            shrink=shrink,
        )
        fit[f"{name}_prob"] = fit_prob
        fit[f"{name}_log_support"] = np.log1p(fit_support)
        validation[f"{name}_prob"] = valid_prob
        validation[f"{name}_log_support"] = np.log1p(valid_support)

    fit["_safe_row_id"] = np.arange(len(fit), dtype=np.int64)
    validation["_safe_row_id"] = np.arange(len(validation), dtype=np.int64)
    fit_defenders = _explode_defenders(fit)
    validation_defenders = _explode_defenders(validation)
    defender_keys = ["defender_id", "season_end_year", "model_bucket"]
    if fit_defenders.empty:
        fit_defender_prob = pd.Series(dtype=float)
        fit_defender_support = pd.Series(dtype=float)
        validation_defender_prob = pd.Series(dtype=float)
        validation_defender_support = pd.Series(dtype=float)
    else:
        fit_prob, fit_support, valid_prob, valid_support = direct_encode_fold(
            fit_defenders,
            validation_defenders,
            keys=defender_keys,
            shrink=500.0,
        )
        fit_defenders["_prob"] = fit_prob
        fit_defenders["_support"] = fit_support
        validation_defenders["_prob"] = valid_prob
        validation_defenders["_support"] = valid_support
        fit_defender_prob = fit_defenders.groupby("_safe_row_id", observed=True)["_prob"].mean()
        fit_defender_support = fit_defenders.groupby("_safe_row_id", observed=True)[
            "_support"
        ].mean()
        validation_defender_prob = validation_defenders.groupby(
            "_safe_row_id", observed=True
        )["_prob"].mean()
        validation_defender_support = validation_defenders.groupby(
            "_safe_row_id", observed=True
        )["_support"].mean()

    fit["safe_defender_allowed_prob"] = (
        fit["_safe_row_id"].map(fit_defender_prob).fillna(fixed_prior(fit))
    )
    fit["safe_defender_allowed_log_support"] = np.log1p(
        fit["_safe_row_id"].map(fit_defender_support).fillna(0.0)
    )
    validation["safe_defender_allowed_prob"] = (
        validation["_safe_row_id"]
        .map(validation_defender_prob)
        .fillna(fixed_prior(validation))
    )
    validation["safe_defender_allowed_log_support"] = np.log1p(
        validation["_safe_row_id"].map(validation_defender_support).fillna(0.0)
    )
    return (
        fit.drop(columns=["_safe_row_id"]),
        validation.drop(columns=["_safe_row_id"]),
    )


def load_joined_rows(
    v3_dir: Path,
    reference_path: Path,
    *,
    years: set[int] | None = None,
) -> tuple[pd.DataFrame, dict]:
    paths = sorted(v3_dir.glob("rich_transfer_v3_NBA*.parquet"))
    # Two-digit filenames need the repo convention: 97-99 -> 1997-99, else 2000+.
    selected: list[Path] = []
    for path in paths:
        token = path.stem.split("NBA", 1)[1].split("_", 1)[0]
        year = 1900 + int(token) if int(token) >= 90 else 2000 + int(token)
        if years is None or year in years:
            selected.append(path)
    paths = selected
    if not paths:
        raise FileNotFoundError(f"No matching v3 files under {v3_dir}")
    v3 = pd.concat(
        [pd.read_parquet(path, columns=V3_COLUMNS) for path in paths],
        ignore_index=True,
    )
    if v3.duplicated(KEYS).any():
        raise ValueError("Duplicate v3 shot identities.")
    source_files = sorted(v3["source_file"].astype(str).unique())
    reference = (
        ds.dataset(reference_path, format="parquet")
        .to_table(
            columns=REFERENCE_COLUMNS,
            filter=ds.field("source_file").isin(source_files),
        )
        .to_pandas()
    )
    if reference.duplicated(KEYS).any():
        raise ValueError("Duplicate canonical shot identities.")
    joined = v3.merge(
        reference,
        on=KEYS,
        how="left",
        suffixes=("_v3", "_reference"),
        indicator=True,
        validate="one_to_one",
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError(
            f"Canonical reference missed {int((~joined['_merge'].eq('both')).sum())} v3 shots."
        )
    y_v3 = pd.to_numeric(joined["is_make_v3"], errors="coerce")
    y_reference = pd.to_numeric(joined["is_make_reference"], errors="coerce")
    if y_v3.ne(y_reference).any():
        raise ValueError("Canonical and v3 outcomes disagree.")
    joined["is_make"] = y_reference.astype(int)
    joined["shot_value"] = pd.to_numeric(joined["shot_value_reference"], errors="coerce")
    joined["_game_key"] = (
        joined["source_file"].astype(str) + "::" + joined["game_id"].astype(str)
    )
    joined["distance_bin"] = (
        (pd.to_numeric(joined["shot_distance_ft"], errors="coerce") / 2.0)
        .round()
        .fillna(-1)
        .astype(int)
        .astype(str)
    )
    joined = joined.drop(columns=["_merge"])
    coverage = {
        "files": len(paths),
        "source_files": source_files,
        "rows": int(len(joined)),
        "games": int(joined["_game_key"].nunique()),
    }
    return joined, coverage


def prepare_categories(rows: pd.DataFrame, *, rare_minimum: int = 30) -> pd.DataFrame:
    out = rows.copy()
    for column in NUMERIC_FEATURES:
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        values = out[column].astype("string").fillna("missing")
        counts = values.value_counts(dropna=False)
        values = values.where(values.map(counts).fillna(0).ge(rare_minimum), "__rare__")
        out[column] = values.astype("category")
        if len(out[column].cat.categories) > 254:
            raise ValueError(f"Categorical feature {column} exceeds HGB's 254-level limit.")
    return out


def build_model(
    *,
    random_seed: int,
    min_samples_leaf: int,
    max_iter: int = 160,
) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=max_iter,
        max_leaf_nodes=15,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=20.0,
        max_bins=127,
        categorical_features=CATEGORICAL_FEATURES,
        early_stopping=False,
        random_state=random_seed,
    )


def select_outer_fit_blend_weight(
    raw_fit_rows: pd.DataFrame,
    *,
    random_seed: int,
    min_samples_leaf: int,
) -> tuple[float, dict]:
    """Select a v3/stack weight on a game-held-out subset of outer-fit rows.

    This selector never reads the outer-validation target. Its own validation
    encodings are rebuilt from selector-training games only.
    """
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=random_seed,
    )
    group_codes = pd.factorize(raw_fit_rows["_game_key"], sort=False)[0]
    selector_fit, selector_validation = next(
        splitter.split(raw_fit_rows, groups=group_codes)
    )
    raw_selector_fit = raw_fit_rows.iloc[selector_fit].copy()
    raw_selector_validation = raw_fit_rows.iloc[selector_validation].copy()
    selector_fit_rows, selector_validation_rows = add_fold_target_encodings(
        raw_selector_fit,
        raw_selector_validation,
    )
    prepared = prepare_categories(
        pd.concat([selector_fit_rows, selector_validation_rows], ignore_index=True)
    )
    selector_fit_rows = prepared.iloc[: len(selector_fit_rows)]
    selector_validation_rows = prepared.iloc[len(selector_fit_rows) :]
    model = build_model(
        random_seed=random_seed,
        min_samples_leaf=min_samples_leaf,
        max_iter=100,
    )
    model.fit(
        selector_fit_rows[FEATURES],
        selector_fit_rows["is_make"].astype(int),
    )
    stack_probability = model.predict_proba(
        selector_validation_rows[FEATURES]
    )[:, 1]
    v3_probability = selector_validation_rows["historical_sq_make_prob"].to_numpy(
        dtype=float
    )
    target = selector_validation_rows["is_make"].to_numpy(dtype=int)
    losses: dict[float, float] = {}
    for weight in np.linspace(0.0, 1.0, 9):
        blended = weight * stack_probability + (1.0 - weight) * v3_probability
        losses[float(weight)] = float(brier_score_loss(target, blended))
    # Prefer the more conservative v3-heavy blend on numerically exact ties.
    selected = min(losses, key=lambda weight: (losses[weight], weight))
    return float(selected), {
        "selector_rows": int(len(selector_validation)),
        "selector_stack_brier": float(brier_score_loss(target, stack_probability)),
        "selector_v3_brier": float(brier_score_loss(target, v3_probability)),
        "selected_stack_weight": float(selected),
        "selected_blend_brier": float(losses[selected]),
    }


def fit_safe_game_crossfit(
    rows: pd.DataFrame,
    *,
    folds: int = 3,
    random_seed: int = 20260723,
    min_samples_leaf: int = 300,
) -> tuple[np.ndarray, pd.DataFrame, list[dict]]:
    predictions = np.full(len(rows), np.nan, dtype=float)
    encoded_validation = pd.DataFrame(index=rows.index, columns=ENCODING_FEATURES, dtype=float)
    audits: list[dict] = []
    for bucket in MODEL_BUCKETS:
        positions = np.flatnonzero(rows["model_bucket"].eq(bucket).to_numpy())
        bucket_rows = rows.iloc[positions]
        splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=random_seed)
        group_codes = pd.factorize(bucket_rows["_game_key"], sort=False)[0]
        for fold, (fit_idx, valid_idx) in enumerate(
            splitter.split(bucket_rows, groups=group_codes)
        ):
            raw_fit = bucket_rows.iloc[fit_idx].copy()
            raw_validation = bucket_rows.iloc[valid_idx].copy()
            blend_weight, selector_audit = select_outer_fit_blend_weight(
                raw_fit,
                random_seed=random_seed + 100 + fold,
                min_samples_leaf=min_samples_leaf,
            )
            fit, validation = add_fold_target_encodings(raw_fit, raw_validation)
            combined = prepare_categories(pd.concat([fit, validation], ignore_index=True))
            fit_model = combined.iloc[: len(fit)]
            validation_model = combined.iloc[len(fit) :]
            model = build_model(
                random_seed=random_seed + fold,
                min_samples_leaf=min_samples_leaf,
            )
            model.fit(fit_model[FEATURES], fit_model["is_make"].astype(int))
            stack_probability = model.predict_proba(
                validation_model[FEATURES]
            )[:, 1]
            v3_probability = validation_model["historical_sq_make_prob"].to_numpy(
                dtype=float
            )
            predictions[positions[valid_idx]] = (
                blend_weight * stack_probability
                + (1.0 - blend_weight) * v3_probability
            )
            encoded_validation.loc[positions[valid_idx], ENCODING_FEATURES] = (
                validation[ENCODING_FEATURES].to_numpy()
            )
            fit_games = set(raw_fit["_game_key"])
            validation_games = set(raw_validation["_game_key"])
            audits.append(
                {
                    "model_bucket": bucket,
                    "fold": fold,
                    "fit_rows": int(len(raw_fit)),
                    "validation_rows": int(len(raw_validation)),
                    "fit_games": int(len(fit_games)),
                    "validation_games": int(len(validation_games)),
                    "game_overlap": int(len(fit_games & validation_games)),
                    **selector_audit,
                }
            )
    if np.isnan(predictions).any():
        raise ValueError("Safe crossfit produced missing predictions.")
    if encoded_validation.isna().any().any():
        raise ValueError("Safe crossfit produced missing validation encodings.")
    return np.clip(predictions, 1e-5, 1 - 1e-5), encoded_validation, audits


def evaluate(
    rows: pd.DataFrame,
    *,
    bootstrap_repetitions: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics: list[dict] = []
    comparisons: list[dict] = []
    groups: list[tuple[str, str, pd.DataFrame]] = [("all", "all", rows)]
    groups.extend(
        ("all", str(bucket), group)
        for bucket, group in rows.groupby("model_bucket", observed=True)
    )
    groups.extend(
        (str(era), "all", group)
        for era, group in rows.groupby("era", observed=True)
    )
    groups.extend(
        (str(era), str(bucket), group)
        for (era, bucket), group in rows.groupby(
            ["era", "model_bucket"], observed=True
        )
    )
    for era_name, bucket_name, group in groups:
        target = group["is_make"].astype(int)
        for model_name, column in {
            "safe_game_crossfit": "safe_crossfit_make_prob",
            "v3_portable": "historical_sq_make_prob",
            "canonical_defense_retrospective": "shot_quality_with_defense_make_prob",
        }.items():
            probability = pd.to_numeric(group[column], errors="coerce").clip(1e-5, 1 - 1e-5)
            eligible = target.notna() & probability.notna()
            metrics.append(
                {
                    "era": era_name,
                    "model_bucket": str(bucket_name),
                    "model": model_name,
                    "shots": int(eligible.sum()),
                    "brier": float(brier_score_loss(target[eligible], probability[eligible])),
                    "log_loss": float(
                        log_loss(target[eligible], probability[eligible], labels=[0, 1])
                    ),
                    "calibration_bias": float(
                        probability[eligible].mean() - target[eligible].mean()
                    ),
                }
            )
        candidate = group["safe_crossfit_make_prob"].astype(float)
        for competitor, baseline_column in {
            "canonical_defense_retrospective": "shot_quality_with_defense_make_prob",
            "v3_portable": "historical_sq_make_prob",
        }.items():
            baseline = group[baseline_column].astype(float)
            eligible = target.notna() & candidate.notna() & baseline.notna()
            interval = paired_game_cluster_interval(
                target[eligible].to_numpy(),
                candidate[eligible].to_numpy(),
                baseline[eligible].to_numpy(),
                group.loc[eligible, "_game_key"].astype(str).to_numpy(),
                repetitions=bootstrap_repetitions,
                seed=20260723 + len(comparisons),
            )
            candidate_brier = float(brier_score_loss(target[eligible], candidate[eligible]))
            baseline_brier = float(brier_score_loss(target[eligible], baseline[eligible]))
            candidate_log = float(
                log_loss(target[eligible], candidate[eligible], labels=[0, 1])
            )
            baseline_log = float(
                log_loss(target[eligible], baseline[eligible], labels=[0, 1])
            )
            comparisons.append(
                {
                    "era": era_name,
                    "model_bucket": str(bucket_name),
                    "competitor": competitor,
                    "shots": int(eligible.sum()),
                    "safe_brier": candidate_brier,
                    "competitor_brier": baseline_brier,
                    "brier_improvement": baseline_brier - candidate_brier,
                    "safe_log_loss": candidate_log,
                    "competitor_log_loss": baseline_log,
                    "log_loss_improvement": baseline_log - candidate_log,
                    **interval,
                }
            )
    return pd.DataFrame(metrics), pd.DataFrame(comparisons)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-dir", type=Path, default=DEFAULT_V3_DIR)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--years", nargs="*", type=int, default=[1997, 1998])
    parser.add_argument(
        "--all-years",
        action="store_true",
        help="Use every currently available v3 file instead of the default 1997-98 prototype.",
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--bootstrap-repetitions", type=int, default=500)
    parser.add_argument("--min-samples-leaf", type=int, default=300)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    years = None if args.all_years else set(args.years)
    rows, coverage = load_joined_rows(args.v3_dir, args.reference, years=years)
    rows["era"] = rows["season_end_year"].astype(int).map(era_label)
    predictions, encodings, audits = fit_safe_game_crossfit(
        rows,
        folds=args.folds,
        min_samples_leaf=args.min_samples_leaf,
    )
    rows["safe_crossfit_make_prob"] = predictions
    rows["safe_crossfit_ev"] = predictions * rows["shot_value"].astype(float)
    for column in ENCODING_FEATURES:
        rows[column] = encodings[column].to_numpy()
    metrics, comparisons = evaluate(
        rows,
        bootstrap_repetitions=args.bootstrap_repetitions,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_columns = [
        *KEYS,
        "shot_id",
        "season_end_year",
        "season_type",
        "model_bucket",
        "shot_value",
        "is_make",
        "historical_sq_make_prob",
        "shot_quality_with_defense_make_prob",
        *ENCODING_FEATURES,
        "safe_crossfit_make_prob",
        "safe_crossfit_ev",
    ]
    rows[prediction_columns].to_parquet(
        args.output_dir / "safe_crossfit_predictions.parquet",
        index=False,
    )
    metrics.to_csv(args.output_dir / "safe_crossfit_metrics.csv", index=False)
    comparisons.to_csv(args.output_dir / "safe_crossfit_comparisons.csv", index=False)
    write_json_atomic(args.output_dir / "safe_crossfit_fold_audit.json", audits)
    write_json_atomic(
        args.output_dir / "safe_crossfit_manifest.json",
        {
            **coverage,
            "years": sorted(years) if years else "all_available",
            "folds": args.folds,
            "target_encoding_contract": (
                "fit rows exclude their entire game; validation rows use fit-game-only maps"
            ),
            "upstream_v3_note": (
                "Frozen v3 probabilities are used as model inputs and are not recomputed from "
                "historical outcomes by this crossfit harness."
            ),
            "outcome_derived_encoding_features": ENCODING_FEATURES,
            "canonical_defense_baseline_label": (
                "retrospective repo-owned comparison; not a prospective leakage-safe baseline"
            ),
        },
    )
    print(metrics.to_string(index=False))
    print(comparisons.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
