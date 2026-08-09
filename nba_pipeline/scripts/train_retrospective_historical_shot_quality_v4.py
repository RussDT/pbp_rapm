#!/usr/bin/env python3
"""Single-anchor retrospective historical ShotQuality v4.

This model is intentionally separate from ``rich_transfer_v3``:

* v3 is the portable, prospective PBP-only model.
* v4 is a deliberately retrospective research surface. Its canonical defense
  anchor is leave-one-shot-out but may use other shots from the same game and
  season.

Combining several nested shot-level leave-one-out aggregates can algebraically
reconstruct the held-out result. V4 therefore permits exactly one canonical
LOO anchor in the learner: the defense-adjusted probability. The context,
prior, career, and season variants remain comparisons only. Game-group OOF
prevents stacker train/validation game overlap, while the temporal benchmark
fits the stacker only on earlier seasons. Neither removes same-game or
same-season information already embedded in the anchor, so this is not the
leakage-controlled prospective lane.
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

import joblib
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold

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
    PIPELINE_ROOT / "results" / "historical_shot_quality" / "retrospective_v4"
)

KEYS = ["source_file", "game_id", "event_num"]
MODEL_BUCKETS = ("midrange", "three")

REFERENCE_PROBABILITIES = {
    "v3_portable": "historical_sq_make_prob",
    "v3_direct": "historical_sq_direct_make_prob",
    "v3_distilled": "historical_sq_distilled_make_prob",
    "repo_context_shot_loo": "shot_context_make_prob",
    "repo_pregame_prior": "shot_quality_prior_make_prob",
    "repo_career_shot_loo": "shot_quality_make_prob",
    "repo_season_shot_loo": "shot_quality_season_make_prob",
    "repo_defense_shot_loo": "shot_quality_with_defense_make_prob",
}
NUMERIC_FEATURES = [
    "historical_sq_make_prob",
    "historical_sq_direct_make_prob",
    "historical_sq_distilled_make_prob",
    "historical_sq_component_gap",
    "shooter_pregame_make_rate",
    "shooter_pregame_log_attempts",
    "shot_quality_with_defense_make_prob",
    "shot_distance_ft",
    "shot_clock_est",
    "shot_clock_fraction",
    "elapsed_since_reset",
    "effective_reset_len",
    "period",
    "season_end_year",
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
    "shooter_name",
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
    "shooter_pregame_make_rate",
    "shooter_pregame_log_attempts",
    "historical_sq_direct_make_prob",
    "historical_sq_distilled_make_prob",
    "historical_sq_component_gap",
    "historical_sq_make_prob",
]

REFERENCE_COLUMNS = [
    *KEYS,
    "is_make",
    "shot_value",
    "talent_bucket",
    "shot_context_make_prob",
    "shot_quality_prior_make_prob",
    "shot_quality_make_prob",
    "shot_quality_season_make_prob",
    "shot_quality_with_defense_make_prob",
]


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


def load_joined_rows(v3_dir: Path, reference_path: Path) -> tuple[pd.DataFrame, dict]:
    paths = sorted(v3_dir.glob("rich_transfer_v3_NBA*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No v3 scored files under {v3_dir}")
    v3 = pd.concat(
        [pd.read_parquet(path, columns=V3_COLUMNS) for path in paths],
        ignore_index=True,
    )
    if v3.duplicated(KEYS).any():
        raise ValueError("Duplicate v3 shot identities.")
    source_files = sorted(v3["source_file"].astype(str).unique().tolist())
    reference_dataset = ds.dataset(reference_path, format="parquet")
    reference = reference_dataset.to_table(
        columns=REFERENCE_COLUMNS,
        filter=ds.field("source_file").isin(source_files),
    ).to_pandas()
    if reference.duplicated(KEYS).any():
        raise ValueError("Duplicate reference shot identities.")
    joined = v3.merge(
        reference,
        on=KEYS,
        how="left",
        suffixes=("_v3", "_reference"),
        indicator=True,
        validate="one_to_one",
    )
    coverage = {
        "v3_files": len(paths),
        "source_files": source_files,
        "v3_rows": int(len(v3)),
        "joined_rows": int(joined["_merge"].eq("both").sum()),
        "joined_pct": float(100.0 * joined["_merge"].eq("both").mean()),
    }
    if not joined["_merge"].eq("both").all():
        missing = joined.loc[~joined["_merge"].eq("both"), KEYS].head(10).to_dict("records")
        raise ValueError(f"Reference join missed v3 rows; examples={missing}")
    y_v3 = pd.to_numeric(joined["is_make_v3"], errors="coerce")
    y_reference = pd.to_numeric(joined["is_make_reference"], errors="coerce")
    mismatch = y_v3.ne(y_reference)
    if mismatch.any():
        raise ValueError(f"Outcome mismatch on {int(mismatch.sum())} joined shots.")
    value_mismatch = pd.to_numeric(joined["shot_value_v3"], errors="coerce").ne(
        pd.to_numeric(joined["shot_value_reference"], errors="coerce")
    )
    if value_mismatch.any():
        raise ValueError(f"Shot-value mismatch on {int(value_mismatch.sum())} joined shots.")
    joined["is_make"] = y_reference.astype(int)
    joined["shot_value"] = pd.to_numeric(joined["shot_value_reference"], errors="coerce")
    joined = joined.drop(columns=["_merge"])
    return joined, coverage


def prepare_model_frame(rows: pd.DataFrame, *, rare_minimum: int = 100) -> pd.DataFrame:
    out = rows.copy()
    for column in NUMERIC_FEATURES:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        values = out[column].astype("string").fillna("missing")
        counts = values.value_counts(dropna=False)
        values = values.where(values.map(counts).fillna(0).ge(rare_minimum), "__rare__")
        out[column] = values.astype("category")
        if len(out[column].cat.categories) > 254:
            raise ValueError(f"Categorical feature {column} has too many levels.")
    return out


def fit_game_group_oof(
    rows: pd.DataFrame,
    *,
    folds: int = 5,
    random_seed: int = 20260723,
) -> tuple[np.ndarray, dict[str, HistGradientBoostingClassifier]]:
    predictions = np.full(len(rows), np.nan, dtype=np.float32)
    models: dict[str, HistGradientBoostingClassifier] = {}
    feature_columns = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    for bucket in MODEL_BUCKETS:
        positions = np.flatnonzero(rows["model_bucket"].eq(bucket).to_numpy())
        bucket_rows = rows.iloc[positions]
        # GroupKFold's object-string membership path is quadratic enough to
        # dominate a multi-million-row build. Integer codes preserve exact
        # game grouping and make split construction linear-time in practice.
        game_codes = pd.factorize(bucket_rows["game_id"].astype(str), sort=True)[0]
        splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=random_seed)
        for fold, (train_idx, valid_idx) in enumerate(
            splitter.split(bucket_rows, groups=game_codes)
        ):
            print(
                f"  {bucket} fold {fold + 1}/{folds}: "
                f"train={len(train_idx):,} valid={len(valid_idx):,}",
                flush=True,
            )
            model = HistGradientBoostingClassifier(
                loss="log_loss",
                learning_rate=0.05,
                max_iter=180,
                max_leaf_nodes=15,
                min_samples_leaf=500,
                l2_regularization=25.0,
                max_bins=127,
                categorical_features=CATEGORICAL_FEATURES,
                early_stopping=False,
                random_state=random_seed + fold,
            )
            model.fit(
                bucket_rows.iloc[train_idx][feature_columns],
                bucket_rows.iloc[train_idx]["is_make"].astype(int),
            )
            predictions[positions[valid_idx]] = model.predict_proba(
                bucket_rows.iloc[valid_idx][feature_columns]
            )[:, 1]
            print(f"    completed {bucket} fold {fold + 1}/{folds}", flush=True)
        full_model = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.05,
            max_iter=180,
            max_leaf_nodes=15,
            min_samples_leaf=500,
            l2_regularization=25.0,
            max_bins=127,
            categorical_features=CATEGORICAL_FEATURES,
            early_stopping=False,
            random_state=random_seed,
        )
        full_model.fit(bucket_rows[feature_columns], bucket_rows["is_make"].astype(int))
        models[bucket] = full_model
    if np.isnan(predictions).any():
        raise ValueError("OOF predictions contain missing values.")
    return np.clip(predictions.astype(float), 1e-5, 1 - 1e-5), models


def fit_temporal_forward_predictions(
    rows: pd.DataFrame,
    *,
    random_seed: int = 20260723,
) -> tuple[np.ndarray, list[dict]]:
    """Score era blocks with stackers fit earlier; the anchor stays retrospective."""
    predictions = np.full(len(rows), np.nan, dtype=np.float32)
    audit: list[dict] = []
    feature_columns = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    blocks = [
        (1997, 1998, 2003),
        (2003, 2004, 2013),
        (2013, 2014, 2023),
        (2023, 2024, 2026),
    ]
    for train_through, valid_start, valid_end in blocks:
        for bucket in MODEL_BUCKETS:
            train_mask = rows["model_bucket"].eq(bucket) & rows["season_end_year"].le(
                train_through
            )
            valid_mask = (
                rows["model_bucket"].eq(bucket)
                & rows["season_end_year"].between(valid_start, valid_end)
            )
            if not train_mask.any() or not valid_mask.any():
                continue
            model = HistGradientBoostingClassifier(
                loss="log_loss",
                learning_rate=0.05,
                max_iter=180,
                max_leaf_nodes=15,
                min_samples_leaf=150,
                l2_regularization=10.0,
                max_bins=127,
                categorical_features=CATEGORICAL_FEATURES,
                early_stopping=False,
                random_state=random_seed + valid_start,
            )
            model.fit(
                rows.loc[train_mask, feature_columns],
                rows.loc[train_mask, "is_make"].astype(int),
            )
            predictions[valid_mask.to_numpy()] = model.predict_proba(
                rows.loc[valid_mask, feature_columns]
            )[:, 1]
            print(
                f"  temporal {bucket}: train through {train_through}, "
                f"score {valid_start}-{valid_end}",
                flush=True,
            )
            audit.append(
                {
                    "model_bucket": bucket,
                    "train_through": train_through,
                    "validation_start": valid_start,
                    "validation_end": valid_end,
                    "train_rows": int(train_mask.sum()),
                    "validation_rows": int(valid_mask.sum()),
                    "train_validation_game_overlap": int(
                        len(
                            set(rows.loc[train_mask, "game_id"].astype(str))
                            & set(rows.loc[valid_mask, "game_id"].astype(str))
                        )
                    ),
                }
            )
    return predictions.astype(float), audit


def metric_record(y: pd.Series, probability: pd.Series) -> dict[str, float]:
    eligible = y.notna() & probability.notna()
    target = y.loc[eligible].astype(int)
    pred = probability.loc[eligible].astype(float).clip(1e-5, 1 - 1e-5)
    return {
        "shots": int(eligible.sum()),
        "brier": float(brier_score_loss(target, pred)),
        "log_loss": float(log_loss(target, pred, labels=[0, 1])),
        "calibration_bias": float(pred.mean() - target.mean()),
    }


def build_evaluation(
    rows: pd.DataFrame,
    *,
    bootstrap_repetitions: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics: list[dict] = []
    comparisons: list[dict] = []
    candidates = {
        "retrospective_v4_game_group_oof": "retrospective_v4_make_prob",
        "retrospective_v4_temporal_forward": "retrospective_v4_temporal_make_prob",
        **REFERENCE_PROBABILITIES,
    }
    for (era, bucket), group in rows.groupby(["era", "model_bucket"], observed=True):
        y = group["is_make"].astype(int)
        for model_name, column in candidates.items():
            record = metric_record(y, group[column])
            metrics.append({"era": era, "model_bucket": bucket, "model": model_name, **record})
        for evaluation, candidate_column in [
            ("game_group_oof", "retrospective_v4_make_prob"),
            ("temporal_forward", "retrospective_v4_temporal_make_prob"),
        ]:
            v4 = group[candidate_column].astype(float)
            for competitor, column in REFERENCE_PROBABILITIES.items():
                eligible = y.notna() & v4.notna() & group[column].notna()
                if not eligible.any():
                    continue
                yy = y.loc[eligible].to_numpy()
                candidate = v4.loc[eligible].to_numpy()
                baseline = group.loc[eligible, column].astype(float).to_numpy()
                interval = paired_game_cluster_interval(
                    yy,
                    candidate,
                    baseline,
                    group.loc[eligible, "game_id"].astype(str).to_numpy(),
                    repetitions=bootstrap_repetitions,
                    seed=20260723 + len(comparisons),
                )
                candidate_brier = float(brier_score_loss(yy, candidate))
                baseline_brier = float(brier_score_loss(yy, baseline))
                candidate_log = float(log_loss(yy, candidate, labels=[0, 1]))
                baseline_log = float(log_loss(yy, baseline, labels=[0, 1]))
                comparisons.append(
                    {
                        "evaluation": evaluation,
                        "era": era,
                        "model_bucket": bucket,
                        "competitor": competitor,
                        "shots": int(eligible.sum()),
                        "v4_brier": candidate_brier,
                        "competitor_brier": baseline_brier,
                        "brier_improvement": baseline_brier - candidate_brier,
                        "v4_log_loss": candidate_log,
                        "competitor_log_loss": baseline_log,
                        "log_loss_improvement": baseline_log - candidate_log,
                        **interval,
                    }
                )
    return pd.DataFrame(metrics), pd.DataFrame(comparisons)


def write_split_outputs(rows: pd.DataFrame, output_dir: Path) -> None:
    output_columns = [
        *KEYS,
        "shot_id",
        "season_end_year",
        "season_type",
        "model_bucket",
        "shot_value",
        "is_make",
        "historical_sq_make_prob",
        "shot_quality_with_defense_make_prob",
        "retrospective_v4_make_prob",
        "retrospective_v4_temporal_make_prob",
        "retrospective_v4_ev",
    ]
    for source_file, group in rows.groupby("source_file", sort=True, observed=True):
        stem = Path(str(source_file)).stem
        path = output_dir / f"retrospective_v4_{stem}.parquet"
        group[output_columns].to_parquet(path, index=False)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-dir", type=Path, default=DEFAULT_V3_DIR)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-repetitions", type=int, default=500)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Permit fewer than the expected 60 RS/PS v3 source files.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    paths = sorted(args.v3_dir.glob("rich_transfer_v3_NBA*.parquet"))
    if len(paths) != 60 and not args.allow_partial:
        raise ValueError(
            f"Expected 60 v3 RS/PS files for 1997-2026; found {len(paths)}. "
            "Use --allow-partial only for development."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading and joining {len(paths)} v3 files...")
    rows, coverage = load_joined_rows(args.v3_dir, args.reference)
    print(f"Joined {len(rows):,} midrange/three attempts ({coverage['joined_pct']:.3f}%).")
    rows = prepare_model_frame(rows)
    print(f"Fitting {args.folds}-fold game-group OOF stacker...")
    predictions, models = fit_game_group_oof(rows, folds=args.folds)
    rows["retrospective_v4_make_prob"] = predictions
    print("Fitting earlier-season temporal stacker benchmarks...")
    temporal_predictions, temporal_audit = fit_temporal_forward_predictions(rows)
    rows["retrospective_v4_temporal_make_prob"] = temporal_predictions
    rows["retrospective_v4_ev"] = predictions * rows["shot_value"].astype(float)
    rows["era"] = rows["season_end_year"].map(era_label)

    metrics, comparisons = build_evaluation(
        rows,
        bootstrap_repetitions=args.bootstrap_repetitions,
    )
    metrics.to_csv(args.output_dir / "retrospective_v4_metrics.csv", index=False)
    comparisons.to_csv(args.output_dir / "retrospective_v4_comparisons.csv", index=False)
    combined_columns = [
        *KEYS,
        "shot_id",
        "season_end_year",
        "season_type",
        "model_bucket",
        "shot_value",
        "is_make",
        *REFERENCE_PROBABILITIES.values(),
        "retrospective_v4_make_prob",
        "retrospective_v4_temporal_make_prob",
        "retrospective_v4_ev",
    ]
    combined_columns = list(dict.fromkeys(combined_columns))
    rows[combined_columns].to_parquet(
        args.output_dir / "retrospective_v4_1997_2026.parquet",
        index=False,
    )
    write_split_outputs(rows, args.output_dir)
    joblib.dump(
        {
            "models": models,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "contract": "retrospective_single_defense_loo_anchor",
        },
        args.output_dir / "retrospective_v4_models.joblib",
    )
    coverage.update(
        {
            "output_rows": int(len(rows)),
            "midrange_rows": int(rows["model_bucket"].eq("midrange").sum()),
            "three_rows": int(rows["model_bucket"].eq("three").sum()),
            "duplicate_keys": int(rows.duplicated(KEYS).sum()),
            "oof_nulls": int(rows["retrospective_v4_make_prob"].isna().sum()),
            "oof_out_of_bounds": int(
                (~rows["retrospective_v4_make_prob"].between(0, 1)).sum()
            ),
            "folds": int(args.folds),
        }
    )
    write_json_atomic(args.output_dir / "retrospective_v4_coverage.json", coverage)
    write_json_atomic(
        args.output_dir / "retrospective_v4_temporal_split_audit.json",
        temporal_audit,
    )
    print(metrics.to_string(index=False))
    print(comparisons.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
