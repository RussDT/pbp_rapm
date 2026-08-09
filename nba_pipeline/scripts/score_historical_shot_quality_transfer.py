#!/usr/bin/env python3
"""Production backfill for the frozen ``rich_transfer_v3`` historical SQ student.

The scorer streams seasons in chronological order so the shooter prior for a
game uses only earlier games. Outputs are written atomically, one RS/PS season
at a time, and can be resumed safely when the model and source fingerprints
match their sidecars.

Audits separate two ideas:

- distribution/calibration proxies, which do not require outcomes (student
  component disagreement and held-out support drift);
- retrospective outcome diagnostics, written only when recorded make/miss
  labels are present. They are never fabricated for unlabeled rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import brier_score_loss, log_loss

from nba_pipeline.scripts.build_historical_shot_quality_transfer import (
    HISTORICAL_CATEGORICAL,
    HISTORICAL_NUMERIC,
    MODEL_BUCKETS,
    load_feature_rows,
    score_student_rows,
)
from nba_pipeline.scripts.build_pbp_shot_quality import discover_raw_files, file_metadata


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PIPELINE_ROOT / "raw_data"
DEFAULT_MODEL = (
    PIPELINE_ROOT
    / "validation"
    / "historical_shot_quality"
    / "rich_transfer_v3_models.joblib"
)
DEFAULT_HELDOUT = (
    PIPELINE_ROOT
    / "validation"
    / "historical_shot_quality"
    / "rich_transfer_v3_heldout_predictions.parquet"
)
DEFAULT_OUTPUT_DIR = (
    PIPELINE_ROOT / "results" / "historical_shot_quality" / "rich_transfer_v3_scored"
)
OUTPUT_SCHEMA_VERSION = 2
FIXED_PRIORS = {"midrange": 0.41, "three": 0.36}
OUTPUT_COLUMNS = [
    "source_file",
    "season_end_year",
    "season_type",
    "game_id",
    "event_num",
    "shot_id",
    "period",
    "seconds_remaining_quarter",
    "shooter_id",
    "shooter_name",
    "shooter_team_id",
    "model_bucket",
    "shot_value",
    "is_make",
    "shot_distance_ft",
    "distance_source",
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
    "historical_sq_ev",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "source_file": path.name,
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
    }


def sidecar_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".manifest.json")


def write_json_atomic(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def add_streaming_pregame_shooter_priors(
    rows: pd.DataFrame,
    state: dict[tuple[int, str], tuple[float, float]],
    *,
    shrink: float = 200.0,
) -> pd.DataFrame:
    """Add game-excluded priors and update career state after the final game."""
    out = rows.copy()
    out["shooter_pregame_make_rate"] = out["model_bucket"].map(FIXED_PRIORS).fillna(0.40)
    out["shooter_pregame_log_attempts"] = 0.0
    if out.empty:
        return out

    valid = (
        pd.to_numeric(out["shooter_id"], errors="coerce").notna()
        & out["model_bucket"].isin(MODEL_BUCKETS)
        & out["is_make"].notna()
    )
    if not valid.any():
        return out

    game_stats = (
        out.loc[valid]
        .assign(
            shooter_id=lambda frame: pd.to_numeric(frame["shooter_id"], errors="coerce").astype(
                np.int64
            ),
            made=lambda frame: frame["is_make"].astype(float),
        )
        .groupby(["shooter_id", "model_bucket", "game_id"], as_index=False, observed=True)
        .agg(game_attempts=("made", "size"), game_makes=("made", "sum"))
        .sort_values(["shooter_id", "model_bucket", "game_id"], kind="stable")
    )
    game_stats["_key"] = list(
        zip(game_stats["shooter_id"].astype(int), game_stats["model_bucket"].astype(str))
    )
    game_stats["start_attempts"] = game_stats["_key"].map(
        lambda key: state.get(key, (0.0, 0.0))[0]
    )
    game_stats["start_makes"] = game_stats["_key"].map(
        lambda key: state.get(key, (0.0, 0.0))[1]
    )
    grouped = game_stats.groupby(["shooter_id", "model_bucket"], observed=True)
    game_stats["prior_attempts"] = (
        grouped["game_attempts"].cumsum() - game_stats["game_attempts"] + game_stats["start_attempts"]
    )
    game_stats["prior_makes"] = (
        grouped["game_makes"].cumsum() - game_stats["game_makes"] + game_stats["start_makes"]
    )
    game_stats["fixed_prior"] = game_stats["model_bucket"].map(FIXED_PRIORS).fillna(0.40)
    game_stats["shooter_pregame_make_rate"] = (
        game_stats["prior_makes"] + shrink * game_stats["fixed_prior"]
    ) / (game_stats["prior_attempts"] + shrink)
    game_stats["shooter_pregame_log_attempts"] = np.log1p(game_stats["prior_attempts"])

    keys = ["shooter_id", "model_bucket", "game_id"]
    prior_columns = ["shooter_pregame_make_rate", "shooter_pregame_log_attempts"]
    out = out.drop(columns=prior_columns).merge(
        game_stats[keys + prior_columns],
        on=keys,
        how="left",
        validate="many_to_one",
    )
    out["shooter_pregame_make_rate"] = out["shooter_pregame_make_rate"].fillna(
        out["model_bucket"].map(FIXED_PRIORS).fillna(0.40)
    )
    out["shooter_pregame_log_attempts"] = out["shooter_pregame_log_attempts"].fillna(0.0)

    totals = (
        game_stats.groupby(["shooter_id", "model_bucket"], observed=True)
        .agg(attempts=("game_attempts", "sum"), makes=("game_makes", "sum"))
        .reset_index()
    )
    for row in totals.itertuples(index=False):
        key = (int(row.shooter_id), str(row.model_bucket))
        previous_attempts, previous_makes = state.get(key, (0.0, 0.0))
        state[key] = (
            float(previous_attempts + row.attempts),
            float(previous_makes + row.makes),
        )
    return out


def output_is_resumable(
    output_path: Path,
    *,
    expected_rows: int,
    model_sha256: str,
    source: dict[str, int | str],
) -> tuple[bool, str]:
    manifest_path = sidecar_path(output_path)
    if not output_path.exists() or not manifest_path.exists():
        return False, "output_or_sidecar_missing"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False, "sidecar_unreadable"
    expected = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "model_sha256": model_sha256,
        "rows": expected_rows,
        **source,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            return False, f"sidecar_{key}_mismatch"
    try:
        parquet = pq.ParquetFile(output_path)
        if parquet.metadata.num_rows != expected_rows:
            return False, "parquet_row_count_mismatch"
        columns = set(parquet.schema_arrow.names)
        missing = set(OUTPUT_COLUMNS) - columns
        if missing:
            return False, f"parquet_columns_missing:{','.join(sorted(missing))}"
        probabilities = pd.read_parquet(
            output_path,
            columns=[
                "historical_sq_make_prob",
                "historical_sq_direct_make_prob",
                "historical_sq_distilled_make_prob",
            ],
        )
        if probabilities.isna().any().any():
            return False, "probability_nulls"
        if not probabilities.apply(lambda column: column.between(0, 1).all()).all():
            return False, "probability_out_of_bounds"
    except Exception as exc:
        return False, f"parquet_validation_error:{exc}"
    return True, "valid"


def missingness_records(rows: pd.DataFrame, season: int, season_type: str) -> list[dict]:
    records: list[dict] = []
    features = [*HISTORICAL_CATEGORICAL, *HISTORICAL_NUMERIC]
    for bucket, group in rows.groupby("model_bucket", observed=True):
        for feature in features:
            series = group[feature] if feature in group else pd.Series(np.nan, index=group.index)
            if feature in HISTORICAL_CATEGORICAL:
                normalized = series.astype("string").str.strip().str.lower()
                # Literal "none" is a meaningful historical category for
                # modifier_bundle (no parsed modifiers), not missing data.
                missing = series.isna() | normalized.isin({"", "missing", "nan", "<na>"})
            else:
                missing = pd.to_numeric(series, errors="coerce").isna()
            records.append(
                {
                    "season_end_year": season,
                    "season_type": season_type,
                    "model_bucket": str(bucket),
                    "feature": feature,
                    "rows": int(len(group)),
                    "missing_rows": int(missing.sum()),
                    "missing_pct": float(100.0 * missing.mean()) if len(group) else np.nan,
                }
            )
    return records


def era_label(season: int) -> str:
    if season <= 2003:
        return "1997-2003"
    if season <= 2013:
        return "2004-2013"
    if season <= 2023:
        return "2014-2023"
    return "2024-2026"


def distribution_summary(
    group: pd.DataFrame,
    *,
    scope: str,
    scope_value: str,
    season_type: str,
    bucket: str,
) -> dict:
    probability = pd.to_numeric(group["historical_sq_make_prob"], errors="coerce").dropna()
    return {
        "scope": scope,
        "scope_value": scope_value,
        "season_type": season_type,
        "model_bucket": bucket,
        "shots": int(len(group)),
        "mean_probability": float(probability.mean()),
        "std_probability": float(probability.std()),
        "p01_probability": float(probability.quantile(0.01)),
        "p05_probability": float(probability.quantile(0.05)),
        "p50_probability": float(probability.quantile(0.50)),
        "p95_probability": float(probability.quantile(0.95)),
        "p99_probability": float(probability.quantile(0.99)),
        "min_probability": float(probability.min()),
        "max_probability": float(probability.max()),
    }


def calibration_proxy_summary(
    group: pd.DataFrame,
    *,
    scope: str,
    scope_value: str,
    season_type: str,
    bucket: str,
    support: dict[str, tuple[float, float]],
) -> dict:
    probability = pd.to_numeric(group["historical_sq_make_prob"], errors="coerce")
    direct = pd.to_numeric(group["historical_sq_direct_make_prob"], errors="coerce")
    distilled = pd.to_numeric(group["historical_sq_distilled_make_prob"], errors="coerce")
    gap = (direct - distilled).abs()
    lower, upper = support.get(bucket, (np.nan, np.nan))
    outside = (
        (probability < lower) | (probability > upper)
        if np.isfinite(lower) and np.isfinite(upper)
        else pd.Series(False, index=group.index)
    )
    labels = pd.to_numeric(group.get("is_make"), errors="coerce")
    labeled = labels.notna() & probability.notna()
    result = {
        "scope": scope,
        "scope_value": scope_value,
        "season_type": season_type,
        "model_bucket": bucket,
        "shots": int(len(group)),
        "component_gap_mean": float(gap.mean()),
        "component_gap_p95": float(gap.quantile(0.95)),
        "heldout_support_low": float(lower) if np.isfinite(lower) else np.nan,
        "heldout_support_high": float(upper) if np.isfinite(upper) else np.nan,
        "outside_heldout_support_pct": float(100.0 * outside.mean()),
        "labeled_rows": int(labeled.sum()),
        "label_coverage_pct": float(100.0 * labeled.mean()),
        "outcome_audit_kind": (
            "recorded_make_miss_outcomes" if labeled.any() else "distribution_proxy_only"
        ),
        "recorded_make_rate": np.nan,
        "outcome_calibration_bias": np.nan,
        "outcome_brier": np.nan,
        "outcome_log_loss": np.nan,
    }
    if labeled.any():
        y = labels.loc[labeled].astype(int)
        p = probability.loc[labeled].clip(1e-5, 1 - 1e-5)
        result.update(
            {
                "recorded_make_rate": float(y.mean()),
                "outcome_calibration_bias": float(p.mean() - y.mean()),
                "outcome_brier": float(brier_score_loss(y, p)),
                "outcome_log_loss": float(log_loss(y, p, labels=[0, 1])),
            }
        )
    return result


def load_heldout_support(path: Path | None) -> dict[str, tuple[float, float]]:
    if path is None or not path.exists():
        return {}
    heldout = pd.read_parquet(path, columns=["model_bucket", "student_probability"])
    return {
        str(bucket): (
            float(group["student_probability"].quantile(0.01)),
            float(group["student_probability"].quantile(0.99)),
        )
        for bucket, group in heldout.groupby("model_bucket", observed=True)
    }


def build_audits(
    output_paths: Iterable[Path],
    missing_records_list: list[dict],
    *,
    heldout_support: dict[str, tuple[float, float]],
    output_dir: Path,
) -> dict[str, str]:
    audit_columns = [
        "season_end_year",
        "season_type",
        "model_bucket",
        "is_make",
        "historical_sq_make_prob",
        "historical_sq_direct_make_prob",
        "historical_sq_distilled_make_prob",
    ]
    frames = [pd.read_parquet(path, columns=audit_columns) for path in output_paths]
    if not frames:
        raise RuntimeError("No scored outputs are available for audits.")
    audit = pd.concat(frames, ignore_index=True)
    audit["era"] = audit["season_end_year"].map(era_label)

    distributions: list[dict] = []
    proxies: list[dict] = []
    for (season, season_type, bucket), group in audit.groupby(
        ["season_end_year", "season_type", "model_bucket"], observed=True
    ):
        kwargs = {
            "scope": "season",
            "scope_value": str(int(season)),
            "season_type": str(season_type),
            "bucket": str(bucket),
        }
        distributions.append(distribution_summary(group, **kwargs))
        proxies.append(calibration_proxy_summary(group, support=heldout_support, **kwargs))
    for (era, season_type, bucket), group in audit.groupby(
        ["era", "season_type", "model_bucket"], observed=True
    ):
        kwargs = {
            "scope": "era",
            "scope_value": str(era),
            "season_type": str(season_type),
            "bucket": str(bucket),
        }
        distributions.append(distribution_summary(group, **kwargs))
        proxies.append(calibration_proxy_summary(group, support=heldout_support, **kwargs))

    bins = [0.0, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 1.0]
    audit["probability_bin"] = pd.cut(
        audit["historical_sq_make_prob"], bins=bins, include_lowest=True
    ).astype("string")
    bin_rows: list[dict] = []
    for (era, season_type, bucket, probability_bin), group in audit.groupby(
        ["era", "season_type", "model_bucket", "probability_bin"],
        observed=True,
        dropna=False,
    ):
        probability = pd.to_numeric(group["historical_sq_make_prob"], errors="coerce")
        labels = pd.to_numeric(group["is_make"], errors="coerce")
        labeled = labels.notna() & probability.notna()
        bin_rows.append(
            {
                "era": era,
                "season_type": season_type,
                "model_bucket": bucket,
                "probability_bin": probability_bin,
                "shots": int(len(group)),
                "predicted_make_rate": float(probability.mean()),
                "labeled_rows": int(labeled.sum()),
                "label_coverage_pct": float(100.0 * labeled.mean()),
                "recorded_make_rate": float(labels.loc[labeled].mean()) if labeled.any() else np.nan,
                "outcome_calibration_gap": (
                    float(probability.loc[labeled].mean() - labels.loc[labeled].mean())
                    if labeled.any()
                    else np.nan
                ),
                "outcome_audit_kind": (
                    "recorded_make_miss_outcomes" if labeled.any() else "distribution_proxy_only"
                ),
            }
        )

    missingness = pd.DataFrame(missing_records_list)
    if not missingness.empty:
        missingness["era"] = missingness["season_end_year"].map(era_label)
        era_missing = (
            missingness.groupby(
                ["era", "season_type", "model_bucket", "feature"],
                as_index=False,
                observed=True,
            )[["rows", "missing_rows"]]
            .sum()
        )
        era_missing["missing_pct"] = 100.0 * era_missing["missing_rows"] / era_missing["rows"]
        era_missing["season_end_year"] = pd.NA
        era_missing["scope"] = "era"
        missingness["scope"] = "season"
        missingness = pd.concat([missingness, era_missing], ignore_index=True, sort=False)

    paths = {
        "feature_missingness": output_dir / "feature_missingness.csv",
        "prediction_distribution": output_dir / "prediction_distribution.csv",
        "calibration_proxy": output_dir / "calibration_proxy.csv",
        "probability_bin_audit": output_dir / "probability_bin_audit.csv",
    }
    missingness.to_csv(paths["feature_missingness"], index=False)
    pd.DataFrame(distributions).to_csv(paths["prediction_distribution"], index=False)
    pd.DataFrame(proxies).to_csv(paths["calibration_proxy"], index=False)
    pd.DataFrame(bin_rows).to_csv(paths["probability_bin_audit"], index=False)
    return {key: str(path) for key, path in paths.items()}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=1997)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--season-types", choices=["all", "rs", "ps"], default="all")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--heldout", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if not args.model.exists():
        raise FileNotFoundError(f"Frozen model package not found: {args.model}")
    artifact = joblib.load(args.model)
    if artifact.get("model_buckets") != list(MODEL_BUCKETS):
        raise ValueError(f"Unexpected model buckets: {artifact.get('model_buckets')}")
    model_sha = sha256_file(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_paths = discover_raw_files(
        args.raw_dir,
        set(range(1997, args.end_year + 1)),
        {"rs", "ps"},
    )
    all_paths = sorted(
        all_paths,
        key=lambda path: (
            file_metadata(path)[0],
            0 if file_metadata(path)[1] == "RS" else 1,
        ),
    )
    selected_types = {"RS", "PS"} if args.season_types == "all" else {args.season_types.upper()}
    selected_paths = [
        path
        for path in all_paths
        if args.start_year <= file_metadata(path)[0] <= args.end_year
        and file_metadata(path)[1] in selected_types
    ]
    if not selected_paths:
        raise FileNotFoundError("No selected raw files found.")

    progress_path = args.output_dir / "run_manifest.json"
    run_manifest: dict[str, object] = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "model_path": str(args.model),
        "model_sha256": model_sha,
        "heldout_path": str(args.heldout) if args.heldout else None,
        "start_year": args.start_year,
        "end_year": args.end_year,
        "season_types": sorted(selected_types),
        "expected_outputs": len(selected_paths),
        "files": [],
        "audit_paths": {},
        "notes": (
            "Distribution proxies do not imply outcome calibration. Retrospective outcome "
            "metrics are populated only where raw make/miss labels are present."
        ),
    }
    write_json_atomic(progress_path, run_manifest)

    state: dict[tuple[int, str], tuple[float, float]] = {}
    output_paths: list[Path] = []
    missingness: list[dict] = []
    failures = 0
    selected_set = set(selected_paths)
    for path in all_paths:
        season, season_type = file_metadata(path)
        try:
            feature_rows = load_feature_rows(path, require_sq=False)
            eligible = feature_rows[feature_rows["model_bucket"].isin(MODEL_BUCKETS)].copy()
            eligible = add_streaming_pregame_shooter_priors(eligible, state)
            if path not in selected_set:
                print(f"Warmup {path.name}: {len(eligible):,} eligible shots")
                continue

            missingness.extend(missingness_records(eligible, season, season_type))
            output_path = args.output_dir / f"rich_transfer_v3_{path.stem}.parquet"
            source = source_fingerprint(path)
            resumable, resume_reason = output_is_resumable(
                output_path,
                expected_rows=len(eligible),
                model_sha256=model_sha,
                source=source,
            )
            if resumable and not args.force:
                status = "resumed"
                print(f"Resume {path.name}: {len(eligible):,} shots already valid")
            else:
                scored = score_student_rows(eligible, artifact)
                scored = scored[OUTPUT_COLUMNS].copy()
                if scored["historical_sq_make_prob"].isna().any():
                    raise ValueError("Scorer produced null probabilities.")
                if not scored["historical_sq_make_prob"].between(0, 1).all():
                    raise ValueError("Scorer produced probabilities outside [0, 1].")
                temp = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp.parquet")
                scored.to_parquet(temp, index=False)
                temp.replace(output_path)
                file_manifest = {
                    "schema_version": OUTPUT_SCHEMA_VERSION,
                    "model_sha256": model_sha,
                    "model_path": str(args.model),
                    "rows": int(len(scored)),
                    "midrange_rows": int(scored["model_bucket"].eq("midrange").sum()),
                    "three_rows": int(scored["model_bucket"].eq("three").sum()),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    **source,
                }
                write_json_atomic(sidecar_path(output_path), file_manifest)
                valid, reason = output_is_resumable(
                    output_path,
                    expected_rows=len(eligible),
                    model_sha256=model_sha,
                    source=source,
                )
                if not valid:
                    raise ValueError(f"Post-write validation failed: {reason}")
                status = "scored"
                print(
                    f"Scored {path.name}: {len(scored):,} shots "
                    f"({scored['model_bucket'].eq('midrange').sum():,} mid, "
                    f"{scored['model_bucket'].eq('three').sum():,} three)"
                )
            output_paths.append(output_path)
            run_manifest["files"].append(
                {
                    "source": str(path),
                    "output": str(output_path),
                    "rows": int(len(eligible)),
                    "status": status,
                    "resume_check_before_score": resume_reason,
                }
            )
        except Exception as exc:
            failures += 1
            run_manifest["files"].append(
                {"source": str(path), "status": "failed", "error": str(exc)}
            )
            run_manifest["status"] = "failed"
            write_json_atomic(progress_path, run_manifest)
            if not args.continue_on_error:
                raise
        write_json_atomic(progress_path, run_manifest)

    audit_paths = build_audits(
        output_paths,
        missingness,
        heldout_support=load_heldout_support(args.heldout),
        output_dir=args.output_dir,
    )
    run_manifest["audit_paths"] = audit_paths
    run_manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    run_manifest["completed_outputs"] = len(output_paths)
    run_manifest["failures"] = failures
    run_manifest["status"] = "complete" if failures == 0 and len(output_paths) == len(selected_paths) else "partial"
    write_json_atomic(progress_path, run_manifest)
    print(
        f"Backfill {run_manifest['status']}: {len(output_paths)}/{len(selected_paths)} outputs; "
        f"{failures} failures"
    )
    print("Audits:", ", ".join(audit_paths.values()))
    return 0 if run_manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
