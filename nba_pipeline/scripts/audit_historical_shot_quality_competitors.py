#!/usr/bin/env python3
"""Compare frozen historical SQ scores with repo-owned retrospective references.

Every comparison uses identical joined shots. Positive improvement means the
frozen ``rich_transfer_v3`` historical student has lower loss. The repo-owned
context, season/talent, and empirical references use future/full-era
information and are therefore labeled retrospective rather than leakage-safe
deployment baselines.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORED_DIR = (
    PIPELINE_ROOT / "results" / "historical_shot_quality" / "rich_transfer_v3_scored"
)
DEFAULT_PBP_SQ = (
    PIPELINE_ROOT / "results" / "pbp_shot_quality" / "pbp_shot_quality_1997_2026.parquet"
)
KEYS = ["source_file", "game_id", "event_num"]
PROBABILITY_COLUMNS = {
    "repo_shot_context": "context_probability",
    "repo_shot_quality_season": "season_talent_probability",
    "repo_shot_quality_with_defense": "defense_adjusted_probability",
    "era_bucket_empirical_loo": "era_bucket_empirical_probability",
    "era_zone_empirical_loo": "era_zone_empirical_probability",
}


def era_label(season: int) -> str:
    if season <= 2003:
        return "1997-2003"
    if season <= 2013:
        return "2004-2013"
    if season <= 2023:
        return "2014-2023"
    return "2024-2026"


def write_json_atomic(path: Path, payload: dict) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def paired_game_cluster_interval(
    y: np.ndarray,
    candidate_probability: np.ndarray,
    baseline_probability: np.ndarray,
    game_ids: np.ndarray,
    *,
    repetitions: int = 1000,
    seed: int = 20260723,
) -> dict[str, float]:
    """Shot-weighted paired bootstrap that resamples complete games."""
    candidate = np.clip(candidate_probability.astype(float), 1e-5, 1 - 1e-5)
    baseline = np.clip(baseline_probability.astype(float), 1e-5, 1 - 1e-5)
    target = y.astype(int)
    brier_delta = (target - baseline) ** 2 - (target - candidate) ** 2
    baseline_log = -(target * np.log(baseline) + (1 - target) * np.log(1 - baseline))
    candidate_log = -(target * np.log(candidate) + (1 - target) * np.log(1 - candidate))
    grouped = (
        pd.DataFrame(
            {
                "game_id": game_ids.astype(str),
                "brier_delta": brier_delta,
                "log_delta": baseline_log - candidate_log,
            }
        )
        .groupby("game_id", observed=True)
        .agg(
            brier_sum=("brier_delta", "sum"),
            log_sum=("log_delta", "sum"),
            shots=("brier_delta", "size"),
        )
    )
    if len(grouped) < 2:
        return {
            "brier_improvement_ci_low": np.nan,
            "brier_improvement_ci_high": np.nan,
            "log_loss_improvement_ci_low": np.nan,
            "log_loss_improvement_ci_high": np.nan,
        }

    rng = np.random.default_rng(seed)
    counts = grouped["shots"].to_numpy(dtype=float)
    brier_sums = grouped["brier_sum"].to_numpy(dtype=float)
    log_sums = grouped["log_sum"].to_numpy(dtype=float)
    brier_samples: list[np.ndarray] = []
    log_samples: list[np.ndarray] = []
    remaining = repetitions
    while remaining:
        batch = min(100, remaining)
        sample_idx = rng.integers(0, len(grouped), size=(batch, len(grouped)))
        denominator = counts[sample_idx].sum(axis=1)
        brier_samples.append(brier_sums[sample_idx].sum(axis=1) / denominator)
        log_samples.append(log_sums[sample_idx].sum(axis=1) / denominator)
        remaining -= batch
    brier_bootstrap = np.concatenate(brier_samples)
    log_bootstrap = np.concatenate(log_samples)
    return {
        "brier_improvement_ci_low": float(np.quantile(brier_bootstrap, 0.025)),
        "brier_improvement_ci_high": float(np.quantile(brier_bootstrap, 0.975)),
        "log_loss_improvement_ci_low": float(np.quantile(log_bootstrap, 0.025)),
        "log_loss_improvement_ci_high": float(np.quantile(log_bootstrap, 0.975)),
    }


def add_empirical_probabilities(rows: pd.DataFrame, *, prior_strength: float = 200.0) -> pd.DataFrame:
    """Add retrospective leave-one-shot-out era/bucket and era/zone baselines."""
    out = rows.copy()
    y = pd.to_numeric(out["is_make"], errors="coerce")
    bucket_totals = (
        out.assign(_make=y)
        .groupby(["era", "model_bucket"], observed=True)["_make"]
        .agg(["count", "sum"])
        .rename(columns={"count": "_bucket_count", "sum": "_bucket_makes"})
        .reset_index()
    )
    out = out.merge(bucket_totals, on=["era", "model_bucket"], how="left", validate="many_to_one")
    global_prior = out["model_bucket"].map({"midrange": 0.41, "three": 0.36}).fillna(0.40)
    out["era_bucket_empirical_probability"] = (
        out["_bucket_makes"] - y + prior_strength * global_prior
    ) / (out["_bucket_count"] - 1 + prior_strength)

    zone_totals = (
        out.assign(_make=y)
        .groupby(["era", "model_bucket", "shot_zone"], observed=True, dropna=False)["_make"]
        .agg(["count", "sum"])
        .rename(columns={"count": "_zone_count", "sum": "_zone_makes"})
        .reset_index()
    )
    out = out.merge(
        zone_totals,
        on=["era", "model_bucket", "shot_zone"],
        how="left",
        validate="many_to_one",
    )
    out["era_zone_empirical_probability"] = (
        out["_zone_makes"] - y + prior_strength * out["era_bucket_empirical_probability"]
    ) / (out["_zone_count"] - 1 + prior_strength)
    return out.drop(
        columns=["_bucket_count", "_bucket_makes", "_zone_count", "_zone_makes"]
    )


def load_identical_rows(scored_dir: Path, pbp_sq_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored_paths = sorted(scored_dir.glob("rich_transfer_v3_NBA*.parquet"))
    if not scored_paths:
        raise FileNotFoundError(f"No frozen student outputs found under {scored_dir}")
    scored = pd.concat(
        [
            pd.read_parquet(
                path,
                columns=[
                    *KEYS,
                    "season_end_year",
                    "season_type",
                    "model_bucket",
                    "is_make",
                    "historical_sq_make_prob",
                ],
            )
            for path in scored_paths
        ],
        ignore_index=True,
    )
    if scored.duplicated(KEYS).any():
        raise ValueError("Frozen student outputs contain duplicate shot keys.")

    reference = pd.read_parquet(
        pbp_sq_path,
        columns=[
            *KEYS,
            "shot_value",
            "shot_zone",
            "is_make",
            "shot_context_ev",
            "shot_quality_season_ev",
            "shot_quality_with_defense_ev",
        ],
    )
    if reference.duplicated(KEYS).any():
        raise ValueError("Repo-owned PBP SQ artifact contains duplicate shot keys.")
    joined = scored.merge(
        reference,
        on=KEYS,
        how="left",
        suffixes=("_v3", "_reference"),
        indicator=True,
        validate="one_to_one",
    )
    coverage = (
        joined.groupby(["season_end_year", "season_type", "model_bucket"], observed=True)
        .agg(
            v3_rows=("game_id", "size"),
            joined_rows=("_merge", lambda values: int(values.eq("both").sum())),
            context_ev_rows=("shot_context_ev", lambda values: int(values.notna().sum())),
            season_talent_ev_rows=(
                "shot_quality_season_ev", lambda values: int(values.notna().sum())
            ),
            defense_adjusted_ev_rows=(
                "shot_quality_with_defense_ev", lambda values: int(values.notna().sum())
            ),
            outcome_rows=("is_make_reference", lambda values: int(values.notna().sum())),
        )
        .reset_index()
    )
    for numerator in [
        "joined_rows",
        "context_ev_rows",
        "season_talent_ev_rows",
        "defense_adjusted_ev_rows",
        "outcome_rows",
    ]:
        coverage[f"{numerator}_pct"] = 100.0 * coverage[numerator] / coverage["v3_rows"]

    matched = joined[joined["_merge"].eq("both")].copy()
    mismatch = (
        pd.to_numeric(matched["is_make_v3"], errors="coerce")
        != pd.to_numeric(matched["is_make_reference"], errors="coerce")
    )
    if mismatch.any():
        raise ValueError(f"Recorded make/miss mismatch on {int(mismatch.sum())} joined shots.")
    matched["is_make"] = pd.to_numeric(matched["is_make_reference"], errors="coerce")
    matched["context_probability"] = (
        pd.to_numeric(matched["shot_context_ev"], errors="coerce")
        / pd.to_numeric(matched["shot_value"], errors="coerce")
    ).clip(1e-5, 1 - 1e-5)
    matched["season_talent_probability"] = (
        pd.to_numeric(matched["shot_quality_season_ev"], errors="coerce")
        / pd.to_numeric(matched["shot_value"], errors="coerce")
    ).clip(1e-5, 1 - 1e-5)
    matched["defense_adjusted_probability"] = (
        pd.to_numeric(matched["shot_quality_with_defense_ev"], errors="coerce")
        / pd.to_numeric(matched["shot_value"], errors="coerce")
    ).clip(1e-5, 1 - 1e-5)
    matched["era"] = matched["season_end_year"].map(era_label)
    matched = add_empirical_probabilities(matched)
    return matched, coverage


def competitor_comparisons(
    joined: pd.DataFrame,
    *,
    bootstrap_repetitions: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for (era, bucket), group in joined.groupby(["era", "model_bucket"], observed=True):
        for competitor, probability_column in PROBABILITY_COLUMNS.items():
            eligible = group[
                group["is_make"].notna()
                & group["historical_sq_make_prob"].notna()
                & group[probability_column].notna()
            ].copy()
            if eligible.empty:
                continue
            y = eligible["is_make"].astype(int).to_numpy()
            v3 = eligible["historical_sq_make_prob"].astype(float).clip(1e-5, 1 - 1e-5)
            baseline = eligible[probability_column].astype(float).clip(1e-5, 1 - 1e-5)
            v3_brier = float(brier_score_loss(y, v3))
            baseline_brier = float(brier_score_loss(y, baseline))
            v3_log = float(log_loss(y, v3, labels=[0, 1]))
            baseline_log = float(log_loss(y, baseline, labels=[0, 1]))
            brier_improvement = baseline_brier - v3_brier
            log_improvement = baseline_log - v3_log
            intervals = paired_game_cluster_interval(
                y,
                v3.to_numpy(),
                baseline.to_numpy(),
                eligible["game_id"].astype(str).to_numpy(),
                repetitions=bootstrap_repetitions,
                seed=20260723 + len(rows),
            )
            rows.append(
                {
                    "era": era,
                    "model_bucket": bucket,
                    "competitor": competitor,
                    "shots": int(len(eligible)),
                    "games": int(eligible["game_id"].nunique()),
                    "v3_brier": v3_brier,
                    "competitor_brier": baseline_brier,
                    "brier_improvement_v3_vs_competitor": brier_improvement,
                    "v3_log_loss": v3_log,
                    "competitor_log_loss": baseline_log,
                    "log_loss_improvement_v3_vs_competitor": log_improvement,
                    **intervals,
                    "v3_loses_brier": bool(brier_improvement < 0),
                    "v3_loses_log_loss": bool(log_improvement < 0),
                    "v3_clear_brier_win": bool(
                        intervals["brier_improvement_ci_low"] > 0
                    ),
                    "v3_clear_log_loss_win": bool(
                        intervals["log_loss_improvement_ci_low"] > 0
                    ),
                    "competitor_is_retrospective": True,
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-dir", type=Path, default=DEFAULT_SCORED_DIR)
    parser.add_argument("--pbp-sq", type=Path, default=DEFAULT_PBP_SQ)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    joined, coverage = load_identical_rows(args.scored_dir, args.pbp_sq)
    comparison = competitor_comparisons(
        joined, bootstrap_repetitions=args.bootstrap_repetitions
    )
    coverage_path = args.scored_dir / "competitor_join_coverage.csv"
    comparison_path = args.scored_dir / "competitor_comparison.csv"
    coverage.to_csv(coverage_path, index=False)
    comparison.to_csv(comparison_path, index=False)

    summary = {
        "joined_rows": int(len(joined)),
        "comparison_rows": int(len(comparison)),
        "v3_losing_comparisons_brier": int(comparison["v3_loses_brier"].sum()),
        "v3_losing_comparisons_log_loss": int(comparison["v3_loses_log_loss"].sum()),
        "retrospective_reference_caveat": (
            "Repo shot_context_ev, shot_quality_season_ev, lineup-defense adjustment, and "
            "empirical LOO references use future/full-era outcomes and are not leakage-safe "
            "deployment baselines."
        ),
        "coverage_path": str(coverage_path),
        "comparison_path": str(comparison_path),
    }
    summary_path = args.scored_dir / "competitor_audit_summary.json"
    write_json_atomic(summary_path, summary)

    manifest_path = args.scored_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("audit_paths", {}).update(
            {
                "competitor_join_coverage": str(coverage_path),
                "competitor_comparison": str(comparison_path),
                "competitor_audit_summary": str(summary_path),
            }
        )
        manifest["competitor_audit"] = summary
        write_json_atomic(manifest_path, manifest)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
