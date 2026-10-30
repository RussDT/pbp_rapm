#!/usr/bin/env python3
"""Rebuild the Russell v2 architecture and score shooter-neutral 3P context.

The frozen ``p_v2`` parquet contains only final talent-aware probabilities;
it cannot be decomposed into shooter talent and shot context after the fact.
This builder reconstructs the documented v2 season-forward architecture from
the surviving PBP, transition, EPM, and career-talent inputs. Each test-season
three is then scored twice by the same fitted model:

1. with its actual as-of shooter-skill inputs;
2. with every shooter-skill input replaced by the test season's league
   reference value while shot context is held fixed.

The resulting player difficulty is the attempt-weighted season neutral
expectation minus the player's neutral-context expectation. Positive means a
harder-than-average three-point attempt mix.
"""

from __future__ import annotations

import argparse
import gc
import json
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, mean_squared_error


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "nba_pipeline"
VPM_DATA_ROOT = REPO_ROOT.parent / "vpm" / "vpm" / "data"
DEFAULT_PBP = (
    PIPELINE_ROOT
    / "results"
    / "pbp_shot_quality"
    / "pbp_shot_quality_1997_2026.parquet"
)
DEFAULT_TRANSITION_DIR = PIPELINE_ROOT / "results" / "sq_transition_backfill"
DEFAULT_PROCESSED_DIR = PIPELINE_ROOT / "processed"
DEFAULT_EPM_DIR = PIPELINE_ROOT / "cache" / "epm"
DEFAULT_CAREER_TALENT = VPM_DATA_ROOT / "career_talent_pre15.parquet"
DEFAULT_FROZEN_V2 = VPM_DATA_ROOT / "russell_sq_v2_ev_21_26.parquet"
DEFAULT_OUTPUT_DIR = (
    PIPELINE_ROOT / "results" / "sq_three_point_difficulty" / "russell_v2_cf"
)

MID_ZONES = ["short_mid", "long_mid", "deep_two"]
DESCRIPTION_FEATURES = [
    "desc_driving",
    "desc_running",
    "desc_cutting",
    "desc_pullup",
    "desc_step_back",
    "desc_turnaround",
    "desc_fadeaway",
    "desc_bank",
    "desc_alley_oop",
    "desc_putback",
    "desc_reverse",
    "desc_finger_roll",
    "desc_floating",
]
SHOOTER_TALENT_FEATURES = [
    "epm_fg3pct",
    "epm_midpct",
    "epm_ftpct",
    "epm_fg3a",
    "sh_zone_rate",
    "sh_zone_n",
    "sh_jumper_rate",
]
MODEL_FEATURES = [
    "shot_distance_ft",
    "period",
    "seconds_remaining_quarter",
    "playoffs",
    "transition_probability",
    "action_code",
    "family_code",
] + SHOOTER_TALENT_FEATURES + DESCRIPTION_FEATURES


def neutralize_shooter_features(
    frame: pd.DataFrame,
    reference: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Return a copy with shooter-skill inputs fixed at league references."""
    missing = sorted(set(SHOOTER_TALENT_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing shooter-talent features: {missing}")
    neutral = frame.copy()
    if reference is None:
        reference = {}
        for column in SHOOTER_TALENT_FEATURES:
            values = pd.to_numeric(frame[column], errors="coerce")
            if column == "sh_zone_n":
                value = values.median()
            else:
                value = values.mean()
            reference[column] = float(value) if pd.notna(value) else 0.0
    for column in SHOOTER_TALENT_FEATURES:
        if column not in reference or not np.isfinite(reference[column]):
            raise ValueError(f"Invalid neutral reference for {column}")
        neutral[column] = float(reference[column])
    return neutral


def aggregate_counterfactual_difficulty(scored: pd.DataFrame) -> pd.DataFrame:
    """Aggregate double-scored threes to an exactly centered player season."""
    required = {
        "season_end_year",
        "game_id",
        "shooter_id",
        "shooter_name",
        "made",
        "p_actual_skill",
        "p_neutral_skill",
    }
    missing = sorted(required - set(scored.columns))
    if missing:
        raise ValueError(f"Missing counterfactual columns: {missing}")
    work = scored.copy()
    game_id = work["game_id"].astype(str).str.zfill(10)
    regular = work[game_id.str.startswith("002")].copy()
    if regular.empty:
        raise ValueError("No regular-season game ids beginning 002")
    baseline = regular.groupby("season_end_year")["p_neutral_skill"].mean()
    player = (
        regular.groupby(
            ["season_end_year", "shooter_id", "shooter_name"], sort=False
        )
        .agg(
            v2_attempts=("made", "size"),
            raw_3p_pct=("made", "mean"),
            v2_actual_expected=("p_actual_skill", "mean"),
            v2_neutral_expected=("p_neutral_skill", "mean"),
        )
        .reset_index()
    )
    player["v2_neutral_baseline"] = player["season_end_year"].map(baseline)
    player["v2_difficulty"] = (
        player["v2_neutral_baseline"] - player["v2_neutral_expected"]
    )
    player["v2_effective_3p_pct"] = (
        player["raw_3p_pct"] + player["v2_difficulty"]
    )
    for season, group in player.groupby("season_end_year"):
        centered = np.average(
            group["v2_difficulty"], weights=group["v2_attempts"]
        )
        if abs(float(centered)) > 1e-12:
            raise ValueError(
                f"Counterfactual difficulty is not centered for {season}: {centered}"
            )
    return player


def _load_transition_features(
    transition_dir: Path, first_year: int, last_year: int
) -> pd.DataFrame:
    parts = []
    for path in sorted(transition_dir.glob("transition_backfill_*.parquet")):
        tokens = path.stem.split("_")
        if len(tokens) < 4:
            continue
        try:
            year = int(tokens[2])
        except ValueError:
            continue
        if first_year <= year <= last_year:
            parts.append(
                pd.read_parquet(
                    path,
                    columns=["game_id", "event_num", "transition_probability"],
                )
            )
    if not parts:
        raise FileNotFoundError(
            f"No transition backfills for {first_year}-{last_year} in {transition_dir}"
        )
    transition = pd.concat(parts, ignore_index=True)
    transition["game_id"] = transition["game_id"].astype(str).str.zfill(10)
    if transition.duplicated(["game_id", "event_num"]).any():
        raise ValueError("Transition sidecars contain duplicate shot keys")
    return transition


def _load_game_dates(
    processed_dir: Path, first_year: int, last_year: int
) -> pd.DataFrame:
    parts = []
    for year in range(first_year, last_year + 1):
        for suffix in ["", "_PS"]:
            path = processed_dir / f"FIRST_CHANCE{str(year)[-2:]}{suffix}.parquet"
            if path.exists():
                parts.append(
                    pd.read_parquet(path, columns=["game_id", "game_date"])
                    .drop_duplicates("game_id")
                )
    if not parts:
        raise FileNotFoundError("No FIRST_CHANCE files available for date mapping")
    dates = pd.concat(parts, ignore_index=True).drop_duplicates("game_id")
    dates["game_id"] = dates["game_id"].astype(str).str.zfill(10)
    dates["game_date"] = pd.to_datetime(dates["game_date"])
    return dates


def _load_epm_snapshots(epm_dir: Path) -> pd.DataFrame:
    parts = []
    for filename in sorted(glob(str(epm_dir / "epm_all_*_go0.csv"))):
        requested_date = pd.Timestamp(Path(filename).name.split("_")[2])
        snapshot = pd.read_csv(
            filename,
            usecols=lambda column: column
            in {
                "player_id",
                "p_fg3pct",
                "p_fgpct_mid",
                "p_ftpct",
                "p_fg3a_100",
            },
        )
        snapshot["snapshot_date"] = requested_date
        parts.append(snapshot)
    if not parts:
        raise FileNotFoundError(f"No EPM snapshots found in {epm_dir}")
    snapshots = pd.concat(parts, ignore_index=True).dropna(subset=["player_id"])
    snapshots["player_id"] = snapshots["player_id"].astype(int)
    snapshots["snapshot_date"] = pd.to_datetime(snapshots["snapshot_date"])
    snapshots.loc[
        snapshots["p_fg3a_100"].fillna(0).lt(0.5), "p_fg3pct"
    ] = np.nan
    return snapshots.sort_values("snapshot_date")


def build_feature_table(
    *,
    pbp_path: Path,
    transition_dir: Path,
    processed_dir: Path,
    epm_dir: Path,
    career_talent_path: Path,
    first_training_year: int,
    last_test_year: int,
) -> pd.DataFrame:
    columns = [
        "season_end_year",
        "season_type",
        "game_id",
        "event_num",
        "period",
        "seconds_remaining_quarter",
        "event_action_type",
        "shooter_id",
        "shooter_name",
        "is_make",
        "shot_value",
        "shot_distance_ft",
        "shot_zone",
        "shot_family",
    ] + DESCRIPTION_FEATURES
    shots = pd.read_parquet(
        pbp_path,
        columns=columns,
        filters=[
            ("season_end_year", ">=", first_training_year),
            ("season_end_year", "<=", last_test_year),
        ],
    )
    shots = shots[
        shots["shot_zone"].isin(MID_ZONES + ["three"])
    ].copy()
    shots["made"] = shots["is_make"].astype(int)
    shots["zone"] = np.where(shots["shot_zone"].eq("three"), "three", "mid")
    shots["game_id"] = shots["game_id"].astype(str).str.zfill(10)

    transition = _load_transition_features(
        transition_dir, first_training_year, last_test_year
    )
    shots = shots.merge(
        transition,
        on=["game_id", "event_num"],
        how="left",
        validate="one_to_one",
    )
    dates = _load_game_dates(processed_dir, first_training_year, last_test_year)
    shots = shots.merge(dates, on="game_id", how="left", validate="many_to_one")
    if shots["game_date"].isna().any():
        raise ValueError("Some v2 shots are missing game dates")

    snapshots = _load_epm_snapshots(epm_dir)
    # pandas 2.x requires exact datetime resolutions for merge_asof keys.
    shots["game_date"] = shots["game_date"].astype("datetime64[ns]")
    snapshots["snapshot_date"] = snapshots["snapshot_date"].astype(
        "datetime64[ns]"
    )
    shots["shooter_id"] = (
        pd.to_numeric(shots["shooter_id"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    shots = shots.sort_values("game_date").reset_index(drop=True)
    shots = pd.merge_asof(
        shots,
        snapshots.rename(
            columns={
                "player_id": "shooter_id",
                "snapshot_date": "game_date",
                "p_fg3pct": "epm_fg3pct",
                "p_fgpct_mid": "epm_midpct",
                "p_ftpct": "epm_ftpct",
                "p_fg3a_100": "epm_fg3a",
            }
        ),
        on="game_date",
        by="shooter_id",
        direction="backward",
        allow_exact_matches=False,
    )

    career = (
        pd.read_parquet(career_talent_path)
        .rename(columns={"personId": "shooter_id"})
        .set_index(["shooter_id", "zone"])
    )
    index = pd.MultiIndex.from_frame(shots[["shooter_id", "zone"]])
    shots["car_m"] = career["makes"].reindex(index).fillna(0).to_numpy() * 0.5
    shots["car_n"] = career["att"].reindex(index).fillna(0).to_numpy() * 0.5
    league_by_zone = shots.groupby("zone")["made"].mean().to_dict()
    zone_league = shots["zone"].map(league_by_zone)
    zone_group = shots.groupby(["shooter_id", "zone"], sort=False)["made"]
    prior_zone_makes = zone_group.cumsum() - shots["made"]
    prior_zone_attempts = zone_group.cumcount()
    shots["sh_zone_rate"] = (
        shots["car_m"] + prior_zone_makes + 100.0 * zone_league
    ) / (shots["car_n"] + prior_zone_attempts + 100.0)
    shots["sh_zone_n"] = shots["car_n"] + prior_zone_attempts
    jumper_group = shots.groupby("shooter_id", sort=False)["made"]
    prior_jumper_makes = jumper_group.cumsum() - shots["made"]
    prior_jumper_attempts = jumper_group.cumcount()
    league_jumper = float(shots["made"].mean())
    shots["sh_jumper_rate"] = (
        shots["car_m"] + prior_jumper_makes + 150.0 * league_jumper
    ) / (shots["car_n"] + prior_jumper_attempts + 150.0)

    shots["playoffs"] = shots["season_type"].astype(str).str.lower().eq("ps")
    shots["action_code"] = pd.factorize(shots["event_action_type"])[0]
    shots["family_code"] = pd.factorize(shots["shot_family"])[0]
    return shots


def score_counterfactual_seasons(
    features: pd.DataFrame,
    *,
    first_test_year: int,
    last_test_year: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    threes = features[features["zone"].eq("three")].copy()
    scored_parts = []
    audits: list[dict[str, object]] = []
    for test_year in range(first_test_year, last_test_year + 1):
        train = threes[threes["season_end_year"].lt(test_year)]
        test = threes[threes["season_end_year"].eq(test_year)].copy()
        if train.empty or test.empty:
            raise ValueError(f"Missing v2 train/test rows for {test_year}")
        x_train = train[MODEL_FEATURES].astype("float32").to_numpy()
        y_train = train["made"].astype("int8").to_numpy()
        x_test_frame = test[MODEL_FEATURES].astype("float32")
        model = HistGradientBoostingClassifier(
            max_iter=2000,
            learning_rate=0.03,
            max_leaf_nodes=63,
            min_samples_leaf=80,
            l2_regularization=2.0,
            early_stopping=True,
            validation_fraction=0.08,
            n_iter_no_change=40,
            random_state=7,
        )
        model.fit(x_train, y_train)
        actual_probability = model.predict_proba(x_test_frame.to_numpy())[:, 1]
        neutral_frame = neutralize_shooter_features(x_test_frame)
        neutral_probability = model.predict_proba(neutral_frame.to_numpy())[:, 1]
        scored = test[
            [
                "season_end_year",
                "season_type",
                "game_id",
                "event_num",
                "shooter_id",
                "shooter_name",
                "made",
                "shot_distance_ft",
                "desc_pullup",
                "desc_step_back",
            ]
        ].copy()
        scored["p_actual_skill"] = actual_probability
        scored["p_neutral_skill"] = neutral_probability
        scored_parts.append(scored)
        audits.append(
            {
                "season_end_year": test_year,
                "training_threes": int(len(train)),
                "scored_threes": int(len(test)),
                "model_iterations": int(model.n_iter_),
                "actual_skill_brier": float(
                    brier_score_loss(test["made"], actual_probability)
                ),
                "neutral_skill_brier_diagnostic": float(
                    brier_score_loss(test["made"], neutral_probability)
                ),
                "mean_actual_probability": float(actual_probability.mean()),
                "mean_neutral_probability": float(neutral_probability.mean()),
            }
        )
        del x_train, y_train, x_test_frame, model, neutral_frame
        gc.collect()
    return pd.concat(scored_parts, ignore_index=True), audits


def add_frozen_reproduction_audit(
    scored: pd.DataFrame,
    audits: list[dict[str, object]],
    frozen_v2_path: Path,
) -> None:
    frozen = pd.read_parquet(
        frozen_v2_path,
        columns=["season_end_year", "game_id", "event_num", "zone", "p_v2"],
    )
    frozen = frozen[frozen["zone"].eq("three")].copy()
    frozen["game_id"] = frozen["game_id"].astype(str).str.zfill(10)
    joined = scored.merge(
        frozen.drop(columns="zone"),
        on=["season_end_year", "game_id", "event_num"],
        how="inner",
        validate="one_to_one",
    )
    for audit in audits:
        year = int(audit["season_end_year"])
        group = joined[joined["season_end_year"].eq(year)]
        gap = group["p_actual_skill"] - group["p_v2"]
        audit.update(
            {
                "frozen_rows_matched": int(len(group)),
                "frozen_reproduction_correlation": float(
                    group["p_actual_skill"].corr(group["p_v2"])
                ),
                "frozen_reproduction_mae": float(gap.abs().mean()),
                "frozen_reproduction_rmse": float(
                    np.sqrt(mean_squared_error(group["p_v2"], group["p_actual_skill"]))
                ),
                "frozen_v2_brier": float(
                    brier_score_loss(group["made"], group["p_v2"])
                ),
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pbp", type=Path, default=DEFAULT_PBP)
    parser.add_argument("--transition-dir", type=Path, default=DEFAULT_TRANSITION_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--epm-dir", type=Path, default=DEFAULT_EPM_DIR)
    parser.add_argument("--career-talent", type=Path, default=DEFAULT_CAREER_TALENT)
    parser.add_argument("--frozen-v2", type=Path, default=DEFAULT_FROZEN_V2)
    parser.add_argument("--first-training-year", type=int, default=2015)
    parser.add_argument("--first-test-year", type=int, default=2021)
    parser.add_argument("--last-test-year", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    features = build_feature_table(
        pbp_path=args.pbp,
        transition_dir=args.transition_dir,
        processed_dir=args.processed_dir,
        epm_dir=args.epm_dir,
        career_talent_path=args.career_talent,
        first_training_year=args.first_training_year,
        last_test_year=args.last_test_year,
    )
    scored, audits = score_counterfactual_seasons(
        features,
        first_test_year=args.first_test_year,
        last_test_year=args.last_test_year,
    )
    add_frozen_reproduction_audit(scored, audits, args.frozen_v2)
    player = aggregate_counterfactual_difficulty(scored)
    luka = player[
        player["season_end_year"].eq(args.last_test_year)
        & player["shooter_id"].eq(1629029)
    ]
    if luka.empty or float(luka.iloc[0]["v2_difficulty"]) <= 0:
        raise ValueError("Luka hard-shot falsification failed")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shot_path = args.output_dir / "russell_v2_counterfactual_shots.parquet"
    player_path = args.output_dir / "russell_v2_counterfactual_player_season.csv"
    audit_path = args.output_dir / "russell_v2_counterfactual_audit.csv"
    manifest_path = args.output_dir / "manifest.json"
    scored.to_parquet(shot_path, index=False)
    player.to_csv(player_path, index=False)
    pd.DataFrame(audits).to_csv(audit_path, index=False)
    manifest = {
        "schema_version": 1,
        "model_family": "RUSSELL_SHOTQUALITY v2 architecture counterfactual rebuild",
        "not_the_accepted_frozen_target": True,
        "difficulty_definition": "season mean neutral p - player mean neutral p",
        "neutralized_features": SHOOTER_TALENT_FEATURES,
        "held_fixed_features": [
            column for column in MODEL_FEATURES if column not in SHOOTER_TALENT_FEATURES
        ],
        "neutral_reference": "test-season shot-weighted mean; sh_zone_n median",
        "training_contract": "season-forward; only prior seasons train each test season",
        "source_note": (
            "Rebuilt from surviving current PBP, historical-safe transition backfill, "
            "EPM snapshot cache, and career seed. Compare the audit with frozen p_v2; "
            "do not silently replace the accepted VPM target."
        ),
        "paths": {
            "shot_output": str(shot_path),
            "player_output": str(player_path),
            "audit_output": str(audit_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(audits).to_string(index=False))
    print("\nLuka:")
    print(luka.to_string(index=False))
    print(f"\nWrote {player_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
