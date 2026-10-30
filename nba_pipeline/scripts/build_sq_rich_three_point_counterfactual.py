#!/usr/bin/env python3
"""Build a shooter-neutral three-point model from refreshed SQ geometry.

The model is trained on 2024-25 regular-season ShotQuality shots and scores
2025-26 regular-season shots. Each test shot is scored with its actual
prior-game shooter history and again with those shooter inputs set to league
references. Release-time context and defender geometry are held fixed.

Geometry is unavailable on many 2025-26 shots. The published full-season
prediction therefore uses the rich model only where closest-defender geometry
exists and a separately fitted context-only model everywhere else.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss

from nba_pipeline.scripts.train_sq_walk_forward_challenger import (
    DESCRIPTOR_TAGS,
    add_game_shifted_prior,
    clustered_delta_interval,
)
from nba_pipeline.scripts.build_sq_three_point_difficulty import (
    build_player_identity_bridge,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "nba_pipeline"
SHOTQUALITY_ROOT = REPO_ROOT.parent / "shotquality"
DEFAULT_TRAIN_SHOTS = SHOTQUALITY_ROOT / "store" / "shots_2025.parquet"
DEFAULT_TEST_SHOTS = SHOTQUALITY_ROOT / "store" / "shots_2026.parquet"
DEFAULT_GAMES = SHOTQUALITY_ROOT / "store" / "games.parquet"
DEFAULT_PLAYERS = SHOTQUALITY_ROOT / "players.csv"
DEFAULT_2026_CROSSWALK = (
    SHOTQUALITY_ROOT
    / "court_analysis"
    / "coordinate_repair"
    / "nba_game_crosswalk_2025_26.csv"
)
DEFAULT_OUTPUT_DIR = (
    PIPELINE_ROOT / "results" / "sq_three_point_difficulty" / "sq_rich_cf"
)

BASE_FEATURE_MAP = {
    "period": "period",
    "minutes": "minutes_remaining",
    "seconds": "seconds_remaining",
    "spacing": "spacing",
    "feature_store.Shot Distance (ft.)": "shot_distance_ft",
    "shooter_prior_make_probability": "shooter_prior_make_probability",
    "shooter_prior_log_attempts": "shooter_prior_log_attempts",
}

GEOMETRY_FEATURE_MAP = {
    "contest_score": "contest_score",
    "feature_store.Defenders in Paint": "defenders_in_paint",
    "feature_store.Closest Defender Behind": "closest_defender_behind",
    "feature_store.Closest Defender Inside": "closest_defender_inside",
    "feature_store.Closest Defender Blocking": "closest_defender_blocking",
    "feature_store.Second-Closest Defender Behind": (
        "second_closest_defender_behind"
    ),
    "feature_store.Harmonic Mean Defender Distance": (
        "harmonic_mean_defender_distance"
    ),
    "feature_store.Average Defender Distance (Feet)": (
        "average_defender_distance"
    ),
    "feature_store.Second-Closest Defender Blocking": (
        "second_closest_defender_blocking"
    ),
    "feature_store.Distance to the Closest Defender (Feet)": (
        "closest_defender_distance"
    ),
    "feature_store.Height of the Closest Defender (Inches)": (
        "closest_defender_height"
    ),
    "feature_store.Number of Defenders Closer to the Basket": (
        "defenders_closer_to_basket"
    ),
    "feature_store.Cosine of the Angle with the Closest Defender": (
        "closest_defender_angle_cosine"
    ),
    "feature_store.Distance to the Second-Closest Defender (Feet)": (
        "second_closest_defender_distance"
    ),
    "feature_store.Cosine of the Angle with the Second-Closest Defender": (
        "second_closest_defender_angle_cosine"
    ),
    "feature_store.Difference in Bounding Box Height with the Closest Defender (Pixels)": (
        "closest_defender_bbox_height_difference"
    ),
}

SOURCE_COLUMNS = sorted(
    {
        "game_id",
        "play_id",
        "team_id",
        "player_id",
        "period",
        "minutes",
        "seconds",
        "made",
        "action_type",
        "play_descriptors",
        "shooter_name",
        "ev.initial_ev.value",
        "shot_taking.initial_shot_taking.value",
        *(
            source
            for source in BASE_FEATURE_MAP
            if not source.startswith("shooter_prior_")
        ),
        *GEOMETRY_FEATURE_MAP,
    }
)


SHOOTER_TALENT_FEATURES = (
    "shooter_prior_make_probability",
    "shooter_prior_log_attempts",
)


def _descriptor_features(values: pd.Series) -> pd.DataFrame:
    def parse(value: object) -> set[str]:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return set()
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (list, tuple, set)):
            return {str(item).strip().lower() for item in value}
        return {
            token.strip().lower()
            for token in str(value).replace("|", ",").split(",")
            if token.strip()
        }

    parsed = values.map(parse)
    return pd.DataFrame(
        {
            f"descriptor_{tag}": parsed.map(
                lambda tags, tag=tag: float(tag in tags)
            )
            for tag in DESCRIPTOR_TAGS
        },
        index=values.index,
    )


def build_model_features(frame: pd.DataFrame, *, rich: bool) -> pd.DataFrame:
    """Construct the frozen numeric feature frame for one model arm."""
    mapping = dict(BASE_FEATURE_MAP)
    if rich:
        mapping.update(GEOMETRY_FEATURE_MAP)
    output = pd.DataFrame(index=frame.index)
    for source, target in mapping.items():
        output[target] = pd.to_numeric(frame[source], errors="coerce")
    return pd.concat(
        [output, _descriptor_features(frame["play_descriptors"])], axis=1
    ).astype("float32")


def _new_model(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=650,
        learning_rate=0.035,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=3.0,
        early_stopping=True,
        validation_fraction=0.08,
        n_iter_no_change=35,
        random_state=seed,
    )


def _regular_season_game_ids(
    games: pd.DataFrame,
    *,
    season_end_year: int,
    current_crosswalk: Path,
) -> set[int]:
    if season_end_year == 2026:
        crosswalk = pd.read_csv(
            current_crosswalk, usecols=["sq_game_id", "playoffs"]
        )
        return set(
            pd.to_numeric(
                crosswalk.loc[crosswalk["playoffs"].eq(False), "sq_game_id"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
        )
    season = games[
        pd.to_numeric(games["season_start_year"], errors="coerce").eq(
            season_end_year - 1
        )
    ]
    regular_labels = {"Regular Season", "NBA Cup Finals"}
    return set(
        pd.to_numeric(
            season.loc[
                season["game_descriptors.event_type_id"].isin(regular_labels),
                "game_id",
            ],
            errors="coerce",
        )
        .dropna()
        .astype(int)
    )


def _player_name_map(path: Path) -> pd.Series:
    players = pd.read_csv(
        path,
        usecols=["player_id", "player_name", "updated_at"],
    )
    players["player_id"] = pd.to_numeric(
        players["player_id"], errors="coerce"
    )
    players = (
        players.dropna(subset=["player_id", "player_name"])
        .sort_values("updated_at", kind="stable")
        .drop_duplicates("player_id", keep="last")
    )
    return players.set_index("player_id")["player_name"]


def load_regular_season_threes(
    path: Path,
    *,
    season_end_year: int,
    games: pd.DataFrame,
    current_crosswalk: Path,
    player_names: pd.Series,
) -> pd.DataFrame:
    """Load unique, player-attributed regular-season threes from the SQ store."""
    shots = pd.read_parquet(path, columns=SOURCE_COLUMNS)
    shots = shots[shots["action_type"].astype(str).str.lower().eq("3pt")].copy()
    if shots.duplicated(["game_id", "play_id"]).any():
        raise ValueError(f"Duplicate ShotQuality game/play keys in {path}")
    game_ids = _regular_season_game_ids(
        games,
        season_end_year=season_end_year,
        current_crosswalk=current_crosswalk,
    )
    shots["game_id"] = pd.to_numeric(shots["game_id"], errors="coerce")
    shots = shots[shots["game_id"].isin(game_ids)].copy()
    shots["sq_player_id"] = pd.to_numeric(
        shots["player_id"], errors="coerce"
    )
    shots = shots.dropna(subset=["sq_player_id", "made"]).copy()
    shots["sq_player_id"] = shots["sq_player_id"].astype(int)
    missing_name = shots["shooter_name"].isna()
    shots.loc[missing_name, "shooter_name"] = shots.loc[
        missing_name, "sq_player_id"
    ].map(player_names)
    missing_name = shots["shooter_name"].isna()
    shots.loc[missing_name, "shooter_name"] = (
        "SQ_" + shots.loc[missing_name, "sq_player_id"].astype(str)
    )
    date_map = games[["game_id", "game_datetime_utc"]].copy()
    date_map["game_id"] = pd.to_numeric(date_map["game_id"], errors="coerce")
    date_map = date_map.drop_duplicates("game_id")
    date_map["game_date"] = pd.to_datetime(
        date_map["game_datetime_utc"], errors="coerce", utc=True
    ).dt.tz_convert(None)
    shots = shots.merge(
        date_map[["game_id", "game_date"]],
        on="game_id",
        how="left",
        validate="many_to_one",
    )
    if shots["game_date"].isna().any():
        raise ValueError(f"Missing game dates in {path}")
    shots["season_end_year"] = season_end_year
    shots["segment"] = "three"
    shots["game_id_normalized"] = shots["game_id"].astype(int).astype(str)
    shots["made"] = shots["made"].astype(int)
    shots["shot_value"] = 3.0
    closest = pd.to_numeric(
        shots["feature_store.Distance to the Closest Defender (Feet)"],
        errors="coerce",
    )
    shots["geometry_available"] = closest.notna().astype(int)
    return shots.sort_values(
        ["game_date", "game_id", "period", "minutes", "seconds", "play_id"],
        ascending=[True, True, True, False, False, True],
        kind="stable",
    ).reset_index(drop=True)


def prepare_model_rows(
    *,
    train_path: Path,
    test_path: Path,
    games_path: Path,
    current_crosswalk: Path,
    players_path: Path,
    shooter_shrink: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    games = pd.read_parquet(
        games_path,
        columns=[
            "game_id",
            "game_datetime_utc",
            "season_start_year",
            "game_descriptors.event_type_id",
        ],
    )
    names = _player_name_map(players_path)
    train = load_regular_season_threes(
        train_path,
        season_end_year=2025,
        games=games,
        current_crosswalk=current_crosswalk,
        player_names=names,
    )
    test = load_regular_season_threes(
        test_path,
        season_end_year=2026,
        games=games,
        current_crosswalk=current_crosswalk,
        player_names=names,
    )
    identity_input = test.assign(
        season="2025-26",
        player_id=test["sq_player_id"],
        made_3=test["made"],
    )
    identity, identity_audit = build_player_identity_bridge(
        identity_input, SHOTQUALITY_ROOT
    )
    identity = (
        identity.dropna(subset=["sq_player_id", "player_name"])
        .drop_duplicates("sq_player_id", keep="last")
        .set_index("sq_player_id")
    )
    fallback_name = test["shooter_name"].astype(str).str.startswith("SQ_")
    recovered_name = test["sq_player_id"].map(identity["player_name"])
    test.loc[fallback_name & recovered_name.notna(), "shooter_name"] = (
        recovered_name[fallback_name & recovered_name.notna()]
    )
    combined = pd.concat([train, test], ignore_index=True)
    combined = add_game_shifted_prior(
        combined,
        player_column="sq_player_id",
        prefix="shooter",
        shrink_attempts=shooter_shrink,
        baseline_by_segment={"three": 0.36},
    )
    train = combined[combined["season_end_year"].eq(2025)].copy()
    test = combined[combined["season_end_year"].eq(2026)].copy()
    audit = {
        "training_threes": int(len(train)),
        "test_threes": int(len(test)),
        "training_games": int(train["game_id"].nunique()),
        "test_games": int(test["game_id"].nunique()),
        "training_geometry_coverage": float(train["geometry_available"].mean()),
        "test_geometry_coverage": float(test["geometry_available"].mean()),
        "test_named_players": int(test["sq_player_id"].nunique()),
        "test_unresolved_player_names": int(
            test["shooter_name"].astype(str).str.startswith("SQ_").sum()
        ),
        "identity_audit": identity_audit,
    }
    return train, test, audit


def score_rich_counterfactual(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit context/rich arms and score actual plus neutral shooter inputs."""
    context_columns = list(BASE_FEATURE_MAP.values()) + [
        f"descriptor_{tag}" for tag in DESCRIPTOR_TAGS
    ]
    rich_columns = context_columns + list(GEOMETRY_FEATURE_MAP.values())
    train_context = build_model_features(train, rich=False)
    test_context = build_model_features(test, rich=False)
    train_rich = build_model_features(train, rich=True)
    test_rich = build_model_features(test, rich=True)
    geometry_train = train["geometry_available"].eq(1).to_numpy()
    geometry_test = test["geometry_available"].eq(1).to_numpy()

    context_model = _new_model(seed)
    context_model.fit(train_context[context_columns], train["made"])
    rich_model = _new_model(seed + 1)
    rich_model.fit(
        train_rich.loc[geometry_train, rich_columns],
        train.loc[geometry_train, "made"],
    )

    neutral_reference = {
        column: float(test_context[column].mean())
        for column in SHOOTER_TALENT_FEATURES
    }
    neutral_context = neutralize_rich_shooter_features(
        test_context, neutral_reference
    )
    neutral_rich = neutralize_rich_shooter_features(
        test_rich, neutral_reference
    )
    p_actual_context = context_model.predict_proba(
        test_context[context_columns]
    )[:, 1]
    p_neutral_context = context_model.predict_proba(
        neutral_context[context_columns]
    )[:, 1]
    p_actual_rich = np.full(len(test), np.nan)
    p_neutral_rich = np.full(len(test), np.nan)
    p_actual_rich[geometry_test] = rich_model.predict_proba(
        test_rich.loc[geometry_test, rich_columns]
    )[:, 1]
    p_neutral_rich[geometry_test] = rich_model.predict_proba(
        neutral_rich.loc[geometry_test, rich_columns]
    )[:, 1]
    p_actual_hybrid = np.where(
        geometry_test, p_actual_rich, p_actual_context
    )
    p_neutral_hybrid = np.where(
        geometry_test, p_neutral_rich, p_neutral_context
    )

    scored = test[
        [
            "season_end_year",
            "game_id",
            "play_id",
            "sq_player_id",
            "shooter_name",
            "made",
            "geometry_available",
            "feature_store.Shot Distance (ft.)",
            "feature_store.Distance to the Closest Defender (Feet)",
            "contest_score",
        ]
    ].copy()
    scored["p_actual_context"] = p_actual_context
    scored["p_neutral_context"] = p_neutral_context
    scored["p_actual_rich"] = p_actual_rich
    scored["p_neutral_rich"] = p_neutral_rich
    scored["p_actual_hybrid"] = p_actual_hybrid
    scored["p_neutral_hybrid"] = p_neutral_hybrid

    target = test["made"].to_numpy(dtype=float)
    provider_initial = (
        pd.to_numeric(test["ev.initial_ev.value"], errors="coerce") / 3.0
    ).to_numpy()
    valid_initial = np.isfinite(provider_initial)
    metrics: dict[str, object] = {
        "context_model_iterations": int(context_model.n_iter_),
        "rich_model_iterations": int(rich_model.n_iter_),
        "context_brier": float(brier_score_loss(target, p_actual_context)),
        "hybrid_brier": float(brier_score_loss(target, p_actual_hybrid)),
        "hybrid_minus_context_brier": float(
            np.mean((target - p_actual_hybrid) ** 2)
            - np.mean((target - p_actual_context) ** 2)
        ),
        "initial_ev_brier": float(
            brier_score_loss(target[valid_initial], provider_initial[valid_initial])
        ),
        "hybrid_minus_initial_ev_brier": float(
            np.mean((target[valid_initial] - p_actual_hybrid[valid_initial]) ** 2)
            - np.mean((target[valid_initial] - provider_initial[valid_initial]) ** 2)
        ),
        "neutral_reference": neutral_reference,
    }
    geometry_frame = test.loc[geometry_test].copy()
    geometry_frame["game_id_normalized"] = geometry_frame["game_id"].astype(str)
    geometry_delta = clustered_delta_interval(
        geometry_frame,
        p_actual_rich[geometry_test],
        p_actual_context[geometry_test],
        metric="points_mse",
        iterations=1000,
        seed=seed + 2,
    )
    metrics["geometry_rich_vs_context_points_mse"] = geometry_delta
    return scored, metrics


def neutralize_rich_shooter_features(
    frame: pd.DataFrame,
    reference: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Hold shot context fixed while setting shooter inputs to league values."""
    missing = sorted(set(SHOOTER_TALENT_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing rich shooter features: {missing}")
    neutral = frame.copy()
    if reference is None:
        reference = {
            column: float(
                pd.to_numeric(frame[column], errors="coerce").mean()
            )
            for column in SHOOTER_TALENT_FEATURES
        }
    for column in SHOOTER_TALENT_FEATURES:
        value = reference.get(column)
        if value is None or not np.isfinite(value):
            raise ValueError(f"Invalid neutral reference for {column}")
        neutral[column] = float(value)
    return neutral


def aggregate_rich_counterfactual_difficulty(
    scored: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate rich double-scored threes to centered player seasons."""
    required = {
        "season_end_year",
        "sq_player_id",
        "shooter_name",
        "made",
        "p_actual_hybrid",
        "p_neutral_hybrid",
        "p_neutral_rich",
        "geometry_available",
    }
    missing = sorted(required - set(scored.columns))
    if missing:
        raise ValueError(f"Missing rich counterfactual columns: {missing}")
    baseline = scored.groupby("season_end_year")["p_neutral_hybrid"].mean()
    player = (
        scored.groupby(
            ["season_end_year", "sq_player_id", "shooter_name"],
            sort=False,
            dropna=False,
        )
        .agg(
            sq_rich_attempts=("made", "size"),
            raw_3p_pct=("made", "mean"),
            sq_rich_actual_expected=("p_actual_hybrid", "mean"),
            sq_rich_neutral_expected=("p_neutral_hybrid", "mean"),
            sq_rich_geometry_attempts=("geometry_available", "sum"),
        )
        .reset_index()
    )
    player["sq_rich_geometry_coverage"] = (
        player["sq_rich_geometry_attempts"] / player["sq_rich_attempts"]
    )
    player["sq_rich_neutral_baseline"] = player["season_end_year"].map(
        baseline
    )
    player["sq_rich_difficulty"] = (
        player["sq_rich_neutral_baseline"]
        - player["sq_rich_neutral_expected"]
    )
    player["sq_rich_effective_3p_pct"] = (
        player["raw_3p_pct"] + player["sq_rich_difficulty"]
    )
    covered = scored[
        scored["geometry_available"].eq(1)
        & scored["p_neutral_rich"].notna()
    ]
    covered_baseline = covered.groupby("season_end_year")[
        "p_neutral_rich"
    ].mean()
    covered_player = (
        covered.groupby(
            ["season_end_year", "sq_player_id", "shooter_name"],
            sort=False,
            dropna=False,
        )["p_neutral_rich"]
        .mean()
        .rename("sq_rich_geometry_only_expected")
        .reset_index()
    )
    player = player.merge(
        covered_player,
        on=["season_end_year", "sq_player_id", "shooter_name"],
        how="left",
        validate="one_to_one",
    )
    player["sq_rich_geometry_only_baseline"] = player[
        "season_end_year"
    ].map(covered_baseline)
    player["sq_rich_geometry_only_difficulty"] = (
        player["sq_rich_geometry_only_baseline"]
        - player["sq_rich_geometry_only_expected"]
    )
    for season, group in player.groupby("season_end_year"):
        centered = np.average(
            group["sq_rich_difficulty"],
            weights=group["sq_rich_attempts"],
        )
        if abs(float(centered)) > 1e-12:
            raise ValueError(
                f"Rich difficulty is not centered for {season}: {centered}"
            )
        covered_group = group[group["sq_rich_geometry_attempts"].gt(0)]
        covered_centered = np.average(
            covered_group["sq_rich_geometry_only_difficulty"],
            weights=covered_group["sq_rich_geometry_attempts"],
        )
        if abs(float(covered_centered)) > 1e-12:
            raise ValueError(
                "Geometry-only rich difficulty is not centered for "
                f"{season}: {covered_centered}"
            )
    return player


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-shots", type=Path, default=DEFAULT_TRAIN_SHOTS)
    parser.add_argument("--test-shots", type=Path, default=DEFAULT_TEST_SHOTS)
    parser.add_argument("--games", type=Path, default=DEFAULT_GAMES)
    parser.add_argument("--players", type=Path, default=DEFAULT_PLAYERS)
    parser.add_argument(
        "--current-crosswalk", type=Path, default=DEFAULT_2026_CROSSWALK
    )
    parser.add_argument("--shooter-shrink", type=float, default=200.0)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train, test, data_audit = prepare_model_rows(
        train_path=args.train_shots,
        test_path=args.test_shots,
        games_path=args.games,
        current_crosswalk=args.current_crosswalk,
        players_path=args.players,
        shooter_shrink=args.shooter_shrink,
    )
    scored, model_audit = score_rich_counterfactual(
        train, test, seed=args.seed
    )
    player = aggregate_rich_counterfactual_difficulty(scored)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shots_path = args.output_dir / "sq_rich_counterfactual_shots.parquet"
    player_path = args.output_dir / "sq_rich_counterfactual_player_season.csv"
    audit_path = args.output_dir / "audit.json"
    manifest_path = args.output_dir / "manifest.json"
    scored.to_parquet(shots_path, index=False)
    player.to_csv(player_path, index=False)
    audit = {"data": data_audit, "model": model_audit}
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "model_family": "ShotQuality-rich three-point counterfactual",
        "status": "descriptive challenger",
        "training_season": "2024-25 regular season",
        "test_season": "2025-26 regular season",
        "training_contract": "season-forward",
        "difficulty_definition": (
            "season mean hybrid neutral p - player mean hybrid neutral p"
        ),
        "hybrid_contract": (
            "rich geometry model when closest-defender distance exists; "
            "context-only model otherwise"
        ),
        "neutralized_features": list(SHOOTER_TALENT_FEATURES),
        "held_fixed_context_features": [
            column
            for column in list(BASE_FEATURE_MAP.values())
            if column not in SHOOTER_TALENT_FEATURES
        ]
        + [f"descriptor_{tag}" for tag in DESCRIPTOR_TAGS]
        + list(GEOMETRY_FEATURE_MAP.values()),
        "forbidden_inputs": [
            "made outcome as feature",
            "initial_ev as feature",
            "initial_shot_taking as feature",
            "shot_x/shot_y collapsed coordinates",
            "same-game shooter outcomes",
        ],
        "source_paths": {
            "train_shots": str(args.train_shots),
            "test_shots": str(args.test_shots),
            "games": str(args.games),
            "current_crosswalk": str(args.current_crosswalk),
        },
        "output_paths": {
            "shots": str(shots_path),
            "player_seasons": str(player_path),
            "audit": str(audit_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    luka = player[
        player["shooter_name"].astype(str).str.contains(
            "Luka Doncic", case=False, na=False
        )
    ]
    print(json.dumps(audit, indent=2))
    print("\nLuka:")
    print(luka.to_string(index=False) if len(luka) else "not matched")
    print(f"\nWrote {player_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
