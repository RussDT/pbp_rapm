#!/usr/bin/env python3
"""Build season-level estimates of championships added by DARKO teammate help."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

try:
    from nba_pipeline.scripts.build_darko_teammate_help import (
        METRICS,
        PlayerRatingHistory,
        attach_names,
        load_darko_history,
        load_name_map,
        lookup_asof_ratings,
    )
except ModuleNotFoundError:  # Direct script execution from the repository root.
    from build_darko_teammate_help import (  # type: ignore[no-redef]
        METRICS,
        PlayerRatingHistory,
        attach_names,
        load_darko_history,
        load_name_map,
        lookup_asof_ratings,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_DIR.parent
PROCESSED_DIR = PIPELINE_DIR / "processed"
RAW_DIR = PIPELINE_DIR / "raw_data"
DEFAULT_DARKO_PATH = PIPELINE_DIR / ".cache" / "teammate_help" / "dpm_history.csv"
DEFAULT_OUTPUT_DIR = PIPELINE_DIR / "results" / "championship_help"
DEFAULT_APP_DATA_DIR = PROJECT_ROOT / "teammate-help-app" / "public" / "data" / "championship-help"
MODEL_SPECS = ("own", "linear", "spline", "spline_interaction")
MODEL_C_VALUES = (0.01, 0.05, 0.2, 1.0)
MODEL_MIN_EXPOSURES = 500
MODEL_MIN_COVERAGE = 50.0
APP_SHARDS = 64
PLAYER_COLS = [f"P{i}" for i in range(1, 6)]
LINEUP_COLS = [f"O{i}" for i in range(1, 6)]
OPPONENT_COLS = [f"D{i}" for i in range(1, 6)]
RAW_HOME_COLS = [f"home_player{i}" for i in range(1, 6)]
RAW_AWAY_COLS = [f"away_player{i}" for i in range(1, 6)]
RAW_CONTEXT_COLS = [
    "game_id",
    "event_num",
    "home_description",
    "visitor_description",
    "player1_id",
    "player1_team_id",
    "player1_team_abbreviation",
    "home_score",
    "away_score",
    *RAW_HOME_COLS,
    *RAW_AWAY_COLS,
]


@dataclass
class ChampionshipOddsModel:
    spec: str
    c_value: float
    estimator: Pipeline

    def _features(self, rows: pd.DataFrame) -> pd.DataFrame:
        features = rows[["own_dpm", "help_dpm"]].copy()
        features["interaction"] = features["own_dpm"] * features["help_dpm"]
        return features

    def predict_probability(self, rows: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(self._features(rows))[:, 1]


def fit_championship_odds_model(
    rows: pd.DataFrame,
    spec: str,
    c_value: float,
) -> ChampionshipOddsModel:
    """Fit a regularized player-season championship-probability model."""
    if spec == "own":
        preprocessor = ColumnTransformer([("own", StandardScaler(), ["own_dpm"])])
    elif spec == "linear":
        preprocessor = ColumnTransformer(
            [("linear", StandardScaler(), ["own_dpm", "help_dpm"])]
        )
    elif spec in {"spline", "spline_interaction"}:
        transformers: list[tuple[str, object, list[str]]] = [
            (
                "own",
                SplineTransformer(n_knots=4, degree=2, include_bias=False),
                ["own_dpm"],
            ),
            (
                "help",
                SplineTransformer(n_knots=4, degree=2, include_bias=False),
                ["help_dpm"],
            ),
        ]
        if spec == "spline_interaction":
            transformers.append(("interaction", StandardScaler(), ["interaction"]))
        preprocessor = ColumnTransformer(transformers)
    else:
        raise ValueError(f"Unknown model spec: {spec}")
    estimator = Pipeline(
        [
            ("features", preprocessor),
            ("logit", LogisticRegression(C=float(c_value), max_iter=2_000)),
        ]
    )
    model = ChampionshipOddsModel(spec=spec, c_value=float(c_value), estimator=estimator)
    features = model._features(rows)
    estimator.fit(features, rows["won_championship"].astype(int))
    return model


def prepare_model_sample(
    player: pd.DataFrame,
    min_exposures: int = MODEL_MIN_EXPOSURES,
    min_coverage: float = MODEL_MIN_COVERAGE,
) -> pd.DataFrame:
    sample = player[
        player["total_exposures"].ge(min_exposures)
        & player["coverage_pct"].ge(min_coverage)
        & player["own_dpm"].notna()
        & player["help_dpm"].notna()
    ].copy()
    return sample.sort_values(["season", "player_id"], kind="mergesort").reset_index(drop=True)


def cross_validated_predictions(
    sample: pd.DataFrame,
    spec: str,
    c_value: float,
) -> pd.DataFrame:
    predictions: list[pd.DataFrame] = []
    for season in sorted(sample["season"].unique()):
        train = sample[sample["season"] != season]
        test = sample[sample["season"] == season]
        model = fit_championship_odds_model(train, spec=spec, c_value=c_value)
        fold = test[["season", "player_id", "team_id", "won_championship"]].copy()
        fold["predicted_probability"] = model.predict_probability(test)
        predictions.append(fold)
    return pd.concat(predictions, ignore_index=True)


def score_probability_predictions(predictions: pd.DataFrame) -> dict[str, float]:
    outcomes = predictions["won_championship"].to_numpy(dtype=int)
    probability = predictions["predicted_probability"].to_numpy(dtype=float)
    return {
        "brier": float(brier_score_loss(outcomes, probability)),
        "log_loss": float(log_loss(outcomes, probability)),
        "roc_auc": float(roc_auc_score(outcomes, probability)),
        "mean_probability": float(probability.mean()),
        "observed_rate": float(outcomes.mean()),
    }


def evaluate_model_grid(
    sample: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    records: list[dict[str, object]] = []
    prediction_cache: dict[tuple[str, float], pd.DataFrame] = {}
    for spec in MODEL_SPECS:
        for c_value in MODEL_C_VALUES:
            predictions = cross_validated_predictions(sample, spec, c_value)
            prediction_cache[(spec, c_value)] = predictions
            records.append({"spec": spec, "c_value": c_value, **score_probability_predictions(predictions)})
    diagnostics = pd.DataFrame(records).sort_values(
        ["log_loss", "brier", "spec", "c_value"], kind="mergesort"
    ).reset_index(drop=True)
    winner = diagnostics.iloc[0]
    selected = {
        "spec": str(winner["spec"]),
        "c_value": float(winner["c_value"]),
        "selection_metric": "leave-one-season-out log loss",
        "brier": float(winner["brier"]),
        "log_loss": float(winner["log_loss"]),
        "roc_auc": float(winner["roc_auc"]),
        "mean_probability": float(winner["mean_probability"]),
        "observed_rate": float(winner["observed_rate"]),
    }
    selected_predictions = prediction_cache[(selected["spec"], selected["c_value"])]
    return diagnostics, selected, selected_predictions


def attribute_model_probabilities(
    sample: pd.DataFrame,
    pair: pd.DataFrame,
    model: ChampionshipOddsModel,
    permutations: int = 128,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    """Estimate actual-vs-neutral title odds and allocate the change to teammates."""
    seasons = sample.copy()
    seasons["title_probability"] = model.predict_probability(seasons)
    neutral = seasons.copy()
    neutral["help_dpm"] = 0.0
    seasons["neutral_teammate_title_probability"] = model.predict_probability(neutral)
    seasons["teammate_help_championships"] = (
        seasons["title_probability"] - seasons["neutral_teammate_title_probability"]
    )

    pair_keys = seasons[["season", "player_id", "team_id"]]
    attributed = pair.merge(pair_keys, on=["season", "player_id", "team_id"], how="inner").copy()
    attributed["championships_added"] = 0.0
    max_gap = 0.0
    for key, group in attributed.groupby(["season", "player_id", "team_id"], sort=False):
        season, player_id, team_id = key
        season_row = seasons[
            seasons["season"].eq(season)
            & seasons["player_id"].eq(player_id)
            & seasons["team_id"].eq(team_id)
        ].iloc[0]
        values = group["contribution_dpm"].fillna(0.0).to_numpy(dtype=float)
        count = len(values)
        rng = np.random.default_rng((int(season) * 1_000_003 + int(player_id)) % (2**32))
        orders = np.vstack([rng.permutation(count) for _ in range(permutations)])
        ordered_values = values[orders]
        cumulative = np.concatenate(
            [np.zeros((permutations, 1)), np.cumsum(ordered_values, axis=1)], axis=1
        )
        states = pd.DataFrame(
            {
                "own_dpm": float(season_row["own_dpm"]),
                "help_dpm": cumulative.ravel(),
            }
        )
        probability = model.predict_probability(states).reshape(permutations, count + 1)
        deltas = np.diff(probability, axis=1)
        shapley = np.zeros(count, dtype=float)
        for position in range(count):
            np.add.at(shapley, orders[:, position], deltas[:, position])
        shapley /= permutations
        attributed.loc[group.index, "championships_added"] = shapley
        target = float(season_row["teammate_help_championships"])
        gap = abs(float(shapley.sum()) - target)
        max_gap = max(max_gap, gap)
    attributed = attributed.merge(
        seasons[
            [
                "season",
                "player_id",
                "team_id",
                "won_championship",
                "title_probability",
                "neutral_teammate_title_probability",
            ]
        ],
        on=["season", "player_id", "team_id"],
        how="left",
    )
    report = {
        "player_seasons": int(len(seasons)),
        "attributed_pair_seasons": int(len(attributed)),
        "permutations": int(permutations),
        "max_shapley_reconciliation_gap": float(max_gap),
    }
    if max_gap > 1e-10:
        raise AssertionError(f"Shapley attribution gap {max_gap} exceeds tolerance")
    return seasons, attributed, report


def build_career_championship_help(
    seasons: pd.DataFrame,
    attributed: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    career = (
        seasons.groupby(["player_id", "player_name"], as_index=False, sort=False)
        .agg(
            start_season=("season", "min"),
            end_season=("season", "max"),
            seasons_modeled=("season", "nunique"),
            actual_championships=("won_championship", "sum"),
            expected_championships=("title_probability", "sum"),
            neutral_expected_championships=("neutral_teammate_title_probability", "sum"),
            teammate_help_championships=("teammate_help_championships", "sum"),
        )
    )
    peak_index = seasons.groupby("player_id")["teammate_help_championships"].idxmax()
    peaks = seasons.loc[
        peak_index,
        ["player_id", "season", "teammate_help_championships", "help_dpm", "title_probability"],
    ].rename(
        columns={
            "season": "peak_help_season",
            "teammate_help_championships": "peak_season_championships_added",
            "help_dpm": "peak_season_help_dpm",
            "title_probability": "peak_season_title_probability",
        }
    )
    career = career.merge(peaks, on="player_id", how="left")
    career = career.sort_values(
        ["teammate_help_championships", "player_id"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    career["championship_help_rank"] = np.arange(1, len(career) + 1)

    teammate = (
        attributed.groupby(
            ["player_id", "player_name", "teammate_id", "teammate_name"],
            as_index=False,
            sort=False,
        )
        .agg(
            start_season=("season", "min"),
            end_season=("season", "max"),
            seasons_together=("season", "nunique"),
            shared_exposures=("shared_exposures", "sum"),
            rated_exposures=("rated_exposures", "sum"),
            weighted_dpm=("weighted_dpm", "sum"),
            championships_together=("won_championship", "sum"),
            championships_added=("championships_added", "sum"),
        )
    )
    teammate["avg_teammate_dpm"] = teammate["weighted_dpm"] / teammate[
        "rated_exposures"
    ].replace(0, np.nan)
    teammate = teammate.sort_values(
        ["player_id", "championships_added", "shared_exposures", "teammate_id"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )
    teammate["championship_help_rank"] = teammate.groupby("player_id").cumcount() + 1
    career_pair_sum = teammate.groupby("player_id")["championships_added"].sum()
    career_gap = (
        career.set_index("player_id")["teammate_help_championships"] - career_pair_sum
    ).abs()
    report = {"max_career_reconciliation_gap": float(career_gap.max())}
    if report["max_career_reconciliation_gap"] > 1e-10:
        raise AssertionError("Career teammate championship-help rows do not reconcile")
    return career, teammate.reset_index(drop=True), report


def rounded_records(frame: pd.DataFrame, columns: list[str], decimals: int = 6) -> list[dict]:
    clean = frame[columns].copy()
    float_columns = clean.select_dtypes(include=["float", "float32", "float64"]).columns
    clean[float_columns] = clean[float_columns].round(decimals)
    return clean.replace({np.nan: None}).to_dict(orient="records")


def write_gzip_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    path.write_bytes(gzip.compress(encoded, compresslevel=9, mtime=0))


def write_app_championship_data(
    output_dir: Path,
    career: pd.DataFrame,
    seasons: pd.DataFrame,
    teammate: pd.DataFrame,
    metadata: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, separators=(",", ":")))
    career_columns = [
        "player_id",
        "player_name",
        "start_season",
        "end_season",
        "seasons_modeled",
        "actual_championships",
        "expected_championships",
        "neutral_expected_championships",
        "teammate_help_championships",
        "peak_help_season",
        "peak_season_championships_added",
        "peak_season_help_dpm",
        "peak_season_title_probability",
        "championship_help_rank",
    ]
    season_columns = [
        "season",
        "player_id",
        "player_name",
        "team_id",
        "team_abbr",
        "won_championship",
        "own_dpm",
        "help_dpm",
        "coverage_pct",
        "total_exposures",
        "title_probability",
        "neutral_teammate_title_probability",
        "teammate_help_championships",
    ]
    teammate_columns = [
        "player_id",
        "teammate_id",
        "teammate_name",
        "start_season",
        "end_season",
        "seasons_together",
        "shared_exposures",
        "avg_teammate_dpm",
        "championships_together",
        "championships_added",
        "championship_help_rank",
    ]
    write_gzip_json(output_dir / "players.json.gz", rounded_records(career, career_columns))
    manifest: list[dict[str, object]] = []
    top_teammates = teammate[teammate["championship_help_rank"].le(20)]
    for shard in range(APP_SHARDS):
        player_ids = set(career.loc[career["player_id"].mod(APP_SHARDS).eq(shard), "player_id"])
        payload = {
            "seasons": rounded_records(
                seasons[seasons["player_id"].isin(player_ids)], season_columns
            ),
            "teammates": rounded_records(
                top_teammates[top_teammates["player_id"].isin(player_ids)], teammate_columns
            ),
        }
        filename = f"{shard:02d}.json.gz"
        write_gzip_json(output_dir / "players" / filename, payload)
        manifest.append(
            {
                "shard": shard,
                "file": f"players/{filename}",
                "players": len(player_ids),
                "season_rows": len(payload["seasons"]),
                "teammate_rows": len(payload["teammates"]),
            }
        )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))


def build_probability_curves(
    model: ChampionshipOddsModel,
) -> list[dict[str, float]]:
    help_values = np.linspace(-6.0, 14.0, 81)
    curves: list[dict[str, float]] = []
    for own_dpm in (-2.0, 0.0, 2.0, 4.0, 6.0):
        grid = pd.DataFrame({"own_dpm": own_dpm, "help_dpm": help_values})
        probability = model.predict_probability(grid)
        curves.extend(
            {
                "own_dpm": own_dpm,
                "help_dpm": float(help_dpm),
                "title_probability": float(title_probability),
            }
            for help_dpm, title_probability in zip(help_values, probability)
        )
    return curves


def normalize_game_ids(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def build_game_context(
    raw: pd.DataFrame,
    game_dates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recover per-game team identities, rosters, dates, and final scores."""
    required = {
        "game_id",
        "event_num",
        "home_description",
        "visitor_description",
        "player1_team_id",
        "player1_team_abbreviation",
        "home_score",
        "away_score",
        *RAW_HOME_COLS,
        *RAW_AWAY_COLS,
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Raw PBP missing columns: {sorted(missing)}")
    dates = game_dates[["game_id", "game_date"]].copy()
    dates["game_id"] = normalize_game_ids(dates["game_id"])
    dates["game_date"] = pd.to_datetime(dates["game_date"], errors="coerce").dt.normalize()
    date_map = dates.dropna().drop_duplicates("game_id", keep="last").set_index("game_id")[
        "game_date"
    ]

    rows = raw.copy()
    rows["game_id"] = normalize_game_ids(rows["game_id"])
    rows["player1_team_id"] = pd.to_numeric(rows["player1_team_id"], errors="coerce")
    rows["event_num"] = pd.to_numeric(rows["event_num"], errors="coerce")
    game_records: list[dict[str, object]] = []
    assignment_records: list[dict[str, object]] = []

    def side_identity(
        group: pd.DataFrame,
        description: str,
        lineup_columns: list[str],
    ) -> tuple[int, str]:
        candidates = pd.DataFrame()
        if "player1_id" in group.columns:
            lineup_numeric = group[lineup_columns].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(dtype=float, na_value=np.nan)
            side_players = np.unique(
                lineup_numeric[np.isfinite(lineup_numeric) & (lineup_numeric > 0)].astype(np.int64)
            )
            actor_ids = pd.to_numeric(group["player1_id"], errors="coerce")
            candidates = group[actor_ids.isin(side_players) & group["player1_team_id"].gt(0)]
        if candidates.empty:
            candidates = group[group[description].notna() & group["player1_team_id"].gt(0)]
        if candidates.empty:
            raise ValueError(f"No team identity for {group['game_id'].iloc[0]} {description}")
        team_id = int(candidates["player1_team_id"].mode().iloc[0])
        abbreviations = candidates.loc[
            candidates["player1_team_id"].eq(team_id), "player1_team_abbreviation"
        ].dropna()
        team_abbr = str(abbreviations.mode().iloc[0]) if not abbreviations.empty else str(team_id)
        return team_id, team_abbr

    for game_id, group in rows.groupby("game_id", sort=False):
        home_team_id, home_team_abbr = side_identity(
            group, "home_description", RAW_HOME_COLS
        )
        away_team_id, away_team_abbr = side_identity(
            group, "visitor_description", RAW_AWAY_COLS
        )
        if home_team_id == away_team_id:
            raise ValueError(f"Same home and away team for {game_id}")
        scores = group.dropna(subset=["home_score", "away_score"]).sort_values(
            "event_num", kind="mergesort"
        )
        if scores.empty:
            home_score = away_score = np.nan
        else:
            final_score = scores.iloc[-1]
            home_score = float(final_score["home_score"])
            away_score = float(final_score["away_score"])
        game_records.append(
            {
                "game_id": game_id,
                "game_date": date_map.get(game_id, pd.NaT),
                "home_team_id": home_team_id,
                "home_team_abbr": home_team_abbr,
                "away_team_id": away_team_id,
                "away_team_abbr": away_team_abbr,
                "home_score": home_score,
                "away_score": away_score,
            }
        )
        side_counts: dict[int, dict[int, int]] = {}
        team_abbreviations = {
            home_team_id: home_team_abbr,
            away_team_id: away_team_abbr,
        }
        for columns, team_id, team_abbr in (
            (RAW_HOME_COLS, home_team_id, home_team_abbr),
            (RAW_AWAY_COLS, away_team_id, away_team_abbr),
        ):
            numeric = group[columns].apply(pd.to_numeric, errors="coerce").to_numpy(
                dtype=float, na_value=np.nan
            ).ravel()
            players, counts = np.unique(
                numeric[np.isfinite(numeric) & (numeric > 0)].astype(np.int64),
                return_counts=True,
            )
            for player_id, count in zip(players, counts):
                side_counts.setdefault(int(player_id), {})[team_id] = int(count)
        for player_id, counts in side_counts.items():
            maximum = max(counts.values())
            candidates = [team_id for team_id, count in counts.items() if count == maximum]
            if len(candidates) == 1:
                team_id = candidates[0]
            elif "player1_id" in group.columns:
                actor_ids = pd.to_numeric(group["player1_id"], errors="coerce")
                actor_teams = pd.to_numeric(group["player1_team_id"], errors="coerce")
                evidence = actor_teams[actor_ids.eq(player_id) & actor_teams.isin(candidates)]
                team_id = int(evidence.mode().iloc[0]) if not evidence.empty else min(candidates)
            else:
                team_id = min(candidates)
            assignment_records.append(
                {
                    "game_id": game_id,
                    "player_id": player_id,
                    "team_id": team_id,
                    "team_abbr": team_abbreviations[team_id],
                }
            )
    games = pd.DataFrame(game_records).sort_values(["game_date", "game_id"]).reset_index(drop=True)
    assignments = pd.DataFrame(assignment_records).drop_duplicates(
        ["game_id", "player_id"], keep="last"
    )
    return games, assignments.reset_index(drop=True)


def collapse_processed_team_exposures(
    possessions: pd.DataFrame,
    player_game_teams: pd.DataFrame,
) -> pd.DataFrame:
    """Collapse processed O/D lineups into dated exposures with stable team IDs."""
    required = {"game_id", "game_date", *LINEUP_COLS, *OPPONENT_COLS}
    missing = required - set(possessions.columns)
    if missing:
        raise ValueError(f"Processed possessions missing columns: {sorted(missing)}")
    team_required = {"game_id", "player_id", "team_id", "team_abbr"}
    team_missing = team_required - set(player_game_teams.columns)
    if team_missing:
        raise ValueError(f"Player-game teams missing columns: {sorted(team_missing)}")

    games = possessions.copy()
    games["game_id"] = normalize_game_ids(games["game_id"])
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce").dt.normalize()
    assignments = player_game_teams.copy()
    assignments["game_id"] = normalize_game_ids(assignments["game_id"])
    assignments = assignments.drop_duplicates(["game_id", "player_id"])
    assignment_map = assignments.set_index(["game_id", "player_id"])["team_id"]
    abbr_map = assignments.drop_duplicates(["game_id", "team_id"]).set_index(
        ["game_id", "team_id"]
    )["team_abbr"]

    frames: list[pd.DataFrame] = []
    unresolved_total = 0
    valid_total = 0
    for columns in (LINEUP_COLS, OPPONENT_COLS):
        numeric = games[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        ids = np.where(np.isfinite(numeric), numeric, -1).astype(np.int64)
        ids.sort(axis=1)
        valid = (
            games["game_date"].notna().to_numpy()
            & np.isfinite(numeric).all(axis=1)
            & (ids > 0).all(axis=1)
            & np.array([len(set(row.tolist())) == 5 for row in ids])
        )
        mapped = np.full((len(games), 5), np.nan, dtype=float)
        for slot in range(5):
            keys = pd.MultiIndex.from_arrays([games["game_id"], ids[:, slot]])
            mapped[:, slot] = assignment_map.reindex(keys).to_numpy(dtype=float)
        same_team = np.isfinite(mapped).all(axis=1) & (mapped == mapped[:, [0]]).all(axis=1)
        unresolved = valid & ~same_team
        unresolved_total += int(unresolved.sum())
        valid_total += int(valid.sum())
        valid &= same_team
        if not np.any(valid):
            continue
        team_ids = mapped[valid, 0].astype(np.int64)
        abbr_keys = pd.MultiIndex.from_arrays([games.loc[valid, "game_id"], team_ids])
        team_abbr = abbr_map.reindex(abbr_keys).astype(str).to_numpy()
        frame = pd.DataFrame(ids[valid], columns=PLAYER_COLS)
        frame.insert(0, "team_abbr", team_abbr)
        frame.insert(0, "team_id", team_ids)
        frame.insert(0, "game_date", games.loc[valid, "game_date"].to_numpy())
        frame["exposures"] = 1.0
        frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=["game_date", "team_id", "team_abbr", *PLAYER_COLS, "exposures"]
        )
    allowed_unresolved = max(10, int(np.ceil(0.001 * valid_total)))
    if unresolved_total > allowed_unresolved:
        raise ValueError(
            f"Could not assign {unresolved_total} of {valid_total} valid lineups to one team"
        )
    collapsed = (
        pd.concat(frames, ignore_index=True)
        .groupby(["game_date", "team_id", "team_abbr", *PLAYER_COLS], as_index=False)[
            "exposures"
        ]
        .sum()
        .sort_values(["game_date", "team_id", *PLAYER_COLS], kind="mergesort")
        .reset_index(drop=True)
    )
    collapsed.attrs["unresolved_lineups"] = unresolved_total
    collapsed.attrs["candidate_lineups"] = valid_total
    return collapsed


def shapley_teammate_attribution(
    teammates: pd.DataFrame,
    probability_for_help: Callable[[float], float],
    permutations: int = 512,
    seed: int = 0,
) -> pd.DataFrame:
    """Attribute actual-versus-neutral title probability across teammates."""
    if permutations <= 0:
        raise ValueError("permutations must be positive")
    output = teammates.copy()
    if output.empty:
        output["championships_added"] = pd.Series(dtype=float)
        return output
    contributions = pd.to_numeric(output["contribution_dpm"], errors="coerce").fillna(0.0)
    values = contributions.to_numpy(float)
    attribution = np.zeros(len(output), dtype=float)
    rng = np.random.default_rng(seed)
    for _ in range(permutations):
        cumulative = 0.0
        probability = float(probability_for_help(cumulative))
        for index in rng.permutation(len(output)):
            cumulative += values[index]
            next_probability = float(probability_for_help(cumulative))
            attribution[index] += next_probability - probability
            probability = next_probability
    output["championships_added"] = attribution / permutations
    return output


def summarize_team_lineup_exposures(
    exposures: pd.DataFrame,
    histories: dict[int, PlayerRatingHistory],
    season: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize common-sample own and teammate DARKO by player/team/season."""
    required = {"game_date", "team_id", "team_abbr", "exposures", *PLAYER_COLS}
    missing = required - set(exposures.columns)
    if missing:
        raise ValueError(f"Team exposures missing columns: {sorted(missing)}")
    if exposures.empty:
        return pd.DataFrame(), pd.DataFrame()

    dates = pd.to_datetime(exposures["game_date"], errors="coerce").to_numpy(
        dtype="datetime64[D]"
    )
    lineups = exposures[PLAYER_COLS].to_numpy(dtype=np.int64)
    weights = pd.to_numeric(exposures["exposures"], errors="coerce").to_numpy(float)
    rating_values = np.empty((len(exposures), 5, len(METRICS)), dtype=float)
    for slot in range(5):
        values, _, _ = lookup_asof_ratings(lineups[:, slot], dates, histories)
        rating_values[:, slot, :] = values
    fully_rated = np.isfinite(rating_values).all(axis=(1, 2))
    lineup_sums = np.where(np.isfinite(rating_values), rating_values, 0.0).sum(axis=1)

    player_parts: list[pd.DataFrame] = []
    pair_parts: list[pd.DataFrame] = []
    base = {
        "team_id": exposures["team_id"].to_numpy(),
        "team_abbr": exposures["team_abbr"].astype(str).to_numpy(),
    }
    for anchor_slot in range(5):
        own = rating_values[:, anchor_slot, :]
        help_values = lineup_sums - np.where(np.isfinite(own), own, 0.0)
        player_parts.append(
            pd.DataFrame(
                {
                    **base,
                    "player_id": lineups[:, anchor_slot],
                    "game_date": dates,
                    "total_exposures": weights,
                    "rated_exposures": np.where(fully_rated, weights, 0.0),
                    **{
                        f"weighted_own_{metric}": np.where(
                            fully_rated, weights * own[:, metric_index], 0.0
                        )
                        for metric_index, metric in enumerate(METRICS)
                    },
                    **{
                        f"weighted_help_{metric}": np.where(
                            fully_rated, weights * help_values[:, metric_index], 0.0
                        )
                        for metric_index, metric in enumerate(METRICS)
                    },
                }
            )
        )
        for teammate_slot in range(5):
            if teammate_slot == anchor_slot:
                continue
            pair_parts.append(
                pd.DataFrame(
                    {
                        **base,
                        "player_id": lineups[:, anchor_slot],
                        "teammate_id": lineups[:, teammate_slot],
                        "shared_exposures": weights,
                        "rated_exposures": np.where(fully_rated, weights, 0.0),
                        **{
                            f"weighted_{metric}": np.where(
                                fully_rated,
                                weights * rating_values[:, teammate_slot, metric_index],
                                0.0,
                            )
                            for metric_index, metric in enumerate(METRICS)
                        },
                    }
                )
            )

    raw_player = pd.concat(player_parts, ignore_index=True)
    player = (
        raw_player.groupby(
            ["team_id", "team_abbr", "player_id"], as_index=False, sort=False
        )
        .agg(
            first_game_date=("game_date", "min"),
            last_game_date=("game_date", "max"),
            total_exposures=("total_exposures", "sum"),
            rated_exposures=("rated_exposures", "sum"),
            **{
                f"weighted_own_{metric}": (f"weighted_own_{metric}", "sum")
                for metric in METRICS
            },
            **{
                f"weighted_help_{metric}": (f"weighted_help_{metric}", "sum")
                for metric in METRICS
            },
        )
    )
    pair = (
        pd.concat(pair_parts, ignore_index=True)
        .groupby(
            ["team_id", "team_abbr", "player_id", "teammate_id"],
            as_index=False,
            sort=False,
        )
        .sum(numeric_only=True)
    )
    denominators = player.set_index(["team_id", "player_id"])
    player_keys = pd.MultiIndex.from_frame(pair[["team_id", "player_id"]])
    pair_player_rated = denominators["rated_exposures"].reindex(player_keys).to_numpy()
    pair_player_total = denominators["total_exposures"].reindex(player_keys).to_numpy()
    for metric in METRICS:
        player[f"own_{metric}"] = player[f"weighted_own_{metric}"] / player[
            "rated_exposures"
        ].replace(0, np.nan)
        player[f"help_{metric}"] = player[f"weighted_help_{metric}"] / player[
            "rated_exposures"
        ].replace(0, np.nan)
        pair[f"avg_teammate_{metric}"] = pair[f"weighted_{metric}"] / pair[
            "rated_exposures"
        ].replace(0, np.nan)
        pair[f"contribution_{metric}"] = pair[f"weighted_{metric}"] / np.where(
            pair_player_rated > 0, pair_player_rated, np.nan
        )
    player["coverage_pct"] = (
        100.0 * player["rated_exposures"] / player["total_exposures"].replace(0, np.nan)
    )
    pair["share_of_player_exposures_pct"] = (
        100.0 * pair["shared_exposures"] / np.where(pair_player_total > 0, pair_player_total, np.nan)
    )
    player.insert(0, "season", int(season))
    pair.insert(0, "season", int(season))
    return player, pair


def extract_champion_from_games(games: pd.DataFrame, season: int) -> dict[str, object]:
    """Return the winner of the chronologically last completed playoff game."""
    completed = games.copy()
    completed["game_date"] = pd.to_datetime(completed["game_date"], errors="coerce")
    completed["home_score"] = pd.to_numeric(completed["home_score"], errors="coerce")
    completed["away_score"] = pd.to_numeric(completed["away_score"], errors="coerce")
    completed = completed.dropna(subset=["game_date", "home_score", "away_score"])
    completed = completed[completed["home_score"] != completed["away_score"]]
    if completed.empty:
        raise ValueError(f"No completed playoff games for {season}")
    final = completed.sort_values(["game_date", "game_id"], kind="mergesort").iloc[-1]
    home_won = final["home_score"] > final["away_score"]
    return {
        "season": int(season),
        "champion_team_id": int(final["home_team_id"] if home_won else final["away_team_id"]),
        "champion_team_abbr": str(
            final["home_team_abbr"] if home_won else final["away_team_abbr"]
        ),
        "clinching_game_id": str(final["game_id"]),
        "clinching_game_date": final["game_date"].date().isoformat(),
    }


def select_final_team_segments(
    player_team: pd.DataFrame,
    pair_team: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the team each player appeared for latest in each regular season."""
    players = player_team.copy()
    players["last_game_date"] = pd.to_datetime(players["last_game_date"], errors="coerce")
    players = players.sort_values(
        ["season", "player_id", "last_game_date", "total_exposures", "team_id"],
        ascending=[True, True, True, True, True],
        kind="mergesort",
    )
    players = players.drop_duplicates(["season", "player_id"], keep="last")
    players = players.sort_values(["season", "player_id"], kind="mergesort").reset_index(drop=True)

    keys = players[["season", "player_id", "team_id"]]
    pairs = pair_team.merge(keys, on=["season", "player_id", "team_id"], how="inner")
    pairs = pairs.sort_values(
        ["season", "player_id", "team_id", "teammate_id"], kind="mergesort"
    ).reset_index(drop=True)
    return players, pairs


def validate_pair_reconciliation(
    player: pd.DataFrame,
    pair: pd.DataFrame,
    tolerance: float = 1e-10,
) -> dict[str, float | int]:
    sums = pair.groupby(["season", "player_id", "team_id"], as_index=False)[
        [f"contribution_{metric}" for metric in METRICS]
    ].sum(min_count=1)
    merged = player.merge(sums, on=["season", "player_id", "team_id"], how="left")
    report: dict[str, float | int] = {"rows": int(len(merged))}
    for metric in METRICS:
        gap = (merged[f"help_{metric}"] - merged[f"contribution_{metric}"]).abs()
        max_gap = float(gap.max(skipna=True)) if gap.notna().any() else 0.0
        report[f"max_{metric}_gap"] = max_gap
        if max_gap > tolerance:
            raise AssertionError(f"{metric} team-pair gap {max_gap} exceeds {tolerance}")
    return report


def season_file(directory: Path, stem: str, season: int, playoffs: bool = False) -> Path:
    suffix = "_PS" if playoffs else ""
    return directory / f"{stem}{season % 100:02d}{suffix}.parquet"


def build_team_season_data(
    start_year: int,
    end_year: int,
    processed_dir: Path,
    raw_dir: Path,
    histories: dict[int, PlayerRatingHistory],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    """Build regular-season player/team context and playoff champion labels."""
    player_frames: list[pd.DataFrame] = []
    pair_frames: list[pd.DataFrame] = []
    champions: list[dict[str, object]] = []
    reports: list[dict[str, object]] = []
    processed_columns = ["game_id", "game_date", *LINEUP_COLS, *OPPONENT_COLS]

    for season in range(start_year, end_year + 1):
        processed_path = season_file(processed_dir, "RAPM", season)
        raw_path = season_file(raw_dir, "NBA", season)
        playoff_processed_path = season_file(processed_dir, "RAPM", season, playoffs=True)
        playoff_raw_path = season_file(raw_dir, "NBA", season, playoffs=True)
        required_paths = [processed_path, raw_path, playoff_processed_path, playoff_raw_path]
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing {season} inputs: {missing}")

        possessions = pd.read_parquet(processed_path, columns=processed_columns)
        if pd.to_datetime(possessions["game_date"], errors="coerce").notna().mean() < 0.999:
            raise ValueError(f"{processed_path} lacks complete game dates")
        raw = pd.read_parquet(raw_path, columns=RAW_CONTEXT_COLS)
        dates = possessions[["game_id", "game_date"]].drop_duplicates("game_id")
        regular_games, assignments = build_game_context(raw, dates)
        exposures = collapse_processed_team_exposures(possessions, assignments)
        player, pair = summarize_team_lineup_exposures(exposures, histories, season=season)
        player_frames.append(player)
        pair_frames.append(pair)

        playoff_dates = pd.read_parquet(
            playoff_processed_path, columns=["game_id", "game_date"]
        ).drop_duplicates("game_id")
        playoff_raw = pd.read_parquet(playoff_raw_path, columns=RAW_CONTEXT_COLS)
        playoff_games, _ = build_game_context(playoff_raw, playoff_dates)
        champion = extract_champion_from_games(playoff_games, season=season)
        champions.append(champion)
        reports.append(
            {
                "season": season,
                "regular_games": int(len(regular_games)),
                "regular_possessions": int(len(possessions)),
                "team_exposures": int(exposures["exposures"].sum()),
                "player_team_rows": int(len(player)),
                "pair_team_rows": int(len(pair)),
                "champion_team_id": int(champion["champion_team_id"]),
                "champion_team_abbr": str(champion["champion_team_abbr"]),
            }
        )
        print(
            f"{season}: {len(possessions):,} RS possessions, {len(player):,} player-team rows, "
            f"champion {champion['champion_team_abbr']}"
        )

    player_team = pd.concat(player_frames, ignore_index=True)
    pair_team = pd.concat(pair_frames, ignore_index=True)
    champion_frame = pd.DataFrame(champions)
    selected_player, selected_pair = select_final_team_segments(player_team, pair_team)
    selected_player = selected_player.merge(
        champion_frame[["season", "champion_team_id"]], on="season", how="left"
    )
    selected_player["won_championship"] = (
        selected_player["team_id"] == selected_player["champion_team_id"]
    ).astype(int)
    return selected_player, selected_pair, champion_frame, reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=1998)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--darko", type=Path, default=DEFAULT_DARKO_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--app-data-dir", type=Path, default=DEFAULT_APP_DATA_DIR)
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument(
        "--reuse-data",
        action="store_true",
        help="Reuse existing player/team, pair, and champion CSVs in --output-dir.",
    )
    parser.add_argument("--shapley-permutations", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.darko.exists():
        raise FileNotFoundError(f"DARKO history not found: {args.darko}")
    darko, histories = load_darko_history(args.darko)
    names = load_name_map(darko)
    player_path = args.output_dir / "player_team_seasons.csv"
    pair_path = args.output_dir / "player_teammate_team_seasons.csv"
    champions_path = args.output_dir / "champions.csv"
    if args.reuse_data:
        missing = [str(path) for path in (player_path, pair_path, champions_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Cannot reuse missing championship-help inputs: {missing}")
        player = pd.read_csv(player_path)
        pair = pd.read_csv(pair_path)
        champions = pd.read_csv(champions_path)
        reports: list[dict[str, object]] = []
    else:
        player, pair, champions, reports = build_team_season_data(
            args.start_year,
            args.end_year,
            args.processed_dir,
            args.raw_dir,
            histories,
        )
        player = attach_names(player, names)
        pair = attach_names(pair, names)
    reconciliation = validate_pair_reconciliation(player, pair)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    player.to_csv(player_path, index=False)
    pair.to_csv(pair_path, index=False)
    champions.to_csv(champions_path, index=False)
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "start_season": int(args.start_year),
        "end_season": int(args.end_year),
        "player_seasons": int(len(player)),
        "pair_seasons": int(len(pair)),
        "common_sample_coverage_pct": float(
            100.0 * player["rated_exposures"].sum() / player["total_exposures"].sum()
        ),
        "reconciliation": reconciliation,
        "seasons": reports,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote data-only championship-help inputs to {args.output_dir}")
    print(f"Reconciliation: {reconciliation}")
    if args.data_only:
        return

    sample = prepare_model_sample(player)
    diagnostics, selected, oof_predictions = evaluate_model_grid(sample)
    model = fit_championship_odds_model(
        sample,
        spec=str(selected["spec"]),
        c_value=float(selected["c_value"]),
    )
    seasons, attributed, shapley_report = attribute_model_probabilities(
        sample,
        pair,
        model,
        permutations=args.shapley_permutations,
    )
    career, teammate, career_report = build_career_championship_help(seasons, attributed)
    own_best = diagnostics[diagnostics["spec"].eq("own")].sort_values("log_loss").iloc[0]
    selected["log_loss_improvement_vs_own_only"] = float(
        own_best["log_loss"] - selected["log_loss"]
    )
    selected["brier_improvement_vs_own_only"] = float(
        own_best["brier"] - selected["brier"]
    )
    selected["model_rows"] = int(len(sample))
    selected["championship_player_seasons"] = int(sample["won_championship"].sum())
    selected["seasons"] = int(sample["season"].nunique())
    selected["minimum_exposures"] = MODEL_MIN_EXPOSURES
    selected["minimum_coverage_pct"] = MODEL_MIN_COVERAGE
    selected["neutral_teammate_baseline_dpm"] = 0.0
    selected["probability_curves"] = build_probability_curves(model)

    diagnostics.to_csv(args.output_dir / "model_cv.csv", index=False)
    oof_predictions.to_csv(args.output_dir / "model_oof_predictions.csv", index=False)
    seasons.to_csv(args.output_dir / "player_season_estimates.csv", index=False)
    attributed.to_csv(args.output_dir / "player_teammate_season_estimates.csv", index=False)
    career.to_csv(args.output_dir / "career_player_estimates.csv", index=False)
    teammate.to_csv(args.output_dir / "career_teammate_estimates.csv", index=False)
    metadata.update(
        {
            "model": selected,
            "shapley": shapley_report,
            "career_reconciliation": career_report,
            "interpretation": (
                "Estimated change in a player's modeled championship probability relative to "
                "the same own DARKO with neutral (0.0 DPM) teammate help. This is contextual "
                "association, not a causal championship count."
            ),
            "leakage_guard": (
                "Only regular-season exposures and backward-looking DARKO snapshots are model "
                "inputs. Playoff data supplies the champion label only."
            ),
        }
    )
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    write_app_championship_data(args.app_data_dir, career, seasons, teammate, metadata)
    print(
        f"Selected {selected['spec']} C={selected['c_value']} by LOYO log loss "
        f"({selected['log_loss']:.6f}); wrote {len(career):,} career estimates and "
        f"{len(teammate):,} career teammate attributions"
    )
    print(f"Shapley reconciliation: {shapley_report}")


if __name__ == "__main__":
    main()
