#!/usr/bin/env python3
"""Build retrospective playoff-rotation estimates of championship title lift."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import gzip
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

try:
    from nba_pipeline.scripts.build_darko_teammate_help import (
        METRICS,
        PlayerRatingHistory,
        lookup_asof_ratings,
    )
except ModuleNotFoundError:  # Direct script execution from the repository root.
    from build_darko_teammate_help import (  # type: ignore[no-redef]
        METRICS,
        PlayerRatingHistory,
        lookup_asof_ratings,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_DIR.parent
PROCESSED_DIR = PIPELINE_DIR / "processed"
RAW_DIR = PIPELINE_DIR / "raw_data"
DEFAULT_EPM_CACHE_DIR = PIPELINE_DIR / "cache" / "epm_title"
DEFAULT_OUTPUT_DIR = PIPELINE_DIR / "results" / "championship_help"
DEFAULT_APP_DATA_DIR = PROJECT_ROOT / "teammate-help-app" / "public" / "data" / "championship-help"
DEFAULT_SALARY_PATH = (
    PROJECT_ROOT.parents[1]
    / "pipelines"
    / "core"
    / "nba_rapm"
    / "data"
    / "inputs"
    / "nba_salaries.csv"
)
MODEL_MIN_EXPOSURES = 1
TEAM_MODEL_SPECS = ("overall", "sides")
TEAM_MODEL_C_VALUES = (0.05, 0.2, 1.0, 5.0, 20.0, 100.0)
APP_SHARDS = 64
TITLE_FIELD_SIZE = 16
# Contract-EPM v1 keeps the prototype assumptions already used by the
# From Impact to Winning reader, expressed in cap-share units so the hurdle is
# comparable across eras. These are declared assumptions, not a fitted market
# price. See docs/TITLE_LIFT_OVER_CONTRACT_DATA.md.
CONTRACT_HURDLE_VERSION = "salary-share-v1"
CONTRACT_REPLACEMENT_EPM = -2.0
CONTRACT_REFERENCE_CAP = 154_647_000.0
CONTRACT_REPLACEMENT_SALARY_SHARE = 2_100_000.0 / CONTRACT_REFERENCE_CAP
CONTRACT_CAP_SHARE_PER_WAR = 8_500_000.0 / CONTRACT_REFERENCE_CAP
CONTRACT_NORMALIZED_SEASON_POSSESSIONS = 60.0 * 82.0
CONTRACT_WINS_PER_POINT = 0.0297496
CONTRACT_WAR_PER_EPM = (
    CONTRACT_NORMALIZED_SEASON_POSSESSIONS / 100.0 * CONTRACT_WINS_PER_POINT
)
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


def contract_epm_from_salary_share(salary_share: float | np.ndarray) -> float | np.ndarray:
    """Translate salary share to the v1 contract-implied EPM hurdle."""
    share = np.asarray(salary_share, dtype=float)
    contract_war = (
        share - CONTRACT_REPLACEMENT_SALARY_SHARE
    ) / CONTRACT_CAP_SHARE_PER_WAR
    result = CONTRACT_REPLACEMENT_EPM + contract_war / CONTRACT_WAR_PER_EPM
    if result.ndim == 0:
        return float(result)
    return result


def load_salary_history(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load the private salary/NBA-ID bridge and retain derived salary share only."""
    if not path.exists():
        raise FileNotFoundError(f"Salary history not found: {path}")
    raw = pd.read_csv(path, low_memory=False)
    aliases = {
        "year": "season",
        "nba_id": "player_id",
        "team": "salary_team_abbr",
        "cap": "salary_cap",
        "cap%": "salary_share_pct",
    }
    salary = raw.rename(columns=aliases)
    required = {
        "season",
        "player_id",
        "salary_team_abbr",
        "salary",
        "salary_cap",
    }
    missing = sorted(required - set(salary.columns))
    if missing:
        raise ValueError(f"Salary history is missing columns: {missing}")
    salary = salary[
        [
            "season",
            "player_id",
            "salary_team_abbr",
            "salary",
            "salary_cap",
        ]
    ].copy()
    for column in ("season", "player_id", "salary", "salary_cap"):
        salary[column] = pd.to_numeric(salary[column], errors="coerce")
    salary = salary.dropna(
        subset=["season", "player_id", "salary", "salary_cap", "salary_team_abbr"]
    ).copy()
    salary = salary[salary["salary_cap"].gt(0) & salary["salary"].ge(0)]
    salary["season"] = salary["season"].astype(np.int64)
    salary["player_id"] = salary["player_id"].astype(np.int64)
    salary["salary_team_abbr"] = salary["salary_team_abbr"].astype(str)
    salary["salary_share"] = salary["salary"] / salary["salary_cap"]
    salary = salary.sort_values(
        ["season", "player_id", "salary_team_abbr", "salary"],
        ascending=[True, True, True, False],
        kind="mergesort",
    ).drop_duplicates(
        ["season", "player_id", "salary_team_abbr"], keep="first"
    )
    report = {
        "source_file": path.name,
        "rows": int(len(salary)),
        "start_season": int(salary["season"].min()),
        "end_season": int(salary["season"].max()),
        "source_unit": "annual salary divided by official NBA salary cap",
    }
    return salary.reset_index(drop=True), report


def attach_salary_share(
    players: pd.DataFrame,
    salary_history: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Attach only exact team-season salary rows; never guess or zero-fill cost."""
    salary = salary_history.rename(columns={"salary_team_abbr": "team_abbr"})
    columns = [
        "season",
        "player_id",
        "team_abbr",
        "salary",
        "salary_cap",
        "salary_share",
    ]
    output = players.merge(
        salary[columns],
        on=["season", "player_id", "team_abbr"],
        how="left",
        validate="many_to_one",
    )
    output["salary_mapping_method"] = np.where(
        output["salary"].notna(), "exact_team_player_season", None
    )
    modeled = len(output)
    covered = int(output["salary_share"].notna().sum())
    by_season = (
        output.assign(covered=output["salary_share"].notna())
        .groupby("season", as_index=False)
        .agg(player_seasons=("player_id", "size"), covered=("covered", "sum"))
    )
    by_season["coverage_pct"] = 100.0 * by_season["covered"] / by_season[
        "player_seasons"
    ]
    report = {
        "player_seasons": modeled,
        "covered_player_seasons": covered,
        "coverage_pct": 100.0 * covered / modeled if modeled else 0.0,
        "by_season": by_season.to_dict(orient="records"),
        "missing_policy": "missing remains missing; no team or player fallback",
    }
    return output, report


def add_title_lift_over_contract(
    seasons: pd.DataFrame,
    neutral_title_model: TeamTitleModel,
) -> pd.DataFrame:
    """Price actual and Contract EPM at identical workload in one fixed field."""
    if neutral_title_model.feature_columns != ("team_dpm",):
        raise ValueError(
            "Title Lift over Contract requires the selected overall-EPM title model"
        )
    output = seasons.copy()
    score_per_epm = float(
        neutral_title_model.coefficients[0] / neutral_title_model.scales[0]
    )
    output["playoff_workload_share"] = (
        pd.to_numeric(output["total_exposures"], errors="coerce")
        / pd.to_numeric(output["team_total_exposures"], errors="coerce").replace(0, np.nan)
    )
    output["contract_epm"] = contract_epm_from_salary_share(output["salary_share"])
    output["epm_over_contract"] = output["own_dpm"] - output["contract_epm"]
    actual_score = (
        score_per_epm * output["own_dpm"] * output["playoff_workload_share"]
    )
    contract_score = (
        score_per_epm * output["contract_epm"] * output["playoff_workload_share"]
    )
    field_log_odds = np.log(TITLE_FIELD_SIZE - 1)
    output["neutral_contract_actual_title_probability"] = np.where(
        output["salary_share"].notna(),
        expit(actual_score - field_log_odds),
        np.nan,
    )
    output["neutral_contract_hurdle_title_probability"] = expit(
        contract_score - field_log_odds
    )
    output["title_lift_over_contract"] = (
        output["neutral_contract_actual_title_probability"]
        - output["neutral_contract_hurdle_title_probability"]
    )
    output["contract_hurdle_version"] = np.where(
        output["salary_share"].notna(), CONTRACT_HURDLE_VERSION, None
    )
    return output


def load_epm_snapshot_history(
    cache_dir: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load checksummed-style game-agnostic EPM snapshots without publishing raw rows."""
    paths = sorted(cache_dir.glob("epm_all_*_go0.csv"))
    if not paths:
        raise FileNotFoundError(f"No game-agnostic EPM snapshots found in {cache_dir}")
    frames: list[pd.DataFrame] = []
    for path in paths:
        match = re.fullmatch(r"epm_all_(\d{4}-\d{2}-\d{2})_go0\.csv", path.name)
        if match is None:
            continue
        frame = pd.read_csv(
            path,
            usecols=lambda column: column
            in {
                "requested_date",
                "date",
                "game_dt",
                "game_optimized",
                "player_id",
                "player_name",
                "oepm",
                "depm",
                "epm",
            },
            low_memory=False,
        )
        frame["snapshot_requested_date"] = pd.Timestamp(match.group(1))
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No canonical EPM snapshot filenames found in {cache_dir}")
    history = pd.concat(frames, ignore_index=True)
    history["nba_id"] = pd.to_numeric(history["player_id"], errors="coerce")
    history["provider_effective_date"] = pd.to_datetime(
        history["date"], errors="coerce"
    ).dt.normalize()
    history["game_dt"] = pd.to_datetime(history["game_dt"], errors="coerce").dt.normalize()
    metric_map = {"epm": "dpm", "oepm": "o_dpm", "depm": "d_dpm"}
    for source, target in metric_map.items():
        history[target] = pd.to_numeric(history[source], errors="coerce")
    history = history.dropna(
        subset=["nba_id", "snapshot_requested_date", "dpm", "o_dpm", "d_dpm"]
    ).copy()
    history["nba_id"] = history["nba_id"].astype(np.int64)
    history = history.sort_values(
        ["snapshot_requested_date", "nba_id"], kind="mergesort"
    ).drop_duplicates(["snapshot_requested_date", "nba_id"], keep="last")
    report = {
        "snapshots": int(history["snapshot_requested_date"].nunique()),
        "first_requested_date": history["snapshot_requested_date"].min().date().isoformat(),
        "last_requested_date": history["snapshot_requested_date"].max().date().isoformat(),
        "rows": int(len(history)),
    }
    return history.reset_index(drop=True), report


def select_epm_snapshot(epm_history: pd.DataFrame, as_of_date: str | pd.Timestamp) -> pd.DataFrame:
    """Select the latest complete monthly snapshot available on/before a cutoff."""
    cutoff = pd.Timestamp(as_of_date).normalize()
    eligible_dates = epm_history.loc[
        epm_history["snapshot_requested_date"].le(cutoff),
        "snapshot_requested_date",
    ]
    if eligible_dates.empty:
        raise ValueError(f"No EPM snapshot is available on or before {cutoff.date()}")
    requested_date = eligible_dates.max()
    snapshot = epm_history[
        epm_history["snapshot_requested_date"].eq(requested_date)
    ].copy()
    snapshot["season"] = int(
        cutoff.year + 1 if cutoff.month >= 7 else cutoff.year
    )
    snapshot["date"] = snapshot["provider_effective_date"]
    return snapshot[
        [
            "nba_id",
            "player_name",
            "season",
            "snapshot_requested_date",
            "provider_effective_date",
            "date",
            "game_dt",
            *METRICS,
        ]
    ].reset_index(drop=True)


def attach_neutral_playoff_baseline(teams: pd.DataFrame) -> pd.DataFrame:
    """Attach one common, season-specific average playoff-rotation baseline."""
    output = teams.copy()
    for metric in METRICS:
        output[f"neutral_team_{metric}"] = output.groupby("season")[
            f"team_{metric}"
        ].transform("mean")
    return output


@dataclass(frozen=True)
class TeamTitleModel:
    """Conditional team-season title model with within-season softmax odds."""

    spec: str
    c_value: float
    feature_columns: tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray
    coefficients: np.ndarray

    def score(self, rows: pd.DataFrame) -> np.ndarray:
        values = rows.loc[:, list(self.feature_columns)].to_numpy(dtype=float)
        standardized = (values - self.means) / self.scales
        return standardized @ self.coefficients

    def predict_probabilities(self, rows: pd.DataFrame) -> np.ndarray:
        scores = self.score(rows)
        probability = np.empty(len(rows), dtype=float)
        seasons = rows["season"].to_numpy()
        for season in pd.unique(seasons):
            indexes = np.flatnonzero(seasons == season)
            group_scores = scores[indexes]
            probability[indexes] = np.exp(group_scores - logsumexp(group_scores))
        return probability

    def target_probabilities(
        self,
        states: pd.DataFrame,
        actual_season_rows: pd.DataFrame,
        team_id: int,
    ) -> np.ndarray:
        """Probability for one team over counterfactual states, opponents fixed."""
        opponent_rows = actual_season_rows[actual_season_rows["team_id"].ne(team_id)]
        opponent_scores = self.score(opponent_rows)
        state_scores = self.score(states)
        return expit(state_scores - float(logsumexp(opponent_scores)))

    def common_field_target_probabilities(
        self,
        states: pd.DataFrame,
        actual_season_rows: pd.DataFrame,
    ) -> np.ndarray:
        """Score target states against one shared 15-opponent season field."""
        field_scores = self.score(actual_season_rows)
        if len(field_scores) <= 1:
            raise ValueError("A common title field requires at least two playoff teams")
        opponent_log_mass = float(
            logsumexp(field_scores) + np.log((len(field_scores) - 1) / len(field_scores))
        )
        return expit(self.score(states) - opponent_log_mass)


def _team_model_columns(spec: str) -> tuple[str, ...]:
    if spec == "overall":
        return ("team_dpm",)
    if spec == "sides":
        return ("team_o_dpm", "team_d_dpm")
    raise ValueError(f"Unknown team model spec: {spec}")


def fit_team_title_model(
    rows: pd.DataFrame,
    spec: str,
    c_value: float,
) -> TeamTitleModel:
    """Fit a regularized conditional-logit model over each season's teams."""
    feature_columns = _team_model_columns(spec)
    values = rows.loc[:, list(feature_columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Team title-model features must be finite")
    means = values.mean(axis=0)
    scales = values.std(axis=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    standardized = (values - means) / scales
    seasons = rows["season"].to_numpy()
    outcomes = rows["won_championship"].to_numpy(dtype=int)
    groups: list[tuple[np.ndarray, int]] = []
    for season in pd.unique(seasons):
        indexes = np.flatnonzero(seasons == season)
        champion_indexes = np.flatnonzero(outcomes[indexes] == 1)
        if len(champion_indexes) != 1:
            raise ValueError(f"Season {season} must contain exactly one champion team")
        groups.append((indexes, int(champion_indexes[0])))

    penalty = 1.0 / float(c_value)

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        scores = standardized @ coefficients
        loss = 0.5 * penalty * float(coefficients @ coefficients)
        gradient = penalty * coefficients.copy()
        for indexes, champion_position in groups:
            group_scores = scores[indexes]
            group_probability = np.exp(group_scores - logsumexp(group_scores))
            loss += float(logsumexp(group_scores) - group_scores[champion_position])
            gradient += standardized[indexes].T @ group_probability
            gradient -= standardized[indexes[champion_position]]
        return loss, gradient

    result = minimize(
        fun=lambda coefficients: objective(coefficients),
        x0=np.full(len(feature_columns), 0.5, dtype=float),
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * len(feature_columns),
    )
    if not result.success:
        raise RuntimeError(f"Team title model failed to converge: {result.message}")
    return TeamTitleModel(
        spec=spec,
        c_value=float(c_value),
        feature_columns=feature_columns,
        means=means,
        scales=scales,
        coefficients=np.asarray(result.x, dtype=float),
    )


def cross_validated_team_predictions(
    teams: pd.DataFrame,
    spec: str,
    c_value: float,
) -> tuple[pd.DataFrame, dict[int, TeamTitleModel]]:
    """Leave one complete season out and return normalized team probabilities."""
    predictions: list[pd.DataFrame] = []
    models: dict[int, TeamTitleModel] = {}
    for season in sorted(pd.to_numeric(teams["season"], errors="raise").astype(int).unique()):
        train = teams[teams["season"].ne(season)]
        test = teams[teams["season"].eq(season)]
        model = fit_team_title_model(train, spec=spec, c_value=c_value)
        models[int(season)] = model
        fold = test[["season", "team_id", "won_championship"]].copy()
        fold["predicted_probability"] = model.predict_probabilities(test)
        predictions.append(fold)
    output = pd.concat(predictions, ignore_index=True)
    annual_mass = output.groupby("season")["predicted_probability"].sum()
    max_gap = float((annual_mass - 1.0).abs().max())
    if max_gap > 1e-12:
        raise AssertionError(f"Annual team title probabilities miss 1 by {max_gap}")
    return output, models


def score_team_probability_predictions(predictions: pd.DataFrame) -> dict[str, float]:
    """Score one mutually exclusive champion choice per season."""
    champion_probability = predictions.loc[
        predictions["won_championship"].eq(1), ["season", "predicted_probability"]
    ].copy()
    champion_loss = -np.log(champion_probability["predicted_probability"].clip(lower=1e-15))
    season_brier = predictions.assign(
        squared_error=(
            predictions["predicted_probability"] - predictions["won_championship"]
        )
        ** 2
    ).groupby("season")["squared_error"].sum()
    top = predictions.sort_values(
        ["season", "predicted_probability", "team_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).drop_duplicates("season")
    annual_mass = predictions.groupby("season")["predicted_probability"].sum()
    return {
        "champion_log_loss": float(champion_loss.mean()),
        "champion_log_loss_se": float(
            champion_loss.std(ddof=1) / np.sqrt(len(champion_loss))
        ),
        "season_brier": float(season_brier.mean()),
        "top_one_accuracy": float(top["won_championship"].mean()),
        "max_annual_probability_sum_gap": float((annual_mass - 1.0).abs().max()),
    }


def select_team_model_candidate(diagnostics: pd.DataFrame) -> dict[str, object]:
    """Apply a one-standard-error rule to avoid unstable title-odds tails."""
    ordered = diagnostics.sort_values(
        ["champion_log_loss", "season_brier", "spec", "c_value"], kind="mergesort"
    ).reset_index(drop=True)
    best = ordered.iloc[0]
    threshold = float(best["champion_log_loss"] + best["champion_log_loss_se"])
    eligible = ordered[ordered["champion_log_loss"].le(threshold)].copy()
    eligible["feature_count"] = eligible["spec"].map(
        lambda value: len(_team_model_columns(str(value)))
    )
    winner = eligible.sort_values(
        ["c_value", "feature_count", "champion_log_loss", "season_brier", "spec"],
        kind="mergesort",
    ).iloc[0]
    return {
        "spec": str(winner["spec"]),
        "c_value": float(winner["c_value"]),
        "selection_metric": (
            "one-standard-error rule on leave-one-season-out champion log loss"
        ),
        "one_se_threshold": threshold,
        "best_cv_log_loss": float(best["champion_log_loss"]),
        "champion_log_loss": float(winner["champion_log_loss"]),
        "champion_log_loss_se": float(winner["champion_log_loss_se"]),
        "season_brier": float(winner["season_brier"]),
        "top_one_accuracy": float(winner["top_one_accuracy"]),
        "max_annual_probability_sum_gap": float(
            winner["max_annual_probability_sum_gap"]
        ),
    }


def evaluate_team_model_grid(
    teams: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[str, object],
    pd.DataFrame,
    dict[int, TeamTitleModel],
]:
    records: list[dict[str, object]] = []
    cache: dict[tuple[str, float], tuple[pd.DataFrame, dict[int, TeamTitleModel]]] = {}
    for spec in TEAM_MODEL_SPECS:
        for c_value in TEAM_MODEL_C_VALUES:
            predictions, models = cross_validated_team_predictions(teams, spec, c_value)
            cache[(spec, c_value)] = (predictions, models)
            records.append(
                {
                    "spec": spec,
                    "c_value": c_value,
                    **score_team_probability_predictions(predictions),
                }
            )
    diagnostics = pd.DataFrame(records).sort_values(
        ["champion_log_loss", "season_brier", "spec", "c_value"], kind="mergesort"
    ).reset_index(drop=True)
    selected = select_team_model_candidate(diagnostics)
    predictions, models = cache[(selected["spec"], selected["c_value"])]
    return diagnostics, selected, predictions, models


def attribute_team_probabilities(
    players: pd.DataFrame,
    teammates: pd.DataFrame,
    teams: pd.DataFrame,
    fold_models: dict[int, TeamTitleModel],
    permutations: int = 128,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    """Allocate non-linear team odds with a split-invariant continuous Shapley path."""
    if permutations <= 0:
        raise ValueError("permutations must be positive")
    team_probability_parts: list[pd.DataFrame] = []
    for season, choices in teams.groupby("season", sort=True):
        model = fold_models[int(season)]
        part = choices[["season", "team_id"]].copy()
        part["title_probability"] = model.predict_probabilities(choices)
        team_probability_parts.append(part)
    team_probabilities = pd.concat(team_probability_parts, ignore_index=True)
    annual_mass = team_probabilities.groupby("season")["title_probability"].sum()
    annual_gap = float((annual_mass - 1.0).abs().max())
    if annual_gap > 1e-12:
        raise AssertionError(f"Actual team title probabilities miss 1 by {annual_gap}")

    seasons = players.merge(
        team_probabilities, on=["season", "team_id"], how="left", validate="many_to_one"
    )
    seasons["neutral_teammate_title_probability"] = np.nan
    seasons["self_context_title_probability"] = np.nan
    seasons["neutral_team_title_probability"] = np.nan
    seasons["neutral_field_neutral_title_probability"] = np.nan
    seasons["neutral_field_hard_carry_title_probability"] = np.nan
    seasons["era_neutral_title_probability"] = np.nan
    seasons["era_neutral_hard_carry_title_probability"] = np.nan
    seasons["normalized_self_title_probability_added"] = np.nan
    seasons["self_replacement_title_probability_added"] = np.nan
    seasons["on_court_teammate_title_probability_added"] = np.nan
    seasons["bench_title_probability_added"] = np.nan
    seasons["teammate_help_championships"] = np.nan
    attributed = teammates.copy()
    attributed["championships_added"] = 0.0
    max_gap = 0.0
    max_actual_state_gap = 0.0
    max_bucket_gap = 0.0
    max_score_component_gap = 0.0

    for season_index, season_row in seasons.iterrows():
        season = int(season_row["season"])
        player_id = int(season_row["player_id"])
        team_id = int(season_row["team_id"])
        model = fold_models[season]
        choices = teams[teams["season"].eq(season)]
        group = attributed[
            attributed["season"].eq(season)
            & attributed["player_id"].eq(player_id)
            & attributed["team_id"].eq(team_id)
        ]
        count = len(group)

        states = pd.DataFrame({"season": [season] * 4, "team_id": [team_id] * 4})
        for metric in METRICS:
            own = float(season_row[f"own_team_contribution_{metric}"])
            own_normalized = float(season_row[f"own_normalized_contribution_{metric}"])
            neutral = float(season_row[f"neutral_team_{metric}"])
            actual = float(season_row[f"team_{metric}"])
            states[f"team_{metric}"] = [
                neutral,
                neutral + own_normalized,
                actual - own,
                actual,
            ]
        state_probability = model.target_probabilities(states, choices, team_id)
        neutral_probability, hard_carry_probability, without_self_probability, full_context = map(
            float, state_probability
        )
        common_field_probability = model.common_field_target_probabilities(
            states.iloc[:2], choices
        )
        common_field_neutral_probability, common_field_hard_carry_probability = map(
            float, common_field_probability
        )
        actual_probability = float(season_row["title_probability"])
        feature_metrics = [column.removeprefix("team_") for column in model.feature_columns]
        score_weights = {
            metric: float(model.coefficients[index] / model.scales[index])
            for index, metric in enumerate(feature_metrics)
        }

        def component_score(prefix: str) -> float:
            return float(
                sum(
                    float(season_row[f"{prefix}_{metric}"]) * score_weights.get(metric, 0.0)
                    for metric in METRICS
                )
            )

        self_score = component_score("own_normalized_contribution")
        on_court_score = component_score("on_court_normalized_help")
        bench_score = component_score("bench_normalized_help")
        era_neutral_probability = 1.0 / TITLE_FIELD_SIZE
        era_neutral_hard_carry_probability = float(
            expit(self_score - np.log(TITLE_FIELD_SIZE - 1))
        )
        total_score_delta = self_score + on_court_score + bench_score
        actual_score_delta = float(model.score(states.iloc[[3]])[0] - model.score(states.iloc[[0]])[0])
        score_gap = abs(total_score_delta - actual_score_delta)
        max_score_component_gap = max(max_score_component_gap, score_gap)
        if abs(total_score_delta) > 1e-12:
            response_multiplier = (actual_probability - neutral_probability) / total_score_delta
        else:
            response_multiplier = neutral_probability * (1.0 - neutral_probability)
        normalized_self = self_score * response_multiplier
        on_court_added = on_court_score * response_multiplier
        bench_added = bench_score * response_multiplier
        bucket_gap = abs(
            neutral_probability
            + normalized_self
            + on_court_added
            + bench_added
            - actual_probability
        )
        max_bucket_gap = max(max_bucket_gap, bucket_gap)
        max_actual_state_gap = max(
            max_actual_state_gap, abs(full_context - actual_probability)
        )
        seasons.loc[season_index, "neutral_teammate_title_probability"] = hard_carry_probability
        seasons.loc[season_index, "self_context_title_probability"] = hard_carry_probability
        seasons.loc[season_index, "neutral_team_title_probability"] = neutral_probability
        seasons.loc[
            season_index, "neutral_field_neutral_title_probability"
        ] = common_field_neutral_probability
        seasons.loc[
            season_index, "neutral_field_hard_carry_title_probability"
        ] = common_field_hard_carry_probability
        seasons.loc[season_index, "era_neutral_title_probability"] = (
            era_neutral_probability
        )
        seasons.loc[
            season_index, "era_neutral_hard_carry_title_probability"
        ] = era_neutral_hard_carry_probability
        seasons.loc[season_index, "normalized_self_title_probability_added"] = normalized_self
        seasons.loc[season_index, "self_replacement_title_probability_added"] = (
            actual_probability - without_self_probability
        )
        seasons.loc[
            season_index, "on_court_teammate_title_probability_added"
        ] = on_court_added
        seasons.loc[season_index, "bench_title_probability_added"] = bench_added
        seasons.loc[season_index, "teammate_help_championships"] = on_court_added + bench_added

        if count == 0:
            if abs(on_court_added + bench_added) > 1e-10:
                raise AssertionError("Missing teammate rows for a non-neutral team context")
            continue

        teammate_score = np.zeros(count, dtype=float)
        for metric in METRICS:
            teammate_score += (
                pd.to_numeric(group[f"normalized_contribution_{metric}"], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=float)
                * score_weights.get(metric, 0.0)
            )
        attribution = teammate_score * response_multiplier
        attributed.loc[group.index, "championships_added"] = attribution
        target = on_court_added + bench_added
        gap = abs(float(attribution.sum()) - target)
        max_gap = max(max_gap, gap)

    if max_actual_state_gap > 1e-10:
        raise AssertionError(
            f"Roster contributions do not reproduce actual team state: {max_actual_state_gap}"
        )
    if max_gap > 1e-10:
        raise AssertionError(f"Shapley attribution gap {max_gap} exceeds tolerance")
    if max_bucket_gap > 1e-10:
        raise AssertionError(f"Three-bucket probability gap {max_bucket_gap} exceeds tolerance")
    if max_score_component_gap > 1e-10:
        raise AssertionError(
            f"Self/on-court/bench score gap {max_score_component_gap} exceeds tolerance"
        )
    attributed = attributed.merge(
        seasons[
            [
                "season",
                "player_id",
                "team_id",
                "won_championship",
                "title_probability",
                "neutral_teammate_title_probability",
                "neutral_team_title_probability",
                "normalized_self_title_probability_added",
            ]
        ],
        on=["season", "player_id", "team_id"],
        how="left",
        validate="many_to_one",
    )
    report = {
        "player_seasons": int(len(seasons)),
        "attributed_pair_seasons": int(len(attributed)),
        "permutations": int(permutations),
        "allocation_method": (
            "continuous Shapley / integrated gradients from the season-average "
            "playoff rotation"
        ),
        "max_actual_annual_probability_sum_gap": annual_gap,
        "max_actual_state_reconciliation_gap": float(max_actual_state_gap),
        "max_shapley_reconciliation_gap": float(max_gap),
        "max_three_bucket_probability_gap": float(max_bucket_gap),
        "max_score_component_gap": float(max_score_component_gap),
    }
    return seasons, attributed, report


def build_career_championship_help(
    seasons: pd.DataFrame,
    attributed: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    seasons = seasons.sort_values(["player_id", "season"], kind="mergesort")
    career = (
        seasons.groupby(["player_id"], as_index=False, sort=False)
        .agg(
            player_name=("player_name", "last"),
            start_season=("season", "min"),
            end_season=("season", "max"),
            seasons_modeled=("season", "nunique"),
            actual_championships=("won_championship", "sum"),
            expected_championships=("title_probability", "sum"),
            neutral_expected_championships=("neutral_teammate_title_probability", "sum"),
            self_context_championships=("self_context_title_probability", "sum"),
            neutral_team_championships=("neutral_team_title_probability", "sum"),
            neutral_field_neutral_championships=(
                "neutral_field_neutral_title_probability",
                "sum",
            ),
            neutral_field_hard_carry_championships=(
                "neutral_field_hard_carry_title_probability",
                "sum",
            ),
            era_neutral_championships=("era_neutral_title_probability", "sum"),
            era_neutral_hard_carry_championships=(
                "era_neutral_hard_carry_title_probability",
                "sum",
            ),
            normalized_self_championships=(
                "normalized_self_title_probability_added",
                "sum",
            ),
            self_replacement_championships=(
                "self_replacement_title_probability_added",
                "sum",
            ),
            on_court_teammate_championships=(
                "on_court_teammate_title_probability_added",
                "sum",
            ),
            bench_help_championships=("bench_title_probability_added", "sum"),
            teammate_help_championships=("teammate_help_championships", "sum"),
        )
    )
    contract_summary = (
        seasons.groupby("player_id", as_index=False, sort=False)
        .agg(
            contract_seasons_covered=("title_lift_over_contract", "count"),
            title_lift_over_contract_championships=(
                "title_lift_over_contract",
                lambda values: values.sum(min_count=1),
            ),
            neutral_contract_actual_championships=(
                "neutral_contract_actual_title_probability",
                lambda values: values.sum(min_count=1),
            ),
            neutral_contract_hurdle_championships=(
                "neutral_contract_hurdle_title_probability",
                lambda values: values.sum(min_count=1),
            ),
        )
    )
    career = career.merge(contract_summary, on="player_id", how="left", validate="one_to_one")
    career["contract_coverage_pct"] = (
        100.0 * career["contract_seasons_covered"] / career["seasons_modeled"]
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
        attributed.sort_values(
            ["player_id", "teammate_id", "season"], kind="mergesort"
        )
        .groupby(["player_id", "teammate_id"], as_index=False, sort=False)
        .agg(
            player_name=("player_name", "last"),
            teammate_name=("teammate_name", "last"),
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
    bucket_gap = (
        career["neutral_team_championships"]
        + career["normalized_self_championships"]
        + career["on_court_teammate_championships"]
        + career["bench_help_championships"]
        - career["expected_championships"]
    ).abs()
    contract_gap = (
        career["neutral_contract_actual_championships"]
        - career["neutral_contract_hurdle_championships"]
        - career["title_lift_over_contract_championships"]
    ).abs()
    report = {
        "max_career_reconciliation_gap": float(career_gap.max()),
        "max_career_three_bucket_gap": float(bucket_gap.max()),
        "max_career_contract_lift_gap": float(contract_gap.max()),
    }
    if report["max_career_reconciliation_gap"] > 1e-10:
        raise AssertionError("Career teammate championship-help rows do not reconcile")
    if report["max_career_three_bucket_gap"] > 1e-10:
        raise AssertionError("Career self/on-court/bench buckets do not reconcile")
    if report["max_career_contract_lift_gap"] > 1e-10:
        raise AssertionError("Career Title Lift over Contract rows do not reconcile")
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
    attributed: pd.DataFrame,
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
        "self_context_championships",
        "neutral_team_championships",
        "neutral_field_neutral_championships",
        "neutral_field_hard_carry_championships",
        "era_neutral_championships",
        "era_neutral_hard_carry_championships",
        "contract_seasons_covered",
        "contract_coverage_pct",
        "title_lift_over_contract_championships",
        "neutral_contract_actual_championships",
        "neutral_contract_hurdle_championships",
        "normalized_self_championships",
        "self_replacement_championships",
        "on_court_teammate_championships",
        "bench_help_championships",
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
        "self_context_title_probability",
        "neutral_team_title_probability",
        "neutral_field_neutral_title_probability",
        "neutral_field_hard_carry_title_probability",
        "era_neutral_title_probability",
        "era_neutral_hard_carry_title_probability",
        "salary",
        "salary_cap",
        "salary_share",
        "salary_mapping_method",
        "playoff_workload_share",
        "contract_epm",
        "epm_over_contract",
        "neutral_contract_actual_title_probability",
        "neutral_contract_hurdle_title_probability",
        "title_lift_over_contract",
        "contract_hurdle_version",
        "normalized_self_title_probability_added",
        "self_replacement_title_probability_added",
        "on_court_teammate_title_probability_added",
        "bench_title_probability_added",
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
    season_teammate_columns = [
        "season",
        "player_id",
        "teammate_id",
        "teammate_name",
        "shared_exposures",
        "avg_teammate_dpm",
        "championships_added",
        "season_championship_help_rank",
    ]
    season_teammates = attributed.copy()
    season_teammates = season_teammates.sort_values(
        ["season", "player_id", "championships_added", "shared_exposures", "teammate_id"],
        ascending=[True, True, False, False, True],
        kind="mergesort",
    )
    season_teammates["season_championship_help_rank"] = (
        season_teammates.groupby(["season", "player_id"]).cumcount() + 1
    )
    write_gzip_json(output_dir / "players.json.gz", rounded_records(career, career_columns))
    write_gzip_json(
        output_dir / "season-players.json.gz",
        rounded_records(seasons, season_columns),
    )
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
            "season_teammates": rounded_records(
                season_teammates[season_teammates["player_id"].isin(player_ids)],
                season_teammate_columns,
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


def summarize_retrospective_team_season(
    exposures: pd.DataFrame,
    rating_snapshot: pd.DataFrame,
    season: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build full-team strength from one frozen rating snapshot and every lineup."""
    required = {"game_date", "team_id", "team_abbr", "exposures", *PLAYER_COLS}
    missing = required - set(exposures.columns)
    if missing:
        raise ValueError(f"Team exposures missing columns: {sorted(missing)}")
    if exposures.empty:
        return pd.DataFrame(), pd.DataFrame()

    ratings = rating_snapshot[
        pd.to_numeric(rating_snapshot["season"], errors="coerce").eq(season)
    ].copy()
    ratings["nba_id"] = pd.to_numeric(ratings["nba_id"], errors="coerce")
    ratings["date"] = pd.to_datetime(ratings["date"], errors="coerce")
    for metric in METRICS:
        ratings[metric] = pd.to_numeric(ratings[metric], errors="coerce")
    ratings = (
        ratings.dropna(subset=["nba_id", "date"])
        .sort_values(["nba_id", "date"], kind="mergesort")
        .drop_duplicates("nba_id", keep="last")
    )
    ratings["nba_id"] = ratings["nba_id"].astype(np.int64)
    rating_maps = {metric: ratings.set_index("nba_id")[metric] for metric in METRICS}
    name_map = ratings.set_index("nba_id")["player_name"]

    frame = exposures.copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    frame["exposures"] = pd.to_numeric(frame["exposures"], errors="coerce").fillna(0.0)
    player_parts: list[pd.DataFrame] = []
    for slot in PLAYER_COLS:
        player_ids = pd.to_numeric(frame[slot], errors="coerce").fillna(-1).astype(np.int64)
        part = pd.DataFrame(
            {
                "team_id": frame["team_id"].to_numpy(),
                "team_abbr": frame["team_abbr"].astype(str).to_numpy(),
                "player_id": player_ids.to_numpy(),
                "game_date": frame["game_date"].to_numpy(),
                "player_exposures": frame["exposures"].to_numpy(dtype=float),
            }
        )
        fully_rated = np.ones(len(part), dtype=bool)
        for metric in METRICS:
            values = player_ids.map(rating_maps[metric]).to_numpy(dtype=float)
            fully_rated &= np.isfinite(values)
            part[f"player_{metric}"] = values
            part[f"weighted_{metric}"] = np.where(
                np.isfinite(values), values * part["player_exposures"], 0.0
            )
        part["rated_exposures"] = np.where(
            fully_rated, part["player_exposures"], 0.0
        )
        player_parts.append(part)

    raw_roster = pd.concat(player_parts, ignore_index=True)
    roster = (
        raw_roster.groupby(
            ["team_id", "team_abbr", "player_id"], as_index=False, sort=False
        )
        .agg(
            first_game_date=("game_date", "min"),
            last_game_date=("game_date", "max"),
            player_exposures=("player_exposures", "sum"),
            rated_exposures=("rated_exposures", "sum"),
            **{
                f"weighted_{metric}": (f"weighted_{metric}", "sum")
                for metric in METRICS
            },
        )
    )
    team_totals = (
        frame.groupby(["team_id", "team_abbr"], as_index=False, sort=False)["exposures"]
        .sum()
        .rename(columns={"exposures": "total_exposures"})
    )
    roster = roster.merge(team_totals, on=["team_id", "team_abbr"], how="left")
    roster["player_name"] = roster["player_id"].map(name_map).fillna(
        roster["player_id"].map(lambda value: f"ID {value}")
    )
    for metric in METRICS:
        roster[f"player_{metric}"] = roster[f"weighted_{metric}"] / roster[
            "rated_exposures"
        ].replace(0, np.nan)
        roster[f"contribution_{metric}"] = roster[f"weighted_{metric}"] / roster[
            "total_exposures"
        ].replace(0, np.nan)
    roster["share_of_team_exposures_pct"] = (
        100.0 * roster["player_exposures"] / roster["total_exposures"].replace(0, np.nan)
    )
    roster.insert(0, "season", int(season))

    aggregate_columns = {
        f"team_{metric}": (f"contribution_{metric}", "sum") for metric in METRICS
    }
    teams = (
        roster.groupby(
            ["season", "team_id", "team_abbr", "total_exposures"],
            as_index=False,
            sort=False,
        )
        .agg(
            rated_player_exposures=("rated_exposures", "sum"),
            **aggregate_columns,
        )
    )
    teams["rating_coverage_pct"] = (
        100.0
        * teams["rated_player_exposures"]
        / (5.0 * teams["total_exposures"]).replace(0, np.nan)
    )
    return (
        teams.sort_values(["season", "team_id"], kind="mergesort").reset_index(drop=True),
        roster.sort_values(["season", "team_id", "player_id"], kind="mergesort").reset_index(
            drop=True
        ),
    )


def summarize_team_pair_exposures(
    exposures: pd.DataFrame,
    rating_snapshot: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    """Allocate teammate strength shared with each focal player on the floor."""
    required = {"team_id", "team_abbr", "exposures", *PLAYER_COLS}
    missing = required - set(exposures.columns)
    if missing:
        raise ValueError(f"Team exposures missing columns: {sorted(missing)}")
    if exposures.empty:
        return pd.DataFrame()

    ratings = rating_snapshot[
        pd.to_numeric(rating_snapshot["season"], errors="coerce").eq(season)
    ].copy()
    ratings["nba_id"] = pd.to_numeric(ratings["nba_id"], errors="coerce")
    ratings["date"] = pd.to_datetime(ratings["date"], errors="coerce")
    for metric in METRICS:
        ratings[metric] = pd.to_numeric(ratings[metric], errors="coerce")
    ratings = (
        ratings.dropna(subset=["nba_id", "date"])
        .sort_values(["nba_id", "date"], kind="mergesort")
        .drop_duplicates("nba_id", keep="last")
    )
    ratings["nba_id"] = ratings["nba_id"].astype(np.int64)
    rating_maps = {metric: ratings.set_index("nba_id")[metric] for metric in METRICS}

    frame = exposures.copy()
    frame["exposures"] = pd.to_numeric(frame["exposures"], errors="coerce").fillna(0.0)
    lineups = frame[PLAYER_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    weights = frame["exposures"].to_numpy(dtype=float)
    totals = frame.groupby("team_id")["exposures"].transform("sum").to_numpy(dtype=float)
    parts: list[pd.DataFrame] = []
    for focal_slot in range(5):
        for teammate_slot in range(5):
            if focal_slot == teammate_slot:
                continue
            focal_ids = pd.Series(lineups[:, focal_slot]).astype("Int64")
            teammate_ids = pd.Series(lineups[:, teammate_slot]).astype("Int64")
            part = pd.DataFrame(
                {
                    "team_id": frame["team_id"].to_numpy(),
                    "team_abbr": frame["team_abbr"].astype(str).to_numpy(),
                    "player_id": focal_ids.to_numpy(),
                    "teammate_id": teammate_ids.to_numpy(),
                    "shared_exposures": weights,
                }
            )
            for metric in METRICS:
                teammate_rating = teammate_ids.map(rating_maps[metric]).to_numpy(dtype=float)
                part[f"on_court_contribution_{metric}"] = np.where(
                    np.isfinite(teammate_rating) & (totals > 0),
                    teammate_rating * weights / totals,
                    0.0,
                )
            parts.append(part)
    pairs = (
        pd.concat(parts, ignore_index=True)
        .dropna(subset=["player_id", "teammate_id"])
        .groupby(
            ["team_id", "team_abbr", "player_id", "teammate_id"],
            as_index=False,
            sort=False,
        )
        .sum(numeric_only=True)
    )
    pairs["player_id"] = pairs["player_id"].astype(np.int64)
    pairs["teammate_id"] = pairs["teammate_id"].astype(np.int64)
    pairs.insert(0, "season", int(season))
    return pairs.sort_values(
        ["season", "team_id", "player_id", "teammate_id"], kind="mergesort"
    ).reset_index(drop=True)


def build_player_title_context(
    teams: pd.DataFrame,
    roster: pd.DataFrame,
    pair_exposures: pd.DataFrame,
    min_exposures: int = MODEL_MIN_EXPOSURES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create final-team focal rows and full-roster teammate contribution rows."""
    teams = teams.copy()
    for metric in METRICS:
        neutral_column = f"neutral_team_{metric}"
        if neutral_column not in teams.columns:
            teams[neutral_column] = 0.0
    final = roster.copy()
    final["last_game_date"] = pd.to_datetime(final["last_game_date"], errors="coerce")
    final = final.sort_values(
        ["season", "player_id", "last_game_date", "player_exposures", "team_id"],
        kind="mergesort",
    ).drop_duplicates(["season", "player_id"], keep="last")
    final = final[
        final["player_exposures"].ge(min_exposures) & final["player_dpm"].notna()
    ].copy()
    final = final.drop(columns=["total_exposures"], errors="ignore")
    team_columns = [
        "season",
        "team_id",
        "team_abbr",
        "team_dpm",
        "team_o_dpm",
        "team_d_dpm",
        "neutral_team_dpm",
        "neutral_team_o_dpm",
        "neutral_team_d_dpm",
        "total_exposures",
        "rating_coverage_pct",
    ]
    for optional in ("champion_team_id", "won_championship"):
        if optional in teams.columns:
            team_columns.append(optional)
    players = final.merge(
        teams[team_columns], on=["season", "team_id", "team_abbr"], how="inner"
    )
    players = players.rename(
        columns={
            "player_dpm": "own_dpm",
            "player_o_dpm": "own_o_dpm",
            "player_d_dpm": "own_d_dpm",
            "contribution_dpm": "own_team_contribution_dpm",
            "contribution_o_dpm": "own_team_contribution_o_dpm",
            "contribution_d_dpm": "own_team_contribution_d_dpm",
            "player_exposures": "total_exposures_player",
            "rated_exposures": "rated_exposures_player",
        }
    )
    players["team_total_exposures"] = players["total_exposures"]
    for metric in METRICS:
        neutral_player_rating = players[f"neutral_team_{metric}"] / 5.0
        players[f"own_neutral_team_contribution_{metric}"] = (
            neutral_player_rating
            * players["total_exposures_player"]
            / players["total_exposures"].replace(0, np.nan)
        )
        players[f"own_normalized_contribution_{metric}"] = (
            players[f"own_team_contribution_{metric}"]
            - players[f"own_neutral_team_contribution_{metric}"]
        )
        players[f"help_{metric}"] = (
            players[f"team_{metric}"] - players[f"own_team_contribution_{metric}"]
        )
    on_court = pair_exposures.groupby(
        ["season", "team_id", "player_id"], as_index=False, sort=False
    )[
        ["shared_exposures", *[f"on_court_contribution_{metric}" for metric in METRICS]]
    ].sum()
    players = players.merge(
        on_court, on=["season", "team_id", "player_id"], how="left", validate="one_to_one"
    )
    for metric in METRICS:
        on_column = f"on_court_contribution_{metric}"
        players[f"on_court_help_{metric}"] = players[on_column].fillna(0.0)
        players[f"on_court_neutral_contribution_{metric}"] = (
            (players[f"neutral_team_{metric}"] / 5.0)
            * players["shared_exposures"].fillna(0.0)
            / players["total_exposures"].replace(0, np.nan)
        )
        players[f"on_court_normalized_help_{metric}"] = (
            players[f"on_court_help_{metric}"]
            - players[f"on_court_neutral_contribution_{metric}"]
        )
        players[f"bench_help_{metric}"] = (
            players[f"team_{metric}"]
            - players[f"own_team_contribution_{metric}"]
            - players[f"on_court_help_{metric}"]
        )
        players[f"bench_normalized_help_{metric}"] = (
            players[f"team_{metric}"]
            - players[f"neutral_team_{metric}"]
            - players[f"own_normalized_contribution_{metric}"]
            - players[f"on_court_normalized_help_{metric}"]
        )
    players["coverage_pct"] = players["rating_coverage_pct"]
    players["total_exposures"] = players["total_exposures_player"]
    players["rated_exposures"] = players["rated_exposures_player"]

    focal = players[
        [
            "season",
            "team_id",
            "player_id",
            "player_name",
            "team_total_exposures",
            *[f"neutral_team_{metric}" for metric in METRICS],
        ]
    ].copy()
    teammate_source = roster.rename(
        columns={
            "player_id": "teammate_id",
            "player_name": "teammate_name",
            "player_exposures": "teammate_team_exposures",
            "rated_exposures": "teammate_rated_exposures",
            "player_dpm": "avg_teammate_dpm",
            "player_o_dpm": "avg_teammate_o_dpm",
            "player_d_dpm": "avg_teammate_d_dpm",
        }
    )
    teammates = focal.merge(teammate_source, on=["season", "team_id"], how="left")
    teammates = teammates[teammates["player_id"].ne(teammates["teammate_id"])].copy()
    teammates = teammates.merge(
        pair_exposures.drop(columns=["team_abbr"], errors="ignore"),
        on=["season", "team_id", "player_id", "teammate_id"],
        how="left",
        validate="one_to_one",
    )
    teammates["shared_exposures"] = teammates["shared_exposures"].fillna(0.0)
    teammates["rated_exposures"] = teammates["teammate_rated_exposures"]
    for metric in METRICS:
        teammates[f"on_court_contribution_{metric}"] = teammates[
            f"on_court_contribution_{metric}"
        ].fillna(0.0)
        neutral_player_rating = teammates[f"neutral_team_{metric}"] / 5.0
        teammates[f"neutral_contribution_{metric}"] = (
            neutral_player_rating
            * teammates["teammate_team_exposures"]
            / teammates["team_total_exposures"].replace(0, np.nan)
        )
        teammates[f"normalized_contribution_{metric}"] = (
            teammates[f"contribution_{metric}"]
            - teammates[f"neutral_contribution_{metric}"]
        )
        teammates[f"on_court_neutral_contribution_{metric}"] = (
            neutral_player_rating
            * teammates["shared_exposures"]
            / teammates["team_total_exposures"].replace(0, np.nan)
        )
        teammates[f"on_court_normalized_contribution_{metric}"] = (
            teammates[f"on_court_contribution_{metric}"]
            - teammates[f"on_court_neutral_contribution_{metric}"]
        )
        teammates[f"bench_contribution_{metric}"] = (
            teammates[f"contribution_{metric}"]
            - teammates[f"on_court_contribution_{metric}"]
        )
        teammates[f"bench_normalized_contribution_{metric}"] = (
            teammates[f"normalized_contribution_{metric}"]
            - teammates[f"on_court_normalized_contribution_{metric}"]
        )
    teammates = teammates.sort_values(
        ["season", "player_id", "team_id", "teammate_id"], kind="mergesort"
    ).reset_index(drop=True)
    players = players.sort_values(["season", "player_id"], kind="mergesort").reset_index(
        drop=True
    )
    return players, teammates


def validate_team_roster_reconciliation(
    teams: pd.DataFrame,
    roster: pd.DataFrame,
    tolerance: float = 1e-10,
) -> dict[str, float | int]:
    sums = roster.groupby(["season", "team_id"], as_index=False)[
        [f"contribution_{metric}" for metric in METRICS]
    ].sum(min_count=1)
    merged = teams.merge(sums, on=["season", "team_id"], how="left")
    report: dict[str, float | int] = {"team_seasons": int(len(merged))}
    for metric in METRICS:
        gap = (merged[f"team_{metric}"] - merged[f"contribution_{metric}"]).abs()
        max_gap = float(gap.max(skipna=True)) if gap.notna().any() else 0.0
        report[f"max_{metric}_gap"] = max_gap
        if max_gap > tolerance:
            raise AssertionError(f"{metric} team-roster gap {max_gap} exceeds {tolerance}")
    return report


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
    epm_history: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[dict[str, object]],
]:
    """Build team strength from pre-postseason EPM and actual playoff rotations."""
    team_frames: list[pd.DataFrame] = []
    roster_frames: list[pd.DataFrame] = []
    pair_frames: list[pd.DataFrame] = []
    champions: list[dict[str, object]] = []
    reports: list[dict[str, object]] = []
    processed_columns = ["game_id", "game_date", *LINEUP_COLS, *OPPONENT_COLS]

    for season in range(start_year, end_year + 1):
        playoff_processed_path = season_file(processed_dir, "RAPM", season, playoffs=True)
        playoff_raw_path = season_file(raw_dir, "NBA", season, playoffs=True)
        required_paths = [playoff_processed_path, playoff_raw_path]
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing {season} inputs: {missing}")

        playoff_possessions = pd.read_parquet(
            playoff_processed_path, columns=processed_columns
        )
        if (
            pd.to_datetime(
                playoff_possessions["game_date"], errors="coerce"
            ).notna().mean()
            < 0.999
        ):
            raise ValueError(f"{playoff_processed_path} lacks complete game dates")
        playoff_dates = playoff_possessions[["game_id", "game_date"]].drop_duplicates(
            "game_id"
        )
        playoff_raw = pd.read_parquet(playoff_raw_path, columns=RAW_CONTEXT_COLS)
        playoff_games, playoff_assignments = build_game_context(
            playoff_raw, playoff_dates
        )
        exposures = collapse_processed_team_exposures(
            playoff_possessions, playoff_assignments
        )
        postseason_start = pd.to_datetime(
            playoff_games["game_date"], errors="coerce"
        ).min()
        if pd.isna(postseason_start):
            raise ValueError(f"No postseason start date for {season}")
        epm_snapshot = select_epm_snapshot(epm_history, postseason_start)
        teams, roster = summarize_retrospective_team_season(
            exposures, epm_snapshot, season=season
        )
        pair_exposures = summarize_team_pair_exposures(
            exposures, epm_snapshot, season=season
        )
        teams = attach_neutral_playoff_baseline(teams)
        team_frames.append(teams)
        roster_frames.append(roster)
        pair_frames.append(pair_exposures)

        champion = extract_champion_from_games(playoff_games, season=season)
        champions.append(champion)
        reports.append(
            {
                "season": season,
                "postseason_start_date": postseason_start.date().isoformat(),
                "epm_snapshot_requested_date": epm_snapshot[
                    "snapshot_requested_date"
                ].iloc[0].date().isoformat(),
                "epm_provider_effective_date": epm_snapshot[
                    "provider_effective_date"
                ].max().date().isoformat(),
                "playoff_games": int(len(playoff_games)),
                "playoff_possessions": int(len(playoff_possessions)),
                "playoff_team_exposures": int(exposures["exposures"].sum()),
                "team_rows": int(len(teams)),
                "roster_rows": int(len(roster)),
                "pair_rows": int(len(pair_exposures)),
                "rating_coverage_pct": float(
                    100.0
                    * teams["rated_player_exposures"].sum()
                    / (5.0 * teams["total_exposures"].sum())
                ),
                "champion_team_id": int(champion["champion_team_id"]),
                "champion_team_abbr": str(champion["champion_team_abbr"]),
            }
        )
        print(
            f"{season}: {len(playoff_possessions):,} PS possessions, "
            f"{len(teams):,} teams, EPM {epm_snapshot['snapshot_requested_date'].iloc[0].date()}, "
            f"champion {champion['champion_team_abbr']}"
        )

    team_seasons = pd.concat(team_frames, ignore_index=True)
    roster = pd.concat(roster_frames, ignore_index=True)
    pair_exposures = pd.concat(pair_frames, ignore_index=True)
    champion_frame = pd.DataFrame(champions)
    team_seasons = team_seasons.merge(
        champion_frame[["season", "champion_team_id", "champion_team_abbr"]],
        on="season",
        how="left",
    )
    team_seasons["won_championship"] = (
        team_seasons["team_id"] == team_seasons["champion_team_id"]
    ).astype(int)
    champion_counts = team_seasons.groupby("season")["won_championship"].sum()
    invalid = champion_counts[champion_counts.ne(1)]
    if not invalid.empty:
        raise ValueError(f"Champion team missing or duplicated: {invalid.to_dict()}")
    return team_seasons, roster, pair_exposures, champion_frame, reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2002)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--epm-cache-dir", type=Path, default=DEFAULT_EPM_CACHE_DIR)
    parser.add_argument(
        "--salary-path",
        type=Path,
        default=DEFAULT_SALARY_PATH,
        help=(
            "Private NBA-ID salary history. Raw rows remain outside the public app; "
            "only derived salary share and contract metrics are emitted."
        ),
    )
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
    epm_history, epm_report = load_epm_snapshot_history(args.epm_cache_dir)
    salary_history, salary_source_report = load_salary_history(args.salary_path)
    team_path = args.output_dir / "team_seasons.csv"
    roster_path = args.output_dir / "team_player_contributions.csv"
    team_pair_path = args.output_dir / "team_player_pair_exposures.csv"
    player_path = args.output_dir / "player_team_seasons.csv"
    pair_path = args.output_dir / "player_teammate_team_seasons.csv"
    champions_path = args.output_dir / "champions.csv"
    if args.reuse_data:
        missing = [
            str(path)
            for path in (team_path, roster_path, team_pair_path, champions_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(f"Cannot reuse missing championship-help inputs: {missing}")
        teams = pd.read_csv(team_path)
        roster = pd.read_csv(roster_path)
        team_pair_exposures = pd.read_csv(team_pair_path)
        champions = pd.read_csv(champions_path)
        prior_metadata_path = args.output_dir / "metadata.json"
        if prior_metadata_path.exists():
            prior_metadata = json.loads(prior_metadata_path.read_text())
            reports = list(prior_metadata.get("seasons", []))
        else:
            reports = []
    else:
        teams, roster, team_pair_exposures, champions, reports = build_team_season_data(
            args.start_year,
            args.end_year,
            args.processed_dir,
            args.raw_dir,
            epm_history,
        )
    player, pair = build_player_title_context(teams, roster, team_pair_exposures)
    player, salary_coverage_report = attach_salary_share(player, salary_history)
    team_reconciliation = validate_team_roster_reconciliation(teams, roster)
    player_reconciliation = validate_pair_reconciliation(player, pair)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    teams.to_csv(team_path, index=False)
    roster.to_csv(roster_path, index=False)
    team_pair_exposures.to_csv(team_pair_path, index=False)
    player.to_csv(player_path, index=False)
    pair.to_csv(pair_path, index=False)
    champions.to_csv(champions_path, index=False)
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "start_season": int(args.start_year),
        "end_season": int(args.end_year),
        "team_seasons": int(len(teams)),
        "roster_seasons": int(len(roster)),
        "team_pair_exposure_rows": int(len(team_pair_exposures)),
        "player_seasons": int(len(player)),
        "pair_seasons": int(len(pair)),
        "rating_source": "Dunks & Threes EPM",
        "epm_snapshot_archive": epm_report,
        "team_rating_coverage_pct": float(
            100.0
            * teams["rated_player_exposures"].sum()
            / (5.0 * teams["total_exposures"].sum())
        ),
        "team_reconciliation": team_reconciliation,
        "player_reconciliation": player_reconciliation,
        "seasons": reports,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote data-only championship-help inputs to {args.output_dir}")
    print(f"Team reconciliation: {team_reconciliation}")
    print(f"Player reconciliation: {player_reconciliation}")
    if args.data_only:
        return

    sample = teams.sort_values(["season", "team_id"], kind="mergesort").reset_index(drop=True)
    diagnostics, selected, oof_predictions, fold_models = evaluate_team_model_grid(sample)
    seasons, attributed, shapley_report = attribute_team_probabilities(
        player,
        pair,
        sample,
        fold_models,
        permutations=args.shapley_permutations,
    )
    neutral_title_model = fit_team_title_model(
        sample, str(selected["spec"]), float(selected["c_value"])
    )
    seasons = add_title_lift_over_contract(seasons, neutral_title_model)
    career, teammate, career_report = build_career_championship_help(seasons, attributed)
    selected["model_team_seasons"] = int(len(sample))
    selected["championship_team_seasons"] = int(sample["won_championship"].sum())
    selected["seasons"] = int(sample["season"].nunique())
    selected["minimum_exposures"] = MODEL_MIN_EXPOSURES
    selected["neutral_teammate_baseline"] = (
        "season-average pre-postseason EPM of the actual playoff rotations"
    )
    selected["hard_carry_comparison_field"] = (
        "one fixed neutral 16-team environment for every player-season: a 1/16 "
        "neutral baseline and 15 equal-strength opponents"
    )
    selected["season_common_field"] = (
        "one shared 15-opponent field within each season, using 15 times the "
        "average conditional-softmax mass of all 16 actual playoff teams"
    )
    selected["probability_unit"] = "team-season conditional softmax"
    selected["rating_timing"] = (
        "latest monthly game-agnostic EPM snapshot on/before postseason start"
    )
    selected["title_lift_over_contract"] = {
        "label": "Title Lift over Contract",
        "cost_unit": "annual salary share, not cap hit",
        "hurdle_version": CONTRACT_HURDLE_VERSION,
        "replacement_epm": CONTRACT_REPLACEMENT_EPM,
        "replacement_salary_share": CONTRACT_REPLACEMENT_SALARY_SHARE,
        "cap_share_per_war": CONTRACT_CAP_SHARE_PER_WAR,
        "normalized_season_possessions": CONTRACT_NORMALIZED_SEASON_POSSESSIONS,
        "wins_per_point": CONTRACT_WINS_PER_POINT,
        "war_per_epm": CONTRACT_WAR_PER_EPM,
        "field": "fixed neutral 16-team field with 0 EPM equal to 1-in-16 odds",
        "workload": "player's actual share of his playoff team's possessions",
        "status": "declared prototype market-price assumptions; not a fitted UFA curve",
    }

    diagnostics.to_csv(args.output_dir / "model_cv.csv", index=False)
    oof_predictions.to_csv(args.output_dir / "model_oof_predictions.csv", index=False)
    team_estimates = teams.merge(
        oof_predictions[["season", "team_id", "predicted_probability"]],
        on=["season", "team_id"],
        how="left",
        validate="one_to_one",
    ).rename(columns={"predicted_probability": "title_probability"})
    team_estimates.to_csv(args.output_dir / "team_season_estimates.csv", index=False)
    seasons.to_csv(args.output_dir / "player_season_estimates.csv", index=False)
    attributed.to_csv(args.output_dir / "player_teammate_season_estimates.csv", index=False)
    career.to_csv(args.output_dir / "career_player_estimates.csv", index=False)
    teammate.to_csv(args.output_dir / "career_teammate_estimates.csv", index=False)
    metadata.update(
        {
            "model": selected,
            "shapley": shapley_report,
            "career_reconciliation": career_report,
            "salary_source": salary_source_report,
            "salary_coverage": salary_coverage_report,
            "interpretation": (
                "Every player on a team receives the same normalized team championship odds. "
                "Those playoff-field odds are split from a common season-average playoff rotation "
                "into the focal player's normalized self lift, shared-floor teammate lift, and "
                "off-floor lift with exact closure. Off-floor means actual playoff minutes while "
                "the focal player sat, not a nominal roster label. This is not causal ring credit."
            ),
            "retrospective_guard": (
                "The latest monthly EPM snapshot available before the postseason is frozen for "
                "each season, then weighted by the lineups that actually played in the playoffs. "
                "Team rotation shares are normalized, so a longer playoff run does not create "
                "more team strength. Non-playoff teams have zero title odds; playoff-team odds "
                "sum to one. These are retrospective allocations, not pregame forecasts."
            ),
        }
    )
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    write_app_championship_data(
        args.app_data_dir, career, seasons, teammate, attributed, metadata
    )
    print(
        f"Selected {selected['spec']} C={selected['c_value']} by LOYO champion log loss "
        f"({selected['champion_log_loss']:.6f}); wrote {len(career):,} career estimates and "
        f"{len(teammate):,} career teammate attributions"
    )
    print(f"Shapley reconciliation: {shapley_report}")


if __name__ == "__main__":
    main()
