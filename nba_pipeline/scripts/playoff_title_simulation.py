"""Salary-peer playoff title value through an explicit best-of-seven bracket."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


PLAYOFF_GAME_MODEL_C_VALUES = (0.05, 0.2, 1.0, 5.0, 20.0, 100.0)


@dataclass(frozen=True)
class PlayoffGameModel:
    """Neutral playoff game model in log-odds units."""

    home_advantage: float
    epm_coefficient: float
    c_value: float = 1.0


def _series_probability_with_home_court(
    rating_difference: np.ndarray,
    model: PlayoffGameModel,
    focal_has_home_court: bool,
) -> np.ndarray:
    schedule = np.array([1, 1, 0, 0, 1, 0, 1], dtype=float)
    if not focal_has_home_court:
        schedule = 1.0 - schedule
    venue_sign = np.where(schedule > 0, 1.0, -1.0)
    game_probability = expit(
        model.epm_coefficient * rating_difference[..., None]
        + model.home_advantage * venue_sign
    )
    shape = rating_difference.shape
    states: dict[tuple[int, int], np.ndarray] = {(0, 0): np.ones(shape, dtype=float)}
    for game_index in range(7):
        updated: dict[tuple[int, int], np.ndarray] = {}
        for (wins, losses), mass in states.items():
            if wins == 4 or losses == 4:
                updated[(wins, losses)] = updated.get((wins, losses), 0.0) + mass
                continue
            win_key = (wins + 1, losses)
            loss_key = (wins, losses + 1)
            win_probability = game_probability[..., game_index]
            updated[win_key] = updated.get(win_key, 0.0) + mass * win_probability
            updated[loss_key] = updated.get(loss_key, 0.0) + mass * (1.0 - win_probability)
        states = updated
    return sum(mass for (wins, _), mass in states.items() if wins == 4)


def best_of_seven_win_probability(
    rating_difference: float | np.ndarray,
    model: PlayoffGameModel,
) -> float | np.ndarray:
    """Average a 2-2-1-1-1 series over either team receiving home court."""
    difference = np.asarray(rating_difference, dtype=float)
    with_home_court = _series_probability_with_home_court(difference, model, True)
    without_home_court = _series_probability_with_home_court(difference, model, False)
    result = 0.5 * (with_home_court + without_home_court)
    if result.ndim == 0:
        return float(result)
    return result


def fit_playoff_game_model(
    rows: pd.DataFrame,
    c_value: float,
) -> PlayoffGameModel:
    """Fit home advantage and a nonnegative EPM slope to playoff game outcomes."""
    required = {"rating_difference", "home_win"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Playoff game-model rows are missing columns: {sorted(missing)}")
    if c_value <= 0:
        raise ValueError("c_value must be positive")
    difference = rows["rating_difference"].to_numpy(dtype=float)
    outcome = rows["home_win"].to_numpy(dtype=float)
    if not np.isfinite(difference).all() or not np.isin(outcome, [0.0, 1.0]).all():
        raise ValueError("Playoff game-model inputs must be finite binary outcomes")

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        home_advantage, epm_coefficient = parameters
        linear = home_advantage + epm_coefficient * difference
        probability = expit(linear)
        loss = float(
            np.logaddexp(0.0, linear).sum()
            - np.dot(outcome, linear)
            + 0.5 * np.dot(parameters, parameters) / c_value
        )
        residual = probability - outcome
        gradient = np.array(
            [
                residual.sum() + home_advantage / c_value,
                np.dot(residual, difference) + epm_coefficient / c_value,
            ]
        )
        return loss, gradient

    result = minimize(
        objective,
        x0=np.array([0.1, 0.1]),
        jac=True,
        method="L-BFGS-B",
        bounds=[(None, None), (0.0, None)],
    )
    if not result.success:
        raise RuntimeError(f"Playoff game model failed to converge: {result.message}")
    return PlayoffGameModel(
        home_advantage=float(result.x[0]),
        epm_coefficient=float(result.x[1]),
        c_value=float(c_value),
    )


def prepare_playoff_game_model_rows(
    games: pd.DataFrame,
    team_seasons: pd.DataFrame,
) -> pd.DataFrame:
    """Attach pre-postseason team EPM and the binary home result to playoff games."""
    required_games = {
        "season",
        "game_id",
        "home_team_id",
        "away_team_id",
        "home_score",
        "away_score",
    }
    required_teams = {"season", "team_id", "team_dpm"}
    if missing := required_games.difference(games.columns):
        raise ValueError(f"Playoff game rows are missing columns: {sorted(missing)}")
    if missing := required_teams.difference(team_seasons.columns):
        raise ValueError(f"Playoff team rows are missing columns: {sorted(missing)}")
    home = team_seasons[list(required_teams)].rename(
        columns={"team_id": "home_team_id", "team_dpm": "home_dpm"}
    )
    away = team_seasons[list(required_teams)].rename(
        columns={"team_id": "away_team_id", "team_dpm": "away_dpm"}
    )
    rows = games.merge(
        home,
        on=["season", "home_team_id"],
        how="inner",
        validate="many_to_one",
    ).merge(
        away,
        on=["season", "away_team_id"],
        how="inner",
        validate="many_to_one",
    )
    if len(rows) != len(games):
        raise ValueError("Some playoff games do not map to both team-season ratings")
    rows["rating_difference"] = rows["home_dpm"] - rows["away_dpm"]
    rows["home_win"] = (rows["home_score"] > rows["away_score"]).astype(int)
    return rows


def select_playoff_game_model(
    rows: pd.DataFrame,
    c_values: tuple[float, ...] = PLAYOFF_GAME_MODEL_C_VALUES,
) -> tuple[PlayoffGameModel, pd.DataFrame, dict[str, float | int | str]]:
    """Select regularization by leave-one-season-out playoff game log loss."""
    records: list[dict[str, float]] = []
    seasons = sorted(pd.to_numeric(rows["season"], errors="raise").astype(int).unique())
    for c_value in c_values:
        season_losses: list[float] = []
        game_losses: list[float] = []
        for season in seasons:
            train = rows.loc[rows["season"].ne(season)]
            test = rows.loc[rows["season"].eq(season)]
            model = fit_playoff_game_model(train, float(c_value))
            probability = expit(
                model.home_advantage
                + model.epm_coefficient * test["rating_difference"].to_numpy(dtype=float)
            )
            outcome = test["home_win"].to_numpy(dtype=float)
            loss = -(
                outcome * np.log(probability.clip(1e-15, 1.0))
                + (1.0 - outcome) * np.log((1.0 - probability).clip(1e-15, 1.0))
            )
            season_losses.append(float(loss.mean()))
            game_losses.extend(loss.tolist())
        records.append(
            {
                "c_value": float(c_value),
                "game_log_loss": float(np.mean(game_losses)),
                "season_mean_log_loss": float(np.mean(season_losses)),
                "season_log_loss_se": float(
                    np.std(season_losses, ddof=1) / np.sqrt(len(season_losses))
                ),
            }
        )
    diagnostics = pd.DataFrame(records).sort_values(
        ["season_mean_log_loss", "c_value"], kind="mergesort"
    ).reset_index(drop=True)
    best = diagnostics.iloc[0]
    threshold = float(best["season_mean_log_loss"] + best["season_log_loss_se"])
    selected_row = diagnostics.loc[
        diagnostics["season_mean_log_loss"].le(threshold)
    ].sort_values("c_value", kind="mergesort").iloc[0]
    model = fit_playoff_game_model(rows, float(selected_row["c_value"]))
    report: dict[str, float | int | str] = {
        "selection_metric": "one-standard-error leave-one-season-out game log loss",
        "c_value": float(selected_row["c_value"]),
        "game_log_loss": float(selected_row["game_log_loss"]),
        "season_mean_log_loss": float(selected_row["season_mean_log_loss"]),
        "one_se_threshold": threshold,
        "home_advantage_log_odds": model.home_advantage,
        "epm_log_odds_per_point": model.epm_coefficient,
        "games": int(len(rows)),
        "seasons": int(len(seasons)),
    }
    return model, diagnostics, report


def select_salary_peer_slots(
    focal_salary_share: float,
    focal_season: int,
    roster: pd.DataFrame,
) -> pd.DataFrame:
    """Select one closest-cap-share playoff rotation slot per historical team."""
    required = {
        "season",
        "team_id",
        "player_id",
        "salary_share",
        "own_dpm",
        "own_team_contribution_dpm",
        "playoff_workload_share",
    }
    missing = required.difference(roster.columns)
    if missing:
        raise ValueError(f"Salary-peer roster is missing columns: {sorted(missing)}")
    if not np.isfinite(focal_salary_share):
        return pd.DataFrame(
            columns=[
                "season",
                "team_id",
                "peer_player_id",
                "peer_salary_share",
                "peer_salary_gap",
                "peer_dpm",
                "peer_contribution_dpm",
                "peer_workload_share",
            ]
        )

    candidates = roster.loc[
        roster["season"].ne(int(focal_season))
        & roster["salary_share"].notna()
        & roster["playoff_workload_share"].gt(0)
    ].copy()
    candidates["peer_salary_gap"] = (
        candidates["salary_share"].astype(float) - float(focal_salary_share)
    ).abs()
    candidates["salary_gap_sort"] = candidates["peer_salary_gap"].round(12)
    candidates = candidates.sort_values(
        [
            "season",
            "team_id",
            "salary_gap_sort",
            "playoff_workload_share",
            "player_id",
        ],
        ascending=[True, True, True, False, True],
        kind="mergesort",
    ).drop_duplicates(["season", "team_id"], keep="first")
    return candidates[
        [
            "season",
            "team_id",
            "player_id",
            "salary_share",
            "peer_salary_gap",
            "own_dpm",
            "own_team_contribution_dpm",
            "playoff_workload_share",
        ]
    ].rename(
        columns={
            "player_id": "peer_player_id",
            "salary_share": "peer_salary_share",
            "own_dpm": "peer_dpm",
            "own_team_contribution_dpm": "peer_contribution_dpm",
            "playoff_workload_share": "peer_workload_share",
        }
    ).reset_index(drop=True)


def build_playoff_series(games: pd.DataFrame) -> pd.DataFrame:
    """Collapse completed playoff games into one audited row per series."""
    required = {
        "season",
        "game_id",
        "home_team_id",
        "away_team_id",
        "home_score",
        "away_score",
    }
    missing = required.difference(games.columns)
    if missing:
        raise ValueError(f"Playoff games are missing columns: {sorted(missing)}")
    frame = games.copy()
    numeric_id = pd.to_numeric(frame["game_id"], errors="raise").astype(np.int64)
    frame["round"] = ((numeric_id // 100) % 10).astype(int)
    frame["series_id"] = (numeric_id // 10).astype(str)
    if not frame["round"].between(1, 4).all():
        raise ValueError("Playoff series contain an invalid round identifier")
    frame["winner_team_id"] = np.where(
        frame["home_score"].astype(float) > frame["away_score"].astype(float),
        frame["home_team_id"],
        frame["away_team_id"],
    ).astype(np.int64)

    records: list[dict[str, int | str]] = []
    for (season, round_number, series_id), group in frame.groupby(
        ["season", "round", "series_id"], sort=True
    ):
        teams = np.unique(
            np.concatenate(
                [
                    group["home_team_id"].to_numpy(dtype=np.int64),
                    group["away_team_id"].to_numpy(dtype=np.int64),
                ]
            )
        )
        if len(teams) != 2:
            raise ValueError(f"Series {series_id} has {len(teams)} teams")
        wins = group["winner_team_id"].value_counts()
        if len(wins) != 2 and int(wins.iloc[0]) < 3:
            raise ValueError(f"Series {series_id} has no completed winner")
        if len(wins) > 1 and int(wins.iloc[0]) == int(wins.iloc[1]):
            raise ValueError(f"Series {series_id} is tied")
        team_a, team_b = map(int, sorted(teams.tolist()))
        winner = int(wins.index[0])
        records.append(
            {
                "season": int(season),
                "round": int(round_number),
                "series_id": str(series_id),
                "team_a_id": team_a,
                "team_b_id": team_b,
                "team_a_wins": int(wins.get(team_a, 0)),
                "team_b_wins": int(wins.get(team_b, 0)),
                "winner_team_id": winner,
                "games": int(len(group)),
            }
        )
    return pd.DataFrame(records).sort_values(
        ["season", "round", "series_id"], kind="mergesort"
    ).reset_index(drop=True)


def build_bracket_nodes(series: pd.DataFrame) -> pd.DataFrame:
    """Recover the fixed NBA bracket tree from actual series advancement."""
    required = {
        "season",
        "round",
        "series_id",
        "team_a_id",
        "team_b_id",
        "winner_team_id",
    }
    missing = required.difference(series.columns)
    if missing:
        raise ValueError(f"Playoff series are missing columns: {sorted(missing)}")
    records: list[dict[str, int | str | None]] = []
    for season, season_series in series.groupby("season", sort=True):
        counts = season_series.groupby("round").size().to_dict()
        if counts != {1: 8, 2: 4, 3: 2, 4: 1}:
            raise ValueError(f"Season {season} does not contain an 8-4-2-1 bracket: {counts}")
        prior_winner_to_series: dict[int, str] = {}
        for round_number in range(1, 5):
            round_rows = season_series.loc[season_series["round"].eq(round_number)].sort_values(
                "series_id", kind="mergesort"
            )
            next_winner_to_series: dict[int, str] = {}
            for row in round_rows.itertuples(index=False):
                team_a = int(row.team_a_id)
                team_b = int(row.team_b_id)
                if round_number == 1:
                    child_a = child_b = None
                else:
                    if team_a not in prior_winner_to_series or team_b not in prior_winner_to_series:
                        raise ValueError(
                            f"Series {row.series_id} participants do not map to prior-round winners"
                        )
                    child_a = prior_winner_to_series[team_a]
                    child_b = prior_winner_to_series[team_b]
                records.append(
                    {
                        "season": int(season),
                        "round": int(round_number),
                        "series_id": str(row.series_id),
                        "team_a_id": team_a,
                        "team_b_id": team_b,
                        "child_a_series_id": child_a,
                        "child_b_series_id": child_b,
                    }
                )
                next_winner_to_series[int(row.winner_team_id)] = str(row.series_id)
            prior_winner_to_series = next_winner_to_series
    return pd.DataFrame(records).sort_values(
        ["season", "round", "series_id"], kind="mergesort"
    ).reset_index(drop=True)


def bracket_title_probabilities(
    nodes: pd.DataFrame,
    team_ratings: pd.Series,
    model: PlayoffGameModel,
) -> pd.Series:
    """Propagate best-of-seven win probabilities through one complete bracket."""
    if nodes["season"].nunique() != 1:
        raise ValueError("Bracket title probabilities require exactly one season")
    winner_distributions: dict[str, dict[int, float]] = {}
    for round_number in range(1, 5):
        round_nodes = nodes.loc[nodes["round"].eq(round_number)].sort_values(
            "series_id", kind="mergesort"
        )
        for row in round_nodes.itertuples(index=False):
            if round_number == 1:
                left = {int(row.team_a_id): 1.0}
                right = {int(row.team_b_id): 1.0}
            else:
                left = winner_distributions[str(row.child_a_series_id)]
                right = winner_distributions[str(row.child_b_series_id)]
            output: dict[int, float] = {}
            for team_a, mass_a in left.items():
                for team_b, mass_b in right.items():
                    try:
                        difference = float(team_ratings.loc[team_a] - team_ratings.loc[team_b])
                    except KeyError as exc:
                        raise ValueError(f"Missing playoff-team rating: {exc}") from exc
                    probability_a = float(best_of_seven_win_probability(difference, model))
                    matchup_mass = float(mass_a * mass_b)
                    output[team_a] = output.get(team_a, 0.0) + matchup_mass * probability_a
                    output[team_b] = output.get(team_b, 0.0) + matchup_mass * (
                        1.0 - probability_a
                    )
            winner_distributions[str(row.series_id)] = output
    finals = nodes.loc[nodes["round"].eq(4)]
    if len(finals) != 1:
        raise ValueError("Bracket requires exactly one Finals node")
    result = pd.Series(winner_distributions[str(finals.iloc[0]["series_id"])], dtype=float)
    return result.sort_index()


def target_title_probability_curve(
    nodes: pd.DataFrame,
    team_ratings: pd.Series,
    target_team_id: int,
    target_ratings: float | np.ndarray,
    model: PlayoffGameModel,
) -> float | np.ndarray:
    """Evaluate one team's exact four-series title path over candidate ratings."""
    if nodes["season"].nunique() != 1:
        raise ValueError("Target title curves require exactly one season")

    winner_distributions: dict[str, dict[int, float]] = {}
    for round_number in range(1, 5):
        for row in nodes.loc[nodes["round"].eq(round_number)].itertuples(index=False):
            if round_number == 1:
                left = {int(row.team_a_id): 1.0}
                right = {int(row.team_b_id): 1.0}
            else:
                left = winner_distributions[str(row.child_a_series_id)]
                right = winner_distributions[str(row.child_b_series_id)]
            output: dict[int, float] = {}
            for team_a, mass_a in left.items():
                for team_b, mass_b in right.items():
                    difference = float(team_ratings.loc[team_a] - team_ratings.loc[team_b])
                    probability_a = float(best_of_seven_win_probability(difference, model))
                    matchup_mass = float(mass_a * mass_b)
                    output[team_a] = output.get(team_a, 0.0) + matchup_mass * probability_a
                    output[team_b] = output.get(team_b, 0.0) + matchup_mass * (
                        1.0 - probability_a
                    )
            winner_distributions[str(row.series_id)] = output

    first_round = nodes.loc[
        nodes["round"].eq(1)
        & (
            nodes["team_a_id"].eq(int(target_team_id))
            | nodes["team_b_id"].eq(int(target_team_id))
        )
    ]
    if len(first_round) != 1:
        raise ValueError(f"Target team {target_team_id} has no unique first-round slot")
    leaf = first_round.iloc[0]
    opponent = (
        int(leaf["team_b_id"])
        if int(leaf["team_a_id"]) == int(target_team_id)
        else int(leaf["team_a_id"])
    )
    opponent_distributions: list[dict[int, float]] = [{opponent: 1.0}]
    parent_by_child: dict[str, tuple[str, str]] = {}
    for row in nodes.loc[nodes["round"].gt(1)].itertuples(index=False):
        parent_by_child[str(row.child_a_series_id)] = (
            str(row.series_id),
            str(row.child_b_series_id),
        )
        parent_by_child[str(row.child_b_series_id)] = (
            str(row.series_id),
            str(row.child_a_series_id),
        )
    current = str(leaf["series_id"])
    while current in parent_by_child:
        parent, sibling = parent_by_child[current]
        opponent_distributions.append(winner_distributions[sibling])
        current = parent
    if len(opponent_distributions) != 4:
        raise ValueError("Target path does not contain four playoff rounds")

    rating = np.asarray(target_ratings, dtype=float)
    probability = np.ones(rating.shape, dtype=float)
    for distribution in opponent_distributions:
        opponent_ids = np.fromiter(distribution.keys(), dtype=np.int64)
        opponent_mass = np.fromiter(distribution.values(), dtype=float)
        opponent_ratings = team_ratings.loc[opponent_ids].to_numpy(dtype=float)
        series_probability = best_of_seven_win_probability(
            rating[..., None] - opponent_ratings,
            model,
        )
        probability *= np.sum(series_probability * opponent_mass, axis=-1)
    if probability.ndim == 0:
        return float(probability)
    return probability


SALARY_PEER_HURDLE_VERSION = "salary-peer-series-v1"


def add_salary_peer_title_lift(
    focal_seasons: pd.DataFrame,
    playoff_roster: pd.DataFrame,
    playoff_teams: pd.DataFrame,
    bracket_nodes: pd.DataFrame,
    model: PlayoffGameModel,
    rating_grid: np.ndarray | None = None,
) -> pd.DataFrame:
    """Average same-bracket title lift over closest-cap-share roster swaps."""
    focal_required = {"season", "player_id", "own_dpm", "salary_share"}
    roster_required = {
        "season",
        "team_id",
        "player_id",
        "salary_share",
        "own_dpm",
        "own_team_contribution_dpm",
        "playoff_workload_share",
    }
    team_required = {"season", "team_id", "team_dpm"}
    for label, frame, required in (
        ("focal seasons", focal_seasons, focal_required),
        ("playoff roster", playoff_roster, roster_required),
        ("playoff teams", playoff_teams, team_required),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Salary-peer {label} are missing columns: {sorted(missing)}")
    grid = (
        np.linspace(-30.0, 40.0, 3501)
        if rating_grid is None
        else np.asarray(rating_grid, dtype=float)
    )
    if grid.ndim != 1 or len(grid) < 2 or not np.all(np.diff(grid) > 0):
        raise ValueError("Salary-peer rating grid must be a strictly increasing vector")

    curves: dict[tuple[int, int], np.ndarray] = {}
    base_ratings: dict[tuple[int, int], float] = {}
    base_probabilities: dict[tuple[int, int], float] = {}
    for season, season_teams in playoff_teams.groupby("season", sort=True):
        season_nodes = bracket_nodes.loc[bracket_nodes["season"].eq(int(season))]
        if season_nodes.empty:
            continue
        ratings = season_teams.set_index("team_id")["team_dpm"].astype(float)
        for team_id, base_rating in ratings.items():
            key = (int(season), int(team_id))
            curve = np.asarray(
                target_title_probability_curve(
                    season_nodes,
                    ratings,
                    target_team_id=int(team_id),
                    target_ratings=grid,
                    model=model,
                ),
                dtype=float,
            )
            curves[key] = curve
            base_ratings[key] = float(base_rating)
            base_probabilities[key] = float(np.interp(float(base_rating), grid, curve))

    eligible_roster = playoff_roster.loc[
        playoff_roster["salary_share"].notna()
        & playoff_roster["playoff_workload_share"].gt(0)
    ].copy()
    roster_groups = {
        (int(season), int(team_id)): group.sort_values("player_id", kind="mergesort")
        for (season, team_id), group in eligible_roster.groupby(
            ["season", "team_id"], sort=True
        )
        if (int(season), int(team_id)) in curves
    }

    output = focal_seasons.copy()
    result_columns = [
        "salary_peer_placement_count",
        "salary_peer_actual_title_probability",
        "salary_peer_baseline_title_probability",
        "salary_peer_zero_title_probability",
        "salary_peer_raw_title_lift",
        "salary_peer_mean_epm",
        "salary_peer_mean_workload_share",
        "salary_peer_median_salary_gap",
        "salary_peer_p90_salary_gap",
        "title_lift_over_contract",
    ]
    for column in result_columns:
        output[column] = np.nan
    output["contract_hurdle_version"] = None

    covered_positions = np.flatnonzero(output["salary_share"].notna().to_numpy())
    if len(covered_positions):
        focal_salary = output.iloc[covered_positions]["salary_share"].to_numpy(dtype=float)
        focal_season = output.iloc[covered_positions]["season"].to_numpy(dtype=int)
        focal_epm = output.iloc[covered_positions]["own_dpm"].to_numpy(dtype=float)
        count = np.zeros(len(covered_positions), dtype=int)
        actual_sum = np.zeros(len(covered_positions), dtype=float)
        baseline_sum = np.zeros(len(covered_positions), dtype=float)
        zero_sum = np.zeros(len(covered_positions), dtype=float)
        peer_epm_sum = np.zeros(len(covered_positions), dtype=float)
        peer_workload_sum = np.zeros(len(covered_positions), dtype=float)
        gap_columns: list[np.ndarray] = []

        for key, candidates in roster_groups.items():
            ordered = candidates.sort_values(
                ["salary_share", "playoff_workload_share", "player_id"],
                ascending=[True, False, True],
                kind="mergesort",
            ).drop_duplicates("salary_share", keep="first")
            shares = ordered["salary_share"].to_numpy(dtype=float)
            workloads = ordered["playoff_workload_share"].to_numpy(dtype=float)
            contributions = ordered["own_team_contribution_dpm"].to_numpy(dtype=float)
            peer_epms = ordered["own_dpm"].to_numpy(dtype=float)
            insertion = np.searchsorted(shares, focal_salary, side="left")
            lower = np.clip(insertion - 1, 0, len(shares) - 1)
            upper = np.clip(insertion, 0, len(shares) - 1)
            lower_gap = np.abs(shares[lower] - focal_salary)
            upper_gap = np.abs(shares[upper] - focal_salary)
            tied_gap = np.isclose(lower_gap, upper_gap, rtol=0.0, atol=1e-12)
            choose_upper = (upper_gap < lower_gap) | (
                tied_gap & (workloads[upper] > workloads[lower])
            )
            chosen = np.where(choose_upper, upper, lower)
            valid = focal_season != int(key[0])
            workload = workloads[chosen]
            contribution = contributions[chosen]
            actual_rating = base_ratings[key] - contribution + focal_epm * workload
            zero_rating = base_ratings[key] - contribution
            if np.any((actual_rating < grid[0]) | (actual_rating > grid[-1])):
                raise ValueError("A salary-peer actual rating is outside the response grid")
            if np.any((zero_rating < grid[0]) | (zero_rating > grid[-1])):
                raise ValueError("A salary-peer zero rating is outside the response grid")
            curve = curves[key]
            actual_probability = np.interp(actual_rating, grid, curve)
            zero_probability = np.interp(zero_rating, grid, curve)
            count += valid.astype(int)
            actual_sum += np.where(valid, actual_probability, 0.0)
            baseline_sum += np.where(valid, base_probabilities[key], 0.0)
            zero_sum += np.where(valid, zero_probability, 0.0)
            peer_epm_sum += np.where(valid, peer_epms[chosen], 0.0)
            peer_workload_sum += np.where(valid, workload, 0.0)
            chosen_gap = np.abs(shares[chosen] - focal_salary)
            gap_columns.append(np.where(valid, chosen_gap, np.nan))

        valid_rows = count > 0
        denominator = np.where(valid_rows, count, 1)
        actual_mean = actual_sum / denominator
        baseline_mean = baseline_sum / denominator
        zero_mean = zero_sum / denominator
        gap_matrix = np.column_stack(gap_columns)
        with np.errstate(invalid="ignore"):
            median_gap = np.nanmedian(gap_matrix, axis=1)
            p90_gap = np.nanquantile(gap_matrix, 0.90, axis=1)
        labels = output.index[covered_positions]
        output.loc[labels, "salary_peer_placement_count"] = np.where(
            valid_rows, count, np.nan
        )
        output.loc[labels, "salary_peer_actual_title_probability"] = np.where(
            valid_rows, actual_mean, np.nan
        )
        output.loc[labels, "salary_peer_baseline_title_probability"] = np.where(
            valid_rows, baseline_mean, np.nan
        )
        output.loc[labels, "salary_peer_zero_title_probability"] = np.where(
            valid_rows, zero_mean, np.nan
        )
        output.loc[labels, "salary_peer_raw_title_lift"] = np.where(
            valid_rows, actual_mean - zero_mean, np.nan
        )
        output.loc[labels, "salary_peer_mean_epm"] = np.where(
            valid_rows, peer_epm_sum / denominator, np.nan
        )
        output.loc[labels, "salary_peer_mean_workload_share"] = np.where(
            valid_rows, peer_workload_sum / denominator, np.nan
        )
        output.loc[labels, "salary_peer_median_salary_gap"] = np.where(
            valid_rows, median_gap, np.nan
        )
        output.loc[labels, "salary_peer_p90_salary_gap"] = np.where(
            valid_rows, p90_gap, np.nan
        )
        output.loc[labels, "title_lift_over_contract"] = np.where(
            valid_rows, actual_mean - baseline_mean, np.nan
        )
        output.loc[labels, "contract_hurdle_version"] = np.where(
            valid_rows, SALARY_PEER_HURDLE_VERSION, None
        )
    output["neutral_contract_actual_title_probability"] = output[
        "salary_peer_actual_title_probability"
    ]
    output["neutral_contract_hurdle_title_probability"] = output[
        "salary_peer_baseline_title_probability"
    ]
    output["era_neutral_hard_carry_title_probability"] = output[
        "salary_peer_actual_title_probability"
    ]
    output["era_neutral_title_probability"] = output[
        "salary_peer_zero_title_probability"
    ]
    output["contract_epm"] = output["salary_peer_mean_epm"]
    output["epm_over_contract"] = output["own_dpm"] - output["salary_peer_mean_epm"]
    return output
