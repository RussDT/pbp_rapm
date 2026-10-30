"""Salary-peer playoff title value through an explicit best-of-seven bracket."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


PLAYOFF_GAME_MODEL_C_VALUES = (0.05, 0.2, 1.0, 5.0, 20.0, 100.0)
EASTERN_CONFERENCE_ABBREVIATIONS = {
    "ATL",
    "BKN",
    "BOS",
    "CHA",
    "CHI",
    "CLE",
    "DET",
    "IND",
    "MIA",
    "MIL",
    "NYK",
    "ORL",
    "PHI",
    "TOR",
    "WAS",
}
WESTERN_CONFERENCE_ABBREVIATIONS = {
    "DAL",
    "DEN",
    "GSW",
    "HOU",
    "LAC",
    "LAL",
    "MEM",
    "MIN",
    "NOP",
    "OKC",
    "PHX",
    "POR",
    "SAC",
    "SAS",
    "UTA",
}


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


def best_of_seven_win_probability_with_home_court(
    rating_difference: float | np.ndarray,
    model: PlayoffGameModel,
    focal_has_home_court: bool | None,
) -> float | np.ndarray:
    """Price a series with known home court, or average it when tied/unknown."""
    if focal_has_home_court is None:
        return best_of_seven_win_probability(rating_difference, model)
    difference = np.asarray(rating_difference, dtype=float)
    result = _series_probability_with_home_court(
        difference, model, focal_has_home_court
    )
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


def regular_season_seed_probabilities(
    games: pd.DataFrame,
    teams: pd.DataFrame,
    model: PlayoffGameModel,
    rating_shift_grid: np.ndarray | None = None,
    simulations: int = 2048,
    seed: int = 20260825,
) -> pd.DataFrame:
    """Simulate each team's qualification and seed over regular-season strength shifts.

    Game probabilities come from the historical schedule and team ratings. Team win
    totals use the matching Poisson-binomial mean and variance with common normal
    draws across the strength grid, which makes counterfactual seed changes smooth
    and reproducible without pretending the realized game results were fixed.
    """
    game_required = {"season", "home_team_id", "away_team_id"}
    team_required = {"season", "team_id", "conference", "team_dpm"}
    if missing := game_required.difference(games.columns):
        raise ValueError(f"Regular-season games are missing columns: {sorted(missing)}")
    if missing := team_required.difference(teams.columns):
        raise ValueError(f"Regular-season teams are missing columns: {sorted(missing)}")
    if simulations <= 0:
        raise ValueError("simulations must be positive")
    grid = (
        np.linspace(-20.0, 20.0, 81)
        if rating_shift_grid is None
        else np.asarray(rating_shift_grid, dtype=float)
    )
    if grid.ndim != 1 or len(grid) < 2 or not np.all(np.diff(grid) > 0):
        raise ValueError("Regular-season rating-shift grid must be strictly increasing")

    rng = np.random.default_rng(seed)
    records: list[dict[str, float | int]] = []
    for season, season_teams in teams.groupby("season", sort=True):
        season_games = games.loc[games["season"].eq(season)].copy()
        if season_games.empty:
            continue
        season_teams = season_teams.sort_values("team_id", kind="mergesort").reset_index(
            drop=True
        )
        team_ids = season_teams["team_id"].to_numpy(dtype=np.int64)
        id_to_position = {int(team_id): index for index, team_id in enumerate(team_ids)}
        home_position = season_games["home_team_id"].map(id_to_position)
        away_position = season_games["away_team_id"].map(id_to_position)
        if home_position.isna().any() or away_position.isna().any():
            raise ValueError(f"Season {season} schedule contains an unrated team")
        home_position_array = home_position.to_numpy(dtype=np.int64)
        away_position_array = away_position.to_numpy(dtype=np.int64)
        ratings = season_teams["team_dpm"].to_numpy(dtype=float)
        if not np.isfinite(ratings).all():
            raise ValueError(f"Season {season} contains a non-finite team rating")
        home_probability = expit(
            model.home_advantage
            + model.epm_coefficient
            * (ratings[home_position_array] - ratings[away_position_array])
        )
        team_mean = np.zeros(len(team_ids), dtype=float)
        team_variance = np.zeros(len(team_ids), dtype=float)
        team_games = np.zeros(len(team_ids), dtype=float)
        np.add.at(team_mean, home_position_array, home_probability)
        np.add.at(team_mean, away_position_array, 1.0 - home_probability)
        game_variance = home_probability * (1.0 - home_probability)
        np.add.at(team_variance, home_position_array, game_variance)
        np.add.at(team_variance, away_position_array, game_variance)
        np.add.at(team_games, home_position_array, 1.0)
        np.add.at(team_games, away_position_array, 1.0)
        normal_draws = rng.standard_normal((simulations, len(team_ids)))
        baseline_samples = np.clip(
            team_mean[None, :]
            + np.sqrt(np.maximum(team_variance, 1e-12))[None, :] * normal_draws,
            0.0,
            team_games[None, :],
        )
        baseline_samples += rng.uniform(-1e-7, 1e-7, size=baseline_samples.shape)

        conferences = season_teams["conference"].astype(str).to_numpy()
        for target_position, target_team_id in enumerate(team_ids):
            target_home = home_position_array == target_position
            target_away = away_position_array == target_position
            target_games = target_home | target_away
            if not np.any(target_games):
                continue
            base_difference = (
                ratings[home_position_array[target_games]]
                - ratings[away_position_array[target_games]]
            )
            target_sign = np.where(target_home[target_games], 1.0, -1.0)
            shifted_home_probability = expit(
                model.home_advantage
                + model.epm_coefficient
                * (base_difference[None, :] + grid[:, None] * target_sign[None, :])
            )
            target_game_probability = np.where(
                target_sign[None, :] > 0,
                shifted_home_probability,
                1.0 - shifted_home_probability,
            )
            target_mean = target_game_probability.sum(axis=1)
            target_variance = (
                target_game_probability * (1.0 - target_game_probability)
            ).sum(axis=1)
            target_samples = np.clip(
                target_mean[:, None]
                + np.sqrt(np.maximum(target_variance, 1e-12))[:, None]
                * normal_draws[:, target_position][None, :],
                0.0,
                float(target_games.sum()),
            )
            conference_positions = np.flatnonzero(
                conferences == conferences[target_position]
            )
            opponent_positions = conference_positions[
                conference_positions != target_position
            ]
            opponent_samples = baseline_samples[:, opponent_positions]
            ranks = 1 + (
                opponent_samples[None, :, :] > target_samples[:, :, None]
            ).sum(axis=2)
            for grid_index, rating_shift in enumerate(grid):
                row: dict[str, float | int] = {
                    "season": int(season),
                    "team_id": int(target_team_id),
                    "rating_shift": float(rating_shift),
                }
                seed_probabilities = np.array(
                    [np.mean(ranks[grid_index] == seed_number) for seed_number in range(1, 9)],
                    dtype=float,
                )
                playoff_probability = float(seed_probabilities.sum())
                row["playoff_probability"] = playoff_probability
                row["conditional_mean_seed"] = (
                    float(
                        np.dot(np.arange(1, 9, dtype=float), seed_probabilities)
                        / playoff_probability
                    )
                    if playoff_probability > 0
                    else np.nan
                )
                for seed_number, probability in enumerate(seed_probabilities, start=1):
                    row[f"seed_{seed_number}_probability"] = float(probability)
                records.append(row)
    return pd.DataFrame.from_records(records).sort_values(
        ["season", "team_id", "rating_shift"], kind="mergesort"
    ).reset_index(drop=True)


def conference_from_abbreviation(team_abbr: str, season: int | None = None) -> str:
    """Return the NBA conference for the normalized historical abbreviation."""
    abbreviation = str(team_abbr).upper()
    if abbreviation == "NOP" and season is not None and int(season) <= 2004:
        return "East"
    if abbreviation in EASTERN_CONFERENCE_ABBREVIATIONS:
        return "East"
    if abbreviation in WESTERN_CONFERENCE_ABBREVIATIONS:
        return "West"
    raise ValueError(f"Unknown NBA conference for team abbreviation {team_abbr!r}")


def build_playoff_seed_fields(
    regular_games: pd.DataFrame,
    playoff_teams: pd.DataFrame,
) -> pd.DataFrame:
    """Rank each realized playoff field by regular-season record and conference."""
    game_required = {
        "season",
        "home_team_id",
        "away_team_id",
        "home_score",
        "away_score",
    }
    team_required = {"season", "team_id", "team_abbr", "team_dpm"}
    if missing := game_required.difference(regular_games.columns):
        raise ValueError(f"Regular-season games are missing columns: {sorted(missing)}")
    if missing := team_required.difference(playoff_teams.columns):
        raise ValueError(f"Playoff teams are missing columns: {sorted(missing)}")
    games = regular_games.copy()
    games["home_win"] = (
        pd.to_numeric(games["home_score"], errors="coerce")
        > pd.to_numeric(games["away_score"], errors="coerce")
    ).astype(float)
    home = games[["season", "home_team_id", "home_win"]].rename(
        columns={"home_team_id": "team_id", "home_win": "win"}
    )
    away = games[["season", "away_team_id", "home_win"]].rename(
        columns={"away_team_id": "team_id"}
    )
    away["win"] = 1.0 - away["home_win"]
    records = (
        pd.concat([home, away[["season", "team_id", "win"]]], ignore_index=True)
        .groupby(["season", "team_id"], as_index=False)
        .agg(regular_season_games=("win", "size"), regular_season_wins=("win", "sum"))
    )
    field = playoff_teams[list(team_required)].merge(
        records, on=["season", "team_id"], how="left", validate="one_to_one"
    )
    if field[["regular_season_games", "regular_season_wins"]].isna().any().any():
        raise ValueError("Some playoff teams do not have a regular-season record")
    field["conference"] = [
        conference_from_abbreviation(team_abbr, int(season))
        for team_abbr, season in zip(field["team_abbr"], field["season"])
    ]
    field = field.sort_values(
        ["season", "conference", "regular_season_wins", "team_dpm", "team_id"],
        ascending=[True, True, False, False, True],
        kind="mergesort",
    )
    field["seed"] = field.groupby(["season", "conference"]).cumcount() + 1
    counts = field.groupby(["season", "conference"]).size()
    invalid = counts[counts.ne(8)]
    if not invalid.empty:
        raise ValueError(f"Playoff seed fields do not contain eight teams: {invalid.to_dict()}")
    return field.sort_values(
        ["season", "conference", "seed"], kind="mergesort"
    ).reset_index(drop=True)


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


def seeded_target_title_probability_curve(
    field: pd.DataFrame,
    team_ratings: pd.Series,
    target_team_id: int,
    target_ratings: float | np.ndarray,
    model: PlayoffGameModel,
) -> float | np.ndarray:
    """Evaluate a target through the standard seeded NBA bracket with home court."""
    required = {"team_id", "conference", "seed"}
    if missing := required.difference(field.columns):
        raise ValueError(f"Seeded playoff field is missing columns: {sorted(missing)}")
    if field["team_id"].duplicated().any():
        raise ValueError("Seeded playoff field contains duplicate teams")
    conferences = sorted(field["conference"].astype(str).unique())
    if len(conferences) != 2:
        raise ValueError("Seeded playoff field requires exactly two conferences")
    for conference in conferences:
        seeds = sorted(
            pd.to_numeric(
                field.loc[field["conference"].astype(str).eq(conference), "seed"],
                errors="raise",
            ).astype(int)
        )
        if seeds != list(range(1, 9)):
            raise ValueError(f"Conference {conference} does not contain seeds 1-8")
    target_team_id = int(target_team_id)
    if target_team_id not in set(field["team_id"].astype(int)):
        raise ValueError(f"Target team {target_team_id} is not in the seeded field")
    rating = np.asarray(target_ratings, dtype=float)
    shape = rating.shape
    seed_by_team = field.set_index("team_id")["seed"].astype(int).to_dict()

    def team_rating(team_id: int) -> np.ndarray:
        if int(team_id) == target_team_id:
            return rating
        try:
            value = float(team_ratings.loc[int(team_id)])
        except KeyError as exc:
            raise ValueError(f"Missing seeded playoff-team rating: {exc}") from exc
        return np.full(shape, value, dtype=float)

    def play_series(
        left: dict[int, np.ndarray], right: dict[int, np.ndarray]
    ) -> dict[int, np.ndarray]:
        output: dict[int, np.ndarray] = {}
        for team_a, mass_a in left.items():
            for team_b, mass_b in right.items():
                seed_a = int(seed_by_team[team_a])
                seed_b = int(seed_by_team[team_b])
                focal_has_home_court = (
                    True if seed_a < seed_b else False if seed_a > seed_b else None
                )
                probability_a = np.asarray(
                    best_of_seven_win_probability_with_home_court(
                        team_rating(team_a) - team_rating(team_b),
                        model,
                        focal_has_home_court,
                    ),
                    dtype=float,
                )
                matchup_mass = mass_a * mass_b
                output[team_a] = output.get(team_a, np.zeros(shape)) + (
                    matchup_mass * probability_a
                )
                output[team_b] = output.get(team_b, np.zeros(shape)) + (
                    matchup_mass * (1.0 - probability_a)
                )
        return output

    conference_winners: list[dict[int, np.ndarray]] = []
    first_round_pairs = ((1, 8), (4, 5), (2, 7), (3, 6))
    for conference in conferences:
        conference_field = field.loc[
            field["conference"].astype(str).eq(conference)
        ].set_index("seed")["team_id"]
        first_round = [
            play_series(
                {int(conference_field.loc[seed_a]): np.ones(shape)},
                {int(conference_field.loc[seed_b]): np.ones(shape)},
            )
            for seed_a, seed_b in first_round_pairs
        ]
        semifinal_a = play_series(first_round[0], first_round[1])
        semifinal_b = play_series(first_round[2], first_round[3])
        conference_winners.append(play_series(semifinal_a, semifinal_b))
    champion = play_series(conference_winners[0], conference_winners[1])
    probability = champion[target_team_id]
    if probability.ndim == 0:
        return float(probability)
    return probability


SALARY_PEER_HURDLE_VERSION = "salary-peer-full-season-v2"
ERA_LOCAL_SALARY_PEER_HURDLE_VERSION = "salary-peer-era-local-budget-neutral-v3"
PLAYOFF_ADJUSTED_SALARY_PEER_HURDLE_VERSION = (
    "salary-peer-era-local-budget-neutral-playoff-prior-v4"
)
ERA_LOCAL_SALARY_PEER_WINDOW = 4
REGULAR_SEASON_CAP_POINT_PRICE = 0.067
PLAYOFF_CAP_POINT_PRICE = 0.078


def add_salary_peer_title_lift(
    focal_seasons: pd.DataFrame,
    playoff_roster: pd.DataFrame,
    playoff_teams: pd.DataFrame,
    bracket_nodes: pd.DataFrame,
    model: PlayoffGameModel,
    rating_grid: np.ndarray | None = None,
    regular_seed_probabilities: pd.DataFrame | None = None,
    playoff_seed_fields: pd.DataFrame | None = None,
    era_window: int = ERA_LOCAL_SALARY_PEER_WINDOW,
    regular_season_cap_point_price: float = REGULAR_SEASON_CAP_POINT_PRICE,
    playoff_cap_point_price: float = PLAYOFF_CAP_POINT_PRICE,
) -> pd.DataFrame:
    """Average salary-peer title lift with full-history and era-local estimates."""
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
    seeded_mode = regular_seed_probabilities is not None or playoff_seed_fields is not None
    if seeded_mode and (
        regular_seed_probabilities is None or playoff_seed_fields is None
    ):
        raise ValueError(
            "Regular-season seed probabilities and playoff seed fields must be supplied together"
        )
    if seeded_mode and "regular_season_workload_share" not in playoff_roster.columns:
        raise ValueError(
            "Salary-peer playoff roster is missing regular_season_workload_share"
        )
    grid = (
        np.linspace(-30.0, 40.0, 3501)
        if rating_grid is None
        else np.asarray(rating_grid, dtype=float)
    )
    if grid.ndim != 1 or len(grid) < 2 or not np.all(np.diff(grid) > 0):
        raise ValueError("Salary-peer rating grid must be a strictly increasing vector")
    if era_window < 1:
        raise ValueError("Salary-peer era window must be at least one season")
    if regular_season_cap_point_price < 0 or playoff_cap_point_price < 0:
        raise ValueError("Salary-peer cap-point prices cannot be negative")

    curves: dict[tuple[int, int], np.ndarray] = {}
    seeded_curves: dict[tuple[int, int, int], np.ndarray] = {}
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
            if seeded_mode:
                assert playoff_seed_fields is not None
                season_field = playoff_seed_fields.loc[
                    playoff_seed_fields["season"].eq(int(season))
                ].copy()
                target_row = season_field.loc[
                    season_field["team_id"].eq(int(team_id))
                ]
                if len(target_row) != 1:
                    raise ValueError(
                        f"Team {team_id} has no unique playoff seed-field row in {season}"
                    )
                target_conference = str(target_row.iloc[0]["conference"])
                for target_seed in range(1, 9):
                    conference_field = season_field.loc[
                        season_field["conference"].astype(str).eq(target_conference)
                    ].sort_values("seed", kind="mergesort")
                    other_teams = conference_field.loc[
                        conference_field["team_id"].ne(int(team_id)), "team_id"
                    ].astype(int).tolist()
                    reordered = other_teams.copy()
                    reordered.insert(target_seed - 1, int(team_id))
                    moved_field = season_field.loc[
                        ~season_field["conference"].astype(str).eq(target_conference)
                    ].copy()
                    moved_conference = pd.DataFrame(
                        {
                            "season": int(season),
                            "team_id": reordered,
                            "conference": target_conference,
                            "seed": np.arange(1, 9, dtype=int),
                        }
                    )
                    moved_field = pd.concat(
                        [moved_field, moved_conference], ignore_index=True
                    )
                    seeded_curves[(int(season), int(team_id), target_seed)] = np.asarray(
                        seeded_target_title_probability_curve(
                            moved_field,
                            ratings,
                            target_team_id=int(team_id),
                            target_ratings=grid,
                            model=model,
                        ),
                        dtype=float,
                    )

    adjusted_columns = {
        "focal": "playoff_adjusted_dpm" in focal_seasons.columns,
        "roster_rating": "playoff_adjusted_dpm" in playoff_roster.columns,
        "roster_contribution": (
            "playoff_adjusted_team_contribution_dpm" in playoff_roster.columns
        ),
        "team": "playoff_adjusted_team_dpm" in playoff_teams.columns,
    }
    if any(adjusted_columns.values()) and not all(adjusted_columns.values()):
        missing = [name for name, present in adjusted_columns.items() if not present]
        raise ValueError(
            "Playoff-adjusted salary-peer inputs must be supplied together; "
            f"missing {missing}"
        )
    has_playoff_adjustment = all(adjusted_columns.values())
    adjusted_curves = curves
    adjusted_seeded_curves = seeded_curves
    adjusted_base_ratings = base_ratings
    adjusted_base_probabilities = base_probabilities
    if has_playoff_adjustment:
        adjusted_curves = {}
        adjusted_seeded_curves = {}
        adjusted_base_ratings = {}
        adjusted_base_probabilities = {}
        for season, season_teams in playoff_teams.groupby("season", sort=True):
            season_nodes = bracket_nodes.loc[bracket_nodes["season"].eq(int(season))]
            if season_nodes.empty:
                continue
            ratings = season_teams.set_index("team_id")[
                "playoff_adjusted_team_dpm"
            ].astype(float)
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
                adjusted_curves[key] = curve
                adjusted_base_ratings[key] = float(base_rating)
                adjusted_base_probabilities[key] = float(
                    np.interp(float(base_rating), grid, curve)
                )
                if seeded_mode:
                    assert playoff_seed_fields is not None
                    season_field = playoff_seed_fields.loc[
                        playoff_seed_fields["season"].eq(int(season))
                    ].copy()
                    target_row = season_field.loc[
                        season_field["team_id"].eq(int(team_id))
                    ]
                    if len(target_row) != 1:
                        raise ValueError(
                            f"Team {team_id} has no unique playoff seed-field row in {season}"
                        )
                    target_conference = str(target_row.iloc[0]["conference"])
                    for target_seed in range(1, 9):
                        conference_field = season_field.loc[
                            season_field["conference"].astype(str).eq(target_conference)
                        ].sort_values("seed", kind="mergesort")
                        other_teams = conference_field.loc[
                            conference_field["team_id"].ne(int(team_id)), "team_id"
                        ].astype(int).tolist()
                        reordered = other_teams.copy()
                        reordered.insert(target_seed - 1, int(team_id))
                        moved_field = season_field.loc[
                            ~season_field["conference"].astype(str).eq(target_conference)
                        ].copy()
                        moved_conference = pd.DataFrame(
                            {
                                "season": int(season),
                                "team_id": reordered,
                                "conference": target_conference,
                                "seed": np.arange(1, 9, dtype=int),
                            }
                        )
                        moved_field = pd.concat(
                            [moved_field, moved_conference], ignore_index=True
                        )
                        adjusted_seeded_curves[
                            (int(season), int(team_id), target_seed)
                        ] = np.asarray(
                            seeded_target_title_probability_curve(
                                moved_field,
                                ratings,
                                target_team_id=int(team_id),
                                target_ratings=grid,
                                model=model,
                            ),
                            dtype=float,
                        )

    seed_surface_groups: dict[tuple[int, int], pd.DataFrame] = {}
    if seeded_mode:
        assert regular_seed_probabilities is not None
        seed_required = {
            "season",
            "team_id",
            "rating_shift",
            "playoff_probability",
            "conditional_mean_seed",
            *{f"seed_{seed}_probability" for seed in range(1, 9)},
        }
        if missing := seed_required.difference(regular_seed_probabilities.columns):
            raise ValueError(
                f"Regular-season seed probabilities are missing columns: {sorted(missing)}"
            )
        seed_surface_groups = {
            (int(season), int(team_id)): group.sort_values(
                "rating_shift", kind="mergesort"
            )
            for (season, team_id), group in regular_seed_probabilities.groupby(
                ["season", "team_id"], sort=True
            )
        }

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
        "salary_peer_actual_playoff_probability",
        "salary_peer_baseline_playoff_probability",
        "salary_peer_zero_playoff_probability",
        "salary_peer_actual_mean_seed",
        "salary_peer_baseline_mean_seed",
        "salary_peer_zero_mean_seed",
        "salary_peer_fixed_bracket_actual_title_probability",
        "salary_peer_fixed_bracket_baseline_title_probability",
        "salary_peer_fixed_bracket_zero_title_probability",
        "salary_peer_fixed_bracket_raw_title_lift",
        "salary_peer_fixed_bracket_title_lift",
        "salary_peer_seeding_title_lift",
        "title_lift_over_contract",
        "era_local_salary_peer_placement_count",
        "era_local_budget_actual_title_probability",
        "era_local_budget_baseline_title_probability",
        "era_local_budget_actual_playoff_probability",
        "era_local_budget_baseline_playoff_probability",
        "era_local_budget_actual_mean_seed",
        "era_local_budget_baseline_mean_seed",
        "era_local_budget_fixed_bracket_actual_title_probability",
        "era_local_budget_fixed_bracket_baseline_title_probability",
        "era_local_budget_fixed_bracket_title_lift",
        "era_local_budget_seeding_title_lift",
        "era_local_budget_mean_signed_salary_gap_cap_points",
        "era_local_budget_median_salary_gap",
        "era_local_budget_p90_salary_gap",
        "era_local_budget_share_salary_gap_over_5",
        "era_local_budget_mean_regular_season_debit",
        "era_local_budget_mean_playoff_debit",
        "title_lift_over_contract_v2",
        "playoff_adjusted_actual_title_probability",
        "playoff_adjusted_baseline_title_probability",
        "playoff_adjusted_fixed_bracket_actual_title_probability",
        "playoff_adjusted_fixed_bracket_baseline_title_probability",
        "playoff_adjusted_fixed_bracket_title_lift",
        "playoff_adjusted_seeding_title_lift",
        "playoff_adjusted_title_lift_over_contract_v2",
    ]
    for column in result_columns:
        output[column] = np.nan
    output["contract_hurdle_version"] = None
    output["contract_hurdle_version_v2"] = None
    output["playoff_adjusted_contract_hurdle_version"] = None

    covered_positions = np.flatnonzero(output["salary_share"].notna().to_numpy())
    if len(covered_positions):
        focal_salary = output.iloc[covered_positions]["salary_share"].to_numpy(dtype=float)
        focal_season = output.iloc[covered_positions]["season"].to_numpy(dtype=int)
        focal_epm = output.iloc[covered_positions]["own_dpm"].to_numpy(dtype=float)
        focal_playoff_epm = (
            output.iloc[covered_positions]["playoff_adjusted_dpm"].to_numpy(dtype=float)
            if has_playoff_adjustment
            else focal_epm
        )
        count = np.zeros(len(covered_positions), dtype=int)
        actual_sum = np.zeros(len(covered_positions), dtype=float)
        baseline_sum = np.zeros(len(covered_positions), dtype=float)
        zero_sum = np.zeros(len(covered_positions), dtype=float)
        fixed_actual_sum = np.zeros(len(covered_positions), dtype=float)
        fixed_baseline_sum = np.zeros(len(covered_positions), dtype=float)
        fixed_zero_sum = np.zeros(len(covered_positions), dtype=float)
        actual_playoff_sum = np.zeros(len(covered_positions), dtype=float)
        baseline_playoff_sum = np.zeros(len(covered_positions), dtype=float)
        zero_playoff_sum = np.zeros(len(covered_positions), dtype=float)
        actual_seed_weighted_sum = np.zeros(len(covered_positions), dtype=float)
        baseline_seed_weighted_sum = np.zeros(len(covered_positions), dtype=float)
        zero_seed_weighted_sum = np.zeros(len(covered_positions), dtype=float)
        peer_epm_sum = np.zeros(len(covered_positions), dtype=float)
        peer_workload_sum = np.zeros(len(covered_positions), dtype=float)
        gap_columns: list[np.ndarray] = []
        era_count = np.zeros(len(covered_positions), dtype=int)
        era_actual_sum = np.zeros(len(covered_positions), dtype=float)
        era_baseline_sum = np.zeros(len(covered_positions), dtype=float)
        era_fixed_actual_sum = np.zeros(len(covered_positions), dtype=float)
        era_fixed_baseline_sum = np.zeros(len(covered_positions), dtype=float)
        era_actual_playoff_sum = np.zeros(len(covered_positions), dtype=float)
        era_baseline_playoff_sum = np.zeros(len(covered_positions), dtype=float)
        era_actual_seed_weighted_sum = np.zeros(len(covered_positions), dtype=float)
        era_baseline_seed_weighted_sum = np.zeros(len(covered_positions), dtype=float)
        era_signed_gap_sum = np.zeros(len(covered_positions), dtype=float)
        era_regular_debit_sum = np.zeros(len(covered_positions), dtype=float)
        era_playoff_debit_sum = np.zeros(len(covered_positions), dtype=float)
        era_gap_columns: list[np.ndarray] = []
        adjusted_actual_sum = np.zeros(len(covered_positions), dtype=float)
        adjusted_baseline_sum = np.zeros(len(covered_positions), dtype=float)
        adjusted_fixed_actual_sum = np.zeros(len(covered_positions), dtype=float)
        adjusted_fixed_baseline_sum = np.zeros(len(covered_positions), dtype=float)

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
            adjusted_peer_epms = (
                ordered["playoff_adjusted_dpm"].to_numpy(dtype=float)
                if has_playoff_adjustment
                else peer_epms
            )
            adjusted_contributions = (
                ordered["playoff_adjusted_team_contribution_dpm"].to_numpy(
                    dtype=float
                )
                if has_playoff_adjustment
                else contributions
            )
            regular_workloads = (
                ordered["regular_season_workload_share"]
                .fillna(0.0)
                .to_numpy(dtype=float)
                if seeded_mode
                else np.zeros(len(ordered), dtype=float)
            )
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
            era_valid = valid & (np.abs(focal_season - int(key[0])) <= era_window)
            workload = workloads[chosen]
            contribution = contributions[chosen]
            peer_epm = peer_epms[chosen]
            regular_workload = regular_workloads[chosen]
            actual_rating = base_ratings[key] - contribution + focal_epm * workload
            zero_rating = base_ratings[key] - contribution
            if np.any((actual_rating < grid[0]) | (actual_rating > grid[-1])):
                raise ValueError("A salary-peer actual rating is outside the response grid")
            if np.any((zero_rating < grid[0]) | (zero_rating > grid[-1])):
                raise ValueError("A salary-peer zero rating is outside the response grid")
            curve = curves[key]
            fixed_actual_probability = np.interp(actual_rating, grid, curve)
            fixed_baseline_probability = np.full(
                len(covered_positions), base_probabilities[key], dtype=float
            )
            fixed_zero_probability = np.interp(zero_rating, grid, curve)
            signed_gap_cap_points = 100.0 * (focal_salary - shares[chosen])
            regular_season_debit = (
                regular_season_cap_point_price * signed_gap_cap_points
            )
            playoff_debit = playoff_cap_point_price * signed_gap_cap_points
            era_actual_rating = actual_rating - playoff_debit
            if np.any(
                era_valid
                & ((era_actual_rating < grid[0]) | (era_actual_rating > grid[-1]))
            ):
                raise ValueError(
                    "An era-local salary-peer actual rating is outside the response grid"
                )
            era_fixed_actual_probability = np.interp(era_actual_rating, grid, curve)
            era_fixed_baseline_probability = fixed_baseline_probability
            adjusted_base_rating = adjusted_base_ratings[key]
            adjusted_actual_rating = (
                adjusted_base_rating
                - adjusted_contributions[chosen]
                + focal_playoff_epm * workload
                - playoff_debit
            )
            if np.any(
                era_valid
                & (
                    (adjusted_actual_rating < grid[0])
                    | (adjusted_actual_rating > grid[-1])
                )
            ):
                raise ValueError(
                    "A playoff-adjusted salary-peer rating is outside the response grid"
                )
            adjusted_curve = adjusted_curves[key]
            adjusted_fixed_actual_probability = np.interp(
                adjusted_actual_rating, grid, adjusted_curve
            )
            adjusted_fixed_baseline_probability = np.full(
                len(covered_positions),
                adjusted_base_probabilities[key],
                dtype=float,
            )
            if seeded_mode:
                surface = seed_surface_groups.get(key)
                if surface is None:
                    raise ValueError(f"Missing regular-season seed surface for {key}")
                shift_grid = surface["rating_shift"].to_numpy(dtype=float)
                actual_shift = (focal_epm - peer_epm) * regular_workload
                baseline_shift = np.zeros(len(covered_positions), dtype=float)
                zero_shift = -peer_epm * regular_workload
                era_actual_shift = actual_shift - regular_season_debit
                for label, shifts in (
                    ("actual", actual_shift),
                    ("zero", zero_shift),
                    ("era-local actual", era_actual_shift),
                ):
                    relevant = era_valid if label == "era-local actual" else np.ones(
                        len(shifts), dtype=bool
                    )
                    if np.any(
                        relevant & ((shifts < shift_grid[0]) | (shifts > shift_grid[-1]))
                    ):
                        raise ValueError(
                            f"A salary-peer {label} regular-season shift is outside the seed grid"
                        )

                def seeded_probability(
                    shifts: np.ndarray, candidate_ratings: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                    title_probability = np.zeros(len(shifts), dtype=float)
                    playoff_probability = np.zeros(len(shifts), dtype=float)
                    seed_weighted = np.zeros(len(shifts), dtype=float)
                    for seed_number in range(1, 9):
                        seed_probability = np.interp(
                            shifts,
                            shift_grid,
                            surface[f"seed_{seed_number}_probability"].to_numpy(
                                dtype=float
                            ),
                        )
                        title_if_seeded = np.interp(
                            candidate_ratings,
                            grid,
                            seeded_curves[(key[0], key[1], seed_number)],
                        )
                        title_probability += seed_probability * title_if_seeded
                        playoff_probability += seed_probability
                        seed_weighted += seed_number * seed_probability
                    return title_probability, playoff_probability, seed_weighted

                actual_probability, actual_playoff, actual_seed_weighted = (
                    seeded_probability(actual_shift, actual_rating)
                )
                baseline_probability, baseline_playoff, baseline_seed_weighted = (
                    seeded_probability(
                        baseline_shift,
                        np.full(len(covered_positions), base_ratings[key], dtype=float),
                    )
                )
                zero_probability, zero_playoff, zero_seed_weighted = seeded_probability(
                    zero_shift, zero_rating
                )
                (
                    era_actual_probability,
                    era_actual_playoff,
                    era_actual_seed_weighted,
                ) = seeded_probability(era_actual_shift, era_actual_rating)
                era_baseline_probability = baseline_probability
                era_baseline_playoff = baseline_playoff
                era_baseline_seed_weighted = baseline_seed_weighted
                adjusted_actual_probability = np.zeros(
                    len(covered_positions), dtype=float
                )
                adjusted_baseline_probability = np.zeros(
                    len(covered_positions), dtype=float
                )
                for seed_number in range(1, 9):
                    adjusted_seed_probability = np.interp(
                        era_actual_shift,
                        shift_grid,
                        surface[f"seed_{seed_number}_probability"].to_numpy(
                            dtype=float
                        ),
                    )
                    baseline_seed_probability = np.interp(
                        baseline_shift,
                        shift_grid,
                        surface[f"seed_{seed_number}_probability"].to_numpy(
                            dtype=float
                        ),
                    )
                    adjusted_actual_probability += adjusted_seed_probability * np.interp(
                        adjusted_actual_rating,
                        grid,
                        adjusted_seeded_curves[(key[0], key[1], seed_number)],
                    )
                    adjusted_baseline_probability += (
                        baseline_seed_probability
                        * np.interp(
                            np.full(
                                len(covered_positions),
                                adjusted_base_rating,
                                dtype=float,
                            ),
                            grid,
                            adjusted_seeded_curves[(key[0], key[1], seed_number)],
                        )
                    )
            else:
                actual_probability = fixed_actual_probability
                baseline_probability = fixed_baseline_probability
                zero_probability = fixed_zero_probability
                actual_playoff = baseline_playoff = zero_playoff = np.ones(
                    len(covered_positions), dtype=float
                )
                actual_seed_weighted = baseline_seed_weighted = zero_seed_weighted = (
                    np.full(len(covered_positions), np.nan)
                )
                era_actual_probability = era_fixed_actual_probability
                era_baseline_probability = era_fixed_baseline_probability
                era_actual_playoff = era_baseline_playoff = np.ones(
                    len(covered_positions), dtype=float
                )
                era_actual_seed_weighted = era_baseline_seed_weighted = np.full(
                    len(covered_positions), np.nan
                )
                adjusted_actual_probability = adjusted_fixed_actual_probability
                adjusted_baseline_probability = adjusted_fixed_baseline_probability
            count += valid.astype(int)
            actual_sum += np.where(valid, actual_probability, 0.0)
            baseline_sum += np.where(valid, baseline_probability, 0.0)
            zero_sum += np.where(valid, zero_probability, 0.0)
            fixed_actual_sum += np.where(valid, fixed_actual_probability, 0.0)
            fixed_baseline_sum += np.where(valid, fixed_baseline_probability, 0.0)
            fixed_zero_sum += np.where(valid, fixed_zero_probability, 0.0)
            actual_playoff_sum += np.where(valid, actual_playoff, 0.0)
            baseline_playoff_sum += np.where(valid, baseline_playoff, 0.0)
            zero_playoff_sum += np.where(valid, zero_playoff, 0.0)
            actual_seed_weighted_sum += np.where(valid, actual_seed_weighted, 0.0)
            baseline_seed_weighted_sum += np.where(valid, baseline_seed_weighted, 0.0)
            zero_seed_weighted_sum += np.where(valid, zero_seed_weighted, 0.0)
            peer_epm_sum += np.where(valid, peer_epm, 0.0)
            peer_workload_sum += np.where(valid, workload, 0.0)
            chosen_gap = np.abs(shares[chosen] - focal_salary)
            gap_columns.append(np.where(valid, chosen_gap, np.nan))
            era_count += era_valid.astype(int)
            era_actual_sum += np.where(era_valid, era_actual_probability, 0.0)
            era_baseline_sum += np.where(era_valid, era_baseline_probability, 0.0)
            era_fixed_actual_sum += np.where(
                era_valid, era_fixed_actual_probability, 0.0
            )
            era_fixed_baseline_sum += np.where(
                era_valid, era_fixed_baseline_probability, 0.0
            )
            era_actual_playoff_sum += np.where(era_valid, era_actual_playoff, 0.0)
            era_baseline_playoff_sum += np.where(
                era_valid, era_baseline_playoff, 0.0
            )
            era_actual_seed_weighted_sum += np.where(
                era_valid, era_actual_seed_weighted, 0.0
            )
            era_baseline_seed_weighted_sum += np.where(
                era_valid, era_baseline_seed_weighted, 0.0
            )
            era_signed_gap_sum += np.where(
                era_valid, signed_gap_cap_points, 0.0
            )
            era_regular_debit_sum += np.where(
                era_valid, regular_season_debit, 0.0
            )
            era_playoff_debit_sum += np.where(era_valid, playoff_debit, 0.0)
            era_gap_columns.append(np.where(era_valid, chosen_gap, np.nan))
            adjusted_actual_sum += np.where(
                era_valid, adjusted_actual_probability, 0.0
            )
            adjusted_baseline_sum += np.where(
                era_valid, adjusted_baseline_probability, 0.0
            )
            adjusted_fixed_actual_sum += np.where(
                era_valid, adjusted_fixed_actual_probability, 0.0
            )
            adjusted_fixed_baseline_sum += np.where(
                era_valid, adjusted_fixed_baseline_probability, 0.0
            )

        valid_rows = count > 0
        denominator = np.where(valid_rows, count, 1)
        actual_mean = actual_sum / denominator
        baseline_mean = baseline_sum / denominator
        zero_mean = zero_sum / denominator
        fixed_actual_mean = fixed_actual_sum / denominator
        fixed_baseline_mean = fixed_baseline_sum / denominator
        fixed_zero_mean = fixed_zero_sum / denominator
        actual_playoff_mean = actual_playoff_sum / denominator
        baseline_playoff_mean = baseline_playoff_sum / denominator
        zero_playoff_mean = zero_playoff_sum / denominator
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
        output.loc[labels, "salary_peer_actual_playoff_probability"] = np.where(
            valid_rows, actual_playoff_mean, np.nan
        )
        output.loc[labels, "salary_peer_baseline_playoff_probability"] = np.where(
            valid_rows, baseline_playoff_mean, np.nan
        )
        output.loc[labels, "salary_peer_zero_playoff_probability"] = np.where(
            valid_rows, zero_playoff_mean, np.nan
        )
        output.loc[labels, "salary_peer_actual_mean_seed"] = np.where(
            valid_rows & (actual_playoff_sum > 0),
            actual_seed_weighted_sum / np.where(actual_playoff_sum > 0, actual_playoff_sum, 1),
            np.nan,
        )
        output.loc[labels, "salary_peer_baseline_mean_seed"] = np.where(
            valid_rows & (baseline_playoff_sum > 0),
            baseline_seed_weighted_sum
            / np.where(baseline_playoff_sum > 0, baseline_playoff_sum, 1),
            np.nan,
        )
        output.loc[labels, "salary_peer_zero_mean_seed"] = np.where(
            valid_rows & (zero_playoff_sum > 0),
            zero_seed_weighted_sum / np.where(zero_playoff_sum > 0, zero_playoff_sum, 1),
            np.nan,
        )
        output.loc[
            labels, "salary_peer_fixed_bracket_actual_title_probability"
        ] = np.where(valid_rows, fixed_actual_mean, np.nan)
        output.loc[
            labels, "salary_peer_fixed_bracket_baseline_title_probability"
        ] = np.where(valid_rows, fixed_baseline_mean, np.nan)
        output.loc[
            labels, "salary_peer_fixed_bracket_zero_title_probability"
        ] = np.where(valid_rows, fixed_zero_mean, np.nan)
        output.loc[labels, "salary_peer_fixed_bracket_raw_title_lift"] = np.where(
            valid_rows, fixed_actual_mean - fixed_zero_mean, np.nan
        )
        fixed_lift = fixed_actual_mean - fixed_baseline_mean
        full_lift = actual_mean - baseline_mean
        output.loc[labels, "salary_peer_fixed_bracket_title_lift"] = np.where(
            valid_rows, fixed_lift, np.nan
        )
        output.loc[labels, "salary_peer_seeding_title_lift"] = np.where(
            valid_rows, full_lift - fixed_lift, np.nan
        )
        output.loc[labels, "title_lift_over_contract"] = np.where(
            valid_rows, full_lift, np.nan
        )
        output.loc[labels, "contract_hurdle_version"] = np.where(
            valid_rows, SALARY_PEER_HURDLE_VERSION, None
        )
        era_valid_rows = era_count > 0
        era_denominator = np.where(era_valid_rows, era_count, 1)
        era_actual_mean = era_actual_sum / era_denominator
        era_baseline_mean = era_baseline_sum / era_denominator
        era_fixed_actual_mean = era_fixed_actual_sum / era_denominator
        era_fixed_baseline_mean = era_fixed_baseline_sum / era_denominator
        era_full_lift = era_actual_mean - era_baseline_mean
        era_fixed_lift = era_fixed_actual_mean - era_fixed_baseline_mean
        era_gap_matrix = np.column_stack(era_gap_columns)
        with np.errstate(invalid="ignore"):
            era_median_gap = np.nanmedian(era_gap_matrix, axis=1)
            era_p90_gap = np.nanquantile(era_gap_matrix, 0.90, axis=1)
            era_share_gap_over_5 = np.nanmean(
                np.where(
                    np.isnan(era_gap_matrix),
                    np.nan,
                    (era_gap_matrix > 0.05).astype(float),
                ),
                axis=1,
            )
        output.loc[labels, "era_local_salary_peer_placement_count"] = np.where(
            era_valid_rows, era_count, np.nan
        )
        output.loc[labels, "era_local_budget_actual_title_probability"] = np.where(
            era_valid_rows, era_actual_mean, np.nan
        )
        output.loc[labels, "era_local_budget_baseline_title_probability"] = np.where(
            era_valid_rows, era_baseline_mean, np.nan
        )
        output.loc[labels, "era_local_budget_actual_playoff_probability"] = np.where(
            era_valid_rows, era_actual_playoff_sum / era_denominator, np.nan
        )
        output.loc[labels, "era_local_budget_baseline_playoff_probability"] = np.where(
            era_valid_rows, era_baseline_playoff_sum / era_denominator, np.nan
        )
        output.loc[labels, "era_local_budget_actual_mean_seed"] = np.where(
            era_valid_rows & (era_actual_playoff_sum > 0),
            era_actual_seed_weighted_sum
            / np.where(era_actual_playoff_sum > 0, era_actual_playoff_sum, 1),
            np.nan,
        )
        output.loc[labels, "era_local_budget_baseline_mean_seed"] = np.where(
            era_valid_rows & (era_baseline_playoff_sum > 0),
            era_baseline_seed_weighted_sum
            / np.where(era_baseline_playoff_sum > 0, era_baseline_playoff_sum, 1),
            np.nan,
        )
        output.loc[
            labels, "era_local_budget_fixed_bracket_actual_title_probability"
        ] = np.where(era_valid_rows, era_fixed_actual_mean, np.nan)
        output.loc[
            labels, "era_local_budget_fixed_bracket_baseline_title_probability"
        ] = np.where(era_valid_rows, era_fixed_baseline_mean, np.nan)
        output.loc[labels, "era_local_budget_fixed_bracket_title_lift"] = np.where(
            era_valid_rows, era_fixed_lift, np.nan
        )
        output.loc[labels, "era_local_budget_seeding_title_lift"] = np.where(
            era_valid_rows, era_full_lift - era_fixed_lift, np.nan
        )
        output.loc[
            labels, "era_local_budget_mean_signed_salary_gap_cap_points"
        ] = np.where(era_valid_rows, era_signed_gap_sum / era_denominator, np.nan)
        output.loc[labels, "era_local_budget_median_salary_gap"] = np.where(
            era_valid_rows, era_median_gap, np.nan
        )
        output.loc[labels, "era_local_budget_p90_salary_gap"] = np.where(
            era_valid_rows, era_p90_gap, np.nan
        )
        output.loc[labels, "era_local_budget_share_salary_gap_over_5"] = np.where(
            era_valid_rows, era_share_gap_over_5, np.nan
        )
        output.loc[labels, "era_local_budget_mean_regular_season_debit"] = np.where(
            era_valid_rows, era_regular_debit_sum / era_denominator, np.nan
        )
        output.loc[labels, "era_local_budget_mean_playoff_debit"] = np.where(
            era_valid_rows, era_playoff_debit_sum / era_denominator, np.nan
        )
        output.loc[labels, "title_lift_over_contract_v2"] = np.where(
            era_valid_rows, era_full_lift, np.nan
        )
        output.loc[labels, "contract_hurdle_version_v2"] = np.where(
            era_valid_rows, ERA_LOCAL_SALARY_PEER_HURDLE_VERSION, None
        )
        adjusted_actual_mean = adjusted_actual_sum / era_denominator
        adjusted_baseline_mean = adjusted_baseline_sum / era_denominator
        adjusted_fixed_actual_mean = adjusted_fixed_actual_sum / era_denominator
        adjusted_fixed_baseline_mean = adjusted_fixed_baseline_sum / era_denominator
        adjusted_full_lift = adjusted_actual_mean - adjusted_baseline_mean
        adjusted_fixed_lift = (
            adjusted_fixed_actual_mean - adjusted_fixed_baseline_mean
        )
        output.loc[labels, "playoff_adjusted_actual_title_probability"] = np.where(
            era_valid_rows, adjusted_actual_mean, np.nan
        )
        output.loc[labels, "playoff_adjusted_baseline_title_probability"] = np.where(
            era_valid_rows, adjusted_baseline_mean, np.nan
        )
        output.loc[
            labels, "playoff_adjusted_fixed_bracket_actual_title_probability"
        ] = np.where(era_valid_rows, adjusted_fixed_actual_mean, np.nan)
        output.loc[
            labels, "playoff_adjusted_fixed_bracket_baseline_title_probability"
        ] = np.where(era_valid_rows, adjusted_fixed_baseline_mean, np.nan)
        output.loc[labels, "playoff_adjusted_fixed_bracket_title_lift"] = np.where(
            era_valid_rows, adjusted_fixed_lift, np.nan
        )
        output.loc[labels, "playoff_adjusted_seeding_title_lift"] = np.where(
            era_valid_rows, adjusted_full_lift - adjusted_fixed_lift, np.nan
        )
        output.loc[
            labels, "playoff_adjusted_title_lift_over_contract_v2"
        ] = np.where(era_valid_rows, adjusted_full_lift, np.nan)
        output.loc[labels, "playoff_adjusted_contract_hurdle_version"] = np.where(
            era_valid_rows, PLAYOFF_ADJUSTED_SALARY_PEER_HURDLE_VERSION, None
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
