#!/usr/bin/env python3
"""Calibrate EPM-to-game/season value and build a current-player table.

The build keeps the additive accounting linear:

    game margin = EPM * actual player possessions per game / 100
    season wins = EPM * actual player possessions / 100 * wins per point

Only the expected-margin to win-probability display is nonlinear.  Historical
regular-season games calibrate that display, while the current player table
uses a date-stamped Dunks & Threes EPM snapshot and exact repo-owned PBP
exposure.  The replacement baseline remains an explicit sensitivity parameter;
it is not identified by this calibration.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, ndtr


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
DEFAULT_PROCESSED_DIR = PIPELINE_DIR / "processed"
DEFAULT_RAW_DIR = PIPELINE_DIR / "raw_data"
DEFAULT_EPM_SNAPSHOT = PIPELINE_DIR / "cache" / "epm" / "epm_all_2026-04-15_go0.csv"
DEFAULT_OUTPUT_DIR = PIPELINE_DIR / "results" / "epm_game_value"
LINEUP_COLS = [
    *[f"home_player{i}" for i in range(1, 6)],
    *[f"away_player{i}" for i in range(1, 6)],
]
OFF_COLS = [f"O{i}" for i in range(1, 6)]
DEF_COLS = [f"D{i}" for i in range(1, 6)]


@dataclass(frozen=True)
class WinCurve:
    """Historical calibration from expected neutral margin to win probability."""

    margin_intercept: float
    strength_coefficient: float
    logistic_k: float
    probit_sigma: float
    linear_slope: float

    def logistic(self, margin: np.ndarray | float) -> np.ndarray:
        return expit(self.logistic_k * np.asarray(margin, dtype=float))

    def probit(self, margin: np.ndarray | float) -> np.ndarray:
        return ndtr(np.asarray(margin, dtype=float) / self.probit_sigma)

    def linear(self, margin: np.ndarray | float) -> np.ndarray:
        values = 0.5 + self.linear_slope * np.asarray(margin, dtype=float)
        return np.clip(values, 0.0, 1.0)


def _mode(series: pd.Series) -> str | None:
    values = series.dropna().astype(str)
    if values.empty:
        return None
    return str(values.mode().iloc[0])


def load_regular_season_games(
    processed_dir: Path,
    start_season: int,
    end_season: int,
) -> pd.DataFrame:
    """Reconstruct one score/team row per regular-season game."""
    records: list[dict[str, object]] = []
    columns = [
        "game_id",
        "season_type",
        "poss_offense",
        "off_team_abbr",
        "home_points",
        "away_points",
    ]
    for season in range(start_season, end_season + 1):
        path = processed_dir / f"LINEUP_NET_RATING_POSSESSIONS{season}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path, columns=columns)
        frame = frame[frame["season_type"].astype(str).str.upper().eq("RS")]
        for game_id, game in frame.groupby("game_id", sort=False):
            home_team = _mode(
                game.loc[game["poss_offense"].astype(str).eq("Home"), "off_team_abbr"]
            )
            away_team = _mode(
                game.loc[game["poss_offense"].astype(str).eq("Away"), "off_team_abbr"]
            )
            home_points = float(pd.to_numeric(game["home_points"], errors="coerce").sum())
            away_points = float(pd.to_numeric(game["away_points"], errors="coerce").sum())
            if home_team is None or away_team is None or home_points == away_points:
                continue
            records.append(
                {
                    "season": season,
                    "game_id": str(game_id),
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_points": home_points,
                    "away_points": away_points,
                    "margin": home_points - away_points,
                    "home_win": float(home_points > away_points),
                }
            )
    games = pd.DataFrame.from_records(records)
    if games.empty:
        raise ValueError("No regular-season game results were reconstructed")
    return games.sort_values(["season", "game_id"], kind="mergesort").reset_index(drop=True)


def build_team_seasons(games: pd.DataFrame) -> pd.DataFrame:
    """Aggregate game results into team-season wins and point differential."""
    home = games[["season", "game_id", "home_team", "margin", "home_win"]].rename(
        columns={"home_team": "team", "margin": "point_diff", "home_win": "win"}
    )
    away = games[["season", "game_id", "away_team", "margin", "home_win"]].rename(
        columns={"away_team": "team"}
    )
    away["point_diff"] = -away["margin"]
    away["win"] = 1.0 - away["home_win"]
    away = away[["season", "game_id", "team", "point_diff", "win"]]
    rows = pd.concat([home, away], ignore_index=True)
    return (
        rows.groupby(["season", "team"], as_index=False)
        .agg(gp=("game_id", "nunique"), wins=("win", "sum"), point_diff=("point_diff", "sum"))
        .sort_values(["season", "team"], kind="mergesort")
        .reset_index(drop=True)
    )


def fit_wins_per_point(team_seasons: pd.DataFrame) -> float:
    """Fit wins - GP/2 = beta * season point differential through the origin."""
    x = team_seasons["point_diff"].to_numpy(dtype=float)
    y = (team_seasons["wins"] - team_seasons["gp"] / 2.0).to_numpy(dtype=float)
    denominator = float(x @ x)
    if denominator <= 0:
        raise ValueError("Point differential has no variance")
    return float((x @ y) / denominator)


def attach_leave_one_game_out_strength(games: pd.DataFrame) -> pd.DataFrame:
    """Attach final-season team quality with the focal game removed."""
    team_seasons = build_team_seasons(games).set_index(["season", "team"])
    output = games.copy()
    home_total = np.array(
        [
            team_seasons.loc[(season, team), "point_diff"]
            for season, team in zip(output["season"], output["home_team"])
        ],
        dtype=float,
    )
    away_total = np.array(
        [
            team_seasons.loc[(season, team), "point_diff"]
            for season, team in zip(output["season"], output["away_team"])
        ],
        dtype=float,
    )
    home_gp = np.array(
        [team_seasons.loc[(s, t), "gp"] for s, t in zip(output["season"], output["home_team"])],
        dtype=float,
    )
    away_gp = np.array(
        [team_seasons.loc[(s, t), "gp"] for s, t in zip(output["season"], output["away_team"])],
        dtype=float,
    )
    margin = output["margin"].to_numpy(dtype=float)
    output["strength_diff"] = (home_total - margin) / (home_gp - 1.0) - (
        away_total + margin
    ) / (away_gp - 1.0)
    return output


def _binary_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    probability = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return float(-np.mean(y * np.log(probability) + (1.0 - y) * np.log(1.0 - probability)))


def fit_win_curve(rows: pd.DataFrame) -> WinCurve:
    """Fit expected margin, then logistic/probit/linear probability mappings."""
    strength = rows["strength_diff"].to_numpy(dtype=float)
    margin = rows["margin"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(rows)), strength])
    margin_intercept, strength_coefficient = np.linalg.lstsq(design, margin, rcond=None)[0]
    expected_margin = margin_intercept + strength_coefficient * strength
    y = rows["home_win"].to_numpy(dtype=float)

    logistic_fit = minimize_scalar(
        lambda k: _binary_log_loss(y, expit(k * expected_margin)),
        bounds=(0.001, 1.0),
        method="bounded",
    )
    probit_fit = minimize_scalar(
        lambda sigma: _binary_log_loss(y, ndtr(expected_margin / sigma)),
        bounds=(1.0, 50.0),
        method="bounded",
    )
    centered = y - 0.5
    linear_slope = float((expected_margin @ centered) / (expected_margin @ expected_margin))
    return WinCurve(
        margin_intercept=float(margin_intercept),
        strength_coefficient=float(strength_coefficient),
        logistic_k=float(logistic_fit.x),
        probit_sigma=float(probit_fit.x),
        linear_slope=linear_slope,
    )


def score_win_curve(rows: pd.DataFrame, curve: WinCurve) -> dict[str, dict[str, float]]:
    expected_margin = (
        curve.margin_intercept
        + curve.strength_coefficient * rows["strength_diff"].to_numpy(dtype=float)
    )
    y = rows["home_win"].to_numpy(dtype=float)
    predictions = {
        "logistic": curve.logistic(expected_margin),
        "probit": curve.probit(expected_margin),
        "linear_clipped": curve.linear(expected_margin),
    }
    return {
        name: {
            "brier": float(np.mean((probability - y) ** 2)),
            "log_loss": _binary_log_loss(y, probability),
        }
        for name, probability in predictions.items()
    }


def reconstruct_player_minutes(raw_path: Path) -> pd.DataFrame:
    """Allocate exact between-event clock intervals to the post-event lineups."""
    columns = ["game_id", "event_num", "period", "time_quarter", *LINEUP_COLS]
    frame = pd.read_parquet(raw_path, columns=columns).drop_duplicates()
    clock_parts = frame["time_quarter"].astype("string").str.extract(r"^(\d+):(\d+(?:\.\d+)?)$")
    frame["clock_sec"] = pd.to_numeric(clock_parts[0], errors="coerce") * 60.0 + pd.to_numeric(
        clock_parts[1], errors="coerce"
    )
    frame["event_num"] = pd.to_numeric(frame["event_num"], errors="coerce")
    frame["period"] = pd.to_numeric(frame["period"], errors="coerce")
    frame = frame.dropna(subset=["game_id", "event_num", "period", "clock_sec"])
    frame = frame.sort_values(
        ["game_id", "period", "clock_sec", "event_num"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    # The last event at a clock contains the lineup after same-clock substitutions.
    intervals = frame.groupby(["game_id", "period", "clock_sec"], sort=False).tail(1).copy()
    next_clock = (
        intervals.groupby(["game_id", "period"], sort=False)["clock_sec"]
        .shift(-1)
        .fillna(0.0)
    )
    intervals["seconds"] = (intervals["clock_sec"] - next_clock).clip(lower=0.0)
    melted = intervals.melt(
        id_vars=["game_id", "seconds"],
        value_vars=LINEUP_COLS,
        value_name="player_id",
    )
    melted["player_id"] = pd.to_numeric(melted["player_id"], errors="coerce")
    melted = melted[melted["player_id"].gt(0) & melted["seconds"].gt(0)].copy()
    melted["player_id"] = melted["player_id"].astype(np.int64)
    return (
        melted.groupby("player_id", as_index=False)
        .agg(minutes=("seconds", lambda x: float(x.sum() / 60.0)), gp=("game_id", "nunique"))
        .sort_values("player_id", kind="mergesort")
        .reset_index(drop=True)
    )


def reconstruct_player_possessions(processed_path: Path) -> pd.DataFrame:
    """Count the average of offensive and defensive possessions on court."""
    columns = ["game_id", "off_poss", "def_poss", *OFF_COLS, *DEF_COLS]
    frame = pd.read_parquet(processed_path, columns=columns)
    pieces: list[pd.DataFrame] = []
    for lineup_columns, weight_column in ((OFF_COLS, "off_poss"), (DEF_COLS, "def_poss")):
        side = frame[["game_id", weight_column, *lineup_columns]].melt(
            id_vars=["game_id", weight_column],
            value_vars=lineup_columns,
            value_name="player_id",
        )
        side["player_id"] = pd.to_numeric(side["player_id"], errors="coerce")
        side["possessions"] = 0.5 * pd.to_numeric(side[weight_column], errors="coerce").fillna(0.0)
        pieces.append(side[["game_id", "player_id", "possessions"]])
    exposure = pd.concat(pieces, ignore_index=True)
    exposure = exposure[exposure["player_id"].gt(0) & exposure["possessions"].gt(0)].copy()
    exposure["player_id"] = exposure["player_id"].astype(np.int64)
    return (
        exposure.groupby("player_id", as_index=False)
        .agg(possessions=("possessions", "sum"), possession_gp=("game_id", "nunique"))
        .sort_values("player_id", kind="mergesort")
        .reset_index(drop=True)
    )


def build_current_player_values(
    epm_snapshot_path: Path,
    raw_path: Path,
    processed_path: Path,
    wins_per_point: float,
    curve: WinCurve,
    replacement_epm: float,
) -> pd.DataFrame:
    """Join current EPM to exact minutes/possessions and derive value columns."""
    epm = pd.read_csv(epm_snapshot_path, low_memory=False)
    epm = epm.rename(columns={"player_id": "nba_id"})
    epm["nba_id"] = pd.to_numeric(epm["nba_id"], errors="coerce")
    epm["epm"] = pd.to_numeric(epm["epm"], errors="coerce")
    epm = epm.dropna(subset=["nba_id", "epm"]).copy()
    epm["nba_id"] = epm["nba_id"].astype(np.int64)
    epm = epm.sort_values(["nba_id", "game_dt"], kind="mergesort").drop_duplicates(
        "nba_id", keep="last"
    )
    minutes = reconstruct_player_minutes(raw_path).rename(columns={"player_id": "nba_id"})
    possessions = reconstruct_player_possessions(processed_path).rename(
        columns={"player_id": "nba_id"}
    )
    players = epm.merge(minutes, on="nba_id", how="inner").merge(
        possessions, on="nba_id", how="inner"
    )
    players["mpg"] = players["minutes"] / players["gp"]
    players["possessions_per_game"] = players["possessions"] / players["gp"]
    players["game_margin"] = players["epm"] * players["possessions_per_game"] / 100.0
    players["net_win_probability"] = curve.logistic(players["game_margin"].to_numpy(dtype=float))
    players["linear_net_win_probability"] = curve.linear(
        players["game_margin"].to_numpy(dtype=float)
    )
    players["season_wins_vs_average"] = (
        players["epm"] * players["possessions"] / 100.0 * wins_per_point
    )
    players["war"] = (
        (players["epm"] - replacement_epm)
        * players["possessions"]
        / 100.0
        * wins_per_point
    )
    players["contender_game_margin_vs_average_alternative"] = players["game_margin"]
    players["contender_win_lift_pp_vs_average_alternative"] = 100.0 * (
        players["net_win_probability"] - 0.5
    )
    keep = [
        "nba_id",
        "player_name",
        "team_id",
        "seasontype",
        "requested_date",
        "date",
        "game_dt",
        "epm",
        "oepm",
        "depm",
        "gp",
        "minutes",
        "mpg",
        "possessions",
        "possessions_per_game",
        "game_margin",
        "net_win_probability",
        "linear_net_win_probability",
        "season_wins_vs_average",
        "war",
        "contender_game_margin_vs_average_alternative",
        "contender_win_lift_pp_vs_average_alternative",
    ]
    return players[[column for column in keep if column in players.columns]].sort_values(
        ["war", "epm"], ascending=[False, False], kind="mergesort"
    ).reset_index(drop=True)


def build_matched_pairs(players: pd.DataFrame, maximum_war_gap: float = 0.10) -> pd.DataFrame:
    """Rank real higher-impact/lower-MPG pairs with nearly equal modeled WAR."""
    eligible = players[
        players["seasontype"].eq(2)
        & players["gp"].ge(30)
        & players["war"].gt(0)
    ].reset_index(drop=True)
    records: list[dict[str, object]] = []
    for left_index in range(len(eligible)):
        for right_index in range(left_index + 1, len(eligible)):
            left = eligible.iloc[left_index]
            right = eligible.iloc[right_index]
            high, low = (left, right) if left["epm"] >= right["epm"] else (right, left)
            war_gap = abs(float(high["war"] - low["war"]))
            epm_gap = float(high["epm"] - low["epm"])
            mpg_gap = float(low["mpg"] - high["mpg"])
            if war_gap > maximum_war_gap or epm_gap < 1.0 or mpg_gap < 3.0:
                continue
            records.append(
                {
                    "higher_impact_player": high["player_name"],
                    "higher_epm": high["epm"],
                    "higher_mpg": high["mpg"],
                    "higher_gp": high["gp"],
                    "higher_war": high["war"],
                    "higher_game_margin": high["game_margin"],
                    "higher_net_win_probability": high["net_win_probability"],
                    "lower_impact_player": low["player_name"],
                    "lower_epm": low["epm"],
                    "lower_mpg": low["mpg"],
                    "lower_gp": low["gp"],
                    "lower_war": low["war"],
                    "lower_game_margin": low["game_margin"],
                    "lower_net_win_probability": low["net_win_probability"],
                    "war_gap": war_gap,
                    "epm_gap": epm_gap,
                    "mpg_gap": mpg_gap,
                    "selection_score": epm_gap + 0.10 * mpg_gap - 5.0 * war_gap,
                }
            )
    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records).sort_values(
        ["selection_score", "war_gap"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)


def bootstrap_intervals(
    team_seasons: pd.DataFrame,
    train_games: pd.DataFrame,
    iterations: int,
    seed: int,
) -> dict[str, list[float]]:
    """Season-cluster bootstrap intervals for the two headline calibrations."""
    rng = np.random.default_rng(seed)
    wpp_values: list[float] = []
    logistic_values: list[float] = []
    team_years = np.sort(team_seasons["season"].unique())
    train_years = np.sort(train_games["season"].unique())
    for _ in range(iterations):
        sampled_team_years = rng.choice(team_years, size=len(team_years), replace=True)
        sampled_teams = pd.concat(
            [team_seasons[team_seasons["season"].eq(year)] for year in sampled_team_years],
            ignore_index=True,
        )
        wpp_values.append(fit_wins_per_point(sampled_teams))
        sampled_train_years = rng.choice(train_years, size=len(train_years), replace=True)
        sampled_games = pd.concat(
            [train_games[train_games["season"].eq(year)] for year in sampled_train_years],
            ignore_index=True,
        )
        logistic_values.append(fit_win_curve(sampled_games).logistic_k)
    return {
        "wins_per_point_95ci": [float(x) for x in np.quantile(wpp_values, [0.025, 0.975])],
        "logistic_k_95ci": [float(x) for x in np.quantile(logistic_values, [0.025, 0.975])],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--epm-snapshot", type=Path, default=DEFAULT_EPM_SNAPSHOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--history-start", type=int, default=14)
    parser.add_argument("--history-end", type=int, default=25)
    parser.add_argument("--train-end", type=int, default=22)
    parser.add_argument("--current-season", type=int, default=26)
    parser.add_argument("--replacement-epm", type=float, default=-2.0)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260815)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    games = load_regular_season_games(args.processed_dir, args.history_start, args.history_end)
    team_seasons = build_team_seasons(games)
    wins_per_point = fit_wins_per_point(team_seasons)
    games = attach_leave_one_game_out_strength(games)
    train = games[games["season"].le(args.train_end)].copy()
    holdout = games[games["season"].gt(args.train_end)].copy()
    curve = fit_win_curve(train)
    holdout_scores = score_win_curve(holdout, curve)
    intervals = bootstrap_intervals(
        team_seasons,
        train,
        iterations=args.bootstrap_iterations,
        seed=args.seed,
    )

    current_processed = (
        args.processed_dir
        / f"LINEUP_NET_RATING_POSSESSIONS{args.current_season}.parquet"
    )
    current_raw = args.raw_dir / f"NBA{args.current_season}.parquet"
    players = build_current_player_values(
        args.epm_snapshot,
        current_raw,
        current_processed,
        wins_per_point,
        curve,
        args.replacement_epm,
    )
    pairs = build_matched_pairs(players)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    players.to_csv(args.output_dir / f"player_values_{args.current_season}.csv", index=False)
    pairs.to_csv(args.output_dir / f"matched_pairs_{args.current_season}.csv", index=False)
    margin_grid = np.arange(-12.0, 12.01, 0.25)
    pd.DataFrame(
        {
            "expected_margin": margin_grid,
            "logistic_probability": curve.logistic(margin_grid),
            "probit_probability": curve.probit(margin_grid),
            "linear_clipped_probability": curve.linear(margin_grid),
        }
    ).to_csv(args.output_dir / "win_curve.csv", index=False)

    minute_total = float(players["minutes"].sum())
    summary = {
        "historical_seasons": [args.history_start, args.history_end],
        "train_seasons": [args.history_start, args.train_end],
        "holdout_seasons": [args.train_end + 1, args.history_end],
        "games": int(len(games)),
        "team_seasons": int(len(team_seasons)),
        "wins_per_point": wins_per_point,
        "points_per_win": 1.0 / wins_per_point,
        "replacement_epm_assumption": args.replacement_epm,
        "win_curve": {
            "margin_intercept": curve.margin_intercept,
            "strength_coefficient": curve.strength_coefficient,
            "logistic_k": curve.logistic_k,
            "logistic_neutral_derivative_pp_per_point": 25.0 * curve.logistic_k,
            "probit_sigma": curve.probit_sigma,
            "linear_slope_probability_per_point": curve.linear_slope,
        },
        "holdout_scores": holdout_scores,
        "bootstrap": intervals,
        "current_snapshot": {
            "path": str(args.epm_snapshot),
            "requested_date": str(players["requested_date"].iloc[0]),
            "provider_effective_date": str(players["date"].iloc[0]),
            "players": int(len(players)),
            "reconstructed_player_minutes": minute_total,
        },
    }
    with (args.output_dir / "calibration_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(json.dumps(summary, indent=2))
    if not pairs.empty:
        display_columns = [
            "higher_impact_player",
            "higher_epm",
            "higher_mpg",
            "higher_war",
            "lower_impact_player",
            "lower_epm",
            "lower_mpg",
            "lower_war",
            "war_gap",
        ]
        print("\nTop matched pairs:\n")
        print(pairs[display_columns].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
