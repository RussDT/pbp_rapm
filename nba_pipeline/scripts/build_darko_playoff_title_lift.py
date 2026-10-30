#!/usr/bin/env python3
"""Build a lagged-DARKO playoff-prior Title Lift over Contract challenger.

Public historical DARKO is one completed-season snapshot rather than a daily
archive.  Target season ``t`` therefore uses DARKO season ``t - 1`` uniformly;
the same-season row is never eligible for the entering-postseason prior.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .build_championship_help import (
        attach_neutral_playoff_baseline,
        attach_playoff_adjusted_ratings,
        attach_salary_share,
        build_player_title_context,
        build_regular_season_environment,
        build_salary_peer_focal_seasons,
        build_team_season_data,
        load_salary_history,
        select_epm_snapshot,
    )
    from .playoff_epm_prior import (
        build_walk_forward_playoff_epm_ratings,
        evaluate_walk_forward_playoff_ratings,
        load_playoff_possessions,
    )
    from .playoff_title_simulation import (
        add_salary_peer_title_lift,
        build_bracket_nodes,
        build_playoff_seed_fields,
        build_playoff_series,
        prepare_playoff_game_model_rows,
        regular_season_seed_probabilities,
        select_playoff_game_model,
    )
except ImportError:  # pragma: no cover - direct script execution
    from build_championship_help import (
        attach_neutral_playoff_baseline,
        attach_playoff_adjusted_ratings,
        attach_salary_share,
        build_player_title_context,
        build_regular_season_environment,
        build_salary_peer_focal_seasons,
        build_team_season_data,
        load_salary_history,
        select_epm_snapshot,
    )
    from playoff_epm_prior import (
        build_walk_forward_playoff_epm_ratings,
        evaluate_walk_forward_playoff_ratings,
        load_playoff_possessions,
    )
    from playoff_title_simulation import (
        add_salary_peer_title_lift,
        build_bracket_nodes,
        build_playoff_seed_fields,
        build_playoff_series,
        prepare_playoff_game_model_rows,
        regular_season_seed_probabilities,
        select_playoff_game_model,
    )


REQUIRED_DARKO_COLUMNS = {
    "nba_id",
    "player_name",
    "season",
    "date",
    "dpm",
    "o_dpm",
    "d_dpm",
}

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "nba_pipeline"
DEFAULT_DARKO_PATH = PIPELINE_ROOT / ".cache" / "teammate_help" / "dpm_history.csv"
DEFAULT_OUTPUT_DIR = PIPELINE_ROOT / "results" / "darko_playoff_title_lift"
DEFAULT_PROCESSED_DIR = PIPELINE_ROOT / "processed"
DEFAULT_RAW_DIR = PIPELINE_ROOT / "raw_data"
DEFAULT_SALARY_PATH = (
    REPO_ROOT.parent.parent / "pipelines" / "core" / "nba_rapm" / "data" / "inputs" / "nba_salaries.csv"
)
DEFAULT_2026_SALARY_ARCHIVE = (
    REPO_ROOT.parent.parent
    / "products"
    / "nbarapm-next"
    / "lib"
    / "control-runway"
    / "previous-season-snapshot.ts"
)
DEFAULT_2026_PLAYER_SHEETS = (
    REPO_ROOT.parent.parent
    / "products"
    / "nbarapm-legacy"
    / "data"
    / "player_sheets"
    / "modern_selected_deduped.csv"
)
DARKO_PLAYOFF_ALPHA_OFFENSE = 24000.0
DARKO_PLAYOFF_ALPHA_DEFENSE = 24000.0
DARKO_PLAYOFF_HISTORY_WINDOW = 8


def build_lagged_darko_snapshot_history(
    darko: pd.DataFrame,
    *,
    start_season: int,
    end_season: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Convert completed-season DARKO into synthetic preseason snapshots.

    Each synthetic snapshot is dated July 1 before its target season.  This
    lets the shared championship builder select it before that postseason while
    preserving the strict one-season lag.
    """
    missing = REQUIRED_DARKO_COLUMNS.difference(darko.columns)
    if missing:
        raise ValueError(f"DARKO history is missing columns: {sorted(missing)}")
    if int(end_season) < int(start_season):
        raise ValueError("end_season must be at least start_season")

    frame = darko[list(REQUIRED_DARKO_COLUMNS)].copy()
    frame["season"] = pd.to_numeric(frame["season"], errors="coerce")
    frame["nba_id"] = pd.to_numeric(frame["nba_id"], errors="coerce")
    for column in ("dpm", "o_dpm", "d_dpm"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame = frame.dropna(
        subset=["season", "nba_id", "player_name", "dpm", "o_dpm", "d_dpm"]
    ).copy()
    frame["season"] = frame["season"].astype(int)
    frame["nba_id"] = frame["nba_id"].astype(int)
    frame = frame.sort_values(
        ["season", "nba_id", "date"], kind="mergesort"
    ).drop_duplicates(["season", "nba_id"], keep="last")

    parts: list[pd.DataFrame] = []
    for target_season in range(int(start_season), int(end_season) + 1):
        source_season = target_season - 1
        snapshot = frame.loc[frame["season"].eq(source_season)].copy()
        if snapshot.empty:
            continue
        requested_date = pd.Timestamp(year=target_season - 1, month=7, day=1)
        snapshot["source_darko_season"] = source_season
        snapshot["snapshot_requested_date"] = requested_date
        snapshot["provider_effective_date"] = snapshot["date"]
        snapshot["game_dt"] = snapshot["date"]
        parts.append(snapshot)

    columns = [
        "nba_id",
        "player_name",
        "source_darko_season",
        "snapshot_requested_date",
        "provider_effective_date",
        "date",
        "game_dt",
        "dpm",
        "o_dpm",
        "d_dpm",
    ]
    history = (
        pd.concat(parts, ignore_index=True)[columns]
        if parts
        else pd.DataFrame(columns=columns)
    )
    report: dict[str, object] = {
        "source": "public historical DARKO completed-season snapshots",
        "lag_seasons": 1,
        "timing": "target season t uses only completed DARKO season t-1",
        "start_target_season": int(start_season),
        "end_target_season": int(end_season),
        "snapshots": int(history["snapshot_requested_date"].nunique()),
        "rows": int(len(history)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return history, report


def load_lagged_darko_snapshot_history(
    path: Path,
    *,
    start_season: int,
    end_season: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"DARKO history not found: {path}")
    return build_lagged_darko_snapshot_history(
        pd.read_csv(path, low_memory=False),
        start_season=start_season,
        end_season=end_season,
    )


def _normalized_player_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def load_2026_salary_archive(
    path: Path,
    darko_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Map the frozen 2025-26 contract archive onto NBA IDs."""
    if not path.exists():
        raise FileNotFoundError(f"Frozen 2025-26 salary archive not found: {path}")
    match = re.search(
        r"export const PREVIOUS_SEASON_CONTRACTS[^=]*=\s*(\[.*?\n\]);",
        path.read_text(),
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError("Could not locate PREVIOUS_SEASON_CONTRACTS in salary archive")
    contracts = pd.DataFrame(json.loads(match.group(1)))
    darko = pd.read_csv(
        darko_path,
        usecols=["nba_id", "player_name", "season"],
        low_memory=False,
    )
    aliases = {
        "nic claxton": "nicolas claxton",
        "robert williams": "robert williams iii",
        "alex sarr": "alexandre sarr",
        "ron holland": "ronald holland ii",
        "xavier tillman sr": "xavier tillman",
        "bones hyland": "nah shon hyland",
        "gg jackson ii": "gg jackson",
        "craig porter jr": "craig porter",
        "didi louzada": "marcos louzada silva",
    }
    darko["name_key"] = darko["player_name"].map(_normalized_player_name)
    id_counts = darko.groupby("name_key")["nba_id"].nunique()
    unique_keys = set(id_counts[id_counts.eq(1)].index)
    name_to_id = (
        darko.loc[darko["name_key"].isin(unique_keys)]
        .sort_values(["name_key", "season"], kind="mergesort")
        .drop_duplicates("name_key", keep="last")
        .set_index("name_key")["nba_id"]
        .astype(int)
        .to_dict()
    )
    contracts["name_key"] = contracts["playerName"].map(_normalized_player_name)
    contracts["name_key"] = contracts["name_key"].replace(aliases)
    contracts["player_id"] = contracts["name_key"].map(name_to_id)
    unmatched = contracts.loc[contracts["player_id"].isna(), "playerName"].tolist()
    mapped = contracts.dropna(subset=["player_id", "salary", "team"]).copy()
    mapped["season"] = 2026
    mapped["player_id"] = mapped["player_id"].astype(int)
    mapped["salary_team_abbr"] = mapped["team"].replace(
        {"BRK": "BKN", "CHO": "CHA", "PHO": "PHX"}
    )
    mapped["salary"] = pd.to_numeric(mapped["salary"], errors="raise")
    mapped["salary_cap"] = 154_647_000
    mapped = (
        mapped.groupby(
            ["season", "player_id", "salary_team_abbr", "salary_cap"],
            as_index=False,
        )["salary"]
        .sum()
    )
    mapped["salary_share"] = mapped["salary"] / mapped["salary_cap"]
    report: dict[str, object] = {
        "source_file": str(path),
        "season": 2026,
        "salary_cap": 154_647_000,
        "contract_rows": int(len(contracts)),
        "mapped_rows": int(len(mapped)),
        "unmatched_names": unmatched,
        "source_note": "frozen post-Finals 2025-26 contract archive",
    }
    return mapped, report


def load_player_sheets_2026_salary(
    path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load canonical 2026 cap share from the pinned player_sheets extract."""
    if not path.exists():
        raise FileNotFoundError(f"player_sheets salary extract not found: {path}")
    required = {
        "nba_id", "PLAYER_NAME", "year", "salary", "SALARY_CAP", "SALARY_PCT_CAP"
    }
    raw = pd.read_csv(path, usecols=list(required), low_memory=False)
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"player_sheets extract is missing columns: {sorted(missing)}")
    frame = raw.loc[pd.to_numeric(raw["year"], errors="coerce").eq(2026)].copy()
    for column in ("nba_id", "salary", "SALARY_CAP", "SALARY_PCT_CAP"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["nba_id", "salary", "SALARY_CAP"])
    frame = frame.loc[frame["SALARY_CAP"].gt(0) & frame["salary"].ge(0)].copy()
    native = frame["SALARY_PCT_CAP"].notna()
    frame["salary_share"] = frame["SALARY_PCT_CAP"].where(
        native, frame["salary"] / frame["SALARY_CAP"]
    )
    frame["season"] = 2026
    frame["player_id"] = frame["nba_id"].astype(int)
    frame["salary_cap"] = frame["SALARY_CAP"]
    frame["salary_team_abbr"] = ""
    if frame.duplicated(["season", "player_id"]).any():
        raise ValueError("player_sheets 2026 salary rows are not unique by player")
    output = frame[
        [
            "season", "player_id", "salary_team_abbr", "salary",
            "salary_cap", "salary_share",
        ]
    ].reset_index(drop=True)
    report: dict[str, object] = {
        "source_file": str(path),
        "source": "gabriel1200/player_sheets year_totals/modern.csv",
        "pinned_commit": "cae13ad",
        "downloaded": "2026-07-09",
        "season": 2026,
        "rows": int(len(output)),
        "native_cap_share_rows": int(native.sum()),
        "fallback_cap_share_rows": int((~native).sum()),
        "rule": "SALARY_PCT_CAP, falling back to salary / SALARY_CAP",
    }
    return output, report


def _build_career_rows(seasons: pd.DataFrame) -> pd.DataFrame:
    metric_map = {
        "darko_title_lift_over_contract": "darko_title_lift_over_contract_championships",
        "darko_fixed_bracket_title_lift": "darko_fixed_bracket_title_lift_championships",
        "darko_seeding_title_lift": "darko_seeding_title_lift_championships",
        "darko_actual_title_probability": "darko_actual_championships",
        "darko_baseline_title_probability": "darko_baseline_championships",
    }
    covered = seasons.loc[seasons["darko_title_lift_over_contract"].notna()].copy()
    if covered.empty:
        return pd.DataFrame(
            columns=["player_id", "player_name", "darko_contract_seasons_covered", *metric_map.values()]
        )
    aggregations: dict[str, tuple[str, str]] = {
        "player_name": ("player_name", "last"),
        "darko_contract_seasons_covered": ("season", "size"),
        "darko_start_season": ("season", "min"),
        "darko_end_season": ("season", "max"),
    }
    aggregations.update(
        {target: (source, "sum") for source, target in metric_map.items()}
    )
    return (
        covered.sort_values(["player_id", "season"], kind="mergesort")
        .groupby("player_id", as_index=False)
        .agg(**aggregations)
    )


def build_playoff_series_from_team_pairs(games: pd.DataFrame) -> pd.DataFrame:
    """Recover rounds when legacy game IDs do not encode the bracket."""
    frame = games.copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="raise")
    frame["team_a_id"] = frame[["home_team_id", "away_team_id"]].min(axis=1).astype(int)
    frame["team_b_id"] = frame[["home_team_id", "away_team_id"]].max(axis=1).astype(int)
    frame["winner_team_id"] = np.where(
        frame["home_score"].astype(float) > frame["away_score"].astype(float),
        frame["home_team_id"],
        frame["away_team_id"],
    ).astype(int)
    records: list[dict[str, object]] = []
    for (season, team_a, team_b), group in frame.groupby(
        ["season", "team_a_id", "team_b_id"], sort=True
    ):
        wins = group["winner_team_id"].value_counts()
        if wins.empty or (len(wins) > 1 and int(wins.iloc[0]) == int(wins.iloc[1])):
            raise ValueError(f"Legacy series {season}-{team_a}-{team_b} has no winner")
        records.append(
            {
                "season": int(season),
                "team_a_id": int(team_a),
                "team_b_id": int(team_b),
                "team_a_wins": int(wins.get(int(team_a), 0)),
                "team_b_wins": int(wins.get(int(team_b), 0)),
                "winner_team_id": int(wins.index[0]),
                "games": int(len(group)),
                "first_game_date": group["game_date"].min(),
                "last_game_date": group["game_date"].max(),
            }
        )
    series = pd.DataFrame(records)
    output: list[pd.DataFrame] = []
    for season, group in series.groupby("season", sort=True):
        group = group.sort_values("first_game_date", kind="mergesort").reset_index(drop=True)
        if len(group) != 15:
            raise ValueError(f"Season {season} has {len(group)} legacy series, expected 15")
        next_index: dict[int, int | None] = {}
        for index, row in group.iterrows():
            later = group.loc[
                (group.index > index)
                & (
                    group["team_a_id"].eq(int(row["winner_team_id"]))
                    | group["team_b_id"].eq(int(row["winner_team_id"]))
                )
            ]
            next_index[index] = int(later.index[0]) if not later.empty else None
        finals = [index for index, successor in next_index.items() if successor is None]
        if len(finals) != 1:
            raise ValueError(f"Season {season} has {len(finals)} terminal legacy series")
        rounds: dict[int, int] = {finals[0]: 4}

        def assign_round(index: int) -> int:
            if index in rounds:
                return rounds[index]
            successor = next_index[index]
            if successor is None:
                raise ValueError("Only the final series may be terminal")
            rounds[index] = assign_round(successor) - 1
            return rounds[index]

        for index in group.index:
            assign_round(int(index))
        group["round"] = [rounds[int(index)] for index in group.index]
        if not group["round"].between(1, 4).all():
            raise ValueError(f"Season {season} legacy bracket did not resolve to four rounds")
        group["series_id"] = [
            f"{int(season)}-{int(row['round'])}-{index}"
            for index, row in group.iterrows()
        ]
        output.append(group)
    return pd.concat(output, ignore_index=True)[
        [
            "season", "round", "series_id", "team_a_id", "team_b_id",
            "team_a_wins", "team_b_wins", "winner_team_id", "games",
        ]
    ]
def build_darko_title_lift(
    *,
    darko_path: Path,
    salary_path: Path,
    player_sheets_2026_path: Path | None,
    processed_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    start_season: int,
    end_season: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build the standalone lagged-DARKO playoff-prior TLOC surface."""
    history, history_report = load_lagged_darko_snapshot_history(
        darko_path, start_season=start_season, end_season=end_season
    )
    salary_history, salary_report = load_salary_history(salary_path)
    salary_2026_report = None
    if end_season >= 2026:
        if player_sheets_2026_path is None:
            raise ValueError("Season 2026 requires the pinned player_sheets salary extract")
        salary_2026, salary_2026_report = load_player_sheets_2026_salary(
            player_sheets_2026_path
        )
        salary_history = pd.concat(
            [salary_history.loc[salary_history["season"].ne(2026)], salary_2026],
            ignore_index=True,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    structural_names = {
        "teams": "team_seasons.csv",
        "roster": "team_player_contributions.csv",
        "team_pair_exposures": "team_player_pair_exposures.csv",
        "playoff_games": "playoff_games.csv",
        "regular_teams": "regular_season_teams.csv",
        "regular_roster": "regular_season_roster.csv",
        "regular_games": "regular_season_games.csv",
    }
    structural_paths = {name: output_dir / filename for name, filename in structural_names.items()}
    cache_covers_range = False
    if all(path.exists() for path in structural_paths.values()):
        cached_seasons = pd.read_csv(
            structural_paths["teams"], usecols=["season"]
        )["season"]
        cache_covers_range = (
            int(cached_seasons.min()) <= int(start_season)
            and int(cached_seasons.max()) >= int(end_season)
        )
    if cache_covers_range:
        loaded = {name: pd.read_csv(path) for name, path in structural_paths.items()}
        teams = loaded["teams"]
        roster = loaded["roster"]
        team_pair_exposures = loaded["team_pair_exposures"]
        playoff_games = loaded["playoff_games"]
        regular_teams = loaded["regular_teams"]
        regular_roster = loaded["regular_roster"]
        regular_games = loaded["regular_games"]
        playoff_reports = []
        regular_reports = []
    else:
        (
            teams,
            roster,
            team_pair_exposures,
            _champions,
            playoff_games,
            playoff_reports,
        ) = build_team_season_data(
            start_season, end_season, processed_dir, raw_dir, history
        )
        postseason_starts = (
            playoff_games.assign(game_date=pd.to_datetime(playoff_games["game_date"]))
            .groupby("season", as_index=False)["game_date"]
            .min()
        )
        regular_teams, regular_roster, regular_games, regular_reports = (
            build_regular_season_environment(
                start_season,
                end_season,
                processed_dir,
                raw_dir,
                history,
                postseason_starts,
            )
        )
    postseason_starts = (
        playoff_games.assign(game_date=pd.to_datetime(playoff_games["game_date"]))
        .groupby("season", as_index=False)["game_date"]
        .min()
    )
    if end_season >= 2026:
        final_teams = regular_roster.loc[regular_roster["season"].eq(2026)].copy()
        final_teams["last_game_date"] = pd.to_datetime(
            final_teams["last_game_date"], errors="coerce"
        )
        final_teams = (
            final_teams.sort_values(
                ["player_id", "last_game_date", "player_exposures"],
                kind="mergesort",
            )
            .drop_duplicates("player_id", keep="last")
            .set_index("player_id")["team_abbr"]
        )
        mask_2026 = salary_history["season"].eq(2026)
        salary_history.loc[mask_2026, "salary_team_abbr"] = salary_history.loc[
            mask_2026, "player_id"
        ].map(final_teams)
        salary_history = salary_history.loc[
            ~mask_2026 | salary_history["salary_team_abbr"].notna()
        ].copy()
    teams.to_csv(output_dir / "team_seasons.csv", index=False)
    roster.to_csv(output_dir / "team_player_contributions.csv", index=False)
    team_pair_exposures.to_csv(output_dir / "team_player_pair_exposures.csv", index=False)
    playoff_games.to_csv(output_dir / "playoff_games.csv", index=False)
    regular_teams.to_csv(output_dir / "regular_season_teams.csv", index=False)
    regular_roster.to_csv(output_dir / "regular_season_roster.csv", index=False)
    regular_games.to_csv(output_dir / "regular_season_games.csv", index=False)
    player, _ = build_player_title_context(teams, roster, team_pair_exposures)
    player["playoff_workload_share"] = (
        pd.to_numeric(player["total_exposures"], errors="coerce")
        / pd.to_numeric(player["team_total_exposures"], errors="coerce").replace(0, np.nan)
    )
    player, playoff_salary_report = attach_salary_share(player, salary_history)
    sample = teams.sort_values(["season", "team_id"], kind="mergesort").reset_index(drop=True)

    rating_snapshots = pd.concat(
        [
            select_epm_snapshot(history, row.game_date, season=int(row.season))
            for row in postseason_starts.itertuples(index=False)
        ],
        ignore_index=True,
    )
    priors = rating_snapshots[
        ["season", "nba_id", "o_dpm", "d_dpm", "dpm"]
    ].rename(columns={"nba_id": "player_id"})
    playoff_possessions = load_playoff_possessions(
        processed_dir, start_season, end_season
    )
    adjusted, prior_report = build_walk_forward_playoff_epm_ratings(
        playoff_possessions,
        priors,
        alpha_offense=DARKO_PLAYOFF_ALPHA_OFFENSE,
        alpha_defense=DARKO_PLAYOFF_ALPHA_DEFENSE,
        history_window=DARKO_PLAYOFF_HISTORY_WINDOW,
    )
    gate_start = max(start_season + DARKO_PLAYOFF_HISTORY_WINDOW, 2006)
    gate_report = (
        evaluate_walk_forward_playoff_ratings(
            playoff_possessions,
            priors,
            adjusted,
            start_season=gate_start,
            end_season=end_season,
        )
        if gate_start <= end_season
        else None
    )
    if gate_report is not None:
        gate_report["mse_delta_vs_lagged_darko"] = gate_report.pop(
            "mse_delta_vs_preplayoff_epm"
        )
        gate_report["interpretation"] = (
            "Negative favors the playoff-adjusted rating over lagged DARKO; "
            "the clustered interval crossing zero means the update remains a research sensitivity."
        )
    prior_report["method"] = (
        "Zero-centered O/D playoff residual ridge added to previous-season DARKO"
    )
    prior_report["timing"] = (
        "Target season t uses completed DARKO season t-1 and playoff possessions only from seasons before t"
    )

    seasons, universe_report = build_salary_peer_focal_seasons(
        player, rating_snapshots, salary_history
    )
    regular_roles = regular_roster[
        [
            "season",
            "team_id",
            "player_id",
            "regular_season_workload_share",
            "regular_season_contribution_dpm",
        ]
    ]
    seasons = seasons.merge(
        regular_roles,
        on=["season", "team_id", "player_id"],
        how="left",
        validate="many_to_one",
    )
    seasons, adjusted_teams = attach_playoff_adjusted_ratings(
        seasons, sample, adjusted
    )
    adjusted_team_model = adjusted_teams.copy()
    adjusted_team_model["team_dpm"] = adjusted_team_model[
        "playoff_adjusted_team_dpm"
    ]
    playoff_rows = prepare_playoff_game_model_rows(
        playoff_games, adjusted_team_model
    )
    playoff_model, playoff_cv, playoff_model_report = select_playoff_game_model(playoff_rows)
    try:
        playoff_series = build_playoff_series(playoff_games)
    except ValueError as exc:
        if "invalid round identifier" not in str(exc):
            raise
        playoff_series = build_playoff_series_from_team_pairs(playoff_games)
    bracket_nodes = build_bracket_nodes(playoff_series)
    regular_rows = prepare_playoff_game_model_rows(regular_games, regular_teams)
    regular_model, regular_cv, regular_model_report = select_playoff_game_model(regular_rows)
    seed_probabilities = regular_season_seed_probabilities(
        regular_games, regular_teams, regular_model
    )
    playoff_seed_fields = build_playoff_seed_fields(
        regular_games, adjusted_team_model
    )
    modeled = add_salary_peer_title_lift(
        seasons,
        seasons.loc[seasons["is_playoff_participant"]],
        adjusted_teams,
        bracket_nodes,
        playoff_model,
        regular_seed_probabilities=seed_probabilities,
        playoff_seed_fields=playoff_seed_fields,
    )
    rename = {
        "own_dpm": "darko_lagged_dpm",
        "playoff_adjusted_dpm": "darko_playoff_adjusted_dpm",
        "playoff_translation_o_dpm": "darko_playoff_translation_o_dpm",
        "playoff_translation_d_dpm": "darko_playoff_translation_d_dpm",
        "playoff_translation_dpm": "darko_playoff_translation_dpm",
        "playoff_prior_seasons": "darko_playoff_prior_seasons",
        "playoff_prior_possessions": "darko_playoff_prior_possessions",
        "playoff_adjusted_actual_title_probability": "darko_actual_title_probability",
        "playoff_adjusted_baseline_title_probability": "darko_baseline_title_probability",
        "playoff_adjusted_fixed_bracket_actual_title_probability": "darko_fixed_bracket_actual_title_probability",
        "playoff_adjusted_fixed_bracket_baseline_title_probability": "darko_fixed_bracket_baseline_title_probability",
        "playoff_adjusted_fixed_bracket_title_lift": "darko_fixed_bracket_title_lift",
        "playoff_adjusted_seeding_title_lift": "darko_seeding_title_lift",
        "playoff_adjusted_title_lift_over_contract_v2": "darko_title_lift_over_contract",
    }
    identity = [
        "season", "player_id", "player_name", "team_id", "team_abbr",
        "is_playoff_participant", "salary", "salary_cap", "salary_share",
        "era_local_salary_peer_placement_count",
        "era_local_budget_mean_signed_salary_gap_cap_points",
    ]
    season_output = modeled[identity + list(rename)].rename(columns=rename)
    career_output = _build_career_rows(season_output)
    report: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label": "Lagged DARKO + playoff prior TLOC",
        "rating_prior": history_report,
        "playoff_prior": prior_report,
        "forward_playoff_gate": gate_report,
        "playoff_game_model": playoff_model_report,
        "regular_season_game_model": regular_model_report,
        "salary_source": salary_report,
        "salary_2026_source": salary_2026_report,
        "salary_coverage": universe_report,
        "playoff_salary_coverage": playoff_salary_report,
        "playoff_seasons": playoff_reports,
        "regular_seasons": regular_reports,
        "interpretation": (
            "Target season t starts from completed-season DARKO t-1, adds only playoff "
            "residual evidence from seasons before t, and prices the result through the same "
            "era-local salary-peer seeding and four-series simulation as EPM TLOC."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    season_output.to_csv(output_dir / "player_season_estimates.csv", index=False)
    career_output.to_csv(output_dir / "career_player_estimates.csv", index=False)
    adjusted.to_csv(output_dir / "darko_playoff_prior_ratings.csv", index=False)
    playoff_cv.to_csv(output_dir / "playoff_game_model_cv.csv", index=False)
    regular_cv.to_csv(output_dir / "regular_season_game_model_cv.csv", index=False)
    (output_dir / "metadata.json").write_text(json.dumps(report, indent=2))
    return season_output, career_output, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--darko-path", type=Path, default=DEFAULT_DARKO_PATH)
    parser.add_argument("--salary-path", type=Path, default=DEFAULT_SALARY_PATH)
    parser.add_argument(
        "--player-sheets-2026", type=Path, default=DEFAULT_2026_PLAYER_SHEETS
    )
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-season", type=int, default=1998)
    parser.add_argument("--end-season", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seasons, careers, report = build_darko_title_lift(
        darko_path=args.darko_path,
        salary_path=args.salary_path,
        player_sheets_2026_path=args.player_sheets_2026,
        processed_dir=args.processed_dir,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        start_season=args.start_season,
        end_season=args.end_season,
    )
    print(
        f"Wrote {len(seasons):,} DARKO player-seasons and {len(careers):,} careers "
        f"to {args.output_dir}"
    )
    if report["forward_playoff_gate"] is not None:
        print(f"Forward gate: {report['forward_playoff_gate']}")


if __name__ == "__main__":
    main()
