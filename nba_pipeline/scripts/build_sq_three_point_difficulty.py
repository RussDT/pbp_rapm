#!/usr/bin/env python3
"""Build player regular-season 3P difficulty from ShotQuality Shot Taking.

ShotQuality's ``shot_taking.initial_shot_taking.value`` is expected points for
the attempt.  For a three-pointer, dividing by three gives the shooter-neutral
make probability.  Player difficulty is the season's attempt-weighted league
expectation minus the player's expectation, so positive values mean a harder
shot diet and the display identity is exact::

    raw 3P% + difficulty = effective 3P%

The richer ``initial_ev``/anchored prediction families are deliberately not
used because they contain shooter-talent information.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from nba_api.stats.static import players as nba_static_players


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SQ_ROOT = Path(
    "/Users/russ/Documents/Codex/basketball/nba/research/shotquality"
)
DEFAULT_OUTPUT_DIR = PIPELINE_ROOT / "results" / "sq_three_point_difficulty"

SEASONS = {
    4857: {
        "label": "2024-25",
        "regular_season_start_utc": "2024-10-22T00:00:00Z",
    },
    149560: {
        "label": "2025-26",
        "regular_season_start_utc": "2025-10-21T00:00:00Z",
    },
}
SHOT_TAKING_COLUMN = "shot_taking.initial_shot_taking.value"
NBA_RAW_BY_SEASON = {
    "2024-25": PIPELINE_ROOT / "raw_data" / "NBA25.parquet",
    "2025-26": PIPELINE_ROOT / "raw_data" / "NBA26.parquet",
}
TEAM_NAME_TO_ABBR = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "LA Clippers": "LAC",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


def _made_bool(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().isin({"true", "1", "yes", "made"})


def load_regular_season_threes(sq_root: Path) -> pd.DataFrame:
    games = pd.read_csv(sq_root / "games.csv")
    games["game_datetime_utc"] = pd.to_datetime(
        games["game_datetime_utc"], utc=True, errors="coerce"
    )
    games = games[
        games["competition_season_id"].isin(SEASONS)
        & games["game_descriptors.event_type_id"].eq("Regular Season")
    ].copy()
    games["season"] = games["competition_season_id"].map(
        lambda value: SEASONS[int(value)]["label"]
    )
    starts = games["competition_season_id"].map(
        lambda value: pd.Timestamp(SEASONS[int(value)]["regular_season_start_utc"])
    )
    games = games[games["game_datetime_utc"].ge(starts)].copy()

    wanted = [
        "game_id",
        "play_id",
        "team_id",
        "player_id",
        "shooter_name",
        "action_type",
        "made",
        SHOT_TAKING_COLUMN,
    ]
    frames: list[pd.DataFrame] = []
    missing_files: list[int] = []
    for game in games.itertuples(index=False):
        path = sq_root / "games" / f"{int(game.game_id)}.csv"
        if not path.exists():
            missing_files.append(int(game.game_id))
            continue
        header = set(pd.read_csv(path, nrows=0).columns)
        usecols = [column for column in wanted if column in header]
        frame = pd.read_csv(path, usecols=usecols, low_memory=False)
        for column in wanted:
            if column not in frame.columns:
                frame[column] = np.nan
        frame = frame[frame["action_type"].eq("3pt")].copy()
        frame["season"] = game.season
        frames.append(frame)

    if missing_files:
        raise FileNotFoundError(
            f"Missing {len(missing_files)} regular-season game files; first: "
            f"{missing_files[:5]}"
        )
    if not frames:
        raise ValueError("No regular-season three-pointers found")

    shots = pd.concat(frames, ignore_index=True)
    duplicate = shots.duplicated(["game_id", "play_id"], keep=False)
    if duplicate.any():
        examples = shots.loc[duplicate, ["game_id", "play_id"]].head().to_dict("records")
        raise ValueError(f"Duplicate provider play keys found: {examples}")
    shots["made_3"] = _made_bool(shots["made"]).astype(int)
    shots["shot_taking_points"] = pd.to_numeric(
        shots[SHOT_TAKING_COLUMN], errors="coerce"
    )
    shots["expected_make_probability"] = (
        shots["shot_taking_points"] / 3.0
    ).clip(0.0, 1.0)
    shots["player_id"] = pd.to_numeric(shots["player_id"], errors="coerce")
    shots["team_id"] = pd.to_numeric(shots["team_id"], errors="coerce")
    players = pd.read_csv(sq_root / "players.csv", usecols=["player_id", "player_name"])
    players["player_id"] = pd.to_numeric(players["player_id"], errors="coerce")
    player_names = players.dropna(subset=["player_id", "player_name"]).drop_duplicates(
        "player_id", keep=False
    ).set_index("player_id")["player_name"]
    shots["shooter_name"] = shots["shooter_name"].fillna(
        shots["player_id"].map(player_names)
    )
    return shots


def _mode_text(values: pd.Series) -> str:
    clean = values.dropna().astype(str)
    if clean.empty:
        return ""
    counts = clean.value_counts()
    return str(sorted(counts[counts.eq(counts.max())].index)[0])


def _nba_three_totals(path: Path) -> pd.DataFrame:
    columns = [
        "game_id",
        "event_num",
        "event_type",
        "home_description",
        "visitor_description",
        "player1_id",
        "player1_name",
        "player1_team_abbreviation",
    ]
    events = pd.read_parquet(path, columns=columns)
    events = events.drop_duplicates(
        ["game_id", "event_num", "event_type", "player1_id"]
    )
    description = events["home_description"].fillna(
        events["visitor_description"]
    ).fillna("")
    events = events[
        events["event_type"].isin([1, 2])
        & description.str.contains("3PT", case=False, na=False)
        & events["player1_id"].notna()
    ].copy()
    events["three_pm"] = events["event_type"].eq(1).astype(int)
    return (
        events.groupby(
            ["player1_id", "player1_name", "player1_team_abbreviation"],
            dropna=False,
        )
        .agg(three_pa=("three_pm", "size"), three_pm=("three_pm", "sum"))
        .reset_index()
    )


def _select_stat_match(
    *,
    sq_three_pa: int,
    sq_three_pm: int,
    candidates: pd.DataFrame,
) -> pd.Series | None:
    if candidates.empty:
        return None
    scored = candidates.copy()
    scored["pa_error"] = (scored["three_pa"] - sq_three_pa).abs()
    scored["pm_error"] = (scored["three_pm"] - sq_three_pm).abs()
    scored["match_cost"] = (
        scored["pa_error"] / max(sq_three_pa, 10)
        + scored["pm_error"] / max(sq_three_pm, 5)
    )
    scored = scored.sort_values(
        ["match_cost", "pa_error", "pm_error", "player1_id"], kind="stable"
    ).reset_index(drop=True)
    best = scored.iloc[0]
    exact = scored[scored["pa_error"].eq(0) & scored["pm_error"].eq(0)]
    if len(exact) == 1:
        return exact.iloc[0]
    margin = (
        float(scored.iloc[1]["match_cost"] - best["match_cost"])
        if len(scored) > 1
        else float("inf")
    )
    pa_tolerance = max(6, int(np.ceil(0.02 * sq_three_pa)))
    if (
        int(best["pa_error"]) <= pa_tolerance
        and int(best["pm_error"]) <= 3
        and margin >= 0.05
    ):
        return best
    return None


def build_player_identity_bridge(
    shots: pd.DataFrame,
    sq_root: Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    provider_players = pd.read_csv(
        sq_root / "players.csv", usecols=["player_id", "player_name"]
    )
    provider_players["player_id"] = pd.to_numeric(
        provider_players["player_id"], errors="coerce"
    )
    provider_players = provider_players.dropna(
        subset=["player_id", "player_name"]
    ).drop_duplicates("player_id", keep=False)

    provider_map = pd.read_csv(sq_root / "nba_id_to_player_id_mapping.csv")
    provider_map["player_id"] = pd.to_numeric(
        provider_map["player_id"], errors="coerce"
    )
    provider_map["nba_id"] = pd.to_numeric(provider_map["nba_id"], errors="coerce")
    provider_map = provider_map.dropna(subset=["player_id", "nba_id"]).drop_duplicates(
        "player_id", keep=False
    )
    identity = provider_players.merge(provider_map, on="player_id", how="outer")
    identity = identity.rename(columns={"player_id": "sq_player_id"})
    identity["identity_source"] = np.where(
        identity["player_name"].notna(),
        "provider_player_table",
        "provider_nba_id_mapping",
    )

    team_names = pd.read_csv(sq_root / "teams.csv").drop_duplicates("team_id")
    team_abbr = team_names.set_index("team_id")["team_name"].map(TEAM_NAME_TO_ABBR)
    static_names = {
        int(player["id"]): str(player["full_name"])
        for player in nba_static_players.get_players()
    }
    static_name_counts = pd.Series(list(static_names.values())).value_counts()
    unique_static_ids_by_name = {
        name: nba_id
        for nba_id, name in static_names.items()
        if int(static_name_counts[name]) == 1
    }
    identity["player_name"] = identity["player_name"].fillna(
        identity["nba_id"].map(static_names)
    )
    exact_name_nba_id = identity["player_name"].map(unique_static_ids_by_name)
    exact_name_fill = identity["nba_id"].isna() & exact_name_nba_id.notna()
    identity.loc[exact_name_fill, "nba_id"] = exact_name_nba_id[exact_name_fill]
    identity.loc[exact_name_fill, "identity_source"] = (
        "provider_name_exact_nba_static"
    )
    existing_ids = set(identity["sq_player_id"].dropna().astype(int))
    recovered_rows: list[dict[str, object]] = []
    for season, raw_path in NBA_RAW_BY_SEASON.items():
        season_shots = shots[
            shots["season"].eq(season)
            & shots["player_id"].notna()
            & ~shots["player_id"].isin(existing_ids)
        ].copy()
        if season_shots.empty or not raw_path.exists():
            continue
        unknown = (
            season_shots.groupby(["player_id", "team_id"], dropna=False)
            .agg(three_pa=("made_3", "size"), three_pm=("made_3", "sum"))
            .reset_index()
        )
        unknown["team_abbr"] = unknown["team_id"].map(team_abbr)
        nba_totals = _nba_three_totals(raw_path)
        evidence: dict[int, list[pd.Series]] = {}
        for row in unknown.itertuples(index=False):
            if pd.isna(row.team_abbr):
                continue
            candidates = nba_totals[
                nba_totals["player1_team_abbreviation"].eq(row.team_abbr)
            ]
            match = _select_stat_match(
                sq_three_pa=int(row.three_pa),
                sq_three_pm=int(row.three_pm),
                candidates=candidates,
            )
            if match is not None:
                evidence.setdefault(int(row.player_id), []).append(match)
        for sq_player_id, matches in evidence.items():
            nba_ids = {int(match["player1_id"]) for match in matches}
            if len(nba_ids) != 1:
                continue
            nba_id = nba_ids.pop()
            match = matches[0]
            recovered_rows.append(
                {
                    "sq_player_id": sq_player_id,
                    "player_name": static_names.get(nba_id, str(match["player1_name"])),
                    "nba_id": nba_id,
                    "identity_source": "team_3pa_3pm_unique_nearest",
                }
            )

    if recovered_rows:
        recovered = pd.DataFrame(recovered_rows)
        consistent = recovered.groupby("sq_player_id")["nba_id"].transform(
            "nunique"
        ).eq(1)
        recovered = recovered[consistent].drop_duplicates("sq_player_id")
        identity = pd.concat([identity, recovered], ignore_index=True)
    identity["sq_player_id"] = pd.to_numeric(
        identity["sq_player_id"], errors="coerce"
    ).astype("Int64")
    identity["nba_id"] = pd.to_numeric(identity["nba_id"], errors="coerce").astype(
        "Int64"
    )
    audit = {
        "provider_identity_rows": int(
            identity["identity_source"].str.startswith("provider").sum()
        ),
        "provider_name_exact_nba_static_rows": int(
            identity["identity_source"].eq(
                "provider_name_exact_nba_static"
            ).sum()
        ),
        "recovered_identity_rows": int(
            identity["identity_source"].eq("team_3pa_3pm_unique_nearest").sum()
        ),
    }
    return identity, audit


def aggregate_player_difficulty(
    shots: pd.DataFrame,
    *,
    season_label: str,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    identified = shots["player_id"].notna() & shots["shooter_name"].notna()
    data = shots[identified].copy()
    if data.empty:
        raise ValueError(f"No player-attributed shots for {season_label}")

    covered = data["expected_make_probability"].notna()
    if not covered.any():
        raise ValueError(f"No Shot Taking coverage for {season_label}")
    league_expected = float(data.loc[covered, "expected_make_probability"].mean())
    data["expected_make_probability_filled"] = data[
        "expected_make_probability"
    ].fillna(league_expected)

    rows: list[dict[str, object]] = []
    for player_id, group in data.groupby("player_id", sort=False):
        expected = float(group["expected_make_probability_filled"].mean())
        raw = float(group["made_3"].mean())
        difficulty = league_expected - expected
        effective = raw + difficulty
        rows.append(
            {
                "season": season_label,
                "sq_player_id": int(player_id),
                "player_name": _mode_text(group["shooter_name"]),
                "team_ids": "|".join(
                    str(int(value))
                    for value in sorted(group["team_id"].dropna().unique())
                ),
                "three_pa": int(len(group)),
                "three_pm": int(group["made_3"].sum()),
                "model_covered_three_pa": int(
                    group["expected_make_probability"].notna().sum()
                ),
                "raw_3p_pct": raw,
                "expected_3p_pct": expected,
                "league_expected_3p_pct": league_expected,
                "difficulty": difficulty,
                "effective_3p_pct": effective,
                "shooting_over_expectation": raw - expected,
                "points_above_expectation": float(
                    3.0
                    * (
                        group["made_3"].sum()
                        - group["expected_make_probability_filled"].sum()
                    )
                ),
            }
        )

    result = pd.DataFrame(rows)
    for column in [
        "raw_3p_pct",
        "expected_3p_pct",
        "league_expected_3p_pct",
        "difficulty",
        "effective_3p_pct",
        "shooting_over_expectation",
    ]:
        result[f"{column}_pp"] = 100.0 * result[column]
    result["model_coverage_pct"] = (
        100.0 * result["model_covered_three_pa"] / result["three_pa"]
    )

    identity_gap = (
        result["raw_3p_pct"]
        + result["difficulty"]
        - result["effective_3p_pct"]
    ).abs()
    weighted_difficulty = float(
        np.average(result["difficulty"], weights=result["three_pa"])
    )
    audit: dict[str, float | int | str] = {
        "season": season_label,
        "games": int(shots["game_id"].nunique()),
        "source_three_pa": int(len(shots)),
        "source_model_covered_three_pa": int(
            shots["expected_make_probability"].notna().sum()
        ),
        "three_pa": int(len(data)),
        "unidentified_three_pa": int(len(shots) - len(data)),
        "three_pm": int(data["made_3"].sum()),
        "players": int(len(result)),
        "model_covered_three_pa": int(covered.sum()),
        "model_coverage_pct": float(100.0 * covered.mean()),
        "league_actual_3p_pct": float(data["made_3"].mean()),
        "league_expected_3p_pct": league_expected,
        "attempt_weighted_difficulty": weighted_difficulty,
        "max_identity_gap": float(identity_gap.max()),
    }
    if abs(weighted_difficulty) > 1e-12:
        raise AssertionError(
            f"Attempt-weighted difficulty did not center for {season_label}: "
            f"{weighted_difficulty}"
        )
    if identity_gap.max() > 1e-12:
        raise AssertionError(f"Effective 3P identity failed for {season_label}")
    return result.sort_values(
        ["three_pa", "effective_3p_pct"], ascending=[False, False]
    ).reset_index(drop=True), audit


def attach_nba_ids(result: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    mapping = identity.dropna(subset=["sq_player_id"]).drop_duplicates(
        "sq_player_id", keep=False
    )
    joined = result.merge(
        mapping[["sq_player_id", "nba_id", "identity_source"]],
        on="sq_player_id",
        how="left",
        validate="many_to_one",
    )
    joined["nba_id"] = joined["nba_id"].astype("Int64")
    front = [
        "season",
        "nba_id",
        "sq_player_id",
        "player_name",
        "team_ids",
        "identity_source",
    ]
    return joined[front + [column for column in joined.columns if column not in front]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sq-root", type=Path, default=DEFAULT_SQ_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--qualified-attempts", type=int, default=100)
    args = parser.parse_args()

    shots = load_regular_season_threes(args.sq_root)
    identity, identity_audit = build_player_identity_bridge(shots, args.sq_root)
    recovered_names = identity.dropna(
        subset=["sq_player_id", "player_name"]
    ).drop_duplicates("sq_player_id", keep=False).set_index("sq_player_id")[
        "player_name"
    ]
    shots["shooter_name"] = shots["shooter_name"].fillna(
        shots["player_id"].map(recovered_names)
    )
    outputs: list[pd.DataFrame] = []
    audits: list[dict[str, float | int | str]] = []
    for season in [SEASONS[key]["label"] for key in SEASONS]:
        result, audit = aggregate_player_difficulty(
            shots[shots["season"].eq(season)], season_label=season
        )
        outputs.append(result)
        audits.append(audit)
    pooled, pooled_audit = aggregate_player_difficulty(
        shots, season_label="2024-26 pooled"
    )
    audits.append(pooled_audit)

    player_seasons = attach_nba_ids(pd.concat(outputs, ignore_index=True), identity)
    pooled = attach_nba_ids(pooled, identity)
    qualified = player_seasons[
        player_seasons["three_pa"].ge(args.qualified_attempts)
    ].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    identity.to_csv(args.output_dir / "player_identity_bridge.csv", index=False)
    player_seasons.to_csv(
        args.output_dir / "sq_three_point_difficulty_player_season.csv", index=False
    )
    qualified.to_csv(
        args.output_dir / "sq_three_point_difficulty_qualified.csv", index=False
    )
    pooled.to_csv(
        args.output_dir / "sq_three_point_difficulty_2024_26_pooled.csv", index=False
    )
    manifest = {
        "model": "shot_taking.initial_shot_taking.value / 3",
        "difficulty_definition": "league expected 3P% - player expected 3P%",
        "effective_identity": "raw 3P% + difficulty = effective 3P%",
        "missing_model_policy": "impute season league expectation (neutral difficulty)",
        "qualified_attempts": int(args.qualified_attempts),
        "identity_audit": identity_audit,
        "audits": audits,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
