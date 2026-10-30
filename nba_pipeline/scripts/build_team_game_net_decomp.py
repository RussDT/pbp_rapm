#!/usr/bin/env python3
"""Build team-vs-team game logs from the possession-level net DECOMP lane.

The output has two exact additive views:

* ``net_continuation``: the eight public-style shot/FT/TOV/second-chance
  components, with miss recoverability priced into the shot and second-chance
  branches.
* ``shot_quality``: repo-owned ``RUSSELL_SHOTQUALITY v2`` expected shot value,
  shotmaking versus that expectation, free throws, turnovers, and gross
  second-chance points.

The processed RAPM rows use expected rather than realized free-throw points.
At team-game grain, ``ft_conversion_vs_expected`` bridges the modeled total to
the actual final score so every displayed view closes exactly to the scoreboard.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from .build_vpm_target_swap import assign_first_fga_to_possessions
except ImportError:  # Direct script execution.
    from build_vpm_target_swap import assign_first_fga_to_possessions


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PIPELINE_ROOT / "raw_data"
DEFAULT_NET_DIR = PIPELINE_ROOT / "processed_netcont_lane"
DEFAULT_OUTPUT_DIR = PIPELINE_ROOT / "results" / "team_game_net_decomp"
DEFAULT_RUSSELL_SHOTQUALITY_PATH = (
    PIPELINE_ROOT
    / "results"
    / "pbp_shot_quality"
    / "pbp_shot_quality_1997_2026.parquet"
)
DEFAULT_RUSSELL_SHOTQUALITY_COLUMN = "shot_quality_season_ev"

ROW_KEYS = ["game_id_norm", "period", "time_quarter", "event_num"]
OFFENSE_COLS = [f"O{i}" for i in range(1, 6)]
HOME_LINEUP_COLS = [f"home_player{i}" for i in range(1, 6)]
AWAY_LINEUP_COLS = [f"away_player{i}" for i in range(1, 6)]

NET_CONTINUATION_COMPONENTS = {
    "rim_frequency": "FC_DECOMP_NET_RIM_FREQ_Diff",
    "rim_finishing": "FC_DECOMP_NET_RIM_FG_Diff",
    "midrange_scoring": "FC_DECOMP_NET_MID_FG_Diff",
    "three_frequency": "FC_DECOMP_NET_THREE_FREQ_Diff",
    "three_finishing": "FC_DECOMP_NET_THREE_FG_Diff",
    "free_throw_value": "FC_FT_Diff",
}
SHOT_QUALITY_COMPONENTS = {
    "shot_quality": "FC_RUSSELL_SQ_V2_Diff",
    "shotmaking_vs_expectation": "FC_RUSSELL_MAKE_V2_Diff",
    "free_throw_value": "FC_FT_Diff",
}
DISPLAY_LABELS = {
    "rim_frequency": "Rim frequency",
    "rim_finishing": "Rim finishing",
    "midrange_scoring": "Midrange scoring",
    "three_frequency": "Three-point frequency",
    "three_finishing": "Three-point finishing",
    "shot_quality": "Russell SQ v2",
    "shotmaking_vs_expectation": "Shotmaking vs Russell SQ v2",
    "free_throw_value": "Free-throw value",
    "turnover_value": "Turnover value",
    "second_chance_value": "Second-chance value",
    "ft_conversion_vs_expected": "FT conversion vs expectation",
}


@dataclass(frozen=True)
class GameMetadata:
    game_id: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    home_players: frozenset[int]
    away_players: frozenset[int]


def normalize_game_id(values: pd.Series) -> pd.Series:
    """Return stable ten-character NBA game IDs."""
    return (
        values.astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .str.zfill(10)
    )


def _numeric_player_set(frame: pd.DataFrame, columns: Iterable[str]) -> frozenset[int]:
    values = pd.to_numeric(frame[list(columns)].stack(), errors="coerce").dropna()
    return frozenset(int(value) for value in values if int(value) > 0)


def _description_side_team(game: pd.DataFrame, side: str) -> str:
    description_col = "home_description" if side == "home" else "visitor_description"
    counts = (
        game.loc[game[description_col].notna(), "player1_team_abbreviation"]
        .dropna()
        .astype(str)
        .str.upper()
        .value_counts()
    )
    if counts.empty:
        raise ValueError(f"Could not infer {side} team from raw descriptions.")
    return str(counts.index[0])


def build_game_metadata(raw: pd.DataFrame) -> dict[str, GameMetadata]:
    required = {
        "game_id", "home_description", "visitor_description",
        "player1_team_abbreviation", "away_score", "home_score",
        *HOME_LINEUP_COLS, *AWAY_LINEUP_COLS,
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Raw PBP is missing game metadata columns: {sorted(missing)}")

    work = raw.copy()
    work["game_id_norm"] = normalize_game_id(work["game_id"])
    metadata: dict[str, GameMetadata] = {}
    for game_id, game in work.groupby("game_id_norm", sort=False):
        home_team = _description_side_team(game, "home")
        away_team = _description_side_team(game, "away")
        if home_team == away_team:
            raise ValueError(f"Game {game_id} has the same inferred home and away team: {home_team}")

        score_rows = game.dropna(subset=["away_score", "home_score"])
        if score_rows.empty:
            raise ValueError(f"Game {game_id} has no final score snapshot.")
        final = score_rows.iloc[-1]
        home_players = _numeric_player_set(game, HOME_LINEUP_COLS)
        away_players = _numeric_player_set(game, AWAY_LINEUP_COLS)
        overlap = home_players & away_players
        if overlap:
            raise ValueError(f"Game {game_id} has players on both venue sides: {sorted(overlap)[:10]}")

        metadata[str(game_id)] = GameMetadata(
            game_id=str(game_id),
            home_team=home_team,
            away_team=away_team,
            home_score=int(round(float(final["home_score"]))),
            away_score=int(round(float(final["away_score"]))),
            home_players=home_players,
            away_players=away_players,
        )
    return metadata


def assign_offense_teams(first_chance: pd.DataFrame, metadata: dict[str, GameMetadata]) -> pd.Series:
    """Identify each processed row's offense from raw home/away lineup membership."""
    required = {"game_id_norm", *OFFENSE_COLS}
    missing = required - set(first_chance.columns)
    if missing:
        raise ValueError(f"FIRST_CHANCE is missing offense-team columns: {sorted(missing)}")

    teams: list[str] = []
    for row in first_chance[["game_id_norm", *OFFENSE_COLS]].itertuples(index=False):
        game_id = str(row[0])
        if game_id not in metadata:
            raise ValueError(f"Processed game {game_id} is absent from raw PBP.")
        game = metadata[game_id]
        offense = {int(value) for value in row[1:] if pd.notna(value) and int(value) > 0}
        home_votes = len(offense & game.home_players)
        away_votes = len(offense & game.away_players)
        if home_votes == away_votes:
            raise ValueError(
                f"Could not resolve offense side for game {game_id}: "
                f"home_votes={home_votes}, away_votes={away_votes}, lineup={sorted(offense)}"
            )
        teams.append(game.home_team if home_votes > away_votes else game.away_team)
    return pd.Series(teams, index=first_chance.index, dtype="string")


def _validate_and_merge(first_chance: pd.DataFrame, second_chance: pd.DataFrame) -> pd.DataFrame:
    fc = first_chance.copy()
    sc = second_chance.copy()
    fc["game_id_norm"] = normalize_game_id(fc["game_id"])
    sc["game_id_norm"] = normalize_game_id(sc["game_id"])

    for name, frame in (("FIRST_CHANCE", fc), ("SECOND_CHANCE_CLEAN", sc)):
        missing = set(ROW_KEYS) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing alignment keys: {sorted(missing)}")
        if frame.duplicated(ROW_KEYS).any():
            examples = frame.loc[frame.duplicated(ROW_KEYS, keep=False), ROW_KEYS].head().to_dict("records")
            raise ValueError(f"{name} has duplicate possession keys: {examples}")

    sc_values = sc[ROW_KEYS + ["Off_Diff", "SC_NET_CONT_Diff"]].rename(
        columns={"Off_Diff": "SC_GROSS_Diff"}
    )
    merged = fc.merge(sc_values, on=ROW_KEYS, how="outer", validate="one_to_one", indicator=True)
    if not merged["_merge"].eq("both").all():
        counts = merged["_merge"].value_counts().to_dict()
        raise ValueError(f"FIRST_CHANCE/SECOND_CHANCE_CLEAN row mismatch: {counts}")
    return merged.drop(columns="_merge")


def load_russell_v2_shots(
    path: Path,
    source_file: str,
    value_column: str = DEFAULT_RUSSELL_SHOTQUALITY_COLUMN,
) -> pd.DataFrame:
    """Load one raw file's canonical Russell ShotQuality v2 FGA values."""
    required = ["source_file", "game_id", "event_num", value_column]
    try:
        shots = pd.read_parquet(
            path,
            columns=required,
            filters=[("source_file", "=", source_file)],
        )
    except Exception:
        shots = pd.read_parquet(path, columns=required)
        shots = shots.loc[shots["source_file"].eq(source_file)].copy()
    if shots.empty:
        raise ValueError(f"No Russell ShotQuality v2 rows found for {source_file} in {path}")

    shots["game_id"] = normalize_game_id(shots["game_id"])
    shots["event_num"] = pd.to_numeric(shots["event_num"], errors="coerce")
    shots[value_column] = pd.to_numeric(shots[value_column], errors="coerce")
    shots = shots.dropna(subset=["event_num", value_column])
    shots["event_num"] = shots["event_num"].astype(np.int64)

    duplicate = shots.duplicated(["game_id", "event_num"], keep=False)
    if duplicate.any():
        conflicts = (
            shots.loc[duplicate]
            .groupby(["game_id", "event_num"], sort=False)[value_column]
            .nunique()
        )
        if conflicts.gt(1).any():
            examples = [f"{game}/{event}" for game, event in conflicts[conflicts.gt(1)].index[:5]]
            raise ValueError(
                "Conflicting Russell ShotQuality v2 game/event keys: " + ", ".join(examples)
            )
        shots = shots.drop_duplicates(["game_id", "event_num"], keep="first")
    return shots[["game_id", "event_num", value_column]].reset_index(drop=True)


def attach_russell_v2_first_chance(
    first_chance: pd.DataFrame,
    raw: pd.DataFrame,
    shots: pd.DataFrame,
    value_column: str = DEFAULT_RUSSELL_SHOTQUALITY_COLUMN,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Replace provider ``initial_ev`` with Russell v2 on the first FGA.

    The public first-chance shot parent also carries the replacement-shot
    baseline on FT possessions. Preserve that baseline inside SQ, keep the
    existing FT value as the incremental FT branch, and put actual-minus-v2
    FGA points in shotmaking. The three pieces therefore retain exact
    first-chance closure.
    """
    required = {
        "FC_DECOMP_EFG_Diff", "FC_FT_Diff", "Net_Diff", "Has_FC_FGA"
    }
    missing = required - set(first_chance.columns)
    if missing:
        raise ValueError(f"FIRST_CHANCE is missing Russell v2 swap columns: {sorted(missing)}")
    shot_missing = {"game_id", "event_num", value_column} - set(shots.columns)
    if shot_missing:
        raise ValueError(f"Russell v2 shot table is missing columns: {sorted(shot_missing)}")

    assignments, stats = assign_first_fga_to_possessions(first_chance, raw)
    shot_values = shots.copy()
    shot_values["game_id"] = normalize_game_id(shot_values["game_id"])
    shot_values["event_num"] = pd.to_numeric(
        shot_values["event_num"], errors="coerce"
    ).astype("Int64")
    shot_values[value_column] = pd.to_numeric(shot_values[value_column], errors="coerce")
    shot_values = shot_values.dropna(subset=["event_num", value_column])
    shot_values["event_num"] = shot_values["event_num"].astype(np.int64)
    if shot_values.duplicated(["game_id", "event_num"]).any():
        raise ValueError("Russell v2 shot table must have unique game/event keys")

    assignments = assignments.merge(
        shot_values[["game_id", "event_num", value_column]],
        on=["game_id", "event_num"],
        how="left",
        validate="one_to_one",
    )
    if assignments[value_column].isna().any():
        examples = assignments.loc[
            assignments[value_column].isna(), ["game_id", "event_num"]
        ].head().to_dict("records")
        raise ValueError(f"First-chance FGAs missing Russell v2 values: {examples}")

    out = first_chance.reset_index(drop=True).copy()
    assigned = np.zeros(len(out), dtype=bool)
    actual_fga = np.zeros(len(out), dtype=np.float64)
    russell_fga_ev = np.zeros(len(out), dtype=np.float64)
    row_index = assignments["_row"].to_numpy(dtype=np.int64)
    assigned[row_index] = True
    actual_fga[row_index] = assignments["_actual_pts"].to_numpy(dtype=np.float64)
    russell_fga_ev[row_index] = assignments[value_column].to_numpy(dtype=np.float64)

    has_fga = pd.to_numeric(out["Has_FC_FGA"], errors="coerce").fillna(0).astype(bool).to_numpy()
    if not np.array_equal(assigned, has_fga):
        raise ValueError(
            "Russell v2 first-FGA assignment disagrees with Has_FC_FGA: "
            f"missing={int((has_fga & ~assigned).sum())}, "
            f"unexpected={int((assigned & ~has_fga).sum())}"
        )

    shot_parent = pd.to_numeric(out["FC_DECOMP_EFG_Diff"], errors="coerce").to_numpy(float)
    ft_baseline = shot_parent - actual_fga
    out["FC_RUSSELL_FGA_EV_V2_Diff"] = russell_fga_ev
    out["FC_RUSSELL_FT_BASELINE_V2_Diff"] = ft_baseline
    out["FC_RUSSELL_SQ_V2_Diff"] = russell_fga_ev + ft_baseline
    out["FC_RUSSELL_MAKE_V2_Diff"] = actual_fga - russell_fga_ev

    identity = (
        out["FC_RUSSELL_SQ_V2_Diff"]
        + out["FC_RUSSELL_MAKE_V2_Diff"]
        + pd.to_numeric(out["FC_FT_Diff"], errors="coerce")
        - pd.to_numeric(out["Net_Diff"], errors="coerce")
    )
    max_gap = float(identity.abs().max())
    if max_gap > 1e-9:
        raise ValueError(f"Russell v2 first-chance identity failed; max gap={max_gap:.12g}")
    stats = dict(stats)
    stats.update({
        "russell_v2_values_attached": int(assigned.sum()),
        "max_abs_russell_v2_first_chance_gap": max_gap,
    })
    return out, stats


def build_row_components(first_chance: pd.DataFrame, second_chance: pd.DataFrame) -> pd.DataFrame:
    """Create exact row-level net-continuation and ShotQuality component values."""
    rows = _validate_and_merge(first_chance, second_chance)
    required = {
        "Net_Diff", "Is_Turnover", "FC_TS_NET_Diff", "SC_GROSS_Diff",
        "SC_NET_CONT_Diff", *NET_CONTINUATION_COMPONENTS.values(),
        *SHOT_QUALITY_COMPONENTS.values(),
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Net DECOMP inputs are missing component columns: {sorted(missing)}")

    is_turnover = pd.to_numeric(rows["Is_Turnover"], errors="coerce").fillna(0).astype(int).eq(1)
    non_turnover = ~is_turnover
    if not non_turnover.any():
        raise ValueError("Cannot price turnover value without non-turnover possessions.")

    output = rows.copy()
    for component, source in NET_CONTINUATION_COMPONENTS.items():
        values = pd.to_numeric(output[source], errors="coerce")
        if values.loc[non_turnover].isna().any():
            raise ValueError(f"{source} has missing non-turnover values.")
        baseline = float(values.loc[non_turnover].mean())
        output[f"netcont__{component}"] = np.where(is_turnover, baseline, values)
    net_turnover_price = float(
        pd.to_numeric(output.loc[non_turnover, "FC_TS_NET_Diff"], errors="coerce").mean()
    )
    output["netcont__turnover_value"] = np.where(is_turnover, -net_turnover_price, 0.0)
    output["netcont__second_chance_value"] = pd.to_numeric(
        output["SC_NET_CONT_Diff"], errors="coerce"
    )

    for component, source in SHOT_QUALITY_COMPONENTS.items():
        values = pd.to_numeric(output[source], errors="coerce")
        if values.loc[non_turnover].isna().any():
            raise ValueError(f"{source} has missing non-turnover values.")
        baseline = float(values.loc[non_turnover].mean())
        output[f"sq__{component}"] = np.where(is_turnover, baseline, values)
    gross_turnover_price = float(
        pd.to_numeric(output.loc[non_turnover, "Net_Diff"], errors="coerce").mean()
    )
    output["sq__turnover_value"] = np.where(is_turnover, -gross_turnover_price, 0.0)
    output["sq__second_chance_value"] = pd.to_numeric(output["SC_GROSS_Diff"], errors="coerce")

    modeled_points = (
        pd.to_numeric(output["Net_Diff"], errors="coerce")
        + pd.to_numeric(output["SC_GROSS_Diff"], errors="coerce")
    )
    output["modeled_points"] = modeled_points
    netcont_cols = [col for col in output.columns if col.startswith("netcont__")]
    sq_cols = [col for col in output.columns if col.startswith("sq__")]
    for view, columns in (("net_continuation", netcont_cols), ("shot_quality", sq_cols)):
        gap = output[columns].sum(axis=1) - modeled_points
        max_gap = float(gap.abs().max())
        if max_gap > 1e-9:
            raise ValueError(f"{view} row closure failed; max gap={max_gap:.12g}")
    return output


def _parse_game_selection(spec: str | None) -> set[int] | None:
    if spec is None or not spec.strip():
        return None
    selected: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if "-" in token:
            start, end = (int(value) for value in token.split("-", 1))
            selected.update(range(start, end + 1))
        else:
            selected.add(int(token))
    return selected


def build_matchup_outputs(
    row_components: pd.DataFrame,
    metadata: dict[str, GameMetadata],
    team: str,
    opponent: str,
    games: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    team = team.upper()
    opponent = opponent.upper()
    matchup_ids = [
        game_id for game_id, game in metadata.items()
        if {game.home_team, game.away_team} == {team, opponent}
    ]
    if not matchup_ids:
        raise ValueError(f"No {team}-{opponent} games were found.")

    game_dates = (
        row_components.loc[row_components["game_id_norm"].isin(matchup_ids), ["game_id_norm", "game_date"]]
        .drop_duplicates("game_id_norm")
        .assign(game_date=lambda frame: pd.to_datetime(frame["game_date"]))
        .sort_values(["game_date", "game_id_norm"])
    )
    ordered_ids = game_dates["game_id_norm"].tolist()
    series_number = {game_id: number for number, game_id in enumerate(ordered_ids, start=1)}
    selected = _parse_game_selection(games)
    if selected is not None:
        matchup_ids = [game_id for game_id in ordered_ids if series_number[game_id] in selected]
    else:
        matchup_ids = ordered_ids
    if not matchup_ids:
        raise ValueError("The game selection removed every matchup game.")

    work = row_components.loc[row_components["game_id_norm"].isin(matchup_ids)].copy()
    work["offense_team"] = assign_offense_teams(work, metadata)
    bad_teams = set(work["offense_team"].unique()) - {team, opponent}
    if bad_teams:
        raise ValueError(f"Unexpected offense teams in matchup rows: {sorted(bad_teams)}")

    view_columns = {
        "net_continuation": [col for col in work.columns if col.startswith("netcont__")],
        "shot_quality": [col for col in work.columns if col.startswith("sq__")],
    }
    summary_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []

    for game_id in matchup_ids:
        game = metadata[game_id]
        game_rows = work.loc[work["game_id_norm"].eq(game_id)]
        date_value = pd.to_datetime(game_rows["game_date"].iloc[0]).date().isoformat()
        team_score = game.home_score if game.home_team == team else game.away_score
        opponent_score = game.home_score if game.home_team == opponent else game.away_score
        team_possessions = int(game_rows["offense_team"].eq(team).sum())
        opponent_possessions = int(game_rows["offense_team"].eq(opponent).sum())
        team_turnovers = int(
            game_rows.loc[game_rows["offense_team"].eq(team), "Is_Turnover"].sum()
        )
        opponent_turnovers = int(
            game_rows.loc[game_rows["offense_team"].eq(opponent), "Is_Turnover"].sum()
        )
        modeled_team = float(game_rows.loc[game_rows["offense_team"].eq(team), "modeled_points"].sum())
        modeled_opponent = float(
            game_rows.loc[game_rows["offense_team"].eq(opponent), "modeled_points"].sum()
        )
        team_ft_bridge = float(team_score - modeled_team)
        opponent_ft_bridge = float(opponent_score - modeled_opponent)

        base_summary: dict[str, object] = {
            "game_id": game_id,
            "game_date": date_value,
            "series_game": series_number[game_id],
            "team": team,
            "opponent": opponent,
            "venue": "home" if game.home_team == team else "away",
            "team_score": team_score,
            "opponent_score": opponent_score,
            "score_margin": team_score - opponent_score,
            "team_offensive_possessions": team_possessions,
            "opponent_offensive_possessions": opponent_possessions,
            "team_first_chance_turnovers": team_turnovers,
            "opponent_first_chance_turnovers": opponent_turnovers,
            "modeled_team_points": modeled_team,
            "modeled_opponent_points": modeled_opponent,
            "modeled_margin": modeled_team - modeled_opponent,
            "ft_conversion_bridge_margin": team_ft_bridge - opponent_ft_bridge,
            "shot_quality_model": "RUSSELL_SHOTQUALITY v2",
        }

        for view, columns in view_columns.items():
            view_net: dict[str, float] = {}
            for column in columns:
                component = column.split("__", 1)[1]
                team_value = float(game_rows.loc[game_rows["offense_team"].eq(team), column].sum())
                opponent_value = float(
                    game_rows.loc[game_rows["offense_team"].eq(opponent), column].sum()
                )
                net_value = team_value - opponent_value
                view_net[component] = net_value
                component_rows.append({
                    "game_id": game_id,
                    "game_date": date_value,
                    "series_game": series_number[game_id],
                    "team": team,
                    "opponent": opponent,
                    "view": view,
                    "component": component,
                    "component_label": DISPLAY_LABELS[component],
                    "team_points": team_value,
                    "opponent_points": opponent_value,
                    "net_points": net_value,
                    "team_per_100": 100.0 * team_value / team_possessions,
                    "opponent_per_100": 100.0 * opponent_value / opponent_possessions,
                    "net_per_100": (
                        100.0 * team_value / team_possessions
                        - 100.0 * opponent_value / opponent_possessions
                    ),
                })

            component = "ft_conversion_vs_expected"
            bridge_net = team_ft_bridge - opponent_ft_bridge
            view_net[component] = bridge_net
            component_rows.append({
                "game_id": game_id,
                "game_date": date_value,
                "series_game": series_number[game_id],
                "team": team,
                "opponent": opponent,
                "view": view,
                "component": component,
                "component_label": DISPLAY_LABELS[component],
                "team_points": team_ft_bridge,
                "opponent_points": opponent_ft_bridge,
                "net_points": bridge_net,
                "team_per_100": 100.0 * team_ft_bridge / team_possessions,
                "opponent_per_100": 100.0 * opponent_ft_bridge / opponent_possessions,
                "net_per_100": (
                    100.0 * team_ft_bridge / team_possessions
                    - 100.0 * opponent_ft_bridge / opponent_possessions
                ),
            })
            closure_gap = float(sum(view_net.values()) - (team_score - opponent_score))
            if abs(closure_gap) > 1e-9:
                raise ValueError(
                    f"{view} game closure failed for {game_id}; gap={closure_gap:.12g}"
                )
            for component, value in view_net.items():
                base_summary[f"{view}__{component}__net_points"] = value
            base_summary[f"{view}__closure_gap"] = closure_gap
            leading = max(view_net, key=view_net.get)
            base_summary[f"{view}__leading_positive_driver"] = leading
            base_summary[f"{view}__leading_positive_driver_points"] = view_net[leading]

        summary_rows.append(base_summary)

    summary = pd.DataFrame(summary_rows).sort_values("series_game").reset_index(drop=True)
    components = pd.DataFrame(component_rows).sort_values(
        ["series_game", "view", "net_points"], ascending=[True, True, False]
    ).reset_index(drop=True)
    series = (
        components.groupby(["team", "opponent", "view", "component", "component_label"], as_index=False)
        [["team_points", "opponent_points", "net_points"]]
        .sum()
    )
    series.insert(2, "games", len(summary))
    series = series.sort_values(["view", "net_points"], ascending=[True, False]).reset_index(drop=True)
    for view in series["view"].unique():
        gap = float(series.loc[series["view"].eq(view), "net_points"].sum() - summary["score_margin"].sum())
        if abs(gap) > 1e-9:
            raise ValueError(f"{view} series closure failed; gap={gap:.12g}")
    numeric_cols = components.select_dtypes(include="number").columns
    components[numeric_cols] = components[numeric_cols].round(6)
    numeric_summary = summary.select_dtypes(include="number").columns
    summary[numeric_summary] = summary[numeric_summary].round(6)
    series[["team_points", "opponent_points", "net_points"]] = series[
        ["team_points", "opponent_points", "net_points"]
    ].round(6)
    return summary, components, series


def input_paths(
    year: int,
    season_type: str,
    raw_dir: Path = DEFAULT_RAW_DIR,
    net_dir: Path = DEFAULT_NET_DIR,
) -> tuple[Path, Path, Path]:
    yy = year % 100
    suffix = "_PS" if season_type.lower() == "ps" else ""
    return (
        raw_dir / f"NBA{yy:02d}{suffix}.parquet",
        net_dir / f"FIRST_CHANCE{yy:02d}{suffix}.parquet",
        net_dir / f"SECOND_CHANCE_CLEAN{yy:02d}{suffix}.parquet",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True, help="Season end year, e.g. 2026")
    parser.add_argument("--season-type", choices=["rs", "ps"], default="ps")
    parser.add_argument("--team", required=True, help="Perspective team abbreviation, e.g. NYK")
    parser.add_argument("--opponent", required=True, help="Opponent abbreviation, e.g. CLE")
    parser.add_argument("--games", help="Series game selection, e.g. 1-3 or 1,3")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--net-dir", type=Path, default=DEFAULT_NET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--russell-shotquality-path",
        type=Path,
        default=DEFAULT_RUSSELL_SHOTQUALITY_PATH,
        help="Canonical per-FGA RUSSELL_SHOTQUALITY v2 parquet.",
    )
    parser.add_argument(
        "--russell-shotquality-column",
        default=DEFAULT_RUSSELL_SHOTQUALITY_COLUMN,
        help="Per-FGA Russell v2 EV column (default: shot_quality_season_ev).",
    )
    args = parser.parse_args()

    raw_path, fc_path, sc_path = input_paths(
        args.year, args.season_type, raw_dir=args.raw_dir, net_dir=args.net_dir
    )
    for path in (raw_path, fc_path, sc_path, args.russell_shotquality_path):
        if not path.exists():
            raise FileNotFoundError(path)

    raw = pd.read_parquet(raw_path)
    first_chance = pd.read_parquet(fc_path)
    second_chance = pd.read_parquet(sc_path)
    russell_shots = load_russell_v2_shots(
        args.russell_shotquality_path,
        source_file=raw_path.name,
        value_column=args.russell_shotquality_column,
    )
    first_chance, russell_stats = attach_russell_v2_first_chance(
        first_chance,
        raw,
        russell_shots,
        value_column=args.russell_shotquality_column,
    )
    print(
        "Russell SQ v2: "
        f"attached {russell_stats['russell_v2_values_attached']:,} first FGAs; "
        "max first-chance gap "
        f"{russell_stats['max_abs_russell_v2_first_chance_gap']:.3g}"
    )
    metadata = build_game_metadata(raw)
    row_components = build_row_components(first_chance, second_chance)
    summary, components, series = build_matchup_outputs(
        row_components,
        metadata,
        team=args.team,
        opponent=args.opponent,
        games=args.games,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.year}_{args.season_type}_{args.team.upper()}_vs_{args.opponent.upper()}"
    summary_path = args.output_dir / f"{stem}_games.csv"
    components_path = args.output_dir / f"{stem}_components.csv"
    series_path = args.output_dir / f"{stem}_series.csv"
    summary.to_csv(summary_path, index=False)
    components.to_csv(components_path, index=False)
    series.to_csv(series_path, index=False)
    print(f"Wrote {summary_path} ({len(summary):,} games)")
    print(f"Wrote {components_path} ({len(components):,} game-component rows)")
    print(f"Wrote {series_path} ({len(series):,} series-component rows)")


if __name__ == "__main__":
    main()
