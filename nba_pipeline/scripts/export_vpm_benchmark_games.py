"""RAPM-owned full-score/minute adapter for the private VPM field benchmark.

No impact ratings or model selection live here. Reconstruct post-event lineup
clock intervals, retain scoreboard targets, and quarantine incomplete games.
"""
from pathlib import Path

import numpy as np
import pandas as pd


SLOTS = {side: [f"{side}_player{i}" for i in range(1, 6)] for side in ("home", "away")}


def normalize_ids(values):
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def reconstruct_games(raw, processed, season):
    """Return game census, player-game-side minutes, and explicit audit rows."""
    raw = raw.drop_duplicates().copy()
    raw["game_id"] = normalize_ids(raw.game_id)
    clock = raw.time_quarter.astype("string").str.extract(r"^(\d+):(\d+(?:\.\d+)?)$")
    raw["clock"] = pd.to_numeric(clock[0]) * 60 + pd.to_numeric(clock[1])
    raw = raw.sort_values(["game_id", "period", "clock", "event_num"],
                          ascending=[True, True, False, True], kind="stable")
    # last() independently carries forward each sparse scoreboard side.
    scores = raw.groupby("game_id")[["home_score", "away_score"]].last()
    finals = raw.groupby("game_id").tail(1).set_index("game_id")
    intervals = raw.groupby(["game_id", "period", "clock"], sort=False).tail(1).copy()
    intervals["seconds"] = intervals.clock - intervals.groupby(["game_id", "period"]).clock.shift(-1).fillna(0)
    intervals = intervals.loc[intervals.seconds.gt(0)].copy()
    pieces = []
    invalid_seconds = pd.Series(0., index=scores.index)
    for side, columns in SLOTS.items():
        ids = intervals[columns].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.int64)
        other = intervals[SLOTS["away" if side == "home" else "home"]].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.int64)
        valid = ((ids > 0) & (ids < 1610612737)).all(axis=1)
        valid &= (np.diff(np.sort(ids, axis=1), axis=1) != 0).all(axis=1)
        valid &= ~(ids[:, :, None] == other[:, None, :]).any(axis=(1, 2))
        invalid_seconds = invalid_seconds.add(intervals.loc[~valid].groupby("game_id").seconds.sum(), fill_value=0)
        part = intervals.loc[valid, ["game_id", "seconds", *columns]].melt(
            id_vars=["game_id", "seconds"], value_vars=columns, value_name="player_id")
        part["side"] = side
        pieces.append(part)
    minutes = pd.concat(pieces).groupby(["game_id", "side", "player_id"], as_index=False).seconds.sum()
    minutes["player_id"] = minutes.player_id.astype(np.int64)
    minutes["minutes"] = minutes.pop("seconds") / 60
    # Team identity comes from scoring actors in the side's own descriptions.
    teams = {}
    for side, desc in [("home", "home_description"), ("away", "visitor_description")]:
        rows = raw.loc[raw.event_type.isin([1, 2, 3]) & raw[desc].fillna("").ne("")].copy()
        rows = rows.loc[pd.to_numeric(rows.player1_team_id, errors="coerce").ge(1610612737)]
        counts = rows.groupby(["game_id", "player1_team_id"]).size().rename("n").reset_index()
        teams[side] = counts.sort_values("n").drop_duplicates("game_id", keep="last").set_index("game_id").player1_team_id
    p = processed.copy()
    p["game_id"] = normalize_ids(p.game_id)
    dates = pd.to_datetime(p.game_date).groupby(p.game_id).agg(["min", "max"])
    possession_counts = p.groupby("game_id").size() / 2
    periods = raw.groupby("game_id").period.max()
    games = scores.join(periods).join(possession_counts.rename("pace")).join(dates)
    games["duration"] = 48 + (games.period - 4).clip(lower=0) * 5
    games["home_team"] = teams["home"]
    games["away_team"] = teams["away"]
    games["game_date"] = games["min"]
    games["margin"] = games.home_score - games.away_score
    mass = minutes.groupby(["game_id", "side"]).minutes.sum().unstack()
    games["minute_mass_error"] = mass.sub(5 * games.duration, axis=0).abs().max(axis=1)
    games["invalid_lineup_seconds"] = invalid_seconds
    games["complete_final_clock"] = finals.clock.eq(0) & finals.period.ge(4)
    games["date_unique"] = games["min"].eq(games["max"])
    games["teams_valid"] = games.home_team.notna() & games.away_team.notna() & games.home_team.ne(games.away_team)
    games["pace_valid"] = (games.pace * 48 / games.duration).between(70, 130)
    games["accepted"] = (games.minute_mass_error.le(1e-7) & games.invalid_lineup_seconds.eq(0)
                         & games.complete_final_clock & games.date_unique & games.teams_valid
                         & games.pace_valid & games.margin.ne(0) & games.home_score.gt(0) & games.away_score.gt(0))
    games["season"] = season
    games = games.reset_index()
    minutes = minutes.merge(games[["game_id", "home_team", "away_team", "season", "game_date", "duration"]], on="game_id", validate="many_to_one")
    minutes["team"] = np.where(minutes.side.eq("home"), minutes.home_team, minutes.away_team)
    return games, minutes


def export_season(pipeline_root: Path, season: int):
    columns = ["game_id", "event_num", "event_type", "period", "time_quarter", "home_score", "away_score",
               "home_description", "visitor_description", "player1_team_id", *SLOTS["home"], *SLOTS["away"]]
    raw_path = pipeline_root / "raw_data" / f"NBA{season % 100:02d}.parquet"
    processed_path = pipeline_root / "processed" / f"RAPM{season % 100:02d}.parquet"
    return reconstruct_games(pd.read_parquet(raw_path, columns=columns),
                             pd.read_parquet(processed_path, columns=["game_id", "game_date"]), season)
