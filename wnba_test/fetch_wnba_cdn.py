"""
Fetch WNBA play-by-play from the current WNBA static liveData CDN.

The old stats API path used by nba_api can time out or fail for WNBA games:
    https://stats.wnba.com/stats/playbyplayv3

This fetcher uses the public CDN JSON that powers wnba.com game pages:
    https://cdn.wnba.com/static/json/liveData/playbyplay/playbyplay_{GAME_ID}.json
    https://cdn.wnba.com/static/json/liveData/boxscore/boxscore_{GAME_ID}.json

It writes the raw CSV shape expected by wnba_test/process_rapm_wnba.py.
Lineups are reconstructed from boxscore starters plus substitution events.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


PBP_URL = "https://cdn.wnba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"
BOXSCORE_URL = "https://cdn.wnba.com/static/json/liveData/boxscore/boxscore_{game_id}.json"

RAW_COLUMNS = [
    "game_id",
    "event_num",
    "clock",
    "period",
    "teamid",
    "teamtricode",
    "personid",
    "playername",
    "playernamei",
    "xlegacy",
    "ylegacy",
    "shotdistance",
    "shotresult",
    "isfieldgoal",
    "home_score",
    "away_score",
    "pointstotal",
    "location",
    "description",
    "actiontype",
    "subtype",
    "videoavailable",
    "shotvalue",
    "actionid",
    "away_player1",
    "home_player1",
    "away_player2",
    "home_player2",
    "away_player3",
    "home_player3",
    "away_player4",
    "home_player4",
    "away_player5",
    "home_player5",
    "player1_id",
    "player1_team_id",
    "home_description",
    "visitor_description",
    "seconds_remaining_quarter",
    "score",
    "event_type",
]


def fetch_json(url: str, timeout: int = 30) -> dict[str, Any]:
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://www.wnba.com/",
        },
    )
    try:
        with urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"failed to fetch {url}: {exc}") from exc


def clock_to_seconds(clock: str | None) -> float:
    if not clock:
        return 0.0
    text = str(clock).replace("PT", "").replace("S", "")
    if "M" not in text:
        try:
            return float(text)
        except ValueError:
            return 0.0
    minutes, seconds = text.split("M", 1)
    return (float(minutes) * 60.0) + (float(seconds) if seconds else 0.0)


def event_type_number(action: dict[str, Any]) -> int:
    action_type = str(action.get("actionType") or "").lower()
    subtype = str(action.get("subType") or "").lower()
    shot_result = str(action.get("shotResult") or "").lower()

    if action_type == "period":
        return 12 if subtype == "start" else 13
    if action_type in {"2pt", "3pt"}:
        return 1 if shot_result == "made" else 2
    if action_type == "freethrow":
        return 3
    if action_type == "rebound":
        return 4
    if action_type == "turnover":
        return 5
    if action_type == "foul":
        return 6
    if action_type == "violation":
        return 7
    if action_type == "substitution":
        return 8
    if action_type == "timeout":
        return 9
    if action_type == "jumpball":
        return 10
    return 0


def legacy_action_type(action: dict[str, Any]) -> str:
    action_type = str(action.get("actionType") or "").lower()
    shot_result = str(action.get("shotResult") or "").lower()
    if action_type in {"2pt", "3pt"}:
        return "Made Shot" if shot_result == "made" else "Missed Shot"
    mapping = {
        "freethrow": "Free Throw",
        "jumpball": "Jump Ball",
        "rebound": "Rebound",
        "turnover": "Turnover",
        "foul": "Foul",
        "violation": "Violation",
        "substitution": "Substitution",
        "timeout": "Timeout",
        "period": "period",
        "game": "game",
        "block": "Block",
        "steal": "Steal",
    }
    return mapping.get(action_type, str(action.get("actionType") or ""))


def shot_value(action: dict[str, Any]) -> int:
    action_type = str(action.get("actionType") or "").lower()
    if action_type == "3pt":
        return 3
    if action_type == "2pt":
        return 2
    if action_type == "freethrow":
        return 1
    return 0


def team_info(boxscore: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    game = boxscore["game"]
    return game["awayTeam"], game["homeTeam"]


def starter_lineup(team: dict[str, Any]) -> list[int]:
    starters = [
        int(player["personId"])
        for player in team.get("players", [])
        if str(player.get("starter")) == "1"
    ]
    if len(starters) != 5:
        raise RuntimeError(f"expected 5 starters for {team.get('teamTricode')}, got {len(starters)}")
    return starters


def apply_substitution(lineup: list[int], action: dict[str, Any]) -> list[int]:
    player_id = int(action.get("personId") or 0)
    subtype = str(action.get("subType") or "").lower()
    updated = list(lineup)
    if subtype == "out":
        updated = [pid for pid in updated if pid != player_id]
    elif subtype == "in" and player_id and player_id not in updated:
        updated.append(player_id)
    return updated[:5]


def apply_substitution_group(lineup: list[int], actions: list[dict[str, Any]]) -> list[int]:
    updated = list(lineup)
    for action in actions:
        if str(action.get("subType") or "").lower() == "out":
            updated = apply_substitution(updated, action)
    for action in actions:
        if str(action.get("subType") or "").lower() == "in":
            updated = apply_substitution(updated, action)
    return updated[:5]


def pad_lineup(lineup: list[int]) -> list[float | int | None]:
    padded: list[float | int | None] = list(lineup[:5])
    while len(padded) < 5:
        padded.append(None)
    return padded


def normalize_game(game_id: str) -> pd.DataFrame:
    pbp = fetch_json(PBP_URL.format(game_id=game_id))
    boxscore = fetch_json(BOXSCORE_URL.format(game_id=game_id))
    away_team, home_team = team_info(boxscore)
    away_id = int(away_team["teamId"])
    home_id = int(home_team["teamId"])
    away_lineup = starter_lineup(away_team)
    home_lineup = starter_lineup(home_team)

    rows: list[dict[str, Any]] = []
    actions = pbp["game"]["actions"]
    idx = 0
    while idx < len(actions):
        action_group = [actions[idx]]
        group_key = (actions[idx].get("period"), actions[idx].get("clock"))
        idx += 1
        while idx < len(actions) and (actions[idx].get("period"), actions[idx].get("clock")) == group_key:
            action_group.append(actions[idx])
            idx += 1

        away_subs = [
            action for action in action_group
            if action.get("actionType") == "substitution" and int(action.get("teamId") or 0) == away_id
        ]
        home_subs = [
            action for action in action_group
            if action.get("actionType") == "substitution" and int(action.get("teamId") or 0) == home_id
        ]
        if away_subs:
            away_lineup = apply_substitution_group(away_lineup, away_subs)
        if home_subs:
            home_lineup = apply_substitution_group(home_lineup, home_subs)

        away_players = pad_lineup(away_lineup)
        home_players = pad_lineup(home_lineup)
        for action in action_group:
            team_id = int(action.get("teamId") or 0)
            description = action.get("description") or ""
            score_away = pd.to_numeric(action.get("scoreAway"), errors="coerce")
            score_home = pd.to_numeric(action.get("scoreHome"), errors="coerce")
            if pd.isna(score_away):
                score_away = 0
            if pd.isna(score_home):
                score_home = 0

            row = {
                "game_id": str(pbp["game"]["gameId"]),
                "event_num": action.get("actionNumber"),
                "clock": action.get("clock"),
                "period": action.get("period"),
                "teamid": team_id,
                "teamtricode": action.get("teamTricode") or "",
                "personid": int(action.get("personId") or 0),
                "playername": action.get("playerName") or "",
                "playernamei": action.get("playerNameI") or "",
                "xlegacy": action.get("xLegacy") or 0,
                "ylegacy": action.get("yLegacy") or 0,
                "shotdistance": action.get("shotDistance") or 0,
                "shotresult": action.get("shotResult") or "",
                "isfieldgoal": action.get("isFieldGoal") or 0,
                "home_score": float(score_home),
                "away_score": float(score_away),
                "pointstotal": action.get("pointsTotal") or 0,
                "location": "h" if team_id == home_id else ("v" if team_id == away_id else ""),
                "description": description,
                "actiontype": legacy_action_type(action),
                "subtype": action.get("subType") or "",
                "videoavailable": 0,
                "shotvalue": shot_value(action),
                "actionid": action.get("orderNumber") or action.get("actionNumber"),
                "player1_id": int(action.get("personId") or 0),
                "player1_team_id": team_id,
                "home_description": description if team_id == home_id else "",
                "visitor_description": description if team_id == away_id else "",
                "seconds_remaining_quarter": clock_to_seconds(action.get("clock")),
                "score": f"{int(score_away)} - {int(score_home)}",
                "event_type": event_type_number(action),
            }
            for player_idx in range(5):
                row[f"away_player{player_idx + 1}"] = away_players[player_idx]
                row[f"home_player{player_idx + 1}"] = home_players[player_idx]
            rows.append(row)

    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch WNBA PBP from cdn.wnba.com liveData JSON")
    parser.add_argument("game_ids", nargs="+", help="WNBA game ids, e.g. 1022500001")
    parser.add_argument("--output", type=Path, required=True, help="CSV output path")
    parser.add_argument("--sleep", type=float, default=0.25, help="Delay between games")
    args = parser.parse_args()

    frames = []
    for game_id in args.game_ids:
        print(f"Fetching {game_id} from WNBA CDN...")
        frame = normalize_game(game_id)
        print(f"  {len(frame)} rows")
        frames.append(frame)
        time.sleep(args.sleep)

    output = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Saved {len(output)} rows to {args.output}")


if __name__ == "__main__":
    main()
