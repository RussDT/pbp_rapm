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
SEASON_TYPE_PREFIX = {
    "Regular Season": "102",
    "Playoffs": "104",
}

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


def try_fetch_json(url: str, timeout: int = 30) -> dict[str, Any] | None:
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
    except HTTPError as exc:
        if exc.code in {403, 404}:
            return None
        raise RuntimeError(f"failed to fetch {url}: {exc}") from exc
    except (URLError, TimeoutError):
        return None


def wnba_game_id(season: int, season_type: str, sequence: int) -> str:
    prefix = SEASON_TYPE_PREFIX[season_type]
    return f"{prefix}{season % 100:02d}{sequence:05d}"


def default_output_path(season: int, season_type: str) -> Path:
    suffix = "_PS" if season_type == "Playoffs" else ""
    return Path(f"wnba_test/WNBA{season % 100:02d}{suffix}.csv")


def should_keep_game(game: dict[str, Any], include_live: bool, include_scheduled: bool) -> bool:
    status = int(game.get("gameStatus") or 0)
    if status == 3:
        return True
    if status == 2 and include_live:
        return True
    if status == 1 and include_scheduled:
        return True
    return False


def discover_game_ids(
    season: int,
    season_type: str,
    start_sequence: int,
    end_sequence: int | None,
    max_misses: int,
    include_live: bool,
    include_scheduled: bool,
    sleep_seconds: float,
) -> list[str]:
    """Discover currently available WNBA CDN game ids by scanning the id sequence."""
    game_ids: list[str] = []
    consecutive_misses = 0
    sequence = start_sequence

    while True:
        if end_sequence is not None and sequence > end_sequence:
            break
        if end_sequence is None and consecutive_misses >= max_misses:
            break

        game_id = wnba_game_id(season, season_type, sequence)
        data = try_fetch_json(BOXSCORE_URL.format(game_id=game_id))
        if data is None:
            consecutive_misses += 1
            sequence += 1
            time.sleep(sleep_seconds)
            continue

        consecutive_misses = 0
        game = data.get("game", {})
        if should_keep_game(game, include_live, include_scheduled):
            game_ids.append(game_id)
            print(
                f"Discovered {game_id}: {game.get('gameStatusText')} "
                f"{game.get('gameTimeUTC')} "
                f"{game.get('awayTeam', {}).get('teamTricode')}@{game.get('homeTeam', {}).get('teamTricode')}"
            )
        else:
            print(
                f"Skipping {game_id}: {game.get('gameStatusText')} "
                f"{game.get('gameTimeUTC')}"
            )

        sequence += 1
        time.sleep(sleep_seconds)

    return game_ids


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


def boxscore_player_lookup(*teams: dict[str, Any]) -> dict[int, dict[str, Any]]:
    lookup: dict[int, dict[str, Any]] = {}
    for team in teams:
        team_tricode = team.get("teamTricode") or ""
        team_id = team.get("teamId")
        for player in team.get("players", []):
            person_id = int(player.get("personId") or 0)
            if not person_id:
                continue
            lookup[person_id] = {
                "name": player.get("name") or "",
                "nameI": player.get("nameI") or "",
                "team": team_tricode,
                "team_id": team_id,
            }
    return lookup


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
    player_lookup = boxscore_player_lookup(away_team, home_team)

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
            person_id = int(action.get("personId") or 0)
            player_info = player_lookup.get(person_id, {})
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
                "personid": person_id,
                "playername": player_info.get("name") or action.get("playerName") or "",
                "playernamei": player_info.get("nameI") or action.get("playerNameI") or "",
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
                "player1_id": person_id,
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


def load_existing_game_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    try:
        existing = pd.read_csv(output_path, usecols=["game_id"], dtype={"game_id": str})
    except Exception:
        return set()
    return set(existing["game_id"].dropna().astype(str).str.replace(r"\.0$", "", regex=True))


def write_games(
    game_ids: list[str],
    output_path: Path,
    overwrite: bool,
    sleep_seconds: float,
) -> pd.DataFrame:
    existing_game_ids = set() if overwrite else load_existing_game_ids(output_path)
    new_game_ids = [game_id for game_id in game_ids if game_id not in existing_game_ids]

    if existing_game_ids:
        print(f"Found {len(existing_game_ids)} existing games in {output_path}")
    print(f"Games requested: {len(game_ids)}; new games to fetch: {len(new_game_ids)}")

    frames = []
    if output_path.exists() and not overwrite:
        frames.append(pd.read_csv(output_path, dtype={"game_id": str}))

    for game_id in new_game_ids:
        print(f"Fetching {game_id} from WNBA CDN...")
        frame = normalize_game(game_id)
        print(f"  {len(frame)} rows")
        frames.append(frame)
        time.sleep(sleep_seconds)

    if not new_game_ids and output_path.exists() and not overwrite:
        print("No new games fetched.")
        return frames[0]

    if not frames:
        raise RuntimeError("No WNBA games were fetched or available to write")

    output = pd.concat(frames, ignore_index=True)
    if "game_id" in output.columns:
        output["game_id"] = output["game_id"].astype(str).str.replace(r"\.0$", "", regex=True)
        output = output.drop_duplicates(subset=["game_id", "event_num"], keep="last")
        output = output.sort_values(["game_id", "event_num"], kind="mergesort")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Saved {len(output)} rows across {output['game_id'].nunique()} games to {output_path}")
    return output


def update_player_index_map(raw: pd.DataFrame, map_path: Path) -> None:
    if raw.empty:
        return
    if map_path.exists():
        player_map = pd.read_csv(map_path)
    else:
        player_map = pd.DataFrame(
            columns=["year_season", "player_id", "EntityId", "ref_name", "pbp_name", "team", "team_id", "match_method"]
        )

    existing_ids = set(player_map.get("EntityId", pd.Series(dtype=str)).astype(str))
    players = raw[["game_id", "personid", "playername", "teamtricode", "teamid"]].copy()
    players["personid"] = pd.to_numeric(players["personid"], errors="coerce")
    players = players.dropna(subset=["personid"])
    players["personid"] = players["personid"].astype(int)
    players = players[
        (players["personid"] > 0)
        & players["playername"].fillna("").astype(str).str.strip().ne("")
        & ~players["personid"].astype(str).isin(existing_ids)
    ]
    if players.empty:
        print(f"No new player ids to add to {map_path}")
        return

    players["year_season"] = (
        "20" + players["game_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.slice(3, 5)
    ).astype(int)
    players = (
        players.sort_values(["year_season", "personid", "teamtricode"])
        .drop_duplicates(subset=["year_season", "personid"], keep="last")
    )
    additions = pd.DataFrame(
        {
            "year_season": players["year_season"],
            "player_id": "",
            "EntityId": players["personid"],
            "ref_name": players["playername"],
            "pbp_name": players["playername"],
            "team": players["teamtricode"],
            "team_id": players["teamid"],
            "match_method": "cdn_boxscore",
        }
    )
    updated = pd.concat([player_map, additions], ignore_index=True)
    updated.to_csv(map_path, index=False)
    print(f"Added {len(additions)} player ids to {map_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch WNBA PBP from cdn.wnba.com liveData JSON")
    parser.add_argument("game_ids", nargs="*", help="WNBA game ids, e.g. 1022500001")
    parser.add_argument("--season", type=int, help="WNBA season to scan, e.g. 2026")
    parser.add_argument(
        "--season-type",
        default="Regular Season",
        choices=sorted(SEASON_TYPE_PREFIX),
        help="Season type to scan when --season is provided",
    )
    parser.add_argument("--start-seq", type=int, default=1, help="First game sequence to scan")
    parser.add_argument("--end-seq", type=int, help="Last game sequence to scan")
    parser.add_argument(
        "--max-misses",
        type=int,
        default=20,
        help="Stop season scanning after this many consecutive missing CDN ids",
    )
    parser.add_argument("--include-live", action="store_true", help="Include in-progress games")
    parser.add_argument("--include-scheduled", action="store_true", help="Include scheduled games")
    parser.add_argument("--output", type=Path, help="CSV output path")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output instead of appending new games")
    parser.add_argument(
        "--update-player-map",
        action="store_true",
        help="Append newly seen WNBA player ids/names to wnba_test/player_index_map.csv",
    )
    parser.add_argument("--sleep", type=float, default=0.25, help="Delay between games")
    args = parser.parse_args()

    game_ids = list(args.game_ids)
    if args.season is not None:
        discovered = discover_game_ids(
            season=args.season,
            season_type=args.season_type,
            start_sequence=args.start_seq,
            end_sequence=args.end_seq,
            max_misses=args.max_misses,
            include_live=args.include_live,
            include_scheduled=args.include_scheduled,
            sleep_seconds=args.sleep,
        )
        game_ids.extend(discovered)

    game_ids = list(dict.fromkeys(str(game_id) for game_id in game_ids))
    if not game_ids:
        parser.error("provide game ids or --season")

    output_path = args.output
    if output_path is None:
        if args.season is None:
            parser.error("--output is required when fetching explicit game ids without --season")
        output_path = default_output_path(args.season, args.season_type)

    output = write_games(
        game_ids=game_ids,
        output_path=output_path,
        overwrite=args.overwrite,
        sleep_seconds=args.sleep,
    )
    if args.update_player_map:
        update_player_index_map(output, Path("wnba_test/player_index_map.csv"))


if __name__ == "__main__":
    main()
